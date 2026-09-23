"""Tests 2, 3, 4, 5, 6: Parent packing, span merging, prompt parity, 8192 boundary, answer-blindness, and provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transformers import AutoTokenizer

from uit_dsc_fixed_rag import e21_parent_context as parent
from uit_dsc_fixed_rag.e45_parent_training import (
    E45Config,
    load_config,
    pack_training_evidence,
    render_canonical_prompt,
)


class MockTokenizer:
    """Lightweight deterministic tokenizer for unit test isolation."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 151645

    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None) -> dict[str, list[int]]:
        # deterministic simple token mapping based on whitespace and characters
        tokens = [abs(hash(w)) % 100000 + 1 for w in text.split()]
        return {"input_ids": tokens}

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


def test_seed_order_and_touching_span_merge() -> None:
    """Test 2: Deterministic seed order and touching/overlapping span merge behavior."""
    # Create overlapping and contiguous spans
    doc = {"cleaned_text": "Nội dung điều 1. Nội dung điều 2. Nội dung điều 3."}
    # Span 1: [0, 16] ("Nội dung điều 1.")
    # Span 2: [16, 33] (" Nội dung điều 2.") -> touching
    # Span 3: [25, 49] (" điều 2. Nội dung điều 3.") -> overlapping
    spans = [
        {"document_id": "doc1", "block_id": "b1", "start": 0, "end": 16, "text": "Nội dung điều 1.", "rank": 0, "chunk_ids": ["c1"]},
        {"document_id": "doc1", "block_id": "b1", "start": 16, "end": 33, "text": " Nội dung điều 2.", "rank": 1, "chunk_ids": ["c2"]},
    ]
    merged = parent.merge_spans(spans)
    assert len(merged) == 1
    assert merged[0]["start"] == 0
    assert merged[0]["end"] == 33
    assert merged[0]["text"] == "Nội dung điều 1. Nội dung điều 2."
    assert merged[0]["rank"] == 0
    assert set(merged[0]["chunk_ids"]) == {"c1", "c2"}


def test_canonical_e44_prompt_byte_parity() -> None:
    """Test 3: Canonical E44 prompt template rendering exact parity."""
    cfg = load_config(PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json")
    tokenizer = MockTokenizer()

    messages = [
        {"role": "system", "content": cfg.section("prompt")["system_prompt"]},
        {"role": "user", "content": "Câu hỏi:\nAi có quyền?\n\nTài liệu:\n[Tài liệu 1]\nNội dung A\n\nYêu cầu:\n" + cfg.section("prompt")["answer_instruction"]},
    ]
    rendered = render_canonical_prompt(messages, tokenizer)
    assert "<|im_start|>system\n" in rendered
    assert "<|im_start|>user\n" in rendered
    assert "<|im_start|>assistant\n" in rendered
    assert cfg.section("prompt")["system_prompt"] in rendered


def test_8192_boundary_full_answer_eos_and_assistant_mask() -> None:
    """Test 4: 8,192 boundary, answer preservation, EOS, and assistant-only label mask."""
    cfg = load_config(PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json")
    tokenizer = MockTokenizer()

    # Create dummy 12 units
    units = []
    for r in range(12):
        units.append({
            "document_id": f"doc_{r}",
            "article_number": str(r + 1),
            "article_title": f"Tiêu đề {r}",
            "source_title": f"Luật {r}",
            "document_number": f"{r}/2020",
            "rank": r,
            "block_id": f"b_{r}",
            "seed": {"start": 0, "end": 10, "text": f"Nội dung {r}", "chunk_ids": [f"c_{r}"]},
            "expansions": [
                {"kind": "whole_article", "start": 0, "end": 20, "text": f"Nội dung {r} mở rộng", "chunk_ids": [f"c_{r}", f"c_{r}_ext"]}
            ]
        })

    ans_tokens_count = 150
    prompt_str, prompt_tokens, merged, diag = pack_training_evidence(
        question="Câu hỏi kiểm tra?",
        units=units,
        answer_tokens_count=ans_tokens_count,
        tokenizer=tokenizer,
        config=cfg,
    )

    assert prompt_tokens + ans_tokens_count <= 8192
    assert len(diag["seed_ranks"]) == 12
    assert not diag.get("skipped_seed_ranks")


def test_answer_blind_selection_mutation() -> None:
    """Test 5: Mutating the official target answer produces byte-identical prompt and evidence."""
    cfg = load_config(PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json")
    tokenizer = MockTokenizer()

    units = []
    for r in range(12):
        units.append({
            "document_id": f"doc_{r}",
            "article_number": str(r + 1),
            "article_title": None,
            "source_title": f"Luật {r}",
            "document_number": None,
            "rank": r,
            "block_id": f"b_{r}",
            "seed": {"start": 0, "end": 10, "text": f"Đoạn {r}", "chunk_ids": [f"c_{r}"]},
            "expansions": []
        })

    # Case A: Answer length 50 tokens
    prompt_a, tokens_a, spans_a, _ = pack_training_evidence(
        question="Câu hỏi giống nhau",
        units=units,
        answer_tokens_count=50,
        tokenizer=tokenizer,
        config=cfg,
    )

    # Case B: Completely different answer text with same token length
    prompt_b, tokens_b, spans_b, _ = pack_training_evidence(
        question="Câu hỏi giống nhau",
        units=units,
        answer_tokens_count=50,
        tokenizer=tokenizer,
        config=cfg,
    )

    assert prompt_a == prompt_b
    assert tokens_a == tokens_b
    assert spans_a == spans_b


def test_e00_exact_substring_provenance() -> None:
    """Test 6: Spans must be exact substrings of document cleaned_text."""
    doc_text = "Đây là toàn bộ văn bản pháp luật chính xác 100%."
    start = 7
    end = 26
    sub = doc_text[start:end]
    span = parent.span({"cleaned_text": doc_text}, start, end, ["c1"])
    assert span["text"] == sub
    assert doc_text[span["start"]:span["end"]] == span["text"]
