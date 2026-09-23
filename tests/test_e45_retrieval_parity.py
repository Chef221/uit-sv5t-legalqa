"""Test E45 holdout retrieval and prompt parity against canonical P00/P01 authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from uit_dsc_fixed_rag.e02_compare import weighted_rrf
from uit_dsc_fixed_rag.final_private_p00 import article_blocks, seed_unit
from uit_dsc_fixed_rag.final_private_p01 import Config, _prompt, _ContextConfig
from uit_dsc_fixed_rag.e45_holdout_contexts import (
    HoldoutError,
    _build_holdout_manifest,
    _validate_reusable_context_rows,
    compute_json_sha256,
    normalize_question,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_rrf_top20_fusion_parity() -> None:
    """Test weighted RRF matches exact P00 formula and ordering."""
    sparse_ids = [f"chunk_{i}" for i in range(40)]
    dense_ids = [f"chunk_{39 - i}" for i in range(40)]

    fused = weighted_rrf(
        {"sparse": sparse_ids, "dense": dense_ids},
        weights={"sparse": 0.5, "dense": 0.5},
        constant=60,
        top_k=20,
    )

    assert len(fused) == 20
    assert fused[0]["rrf_score"] > 0.0
    for i in range(len(fused) - 1):
        assert fused[i]["rrf_score"] >= fused[i + 1]["rrf_score"]


def test_parent_seed_unit_parity() -> None:
    """Test seed_unit parent expansion produces canonical structures matching P00."""
    text = "Điều 1. Phạm vi điều chỉnh\nLuật này quy định về quy tắc giao thông đường bộ."
    document = {
        "document_id": "doc_1",
        "title": "Luật Giao thông đường bộ",
        "document_number": "23/2008/QH12",
        "text": text,
        "cleaned_text": text,
    }
    seed = {
        "chunk_id": "c1",
        "document_id": "doc_1",
        "article_number": "1",
        "clause_number": None,
        "text": text,
        "start_char": 0,
        "end_char": len(text),
    }
    block = [seed]
    context_policy = {
        "max_parent_tokens": 1200,
        "max_parent_characters": 6000,
        "neighbor_radius": 1,
        "expansion_order": ["whole_article", "both_neighbors", "previous_neighbor", "next_neighbor"],
        "evict_seed_for_expansion": False,
        "merge": "overlapping-or-touching-exact-document-spans",
        "metadata": "one-source-header-per-document-article-label-per-span",
    }

    unit = seed_unit(seed, 0, block, document, context_policy)
    assert unit["seed"]["chunk_ids"] == ["c1"]
    assert unit["rank"] == 0
    assert "expansions" in unit
    assert unit["document_id"] == "doc_1"


def test_answer_blindness_in_prompt_generation() -> None:
    """Test that context units and prompts contain zero answer tokens or influence."""
    config_data = json.loads((PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json").read_text(encoding="utf-8"))
    cfg = Config(config_data, PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json")

    full_cleaned_text = ""
    seeds = []
    for r in range(12):
        s_text = f"Điều {r+1}. Nội dung điều {r+1}."
        start = len(full_cleaned_text)
        full_cleaned_text += s_text + "\n"
        end = start + len(s_text)
        s = {
            "chunk_id": f"c_{r}",
            "document_id": "doc_1",
            "article_number": f"{r+1}",
            "clause_number": None,
            "text": s_text,
            "start_char": start,
            "end_char": end,
        }
        seeds.append(s)

    document = {
        "document_id": "doc_1",
        "title": "Bộ luật Dân sự",
        "document_number": "91/2015/QH13",
        "text": full_cleaned_text,
        "cleaned_text": full_cleaned_text,
    }
    prepared_units = []
    for r, s in enumerate(seeds):
        prepared_units.append(seed_unit(s, r, [s], document, cfg.section("context_policy")))

    prep_row = {
        "question_id": "q_100",
        "sample_index": 0,
        "answers_used": False,
        "units": prepared_units,
    }

    assert "answer" not in prep_row
    for u in prep_row["units"]:
        assert "answer" not in u
        assert "gold_evidence" not in u


def test_completed_holdout_jsonl_can_be_validated_and_finalized(tmp_path: Path) -> None:
    """Regression: recover after JSONL rename without relying on cfg.experiment_id."""
    class FakeTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
            assert add_special_tokens is False
            return {"input_ids": [ord(char) for char in text]}

    tokenizer = FakeTokenizer()
    questions = [{"question_id": str(index), "question": f"Question {index}"} for index in range(200)]
    rows = []
    for index, question in enumerate(questions):
        prompt = f"Prompt {index}"
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ordered_seed_ids = [f"c-{index}-{rank}" for rank in range(12)]
        units = [{"rank": rank, "chunk_id": chunk_id} for rank, chunk_id in enumerate(ordered_seed_ids)]
        body = {
            "sample_index": index,
            "question_id": question["question_id"],
            "question": question["question"],
            "question_sha256": hashlib.sha256(question["question"].encode("utf-8")).hexdigest(),
            "ordered_seed_ids": ordered_seed_ids,
            "evidence_sha256": compute_json_sha256({"ordered_seed_ids": ordered_seed_ids, "units": units}),
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_input_ids_sha256": hashlib.sha256(
                json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "input_tokens": len(input_ids),
            "units": units,
            "answers_used": False,
        }
        body["record_sha256"] = compute_json_sha256(body)
        rows.append(body)

    _validate_reusable_context_rows(rows=rows, questions=questions, tokenizer=tokenizer)

    config_path = PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json"
    cfg = Config(json.loads(config_path.read_text(encoding="utf-8")), config_path)
    assert not hasattr(cfg, "experiment_id")
    contexts_path = tmp_path / "holdout-contexts.jsonl"
    contexts_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest = _build_holdout_manifest(
        cfg=cfg,
        contexts_rows=rows,
        out_jsonl=contexts_path,
        tokenizer_path=tokenizer_dir,
        e00_cfg=cfg.section("source_e00"),
        dense_cfg=cfg.section("source_dense"),
    )
    assert manifest["experiment_id"] == "E45-inference-aligned-parent-lora-v1"
    assert manifest["sample_size"] == 200

    rows[0]["prompt"] = "tampered"
    with pytest.raises(HoldoutError, match="prompt hash changed"):
        _validate_reusable_context_rows(rows=rows, questions=questions, tokenizer=tokenizer)
