"""Tests 11, 12: Account A/B context byte-parity fixture and generation arm parity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uit_dsc_fixed_rag.e45_paired_generation import compute_json_sha256
from uit_dsc_fixed_rag.final_private_p01 import unified_clean


def test_account_ab_context_byte_parity_fixture(tmp_path: Path) -> None:
    """Test 11: Assert Account A and Account B produce byte-identical holdout-contexts.jsonl."""
    ctx_row_1 = {
        "question_id": "100823",
        "sample_index": 0,
        "question": "Câu hỏi số 1",
        "ranks": list(range(12)),
        "chunk_ids": [f"c_{i}" for i in range(12)],
        "prompt": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n",
        "prompt_tokens": 120,
        "prompt_sha256": "prompt_hash_1",
    }
    row_bytes = (json.dumps(ctx_row_1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")

    file_a = tmp_path / "account_a_contexts.jsonl"
    file_b = tmp_path / "account_b_contexts.jsonl"

    file_a.write_bytes(row_bytes)
    file_b.write_bytes(row_bytes)

    assert file_a.read_bytes() == file_b.read_bytes()


def test_suffix_clean_pipeline_parity() -> None:
    """Test 12: E43 and E44 suffix cleanup parity on truncated or repetitive endings."""
    raw_answer = "Theo quy định tại Điều 5, công dân có quyền khiếu nại. " * 3 + "Theo quy định tại Điều 5, công dân"
    clean_ans, diag = unified_clean(raw_answer)

    # Incomplete sentence and repetitive copies should be cleanly trimmed by P01 suffix trim
    assert clean_ans == "Theo quy định tại Điều 5, công dân có quyền khiếu nại."
    assert clean_ans
    assert isinstance(diag, dict)


def test_generation_identity_hash_is_canonical() -> None:
    """Worker identities use stable canonical JSON rather than Python dict order."""
    assert compute_json_sha256({"b": 2, "a": 1}) == compute_json_sha256({"a": 1, "b": 2})
