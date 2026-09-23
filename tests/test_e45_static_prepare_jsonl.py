"""Regression tests for the E45 static-preparation JSONL boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_static_prepare_module():
    script = PROJECT_ROOT / "scripts/run_e45_static_prepare.py"
    spec = importlib.util.spec_from_file_location("e45_static_prepare_test_module", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jsonl_reader_preserves_u0085_inside_answer(tmp_path: Path) -> None:
    """A valid U+0085 inside JSON text must not become a record boundary."""
    records = [
        {"question_id": "q1", "answer": "trước\u0085sau"},
        {"question_id": "q2", "answer": "bình thường"},
    ]
    path = tmp_path / "records.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )

    module = _load_static_prepare_module()
    assert module.read_jsonl_records(path) == records
