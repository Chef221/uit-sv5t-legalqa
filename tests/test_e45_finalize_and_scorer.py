"""Golden scorer vectors and fail-closed archive prechecks for the E45 merger."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uit_dsc_fixed_rag.e45_finalize import FinalizeError, _official_rouge_tokens, _validate_sidecar, compute_official_rouge_l


def test_btc_rouge_ascii_tokenizer_golden_vectors() -> None:
    """BTC's vendored ROUGE tokenizer lowercases and retains ASCII alphanumerics only."""
    assert _official_rouge_tokens("Việt Nam, ĐIỀU 12-A!") == ["vi", "t", "nam", "i", "u", "12", "a"]
    assert _official_rouge_tokens("  A\tB\nC  ") == ["a", "b", "c"]
    assert _official_rouge_tokens("...!!!") == []
    assert compute_official_rouge_l("ABC 12", "abc---12") == 1.0
    assert compute_official_rouge_l("abc", "") == 0.0


def test_archive_sidecar_mismatch_rejected_before_extraction(tmp_path: Path) -> None:
    """The scorer never opens an archive whose SHA-256 sidecar is false."""
    archive = tmp_path / "account.bin"
    archive.write_bytes(b"not a zip")
    Path(str(archive) + ".sha256").write_text("0" * 64 + "  account.bin\n", encoding="utf-8")
    with pytest.raises(FinalizeError, match="sidecar mismatch"):
        _validate_sidecar(archive)
