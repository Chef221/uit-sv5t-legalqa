"""Test 15: Executable CPU notebook-orchestration test detecting nested archive roots and duplicate basenames."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uit_dsc_fixed_rag.e45_input_resolver import (
    compute_file_sha256,
    extract_system_archive,
    resolve_required_artifacts,
    resolve_system_archive,
)


def test_notebook_orchestration_extraction_and_disambiguation(tmp_path: Path) -> None:
    """Execute real notebook cell logic against duplicate basenames and nested archive roots."""
    input_dir = tmp_path / "kaggle/input"
    working_dir = tmp_path / "kaggle/working"
    project_root = working_dir / "project"
    input_dir.mkdir(parents=True)
    working_dir.mkdir(parents=True)

    # 1. Create a mock system archive with nested "project/" prefix
    bin_path = input_dir / "E45_TWO_ACCOUNT_SYSTEM_TEST.bin"
    with zipfile.ZipFile(bin_path, "w") as zf:
        zf.writestr("project/src/uit_dsc_fixed_rag/__init__.py", "# init\n")
        zf.writestr("project/src/mock_extracted_pkg/__init__.py", "MOCK_VAL = 42\n")
        zf.writestr("project/configs/e45-inference-aligned-parent-lora-v1.json", '{"schema_version": "1.0"}\n')

    bin_sha = compute_file_sha256(bin_path)
    sidecar_path = input_dir / "E45_TWO_ACCOUNT_SYSTEM_TEST.bin.sha256"
    sidecar_path.write_text(f"{bin_sha}  {bin_path.name}\n", encoding="utf-8")

    # 2. Attach duplicate basenames (e.g. official train.json vs split train.json)
    dir_official = input_dir / "official_data"
    dir_splits = input_dir / "split_data"
    dir_official.mkdir()
    dir_splits.mkdir()

    official_train = dir_official / "train.json"
    official_train.write_text('{"official_data": [1, 2, 3]}', encoding="utf-8")
    official_sha = compute_file_sha256(official_train)

    split_train = dir_splits / "train.json"
    split_train.write_text('{"split_data": [4, 5, 6]}', encoding="utf-8")
    split_sha = compute_file_sha256(split_train)

    assert official_train.name == split_train.name == "train.json"
    assert official_sha != split_sha

    # 3. Test Cell 2 extraction logic from real notebook
    # Verify resolve_system_archive finds and validates the bin archive
    resolved_bin, resolved_sidecar = resolve_system_archive([input_dir], bin_sha)
    assert resolved_bin == bin_path
    assert resolved_sidecar == sidecar_path

    # Extract stripping project/ prefix
    extract_system_archive(resolved_bin, project_root)

    # Crucial assertion: no nested project/project/src directory!
    assert (project_root / "src/uit_dsc_fixed_rag/__init__.py").is_file()
    assert (project_root / "src/mock_extracted_pkg/__init__.py").is_file()
    assert (project_root / "configs/e45-inference-aligned-parent-lora-v1.json").is_file()
    assert not (project_root / "project/src").is_dir(), "Nested project/project/src directory detected!"

    # 4. Test Cell 3 input disambiguation by SHA-256 (no basename collisions)
    required_specs = {
        "official_train": official_sha,
        "splits_train": split_sha,
    }
    resolved_inputs = resolve_required_artifacts([input_dir], required_specs)
    assert resolved_inputs["official_train"] == official_train
    assert resolved_inputs["splits_train"] == split_train
    assert resolved_inputs["official_train"] != resolved_inputs["splits_train"]

    # 5. Verify sys.path addition allows importing without error
    sys.path.insert(0, str(project_root / "src"))
    import mock_extracted_pkg
    assert mock_extracted_pkg.MOCK_VAL == 42
