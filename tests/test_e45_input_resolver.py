"""Test E45 input resolver and safe system archive extraction."""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
import pytest

from uit_dsc_fixed_rag.e45_input_resolver import (
    InputResolverError,
    Sha256Index,
    compute_file_sha256,
    extract_system_archive,
    resolve_required_artifacts,
    resolve_system_archive,
)


def test_input_resolver_duplicate_basenames() -> None:
    """Test resolving artifacts when duplicate basenames exist in input directory."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dir1 = root / "official"
        dir2 = root / "splits"
        dir1.mkdir()
        dir2.mkdir()

        # Two train.json files with different content
        f1 = dir1 / "train.json"
        f1.write_text('{"official": true}', encoding="utf-8")
        sha1 = compute_file_sha256(f1)

        f2 = dir2 / "train.json"
        f2.write_text('{"split": true}', encoding="utf-8")
        sha2 = compute_file_sha256(f2)

        assert sha1 != sha2

        specs = {
            "official_train": sha1,
            "splits_train": sha2,
        }

        resolved = resolve_required_artifacts([root], specs)
        assert resolved["official_train"] == f1
        assert resolved["splits_train"] == f2


def test_input_resolver_missing_and_ambiguous() -> None:
    """Test that missing or duplicate files fail closed."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        f1 = root / "a.txt"
        f1.write_text("hello", encoding="utf-8")
        sha_hello = compute_file_sha256(f1)

        # Missing artifact
        with pytest.raises(InputResolverError, match="Missing required artifact"):
            resolve_required_artifacts([root], {"missing": "0" * 64})

        # Ambiguous artifact (two identical files with same hash)
        f2 = root / "b.txt"
        f2.write_text("hello", encoding="utf-8")
        with pytest.raises(InputResolverError, match="Ambiguous artifact match"):
            resolve_required_artifacts([root], {"dup": sha_hello})


def test_system_archive_resolution_and_extraction() -> None:
    """Test resolving system archive by hash and safe extraction enforcing directory layout."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bin_path = root / "E45_TWO_ACCOUNT_SYSTEM_TEST.bin"
        sidecar_path = root / "E45_TWO_ACCOUNT_SYSTEM_TEST.bin.sha256"

        with zipfile.ZipFile(bin_path, "w") as zf:
            zf.writestr("project/src/uit_dsc_fixed_rag/__init__.py", "# init\n")
            zf.writestr("project/configs/e45-inference-aligned-parent-lora-v1.json", '{"schema_version": "1.0"}\n')

        bin_sha = compute_file_sha256(bin_path)
        sidecar_path.write_text(f"{bin_sha}  {bin_path.name}\n", encoding="utf-8")

        res_bin, res_sidecar = resolve_system_archive([root], bin_sha)
        assert res_bin == bin_path
        assert res_sidecar == sidecar_path

        # Test extraction
        extract_dest = root / "working/project"
        extract_system_archive(bin_path, extract_dest)

        # Invariant: src and configs must exist directly under extract_dest
        assert (extract_dest / "src/uit_dsc_fixed_rag/__init__.py").is_file()
        assert (extract_dest / "configs/e45-inference-aligned-parent-lora-v1.json").is_file()
        assert not (extract_dest / "project/src").is_dir()
