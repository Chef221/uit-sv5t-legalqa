"""Test 10: Checkpoint resume positive test and mutation rejection matrix."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e45_checkpoint import (
    CheckpointError,
    copy_frozen_tokenizer_files,
    create_checkpoint_manifest,
    package_checkpoint_bin,
    safe_extract_archive,
    tokenizer_aggregate_sha256,
    tokenizer_file_hashes,
    validate_checkpoint_resume,
)


def test_frozen_tokenizer_is_copied_byte_exactly_over_reserialized_files(tmp_path: Path) -> None:
    """Checkpoint creation uses frozen bytes even if serialization produced different JSON."""
    source = tmp_path / "frozen-tokenizer"
    destination = tmp_path / "checkpoint-32"
    source.mkdir()
    destination.mkdir()
    (source / "tokenizer.json").write_bytes(b'{"frozen":true}\n')
    (source / "tokenizer_config.json").write_bytes(b'{"padding_side":"right"}\n')
    (source / "chat_template.jinja").write_bytes(b"{{ messages }}\n")

    # This models the exact failure seen on Kaggle: save_pretrained rewrites
    # the same logical tokenizer to different bytes and adds a recognized file.
    (destination / "tokenizer.json").write_bytes(b'{"frozen": true}\n')
    (destination / "tokenizer_config.json").write_bytes(b'{"padding_side": "right"}\n')
    (destination / "special_tokens_map.json").write_bytes(b"{}\n")

    expected = tokenizer_file_hashes(source)
    copied = copy_frozen_tokenizer_files(source, destination, expected)

    assert copied == expected
    assert tokenizer_file_hashes(destination) == expected
    assert not (destination / "special_tokens_map.json").exists()
    for name in expected:
        assert (destination / name).read_bytes() == (source / name).read_bytes()


def _make_dummy_checkpoint(ckpt_dir: Path) -> dict[str, str]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Create required files
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"ADAPTER_BYTES_123")
    (ckpt_dir / "adapter_config.json").write_text('{"r": 8, "lora_alpha": 16}', encoding="utf-8")
    (ckpt_dir / "optimizer.pt").write_bytes(b"OPT_STATE_BYTES")
    (ckpt_dir / "scheduler.pt").write_bytes(b"SCHED_STATE_BYTES")
    (ckpt_dir / "trainer_state.json").write_text('{"global_step": 32, "epoch": 0.045}', encoding="utf-8")
    (ckpt_dir / "rng_state_0.pth").write_bytes(b"RNG_0")
    (ckpt_dir / "rng_state_1.pth").write_bytes(b"RNG_1")
    (ckpt_dir / "tokenizer.json").write_text('{"vocab": {}}', encoding="utf-8")

    expected_identity = {
        "experiment_id": "E45-inference-aligned-parent-lora-v1",
        "config_sha256": "conf_sha_1",
        "code_manifest_sha256": "code_sha_1",
        "training_records_jsonl_sha256": "train_jsonl_1",
        "aggregate_records_sha256": "agg_rec_1",
        "base_model": "AITeamVN/Vi-Qwen2-3B-RAG",
        "base_revision": "eaf427c24d86066a2b35828c499b7db3af321227",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj"],
        "expected_trainable_parameters": 14966784,
        "world_size": 2,
        "per_device_train_batch": 1,
        "gradient_accumulation": 4,
        "seed": 20260830,
        "maximum_total_sequence": 8192,
        "expected_total_steps": 705,
        "tokenizer_sha256": file_sha256(ckpt_dir / "tokenizer.json"),
        "tokenizer_aggregate_sha256": tokenizer_aggregate_sha256(tokenizer_file_hashes(ckpt_dir)),
        "runtime_versions": {"torch": "2.10.0+cu128"},
    }
    return expected_identity


def test_checkpoint_positive_manifest_and_resume(tmp_path: Path) -> None:
    """Positive test: valid checkpoint manifest creation, verification, packaging, and extraction."""
    ckpt_dir = tmp_path / "checkpoint-32"
    expected = _make_dummy_checkpoint(ckpt_dir)

    manifest = create_checkpoint_manifest(ckpt_dir, expected)
    assert manifest["global_step"] == 32
    assert (ckpt_dir / "checkpoint-manifest.json").is_file()

    # Verify resume succeeds
    validated = validate_checkpoint_resume(ckpt_dir, expected)
    assert validated["global_step"] == 32

    # Test packaging
    bin_path = tmp_path / "E45_ACCOUNT_A_CHECKPOINT_STEP_32.bin"
    packaged_bin, sha = package_checkpoint_bin(ckpt_dir, bin_path, {"step": 32})
    assert packaged_bin.is_file()
    assert (tmp_path / "E45_ACCOUNT_A_CHECKPOINT_STEP_32.bin.sha256").is_file()

    # Test extraction
    extract_dir = tmp_path / "extracted"
    members = safe_extract_archive(bin_path, extract_dir)
    assert "checkpoint/adapter_model.safetensors" in members


def test_checkpoint_mutation_matrix(tmp_path: Path) -> None:
    """Mutation tests: reject modified config, code, data, model, LoRA, world size, step, optimizer, RNG, missing and extra files."""
    ckpt_dir = tmp_path / "ckpt_mut"
    expected = _make_dummy_checkpoint(ckpt_dir)
    create_checkpoint_manifest(ckpt_dir, expected)

    # 1. Mutate config_sha256 in expected identity
    bad_expected = copy.deepcopy(expected)
    bad_expected["config_sha256"] = "MUTATED_CONFIG"
    with pytest.raises(CheckpointError):
        validate_checkpoint_resume(ckpt_dir, bad_expected)

    # 2. Mutate code_manifest_sha256
    bad_expected = copy.deepcopy(expected)
    bad_expected["code_manifest_sha256"] = "MUTATED_CODE"
    with pytest.raises(CheckpointError):
        validate_checkpoint_resume(ckpt_dir, bad_expected)

    # 3. Mutate training records hash
    bad_expected = copy.deepcopy(expected)
    bad_expected["training_records_jsonl_sha256"] = "MUTATED_DATA"
    with pytest.raises(CheckpointError):
        validate_checkpoint_resume(ckpt_dir, bad_expected)

    # 4. Mutate base model revision
    bad_expected = copy.deepcopy(expected)
    bad_expected["base_revision"] = "MUTATED_REV"
    with pytest.raises(CheckpointError):
        validate_checkpoint_resume(ckpt_dir, bad_expected)

    # 5. Mutate LoRA rank
    bad_expected = copy.deepcopy(expected)
    bad_expected["lora_rank"] = 16
    with pytest.raises(CheckpointError):
        validate_checkpoint_resume(ckpt_dir, bad_expected)

    # 6. Mutate optimizer file on disk
    (ckpt_dir / "optimizer.pt").write_bytes(b"CORRUPTED_OPTIMIZER")
    with pytest.raises(CheckpointError):
        validate_checkpoint_resume(ckpt_dir, expected)
    (ckpt_dir / "optimizer.pt").write_bytes(b"OPT_STATE_BYTES")  # restore

    # 7. Mutate RNG state rank 1
    (ckpt_dir / "rng_state_1.pth").write_bytes(b"CORRUPTED_RNG")
    with pytest.raises(CheckpointError):
        validate_checkpoint_resume(ckpt_dir, expected)
    (ckpt_dir / "rng_state_1.pth").write_bytes(b"RNG_1")  # restore

    # 8. Missing mandatory file
    (ckpt_dir / "scheduler.pt").unlink()
    with pytest.raises(CheckpointError):
        validate_checkpoint_resume(ckpt_dir, expected)
    (ckpt_dir / "scheduler.pt").write_bytes(b"SCHED_STATE_BYTES")  # restore

    # 9. Unexpected extra file
    (ckpt_dir / "unexpected_file.txt").write_text("rogue", encoding="utf-8")
    with pytest.raises(CheckpointError):
        validate_checkpoint_resume(ckpt_dir, expected)
    (ckpt_dir / "unexpected_file.txt").unlink()
