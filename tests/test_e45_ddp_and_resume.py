"""Test E45 DDP world size contract enforcement and atomic multi-rank resume restoration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest
import torch

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e45_checkpoint import (
    CheckpointError,
    create_checkpoint_manifest,
    package_checkpoint_bin,
    validate_checkpoint_resume,
    tokenizer_aggregate_sha256,
    tokenizer_file_hashes,
)
from uit_dsc_fixed_rag.e45_parent_training import E45Error

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ddp_world_size_enforcement() -> None:
    """Test that training rejects single-process execution when WORLD_SIZE != 2."""
    script_path = PROJECT_ROOT / "scripts/run_e45_train_kaggle.py"
    env = os.environ.copy()
    env["WORLD_SIZE"] = "1"

    # Running with world_size=1 must fail with world size contract error
    proc = subprocess.run(
        [sys.executable, str(script_path), "--records-jsonl", "dummy.jsonl", "--manifest-json", "dummy.json",
         "--base-model-path", "dummy", "--tokenizer-path", "dummy", "--output-dir", "dummy"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Live world size 1 != expected 2" in proc.stderr or "Live world size 1 != expected 2" in proc.stdout


def test_miniature_ddp_checkpoint_and_resume() -> None:
    """Test full atomic checkpoint creation, packaging, and strict multi-rank resume validation."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp) / "checkpoint-32"
        ckpt_dir.mkdir()

        # Create dummy weights and tensors
        (ckpt_dir / "adapter_model.safetensors").write_bytes(b"ADAPTER_TENSORS")
        (ckpt_dir / "adapter_config.json").write_text('{"r": 8, "lora_alpha": 16}\n', encoding="utf-8")
        torch.save({"opt": 1}, ckpt_dir / "optimizer.pt")
        torch.save({"sched": 1}, ckpt_dir / "scheduler.pt")
        (ckpt_dir / "trainer_state.json").write_text(json.dumps({"global_step": 32, "epoch": 0.045}), encoding="utf-8")

        # Per-rank RNG states for world_size=2
        torch.save({"rank_0_rng": 123}, ckpt_dir / "rng_state_0.pth")
        torch.save({"rank_1_rng": 456}, ckpt_dir / "rng_state_1.pth")

        # Tokenizer files
        (ckpt_dir / "tokenizer.json").write_text('{"version": "1.0"}\n', encoding="utf-8")
        (ckpt_dir / "tokenizer_config.json").write_text('{"model_type": "qwen2"}\n', encoding="utf-8")

        expected_identity = {
            "experiment_id": "E45-inference-aligned-parent-lora-v1",
            "config_sha256": "0" * 64,
            "code_manifest_sha256": "1" * 64,
            "training_records_jsonl_sha256": "2" * 64,
            "aggregate_records_sha256": "3" * 64,
            "base_model": "AITeamVN/Vi-Qwen2-3B-RAG",
            "base_revision": "eaf427c24d86066a2b35828c499b7db3af321227",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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

        # Create manifest
        manifest = create_checkpoint_manifest(ckpt_dir, expected_identity)
        assert manifest["global_step"] == 32
        assert "rng_state_0.pth" in manifest["rng_states"]
        assert "rng_state_1.pth" in manifest["rng_states"]
        assert "tokenizer.json" in manifest["tokenizer_files"]

        # Validate resume
        validated = validate_checkpoint_resume(ckpt_dir, expected_identity)
        assert validated["global_step"] == 32

        # Package into .bin archive and test extraction
        bin_path = Path(tmp) / "E45_ACCOUNT_A_CHECKPOINT_STEP_32.bin"
        package_checkpoint_bin(ckpt_dir, bin_path)
        assert bin_path.is_file()
        assert Path(str(bin_path) + ".sha256").is_file()

        # Mutation 1: altered RNG state for rank 1
        torch.save({"corrupted": True}, ckpt_dir / "rng_state_1.pth")
        with pytest.raises(CheckpointError, match="RNG state hash altered"):
            validate_checkpoint_resume(ckpt_dir, expected_identity)
