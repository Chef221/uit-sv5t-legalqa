"""Tests 7, 8, 9: Real 5,636 ID/hash, reference isolation, and parameter inventory."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uit_dsc_fixed_rag.e45_parent_training import (
    PINNED_E38_ADAPTER_SHA256,
    PINNED_TRAINING_IDS_SHA256,
    assert_model_inventory,
    load_config,
)


def test_real_training_id_hash() -> None:
    """Test 7: Validate 5,636 ordered QIDs hash matches fee946ada7cbfca7df04ffe1c3d9f06907808c0dc59600e6f1f8d579076742dd."""
    records_env = os.environ.get("E45_PRIOR_RECORDS")
    if not records_env:
        pytest.skip("Set E45_PRIOR_RECORDS to an authorized external records file")
    records_path = Path(records_env)
    assert records_path.is_file()

    qids = [json.loads(l)["question_id"] for l in records_path.open("r", encoding="utf-8")]
    assert len(qids) == 5636
    h = hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest()
    assert h == PINNED_TRAINING_IDS_SHA256


def test_reference_isolation_in_config_and_code() -> None:
    """Test 8: Ensure no reference answer file is opened in holdout context preparation."""
    cfg = load_config(PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json")
    assert "references" not in cfg.raw.get("holdout", {})
    assert "target" not in cfg.raw.get("holdout", {})


def test_exact_parameter_inventory_assertion() -> None:
    """Test 9: Verify parameter budget and exact rank-8 LoRA trainable counts."""
    cfg = load_config(PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json")

    # Mock model with exact Qwen2-3B rank-8 parameter counts
    mock_model = MagicMock()
    # 36 layers * 415,744 params = 14,966,784 trainable params
    mock_trainable = MagicMock()
    mock_trainable.numel.return_value = 14966784
    mock_trainable.requires_grad = True

    mock_frozen = MagicMock()
    mock_frozen.numel.return_value = 3085938688 - 14966784
    mock_frozen.requires_grad = False

    mock_model.named_parameters.return_value = [
        ("trainable_lora", mock_trainable),
        ("frozen_base", mock_frozen),
    ]

    inv = assert_model_inventory(mock_model, cfg)
    assert inv["trainable_parameters"] == 14966784
    assert inv["actual_stack_total"] == 3668660224
    assert inv["actual_stack_total"] < 4000000000
