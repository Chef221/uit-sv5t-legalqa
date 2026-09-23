"""Regression coverage for the real-row E45 memory-smoke path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from uit_dsc_fixed_rag.e45_parent_training import E45CausalCollator, E45TokenizedDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_training_module():
    script = PROJECT_ROOT / "scripts/run_e45_train_kaggle.py"
    spec = importlib.util.spec_from_file_location("e45_train_smoke_test_module", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2


class _TinyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask, labels, logits_to_keep, shift_labels):
        assert self.training, "smoke must activate the training/checkpointing path"
        assert torch.equal(shift_labels, labels[:, logits_to_keep + 1])
        loss = self.scale * input_ids.float().sum()
        return SimpleNamespace(loss=loss)


def test_smoke_uses_model_input_length_and_complete_projection() -> None:
    """Dataset items omit provenance fields; smoke must still select the longest row."""
    records = [
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1] * 5, "labels": [-100] * 4 + [5]},
    ]
    dataset = E45TokenizedDataset(records)
    collator = E45CausalCollator(_Tokenizer())
    module = _load_training_module()

    model = _TinyCausalModel()
    model.eval()
    result = module.run_max_length_smoke(
        dataset,
        model,
        collator,
        gradient_accumulation_steps=4,
        expected_optimizer_steps=705,
        max_hours_threshold=30.0,
        projection_safety_factor=1.25,
    )

    assert result["longest_row_tokens"] == 5
    assert result["gradient_accumulation_steps"] == 4
    assert result["expected_optimizer_steps"] == 705
    assert result["projection_safety_factor"] == 1.25
    assert model.training is True


def test_selective_answer_logits_match_full_qwen_peft_loss_and_gradients() -> None:
    """Answer-only logits must exactly reproduce ignored-prompt causal loss."""
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(20260830)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        attention_dropout=0.0,
    )
    model = get_peft_model(
        Qwen2ForCausalLM(config),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], lora_dropout=0.0, task_type="CAUSAL_LM"),
    )
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    labels = torch.tensor([[-100, -100, -100, -100, 5, 6, 7]])

    full_loss = model(input_ids=input_ids, labels=labels).loss
    full_loss.backward()
    lora_parameter = next(parameter for name, parameter in model.named_parameters() if "lora_B" in name)
    full_gradient = lora_parameter.grad.clone()
    model.zero_grad(set_to_none=True)

    active = torch.nonzero(labels[0].ne(-100), as_tuple=False).flatten()
    selective_loss = model(
        input_ids=input_ids,
        labels=labels,
        logits_to_keep=active - 1,
        shift_labels=labels[:, active],
    ).loss
    selective_loss.backward()

    torch.testing.assert_close(selective_loss, full_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(lora_parameter.grad, full_gradient, rtol=1e-5, atol=1e-6)


def test_training_uses_non_reentrant_checkpointing_for_ddp() -> None:
    """The pinned T4 path must select both required exact-memory repairs."""
    source = (PROJECT_ROOT / "scripts/run_e45_train_kaggle.py").read_text(encoding="utf-8")
    assert 'gradient_checkpointing_kwargs={"use_reentrant": False}' in source
    assert "register_e45_t4_attention()" in source
    assert '"attn_implementation": attention_implementation' in source
    assert 'warmup_steps=training_cfg["warmup_ratio"]' in source
    assert 'warmup_ratio=training_cfg["warmup_ratio"]' not in source
    assert source.index("training_args = TrainingArguments(") < source.index("def fresh_lora_stack()")
