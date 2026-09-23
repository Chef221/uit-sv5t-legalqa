"""Numerical parity tests for the exact E45 T4 attention adapter."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from uit_dsc_fixed_rag.e45_t4_attention import (
    e45_t4_attention_forward,
    register_e45_t4_attention,
    repeat_key_value_heads,
)


def test_expanded_kv_attention_forward_and_backward_parity() -> None:
    """Explicit KV expansion must match the reference full SDPA computation."""
    torch.manual_seed(20260830)
    module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
    query = torch.randn(1, 4, 7, 8, requires_grad=True)
    key = torch.randn(1, 2, 7, 8, requires_grad=True)
    value = torch.randn(1, 2, 7, 8, requires_grad=True)

    actual, weights = e45_t4_attention_forward(
        module, query, key, value, None, dropout=0.0, scaling=8 ** -0.5
    )
    reference = torch.nn.functional.scaled_dot_product_attention(
        query,
        repeat_key_value_heads(key, 2),
        repeat_key_value_heads(value, 2),
        dropout_p=0.0,
        scale=8 ** -0.5,
        is_causal=True,
    ).transpose(1, 2).contiguous()

    assert weights is None
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    actual.sum().backward(retain_graph=True)
    actual_gradients = (query.grad.clone(), key.grad.clone(), value.grad.clone())
    query.grad = key.grad = value.grad = None
    reference.sum().backward()
    for observed, expected in zip(actual_gradients, (query.grad, key.grad, value.grad)):
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)

    # Exercise the production Transformers registry and the actual Qwen2
    # attention call, not only the standalone kernel adapter.
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    implementation = register_e45_t4_attention()
    assert register_e45_t4_attention() == implementation  # smoke and train reload safely
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    config._attn_implementation = implementation
    model = Qwen2ForCausalLM(config)
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    model(input_ids=token_ids, labels=token_ids).loss.backward()
