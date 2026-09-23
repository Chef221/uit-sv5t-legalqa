"""Exact Qwen2 GQA SDPA adapter that avoids the T4 math-backend fallback."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any


ATTENTION_IMPLEMENTATION = "e45_t4_expanded_kv_sdpa"


class E45AttentionError(RuntimeError):
    """Raised when the frozen memory-efficient attention path is unavailable."""


def repeat_key_value_heads(tensor: Any, groups: int) -> Any:
    """Expand KV heads exactly as Qwen2 eager attention does."""
    if groups == 1:
        return tensor
    batch, key_value_heads, sequence, head_dim = tensor.shape
    return (
        tensor[:, :, None, :, :]
        .expand(batch, key_value_heads, groups, sequence, head_dim)
        .reshape(batch, key_value_heads * groups, sequence, head_dim)
    )


def e45_t4_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Run full attention after explicit KV expansion, excluding math SDPA on CUDA.

    PyTorch's native GQA route can select the quadratic-memory math backend on
    Turing GPUs.  Explicit expansion is algebraically identical to Qwen2 eager
    attention and makes the fused efficient backend eligible.
    """
    import torch

    if kwargs.get("output_attentions", False):
        raise E45AttentionError("E45 training does not permit output_attentions=True")
    groups = int(getattr(module, "num_key_value_groups", 1))
    key = repeat_key_value_heads(key, groups)
    value = repeat_key_value_heads(value, groups)
    causal = is_causal if is_causal is not None else bool(getattr(module, "is_causal", True))
    causal = bool(query.shape[2] > 1 and attention_mask is None and causal)

    context = nullcontext()
    if query.is_cuda:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backends = [SDPBackend.EFFICIENT_ATTENTION]
        if hasattr(SDPBackend, "CUDNN_ATTENTION"):
            backends.append(SDPBackend.CUDNN_ATTENTION)
        # FLASH_ATTENTION is harmless in the allowlist but is not expected on
        # T4.  Math is deliberately absent so an unsupported fused path fails
        # explicitly instead of consuming quadratic memory.
        if hasattr(SDPBackend, "FLASH_ATTENTION"):
            backends.append(SDPBackend.FLASH_ATTENTION)
        context = sdpa_kernel(backends)

    with context:
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=causal,
        )
    return output.transpose(1, 2).contiguous(), None


def register_e45_t4_attention() -> str:
    """Register the E45 backend and matching canonical SDPA mask builder."""
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask

    AttentionInterface.register(ATTENTION_IMPLEMENTATION, e45_t4_attention_forward)
    AttentionMaskInterface.register(ATTENTION_IMPLEMENTATION, sdpa_mask)
    return ATTENTION_IMPLEMENTATION
