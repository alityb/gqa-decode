"""FP8 E4M3 quantization utilities for KV cache."""

from __future__ import annotations

import torch


def quantize_kv_fp8(
    k_bf16: torch.Tensor, v_bf16: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-head FP8 E4M3 quantization of KV cache.

    Args:
        k_bf16: [num_kv_heads, seq_len, head_dim] BF16
        v_bf16: [num_kv_heads, seq_len, head_dim] BF16

    Returns:
        k_fp8:   [num_kv_heads, seq_len, head_dim] FP8 E4M3
        v_fp8:   [num_kv_heads, seq_len, head_dim] FP8 E4M3
        k_scale: [num_kv_heads] FP32 — multiply to dequantize
        v_scale: [num_kv_heads] FP32 — multiply to dequantize
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0

    # One scale per KV head (simplest scheme)
    k_amax = k_bf16.float().abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)
    v_amax = v_bf16.float().abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)

    k_scale = (k_amax / fp8_max).squeeze(-1).squeeze(-1).float()  # [num_kv_heads]
    v_scale = (v_amax / fp8_max).squeeze(-1).squeeze(-1).float()  # [num_kv_heads]

    k_fp8 = (
        (k_bf16.float() / k_amax * fp8_max)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )
    v_fp8 = (
        (v_bf16.float() / v_amax * fp8_max)
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )

    return k_fp8, v_fp8, k_scale, v_scale


def dequantize_kv_fp8(
    k_fp8: torch.Tensor,
    v_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize FP8 KV cache back to BF16 for reference comparison."""
    num_kv_heads = k_fp8.shape[0]
    k_bf16 = (k_fp8.float() * k_scale.view(num_kv_heads, 1, 1)).to(torch.bfloat16)
    v_bf16 = (v_fp8.float() * v_scale.view(num_kv_heads, 1, 1)).to(torch.bfloat16)
    return k_bf16, v_bf16
