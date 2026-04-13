import pytest
import torch

from gqa_decode import gqa_decode_attention, reference_gqa_decode
from gqa_decode.gqa_decode import gqa_decode_attention_fp8
from gqa_decode.fp8_utils import quantize_kv_fp8, dequantize_kv_fp8


TEST_CONFIGS = [
    (32, 32, 128),  # MHA
    (32, 8, 128),  # Llama3-8B
    (28, 4, 128),  # Qwen2.5-7B
    (64, 8, 128),  # Llama3-70B
]

TEST_SEQ_LENS = [1024, 4096, 16384]


@pytest.mark.parametrize("num_q_heads,num_kv_heads,head_dim", TEST_CONFIGS)
@pytest.mark.parametrize("seq_len", TEST_SEQ_LENS)
def test_fp8_correctness(
    num_q_heads: int, num_kv_heads: int, head_dim: int, seq_len: int
):
    """FP8 kernel output should match dequantized BF16 reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(42)
    q = torch.randn(num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16
    )

    k_fp8, v_fp8, k_scale, v_scale = quantize_kv_fp8(k, v)

    # Reference: dequantize then run BF16 reference
    k_deq, v_deq = dequantize_kv_fp8(k_fp8, v_fp8, k_scale, v_scale)
    ref = reference_gqa_decode(q, k_deq, v_deq, num_kv_heads=num_kv_heads)

    out = gqa_decode_attention_fp8(q, k_fp8, v_fp8, k_scale, v_scale)
    max_err = (ref.float() - out.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), out.float().flatten(), dim=0
    ).item()

    assert max_err < 0.05, (
        f"FP8 max_err={max_err:.4f} for "
        f"({num_q_heads},{num_kv_heads},{head_dim}) seq={seq_len}"
    )
    assert cos_sim > 0.99, (
        f"FP8 cosine_sim={cos_sim:.6f} for "
        f"({num_q_heads},{num_kv_heads},{head_dim}) seq={seq_len}"
    )


@pytest.mark.parametrize("num_q_heads,num_kv_heads,head_dim", TEST_CONFIGS[:2])
@pytest.mark.parametrize("seq_len", [1024, 4096])
def test_fp8_python_reduction_matches_fused(
    num_q_heads: int, num_kv_heads: int, head_dim: int, seq_len: int
):
    """Fused and Python reductions should give the same FP8 result."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(123)
    q = torch.randn(num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16
    )

    k_fp8, v_fp8, k_scale, v_scale = quantize_kv_fp8(k, v)

    out_fused = gqa_decode_attention_fp8(
        q, k_fp8, v_fp8, k_scale, v_scale, reduction="fused"
    )
    out_python = gqa_decode_attention_fp8(
        q, k_fp8, v_fp8, k_scale, v_scale, reduction="python"
    )

    max_err = (out_fused.float() - out_python.float()).abs().max().item()
    assert max_err < 1e-2, f"FP8 fused vs python reduction max_err={max_err}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
