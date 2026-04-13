import pytest
import torch

from gqa_decode import gqa_decode_attention_persistent, reference_gqa_decode


TEST_CONFIGS = [
    (32, 32, 128),  # MHA
    (32, 8, 128),  # Llama3-8B
    (64, 8, 128),  # Llama3-70B
    (28, 4, 128),  # Qwen2.5-7B
    (8, 1, 128),  # edge: group_size=8
    (16, 4, 64),  # head_dim=64
]

TEST_SEQ_LENS = [64, 512, 1024, 4096, 16384]


@pytest.mark.parametrize("num_q_heads,num_kv_heads,head_dim", TEST_CONFIGS)
@pytest.mark.parametrize("seq_len", TEST_SEQ_LENS)
def test_persistent_matches_reference(
    num_q_heads: int, num_kv_heads: int, head_dim: int, seq_len: int
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(42)
    dtype = torch.bfloat16
    q = torch.randn(num_q_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)

    ref = reference_gqa_decode(q, k, v, num_kv_heads=num_kv_heads)
    out = gqa_decode_attention_persistent(q, k, v)

    max_err = (out.float() - ref.float()).abs().max().item()
    assert max_err < 1e-2, (
        f"max_err={max_err:.4f} for ({num_q_heads},{num_kv_heads},{head_dim}) seq={seq_len}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
