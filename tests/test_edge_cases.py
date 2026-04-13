import pytest
import torch

from gqa_decode import GQADecodeConfig, gqa_decode_attention, reference_gqa_decode


@pytest.mark.parametrize("seq_len", [1, 31, 32, 33, 63, 64, 65])
def test_block_boundaries(seq_len: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    dtype = torch.bfloat16
    q = torch.randn(32, 128, device="cuda", dtype=dtype)
    k = torch.randn(8, seq_len, 128, device="cuda", dtype=dtype)
    v = torch.randn(8, seq_len, 128, device="cuda", dtype=dtype)
    out = gqa_decode_attention(q, k, v, config=GQADecodeConfig(block_seq=32), backend="cute")
    ref = reference_gqa_decode(q, k, v)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_forced_torch_backend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    dtype = torch.bfloat16
    q = torch.randn(28, 128, device="cuda", dtype=dtype)
    k = torch.randn(4, 257, 128, device="cuda", dtype=dtype)
    v = torch.randn(4, 257, 128, device="cuda", dtype=dtype)
    out = gqa_decode_attention(q, k, v, backend="torch")
    ref = reference_gqa_decode(q, k, v)
    assert torch.equal(out, ref)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
