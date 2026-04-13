from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import lru_cache

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from gqa_decode.cute_dsl_utils import make_fake_tensor, torch2cute_dtype_map


CUDA_STREAM = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
LOG2_E = math.log2(math.e)
HBM_PEAK_BW = 3.35e12
DEFAULT_BLOCK_SEQ = 32
DEFAULT_TARGET_BLOCKS_PER_SM = 10
NUM_SMS_H100_SXM = 132
SMEM_PAD = 8


@dataclass(frozen=True)
class GQADecodeConfig:
    block_seq: int = DEFAULT_BLOCK_SEQ
    smem_pad: int = SMEM_PAD
    target_blocks_per_sm: int = DEFAULT_TARGET_BLOCKS_PER_SM


def reference_gqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_kv_heads: int | None = None,
) -> torch.Tensor:
    if q.dim() != 2 or k_cache.dim() != 3 or v_cache.dim() != 3:
        raise ValueError("Expected q=(num_q_heads, head_dim), k/v=(num_kv_heads, seq_len, head_dim)")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes")
    num_q_heads, head_dim = q.shape
    kv_heads, seq_len, kv_head_dim = k_cache.shape
    if kv_head_dim != head_dim:
        raise ValueError("Q/K/V head_dim mismatch")
    if num_kv_heads is None:
        num_kv_heads = kv_heads
    if num_kv_heads != kv_heads:
        raise ValueError("num_kv_heads must match k_cache.shape[0]")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    group_size = num_q_heads // num_kv_heads
    kv_head_index = torch.arange(num_q_heads, device=q.device) // group_size
    scale = head_dim**-0.5
    gathered_k = k_cache.float()[kv_head_index]
    scores = torch.einsum("hd,hsd->hs", q.float(), gathered_k) * scale
    weights = torch.softmax(scores, dim=-1)
    gathered_v = v_cache.float()[kv_head_index]
    out = torch.einsum("hs,hsd->hd", weights, gathered_v)
    return out.to(dtype=q.dtype)


def select_num_splits(
    num_kv_heads: int,
    seq_len: int,
    block_seq: int = DEFAULT_BLOCK_SEQ,
    target_blocks_per_sm: int = DEFAULT_TARGET_BLOCKS_PER_SM,
    num_sms: int = NUM_SMS_H100_SXM,
) -> int:
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    min_splits = max(1, math.ceil(num_sms * target_blocks_per_sm / num_kv_heads))
    max_tiles = max(1, math.ceil(seq_len / block_seq))
    return min(min_splits, max_tiles)


def _reduce_partial_results(
    partial_out: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    max_per_head = partial_max.max(dim=0).values
    safe_max = torch.where(torch.isfinite(max_per_head), max_per_head, torch.zeros_like(max_per_head))
    rescale = torch.exp2((partial_max - safe_max.unsqueeze(0)) * LOG2_E)
    denom = (partial_sum * rescale).sum(dim=0)
    numer = (partial_out * rescale[:, :, None]).sum(dim=0)
    out = numer / denom.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    zero_mask = denom == 0
    if zero_mask.any():
        out[zero_mask] = 0
    return out.to(out_dtype)


class DecodePartialKernel:
    def __init__(self, dtype, head_dim: int, group_size: int, block_seq: int, smem_pad: int):
        if head_dim not in (64, 128):
            raise ValueError(f"Unsupported head_dim={head_dim}; expected 64 or 128")
        if head_dim % cute.arch.WARP_SIZE != 0:
            raise ValueError("head_dim must be divisible by warp size")
        if group_size < 1 or group_size > 8:
            raise ValueError("group_size must be in [1, 8]")
        self.dtype = dtype
        self.head_dim = head_dim
        self.group_size = group_size
        self.block_seq = block_seq
        self.smem_pad = smem_pad
        self.frag_elems = head_dim // cute.arch.WARP_SIZE
        self.scale = float(head_dim**-0.5)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPartialO: cute.Tensor,
        mPartialMax: cute.Tensor,
        mPartialSum: cute.Tensor,
        stream=None,
    ):
        assert mQ.element_type == self.dtype
        assert mK.element_type == self.dtype
        assert mV.element_type == self.dtype
        assert mPartialO.element_type == Float32
        assert mPartialMax.element_type == Float32
        assert mPartialSum.element_type == Float32
        self.kernel(mQ, mK, mV, mPartialO, mPartialMax, mPartialSum).launch(
            grid=[mK.shape[0], mPartialO.shape[0], 1],
            block=[cute.arch.WARP_SIZE, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPartialO: cute.Tensor,
        mPartialMax: cute.Tensor,
        mPartialSum: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        kv_head, split_idx, _ = cute.arch.block_idx()
        frag_base = tidx * self.frag_elems
        seq_len = mK.shape[1]
        num_splits = mPartialO.shape[0]
        split_stride = num_splits * self.block_seq
        q_head_base = kv_head * self.group_size
        gPartialO = mPartialO[split_idx, None, None]
        gPartialMax = mPartialMax[split_idx, None]
        gPartialSum = mPartialSum[split_idx, None]

        tile_start = split_idx * self.block_seq
        for tile_start in cutlass.range(tile_start, seq_len, split_stride, unroll=1):
            for row in cutlass.range_constexpr(self.block_seq):
                seq_idx = tile_start + row
                if seq_idx < seq_len:
                    for h in cutlass.range_constexpr(self.group_size):
                        qh = q_head_base + h
                        score = 0.0
                        for i in cutlass.range_constexpr(self.frag_elems):
                            dim = frag_base + i
                            score += (
                                mQ[qh, dim].to(Float32) * mK[kv_head, seq_idx, dim].to(Float32)
                            )
                        score = cute.arch.warp_reduction(score, operator.add) * self.scale
                        max_prev = gPartialMax[qh]
                        sum_prev = gPartialSum[qh]
                        max_next = cute.arch.fmax(max_prev, score)
                        old_scale = cute.math.exp2(
                            (max_prev - max_next) * LOG2_E,
                            fastmath=True,
                        )
                        score_scale = cute.math.exp2((score - max_next) * LOG2_E, fastmath=True)
                        for i in cutlass.range_constexpr(self.frag_elems):
                            dim = frag_base + i
                            gPartialO[qh, dim] = (
                                gPartialO[qh, dim] * old_scale
                                + mV[kv_head, seq_idx, dim].to(Float32) * score_scale
                            )
                        if tidx == 0:
                            gPartialSum[qh] = sum_prev * old_scale + score_scale
                            gPartialMax[qh] = max_next
                        cute.arch.barrier()


@lru_cache(maxsize=None)
def _compile_decode_partial(dtype, head_dim: int, group_size: int, block_seq: int, smem_pad: int):
    q_heads = cute.sym_int()
    kv_heads = cute.sym_int()
    seq_len = cute.sym_int()
    num_splits = cute.sym_int()
    q_cute = make_fake_tensor(dtype, (q_heads, head_dim), divisibility=16 // (dtype.width // 8))
    k_cute = make_fake_tensor(dtype, (kv_heads, seq_len, head_dim), divisibility=16 // (dtype.width // 8))
    v_cute = make_fake_tensor(dtype, (kv_heads, seq_len, head_dim), divisibility=16 // (dtype.width // 8))
    po_cute = make_fake_tensor(Float32, (num_splits, q_heads, head_dim), divisibility=1)
    pm_cute = make_fake_tensor(Float32, (num_splits, q_heads), divisibility=1)
    ps_cute = make_fake_tensor(Float32, (num_splits, q_heads), divisibility=1)
    kernel = DecodePartialKernel(dtype, head_dim=head_dim, group_size=group_size, block_seq=block_seq, smem_pad=smem_pad)
    return cute.compile(
        kernel,
        q_cute,
        k_cute,
        v_cute,
        po_cute,
        pm_cute,
        ps_cute,
        CUDA_STREAM,
        options="--enable-tvm-ffi",
    )


def _launch_partial_kernel(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_splits: int,
    config: GQADecodeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[0]
    group_size = num_q_heads // num_kv_heads
    partial_out = torch.zeros(
        (num_splits, num_q_heads, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    partial_max = torch.full(
        (num_splits, num_q_heads),
        fill_value=-torch.inf,
        device=q.device,
        dtype=torch.float32,
    )
    partial_sum = torch.zeros(
        (num_splits, num_q_heads),
        device=q.device,
        dtype=torch.float32,
    )
    dtype = torch2cute_dtype_map[q.dtype]
    compiled = _compile_decode_partial(dtype, head_dim, group_size, config.block_seq, config.smem_pad)
    compiled(q, k_cache, v_cache, partial_out, partial_max, partial_sum)
    return partial_out, partial_max, partial_sum


def gqa_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    num_splits: int | None = None,
    config: GQADecodeConfig | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    if q.dim() != 2 or k_cache.dim() != 3 or v_cache.dim() != 3:
        raise ValueError("Expected q=(num_q_heads, head_dim), k/v=(num_kv_heads, seq_len, head_dim)")
    if q.device.type != "cuda" or k_cache.device.type != "cuda" or v_cache.device.type != "cuda":
        raise ValueError("All tensors must be CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Only fp16 and bf16 are supported")
    if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise ValueError("Q/K/V dtypes must match")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    if not (q.is_contiguous() and k_cache.is_contiguous() and v_cache.is_contiguous()):
        raise ValueError("Q/K/V must be contiguous")
    num_q_heads, head_dim = q.shape
    num_kv_heads, seq_len, kv_head_dim = k_cache.shape
    if kv_head_dim != head_dim:
        raise ValueError("Q/K/V head_dim mismatch")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if backend not in {"auto", "cute", "torch"}:
        raise ValueError("backend must be one of {'auto', 'cute', 'torch'}")
    if config is None:
        config = GQADecodeConfig()
    if num_splits is None:
        num_splits = select_num_splits(
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
            block_seq=config.block_seq,
            target_blocks_per_sm=config.target_blocks_per_sm,
        )
    if backend == "torch":
        return reference_gqa_decode(q, k_cache, v_cache, num_kv_heads=num_kv_heads)
    try:
        partial_out, partial_max, partial_sum = _launch_partial_kernel(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            num_splits=num_splits,
            config=config,
        )
        return _reduce_partial_results(partial_out, partial_max, partial_sum, q.dtype)
    except Exception:
        if backend == "cute":
            raise
        return reference_gqa_decode(q, k_cache, v_cache, num_kv_heads=num_kv_heads)
