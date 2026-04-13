from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import lru_cache

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from quack import copy_utils

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
    kv_heads, _, kv_head_dim = k_cache.shape
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


class DirectDecodeKernel:
    def __init__(self, dtype, head_dim: int, group_size: int, block_seq: int, num_splits: int = 1):
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
        self.num_splits = num_splits
        self.vec = head_dim // cute.arch.WARP_SIZE
        self.scale = float(head_dim**-0.5)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        stream=None,
    ):
        tiled_in = copy_utils.tiled_copy_2d(self.dtype, 32, 32, self.vec)
        self.kernel(mQ, mK, mV, mO, tiled_in).launch(
            grid=[mK.shape[0], self.num_splits, 1],
            block=[32, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        tiled_in: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        kv_head, split_idx, _ = cute.arch.block_idx()
        thr = tiled_in.get_slice(tidx)

        q_tile = cute.local_tile(mQ, (self.group_size, self.head_dim), (kv_head, 0))
        o_tile = cute.local_tile(mO, (self.group_size, self.head_dim), (kv_head, 0))

        q_regs = []
        acc_regs = []
        running_max = []
        running_sum = []

        for h in cutlass.range_constexpr(self.group_size):
            q_row = cute.local_tile(q_tile, (1, self.head_dim), (h, 0))
            tQgQ = thr.partition_S(q_row)
            tQr = cute.make_rmem_tensor_like(tQgQ)
            cute.copy(tiled_in, tQgQ, tQr)
            q_vec = tQr.load().to(Float32)
            q_regs.append(q_vec)
            acc_regs.append(q_vec * 0.0)
            running_max.append(-Float32.inf)
            running_sum.append(0.0)

        seq_len = mK.shape[1]
        total_tiles = cute.ceil_div(seq_len, self.block_seq)
        tiles_per_split = cute.ceil_div(total_tiles, self.num_splits)
        first_tile = split_idx * tiles_per_split
        k_head = mK[kv_head, None, None]
        v_head = mV[kv_head, None, None]

        for local_tile_idx in cutlass.range(tiles_per_split, unroll=1):
            tile_idx = first_tile + local_tile_idx
            if tile_idx < total_tiles:
                k_tile = cute.local_tile(k_head, (self.block_seq, self.head_dim), (tile_idx, 0))
                v_tile = cute.local_tile(v_head, (self.block_seq, self.head_dim), (tile_idx, 0))
                tile_start = tile_idx * self.block_seq
                for row in cutlass.range_constexpr(self.block_seq):
                    seq_idx = tile_start + row
                    if seq_idx < seq_len:
                        k_row = cute.local_tile(k_tile, (1, self.head_dim), (row, 0))
                        v_row = cute.local_tile(v_tile, (1, self.head_dim), (row, 0))
                        tKgK = thr.partition_S(k_row)
                        tVgV = thr.partition_S(v_row)
                        tKr = cute.make_rmem_tensor_like(tKgK)
                        tVr = cute.make_rmem_tensor_like(tVgV)
                        cute.copy(tiled_in, tKgK, tKr)
                        cute.copy(tiled_in, tVgV, tVr)
                        k_vec = tKr.load().to(Float32)
                        v_vec = tVr.load().to(Float32)

                        for h in cutlass.range_constexpr(self.group_size):
                            score = (q_regs[h] * k_vec).reduce(
                                cute.ReductionOp.ADD,
                                init_val=0.0,
                                reduction_profile=0,
                            )
                            score = cute.arch.warp_reduction(score, operator.add) * self.scale
                            max_next = cute.arch.fmax(running_max[h], score)
                            old_scale = cute.math.exp2(
                                (running_max[h] - max_next) * LOG2_E,
                                fastmath=True,
                            )
                            score_scale = cute.math.exp2((score - max_next) * LOG2_E, fastmath=True)
                            running_max[h] = max_next
                            running_sum[h] = running_sum[h] * old_scale + score_scale
                            acc_regs[h] = acc_regs[h] * old_scale + v_vec * score_scale

        for h in cutlass.range_constexpr(self.group_size):
            out_row = cute.local_tile(o_tile, (1, self.head_dim), (h, 0))
            tOgO = thr.partition_D(out_row)
            tOrO = cute.make_rmem_tensor_like(tOgO)
            out = acc_regs[h] * cute.arch.rcp_approx(running_sum[h])
            tOrO.store(out.to(tOrO.element_type))
            copy_utils.copy(tOrO, tOgO)


@lru_cache(maxsize=None)
def _compile_decode(dtype, head_dim: int, group_size: int, block_seq: int, num_splits: int = 1):
    q_heads = cute.sym_int()
    kv_heads = cute.sym_int()
    seq_len = cute.sym_int()
    q_cute = make_fake_tensor(dtype, (q_heads, head_dim), divisibility=16 // (dtype.width // 8))
    k_cute = make_fake_tensor(
        dtype,
        (kv_heads, seq_len, head_dim),
        divisibility=16 // (dtype.width // 8),
    )
    v_cute = make_fake_tensor(
        dtype,
        (kv_heads, seq_len, head_dim),
        divisibility=16 // (dtype.width // 8),
    )
    o_cute = make_fake_tensor(dtype, (q_heads, head_dim), divisibility=16 // (dtype.width // 8))
    kernel = DirectDecodeKernel(dtype, head_dim=head_dim, group_size=group_size, block_seq=block_seq, num_splits=num_splits)
    return cute.compile(
        kernel,
        q_cute,
        k_cute,
        v_cute,
        o_cute,
        CUDA_STREAM,
        options="--enable-tvm-ffi",
    )


def gqa_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    num_splits: int | None = None,
    config: GQADecodeConfig | None = None,
    backend: str = "cute",
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
    num_kv_heads, _, kv_head_dim = k_cache.shape
    if kv_head_dim != head_dim:
        raise ValueError("Q/K/V head_dim mismatch")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if backend not in {"cute", "torch"}:
        raise ValueError("backend must be one of {'cute', 'torch'}")
    if backend == "torch":
        return reference_gqa_decode(q, k_cache, v_cache, num_kv_heads=num_kv_heads)
    if config is None:
        config = GQADecodeConfig()
    if num_splits not in (None, 1):
        raise NotImplementedError("split-K path is not implemented in the current CuTe kernel")

    group_size = num_q_heads // num_kv_heads
    dtype = torch2cute_dtype_map[q.dtype]
    out = torch.empty_like(q)
    compiled = _compile_decode(dtype, head_dim, group_size, config.block_seq)
    compiled(q, k_cache, v_cache, out)
    return out
