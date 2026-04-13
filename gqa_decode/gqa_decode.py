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


class SplitKDecodeKernel:
    """Split-K kernel that writes partial (unnormalized) results."""

    def __init__(self, dtype, head_dim: int, group_size: int, block_seq: int, num_splits: int):
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
        mPartialO: cute.Tensor,
        mPartialMax: cute.Tensor,
        mPartialSum: cute.Tensor,
        stream=None,
    ):
        tiled_in = copy_utils.tiled_copy_2d(self.dtype, 32, 32, self.vec)
        tiled_f32 = copy_utils.tiled_copy_2d(Float32, 32, 32, self.vec)
        self.kernel(mQ, mK, mV, mPartialO, mPartialMax, mPartialSum, tiled_in, tiled_f32).launch(
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
        mPartialO: cute.Tensor,
        mPartialMax: cute.Tensor,
        mPartialSum: cute.Tensor,
        tiled_in: cute.TiledCopy,
        tiled_f32: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        kv_head, split_idx, _ = cute.arch.block_idx()
        thr = tiled_in.get_slice(tidx)
        thr_f32 = tiled_f32.get_slice(tidx)

        q_tile = cute.local_tile(mQ, (self.group_size, self.head_dim), (kv_head, 0))

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

        # Allocate SMEM for K/V tile staging
        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            self.dtype,
            cute.make_ordered_layout((self.block_seq, self.head_dim), order=(1, 0)),
            byte_alignment=16,
        )
        sV = smem.allocate_tensor(
            self.dtype,
            cute.make_ordered_layout((self.block_seq, self.head_dim), order=(1, 0)),
            byte_alignment=16,
        )

        # Pre-partition SMEM as copy destination (reused across tiles)
        tKsK = thr.partition_D(sK)
        tVsV = thr.partition_D(sV)

        for local_tile_idx in cutlass.range(tiles_per_split, unroll=1):
            tile_idx = first_tile + local_tile_idx
            if tile_idx < total_tiles:
                k_tile = cute.local_tile(k_head, (self.block_seq, self.head_dim), (tile_idx, 0))
                v_tile = cute.local_tile(v_head, (self.block_seq, self.head_dim), (tile_idx, 0))
                tile_start = tile_idx * self.block_seq

                # Bulk async copy K and V tiles from global to SMEM
                tKgK = thr.partition_S(k_tile)
                tVgV = thr.partition_S(v_tile)
                copy_utils.copy(tKgK, tKsK, is_async=True)
                copy_utils.copy(tVgV, tVsV, is_async=True)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                # Process rows from SMEM
                for row in cutlass.range_constexpr(self.block_seq):
                    seq_idx = tile_start + row
                    if seq_idx < seq_len:
                        sK_row = cute.local_tile(sK, (1, self.head_dim), (row, 0))
                        sV_row = cute.local_tile(sV, (1, self.head_dim), (row, 0))
                        tKsKr = thr.partition_S(sK_row)
                        tVsVr = thr.partition_S(sV_row)
                        tKr = cute.make_rmem_tensor_like(tKsKr)
                        tVr = cute.make_rmem_tensor_like(tVsVr)
                        cute.copy(tiled_in, tKsKr, tKr)
                        cute.copy(tiled_in, tVsVr, tVr)
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

        # Write partial results for split-K reduction
        for h in cutlass.range_constexpr(self.group_size):
            q_head_global = kv_head * self.group_size + h
            partial_row = q_head_global * self.num_splits + split_idx

            # Unnormalized accumulator (BF16)
            po_row = cute.local_tile(mPartialO, (1, self.head_dim), (partial_row, 0))
            tOgO = thr.partition_D(po_row)
            tOrO = cute.make_rmem_tensor_like(tOgO)
            tOrO.store(acc_regs[h].to(tOrO.element_type))
            copy_utils.copy(tOrO, tOgO)

            # Broadcast running_max to vector and write (FP32)
            max_vec = q_regs[h] * 0.0 + running_max[h]
            pm_row = cute.local_tile(mPartialMax, (1, self.head_dim), (partial_row, 0))
            tMgM = thr_f32.partition_D(pm_row)
            tMrM = cute.make_rmem_tensor_like(tMgM)
            tMrM.store(max_vec)
            copy_utils.copy(tMrM, tMgM)

            # Broadcast running_sum to vector and write (FP32)
            sum_vec = q_regs[h] * 0.0 + running_sum[h]
            ps_row = cute.local_tile(mPartialSum, (1, self.head_dim), (partial_row, 0))
            tSgS = thr_f32.partition_D(ps_row)
            tSrS = cute.make_rmem_tensor_like(tSgS)
            tSrS.store(sum_vec)
            copy_utils.copy(tSrS, tSgS)


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


@lru_cache(maxsize=None)
def _compile_splitk(dtype, head_dim: int, group_size: int, block_seq: int, num_splits: int):
    q_heads = cute.sym_int()
    kv_heads = cute.sym_int()
    seq_len = cute.sym_int()
    total_partials = cute.sym_int()
    div = 16 // (dtype.width // 8)
    q_cute = make_fake_tensor(dtype, (q_heads, head_dim), divisibility=div)
    k_cute = make_fake_tensor(dtype, (kv_heads, seq_len, head_dim), divisibility=div)
    v_cute = make_fake_tensor(dtype, (kv_heads, seq_len, head_dim), divisibility=div)
    po_cute = make_fake_tensor(dtype, (total_partials, head_dim), divisibility=div)
    pm_cute = make_fake_tensor(Float32, (total_partials, head_dim), divisibility=16 // (Float32.width // 8))
    ps_cute = make_fake_tensor(Float32, (total_partials, head_dim), divisibility=16 // (Float32.width // 8))
    kernel = SplitKDecodeKernel(
        dtype, head_dim=head_dim, group_size=group_size, block_seq=block_seq, num_splits=num_splits,
    )
    return cute.compile(
        kernel,
        q_cute, k_cute, v_cute, po_cute, pm_cute, ps_cute,
        CUDA_STREAM,
        options="--enable-tvm-ffi",
    )


def reduce_partials(
    partial_out: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    num_q_heads: int,
    num_splits: int,
    head_dim: int,
) -> torch.Tensor:
    """Combine split-K partial results using online softmax correction."""
    out_dtype = partial_out.dtype
    partial_out = partial_out.view(num_q_heads, num_splits, head_dim).float()
    # Max/sum are broadcast across head_dim; extract column 0 for the scalar
    partial_max = partial_max.view(num_q_heads, num_splits, head_dim)[:, :, 0]
    partial_sum = partial_sum.view(num_q_heads, num_splits, head_dim)[:, :, 0]

    global_max = partial_max.max(dim=1, keepdim=True).values
    correction = torch.exp(partial_max - global_max)
    corrected_sum = (partial_sum * correction).sum(dim=1, keepdim=True)
    corrected_out = (partial_out * correction.unsqueeze(-1)).sum(dim=1)

    return (corrected_out / corrected_sum).to(out_dtype)


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
    group_size = num_q_heads // num_kv_heads
    dtype = torch2cute_dtype_map[q.dtype]
    seq_len = k_cache.shape[1]

    if num_splits is None:
        actual_splits = select_num_splits(
            num_kv_heads, seq_len, config.block_seq, config.target_blocks_per_sm,
        )
    else:
        actual_splits = num_splits

    total_partials = num_q_heads * actual_splits
    partial_out = torch.empty(total_partials, head_dim, dtype=q.dtype, device=q.device)
    partial_max = torch.empty(total_partials, head_dim, dtype=torch.float32, device=q.device)
    partial_sum = torch.empty(total_partials, head_dim, dtype=torch.float32, device=q.device)

    compiled = _compile_splitk(dtype, head_dim, group_size, config.block_seq, actual_splits)
    compiled(q, k_cache, v_cache, partial_out, partial_max, partial_sum)

    return reduce_partials(partial_out, partial_max, partial_sum, num_q_heads, actual_splits, head_dim)
