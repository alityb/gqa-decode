from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import lru_cache

import torch

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Float8E4M3FN, Int32

from quack import copy_utils

from gqa_decode.cute_dsl_utils import make_fake_tensor, torch2cute_dtype_map


CUDA_STREAM = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
LOG2_E = math.log2(math.e)
HBM_PEAK_BW = 3.35e12
DEFAULT_BLOCK_SEQ = 16
DEFAULT_TARGET_BLOCKS_PER_SM = 16
NUM_SMS_H100_SXM = 132
SMEM_PAD = 8
WARP_SIZE = 32


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
        raise ValueError(
            "Expected q=(num_q_heads, head_dim), k/v=(num_kv_heads, seq_len, head_dim)"
        )
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


def _smem_bytes_per_block(
    block_seq: int, head_dim: int, smem_pad: int, elem_bytes: int = 2
) -> int:
    """SMEM usage: two padded tiles for K and V, plus allocator alignment."""
    tile_elems = (block_seq - 1) * (head_dim + smem_pad) + head_dim
    tile_bytes = tile_elems * elem_bytes
    # SmemAllocator aligns each allocation to 16 bytes
    tile_bytes = (tile_bytes + 15) & ~15
    return 2 * tile_bytes


def select_num_splits(
    num_kv_heads: int,
    seq_len: int,
    block_seq: int = DEFAULT_BLOCK_SEQ,
    target_blocks_per_sm: int = DEFAULT_TARGET_BLOCKS_PER_SM,
    num_sms: int = NUM_SMS_H100_SXM,
) -> int:
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")

    # Compute SMEM-limited max concurrent blocks per SM.
    # Subtract 1 from theoretical max — allocator overhead uses ~1-2 KB
    # that the cosize calculation doesn't account for.
    smem_per_block = _smem_bytes_per_block(block_seq, 128, SMEM_PAD)
    smem_per_sm = 228 * 1024  # H100 SXM
    max_blocks_per_sm = max(1, smem_per_sm // smem_per_block - 1)

    # Use target occupancy, capped by SMEM ceiling to stay within one wave.
    blocks_per_sm = min(target_blocks_per_sm, max_blocks_per_sm)
    max_grid = num_sms * blocks_per_sm
    min_splits = max(1, max_grid // num_kv_heads)
    max_tiles = max(1, math.ceil(seq_len / block_seq))
    return min(min_splits, max_tiles)


class DirectDecodeKernel:
    def __init__(
        self, dtype, head_dim: int, group_size: int, block_seq: int, num_splits: int = 1
    ):
        if head_dim not in (64, 128, 256):
            raise ValueError(
                f"Unsupported head_dim={head_dim}; expected 64, 128, or 256"
            )
        if head_dim % cute.arch.WARP_SIZE != 0:
            raise ValueError("head_dim must be divisible by warp size")
        if group_size < 1 or group_size > 16:
            raise ValueError("group_size must be in [1, 16]")
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
                k_tile = cute.local_tile(
                    k_head, (self.block_seq, self.head_dim), (tile_idx, 0)
                )
                v_tile = cute.local_tile(
                    v_head, (self.block_seq, self.head_dim), (tile_idx, 0)
                )
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
                            score = (
                                cute.arch.warp_reduction(score, operator.add)
                                * self.scale
                            )
                            max_next = cute.arch.fmax(running_max[h], score)
                            old_scale = cute.math.exp2(
                                (running_max[h] - max_next) * LOG2_E,
                                fastmath=True,
                            )
                            score_scale = cute.math.exp2(
                                (score - max_next) * LOG2_E, fastmath=True
                            )
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

    def __init__(
        self, dtype, head_dim: int, group_size: int, block_seq: int, num_splits: int
    ):
        if head_dim not in (64, 128, 256):
            raise ValueError(
                f"Unsupported head_dim={head_dim}; expected 64, 128, or 256"
            )
        if head_dim % cute.arch.WARP_SIZE != 0:
            raise ValueError("head_dim must be divisible by warp size")
        if group_size < 1 or group_size > 16:
            raise ValueError("group_size must be in [1, 16]")
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
        self.kernel(
            mQ, mK, mV, mPartialO, mPartialMax, mPartialSum, tiled_in, tiled_f32
        ).launch(
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

        # Allocate SMEM for K/V tile staging with PAD to avoid bank conflicts
        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_layout(
            (self.block_seq, self.head_dim),
            stride=(self.head_dim + SMEM_PAD, 1),
        )
        sK = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=16)
        sV = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=16)

        # Pre-partition SMEM as copy destination (reused across tiles)
        tKsK = thr.partition_D(sK)
        tVsV = thr.partition_D(sV)

        for local_tile_idx in cutlass.range(tiles_per_split, unroll=1):
            tile_idx = first_tile + local_tile_idx
            if tile_idx < total_tiles:
                k_tile = cute.local_tile(
                    k_head, (self.block_seq, self.head_dim), (tile_idx, 0)
                )
                v_tile = cute.local_tile(
                    v_head, (self.block_seq, self.head_dim), (tile_idx, 0)
                )
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
                            score = (
                                cute.arch.warp_reduction(score, operator.add)
                                * self.scale
                            )
                            max_next = cute.arch.fmax(running_max[h], score)
                            old_scale = cute.math.exp2(
                                (running_max[h] - max_next) * LOG2_E,
                                fastmath=True,
                            )
                            score_scale = cute.math.exp2(
                                (score - max_next) * LOG2_E, fastmath=True
                            )
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


class ReductionKernel:
    """Fused split-K reduction: one kernel launch replaces ~8 PyTorch ops."""

    def __init__(self, dtype, head_dim: int, num_splits: int):
        self.dtype = dtype
        self.head_dim = head_dim
        self.num_splits = num_splits
        self.vecsize = head_dim // WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mPartialO: cute.Tensor,  # [total_partials, head_dim] BF16
        mPartialMax: cute.Tensor,  # [total_partials, head_dim] FP32 (col 0 = scalar)
        mPartialSum: cute.Tensor,  # [total_partials, head_dim] FP32 (col 0 = scalar)
        mOutput: cute.Tensor,  # [num_q_heads, head_dim] BF16
        num_q_heads: Int32,
        num_splits: Int32,
        stream=None,
    ):
        self.kernel(
            mPartialO,
            mPartialMax,
            mPartialSum,
            mOutput,
            num_q_heads,
            num_splits,
        ).launch(
            grid=[num_q_heads, 1, 1],
            block=[WARP_SIZE, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mPartialO: cute.Tensor,
        mPartialMax: cute.Tensor,
        mPartialSum: cute.Tensor,
        mOutput: cute.Tensor,
        num_q_heads: Int32,
        num_splits: Int32,
    ):
        qh = cute.arch.block_idx()[0]
        lane = cute.arch.lane_idx()
        col_start = lane * self.vecsize

        # Use list-element assignment so values carry across range_constexpr iterations.
        running_max = [-Float32.inf]

        # Step 1: Find global max across all splits
        for s in cutlass.range_constexpr(self.num_splits):
            partial_row = qh * num_splits + s
            m = mPartialMax[partial_row, 0]
            running_max[0] = cute.arch.fmax(running_max[0], m)

        # Step 2: Accumulate corrected outputs and sums
        corrected_sum = [Float32(0.0)]
        acc = [Float32(0.0) for _ in range(self.vecsize)]

        for s in cutlass.range_constexpr(self.num_splits):
            partial_row = qh * num_splits + s
            correction = cute.math.exp(
                mPartialMax[partial_row, 0] - running_max[0], fastmath=True
            )
            corrected_sum[0] = (
                corrected_sum[0] + mPartialSum[partial_row, 0] * correction
            )
            for e in cutlass.range_constexpr(self.vecsize):
                acc[e] = (
                    acc[e] + Float32(mPartialO[partial_row, col_start + e]) * correction
                )

        # Step 3: Normalize and write output
        inv_sum = Float32(1.0) / corrected_sum[0]
        for e in cutlass.range_constexpr(self.vecsize):
            mOutput[qh, col_start + e] = BFloat16(acc[e] * inv_sum)


@lru_cache(maxsize=None)
def _compile_reduction(dtype, head_dim: int, num_splits: int):
    """Compile the fused reduction kernel."""
    partial_sym = cute.sym_int()
    output_sym = cute.sym_int()
    div = 16 // (dtype.width // 8)

    mPO = make_fake_tensor(dtype, (partial_sym, head_dim), divisibility=div)
    mPMax = make_fake_tensor(
        Float32, (partial_sym, head_dim), divisibility=16 // (Float32.width // 8)
    )
    mPSum = make_fake_tensor(
        Float32, (partial_sym, head_dim), divisibility=16 // (Float32.width // 8)
    )
    mOut = make_fake_tensor(dtype, (output_sym, head_dim), divisibility=div)

    kernel_op = ReductionKernel(dtype, head_dim, num_splits)
    return cute.compile(
        kernel_op,
        mPO,
        mPMax,
        mPSum,
        mOut,
        Int32(0),
        Int32(0),
        CUDA_STREAM,
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=None)
def _compile_decode(
    dtype, head_dim: int, group_size: int, block_seq: int, num_splits: int = 1
):
    q_heads = cute.sym_int()
    kv_heads = cute.sym_int()
    seq_len = cute.sym_int()
    q_cute = make_fake_tensor(
        dtype, (q_heads, head_dim), divisibility=16 // (dtype.width // 8)
    )
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
    o_cute = make_fake_tensor(
        dtype, (q_heads, head_dim), divisibility=16 // (dtype.width // 8)
    )
    kernel = DirectDecodeKernel(
        dtype,
        head_dim=head_dim,
        group_size=group_size,
        block_seq=block_seq,
        num_splits=num_splits,
    )
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
def _compile_splitk(
    dtype, head_dim: int, group_size: int, block_seq: int, num_splits: int
):
    q_heads = cute.sym_int()
    kv_heads = cute.sym_int()
    seq_len = cute.sym_int()
    total_partials = cute.sym_int()
    div = 16 // (dtype.width // 8)
    q_cute = make_fake_tensor(dtype, (q_heads, head_dim), divisibility=div)
    k_cute = make_fake_tensor(dtype, (kv_heads, seq_len, head_dim), divisibility=div)
    v_cute = make_fake_tensor(dtype, (kv_heads, seq_len, head_dim), divisibility=div)
    po_cute = make_fake_tensor(dtype, (total_partials, head_dim), divisibility=div)
    pm_cute = make_fake_tensor(
        Float32, (total_partials, head_dim), divisibility=16 // (Float32.width // 8)
    )
    ps_cute = make_fake_tensor(
        Float32, (total_partials, head_dim), divisibility=16 // (Float32.width // 8)
    )
    kernel = SplitKDecodeKernel(
        dtype,
        head_dim=head_dim,
        group_size=group_size,
        block_seq=block_seq,
        num_splits=num_splits,
    )
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


@lru_cache(maxsize=None)
def _compile_double_buffer_splitk(
    dtype, head_dim: int, group_size: int, block_seq: int, num_splits: int
):
    q_heads = cute.sym_int()
    kv_heads = cute.sym_int()
    seq_len = cute.sym_int()
    total_partials = cute.sym_int()
    div = 16 // (dtype.width // 8)
    q_cute = make_fake_tensor(dtype, (q_heads, head_dim), divisibility=div)
    k_cute = make_fake_tensor(dtype, (kv_heads, seq_len, head_dim), divisibility=div)
    v_cute = make_fake_tensor(dtype, (kv_heads, seq_len, head_dim), divisibility=div)
    po_cute = make_fake_tensor(dtype, (total_partials, head_dim), divisibility=div)
    pm_cute = make_fake_tensor(
        Float32, (total_partials, head_dim), divisibility=16 // (Float32.width // 8)
    )
    ps_cute = make_fake_tensor(
        Float32, (total_partials, head_dim), divisibility=16 // (Float32.width // 8)
    )
    kernel = DoubleBufferSplitKKernel(
        dtype,
        head_dim=head_dim,
        group_size=group_size,
        block_seq=block_seq,
        num_splits=num_splits,
    )
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
    # Max/sum are broadcast across head_dim; column 0 holds the scalar
    partial_max = partial_max[:, 0].view(num_q_heads, num_splits)
    partial_sum = partial_sum[:, 0].view(num_q_heads, num_splits)

    global_max = partial_max.max(dim=1, keepdim=True).values
    correction = torch.exp(partial_max - global_max)  # [q_heads, splits]
    corrected_sum = (partial_sum * correction).sum(dim=1, keepdim=True)  # [q_heads, 1]

    # Apply correction to output and sum in one pass
    partial_out = partial_out.view(num_q_heads, num_splits, head_dim).float()
    corrected_out = torch.einsum("qsd,qs->qd", partial_out, correction)

    return (corrected_out / corrected_sum).to(out_dtype)


def gqa_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    num_splits: int | None = None,
    config: GQADecodeConfig | None = None,
    backend: str = "cute",
    reduction: str = "fused",
) -> torch.Tensor:
    if q.dim() != 2 or k_cache.dim() != 3 or v_cache.dim() != 3:
        raise ValueError(
            "Expected q=(num_q_heads, head_dim), k/v=(num_kv_heads, seq_len, head_dim)"
        )
    if (
        q.device.type != "cuda"
        or k_cache.device.type != "cuda"
        or v_cache.device.type != "cuda"
    ):
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
            num_kv_heads,
            seq_len,
            config.block_seq,
            config.target_blocks_per_sm,
        )
    else:
        actual_splits = num_splits

    if actual_splits == 1:
        # Fast path: direct output, no partial buffers or reduction needed
        out = torch.empty_like(q)
        compiled = _compile_decode(
            dtype, head_dim, group_size, config.block_seq, num_splits=1
        )
        compiled(q, k_cache, v_cache, out)
        return out

    total_partials = num_q_heads * actual_splits
    partial_out = torch.empty(total_partials, head_dim, dtype=q.dtype, device=q.device)
    partial_max = torch.empty(
        total_partials, head_dim, dtype=torch.float32, device=q.device
    )
    partial_sum = torch.empty(
        total_partials, head_dim, dtype=torch.float32, device=q.device
    )

    compiled = (
        _compile_splitk(dtype, head_dim, group_size, config.block_seq, actual_splits)
        if group_size == 1
        else _compile_double_buffer_splitk(
            dtype, head_dim, group_size, config.block_seq, actual_splits
        )
    )
    compiled(q, k_cache, v_cache, partial_out, partial_max, partial_sum)

    if reduction == "python":
        return reduce_partials(
            partial_out, partial_max, partial_sum, num_q_heads, actual_splits, head_dim
        )

    # Fused CuTe-DSL reduction: single kernel launch
    out = torch.empty(num_q_heads, head_dim, dtype=q.dtype, device=q.device)
    compiled_red = _compile_reduction(dtype, head_dim, actual_splits)
    compiled_red(partial_out, partial_max, partial_sum, out, num_q_heads, actual_splits)
    return out


# ---------------------------------------------------------------------------
# Persistent kernel: 132 blocks, each processes a contiguous work range.
# Eliminates split-K reduction overhead for GQA configs.
# ---------------------------------------------------------------------------


class PersistentDecodeKernel:
    """Persistent GQA decode: NUM_BLOCKS blocks, each loops over assigned tiles.

    Work assignment: block b processes work items [b*items_per_block,
    (b+1)*items_per_block). Work items are ordered kv-head-major so most
    blocks only touch 1-2 KV heads, keeping flush-and-reload rare.

    Partial buffer layout: [num_q_heads * NUM_BLOCKS, head_dim], indexed by
    partial_row = q_head_global * NUM_BLOCKS + block_idx. Compatible with
    the existing ReductionKernel (num_splits = NUM_BLOCKS).
    """

    NUM_BLOCKS = NUM_SMS_H100_SXM  # 132

    def __init__(
        self,
        dtype,
        head_dim: int,
        group_size: int,
        block_seq: int,
        items_per_block: int,
    ):
        if head_dim not in (64, 128):
            raise ValueError(f"Unsupported head_dim={head_dim}")
        if group_size < 1 or group_size > 8:
            raise ValueError("group_size must be in [1, 8]")
        self.dtype = dtype
        self.head_dim = head_dim
        self.group_size = group_size
        self.block_seq = block_seq
        self.items_per_block = items_per_block  # compile-time constant
        self.vec = head_dim // WARP_SIZE
        self.attn_scale = float(head_dim**-0.5)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPartialO: cute.Tensor,
        mPartialMax: cute.Tensor,
        mPartialSum: cute.Tensor,
        num_kv_heads: Int32,
        num_q_heads: Int32,
        seq_len: Int32,
        tiles_per_head: Int32,
        total_work_items: Int32,
        stream=None,
    ):
        tiled_in = copy_utils.tiled_copy_2d(self.dtype, 32, 32, self.vec)
        tiled_f32 = copy_utils.tiled_copy_2d(Float32, 32, 32, self.vec)
        self.kernel(
            mQ,
            mK,
            mV,
            mPartialO,
            mPartialMax,
            mPartialSum,
            num_kv_heads,
            num_q_heads,
            seq_len,
            tiles_per_head,
            total_work_items,
            tiled_in,
            tiled_f32,
        ).launch(
            grid=[self.NUM_BLOCKS, 1, 1],
            block=[WARP_SIZE, 1, 1],
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
        num_kv_heads: Int32,
        num_q_heads: Int32,
        seq_len: Int32,
        tiles_per_head: Int32,
        total_work_items: Int32,
        tiled_in: cute.TiledCopy,
        tiled_f32: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx = cute.arch.block_idx()[0]
        thr = tiled_in.get_slice(tidx)
        thr_f32 = tiled_f32.get_slice(tidx)

        block_start = block_idx * Int32(self.items_per_block)

        # SMEM for K/V staging — reused every tile, same as SplitK kernel
        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_layout(
            (self.block_seq, self.head_dim),
            stride=(self.head_dim + SMEM_PAD, 1),
        )
        sK = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=16)
        sV = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=16)
        tKsK = thr.partition_D(sK)
        tVsV = thr.partition_D(sV)

        # Dummy Q load for type initialization — always overwritten before use
        # (first work item triggers a kv_head transition from sentinel -1)
        q_tile_dummy = cute.local_tile(
            mQ, (self.group_size, self.head_dim), (Int32(0), 0)
        )
        q_regs = []
        acc_regs = []
        running_max = []
        running_sum = []
        for h in cutlass.range_constexpr(self.group_size):
            q_row = cute.local_tile(q_tile_dummy, (1, self.head_dim), (h, 0))
            tQgQ = thr.partition_S(q_row)
            tQr = cute.make_rmem_tensor_like(tQgQ)
            cute.copy(tiled_in, tQgQ, tQr)
            q_vec = tQr.load().to(Float32)
            q_regs.append(q_vec)
            acc_regs.append(q_vec * Float32(0.0))
            running_max.append(-Float32.inf)
            running_sum.append(Float32(0.0))

        # Sentinel: -1 means no kv_head has been started yet
        current_kv_head = [Int32(-1)]

        # ---- Main persistent loop ----
        for item_offset in cutlass.range(self.items_per_block, unroll=1):
            work_idx = block_start + Int32(item_offset)
            if work_idx < total_work_items:
                kv_head_new = work_idx // tiles_per_head
                tile_idx = work_idx % tiles_per_head
                tile_start = tile_idx * Int32(self.block_seq)

                # ---- KV head transition ----
                if kv_head_new != current_kv_head[0]:
                    # Flush partials for the previous kv_head (skip on first item)
                    if current_kv_head[0] >= Int32(0):
                        for h in cutlass.range_constexpr(self.group_size):
                            q_hg = current_kv_head[0] * Int32(self.group_size) + Int32(
                                h
                            )
                            pr = q_hg * Int32(self.NUM_BLOCKS) + block_idx

                            po_row = cute.local_tile(
                                mPartialO, (1, self.head_dim), (pr, 0)
                            )
                            tOgO = thr.partition_D(po_row)
                            tOrO = cute.make_rmem_tensor_like(tOgO)
                            tOrO.store(acc_regs[h].to(tOrO.element_type))
                            copy_utils.copy(tOrO, tOgO)

                            max_vec = q_regs[h] * Float32(0.0) + running_max[h]
                            pm_row = cute.local_tile(
                                mPartialMax, (1, self.head_dim), (pr, 0)
                            )
                            tMgM = thr_f32.partition_D(pm_row)
                            tMrM = cute.make_rmem_tensor_like(tMgM)
                            tMrM.store(max_vec)
                            copy_utils.copy(tMrM, tMgM)

                            sum_vec = q_regs[h] * Float32(0.0) + running_sum[h]
                            ps_row = cute.local_tile(
                                mPartialSum, (1, self.head_dim), (pr, 0)
                            )
                            tSgS = thr_f32.partition_D(ps_row)
                            tSrS = cute.make_rmem_tensor_like(tSgS)
                            tSrS.store(sum_vec)
                            copy_utils.copy(tSrS, tSgS)

                    # Load Q for new kv_head and reset accumulators
                    q_tile_new = cute.local_tile(
                        mQ, (self.group_size, self.head_dim), (kv_head_new, 0)
                    )
                    for h in cutlass.range_constexpr(self.group_size):
                        q_row_new = cute.local_tile(
                            q_tile_new, (1, self.head_dim), (h, 0)
                        )
                        tQgQ_new = thr.partition_S(q_row_new)
                        tQr_new = cute.make_rmem_tensor_like(tQgQ_new)
                        cute.copy(tiled_in, tQgQ_new, tQr_new)
                        q_regs[h] = tQr_new.load().to(Float32)
                        acc_regs[h] = q_regs[h] * Float32(0.0)
                        running_max[h] = -Float32.inf
                        running_sum[h] = Float32(0.0)

                    current_kv_head[0] = kv_head_new

                # ---- Load K/V tile into SMEM ----
                k_head_slice = mK[kv_head_new, None, None]
                v_head_slice = mV[kv_head_new, None, None]
                k_tile = cute.local_tile(
                    k_head_slice, (self.block_seq, self.head_dim), (tile_idx, 0)
                )
                v_tile = cute.local_tile(
                    v_head_slice, (self.block_seq, self.head_dim), (tile_idx, 0)
                )
                tKgK = thr.partition_S(k_tile)
                tVgV = thr.partition_S(v_tile)
                copy_utils.copy(tKgK, tKsK, is_async=True)
                copy_utils.copy(tVgV, tVsV, is_async=True)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                # ---- Compute attention over SMEM rows ----
                for row in cutlass.range_constexpr(self.block_seq):
                    seq_idx = tile_start + Int32(row)
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
                            score = (
                                cute.arch.warp_reduction(score, operator.add)
                                * self.attn_scale
                            )
                            max_next = cute.arch.fmax(running_max[h], score)
                            old_scale = cute.math.exp2(
                                (running_max[h] - max_next) * LOG2_E, fastmath=True
                            )
                            score_scale = cute.math.exp2(
                                (score - max_next) * LOG2_E, fastmath=True
                            )
                            running_max[h] = max_next
                            running_sum[h] = running_sum[h] * old_scale + score_scale
                            acc_regs[h] = acc_regs[h] * old_scale + v_vec * score_scale

        # ---- Final flush for the last kv_head processed ----
        if current_kv_head[0] >= Int32(0):
            for h in cutlass.range_constexpr(self.group_size):
                q_hg = current_kv_head[0] * Int32(self.group_size) + Int32(h)
                pr = q_hg * Int32(self.NUM_BLOCKS) + block_idx

                po_row = cute.local_tile(mPartialO, (1, self.head_dim), (pr, 0))
                tOgO = thr.partition_D(po_row)
                tOrO = cute.make_rmem_tensor_like(tOgO)
                tOrO.store(acc_regs[h].to(tOrO.element_type))
                copy_utils.copy(tOrO, tOgO)

                max_vec = q_regs[h] * Float32(0.0) + running_max[h]
                pm_row = cute.local_tile(mPartialMax, (1, self.head_dim), (pr, 0))
                tMgM = thr_f32.partition_D(pm_row)
                tMrM = cute.make_rmem_tensor_like(tMgM)
                tMrM.store(max_vec)
                copy_utils.copy(tMrM, tMgM)

                sum_vec = q_regs[h] * Float32(0.0) + running_sum[h]
                ps_row = cute.local_tile(mPartialSum, (1, self.head_dim), (pr, 0))
            tSgS = thr_f32.partition_D(ps_row)
            tSrS = cute.make_rmem_tensor_like(tSgS)
            tSrS.store(sum_vec)
            copy_utils.copy(tSrS, tSgS)


class DoubleBufferSplitKKernel:
    """Split-K GQA decode with double-buffered cp.async pipeline.

    Overlaps K/V tile load for tile T+1 with attention compute for tile T,
    keeping the HBM DMA engine busy during compute to close the 3× gap vs
    FlashInfer on production GQA configs (Llama, Qwen).

    Work assignment is identical to SplitKDecodeKernel (used for GQA only,
    i.e. group_size > 1). MHA stays on the single-buffer kernel.

    Buffer layout:
        A (sK_A / sV_A) holds the EVEN tile within each pair.
        B (sK_B / sV_B) holds the ODD tile within each pair.

    Pipeline per pair (tiles 2p, 2p+1):
        1. A is already ready (prologue or previous pair waited for it).
        2. Issue async load of tile 2p+1 → B  [non-blocking]
        3. Commit group.
        4. Compute tile 2p from A              [overlaps with step 2]
        5. Wait for B (tile 2p+1 ready).
        6. Issue async load of tile 2(p+1) → A  [non-blocking]
        7. Commit group.
        8. Compute tile 2p+1 from B            [overlaps with step 6]
        9. Wait for A (tile 2(p+1) ready).  ← A ready for next pair.
    """

    def __init__(
        self, dtype, head_dim: int, group_size: int, block_seq: int, num_splits: int
    ):
        if head_dim not in (64, 128, 256):
            raise ValueError(
                f"Unsupported head_dim={head_dim}; expected 64, 128, or 256"
            )
        if head_dim % cute.arch.WARP_SIZE != 0:
            raise ValueError("head_dim must be divisible by warp size")
        if group_size < 1 or group_size > 16:
            raise ValueError("group_size must be in [1, 16]")
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
        self.kernel(
            mQ, mK, mV, mPartialO, mPartialMax, mPartialSum, tiled_in, tiled_f32
        ).launch(
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

        # Two SMEM buffers: A for even tiles, B for odd tiles
        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_layout(
            (self.block_seq, self.head_dim),
            stride=(self.head_dim + SMEM_PAD, 1),
        )
        sK_A = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=16)
        sV_A = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=16)
        sK_B = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=16)
        sV_B = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=16)

        tKsK_A = thr.partition_D(sK_A)
        tVsV_A = thr.partition_D(sV_A)
        tKsK_B = thr.partition_D(sK_B)
        tVsV_B = thr.partition_D(sV_B)

        # ---- PROLOGUE: issue async load of tile 0 (even) into A ----
        if first_tile < total_tiles:
            k_t = cute.local_tile(
                k_head, (self.block_seq, self.head_dim), (first_tile, 0)
            )
            v_t = cute.local_tile(
                v_head, (self.block_seq, self.head_dim), (first_tile, 0)
            )
            copy_utils.copy(thr.partition_S(k_t), tKsK_A, is_async=True)
            copy_utils.copy(thr.partition_S(v_t), tVsV_A, is_async=True)
            cute.arch.cp_async_commit_group()

        total_pairs = cute.ceil_div(tiles_per_split, 2)

        # ---- PAIR LOOP: A always has the even tile, B always has the odd tile ----
        for pair in cutlass.range(total_pairs, unroll=1):
            even_tile = first_tile + pair * 2
            odd_tile = first_tile + pair * 2 + 1
            split_end = first_tile + tiles_per_split

            # Wait for A (even tile) — committed in prologue or end of previous pair
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # Issue async load of odd tile into B (non-blocking)
            if odd_tile < total_tiles and odd_tile < split_end:
                k_odd = cute.local_tile(
                    k_head, (self.block_seq, self.head_dim), (odd_tile, 0)
                )
                v_odd = cute.local_tile(
                    v_head, (self.block_seq, self.head_dim), (odd_tile, 0)
                )
                copy_utils.copy(thr.partition_S(k_odd), tKsK_B, is_async=True)
                copy_utils.copy(thr.partition_S(v_odd), tVsV_B, is_async=True)
                cute.arch.cp_async_commit_group()

            # Compute even tile from A (overlaps with odd tile loading above)
            if even_tile < total_tiles:
                even_start = even_tile * self.block_seq
                for row in cutlass.range_constexpr(self.block_seq):
                    seq_idx = even_start + row
                    if seq_idx < seq_len:
                        sK_row = cute.local_tile(sK_A, (1, self.head_dim), (row, 0))
                        sV_row = cute.local_tile(sV_A, (1, self.head_dim), (row, 0))
                        tKr = cute.make_rmem_tensor_like(thr.partition_S(sK_row))
                        tVr = cute.make_rmem_tensor_like(thr.partition_S(sV_row))
                        cute.copy(tiled_in, thr.partition_S(sK_row), tKr)
                        cute.copy(tiled_in, thr.partition_S(sV_row), tVr)
                        k_vec = tKr.load().to(Float32)
                        v_vec = tVr.load().to(Float32)
                        for h in cutlass.range_constexpr(self.group_size):
                            score = (q_regs[h] * k_vec).reduce(
                                cute.ReductionOp.ADD,
                                init_val=0.0,
                                reduction_profile=0,
                            )
                            score = (
                                cute.arch.warp_reduction(score, operator.add)
                                * self.scale
                            )
                            max_next = cute.arch.fmax(running_max[h], score)
                            old_scale = cute.math.exp2(
                                (running_max[h] - max_next) * LOG2_E, fastmath=True
                            )
                            score_scale = cute.math.exp2(
                                (score - max_next) * LOG2_E, fastmath=True
                            )
                            running_max[h] = max_next
                            running_sum[h] = running_sum[h] * old_scale + score_scale
                            acc_regs[h] = acc_regs[h] * old_scale + v_vec * score_scale

            # Handle odd tile if it exists
            if odd_tile < total_tiles and odd_tile < split_end:
                # Wait for B (odd tile ready)
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                # Issue async load of next even tile into A (non-blocking)
                next_even = first_tile + (pair + 1) * 2
                if next_even < total_tiles and next_even < split_end:
                    k_ne = cute.local_tile(
                        k_head, (self.block_seq, self.head_dim), (next_even, 0)
                    )
                    v_ne = cute.local_tile(
                        v_head, (self.block_seq, self.head_dim), (next_even, 0)
                    )
                    copy_utils.copy(thr.partition_S(k_ne), tKsK_A, is_async=True)
                    copy_utils.copy(thr.partition_S(v_ne), tVsV_A, is_async=True)
                    cute.arch.cp_async_commit_group()

                # Compute odd tile from B (overlaps with next even tile loading above)
                odd_start = odd_tile * self.block_seq
                for row in cutlass.range_constexpr(self.block_seq):
                    seq_idx = odd_start + row
                    if seq_idx < seq_len:
                        sK_row = cute.local_tile(sK_B, (1, self.head_dim), (row, 0))
                        sV_row = cute.local_tile(sV_B, (1, self.head_dim), (row, 0))
                        tKr = cute.make_rmem_tensor_like(thr.partition_S(sK_row))
                        tVr = cute.make_rmem_tensor_like(thr.partition_S(sV_row))
                        cute.copy(tiled_in, thr.partition_S(sK_row), tKr)
                        cute.copy(tiled_in, thr.partition_S(sV_row), tVr)
                        k_vec = tKr.load().to(Float32)
                        v_vec = tVr.load().to(Float32)
                        for h in cutlass.range_constexpr(self.group_size):
                            score = (q_regs[h] * k_vec).reduce(
                                cute.ReductionOp.ADD,
                                init_val=0.0,
                                reduction_profile=0,
                            )
                            score = (
                                cute.arch.warp_reduction(score, operator.add)
                                * self.scale
                            )
                            max_next = cute.arch.fmax(running_max[h], score)
                            old_scale = cute.math.exp2(
                                (running_max[h] - max_next) * LOG2_E, fastmath=True
                            )
                            score_scale = cute.math.exp2(
                                (score - max_next) * LOG2_E, fastmath=True
                            )
                            running_max[h] = max_next
                            running_sum[h] = running_sum[h] * old_scale + score_scale
                            acc_regs[h] = acc_regs[h] * old_scale + v_vec * score_scale

                # Wait for A (next even tile) so it's ready at start of next pair
                next_even2 = first_tile + (pair + 1) * 2
                if next_even2 < total_tiles and next_even2 < split_end:
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()

        # Write partial results (identical to SplitKDecodeKernel)
        for h in cutlass.range_constexpr(self.group_size):
            q_head_global = kv_head * self.group_size + h
            partial_row = q_head_global * self.num_splits + split_idx

            po_row = cute.local_tile(mPartialO, (1, self.head_dim), (partial_row, 0))
            tOgO = thr.partition_D(po_row)
            tOrO = cute.make_rmem_tensor_like(tOgO)
            tOrO.store(acc_regs[h].to(tOrO.element_type))
            copy_utils.copy(tOrO, tOgO)

            max_vec = q_regs[h] * 0.0 + running_max[h]
            pm_row = cute.local_tile(mPartialMax, (1, self.head_dim), (partial_row, 0))
            tMgM = thr_f32.partition_D(pm_row)
            tMrM = cute.make_rmem_tensor_like(tMgM)
            tMrM.store(max_vec)
            copy_utils.copy(tMrM, tMgM)

            sum_vec = q_regs[h] * 0.0 + running_sum[h]
            ps_row = cute.local_tile(mPartialSum, (1, self.head_dim), (partial_row, 0))
            tSgS = thr_f32.partition_D(ps_row)
            tSrS = cute.make_rmem_tensor_like(tSgS)
            tSrS.store(sum_vec)
            copy_utils.copy(tSrS, tSgS)


@lru_cache(maxsize=None)
def _compile_persistent(
    dtype, head_dim: int, group_size: int, block_seq: int, items_per_block: int
):
    q_heads = cute.sym_int()
    kv_heads = cute.sym_int()
    seq_len = cute.sym_int()
    total_partials = cute.sym_int()
    div = 16 // (dtype.width // 8)

    q_cute = make_fake_tensor(dtype, (q_heads, head_dim), divisibility=div)
    k_cute = make_fake_tensor(dtype, (kv_heads, seq_len, head_dim), divisibility=div)
    v_cute = make_fake_tensor(dtype, (kv_heads, seq_len, head_dim), divisibility=div)
    po_cute = make_fake_tensor(dtype, (total_partials, head_dim), divisibility=div)
    pm_cute = make_fake_tensor(
        Float32, (total_partials, head_dim), divisibility=16 // (Float32.width // 8)
    )
    ps_cute = make_fake_tensor(
        Float32, (total_partials, head_dim), divisibility=16 // (Float32.width // 8)
    )

    kernel = PersistentDecodeKernel(
        dtype, head_dim, group_size, block_seq, items_per_block
    )
    return cute.compile(
        kernel,
        q_cute,
        k_cute,
        v_cute,
        po_cute,
        pm_cute,
        ps_cute,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        CUDA_STREAM,
        options="--enable-tvm-ffi",
    )


def gqa_decode_attention_persistent(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    config: GQADecodeConfig | None = None,
) -> torch.Tensor:
    """GQA decode using persistent kernel (132 blocks, block-contiguous work).

    Eliminates split-K reduction overhead by reducing partial count from
    ~hundreds to at most 132 per Q head. Best for GQA configs with few
    KV heads (e.g. Llama 8B: 8 KV heads, Qwen 2.5: 4 KV heads).
    """
    if q.dim() != 2 or k_cache.dim() != 3 or v_cache.dim() != 3:
        raise ValueError(
            "Expected q=(num_q_heads, head_dim), k/v=(num_kv_heads, seq_len, head_dim)"
        )
    if q.device.type != "cuda":
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

    if config is None:
        config = GQADecodeConfig()

    group_size = num_q_heads // num_kv_heads
    dtype = torch2cute_dtype_map[q.dtype]
    block_seq = config.block_seq

    # Work distribution: block-contiguous, kv-head-major ordering
    tiles_per_head = math.ceil(seq_len / block_seq)
    total_work_items = num_kv_heads * tiles_per_head
    items_per_block = math.ceil(total_work_items / NUM_SMS_H100_SXM)

    # Partial buffers: [num_q_heads * NUM_BLOCKS, head_dim]
    # Unwritten entries stay at init values (max=-inf, sum=0, out=0)
    # and contribute nothing to the reduction.
    num_blocks = NUM_SMS_H100_SXM
    total_partials = num_q_heads * num_blocks
    partial_out = torch.zeros(total_partials, head_dim, dtype=q.dtype, device=q.device)
    partial_max = torch.full(
        (total_partials, head_dim), float("-inf"), dtype=torch.float32, device=q.device
    )
    partial_sum = torch.zeros(
        total_partials, head_dim, dtype=torch.float32, device=q.device
    )

    compiled = _compile_persistent(
        dtype, head_dim, group_size, block_seq, items_per_block
    )
    compiled(
        q,
        k_cache,
        v_cache,
        partial_out,
        partial_max,
        partial_sum,
        num_kv_heads,
        num_q_heads,
        seq_len,
        tiles_per_head,
        total_work_items,
    )

    # Reduce: at most 132 partials per Q head (vs hundreds in split-K)
    out = torch.empty(num_q_heads, head_dim, dtype=q.dtype, device=q.device)
    compiled_red = _compile_reduction(dtype, head_dim, num_blocks)
    compiled_red(partial_out, partial_max, partial_sum, out, num_q_heads, num_blocks)
    return out


# ---------------------------------------------------------------------------
# FP8 KV Cache kernels
# ---------------------------------------------------------------------------


class FP8DirectDecodeKernel:
    """Direct decode kernel for FP8 KV cache with per-head dequantization."""

    def __init__(
        self,
        kv_dtype,
        q_dtype,
        head_dim: int,
        group_size: int,
        block_seq: int,
        num_splits: int = 1,
    ):
        if head_dim not in (64, 128):
            raise ValueError(f"Unsupported head_dim={head_dim}; expected 64 or 128")
        self.kv_dtype = kv_dtype
        self.q_dtype = q_dtype
        self.head_dim = head_dim
        self.group_size = group_size
        self.block_seq = block_seq
        self.num_splits = num_splits
        self.vec = head_dim // WARP_SIZE
        self.scale = float(head_dim**-0.5)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        mO: cute.Tensor,
        stream=None,
    ):
        tiled_q = copy_utils.tiled_copy_2d(self.q_dtype, 32, 32, self.vec)
        tiled_kv = copy_utils.tiled_copy_2d(self.kv_dtype, 32, 32, self.vec)
        self.kernel(mQ, mK, mV, mKScale, mVScale, mO, tiled_q, tiled_kv).launch(
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
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        mO: cute.Tensor,
        tiled_q: cute.TiledCopy,
        tiled_kv: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        kv_head, split_idx, _ = cute.arch.block_idx()
        thr_q = tiled_q.get_slice(tidx)
        thr_kv = tiled_kv.get_slice(tidx)

        # Load per-head scales once
        k_scale_val = mKScale[kv_head]
        v_scale_val = mVScale[kv_head]

        q_tile = cute.local_tile(mQ, (self.group_size, self.head_dim), (kv_head, 0))
        o_tile = cute.local_tile(mO, (self.group_size, self.head_dim), (kv_head, 0))

        q_regs = []
        acc_regs = []
        running_max = []
        running_sum = []

        for h in cutlass.range_constexpr(self.group_size):
            q_row = cute.local_tile(q_tile, (1, self.head_dim), (h, 0))
            tQgQ = thr_q.partition_S(q_row)
            tQr = cute.make_rmem_tensor_like(tQgQ)
            cute.copy(tiled_q, tQgQ, tQr)
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
                k_tile = cute.local_tile(
                    k_head, (self.block_seq, self.head_dim), (tile_idx, 0)
                )
                v_tile = cute.local_tile(
                    v_head, (self.block_seq, self.head_dim), (tile_idx, 0)
                )
                tile_start = tile_idx * self.block_seq
                for row in cutlass.range_constexpr(self.block_seq):
                    seq_idx = tile_start + row
                    if seq_idx < seq_len:
                        k_row = cute.local_tile(k_tile, (1, self.head_dim), (row, 0))
                        v_row = cute.local_tile(v_tile, (1, self.head_dim), (row, 0))
                        tKgK = thr_kv.partition_S(k_row)
                        tVgV = thr_kv.partition_S(v_row)
                        tKr = cute.make_rmem_tensor_like(tKgK)
                        tVr = cute.make_rmem_tensor_like(tVgV)
                        cute.copy(tiled_kv, tKgK, tKr)
                        cute.copy(tiled_kv, tVgV, tVr)
                        # FP8 → FP32 + dequant
                        k_vec = tKr.load().to(Float32) * k_scale_val
                        v_vec = tVr.load().to(Float32) * v_scale_val

                        for h in cutlass.range_constexpr(self.group_size):
                            score = (q_regs[h] * k_vec).reduce(
                                cute.ReductionOp.ADD,
                                init_val=0.0,
                                reduction_profile=0,
                            )
                            score = (
                                cute.arch.warp_reduction(score, operator.add)
                                * self.scale
                            )
                            max_next = cute.arch.fmax(running_max[h], score)
                            old_scale = cute.math.exp2(
                                (running_max[h] - max_next) * LOG2_E,
                                fastmath=True,
                            )
                            score_scale = cute.math.exp2(
                                (score - max_next) * LOG2_E, fastmath=True
                            )
                            running_max[h] = max_next
                            running_sum[h] = running_sum[h] * old_scale + score_scale
                            acc_regs[h] = acc_regs[h] * old_scale + v_vec * score_scale

        for h in cutlass.range_constexpr(self.group_size):
            out_row = cute.local_tile(o_tile, (1, self.head_dim), (h, 0))
            tOgO = thr_q.partition_D(out_row)
            tOrO = cute.make_rmem_tensor_like(tOgO)
            out = acc_regs[h] * cute.arch.rcp_approx(running_sum[h])
            tOrO.store(out.to(tOrO.element_type))
            copy_utils.copy(tOrO, tOgO)


class FP8SplitKDecodeKernel:
    """Split-K kernel for FP8 KV cache with SMEM staging and per-head dequantization."""

    def __init__(
        self,
        kv_dtype,
        q_dtype,
        head_dim: int,
        group_size: int,
        block_seq: int,
        num_splits: int,
    ):
        if head_dim not in (64, 128):
            raise ValueError(f"Unsupported head_dim={head_dim}; expected 64 or 128")
        self.kv_dtype = kv_dtype
        self.q_dtype = q_dtype
        self.head_dim = head_dim
        self.group_size = group_size
        self.block_seq = block_seq
        self.num_splits = num_splits
        self.vec = head_dim // WARP_SIZE
        self.scale = float(head_dim**-0.5)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        mPartialO: cute.Tensor,
        mPartialMax: cute.Tensor,
        mPartialSum: cute.Tensor,
        stream=None,
    ):
        tiled_q = copy_utils.tiled_copy_2d(self.q_dtype, 32, 32, self.vec)
        tiled_kv = copy_utils.tiled_copy_2d(self.kv_dtype, 32, 32, self.vec)
        tiled_f32 = copy_utils.tiled_copy_2d(Float32, 32, 32, self.vec)
        self.kernel(
            mQ,
            mK,
            mV,
            mKScale,
            mVScale,
            mPartialO,
            mPartialMax,
            mPartialSum,
            tiled_q,
            tiled_kv,
            tiled_f32,
        ).launch(
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
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        mPartialO: cute.Tensor,
        mPartialMax: cute.Tensor,
        mPartialSum: cute.Tensor,
        tiled_q: cute.TiledCopy,
        tiled_kv: cute.TiledCopy,
        tiled_f32: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        kv_head, split_idx, _ = cute.arch.block_idx()
        thr_q = tiled_q.get_slice(tidx)
        thr_kv = tiled_kv.get_slice(tidx)
        thr_f32 = tiled_f32.get_slice(tidx)

        # Load per-head scales once
        k_scale_val = mKScale[kv_head]
        v_scale_val = mVScale[kv_head]

        q_tile = cute.local_tile(mQ, (self.group_size, self.head_dim), (kv_head, 0))

        q_regs = []
        acc_regs = []
        running_max = []
        running_sum = []

        for h in cutlass.range_constexpr(self.group_size):
            q_row = cute.local_tile(q_tile, (1, self.head_dim), (h, 0))
            tQgQ = thr_q.partition_S(q_row)
            tQr = cute.make_rmem_tensor_like(tQgQ)
            cute.copy(tiled_q, tQgQ, tQr)
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

        # Allocate SMEM for FP8 K/V tile staging (1 byte/element = half of BF16)
        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_layout(
            (self.block_seq, self.head_dim),
            stride=(self.head_dim + SMEM_PAD, 1),
        )
        sK = smem.allocate_tensor(self.kv_dtype, smem_layout, byte_alignment=16)
        sV = smem.allocate_tensor(self.kv_dtype, smem_layout, byte_alignment=16)

        # Pre-partition SMEM as copy destination
        tKsK = thr_kv.partition_D(sK)
        tVsV = thr_kv.partition_D(sV)

        for local_tile_idx in cutlass.range(tiles_per_split, unroll=1):
            tile_idx = first_tile + local_tile_idx
            if tile_idx < total_tiles:
                k_tile = cute.local_tile(
                    k_head, (self.block_seq, self.head_dim), (tile_idx, 0)
                )
                v_tile = cute.local_tile(
                    v_head, (self.block_seq, self.head_dim), (tile_idx, 0)
                )
                tile_start = tile_idx * self.block_seq

                # Bulk async copy K and V tiles from global to SMEM
                tKgK = thr_kv.partition_S(k_tile)
                tVgV = thr_kv.partition_S(v_tile)
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
                        tKsKr = thr_kv.partition_S(sK_row)
                        tVsVr = thr_kv.partition_S(sV_row)
                        tKr = cute.make_rmem_tensor_like(tKsKr)
                        tVr = cute.make_rmem_tensor_like(tVsVr)
                        cute.copy(tiled_kv, tKsKr, tKr)
                        cute.copy(tiled_kv, tVsVr, tVr)
                        # FP8 → FP32 + dequant
                        k_vec = tKr.load().to(Float32) * k_scale_val
                        v_vec = tVr.load().to(Float32) * v_scale_val

                        for h in cutlass.range_constexpr(self.group_size):
                            score = (q_regs[h] * k_vec).reduce(
                                cute.ReductionOp.ADD,
                                init_val=0.0,
                                reduction_profile=0,
                            )
                            score = (
                                cute.arch.warp_reduction(score, operator.add)
                                * self.scale
                            )
                            max_next = cute.arch.fmax(running_max[h], score)
                            old_scale = cute.math.exp2(
                                (running_max[h] - max_next) * LOG2_E,
                                fastmath=True,
                            )
                            score_scale = cute.math.exp2(
                                (score - max_next) * LOG2_E, fastmath=True
                            )
                            running_max[h] = max_next
                            running_sum[h] = running_sum[h] * old_scale + score_scale
                            acc_regs[h] = acc_regs[h] * old_scale + v_vec * score_scale

        # Write partial results for split-K reduction (same format as BF16)
        for h in cutlass.range_constexpr(self.group_size):
            q_head_global = kv_head * self.group_size + h
            partial_row = q_head_global * self.num_splits + split_idx

            # Unnormalized accumulator (BF16 output dtype)
            po_row = cute.local_tile(mPartialO, (1, self.head_dim), (partial_row, 0))
            tOgO = thr_q.partition_D(po_row)
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
def _compile_fp8_decode(
    kv_dtype,
    q_dtype,
    head_dim: int,
    group_size: int,
    block_seq: int,
    num_splits: int = 1,
):
    q_heads = cute.sym_int()
    kv_heads = cute.sym_int()
    seq_len = cute.sym_int()
    q_div = 16 // (q_dtype.width // 8)
    kv_div = 16 // (kv_dtype.width // 8)
    q_cute = make_fake_tensor(q_dtype, (q_heads, head_dim), divisibility=q_div)
    k_cute = make_fake_tensor(
        kv_dtype, (kv_heads, seq_len, head_dim), divisibility=kv_div
    )
    v_cute = make_fake_tensor(
        kv_dtype, (kv_heads, seq_len, head_dim), divisibility=kv_div
    )
    ks_cute = make_fake_tensor(Float32, (kv_heads,), divisibility=1)
    vs_cute = make_fake_tensor(Float32, (kv_heads,), divisibility=1)
    o_cute = make_fake_tensor(q_dtype, (q_heads, head_dim), divisibility=q_div)
    kernel = FP8DirectDecodeKernel(
        kv_dtype,
        q_dtype,
        head_dim=head_dim,
        group_size=group_size,
        block_seq=block_seq,
        num_splits=num_splits,
    )
    return cute.compile(
        kernel,
        q_cute,
        k_cute,
        v_cute,
        ks_cute,
        vs_cute,
        o_cute,
        CUDA_STREAM,
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=None)
def _compile_fp8_splitk(
    kv_dtype, q_dtype, head_dim: int, group_size: int, block_seq: int, num_splits: int
):
    q_heads = cute.sym_int()
    kv_heads = cute.sym_int()
    seq_len = cute.sym_int()
    total_partials = cute.sym_int()
    q_div = 16 // (q_dtype.width // 8)
    kv_div = 16 // (kv_dtype.width // 8)
    q_cute = make_fake_tensor(q_dtype, (q_heads, head_dim), divisibility=q_div)
    k_cute = make_fake_tensor(
        kv_dtype, (kv_heads, seq_len, head_dim), divisibility=kv_div
    )
    v_cute = make_fake_tensor(
        kv_dtype, (kv_heads, seq_len, head_dim), divisibility=kv_div
    )
    ks_cute = make_fake_tensor(Float32, (kv_heads,), divisibility=1)
    vs_cute = make_fake_tensor(Float32, (kv_heads,), divisibility=1)
    po_cute = make_fake_tensor(q_dtype, (total_partials, head_dim), divisibility=q_div)
    pm_cute = make_fake_tensor(
        Float32, (total_partials, head_dim), divisibility=16 // (Float32.width // 8)
    )
    ps_cute = make_fake_tensor(
        Float32, (total_partials, head_dim), divisibility=16 // (Float32.width // 8)
    )
    kernel = FP8SplitKDecodeKernel(
        kv_dtype,
        q_dtype,
        head_dim=head_dim,
        group_size=group_size,
        block_seq=block_seq,
        num_splits=num_splits,
    )
    return cute.compile(
        kernel,
        q_cute,
        k_cute,
        v_cute,
        ks_cute,
        vs_cute,
        po_cute,
        pm_cute,
        ps_cute,
        CUDA_STREAM,
        options="--enable-tvm-ffi",
    )


def gqa_decode_attention_fp8(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    v_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    num_splits: int | None = None,
    config: GQADecodeConfig | None = None,
    reduction: str = "fused",
) -> torch.Tensor:
    """GQA decode attention with FP8 E4M3 KV cache.

    Args:
        q:       [num_q_heads, head_dim] BF16
        k_fp8:   [num_kv_heads, seq_len, head_dim] FP8 E4M3
        v_fp8:   [num_kv_heads, seq_len, head_dim] FP8 E4M3
        k_scale: [num_kv_heads] FP32
        v_scale: [num_kv_heads] FP32
    Returns:
        [num_q_heads, head_dim] BF16
    """
    if q.dim() != 2 or k_fp8.dim() != 3 or v_fp8.dim() != 3:
        raise ValueError(
            "Expected q=(num_q_heads, head_dim), k/v=(num_kv_heads, seq_len, head_dim)"
        )
    if q.device.type != "cuda":
        raise ValueError("All tensors must be CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Q must be fp16 or bf16")
    if k_fp8.dtype != torch.float8_e4m3fn or v_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("K/V must be float8_e4m3fn")
    if k_fp8.shape != v_fp8.shape:
        raise ValueError("k and v must have the same shape")
    if not (q.is_contiguous() and k_fp8.is_contiguous() and v_fp8.is_contiguous()):
        raise ValueError("Q/K/V must be contiguous")

    num_q_heads, head_dim = q.shape
    num_kv_heads, seq_len, kv_head_dim = k_fp8.shape
    if kv_head_dim != head_dim:
        raise ValueError("Q/K/V head_dim mismatch")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if k_scale.shape != (num_kv_heads,) or v_scale.shape != (num_kv_heads,):
        raise ValueError("Scales must be [num_kv_heads]")

    if config is None:
        config = GQADecodeConfig()
    group_size = num_q_heads // num_kv_heads
    q_dtype = torch2cute_dtype_map[q.dtype]
    kv_dtype = Float8E4M3FN

    # IMPORTANT: Use the SAME split heuristic as BF16 — do NOT let smaller
    # FP8 SMEM cause more splits (that adds reduction overhead).
    if num_splits is None:
        actual_splits = select_num_splits(
            num_kv_heads,
            seq_len,
            config.block_seq,
            config.target_blocks_per_sm,
        )
    else:
        actual_splits = num_splits

    if actual_splits == 1:
        out = torch.empty_like(q)
        compiled = _compile_fp8_decode(
            kv_dtype,
            q_dtype,
            head_dim,
            group_size,
            config.block_seq,
            num_splits=1,
        )
        compiled(q, k_fp8, v_fp8, k_scale, v_scale, out)
        return out

    total_partials = num_q_heads * actual_splits
    partial_out = torch.empty(total_partials, head_dim, dtype=q.dtype, device=q.device)
    partial_max = torch.empty(
        total_partials, head_dim, dtype=torch.float32, device=q.device
    )
    partial_sum = torch.empty(
        total_partials, head_dim, dtype=torch.float32, device=q.device
    )

    compiled = _compile_fp8_splitk(
        kv_dtype,
        q_dtype,
        head_dim,
        group_size,
        config.block_seq,
        actual_splits,
    )
    compiled(q, k_fp8, v_fp8, k_scale, v_scale, partial_out, partial_max, partial_sum)

    if reduction == "python":
        return reduce_partials(
            partial_out,
            partial_max,
            partial_sum,
            num_q_heads,
            actual_splits,
            head_dim,
        )

    # Fused CuTe-DSL reduction
    out = torch.empty(num_q_heads, head_dim, dtype=q.dtype, device=q.device)
    compiled_red = _compile_reduction(q_dtype, head_dim, actual_splits)
    compiled_red(partial_out, partial_max, partial_sum, out, num_q_heads, actual_splits)
    return out
