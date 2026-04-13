from __future__ import annotations

import argparse
import time

import torch

from gqa_decode import gqa_decode_attention


HBM_PEAK_BW = 3.35e12
CONFIGS = [
    ("Llama3-8B", 32, 8, 128),
    ("Llama3-70B", 64, 8, 128),
    ("Qwen2.5-7B", 28, 4, 128),
    ("MHA", 32, 32, 128),
]
SEQ_LENS = [1024, 4096, 16384, 32768, 65536]


def benchmark_config(kernel_fn, num_q_heads, num_kv_heads, seq_len, head_dim, dtype, warmup, iters):
    q = torch.randn(num_q_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
    for _ in range(warmup):
        kernel_fn(q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        kernel_fn(q, k, v)
    end.record()
    torch.cuda.synchronize()

    time_us = start.elapsed_time(end) / iters * 1000
    kv_bytes = 2 * num_kv_heads * seq_len * head_dim * q.element_size()
    bw = kv_bytes / (time_us / 1e6)
    util = bw / HBM_PEAK_BW * 100
    return time_us, bw, util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    for name, num_q_heads, num_kv_heads, head_dim in CONFIGS:
        print(name)
        for seq_len in SEQ_LENS:
            time_us, bw, util = benchmark_config(
                gqa_decode_attention,
                num_q_heads,
                num_kv_heads,
                seq_len,
                head_dim,
                dtype,
                args.warmup,
                args.iters,
            )
            print(
                f"  {num_q_heads}qh/{num_kv_heads}kvh seq={seq_len:>6} d={head_dim}"
                f"  {time_us:>8.1f}us  {bw/1e12:.3f} TB/s  {util:.1f}%"
            )


if __name__ == "__main__":
    main()
