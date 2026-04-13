from __future__ import annotations

import argparse

import torch

from gqa_decode import gqa_decode_attention


CONFIGS = [
    ("Llama3-8B", 32, 8, 128),
    ("Llama3-70B", 64, 8, 128),
    ("Qwen2.5-7B", 28, 4, 128),
    ("MHA", 32, 32, 128),
]
SEQ_LENS = [1024, 4096, 16384, 32768, 65536]


def time_fn(fn, q, k, v, warmup, iters):
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(q, k, v)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    try:
        import flashinfer
    except ImportError as exc:
        raise SystemExit("flashinfer is not installed") from exc

    dtype = getattr(torch, args.dtype)
    for name, num_q_heads, num_kv_heads, head_dim in CONFIGS:
        print(name)
        for seq_len in SEQ_LENS:
            q = torch.randn(num_q_heads, head_dim, device="cuda", dtype=dtype)
            k = torch.randn(num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
            v = torch.randn(num_kv_heads, seq_len, head_dim, device="cuda", dtype=dtype)
            ours = time_fn(gqa_decode_attention, q, k, v, args.warmup, args.iters)
            ref = time_fn(
                lambda q_, k_, v_: flashinfer.single_decode_with_kv_cache(q_, k_, v_),
                q,
                k,
                v,
                args.warmup,
                args.iters,
            )
            print(f"  seq={seq_len:>6}  ours={ours:>8.1f}us  flashinfer={ref:>8.1f}us")


if __name__ == "__main__":
    main()
