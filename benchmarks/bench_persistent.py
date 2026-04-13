"""Benchmark: persistent vs split-K vs FlashInfer."""

from __future__ import annotations

import torch

from gqa_decode import gqa_decode_attention, gqa_decode_attention_persistent


HBM_PEAK_BW = 3.35e12

CONFIGS = [
    ("Llama3-8B", 32, 8, 128),
    ("Llama3-70B", 64, 8, 128),
    ("Qwen2.5-7B", 28, 4, 128),
    ("MHA", 32, 32, 128),
]
SEQ_LENS = [4096, 16384, 32768, 65536]


def bench(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


def bw_pct(time_us, nkv, seq, d, elem_bytes):
    kv_bytes = 2 * nkv * seq * d * elem_bytes
    return kv_bytes / (time_us / 1e6) / HBM_PEAK_BW * 100


def main():
    try:
        import flashinfer

        has_fi = True
    except ImportError:
        print("FlashInfer not installed")
        has_fi = False

    print(
        f"{'Config':<14} {'Seq':>7} "
        f"{'Persist(us)':>12} {'SplitK(us)':>11} {'P/SK':>6} "
        f"{'FI(us)':>8} {'P/FI':>6} "
        f"{'P BW%':>7} {'SK BW%':>7}"
    )
    print("-" * 95)

    for name, nq, nkv, d in CONFIGS:
        for seq in SEQ_LENS:
            q = torch.randn(nq, d, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(nkv, seq, d, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(nkv, seq, d, device="cuda", dtype=torch.bfloat16)

            persist_us = bench(lambda: gqa_decode_attention_persistent(q, k, v))
            splitk_us = bench(lambda: gqa_decode_attention(q, k, v))

            elem = q.element_size()
            p_bw = bw_pct(persist_us, nkv, seq, d, elem)
            sk_bw = bw_pct(splitk_us, nkv, seq, d, elem)
            p_vs_sk = splitk_us / persist_us

            fi_us = float("nan")
            p_vs_fi = float("nan")
            if has_fi:
                q_fi = torch.randn(nq, d, device="cuda", dtype=torch.float16)
                k_fi = torch.randn(seq, nkv, d, device="cuda", dtype=torch.float16)
                v_fi = torch.randn(seq, nkv, d, device="cuda", dtype=torch.float16)
                try:
                    fi_us = bench(
                        lambda: flashinfer.single_decode_with_kv_cache(q_fi, k_fi, v_fi)
                    )
                    p_vs_fi = fi_us / persist_us
                except Exception:
                    pass

            print(
                f"{name:<14} {seq:>7} "
                f"{persist_us:>11.1f} {splitk_us:>10.1f} {p_vs_sk:>5.2f}x "
                f"{fi_us:>7.1f} {p_vs_fi:>5.2f}x "
                f"{p_bw:>6.1f}% {sk_bw:>6.1f}%"
            )
        print()


if __name__ == "__main__":
    main()
