from __future__ import annotations

import torch

from gqa_decode import gqa_decode_attention


HBM_PEAK_BW = 3.35e12


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


def run_comparison():
    try:
        import flashinfer

        has_fi = True
    except ImportError:
        print("FlashInfer not installed — skipping comparison")
        has_fi = False

    configs = [
        ("Llama3-8B", 32, 8, 128),
        ("Llama3-70B", 64, 8, 128),
        ("Qwen2.5-7B", 28, 4, 128),
        ("MHA", 32, 32, 128),
    ]
    seq_lens = [4096, 16384, 32768, 65536]

    print(
        f"{'Config':<14} {'SeqLen':>7} {'Ours(us)':>9} {'FI(us)':>9} "
        f"{'Ours BW%':>9} {'FI BW%':>9} {'Speedup':>8}"
    )
    print("-" * 75)

    for name, nq, nkv, d in configs:
        for seq in seq_lens:
            q_ours = torch.randn(nq, d, device="cuda", dtype=torch.bfloat16)
            k_ours = torch.randn(nkv, seq, d, device="cuda", dtype=torch.bfloat16)
            v_ours = torch.randn(nkv, seq, d, device="cuda", dtype=torch.bfloat16)

            ours_us = bench(lambda: gqa_decode_attention(q_ours, k_ours, v_ours, backend="cute"))
            kv_bytes = 2 * nkv * seq * d * q_ours.element_size()
            ours_bw = kv_bytes / (ours_us / 1e6) / HBM_PEAK_BW * 100

            if has_fi:
                q_fi = torch.randn(nq, d, device="cuda", dtype=torch.float16)
                k_fi = torch.randn(seq, nkv, d, device="cuda", dtype=torch.float16)
                v_fi = torch.randn(seq, nkv, d, device="cuda", dtype=torch.float16)
                try:
                    fi_us = bench(lambda: flashinfer.single_decode_with_kv_cache(q_fi, k_fi, v_fi))
                    kv_bytes_fi = 2 * nkv * seq * d * q_fi.element_size()
                    fi_bw = kv_bytes_fi / (fi_us / 1e6) / HBM_PEAK_BW * 100
                    speedup = fi_us / ours_us
                except Exception:
                    fi_us = float("nan")
                    fi_bw = float("nan")
                    speedup = float("nan")
            else:
                fi_us = float("nan")
                fi_bw = float("nan")
                speedup = float("nan")

            print(
                f"{name:<14} {seq:>7} {ours_us:>8.1f} {fi_us:>8.1f} "
                f"{ours_bw:>8.1f}% {fi_bw:>8.1f}% {speedup:>7.2f}x"
            )
        print()


if __name__ == "__main__":
    run_comparison()
