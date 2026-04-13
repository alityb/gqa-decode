"""FP8 KV cache benchmark: our FP8 vs our BF16 vs FlashInfer."""

from __future__ import annotations

import torch

from gqa_decode import gqa_decode_attention
from gqa_decode.gqa_decode import gqa_decode_attention_fp8
from gqa_decode.fp8_utils import quantize_kv_fp8


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


def main():
    try:
        import flashinfer

        has_fi = True
    except ImportError:
        print("FlashInfer not installed — skipping FlashInfer columns")
        has_fi = False

    configs = [
        ("Llama3-8B", 32, 8, 128),
        ("Llama3-70B", 64, 8, 128),
        ("Qwen2.5-7B", 28, 4, 128),
        ("MHA", 32, 32, 128),
    ]
    seq_lens = [4096, 16384, 32768, 65536]

    print(
        f"{'Config':<14} {'SeqLen':>7} "
        f"{'FP8(us)':>9} {'BF16(us)':>9} {'FP8/BF16':>9} "
        f"{'FI-BF16':>9} {'FP8vsFI':>9}"
    )
    print("-" * 85)

    for name, nq, nkv, d in configs:
        for seq in seq_lens:
            # BF16 tensors
            q = torch.randn(nq, d, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(nkv, seq, d, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(nkv, seq, d, device="cuda", dtype=torch.bfloat16)

            # FP8 tensors
            k_fp8, v_fp8, k_scale, v_scale = quantize_kv_fp8(k, v)

            # Our BF16
            bf16_us = bench(lambda: gqa_decode_attention(q, k, v))

            # Our FP8
            fp8_us = bench(
                lambda: gqa_decode_attention_fp8(q, k_fp8, v_fp8, k_scale, v_scale)
            )

            fp8_speedup = bf16_us / fp8_us

            # FlashInfer BF16
            fi_us = float("nan")
            fp8_vs_fi = float("nan")
            if has_fi:
                q_fi = q.to(torch.float16)
                k_fi = torch.randn(seq, nkv, d, device="cuda", dtype=torch.float16)
                v_fi = torch.randn(seq, nkv, d, device="cuda", dtype=torch.float16)
                try:
                    fi_us = bench(
                        lambda: flashinfer.single_decode_with_kv_cache(q_fi, k_fi, v_fi)
                    )
                    fp8_vs_fi = fi_us / fp8_us
                except Exception:
                    pass

            print(
                f"{name:<14} {seq:>7} "
                f"{fp8_us:>8.1f} {bf16_us:>8.1f} {fp8_speedup:>8.2f}x "
                f"{fi_us:>8.1f} {fp8_vs_fi:>8.2f}x"
            )
        print()

    # Also check if FlashInfer supports FP8 decode
    if has_fi:
        print("--- FlashInfer FP8 support check ---")
        try:
            q_test = torch.randn(32, 128, device="cuda", dtype=torch.float16)
            k_test = torch.randn(1024, 32, 128, device="cuda").to(torch.float8_e4m3fn)
            v_test = torch.randn(1024, 32, 128, device="cuda").to(torch.float8_e4m3fn)
            out = flashinfer.single_decode_with_kv_cache(q_test, k_test, v_test)
            print(f"FlashInfer FP8 decode works: {out.shape}")

            # Benchmark it
            fi_fp8_us = bench(
                lambda: flashinfer.single_decode_with_kv_cache(q_test, k_test, v_test)
            )
            print(f"FlashInfer FP8 1K seq: {fi_fp8_us:.1f}us")
        except Exception as e:
            print(f"FlashInfer FP8 decode failed: {e}")


if __name__ == "__main__":
    main()
