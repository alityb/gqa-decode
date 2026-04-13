from __future__ import annotations

HBM_PEAK_BW = 3.35e12
CONFIGS = [
    ("Llama3-8B", 32, 8, 128),
    ("Llama3-70B", 64, 8, 128),
    ("Qwen2.5-7B", 28, 4, 128),
    ("MHA", 32, 32, 128),
]
SEQ_LENS = [1024, 4096, 16384, 32768, 65536]


def theoretical_min_time_us(num_kv_heads: int, seq_len: int, head_dim: int, bytes_per_elem: int = 2):
    total_bytes = 2 * num_kv_heads * seq_len * head_dim * bytes_per_elem
    return total_bytes / HBM_PEAK_BW * 1e6


def main():
    for name, _, num_kv_heads, head_dim in CONFIGS:
        print(name)
        for seq_len in SEQ_LENS:
            time_us = theoretical_min_time_us(num_kv_heads, seq_len, head_dim)
            print(f"  seq={seq_len:>6}  theoretical_min={time_us:>8.2f}us")


if __name__ == "__main__":
    main()
