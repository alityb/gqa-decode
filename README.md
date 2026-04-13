# GQA Decode Attention Kernel

Single-warp CuTe-DSL decode attention experiments for H100 SM90, with a PyTorch
reference path, a split-K partial kernel, correctness tests, and benchmark
scripts.

## Layout

- `gqa_decode/`: public API and kernel implementation
- `tests/`: correctness and edge-case coverage
- `benchmarks/`: achieved-bandwidth and FlashInfer comparison scripts
- `analysis/roofline.py`: theoretical bandwidth floor

## Environment

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e './quack[dev,cu13]'
```

## Usage

```python
import torch
from gqa_decode import gqa_decode_attention

q = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(8, 16384, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(8, 16384, 128, device="cuda", dtype=torch.bfloat16)
o = gqa_decode_attention(q, k, v)
```
