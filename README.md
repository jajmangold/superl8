# SuperL8

INT8 DP4A FlashAttention-2 and GEMM kernels for NVIDIA Volta GPUs.

## Why dp4a?

The NVIDIA CMP 100-210 (GV100 silicon) has firmware-disabled fp16 tensor cores. On
this card, pure integer `__dp4a` on the CUDA cores is ~6.7x faster than fp16 tensor
core math. sm_70 has no INT8 tensor cores either (IMMA starts at Turing), so `__dp4a`
is the only fast INT8 matmul primitive — and int8 also halves K/V cache footprint.

| Path | Real V100 | CMP 100-210 |
| --- | --: | --: |
| FP16 tensor core (HMMA) | ~112 TFLOP/s | 6.9 TFLOP/s |
| INT8 `__dp4a` (CUDA core) | ~63 TOP/s | 46 TOP/s |

## Features

- **W8A8 DP4A FlashAttention-2** — forward, backward (gradient-checked), causal,
  GQA/MQA, head dims 32/64/128/256, varlen prefill, split-KV decode with int8 KV
  cache, and non-causal / cross-attention for diffusion.
- **W8A8 / W4A8 GEMM** — int8 dp4a linear layers for inference and training, plus a
  grouped GEMM for MoE (runs every active expert in one launch).
- **GGUF native-fused kernels** — Q2_K through Q6_K and IQ quant types decoded
  in-kernel with dp4a (no dequant-to-int8 round-trip).
- **Hybrid linear attention** — Gated-DeltaNet, Lightning, and MLA (DeepSeek-V2/V3)
  int8 decode kernels.
- **Quantization** — symmetric per-row int8, optional Hadamard rotation for
  outlier-heavy activations, and per-channel K-smoothing.
- **`.superl8` weight format** — on-disk bytes are the resident dp4a VRAM layout
  (`mmap + cudaMemcpy`, no dequant or repack).

## Install

```bash
pip install https://github.com/jajmangold/superl8/releases/download/v0.1.0/superl8-0.1.0-cp312-cp312-linux_x86_64.whl
```

Requires Python 3.12, CUDA 12.9, and a Volta-capable GPU (sm_70).

```python
import torch
import superl8

q = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1,  8, 4096, 128, device="cuda", dtype=torch.float16)
v = torch.randn(1,  8, 4096, 128, device="cuda", dtype=torch.float16)

# INT8 FlashAttention-2 forward (quantizes internally)
out = superl8.attn_int8_fwd(q, k, v, causal=True)
```

## Supported models

Tested with Qwen, Llama, Gemma, and DeepSeek-family models. Any model with
standard transformer attention layers works through the generic linear/attention
APIs. GGUF quantized weights (Q2_K–Q6_K, IQ types) are natively fitted.

## Build from source

```bash
git clone https://github.com/jajmangold/superl8.git
cd superl8
pip install -e ".[dev]"
```

## Tests

```bash
pytest tests/ -ra
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).

Quantization approach informed by SDNQ and SageAttention. The 3-bit Lloyd-Max KV
quantizer and transport bridge are informed by
[qengine](https://github.com/Haru-neo/qengine) (Apache-2.0).
