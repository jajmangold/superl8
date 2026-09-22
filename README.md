# SuperL8

**The fastest way to run INT8 inference on Volta GPUs.**

SuperL8 is a collection of hand-written DP4A FlashAttention-2 and GEMM kernels that exploit a quirk of NVIDIA hardware: on Volta GPUs where tensor cores are disabled or slow, integer `__dp4a` on CUDA cores is **6.7x faster**. If you're running inference on a CMP 100-210, a V100 with disabled tensor cores, or any sm_70 card where fp16 throughput is disappointing — this is the math library that makes it fast.

## The hardware insight

| Math Path | Real V100 | CMP 100-210 |
| --- | --: | --: |
| FP16 tensor core (HMMA) | ~112 TFLOP/s | 6.9 TFLOP/s |
| INT8 `__dp4a` (CUDA core) | ~63 TOP/s | **46 TOP/s** |

On a CMP 100-210, `__dp4a` is the only fast INT8 matmul primitive. IMMA (INT8 tensor cores) doesn't exist until Turing. SuperL8 builds an entire inference stack on top of this instruction — attention, linear layers, quantization, and a custom weight format designed for zero-copy loading.

## What you get

- **W8A8 DP4A FlashAttention-2** — forward, backward (gradient-checked), causal, GQA/MQA, head dims 32/64/128/256, varlen prefill, split-KV decode with int8 KV cache, and cross-attention for diffusion.
- **W8A8 / W4A8 GEMM** — int8 dp4a linear layers for inference and training, plus grouped GEMM for MoE (every active expert in one launch).
- **GGUF native-fused kernels** — Q2_K through Q6_K and IQ quant types decoded in-kernel with dp4a (no dequant-to-int8 round-trip).
- **Hybrid linear attention** — Gated-DeltaNet, Lightning, and MLA (DeepSeek-V2/V3) int8 decode kernels.
- **Quantization** — symmetric per-row int8, optional Hadamard rotation for outlier-heavy activations, per-channel K-smoothing.
- **`.superl8` weight format** — on-disk bytes are the resident dp4a VRAM layout (`mmap + cudaMemcpy`, no dequant or repack). Load weights at memory speed.

## Quick start

```bash
pip install https://github.com/jajmangold/superl8/releases/download/v0.1.0/superl8-0.1.0-cp312-cp312-linux_x86_64.whl
```

```python
import torch
import superl8

q = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1,  8, 4096, 128, device="cuda", dtype=torch.float16)
v = torch.randn(1,  8, 4096, 128, device="cuda", dtype=torch.float16)

# INT8 FlashAttention-2 forward (quantizes internally)
out = superl8.attn_int8_fwd(q, k, v, causal=True)
```

Requires Python 3.12, CUDA 12.9, and a Volta-capable GPU (sm_70).

## Performance

Tested on a CMP 100-210 (V100-labelled fleet card):

| Model | Metric | Value |
|---|---|---|
| Qwen3.6-27B Q3_K_S | Steady decode | 21.3 tok/s |
| Qwen3.6-27B Q3_K_S | End-to-end | 7.2 tok/s |
| Qwen3.6-27B Q3_K_S | Peak VRAM | 14.3 GiB |

See `bench/qwen3-scoreboard.json` for the full provenance-locked benchmark data.

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

## Related repos

- [**SuperL8 Serve**](https://github.com/jajmangold/superl8-serve) — OpenAI-compatible inference server built on SuperL8. Continuous batching, CUDA graphs, paged KV cache, speculative decode.
- [**ComfyUI-SuperL8**](https://github.com/jajmangold/ComfyUI-superl8) — ComfyUI nodes for INT8 quantized diffusion DiTs. SQNR accuracy gating, GGUF loading, multi-GPU.

## License

BSD-3-Clause. See [LICENSE](LICENSE).

Quantization approach informed by SDNQ and SageAttention. The 3-bit Lloyd-Max KV quantizer and transport bridge are informed by [qengine](https://github.com/Haru-neo/qengine) (Apache-2.0).
