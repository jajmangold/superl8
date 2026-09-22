# SuperL8 — Design Document

## 1. Problem Statement

### The Volta/CMP Hardware Gap

NVIDIA's CMP 100-210 and V100-with-disabled-tensor-cores represent a large deployed fleet that most inference frameworks ignore. The hardware quirk that matters:

| Math Path | Real V100 | CMP 100-210 |
|-----------|-----------|-------------|
| FP16 tensor core (HMMA) | ~112 TFLOP/s | **6.9 TFLOP/s** |
| INT8 `__dp4a` (CUDA core) | ~63 TOP/s | **46 TOP/s** |

Tensor cores on CMP cards are firmware-gimped to 6% of their V100 throughput. But the integer CUDA core pipe — specifically the `__dp4a` instruction (4-element signed int8 dot product with int32 accumulate) — runs at full speed. This isn't a rounding error: it's a **6.7× advantage** for INT8 dp4a over FP16 tensor cores on this fleet.

Standard inference frameworks (vLLM, TGI, TensorRT-LLM) assume tensor cores are the fast path. They quantize to fp16 and use HMMA. On CMP hardware, this is the slow path. SuperL8 inverts the assumption: treat `__dp4a` as the native compute primitive and build the entire stack around it.

### What doesn't work

The obvious alternatives fail on this hardware:

- **FP16 + HMMA**: 6.9 TFLOP/s on CMP. The tensor cores are intentionally rate-limited by firmware. Nothing fixes this except the integer pipe.
- **FP8**: Doesn't exist on sm_70. Requires Hopper/Ada.
- **IMMA (INT8 tensor cores)**: Doesn't exist until Turing (sm_75). Not available on Volta/CMP.
- **INT4 + dp4a**: Viable for storage (AWQ, GGUF Q4_K), but the reconstruct→dp4a path adds latency. W8A8 is the compute-optimal sweet spot for this instruction.

## 2. Architecture Decisions

### 2.1 Why dp4a?

`__dp4a` computes `a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]` on 4 int8 values with int32 accumulation, in one instruction. On sm_70, it runs on the CUDA core pipe (not tensor cores) at 46 TOP/s.

Why this is the right primitive:

1. **It's the only fast INT8 matmul on this hardware.** IMMA doesn't exist until Turing. Tensor cores are firmware-limited. The integer CUDA core pipe is intact.
2. **It composes naturally with per-row quantized weights.** Each dp4a produces a 4-element partial dot product. Tile them across output dimensions and you get a full GEMM.
3. **It handles asymmetric quantization.** The min-correction term (`xsum = dp4a(x, 0x01010101)`) is computed inline at no extra cost — the same instruction, just with a constant weight.
4. **It's already production-proven.** SuperL8 has been running dp4a GEMM and attention on real workloads for months.

### 2.2 Why W8A8 (per-row int8 + scale)?

The native weight format is:

```
weight = scale × index
```

- Index: int8 (1 byte per weight)
- Scale: fp32 (one per row)
- Reconstruction: multiply scale × index

This is not a new representation. It's the simplest possible int8 quantization. Why it wins:

1. **Near-lossless.** PPL 21.19 vs 21.22 baseline (perplexity delta < 0.15%). The Pareto frontier analysis measured this.
2. **Per-row scale adapts.** Each row gets its own scale, so the quantizer adapts to each row's distribution. No shared codebook needed.
3. **Simple to implement.** Symmetric per-row RTN (round-to-nearest). No calibration data, no entropy coding, no codebook training.
4. **Half the memory of fp16.** 1 byte vs 2 bytes per weight. A 9B model is 752 MB in W8A8 vs 1503 MB in FP16.

### 2.3 Why GGUF native loading?

GGUF files (Q2_K through Q6_K, IQ types) are the standard format for distributing quantized models. The alternative is to dequantize to int8 at load time, but:

1. **Dequantization busts memory.** A 22B model in Q4_K_M is ~13 GB resident. Requantizing to int8 would require ~21 GB, exceeding a 16 GB card.
2. **Dequantization is slow.** The fp32-dequant→matmul path takes 143s/step on real hardware. The fused dp4a kernel does the same work by unpacking k-quant super-blocks to int8 in registers and running `__dp4a` inline.
3. **Zero-transcode loading.** GGUF bytes can be mmapped straight into VRAM. The kernel reads native super-block bytes, unpacks them on the fly, and runs the dp4a dot product. No offline conversion step.

### 2.4 Why custom weight format (.superl8)?

The `.superl8` format exists for:

1. **mmap zero-copy loading.** Weights are placed in memory-mapped files. The kernel reads them directly without host-side deserialization.
2. **Shard indexing.** Multi-file sharding with CRC validation. Critical for 27B+ models that exceed single-file sizes.
3. **QKV merge optimization.** `consume_on_merge()` pops source rows during the QKV projection merge, bounding the transient memory spike so a card-filling 27B fits on one 16 GB GPU.
4. **Pre-fused weights.** The format stores already-merged QKV projections, eliminating the per-load merge overhead.

## 3. Quantization Strategy

### 3.1 Weight Quantization (W8A8)

**Method**: Symmetric per-row int8 quantization with K-smoothing and Q-outlier detection.

```
For each weight row w (fp16):
  1. Compute absmax
  2. Detect outliers (values > 6× median)
  3. If outliers: apply K-smoothing (downscale outlier channels)
  4. Scale = absmax / 127
  5. Quantized = round(w / scale) → int8
```

**SQNR gating**: Per-layer signal-to-quantization-noise ratio measured against the fp16 original. Gate: SQNR ≥ 40 dB. A layer that fails gets wider scales or is kept in fp16.

### 3.2 Activation Quantization (A8)

**Method**: Symmetric per-row int8 with fp32 per-row scale.

```
For each activation row x (fp16):
  Scale = absmax(x) / 127
  Quantized = round(x / scale) → int8
```

No stored scale — computed on the fly from the activation. This is the "A8" in W8A8.

### 3.3 KV Cache Quantization

Three formats, selected per-model:

| Format | K | V | Bytes per KV pair | Use case |
|--------|---|---|-------------------|----------|
| int8 | int8 + fp32 scale | int8 + fp32 scale | 8 bytes | Default |
| k8v8 | int8 + fp32 scale | int8 + fp32 scale | 8 bytes | Fallback for K8V3 |
| k8v3 | int8 + fp32 scale | 3-bit Lloyd-Max codes + fp32 norms | 3.875 bytes | Long-context (262k) |

**K8V3 rationale**: For models with 262k context windows, the KV cache dominates VRAM. K8V3 reduces V storage from 4 bytes to 1.875 bytes per weight (3-bit codes + per-128-token fp32 norms). The 3-bit Lloyd-Max codebook is a fixed Gaussian table (no per-model training). Quality is validated by a recall gate: cosine similarity between full-KV and evicted-KV decode logits must exceed 0.99.

**Quantize-on-write**: K/V are quantized at write time (inside `quantize_kv_write_paged`), never stored as fp16. This is a key design choice: the decode path reads int8 directly, halving the memory bandwidth bottleneck.

**Hadamard rotation**: Before quantizing K, apply a Hadamard rotation (`rotate_last`). This is an involution (self-inverse) that rotates the weight distribution to improve quantization quality on outlier-heavy activations. Applied at write, undone at dequant for eviction compaction.

### 3.4 Lloyd-Max 3-bit KV

The Lloyd-Max quantizer is a nearest-neighbor codebook quantizer:

1. **Codebook**: Fixed Gaussian table (8 entries for 3-bit), pre-computed, stored per-layer.
2. **Quantize**: Block-L2-normalize the V vector, bucketize onto the codebook, bit-pack the codes.
3. **Dequantize**: Unpack codes, look up codebook entries, multiply by block norm.

The codebook is identical across layers (a fixed Gaussian distribution). Per-layer codebooks slot into the same storage format without changes — the per-layer `v_codebook` tensor holds the fixed table replicated per layer.

### 3.5 Alternatives Considered

| Approach | Why not |
|----------|---------|
| FP16 KV cache | 2× memory, 2× bandwidth. Decode is memory-bound on KV reads. |
| FP8 KV cache | Doesn't exist on sm_70. Requires Hopper/Ada hardware. |
| NF4 KV cache | 4-bit codebook adds dequant latency. Q2_K/Q4_K storage is viable but the reconstruct→matmul path is slower than native int8 dp4a. |
| Per-channel V quant | More complex, marginal quality gain. Per-token is simpler and sufficient. |
| Entropy-coded KV | Huffman/arithmetic coding adds decode latency. The codebook lookup in Lloyd-Max is O(1). |

## 4. Tradeoffs Considered

### 4.1 dp4a vs IMMA vs FP8

| Primitive | CMP 100-210 | Real V100 | Hopper | Notes |
|-----------|-------------|-----------|--------|-------|
| dp4a (CUDA core) | **46 TOP/s** | 63 TOP/s | ~100 TOP/s | Always available on sm_70+ |
| IMMA (INT8 tensor core) | N/A | 125 TOP/s | 3958 TOP/s | Requires Turing+ (sm_75) |
| FP8 (tensor core) | N/A | N/A | 989 TFLOP/s | Requires Hopper (sm_90) |
| HMMA (FP16 tensor core) | 6.9 TFLOP/s | 112 TFLOP/s | 989 TFLOP/s | Firmware-limited on CMP |

**Decision**: dp4a is the only viable fast path on CMP. On Hopper, FP8 tensor cores would win for compute-bound workloads, but SuperL8's fleet is CMP/Volta.

### 4.2 mmap vs Streaming

| Approach | Latency | Memory | Complexity |
|----------|---------|--------|------------|
| mmap zero-copy | O(1) page fault | Demand-paged | Simple (OS handles I/O) |
| Streaming | O(model_size / bandwidth) | Fixed allocation | Complex (async I/O, double-buffering) |
| mmap + `MAP_POPULATE` | O(1) fault-free | Pre-faulted | Simple, fastest warm-up |

**Decision**: mmap with optional `MAP_POPULATE` for pre-faulting. The `.superl8` format is designed for this: shard index points to file offsets, weights are page-aligned, CRC validated at load time. Streaming is the fallback for network-mounted models (NFS, object storage) where mmap faults are unreliable.

### 4.3 Continuous vs Static Batching

| Approach | Throughput | Latency | Complexity |
|----------|-----------|---------|------------|
| Static batching | Low (waits for longest) | Bounded per batch | Simple |
| Continuous batching | **High** (admit on slot free) | Low for short requests | Moderate |
| Chunked continuous | **High** + fair decode | Low, no decode starvation | Moderate+ |

**Decision**: Continuous batching with chunked prefill. The scheduler admits new requests whenever a slot opens (max_num_seqs cap). Prefill is chunked (`chunked_prefill_size`) to prevent long prompts from starving decode of GPU time. This is the nano-vllm shape.

### 4.4 KV Cache: int8 vs Lloyd-Max 3-bit

| Format | V bytes/weight | Quality | Decode kernel |
|--------|----------------|---------|---------------|
| int8 | 4 | near-lossless | `attn_paged_decode_cached` (fused) |
| k8v3 (3-bit Lloyd-Max) | 1.875 | slightly lower | `attn_paged_decode_k8v3` (fused) OR per-slot fp16 fallback |

**Decision**: int8 default. k8v3 for long-context models (262k) where KV cache dominates VRAM. The k8v3 path has a fused kernel for head_dim ∈ {128, 256} with Lloyd block=128, and a per-slot fp16 dequant fallback for other configurations.

## 5. Kernel Design

### 5.1 The dp4a GEMM Spine

The core GEMM kernel (`gemm_dp4a`) follows a BM=BN=BK=64 tiling with 128 threads:

```
Thread grid: 16 × 8 (output tiles)
Micro-tile: TM=4 × TN=8 (per-thread output)
K-step: 32 elements (one dp4a per 4 int8, 8 dp4as per 32-element step)
Accumulator: int32, flushed to fp16 with per-group scale in epilogue
```

**Key properties**:
- Bank-conflict-free via XOR swizzle on shared memory
- Handles ragged M and N (non-tile-multiple shapes) with zero-pad guards
- K is always a multiple of 256 for quantized tensors (GGUF stores non-aligned tensors as Q8_0/F16)

### 5.2 Decode-Optimized GEMM

At M=1 (autoregressive decode), the tiled GEMM launches too few threadblocks to fill the GPU. The decode variant (`gemm_decode_dp4a`) uses warp-per-column layout:

```
One warp per output column
Full K-reduction within the warp
__shfl_xor butterfly for warp-wide reduction
GPU-saturating at M=1 (7-8× faster than tiled at decode)
```

### 5.3 Fused GGUF K-Quant Kernels

The k-quant kernels (`gemm_q{2,3,4,5,6}k`) extend the dp4a spine with:

1. **Weight staging**: Read native GGUF super-block bytes, unpack to int8 in registers/smem
2. **Sub-scale staging**: Decode per-32-element scales from the k-quant header
3. **Affine flush**: `iacc = dp4a(x, w)`, `xsum = dp4a(x, 0x01010101)`, `output = scale * iacc - min * xsum`

The math: k-quant reconstruction is a per-32-element affine dequant. For Q4_K:

```
w_k = d · sc_j · q_k − dmin · m_j
```

where `d`/`dmin` are fp16 super-block scales, `sc_j`/`m_j` are 6-bit sub-block scales, and `q_k` is the 4-bit code. The dp4a computes `Σ x̂·q` (the `sumi` term), and `dp4a(x, 0x01010101)` computes `Σ x̂` (the min-correction `xsum` term). The affine combine is two fp32 multiplies.

### 5.4 Attention Kernels

| Kernel | Path | Notes |
|--------|------|-------|
| `attn_int8_fwd` | Prefill | W8A8 DP4A FlashAttention-2. Causal, GQA/MQA. Head dims 32-256. |
| `attn_varlen_fwd` | Prefill | Variable-length batched prefill (ragged shapes). |
| `attn_paged_decode_cached` | Decode | ONE launch for entire ragged batch. Reads block table + context_lens. |
| `attn_paged_decode_k8v3` | Decode | K8V3 format: int8 K + 3-bit V fused decode. |
| `attn_tree_fwd` | Verify | Tree-structured speculative decode verification. |

## 6. Performance Characteristics

### 6.1 What's Memory-Bound

**Decode** is memory-bound, not compute-bound. The bottleneck is KV cache reads:

```
Per decode step (batch=B, context_len=N):
  KV read: B × N × num_kv_heads × head_dim × 2 (K+V) × bytes_per_element
  At B=8, N=4096, Hkv=8, D=128, int8:  8 × 4096 × 8 × 128 × 2 = 64 MB
  At 829 GB/s HBM BW: ~77 µs
```

Compare to compute:
```
Per decode step (B=8, hidden=4096, 32 layers):
  QKV GEMM: 8 × 4096 × 12288 × 2 (Q+K+V) = 805M dp4a operations
  At 46 TOP/s: ~17.5 ms total (all layers)
  Per layer: ~0.55 ms
```

**The decode bottleneck is KV read bandwidth, not GEMM compute.** This is why int8 KV cache matters — it halves the bytes read per attention head.

### 6.2 What's Compute-Bound

**Prefill** is compute-bound for long prompts:

```
Per prefill step (S=2048, hidden=4096, 32 layers):
  Attention: O(S² × H × D) — quadratic in sequence length
  QKV GEMM: S × 4096 × 12288 × 2 = 205B dp4a operations
  At 46 TOP/s: ~4.5 seconds total
  Per layer: ~140 ms
```

At S=2048, the attention kernel's O(S²) cost dominates. For S < 512, the GEMM compute dominates.

### 6.3 Dispatch Overhead (CUDA Graphs)

On a 0.6B model (28 layers), decode step breakdown without graphs:

```
GPU kernel time:   ~18 ms
Kernel launch:     ~3,200 launches × ~30 µs each = ~96 ms
Total step time:   ~131 ms
GPU utilization:   14% (86% is launch overhead)
```

With CUDA graph capture, the 3,200 launches collapse to 1 `cudaGraphLaunch`. Step time drops to ~20-25ms (dominated by actual kernel time + sampling sync).

### 6.4 Bottleneck Ladder

Priority-ordered bottlenecks for a typical workload:

1. **KV cache bandwidth** (decode): The #1 bottleneck for long-context workloads. Addressed by int8 KV, K8V3, and paged cache.
2. **Kernel launch overhead** (decode): The #1 bottleneck for short-context / small-batch workloads. Addressed by CUDA graphs.
3. **Weight bandwidth** (prefill): The bottleneck for model loading and prefill of very long prompts. Addressed by mmap, .superl8 format, and GGUF native loading.
4. **PCIe bandwidth** (multi-GPU): The bottleneck for pipeline parallelism. Addressed by compression codecs (int8/int4/NF4 + Hadamard rotation) and avoiding TP entirely.
5. **Host memory** (large models): The bottleneck when a model exceeds GPU VRAM. Addressed by weight streaming (not yet implemented) and expert offloading.

## 7. Known Limitations

1. **No FP8 fallback for Hopper/Ada.** The dp4a path is optimal on CMP/Volta, but on Hopper, FP8 tensor cores would be faster for compute-bound workloads. A future `fp8_fallback` path would dispatch to HMMA when available.

2. **No O(1) memory-efficient backward.** The backward pass (`attn_bwd`) stores full attention matrices. Training long contexts requires O(N²) memory. An O(1) tiled backward is on the roadmap.

3. **No compile-time kernel selection.** Tile sizes and split factors are fixed at compile time. Auto-tuning per GPU at install time would improve performance across hardware variants.

4. **No persistent kernel launch.** Each prefill/decode cycle incurs CUDA context overhead. A persistent kernel approach (keep the GPU context alive across calls) would eliminate this.

5. **GGUF k-quant decode is M=1 only (currently).** The fused k-quant kernels (`gemm_q{2,3,4,5,6}k`) are warp-per-column for M=1 decode. The tiled variant for M>1 prefill exists but is less optimized than the int8 path.

6. **KV eviction not supported for K8V3.** The compact path (dequantize → requantize) is not implemented for the packed 3-bit V layout. Eviction is int8-only.

7. **No training-aware quantization.** W8A8 is post-training quantized. Quantization-aware training (QAT) with simulated INT8 would recover the last fraction of quality lost during PTQ.

## 8. Design Provenance

The design decisions in SuperL8 are validated by measured evidence, not assumption. Key experiments:

| Hypothesis | Result | Source |
|-----------|--------|--------|
| Shared codebook across models | NO-GO (PPL 8M vs 21) | `superl8-serve/docs/scale-codebook-spike.md` |
| Scalar/vector PQ on weights | NO-GO (PPL 300, accuracy 60%) | `superl8-serve/docs/pq-rigorous-spike.md` |
| Shared prototype across models | NO-GO (residuals full-sized) | `superl8-serve/docs/shared-prototype-spike.md` |
| Per-row INT8 + scale | **GO** (PPL 21.19, near-lossless) | `superl8-serve/docs/weight-stationary-architecture.md` |
| W8A8 with CUDA graphs | **GO** (11.4× speedup over FP16) | Pareto frontier measurement |

The universal codebook hypothesis (shared prototypes across models) was **falsified**. The weight-stationary runtime hypothesis (minimize weight movement per token) **remains intact**. The innovation is in execution, not compression.
