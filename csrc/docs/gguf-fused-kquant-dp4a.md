# Fused GGUF k-quant → dp4a GEMM for sm_70 (native-GGUF-on-the-fly)

**Status:** All 5 k-quants implemented + quality-gated, each with a **prefill tile** kernel
(`gemm_q{2,3,4,5,6}k`) AND a **warp-per-column MMVQ decode** kernel (`gemm_decode_q{2,3,4,5,6}k`,
M=1 GPU-saturating). Covers the full LTX Q4_K_M mix (Q4_K/Q6_K/Q5_K) AND **Qwen3.6-27B-Q3_K_S**
(353 Q3_K tensors — native residency fits ONE 16 GB card; requant→int8 would bust it) AND the
aggressive UD-Q2 bulk (Q2_K). Marquee capability: **load a GGUF file and matmul it directly** —
the k-quant
super-blocks stay resident in their native GGUF layout (~13 GB for LTX-2.3-22B-Q4_K_M, fits
ONE 16 GB card), and the GEMM kernel unpacks each super-block to int8 in registers/smem and
runs `__dp4a` inline, honoring the native per-sub-block scales exactly. No offline conversion,
no resident re-quant, no per-forward fp32 dequant materialization.

## Reference (MIT — adapted in fni8 BSD-3 style, attributed in each file header)

llama.cpp checkout (see llama.cpp git history). The MMQ/MMVQ quantized matmul
does exactly this and has a Volta `__dp4a` path (pre-Turing). Pieces adapted:

| fni8 piece | llama.cpp source |
|---|---|
| `block_q4_K` layout (d, dmin, scales[12], qs[128]) | `ggml/src/ggml-common.h:408-419` |
| `block_q5_K` (d, dmin, scales[12], qh[32], qs[128]) | `ggml-common.h:425-437` |
| `block_q6_K` (ql[128], qh[64], scales[16] int8, d) | `ggml-common.h:443-449` |
| 6-bit sub-scale/min unpack `get_scale_min_k4` | `ggml-cuda/convert.cu:195-202`, `ggml-quants.c:818-825` |
| Q4_K dp4a dot (`sumi`, min-correction, `d·Σd − dmin·Σm`) | `ggml-cuda/vecdotq.cuh:505-527` |
| Q5_K dp4a dot (+ high-bit assembly) | `vecdotq.cuh:561-590` |
| Q6_K dp4a dot (`__vsubss4(...,0x20)`, signed, no min) | `vecdotq.cuh:624-644` |
| q8_1 activation `s = d·Σqs` (min-correction sum) | `ggml-common.h:248-259`, `ggml-quants.c:263-292` |

## The math (why this is the RIGHT primitive on this fleet)

k-quant reconstruction is a **per-32-element affine** dequant. For Q4_K, sub-block `j` (32
weights) has a 6-bit scale `sc_j` and 6-bit min `m_j`, plus super-block fp16 `d` and `dmin`;
each 4-bit weight `q ∈ [0,15]` reconstructs as

    w_k = d·sc_j·q_k − dmin·m_j            (asymmetric / min-offset; NO −8 centering)

A linear layer's output contribution from that sub-block, with the activation quantized per row
to int8 (`x_k = x̂_k · xs_m`, `x̂ ∈ [−127,127]`, `xs_m` the per-row fp32 scale), is

    Σ_k x_k·w_k = xs_m · [ d·sc_j · Σ_k x̂_k·q_k  −  dmin·m_j · Σ_k x̂_k ]
                = xs_m · [ d·sc_j · sumi_j        −  dmin·m_j · xsum_j ]

* `sumi_j = Σ x̂·q` — a **`__dp4a`** of 8 int32 words (32 int8 activations × 32 unsigned-4-bit
  weights unpacked to int8). This is the hot compute.
* `xsum_j = Σ x̂` — the **min-correction** term. Computed inline as `__dp4a(x̂_word, 0x01010101)`
  accumulated over the 8 words (exactly llama.cpp MMVQ's `dot2`, `vecdotq.cuh:512`). It depends
  only on the activation, so it is computed once per (activation-row, sub-block) and reused
  across all N output columns.

The whole sub-block result accumulates into an **fp32** accumulator; `xs_m` is factored out to
the epilogue. Softmax/LSE are not involved — this is a weight-only linear, activations stay
dynamic per-row int8 (`quantize_i8_rowwise`), so nothing extra is stored (AGENTS.md: int32
accumulate, single dequant multiply; fp32 scales).

Q6_K is **symmetric** (no min term): `w_k = d·sc_j·(q6_k − 32)`, `sc_j` a signed int8 per-16
scale. The `−32` centers the 6-bit code to `[−32,31]` before the dp4a (llama uses per-byte
saturating `__vsubss4(code, 0x20202020)`), so it is a plain signed `__dp4a` with no `xsum`
correction — `y = xs_m · d · Σ_j sc_j·sumi_j`.

**Fleet rationale (AGENTS.md HARD RULE):** the CMP 100-210 tensor cores are firmware-gimped
(fp16 TC 6.9 TFLOP/s), while `__dp4a` runs at 46 TOP/s on the intact INT/CUDA-core pipe. So
fusing the k-quant unpack into a dp4a matmul is the correct primitive, not a compromise — and
the current per-forward fp32-dequant GGUF path (143 s/step) is exactly the tensor-core-bound
loser we avoid. int8 also buys the memory win (native 4.5-bit resident vs 21 GB int8 re-quant
that busts a 16 GB card).

## Tiling — reuse the `gemm_w4a8_kernel` spine

`csrc/include/gemm_dp4a.cuh::gemm_w4a8_kernel` already does *precisely* the structure needed:
BM=BN=BK=64, 128 threads, 16×8 thread grid, TM=4×TN=8 register micro-tile, **per-32-element
step flush into an fp32 accumulator with a per-group scale** (`GEMM_STEP=32`, `GEMM_STEP4=8`).
A Q4_K super-block is 256 K-elements = 4 BK-blocks; each BK-block = 2 sub-blocks of 32 = 2 step
flushes. So the k-quant kernel is the w4a8 kernel with three changes:

1. **Weight staging** reads the native super-block bytes instead of a `[N,K/2]` nibble array.
   For weight row `gn`, BK-block `kb`: super-block `sb = kb/4` (block stride 144 B), group
   `gsb = kb%4`, the 32 qs bytes at `qs + gsb*32`. The two step sub-blocks are `j0 = 2·gsb`
   (low nibbles of those 32 bytes) and `j1 = 2·gsb+1` (high nibbles) — this exactly matches the
   Q4_K qs interleave (`dequantize_row_q4_K`: `y[l]=d1·(q[l]&0xF)-m1; y[l+32]=d2·(q[l]>>4)-m2`).
   Unpacked to int8 into `s_w[BN][BK4]` (unsigned 0..15, positive → signed dp4a is exact).
   Q4_K rows are NOT 4-byte aligned (144 is even, not %4), so the weight is read byte-wise — no
   `int32` reinterpret on the weight (only on the already-`%4` activation).
2. **Sub-scale staging**: once per BK-block, decode `d·sc`, `dmin·m` for `j0,j1` per weight row
   via `get_scale_min_k4` from the 12 scale bytes + fp16 `d,dmin`, into small smem
   `s_dsc[BN][2]`, `s_dm[BN][2]`.
3. **Affine flush**: alongside `iacc[i][j] = dp4a(x,w)`, accumulate `xsum[i] = dp4a(x,0x0101…)`
   per activation row per step; flush `f_acc[i][j] += s_dsc[wrow][sbl]·iacc − s_dm[wrow][sbl]·xsum[i]`.
   Epilogue: `y = xs_m · f_acc`.

Ragged M and N handled by the same zero-pad guards. **K is always %256** for a k-quant tensor
(GGUF stores in%256≠0 tensors as Q8_0/F16, never a k-quant), asserted in the launcher.

## Batched M>1 (prefill) vs decode M=1 — one tiled GEMM serves both

This kernel is a **batched tiled GEMM**, not a GEMV — it stages a `BM=64`-row activation tile
into smem and dp4a's it against the staged weight tile, so `M>1` prefill and batched decode are
the native case (`grid.y = ceil(M/BM)`), and `M=1` autoregressive decode is just the ragged-M
tail of the same launch. This is exactly llama.cpp's **MMQ** structure (load k-quant weight tile
→ int8 in smem → dp4a against the activation tile → per-sub-block-scale accumulate), which is
the right reference for the GGUF-native **batched engine** (vLLM×ggml on dp4a): efficient
prefill AND batched decode from a single code path. The decode SQNR/speed proof (M=1) and the
prefill path (M=2048 in the test sweep) share this kernel — verified by the same fidelity gate
across the M sweep.

Two deliberately-deferred optimizations (each a later stacked PR, neither changes the math):

* **Per-32 q8_1 activation tile.** We currently quantize the activation **per-row** (one fp32
  scale/row) and recover the min-correction sum inline (`dp4a(x,0x0101…)`). llama.cpp instead
  block-quantizes the activation to **q8_1** (per-32 int8 + `{d, s=d·Σq}`), which gives a
  tighter per-32 activation scale (better SQNR on outlier-heavy activation blocks) and
  precomputes `s` so the min term is a single multiply. The kernel already isolates the sum per
  sub-block, so switching to a per-32 activation quant is a prologue swap + reading `s` instead
  of computing it — no change to the dp4a spine. Per-row is correct and simpler for v1; q8_1 is
  the accuracy/throughput follow-up for the batched engine.
* **Decode-specialized warp-per-column variant** (mirrors `gemm_decode_w4a8`, issue #173). At
  `M≤16` the 64×64 output tile launches too few threadblocks to fill the GPU; a warp-per-output-
  column launch (with K-splitting) fills it. Same Q4_K unpack + affine dp4a, just remapped —
  a pure occupancy win for latency-bound single-stream decode. The tile kernel here is
  correct and already fast for batched decode; the warp-per-column variant is the single-stream
  latency lever.

## fni8 integration

* **`QTensor` scheme `gguf_kquant`** (`fni8/format.py`): `data` = raw GGUF k-quant bytes uint8
  `[N, n_superblocks·type_size]`; `codebook` carries the type tag (`"q4_k"`/`"q5_k"`/`"q6_k"`);
  `group_size=256`; no separate `scale` blob (sub-scales live in the bytes). Byte-identical to
  the resident VRAM layout — `.fni8` can now store a GGUF tensor with zero transcode, and a GGUF
  file can be mmapped straight into VRAM.
* **Op**: `fni8._C.gemm_q4k(x_i8, x_scale, w_bytes, out_dtype)` → `[M,N]`
  (`csrc/kernel/gemm_dp4a.cu` launcher, `gemm_q4k_kernel` in `gemm_q4k_dp4a.cuh`).
* **Python**: `fni8.linear_q4k(x, w_bytes, out_dtype)` quantizes the activation per-row and calls
  the op; `fni8.linear(x, qt)` dispatches `gguf_kquant` by type tag.
* **Loader path**: a GGUF reader (gguf-py `GGUFReader`, or the fni8-serve `gguf_import.py`
  name-map) yields `(hf_name, raw_bytes, ggml_type)` per tensor; wrap in a `gguf_kquant`
  `QTensor` and hand to `fni8.linear`. No dequant. Q6_K `to_v` outliers, Q4_K bulk, Q5_K
  remainder each run their native precision — GGUF's per-tensor type IS the mixed-precision map.

## Test / quality gate

`tests/test_gemm_q4k.py`, `@pytest.mark.correctness`, int8 gates (SQNR/cos/rel-L1, never
allclose — AGENTS.md):

1. **Kernel fidelity** vs a pure-torch Q4_K dequant→matmul of the SAME synthesized bytes:
   SQNR ≥ 40 dB / cos ≥ 0.999 (the kernel parses the layout + does the affine dp4a exactly, so
   only fp-store rounding separates it — mirrors `test_gemm_w4a8_reproduces_grouped_dequant`).
   Synthesized bytes exercise the real 6-bit sub-scale/min packing and qs interleave.
2. **fp32-oracle quality** on a Q4_K quant of a real fp weight: cos/SQNR at Q4_K's intrinsic
   error (documented bar, not weakened).
3. **Ragged** N and M (non-tile-multiple); **determinism** ×3 bitwise.
4. **Real-LTX** (gated on `gguf` + file present): dequant vs fused on actual Q4_K tensors from
   `ltx-2.3-22b-distilled-Q4_K_M_light.gguf`.

## Perf

`@pytest.mark.perf`: fused `gemm_q4k` vs the fp32-dequant→matmul path at LTX DiT linear shapes.
Win is (a) O(native-bytes) VRAM (single card) and (b) dp4a on the intact pipe vs the gimped
tensor cores the dequant→fp16-matmul path lands on. ncu artifact on real V100 idx 7.

## The five k-quants (all implemented, tile + decode)

| type | block | sub | scale | affine? | reconstruction | ref |
|---|---|---|---|---|---|---|
| Q2_K | 84 B | 16×16 | 4-bit +4-bit min | yes (`xsum`) | `d·sc·q − dmin·m`, q∈[0,3] | `vecdotq.cuh:364-389` |
| Q3_K | 110 B | 16×16 | signed 6-bit | no | `d·sc·(q−4)`, q∈[0,7], `-4` in-lane | `vecdotq.cuh:447-477` |
| Q4_K | 144 B | 8×32 | 6-bit +6-bit min | yes (`xsum`) | `d·sc·q − dmin·m`, q∈[0,15] | `vecdotq.cuh:505-527` |
| Q5_K | 176 B | 8×32 | 6-bit +6-bit min | yes (`xsum`) | Q4_K + 5th bit from qh[32] | `vecdotq.cuh:561-590` |
| Q6_K | 210 B | 16×16 | signed int8 | no | `d·sc·(q−32)`, q∈[0,63] | `vecdotq.cuh:624-644` |

* **Q3_K** (the 27B-Q3_K_S type): `hmask[32] qs[64] scales[12] d`. 3-bit code = `(qs 2-bit) |
  (hmask bit << 2)`, centered `−4`. The signed 6-bit per-16 scale is rebuilt from `scales[12]`
  (low 4 bits from `scales[0..7]`, high 2 bits from `scales[8..11]`, `−32`) — same `q3k_scale`
  helper in the tile and decode kernels. Symmetric (no min). The per-16 sub-block flushes every
  16 K-elements (mirrors the Q6_K spine).
* **Q2_K** (aggressive UD-Q2): `scales[16] qs[64] d dmin`. Affine like Q4_K but 2-bit q + 4-bit
  scale/min per-16, with the `xsum` min-correction (`__dp4a(x, 0x01010101)`).

The **decode** variants (`gemm_decode_q{2,3,4,5,6}k`) are warp-per-column MMVQ (one warp per
output column, full-K reduction, one `__shfl_xor` butterfly) — byte-identical math to the tile,
GPU-saturating at M=1 (7–8× faster than the tile at decode; native then beats dequant→int8 on
the memory-bound GEMV). See gemm_decode_dp4a.cuh.
