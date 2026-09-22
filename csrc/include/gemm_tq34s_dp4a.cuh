// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Fused GGUF TQ3_4S -> int8 dp4a GEMM for sm_70 (native TurboQuant, type 46).
//
// TQ3_4S reconstruction (turbo-tan/llama.cpp-tq3 fork, ggml-quants.c:2706-2840):
//   1. unpack 3-bit codes (4 groups x 8 per 32-value block),
//   2. v_j = centroid[code_j] * E3M5(scale_g),  per-8 E3M5 u8 scale,
//   3. w = RHT_inv(v) = diag(SIGNS) . H / sqrt(32) . v.
//
// The dp4a-fusion identity (fni8#270 charter, F = H.diag(SIGNS)/sqrt(32)):
//
//     x_act^T . w = x_act^T . F_inv(v) = (F . x_act)^T . v
//
// so the ACTIVATION is rotated with the FORWARD RHT per 32-block on the fp
// values BEFORE the per-32 q8_1 int8 quantization (signs -> WHT butterfly ->
// 1/sqrt(32); the butterfly mixes all 32 values and the range grows past int8,
// so it cannot run on the int8 codes — rotation placement pinned in qwen38-tq3
// review). The per-32 activation scale is computed on the ROTATED values, so it
// varies along K and must multiply into the fp32 accumulator per block (NOT the
// per-row epilogue factor of the q4k spine).
//
// Weight side is a plain __dp4a against the CORRECTED int8 centroid levels
// (NOT the fork's stale codebook): levels[c] = round(centroid[c]*127/1.996684)
// = {-127,-82,-47,-16,15,46,81,127}, with per-block fp32 factor
// d_weight = E3M5(scale_byte) * max|centroid|/127. Per 8-element group the
// int32 dp4a partial is flushed with d_weight (and the per-32 activation scale):
//
//     out[m,n] = sum_{32-blocks b} xs_b * sum_{groups g}
//                          E3M5(g)*maxc/127 * dp4a(xhat_b, levels_g)
//
// The weight stays RESIDENT in its native GGUF TQ3_4S layout: 16 B / 32 weights
// (4 E3M5 scale bytes + 12 packed 3-bit code bytes), rows [out, (in/32)*16],
// in%32==0. Block QK_TQ3_0 = 32 — NOT the 256-elem k-quant super-block — so a
// GEMM_BK=64 k-block holds 2 TQ3 blocks = 8 per-8 scale flushes, and the
// tile-staging guards must generalize past K%256==0 to K%32==0.
//
// Volta rules: __dp4a + __shfl_xor_sync only (no mma/wmma/cp.async/ldmatrix);
// synchronous 128-bit loads, fp32 accumulate.
//
// Perf structure (fni8#281): the forward RHT + per-32 q8_1 quant runs ONCE per
// activation in `tq34s_rht_prepass_kernel` (global xq [M,K] int8 + xs [M,nblk]
// fp32) instead of being re-computed per output block — at M=1 decode it was
// 1280x redundant and at prefill 80x (N/BN), measuring 1.2x the dp4a work.
// Both the tile kernel here and gemm_decode_tq34s_dp4a.cuh stage the PRE-ROTATED
// int8 activation (the proven q4k spine), and the 3-bit code unpack is a
// branch-free 64-entry uint16 smem LUT (tq34s_group_words_pair) instead of the
// old per-level switch (~19M branch instructions at decode).
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "compute_dtype.cuh"
#include "gemm_dp4a_config.cuh"   // GEMM_BM/BN/BK/THREADS/TM/TN/BK4

namespace fni8 {

constexpr int TQ3_QK = 32;          // values per block (QK_TQ3_0, NOT QK_K=256)
constexpr int TQ3_TYPE_SIZE = 16;   // 4 E3M5 scale bytes + 12 code bytes
constexpr int TQ3_GROUPS = 4;       // per-8 groups per 32-block
constexpr float TQ3_RHT_NORM = 0.17677669529663687f;  // 1/sqrt(32)
constexpr float TQ3_KMAX_OVER_127 = 1.9966840744018555f / 127.0f;  // max|centroid|/127

// Corrected fni8 int8 centroid levels (K = 127/max|centroid| = 127/1.996684).
// The fork's decode dot uses a STALE codebook — fni8 gates against the fp32
// CPU dequant of the same bytes, so this table must match tq34s_levels().
__device__ __forceinline__ int tq34s_level(int idx) {
  // Only the one-time per-block LUT build calls this (tq34s_build_lut16); the
  // per-8 hot-path unpack is the branch-free LUT16 lookup.
  switch (idx) {
    case 0: return -127;
    case 1: return -82;
    case 2: return -47;
    case 3: return -16;
    case 4: return 15;
    case 5: return 46;
    case 6: return 81;
    default: return 127;
  }
}

// TQ3_SIGNS packed as a 32-bit mask (bit i set => lane i is -1), =0x696B4B4A.
constexpr uint32_t TQ3_SIGNS_MASK = 0x696B4B4Au;

// E3M5 mini-float scale decode, exact fp32 (fork `tq3_4s_ratio4s`): byte==0 ->
// 0.0; else 2^(exp-9)*(1+mant/32) via the fp32 bit pattern ((b>>5)+118)<<23 |
// (b&31)<<18. No ldexpf in the hot loop.
__device__ __forceinline__ float tq34s_decode_e3m5(uint8_t b) {
  if (b == 0) return 0.0f;
  const uint32_t bits = ((uint32_t)((b >> 5) + 118) << 23) | ((uint32_t)(b & 31) << 18);
  return __uint_as_float(bits);
}

// ── Branch-free 3-bit -> int8-level unpack (perf pass, fni8#281) ────────────
// The per-8 group's 3 code bytes encode 8 three-bit indices; the corrected
// int8 centroid levels are {-127,-82,-47,-16,15,46,81,127}. The old switch
// lookup compiled to ~19M branch instructions (31% of the decode kernel's mix,
// divergent per lane) — the single biggest decode stall. Replace it with a
// 64-entry uint16 smem LUT keyed on 6-bit PAIRS of codes:
//
//   LUT16[key] = byte0 = level[key&7], byte1 = level[(key>>3)&7]
//
// Every two contiguous codes in the fork's bit layout (idx0,1 in qp0 bits 0-5;
// idx2,3 spanning qp0/qp1; idx4,5 spanning qp1/qp2; idx6,7 in qp2 bits 2-7)
// map to a single 6-bit key, so each per-8 group's two dp4a words are four
// smem LUT loads + shifts — branch-free, no divergence, ~4x fewer instructions.
__device__ __forceinline__ void tq34s_group_words_pair(uint32_t qp0, uint32_t qp1,
                                                       uint32_t qp2,
                                                       const uint16_t* lut,
                                                       int32_t& w0, int32_t& w1) {
  const uint16_t a = lut[qp0 & 63u];
  const uint16_t b = lut[((qp0 >> 6) & 3u) | ((qp1 & 0xFu) << 2)];
  // idx4 = qp1 bits 4-6; idx5 = bit7(qp1) | bit0(qp2)<<1 | bit1(qp2)<<2
  // (the fork's idx5 = ((qp1>>7)|(qp2<<1))&7 — do NOT drop the qp2 bit1 term).
  const uint16_t c =
      lut[((qp1 >> 4) & 7u) | (((qp1 >> 7) & 1u) << 3) | ((qp2 & 1u) << 4)
          | ((qp2 & 2u) << 4)];
  const uint16_t d = lut[(qp2 >> 2) & 63u];
  w0 = (int32_t)a | ((int32_t)b << 16);
  w1 = (int32_t)c | ((int32_t)d << 16);
}

// Requantize four signed int8 centroid levels to a block-common Q8 scale.
// This is the Volta DP4A path used by llama.cpp's TQ3_4S MMQ loader: per-group
// scales are folded into the staged Q8 bytes so the dot product needs one fp32
// scale flush per 32 weights instead of four.
__device__ __forceinline__ int32_t tq34s_requant_word_common(int32_t word,
                                                             float ratio) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = __float2int_rn((float)(int8_t)((uint32_t)word >> (8 * i)) * ratio);
    out |= (uint32_t)(uint8_t)max(-127, min(127, q)) << (8 * i);
  }
  return (int32_t)out;
}

// One-time per block: build the 64-entry uint16 pair LUT from the 8 levels.
__device__ __forceinline__ void tq34s_build_lut16(uint16_t* lut, int tid,
                                                  int threads) {
  for (int i = tid; i < 64; i += threads) {
    const int lo = i & 7, hi = (i >> 3) & 7;
    lut[i] = (uint16_t)(uint8_t)tq34s_level(lo)
        | ((uint16_t)(uint8_t)tq34s_level(hi) << 8);
  }
}

// Forward RHT of a 32-value block: multiply by SIGNS, WHT butterfly, 1/sqrt(32).
// One lane per value; the 5-stage butterfly pairs lane j with lane j^mask using
// __shfl_xor_sync — identical pairing/order to fni8.quant.tq34s._wht.
__device__ __forceinline__ void tq34s_rht_fwd_warp(float& v, int lane) {
  v *= (((TQ3_SIGNS_MASK >> lane) & 1u) ? -1.0f : 1.0f);
#pragma unroll
  for (int mask = 1; mask < TQ3_QK; mask <<= 1) {
    const float o = __shfl_xor_sync(0xFFFFFFFFu, v, mask);
    v = (lane & mask) ? (o - v) : (v + o);
  }
  v *= TQ3_RHT_NORM;
}

// ── Activation forward-RHT + per-32 q8_1 quant PRE-PASS (perf pass, fni8#281) ─
// The rotation+quant of the fp activation is O(M*K/32) work but the decode
// kernel previously re-ran it in EVERY block (1280x redundancy at M=1) and the
// tile kernel in every N-block (80x at prefill) — it measured 1.2x the dp4a
// work itself and was the dominant non-dp4a cost. This kernel does it ONCE per
// activation into global (xq [M,K] int8 + xs [M,nblk] fp32); both the tile and
// the decode kernels then stage the PRE-ROTATED int8 activation exactly like
// the proven q4k spine. Math is byte-identical to the old in-kernel phase:
// signs -> WHT butterfly -> 1/sqrt(32), per-32 scale = max|rotated|/127.
template <typename InT>
__global__ void __launch_bounds__(256)
tq34s_rht_prepass_kernel(const InT* __restrict__ x, int8_t* __restrict__ xq,
                         float* __restrict__ xs, int M, int K, int nblk) {
  const int lane = threadIdx.x & 31;
  const int warp = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  const int total = M * nblk;
  const int n_warps = gridDim.x * (blockDim.x / 32);
  for (int t = warp; t < total; t += n_warps) {
    const int m = t / nblk, b = t % nblk;
    const int64_t base = (int64_t)m * K + (int64_t)b * TQ3_QK;
    float v = to_float(x[base + lane]);
    tq34s_rht_fwd_warp(v, lane);
    float a = fabsf(v);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, mask));
    const float scale = a * (1.0f / 127.0f);
    const float inv = (scale > 0.f) ? (1.0f / scale) : 0.f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    xq[base + lane] = (int8_t)q;
    if (lane == 0) xs[(int64_t)m * nblk + b] = scale;
  }
}

// Grid: (ceil(N/BN), ceil(M/BM)). xq [M,K] int8 + xs [M, K/32] fp32 (the
// tq34s_rht_prepass_kernel output), w [N, (K/32)*16] uint8 native TQ3_4S bytes
// -> out [M,N] OutT. nblk = K/32. The activation is already ROTATED+quantized
// per 32-block by the prepass; this kernel is the q4k spine with a branch-free
// LUT16 3-bit unpack and per-8 E3M5 flush (see the file header).
template <typename OutT, bool CommonScale = false, int TQ_THREADS = GEMM_THREADS,
          int TQ_TM = GEMM_TM, int TQ_TN = GEMM_TN>
__global__ void __launch_bounds__(TQ_THREADS)
gemm_tq34s_kernel(const int8_t* __restrict__ xq, const float* __restrict__ xs,
                  const uint8_t* __restrict__ w, OutT* __restrict__ out,
                  int M, int N, int K, int nblk) {
  __shared__ int32_t s_x[GEMM_BM][GEMM_BK4 + 1];  // rotated+quantized activation words
  __shared__ float s_xs[GEMM_BM][2];          // per-32 activation scale (2 TQ3 blocks/k-block)
  __shared__ int32_t s_w[GEMM_BN][GEMM_BK4 + 1];  // int8 centroid levels (weights)
  __shared__ float s_dsc[GEMM_BN][8];         // E3M5*maxc/127 for the 8 per-8 groups
  __shared__ uint16_t s_lut16[64];            // 6-bit-code pair -> 2 int8 levels

  static_assert(TQ_THREADS == (GEMM_BM / TQ_TM) * (GEMM_BN / TQ_TN),
                "TQ3 thread geometry must cover the full output tile");
  const int tid = threadIdx.x;
  const int tr = tid / (GEMM_BN / TQ_TN);
  const int tc = tid % (GEMM_BN / TQ_TN);
  const int m_block = blockIdx.y * GEMM_BM;
  const int n_block = blockIdx.x * GEMM_BN;

  const int K4 = K / 4;
  const int64_t row_bytes = (int64_t)nblk * TQ3_TYPE_SIZE;
  const int n_kblocks = (K + GEMM_BK - 1) / GEMM_BK;  // BK=64; 2 TQ3 blocks each

  tq34s_build_lut16(s_lut16, tid, TQ_THREADS);
  __syncthreads();  // s_lut16 is read by every thread in stage W+S below; the
                    // per-block build is split across warps, so the FIRST k-block
                    // needs this barrier before any thread consumes the LUT.

  float f_acc[TQ_TM][TQ_TN];
#pragma unroll
  for (int i = 0; i < TQ_TM; ++i)
#pragma unroll
    for (int j = 0; j < TQ_TN; ++j) f_acc[i][j] = 0.f;

  for (int kb = 0; kb < n_kblocks; ++kb) {
    const int kbase = kb * GEMM_BK;           // element base of this k-block

    // ---- stage X: int8 activation words (pre-rotated by the prepass) ----
    for (int idx = tid; idx < GEMM_BM * GEMM_BK4; idx += TQ_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gm = m_block + r, gk4 = kbase / 4 + c;
      s_x[r][c] = (gm < M && gk4 < K4)
                      ? reinterpret_cast<const int32_t*>(xq + (int64_t)gm * K)[gk4] : 0;
    }
    // ---- stage W+S: one 128-bit TQ3 block load per thread -> 8 level words
    //      (LUT16 branch-free) + the 4 E3M5 scales from the same 16 bytes.
    // 128 threads = 64 rows x 2 TQ3 blocks (BK=64), one block each. ----
    if (tid < GEMM_BN * 2) {
      const int r = tid >> 1, tq3c = tid & 1;
      const int gn = n_block + r;
      const int tq3 = kb * 2 + tq3c;
      uint4 bv = make_uint4(0u, 0u, 0u, 0u);
      if (gn < N && tq3 < nblk)
        bv = *reinterpret_cast<const uint4*>(w + (int64_t)gn * row_bytes
                                                 + (int64_t)tq3 * TQ3_TYPE_SIZE);
      const uint32_t y = bv.y, z = bv.z, w4 = bv.w;
      int32_t ww[8];
      tq34s_group_words_pair(y & 0xFFu, (y >> 8) & 0xFFu, (y >> 16) & 0xFFu,
                             s_lut16, ww[0], ww[1]);
      tq34s_group_words_pair((y >> 24) & 0xFFu, z & 0xFFu, (z >> 8) & 0xFFu,
                             s_lut16, ww[2], ww[3]);
      tq34s_group_words_pair((z >> 16) & 0xFFu, (z >> 24) & 0xFFu, w4 & 0xFFu,
                             s_lut16, ww[4], ww[5]);
      tq34s_group_words_pair((w4 >> 8) & 0xFFu, (w4 >> 16) & 0xFFu,
                             (w4 >> 24) & 0xFFu, s_lut16, ww[6], ww[7]);
      if constexpr (CommonScale) {
        float scale[TQ3_GROUPS];
        float max_scale = 0.f;
#pragma unroll
        for (int g = 0; g < TQ3_GROUPS; ++g) {
          scale[g] = tq34s_decode_e3m5((uint8_t)(bv.x >> (8 * g)));
          max_scale = fmaxf(max_scale, scale[g]);
        }
        const float inv_max = max_scale > 0.f ? 1.f / max_scale : 0.f;
#pragma unroll
        for (int g = 0; g < TQ3_GROUPS; ++g) {
          const float ratio = scale[g] * inv_max;
          ww[2 * g] = tq34s_requant_word_common(ww[2 * g], ratio);
          ww[2 * g + 1] = tq34s_requant_word_common(ww[2 * g + 1], ratio);
        }
        s_dsc[r][tq3c] = max_scale * TQ3_KMAX_OVER_127;
      }
#pragma unroll
      for (int c = 0; c < 8; ++c) s_w[r][tq3c * 8 + c] = ww[c];
      if constexpr (!CommonScale) {
#pragma unroll
        for (int g = 0; g < TQ3_GROUPS; ++g)
          s_dsc[r][tq3c * TQ3_GROUPS + g] =
              tq34s_decode_e3m5((uint8_t)(bv.x >> (8 * g))) * TQ3_KMAX_OVER_127;
      }
    }
    // ---- stage scales: per-32 activation scale xs[m][kb*2+sub] ----
    for (int idx = tid; idx < GEMM_BM * 2; idx += TQ_THREADS) {
      const int r = idx >> 1, sub = idx & 1;
      const int gm = m_block + r;
      s_xs[r][sub] =
          (gm < M && kb * 2 + sub < nblk) ? xs[(int64_t)gm * nblk + kb * 2 + sub] : 0.f;
    }
    __syncthreads();

    // ---- 8 per-8 group flushes; each = 2 dp4a words (8 elements), flushed
    //      with xs_block * E3M5*maxc/127 into the fp32 accumulator. ----
    constexpr int flushes = CommonScale ? 2 : 8;
    constexpr int words_per_flush = CommonScale ? 8 : 2;
#pragma unroll
    for (int gidx = 0; gidx < flushes; ++gidx) {
      int32_t iacc[TQ_TM][TQ_TN];
#pragma unroll
      for (int i = 0; i < TQ_TM; ++i)
#pragma unroll
        for (int j = 0; j < TQ_TN; ++j) iacc[i][j] = 0;
#pragma unroll
      for (int cc = 0; cc < words_per_flush; ++cc) {
        const int c = gidx * words_per_flush + cc;
        int32_t xr[TQ_TM], wr[TQ_TN];
#pragma unroll
        for (int i = 0; i < TQ_TM; ++i) xr[i] = s_x[tr * TQ_TM + i][c];
#pragma unroll
        for (int j = 0; j < TQ_TN; ++j) wr[j] = s_w[tc * TQ_TN + j][c];
#pragma unroll
        for (int i = 0; i < TQ_TM; ++i)
#pragma unroll
          for (int j = 0; j < TQ_TN; ++j)
            iacc[i][j] = __dp4a(xr[i], wr[j], iacc[i][j]);
      }
      const int sub = CommonScale ? gidx : (gidx >> 2);  // TQ3 block within k-block
      float xsr[TQ_TM];
#pragma unroll
      for (int i = 0; i < TQ_TM; ++i) xsr[i] = s_xs[tr * TQ_TM + i][sub];
#pragma unroll
      for (int j = 0; j < TQ_TN; ++j) {
        const int wrow = tc * TQ_TN + j;
        const float dsc = s_dsc[wrow][gidx];
#pragma unroll
        for (int i = 0; i < TQ_TM; ++i)
          f_acc[i][j] += xsr[i] * dsc * (float)iacc[i][j];
      }
    }
    __syncthreads();
  }

  // ---- epilogue: activation scales are already folded into f_acc (per-32, not
  // per-row), so the store is a plain OutT convert.
#pragma unroll
  for (int i = 0; i < TQ_TM; ++i) {
    const int gm = m_block + tr * TQ_TM + i;
    if (gm >= M) continue;
    OutT* out_row = out + (int64_t)gm * N;
#pragma unroll
    for (int j = 0; j < TQ_TN; ++j) {
      const int gn = n_block + tc * TQ_TN + j;
      if (gn < N) out_row[gn] = float_to<OutT>(f_acc[i][j]);
    }
  }
}

}  // namespace fni8
