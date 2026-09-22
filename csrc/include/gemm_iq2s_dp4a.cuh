// SPDX-License-Identifier: BSD-3-Clause
// ===========================================================================
// IQ2_S TILE (prefill) GEMM — the M > 16 sibling of gemm_decode_iq2s.
// Same spine as gemm_iq3s/gemm_iq3xxs: stage a GEMM_BK=64-wide k-chunk of
// activations into s_x, the UNPACKED weight codes (4 per int32, dp4a-ready) into
// s_w, and the per-sub-block scales into s_dsc, then accumulate a
// GEMM_TM x GEMM_TN register tile.
//
// Block (82 B / 256 weights, 2.5625 bpw):
//     +0   d         fp16 super-block scale
//     +2   qs[32]    low 8 bits of the grid index (one byte per 8 weights)
//     +34  signs[32] one byte per grid entry: 8 EXPLICIT sign bits (set = negative)
//     +66  qh[8]     one byte per 32-weight sub-block; 2 bits per entry (index bits 8-9)
//     +74  scales[8] two 4-bit fields per byte -> SIXTEEN scales, one per 16 weights
//
// TWO DIFFERENCES FROM IQ3_S, both of which fail *plausibly* if the IQ3_S mapping
// is copied across (see gemm_decode_iq2s.cuh's own warning):
//
//   1. A grid entry here is EIGHT magnitudes (kIq2sGrid[idx][2], two dp4a words),
//      not four. So an int32 word is HALF an entry, not a whole one, and the
//      IQ3_S mapping `p = c % 8` is wrong. Word c covers weights 4c..4c+3, hence
//          e = kb4*8 + (c >> 1)      grid entry within the super-block
//          p = (c >> 1) & 3          entry within the 32-weight sub-block
//          h = c & 1                 WHICH HALF of the entry -> kIq2sGrid[idx][h]
//
//   2. The scale granularity is 16 weights, so a 64-wide k-block carries FOUR
//      scales, not two -- s_dsc is [4] here and the accumulate loop runs 4
//      sub-groups of 4 words each (4 words = 16 weights), not 2 of 8. Applying one
//      scale per 32 weights still produces well-shaped output, so this is exactly
//      the kind of error that survives a smoke test.
//
// Mapping actually used (verified case-by-case against the decode kernel):
//     ib32 = kb4*2 + (c >> 3)                     32-weight sub-block
//     p    = (c >> 1) & 3,  h = c & 1
//     idx  = qs[4*ib32 + p] | (((qh[ib32] >> (2*p)) & 3) << 8)   // 10-bit
//     g    = kIq2sGrid[idx][h]
//     mb   = (signs[4*ib32 + p] >> (4*h)) & 0xF   explicit bits, set = negative
//     si   = kb4*4 + (c >> 2)                     scale index, si >> 1 == ib32
//     dsc  = d * (0.5f + ((scales[si >> 1] >> (4*(si & 1))) & 0xF)) * 0.25f
//
// IQ2_S carries explicit sign BITS, so unlike IQ2_XS/IQ2_XXS/IQ3_XXS it does NOT
// use the shared kSignsIq2xs LUT.
// Oracle: fni8.quant.iq2s (fni8#317), bit-exact vs gguf-py.
// Adapted (MIT->BSD-3) from llama.cpp's IQ2_S dot product.
// ===========================================================================
#pragma once

#include "gemm_dp4a_config.cuh"
#include "gemm_decode_iq2s.cuh"   // kIq2sGrid + IQ2S_TYPE_SIZE

namespace fni8 {

template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_iq2s_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                 const uint8_t* __restrict__ w, OutT* __restrict__ out,
                 int M, int N, int K, int num_sb) {
  __shared__ int32_t s_x[GEMM_BM][GEMM_BK4 + 1];
  __shared__ int32_t s_w[GEMM_BN][GEMM_BK4 + 1];
  __shared__ float s_dsc[GEMM_BN][4];          // 4 scale groups per 64-wide k-block

  const int tid = threadIdx.x;
  const int tr = tid / (GEMM_BN / GEMM_TN);
  const int tc = tid % (GEMM_BN / GEMM_TN);
  const int m_block = blockIdx.y * GEMM_BM;
  const int n_block = blockIdx.x * GEMM_BN;
  const int K4 = K / 4;
  const int64_t row_bytes = (int64_t)num_sb * IQ2S_TYPE_SIZE;
  const int n_kblocks = (K + GEMM_BK - 1) / GEMM_BK;

  float f_acc[GEMM_TM][GEMM_TN];
#pragma unroll
  for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) f_acc[i][j] = 0.f;

  for (int kb = 0; kb < n_kblocks; ++kb) {
    const int k4_base = kb * GEMM_BK4;
    const int sb = kb / 4, kb4 = kb % 4;
    for (int idx = tid; idx < GEMM_BM * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gm = m_block + r, gk4 = k4_base + c;
      s_x[r][c] = (gm < M && gk4 < K4)
                      ? reinterpret_cast<const int32_t*>(x + (int64_t)gm * K)[gk4] : 0;
    }
    for (int idx = tid; idx < GEMM_BN * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gn = n_block + r;
      int32_t packed = 0;
      if (gn < N) {
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * IQ2S_TYPE_SIZE;
        const int ib32 = kb4 * 2 + (c >> 3);
        const int p = (c >> 1) & 3;      // grid entry within the sub-block
        const int h = c & 1;             // which HALF of the 8-magnitude entry
        const uint8_t* qs = blk + 2 + 4 * ib32;    // 4 grid entries per sub-block
        const uint8_t* sg = blk + 34 + 4 * ib32;   // one sign byte per entry
        const int qh = blk[66 + ib32];
        const int gidx = qs[p] | (((qh >> (2 * p)) & 3) << 8);
        const uint32_t g = __ldg(&kIq2sGrid[gidx][h]);
        const int mb = (sg[p] >> (4 * h)) & 0xF;
        const uint32_t sbits = (uint32_t)(mb & 1) | ((uint32_t)((mb >> 1) & 1) << 8) |
                               ((uint32_t)((mb >> 2) & 1) << 16) |
                               ((uint32_t)((mb >> 3) & 1) << 24);
        const int neg = __vcmpne4(sbits, 0x00000000u);
        packed = __vsub4((int)(g ^ (uint32_t)neg), neg);
      }
      s_w[r][c] = packed;
    }
    for (int r = tid; r < GEMM_BN; r += GEMM_THREADS) {
      const int gn = n_block + r;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        float dsc = 0.f;
        if (gn < N) {
          const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * IQ2S_TYPE_SIZE;
          const int si = kb4 * 4 + s;               // scale index, si >> 1 == ib32
          const int ls = (blk[74 + (si >> 1)] >> (4 * (si & 1))) & 0xF;
          dsc = __half2float(*reinterpret_cast<const __half*>(blk)) *
                (0.5f + (float)ls) * 0.25f;
        }
        s_dsc[r][s] = dsc;
      }
    }
    __syncthreads();
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {          // 4 scale groups of 16 weights
      int32_t iacc[GEMM_TM][GEMM_TN];
#pragma unroll
      for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) iacc[i][j] = 0;
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {           // 4 int32 words = 16 weights
        const int c = sub * 4 + cc;
        int32_t xr[GEMM_TM], wr[GEMM_TN];
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i) xr[i] = s_x[tr * GEMM_TM + i][c];
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) wr[j] = s_w[tc * GEMM_TN + j][c];
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
          for (int j = 0; j < GEMM_TN; ++j) iacc[i][j] = __dp4a(xr[i], wr[j], iacc[i][j]);
      }
#pragma unroll
      for (int j = 0; j < GEMM_TN; ++j) {
        const float dsc = s_dsc[tc * GEMM_TN + j][sub];
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i) f_acc[i][j] += dsc * (float)iacc[i][j];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < GEMM_TM; ++i) {
    const int gm = m_block + tr * GEMM_TM + i;
    if (gm >= M) continue;
    const float xs = x_scale[gm];
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) {
      const int gn = n_block + tc * GEMM_TN + j;
      if (gn < N) out[(int64_t)gm * N + gn] = float_to<OutT>(f_acc[i][j] * xs);
    }
  }
}

}  // namespace fni8
