// SPDX-License-Identifier: BSD-3-Clause
// ===========================================================================
// IQ2_XS TILE (prefill) GEMM — the M > 16 sibling of gemm_decode_iq2xs.
//
// Same spine as gemm_iq3s/gemm_iq3xxs: stage a GEMM_BK=64-wide k-chunk of
// activations into s_x, the UNPACKED weight codes (4 per int32, dp4a-ready) into
// s_w, and the scales into s_dsc, then accumulate a GEMM_TM x GEMM_TN tile.
//
// TWO DELIBERATE DIVERGENCES from the IQ3_S/IQ3_XXS tile kernels. Both are forced
// by the format; copying those kernels verbatim produces right-shaped, wrong-valued
// output that a cosine check can hide.
//
// (1) A GRID ENTRY IS EIGHT VALUES, NOT FOUR. kIq2xsGrid is [512][2] — two dp4a
//     words per entry — where kIq3sGrid/kIq3xxsGrid are one. So an int32 word is
//     HALF an entry, not a whole one, and the familiar
//         ib32 = (kb%4)*2 + c/8 ; p = c%8
//     mapping does NOT apply here. For IQ2_XS, word c of the k-block is
//         e = kb4*8 + (c >> 1)      entry within the 256-weight super-block
//         h = c & 1                 which half (which of the entry's 2 words)
//     since 64 weights = 8 entries = 16 int32 words.
//
// (2) SCALE GRANULARITY IS 16 WEIGHTS, NOT 32. The 8 scale bytes hold SIXTEEN
//     nibbles. A 64-wide k-block therefore spans FOUR scale groups, so s_dsc is
//     [GEMM_BN][4] and the accumulate loop runs 4 sub-groups of 4 words (16
//     weights) each — not 2 sub-groups of 8 words. Treating the scales as eight
//     per-32 nibbles would read only half the scale bytes.
//
// Block (74 B / 256 weights, 2.3125 bpw):
//     +0   d          fp16 super-block scale
//     +2   qs[32]     uint16: low 9 bits = grid index, top 7 bits = ksigns index
//     +66  scales[8]  TWO 4-bit scales per byte (16 groups of 16 weights)
//   entry e : ent = qs[e]; grid = kIq2xsGrid[ent & 511]; sgn = kSignsIq2xs[ent >> 9]
//             half h takes sign bits (4*h .. 4*h+3) of sgn
//   group g = e >> 1 : nib = (scales[g >> 1] >> (4 * (g & 1))) & 0xF
//             db  = d * (0.5f + nib) * 0.25f
//
// Grid magnitudes are {8, 25, 43}, so sign-flipped bytes stay in int8 range and
// feed __dp4a directly. Oracle: fni8.quant.iq2xs (fni8#317).
// Adapted (MIT->BSD-3) from llama.cpp's IQ2_XS dot product.
//
// NOTE FOR THE LAUNCHER (this bit us in #327, where every cloned tile launcher
// carried Q3K_TYPE_SIZE and the tag "Q3_K" — IQ3_S passed its whole suite anyway
// because it happens to share Q3_K's 110-byte block): this kernel must be launched
// with IQ2XS_TYPE_SIZE (74) and the tag "IQ2_XS", never a cloned constant.
// ===========================================================================
#pragma once

#include "gemm_dp4a_config.cuh"
#include "gemm_decode_iq2xs.cuh"   // kIq2xsGrid + IQ2XS_TYPE_SIZE (+ kSignsIq2xs)

namespace fni8 {

template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_iq2xs_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
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
  const int64_t row_bytes = (int64_t)num_sb * IQ2XS_TYPE_SIZE;
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
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * IQ2XS_TYPE_SIZE;
        // 74 is even and qs starts at +2, so the uint16 reads are 2-byte aligned --
        // the same assumption gemm_decode_iq2xs already makes.
        const int e = kb4 * 8 + (c >> 1);     // entry within the super-block
        const int h = c & 1;                  // which of the entry's two dp4a words
        const uint32_t ent = reinterpret_cast<const uint16_t*>(blk + 2)[e];
        const uint32_t g = __ldg(&kIq2xsGrid[ent & 511][h]);
        const int mb = __ldg(&kSignsIq2xs[ent >> 9]) >> (4 * h);
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
          const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * IQ2XS_TYPE_SIZE;
          const int grp = kb4 * 4 + s;        // 16-weight scale group in the block
          const int nib = (blk[66 + (grp >> 1)] >> (4 * (grp & 1))) & 0x0F;
          dsc = __half2float(*reinterpret_cast<const __half*>(blk)) *
                (0.5f + (float)nib) * 0.25f;
        }
        s_dsc[r][s] = dsc;
      }
    }
    __syncthreads();
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {       // 4 words = 16 weights = one scale group
      int32_t iacc[GEMM_TM][GEMM_TN];
#pragma unroll
      for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) iacc[i][j] = 0;
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
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
