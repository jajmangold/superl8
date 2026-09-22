// SPDX-License-Identifier: BSD-3-Clause
// ===========================================================================
// IQ2_XXS TILE (prefill) GEMM — the M > 16 sibling of gemm_decode_iq2xxs.
//
// Same spine as gemm_iq3s/gemm_iq3xxs: stage a GEMM_BK=64-wide k-chunk of
// activations into s_x, the UNPACKED weight codes (4 per int32, dp4a-ready) into
// s_w, and the per-sub-block scales into s_dsc, then accumulate a
// GEMM_TM x GEMM_TN register tile.
//
// MAPPING — this is where IQ2_XXS DIVERGES from the IQ3 family, so read before
// copying: IQ3_XXS packs a 32-weight sub-block as EIGHT grid entries x 4 values,
// making "one int32 word == one grid entry". IQ2_XXS packs it as FOUR entries x
// EIGHT values, so one grid entry spans TWO dp4a words and the word index must be
// split in two. With c in [0, GEMM_BK4) the word index inside the sub-block is
// cc = c % 8, and cc = 2*l + h:
//     ib32 = (kb % 4) * 2 + (c / 8)   sub-block within the 256-weight super-block
//     l    = (c % 8) / 2              which of the FOUR grid entries (0..3)
//     h    = (c % 8) % 2              which half: h=0 -> values 0-3, h=1 -> 4-7
// This is exactly the q[2*l + h] ordering gemm_decode_iq2xxs feeds to dp4a.
//
// Block (66 B / 256 weights, 2.0625 bpw):
//     +0   d        fp16 super-block scale
//     +2   qs[64]   EIGHT sub-blocks x TWO uint32 (aux0, aux1)
//   aux0 = u32 at 2 + 8*ib32        FOUR 8-bit grid indices, one per byte
//   aux1 = u32 at 2 + 8*ib32 + 4    top nibble = scale, four 7-bit sign indices
//   db   = d * (0.5 + (aux1 >> 28)) * 0.25        // quarter-offset, NOT 0.5 like IQ3_XXS
//   idx  = (aux0 >> 8*l) & 0xFF   ->  g = kIq2xxsGrid[idx][h]
//   sgn  = ksigns_iq2xs[(aux1 >> 7*l) & 127], the h-th nibble driving these 4 values
// Note the sign handling is bit-identical IN FORM to IQ3_XXS (LUT index from a
// 7-bit field, nibble select), only the grid lookup differs.
//
// The 66-B block puts every aux word at an offset == 2 (mod 4), so they are
// 2-byte aligned ONLY and must go through dec_ld_u32, never a u32 reinterpret.
// Grid magnitudes are {8, 25, 43}, so sign-flipped bytes stay well inside int8
// and feed __dp4a directly.
//
// kIq2xxsGrid / IQ2XXS_TYPE_SIZE / kSignsIq2xs / dec_ld_u32 are all REUSED from
// the decode header rather than duplicated (duplicated tables and cloned
// constants have caused several bugs on this path already — see fni8#327).
// Oracle: fni8.quant.iq2xxs. Adapted (MIT->BSD-3) from llama.cpp's IQ2_XXS dot.
// ===========================================================================
#pragma once

#include "gemm_dp4a_config.cuh"
#include "gemm_decode_iq2xxs.cuh"   // kIq2xxsGrid + IQ2XXS_TYPE_SIZE + kSignsIq2xs
                                    // (and, transitively, kSignsIq2xs/dec_ld_u32)

namespace fni8 {

template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_iq2xxs_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                   const uint8_t* __restrict__ w, OutT* __restrict__ out,
                   int M, int N, int K, int num_sb) {
  __shared__ int32_t s_x[GEMM_BM][GEMM_BK4 + 1];
  __shared__ int32_t s_w[GEMM_BN][GEMM_BK4 + 1];
  __shared__ float s_dsc[GEMM_BN][2];          // 2 sub-blocks per 64-wide k-block

  const int tid = threadIdx.x;
  const int tr = tid / (GEMM_BN / GEMM_TN);
  const int tc = tid % (GEMM_BN / GEMM_TN);
  const int m_block = blockIdx.y * GEMM_BM;
  const int n_block = blockIdx.x * GEMM_BN;
  const int K4 = K / 4;
  const int64_t row_bytes = (int64_t)num_sb * IQ2XXS_TYPE_SIZE;
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
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * IQ2XXS_TYPE_SIZE;
        const int ib32 = kb4 * 2 + (c >> 3);
        const int cc = c & 7;
        const int l = cc >> 1;          // grid entry 0..3 (each spans 8 values)
        const int h = cc & 1;           // half of that entry: 0 -> vals 0-3, 1 -> 4-7
        const uint8_t* pair = blk + 2 + 8 * ib32;
        const uint32_t aux0 = dec_ld_u32(pair);        // four 8-bit grid indices
        const uint32_t aux1 = dec_ld_u32(pair + 4);    // scale nibble + four sign idx
        const uint32_t gidx = (aux0 >> (8 * l)) & 0xFFu;
        const uint32_t g = __ldg(&kIq2xxsGrid[gidx][h]);
        const int mb = __ldg(&kSignsIq2xs[(aux1 >> (7 * l)) & 127]) >> (4 * h);
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
      for (int s = 0; s < 2; ++s) {
        float dsc = 0.f;
        if (gn < N) {
          const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * IQ2XXS_TYPE_SIZE;
          const int ib32 = kb4 * 2 + s;
          const uint32_t aux1 = dec_ld_u32(blk + 2 + 8 * ib32 + 4);
          dsc = __half2float(*reinterpret_cast<const __half*>(blk)) *
                (0.5f + (float)(aux1 >> 28)) * 0.25f;
        }
        s_dsc[r][s] = dsc;
      }
    }
    __syncthreads();
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      int32_t iacc[GEMM_TM][GEMM_TN];
#pragma unroll
      for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) iacc[i][j] = 0;
#pragma unroll
      for (int cc = 0; cc < 8; ++cc) {
        const int c = sub * 8 + cc;
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
