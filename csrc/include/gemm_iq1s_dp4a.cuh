// SPDX-License-Identifier: BSD-3-Clause
// ===========================================================================
// IQ1_S TILE (prefill) GEMM — the M > 16 sibling of gemm_decode_iq1s.
// Same spine as gemm_iq3s_kernel: stage a GEMM_BK=64-wide k-chunk of
// activations into s_x, the UNPACKED weight codes (4 per int32, dp4a-ready)
// into s_w, and the per-sub-block scales into s_dsc, then accumulate a
// GEMM_TM x GEMM_TN register tile.
//
// THE ONE STRUCTURAL DIFFERENCE from IQ3_S/IQ3_XXS/IQ4_XS: their grid entries
// hold 4 codes, so one int32 word of a k-block is exactly one grid entry. An
// IQ1_S entry holds EIGHT codes (kIq1sGrid is [2048][8]), so a grid entry spans
// TWO int32 words and the mapping needs a half-entry selector. For word c in
// [0, GEMM_BK4) of k-block kb, the weights covered are 4c..4c+3, i.e. super-block
// offset (kb%4)*64 + 4c, so the entry index within the super-block is
//     e_glob = ((kb%4)*64 + 4c) / 8 = (kb%4)*8 + c/2
// and therefore
//     ib32 = e_glob / 4 = (kb % 4) * 2 + (c >> 3)   // sub-block   (as IQ3_S)
//     e    = e_glob % 4 = (c >> 1) & 3              // entry within the sub-block
//     h    = c & 1                                  // which half of the entry
//     idx  = qs[e] | (((qh >> 3*e) & 7) << 8)       // 11 bits, no sign bits
//     dsc  = d * (2*((qh >> 12) & 7) + 1) * 0.125f
// Boundaries line up: c=7 is (e=3,h=1), the last half-entry of sub-block
// kb4*2+0, and c=8 opens sub-block kb4*2+1 at (e=0,h=0).
//
// THE DELTA TRICK (why IQ1_S can use __dp4a at all): the per-sub-block +/-0.125
// delta makes the effective weight non-integer, but
//     8 * (grid_value + delta) == 8*grid_value +- 1  in  {-9,-7,-1,+1,+7,+9}
// exactly. kIq1sGrid already stores 8*grid_value (its entries are -8/0/+8), so
// the delta is a per-byte +-1 via a single __vadd4, and the compensating 1/8
// folds into the scale ONCE, as the trailing * 0.125f on s_dsc. Nothing in the
// inner loop multiplies or unpacks. Codes are bounded by 9, so a 32-weight
// sub-block accumulates at most 32*9*127 = 36576 — far inside int32.
//
// qh lives at byte 34 and is only 2-byte aligned, so it is assembled from two
// bytes rather than cast to a uint16 (same as the decode kernel).
// Oracle: fni8.quant.iq1s. Adapted (MIT->BSD-3) from llama.cpp's IQ1_S dot.
// ===========================================================================
#pragma once

#include "gemm_dp4a_config.cuh"
#include "gemm_decode_iq1s.cuh"   // kIq1sGrid + IQ1S_TYPE_SIZE

namespace fni8 {

template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_iq1s_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
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
  const int64_t row_bytes = (int64_t)num_sb * IQ1S_TYPE_SIZE;
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
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * IQ1S_TYPE_SIZE;
        const int ib32 = kb4 * 2 + (c >> 3);   // sub-block within the super-block
        const int e = (c >> 1) & 3;            // grid entry within the sub-block
        const int h = c & 1;                   // half of the 8-code entry
        const uint8_t* qs = blk + 2 + 4 * ib32;
        // qh sits at byte 34, so it is only 2-byte aligned; assemble it from bytes.
        const uint8_t* qhp = blk + 34 + 2 * ib32;
        const uint32_t qh = (uint32_t)qhp[0] | ((uint32_t)qhp[1] << 8);
        const uint32_t gidx = (uint32_t)qs[e] | (((qh >> (3 * e)) & 7u) << 8);
        // +delta -> +1 in every byte lane; -delta -> -1 (0xFF per byte).
        const uint32_t dword = (qh & 0x8000u) ? 0xFFFFFFFFu : 0x01010101u;
        const uint32_t* gw = reinterpret_cast<const uint32_t*>(&kIq1sGrid[gidx][0]);
        packed = (int32_t)__vadd4(__ldg(&gw[h]), dword);
      }
      s_w[r][c] = packed;
    }
    for (int r = tid; r < GEMM_BN; r += GEMM_THREADS) {
      const int gn = n_block + r;
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        float dsc = 0.f;
        if (gn < N) {
          const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * IQ1S_TYPE_SIZE;
          const int ib32 = kb4 * 2 + s;
          const uint8_t* qhp = blk + 34 + 2 * ib32;
          const uint32_t qh = (uint32_t)qhp[0] | ((uint32_t)qhp[1] << 8);
          // The 1/8 that pays for storing 8*grid_value lands HERE, exactly once.
          dsc = __half2float(*reinterpret_cast<const __half*>(blk)) *
                (float)(2 * ((qh >> 12) & 7) + 1) * 0.125f;
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
