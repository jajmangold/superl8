// SPDX-License-Identifier: BSD-3-Clause
// ===========================================================================
// IQ4_XS TILE (prefill) GEMM — the M > 16 sibling of gemm_decode_iq4xs.
//
// IQ4_XS interleaves each 32-weight sub-block BY NIBBLE: weight j is the low
// nibble of qs[j] and weight j+16 the high nibble. That looked like it would
// break the shared tile spine, and it does not: the four weights 4c..4c+3 still
// come from four CONTIGUOUS bytes qs[4*(p%4)+j]; only which nibble is read
// varies, selected by half = p/4. Verified against fni8.quant.iq4xs in NumPy
// (0.0 error) before build.
//
// Block (136 B): d fp16 (+0), scales_h uint16 (+2), scales_l[4] (+4), qs[128] (+8).
//   ls = ((scales_l[ib32/2] >> 4*(ib32%2)) & 0xF) | (((scales_h >> 2*ib32) & 3) << 4)
//   dl = d * (ls - 32)                      // 6-bit scale, split across two fields
// Adapted (MIT->BSD-3) from llama.cpp's IQ4_XS dot product.
// ===========================================================================
#pragma once

#include "gemm_dp4a_config.cuh"
#include "gemm_decode_iq4xs.cuh"   // kIq4nlValues + IQ4XS_TYPE_SIZE

namespace fni8 {

template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_iq4xs_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                  const uint8_t* __restrict__ w, OutT* __restrict__ out,
                  int M, int N, int K, int num_sb) {
  __shared__ int32_t s_x[GEMM_BM][GEMM_BK4 + 1];
  __shared__ int32_t s_w[GEMM_BN][GEMM_BK4 + 1];
  __shared__ float s_dsc[GEMM_BN][2];

  const int tid = threadIdx.x;
  const int tr = tid / (GEMM_BN / GEMM_TN);
  const int tc = tid % (GEMM_BN / GEMM_TN);
  const int m_block = blockIdx.y * GEMM_BM;
  const int n_block = blockIdx.x * GEMM_BN;
  const int K4 = K / 4;
  const int64_t row_bytes = (int64_t)num_sb * IQ4XS_TYPE_SIZE;
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
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * IQ4XS_TYPE_SIZE;
        const int ib32 = kb4 * 2 + (c >> 3);
        const int p = c & 7;
        const uint8_t* qs = blk + 8 + 16 * ib32;
        const int half = p >> 2, base = 4 * (p & 3);
        const int v0 = __ldg(&kIq4nlValues[half ? (qs[base + 0] >> 4) : (qs[base + 0] & 0xF)]);
        const int v1 = __ldg(&kIq4nlValues[half ? (qs[base + 1] >> 4) : (qs[base + 1] & 0xF)]);
        const int v2 = __ldg(&kIq4nlValues[half ? (qs[base + 2] >> 4) : (qs[base + 2] & 0xF)]);
        const int v3 = __ldg(&kIq4nlValues[half ? (qs[base + 3] >> 4) : (qs[base + 3] & 0xF)]);
        packed = (v0 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16) |
                 ((v3 & 0xFF) << 24);
      }
      s_w[r][c] = packed;
    }
    for (int r = tid; r < GEMM_BN; r += GEMM_THREADS) {
      const int gn = n_block + r;
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        float dsc = 0.f;
        if (gn < N) {
          const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * IQ4XS_TYPE_SIZE;
          const int ib32 = kb4 * 2 + s;
          const uint32_t sh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8);
          const int ls = (int)(((blk[4 + (ib32 >> 1)] >> (4 * (ib32 & 1))) & 0xF) |
                               (((sh >> (2 * ib32)) & 3) << 4));
          dsc = __half2float(*reinterpret_cast<const __half*>(blk)) * (float)(ls - 32);
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
