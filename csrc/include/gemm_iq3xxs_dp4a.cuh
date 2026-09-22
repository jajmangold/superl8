// SPDX-License-Identifier: BSD-3-Clause
// ===========================================================================
// IQ3_XXS TILE (prefill) GEMM — the M > 16 sibling of gemm_decode_iq3xxs.
//
// Same spine as gemm_iq3s: one int32 word of the k-block is exactly one 4-value
// grid entry, so word c maps to sub-block ib32 = (kb%4)*2 + c/8 and grid position
// p = c%8. Verified against fni8.quant.iq3xxs in NumPy (0.0 error) before build.
//
// Block (98 B): d fp16 (+0), qs[64] plain 8-bit grid indices (+2),
//               scales_and_signs 8 x uint32 (+66).
//   aux  = u32 at 66 + 4*ib32          (assembled byte-wise: NOT 4-byte aligned)
//   db   = d * (0.5 + (aux >> 28)) * 0.5
//   sgn  = ksigns_iq2xs[(aux >> 7*(p/2)) & 127], bits (p%2)*4 + j
// Adapted (MIT->BSD-3) from llama.cpp's IQ3_XXS dot product.
// ===========================================================================
#pragma once

#include "gemm_dp4a_config.cuh"
#include "gemm_decode_iq3xxs.cuh"   // kIq3xxsGrid + kSignsIq2xs + IQ3XXS_TYPE_SIZE

namespace fni8 {

__device__ __forceinline__ uint32_t iq3xxs_aux(const uint8_t* p) {
  // The 8 aux words start at byte 66, so they are only 2-byte aligned; a u32 load
  // would be misaligned. Assemble little-endian from bytes.
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_iq3xxs_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
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
  const int64_t row_bytes = (int64_t)num_sb * IQ3XXS_TYPE_SIZE;
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
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * IQ3XXS_TYPE_SIZE;
        const int ib32 = kb4 * 2 + (c >> 3);
        const int p = c & 7;
        const uint32_t aux = iq3xxs_aux(blk + 66 + 4 * ib32);
        const uint32_t g = __ldg(&kIq3xxsGrid[blk[2 + 8 * ib32 + p]]);
        const int mb = __ldg(&kSignsIq2xs[(aux >> (7 * (p >> 1))) & 127]) >> ((p & 1) * 4);
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
          const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * IQ3XXS_TYPE_SIZE;
          const int ib32 = kb4 * 2 + s;
          const uint32_t aux = iq3xxs_aux(blk + 66 + 4 * ib32);
          dsc = __half2float(*reinterpret_cast<const __half*>(blk)) *
                (0.5f + (float)(aux >> 28)) * 0.5f;
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
