// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// MoE grouped/batched int8 dp4a GEMM for sm_70 — all active experts for a
// token batch in ONE launch instead of looping `gemm_w8a8` per expert.
//
// Tokens are assumed pre-sorted so each expert's rows are CONTIGUOUS in x (a
// stable argsort by expert id, done once in the Python wrapper). The launcher
// walks `group_sizes[E]` on the host and emits one M-tile descriptor
// (expert, row_start, row_end) per BLOCK_M-sized chunk of each expert's
// segment — a ragged expert (or a zero-token expert) costs exactly as many
// blocks as it needs, never GEMM_BM-padded waste across an expert boundary.
// The N/K tile math is identical to gemm_w8a8_kernel; only the M-tile lookup
// and the per-expert W/W_scale slice are new, so it reuses the same tile
// constants for a matching register/smem footprint.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "gemm_dp4a_config.cuh"   // shared tile geometry only — NOT the dense
                                  // __global__ kernels (would multiply-define)

namespace fni8 {

// Grid: (n_m_tiles, ceil(N/BN)). x [total_M,K] int8 (rows contiguous per
// expert), x_scale [total_M] fp32, w [E,N,K] int8, w_scale [E,N] fp32,
// tile_expert/tile_row_start/tile_row_end [n_m_tiles] int32 (row_end
// exclusive, row_end - row_start <= GEMM_BM), out [total_M,N] fp16.
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_grouped_w8a8_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                         const int8_t* __restrict__ w, const float* __restrict__ w_scale,
                         const int32_t* __restrict__ tile_expert,
                         const int32_t* __restrict__ tile_row_start,
                         const int32_t* __restrict__ tile_row_end,
                         __half* __restrict__ out, int N, int K) {
  __shared__ int32_t s_x[GEMM_BM][GEMM_BK4];
  __shared__ int32_t s_w[GEMM_BN][GEMM_BK4];

  const int tid = threadIdx.x;
  const int tr = tid / (GEMM_BN / GEMM_TN);   // 0..15  (thread-row)
  const int tc = tid % (GEMM_BN / GEMM_TN);   // 0..7   (thread-col)

  const int m_tile = blockIdx.x;
  const int row_start = tile_row_start[m_tile];
  const int row_end = tile_row_end[m_tile];
  const int e = tile_expert[m_tile];
  const int n_block = blockIdx.y * GEMM_BN;

  const int8_t* w_e = w + (int64_t)e * N * K;
  const float* w_scale_e = w_scale + (int64_t)e * N;

  const int K4 = K / 4;
  const int n_kblocks = (K + GEMM_BK - 1) / GEMM_BK;

  int32_t acc[GEMM_TM][GEMM_TN];
#pragma unroll
  for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) acc[i][j] = 0;

  for (int kb = 0; kb < n_kblocks; ++kb) {
    const int k4_base = kb * GEMM_BK4;
    // ---- stage X and W tiles (int8 rows as int32; zero-pad ragged tile/N/K) ----
    for (int idx = tid; idx < GEMM_BM * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gm = row_start + r, gk4 = k4_base + c;
      s_x[r][c] = (gm < row_end && gk4 < K4)
                      ? reinterpret_cast<const int32_t*>(x + (int64_t)gm * K)[gk4]
                      : 0;
    }
    for (int idx = tid; idx < GEMM_BN * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gn = n_block + r, gk4 = k4_base + c;
      s_w[r][c] = (gn < N && gk4 < K4)
                      ? reinterpret_cast<const int32_t*>(w_e + (int64_t)gn * K)[gk4]
                      : 0;
    }
    __syncthreads();

    // ---- register-blocked dp4a: 4 X words x 8 W words -> 32 MACs per k4 ----
#pragma unroll 4
    for (int c = 0; c < GEMM_BK4; ++c) {
      int32_t xr[GEMM_TM], wr[GEMM_TN];
#pragma unroll
      for (int i = 0; i < GEMM_TM; ++i) xr[i] = s_x[tr * GEMM_TM + i][c];
#pragma unroll
      for (int j = 0; j < GEMM_TN; ++j) wr[j] = s_w[tc * GEMM_TN + j][c];
#pragma unroll
      for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) acc[i][j] = __dp4a(xr[i], wr[j], acc[i][j]);
    }
    __syncthreads();  // tiles reused next k-block
  }

  // ---- epilogue: dequant (single multiply per element) + fp16 store ----
#pragma unroll
  for (int i = 0; i < GEMM_TM; ++i) {
    const int gm = row_start + tr * GEMM_TM + i;
    if (gm >= row_end) continue;
    const float xs = x_scale[gm];
    __half* out_row = out + (int64_t)gm * N;
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) {
      const int gn = n_block + tc * GEMM_TN + j;
      if (gn < N) out_row[gn] = __float2half((float)acc[i][j] * xs * w_scale_e[gn]);
    }
  }
}

}  // namespace fni8
