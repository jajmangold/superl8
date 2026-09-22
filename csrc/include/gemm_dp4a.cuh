// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// int8 dp4a GEMM for sm_70 — the linear-layer primitive (QKV/O projections,
// FFN, MoE experts) behind fni8-serve.
//
//   Y[M,N] = (X_i8[M,K] . W_i8[N,K]^T) * x_scale[m] * w_scale[n]
//
// sm_70 has NO int8 tensor cores; the contraction runs as a SIMT tile GEMM on
// __dp4a (int8x4 dot -> int32 accumulate on the CUDA cores). Both operands are
// K-major (X row-major [M,K]; W row-major [N,K] == W^T contraction), so dp4a
// runs along K with no transpose staging — 4 consecutive int8 read as one int32,
// little-endian, exactly matching the .fni8 `per_row_i8` resident layout.
//
// Tile (BLOCK_M=BLOCK_N=BLOCK_K=64, 128 threads): threads form a 16x8 grid; each
// owns a 4x8 register micro-tile (TM*TN=32 int32 accumulators). Per k4 step a
// thread loads 4 X words + 8 W words and issues 32 dp4a — register-blocked reuse.
// Two smem tiles (s_x + s_w) = 8 KB, well under the 48 KB static cap.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "compute_dtype.cuh"          // float_to<OutT> for fp16/bf16 store
#include "gemm_dp4a_config.cuh"        // shared tile geometry (GEMM_BM/BN/BK/THREADS/TM/TN/BK4)

namespace fni8 {

// Shared-memory XOR bank-swizzle for the dp4a tiles (s_x / s_w). The inner loop
// reads a smem COLUMN (fixed k4-word `c`, varying tile row), so with a plain
// [row][BK4] layout every 4th (s_x) / 8th (s_w) row aliases the same 32-bank set
// (BK4=16 divides 32) -> up to 8-way bank conflicts. Profiled: 274M conflicts on
// 14M shared-load requests (~19x), LSU wavefront pipe pegged at 93% replaying
// them while the dp4a/FMA pipe sat at 28%. Permuting the physical column by
// `(row>>2)` bits spreads those rows across distinct banks: verified 1-way (zero
// conflict) for BOTH s_x (rows step TM=4) and s_w (rows step TN=8) over a warp.
// XOR is a per-row bijection on [0,BK4), so the stored/loaded VALUES are
// unchanged (store and load apply the identical map) — pure layout, no numerics
// change. AGENTS.md names XOR bank-swizzle as the mandatory Volta smem pattern.
__device__ __forceinline__ int gemm_swz_col(int row, int c) {
  return c ^ ((row >> 2) & (GEMM_BK4 - 1));
}

// Grid: (ceil(N/BN), ceil(M/BM)). x [M,K] int8, w [N,K] int8 (both contiguous,
// K%4==0), x_scale [M] fp32, w_scale [N] fp32, out [M,N] OutT (fp16 or bf16 —
// the int32 accumulate/fp32 dequant are unaffected; only the store differs).
template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_w8a8_kernel(const int8_t* __restrict__ x,       // [M,K] int8
                 const float* __restrict__ x_scale,  // [M] fp32 (per activation row)
                 const int8_t* __restrict__ w,       // [N,K] int8 (per output channel)
                 const float* __restrict__ w_scale,  // [N] fp32
                 OutT* __restrict__ out,             // [M,N] fp16 or bf16
                 int M, int N, int K) {
  __shared__ int32_t s_x[GEMM_BM][GEMM_BK4];
  __shared__ int32_t s_w[GEMM_BN][GEMM_BK4];

  const int tid = threadIdx.x;
  const int tr = tid / (GEMM_BN / GEMM_TN);   // 0..15  (thread-row)
  const int tc = tid % (GEMM_BN / GEMM_TN);   // 0..7   (thread-col)
  const int m_block = blockIdx.y * GEMM_BM;
  const int n_block = blockIdx.x * GEMM_BN;

  const int K4 = K / 4;                        // total int32 words per row (K%4==0)
  const int n_kblocks = (K + GEMM_BK - 1) / GEMM_BK;

  int32_t acc[GEMM_TM][GEMM_TN];
#pragma unroll
  for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) acc[i][j] = 0;

  for (int kb = 0; kb < n_kblocks; ++kb) {
    const int k4_base = kb * GEMM_BK4;
    // ---- stage X and W tiles (int8 rows as int32; zero-pad ragged M/N/K) ----
    for (int idx = tid; idx < GEMM_BM * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gm = m_block + r, gk4 = k4_base + c;
      s_x[r][gemm_swz_col(r, c)] = (gm < M && gk4 < K4)
                      ? reinterpret_cast<const int32_t*>(x + (int64_t)gm * K)[gk4]
                      : 0;
    }
    for (int idx = tid; idx < GEMM_BN * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gn = n_block + r, gk4 = k4_base + c;
      s_w[r][gemm_swz_col(r, c)] = (gn < N && gk4 < K4)
                      ? reinterpret_cast<const int32_t*>(w + (int64_t)gn * K)[gk4]
                      : 0;
    }
    __syncthreads();

    // ---- register-blocked dp4a: 4 X words x 8 W words -> 32 MACs per k4 ----
#pragma unroll 4
    for (int c = 0; c < GEMM_BK4; ++c) {
      int32_t xr[GEMM_TM], wr[GEMM_TN];
#pragma unroll
      for (int i = 0; i < GEMM_TM; ++i) {
        const int xrow = tr * GEMM_TM + i;
        xr[i] = s_x[xrow][gemm_swz_col(xrow, c)];
      }
#pragma unroll
      for (int j = 0; j < GEMM_TN; ++j) {
        const int wrow = tc * GEMM_TN + j;
        wr[j] = s_w[wrow][gemm_swz_col(wrow, c)];
      }
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
    const int gm = m_block + tr * GEMM_TM + i;
    if (gm >= M) continue;
    const float xs = x_scale[gm];
    OutT* out_row = out + (int64_t)gm * N;
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) {
      const int gn = n_block + tc * GEMM_TN + j;
      if (gn < N) out_row[gn] = float_to<OutT>((float)acc[i][j] * xs * w_scale[gn]);
    }
  }
}

// ---------------------------------------------------------------------------
// W4A8 variant: 4-bit WEIGHTS (storage-only int4), unpacked to int8 for dp4a.
// sm_70 has no int4 matmul, so W is packed 2 signed nibbles/byte (even col = low
// nibble, matching .fni8 `per_group_i4`) and unpacked to int8 on smem-stage; the
// contraction is the same dp4a. Weights use per-GROUP scales w_scale[N, K/G]:
// the int32 accumulator is flushed into the fp32 accumulator every 32-element
// K-step with that step's group scale (G % 32 == 0 keeps each step inside one
// group), so any G in {32, 64, 128, ...} is handled without divergent logic.
//   Y[m,n] = x_scale[m] * sum_g w_scale[n,g] * (sum_{k in g} x_i8[m,k] w_i4[n,k])
constexpr int GEMM_STEP = 32;                    // K-elements per scale-flush step
constexpr int GEMM_STEP4 = GEMM_STEP / 4;        // = 8 int32 words
constexpr int GEMM_BKP = GEMM_BK / 8;            // packed uint32 words per k-block row

// Sign-extend a 4-bit nibble (0..15) to a signed int.
__device__ __forceinline__ int gemm_sx4(unsigned nib) {
  return ((int)(nib << 28)) >> 28;
}
// Pack four signed int8-range values into one dp4a int32 (little-endian).
__device__ __forceinline__ int32_t gemm_pack4(int a, int b, int c, int d) {
  return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_w4a8_kernel(const int8_t* __restrict__ x,        // [M,K] int8
                 const float* __restrict__ x_scale,   // [M] fp32
                 const uint8_t* __restrict__ w,       // [N,K/2] uint8 (2 nibbles/byte)
                 const float* __restrict__ w_scale,   // [N,K/G] fp32 (per output chan, per group)
                 OutT* __restrict__ out,              // [M,N] fp16 or bf16
                 int M, int N, int K, int G) {
  __shared__ int32_t s_x[GEMM_BM][GEMM_BK4];   // int8 activation, int32-packed
  __shared__ int32_t s_w[GEMM_BN][GEMM_BK4];   // int8 (unpacked from int4)

  const int tid = threadIdx.x;
  const int tr = tid / (GEMM_BN / GEMM_TN);
  const int tc = tid % (GEMM_BN / GEMM_TN);
  const int m_block = blockIdx.y * GEMM_BM;
  const int n_block = blockIdx.x * GEMM_BN;

  const int K4 = K / 4;
  const int num_groups = K / G;
  const int n_kblocks = (K + GEMM_BK - 1) / GEMM_BK;

  float f_acc[GEMM_TM][GEMM_TN];
#pragma unroll
  for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) f_acc[i][j] = 0.f;

  for (int kb = 0; kb < n_kblocks; ++kb) {
    const int k4_base = kb * GEMM_BK4;
    // ---- stage X (int8 -> int32-packed) ----
    for (int idx = tid; idx < GEMM_BM * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gm = m_block + r, gk4 = k4_base + c;
      s_x[r][c] = (gm < M && gk4 < K4)
                      ? reinterpret_cast<const int32_t*>(x + (int64_t)gm * K)[gk4]
                      : 0;
    }
    // ---- stage W: read packed uint32 (8 nibbles) -> two unpacked int32 ----
    const int kcol0 = kb * GEMM_BK;              // first K column of this block
    const int wp_base = kb * (GEMM_BK / 8);      // packed-uint32 column base per row
    for (int idx = tid; idx < GEMM_BN * GEMM_BKP; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BKP, pc = idx % GEMM_BKP;
      const int gn = n_block + r;
      unsigned p = 0u;
      if (gn < N && (kcol0 + pc * 8) < K)
        p = reinterpret_cast<const uint32_t*>(w + (int64_t)gn * (K / 2))[wp_base + pc];
      // nibble k (k=0..7) == column (kcol0 + pc*8 + k); low nibble first.
      s_w[r][2 * pc] = gemm_pack4(gemm_sx4(p & 0xF), gemm_sx4((p >> 4) & 0xF),
                                  gemm_sx4((p >> 8) & 0xF), gemm_sx4((p >> 12) & 0xF));
      s_w[r][2 * pc + 1] = gemm_pack4(gemm_sx4((p >> 16) & 0xF), gemm_sx4((p >> 20) & 0xF),
                                      gemm_sx4((p >> 24) & 0xF), gemm_sx4((p >> 28) & 0xF));
    }
    __syncthreads();

    // ---- two 32-element steps; flush each with its group scale ----
#pragma unroll
    for (int sub = 0; sub < GEMM_BK / GEMM_STEP; ++sub) {
      const int k_sub = kcol0 + sub * GEMM_STEP;
      if (k_sub >= K) continue;                  // whole step absent (K % 32 == 0)
      const int g = k_sub / G;
      int32_t iacc[GEMM_TM][GEMM_TN];
#pragma unroll
      for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) iacc[i][j] = 0;
#pragma unroll
      for (int cc = 0; cc < GEMM_STEP4; ++cc) {
        const int c = sub * GEMM_STEP4 + cc;
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
        const int gn = n_block + tc * GEMM_TN + j;
        const float wsj = (gn < N) ? w_scale[(int64_t)gn * num_groups + g] : 0.f;
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i) f_acc[i][j] += wsj * (float)iacc[i][j];
      }
    }
    __syncthreads();
  }

  // ---- epilogue: y = x_scale[m] * f_acc ----
#pragma unroll
  for (int i = 0; i < GEMM_TM; ++i) {
    const int gm = m_block + tr * GEMM_TM + i;
    if (gm >= M) continue;
    const float xs = x_scale[gm];
    OutT* out_row = out + (int64_t)gm * N;
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) {
      const int gn = n_block + tc * GEMM_TN + j;
      if (gn < N) out_row[gn] = float_to<OutT>(f_acc[i][j] * xs);
    }
  }
}

}  // namespace fni8
