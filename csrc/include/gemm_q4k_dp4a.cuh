// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Fused GGUF Q4_K -> int8 dp4a GEMM for sm_70 (native-GGUF-on-the-fly). The
// weight stays RESIDENT in its native Q4_K super-block layout (144 B / 256
// weights); this kernel unpacks each 32-element sub-block to int8 in shared
// memory and runs __dp4a inline, honoring the native 6-bit per-sub-block scale
// and min EXACTLY. No offline conversion, no per-forward fp32 dequant.
//
//   Y[m,n] = x_scale[m] * sum_sub ( d*sc_sub * <x_i8, q4_sub>  -  dmin*m_sub * sum(x_i8_sub) )
//
// where per Q4_K sub-block: d,dmin = super-block fp16 scales; sc_sub,m_sub =
// 6-bit scale/min; q4 in [0,15] (asymmetric, NO -8 centering). The min-
// correction term uses the per-sub-block activation SUM, computed inline via
// __dp4a(x, 0x01010101) — exactly llama.cpp's MMVQ `dot2`.
//
// Adapted (MIT -> BSD-3, rewritten in fni8 style) from llama.cpp:
//   block_q4_K            ggml/src/ggml-common.h:408-419
//   get_scale_min_k4      ggml/src/ggml-cuda/convert.cu:195-202
//   vec_dot_q4_K_q8_1     ggml/src/ggml-cuda/vecdotq.cuh:505-527
//   q8_1 sum (dot2)       ggml/src/ggml-cuda/vecdotq.cuh:512
// This is the same dp4a tile spine as gemm_w4a8_kernel (per-32 group-scale flush
// into an fp32 accumulator), with native-Q4_K staging + the affine min term.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "compute_dtype.cuh"
#include "gemm_dp4a_config.cuh"   // GEMM_BM/BN/BK/THREADS/TM/TN/BK4

namespace fni8 {

constexpr int Q4K_QK = 256;        // weights per super-block
constexpr int Q4K_TYPE_SIZE = 144; // 2(d)+2(dmin)+12(scales)+128(qs) bytes
constexpr int Q4K_STEP = 32;       // K-elements per scale-flush (== one sub-block)
constexpr int Q4K_STEP4 = Q4K_STEP / 4;  // = 8 int32 words

// get_scale_min_k4 (convert.cu:195-202): 6-bit sub-block scale `sc` and min `m`
// for sub-block j (0..7) from the 12 packed `scales` bytes.
__device__ __forceinline__ void q4k_scale_min(int j, const uint8_t* s,
                                               int& sc, int& m) {
  if (j < 4) {
    sc = s[j] & 63;
    m = s[j + 4] & 63;
  } else {
    sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
    m = (s[j + 4] >> 4) | ((s[j - 0] >> 6) << 4);
  }
}

// Pack four values (each 0..15 here, positive) into one dp4a int32 (little-endian).
__device__ __forceinline__ int32_t q4k_pack4(int a, int b, int c, int d) {
  return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// Grid: (ceil(N/BN), ceil(M/BM)). x [M,K] int8 (K%256==0), x_scale [M] fp32,
// w [N, (K/256)*144] uint8 native Q4_K, out [M,N] OutT (fp16/bf16). num_sb=K/256.
template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_q4k_kernel(const int8_t* __restrict__ x,       // [M,K] int8
                const float* __restrict__ x_scale,  // [M] fp32
                const uint8_t* __restrict__ w,      // [N, num_sb*144] uint8 (Q4_K)
                OutT* __restrict__ out,             // [M,N]
                int M, int N, int K, int num_sb) {
  __shared__ int32_t s_x[GEMM_BM][GEMM_BK4 + 1];  // pad to reduce bank conflicts
  __shared__ int32_t s_w[GEMM_BN][GEMM_BK4 + 1];  // pad to reduce bank conflicts
  __shared__ float s_dsc[GEMM_BN][2];          // d*sc for the 2 sub-blocks of this k-block
  __shared__ float s_dm[GEMM_BN][2];           // dmin*m for the 2 sub-blocks

  const int tid = threadIdx.x;
  const int tr = tid / (GEMM_BN / GEMM_TN);    // 0..15  thread-row
  const int tc = tid % (GEMM_BN / GEMM_TN);    // 0..7   thread-col
  const int m_block = blockIdx.y * GEMM_BM;
  const int n_block = blockIdx.x * GEMM_BN;

  const int K4 = K / 4;
  const int64_t row_bytes = (int64_t)num_sb * Q4K_TYPE_SIZE;
  const int n_kblocks = (K + GEMM_BK - 1) / GEMM_BK;  // BK=64; 4 per super-block

  float f_acc[GEMM_TM][GEMM_TN];
#pragma unroll
  for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) f_acc[i][j] = 0.f;

  for (int kb = 0; kb < n_kblocks; ++kb) {
    const int k4_base = kb * GEMM_BK4;
    const int sb = kb / 4;            // super-block index
    const int gsb = kb % 4;           // 64-elem group within super-block (0..3)
    const int j0 = 2 * gsb;           // sub-block for step 0 (low nibbles)
    const int j1 = 2 * gsb + 1;       // sub-block for step 1 (high nibbles)

    // ---- stage X (int8 -> int32-packed), zero-pad ragged M/K ----
    for (int idx = tid; idx < GEMM_BM * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gm = m_block + r, gk4 = k4_base + c;
      s_x[r][c] = (gm < M && gk4 < K4)
                      ? reinterpret_cast<const int32_t*>(x + (int64_t)gm * K)[gk4]
                      : 0;
    }
    // ---- stage W: unpack Q4_K nibbles -> int8 for this k-block's 64 columns ----
    // col c (0..15): sub-block sbl=c/8 (0=>j0 low nibble, 1=>j1 high nibble); the
    // 4 int8 come from qs[gsb*32 + (c%8)*4 .. +4].
    for (int idx = tid; idx < GEMM_BN * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gn = n_block + r;
      int32_t packed = 0;
      if (gn < N) {
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * Q4K_TYPE_SIZE;
        const uint8_t* qs = blk + 16 + gsb * 32 + (c % 8) * 4;
        const int shift = (c / 8) == 0 ? 0 : 4;   // low vs high nibble
        uint32_t qs_word = __ldg(reinterpret_cast<const uint32_t*>(qs));
        // Extract 4 nibbles: shift=0 -> low nibble (bits 0-3), shift=4 -> high (bits 4-7)
        int b0 = (qs_word >> (0 + shift)) & 0xF;
        int b1 = (qs_word >> (8 + shift)) & 0xF;
        int b2 = (qs_word >> (16 + shift)) & 0xF;
        int b3 = (qs_word >> (24 + shift)) & 0xF;
        packed = q4k_pack4(b0, b1, b2, b3);
      }
      s_w[r][c] = packed;
    }
    // ---- stage sub-block scales/mins (d*sc, dmin*m) for j0,j1 per weight row ----
    for (int r = tid; r < GEMM_BN; r += GEMM_THREADS) {
      const int gn = n_block + r;
      if (gn < N) {
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * Q4K_TYPE_SIZE;
        uint32_t d_dmin = __ldg(reinterpret_cast<const uint32_t*>(blk));
        const float d = __half2float(*reinterpret_cast<const __half*>(&d_dmin));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(reinterpret_cast<const char*>(&d_dmin) + 2));
        const uint8_t* sca = blk + 4;
        int sc0, m0, sc1, m1;
        q4k_scale_min(j0, sca, sc0, m0);
        q4k_scale_min(j1, sca, sc1, m1);
        s_dsc[r][0] = d * sc0;  s_dm[r][0] = dmin * m0;
        s_dsc[r][1] = d * sc1;  s_dm[r][1] = dmin * m1;
      } else {
        s_dsc[r][0] = s_dsc[r][1] = 0.f;
        s_dm[r][0] = s_dm[r][1] = 0.f;
      }
    }
    __syncthreads();

    // ---- two 32-element steps (sub-blocks); flush each with its affine scale ----
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      int32_t iacc[GEMM_TM][GEMM_TN];
      int32_t xsum[GEMM_TM];
#pragma unroll
      for (int i = 0; i < GEMM_TM; ++i) {
        xsum[i] = 0;
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) iacc[i][j] = 0;
      }
#pragma unroll
      for (int cc = 0; cc < Q4K_STEP4; ++cc) {
        const int c = sub * Q4K_STEP4 + cc;
        int32_t xr[GEMM_TM], wr[GEMM_TN];
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i) xr[i] = s_x[tr * GEMM_TM + i][c];
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) wr[j] = s_w[tc * GEMM_TN + j][c];
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i) {
          xsum[i] = __dp4a(xr[i], 0x01010101, xsum[i]);   // Sum of 4 int8 activations
#pragma unroll
          for (int j = 0; j < GEMM_TN; ++j)
            iacc[i][j] = __dp4a(xr[i], wr[j], iacc[i][j]);
        }
      }
      // affine flush: f += d*sc*<x,q> - dmin*m*sum(x)
#pragma unroll
      for (int j = 0; j < GEMM_TN; ++j) {
        const int wrow = tc * GEMM_TN + j;
        const float dsc = s_dsc[wrow][sub];
        const float dm = s_dm[wrow][sub];
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i)
          f_acc[i][j] += dsc * (float)iacc[i][j] - dm * (float)xsum[i];
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

// ===========================================================================
// Q5_K — identical affine (d*sc*q - dmin*m) + min-correction spine as Q4_K, but
// each weight is 5-bit: the low 4 bits from qs (same nibble interleave as Q4_K)
// plus a 5th high bit from qh[32] (1 bit/weight). Block layout (176 B):
//   d(2) dmin(2) scales[12] qh[32] qs[128]   (ggml-common.h:425-437)
// High bit of (sub-block j, position l) = (qh[l] >> j) & 1  (vecdotq.cuh:561-590,
// dequantize_row_q5_K). qs/scales share Q4_K's packing (offsets shifted by qh).
// ===========================================================================
constexpr int Q5K_TYPE_SIZE = 176;  // 2+2+12+32+128

template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_q5k_kernel(const int8_t* __restrict__ x,
                const float* __restrict__ x_scale,
                const uint8_t* __restrict__ w,      // [N, num_sb*176] uint8 (Q5_K)
                OutT* __restrict__ out,
                int M, int N, int K, int num_sb) {
  __shared__ int32_t s_x[GEMM_BM][GEMM_BK4 + 1];  // pad to reduce bank conflicts
  __shared__ int32_t s_w[GEMM_BN][GEMM_BK4 + 1];  // pad to reduce bank conflicts
  __shared__ float s_dsc[GEMM_BN][2];
  __shared__ float s_dm[GEMM_BN][2];

  const int tid = threadIdx.x;
  const int tr = tid / (GEMM_BN / GEMM_TN);
  const int tc = tid % (GEMM_BN / GEMM_TN);
  const int m_block = blockIdx.y * GEMM_BM;
  const int n_block = blockIdx.x * GEMM_BN;

  const int K4 = K / 4;
  const int64_t row_bytes = (int64_t)num_sb * Q5K_TYPE_SIZE;
  const int n_kblocks = (K + GEMM_BK - 1) / GEMM_BK;

  float f_acc[GEMM_TM][GEMM_TN];
#pragma unroll
  for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) f_acc[i][j] = 0.f;

  for (int kb = 0; kb < n_kblocks; ++kb) {
    const int k4_base = kb * GEMM_BK4;
    const int sb = kb / 4;
    const int gsb = kb % 4;
    const int j0 = 2 * gsb, j1 = 2 * gsb + 1;

    for (int idx = tid; idx < GEMM_BM * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gm = m_block + r, gk4 = k4_base + c;
      s_x[r][c] = (gm < M && gk4 < K4)
                      ? reinterpret_cast<const int32_t*>(x + (int64_t)gm * K)[gk4]
                      : 0;
    }
    for (int idx = tid; idx < GEMM_BN * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gn = n_block + r;
      int32_t packed = 0;
      if (gn < N) {
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * Q5K_TYPE_SIZE;
        const uint8_t* qh = blk + 16;                 // 32 bytes, 1 bit/weight
        const uint8_t* qs = blk + 48 + gsb * 32 + (c % 8) * 4;
        const int sbl = c / 8;                        // 0=>j0 low nibble, 1=>j1 high
        const int jsub = sbl == 0 ? j0 : j1;          // sub-block index for the qh bit
        const int lo_shift = sbl == 0 ? 0 : 4;
        uint32_t qs_word = __ldg(reinterpret_cast<const uint32_t*>(qs));
        uint32_t qh_word = __ldg(reinterpret_cast<const uint32_t*>(qh + (c % 8) * 4));
        int b[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          uint8_t qbyte = (qs_word >> (t * 8)) & 0xFF;
          uint8_t qhbyte = (qh_word >> (t * 8)) & 0xFF;
          int lo = (qbyte >> lo_shift) & 0xF;
          int hi = (qhbyte >> jsub) & 1;
          b[t] = lo | (hi << 4);
        }
        packed = q4k_pack4(b[0], b[1], b[2], b[3]);
      }
      s_w[r][c] = packed;
    }
    for (int r = tid; r < GEMM_BN; r += GEMM_THREADS) {
      const int gn = n_block + r;
      if (gn < N) {
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * Q5K_TYPE_SIZE;
        uint32_t d_dmin = __ldg(reinterpret_cast<const uint32_t*>(blk));
        const float d = __half2float(*reinterpret_cast<const __half*>(&d_dmin));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(reinterpret_cast<const char*>(&d_dmin) + 2));
        const uint8_t* sca = blk + 4;
        int sc0, m0, sc1, m1;
        q4k_scale_min(j0, sca, sc0, m0);
        q4k_scale_min(j1, sca, sc1, m1);
        s_dsc[r][0] = d * sc0;  s_dm[r][0] = dmin * m0;
        s_dsc[r][1] = d * sc1;  s_dm[r][1] = dmin * m1;
      } else {
        s_dsc[r][0] = s_dsc[r][1] = 0.f;
        s_dm[r][0] = s_dm[r][1] = 0.f;
      }
    }
    __syncthreads();

#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      int32_t iacc[GEMM_TM][GEMM_TN];
      int32_t xsum[GEMM_TM];
#pragma unroll
      for (int i = 0; i < GEMM_TM; ++i) {
        xsum[i] = 0;
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) iacc[i][j] = 0;
      }
#pragma unroll
      for (int cc = 0; cc < Q4K_STEP4; ++cc) {
        const int c = sub * Q4K_STEP4 + cc;
        int32_t xr[GEMM_TM], wr[GEMM_TN];
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i) xr[i] = s_x[tr * GEMM_TM + i][c];
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) wr[j] = s_w[tc * GEMM_TN + j][c];
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i) {
          xsum[i] = __dp4a(xr[i], 0x01010101, xsum[i]);
#pragma unroll
          for (int j = 0; j < GEMM_TN; ++j)
            iacc[i][j] = __dp4a(xr[i], wr[j], iacc[i][j]);
        }
      }
#pragma unroll
      for (int j = 0; j < GEMM_TN; ++j) {
        const int wrow = tc * GEMM_TN + j;
        const float dsc = s_dsc[wrow][sub];
        const float dm = s_dm[wrow][sub];
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i)
          f_acc[i][j] += dsc * (float)iacc[i][j] - dm * (float)xsum[i];
      }
    }
    __syncthreads();
  }

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

// ===========================================================================
// Q6_K — SYMMETRIC (no min term): y = d * sc_i * (q6 - 32), sc_i signed int8
// per-16-element scale. Block (210 B): ql[128] qh[64] scales[16](int8) d
// (ggml-common.h:443-449). 6-bit code = (ql nibble) | (qh 2-bit << 4), centered
// by -32 (vecdotq.cuh:624-644, __vsubss4(code,0x20)). Natural element order maps
// to (group g6=e/128, quadrant qd=(e%128)/32, l=(e%128)%32): qd0/2 use ql[l],
// qd1/3 use ql[l+32]; qd0/1 low nibble, qd2/3 high; qh[l] bits[2qd..2qd+1]; the
// per-16 scale index = 8*g6 + (l/16) + 2*qd. A 64-elem k-block holds 2 quadrants
// x 2 per-16 halves = 4 signed-int8 scale steps (STEP4=4 words each), no xsum.
// ===========================================================================
constexpr int Q6K_TYPE_SIZE = 210;  // 128+64+16+2

template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_q6k_kernel(const int8_t* __restrict__ x,
                const float* __restrict__ x_scale,
                const uint8_t* __restrict__ w,      // [N, num_sb*210] uint8 (Q6_K)
                OutT* __restrict__ out,
                int M, int N, int K, int num_sb) {
  __shared__ int32_t s_x[GEMM_BM][GEMM_BK4 + 1];  // pad to reduce bank conflicts
  __shared__ int32_t s_w[GEMM_BN][GEMM_BK4 + 1];  // pad to reduce bank conflicts
  __shared__ float s_dsc[GEMM_BN][4];               // d * signed-int8 scale, 4 steps/k-block

  const int tid = threadIdx.x;
  const int tr = tid / (GEMM_BN / GEMM_TN);
  const int tc = tid % (GEMM_BN / GEMM_TN);
  const int m_block = blockIdx.y * GEMM_BM;
  const int n_block = blockIdx.x * GEMM_BN;

  const int K4 = K / 4;
  const int64_t row_bytes = (int64_t)num_sb * Q6K_TYPE_SIZE;
  const int n_kblocks = (K + GEMM_BK - 1) / GEMM_BK;

  float f_acc[GEMM_TM][GEMM_TN];
#pragma unroll
  for (int i = 0; i < GEMM_TM; ++i)
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) f_acc[i][j] = 0.f;

  for (int kb = 0; kb < n_kblocks; ++kb) {
    const int k4_base = kb * GEMM_BK4;
    const int sb = kb / 4;
    const int kb4 = kb % 4;
    const int g6 = kb4 / 2;            // group of 128 (0 or 1)
    const int base_p = (kb4 % 2) * 64; // position offset within the 128-group

    for (int idx = tid; idx < GEMM_BM * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gm = m_block + r, gk4 = k4_base + c;
      s_x[r][c] = (gm < M && gk4 < K4)
                      ? reinterpret_cast<const int32_t*>(x + (int64_t)gm * K)[gk4]
                      : 0;
    }
    for (int idx = tid; idx < GEMM_BN * GEMM_BK4; idx += GEMM_THREADS) {
      const int r = idx / GEMM_BK4, c = idx % GEMM_BK4;
      const int gn = n_block + r;
      int32_t packed = 0;
      if (gn < N) {
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * Q6K_TYPE_SIZE;
        const uint8_t* ql = blk + 0 + 64 * g6;
        const uint8_t* qh = blk + 128 + 32 * g6;
        const int p0 = base_p + 4 * c;    // natural position of this column's 1st elem
        const int qd = p0 / 32;           // quadrant 0..3
        int b[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int l = (base_p + 4 * c + t) % 32;
          const uint8_t qlb = ql[(qd & 1) ? l + 32 : l];
          const int nib = (qd < 2) ? (qlb & 0xF) : (qlb >> 4);
          const int hb = (qh[l] >> (2 * qd)) & 3;
          b[t] = (nib | (hb << 4)) - 32;   // centered signed [-32,31]
        }
        packed = q4k_pack4(b[0], b[1], b[2], b[3]);
      }
      s_w[r][c] = packed;
    }
    for (int r = tid; r < GEMM_BN; r += GEMM_THREADS) {
      const int gn = n_block + r;
      if (gn < N) {
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * Q6K_TYPE_SIZE;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
        const int8_t* sca = reinterpret_cast<const int8_t*>(blk + 192);
        // Step `s` (columns [4s,4s+4)) covers natural elements [kb*64+16s, +16),
        // i.e. natural per-16 sub-block kb4*4 + s -> scales[kb4*4 + s]. (NOT
        // 8*g6+s: that mis-assigns the two odd-kb4 groups, base_p=64.)
#pragma unroll
        for (int s = 0; s < 4; ++s)
          s_dsc[r][s] = d * (float)sca[kb4 * 4 + s];
      } else {
#pragma unroll
        for (int s = 0; s < 4; ++s) s_dsc[r][s] = 0.f;
      }
    }
    __syncthreads();

    // 4 steps of 4 int32 words (16 elements = one per-16 scale group each).
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
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
    OutT* out_row = out + (int64_t)gm * N;
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) {
      const int gn = n_block + tc * GEMM_TN + j;
      if (gn < N) out_row[gn] = float_to<OutT>(f_acc[i][j] * xs);
    }
  }
}

// ===========================================================================
// Q3_K — SYMMETRIC per-16 sub-block, signed 6-bit scale, 3-bit code centered -4.
// Block (110 B): hmask[32] qs[64] scales[12] d  (ggml-common.h:396-402).
// 16 sub-blocks of 16. code = (qs 2-bit) | (hmask bit << 2), then -4 -> [-4,3].
// scale_j (signed [-32,31]) from the 12-byte scales: low 4 bits from scales[0..7],
// high 2 bits from scales[8..11], minus 32. y = d * scale_j * (q3-4). No min.
// Adapted (MIT->BSD-3) from llama.cpp vec_dot_q3_K_q8_1 (vecdotq.cuh:447-477) +
// dequantize_row_q3_K (ggml-quants.c:1243-1291). THE type for Qwen3.6-27B-Q3_K_S.
// ===========================================================================
constexpr int Q3K_TYPE_SIZE = 110;

// 6-bit signed scale for sub-block is (0..15) from the 12-byte scales block.
__device__ __forceinline__ int q3k_scale(const uint8_t* sc, int is) {
  const int lo = (sc[is % 8] >> (4 * (is / 8))) & 0xF;
  const int hi = ((sc[8 + is % 4] >> (2 * (is / 4))) & 3) << 4;
  return (lo | hi) - 32;
}

template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_q3k_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                const uint8_t* __restrict__ w, OutT* __restrict__ out,
                int M, int N, int K, int num_sb) {
  __shared__ int32_t s_x[GEMM_BM][GEMM_BK4 + 1];  // pad to reduce bank conflicts
  __shared__ int32_t s_w[GEMM_BN][GEMM_BK4 + 1];  // pad to reduce bank conflicts
  __shared__ float s_dsc[GEMM_BN][4];               // d * signed scale, 4 sub-blocks/k-block

  const int tid = threadIdx.x;
  const int tr = tid / (GEMM_BN / GEMM_TN);
  const int tc = tid % (GEMM_BN / GEMM_TN);
  const int m_block = blockIdx.y * GEMM_BM;
  const int n_block = blockIdx.x * GEMM_BN;
  const int K4 = K / 4;
  const int64_t row_bytes = (int64_t)num_sb * Q3K_TYPE_SIZE;
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
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * Q3K_TYPE_SIZE;
        const uint8_t* hmask = blk;
        const uint8_t* qs = blk + 32;
        const int is = kb4 * 4 + (c / 4);
        const int g_ = is / 8, sig = is % 8, shift = sig & 6;
        const uint8_t* qsp = qs + 32 * g_ + (sig & 1) * 16;
        const uint8_t* hmp = hmask + (sig & 1) * 16;
        const int m_shift = g_ * 4 + (sig >> 1);
        int b[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int i16 = 4 * (c % 4) + t;
          const int low2 = (qsp[i16] >> shift) & 3;
          const int hbit = (hmp[i16] >> m_shift) & 1;
          b[t] = (low2 | (hbit << 2)) - 4;
        }
        packed = q4k_pack4(b[0], b[1], b[2], b[3]);
      }
      s_w[r][c] = packed;
    }
    for (int r = tid; r < GEMM_BN; r += GEMM_THREADS) {
      const int gn = n_block + r;
      if (gn < N) {
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * Q3K_TYPE_SIZE;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk + 108));
#pragma unroll
        for (int s = 0; s < 4; ++s) s_dsc[r][s] = d * (float)q3k_scale(blk + 96, kb4 * 4 + s);
      } else {
#pragma unroll
        for (int s = 0; s < 4; ++s) s_dsc[r][s] = 0.f;
      }
    }
    __syncthreads();
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
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
    OutT* out_row = out + (int64_t)gm * N;
#pragma unroll
    for (int j = 0; j < GEMM_TN; ++j) {
      const int gn = n_block + tc * GEMM_TN + j;
      if (gn < N) out_row[gn] = float_to<OutT>(f_acc[i][j] * xs);
    }
  }
}

// ===========================================================================
// Q2_K — AFFINE per-16 sub-block, 4-bit scale + 4-bit min, 2-bit code [0,3].
// Block (84 B): scales[16] qs[64] d dmin  (ggml-common.h:379-390). y = d*sc*q -
// dmin*m; sc = scales[is]&0xF, m = scales[is]>>4. Min-correction uses the per-16
// activation sum (dp4a x,0x01010101). Adapted from vec_dot_q2_K (vecdotq.cuh:
// 364-389) + dequantize_row_q2_K (ggml-quants.c:899-929). For aggressive UD-Q2.
// ===========================================================================
constexpr int Q2K_TYPE_SIZE = 84;

template <typename OutT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_q2k_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                const uint8_t* __restrict__ w, OutT* __restrict__ out,
                int M, int N, int K, int num_sb) {
  __shared__ int32_t s_x[GEMM_BM][GEMM_BK4 + 1];  // pad to reduce bank conflicts
  __shared__ int32_t s_w[GEMM_BN][GEMM_BK4 + 1];  // pad to reduce bank conflicts
  __shared__ float s_dsc[GEMM_BN][4];               // d * 4-bit scale
  __shared__ float s_dm[GEMM_BN][4];                // dmin * 4-bit min

  const int tid = threadIdx.x;
  const int tr = tid / (GEMM_BN / GEMM_TN);
  const int tc = tid % (GEMM_BN / GEMM_TN);
  const int m_block = blockIdx.y * GEMM_BM;
  const int n_block = blockIdx.x * GEMM_BN;
  const int K4 = K / 4;
  const int64_t row_bytes = (int64_t)num_sb * Q2K_TYPE_SIZE;
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
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * Q2K_TYPE_SIZE;
        const uint8_t* qs = blk + 16;
        const int is = kb4 * 4 + (c / 4);
        const int g_ = is / 8, sig = is % 8, shift = sig & 6;
        const uint8_t* qsp = qs + 32 * g_ + (sig & 1) * 16;
        int b[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int i16 = 4 * (c % 4) + t;
          b[t] = (qsp[i16] >> shift) & 3;
        }
        packed = q4k_pack4(b[0], b[1], b[2], b[3]);
      }
      s_w[r][c] = packed;
    }
    for (int r = tid; r < GEMM_BN; r += GEMM_THREADS) {
      const int gn = n_block + r;
      if (gn < N) {
        const uint8_t* blk = w + (int64_t)gn * row_bytes + (int64_t)sb * Q2K_TYPE_SIZE;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk + 80));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 82));
        const uint8_t* sca = blk;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const int byte = sca[kb4 * 4 + s];
          s_dsc[r][s] = d * (float)(byte & 0xF);
          s_dm[r][s] = dmin * (float)(byte >> 4);
        }
      } else {
#pragma unroll
        for (int s = 0; s < 4; ++s) { s_dsc[r][s] = 0.f; s_dm[r][s] = 0.f; }
      }
    }
    __syncthreads();
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      int32_t iacc[GEMM_TM][GEMM_TN];
      int32_t xsum[GEMM_TM];
#pragma unroll
      for (int i = 0; i < GEMM_TM; ++i) {
        xsum[i] = 0;
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) iacc[i][j] = 0;
      }
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        const int c = sub * 4 + cc;
        int32_t xr[GEMM_TM], wr[GEMM_TN];
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i) xr[i] = s_x[tr * GEMM_TM + i][c];
#pragma unroll
        for (int j = 0; j < GEMM_TN; ++j) wr[j] = s_w[tc * GEMM_TN + j][c];
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i) {
          xsum[i] = __dp4a(xr[i], 0x01010101, xsum[i]);
#pragma unroll
          for (int j = 0; j < GEMM_TN; ++j) iacc[i][j] = __dp4a(xr[i], wr[j], iacc[i][j]);
        }
      }
#pragma unroll
      for (int j = 0; j < GEMM_TN; ++j) {
        const int wrow = tc * GEMM_TN + j;
        const float dsc = s_dsc[wrow][sub], dm = s_dm[wrow][sub];
#pragma unroll
        for (int i = 0; i < GEMM_TM; ++i)
          f_acc[i][j] += dsc * (float)iacc[i][j] - dm * (float)xsum[i];
      }
    }
    __syncthreads();
  }
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
