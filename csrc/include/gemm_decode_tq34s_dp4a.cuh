// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Decode-specialized fused GGUF TQ3_4S -> int8 dp4a GEMM for sm_70 — warp-per-
// column MMVQ (mirror of gemm_decode_q*k). Same math as the tile kernel
// (gemm_tq34s_dp4a.cuh), different launch/tiling: at M<=16 the BM=BN=64 tile
// launches only ceil(N/64) blocks and starves the GPU, so ONE warp per output
// column (with K-splitting handled by the tile fallback) fills the SMs.
//
// The activation arrives PRE-ROTATED + per-32 quantized from
// tq34s_rht_prepass_kernel (xq [M,K] int8, xs [M,nblk] fp32 — the prepass split
// of fni8#281, which removed the per-block redundant RHT that cost ~9x the
// memory bound at M=1). Phase 1 stages xq/xs into dynamic smem; Phase 2: one
// warp per output column streams the native TQ3_4S blocks (16 B: 4 E3M5 scale
// bytes + 12 code bytes) as one 128-bit load, unpacks each 32-block's 8 dp4a
// words with the branch-free 64-entry uint16 smem LUT, dp4a's against the smem
// activation, flushes each per-8 E3M5 scale (times max|centroid|/127 and the
// per-32 activation scale) into a per-lane fp32 accumulator, then ONE __shfl_xor
// butterfly reduces the warp. Byte-identical int math to the tile kernel.
//
// The launcher guards smem: M*K + M*(K/32)*4 <= 98304 (else the wrapper routes
// to the tile kernel, which needs no big smem). Volta rules: __dp4a +
// __shfl_xor_sync, synchronous 128-bit loads, fp32 accumulate.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "compute_dtype.cuh"
#include "gemm_decode_dp4a.cuh"   // DEC_MAX_M / DEC_WARP / DEC_WARPS_PER_BLOCK / DEC_THREADS
#include "gemm_tq34s_dp4a.cuh"    // TQ3 constants + tq34s_* device helpers

namespace fni8 {

// xq [M,K] int8 + xs [M,K/32] fp32 (prepass output), w [N, (K/32)*16] uint8
// native TQ3_4S bytes -> out [M,N] OutT. nblk = K/32. Dynamic smem: [M*K] int8
// activation + [M*nblk] fp32 per-32 scales + the 128-B uint16 pair LUT (appended
// so the launcher's single dynamic-smem budget/attribute stays exact).
template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_tq34s_kernel(const int8_t* __restrict__ xq, const float* __restrict__ xs,
                         const uint8_t* __restrict__ w, OutT* __restrict__ out,
                         int M, int N, int K, int nblk) {
  extern __shared__ __align__(16) char fused_smem[];
  int8_t* xqs = reinterpret_cast<int8_t*>(fused_smem);             // [M*K] int8
  float* xss = reinterpret_cast<float*>(xqs + ((M * K + 15) & ~15));  // [M*nblk] fp32
  uint16_t* s_lut16 = reinterpret_cast<uint16_t*>(xss + M * nblk);  // 128-B pair LUT
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  tq34s_build_lut16(s_lut16, tid, DEC_THREADS);

  // ---- Phase 1: stage the prepass xq (int8) + xs into smem (no RHT here).
  for (int t = warp; t < M * nblk; t += DEC_WARPS_PER_BLOCK) {
    const int m = t / nblk, b = t % nblk;
    const int64_t base = (int64_t)m * K + (int64_t)b * TQ3_QK;
    xqs[base + lane] = xq[base + lane];
    if (lane == 0) xss[(int64_t)m * nblk + b] = xs[(int64_t)m * nblk + b];
  }
  __syncthreads();

  // ---- Phase 2: one warp per output column, full-K dp4a from smem.
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp;
  if (n >= N) return;
  const uint8_t* wrow = w + (int64_t)n * ((int64_t)nblk * TQ3_TYPE_SIZE);

  float f_acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) f_acc[m] = 0.f;

#pragma unroll 1
  for (int blk = lane; blk < nblk; blk += DEC_WARP) {
    // One 128-bit load for the whole 16-byte native block (16-byte aligned).
    const uint4 bv = *reinterpret_cast<const uint4*>(wrow + (int64_t)blk * TQ3_TYPE_SIZE);
    float dsc[4];
#pragma unroll
    for (int g = 0; g < TQ3_GROUPS; ++g)
      dsc[g] = tq34s_decode_e3m5((uint8_t)(bv.x >> (8 * g))) * TQ3_KMAX_OVER_127;
    int32_t ww[8];
    const uint32_t y = bv.y, z = bv.z, w4 = bv.w;
    tq34s_group_words_pair(y & 0xFFu, (y >> 8) & 0xFFu, (y >> 16) & 0xFFu,
                           s_lut16, ww[0], ww[1]);
    tq34s_group_words_pair((y >> 24) & 0xFFu, z & 0xFFu, (z >> 8) & 0xFFu,
                           s_lut16, ww[2], ww[3]);
    tq34s_group_words_pair((z >> 16) & 0xFFu, (z >> 24) & 0xFFu, w4 & 0xFFu,
                           s_lut16, ww[4], ww[5]);
    tq34s_group_words_pair((w4 >> 8) & 0xFFu, (w4 >> 16) & 0xFFu,
                           (w4 >> 24) & 0xFFu, s_lut16, ww[6], ww[7]);
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m >= M) break;
      const int4* xr = reinterpret_cast<const int4*>(xqs + (int64_t)m * K);
      const int4 xv0 = xr[blk * 2], xv1 = xr[blk * 2 + 1];   // 32 int8 = 8 dp4a words
      int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      s0 = __dp4a(xv0.x, ww[0], s0); s0 = __dp4a(xv0.y, ww[1], s0);
      s1 = __dp4a(xv0.z, ww[2], s1); s1 = __dp4a(xv0.w, ww[3], s1);
      s2 = __dp4a(xv1.x, ww[4], s2); s2 = __dp4a(xv1.y, ww[5], s2);
      s3 = __dp4a(xv1.z, ww[6], s3); s3 = __dp4a(xv1.w, ww[7], s3);
      const float xscale = xss[(int64_t)m * nblk + blk];
      f_acc[m] += xscale * (dsc[0] * (float)s0 + dsc[1] * (float)s1
                          + dsc[2] * (float)s2 + dsc[3] * (float)s3);
    }
  }

#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    float v = f_acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) out[(int64_t)m * N + n] = float_to<OutT>(v);
  }
}

}  // namespace fni8
