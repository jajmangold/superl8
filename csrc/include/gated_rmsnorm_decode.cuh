// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Fused gated output RMSNorm for the Gated-DeltaNet DECODE step — HF
// `Qwen3_5RMSNormGated` ("norm BEFORE gate"): a PER-HEAD RMS over head_v_dim
// (vd), then the per-head gain, then the z gate (SiLU), in that order. One
// launch replaces the eager pow/mean/rsqrt/mul/silu swarm (~6 ops).
//
//   r      = rsqrt( mean_d( o[h,d]^2 ) + eps )
//   out    = o[h,d] * r * gain[d] * silu(z[h,d])        (z optional)
//
// Mapped one WARP per (batch, head) row (the same warp-per-output-row occupancy
// fix as deltanet_decode.cuh / gemm_decode_dp4a.cuh): at decode B*nv is tiny
// (e.g. 16), so a block-per-row launch starves the GPU. Lane `l` owns elements
// d = l + 32*j; the sum-of-squares is a `__shfl_xor_sync` tree reduction. State
// lives in registers — ZERO dynamic shared memory, no cudaFuncSetAttribute ->
// CUDA-graph-capturable. fp32 throughout (the gated norm is numerically
// load-bearing, never quantized, per AGENTS.md).
#pragma once

#include <cstdint>

namespace fni8 {

constexpr int GRN_WARP = 32;
constexpr int GRN_WARPS_PER_BLOCK = 4;
constexpr int GRN_THREADS = GRN_WARP * GRN_WARPS_PER_BLOCK;  // 128
constexpr int GRN_MAX_CHUNK = 4;  // ceil(128/32): max vd elements per lane

// o:    [N, vd]  (N = B*nv rows)   fp32
// gain: [vd]                       fp32
// z:    [N, vd] or nullptr         fp32
// out:  [N, vd]                    fp32
__global__ void __launch_bounds__(GRN_THREADS) gated_rmsnorm_decode_kernel(
    const float* __restrict__ o, const float* __restrict__ gain,
    const float* __restrict__ z, float* __restrict__ out, int N, int vd, float eps) {
  const int lane = threadIdx.x & (GRN_WARP - 1);
  const int warp_id = threadIdx.x >> 5;
  const int row = blockIdx.x * GRN_WARPS_PER_BLOCK + warp_id;
  if (row >= N) return;
  const int nchunk = (vd + GRN_WARP - 1) / GRN_WARP;  // <= GRN_MAX_CHUNK
  const int64_t row_off = (int64_t)row * vd;

  float ov[GRN_MAX_CHUNK];
  float sumsq = 0.f;
#pragma unroll
  for (int j = 0; j < GRN_MAX_CHUNK; ++j) {
    const int d = lane + GRN_WARP * j;
    const bool ok = (j < nchunk) && (d < vd);
    ov[j] = ok ? o[row_off + d] : 0.f;
    sumsq += ov[j] * ov[j];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sumsq += __shfl_xor_sync(0xFFFFFFFFu, sumsq, off);
  const float r = rsqrtf(sumsq / (float)vd + eps);

#pragma unroll
  for (int j = 0; j < GRN_MAX_CHUNK; ++j) {
    const int d = lane + GRN_WARP * j;
    if (j < nchunk && d < vd) {
      float val = ov[j] * r * gain[d];
      if (z != nullptr) {
        const float zv = z[row_off + d];
        val *= zv / (1.0f + __expf(-zv));  // SiLU gate
      }
      out[row_off + d] = val;
    }
  }
}

}  // namespace fni8
