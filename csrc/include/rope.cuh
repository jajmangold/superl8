// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Fused RoPE — one launch per tensor replaces the eager gather/unsqueeze/cast +
// `_rotate_half` (chunk/neg/cat) + mul/mul/add chain. Rotates in place: each
// thread owns one pair (j, j+half), reads both before writing, so no cross-thread
// dependency. cos/sin come from fp32 tables (more accurate than the reference's
// fp16-cast); dims >= rotary_dim pass through untouched (partial rotary).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "compute_dtype.cuh"

namespace fni8 {

__device__ __forceinline__ float rope_load(const __half* p, int64_t i) {
  return to_float<__half>(p[i]);
}
__device__ __forceinline__ float rope_load(const __nv_bfloat16* p, int64_t i) {
  return to_float<__nv_bfloat16>(p[i]);
}

// grid = (N, H), block = min(half, 256). x[N,H,D] rotated in place; positions[N];
// cos/sin[P, rotary_dim] fp32. half = rotary_dim/2.
template <typename T>
__global__ void rope_kernel(T* __restrict__ x, const int64_t* __restrict__ positions,
                            const float* __restrict__ cos, const float* __restrict__ sin,
                            int N, int H, int D, int rotary_dim) {
  const int n = blockIdx.x, h = blockIdx.y;
  if (n >= N || h >= H) return;
  const int half = rotary_dim >> 1;
  T* __restrict__ row = x + ((int64_t)n * H + h) * D;
  const int64_t pos = positions[n];
  const float* __restrict__ crow = cos + pos * rotary_dim;
  const float* __restrict__ srow = sin + pos * rotary_dim;

  for (int j = threadIdx.x; j < half; j += blockDim.x) {
    const float xl = rope_load(row, j);
    const float xh = rope_load(row, j + half);
    // cos[j]==cos[j+half], sin[j]==sin[j+half] (emb = cat(freqs,freqs)); read both
    // to mirror the reference exactly.
    row[j] = float_to<T>(xl * crow[j] - xh * srow[j]);
    row[j + half] = float_to<T>(xh * crow[j + half] + xl * srow[j + half]);
  }
}

// ---- half2 path (fp16 only) ----
// Each thread owns one pair (j, j+half) and processes it via __hmul2/__hfma2
// on the healthy half2 CUDA-core pipe (~27 TFLOP/s, 2.3x fp32 on this fleet).
// cos/sin loaded as fp32 (table stays fp32 for accuracy), then converted to
// half and broadcast. Reductions/scales in fp32 per AGENTS.md (none here).
__global__ void rope_half2_kernel(__half* __restrict__ x,
                                   const int64_t* __restrict__ positions,
                                   const float* __restrict__ cos,
                                   const float* __restrict__ sin,
                                   int N, int H, int D, int rotary_dim) {
  const int n = blockIdx.x, h = blockIdx.y;
  if (n >= N || h >= H) return;
  const int half = rotary_dim >> 1;
  __half* __restrict__ row = x + ((int64_t)n * H + h) * D;
  const int64_t pos = positions[n];
  const float* __restrict__ crow = cos + pos * rotary_dim;
  const float* __restrict__ srow = sin + pos * rotary_dim;

  for (int j = threadIdx.x; j < half; j += blockDim.x) {
    __half xl = row[j];
    __half xh = row[j + half];
    // half2 pair: (xl, xh); cos/sin broadcast to half2
    __half2 h2_x = __halves2half2(xl, xh);
    __half ch = __float2half(crow[j]);
    __half sh = __float2half(srow[j]);
    __half2 h2_c = __half2half2(ch);
    __half2 h2_s = __half2half2(sh);
    // out[j]=xl*ch-xh*sh, out[j+half]=xh*ch+xl*sh
    __half2 h2_mul_c = __hmul2(h2_x, h2_c);
    __half2 rev_neg = __halves2half2(__hneg(xh), xl);
    __half2 out = __hfma2(rev_neg, h2_s, h2_mul_c);
    row[j]          = __low2half(out);
    row[j + half]   = __high2half(out);
  }
}

}  // namespace fni8
