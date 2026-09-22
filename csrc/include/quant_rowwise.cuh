// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Fused per-row symmetric-RTN int8 activation quantizer — the single-launch
// replacement for the ~11-op eager `quantize_int8_rowwise` torch prologue that
// runs once per int8 linear (~200x/decode step). One block per row: reduce the
// row abs-max (warp `__shfl_xor_sync` + a tiny shared cross-warp pass, the
// mandated Volta reduction — no `wmma`/`cp.async`), then a second pass writes
// int8. Byte-identical to the torch reference: symmetric RTN, Q_MAX=127, CUDA
// `rintf` == torch round-half-to-even, so the store is exact after clamp.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "compute_dtype.cuh"

namespace fni8 {

// fp32 input is passed straight through; half/bf16 go through the shared
// to_float<> specializations (compute_dtype.cuh).
__device__ __forceinline__ float load_as_float(const float* p, int i) { return p[i]; }
__device__ __forceinline__ float load_as_float(const __half* p, int i) {
  return to_float<__half>(p[i]);
}
__device__ __forceinline__ float load_as_float(const __nv_bfloat16* p, int i) {
  return to_float<__nv_bfloat16>(p[i]);
}

constexpr float FNI8_Q_MAX = 127.0f;
// torch implements tensor/scalar (`amax / 127.0`) as reciprocal-MULTIPLY, so to
// match the reference scale bit-for-bit we do the same with the fp32-rounded
// reciprocal constant (NOT __fdiv_rn true division — that drifts ~1 ULP and
// flips near-.5 activations by 1). The per-element x/safe below is tensor/tensor
// = true division, so that one uses __fdiv_rn.
constexpr float FNI8_Q_MAX_INV = 1.0f / 127.0f;
constexpr int QUANT_ROW_THREADS = 256;

// grid = (M,), block = QUANT_ROW_THREADS. Each block quantizes one row of x[M,K].
template <typename T>
__global__ void quantize_i8_rowwise_kernel(const T* __restrict__ x,   // [M,K]
                                           int8_t* __restrict__ q,     // [M,K]
                                           float* __restrict__ scale,  // [M]
                                           int M, int K) {
  const int row = blockIdx.x;
  if (row >= M) return;
  const T* __restrict__ xr = x + (int64_t)row * K;
  int8_t* __restrict__ qr = q + (int64_t)row * K;
  const int tid = threadIdx.x;

  // Pass 1: per-thread abs-max, strided over K.
  float amax = 0.0f;
  for (int i = tid; i < K; i += QUANT_ROW_THREADS) {
    amax = fmaxf(amax, fabsf(load_as_float(xr, i)));
  }
  // Warp reduce, then cross-warp through shared memory.
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  __shared__ float warp_amax[QUANT_ROW_THREADS / 32];
  const int lane = tid & 31, warp = tid >> 5;
  if (lane == 0) warp_amax[warp] = amax;
  __syncthreads();
  // First warp reduces the per-warp maxima to the row amax, broadcast via shared.
  __shared__ float row_safe;
  if (warp == 0) {
    float v = (lane < (QUANT_ROW_THREADS / 32)) ? warp_amax[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if (lane == 0) {
      const float s = v * FNI8_Q_MAX_INV;   // reciprocal-multiply, == torch amax/127
      // Zero row: keep scale finite (== torch `where(scale==0, 1, scale)`), row
      // quantizes to exactly 0.
      row_safe = (s == 0.0f) ? 1.0f : s;
      scale[row] = row_safe;
    }
  }
  __syncthreads();

  // Pass 2: quantize. rintf == torch round-half-to-even; clamp then store int8.
  const float inv = row_safe;
  for (int i = tid; i < K; i += QUANT_ROW_THREADS) {
    // __fdiv_rn: correctly-rounded (round-to-nearest-even) IEEE division, matching
    // torch's aten division bit-for-bit. Plain `/` compiles to nvcc's approximate
    // division, which can drift ~1 ULP and flip a value across an N.5 rounding
    // boundary (observed: 14.4999998 -> 15 instead of 14).
    float v = rintf(__fdiv_rn(load_as_float(xr, i), inv));
    v = fminf(fmaxf(v, -FNI8_Q_MAX), FNI8_Q_MAX);
    qr[i] = (int8_t)v;
  }
}

}  // namespace fni8
