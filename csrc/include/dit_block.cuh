// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Fused DiT post-attention block kernel: RMSNorm → adaLN scale/shift/gate
// apply → residual add → RoPE, all in fp32 registers, bf16 storage, ONE
// HBM round-trip for the hidden state x — collapsing N bandwidth-bound
// passes into 1 (sota-research-2026-07.md, roofline analysis).
//
// Per-token operations (memory-bound, ~1–5 FLOP/byte) dominate the DiT
// profile when each op round-trips through storage.  This kernel reads x
// from HBM ONCE, does every elementwise operation in fp32 registers, and
// writes bf16 ONCE.  RMSNorm stays fp32 (AGENTS.md: numerically
// load-bearing) — never quantized.  The adaLN scale/shift/gate use fp32
// opmath (cast on load).  int8 does NOT belong here.
//
// fp16 path: half2 vectorized loads/stores (128-bit ld.global.v4 via
// packed half2/bf162 reads) for the HBM load (phase 1) and HBM store
// (phase 5) — the Volta mandatory pattern per AGENTS.md.  ALL arithmetic
// stays fp32 (the sum-of-squares reduction is load-bearing) — only memory
// access is vectorized.  The vec2 kernel launches for even D; odd D falls
// back to the scalar kernel.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "compute_dtype.cuh"

namespace fni8 {

constexpr int DIT_BLOCK_THREADS = 256;

// ---- half2/bf162 vectorization helpers (issue #126, AGENTS.md Volta patterns) ----
// The fp16/bf16 packed CUDA-core pipe is healthy on this fleet, so 32-bit
// vec2 loads/stores cut memory transactions in half for the (memory-bound)
// DiT block.  ALL arithmetic stays fp32.
template <typename T> struct dit_vec2 {};
template <> struct dit_vec2<__half> { using type = __half2; };
template <> struct dit_vec2<__nv_bfloat16> { using type = __nv_bfloat162; };

__device__ __forceinline__ float2 dit_vec2_to_f2(__half2 v) { return __half22float2(v); }
__device__ __forceinline__ float2 dit_vec2_to_f2(__nv_bfloat162 v) { return __bfloat1622float2(v); }

template <typename Vec>
__device__ __forceinline__ Vec dit_f2_to_vec(float2 f);
template <>
__device__ __forceinline__ __half2 dit_f2_to_vec<__half2>(float2 f) {
  return __float22half2_rn(f);
}
template <>
__device__ __forceinline__ __nv_bfloat162 dit_f2_to_vec<__nv_bfloat162>(float2 f) {
  return __float22bfloat162_rn(f);
}

// ---- scalar kernel (odd D fallback) ----
// grid = (M,), block = DIT_BLOCK_THREADS.  x[M,D], rms_weight[D],
// scale[M,D], shift[M,D].  Gate (optional) [M,D]: if HAS_GATE, applies
// sigmoid(gate) * modulated.  RoPE (optional): if HAS_ROPE, applies
// rotary to the first `rotary_dim` elements per row.
// Shared memory holds D floats for the intermediate row state.
template <typename T, bool HAS_GATE, bool HAS_ROPE>
__global__ void dit_block_kernel(const T* __restrict__ x,
                                 const T* __restrict__ rms_weight,
                                 const T* __restrict__ scale,
                                 const T* __restrict__ shift,
                                 const T* __restrict__ gate,
                                 T* __restrict__ out,
                                 float eps,
                                 const int64_t* __restrict__ positions,
                                 const float* __restrict__ cos_table,
                                 const float* __restrict__ sin_table,
                                 int rotary_dim,
                                 int M, int D) {
  const int row = blockIdx.x;
  if (row >= M) return;
  const int64_t base = (int64_t)row * D;
  const int tid = threadIdx.x;

  extern __shared__ float smem_float[];  // [D] intermediate row values in fp32

  // ---- Phase 1: load x from HBM, accumulate ssq, store fp32 in smem ----
  float ssq = 0.0f;
  for (int i = tid; i < D; i += DIT_BLOCK_THREADS) {
    float v = to_float<T>(x[base + i]);
    smem_float[i] = v;
    ssq += v * v;
  }

  // ---- Phase 2: warp-reduce ssq → inv_rms ----
  for (int off = 16; off > 0; off >>= 1)
    ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
  __shared__ float warp_ssq[DIT_BLOCK_THREADS / 32];
  const int lane = tid & 31, warp = tid >> 5;
  if (lane == 0) warp_ssq[warp] = ssq;
  __syncthreads();

  __shared__ float inv_rms;
  if (warp == 0) {
    float v = (lane < (DIT_BLOCK_THREADS / 32)) ? warp_ssq[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) inv_rms = rsqrtf(v / (float)D + eps);
  }
  __syncthreads();

  // ---- Phase 3: norm → adaLN → gate → residual, write back to smem ----
  const float inv = inv_rms;
  for (int i = tid; i < D; i += DIT_BLOCK_THREADS) {
    float v = smem_float[i];
    float w = to_float<T>(rms_weight[i]);
    // 1. RMSNorm
    float normed = v * inv * w;
    // 2. adaLN: modulated = normed * (1 + scale) + shift
    float s = to_float<T>(scale[base + i]);
    float sh = to_float<T>(shift[base + i]);
    float modulated = normed * (1.0f + s) + sh;
    // 3. Gate (optional): modulated *= sigmoid(gate)
    if constexpr (HAS_GATE) {
      float g = to_float<T>(gate[base + i]);
      modulated *= 1.0f / (1.0f + __expf(-g));
    }
    // 4. Residual add
    smem_float[i] = modulated + v;
  }
  __syncthreads();

  // ---- Phase 4: RoPE (optional) ----
  if constexpr (HAS_ROPE) {
    int half = rotary_dim >> 1;
    int64_t pos = positions[row];
    const float* crow = cos_table + pos * rotary_dim;
    const float* srow = sin_table + pos * rotary_dim;
    for (int j = tid; j < half; j += DIT_BLOCK_THREADS) {
      float xl = smem_float[j];
      float xh = smem_float[j + half];
      smem_float[j]          = xl * crow[j] - xh * srow[j];
      smem_float[j + half]   = xh * crow[j + half] + xl * srow[j + half];
    }
    __syncthreads();
  }

  // ---- Phase 5: write back to HBM as T (bf16/fp16) ----
  for (int i = tid; i < D; i += DIT_BLOCK_THREADS) {
    out[base + i] = float_to<T>(smem_float[i]);
  }
}

// ---- vec2 kernel (even D — vectorized 128-bit HBM loads/stores) ----
// Duplicates the scalar kernel's logic but uses half2/bf162 vec2 loads
// and stores for the HBM phases.  The smem compute and RoPE phases are
// identical to the scalar kernel.  Launched ONLY for even D (D2 = D/2
// integer, and every row's HBM address is 4-byte-aligned for vec2 access).
template <typename T, bool HAS_GATE, bool HAS_ROPE>
__global__ void dit_block_kernel_vec2(const T* __restrict__ x,
                                      const T* __restrict__ rms_weight,
                                      const T* __restrict__ scale,
                                      const T* __restrict__ shift,
                                      const T* __restrict__ gate,
                                      T* __restrict__ out,
                                      float eps,
                                      const int64_t* __restrict__ positions,
                                      const float* __restrict__ cos_table,
                                      const float* __restrict__ sin_table,
                                      int rotary_dim,
                                      int M, int D) {
  using Vec = typename dit_vec2<T>::type;
  const int row = blockIdx.x;
  if (row >= M) return;
  const int64_t base = (int64_t)row * D;
  const int tid = threadIdx.x;
  const int D2 = D >> 1;  // D is even (guaranteed by dispatcher)

  extern __shared__ float smem_float[];  // [D] intermediate row values in fp32
  const Vec* __restrict__ xv = reinterpret_cast<const Vec*>(x + base);

  // ---- Phase 1: load x from HBM (vec2), accumulate ssq, store fp32 in smem ----
  float ssq = 0.0f;
  for (int j = tid; j < D2; j += DIT_BLOCK_THREADS) {
    float2 f = dit_vec2_to_f2(xv[j]);
    smem_float[2 * j]     = f.x;
    smem_float[2 * j + 1] = f.y;
    ssq += f.x * f.x + f.y * f.y;
  }

  // ---- Phase 2: warp-reduce ssq → inv_rms ----
  for (int off = 16; off > 0; off >>= 1)
    ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
  __shared__ float warp_ssq[DIT_BLOCK_THREADS / 32];
  const int lane = tid & 31, warp = tid >> 5;
  if (lane == 0) warp_ssq[warp] = ssq;
  __syncthreads();

  __shared__ float inv_rms;
  if (warp == 0) {
    float v = (lane < (DIT_BLOCK_THREADS / 32)) ? warp_ssq[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) inv_rms = rsqrtf(v / (float)D + eps);
  }
  __syncthreads();

  // ---- Phase 3: norm → adaLN → gate → residual, write back to smem ----
  const float inv = inv_rms;
  for (int i = tid; i < D; i += DIT_BLOCK_THREADS) {
    float v = smem_float[i];
    float w = to_float<T>(rms_weight[i]);
    float normed = v * inv * w;
    float s = to_float<T>(scale[base + i]);
    float sh = to_float<T>(shift[base + i]);
    float modulated = normed * (1.0f + s) + sh;
    if constexpr (HAS_GATE) {
      float g = to_float<T>(gate[base + i]);
      modulated *= 1.0f / (1.0f + __expf(-g));
    }
    smem_float[i] = modulated + v;
  }
  __syncthreads();

  // ---- Phase 4: RoPE (optional) — same as scalar kernel ----
  if constexpr (HAS_ROPE) {
    int half = rotary_dim >> 1;
    int64_t pos = positions[row];
    const float* crow = cos_table + pos * rotary_dim;
    const float* srow = sin_table + pos * rotary_dim;
    for (int j = tid; j < half; j += DIT_BLOCK_THREADS) {
      float xl = smem_float[j];
      float xh = smem_float[j + half];
      smem_float[j]          = xl * crow[j] - xh * srow[j];
      smem_float[j + half]   = xh * crow[j + half] + xl * srow[j + half];
    }
    __syncthreads();
  }

  // ---- Phase 5: write back to HBM (vec2) ----
  Vec* __restrict__ ov = reinterpret_cast<Vec*>(out + base);
  for (int j = tid; j < D2; j += DIT_BLOCK_THREADS) {
    float2 f = {smem_float[2 * j], smem_float[2 * j + 1]};
    ov[j] = dit_f2_to_vec<Vec>(f);
  }
}

}  // namespace fni8
