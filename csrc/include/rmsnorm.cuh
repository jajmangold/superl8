// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Fused RMSNorm — one launch replaces the eager float()/pow/mean/rsqrt/mul/mul/
// cast chain (+ optional residual add) that ran twice per decoder layer plus
// per-head QK-norm. One block per row: (optional) residual add in fp32 opmath
// (bit-identical to torch's half/bf16 add, which also upcasts to float), fp32
// sum-of-squares reduction (`__shfl_xor_sync` + shared cross-warp — the mandated
// Volta pattern), `rsqrt`, scale by the (optionally 1+w Gemma) weight. The
// reduction stays fp32 (AGENTS.md: numerically load-bearing) — never quantized.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "compute_dtype.cuh"

namespace fni8 {

__device__ __forceinline__ float rms_load(const float* p, int i) { return p[i]; }
__device__ __forceinline__ float rms_load(const __half* p, int i) { return to_float<__half>(p[i]); }
__device__ __forceinline__ float rms_load(const __nv_bfloat16* p, int i) {
  return to_float<__nv_bfloat16>(p[i]);
}

// half2/FP16×2 vectorization helpers (issue #126). The fp16/bf16 packed CUDA-core pipe
// is healthy on this fleet, so wide 32-bit vec2 loads/stores cut memory transactions in
// half for the (memory-bound) RMSNorm. ALL arithmetic stays fp32 (the sum-of-squares
// reduction is load-bearing per AGENTS.md; an fp16 normalize would blow the cos>0.9999
// gate) — only the memory access is vectorized.
template <typename T> struct rms_vec2 {};
template <> struct rms_vec2<__half> { using type = __half2; };
template <> struct rms_vec2<__nv_bfloat16> { using type = __nv_bfloat162; };

__device__ __forceinline__ float2 rms_vec2_to_f2(__half2 v) { return __half22float2(v); }
__device__ __forceinline__ float2 rms_vec2_to_f2(__nv_bfloat162 v) { return __bfloat1622float2(v); }

template <typename Vec> __device__ __forceinline__ Vec rms_f2_to_vec(float2 f);
template <> __device__ __forceinline__ __half2 rms_f2_to_vec<__half2>(float2 f) {
  return __float22half2_rn(f);
}
template <> __device__ __forceinline__ __nv_bfloat162 rms_f2_to_vec<__nv_bfloat162>(float2 f) {
  return __float22bfloat162_rn(f);
}

constexpr int RMSNORM_THREADS = 256;

// grid = (M,), block = RMSNORM_THREADS. x[M,D], weight[D]. If `res` is non-null,
// writes xr[i] = x[i] + res[i] (fp32 opmath, cast to T) and normalizes over xr;
// otherwise normalizes over x. `unit_offset` uses (1 + w) (Gemma).
template <typename T, bool HAS_RES, bool UNIT_OFFSET>
__global__ void rmsnorm_kernel(const T* __restrict__ x,        // [M,D]
                               const T* __restrict__ res,       // [M,D] or null
                               const T* __restrict__ weight,    // [D]
                               T* __restrict__ out,             // [M,D] normed
                               T* __restrict__ xr_out,          // [M,D] x+res, or null
                               float eps, int M, int D) {
  const int row = blockIdx.x;
  if (row >= M) return;
  const int64_t base = (int64_t)row * D;
  const int tid = threadIdx.x;

  // Pass 1: (optional) residual add, accumulate fp32 sum of squares.
  float ssq = 0.0f;
  for (int i = tid; i < D; i += RMSNORM_THREADS) {
    float v = rms_load(x, base + i);
    if (HAS_RES) {
      v += rms_load(res, base + i);          // fp32 opmath == torch half/bf16 add
      xr_out[base + i] = float_to<T>(v);
    }
    ssq += v * v;
  }
  for (int off = 16; off > 0; off >>= 1) ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
  __shared__ float warp_ssq[RMSNORM_THREADS / 32];
  const int lane = tid & 31, warp = tid >> 5;
  if (lane == 0) warp_ssq[warp] = ssq;
  __syncthreads();
  __shared__ float inv_rms;
  if (warp == 0) {
    float v = (lane < (RMSNORM_THREADS / 32)) ? warp_ssq[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) inv_rms = rsqrtf(v / (float)D + eps);
  }
  __syncthreads();

  // Pass 2: normalize + weight, cast back. Re-derive v (from xr if residual).
  const float inv = inv_rms;
  const T* __restrict__ src = HAS_RES ? xr_out : x;
  for (int i = tid; i < D; i += RMSNORM_THREADS) {
    float v = rms_load(src, base + i) * inv;
    float w = rms_load(weight, i);
    if (UNIT_OFFSET) w += 1.0f;
    out[base + i] = float_to<T>(v * w);
  }
}

// half2-vectorized RMSNorm — identical math to rmsnorm_kernel but 2 elements per step via
// vec2 (half2/bf162) loads/stores. Launched ONLY for even D (issue #126): then every row
// base = row*D is even, so (ptr + base) meets half2's 4-byte alignment and D splits evenly
// into D/2 pairs with no tail. Odd D falls back to the scalar rmsnorm_kernel (see launcher).
// All arithmetic is fp32 → output is bit-identical to the scalar kernel.
template <typename T, bool HAS_RES, bool UNIT_OFFSET>
__global__ void rmsnorm_kernel_vec2(const T* __restrict__ x,       // [M,D]
                                    const T* __restrict__ res,      // [M,D] or null
                                    const T* __restrict__ weight,   // [D]
                                    T* __restrict__ out,            // [M,D] normed
                                    T* __restrict__ xr_out,         // [M,D] x+res, or null
                                    float eps, int M, int D) {
  using Vec = typename rms_vec2<T>::type;
  const int row = blockIdx.x;
  if (row >= M) return;
  const int64_t base = (int64_t)row * D;
  const int tid = threadIdx.x;
  const int D2 = D >> 1;  // D is even (launcher guarantees it)

  const Vec* __restrict__ xv = reinterpret_cast<const Vec*>(x + base);
  const Vec* __restrict__ rv = HAS_RES ? reinterpret_cast<const Vec*>(res + base) : nullptr;
  Vec* __restrict__ xrv = HAS_RES ? reinterpret_cast<Vec*>(xr_out + base) : nullptr;

  // Pass 1: (optional) residual add in fp32, accumulate fp32 sum of squares, 2 elems/step.
  float ssq = 0.0f;
  for (int j = tid; j < D2; j += RMSNORM_THREADS) {
    float2 f = rms_vec2_to_f2(xv[j]);
    if (HAS_RES) {
      float2 rf = rms_vec2_to_f2(rv[j]);
      f.x += rf.x;                             // fp32 opmath == torch half/bf16 add
      f.y += rf.y;
      xrv[j] = rms_f2_to_vec<Vec>(f);
    }
    ssq += f.x * f.x + f.y * f.y;
  }
  for (int off = 16; off > 0; off >>= 1) ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
  __shared__ float warp_ssq[RMSNORM_THREADS / 32];
  const int lane = tid & 31, warp = tid >> 5;
  if (lane == 0) warp_ssq[warp] = ssq;
  __syncthreads();
  __shared__ float inv_rms;
  if (warp == 0) {
    float v = (lane < (RMSNORM_THREADS / 32)) ? warp_ssq[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) inv_rms = rsqrtf(v / (float)D + eps);
  }
  __syncthreads();

  // Pass 2: normalize + weight in fp32, pack back to vec2. Weight indexed [0,D) directly.
  const float inv = inv_rms;
  const Vec* __restrict__ srcv =
      HAS_RES ? reinterpret_cast<const Vec*>(xr_out + base) : reinterpret_cast<const Vec*>(x + base);
  const Vec* __restrict__ wv = reinterpret_cast<const Vec*>(weight);
  Vec* __restrict__ ov = reinterpret_cast<Vec*>(out + base);
  for (int j = tid; j < D2; j += RMSNORM_THREADS) {
    float2 f = rms_vec2_to_f2(srcv[j]);
    float2 w = rms_vec2_to_f2(wv[j]);
    if (UNIT_OFFSET) { w.x += 1.0f; w.y += 1.0f; }
    f.x = f.x * inv * w.x;
    f.y = f.y * inv * w.y;
    ov[j] = rms_f2_to_vec<Vec>(f);
  }
}

}  // namespace fni8
