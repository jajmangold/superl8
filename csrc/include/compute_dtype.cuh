// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Output-dtype dispatch for the dp4a kernels: int32 accumulate and the fp32
// dequant/softmax math never change (AGENTS.md — those stay fp32/fp16, never
// quantized); only the FINAL store dtype is templated. fp16 (max 65504)
// overflows on bf16-native models (Gemma, most diffusion DiTs) -> inf/NaN ->
// black output, so bf16 is a first-class store target alongside fp16.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace fni8 {

template <typename T>
__device__ __forceinline__ T float_to(float x);
template <>
__device__ __forceinline__ __half float_to<__half>(float x) {
  return __float2half(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 float_to<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

template <typename T>
__device__ __forceinline__ float to_float(T x);
template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
  return __half2float(x);
}
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T zero_val();
template <>
__device__ __forceinline__ __half zero_val<__half>() {
  return __half(0.f);
}
template <>
__device__ __forceinline__ __nv_bfloat16 zero_val<__nv_bfloat16>() {
  return __float2bfloat16(0.f);   // __nv_bfloat16 has no float ctor on sm_70
}

// Pair-store two fp32 values to `out[0..1]` as a single packed store when
// the dtype supports it (fp16: __half2 via __float22half2_rn, halving stores).
// bf16 has no native vector store on sm_70 → two scalar stores (correct but
// not a bandwidth win). COLS MUST be even (always true — D is always even).
template <typename VT>
__device__ __forceinline__ void pair_store(VT* out, float v0, float v1);

template <>
__device__ __forceinline__ void pair_store<__half>(__half* out, float v0, float v1) {
  *reinterpret_cast<__half2*>(out) = __float22half2_rn(float2{v0, v1});
}

template <>
__device__ __forceinline__ void pair_store<__nv_bfloat16>(__nv_bfloat16* out, float v0, float v1) {
  out[0] = __float2bfloat16(v0);
  out[1] = __float2bfloat16(v1);
}

}  // namespace fni8
