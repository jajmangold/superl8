// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Fused gated activation on the merged gate_up projection output — one launch
// replaces the eager chunk/silu/mul (SwiGLU) or chunk/gelu-tanh/mul (GeGLU).
// Input x[..., 2I], output[..., I]: out[i] = act(x[i]) * x[i+I].
//
// fp16 path: half2 vectorized loads/stores + __hmul2 multiply on the healthy
// CUDA-core pipe (~27 TFLOP/s); activation (sigmoid/tanh) stays fp32 for accuracy.
// bf16 path: scalar fp32 math (same as before).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "compute_dtype.cuh"

namespace fni8 {

enum class ActKind { Silu = 0, GeluTanh = 1 };

__device__ __forceinline__ float act_load(const __half* p, int64_t i) {
  return to_float<__half>(p[i]);
}
__device__ __forceinline__ float act_load(const __nv_bfloat16* p, int64_t i) {
  return to_float<__nv_bfloat16>(p[i]);
}

__device__ __forceinline__ float apply_act(float g, ActKind kind) {
  if (kind == ActKind::Silu) return g / (1.0f + __expf(-g));
  const float k = 0.7978845608028654f;   // sqrt(2/pi)
  return 0.5f * g * (1.0f + tanhf(k * (g + 0.044715f * g * g * g)));
}

// ---- scalar path (bf16) ----
// out[M,I] = act(x[:, :I]) * x[:, I:2I]. grid-stride over M*I elements.
template <typename T, ActKind KIND>
__global__ void act_and_mul_kernel(const T* __restrict__ x, T* __restrict__ out,
                                   int64_t M, int64_t I) {
  const int64_t total = M * I;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    const int64_t m = idx / I, i = idx - m * I;
    const int64_t base = m * 2 * I;
    const float g = act_load(x, base + i);
    const float u = act_load(x, base + I + i);
    out[idx] = float_to<T>(apply_act(g, KIND) * u);
  }
}

// ---- half2 path (fp16, I even only) ----
// Each thread processes a pair of consecutive elements via half2 vectorized
// loads/stores and __hmul2. Activation (sigmoid/tanh) computed in fp32 for
// accuracy; the final gate*sigmoid(gate)*up multiply uses half2 ops.
// I is guaranteed even by the launcher (needed for aligned half2 stores).
template <ActKind KIND>
__global__ void act_and_mul_kernel_half2(const __half* __restrict__ x,
                                          __half* __restrict__ out,
                                          int64_t M, int64_t I) {
  const int64_t pairs_per_row = I >> 1;
  const int64_t total_pairs = M * pairs_per_row;

  for (int64_t pid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       pid < total_pairs;
       pid += (int64_t)gridDim.x * blockDim.x) {
    const int64_t m = pid / pairs_per_row;
    const int64_t p = pid - m * pairs_per_row;
    const int64_t i = p << 1;
    const int64_t base = m * 2 * I;

    __half2 gate = *reinterpret_cast<const __half2*>(x + base + i);
    __half2 up   = *reinterpret_cast<const __half2*>(x + base + I + i);

    float g0 = __half2float(gate.x);
    float g1 = __half2float(gate.y);

    // Compute sigmoid(g) / gelu-tanh-weight in fp32; the gate multiply and up
    // multiply run as half2 __hmul2 ops on the healthy CUDA-core pipe.
    if constexpr (KIND == ActKind::Silu) {
      float s0 = 1.0f / (1.0f + __expf(-g0));
      float s1 = 1.0f / (1.0f + __expf(-g1));
      __half2 a2 = __halves2half2(__float2half(s0), __float2half(s1));
      __half2 res = __hmul2(__hmul2(gate, a2), up);
      *reinterpret_cast<__half2*>(out + m * I + i) = res;
    } else {
      const float k = 0.7978845608028654f;
      float w0 = 0.5f * (1.0f + tanhf(k * (g0 + 0.044715f * g0 * g0 * g0)));
      float w1 = 0.5f * (1.0f + tanhf(k * (g1 + 0.044715f * g1 * g1 * g1)));
      __half2 w2 = __halves2half2(__float2half(w0), __float2half(w1));
      __half2 res = __hmul2(__hmul2(gate, w2), up);
      *reinterpret_cast<__half2*>(out + m * I + i) = res;
    }
  }
}

}  // namespace fni8
