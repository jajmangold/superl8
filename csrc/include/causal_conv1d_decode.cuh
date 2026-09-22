// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Fused causal depthwise conv1d(kernel=K) + SiLU for the L==1 DECODE token
// shift of a Gated-DeltaNet / short-conv block (vLLM's `causal_conv1d_update`
// analogue). One launch replaces the eager cat/conv1d/slice/silu swarm.
//
// For one decode token the causal window of channel w is
//   window = [ tail[0,w], tail[1,w], ..., tail[K-2,w], x[w] ]   (length K)
// and the (depthwise, no-bias) conv + SiLU output is
//   out[w] = silu( sum_{j=0..K-1} window[j] * weight[w,j] ).
// The rolled-forward history is new_tail[j,w] = window[j+1] (drop the oldest
// sample, append x) — a pure copy, no arithmetic, so it is bit-exact.
//
// One thread per (batch, channel): K strided loads + a length-K dot in fp32 +
// one SiLU + K stores. ZERO dynamic shared memory -> no cudaFuncSetAttribute
// -> CUDA-graph-capturable (same design rule as deltanet_decode.cuh). Math is
// fp32 internally; the store dtype follows the input (fp16/bf16/fp32 — this is
// an elementwise pre/epilogue, not the load-bearing recurrence, per AGENTS.md).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "compute_dtype.cuh"

namespace fni8 {

// fp32 passthrough specializations so the kernel templates over fp16/bf16/fp32.
template <>
__device__ __forceinline__ float float_to<float>(float x) {
  return x;
}
template <>
__device__ __forceinline__ float to_float<float>(float x) {
  return x;
}

constexpr int CONV_MAX_K = 8;  // K-1 register slots for the window tail

// x:        [B, Wc]          — this token's pre-conv activation
// tail:     [B, K-1, Wc]     — the previous K-1 raw samples (decode history)
// weight:   [Wc, K]          — depthwise conv filter, one row per channel
// out:      [B, Wc]          — silu(conv) for this token
// new_tail: [B, K-1, Wc]     — window[1:] = rolled history for the next step
template <typename T>
__global__ void causal_conv1d_silu_decode_kernel(
    const T* __restrict__ x, const T* __restrict__ tail, const T* __restrict__ weight,
    T* __restrict__ out, T* __restrict__ new_tail, int B, int Wc, int K) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = (int64_t)B * Wc;
  if (idx >= total) return;
  const int b = (int)(idx / Wc);
  const int w = (int)(idx - (int64_t)b * Wc);

  const int Km1 = K - 1;
  const int64_t tail_base = (int64_t)b * Km1 * Wc + w;  // tail[b, j, w] at + j*Wc
  const int64_t wt_base = (int64_t)w * K;               // weight[w, j] at + j

  // Load the length-K window: [tail[0..K-2], x] into registers.
  float win[CONV_MAX_K];
#pragma unroll
  for (int j = 0; j < CONV_MAX_K; ++j) {
    if (j < Km1) win[j] = to_float<T>(tail[tail_base + (int64_t)j * Wc]);
  }
  const float xv = to_float<T>(x[idx]);
  win[Km1] = xv;  // guaranteed K-1 < CONV_MAX_K by the launcher check

  // Depthwise dot + SiLU (fp32 accumulate).
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < CONV_MAX_K; ++j) {
    if (j < K) acc += win[j] * to_float<T>(weight[wt_base + j]);
  }
  const float o = acc / (1.0f + __expf(-acc));  // SiLU
  out[idx] = float_to<T>(o);

  // Roll the history forward: new_tail[b, j, w] = window[j+1].
#pragma unroll
  for (int j = 0; j < CONV_MAX_K; ++j) {
    if (j < Km1) new_tail[tail_base + (int64_t)j * Wc] = float_to<T>(win[j + 1]);
  }
}

}  // namespace fni8
