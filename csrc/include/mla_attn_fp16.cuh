// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// v2 (issue #41): fp16 on-device MLA absorb-path decode kernel.
//
// This is the fp16 variant of the v1 fp32 ground-truth kernel in mla_attn.cuh.
// Same algorithm (online-softmax sequential scan, one warp per (batch,head),
// channel-striping over MLA_WARP lanes), but the KV cache is read in fp16 for
// half the HBM bandwidth of the fp32 path --- the main decode bottleneck on
// Volta's crippled tensor-core fleet (AGENTS.md: dp4a == 46 TOP/s, fp16 TC ==
// 6.9 TFLOP/s, so the CUDA-core fp16-fmul dot products here are actually the
// right compute primitive too).
//
// Internal accumulation (q_abs_s, q_rope_s, o_s, online-softmax state m_i/l_i)
// stays fp32 --- softmax rescaling is numerically load-bearing per AGENTS.md.
// The kernel casts half -> float on global-memory load and float -> half on
// global-memory store; all shared-mem math is fp32.
#pragma once

#include <cfloat>
#include <cuda_fp16.h>

namespace fni8 {

constexpr int MLA_FP16_WARP = 32;

__global__ void mla_decode_fp16_kernel(
    const half* __restrict__ q_abs,         // [B,H,d_c] fp16
    const half* __restrict__ q_rope,        // [B,H,d_r] fp16
    const half* __restrict__ c_kv_cache,    // [B,N,d_c] fp16
    const half* __restrict__ k_rope_cache,  // [B,N,d_r] fp16
    half* __restrict__ out_abs,             // [B,H,d_c] fp16
    int h, int n_len, int d_c, int d_r, float scale) {
  extern __shared__ float smem[];
  float* q_abs_s = smem;           // [d_c]  fp32 cache of the query
  float* q_rope_s = smem + d_c;    // [d_r]  fp32 cache of the query
  float* o_s = smem + d_c + d_r;   // [d_c]  running (unnormalized) PV accumulator

  const int lane = threadIdx.x;
  const int64_t bh = blockIdx.x;
  const int64_t b = bh / h;

  const half* q_abs_bh = q_abs + bh * (int64_t)d_c;
  const half* q_rope_bh = q_rope + bh * (int64_t)d_r;
  const half* ckv_b = c_kv_cache + b * (int64_t)n_len * d_c;
  const half* krope_b = k_rope_cache + b * (int64_t)n_len * d_r;

  // Pre-load the query into fp32 shared memory once; all subsequent N-loop
  // iterations read from shared memory (no fp16->fp32 conversion per iter).
  for (int c = lane; c < d_c; c += MLA_FP16_WARP) {
    q_abs_s[c] = __half2float(q_abs_bh[c]);
    o_s[c] = 0.f;
  }
  for (int c = lane; c < d_r; c += MLA_FP16_WARP)
    q_rope_s[c] = __half2float(q_rope_bh[c]);

  float m_i = -FLT_MAX, l_i = 0.f;

  for (int n = 0; n < n_len; ++n) {
    const half* ckv_row = ckv_b + (int64_t)n * d_c;
    const half* krope_row = krope_b + (int64_t)n * d_r;

    float partial = 0.f;
    for (int c = lane; c < d_c; c += MLA_FP16_WARP)
      partial += q_abs_s[c] * __half2float(ckv_row[c]);
    for (int c = lane; c < d_r; c += MLA_FP16_WARP)
      partial += q_rope_s[c] * __half2float(krope_row[c]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      partial += __shfl_xor_sync(0xffffffffu, partial, o);
    const float score = partial * scale;

    const float m_new = fmaxf(m_i, score);
    const float rescale = expf(m_i - m_new);
    const float p = expf(score - m_new);
    for (int c = lane; c < d_c; c += MLA_FP16_WARP)
      o_s[c] = o_s[c] * rescale + p * __half2float(ckv_row[c]);
    l_i = l_i * rescale + p;
    m_i = m_new;
  }

  const float inv_l = (l_i > 0.f) ? (1.f / l_i) : 0.f;
  half* out_bh = out_abs + bh * (int64_t)d_c;
  for (int c = lane; c < d_c; c += MLA_FP16_WARP)
    out_bh[c] = __float2half(o_s[c] * inv_l);
}

}  // namespace fni8
