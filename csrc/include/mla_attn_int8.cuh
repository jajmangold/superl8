// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// v3 (issue #57): int8 dp4a MLA absorb-path decode kernel.
//
// Same warp-per-(batch,head) online-softmax sequential scan as v1/v2, but the
// QK dot product runs via int8 dp4a: Q_abs and the per-token C_KV row are
// quantized symmetric-per-row (fp32 scales, SDNQ recipe), then the dot product
// is accumulated as int32 and dequantized with q_scale * kv_scale.
//
// The decoupled RoPE score, softmax (expf), and the PV weighted-sum accumulator
// stay fp32 (numerically load-bearing per AGENTS.md §Softmax). C_KV is loaded
// from fp16 global memory, quantized to int8 in shared memory, and used for
// both QK (dp4a) and PV (fp32 weighted-add — P[n] is a scalar so dp4a PV does
// not commute with online-softmax rescaling).
//
// Shared memory per warp:
//   q_abs_s   [d_c]   fp32   Q-abs query, cached once
//   q_rope_s  [d_r]   fp32   Q-rope query, cached once
//   o_s       [d_c]   fp32   running fp32 PV accumulator
//   q_int8    [d_c]   int8   int32-packed int8 Q  (d_c/4 int32 words)
//   ckv_f16   [d_c]   half   scratch: per-token C_KV row loaded from global
//   ckv_int8  [d_c]   int8   int32-packed int8 C_KV per token (d_c/4 int32 words)
// ============================================================================
#pragma once

#include <cfloat>
#include <cuda_fp16.h>
#include <cstdint>

#include "quant_rowwise.cuh"

namespace fni8 {

constexpr int MLA_INT8_WARP = 32;

__global__ void mla_decode_int8_kernel(
    const half* __restrict__ q_abs,         // [B,H,d_c] fp16
    const half* __restrict__ q_rope,        // [B,H,d_r] fp16
    const half* __restrict__ c_kv_cache,    // [B,N,d_c] fp16
    const half* __restrict__ k_rope_cache,  // [B,N,d_r] fp16
    half* __restrict__ out_abs,             // [B,H,d_c] fp16
    int h, int n_len, int d_c, int d_r, float scale) {

  const int d_c4 = d_c / 4;  // number of int32-packed words

  extern __shared__ float smem[];
  float*    q_abs_s  = smem;                                 // [d_c] float, offset 0
  float*    q_rope_s = smem + d_c;                            // [d_r] float, offset 4*Dc
  float*    o_s      = smem + d_c + d_r;                      // [d_c] float, offset 4*(Dc+Dr)
  int32_t*  q_int8_p = reinterpret_cast<int32_t*>(smem + 2 * d_c + d_r);  // [d_c4], after o_s
  half*     ckv_f16  = reinterpret_cast<half*>(q_int8_p + d_c4);          // [d_c] half
  int32_t*  ckv_int8_p = reinterpret_cast<int32_t*>(ckv_f16 + d_c);       // [d_c4]

  const int lane = threadIdx.x;
  const int64_t bh = blockIdx.x;
  const int64_t b  = bh / h;

  const half* q_abs_bh  = q_abs  + bh * (int64_t)d_c;
  const half* q_rope_bh = q_rope + bh * (int64_t)d_r;
  const half* ckv_b     = c_kv_cache    + b * (int64_t)n_len * d_c;
  const half* krope_b   = k_rope_cache  + b * (int64_t)n_len * d_r;

  // === Step 1: load Q into fp32 shared, compute Q int8 scale ===
  float q_amax = 0.f;
  for (int c = lane; c < d_c; c += MLA_INT8_WARP) {
    const float v = __half2float(q_abs_bh[c]);
    q_abs_s[c] = v;
    o_s[c]     = 0.f;
    q_amax = fmaxf(q_amax, fabsf(v));
  }
  for (int c = lane; c < d_r; c += MLA_INT8_WARP)
    q_rope_s[c] = __half2float(q_rope_bh[c]);

  for (int off = 16; off > 0; off >>= 1)
    q_amax = fmaxf(q_amax, __shfl_xor_sync(0xffffffffu, q_amax, off));
  float q_scale = q_amax * FNI8_Q_MAX_INV;
  if (q_scale == 0.f) q_scale = 1.f;

  // === Step 2: quantize Q to int8, write int32-packed (strided layout) ===
  // Each thread: d_c4/32 = d_c/128 int32 words. For d_c=512: 4 words/thread.
  {
    for (int idx = lane; idx < d_c4; idx += MLA_INT8_WARP) {
      const int base = idx * 4;
      int32_t packed = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = base + j;
        float v = rintf(__fdiv_rn(q_abs_s[c], q_scale));
        v = fminf(fmaxf(v, -FNI8_Q_MAX), FNI8_Q_MAX);
        packed |= ((int32_t)(int8_t)v & 0xFF) << (j * 8);
      }
      q_int8_p[idx] = packed;
    }
  }

  // === Main loop over N cached tokens ===
  float m_i = -FLT_MAX, l_i = 0.f;

  for (int n = 0; n < n_len; ++n) {
    const half* ckv_row  = ckv_b   + (int64_t)n * d_c;
    const half* krope_row = krope_b + (int64_t)n * d_r;

    // -- 3a: load C_KV row into shared fp16 (lane-strided) --
    for (int c = lane; c < d_c; c += MLA_INT8_WARP)
      ckv_f16[c] = ckv_row[c];

    // -- 3b: quantize C_KV row: compute abs-max, warp-reduce, get kv_scale --
    float kv_amax = 0.f;
    for (int c = lane; c < d_c; c += MLA_INT8_WARP)
      kv_amax = fmaxf(kv_amax, fabsf(__half2float(ckv_f16[c])));
    for (int off = 16; off > 0; off >>= 1)
      kv_amax = fmaxf(kv_amax, __shfl_xor_sync(0xffffffffu, kv_amax, off));
    float kv_scale = kv_amax * FNI8_Q_MAX_INV;
    if (kv_scale == 0.f) kv_scale = 1.f;

    // -- 3c: quantize C_KV to int8, write int32-packed (strided layout) --
    {
      for (int idx = lane; idx < d_c4; idx += MLA_INT8_WARP) {
        const int base = idx * 4;
        int32_t packed = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int c = base + j;
          float v = rintf(__fdiv_rn(__half2float(ckv_f16[c]), kv_scale));
          v = fminf(fmaxf(v, -FNI8_Q_MAX), FNI8_Q_MAX);
          packed |= ((int32_t)(int8_t)v & 0xFF) << (j * 8);
        }
        ckv_int8_p[idx] = packed;
      }
    }

    // -- 3d: dp4a QK dot product --
    int32_t dot = 0;
    for (int i = lane; i < d_c4; i += MLA_INT8_WARP)
      dot = __dp4a(q_int8_p[i], ckv_int8_p[i], dot);
    for (int off = 16; off > 0; off >>= 1)
      dot += __shfl_xor_sync(0xffffffffu, dot, off);
    float score_c = (float)dot * q_scale * kv_scale;  // dequantize

    // -- 3e: add RoPE score --
    float score_r = 0.f;
    for (int c = lane; c < d_r; c += MLA_INT8_WARP)
      score_r += q_rope_s[c] * __half2float(krope_row[c]);
    for (int off = 16; off > 0; off >>= 1)
      score_r += __shfl_xor_sync(0xffffffffu, score_r, off);

    const float score = (score_c + score_r) * scale;

    // -- 3f: online softmax (fp32) --
    const float m_new = fmaxf(m_i, score);
    const float rescale = expf(m_i - m_new);
    const float p = expf(score - m_new);

    // -- 3g: PV weighted-add (fp32, using quantized C_KV dequantized) --
    for (int c = lane; c < d_c; c += MLA_INT8_WARP) {
      const int w_idx = c / 4;
      const int byte_off = c & 3;
      const int8_t ckv_byte =
          (int8_t)((ckv_int8_p[w_idx] >> (byte_off * 8)) & 0xFF);
      const float ckv_val = (float)(int)ckv_byte * kv_scale;
      o_s[c] = o_s[c] * rescale + p * ckv_val;
    }
    l_i = l_i * rescale + p;
    m_i = m_new;
  }

  // === Epilogue: normalize and store fp16 output ===
  const float inv_l = (l_i > 0.f) ? (1.f / l_i) : 0.f;
  half* out_bh = out_abs + bh * (int64_t)d_c;
  for (int c = lane; c < d_c; c += MLA_INT8_WARP)
    out_bh[c] = __float2half(o_s[c] * inv_l);
}

}  // namespace fni8
