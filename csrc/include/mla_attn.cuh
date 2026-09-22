// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Track-2 (issue #7): int8 dp4a MLA (Multi-head Latent Attention, DeepSeek-
// V2/V3) absorb-path decode kernel.
//
// v1 (this file): the fp32 ground-truth port of the absorb-path decode math
// in `tests/reference_mla.py::mla_absorb` — one warp per (batch, head), a
// sequential online-softmax scan over the N cached tokens. Not the fast path
// (no dp4a, no tiling, no split-KV): it exists to be the on-device oracle v2
// (fp16) and v3 (int8 dp4a the latent QK/PV) get validated against, same
// staging discipline `deltanet_chunk.cuh` used for issue #6.
//
// MLA's decode-time reformulation folds W_UK into the query
// (`q'_h = W_UK_h^T q_nope_h`, done per-token in fni8/ops.py — a tiny GEMM)
// and W_UV.W_O offline (once, also fni8/ops.py), so this kernel's job is
// purely the attention math: QK/softmax/PV run MQA-style against the SHARED
// d_c-dim latent cache — no per-head K/V is ever materialized in the kernel.
// The two score components (latent + decoupled RoPE) are summed before the
// softmax, matching `scores = (scores_c + scores_r) * scale` in the oracle.
// Decode-step convention (see the oracle's docstring): the query occupies the
// cache's last (highest-position) row, so it always attends every cached key
// — no causal mask is needed here.
//
// One warp (MLA_WARP lanes) per (b, h) block. Lane L owns latent-cache
// channels {L, L+32, ...} for both the QK dot-product partial sum and the PV
// weighted-sum accumulator — the same channel-striping convention
// `attn_decode.cuh` uses, just fp32 dot products instead of dp4a. Every
// per-(b,h) accumulator (q_abs, q_rope, running output) lives in dynamic
// shared memory rather than fixed-size registers, since d_c/d_r are runtime
// values here (v1 favors correctness/generality over the templated
// compile-time specialization the fast paths use).
#pragma once

#include <cfloat>

namespace fni8 {

constexpr int MLA_WARP = 32;

// grid = (B*H,), block = (MLA_WARP,). Dynamic shared mem: (2*d_c + d_r) floats.
__global__ void mla_decode_kernel(
    const float* __restrict__ q_abs,         // [B,H,d_c]  q_nope . W_UK^T (per head)
    const float* __restrict__ q_rope,        // [B,H,d_r]  RoPE'd, per head
    const float* __restrict__ c_kv_cache,    // [B,N,d_c]  shared latent cache (MQA)
    const float* __restrict__ k_rope_cache,  // [B,N,d_r]  shared decoupled-RoPE key cache
    float* __restrict__ out_abs,             // [B,H,d_c]  weighted-latent output (pre W_OV)
    int h, int n_len, int d_c, int d_r, float scale) {
  extern __shared__ float smem[];
  float* q_abs_s = smem;           // [d_c]
  float* q_rope_s = smem + d_c;    // [d_r]
  float* o_s = smem + d_c + d_r;   // [d_c]  running (unnormalized) PV accumulator

  const int lane = threadIdx.x;
  const int64_t bh = blockIdx.x;   // 0 .. B*H-1
  const int64_t b = bh / h;

  const float* q_abs_bh = q_abs + bh * (int64_t)d_c;
  const float* q_rope_bh = q_rope + bh * (int64_t)d_r;
  const float* ckv_b = c_kv_cache + b * (int64_t)n_len * d_c;
  const float* krope_b = k_rope_cache + b * (int64_t)n_len * d_r;

  // Each lane only ever reads shared-mem indices it wrote itself (same
  // lane-strided pattern below) -> no cross-lane dependency, no sync needed.
  for (int c = lane; c < d_c; c += MLA_WARP) {
    q_abs_s[c] = q_abs_bh[c];
    o_s[c] = 0.f;
  }
  for (int c = lane; c < d_r; c += MLA_WARP) q_rope_s[c] = q_rope_bh[c];

  float m_i = -FLT_MAX, l_i = 0.f;

  for (int n = 0; n < n_len; ++n) {
    const float* ckv_row = ckv_b + (int64_t)n * d_c;
    const float* krope_row = krope_b + (int64_t)n * d_r;

    float partial = 0.f;
    for (int c = lane; c < d_c; c += MLA_WARP) partial += q_abs_s[c] * ckv_row[c];
    for (int c = lane; c < d_r; c += MLA_WARP) partial += q_rope_s[c] * krope_row[c];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) partial += __shfl_xor_sync(0xffffffffu, partial, o);
    const float score = partial * scale;

    // rescale = expf(m_i - m_new); on the first iter m_i == -FLT_MAX so this
    // underflows to exactly 0.f (no branch needed -- o_s/l_i are already 0).
    const float m_new = fmaxf(m_i, score);
    const float rescale = expf(m_i - m_new);
    const float p = expf(score - m_new);
    for (int c = lane; c < d_c; c += MLA_WARP)
      o_s[c] = o_s[c] * rescale + p * ckv_row[c];
    l_i = l_i * rescale + p;
    m_i = m_new;
  }

  const float inv_l = (l_i > 0.f) ? (1.f / l_i) : 0.f;
  float* out_bh = out_abs + bh * (int64_t)d_c;
  for (int c = lane; c < d_c; c += MLA_WARP) out_bh[c] = o_s[c] * inv_l;
}

}  // namespace fni8
