// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Decode (M=1) FlashAttention-2 for sm_70 — flash-decoding / split-KV.
//
// Autoregressive decode runs ONE query row against a long K/V cache. The prefill
// kernel would launch a BLOCK_M=64 tile (masking 63/64 rows) and parallelise only
// over B*H_q — far too few blocks to fill 80 SMs, and each serialises the whole
// cache. This kernel splits the KEY dim across blocks so the GPU stays busy:
//
//   split kernel   — grid (n_splits, B*H_q), ONE warp/block. Each block reduces
//                    its key chunk to a partial (O[D] fp32, running max m, sum l)
//                    via the same INT8 dp4a QK + fp32 exp2 online softmax as the
//                    prefill kernel. Thread map: lane L holds Q int32-chunk L (the
//                    dp4a operand, warp-reduced to the full logit) and OWNS output
//                    channels {L, L+32, ...} (D/32 per lane) — registers stay O(D/32).
//   combine kernel — grid (B*H_q,), ONE warp. Merges the n_splits partials for a
//                    row with the log-sum-exp trick: global m = max_s m_s, then
//                    O = Σ_s O_s·exp2(m_s−m) / Σ_s l_s·exp2(m_s−m).
//
// Semantics: the single query attends to ALL N keys (decode = full attention over
// the cache). V stays fp16 (int8 KV cache is a separate change); Q/K are int8 dp4a.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace fni8 {

constexpr int DEC_WARP = 32;
constexpr int DEC_I4V_GROUP = 32;             // int4-V: per-channel V scale per key-block
constexpr float DEC_NEG_INF = -1e30f;

// NF4 codebook (QLoRA): 16 non-uniform levels at the quantiles of a normal, in
// [-1,1]. V is ~Gaussian, so NF4 spends its 16 codes far better than a uniform
// grid -> lower int4-V error (measured: g32 uniform 0.096 -> NF4 0.088 rel-L1).
__device__ const float DEC_NF4[16] = {
    -1.0f, -0.6961928f, -0.52507305f, -0.3949175f, -0.28444138f, -0.18477343f,
    -0.09105004f, 0.0f, 0.07958030f, 0.16093020f, 0.24611230f, 0.33791524f,
    0.44070983f, 0.562617f, 0.72295684f, 1.0f};

// ---------------------------------------------------------------------------
// Split kernel: one warp reduces this block's key chunk to a partial (O,m,l).
// scratch layouts (row-major): o_part[n_splits][BHq][D], m_part/l_part[n_splits][BHq].
// ---------------------------------------------------------------------------
template <int D>
__global__ void decode_split_kernel(
    const int8_t* __restrict__ q,       // [B,H_q,1,D] int8
    const float* __restrict__ q_scale,  // [B,H_q,1] folded (softmax_scale*log2e)
    const int8_t* __restrict__ k,       // [B,H_kv,N,D] int8 (smoothed)
    const float* __restrict__ k_scale,  // [B,H_kv,N]
    const __half* __restrict__ v,       // [B,H_kv,N,D] fp16
    float* __restrict__ o_part,         // [n_splits, B*H_q, D] fp32
    float* __restrict__ m_part,         // [n_splits, B*H_q] fp32
    float* __restrict__ l_part,         // [n_splits, B*H_q] fp32
    int n_len, int n_splits, int h_q, int gqa_group) {
  constexpr int D4 = D / 4;         // int32-packed head dim (32 for D<=128, 64 for D=256)
  constexpr int QCH = (D4 + DEC_WARP - 1) / DEC_WARP;  // dp4a int32-chunks/lane (1, 2 @ D=256)
  constexpr int CH = D / DEC_WARP;  // output channels per lane (1,2,4,8 for D=32,64,128,256)
  constexpr unsigned FULL = 0xffffffffu;

  const int lane = threadIdx.x;            // 0..31 (one warp)
  const int split = blockIdx.x;            // 0..n_splits-1
  const int64_t bh = blockIdx.y;           // over B*H_q
  const int64_t kv_bh = (bh / h_q) * (h_q / gqa_group) + (bh % h_q) / gqa_group;

  // even key partition: split s owns [s*per, min(N,(s+1)*per)).
  const int per = (n_len + n_splits - 1) / n_splits;
  const int n0 = split * per;
  const int n1 = min(n_len, n0 + per);

  const int8_t* q_bh = q + bh * (int64_t)D;
  const int8_t* k_bh = k + kv_bh * (int64_t)n_len * D;
  const __half* v_bh = v + kv_bh * (int64_t)n_len * D;
  const float* ks_bh = k_scale + kv_bh * (int64_t)n_len;
  const float my_qs = q_scale[bh];

  // lane L holds Q int32-chunk(s) {L, L+32, ...} (QCH per lane; D<=128 -> QCH=1,
  // unchanged); q packed as int32 over D.
  int32_t q_pk[QCH];
#pragma unroll
  for (int qc = 0; qc < QCH; ++qc) {
    const int idx = lane + qc * DEC_WARP;
    q_pk[qc] = (idx < D4) ? reinterpret_cast<const int32_t*>(q_bh)[idx] : 0;
  }

  float m_i = DEC_NEG_INF, l_i = 0.f;
  float o_acc[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) o_acc[c] = 0.f;

  for (int n = n0; n < n1; ++n) {
    // logit s = Q.K_n via dp4a (QCH chunks/lane), warp-reduced across all D-chunks.
    const int8_t* k_row = k_bh + (int64_t)n * D;
    int32_t acc = 0;
#pragma unroll
    for (int qc = 0; qc < QCH; ++qc) {
      const int idx = lane + qc * DEC_WARP;
      const int32_t k_pk = (idx < D4) ? reinterpret_cast<const int32_t*>(k_row)[idx] : 0;
      acc = __dp4a(q_pk[qc], k_pk, acc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(FULL, acc, o);
    const float s = (float)acc * my_qs * ks_bh[n];  // exp2 domain (log2e folded)

    // The rescale exp2f is only needed on a NEW running max (~log N times), but
    // the online loop would pay it every key. Decode is XU/exp2f-bound (ncu: XU
    // 44% is the tallest pole, LSU 9%), so guarding it ~halves the exp2f count.
    // The compare is warp-uniform (s, m_i identical across lanes) -> no divergence.
    const float m_new = fmaxf(m_i, s);
    float rescale = 1.f;
    if (m_i != DEC_NEG_INF && m_new > m_i) rescale = exp2f(m_i - m_new);
    const float p = exp2f(s - m_new);
    // lane L accumulates its channels {L, L+32, ...}; V read is coalesced.
    const __half* v_row = v_bh + (int64_t)n * D;
#pragma unroll
    for (int c = 0; c < CH; ++c)
      o_acc[c] = o_acc[c] * rescale + p * __half2float(v_row[lane + c * DEC_WARP]);
    l_i = l_i * rescale + p;
    m_i = m_new;
  }

  // write this split's partial (empty split -> m=-inf, l=0, O=0).
  float* op = o_part + ((int64_t)split * gridDim.y + bh) * D;
#pragma unroll
  for (int c = 0; c < CH; ++c) op[lane + c * DEC_WARP] = o_acc[c];
  if (lane == 0) {
    m_part[(int64_t)split * gridDim.y + bh] = (n1 > n0) ? m_i : DEC_NEG_INF;
    l_part[(int64_t)split * gridDim.y + bh] = l_i;
  }
}

// ---------------------------------------------------------------------------
// Split kernel, INT8 KV-cache variant: V is stored int8 with a per-channel
// scale (persistent int8 cache -> half the HBM footprint and half the V read
// bandwidth, the dominant cost of memory-bound decode). O accumulates
// p * (float)v_i8 and folds the per-channel v_scale into the partial write
// (v_scale[d] factors out of the key-sum, so it commutes with the LSE combine).
// ---------------------------------------------------------------------------
template <int D>
__global__ void decode_split_i8v_kernel(
    const int8_t* __restrict__ q,       // [B,H_q,1,D] int8
    const float* __restrict__ q_scale,  // [B,H_q,1] folded
    const int8_t* __restrict__ k,       // [B,H_kv,N,D] int8 (smoothed)
    const float* __restrict__ k_scale,  // [B,H_kv,N]
    const int8_t* __restrict__ v,       // [B,H_kv,N,D] int8
    const float* __restrict__ v_scale,  // [B,H_kv,D] per-channel
    float* __restrict__ o_part,         // [n_splits, B*H_q, D] fp32
    float* __restrict__ m_part,         // [n_splits, B*H_q] fp32
    float* __restrict__ l_part,         // [n_splits, B*H_q] fp32
    int n_len, int n_splits, int h_q, int gqa_group) {
  constexpr int D4 = D / 4;
  constexpr int QCH = (D4 + DEC_WARP - 1) / DEC_WARP;  // dp4a int32-chunks/lane (1, 2 @ D=256)
  constexpr int CH = D / DEC_WARP;
  constexpr unsigned FULL = 0xffffffffu;

  const int lane = threadIdx.x;
  const int split = blockIdx.x;
  const int64_t bh = blockIdx.y;
  const int64_t kv_bh = (bh / h_q) * (h_q / gqa_group) + (bh % h_q) / gqa_group;

  const int per = (n_len + n_splits - 1) / n_splits;
  const int n0 = split * per;
  const int n1 = min(n_len, n0 + per);

  const int8_t* q_bh = q + bh * (int64_t)D;
  const int8_t* k_bh = k + kv_bh * (int64_t)n_len * D;
  const int8_t* v_bh = v + kv_bh * (int64_t)n_len * D;
  const float* ks_bh = k_scale + kv_bh * (int64_t)n_len;
  const float* vsc_bh = v_scale + kv_bh * (int64_t)D;
  const float my_qs = q_scale[bh];

  int32_t q_pk[QCH];
#pragma unroll
  for (int qc = 0; qc < QCH; ++qc) {
    const int idx = lane + qc * DEC_WARP;
    q_pk[qc] = (idx < D4) ? reinterpret_cast<const int32_t*>(q_bh)[idx] : 0;
  }

  float m_i = DEC_NEG_INF, l_i = 0.f;
  float o_acc[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) o_acc[c] = 0.f;

  for (int n = n0; n < n1; ++n) {
    const int8_t* k_row = k_bh + (int64_t)n * D;
    int32_t acc = 0;
#pragma unroll
    for (int qc = 0; qc < QCH; ++qc) {
      const int idx = lane + qc * DEC_WARP;
      const int32_t k_pk = (idx < D4) ? reinterpret_cast<const int32_t*>(k_row)[idx] : 0;
      acc = __dp4a(q_pk[qc], k_pk, acc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(FULL, acc, o);
    const float s = (float)acc * my_qs * ks_bh[n];

    // Guard the rescale exp2f (only fires on a new max) — decode is XU/exp2f-bound
    // (ncu). Warp-uniform compare, no divergence.
    const float m_new = fmaxf(m_i, s);
    float rescale = 1.f;
    if (m_i != DEC_NEG_INF && m_new > m_i) rescale = exp2f(m_i - m_new);
    const float p = exp2f(s - m_new);
    const int8_t* v_row = v_bh + (int64_t)n * D;
#pragma unroll
    for (int c = 0; c < CH; ++c)
      o_acc[c] = o_acc[c] * rescale + p * (float)v_row[lane + c * DEC_WARP];
    l_i = l_i * rescale + p;
    m_i = m_new;
  }

  // fold per-channel v_scale into the partial (commutes with the LSE combine).
  float* op = o_part + ((int64_t)split * gridDim.y + bh) * D;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const int d = lane + c * DEC_WARP;
    op[d] = o_acc[c] * vsc_bh[d];
  }
  if (lane == 0) {
    m_part[(int64_t)split * gridDim.y + bh] = (n1 > n0) ? m_i : DEC_NEG_INF;
    l_part[(int64_t)split * gridDim.y + bh] = l_i;
  }
}

// ---------------------------------------------------------------------------
// VERIFY variant of the INT8-KV split kernel: M = k draft queries per (B,H_q)
// instead of the single decode row, with an END-ALIGNED CAUSAL mask. This is the
// speculative-decode / MTP verify shape — the k drafts of a chain are verified in
// ONE pass against the persistent int8 cache (length N = prefix + k). Draft i
// attends keys [0, causal_diag + i] (causal_diag = prefix = N - k), i.e. the exact
// tril(N-k) mask (mirrors attn_w8a8_fwd's end-aligned causal).
//
// Why re-use the decode machinery: the dense-tile W8A8 verify kernel launches only
// B*H_q blocks (tiny-M, large-N) -> ~6% occupancy, latency-bound, masking the int8
// bandwidth win. Mapping the drafts onto flash-decoding gives an (n_splits, B*H_q*M)
// grid -> n_splits*M more blocks to fill the SMs; the LSE combine is split-invariant
// so the result is numerically the same as the non-split kernel (see the combine
// kernel below, reused unchanged with BHq = B*H_q*M). One warp/block, same int8 dp4a
// QK + fp32 exp2 online softmax + int8 dp4a-domain PV as decode_split_i8v_kernel.
// ---------------------------------------------------------------------------
template <int D>
__global__ void decode_split_i8v_verify_kernel(
    const int8_t* __restrict__ q,       // [B,H_q,M,D] int8 (M = k drafts)
    const float* __restrict__ q_scale,  // [B,H_q,M] folded (softmax_scale*log2e)
    const int8_t* __restrict__ k,       // [B,H_kv,N,D] int8 (smoothed cache)
    const float* __restrict__ k_scale,  // [B,H_kv,N]
    const int8_t* __restrict__ v,       // [B,H_kv,N,D] int8
    const float* __restrict__ v_scale,  // [B,H_kv,D] per-channel
    float* __restrict__ o_part,         // [n_splits, B*H_q*M, D] fp32
    float* __restrict__ m_part,         // [n_splits, B*H_q*M] fp32
    float* __restrict__ l_part,         // [n_splits, B*H_q*M] fp32
    int n_len, int n_splits, int m_len, int h_q, int gqa_group, int causal_diag) {
  constexpr int D4 = D / 4;
  constexpr int QCH = (D4 + DEC_WARP - 1) / DEC_WARP;  // dp4a int32-chunks/lane (1, 2 @ D=256)
  constexpr int CH = D / DEC_WARP;
  constexpr unsigned FULL = 0xffffffffu;

  const int lane = threadIdx.x;
  const int split = blockIdx.x;
  const int64_t bhm = blockIdx.y;          // over B*H_q*M
  const int64_t bh = bhm / m_len;          // over B*H_q (owns the K/V head)
  const int draft = (int)(bhm % m_len);    // 0..M-1
  const int64_t kv_bh = (bh / h_q) * (h_q / gqa_group) + (bh % h_q) / gqa_group;

  // End-aligned causal: draft `draft` attends keys [0, causal_diag + draft]
  // (inclusive) -> exclusive bound n_causal. Always >= 1 (split 0 is never empty).
  const int n_causal = causal_diag + draft + 1;

  // even key partition over the FULL cache, then clamp to the causal bound. Splits
  // that start past n_causal are empty (m=-inf, l=0) and drop out of the combine.
  const int per = (n_len + n_splits - 1) / n_splits;
  const int n0 = split * per;
  int n1 = min(n_len, n0 + per);
  if (n1 > n_causal) n1 = n_causal;

  const int8_t* q_bh = q + bhm * (int64_t)D;   // one int8 query row per (B,H_q,draft)
  const int8_t* k_bh = k + kv_bh * (int64_t)n_len * D;
  const int8_t* v_bh = v + kv_bh * (int64_t)n_len * D;
  const float* ks_bh = k_scale + kv_bh * (int64_t)n_len;
  const float* vsc_bh = v_scale + kv_bh * (int64_t)D;
  const float my_qs = q_scale[bhm];

  int32_t q_pk[QCH];
#pragma unroll
  for (int qc = 0; qc < QCH; ++qc) {
    const int idx = lane + qc * DEC_WARP;
    q_pk[qc] = (idx < D4) ? reinterpret_cast<const int32_t*>(q_bh)[idx] : 0;
  }

  float m_i = DEC_NEG_INF, l_i = 0.f;
  float o_acc[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) o_acc[c] = 0.f;

  for (int n = n0; n < n1; ++n) {
    const int8_t* k_row = k_bh + (int64_t)n * D;
    int32_t acc = 0;
#pragma unroll
    for (int qc = 0; qc < QCH; ++qc) {
      const int idx = lane + qc * DEC_WARP;
      const int32_t k_pk = (idx < D4) ? reinterpret_cast<const int32_t*>(k_row)[idx] : 0;
      acc = __dp4a(q_pk[qc], k_pk, acc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(FULL, acc, o);
    const float s = (float)acc * my_qs * ks_bh[n];

    const float m_new = fmaxf(m_i, s);
    float rescale = 1.f;
    if (m_i != DEC_NEG_INF && m_new > m_i) rescale = exp2f(m_i - m_new);
    const float p = exp2f(s - m_new);
    const int8_t* v_row = v_bh + (int64_t)n * D;
#pragma unroll
    for (int c = 0; c < CH; ++c)
      o_acc[c] = o_acc[c] * rescale + p * (float)v_row[lane + c * DEC_WARP];
    l_i = l_i * rescale + p;
    m_i = m_new;
  }

  // fold per-channel v_scale into the partial (commutes with the LSE combine).
  float* op = o_part + ((int64_t)split * gridDim.y + bhm) * D;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const int d = lane + c * DEC_WARP;
    op[d] = o_acc[c] * vsc_bh[d];
  }
  if (lane == 0) {
    m_part[(int64_t)split * gridDim.y + bhm] = (n1 > n0) ? m_i : DEC_NEG_INF;
    l_part[(int64_t)split * gridDim.y + bhm] = l_i;
  }
}

// ---------------------------------------------------------------------------
// Split kernel, INT4-V (NF4) KV-cache variant: V stored 4-bit, 2 channels/byte,
// non-uniform NF4 levels, per-(channel, key-block) absmax scale. Halves the V cache
// vs int8 (more context in 16 GB). The V read unpacks a 4-bit index (byte d/2, nibble
// d%2) and looks it up in DEC_NF4; the per-key-block absmax scale is applied in-loop.
// v is [B,H_kv,N,D/2] packed: byte j holds channels 2j (low nibble), 2j+1 (high).
// ---------------------------------------------------------------------------
template <int D>
__global__ void decode_split_i4v_kernel(
    const int8_t* __restrict__ q, const float* __restrict__ q_scale,
    const int8_t* __restrict__ k, const float* __restrict__ k_scale,
    const int8_t* __restrict__ v,       // [B,H_kv,N,D/2] int4-packed
    const float* __restrict__ v_scale,  // [B,H_kv,nblocks,D] per (key-block, channel)
    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int n_len, int n_splits, int nblocks, int h_q, int gqa_group) {
  constexpr int D4 = D / 4;
  constexpr int QCH = (D4 + DEC_WARP - 1) / DEC_WARP;  // dp4a int32-chunks/lane (1, 2 @ D=256)
  constexpr int CH = D / DEC_WARP;
  constexpr int DP = D / 2;              // packed bytes per key
  constexpr unsigned FULL = 0xffffffffu;

  const int lane = threadIdx.x;
  const int split = blockIdx.x;
  const int64_t bh = blockIdx.y;
  const int64_t kv_bh = (bh / h_q) * (h_q / gqa_group) + (bh % h_q) / gqa_group;

  const int per = (n_len + n_splits - 1) / n_splits;
  const int n0 = split * per;
  const int n1 = min(n_len, n0 + per);

  const int8_t* q_bh = q + bh * (int64_t)D;
  const int8_t* k_bh = k + kv_bh * (int64_t)n_len * D;
  const int8_t* v_bh = v + kv_bh * (int64_t)n_len * DP;   // int4-packed
  const float* ks_bh = k_scale + kv_bh * (int64_t)n_len;
  const float* vsc_bh = v_scale + kv_bh * (int64_t)nblocks * D;  // per (block, channel)
  const float my_qs = q_scale[bh];

  int32_t q_pk[QCH];
#pragma unroll
  for (int qc = 0; qc < QCH; ++qc) {
    const int idx = lane + qc * DEC_WARP;
    q_pk[qc] = (idx < D4) ? reinterpret_cast<const int32_t*>(q_bh)[idx] : 0;
  }

  float m_i = DEC_NEG_INF, l_i = 0.f;
  float o_acc[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) o_acc[c] = 0.f;

  for (int n = n0; n < n1; ++n) {
    const int8_t* k_row = k_bh + (int64_t)n * D;
    int32_t acc = 0;
#pragma unroll
    for (int qc = 0; qc < QCH; ++qc) {
      const int idx = lane + qc * DEC_WARP;
      const int32_t k_pk = (idx < D4) ? reinterpret_cast<const int32_t*>(k_row)[idx] : 0;
      acc = __dp4a(q_pk[qc], k_pk, acc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(FULL, acc, o);
    const float s = (float)acc * my_qs * ks_bh[n];

    const float m_new = fmaxf(m_i, s);
    float rescale = 1.f;
    if (m_i != DEC_NEG_INF && m_new > m_i) rescale = exp2f(m_i - m_new);
    const float p = exp2f(s - m_new);
    // per-(key-block, channel) V scale applied inside the loop (block-grouped int4
    // is much tighter than a single per-channel scale) -> can't fold at the end.
    const float* vsc = vsc_bh + (int64_t)(n / DEC_I4V_GROUP) * D;
    const int8_t* v_row = v_bh + (int64_t)n * DP;
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      const int d = lane + c * DEC_WARP;
      const int byte = v_row[d >> 1];                    // channels (d&~1, d|1)
      const int nib = (d & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);  // NF4 index 0..15
      o_acc[c] = o_acc[c] * rescale + p * DEC_NF4[nib] * vsc[d];     // vsc = per-block absmax
    }
    l_i = l_i * rescale + p;
    m_i = m_new;
  }

  float* op = o_part + ((int64_t)split * gridDim.y + bh) * D;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const int d = lane + c * DEC_WARP;
    op[d] = o_acc[c];   // v_scale already applied per key-block
  }
  if (lane == 0) {
    m_part[(int64_t)split * gridDim.y + bh] = (n1 > n0) ? m_i : DEC_NEG_INF;
    l_part[(int64_t)split * gridDim.y + bh] = l_i;
  }
}

// ---------------------------------------------------------------------------
// Combine kernel: one warp merges the n_splits partials for a row via LSE.
// ---------------------------------------------------------------------------
template <int D>
__global__ void decode_combine_kernel(
    const float* __restrict__ o_part,   // [n_splits, B*H_q, D]
    const float* __restrict__ m_part,   // [n_splits, B*H_q]
    const float* __restrict__ l_part,   // [n_splits, B*H_q]
    __half* __restrict__ out,           // [B,H_q,1,D] fp16
    int n_splits) {
  constexpr int CH = D / DEC_WARP;
  const int lane = threadIdx.x;
  const int64_t bh = blockIdx.x;         // over B*H_q
  const int64_t BHq = gridDim.x;

  // global max over splits.
  float gm = DEC_NEG_INF;
  for (int s = 0; s < n_splits; ++s) gm = fmaxf(gm, m_part[(int64_t)s * BHq + bh]);

  float denom = 0.f;
  float acc[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) acc[c] = 0.f;

  for (int s = 0; s < n_splits; ++s) {
    const float ms = m_part[(int64_t)s * BHq + bh];
    if (ms == DEC_NEG_INF) continue;
    const float w = exp2f(ms - gm);
    denom += l_part[(int64_t)s * BHq + bh] * w;
    const float* op = o_part + ((int64_t)s * BHq + bh) * D;
#pragma unroll
    for (int c = 0; c < CH; ++c) acc[c] += op[lane + c * DEC_WARP] * w;
  }

  const float inv = (denom > 0.f) ? 1.f / denom : 0.f;
  __half* out_row = out + bh * (int64_t)D;
#pragma unroll
  for (int c = 0; c < CH; ++c)
    out_row[lane + c * DEC_WARP] = __float2half(acc[c] * inv);
}

}  // namespace fni8
