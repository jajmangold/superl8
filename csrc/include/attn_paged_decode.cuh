// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Paged-KV block-table decode + quantize-on-write KV store for sm_70.
//
// `attn_decode_cached` (attn_decode.cuh) needs a CONTIGUOUS [B,H_kv,N,D] cache
// slice, so a serving engine batching sequences of different lengths must loop
// the decode kernel once per distinct N. Real batched decode instead keeps a
// pool of fixed-size physical KV "blocks" and gives each sequence a table of
// which blocks (in any order, possibly non-contiguous/reused) hold its tokens
// — the vLLM/PagedAttention scheme. This header adds:
//
//   kv_write_paged_kernel   — quantize-on-write: commit ONE new token's K/V
//                             into the paged int8 cache at a caller-computed
//                             flat slot (block_id * block_size + offset).
//   paged_decode_split_kernel — the split-KV decode split stage addressed
//                             through a per-sequence block_table + a per-
//                             sequence context_len, so ONE launch (grid.y =
//                             B*H_q, same as the contiguous kernel) serves a
//                             whole batch of mixed-length sequences.
//
// The combine stage is layout-agnostic (it only touches the [n_splits,B*H_q]
// partial scratch) so `decode_combine_kernel<D>` from attn_decode.cuh is
// reused unmodified.
//
// Cache layout: k_cache/v_cache [num_blocks, H_kv, block_size, D] int8,
// k_scale/v_scale [num_blocks, H_kv, block_size] fp32 — PER-TOKEN (per-row)
// scale for BOTH K and V. `quantize_kv_cache`'s per-CHANNEL V scale needs the
// amax over the whole cache up front (quantize_v_perchannel reduces over the
// key dim); quantize-on-write only ever sees one new token, so that scheme is
// unavailable here — V uses the same per-row symmetric RTN as K. Likewise
// K-smoothing's per-channel MEAN (smooth_k) needs every key up front, so the
// paged write path leans on the Hadamard incoherence rotation instead
// (`rotate_last`, quant-prologue only, already used by quantize_kv_cache's
// `rotate=True`): it is a fixed per-token linear map, so it works one row at a
// time and stays logit-invariant when Q is rotated the same way at decode.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "attn_decode.cuh"  // DEC_WARP, DEC_NEG_INF, decode_combine_kernel<D>

namespace fni8 {

// ---------------------------------------------------------------------------
// Quantize-on-write: commits one new token's K/V for every (batch, H_kv) row
// in ONE launch. One warp per row; lane L owns channels {L, L+32, ...} (CH per
// lane), matching the decode kernels' output-channel thread map. Each lane
// reduces its own channels' abs-max, a warp shuffle-xor combines them into the
// row's amax, then every lane independently rescales+writes its channels — no
// shared memory, no second pass over global memory.
// ---------------------------------------------------------------------------
template <int D>
__global__ void kv_write_paged_kernel(
    const __half* __restrict__ k_new,          // [B, H_kv, D] fp16 (pre-rotated if rotate=True)
    const __half* __restrict__ v_new,          // [B, H_kv, D] fp16
    const int32_t* __restrict__ slot_mapping,  // [B] flat slot = block_id*block_size + offset
    int8_t* __restrict__ k_cache,              // [num_blocks, H_kv, block_size, D] int8
    float* __restrict__ k_scale,               // [num_blocks, H_kv, block_size] fp32
    int8_t* __restrict__ v_cache,              // [num_blocks, H_kv, block_size, D] int8
    float* __restrict__ v_scale,               // [num_blocks, H_kv, block_size] fp32
    int h_kv, int block_size) {
  constexpr int CH = D / DEC_WARP;
  constexpr float Q_MAX = 127.f;
  constexpr unsigned FULL = 0xffffffffu;

  const int lane = threadIdx.x;
  const int64_t bh = blockIdx.x;   // over B*H_kv
  const int64_t b = bh / h_kv;
  const int64_t h = bh % h_kv;

  const __half* k_row = k_new + bh * (int64_t)D;
  const __half* v_row = v_new + bh * (int64_t)D;

  float k_local[CH], v_local[CH];
  float k_amax = 0.f, v_amax = 0.f;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const int d = lane + c * DEC_WARP;
    k_local[c] = __half2float(k_row[d]);
    v_local[c] = __half2float(v_row[d]);
    k_amax = fmaxf(k_amax, fabsf(k_local[c]));
    v_amax = fmaxf(v_amax, fabsf(v_local[c]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    k_amax = fmaxf(k_amax, __shfl_xor_sync(FULL, k_amax, o));
    v_amax = fmaxf(v_amax, __shfl_xor_sync(FULL, v_amax, o));
  }
  const float ks = (k_amax > 0.f) ? k_amax / Q_MAX : 1.f;
  const float vs = (v_amax > 0.f) ? v_amax / Q_MAX : 1.f;

  const int64_t slot = slot_mapping[b];
  const int64_t block = slot / block_size;
  const int64_t offset = slot % block_size;
  const int64_t row = (block * (int64_t)h_kv + h) * block_size + offset;

  int8_t* k_dst = k_cache + row * D;
  int8_t* v_dst = v_cache + row * D;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const int d = lane + c * DEC_WARP;
    k_dst[d] = (int8_t)__float2int_rn(fminf(fmaxf(k_local[c] / ks, -Q_MAX), Q_MAX));
    v_dst[d] = (int8_t)__float2int_rn(fminf(fmaxf(v_local[c] / vs, -Q_MAX), Q_MAX));
  }
  if (lane == 0) {
    k_scale[row] = ks;
    v_scale[row] = vs;
  }
}

// ---------------------------------------------------------------------------
// Paged split kernel: same online-softmax reduction as decode_split_i8v_kernel
// (attn_decode.cuh), but each key position n is addressed through this
// sequence's block_table instead of a contiguous `k + n*D` offset, and n_len
// is PER-SEQUENCE (context_lens[b]) instead of one shared value for the whole
// launch — the two changes that let one launch serve a ragged batch.
//
// `LowbitV=true` (fni8#295) additionally fuses the LloydMax3 V decode into the
// kernel: V is no longer read as int8 rows with per-token scale but as packed
// 3-bit codes + fp32 block norms + a fixed Gaussian codebook (the K8V3 cache
// store, fni8-serve `_write_v_lloydmax3`). Per (n, head) the kernel unpacks
// `idx = ((words[3d>>5] >> (3d&31)) | (words[3d>>5+1] << (32-(3d&31)))) & 7`
// (LSB-first stream, elements straddle words only at shift 30/31, and the last
// word never straddles since 3D%32==0) and dequantizes
// `v = __float2half_rn(codebook[idx] * norm[d/128])` — the same fp32-dequant
// then fp16 rounding the dense `_decode_lloydmax3` fallback applies before its
// kernel, so logits are identical up to softmax reduction order. The K side is
// byte-identical to the int8 variant: the cache stores ROTATED K, and
// `<R(q), R(k)> = <q,k>` keeps the attention logits invariant.
// ---------------------------------------------------------------------------
template <int D, bool LowbitV = false>
__global__ void paged_decode_split_kernel(
    const int8_t* __restrict__ q,          // [B,H_q,1,D] int8
    const float* __restrict__ q_scale,     // [B,H_q,1] fp32 folded
    const int8_t* __restrict__ k_cache,    // [num_blocks,H_kv,block_size,D] int8
    const float* __restrict__ k_scale,     // [num_blocks,H_kv,block_size] fp32
    const int8_t* __restrict__ v_cache,    // [num_blocks,H_kv,block_size,D] int8 (int8 V)
    const float* __restrict__ v_scale,     // [num_blocks,H_kv,block_size] fp32 (int8 V)
    const int32_t* __restrict__ v_packed,  // [num_blocks,H_kv,block_size,D*3/32] int32 (LowbitV)
    const float* __restrict__ v_norm,      // [num_blocks,H_kv,block_size,D/128] fp32 (LowbitV)
    const float* __restrict__ v_codebook,  // [8] fp32 (LowbitV)
    const int32_t* __restrict__ block_table,    // [B, max_blocks_per_seq]
    const int32_t* __restrict__ context_lens,   // [B]
    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int n_len_max, int n_splits, int max_blocks_per_seq, int block_size, int h_kv,
    int h_q, int gqa_group) {
  constexpr int D4 = D / 4;
  constexpr int QCH = (D4 + DEC_WARP - 1) / DEC_WARP;
  constexpr int CH = D / DEC_WARP;
  constexpr int VWORDS = (D * 3) / 32;  // packed 3-bit words per row (12 @ D=128, 24 @ D=256)
  constexpr int VBLOCKS = D / 128;      // LloydMax block norms per row (1 @ D=128, 2 @ D=256)
  constexpr unsigned FULL = 0xffffffffu;

  const int lane = threadIdx.x;
  const int split = blockIdx.x;
  const int64_t bh = blockIdx.y;             // over B*H_q
  const int64_t b = bh / h_q;
  const int64_t kv_h = (bh % h_q) / gqa_group;

  const int n_len = context_lens[b];
  const int per = (n_len_max + n_splits - 1) / n_splits;
  const int n0 = split * per;
  const int n1 = min(n_len, n0 + per);

  const int8_t* q_bh = q + bh * (int64_t)D;
  const int32_t* bt_b = block_table + b * (int64_t)max_blocks_per_seq;
  const float my_qs = q_scale[bh];

  int32_t q_pk[QCH];
#pragma unroll
  for (int qc = 0; qc < QCH; ++qc) {
    const int idx = lane + qc * DEC_WARP;
    q_pk[qc] = (idx < D4) ? reinterpret_cast<const int32_t*>(q_bh)[idx] : 0;
  }

  float cb[8];
  if constexpr (LowbitV) {
#pragma unroll
    for (int i = 0; i < 8; ++i) cb[i] = v_codebook[i];
  }

  float m_i = DEC_NEG_INF, l_i = 0.f;
  float o_acc[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) o_acc[c] = 0.f;

  for (int n = n0; n < n1; ++n) {
    const int logical_block = n / block_size;
    const int offset = n % block_size;
    const int64_t phys_block = bt_b[logical_block];
    const int64_t row = (phys_block * (int64_t)h_kv + kv_h) * block_size + offset;

    const int8_t* k_row = k_cache + row * D;
    int32_t acc = 0;
#pragma unroll
    for (int qc = 0; qc < QCH; ++qc) {
      const int idx = lane + qc * DEC_WARP;
      const int32_t k_pk = (idx < D4) ? reinterpret_cast<const int32_t*>(k_row)[idx] : 0;
      acc = __dp4a(q_pk[qc], k_pk, acc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(FULL, acc, o);
    const float s = (float)acc * my_qs * k_scale[row];

    // Guard the rescale exp2f (only fires on a new max) — decode is
    // XU/exp2f-bound (ncu). Warp-uniform compare, no divergence.
    const float m_new = fmaxf(m_i, s);
    float rescale = 1.f;
    if (m_i != DEC_NEG_INF && m_new > m_i) rescale = exp2f(m_i - m_new);
    const float p = exp2f(s - m_new);
    if constexpr (LowbitV) {
      // LloydMax3 V: per-TOKEN packed codes + block norms. Each lane unpacks
      // only its own channels' 3-bit codes (LSB-first stream, word 3d>>5,
      // shift 3d&31; a code straddles into the next word only at shift 30/31,
      // and the last word of a row never straddles because 3D%32==0). The
      // codebook and norm loads are L1-broadcast across the warp.
      const int32_t* vw = v_packed + row * VWORDS;
      const float* vn = v_norm + row * VBLOCKS;
      float nv[VBLOCKS];
#pragma unroll
      for (int j = 0; j < VBLOCKS; ++j) nv[j] = vn[j];
#pragma unroll
      for (int c = 0; c < CH; ++c) {
        const int d = lane + c * DEC_WARP;
        const int w = (3 * d) >> 5;
        const int s_ = (3 * d) & 31;
        const uint32_t w0 = (uint32_t)vw[w];
        const uint32_t w1 = (s_ > 29) ? (uint32_t)vw[w + 1] : 0u;
        const int idx = (int)(((w0 >> s_) & 7u) |
                              (((s_ > 29) ? (w1 << (32 - s_)) : 0u) & 7u));
        const float vv = __half2float(__float2half_rn(cb[idx] * nv[c >> 2]));
        o_acc[c] = o_acc[c] * rescale + p * vv;
      }
    } else {
      // V scale is per-TOKEN here (unlike the contiguous int8-KV kernel's
      // per-channel fold-at-the-end), so it must apply inside the sum over n.
      const int8_t* v_row = v_cache + row * D;
      const float vs = v_scale[row];
#pragma unroll
      for (int c = 0; c < CH; ++c)
        o_acc[c] = o_acc[c] * rescale + p * (float)v_row[lane + c * DEC_WARP] * vs;
    }
    l_i = l_i * rescale + p;
    m_i = m_new;
  }

  float* op = o_part + ((int64_t)split * gridDim.y + bh) * D;
#pragma unroll
  for (int c = 0; c < CH; ++c) op[lane + c * DEC_WARP] = o_acc[c];
  if (lane == 0) {
    m_part[(int64_t)split * gridDim.y + bh] = (n1 > n0) ? m_i : DEC_NEG_INF;
    l_part[(int64_t)split * gridDim.y + bh] = l_i;
  }
}

}  // namespace fni8
