// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Varlen (cu_seqlens-packed) int8 dp4a FlashAttention-2 forward for sm_70 —
// the serving PREFILL path. INT8 QK + fp32 exp2 softmax + fp16 PV.
//
// Serving packs variable-length prompts contiguously (no padding). Layout is the
// flash_attn_varlen convention: q/k/v are [total_tokens, H, D] (token-major,
// heads interleaved); cu_seqlens[b] is sequence b's start offset. Each sequence
// attends only within itself. The per-tile math is identical to attn_int8_fwd;
// only the addressing (token-major, sequence-relative) and the causal diagonal
// (aligned to the sequence end so KV-cache prefixes work) differ.
//
// Grid: (ceil(max_seqlen_q / BLOCK_M), batch, H_q). A block owns (m_tile, seq,
// head); tiles past its sequence length early-exit. Same lane-pair thread map as
// attn_int8_fwd (BLOCK_M=BLOCK_N=64, 128 threads).
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "attn_int8_fwd.cuh"  // BLOCK_M, BLOCK_N, THREADS

namespace fni8 {

template <int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(THREADS)
attn_int8_varlen_fwd_kernel(const int8_t* __restrict__ q,       // [total_q, H_q, D] int8
                            const float* __restrict__ q_scale,  // [total_q, H_q] folded
                            const int8_t* __restrict__ k,       // [total_k, H_kv, D] int8
                            const float* __restrict__ k_scale,  // [total_k, H_kv]
                            const __half* __restrict__ v,       // [total_k, H_kv, D] fp16
                            __half* __restrict__ out,           // [total_q, H_q, D] fp16
                            const int* __restrict__ cu_q,       // [batch+1]
                            const int* __restrict__ cu_k,       // [batch+1]
                            int h_q, int h_kv, int gqa_group) {
  constexpr int D4 = D / 4;
  constexpr int COLS = D / 2;
  constexpr unsigned FULL = 0xffffffffu;

  __shared__ int32_t s_q[BLOCK_M][D4];
  __shared__ int32_t s_k[BLOCK_N][D4];
  __shared__ __half s_v[BLOCK_N][D];
  __shared__ float s_ksc[BLOCK_N];

  const int tid = threadIdx.x;
  const int row = tid >> 1;
  const int half_id = tid & 1;
  const int m_block = blockIdx.x * BLOCK_M;
  const int seq = blockIdx.y;             // sequence in the batch
  const int head = blockIdx.z;            // query head
  const int hk = head / gqa_group;        // shared K/V head (GQA)

  const int q0 = cu_q[seq], q1 = cu_q[seq + 1];
  const int k0 = cu_k[seq], k1 = cu_k[seq + 1];
  const int seqlen_q = q1 - q0;
  const int seqlen_k = k1 - k0;
  if (m_block >= seqlen_q) return;        // tile past this sequence

  // token-major bases: element (token t, head h, d) at (t*H + h)*D + d.
  const int8_t* q_base = q + ((int64_t)q0 * h_q + head) * D;
  const int8_t* k_base = k + ((int64_t)k0 * h_kv + hk) * D;
  const __half* v_base = v + ((int64_t)k0 * h_kv + hk) * D;
  const float* qs_base = q_scale + (int64_t)q0 * h_q + head;
  const float* ks_base = k_scale + (int64_t)k0 * h_kv + hk;
  __half* out_base = out + ((int64_t)q0 * h_q + head) * D;

  // ---- load Q tile (rows are seqlen_q-bounded; token stride is H_q*D) ----
  for (int idx = tid; idx < BLOCK_M * D4; idx += THREADS) {
    const int r = idx / D4, c = idx % D4;
    const int gm = m_block + r;
    s_q[r][c] = (gm < seqlen_q)
                    ? reinterpret_cast<const int32_t*>(q_base + (int64_t)gm * h_q * D)[c]
                    : 0;
  }
  __syncthreads();

  const int my_m = m_block + row;
  const float my_qs = (my_m < seqlen_q) ? qs_base[(int64_t)my_m * h_q] : 0.f;

  float o_acc[COLS];
#pragma unroll
  for (int i = 0; i < COLS; ++i) o_acc[i] = 0.f;
  float m_i = -INFINITY, l_i = 0.f;

  // causal diagonal aligned to the sequence end (KV-cache prefix convention):
  // query my_m attends key gn <= my_m + (seqlen_k - seqlen_q).
  const int diag = seqlen_k - seqlen_q;
  int n_end = seqlen_k;
  if (IS_CAUSAL) {
    n_end = m_block + BLOCK_M + diag;
    if (n_end > seqlen_k) n_end = seqlen_k;
    if (n_end < 0) n_end = 0;
  }

  for (int n_block = 0; n_block < n_end; n_block += BLOCK_N) {
    for (int idx = tid; idx < BLOCK_N * D4; idx += THREADS) {
      const int r = idx / D4, c = idx % D4;
      const int gn = n_block + r;
      s_k[r][c] = (gn < seqlen_k)
                      ? reinterpret_cast<const int32_t*>(k_base + (int64_t)gn * h_kv * D)[c]
                      : 0;
    }
    for (int idx = tid; idx < BLOCK_N * D; idx += THREADS) {
      const int r = idx / D, c = idx % D;
      const int gn = n_block + r;
      s_v[r][c] = (gn < seqlen_k) ? v_base[(int64_t)gn * h_kv * D + c] : __half(0.f);
    }
    if (tid < BLOCK_N) {
      const int gn = n_block + tid;
      s_ksc[tid] = (gn < seqlen_k) ? ks_base[(int64_t)gn * h_kv] : 0.f;
    }
    __syncthreads();

    // ---- S = Q.K^T via dp4a; dequant to exp2-ready fp32 logits ----
    float s_row[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int col = half_id * 32 + j;
      int32_t acc = 0;
#pragma unroll
      for (int c = 0; c < D4; ++c) acc = __dp4a(s_q[row][c], s_k[col][c], acc);
      const int gn = n_block + col;
      const bool valid = (gn < seqlen_k) && (!IS_CAUSAL || gn <= my_m + diag);
      s_row[j] = valid ? (float)acc * my_qs * s_ksc[col] : -INFINITY;
    }

    // ---- online softmax (lane pair covers the full row) ----
    float tile_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < 32; ++j) tile_max = fmaxf(tile_max, s_row[j]);
    tile_max = fmaxf(tile_max, __shfl_xor_sync(FULL, tile_max, 1));
    const float m_new = fmaxf(m_i, tile_max);

    float p_row[32];
    float p_sum = 0.f;
    if (m_new == -INFINITY) {
#pragma unroll
      for (int j = 0; j < 32; ++j) p_row[j] = 0.f;
    } else {
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        p_row[j] = (s_row[j] == -INFINITY) ? 0.f : exp2f(s_row[j] - m_new);
        p_sum += p_row[j];
      }
    }
    p_sum += __shfl_xor_sync(FULL, p_sum, 1);

    const float rescale = (m_i == -INFINITY || m_new == -INFINITY) ? 1.f : exp2f(m_i - m_new);
    if (m_i != -INFINITY) {
#pragma unroll
      for (int i = 0; i < COLS; ++i) o_acc[i] *= rescale;
      l_i *= rescale;
    }
    l_i += p_sum;
    m_i = fmaxf(m_i, m_new);

    // ---- O += P.V over the full key tile (my 32 P cols + partner's via shfl) ----
    const int d0 = half_id * COLS;
#pragma unroll 4
    for (int j = 0; j < 32; ++j) {
      const float p_mine = p_row[j];
      const float p_part = __shfl_xor_sync(FULL, p_row[j], 1);
      const int col_mine = half_id * 32 + j;
      const int col_part = (1 - half_id) * 32 + j;
      if (p_mine != 0.f) {
        const __half* v_row = &s_v[col_mine][d0];
#pragma unroll
        for (int i = 0; i < COLS; ++i) o_acc[i] = fmaf(p_mine, __half2float(v_row[i]), o_acc[i]);
      }
      if (p_part != 0.f) {
        const __half* v_row = &s_v[col_part][d0];
#pragma unroll
        for (int i = 0; i < COLS; ++i) o_acc[i] = fmaf(p_part, __half2float(v_row[i]), o_acc[i]);
      }
    }
    __syncthreads();
  }

  if (my_m < seqlen_q) {
    const float inv_l = (l_i > 0.f) ? 1.f / l_i : 0.f;
    __half* out_row = out_base + (int64_t)my_m * h_q * D;
#pragma unroll
    for (int i = 0; i < COLS; i += 2)
      pair_store(out_row + half_id * COLS + i, o_acc[i] * inv_l, o_acc[i + 1] * inv_l);
  }
}

}  // namespace fni8
