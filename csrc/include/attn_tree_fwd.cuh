// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// TREE-attention verify for sm_70 — the EAGLE / tree-speculative-decoding kernel.
//
// Tree spec-decode (EAGLE-2/3) drafts a TREE of candidate tokens and verifies the
// whole tree in ONE forward. The mask is not causal: each tree node attends (a) the
// shared prefix in the KV cache, and (b) its ANCESTORS in the tree (incl. itself) —
// NOT siblings or unrelated branches. That "attend prefix + ancestors" pattern is a
// custom mask, which is exactly why tree verification "precludes FlashAttention"
// in eager implementations. This kernel bakes it in.
//
// Layout: Q = the T tree nodes [B,H,T,D]; K/V = [prefix + T tree nodes], N = N_p+T.
// tree_mask[qi*T + kj] = 1 iff tree node kj is an ancestor-or-self of node qi (the
// drafter provides it, in the same node order as Q and the tree-key block). A key at
// global position gn is valid iff gn < tree_offset (prefix — always) OR
// tree_mask[my_m][gn - tree_offset]. int8 dp4a QK + fp32 exp2 softmax + fp16 PV;
// same lane-pair map as attn_int8_fwd.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "attn_int8_fwd.cuh"  // BLOCK_M, BLOCK_N, THREADS

namespace fni8 {

template <int D>
__global__ void __launch_bounds__(THREADS)
attn_int8_tree_kernel(const int8_t* __restrict__ q,      // [B,H,T,D] int8 (drafts)
                      const float* __restrict__ q_scale, // [B,H,T] folded
                      const int8_t* __restrict__ k,      // [B,H_kv,N,D] int8
                      const float* __restrict__ k_scale, // [B,H_kv,N]
                      const __half* __restrict__ v,      // [B,H_kv,N,D] fp16
                      __half* __restrict__ out,          // [B,H,T,D] fp16
                      const int8_t* __restrict__ tree_mask,  // [T*T], shared across B/H
                      int tree_offset, int t_len, int mask_batch_stride, int m_len,
                      int n_len, int h_q, int gqa_group) {
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
  const int64_t bh = blockIdx.y;
  const int64_t kv_bh = (bh / h_q) * (h_q / gqa_group) + (bh % h_q) / gqa_group;

  const int8_t* q_bh = q + bh * (int64_t)m_len * D;
  const int8_t* k_bh = k + kv_bh * (int64_t)n_len * D;
  const __half* v_bh = v + kv_bh * (int64_t)n_len * D;
  const float* qs_bh = q_scale + bh * m_len;
  const float* ks_bh = k_scale + kv_bh * n_len;

  for (int idx = tid; idx < BLOCK_M * D4; idx += THREADS) {
    const int r = idx / D4, c = idx % D4;
    const int gm = m_block + r;
    s_q[r][c] =
        (gm < m_len) ? reinterpret_cast<const int32_t*>(q_bh + (int64_t)gm * D)[c] : 0;
  }
  __syncthreads();

  const int my_m = m_block + row;   // this row's tree-node index (0..T-1)
  const float my_qs = (my_m < m_len) ? qs_bh[my_m] : 0.f;

  float o_acc[COLS];
#pragma unroll
  for (int i = 0; i < COLS; ++i) o_acc[i] = 0.f;
  float m_i = -INFINITY, l_i = 0.f;

  for (int n_block = 0; n_block < n_len; n_block += BLOCK_N) {
    for (int idx = tid; idx < BLOCK_N * D4; idx += THREADS) {
      const int r = idx / D4, c = idx % D4;
      const int gn = n_block + r;
      s_k[r][c] =
          (gn < n_len) ? reinterpret_cast<const int32_t*>(k_bh + (int64_t)gn * D)[c] : 0;
    }
    for (int idx = tid; idx < BLOCK_N * D; idx += THREADS) {
      const int r = idx / D, c = idx % D;
      const int gn = n_block + r;
      s_v[r][c] = (gn < n_len) ? v_bh[(int64_t)gn * D + c] : __half(0.f);
    }
    if (tid < BLOCK_N) {
      const int gn = n_block + tid;
      s_ksc[tid] = (gn < n_len) ? ks_bh[gn] : 0.f;
    }
    __syncthreads();

    float s_row[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int col = half_id * 32 + j;
      int32_t acc = 0;
#pragma unroll
      for (int c = 0; c < D4; ++c) acc = __dp4a(s_q[row][c], s_k[col][c], acc);
      const int gn = n_block + col;
      // valid: prefix key (always) OR tree ancestor-or-self per the tree mask.
      bool valid = (gn < n_len) && (my_m < m_len);
      if (valid && gn >= tree_offset)
        valid = tree_mask[(int64_t)(bh / h_q) * mask_batch_stride +
                          (int64_t)my_m * t_len + (gn - tree_offset)] != 0;
      s_row[j] = valid ? (float)acc * my_qs * s_ksc[col] : -INFINITY;
    }

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

  if (my_m < m_len) {
    const float inv_l = (l_i > 0.f) ? 1.f / l_i : 0.f;
    __half* out_row = out + bh * (int64_t)m_len * D + (int64_t)my_m * D;
#pragma unroll
    for (int i = 0; i < COLS; ++i)
      out_row[half_id * COLS + i] = __float2half(o_acc[i] * inv_l);
  }
}

}  // namespace fni8
