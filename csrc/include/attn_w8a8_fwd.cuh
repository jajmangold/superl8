// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Full-W8A8 int8 dp4a FlashAttention-2 forward for sm_70: BOTH matmuls int8.
//
// QK^T and online softmax are identical to attn_int8_fwd (INT8 QK, fp32 exp2
// softmax). The difference is PV: P is re-quantized per-row to int8 IN-LOOP
// (SDNQ triton_atten recipe: p_scale = rowmax/127 with a tiny floor) and V is
// pre-quantized PER-CHANNEL to int8; the P.V contraction runs on __dp4a
// (int8x4 -> int32) on the CUDA cores, then dequants by p_scale * v_scale[d].
//
// V is staged TRANSPOSED and packed over the key dim so dp4a contracts over
// keys: s_vt[d][kp] holds keys [4kp..4kp+3] of channel d as one int32.
// P packs the same way: each lane owns 32 keys -> 8 packs; the partner's 8
// arrive via __shfl_xor, giving every lane all 16 key-packs.
//
// This is the numerically harder matmul (P peaked, V the "free" side per
// turboquant's asymmetric-KV finding). The op falls back to fp16 PV if this
// path cannot hold its accuracy gate — the gate decides, not ideology.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "attn_int8_fwd.cuh"  // BLOCK_M, BLOCK_N, THREADS

namespace fni8 {

template <int D, bool IS_CAUSAL, bool PER_WARP = false>
__global__ void __launch_bounds__(THREADS)
attn_w8a8_fwd_kernel(const int8_t* __restrict__ q,       // [B,H,M,D] int8
                     const float* __restrict__ q_scale,  // [B,H,M] folded
                     const int8_t* __restrict__ k,       // [B,H,N,D] int8 (smoothed)
                     const float* __restrict__ k_scale,  // [B,H,N]
                     const int8_t* __restrict__ v,       // [B,H,N,D] int8
                     const float* __restrict__ v_scale,  // [B,H,D] per-channel
                     __half* __restrict__ out,           // [B,H,M,D] fp16
                     int m_len, int n_len, int h_q, int gqa_group, int causal_diag) {
  constexpr int D4 = D / 4;
  constexpr int COLS = D / 2;             // per-lane O slice
  constexpr int NPACK = BLOCK_N / 4;      // key-packs for the full tile (16)
  constexpr int MYPACK = NPACK / 2;       // packs a lane builds from its 32 keys (8)
  constexpr unsigned FULL = 0xffffffffu;

  __shared__ int32_t s_q[BLOCK_M][D4];
  __shared__ int32_t s_k[BLOCK_N][D4];
  __shared__ int32_t s_vt[D][NPACK];      // V transposed+packed over keys
  __shared__ float s_ksc[BLOCK_N];

  const int tid = threadIdx.x;
  const int row = tid >> 1;
  const int half_id = tid & 1;
  const int m_block = blockIdx.x * BLOCK_M;
  const int64_t bh = blockIdx.y;                         // over B * H_q
  const int64_t kv_bh = (bh / h_q) * (h_q / gqa_group) + (bh % h_q) / gqa_group;

  const int8_t* q_bh = q + bh * (int64_t)m_len * D;
  const int8_t* k_bh = k + kv_bh * (int64_t)n_len * D;
  const int8_t* v_bh = v + kv_bh * (int64_t)n_len * D;
  const float* qs_bh = q_scale + bh * m_len;
  const float* ks_bh = k_scale + kv_bh * n_len;
  const float* vsc_bh = v_scale + kv_bh * D;   // per-channel, shared across keys

  for (int idx = tid; idx < BLOCK_M * D4; idx += THREADS) {
    const int r = idx / D4, c = idx % D4;
    const int gm = m_block + r;
    s_q[r][c] =
        (gm < m_len) ? reinterpret_cast<const int32_t*>(q_bh + (int64_t)gm * D)[c] : 0;
  }
  __syncthreads();

  const int my_m = m_block + row;
  const float my_qs = (my_m < m_len) ? qs_bh[my_m] : 0.f;

  float o_acc[COLS];
#pragma unroll
  for (int i = 0; i < COLS; ++i) o_acc[i] = 0.f;
  float m_i = -INFINITY, l_i = 0.f;

  // causal_diag shifts the diagonal to the sequence END (query i attends keys
  // <= i + causal_diag). 0 = standard self-attention (M==N). For spec-decode/MTP
  // verify, causal_diag = N - M (drafts attend the cache prefix + preceding drafts).
  int n_end = n_len;
  if (IS_CAUSAL) { n_end = m_block + BLOCK_M + causal_diag; n_end = max(0, min(n_len, n_end)); }

  for (int n_block = 0; n_block < n_end; n_block += BLOCK_N) {
    // ---- stage K (+scales) ----
    for (int idx = tid; idx < BLOCK_N * D4; idx += THREADS) {
      const int r = idx / D4, c = idx % D4;
      const int gn = n_block + r;
      s_k[r][c] =
          (gn < n_len) ? reinterpret_cast<const int32_t*>(k_bh + (int64_t)gn * D)[c] : 0;
    }
    if (tid < BLOCK_N) {
      const int gn = n_block + tid;
      s_ksc[tid] = (gn < n_len) ? ks_bh[gn] : 0.f;
    }
    // ---- stage V transposed+packed: each thread packs 4 keys of one channel ----
    for (int idx = tid; idx < D * NPACK; idx += THREADS) {
      const int d = idx / NPACK, kp = idx % NPACK;
      int8_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
      const int n0 = n_block + kp * 4;
      if (n0 + 0 < n_len) b0 = v_bh[(int64_t)(n0 + 0) * D + d];
      if (n0 + 1 < n_len) b1 = v_bh[(int64_t)(n0 + 1) * D + d];
      if (n0 + 2 < n_len) b2 = v_bh[(int64_t)(n0 + 2) * D + d];
      if (n0 + 3 < n_len) b3 = v_bh[(int64_t)(n0 + 3) * D + d];
      s_vt[d][kp] = (uint8_t)b0 | ((uint8_t)b1 << 8) | ((uint8_t)b2 << 16) | ((uint8_t)b3 << 24);
    }
    __syncthreads();

    // ---- S = Q.K^T via dp4a -> exp2-ready logits ----
    float s_row[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int col = half_id * 32 + j;
      int32_t acc = 0;
#pragma unroll
      for (int c = 0; c < D4; ++c) acc = __dp4a(s_q[row][c], s_k[col][c], acc);
      const int gn = n_block + col;
      const bool valid = (gn < n_len) && (!IS_CAUSAL || gn <= my_m + causal_diag);
      s_row[j] = valid ? (float)acc * my_qs * s_ksc[col] : -INFINITY;
    }

    // ---- online softmax ----
    float tile_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < 32; ++j) tile_max = fmaxf(tile_max, s_row[j]);
    tile_max = fmaxf(tile_max, __shfl_xor_sync(FULL, tile_max, 1));
    const float m_new = fmaxf(m_i, tile_max);

    float p_row[32];
    float p_max = 0.f;
    if (m_new == -INFINITY) {
#pragma unroll
      for (int j = 0; j < 32; ++j) p_row[j] = 0.f;
    } else {
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        p_row[j] = (s_row[j] == -INFINITY) ? 0.f : exp2f(s_row[j] - m_new);
        p_max = fmaxf(p_max, p_row[j]);
      }
    }
    if constexpr (!PER_WARP) {
      p_max = fmaxf(p_max, __shfl_xor_sync(FULL, p_max, 1));  // lane-pair merge -> per-row
    }

    // ---- requantize P to int8 (per-row or per-warp) ----
    // PER_WARP: each lane keeps its own p_max over 32 keys -> per-lane scale.
    // !PER_WARP: p_max merged across the lane pair -> one scale per 64-key row.
    const float p_scale = fmaxf(p_max * (1.f / 127.f), 2e-38f);  // SDNQ floor
    const float inv_ps = 1.f / p_scale;
    int32_t my_pk[MYPACK];
#pragma unroll
    for (int pk = 0; pk < MYPACK; ++pk) {
      int8_t q0 = (int8_t)__float2int_rn(p_row[pk * 4 + 0] * inv_ps);
      int8_t q1 = (int8_t)__float2int_rn(p_row[pk * 4 + 1] * inv_ps);
      int8_t q2 = (int8_t)__float2int_rn(p_row[pk * 4 + 2] * inv_ps);
      int8_t q3 = (int8_t)__float2int_rn(p_row[pk * 4 + 3] * inv_ps);
      my_pk[pk] = (uint8_t)q0 | ((uint8_t)q1 << 8) | ((uint8_t)q2 << 16) | ((uint8_t)q3 << 24);
    }
    // partner's 8 packs; assemble all 16 (half_id 0 owns keys 0..31 = packs 0..7)
    int32_t packs[NPACK];
#pragma unroll
    for (int pk = 0; pk < MYPACK; ++pk) {
      const int32_t partner = __shfl_xor_sync(FULL, my_pk[pk], 1);
      if (half_id == 0) {
        packs[pk] = my_pk[pk];
        packs[MYPACK + pk] = partner;
      } else {
        packs[pk] = partner;
        packs[MYPACK + pk] = my_pk[pk];
      }
    }

    // p_sum for the running denominator (fp32, exact)
    float p_sum = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) p_sum += p_row[j];
    p_sum += __shfl_xor_sync(FULL, p_sum, 1);

    // ---- rescale running state ----
    const float rescale = (m_i == -INFINITY || m_new == -INFINITY) ? 1.f : exp2f(m_i - m_new);
    if (m_i != -INFINITY) {
#pragma unroll
      for (int i = 0; i < COLS; ++i) o_acc[i] *= rescale;
      l_i *= rescale;
    }
    l_i += p_sum;
    m_i = fmaxf(m_i, m_new);

    // ---- O += (P.V) via dp4a, dequant by p_scale * v_scale[d] ----
    // PER_WARP: each half of packs[] was quantized with a different p_scale
    // (my 32-key half uses this lane's scale; partner's half uses partner's).
    // Split the dp4a into two accumulators and dequant each with its own scale.
    const int d0 = half_id * COLS;
    if (m_new != -INFINITY) {
      if constexpr (PER_WARP) {
        const float p_scale_part = __shfl_xor_sync(FULL, p_scale, 1);
        const float ps_low = (half_id == 0) ? p_scale : p_scale_part;
        const float ps_high = (half_id == 0) ? p_scale_part : p_scale;
#pragma unroll
        for (int i = 0; i < COLS; ++i) {
          const int d = d0 + i;
          int32_t acc = 0;
#pragma unroll
          for (int pk = 0; pk < MYPACK; ++pk) acc = __dp4a(packs[pk], s_vt[d][pk], acc);
          float val = (float)acc * ps_low;
          acc = 0;
#pragma unroll
          for (int pk = MYPACK; pk < NPACK; ++pk) acc = __dp4a(packs[pk], s_vt[d][pk], acc);
          o_acc[i] += (val + (float)acc * ps_high) * vsc_bh[d];
        }
      } else {
#pragma unroll
        for (int i = 0; i < COLS; ++i) {
          const int d = d0 + i;
          int32_t acc = 0;
#pragma unroll
          for (int pk = 0; pk < NPACK; ++pk) acc = __dp4a(packs[pk], s_vt[d][pk], acc);
          o_acc[i] += (float)acc * p_scale * vsc_bh[d];
        }
      }
    }
    __syncthreads();
  }

  if (my_m < m_len) {
    const float inv_l = (l_i > 0.f) ? 1.f / l_i : 0.f;
    __half* out_row = out + bh * (int64_t)m_len * D + (int64_t)my_m * D;
#pragma unroll
    for (int i = 0; i < COLS; i += 2)
      pair_store(out_row + half_id * COLS + i, o_acc[i] * inv_l, o_acc[i + 1] * inv_l);
  }
}

// ---------------------------------------------------------------------------
// DYNAMIC-SMEM variant, for head dims whose W8A8 tiles exceed the 48 KB static
// cap. At D=256 the tiles are s_q(16 KB) + s_k(16 KB) + s_vt(16 KB) + s_ksc =
// ~48.25 KB > 49152, so the static kernel above will not launch. This variant is
// numerically IDENTICAL (same lane-pair map, int8 dp4a QK, fp32 exp2 softmax,
// int8 dp4a PV) — only the shared tiles move to a dynamic `extern __shared__`
// buffer opted in via cudaFuncSetAttribute. Unblocks head_dim 256 (Gemma-family
// and the 9B/27B MTP speculative-decode VERIFY path). Correctness-first: at D=256
// the per-thread O accumulator is COLS=D/2=128 wide, so occupancy is low — but
// the verify shape (M = k drafts, small) leans on the B*H grid for parallelism.
template <int D, bool PER_WARP>
constexpr int w8a8_dyn_smem_bytes() {
  constexpr int D4 = D / 4;
  constexpr int NPACK = BLOCK_N / 4;
  return (BLOCK_M + BLOCK_N) * D4 * (int)sizeof(int32_t)  // s_q + s_k (int8 packed)
         + D * NPACK * (int)sizeof(int32_t)               // s_vt (V transposed+packed)
         + BLOCK_N * (int)sizeof(float);                  // s_ksc
}

template <int D, bool IS_CAUSAL, bool PER_WARP = false>
__global__ void __launch_bounds__(THREADS)
attn_w8a8_fwd_dyn_kernel(const int8_t* __restrict__ q, const float* __restrict__ q_scale,
                         const int8_t* __restrict__ k, const float* __restrict__ k_scale,
                         const int8_t* __restrict__ v, const float* __restrict__ v_scale,
                         __half* __restrict__ out, int m_len, int n_len, int h_q,
                         int gqa_group, int causal_diag) {
  constexpr int D4 = D / 4;
  constexpr int COLS = D / 2;
  constexpr int NPACK = BLOCK_N / 4;
  constexpr int MYPACK = NPACK / 2;
  constexpr unsigned FULL = 0xffffffffu;

  extern __shared__ char smem_dyn_w8a8[];
  auto s_q = reinterpret_cast<int32_t(*)[D4]>(smem_dyn_w8a8);
  auto s_k = reinterpret_cast<int32_t(*)[D4]>(smem_dyn_w8a8 + BLOCK_M * D4 * sizeof(int32_t));
  auto s_vt = reinterpret_cast<int32_t(*)[NPACK]>(
      smem_dyn_w8a8 + (BLOCK_M + BLOCK_N) * D4 * sizeof(int32_t));
  auto s_ksc = reinterpret_cast<float*>(
      smem_dyn_w8a8 + (BLOCK_M + BLOCK_N) * D4 * sizeof(int32_t) + D * NPACK * sizeof(int32_t));

  const int tid = threadIdx.x;
  const int row = tid >> 1;
  const int half_id = tid & 1;
  const int m_block = blockIdx.x * BLOCK_M;
  const int64_t bh = blockIdx.y;
  const int64_t kv_bh = (bh / h_q) * (h_q / gqa_group) + (bh % h_q) / gqa_group;

  const int8_t* q_bh = q + bh * (int64_t)m_len * D;
  const int8_t* k_bh = k + kv_bh * (int64_t)n_len * D;
  const int8_t* v_bh = v + kv_bh * (int64_t)n_len * D;
  const float* qs_bh = q_scale + bh * m_len;
  const float* ks_bh = k_scale + kv_bh * n_len;
  const float* vsc_bh = v_scale + kv_bh * D;

  for (int idx = tid; idx < BLOCK_M * D4; idx += THREADS) {
    const int r = idx / D4, c = idx % D4;
    const int gm = m_block + r;
    s_q[r][c] =
        (gm < m_len) ? reinterpret_cast<const int32_t*>(q_bh + (int64_t)gm * D)[c] : 0;
  }
  __syncthreads();

  const int my_m = m_block + row;
  const float my_qs = (my_m < m_len) ? qs_bh[my_m] : 0.f;

  float o_acc[COLS];
#pragma unroll
  for (int i = 0; i < COLS; ++i) o_acc[i] = 0.f;
  float m_i = -INFINITY, l_i = 0.f;

  int n_end = n_len;
  if (IS_CAUSAL) { n_end = m_block + BLOCK_M + causal_diag; n_end = max(0, min(n_len, n_end)); }

  for (int n_block = 0; n_block < n_end; n_block += BLOCK_N) {
    for (int idx = tid; idx < BLOCK_N * D4; idx += THREADS) {
      const int r = idx / D4, c = idx % D4;
      const int gn = n_block + r;
      s_k[r][c] =
          (gn < n_len) ? reinterpret_cast<const int32_t*>(k_bh + (int64_t)gn * D)[c] : 0;
    }
    if (tid < BLOCK_N) {
      const int gn = n_block + tid;
      s_ksc[tid] = (gn < n_len) ? ks_bh[gn] : 0.f;
    }
    for (int idx = tid; idx < D * NPACK; idx += THREADS) {
      const int d = idx / NPACK, kp = idx % NPACK;
      int8_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
      const int n0 = n_block + kp * 4;
      if (n0 + 0 < n_len) b0 = v_bh[(int64_t)(n0 + 0) * D + d];
      if (n0 + 1 < n_len) b1 = v_bh[(int64_t)(n0 + 1) * D + d];
      if (n0 + 2 < n_len) b2 = v_bh[(int64_t)(n0 + 2) * D + d];
      if (n0 + 3 < n_len) b3 = v_bh[(int64_t)(n0 + 3) * D + d];
      s_vt[d][kp] = (uint8_t)b0 | ((uint8_t)b1 << 8) | ((uint8_t)b2 << 16) | ((uint8_t)b3 << 24);
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
      const bool valid = (gn < n_len) && (!IS_CAUSAL || gn <= my_m + causal_diag);
      s_row[j] = valid ? (float)acc * my_qs * s_ksc[col] : -INFINITY;
    }

    float tile_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < 32; ++j) tile_max = fmaxf(tile_max, s_row[j]);
    tile_max = fmaxf(tile_max, __shfl_xor_sync(FULL, tile_max, 1));
    const float m_new = fmaxf(m_i, tile_max);

    float p_row[32];
    float p_max = 0.f;
    if (m_new == -INFINITY) {
#pragma unroll
      for (int j = 0; j < 32; ++j) p_row[j] = 0.f;
    } else {
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        p_row[j] = (s_row[j] == -INFINITY) ? 0.f : exp2f(s_row[j] - m_new);
        p_max = fmaxf(p_max, p_row[j]);
      }
    }
    if constexpr (!PER_WARP) {
      p_max = fmaxf(p_max, __shfl_xor_sync(FULL, p_max, 1));
    }

    const float p_scale = fmaxf(p_max * (1.f / 127.f), 2e-38f);
    const float inv_ps = 1.f / p_scale;
    int32_t my_pk[MYPACK];
#pragma unroll
    for (int pk = 0; pk < MYPACK; ++pk) {
      int8_t q0 = (int8_t)__float2int_rn(p_row[pk * 4 + 0] * inv_ps);
      int8_t q1 = (int8_t)__float2int_rn(p_row[pk * 4 + 1] * inv_ps);
      int8_t q2 = (int8_t)__float2int_rn(p_row[pk * 4 + 2] * inv_ps);
      int8_t q3 = (int8_t)__float2int_rn(p_row[pk * 4 + 3] * inv_ps);
      my_pk[pk] = (uint8_t)q0 | ((uint8_t)q1 << 8) | ((uint8_t)q2 << 16) | ((uint8_t)q3 << 24);
    }
    int32_t packs[NPACK];
#pragma unroll
    for (int pk = 0; pk < MYPACK; ++pk) {
      const int32_t partner = __shfl_xor_sync(FULL, my_pk[pk], 1);
      if (half_id == 0) {
        packs[pk] = my_pk[pk];
        packs[MYPACK + pk] = partner;
      } else {
        packs[pk] = partner;
        packs[MYPACK + pk] = my_pk[pk];
      }
    }

    float p_sum = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) p_sum += p_row[j];
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
    if (m_new != -INFINITY) {
      if constexpr (PER_WARP) {
        const float p_scale_part = __shfl_xor_sync(FULL, p_scale, 1);
        const float ps_low = (half_id == 0) ? p_scale : p_scale_part;
        const float ps_high = (half_id == 0) ? p_scale_part : p_scale;
#pragma unroll
        for (int i = 0; i < COLS; ++i) {
          const int d = d0 + i;
          int32_t acc = 0;
#pragma unroll
          for (int pk = 0; pk < MYPACK; ++pk) acc = __dp4a(packs[pk], s_vt[d][pk], acc);
          float val = (float)acc * ps_low;
          acc = 0;
#pragma unroll
          for (int pk = MYPACK; pk < NPACK; ++pk) acc = __dp4a(packs[pk], s_vt[d][pk], acc);
          o_acc[i] += (val + (float)acc * ps_high) * vsc_bh[d];
        }
      } else {
#pragma unroll
        for (int i = 0; i < COLS; ++i) {
          const int d = d0 + i;
          int32_t acc = 0;
#pragma unroll
          for (int pk = 0; pk < NPACK; ++pk) acc = __dp4a(packs[pk], s_vt[d][pk], acc);
          o_acc[i] += (float)acc * p_scale * vsc_bh[d];
        }
      }
    }
    __syncthreads();
  }

  if (my_m < m_len) {
    const float inv_l = (l_i > 0.f) ? 1.f / l_i : 0.f;
    __half* out_row = out + bh * (int64_t)m_len * D + (int64_t)my_m * D;
#pragma unroll
    for (int i = 0; i < COLS; i += 2)
      pair_store(out_row + half_id * COLS + i, o_acc[i] * inv_l, o_acc[i + 1] * inv_l);
  }
}

}  // namespace fni8
