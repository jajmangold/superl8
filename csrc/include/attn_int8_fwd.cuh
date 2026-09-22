// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// int8 dp4a QK^T FlashAttention-2 forward for sm_70 (MVP tile config).
//
// sm_70 has NO int8 tensor cores; QK^T runs as a SIMT tile GEMM on __dp4a
// (int8x4 dot -> int32 accumulate on the CUDA cores). Softmax is fp32 with
// exp2 (softmax_scale * log2(e) is pre-folded into q_scale by the Python
// prologue; K is pre-smoothed by subtracting its per-channel mean, which
// shifts every logit in a row equally and so leaves the SOFTMAX ATTENTION
// OUTPUT invariant — it does NOT preserve raw logits). This is INT8 QK with
// fp32-accumulated fp16 PV; it is NOT full-INT8 attention (int8 PV is PR4).
//
// Thread map (BLOCK_M=64, BLOCK_N=64, 128 threads = 4 warps):
//   - Lane pair (tid, tid^1) owns one S row: each lane computes 32 of the 64
//     S columns and owns the row's D/2 output slice. Row max/sum finish with
//     __shfl_xor_sync(., 1). In PV, a lane needs P for all 64 keys against its
//     D-slice; the partner's 32 P values arrive via one __shfl_xor — no smem.
//
// PERF NOTE (measured on V100, PR3): a 4-lanes-per-row split that bounds
// o_acc to 32 regs was tried to relieve register pressure, but ptxas already
// caps this kernel at 128 regs via __launch_bounds__, so the split only added
// __shfl traffic and REGRESSED both D (D64 1.06x->0.88x, D128 0.91x->0.68x).
// Register pressure is real but not the binding constraint here; finding the
// true optimum (tile sizes, threads, maxrregcount, PV strategy) is an
// autotuning search deferred to PR7. This lane-pair map is the measured best.
//
// Smem: Q int8 64xD + K int8 64xD + V half 64xD + k_scale (<= 24.25 KB @ D=128).
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "compute_dtype.cuh"

namespace fni8 {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int THREADS = 128;

// Grid: (ceil(M / BLOCK_M), B * H). Layout: [B, H, M, D] contiguous. `VT` is V's
// (and thus O's — the output dtype always matches V's, per the torch wrapper)
// storage type: __half or __nv_bfloat16. fp16 overflows (max 65504) on
// bf16-native models' V (Gemma, most diffusion DiTs) -> inf/NaN -> black output.
template <int D, bool IS_CAUSAL, typename VT>
__global__ void __launch_bounds__(THREADS)
attn_int8_fwd_kernel(const int8_t* __restrict__ q,      // [B,H,M,D] int8
                     const float* __restrict__ q_scale, // [B,H,M] folded: sq*softmax_scale*log2e
                     const int8_t* __restrict__ k,      // [B,H,N,D] int8 (K-smoothed)
                     const float* __restrict__ k_scale, // [B,H,N]
                     const VT* __restrict__ v,          // [B,H,N,D] fp16 or bf16
                     VT* __restrict__ out,              // [B,H,M,D] fp16 or bf16
                     float* __restrict__ lse,           // [B,H,M] fp32, or nullptr
                     int m_len, int n_len, int h_q, int gqa_group, int window_left) {
  constexpr int D4 = D / 4;    // int32-packed head dim
  constexpr int COLS = D / 2;  // per-lane O slice width
  constexpr unsigned FULL = 0xffffffffu;

  __shared__ int32_t s_q[BLOCK_M][D4];
  __shared__ int32_t s_k[BLOCK_N][D4];
  __shared__ VT s_v[BLOCK_N][D];
  __shared__ float s_ksc[BLOCK_N];

  const int tid = threadIdx.x;
  const int row = tid >> 1;      // S row owned by this lane pair (0..63)
  const int half_id = tid & 1;   // which 32-col S slice / D-half of O
  const int m_block = blockIdx.x * BLOCK_M;
  const int64_t bh = blockIdx.y;                         // over B * H_q
  // GQA/MQA: this Q head shares a K/V head. kv_bh maps into the H_kv-head K/V.
  const int64_t kv_bh = (bh / h_q) * (h_q / gqa_group) + (bh % h_q) / gqa_group;

  const int8_t* q_bh = q + bh * (int64_t)m_len * D;
  const int8_t* k_bh = k + kv_bh * (int64_t)n_len * D;
  const VT* v_bh = v + kv_bh * (int64_t)n_len * D;
  const float* qs_bh = q_scale + bh * m_len;
  const float* ks_bh = k_scale + kv_bh * n_len;

  // ---- load Q tile (int8 rows packed as int32; zero-pad ragged tail) ----
  for (int idx = tid; idx < BLOCK_M * D4; idx += THREADS) {
    const int r = idx / D4, c = idx % D4;
    const int gm = m_block + r;
    s_q[r][c] =
        (gm < m_len) ? reinterpret_cast<const int32_t*>(q_bh + (int64_t)gm * D)[c] : 0;
  }
  __syncthreads();

  const int my_m = m_block + row;
  // FlashAttention convention for rectangular causal attention: align the
  // causal diagonal to the bottom-right. Cached chunk query i maps to absolute
  // key position i + (N - M).
  const int causal_diag = IS_CAUSAL ? n_len - m_len : 0;
  const int my_q_pos = IS_CAUSAL ? my_m + causal_diag : my_m;
  const float my_qs = (my_m < m_len) ? qs_bh[my_m] : 0.f;

  float o_acc[COLS];
#pragma unroll
  for (int i = 0; i < COLS; ++i) o_acc[i] = 0.f;
  float m_i = -INFINITY;  // running row max (exp2 domain)
  float l_i = 0.f;        // running row sum

  const int n_end = IS_CAUSAL
                        ? max(0, min(n_len, m_block + BLOCK_M + causal_diag))
                        : n_len;

  for (int n_block = 0; n_block < n_end; n_block += BLOCK_N) {
    // ---- stage K (+scales) and V tiles ----
    for (int idx = tid; idx < BLOCK_N * D4; idx += THREADS) {
      const int r = idx / D4, c = idx % D4;
      const int gn = n_block + r;
      s_k[r][c] =
          (gn < n_len) ? reinterpret_cast<const int32_t*>(k_bh + (int64_t)gn * D)[c] : 0;
    }
    for (int idx = tid; idx < BLOCK_N * D; idx += THREADS) {
      const int r = idx / D, c = idx % D;
      const int gn = n_block + r;
      s_v[r][c] = (gn < n_len) ? v_bh[(int64_t)gn * D + c] : zero_val<VT>();
    }
    if (tid < BLOCK_N) {
      const int gn = n_block + tid;
      s_ksc[tid] = (gn < n_len) ? ks_bh[gn] : 0.f;
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
      const bool valid = (gn < n_len) && (!IS_CAUSAL || gn <= my_q_pos) &&
                         (window_left < 0 || gn > my_q_pos - window_left);
      s_row[j] = valid ? (float)acc * my_qs * s_ksc[col] : -INFINITY;
    }

    // ---- online softmax update (lane pair covers the full row) ----
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

    // ---- O += P.V over the FULL key tile: my 32 P cols + partner's via shfl ----
    const int d0 = half_id * COLS;
#pragma unroll 4
    for (int j = 0; j < 32; ++j) {
      const float p_mine = p_row[j];
      const float p_part = __shfl_xor_sync(FULL, p_row[j], 1);
      const int col_mine = half_id * 32 + j;
      const int col_part = (1 - half_id) * 32 + j;
      if (p_mine != 0.f) {
        const VT* v_row = &s_v[col_mine][d0];
#pragma unroll
        for (int i = 0; i < COLS; ++i) o_acc[i] = fmaf(p_mine, to_float<VT>(v_row[i]), o_acc[i]);
      }
      if (p_part != 0.f) {
        const VT* v_row = &s_v[col_part][d0];
#pragma unroll
        for (int i = 0; i < COLS; ++i) o_acc[i] = fmaf(p_part, to_float<VT>(v_row[i]), o_acc[i]);
      }
    }
    __syncthreads();  // tiles reused next iteration
  }

  // ---- epilogue: O /= l, write fp16/bf16 (each lane writes its own D-half) ----
  // For __half output: use packed __half2 stores (__float22half2_rn + st.v2.f16)
  // halving the store count — the healthy CUDA-core half2 pipe (~27 TFLOP/s) vs
  // the firmware-gimped tensor cores. bf16 keeps scalar stores (no native sm_70
  // bf16 vector store).
  if (my_m < m_len) {
    const float inv_l = (l_i > 0.f) ? 1.f / l_i : 0.f;  // N==0 / fully-masked -> 0
    VT* out_row = out + bh * (int64_t)m_len * D + (int64_t)my_m * D;
#pragma unroll
    for (int i = 0; i < COLS; i += 2)
      pair_store(out_row + half_id * COLS + i, o_acc[i] * inv_l, o_acc[i + 1] * inv_l);
    // logsumexp in NATURAL log (kernel logits carry a log2(e) fold -> * ln2).
    // lane 0 of the pair writes; both hold the same m_i/l_i.
    if (lse != nullptr && half_id == 0)
      lse[bh * (int64_t)m_len + my_m] =
          (l_i > 0.f) ? (m_i * 0.6931471805599453f + logf(l_i)) : -INFINITY;
  }
}

// ---------------------------------------------------------------------------
// DYNAMIC-SMEM variant, for head dims whose tiles exceed the 48 KB static cap
// (D=256: s_q+s_k+s_v+scales ~= 64 KB). Identical math to attn_int8_fwd_kernel
// (lane-pair, int8 dp4a QK, fp32 exp2 softmax, fp16 PV); only the shared tiles
// move to a dynamic `extern __shared__` buffer opted in via cudaFuncSetAttribute.
// Unblocks Gemma-family (head_dim 256) and some diffusion DiTs. Correctness-first:
// at D=256 the per-thread O accumulator is COLS=D/2 wide, so occupancy is low —
// a threads-per-row treatment (perf) is a follow-up; the SHAPE works today.
// ---------------------------------------------------------------------------
template <int D, typename VT>
constexpr int fwd_dyn_smem_bytes() {
  constexpr int D4 = D / 4;
  return (BLOCK_M + BLOCK_N) * D4 * (int)sizeof(int32_t)   // s_q + s_k (int8 packed)
         + BLOCK_N * D * (int)sizeof(VT)                   // s_v
         + BLOCK_N * (int)sizeof(float);                   // s_ksc
}

template <int D, bool IS_CAUSAL, typename VT>
__global__ void __launch_bounds__(THREADS)
attn_int8_fwd_dyn_kernel(const int8_t* __restrict__ q, const float* __restrict__ q_scale,
                         const int8_t* __restrict__ k, const float* __restrict__ k_scale,
                         const VT* __restrict__ v, VT* __restrict__ out,
                         float* __restrict__ lse, int m_len, int n_len, int h_q,
                         int gqa_group, int window_left) {
  constexpr int D4 = D / 4;
  constexpr int COLS = D / 2;
  constexpr unsigned FULL = 0xffffffffu;

  extern __shared__ char smem_dyn[];
  auto s_q = reinterpret_cast<int32_t(*)[D4]>(smem_dyn);
  auto s_k = reinterpret_cast<int32_t(*)[D4]>(smem_dyn + BLOCK_M * D4 * sizeof(int32_t));
  auto s_v = reinterpret_cast<VT(*)[D]>(
      smem_dyn + (BLOCK_M + BLOCK_N) * D4 * sizeof(int32_t));
  auto s_ksc = reinterpret_cast<float*>(
      smem_dyn + (BLOCK_M + BLOCK_N) * D4 * sizeof(int32_t) + BLOCK_N * D * sizeof(VT));

  const int tid = threadIdx.x;
  const int row = tid >> 1;
  const int half_id = tid & 1;
  const int m_block = blockIdx.x * BLOCK_M;
  const int64_t bh = blockIdx.y;
  const int64_t kv_bh = (bh / h_q) * (h_q / gqa_group) + (bh % h_q) / gqa_group;

  const int8_t* q_bh = q + bh * (int64_t)m_len * D;
  const int8_t* k_bh = k + kv_bh * (int64_t)n_len * D;
  const VT* v_bh = v + kv_bh * (int64_t)n_len * D;
  const float* qs_bh = q_scale + bh * m_len;
  const float* ks_bh = k_scale + kv_bh * n_len;

  for (int idx = tid; idx < BLOCK_M * D4; idx += THREADS) {
    const int r = idx / D4, c = idx % D4;
    const int gm = m_block + r;
    s_q[r][c] =
        (gm < m_len) ? reinterpret_cast<const int32_t*>(q_bh + (int64_t)gm * D)[c] : 0;
  }
  __syncthreads();

  const int my_m = m_block + row;
  const int causal_diag = IS_CAUSAL ? n_len - m_len : 0;
  const int my_q_pos = IS_CAUSAL ? my_m + causal_diag : my_m;
  const float my_qs = (my_m < m_len) ? qs_bh[my_m] : 0.f;

  float o_acc[COLS];
#pragma unroll
  for (int i = 0; i < COLS; ++i) o_acc[i] = 0.f;
  float m_i = -INFINITY, l_i = 0.f;

  const int n_end = IS_CAUSAL
                        ? max(0, min(n_len, m_block + BLOCK_M + causal_diag))
                        : n_len;

  for (int n_block = 0; n_block < n_end; n_block += BLOCK_N) {
    for (int idx = tid; idx < BLOCK_N * D4; idx += THREADS) {
      const int r = idx / D4, c = idx % D4;
      const int gn = n_block + r;
      s_k[r][c] =
          (gn < n_len) ? reinterpret_cast<const int32_t*>(k_bh + (int64_t)gn * D)[c] : 0;
    }
    for (int idx = tid; idx < BLOCK_N * D; idx += THREADS) {
      const int r = idx / D, c = idx % D;
      const int gn = n_block + r;
      s_v[r][c] = (gn < n_len) ? v_bh[(int64_t)gn * D + c] : zero_val<VT>();
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
      const bool valid = (gn < n_len) && (!IS_CAUSAL || gn <= my_q_pos) &&
                         (window_left < 0 || gn > my_q_pos - window_left);
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
        const VT* v_row = &s_v[col_mine][d0];
#pragma unroll
        for (int i = 0; i < COLS; ++i) o_acc[i] = fmaf(p_mine, to_float<VT>(v_row[i]), o_acc[i]);
      }
      if (p_part != 0.f) {
        const VT* v_row = &s_v[col_part][d0];
#pragma unroll
        for (int i = 0; i < COLS; ++i) o_acc[i] = fmaf(p_part, to_float<VT>(v_row[i]), o_acc[i]);
      }
    }
    __syncthreads();
  }

  if (my_m < m_len) {
    const float inv_l = (l_i > 0.f) ? 1.f / l_i : 0.f;
    VT* out_row = out + bh * (int64_t)m_len * D + (int64_t)my_m * D;
#pragma unroll
    for (int i = 0; i < COLS; i += 2)
      pair_store(out_row + half_id * COLS + i, o_acc[i] * inv_l, o_acc[i + 1] * inv_l);
    if (lse != nullptr && half_id == 0)
      lse[bh * (int64_t)m_len + my_m] =
          (l_i > 0.f) ? (m_i * 0.6931471805599453f + logf(l_i)) : -INFINITY;
  }
}

}  // namespace fni8
