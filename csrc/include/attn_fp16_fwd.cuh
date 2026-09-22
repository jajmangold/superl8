// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// fp16/bf16 half2 QK^T FlashAttention-2 forward for sm_70 (PREFILL).
//
// The full-precision sibling of attn_int8_fwd.cuh: IDENTICAL tiling (BLOCK_M=
// BLOCK_N=64, 128 threads = 4 warps, lane-pair-per-row, online exp2 softmax,
// __shfl_xor pair reductions, causal/window/GQA, dynamic-smem D=256 path,
// pair_store/to_float epilogue). The ONLY compute change vs the int8 kernel:
// the QK^T inner loop is a half2 CUDA-core dot (fp16: __hmul2 on the healthy
// Volta half2 pipe -> ~27 TFLOP/s, a SEPARATE pipe from the firmware-dead
// tensor cores; NEVER wmma/HMMA) with fp32 accumulation, instead of int8
// __dp4a. bf16 QK dots upcast to fp32 (sm_70 has no bf162 arithmetic). No quant
// prologue, no per-row scales: Q/K/V are fp16 or bf16 and the softmax_scale *
// log2(e) fold is a single scalar. Softmax/LSE stay fp32 (load-bearing).
//
// Why it exists: torch SDPA has no flash/mem-efficient backend on Volta, so it
// materializes the O(N^2) score matrix and OOMs on ~10k-17k-token video-DiT
// attention (LTX/Wan). This streams the K/V tiles -> O(N) memory.
//
// Thread map (same as int8): lane pair (tid, tid^1) owns one S row; each lane
// computes 32 of the 64 S columns and owns the row's D/2 output slice. Row
// max/sum finish with __shfl_xor_sync(., 1); in PV a lane gets the partner's 32
// P values via one __shfl_xor.
//
// Smem: Q + K + V tiles, all in the (single) input dtype T (== v's dtype). At
// D=128 that is 48 KB (static cap); D=256 -> 96 KB via the dynamic-smem kernel.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "compute_dtype.cuh"

namespace fni8 {

#ifndef FNI8_FP16_BLOCK_M
constexpr int FP16_BLOCK_M = 64;
constexpr int FP16_BLOCK_N = 64;
constexpr int FP16_THREADS = 128;
#endif

// half2/bf162 vector type for the packed QK dot.
template <typename T>
struct half_vec2;
template <>
struct half_vec2<__half> {
  using type = __half2;
};
template <>
struct half_vec2<__nv_bfloat16> {
  using type = __nv_bfloat162;
};

// Packed pair multiply -> fp32 lanes. fp16 uses __hmul2 (the healthy Volta
// half2 CUDA-core pipe), then converts the fp16 products to fp32 for
// accumulation. bf16 has no sm_70 bf162 multiply, so it upcasts to fp32 first.
__device__ __forceinline__ float2 fp16_mul2(__half2 a, __half2 b) {
  return __half22float2(__hmul2(a, b));
}
__device__ __forceinline__ float2 fp16_mul2(__nv_bfloat162 a, __nv_bfloat162 b) {
  const float2 af = __bfloat1622float2(a);
  const float2 bf = __bfloat1622float2(b);
  return make_float2(af.x * bf.x, af.y * bf.y);
}

// Single-row Q.K dot over D elements (D/2 packed pairs), fp32-accumulated.
template <int D2, typename T>
__device__ __forceinline__ float fp16_qk_dot(const T* __restrict__ qrow,
                                             const T* __restrict__ krow) {
  using V2 = typename half_vec2<T>::type;
  const V2* q2 = reinterpret_cast<const V2*>(qrow);
  const V2* k2 = reinterpret_cast<const V2*>(krow);
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < D2; ++c) {
    const float2 p = fp16_mul2(q2[c], k2[c]);
    acc += p.x + p.y;
  }
  return acc;
}

// Grid: (ceil(M / BLOCK_M), B * H). Layout: [B, H, M, D] contiguous. `T` is the
// single input/output dtype (__half or __nv_bfloat16); out shares it (the torch
// wrapper allocates out with v.options()). fp16 overflows (max 65504) on
// bf16-native models' activations -> use bf16 there.
template <int D, bool IS_CAUSAL, typename T>
__global__ void __launch_bounds__(FP16_THREADS)
attn_fp16_fwd_kernel(const T* __restrict__ q,    // [B,H,M,D]
                     const T* __restrict__ k,    // [B,H,N,D]
                     const T* __restrict__ v,    // [B,H,N,D]
                     T* __restrict__ out,        // [B,H,M,D]
                     float* __restrict__ lse,    // [B,H,M] fp32, or nullptr
                     const T* __restrict__ mask, // additive [*,M,N] (natural log) or nullptr
                     int64_t mask_sb, int64_t mask_sh, int64_t mask_sm, int64_t mask_sn,
                     float scale,                // softmax_scale * log2(e)
                     int m_len, int n_len, int h_q, int gqa_group, int window_left) {
  constexpr int D2 = D / 2;    // packed-pair head dim
  constexpr int COLS = D / 2;  // per-lane O slice width
  constexpr unsigned FULL = 0xffffffffu;
  // Static-smem cap on sm_70 is 48 KB; the dynamic kernel handles the rest.
  static_assert((FP16_BLOCK_M + FP16_BLOCK_N) * D * (int)sizeof(T) +
                        FP16_BLOCK_N * D * (int)sizeof(T) <=
                    98304,
                "fp16 fwd smem exceeds the 96 KB Volta cap");

  __shared__ T s_q[FP16_BLOCK_M][D];
  __shared__ T s_k[FP16_BLOCK_N][D];
  __shared__ T s_v[FP16_BLOCK_N][D];

  const int tid = threadIdx.x;
  const int row = tid >> 1;      // S row owned by this lane pair (0..63)
  const int half_id = tid & 1;   // which 32-col S slice / D-half of O
  const int m_block = blockIdx.x * FP16_BLOCK_M;
  const int64_t bh = blockIdx.y;                         // over B * H_q
  // GQA/MQA: this Q head shares a K/V head. kv_bh maps into the H_kv-head K/V.
  const int64_t kv_bh = (bh / h_q) * (h_q / gqa_group) + (bh % h_q) / gqa_group;

  const T* q_bh = q + bh * (int64_t)m_len * D;
  const T* k_bh = k + kv_bh * (int64_t)n_len * D;
  const T* v_bh = v + kv_bh * (int64_t)n_len * D;

  // ---- load Q tile (zero-pad ragged tail) ----
  for (int idx = tid; idx < FP16_BLOCK_M * D; idx += FP16_THREADS) {
    const int r = idx / D, c = idx % D;
    const int gm = m_block + r;
    s_q[r][c] = (gm < m_len) ? q_bh[(int64_t)gm * D + c] : zero_val<T>();
  }
  __syncthreads();

  const int my_m = m_block + row;
  const int causal_diag = IS_CAUSAL ? n_len - m_len : 0;
  const int my_q_pos = IS_CAUSAL ? my_m + causal_diag : my_m;
  // Additive mask base for this lane's row (natural-log bias, indexed by Q head).
  const bool has_mask = (mask != nullptr) && (my_m < m_len);
  const int64_t my_row_mask_base =
      has_mask ? (bh / h_q) * mask_sb + (bh % h_q) * mask_sh + (int64_t)my_m * mask_sm : 0;

  float o_acc[COLS];
#pragma unroll
  for (int i = 0; i < COLS; ++i) o_acc[i] = 0.f;
  float m_i = -INFINITY;  // running row max (exp2 domain)
  float l_i = 0.f;        // running row sum

  const int n_end = IS_CAUSAL
                        ? max(0, min(n_len, m_block + FP16_BLOCK_M + causal_diag))
                        : n_len;

  for (int n_block = 0; n_block < n_end; n_block += FP16_BLOCK_N) {
    // ---- stage K and V tiles ----
    for (int idx = tid; idx < FP16_BLOCK_N * D; idx += FP16_THREADS) {
      const int r = idx / D, c = idx % D;
      const int gn = n_block + r;
      s_k[r][c] = (gn < n_len) ? k_bh[(int64_t)gn * D + c] : zero_val<T>();
      s_v[r][c] = (gn < n_len) ? v_bh[(int64_t)gn * D + c] : zero_val<T>();
    }
    __syncthreads();

    // ---- S = Q.K^T via half2 CUDA-core dot; scale to exp2-ready fp32 logits ----
    float s_row[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int col = half_id * 32 + j;
      const float acc = fp16_qk_dot<D2, T>(&s_q[row][0], &s_k[col][0]);
      const int gn = n_block + col;
      const bool valid = (gn < n_len) && (!IS_CAUSAL || gn <= my_q_pos) &&
                         (window_left < 0 || gn > my_q_pos - window_left);
      float logit = valid ? acc * scale : -INFINITY;
      // additive mask is a natural-log bias -> fold log2(e) to match the exp2 domain
      if (has_mask && valid)
        logit += to_float<T>(mask[my_row_mask_base + (int64_t)gn * mask_sn]) * 1.4426950408889634f;
      s_row[j] = logit;
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
        const T* v_row = &s_v[col_mine][d0];
#pragma unroll
        for (int i = 0; i < COLS; ++i) o_acc[i] = fmaf(p_mine, to_float<T>(v_row[i]), o_acc[i]);
      }
      if (p_part != 0.f) {
        const T* v_row = &s_v[col_part][d0];
#pragma unroll
        for (int i = 0; i < COLS; ++i) o_acc[i] = fmaf(p_part, to_float<T>(v_row[i]), o_acc[i]);
      }
    }
    __syncthreads();  // tiles reused next iteration
  }

  // ---- epilogue: O /= l, write fp16/bf16 (each lane writes its own D-half) ----
  if (my_m < m_len) {
    const float inv_l = (l_i > 0.f) ? 1.f / l_i : 0.f;  // N==0 / fully-masked -> 0
    T* out_row = out + bh * (int64_t)m_len * D + (int64_t)my_m * D;
#pragma unroll
    for (int i = 0; i < COLS; i += 2)
      pair_store(out_row + half_id * COLS + i, o_acc[i] * inv_l, o_acc[i + 1] * inv_l);
    // logsumexp in NATURAL log (kernel logits carry a log2(e) fold -> * ln2).
    if (lse != nullptr && half_id == 0)
      lse[bh * (int64_t)m_len + my_m] =
          (l_i > 0.f) ? (m_i * 0.6931471805599453f + logf(l_i)) : -INFINITY;
  }
}

// ---------------------------------------------------------------------------
// DYNAMIC-SMEM variant for head dims whose tiles exceed the 48 KB static cap
// (D=256: s_q+s_k+s_v ~= 96 KB). Identical math to attn_fp16_fwd_kernel; only
// the shared tiles move to a dynamic `extern __shared__` buffer opted in via
// cudaFuncSetAttribute.
// ---------------------------------------------------------------------------
template <int D, typename T>
constexpr int fp16_fwd_dyn_smem_bytes() {
  return (FP16_BLOCK_M + FP16_BLOCK_N) * D * (int)sizeof(T)   // s_q + s_k
         + FP16_BLOCK_N * D * (int)sizeof(T);                 // s_v
}

template <int D, bool IS_CAUSAL, typename T>
__global__ void __launch_bounds__(FP16_THREADS)
attn_fp16_fwd_dyn_kernel(const T* __restrict__ q, const T* __restrict__ k,
                         const T* __restrict__ v, T* __restrict__ out,
                         float* __restrict__ lse, const T* __restrict__ mask,
                         int64_t mask_sb, int64_t mask_sh, int64_t mask_sm, int64_t mask_sn,
                         float scale, int m_len, int n_len,
                         int h_q, int gqa_group, int window_left) {
  constexpr int D2 = D / 2;
  constexpr int COLS = D / 2;
  constexpr unsigned FULL = 0xffffffffu;

  extern __shared__ char fp16_smem_dyn[];
  auto s_q = reinterpret_cast<T(*)[D]>(fp16_smem_dyn);
  auto s_k = reinterpret_cast<T(*)[D]>(fp16_smem_dyn + FP16_BLOCK_M * D * sizeof(T));
  auto s_v = reinterpret_cast<T(*)[D]>(
      fp16_smem_dyn + (FP16_BLOCK_M + FP16_BLOCK_N) * D * sizeof(T));

  const int tid = threadIdx.x;
  const int row = tid >> 1;
  const int half_id = tid & 1;
  const int m_block = blockIdx.x * FP16_BLOCK_M;
  const int64_t bh = blockIdx.y;
  const int64_t kv_bh = (bh / h_q) * (h_q / gqa_group) + (bh % h_q) / gqa_group;

  const T* q_bh = q + bh * (int64_t)m_len * D;
  const T* k_bh = k + kv_bh * (int64_t)n_len * D;
  const T* v_bh = v + kv_bh * (int64_t)n_len * D;

  for (int idx = tid; idx < FP16_BLOCK_M * D; idx += FP16_THREADS) {
    const int r = idx / D, c = idx % D;
    const int gm = m_block + r;
    s_q[r][c] = (gm < m_len) ? q_bh[(int64_t)gm * D + c] : zero_val<T>();
  }
  __syncthreads();

  const int my_m = m_block + row;
  const int causal_diag = IS_CAUSAL ? n_len - m_len : 0;
  const int my_q_pos = IS_CAUSAL ? my_m + causal_diag : my_m;
  const bool has_mask = (mask != nullptr) && (my_m < m_len);
  const int64_t my_row_mask_base =
      has_mask ? (bh / h_q) * mask_sb + (bh % h_q) * mask_sh + (int64_t)my_m * mask_sm : 0;

  float o_acc[COLS];
#pragma unroll
  for (int i = 0; i < COLS; ++i) o_acc[i] = 0.f;
  float m_i = -INFINITY, l_i = 0.f;

  const int n_end = IS_CAUSAL
                        ? max(0, min(n_len, m_block + FP16_BLOCK_M + causal_diag))
                        : n_len;

  for (int n_block = 0; n_block < n_end; n_block += FP16_BLOCK_N) {
    for (int idx = tid; idx < FP16_BLOCK_N * D; idx += FP16_THREADS) {
      const int r = idx / D, c = idx % D;
      const int gn = n_block + r;
      s_k[r][c] = (gn < n_len) ? k_bh[(int64_t)gn * D + c] : zero_val<T>();
      s_v[r][c] = (gn < n_len) ? v_bh[(int64_t)gn * D + c] : zero_val<T>();
    }
    __syncthreads();

    float s_row[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int col = half_id * 32 + j;
      const float acc = fp16_qk_dot<D2, T>(&s_q[row][0], &s_k[col][0]);
      const int gn = n_block + col;
      const bool valid = (gn < n_len) && (!IS_CAUSAL || gn <= my_q_pos) &&
                         (window_left < 0 || gn > my_q_pos - window_left);
      float logit = valid ? acc * scale : -INFINITY;
      // additive mask is a natural-log bias -> fold log2(e) to match the exp2 domain
      if (has_mask && valid)
        logit += to_float<T>(mask[my_row_mask_base + (int64_t)gn * mask_sn]) * 1.4426950408889634f;
      s_row[j] = logit;
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
        const T* v_row = &s_v[col_mine][d0];
#pragma unroll
        for (int i = 0; i < COLS; ++i) o_acc[i] = fmaf(p_mine, to_float<T>(v_row[i]), o_acc[i]);
      }
      if (p_part != 0.f) {
        const T* v_row = &s_v[col_part][d0];
#pragma unroll
        for (int i = 0; i < COLS; ++i) o_acc[i] = fmaf(p_part, to_float<T>(v_row[i]), o_acc[i]);
      }
    }
    __syncthreads();
  }

  if (my_m < m_len) {
    const float inv_l = (l_i > 0.f) ? 1.f / l_i : 0.f;
    T* out_row = out + bh * (int64_t)m_len * D + (int64_t)my_m * D;
#pragma unroll
    for (int i = 0; i < COLS; i += 2)
      pair_store(out_row + half_id * COLS + i, o_acc[i] * inv_l, o_acc[i + 1] * inv_l);
    if (lse != nullptr && half_id == 0)
      lse[bh * (int64_t)m_len + my_m] =
          (l_i > 0.f) ? (m_i * 0.6931471805599453f + logf(l_i)) : -INFINITY;
  }
}

}  // namespace fni8
