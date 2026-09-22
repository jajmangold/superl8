// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Track-2 (issue #42): Lightning (MiniMax un-gated linear) attention — fp32
// naive sequential CUDA kernel.
//
// v1 (this file): the naive SEQUENTIAL fp32 recurrence, ported to a single
// CUDA block per (batch, head) — the on-device analogue of
// `tests/reference_lightning.py::lightning_attn_oracle`. The int8 dp4a variant
// (next PR) is validated against this kernel before it is trusted.
//
// Recurrence (standard linear attention, no k-norm, no delta rule):
//   S_t = S_{t-1} + v_t (x) k_t    (x = outer product)
//   o_t = S_t @ q_t
//
// One block per (b, h); the block loops over T sequentially. State S [Dv, Dk]
// lives in dynamic shared memory, ROW-MAJOR, one row per lane index v (v < Dv):
// lane v owns S[v, :] exclusively for the whole kernel.
//
// Deliberately NOT the fast path: runtime (not compile-time) Dv/Dk, no dp4a,
// no tiling. v1's only job is to be an unambiguous, trustworthy on-device
// ground truth for v2+ (int8) to be measured against.
#pragma once

#include <cstdint>

namespace fni8 {

constexpr int LIGHTNING_V1_THREADS = 128;   // covers Dv, Dk <= 128

__global__ void lightning_attn_fwd_kernel(
    const float* __restrict__ q,           // [B,H,T,Dk]
    const float* __restrict__ k,           // [B,H,T,Dk]
    const float* __restrict__ v,           // [B,H,T,Dv]
    const float* __restrict__ init_state,  // [B,H,Dv,Dk] or nullptr (zero init)
    float* __restrict__ out,               // [B,H,T,Dv]
    float* __restrict__ final_state,       // [B,H,Dv,Dk]
    int H, int T, int Dk, int Dv) {
  extern __shared__ float smem[];
  float* state = smem;               // Dv*Dk
  float* buf_k = state + Dv * Dk;    // Dk
  float* buf_q = buf_k + Dk;         // Dk
  float* buf_v = buf_q + Dk;         // Dv

  const int bh = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t state_off = (int64_t)bh * Dv * Dk;

  for (int i = tid; i < Dv * Dk; i += blockDim.x)
    state[i] = init_state ? init_state[state_off + i] : 0.f;
  __syncthreads();

  const float* q_bh = q + (int64_t)bh * T * Dk;
  const float* k_bh = k + (int64_t)bh * T * Dk;
  const float* v_bh = v + (int64_t)bh * T * Dv;
  float* out_bh = out + (int64_t)bh * T * Dv;

  for (int t = 0; t < T; ++t) {
    if (tid < Dk) {
      buf_k[tid] = k_bh[t * Dk + tid];
      buf_q[tid] = q_bh[t * Dk + tid];
    }
    if (tid < Dv) buf_v[tid] = v_bh[t * Dv + tid];
    __syncthreads();

    // State update: S += v_t (x) k_t
    if (tid < Dv) {
      float* srow = state + tid * Dk;
      const float vt = buf_v[tid];
      for (int kk = 0; kk < Dk; ++kk)
        srow[kk] += vt * buf_k[kk];
    }
    __syncthreads();

    // Output: o_t = S_t @ q_t
    if (tid < Dv) {
      float* srow = state + tid * Dk;
      float ov = 0.f;
#pragma unroll 4
      for (int kk = 0; kk < Dk; ++kk)
        ov += srow[kk] * buf_q[kk];
      out_bh[t * Dv + tid] = ov;
    }
    __syncthreads();
  }

  for (int i = tid; i < Dv * Dk; i += blockDim.x)
    final_state[state_off + i] = state[i];
}

// ============================================================================
// v2 (issue #42): int8 dp4a chunked Lightning attention.
//
// Same Lightning recurrence (ungated linear attention — no α/β gates, no L2
// key-norm, no delta rule) as the v1 sequential kernel, but the intra-chunk
// k·q dot products are computed with the __dp4a int8×4 CUDA-core intrinsic
// (sm_70 has no int8 tensor cores; dp4a is the primitive, per AGENTS.md).
//
// Keys and queries are symmetric-RTN quantized per-row to int8 (scale =
// amax/127, fp32), zero-padded to Dkp = round_up(Dk, 4) so dp4a consumes
// whole int32 words. Dequant is a single multiply by the two row scales.
//
// Recurrence within a chunk [0, L):
//   o_t = S_prev @ q_t  +  Σ_{j≤t} v_j * (k_j · q_t)    (k_j·q_t via dp4a)
// State update: S_new = S_prev + Σ_j v_j ⊗ k_j           (stays fp32)
//
// The numerically load-bearing pieces stay fp32: the state S, the values V,
// the output base S@q_t, and the state update v⊗k. Only the pure k·q dot
// products go through int8 dp4a.
//
// Validated against lightning_attn_oracle (fp32 sequential reference) at int8
// SQNR/cos tolerance (NOT allclose): cos ≥ 0.999, rel-L1 ≤ 0.02, SQNR ≥ 30 dB.
// ============================================================================

constexpr int LIGHTNING_V2_THREADS = 128;
constexpr int LIGHTNING_V2_CHUNK_SIZE = 64;

// Pack four signed int8-range values into one dp4a int32 word (little-endian).
__device__ __forceinline__ int32_t ltn_pack4(int a, int b, int c, int d) {
  return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// int8×int8 dot over Dkp (multiple of 4) elements via dp4a → int32 accumulate.
__device__ __forceinline__ int ltn_i8dot(const int8_t* __restrict__ a,
                                          const int8_t* __restrict__ b, int Dkp) {
  int acc = 0;
  for (int w = 0; w < Dkp; w += 4) {
    const int32_t pa = ltn_pack4(a[w], a[w + 1], a[w + 2], a[w + 3]);
    const int32_t pb = ltn_pack4(b[w], b[w + 1], b[w + 2], b[w + 3]);
    acc = __dp4a(pa, pb, acc);
  }
  return acc;
}

__global__ void lightning_attn_int8_fwd_kernel(
    const float* __restrict__ q,           // [B,H,T,Dk]
    const float* __restrict__ k,           // [B,H,T,Dk]
    const float* __restrict__ v,           // [B,H,T,Dv]
    const float* __restrict__ init_state,  // [B,H,Dv,Dk] or nullptr
    float* __restrict__ out,               // [B,H,T,Dv]
    float* __restrict__ final_state,       // [B,H,Dv,Dk]
    int H, int T, int Dk, int Dv, int C, int Dkp) {
  extern __shared__ float smem[];
  float* state    = smem;                    // Dv*Dk
  float* k_chunk  = state + Dv * Dk;         // C*Dk
  float* ks       = k_chunk + C * Dk;        // C (key row scales)
  float* qs       = ks + C;                  // C (query row scales)
  float* gram_row = qs + C;                  // C (reused per-t for k·q products)
  int8_t* k_i8    = reinterpret_cast<int8_t*>(gram_row + C);    // C*Dkp
  int8_t* q_i8    = k_i8 + (int64_t)C * Dkp;                    // C*Dkp

  const int bh = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t state_off = (int64_t)bh * Dv * Dk;

  for (int i = tid; i < Dv * Dk; i += blockDim.x)
    state[i] = init_state ? init_state[state_off + i] : 0.f;
  __syncthreads();

  const float* q_bh = q + (int64_t)bh * T * Dk;
  const float* k_bh = k + (int64_t)bh * T * Dk;
  const float* v_bh = v + (int64_t)bh * T * Dv;
  float* out_bh = out + (int64_t)bh * T * Dv;

  const int num_chunks = (T + C - 1) / C;
  constexpr float LTN_Q_MAX = 127.0f;

  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    const int c_start = chunk * C;
    const int c_end   = min(c_start + C, T);
    const int L       = c_end - c_start;

    // ---- 1. Load K_chunk -------------------------------------------------
    for (int idx = tid; idx < L * Dk; idx += blockDim.x) {
      const int t = idx / Dk, d = idx % Dk;
      k_chunk[t * Dk + d] = k_bh[(c_start + t) * Dk + d];
    }
    __syncthreads();

    // ---- 1b. Quantize K rows → int8 (per-row symmetric RTN) --------------
    for (int t = 0; t < L; ++t) {
      if (tid == 0) {
        float amax = 0.f;
        for (int d = 0; d < Dk; ++d)
          amax = fmaxf(amax, fabsf(k_chunk[t * Dk + d]));
        const float s = amax / LTN_Q_MAX;
        ks[t] = (s == 0.f) ? 1.f : s;
      }
      __syncthreads();
      const float inv = ks[t];
      for (int d = tid; d < Dkp; d += blockDim.x) {
        float qv = (d < Dk) ? rintf(__fdiv_rn(k_chunk[t * Dk + d], inv)) : 0.f;
        qv = fminf(fmaxf(qv, -LTN_Q_MAX), LTN_Q_MAX);
        k_i8[t * Dkp + d] = (int8_t)qv;
      }
      __syncthreads();
    }

    // ---- 1c. Quantize Q rows → int8 (per-row symmetric RTN) --------------
    for (int t = 0; t < L; ++t) {
      const float* qt = q_bh + (c_start + t) * Dk;
      if (tid == 0) {
        float amax = 0.f;
        for (int d = 0; d < Dk; ++d)
          amax = fmaxf(amax, fabsf(qt[d]));
        const float s = amax / LTN_Q_MAX;
        qs[t] = (s == 0.f) ? 1.f : s;
      }
      __syncthreads();
      const float inv = qs[t];
      for (int d = tid; d < Dkp; d += blockDim.x) {
        float qv = (d < Dk) ? rintf(__fdiv_rn(qt[d], inv)) : 0.f;
        qv = fminf(fmaxf(qv, -LTN_Q_MAX), LTN_Q_MAX);
        q_i8[t * Dkp + d] = (int8_t)qv;
      }
      __syncthreads();
    }

    // ---- 2. Output: o_t = S@q_t + Σ_{j≤t} v_j·(k_j·q_t) ---------------
    for (int t = 0; t < L; ++t) {
      // Compute k_j·q_t for all j via dp4a (distributed across threads)
      for (int j = tid; j < L; j += blockDim.x) {
        const int dot = ltn_i8dot(&k_i8[j * Dkp], &q_i8[t * Dkp], Dkp);
        gram_row[j] = (float)dot * ks[j] * qs[t];
      }
      __syncthreads();

      if (tid < Dv) {
        const float* qt = q_bh + (c_start + t) * Dk;
        // Base: S@q_t (fp32)
        float acc = 0.f;
        {
          float* srow = state + tid * Dk;
          for (int d = 0; d < Dk; ++d)
            acc += srow[d] * qt[d];
        }
        // Add intra-chunk contributions: Σ_{j≤t} v_j·(k_j·q_t)
        for (int j = 0; j <= t; ++j)
          acc += v_bh[(c_start + j) * Dv + tid] * gram_row[j];
        out_bh[(c_start + t) * Dv + tid] = acc;
      }
      __syncthreads();
    }

    // ---- 3. State update: S += Σ_j v_j ⊗ k_j  (pure fp32) ---------------
    if (tid < Dv) {
      float* srow = state + tid * Dk;
      for (int j = 0; j < L; ++j) {
        const float vj = v_bh[(c_start + j) * Dv + tid];
        for (int d = 0; d < Dk; ++d)
          srow[d] += vj * k_chunk[j * Dk + d];
      }
    }
    __syncthreads();
  }

  for (int i = tid; i < Dv * Dk; i += blockDim.x)
    final_state[state_off + i] = state[i];
}

}  // namespace fni8
