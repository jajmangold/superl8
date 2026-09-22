// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Track-2 (issue #6): int8 dp4a Gated-DeltaNet chunked kernel.
//
// v1 (this file): the naive SEQUENTIAL fp32 recurrence, ported to a single CUDA
// block per (batch, head) — the on-device analogue of
// `tests/reference_linear_attn.py::gated_delta_rule_oracle`. Every later
// parallel/chunked form (v2 ungated chunked, v3 gated, v4 int8 dp4a) is
// validated against this kernel (which is itself validated against the fp32
// Python oracle) before it is trusted — same staging discipline the Python
// oracle used against the delta-rule algebraic invariants.
//
// One block per (b, h); the block loops over T sequentially (T is an inherently
// serial recurrence at this stage — v2's WY/UT parallel form is what removes
// that serialization within a chunk). State S [Dv, Dk] lives in dynamic shared
// memory, ROW-MAJOR, one row per lane index `v` (v < Dv): lane v owns S[v, :]
// exclusively for the whole kernel, so the per-step state update and output
// read need no cross-lane communication beyond the shared k_t/q_t/v_t/alpha/
// beta broadcasts already staged in shared memory. The only cross-lane
// dependency is the k_t L2-norm reduction, done by lane 0 (Dk is small; not
// worth a parallel reduction in this non-perf-critical baseline).
//
// Recurrence (must match gated_delta_rule_oracle exactly, fp32 throughout):
//   k_t <- k_t / max(||k_t||_2, 1e-12)
//   sk_t = alpha_t * (S_{t-1} @ k_t)
//   write_t = beta_t * (v_t - sk_t)
//   S_t = alpha_t * S_{t-1} + write_t (x) k_t        ((x) = outer product)
//   o_t = S_t @ q_t
//
// Deliberately NOT the fast path: runtime (not compile-time) Dv/Dk, no dp4a,
// no tiling. v1's only job is to be an unambiguous, trustworthy on-device
// ground truth for v2+ to be measured against.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace fni8 {

// ============================================================================
// half2 helper: fp32 elementwise dot product recast onto __hfma2.
// Operands a/b are pointers into fp32 shared memory; pairs are converted on
// the fly to half2, multiplied via __hfma2, and the final half2 partial sum
// is reduced to fp32.  For Dk ≤ 128 the half2 accumulator is safe (at most
// 64 half2 products summing to ~64, well within fp16 range).
// ============================================================================
__device__ __forceinline__ float fni8_h2_dot(const float* a, const float* b, int Dk) {
  half2 hsum = __float2half2_rn(0.0f);
  for (int d = 0; d + 1 < Dk; d += 2) {
    half2 av = __floats2half2_rn(a[d], a[d + 1]);
    half2 bv = __floats2half2_rn(b[d], b[d + 1]);
    hsum = __hfma2(av, bv, hsum);
  }
  float acc = __half2float(hsum.x) + __half2float(hsum.y);
  if (Dk & 1)
    acc += a[Dk - 1] * b[Dk - 1];
  return acc;
}

constexpr int DELTANET_V1_THREADS = 128;   // covers Dv, Dk <= 128 (v1's supported range)

__global__ void deltanet_recurrent_fwd_kernel(
    const float* __restrict__ q,           // [B,H,T,Dk]
    const float* __restrict__ k,           // [B,H,T,Dk]
    const float* __restrict__ v,           // [B,H,T,Dv]
    const float* __restrict__ alpha,       // [B,H,T]
    const float* __restrict__ beta,        // [B,H,T]
    const float* __restrict__ init_state,  // [B,H,Dv,Dk] or nullptr (zero init)
    float* __restrict__ out,               // [B,H,T,Dv]
    float* __restrict__ final_state,       // [B,H,Dv,Dk]
    int H, int T, int Dk, int Dv) {
  extern __shared__ float smem[];
  float* state = smem;               // Dv*Dk
  float* buf_k = state + Dv * Dk;    // Dk (raw, then L2-normalized in place)
  float* buf_q = buf_k + Dk;         // Dk
  float* buf_v = buf_q + Dk;         // Dv
  float* sq = buf_v + Dv;            // Dk scratch (k_t element-wise squares)
  __shared__ float s_alpha, s_beta, s_sumsq, s_inv_norm;

  const int bh = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t state_off = (int64_t)bh * Dv * Dk;

  for (int i = tid; i < Dv * Dk; i += blockDim.x)
    state[i] = init_state ? init_state[state_off + i] : 0.f;
  __syncthreads();

  const float* q_bh = q + (int64_t)bh * T * Dk;
  const float* k_bh = k + (int64_t)bh * T * Dk;
  const float* v_bh = v + (int64_t)bh * T * Dv;
  const float* a_bh = alpha + (int64_t)bh * T;
  const float* b_bh = beta + (int64_t)bh * T;
  float* out_bh = out + (int64_t)bh * T * Dv;

  for (int t = 0; t < T; ++t) {
    if (tid < Dk) {
      const float kv = k_bh[t * Dk + tid];
      buf_k[tid] = kv;
      sq[tid] = kv * kv;
      buf_q[tid] = q_bh[t * Dk + tid];
    }
    if (tid < Dv) buf_v[tid] = v_bh[t * Dv + tid];
    if (tid == 0) {
      float sum = 0.f;
      for (int i = 0; i < Dk; ++i) sum += sq[i];
      s_sumsq = sum;
      s_alpha = a_bh[t];
      s_beta = b_bh[t];
    }
    __syncthreads();
    // clamp on the NORM (not sum-of-squares) at 1e-12, matching
    // `kf.norm(...).clamp_min(1e-12)` in the fp32 oracle: norm^2 = 1e-24.
    if (tid == 0) s_inv_norm = rsqrtf(fmaxf(s_sumsq, 1e-24f));
    __syncthreads();
    if (tid < Dk) buf_k[tid] *= s_inv_norm;
    __syncthreads();

    if (tid < Dv) {
      float* srow = state + tid * Dk;
      float sk = 0.f;
#pragma unroll 4
      for (int kk = 0; kk < Dk; ++kk) sk += srow[kk] * buf_k[kk];
      sk *= s_alpha;
      const float write = s_beta * (buf_v[tid] - sk);
      float ov = 0.f;
#pragma unroll 4
      for (int kk = 0; kk < Dk; ++kk) {
        const float ns = s_alpha * srow[kk] + write * buf_k[kk];
        srow[kk] = ns;
        ov += ns * buf_q[kk];
      }
      out_bh[t * Dv + tid] = ov;
    }
    __syncthreads();
  }

  for (int i = tid; i < Dv * Dk; i += blockDim.x) final_state[state_off + i] = state[i];
}

// ============================================================================
// v2 (issue #40): ungated chunked WY/UT parallel form (fp32, C ≤ 64).
//
// One block per (b, h); the sequence is split into chunks of size C (computed
// by the launcher to fit Volta's 96 KB shared-mem cap).  Within each chunk the
// kernel uses the WY decomposition to parallelise across Dv and compute the
// intra-chunk interactions via forward substitution (the only serial part,
// length C ≤ 64).  The algorithm is:
//
//   k_t <- k_t / ||k_t||_2                                     (L2-norm)
//   for each chunk of length L ≤ C:
//     1.  load K_chunk, L2-normalise in-place
//     2.  W  = V_chunk - S_0 @ K_chunk^T        (initial residuals, Dv×L)
//     3.  solve  R = W - tril(K_chunk @ K_chunk^T) @ R   by forward substitution
//     4.  o_t = S_0 @ q_t + Σ_{j≤t} (k_j @ q_t) · R[:,j]   (t = 0…L-1)
//     5.  S_0 = S_0 + R @ K_chunk
//
// This is mathematically identical to the sequential recurrence with α=1, β=1.
// ============================================================================
constexpr int DELTANET_V2_THREADS = 128;
constexpr int DELTANET_V2_CHUNK_SIZE = 64;   // max C the launcher will pass
constexpr int DELTANET_V2_MAX_SMEM = 98304;  // Volta opt-in cap

__global__ void deltanet_chunk_fwd_kernel(
    const float* __restrict__ q,           // [B,H,T,Dk]
    const float* __restrict__ k,           // [B,H,T,Dk]
    const float* __restrict__ v,           // [B,H,T,Dv]
    const float* __restrict__ init_state,  // [B,H,Dv,Dk] or nullptr
    float* __restrict__ out,               // [B,H,T,Dv]
    float* __restrict__ final_state,       // [B,H,Dv,Dk]
    int H, int T, int Dk, int Dv, int C) {  // C = computed chunk size
  extern __shared__ float smem[];
  float* state    = smem;                  // Dv*Dk
  float* k_chunk  = state + Dv * Dk;       // C*Dk
  float* r_buf    = k_chunk + C * Dk;      // Dv*C  (W then R)
  float* gram_row = r_buf + Dv * C;        // C     (scratch for one Gram / QK row)

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
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    const int c_start = chunk * C;
    const int c_end   = min(c_start + C, T);
    const int L       = c_end - c_start;   // actual chunk length (L ≤ C)

    // ---- 1.  Load K_chunk + L2-normalise each key -------------------------
    for (int idx = tid; idx < L * Dk; idx += blockDim.x) {
      const int t = idx / Dk, d = idx % Dk;
      k_chunk[t * Dk + d] = k_bh[(c_start + t) * Dk + d];
    }
    __syncthreads();
    for (int t = 0; t < L; ++t) {
      if (tid == 0) {
        float ssq = 0.f;
        for (int d = 0; d < Dk; ++d) {
          const float kv = k_chunk[t * Dk + d];
          ssq += kv * kv;
        }
        gram_row[t] = rsqrtf(fmaxf(ssq, 1e-24f));    // reuse gram_row for inv_norm
      }
      __syncthreads();
      const float inv = gram_row[t];
      for (int d = tid; d < Dk; d += blockDim.x)
        k_chunk[t * Dk + d] *= inv;
      __syncthreads();
    }

    // ---- 2.  W = V_chunk - S_0 @ K_chunk^T   (-> r_buf) ------------------
    if (tid < Dv) {
      float* srow = state + tid * Dk;
      for (int t = 0; t < L; ++t) {
        float dot = 0.f;
        for (int d = 0; d < Dk; ++d)
          dot += srow[d] * k_chunk[t * Dk + d];
        r_buf[tid * C + t] = v_bh[(c_start + t) * Dv + tid] - dot;
      }
    }
    __syncthreads();

    // ---- 3.  Forward substitution: solve R from W -------------------------
    //   R[t] = W[t] - Σ_{j<t} (k_t · k_j) R[j]
    for (int t = 0; t < L; ++t) {
      for (int j = tid; j < t; j += blockDim.x) {
        float dot = 0.f;
        for (int d = 0; d < Dk; ++d)
          dot += k_chunk[t * Dk + d] * k_chunk[j * Dk + d];
        gram_row[j] = dot;
      }
      __syncthreads();
      if (tid < Dv) {
        float correction = 0.f;
        for (int j = 0; j < t; ++j)
          correction += gram_row[j] * r_buf[tid * C + j];
        r_buf[tid * C + t] -= correction;
      }
      __syncthreads();
    }

    // ---- 4.  Compute outputs  o_t = S_0@q_t + Σ_{j≤t}(k_j·q_t)R[:,j] -----
    for (int t = 0; t < L; ++t) {
      // compute QK row: q_t · k_j  for all j, store in gram_row
      const float* qt = q_bh + (c_start + t) * Dk;
      for (int j = tid; j < L; j += blockDim.x) {
        float dot = 0.f;
        for (int d = 0; d < Dk; ++d)
          dot += qt[d] * k_chunk[j * Dk + d];
        gram_row[j] = dot;
      }
      __syncthreads();
      if (tid < Dv) {
        float base = 0.f;
        {
          float* srow = state + tid * Dk;
          for (int d = 0; d < Dk; ++d)
            base += srow[d] * qt[d];
        }
        float acc = base;
        for (int j = 0; j <= t; ++j)
          acc += gram_row[j] * r_buf[tid * C + j];
        out_bh[(c_start + t) * Dv + tid] = acc;
      }
      __syncthreads();
    }

    // ---- 5.  Update state  S_0 = S_0 + R @ K_chunk -----------------------
    if (tid < Dv) {
      float* srow = state + tid * Dk;
      for (int t = 0; t < L; ++t) {
        const float rit = r_buf[tid * C + t];
        for (int d = 0; d < Dk; ++d)
          srow[d] += rit * k_chunk[t * Dk + d];
      }
    }
    __syncthreads();
  }

  for (int i = tid; i < Dv * Dk; i += blockDim.x)
    final_state[state_off + i] = state[i];
}

// ============================================================================
// v3 (issue #56): gated chunked WY/UT parallel form with log-space γ-cumprod.
//
// Extends v2's ungated chunked kernel with the α (decay) / β (write-strength)
// gates.  The per-step decay factors α_t compose multiplicatively within a
// chunk; we track the cumulative product γ[i] = ∏_{j=0}^{i-1} α_j in log-space
// to avoid fp32 underflow over long chunks.
//
// Algorithm per chunk of length L ≤ C:
//   1.  Load K_chunk, L2-normalise.
//   2.  Load alpha, beta.
//   3.  Compute log γ[i] = Σ_{j=0}^{i-1} log α_j  (i=0..L),  γ[i]=exp(log γ[i]).
//   4.  w[i] = v[i] − γ[i]·S₀@k[i]       (initial residuals, Dv×L).
//   5.  Forward-substitute r[i] from w[i] with (γ[i]/γ[j+1])·β[j]·(kⱼ·kᵢ) coeffs.
//   6.  o[i] = γ[i+1]·S₀@q[i] + Σ_{j≤i} (γ[i+1]/γ[j+1])·β[j]·(kⱼ·q[i])·r[j].
//   7.  S₀ ← γ[L]·S₀ + Σ_{j<L} (γ[L]/γ[j+1])·β[j]·r[j]⊗k[j].
//
// Mathematically identical to the sequential gated recurrence; matches
// gated_delta_rule_oracle at fp32 reassociation tolerance.
// ============================================================================
// v3 launch geometry (issue: GDN prefill occupancy).
//
// The block is B*H wide and each block needs ~96 KB of shared memory, so
// exactly ONE block is resident per SM -- the only lever on occupancy is the
// block itself.  At 128 threads that is 4 warps for 4 warp schedulers, i.e. one
// warp per scheduler and no latency hiding at all.  v3 therefore runs 512
// threads and has G = blockDim/Dv threads cooperate on each Dv row:
//
//     G   = largest power of two <= 32 with G*Dv <= blockDim
//     row = tid / G   (the Dv row, or the Gram column, this group owns)
//     gl  = tid % G   (lane within the group; groups are consecutive tids, so
//                      a group never straddles a warp)
//
// Every per-row dot product splits its Dk accumulation across the G lanes and
// finishes with a __shfl_xor_sync butterfly over G lanes.  The step-7 state
// update needs no reduction at all -- each lane owns its own slice of d.
//
// Shared memory is padded (see deltanet_v3_row_stride): with an unpadded row
// stride of Dk (128, a multiple of the 32 4-byte banks) every row of `state`
// starts in bank 0, so the Dv rows of the S@k / S@q dots collide 32 ways.
// The padding costs ~2 KB of the 96 KB budget, which pulls the dynamic chunk
// size at Dk=Dv=128 from C=31 to C=29 (18 chunks per 512 tokens instead of
// 17).  Steps 5 and 6 scale with C, so that is a real cost, and it is paid
// for: measured on the fleet's V100 lane at B=1 H=48 T=512 Dk=Dv=128,
// 10.92 ms/call unpadded vs 6.06 ms/call padded (1.80x for the stride alone).
//
// Numerics are unchanged in kind (AGENTS.md): state S, the log-space gamma
// cumprod, the L2 norm, the alpha/beta gates, V, the residuals r/W and the
// step-7 state update all stay fp32.  Only the *association* of the fp32 dot
// products changes (strided partial sums + butterfly instead of a single
// sequential sweep).
constexpr int DELTANET_V3_THREADS = 512;
constexpr int DELTANET_V3_CHUNK_SIZE = 64;
constexpr int DELTANET_V3_MAX_SMEM = 98304;

// G threads cooperate on one Dv row: the largest power of two <= 32 (one warp)
// with G*Dv <= blockDim.  Falls back to G=1 (one thread per row, the previous
// behaviour) when Dv >= blockDim.
__host__ __device__ __forceinline__ int deltanet_v3_group_width(int nthreads, int Dv) {
  int g = 1;
  while (g < 32 && (g << 1) * Dv <= nthreads) g <<= 1;
  return g;
}

// Shared-memory row stride for `state` and `k_chunk`.  A group reads
// d = gl, gl+G, gl+2G, ... of row `row`, so thread (row, gl) touches word
// row*stride + gl + i*G; making stride == G (mod 32) spreads the 32 threads of
// a warp over all 32 banks.  Costs at most 31 floats per row.
__host__ __device__ __forceinline__ int deltanet_v3_row_stride(int Dk, int G) {
  return Dk + (((G % 32) - (Dk % 32) + 32) % 32);
}

// r_buf is indexed [row][j] with j warp-uniform, so an odd row stride is
// enough to keep the Dv rows in distinct banks.
__host__ __device__ __forceinline__ int deltanet_v3_rbuf_stride(int C) {
  return C | 1;
}

// state(Dv*SS) + k_chunk(C*SS) + r_buf(Dv*RS) + gram_row(C) + beta_chunk(C) +
// gexp(C) + log_gamma(C+1), in floats.
__host__ __device__ __forceinline__ int64_t deltanet_v3_smem_floats(
    int Dk, int Dv, int C, int nthreads) {
  const int G  = deltanet_v3_group_width(nthreads, Dv);
  const int SS = deltanet_v3_row_stride(Dk, G);
  const int RS = deltanet_v3_rbuf_stride(C);
  return (int64_t)Dv * SS + (int64_t)C * SS + (int64_t)Dv * RS + 3 * (int64_t)C + (C + 1);
}

__global__ void deltanet_gated_chunk_fwd_kernel(
    const float* __restrict__ q,           // [B,H,T,Dk]
    const float* __restrict__ k,           // [B,H,T,Dk]
    const float* __restrict__ v,           // [B,H,T,Dv]
    const float* __restrict__ alpha,       // [B,H,T]
    const float* __restrict__ beta,        // [B,H,T]
    const float* __restrict__ init_state,  // [B,H,Dv,Dk] or nullptr
    float* __restrict__ out,               // [B,H,T,Dv]
    float* __restrict__ final_state,       // [B,H,Dv,Dk]
    int H, int T, int Dk, int Dv, int C) {
  extern __shared__ float smem[];
  const int nthreads = blockDim.x;
  const int G  = deltanet_v3_group_width(nthreads, Dv);
  const int nG = nthreads / G;                 // number of cooperating groups
  const int SS = deltanet_v3_row_stride(Dk, G);
  const int RS = deltanet_v3_rbuf_stride(C);

  float* state      = smem;                 // Dv*SS
  float* k_chunk    = state + Dv * SS;      // C*SS
  float* r_buf      = k_chunk + C * SS;     // Dv*RS  (W then R)
  float* gram_row   = r_buf + Dv * RS;      // C   (Gram/QK row, then step-7 coeff)
  float* beta_chunk = gram_row + C;         // C   (write-strength gates)
  float* gexp       = beta_chunk + C;       // C   exp(log gamma[t+1])
  float* log_gamma  = gexp + C;             // C+1 (log-space gamma cumprod)

  const int bh  = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = nthreads >> 5;
  const int row = tid / G;   // Dv row (steps 4,6,7) / Gram column (steps 5,6)
  const int gl  = tid % G;   // lane within the group
  const int64_t state_off = (int64_t)bh * Dv * Dk;
  float* srow = state + (row < Dv ? row : 0) * SS;

  for (int r = warp; r < Dv; r += nwarps)
    for (int d = lane; d < Dk; d += 32)
      state[r * SS + d] =
          init_state ? init_state[state_off + (int64_t)r * Dk + d] : 0.f;
  __syncthreads();

  const float* q_bh = q + (int64_t)bh * T * Dk;
  const float* k_bh = k + (int64_t)bh * T * Dk;
  const float* v_bh = v + (int64_t)bh * T * Dv;
  const float* a_bh = alpha + (int64_t)bh * T;
  const float* b_bh = beta + (int64_t)bh * T;
  float* out_bh = out + (int64_t)bh * T * Dv;

  const int num_chunks = (T + C - 1) / C;
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    const int c_start = chunk * C;
    const int c_end   = min(c_start + C, T);
    const int L       = c_end - c_start;

    // ---- 1.  Load K_chunk + L2-normalise (one warp per row) ----------------
    for (int t = warp; t < L; t += nwarps) {
      float ssq = 0.f;
      for (int d = lane; d < Dk; d += 32) {
        const float x = k_bh[(int64_t)(c_start + t) * Dk + d];
        k_chunk[t * SS + d] = x;
        ssq += x * x;
      }
      for (int off = 16; off > 0; off >>= 1)
        ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
      const float inv = rsqrtf(fmaxf(ssq, 1e-24f));
      for (int d = lane; d < Dk; d += 32) k_chunk[t * SS + d] *= inv;
    }
    __syncthreads();

    // ---- 2.  Load beta; take the logs of alpha in parallel -----------------
    for (int t = tid; t < L; t += nthreads) {
      beta_chunk[t] = b_bh[c_start + t];
      log_gamma[t + 1] = logf(fmaxf(a_bh[c_start + t], 1e-12f));
    }
    __syncthreads();

    // ---- 3.  Prefix-sum the logs (same association as the scalar path) -----
    if (tid == 0) {
      log_gamma[0] = 0.f;
      for (int i = 1; i <= L; ++i) log_gamma[i] += log_gamma[i - 1];
    }
    __syncthreads();
    for (int t = tid; t < L; t += nthreads) gexp[t] = expf(log_gamma[t + 1]);
    __syncthreads();

    // ---- 4.  W = V_chunk - gamma[t+1]*S0@k[t]  (-> r_buf) ------------------
    for (int t = 0; t < L; ++t) {
      float dot = 0.f;
      if (row < Dv)
        for (int d = gl; d < Dk; d += G) dot += srow[d] * k_chunk[t * SS + d];
      for (int off = G >> 1; off > 0; off >>= 1)
        dot += __shfl_xor_sync(0xffffffffu, dot, off);
      if (gl == 0 && row < Dv)
        r_buf[row * RS + t] =
            v_bh[(int64_t)(c_start + t) * Dv + row] - gexp[t] * dot;
    }
    __syncthreads();

    // ---- 5.  Forward substitution: r from w --------------------------------
    //   r[t] = w[t] - sum_{j<t} exp(logg[t+1]-logg[j+1])*beta[j]*(k_j.k_t)*r[j]
    // The decay*beta factors are folded into gram_row as it is built, so the
    // transcendental is evaluated once per (t,j) instead of once per (t,j,row).
    for (int t = 0; t < L; ++t) {
      const float lg_t = log_gamma[t + 1];
      for (int jb = 0; jb < t; jb += nG) {
        const int j = jb + row;
        float dot = 0.f;
        if (j < t)
          for (int d = gl; d < Dk; d += G)
            dot += k_chunk[t * SS + d] * k_chunk[j * SS + d];
        for (int off = G >> 1; off > 0; off >>= 1)
          dot += __shfl_xor_sync(0xffffffffu, dot, off);
        if (gl == 0 && j < t)
          gram_row[j] = expf(lg_t - log_gamma[j + 1]) * beta_chunk[j] * dot;
      }
      __syncthreads();
      float corr = 0.f;
      if (row < Dv)
        for (int j = gl; j < t; j += G) corr += gram_row[j] * r_buf[row * RS + j];
      for (int off = G >> 1; off > 0; off >>= 1)
        corr += __shfl_xor_sync(0xffffffffu, corr, off);
      if (gl == 0 && row < Dv) r_buf[row * RS + t] -= corr;
      __syncthreads();
    }

    // ---- 6.  Compute outputs -----------------------------------------------
    //   o[t] = gamma[t+1]*S0@q[t] + sum_{j<=t} decay*beta[j]*(k_j.q_t)*r[j]
    // Only j <= t is needed, so the Gram row stops at t (the previous kernel
    // computed the full L-wide row and threw half of it away).
    for (int t = 0; t < L; ++t) {
      const float* qt = q_bh + (int64_t)(c_start + t) * Dk;
      const float lg_t = log_gamma[t + 1];
      for (int jb = 0; jb <= t; jb += nG) {
        const int j = jb + row;
        float dot = 0.f;
        if (j <= t)
          for (int d = gl; d < Dk; d += G) dot += qt[d] * k_chunk[j * SS + d];
        for (int off = G >> 1; off > 0; off >>= 1)
          dot += __shfl_xor_sync(0xffffffffu, dot, off);
        if (gl == 0 && j <= t)
          gram_row[j] = expf(lg_t - log_gamma[j + 1]) * beta_chunk[j] * dot;
      }
      __syncthreads();
      float base = 0.f, acc = 0.f;
      if (row < Dv) {
        for (int d = gl; d < Dk; d += G) base += srow[d] * qt[d];
        for (int j = gl; j <= t; j += G) acc += gram_row[j] * r_buf[row * RS + j];
      }
      for (int off = G >> 1; off > 0; off >>= 1) {
        base += __shfl_xor_sync(0xffffffffu, base, off);
        acc  += __shfl_xor_sync(0xffffffffu, acc, off);
      }
      if (gl == 0 && row < Dv)
        out_bh[(int64_t)(c_start + t) * Dv + row] = base * gexp[t] + acc;
      __syncthreads();
    }

    // ---- 7.  State update S0 <- g[L]*S0 + sum_j (g[L]/g[j+1])*b[j]*r[j]x k[j]
    // Each lane owns its own slice of d, so there is no reduction here.
    for (int t = tid; t < L; t += nthreads)
      gram_row[t] = expf(log_gamma[L] - log_gamma[t + 1]) * beta_chunk[t];
    __syncthreads();
    if (row < Dv) {
      const float gL = expf(log_gamma[L]);
      for (int d = gl; d < Dk; d += G) srow[d] *= gL;
      for (int t = 0; t < L; ++t) {
        const float coeff = gram_row[t] * r_buf[row * RS + t];
        for (int d = gl; d < Dk; d += G) srow[d] += coeff * k_chunk[t * SS + d];
      }
    }
    __syncthreads();
  }

  for (int r = warp; r < Dv; r += nwarps)
    for (int d = lane; d < Dk; d += 32)
      final_state[state_off + (int64_t)r * Dk + d] = state[r * SS + d];
}

// ============================================================================
// v3.5 (issue #122): half2 (FP16x2) CUDA-core gated chunked DeltaNet.
//
// Recasts the four dot-product inner loops (S@k in step 4, k·k Gram in step 5,
// q·k in step 6, and S@q base in step 6) onto __hfma2 half2 packed math with
// fp32 accumulation.  Everything numerically load-bearing stays fp32: state S,
// log-space γ-cumprod, L2-norm, α/β gates, values V, residuals r, W, and the
// state update r@K (step 7).  This is the half2 CUDA-core pipe (healthy at
// ~27 TFLOP/s on this fleet), NOT the firmware-dead fp16 tensor cores.
//
// Shared memory layout is identical to v3 (fp32 throughout).  Only the dot
// product reductions use the fni8_h2_dot helper; the AXPY state update and all
// scalar logic stay fp32.
// ============================================================================
constexpr int DELTANET_V35_THREADS = 128;
constexpr int DELTANET_V35_CHUNK_SIZE = 64;
constexpr int DELTANET_V35_MAX_SMEM = 98304;

__global__ void deltanet_gated_chunk_h2_fwd_kernel(
    const float* __restrict__ q,           // [B,H,T,Dk]
    const float* __restrict__ k,           // [B,H,T,Dk]
    const float* __restrict__ v,           // [B,H,T,Dv]
    const float* __restrict__ alpha,       // [B,H,T]
    const float* __restrict__ beta,        // [B,H,T]
    const float* __restrict__ init_state,  // [B,H,Dv,Dk] or nullptr
    float* __restrict__ out,               // [B,H,T,Dv]
    float* __restrict__ final_state,       // [B,H,Dv,Dk]
    int H, int T, int Dk, int Dv, int C) {
  extern __shared__ float smem[];
  float* state     = smem;                  // Dv*Dk
  float* k_chunk   = state + Dv * Dk;       // C*Dk
  float* r_buf     = k_chunk + C * Dk;      // Dv*C  (W then R)
  float* gram_row  = r_buf + Dv * C;        // C     (scratch: L2-norm/Gram/QK)
  float* beta_chunk = gram_row + C;         // C     (write-strength gates)
  float* log_gamma = beta_chunk + C;        // C+1   (log-space γ cumprod)

  const int bh = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t state_off = (int64_t)bh * Dv * Dk;

  for (int i = tid; i < Dv * Dk; i += blockDim.x)
    state[i] = init_state ? init_state[state_off + i] : 0.f;
  __syncthreads();

  const float* q_bh = q + (int64_t)bh * T * Dk;
  const float* k_bh = k + (int64_t)bh * T * Dk;
  const float* v_bh = v + (int64_t)bh * T * Dv;
  const float* a_bh = alpha + (int64_t)bh * T;
  const float* b_bh = beta + (int64_t)bh * T;
  float* out_bh = out + (int64_t)bh * T * Dv;

  const int num_chunks = (T + C - 1) / C;
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    const int c_start = chunk * C;
    const int c_end   = min(c_start + C, T);
    const int L       = c_end - c_start;

    // ---- 1.  Load K_chunk + L2-normalise -----------------------------------
    for (int idx = tid; idx < L * Dk; idx += blockDim.x) {
      const int t = idx / Dk, d = idx % Dk;
      k_chunk[t * Dk + d] = k_bh[(c_start + t) * Dk + d];
    }
    __syncthreads();
    for (int t = 0; t < L; ++t) {
      if (tid == 0) {
        float ssq = 0.f;
        for (int d = 0; d < Dk; ++d) {
          const float kv = k_chunk[t * Dk + d];
          ssq += kv * kv;
        }
        gram_row[t] = rsqrtf(fmaxf(ssq, 1e-24f));
      }
      __syncthreads();
      const float inv = gram_row[t];
      for (int d = tid; d < Dk; d += blockDim.x)
        k_chunk[t * Dk + d] *= inv;
      __syncthreads();
    }

    // ---- 2.  Load alpha, beta into shared memory ---------------------------
    for (int t = tid; t < L; t += blockDim.x) {
      beta_chunk[t] = b_bh[c_start + t];
      gram_row[t] = a_bh[c_start + t];
    }
    __syncthreads();

    // ---- 3.  Compute log γ cumprod -----------------------------------------
    if (tid == 0) {
      log_gamma[0] = 0.f;
      for (int i = 1; i <= L; ++i) {
        log_gamma[i] = log_gamma[i - 1] + logf(fmaxf(gram_row[i - 1], 1e-12f));
      }
    }
    __syncthreads();

    // ---- 4.  W = V_chunk − γ[i+1]·S₀@k[i]  (→ r_buf)  -- half2 dot -----
    if (tid < Dv) {
      float* srow = state + tid * Dk;
      for (int t = 0; t < L; ++t) {
        const float gi = expf(log_gamma[t + 1]);
        r_buf[tid * C + t] = v_bh[(c_start + t) * Dv + tid] -
                             gi * fni8_h2_dot(srow, &k_chunk[t * Dk], Dk);
      }
    }
    __syncthreads();

    // ---- 5.  Forward substitution: r from w  -- half2 Gram dot ------------
    for (int t = 0; t < L; ++t) {
      for (int j = tid; j < t; j += blockDim.x)
        gram_row[j] = fni8_h2_dot(&k_chunk[t * Dk], &k_chunk[j * Dk], Dk);
      __syncthreads();
      if (tid < Dv) {
        float correction = 0.f;
        for (int j = 0; j < t; ++j) {
          const float decay = expf(log_gamma[t + 1] - log_gamma[j + 1]);
          const float coeff = decay * beta_chunk[j] * gram_row[j];
          correction += coeff * r_buf[tid * C + j];
        }
        r_buf[tid * C + t] -= correction;
      }
      __syncthreads();
    }

    // ---- 6.  Compute outputs  -- half2 Q·K dot ----------------------------
    for (int t = 0; t < L; ++t) {
      const float* qt = q_bh + (c_start + t) * Dk;
      for (int j = tid; j < L; j += blockDim.x)
        gram_row[j] = fni8_h2_dot(qt, &k_chunk[j * Dk], Dk);
      __syncthreads();
      if (tid < Dv) {
        float* srow = state + tid * Dk;
        float base = fni8_h2_dot(srow, qt, Dk);
        base *= expf(log_gamma[t + 1]);
        float acc = base;
        for (int j = 0; j <= t; ++j) {
          const float decay = expf(log_gamma[t + 1] - log_gamma[j + 1]);
          const float coeff = decay * beta_chunk[j];
          acc += coeff * gram_row[j] * r_buf[tid * C + j];
        }
        out_bh[(c_start + t) * Dv + tid] = acc;
      }
      __syncthreads();
    }

    // ---- 7.  State update S₀ ← γ[L]·S₀ + Σ_j (γ[L]/γ[j+1])·β[j]·r[j]⊗k[j] -
    if (tid < Dv) {
      const float gL = expf(log_gamma[L]);
      float* srow = state + tid * Dk;
      for (int d = 0; d < Dk; ++d)
        srow[d] *= gL;
      for (int t = 0; t < L; ++t) {
        const float decay = expf(log_gamma[L] - log_gamma[t + 1]);
        const float coeff = decay * beta_chunk[t] * r_buf[tid * C + t];
        for (int d = 0; d < Dk; ++d)
          srow[d] += coeff * k_chunk[t * Dk + d];
      }
    }
    __syncthreads();
  }

  for (int i = tid; i < Dv * Dk; i += blockDim.x)
    final_state[state_off + i] = state[i];
}

// ============================================================================
// v4 (issue #83): int8 dp4a ungated chunked WY/UT parallel DeltaNet.
//
// Same chunked ungated delta-rule algebra as v2, but the two Dk-contraction
// score matrices — the intra-chunk K-Gram (kᵢ·kⱼ, feeds forward substitution)
// and the Q·K read (qᵢ·kⱼ, feeds the output) — are computed with the `__dp4a`
// int8×4 CUDA-core intrinsic (sm_70 has NO int8 tensor cores; dp4a is the
// primitive, per AGENTS.md). Keys are L2-normalised BEFORE quantization; that
// per-token normalisation is the delta-rule analogue of K-smoothing — it bounds
// each key vector so symmetric per-row int8 is well-conditioned. (The softmax
// mean-subtraction "K-smoothing" is INVALID here: there is no softmax row-shift
// to cancel it, and no per-channel diagonal transform preserves the symmetric
// K-Gram kᵢ·kⱼ; measured, mean-subtraction collapses SQNR to ~14 dB.)
//
// The numerically load-bearing pieces stay fp32 and are NOT quantized: the state
// S, the values V, the residuals r, W = V − S@Kᵀ, the output base S@q, and the
// state update r@K. Only the pure q/k dot products go through int8 dp4a.
//
// int8 layout: q,k rows are quantized symmetric-RTN per row over Dk (scale =
// amax/127, fp32), zero-padded to Dkp = round_up(Dk,4) so dp4a consumes whole
// int32 words. Dequant is a single multiply by the two row scales.
//
// Validated against gated_delta_rule_oracle (α=β=1) at int8 SQNR/cos tolerance
// (NOT allclose); the Python int8 sim ungated_delta_rule_int8_reference mirrors
// this arithmetic. Falls back to the fp v2 kernel where a shape misses the bar.
// ============================================================================
constexpr int DELTANET_V4_THREADS = 128;
constexpr int DELTANET_V4_CHUNK_SIZE = 64;
constexpr int DELTANET_V4_MAX_SMEM = 98304;

// Pack four signed int8-range values into one dp4a int32 word (little-endian).
__device__ __forceinline__ int32_t dn_pack4(int a, int b, int c, int d) {
  return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// int8×int8 dot over Dkp (multiple of 4) elements via dp4a → int32 accumulate.
__device__ __forceinline__ int dn_i8dot(const int8_t* __restrict__ a,
                                         const int8_t* __restrict__ b, int Dkp) {
  int acc = 0;
  for (int w = 0; w < Dkp; w += 4) {
    const int32_t pa = dn_pack4(a[w], a[w + 1], a[w + 2], a[w + 3]);
    const int32_t pb = dn_pack4(b[w], b[w + 1], b[w + 2], b[w + 3]);
    acc = __dp4a(pa, pb, acc);
  }
  return acc;
}

constexpr float DN_Q_MAX = 127.0f;
constexpr float DN_Q_MAX_INV = 1.0f / 127.0f;

__global__ void deltanet_chunk_int8_fwd_kernel(
    const float* __restrict__ q,           // [B,H,T,Dk]
    const float* __restrict__ k,           // [B,H,T,Dk]
    const float* __restrict__ v,           // [B,H,T,Dv]
    const float* __restrict__ init_state,  // [B,H,Dv,Dk] or nullptr
    float* __restrict__ out,               // [B,H,T,Dv]
    float* __restrict__ final_state,       // [B,H,Dv,Dk]
    int H, int T, int Dk, int Dv, int C, int Dkp) {
  extern __shared__ float smem[];
  float* state    = smem;                  // Dv*Dk
  float* k_chunk  = state + Dv * Dk;       // C*Dk (L2-normalised keys, fp32)
  float* r_buf    = k_chunk + C * Dk;      // Dv*C  (W then R)
  float* gram_row = r_buf + Dv * C;        // C     (scratch: L2-norm / Gram / QK)
  float* ks       = gram_row + C;          // C     (int8 key row scales)
  float* qs       = ks + C;                // C     (int8 query row scales)
  int8_t* k_i8    = reinterpret_cast<int8_t*>(qs + C);  // C*Dkp bytes
  int8_t* q_i8    = k_i8 + (int64_t)C * Dkp;            // C*Dkp bytes

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
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    const int c_start = chunk * C;
    const int c_end   = min(c_start + C, T);
    const int L       = c_end - c_start;

    // ---- 1.  Load K_chunk + L2-normalise each key -------------------------
    for (int idx = tid; idx < L * Dk; idx += blockDim.x) {
      const int t = idx / Dk, d = idx % Dk;
      k_chunk[t * Dk + d] = k_bh[(c_start + t) * Dk + d];
    }
    __syncthreads();
    for (int t = 0; t < L; ++t) {
      if (tid == 0) {
        float ssq = 0.f;
        for (int d = 0; d < Dk; ++d) {
          const float kv = k_chunk[t * Dk + d];
          ssq += kv * kv;
        }
        gram_row[t] = rsqrtf(fmaxf(ssq, 1e-24f));
      }
      __syncthreads();
      const float inv = gram_row[t];
      for (int d = tid; d < Dk; d += blockDim.x)
        k_chunk[t * Dk + d] *= inv;
      __syncthreads();
    }

    // ---- 1b.  Quantize normalised K rows → int8 (per-row scale over Dk) ----
    for (int t = 0; t < L; ++t) {
      if (tid == 0) {
        float amax = 0.f;
        for (int d = 0; d < Dk; ++d) amax = fmaxf(amax, fabsf(k_chunk[t * Dk + d]));
        const float s = amax * DN_Q_MAX_INV;
        ks[t] = (s == 0.f) ? 1.f : s;
      }
      __syncthreads();
      const float inv = ks[t];
      for (int d = tid; d < Dkp; d += blockDim.x) {
        float qv = (d < Dk) ? rintf(__fdiv_rn(k_chunk[t * Dk + d], inv)) : 0.f;
        qv = fminf(fmaxf(qv, -DN_Q_MAX), DN_Q_MAX);
        k_i8[t * Dkp + d] = (int8_t)qv;
      }
      __syncthreads();
    }

    // ---- 1c.  Quantize Q rows → int8 (per-row scale over Dk) ---------------
    for (int t = 0; t < L; ++t) {
      const float* qt = q_bh + (c_start + t) * Dk;
      if (tid == 0) {
        float amax = 0.f;
        for (int d = 0; d < Dk; ++d) amax = fmaxf(amax, fabsf(qt[d]));
        const float s = amax * DN_Q_MAX_INV;
        qs[t] = (s == 0.f) ? 1.f : s;
      }
      __syncthreads();
      const float inv = qs[t];
      for (int d = tid; d < Dkp; d += blockDim.x) {
        float qv = (d < Dk) ? rintf(__fdiv_rn(qt[d], inv)) : 0.f;
        qv = fminf(fmaxf(qv, -DN_Q_MAX), DN_Q_MAX);
        q_i8[t * Dkp + d] = (int8_t)qv;
      }
      __syncthreads();
    }

    // ---- 2.  W = V_chunk − S₀ @ K_chunkᵀ   (→ r_buf, fp32) ----------------
    if (tid < Dv) {
      float* srow = state + tid * Dk;
      for (int t = 0; t < L; ++t) {
        float dot = 0.f;
        for (int d = 0; d < Dk; ++d)
          dot += srow[d] * k_chunk[t * Dk + d];
        r_buf[tid * C + t] = v_bh[(c_start + t) * Dv + tid] - dot;
      }
    }
    __syncthreads();

    // ---- 3.  Forward substitution with int8 dp4a K-Gram -------------------
    //   r[t] = W[t] − Σ_{j<t} (kₜ·kⱼ) r[j]      (kₜ·kⱼ via dp4a + dequant)
    for (int t = 0; t < L; ++t) {
      for (int j = tid; j < t; j += blockDim.x) {
        const int g = dn_i8dot(&k_i8[t * Dkp], &k_i8[j * Dkp], Dkp);
        gram_row[j] = (float)g * ks[t] * ks[j];
      }
      __syncthreads();
      if (tid < Dv) {
        float correction = 0.f;
        for (int j = 0; j < t; ++j)
          correction += gram_row[j] * r_buf[tid * C + j];
        r_buf[tid * C + t] -= correction;
      }
      __syncthreads();
    }

    // ---- 4.  Outputs with int8 dp4a Q·K read -----------------------------
    //   o_t = S₀@q_t + Σ_{j≤t} (q_t·kⱼ) R[:,j]   (q_t·kⱼ via dp4a + dequant)
    for (int t = 0; t < L; ++t) {
      for (int j = tid; j < L; j += blockDim.x) {
        const int a = dn_i8dot(&q_i8[t * Dkp], &k_i8[j * Dkp], Dkp);
        gram_row[j] = (float)a * qs[t] * ks[j];
      }
      __syncthreads();
      if (tid < Dv) {
        const float* qt = q_bh + (c_start + t) * Dk;
        float base = 0.f;
        {
          float* srow = state + tid * Dk;
          for (int d = 0; d < Dk; ++d)
            base += srow[d] * qt[d];
        }
        float acc = base;
        for (int j = 0; j <= t; ++j)
          acc += gram_row[j] * r_buf[tid * C + j];
        out_bh[(c_start + t) * Dv + tid] = acc;
      }
      __syncthreads();
    }

    // ---- 5.  State update  S₀ = S₀ + R @ K_chunk  (fp32) -----------------
    if (tid < Dv) {
      float* srow = state + tid * Dk;
      for (int t = 0; t < L; ++t) {
        const float rit = r_buf[tid * C + t];
        for (int d = 0; d < Dk; ++d)
          srow[d] += rit * k_chunk[t * Dk + d];
      }
    }
    __syncthreads();
  }

  for (int i = tid; i < Dv * Dk; i += blockDim.x)
    final_state[state_off + i] = state[i];
}

}  // namespace fni8
