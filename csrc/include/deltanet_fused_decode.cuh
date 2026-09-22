// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Fused Gated-DeltaNet DECODE kernel for sm_70 — collapses the per-layer glue
// (L2-norm(q,k), GQA expand, sigmoid(beta), g=-softplus(dt+dt_bias)*exp(A_log),
// the delta-rule recurrence, and the gated output RMSNorm silu(z)*o) into ONE
// launch. Replaces the ~15-19 tiny PyTorch ops fni8serve runs per linear-attn
// layer at decode with a single graph-capturable kernel.
//
// Adapted (with attribution) from qengine (https://github.com/Haru-neo/qengine),
// Apache License 2.0 — `gdn_recurrent_step_v2<128,128>` (fused L2-norm + score +
// gate + beta + state update, fp32 register state) and `gdn_fused_recur_rmsg`
// (additionally folds the gated RMSNorm + z-gate into the final output, one
// launch per layer). fni8's NOTICE already carries qengine's attribution.
// This port matches fni8's OWN numeric conventions, NOT qengine's: q AND k are
// L2-normalised with the `_l2norm` clamp (norm >= 1e-6), there is NO 1/sqrt(Dk)
// query scaling and NO state clamp (fni8's fp32 delta-rule oracle has neither),
// the decay gate is `alpha = exp(-softplus(dt+dt_bias)*exp(A_log))` and the GQA
// map is `repeat_interleave` semantics (value head hv -> key head hv/rep).
//
// Mapping: ONE BLOCK per (batch, value-head), blockDim.x = DV threads, thread
// `vi` owns output row v[vi] (as qengine's rmsg variant does) — the block-level
// map is required because the gated RMSNorm reduces over ALL DV rows of the head,
// which a warp-per-row map (deltanet_decode.cuh) cannot do in one kernel. State
// lives in fp32; smem is a tiny STATIC buffer (2*DK + DV + a few floats, ~1.5 KB
// for 128/128) — well under the 48 KB default, so NO cudaFuncSetAttribute is
// needed and the kernel stays CUDA-graph-capturable, the same contract as
// deltanet_recurrent_decode. State layout is byte-identical to that kernel
// ([B,H,Dv,Dk]) so the two are interchangeable oracles.
//
// fp32 throughout the state / recurrence / norm (numerically load-bearing, per
// AGENTS.md — never fp64, never quantized). Decode only (T == 1). Dk == Dv == 128.
#pragma once

#include <cstdint>

namespace fni8 {

constexpr int DFD_DIM = 128;              // Dk == Dv specialization
constexpr int DFD_THREADS = DFD_DIM;      // one thread per (dim / v-row)
constexpr int DFD_NWARPS = DFD_THREADS / 32;

// Numerically-stable softplus, matching torch.nn.functional.softplus:
//   softplus(x) = max(x,0) + log1p(exp(-|x|))
__device__ __forceinline__ float dfd_softplus(float x) {
  return fmaxf(x, 0.f) + log1pf(__expf(-fabsf(x)));
}

// Block reduction (sum) over DFD_THREADS lanes via warp shuffles + a smem hop.
// `scratch` must hold at least DFD_NWARPS floats. Result is broadcast to all
// threads. A __syncthreads() precedes the scratch reuse so the caller may pass
// the same buffer repeatedly.
__device__ __forceinline__ float dfd_block_sum(float val, float* scratch, int tid) {
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) val += __shfl_xor_sync(0xFFFFFFFFu, val, off);
  __syncthreads();
  if (lane == 0) scratch[warp_id] = val;
  __syncthreads();
  float total = 0.f;
#pragma unroll
  for (int w = 0; w < DFD_NWARPS; ++w) total += scratch[w];
  return total;
}

// q,k:   [B, nk, DK]  (T==1 squeezed)     fp32
// v:     [B, nv, DV]                       fp32
// dt:    [B, nv]   (gate_proj output)      fp32
// blog:  [B, nv]   (beta_proj output)      fp32
// a_log: [nv]                              fp32
// dtb:   [nv]      (dt_bias)               fp32
// gain:  [DV]      (RMSNorm weight)        fp32
// z:     [B, nv, DV] or nullptr            fp32
// init:  [B, nv, DV, DK] or nullptr        fp32
// out:   [B, nv, DV]                       fp32  (post gated-RMSNorm)
// fstate:[B, nv, DV, DK]                   fp32
template <int DK, int DV>
__global__ void __launch_bounds__(DV) deltanet_fused_decode_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ dt, const float* __restrict__ blog,
    const float* __restrict__ a_log, const float* __restrict__ dtb,
    const float* __restrict__ gain, const float* __restrict__ z,
    const float* __restrict__ init, float* __restrict__ out, float* __restrict__ fstate,
    int nk, int nv, int rep, float q_scale, float eps) {
  const int bh = blockIdx.x;         // over B * nv
  const int b = bh / nv;
  const int hv = bh - b * nv;        // value-head index within the batch element
  const int hk = b * nk + hv / rep;  // global key-head index (repeat_interleave map)
  const int tid = threadIdx.x;       // 0..DV-1  (== 0..DK-1 since DK == DV)

  __shared__ float sQ[DK];           // L2-normalised query
  __shared__ float sK[DK];           // L2-normalised key
  __shared__ float sRed[DFD_NWARPS]; // block-reduction scratch
  __shared__ float sScalar;          // q.k dot (broadcast)

  // ---- Phase 1: L2-norm q,k (per head) + the q.k dot ----------------------
  const float qv = q[(int64_t)hk * DK + tid];
  const float kv = k[(int64_t)hk * DK + tid];
  const float qsq = dfd_block_sum(qv * qv, sRed, tid);
  const float ksq = dfd_block_sum(kv * kv, sRed, tid);
  // rsqrt(max(sumsq,1e-12)) == 1 / max(norm, 1e-6): matches `_l2norm`.
  const float q_inv = rsqrtf(fmaxf(qsq, 1e-12f));
  const float k_inv = rsqrtf(fmaxf(ksq, 1e-12f));
  // q_scale (= 1/sqrt(Dk) for HF gated-delta-rule) is applied to the *normalised*
  // query — the readout scale must land after L2-norm (which would otherwise
  // divide it out), so it cannot be pre-multiplied by the caller. It feeds both
  // sQ (readout) and the q.k dot, and being a single global scalar it very nearly
  // cancels through the per-head RMSNorm; carrying it makes the match exact.
  const float qn = qv * q_inv * q_scale;
  const float kn = kv * k_inv;
  sQ[tid] = qn;
  sK[tid] = kn;
  const float kq = dfd_block_sum(qn * kn, sRed, tid);
  if (tid == 0) sScalar = kq;
  __syncthreads();
  const float kq_dot = sScalar;

  // ---- Phase 2: gate / beta / delta-rule recurrence -----------------------
  // g = -softplus(dt + dt_bias) * exp(A_log);  alpha = exp(g) in (0,1].
  const float g = -dfd_softplus(dt[bh] + dtb[hv]) * __expf(a_log[hv]);
  const float alpha = __expf(g);
  const float beta = 1.0f / (1.0f + __expf(-blog[bh]));

  const int vi = tid;                // this block covers exactly DV rows
  const int64_t st_off = (int64_t)bh * DV * DK + (int64_t)vi * DK;
  const float* st_in = init ? init + st_off : nullptr;

  // sum1 = S_{t-1}[vi,:] . k_n ;  sum2 = S_{t-1}[vi,:] . q_n
  float sum1 = 0.f, sum2 = 0.f;
#pragma unroll
  for (int kd = 0; kd < DK; ++kd) {
    const float s = st_in ? st_in[kd] : 0.f;
    sum1 += s * sK[kd];
    sum2 += s * sQ[kd];
  }
  const float sk = alpha * sum1;
  const float sv_new = beta * (v[(int64_t)bh * DV + vi] - sk);  // delta-rule write
  // o[vi] = sum_kd (alpha*S[vi,kd] + sv_new*k_n[kd]) * q_n[kd]
  //       = alpha*sum2 + sv_new*(q_n . k_n)
  float out_val = alpha * sum2 + sv_new * kq_dot;
  // Single-write the updated state row (re-read st_in — cache hit expected).
#pragma unroll
  for (int kd = 0; kd < DK; ++kd) {
    const float s = st_in ? st_in[kd] : 0.f;
    fstate[st_off + kd] = alpha * s + sv_new * sK[kd];
  }

  // ---- Phase 3: gated output RMSNorm (per head, over DV) ------------------
  //   out = out_val * rsqrt(mean_v(out_val^2) + eps) * gain[vi] * silu(z[vi])
  const float ms = dfd_block_sum(out_val * out_val, sRed, tid) / (float)DV;
  const float rms = rsqrtf(ms + eps);
  float normed = out_val * rms * gain[vi];
  if (z != nullptr) {
    const float zv = z[(int64_t)bh * DV + vi];
    normed *= zv / (1.0f + __expf(-zv));  // SiLU gate
  }
  out[(int64_t)bh * DV + vi] = normed;
}

}  // namespace fni8
