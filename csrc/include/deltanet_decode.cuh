// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Decode-specialized fp32 Gated-DeltaNet recurrence for sm_70.
//
// Same math as the v1 `deltanet_recurrent_fwd_kernel` (validated against
// `tests/reference_linear_attn.py::gated_delta_rule_oracle`), but re-mapped for
// the autoregressive decode step where B*H is tiny. v1 launches ONE block per
// (batch, head) -- for a hybrid model at decode (B=1, H=16) that's 16 blocks on
// 80 SMs (~80% idle), and within a block only Dv lanes work with serial per-lane
// reductions. This kernel instead launches **one WARP per (batch, head, v-row)**:
// B=1,H=16,Dv=128 -> 2048 warps, filling the GPU (same occupancy fix as
// gemm_decode_dp4a.cuh's warp-per-column decode GEMV, and the sm_70 "32 blocks
// OR 64 warps" trap noted there).
//
// State row S[v_row, :] (Dk floats) lives entirely in REGISTERS -- lane `l` owns
// elements k = l + 32*j for j in [0, ceil(Dk/32)). The S*k and S*q dots are warp
// `__shfl_xor_sync` tree reductions over the 32 lanes; the state update is
// per-lane (each lane owns distinct k), no cross-lane traffic. Because state is
// in registers there is **zero dynamic shared memory**, so the launcher makes no
// `cudaFuncSetAttribute` call -- unlike v1 (64 KB smem state -> a per-invocation
// runtime CUDA call that is not stable inside a CUDA-graph capture). This kernel
// is therefore CUDA-graph-capturable, which is what lets fni8serve replay it
// inside its graphed decode instead of falling back to eager.
//
// fp32 throughout (the recurrence is numerically load-bearing, per AGENTS.md).
// Supports Dk, Dv <= 128 (ceil(128/32)=4 register slots per lane). General T
// (loops sequentially), but the intended caller is L==1 decode.
#pragma once

#include <cstdint>

namespace fni8 {

constexpr int DND_WARP = 32;
constexpr int DND_WARPS_PER_BLOCK = 4;
constexpr int DND_THREADS = DND_WARP * DND_WARPS_PER_BLOCK;  // 128
constexpr int DND_MAX_CHUNK = 4;  // ceil(128/32): max Dk (or Dv) elements per lane

__global__ void __launch_bounds__(DND_THREADS) deltanet_recurrent_decode_kernel(
    const float* __restrict__ q,           // [B,H,T,Dk]
    const float* __restrict__ k,           // [B,H,T,Dk]
    const float* __restrict__ v,           // [B,H,T,Dv]
    const float* __restrict__ alpha,       // [B,H,T]
    const float* __restrict__ beta,        // [B,H,T]
    const float* __restrict__ init_state,  // [B,H,Dv,Dk] or nullptr (zero init)
    float* __restrict__ out,               // [B,H,T,Dv]
    float* __restrict__ final_state,       // [B,H,Dv,Dk]
    int BH, int T, int Dk, int Dv) {
  const int lane = threadIdx.x & (DND_WARP - 1);
  const int warp_id = threadIdx.x >> 5;
  const int warp_global = blockIdx.x * DND_WARPS_PER_BLOCK + warp_id;
  if (warp_global >= BH * Dv) return;
  const int bh = warp_global / Dv;
  const int v_row = warp_global - bh * Dv;
  const int nchunk = (Dk + DND_WARP - 1) / DND_WARP;  // <= DND_MAX_CHUNK

  // State row S[v_row, :] in registers: srow[j] holds k = lane + 32*j.
  const int64_t srow_off = (int64_t)bh * Dv * Dk + (int64_t)v_row * Dk;
  float srow[DND_MAX_CHUNK];
#pragma unroll
  for (int j = 0; j < DND_MAX_CHUNK; ++j) {
    const int kk = lane + DND_WARP * j;
    srow[j] = (j < nchunk && kk < Dk && init_state) ? init_state[srow_off + kk] : 0.f;
  }

  const float* q_bh = q + (int64_t)bh * T * Dk;
  const float* k_bh = k + (int64_t)bh * T * Dk;
  const float* v_bh = v + (int64_t)bh * T * Dv;
  const float* a_bh = alpha + (int64_t)bh * T;
  const float* b_bh = beta + (int64_t)bh * T;
  float* out_bh = out + (int64_t)bh * T * Dv;

  for (int t = 0; t < T; ++t) {
    float kbuf[DND_MAX_CHUNK], qbuf[DND_MAX_CHUNK];
    float sumsq = 0.f;
#pragma unroll
    for (int j = 0; j < DND_MAX_CHUNK; ++j) {
      const int kk = lane + DND_WARP * j;
      const bool ok = (j < nchunk) && (kk < Dk);
      kbuf[j] = ok ? k_bh[t * Dk + kk] : 0.f;
      qbuf[j] = ok ? q_bh[t * Dk + kk] : 0.f;
      sumsq += kbuf[j] * kbuf[j];
    }
    // clamp on the NORM at 1e-12 (norm^2 = 1e-24), matching the fp32 oracle.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sumsq += __shfl_xor_sync(0xFFFFFFFFu, sumsq, off);
    const float inv_norm = rsqrtf(fmaxf(sumsq, 1e-24f));
#pragma unroll
    for (int j = 0; j < DND_MAX_CHUNK; ++j) kbuf[j] *= inv_norm;

    const float a_t = a_bh[t];
    const float b_t = b_bh[t];
    const float v_t = v_bh[t * Dv + v_row];  // this row's value scalar

    float sk = 0.f;
#pragma unroll
    for (int j = 0; j < DND_MAX_CHUNK; ++j) sk += srow[j] * kbuf[j];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sk += __shfl_xor_sync(0xFFFFFFFFu, sk, off);
    sk *= a_t;
    const float write = b_t * (v_t - sk);

    float ov = 0.f;
#pragma unroll
    for (int j = 0; j < DND_MAX_CHUNK; ++j) {
      const float ns = a_t * srow[j] + write * kbuf[j];
      srow[j] = ns;
      ov += ns * qbuf[j];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ov += __shfl_xor_sync(0xFFFFFFFFu, ov, off);
    if (lane == 0) out_bh[t * Dv + v_row] = ov;
  }

#pragma unroll
  for (int j = 0; j < DND_MAX_CHUNK; ++j) {
    const int kk = lane + DND_WARP * j;
    if (j < nchunk && kk < Dk) final_state[srow_off + kk] = srow[j];
  }
}

}  // namespace fni8
