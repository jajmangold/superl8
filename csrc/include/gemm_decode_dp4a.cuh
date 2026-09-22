// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Decode-specialized int8 dp4a GEMM for sm_70 -- split-K GEMV for small M
// (1-16), the fni8-serve autoregressive decode step (issue #27).
//
// The prefill tile GEMM (gemm_w8a8_kernel, BM=BN=BK=64) launches only
// ceil(N/64) x ceil(M/64) threadblocks. At decode M<=16 that collapses to ONE
// M-tile, and for a typical projection N (e.g. down_proj N=1024) that's
// ceil(N/64)=16 blocks on 80 SMs -- ~20% occupancy, and 7/8+ of the M tile is
// wasted padding. Root cause: not enough independent work to fill the GPU.
//
// Fix: one WARP per output column n. Each warp streams a contiguous K-slice
// of ONE weight row (coalesced: lane L reads w[n, k4_base + L], consecutive
// lanes = consecutive int32 words) and dp4a's it against the same K range of
// every X row. X is tiny (M*K int8, M<=16) so it stays L2-resident;
// redundant reads across warps/blocks are cheap, the same reasoning as the
// GQA-shared-read finding in utils/docs/decode-profiling.md. This alone turns
// N independent warps (e.g. 1024 for a down_proj) instead of ceil(N/64)=16.
//
// sm_70 caps a resident SM at 32 blocks OR 64 warps, whichever binds first --
// a naive 1-warp/block launch hits the 32-BLOCK cap at only 32 resident
// warps/SM, half the SM's warp capacity. `DEC_WARPS_PER_BLOCK` (4) packs
// several columns' warps into one block so blocks stay warp-bound instead of
// block-count-bound.
//
// Two kernel variants, chosen by the launcher on `split_k`:
//   - split_k == 1 (the common case: N alone already gives >= the SM-filling
//     block target): `gemm_decode_single_kernel` does the FULL K reduction in
//     one warp and writes the dequantized OutT output directly -- no atomics,
//     no int32 workspace, ONE kernel launch.
//   - split_k > 1 (N itself too small to fill the GPU): `gemm_decode_accum_kernel`
//     partitions K across `split_k` blocks per column, each contributing a
//     partial int32 sum via `atomicAdd` into an `acc32[M,N]` workspace (integer
//     add is associative/commutative, so this stays bitwise-deterministic --
//     no float non-associativity concern), then `gemm_decode_epilogue_kernel`
//     (one thread per output element) applies the dequant scale once kernel
//     1's grid has fully retired (same-stream launches serialize on the whole
//     prior grid, so no explicit sync is needed between the two).
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "compute_dtype.cuh"   // float_to<OutT> for fp16/bf16 store

namespace fni8 {

constexpr int DEC_MAX_M = 16;   // decode-shape cap; larger M should use gemm_w8a8
constexpr int DEC_WARP = 32;
constexpr int DEC_WARPS_PER_BLOCK = 4;
constexpr int DEC_THREADS = DEC_WARP * DEC_WARPS_PER_BLOCK;   // 128

// Per-warp K-range dot product of every X row against ONE weight row,
// int32-exact. `MAX_M` is a COMPILE-TIME upper bound on the runtime M (the
// host dispatches to the smallest MAX_M template that covers M -- see
// launch_decode_single/launch_decode_accum in gemm_decode_dp4a.cu) so the
// accumulator array, and therefore register footprint, scales with the
// actual batch size instead of always paying for DEC_MAX_M=16 registers.
template <int MAX_M>
__device__ __forceinline__ void dec_warp_dot(const int8_t* __restrict__ x,
                                             const int32_t* __restrict__ w_row, int M, int K,
                                             int k4_start, int k4_end, int lane,
                                             int32_t (&acc)[MAX_M]) {
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) acc[m] = 0;

  // 128-bit vectorized bulk: each lane loads an int4 (4 int32 = 16 int8) per step
  // -- 4x the bytes/instruction of a scalar int32 load, so the weight stream
  // saturates HBM instead of being issue-bound. Guarded on K % 16 == 0 (so every
  // row base w + n*K stays 16-byte aligned) and k4_start % 4 == 0 (int4 group
  // alignment within the split); otherwise the scalar loop below does everything.
  int k4 = k4_start;
  if (((K & 15) == 0) && ((k4_start & 3) == 0)) {
    const int g_start = k4_start >> 2, g_end = k4_end >> 2;   // int4 groups
    const int4* w4 = reinterpret_cast<const int4*>(w_row);
    for (int g = g_start + lane; g < g_end; g += DEC_WARP) {
      const int4 wv = w4[g];
#pragma unroll
      for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
          const int4 xv = reinterpret_cast<const int4*>(x + (int64_t)m * K)[g];
          acc[m] = __dp4a(xv.x, wv.x, acc[m]);
          acc[m] = __dp4a(xv.y, wv.y, acc[m]);
          acc[m] = __dp4a(xv.z, wv.z, acc[m]);
          acc[m] = __dp4a(xv.w, wv.w, acc[m]);
        }
      }
    }
    k4 = g_end << 2;   // aligned bulk consumed; scalar tail handles the last 0-3 words
  }

  // Scalar int32 tail (and the whole reduction when not vectorizable).
  for (int kk = k4 + lane; kk < k4_end; kk += DEC_WARP) {
    const int32_t w_word = w_row[kk];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m < M) {
        const int32_t x_word = reinterpret_cast<const int32_t*>(x + (int64_t)m * K)[kk];
        acc[m] = __dp4a(x_word, w_word, acc[m]);
      }
    }
  }
}

// Fast path (split_k == 1): one warp computes the WHOLE K reduction for its
// column and writes the dequantized result directly. x [M,K] int8, x_scale
// [M] fp32, w [N,K] int8 (K%4==0), w_scale [N] fp32 -> out [M,N] OutT.
template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_single_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                          const int8_t* __restrict__ w, const float* __restrict__ w_scale,
                          OutT* __restrict__ out, int M, int N, int K) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;
  if (n >= N) return;

  const int K4 = K / 4;
  const int32_t* w_row = reinterpret_cast<const int32_t*>(w + (int64_t)n * K);

  int32_t acc[MAX_M];
  dec_warp_dot<MAX_M>(x, w_row, M, K, 0, K4, lane, acc);

  const float ws = w_scale[n];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    int32_t v = acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) out[(int64_t)m * N + n] = float_to<OutT>((float)v * x_scale[m] * ws);
  }
}

// ── W4A8 decode: warp-per-column, int4 unpack + per-group scale ─────────────
// 4-bit weights (w_packed [N,K/2] uint8, 2 signed nibbles/byte, even col = low
// nibble) with per-group scales (w_scale [N,K/G] fp32). Same warp-per-column
// structure as gemm_decode_single_kernel, but the int32 accumulator is warp-
// reduced and flushed to fp32 PER GROUP with that group's scale, matching the
// tile gemm_w4a8_kernel's math:
//   out[m,n] = x_scale[m] * sum_g w_scale[n,g] * (sum_{k in group g} x_i8*w_i4)
// split_k == 1 only (large-N decode projections; small N keeps the tile GEMM).
__device__ __forceinline__ int dec_sx4(unsigned nib) { return ((int)(nib << 28)) >> 28; }
__device__ __forceinline__ int32_t dec_pack4(int a, int b, int c, int d) {
  return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// Unpack one packed uint32 (8 nibbles = 8 K-elems, low nibble = lower column) into
// two dp4a int32 words (K-elems [0..3] and [4..7]).
__device__ __forceinline__ void dec_unpack8(uint32_t p, int32_t& lo, int32_t& hi) {
  lo = dec_pack4(dec_sx4(p & 0xF), dec_sx4((p >> 4) & 0xF),
                 dec_sx4((p >> 8) & 0xF), dec_sx4((p >> 12) & 0xF));
  hi = dec_pack4(dec_sx4((p >> 16) & 0xF), dec_sx4((p >> 20) & 0xF),
                 dec_sx4((p >> 24) & 0xF), dec_sx4((p >> 28) & 0xF));
}

// Vectorized W4A8 decode: each lane streams the packed weight row in 128-bit uint4
// loads (one uint4 = 32 nibbles = 32 K-elems = 8 dp4a words = exactly one 32-K scale
// step, matching gemm_w4a8's GEMM_STEP) and dp4a's against 128-bit int4 activation
// loads. int32 within each 32-K step, scaled by that step's group w_scale[n,g] into a
// per-lane fp32 accumulator, then ONE warp reduction at the end. Same math as the tile
// gemm_w4a8; 4x fewer load instructions + 1 reduction (vs per-group) than the scalar
// path -> half the weight bytes of W8A8 actually translate to bandwidth, not overhead.
// 16-byte alignment of x/w and K%32==0 are launcher-enforced (contiguous .fni8 weights
// + fresh rowwise-quant activations always satisfy this).
template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_w4a8_single_kernel(const int8_t* __restrict__ x,        // [M,K] int8
                               const float* __restrict__ x_scale,   // [M] fp32
                               const uint8_t* __restrict__ w,       // [N,K/2] packed int4
                               const float* __restrict__ w_scale,   // [N,K/G] fp32
                               OutT* __restrict__ out, int M, int N, int K, int G) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;
  if (n >= N) return;

  const int num_groups = K / G;
  const int num_u4 = K / 32;                          // uint4 chunks (32 K-elems each)
  const int u4_per_group = G / 32;                    // uint4 chunks per scale group
  const uint4* w4 = reinterpret_cast<const uint4*>(w + (int64_t)n * (K / 2));

  float f_acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) f_acc[m] = 0.0f;

  for (int u = lane; u < num_u4; u += DEC_WARP) {
    const uint4 wv = w4[u];
    int32_t ww[8];
    dec_unpack8(wv.x, ww[0], ww[1]);
    dec_unpack8(wv.y, ww[2], ww[3]);
    dec_unpack8(wv.z, ww[4], ww[5]);
    dec_unpack8(wv.w, ww[6], ww[7]);
    const float sc = w_scale[(int64_t)n * num_groups + (u / u4_per_group)];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m >= M) break;
      const int4* xr = reinterpret_cast<const int4*>(x + (int64_t)m * K);
      const int4 xv0 = xr[u * 2], xv1 = xr[u * 2 + 1];   // 32 int8 = 8 dp4a words
      int32_t s = 0;
      s = __dp4a(xv0.x, ww[0], s); s = __dp4a(xv0.y, ww[1], s);
      s = __dp4a(xv0.z, ww[2], s); s = __dp4a(xv0.w, ww[3], s);
      s = __dp4a(xv1.x, ww[4], s); s = __dp4a(xv1.y, ww[5], s);
      s = __dp4a(xv1.z, ww[6], s); s = __dp4a(xv1.w, ww[7], s);
      f_acc[m] += (float)s * sc;
    }
  }

  // ONE fp32 warp reduction (deterministic butterfly order).
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    float v = f_acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    f_acc[m] = v;
  }

#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    if (lane == 0) out[(int64_t)m * N + n] = float_to<OutT>(f_acc[m] * x_scale[m]);
  }
}

// ── W3A8 decode: warp-per-column, uniform 3-bit bit-plane unpack + per-group ─
// scale. Uniform (symmetric-ish, NOT a codebook) 3-bit weights stored in a
// Q3_K-style bit-plane split (llama.cpp Q3_K layout; MIT, adapted for uniform
// 3-bit — study/attribution only, no code pasted). Per 32-value packing group:
// three uint32 (96 bits = 3.0 bit/wt exactly, byte-lane-aligned for a 3-op
// unpack), laid out [N, (K/32)*3] uint32 (== int32 in torch, same bytes):
//   qs0 : low-2-bits of values 0..15   (byte b bits[2s] = value(4s+b), s=0..3)
//   qs1 : low-2-bits of values 16..31  (s=4..7)
//   hp  : INVERTED high-bit plane       (byte b bit s = (high(value 4s+b)==0))
// Device unpack (per lane): vil=(qs>>2i)&0x03030303; vih=((hp>>i)<<2)&0x04040404;
//   vi = __vsubss4(vil,vih) == (high<<2|low2)-4  in [-4,3] per int8 lane.
// This is the PRODUCTION form of the issue #181 spike (csrc/spike/w3a8_spike.cu,
// cos 0.99996 vs fp-dequant). It is a VRAM/context lever, NOT a speed lever: the
// sub-byte unpack ALU contends with dp4a on the INT pipe, so decode is measured
// at PARITY with W4A8 (not faster) while moving 0.75x the weight bytes (3.0 vs
// 4.0 bpw) — buys longer KV context / more batch slots before OOM, decode-neutral.
// split_k == 1 only (large-N MLP decode projections). Same per-group dequant math
// as gemm_decode_w4a8_single_kernel: out[m,n]=x_scale[m]*sum_g ws[n,g]*(sum x_i8*w_i3).

// Unpack one 32-value bit-plane group (qs0,qs1,hp) into 8 dp4a int32 words
// (natural K order, each lane holding a signed 3-bit value in [-4,3]).
__device__ __forceinline__ void dec_w3_unpack32(uint32_t qs0, uint32_t qs1, uint32_t hp,
                                                int32_t ww[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t vl = (i < 4) ? qs0 : qs1;
    const uint32_t vil = (vl >> (2 * (i & 3))) & 0x03030303u;   // low 2 bits -> 4 int8 lanes
    const uint32_t vih = ((hp >> i) << 2) & 0x04040404u;        // inverted high bit -> bit2
    ww[i] = __vsubss4((int)vil, (int)vih);                      // (high<<2|low2)-4 in [-4,3]
  }
}

// x [M,K] int8, x_scale [M] fp32, w [N,(K/32)*3] uint32 bit-planes, w_scale
// [N,K/G] fp32 -> out [M,N] OutT. One warp per output column; each lane streams
// one 32-K packing group (3 uint32 = 12 B) and dp4a's it against 128-bit int4
// activation loads. int32 within each group, scaled into a per-lane fp32
// accumulator by that group's scale, then ONE warp reduction at the end. K%128==0
// (G a multiple of 128, K a multiple of G) and 16-byte x alignment are launcher-
// enforced. Same math as gemm_decode_w4a8_single_kernel with a 3-bit unpack.
template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_w3a8_single_kernel(const int8_t* __restrict__ x,        // [M,K] int8
                               const float* __restrict__ x_scale,   // [M] fp32
                               const uint32_t* __restrict__ w,      // [N,(K/32)*3] bit-planes
                               const float* __restrict__ w_scale,   // [N,K/G] fp32
                               OutT* __restrict__ out, int M, int N, int K, int G) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;
  if (n >= N) return;

  const int ng = K / 32;                              // packing groups (3 uint32 each)
  const int sg = G / 32;                              // packing groups per scale group
  const int num_sgroups = K / G;
  const uint32_t* wr = w + (int64_t)n * ((int64_t)ng * 3);

  float f_acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) f_acc[m] = 0.0f;

  for (int g = lane; g < ng; g += DEC_WARP) {
    const uint32_t qs0 = wr[(int64_t)g * 3 + 0];
    const uint32_t qs1 = wr[(int64_t)g * 3 + 1];
    const uint32_t hp = wr[(int64_t)g * 3 + 2];
    int32_t ww[8];
    dec_w3_unpack32(qs0, qs1, hp, ww);
    const float sc = w_scale[(int64_t)n * num_sgroups + (g / sg)];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m >= M) break;
      const int4* xr = reinterpret_cast<const int4*>(x + (int64_t)m * K);
      const int4 xv0 = xr[g * 2], xv1 = xr[g * 2 + 1];   // 32 int8 = 8 dp4a words
      int32_t s = 0;
      s = __dp4a(xv0.x, ww[0], s); s = __dp4a(xv0.y, ww[1], s);
      s = __dp4a(xv0.z, ww[2], s); s = __dp4a(xv0.w, ww[3], s);
      s = __dp4a(xv1.x, ww[4], s); s = __dp4a(xv1.y, ww[5], s);
      s = __dp4a(xv1.z, ww[6], s); s = __dp4a(xv1.w, ww[7], s);
      f_acc[m] += (float)s * sc;
    }
  }

  // ONE fp32 warp reduction (deterministic butterfly order).
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    float v = f_acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    f_acc[m] = v;
  }

#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    if (lane == 0) out[(int64_t)m * N + n] = float_to<OutT>(f_acc[m] * x_scale[m]);
  }
}

// Split-K path (split_k > 1): grid (ceil(N/DEC_WARPS_PER_BLOCK), split_k).
// x [M,K] int8, w [N,K] int8 (K%4==0), acc32 [M,N] int32 (pre-zeroed).
template <int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_accum_kernel(const int8_t* __restrict__ x, const int8_t* __restrict__ w,
                         int32_t* __restrict__ acc32, int M, int N, int K,
                         int split_k) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;
  const int ks = blockIdx.y;
  if (n >= N) return;

  const int K4 = K / 4;
  const int k4_per_split = (K4 + split_k - 1) / split_k;
  const int k4_start = ks * k4_per_split;
  const int k4_end = min(K4, k4_start + k4_per_split);
  if (k4_start >= k4_end) return;

  const int32_t* w_row = reinterpret_cast<const int32_t*>(w + (int64_t)n * K);
  int32_t acc[MAX_M];
  dec_warp_dot<MAX_M>(x, w_row, M, K, k4_start, k4_end, lane, acc);

  // Warp-reduce each row's partial sum, then lane 0 flushes it into the
  // global int32 accumulator (exact -- integer add has no rounding/order
  // sensitivity, so the atomics are bitwise-deterministic across launches).
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    int32_t v = acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) atomicAdd(&acc32[(int64_t)m * N + n], v);
  }
}

// ── Fused fp16-input decode GEMV (issue #130 rung-1) ────────────────────────
// Folds the per-row activation quantization into the GEMV prologue: read fp16
// x, compute the per-row max in a cooperative block pass, quantize x -> int8
// into SHARED memory, then run the same warp-per-column dp4a (dec_warp_dot)
// against the smem int8 x. Removes the standalone quantize_int8_rowwise kernel
// AND the HBM round-trip of the int8 activation per linear. Same rowwise quant
// (max/127, round-to-nearest) + same dp4a as the unfused path -> matches it.
//
// Only for split_k==1 AND M*K int8 fitting in dynamic smem (launcher guards;
// larger shapes fall back to the unfused quant->gemm_decode path in the Python
// op). The int8 x buffer is 16-byte aligned so dec_warp_dot's int4 bulk load
// (guarded on K%16==0) works from smem.
template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_single_fp16in_kernel(const __half* __restrict__ x_h,   // [M,K] fp16
                                 const int8_t* __restrict__ w, const float* __restrict__ w_scale,
                                 OutT* __restrict__ out, int M, int N, int K) {
  extern __shared__ __align__(16) char fused_smem[];
  int8_t* xq = reinterpret_cast<int8_t*>(fused_smem);          // [M*K] int8, 16B-aligned
  float* xs = reinterpret_cast<float*>(xq + ((M * K + 15) & ~15));  // [M] fp32 scales
  const int tid = threadIdx.x;

  // Phase 1: per-row max|x| (fp32), block-reduced -> x_scale[m] = max/127.
  __shared__ float warp_max[DEC_WARPS_PER_BLOCK];
  const int lane = tid & 31, warp = tid >> 5;
  for (int m = 0; m < M; ++m) {
    float lm = 0.0f;
    for (int k = tid; k < K; k += DEC_THREADS) lm = fmaxf(lm, fabsf(__half2float(x_h[(int64_t)m * K + k])));
    for (int off = 16; off > 0; off >>= 1) lm = fmaxf(lm, __shfl_xor_sync(0xFFFFFFFFu, lm, off));
    if (lane == 0) warp_max[warp] = lm;
    __syncthreads();
    if (warp == 0) {
      float v = (lane < DEC_WARPS_PER_BLOCK) ? warp_max[lane] : 0.0f;
      for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
      if (lane == 0) xs[m] = v * (1.0f / 127.0f);
    }
    __syncthreads();
  }

  // Phase 2: quantize x -> int8 in smem (round-to-nearest, clamp ±127).
  for (int m = 0; m < M; ++m) {
    const float inv = (xs[m] > 0.0f) ? (1.0f / xs[m]) : 0.0f;
    for (int k = tid; k < K; k += DEC_THREADS) {
      int q = __float2int_rn(__half2float(x_h[(int64_t)m * K + k]) * inv);
      q = max(-127, min(127, q));
      xq[(int64_t)m * K + k] = (int8_t)q;
    }
  }
  __syncthreads();

  // Phase 3: warp-per-column dp4a over the smem int8 x (identical to the single
  // kernel; warps with n>=N have already done their share of phases 1-2).
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp;
  if (n >= N) return;
  const int K4 = K / 4;
  const int32_t* w_row = reinterpret_cast<const int32_t*>(w + (int64_t)n * K);
  int32_t acc[MAX_M];
  dec_warp_dot<MAX_M>(xq, w_row, M, K, 0, K4, lane, acc);

  const float ws = w_scale[n];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    int32_t v = acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) out[(int64_t)m * N + n] = float_to<OutT>((float)v * xs[m] * ws);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Native GGUF k-quant DECODE (warp-per-column MMVQ) — Q4_K / Q5_K / Q6_K.
//
// The prefill tile gemm_q{4,5,6}k_kernel (BM=BN=64) launches only ceil(N/64)
// blocks at M=1 -> the GPU sits idle and native GGUF decode measured 6.7x SLOWER
// than the dequant->per_row_i8 path (join harness, Qwen3-8B-Q4_K_M). Fix: ONE
// warp per output column (llama.cpp's MMVQ / mul_mat_vec_q decode design,
// ggml-cuda/mmvq.cu — distinct from MMQ tiles for prefill; MIT, adapted to
// fni8 style). Each warp does the FULL K reduction for its column n: lane L
// streams sub-blocks L, L+32, ... of the native weight row (128-bit uint4
// loads), unpacks each to int8, dp4a's it against the same-K activation slice,
// and applies the native per-sub-block scale into a per-lane fp32 accumulator;
// ONE __shfl_xor butterfly reduces the warp. Byte-identical math to the tile
// kernel (same SQNR), only the launch/tiling differs. The decode GEMV is
// memory-bound, so native (~1.78x fewer resident bytes than int8) should WIN
// once the launch saturates the GPU. split_k==1 (large-N decode projections).
// ═══════════════════════════════════════════════════════════════════════════

// get_scale_min_k4 (convert.cu:195-202) — decode copy (independent TU from the
// tile kernel's q4k_scale_min; identical body).
__device__ __forceinline__ void dec_kq_scale_min(int j, const uint8_t* s, int& sc, int& m) {
  if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
  else {
    sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
    m = (s[j + 4] >> 4) | ((s[j - 0] >> 6) << 4);
  }
}

// Extract 4 nibbles (low if shift==0, high if shift==4) of a packed uint32's 4
// bytes -> one dp4a int32 (element order preserved: byte b -> lane b).
__device__ __forceinline__ int32_t dec_nib4(uint32_t p, int shift) {
  return dec_pack4((int)((p >> shift) & 0xF), (int)((p >> (shift + 8)) & 0xF),
                   (int)((p >> (shift + 16)) & 0xF), (int)((p >> (shift + 24)) & 0xF));
}

// Read 4 bytes at a 2-byte-aligned address as little-endian uint32 via two aligned
// uint16 loads (native Q2_K/Q3_K/Q6_K blocks are only 2-byte-aligned, so a uint32
// reinterpret would be misaligned). Lets the per-16 unpack run 4 elems/instruction.
// Defined here (before all decode kernels) so Q6_K's SIMD unpack can use it too.
__device__ __forceinline__ uint32_t dec_ld_u32(const uint8_t* p) {
  const uint16_t* p16 = reinterpret_cast<const uint16_t*>(p);
  return (uint32_t)p16[0] | ((uint32_t)p16[1] << 16);
}

// ── Q4_K decode: one warp per column, native 144-B super-blocks ──────────────
// out[m,n] = x_scale[m] * sum_sub ( d*sc*<x,q4> - dmin*m*sum(x) )  over K/32
// sub-blocks. x [M,K] int8 (K%256==0), x_scale [M], w [N,num_sb*144] uint8.
template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_q4k_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                       const uint8_t* __restrict__ w, OutT* __restrict__ out,
                       int M, int N, int K, int num_sb) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;
  if (n >= N) return;
  const int nsub = K / 32;                       // 32-elem sub-blocks along K
  const int64_t row_bytes = (int64_t)num_sb * 144;
  const uint8_t* wrow = w + (int64_t)n * row_bytes;

  float f_acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) f_acc[m] = 0.f;

  for (int j = lane; j < nsub; j += DEC_WARP) {
    const int sb = j >> 3, js = j & 7;           // super-block, sub-index 0..7
    const uint8_t* blk = wrow + (int64_t)sb * 144;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
    int sc, mm; dec_kq_scale_min(js, blk + 4, sc, mm);
    const uint8_t* qs = blk + 16 + (js >> 1) * 32;   // 32 bytes (shared by js & js^1)
    const int shift = (js & 1) ? 4 : 0;              // high nibble for odd sub-index
    const uint4 wv0 = *reinterpret_cast<const uint4*>(qs);
    const uint4 wv1 = *reinterpret_cast<const uint4*>(qs + 16);
    int32_t q[8];
    q[0] = dec_nib4(wv0.x, shift); q[1] = dec_nib4(wv0.y, shift);
    q[2] = dec_nib4(wv0.z, shift); q[3] = dec_nib4(wv0.w, shift);
    q[4] = dec_nib4(wv1.x, shift); q[5] = dec_nib4(wv1.y, shift);
    q[6] = dec_nib4(wv1.z, shift); q[7] = dec_nib4(wv1.w, shift);
    const float dsc = d * sc, dm = dmin * mm;
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m >= M) break;
      const int4* xr = reinterpret_cast<const int4*>(x + (int64_t)m * K);
      const int4 xv0 = xr[j * 2], xv1 = xr[j * 2 + 1];     // 32 int8 = 8 dp4a words
      int32_t si = 0, xs = 0;
      si = __dp4a(xv0.x, q[0], si); xs = __dp4a(xv0.x, 0x01010101, xs);
      si = __dp4a(xv0.y, q[1], si); xs = __dp4a(xv0.y, 0x01010101, xs);
      si = __dp4a(xv0.z, q[2], si); xs = __dp4a(xv0.z, 0x01010101, xs);
      si = __dp4a(xv0.w, q[3], si); xs = __dp4a(xv0.w, 0x01010101, xs);
      si = __dp4a(xv1.x, q[4], si); xs = __dp4a(xv1.x, 0x01010101, xs);
      si = __dp4a(xv1.y, q[5], si); xs = __dp4a(xv1.y, 0x01010101, xs);
      si = __dp4a(xv1.z, q[6], si); xs = __dp4a(xv1.z, 0x01010101, xs);
      si = __dp4a(xv1.w, q[7], si); xs = __dp4a(xv1.w, 0x01010101, xs);
      f_acc[m] += dsc * (float)si - dm * (float)xs;
    }
  }
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    float v = f_acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) out[(int64_t)m * N + n] = float_to<OutT>(v * x_scale[m]);
  }
}

// ── Q5_K decode: Q4_K + a 5th bit per weight from qh[32] ─────────────────────
// 5-bit = (qs nibble) | (((qh[l]>>js)&1)<<4). Block: d dmin scales[12] qh[32]
// qs[128] (176 B). Same affine + min-correction spine as Q4_K.
template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_q5k_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                       const uint8_t* __restrict__ w, OutT* __restrict__ out,
                       int M, int N, int K, int num_sb) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;
  if (n >= N) return;
  const int nsub = K / 32;
  const int64_t row_bytes = (int64_t)num_sb * 176;
  const uint8_t* wrow = w + (int64_t)n * row_bytes;

  float f_acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) f_acc[m] = 0.f;

  for (int j = lane; j < nsub; j += DEC_WARP) {
    const int sb = j >> 3, js = j & 7;
    const uint8_t* blk = wrow + (int64_t)sb * 176;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
    int sc, mm; dec_kq_scale_min(js, blk + 4, sc, mm);
    const uint8_t* qh = blk + 16;                     // 32 bytes (1 bit/weight)
    const uint8_t* qs = blk + 48 + (js >> 1) * 32;
    const int shift = (js & 1) ? 4 : 0;
    const uint4 wv0 = *reinterpret_cast<const uint4*>(qs);
    const uint4 wv1 = *reinterpret_cast<const uint4*>(qs + 16);
    const uint4 hv0 = *reinterpret_cast<const uint4*>(qh);
    const uint4 hv1 = *reinterpret_cast<const uint4*>(qh + 16);
    // 5th bit: bit js of each qh byte -> bit 4 of the int8 lane.
    const uint32_t hb = 0x01010101u;
    int32_t q[8];
    q[0] = dec_nib4(wv0.x, shift) | (int)(((hv0.x >> js) & hb) << 4);
    q[1] = dec_nib4(wv0.y, shift) | (int)(((hv0.y >> js) & hb) << 4);
    q[2] = dec_nib4(wv0.z, shift) | (int)(((hv0.z >> js) & hb) << 4);
    q[3] = dec_nib4(wv0.w, shift) | (int)(((hv0.w >> js) & hb) << 4);
    q[4] = dec_nib4(wv1.x, shift) | (int)(((hv1.x >> js) & hb) << 4);
    q[5] = dec_nib4(wv1.y, shift) | (int)(((hv1.y >> js) & hb) << 4);
    q[6] = dec_nib4(wv1.z, shift) | (int)(((hv1.z >> js) & hb) << 4);
    q[7] = dec_nib4(wv1.w, shift) | (int)(((hv1.w >> js) & hb) << 4);
    const float dsc = d * sc, dm = dmin * mm;
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m >= M) break;
      const int4* xr = reinterpret_cast<const int4*>(x + (int64_t)m * K);
      const int4 xv0 = xr[j * 2], xv1 = xr[j * 2 + 1];
      int32_t si = 0, xs = 0;
      si = __dp4a(xv0.x, q[0], si); xs = __dp4a(xv0.x, 0x01010101, xs);
      si = __dp4a(xv0.y, q[1], si); xs = __dp4a(xv0.y, 0x01010101, xs);
      si = __dp4a(xv0.z, q[2], si); xs = __dp4a(xv0.z, 0x01010101, xs);
      si = __dp4a(xv0.w, q[3], si); xs = __dp4a(xv0.w, 0x01010101, xs);
      si = __dp4a(xv1.x, q[4], si); xs = __dp4a(xv1.x, 0x01010101, xs);
      si = __dp4a(xv1.y, q[5], si); xs = __dp4a(xv1.y, 0x01010101, xs);
      si = __dp4a(xv1.z, q[6], si); xs = __dp4a(xv1.z, 0x01010101, xs);
      si = __dp4a(xv1.w, q[7], si); xs = __dp4a(xv1.w, 0x01010101, xs);
      f_acc[m] += dsc * (float)si - dm * (float)xs;
    }
  }
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    float v = f_acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) out[(int64_t)m * N + n] = float_to<OutT>(v * x_scale[m]);
  }
}

// ── Q6_K decode: symmetric, per-16 sub-block, quadrant ql/qh unpack ──────────
// out[m,n] = x_scale[m] * d * sum_i sc[i] * <x, q6-32>  over K/16 sub-blocks.
// Natural per-16 sub-block i -> super sb=i/16, is=i%16; is = 8*g6 + 2*qd + h.
// code = ((ql nibble) | (qh 2-bit<<4)) - 32. Block: ql[128] qh[64] scales[16]
// (int8) d (210 B). Scalar per-element gather (16 elems); Q6_K is the minority.
template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_q6k_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                       const uint8_t* __restrict__ w, OutT* __restrict__ out,
                       int M, int N, int K, int num_sb) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;
  if (n >= N) return;
  const int nsub = K / 16;                       // 16-elem sub-blocks along K
  const int64_t row_bytes = (int64_t)num_sb * 210;
  const uint8_t* wrow = w + (int64_t)n * row_bytes;

  float f_acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) f_acc[m] = 0.f;

  for (int i = lane; i < nsub; i += DEC_WARP) {
    const int sb = i >> 4, is = i & 15;          // super-block, per-16 index 0..15
    const uint8_t* blk = wrow + (int64_t)sb * 210;
    const int g6 = is >> 3, rem = is & 7, qd = rem >> 1, h = rem & 1;
    const uint8_t* ql = blk + 64 * g6;
    const uint8_t* qh = blk + 128 + 32 * g6;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
    const float dsc = d * (float)((const int8_t*)(blk + 192))[is];
    // 16 codes -> 4 dp4a words.
    // SIMD unpack: the 4 elements of word wch have consecutive l (l0..l0+3) ->
    // consecutive ql/qh bytes, so 4 codes/instruction (vs the old per-element
    // scalar gather that made Q6_K decode ~2x slower than Q3_K/Q5_K). ql nibble
    // (low if qd<2 else high) | (qh 2-bit << 4), centered -32 (__vsubss4). qd and
    // h are fixed per sub-block. Byte-identical int8 codes -> same dp4a.
    int32_t q[4];
#pragma unroll
    for (int wch = 0; wch < 4; ++wch) {
      const int l0 = h * 16 + 4 * wch;
      const uint32_t qlw = dec_ld_u32(ql + ((qd & 1) ? l0 + 32 : l0));
      const uint32_t nib = (qd < 2) ? (qlw & 0x0F0F0F0Fu) : ((qlw >> 4) & 0x0F0F0F0Fu);
      const uint32_t hb = ((dec_ld_u32(qh + l0) >> (2 * qd)) & 0x03030303u) << 4;
      q[wch] = __vsubss4((int)(nib | hb), 0x20202020);
    }
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m >= M) break;
      const int4 xv = reinterpret_cast<const int4*>(x + (int64_t)m * K)[i];  // 16 int8
      int32_t si = 0;
      si = __dp4a(xv.x, q[0], si);
      si = __dp4a(xv.y, q[1], si);
      si = __dp4a(xv.z, q[2], si);
      si = __dp4a(xv.w, q[3], si);
      f_acc[m] += dsc * (float)si;
    }
  }
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    float v = f_acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) out[(int64_t)m * N + n] = float_to<OutT>(v * x_scale[m]);
  }
}

// ── Q3_K decode: warp-per-column, symmetric, per-16, signed 6-bit scale ──────
// Block (110 B): hmask[32] qs[64] scales[12] d. code = (qs 2-bit)|(hmask bit<<2)
// then -4; scale_j signed [-32,31]. y = d*scale_j*(q3-4). No min. THE type for
// Qwen3.6-27B-Q3_K_S single-card residency. Adapted from vec_dot_q3_K (MIT).
__device__ __forceinline__ int dec_q3k_scale(const uint8_t* sc, int is) {
  const int lo = (sc[is & 7] >> ((is >> 3) << 2)) & 0xF;
  const int hi = ((sc[8 + (is & 3)] >> ((is >> 2) << 1)) & 3) << 4;
  return (lo | hi) - 32;
}


template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_q3k_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                       const uint8_t* __restrict__ w, OutT* __restrict__ out,
                       int M, int N, int K, int num_sb) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;
  if (n >= N) return;
  const int nsub = K / 16;
  const int64_t row_bytes = (int64_t)num_sb * 110;
  const uint8_t* wrow = w + (int64_t)n * row_bytes;

  float f_acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) f_acc[m] = 0.f;

  for (int i = lane; i < nsub; i += DEC_WARP) {
    const int sb = i >> 4, is = i & 15;
    const uint8_t* blk = wrow + (int64_t)sb * 110;
    const uint8_t* hmask = blk;
    const uint8_t* qs = blk + 32;
    const int g_ = is >> 3, sig = is & 7, shift = sig & 6;
    const uint8_t* qsp = qs + 32 * g_ + (sig & 1) * 16;
    const uint8_t* hmp = hmask + (sig & 1) * 16;
    const int m_shift = g_ * 4 + (sig >> 1);
    const float dsc = __half2float(*reinterpret_cast<const __half*>(blk + 108))
                    * (float)dec_q3k_scale(blk + 96, is);
    int32_t q[4];
#pragma unroll
    for (int wch = 0; wch < 4; ++wch) {
      const uint32_t qw = dec_ld_u32(qsp + 4 * wch);
      const uint32_t hw = dec_ld_u32(hmp + 4 * wch);
      const uint32_t vl = (qw >> shift) & 0x03030303u;
      const uint32_t vh = ((hw >> m_shift) & 0x01010101u) << 2;
      q[wch] = __vsubss4((int)(vl | vh), 0x04040404);
    }
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m >= M) break;
      const int4 xv = reinterpret_cast<const int4*>(x + (int64_t)m * K)[i];
      int32_t si = 0;
      si = __dp4a(xv.x, q[0], si);
      si = __dp4a(xv.y, q[1], si);
      si = __dp4a(xv.z, q[2], si);
      si = __dp4a(xv.w, q[3], si);
      f_acc[m] += dsc * (float)si;
    }
  }
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    float v = f_acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) out[(int64_t)m * N + n] = float_to<OutT>(v * x_scale[m]);
  }
}

// ── Q2_K decode: warp-per-column, affine, per-16, 4-bit scale + 4-bit min ────
// Block (84 B): scales[16] qs[64] d dmin. y = d*sc*q - dmin*m; sc=scales[is]&0xF,
// m=scales[is]>>4, q in [0,3]. Min-correction via per-16 activation sum.
template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_q2k_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                       const uint8_t* __restrict__ w, OutT* __restrict__ out,
                       int M, int N, int K, int num_sb) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;
  if (n >= N) return;
  const int nsub = K / 16;
  const int64_t row_bytes = (int64_t)num_sb * 84;
  const uint8_t* wrow = w + (int64_t)n * row_bytes;

  float f_acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) f_acc[m] = 0.f;

  for (int i = lane; i < nsub; i += DEC_WARP) {
    const int sb = i >> 4, is = i & 15;
    const uint8_t* blk = wrow + (int64_t)sb * 84;
    const uint8_t* qs = blk + 16;
    const int g_ = is >> 3, sig = is & 7, shift = sig & 6;
    const uint8_t* qsp = qs + 32 * g_ + (sig & 1) * 16;
    const int scbyte = blk[is];
    const float dsc = __half2float(*reinterpret_cast<const __half*>(blk + 80)) * (float)(scbyte & 0xF);
    const float dm = __half2float(*reinterpret_cast<const __half*>(blk + 82)) * (float)(scbyte >> 4);
    int32_t q[4];
#pragma unroll
    for (int wch = 0; wch < 4; ++wch)
      q[wch] = (int)((dec_ld_u32(qsp + 4 * wch) >> shift) & 0x03030303u);
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m >= M) break;
      const int4 xv = reinterpret_cast<const int4*>(x + (int64_t)m * K)[i];
      int32_t si = 0, xs = 0;
      si = __dp4a(xv.x, q[0], si); xs = __dp4a(xv.x, 0x01010101, xs);
      si = __dp4a(xv.y, q[1], si); xs = __dp4a(xv.y, 0x01010101, xs);
      si = __dp4a(xv.z, q[2], si); xs = __dp4a(xv.z, 0x01010101, xs);
      si = __dp4a(xv.w, q[3], si); xs = __dp4a(xv.w, 0x01010101, xs);
      f_acc[m] += dsc * (float)si - dm * (float)xs;
    }
  }
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    if (m >= M) break;
    float v = f_acc[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) out[(int64_t)m * N + n] = float_to<OutT>(v * x_scale[m]);
  }
}

// One thread per output element: out[m,n] = acc32[m,n] * x_scale[m] * w_scale[n].
template <typename OutT>
__global__ void gemm_decode_epilogue_kernel(const int32_t* __restrict__ acc32,
                                            const float* __restrict__ x_scale,
                                            const float* __restrict__ w_scale,
                                            OutT* __restrict__ out, int M, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  const int m = idx / N, n = idx % N;
  out[idx] = float_to<OutT>((float)acc32[idx] * x_scale[m] * w_scale[n]);
}

// ── Fused requant epilogue (issue #118) ─────────────────────────────────────
// Reads the int32 partial workspace filled by gemm_decode_accum_kernel,
// applies per-row x_scale and per-col w_scale, computes the per-row absmax,
// then requantizes every element to int8 in [−127,127] with the row's own
// fp32 out_scale, so the output pair (out_i8 [M,N] int8, out_scale [M] fp32)
// can be consumed by the NEXT int8 op without a standalone quantize kernel
// and without an fp16-to-HBM round-trip — the "int8 float-tax" eliminated.
// One block per output row (M ≤ 16); two passes over the int32 workspace
// (pass 1 = row absmax, pass 2 = requantize) to keep register pressure low.
template <int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_requant_epilogue_kernel(const int32_t* __restrict__ acc32,
                                    const float* __restrict__ x_scale,
                                    const float* __restrict__ w_scale,
                                    int8_t* __restrict__ out_i8,
                                    float* __restrict__ out_scale,
                                    int M, int N) {
  (void)(int)MAX_M;
  const int m = blockIdx.x;
  if (m >= M) return;

  const int tid = threadIdx.x;
  const int lane = tid & 31;

  __shared__ float warp_best[DEC_WARPS_PER_BLOCK];

  const float xs = x_scale[m];
  float local_best = 0.0f;

  for (int n = tid; n < N; n += DEC_THREADS) {
    const float v = (float)acc32[(int64_t)m * N + n] * xs * w_scale[n];
    local_best = fmaxf(local_best, fabsf(v));
  }

  for (int off = 16; off > 0; off >>= 1)
    local_best = fmaxf(local_best, __shfl_xor_sync(0xFFFFFFFFu, local_best, off));

  if (lane == 0) warp_best[tid / DEC_WARP] = local_best;
  __syncthreads();

  float row_best = 0.0f;
  if (tid < DEC_WARPS_PER_BLOCK) row_best = warp_best[tid];
  for (int off = 16; off > 0; off >>= 1)
    row_best = fmaxf(row_best, __shfl_xor_sync(0xFFFFFFFFu, row_best, off));

  if (lane == 0 && tid < DEC_WARPS_PER_BLOCK) warp_best[0] = row_best;
  __syncthreads();

  row_best = warp_best[0];
  const float inv_scale = (row_best > 0.0f) ? (127.0f / row_best) : 0.0f;
  out_scale[m] = (row_best > 0.0f) ? (row_best / 127.0f) : 2e-38f;

  for (int n = tid; n < N; n += DEC_THREADS) {
    const float v = (float)acc32[(int64_t)m * N + n] * xs * w_scale[n];
    int q = __float2int_rn(v * inv_scale);
    q = max(-127, min(127, q));
    out_i8[(int64_t)m * N + n] = (int8_t)q;
  }
}

}  // namespace fni8
