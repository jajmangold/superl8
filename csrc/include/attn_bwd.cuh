// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// FlashAttention-2 backward for sm_70 — PR5b-2: int8 dp4a S=Q.K^T recompute.
//
// Precision map (SageBwd, arXiv 2505.11594 / 2603.02170; see
// utils/docs/backward-design.md):
//   recompute S = Q.K^T -> INT8 dp4a   (contracts over d, d-contiguous, no transpose)
//   dQ = dS'.K          -> INT8 dp4a   (dq pass; K transposed+packed over keys)
//   dV = P^T.dO         -> INT8 dp4a   (dkv pass; dO transposed+packed over m,
//                          per-key P scale via a cheap int8-S pre-pass so the
//                          softmax tail is not zeroed)
//   dP = dO.V^T         -> fp on CUDA cores (KEEP — the sensitive matmul; its
//                          error blows up through dS = P.(dP - D) cancellation)
//   dK = dS^T.Q         -> fp accumulate. int8 dK needs max_m|dS| (a dP-pre-pass,
//                          since dS needs dP) and saves little — dP dominates the
//                          dkv cost. The accuracy/perf gate, not ideology, keeps
//                          it fp (AGENTS.md).
//
// PERF NOTE: the fused backward does NOT beat cuBLAS here — the mandatory fp dP
// (hand-scalar on CUDA cores) loses to a tuned fp32 GEMM. int8 S/dQ/dV speed
// their parts but dP is the wall. The wins are (a) MEMORY (never materializes
// [M,N] -> long seq / 16 GB), (b) the foundation for PR7's Volta split-pipe
// overlap (fp dP on FP32 cores || int8 dp4a on INT32 cores, which ARE independent).
//
// S is recomputed via __dp4a on per-tile int8 copies of the Q and K tiles
// (per-ROW symmetric quant, scale = amax/127 in fp32, mirroring the forward's
// proven per-row recipe — finer than per-block, keeps the accuracy gate happy).
// The fp16 tiles are kept resident because dP/dV/dQ/dK still consume them.
//
// Two atomic-free kernels with swapped loop orders (deterministic, no atomicAdd):
//   bwd_dq_kernel  — Q-outer/KV-inner: each block owns a Q tile, accumulates dQ
//                    in registers (mirrors the forward's O accumulation, with
//                    dS as P and K as V).
//   bwd_dkv_kernel — KV-outer/Q-inner: each block owns a K/V tile, accumulates
//                    dK, dV across all Q tiles (both lanes recompute S/dP; each
//                    owns a D-half of the accumulators).
// Both recompute S = Q.K^T (int8) and P = exp2((S - LSE)*log2e), and dP = dO.V^T
// (fp); D_i = rowsum(dO_i . O_i). Dynamic shared memory (tiles exceed the 48 KB
// static cap at D=128); the launcher opts in via cudaFuncSetAttribute. Tiles are
// 64x64; perf/tile-tuning is PR7.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace fni8 {

// Tile config — overridable at compile time by the PR7 autotuner (setup.py maps
// FNI8_BWD_BM/BN env vars to -D defines). Defaults reproduce the committed kernel.
// Square tiles + lane-pair (2 threads/row) for now: THREADS = 2*TILE. Raising
// threads-per-row (the occupancy lever) is the next, separately-gated step.
#ifndef FNI8_BWD_BM
#define FNI8_BWD_BM 64
#endif
#ifndef FNI8_BWD_BN
#define FNI8_BWD_BN 64
#endif
#ifndef FNI8_BWD_TPR
#define FNI8_BWD_TPR 8   // threads/row: occupancy lever. 8 -> 48% occ (was 12% @2),
#endif                    // 64 regs (was 127), d128 bwd 150->121 ms. Autotuned default.
constexpr int BWD_BM = FNI8_BWD_BM;
constexpr int BWD_BN = FNI8_BWD_BN;
constexpr int BWD_TPR = FNI8_BWD_TPR;
static_assert(BWD_BM == BWD_BN, "square tiles (BM==BN) required for now");
static_assert(BWD_TPR >= 2 && (BWD_TPR & (BWD_TPR - 1)) == 0, "TPR must be a power of 2 >= 2");
static_assert(32 % BWD_TPR == 0, "TPR must divide the warp (groups stay in-warp)");
constexpr int BWD_THREADS = BWD_BM * BWD_TPR;   // rows * threads-per-row
constexpr int BWD_HALF = BWD_BN / BWD_TPR;      // contraction cols per lane (dq keys/lane)
constexpr float BWD_LOG2E = 1.4426950408889634f;

// Reduce val across a lane's TPR-group (consecutive lanes, group aligned in-warp).
// A per-GROUP shfl mask is REQUIRED, not the full warp: dkv skips whole groups on
// a masked key (group-uniform `continue`), so different groups in a warp diverge;
// a full-warp mask would wait forever on the skipped lanes. mask covers exactly
// this thread's TPR lanes: ((1<<TPR)-1) << (lane - lane%TPR).
__device__ __forceinline__ unsigned bwd_group_mask(int tpr) {
  const int lane = threadIdx.x & 31;
  return ((1u << tpr) - 1u) << (lane - (lane % tpr));
}
template <int TPR>
__device__ __forceinline__ float bwd_group_max(float v) {
  const unsigned m = bwd_group_mask(TPR);
#pragma unroll
  for (int o = 1; o < TPR; o <<= 1) v = fmaxf(v, __shfl_xor_sync(m, v, o));
  return v;
}
template <int TPR>
__device__ __forceinline__ float bwd_group_sum(float v) {
  const unsigned m = bwd_group_mask(TPR);
#pragma unroll
  for (int o = 1; o < TPR; o <<= 1) v += __shfl_xor_sync(m, v, o);
  return v;
}
template <int TPR>
__device__ __forceinline__ int bwd_group_sum_i32(int v) {
  const unsigned m = bwd_group_mask(TPR);
#pragma unroll
  for (int o = 1; o < TPR; o <<= 1) v += __shfl_xor_sync(m, v, o);
  return v;
}

// Dynamic-smem byte size for both passes: fp16 tiles (sQ,sdO,sK,sV) + fp32 rows
// (Drow,Lse) + int8 tiles (sQ8,sK8 as int32-packed) + per-row scales (sQsc,sKsc).
template <int D>
constexpr int bwd_smem_bytes() {
  constexpr int D4 = D / 4;
  // fp16 tiles + fp32 rows + int8 packed tiles (over D) + per-row scales +
  // Kt8 (K transposed+packed over n, for the int8 dQ = dS'.K dp4a).
  return (2 * BWD_BM + 2 * BWD_BN) * D * (int)sizeof(__half) +
         (2 * BWD_BM) * (int)sizeof(float) +
         (BWD_BM + BWD_BN) * D4 * (int)sizeof(int32_t) +
         (BWD_BM + BWD_BN) * (int)sizeof(float) +
         D * (BWD_BN / 4) * (int)sizeof(int32_t) +   // Kt8 (dq) / dOt8 (dkv), mutually exclusive
         BWD_THREADS * (int)sizeof(float);            // block-reduction scratch
}

// ---------------------------------------------------------------------------
// Per-row symmetric int8 quantize + pack an fp16 tile [R][D] (smem) into an
// int32-packed int8 tile [R][D/4] (smem), writing the per-row fp32 scale
// (amax/127). Little-endian byte order matches the forward kernel's int32
// reinterpret, so __dp4a dots are layout-consistent. Zero-padded ragged rows
// (loaded as 0) quantize to 0 with scale 1. Ends with __syncthreads.
// ---------------------------------------------------------------------------
template <int R, int D>
__device__ __forceinline__ void quant_pack_tile_rowwise(
    const __half (*src)[D], int32_t (*dst)[D / 4], float* row_scale, int tid) {
  constexpr int D4 = D / 4;
  if (tid < R) {
    float a = 0.f;
#pragma unroll
    for (int d = 0; d < D; ++d) a = fmaxf(a, fabsf(__half2float(src[tid][d])));
    row_scale[tid] = (a > 0.f) ? a * (1.f / 127.f) : 1.f;
  }
  __syncthreads();
  for (int idx = tid; idx < R * D4; idx += BWD_THREADS) {
    const int r = idx / D4, c = idx % D4;
    const float inv = 1.f / row_scale[r];
    int32_t packed = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      int qi = __float2int_rn(__half2float(src[r][c * 4 + b]) * inv);
      qi = max(-127, min(127, qi));
      packed |= (qi & 0xff) << (b * 8);
    }
    dst[r][c] = packed;
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// dQ pass: grid (ceil(M/BM), B*H). Lane pair owns one query row; each lane owns
// D/2 of dQ. dQ[m,d] = scale * sum_n dS[m,n] K[n,d].
// ---------------------------------------------------------------------------
template <int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(BWD_THREADS)
bwd_dq_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
              const __half* __restrict__ v, const __half* __restrict__ o,
              const __half* __restrict__ do_, const float* __restrict__ lse,
              __half* __restrict__ dq, float scale, int m_len, int n_len,
              int h_q, int gqa_group) {
  constexpr int COLS = D / BWD_TPR;   // output dims per lane
  constexpr int D4 = D / 4;
  constexpr unsigned FULL = 0xffffffffu;

  extern __shared__ char smem[];
  auto sQ = reinterpret_cast<__half(*)[D]>(smem);
  auto sdO = reinterpret_cast<__half(*)[D]>(smem + BWD_BM * D * sizeof(__half));
  auto sK = reinterpret_cast<__half(*)[D]>(smem + 2 * BWD_BM * D * sizeof(__half));
  auto sV = reinterpret_cast<__half(*)[D]>(smem + (2 * BWD_BM + BWD_BN) * D * sizeof(__half));
  auto sDrow = reinterpret_cast<float*>(smem + (2 * BWD_BM + 2 * BWD_BN) * D * sizeof(__half));
  auto sLse = sDrow + BWD_BM;
  // int8 tiles + per-row scales, appended after the fp32 rows.
  char* i8base = smem + (2 * BWD_BM + 2 * BWD_BN) * D * sizeof(__half) +
                 2 * BWD_BM * sizeof(float);
  auto sQ8 = reinterpret_cast<int32_t(*)[D4]>(i8base);
  auto sK8 = reinterpret_cast<int32_t(*)[D4]>(i8base + BWD_BM * D4 * sizeof(int32_t));
  auto sQsc = reinterpret_cast<float*>(i8base + (BWD_BM + BWD_BN) * D4 * sizeof(int32_t));
  auto sKsc = sQsc + BWD_BM;
  // Kt8: K transposed+packed over keys, [D][BN/4] int32 — the int8 dQ operand.
  auto sKt8 = reinterpret_cast<int32_t(*)[BWD_BN / 4]>(
      reinterpret_cast<char*>(sKsc + BWD_BN));

  constexpr int NPACK = BWD_BN / 4;      // key-packs for the full tile
  constexpr int MYPACK = BWD_HALF / 4;   // packs a lane builds from its BN/TPR keys

  const int tid = threadIdx.x;
  const int row = tid / BWD_TPR;         // query row owned by this lane-group
  const int gid = tid % BWD_TPR;         // lane index within the group (0..TPR-1)
  const int gbase = (tid & 31) - ((tid & 31) % BWD_TPR);  // group's base warp-lane
  const int m_block = blockIdx.x * BWD_BM;
  const int64_t bh = blockIdx.y;                         // over B * H_q
  // GQA: dQ is per Q head; K/V come from the shared K/V head.
  const int64_t kv_bh = (bh / h_q) * (h_q / gqa_group) + (bh % h_q) / gqa_group;

  const __half* q_bh = q + bh * (int64_t)m_len * D;
  const __half* o_bh = o + bh * (int64_t)m_len * D;
  const __half* do_bh = do_ + bh * (int64_t)m_len * D;
  const __half* k_bh = k + kv_bh * (int64_t)n_len * D;
  const __half* v_bh = v + kv_bh * (int64_t)n_len * D;
  const float* lse_bh = lse + bh * m_len;

  for (int idx = tid; idx < BWD_BM * D; idx += BWD_THREADS) {
    const int r = idx / D, c = idx % D;
    const int gm = m_block + r;
    sQ[r][c] = (gm < m_len) ? q_bh[(int64_t)gm * D + c] : __half(0.f);
    sdO[r][c] = (gm < m_len) ? do_bh[(int64_t)gm * D + c] : __half(0.f);
  }
  __syncthreads();
  // int8-quantize the (resident) Q tile once for the S=Q.K^T dp4a recompute.
  quant_pack_tile_rowwise<BWD_BM, D>(sQ, sQ8, sQsc, tid);
  {
    const int d0 = gid * COLS;
    float acc = 0.f;
    const int gm = m_block + row;
    if (gm < m_len)
      for (int i = 0; i < COLS; ++i) {
        const int d = d0 + i;
        acc += __half2float(sdO[row][d]) * __half2float(o_bh[(int64_t)gm * D + d]);
      }
    acc = bwd_group_sum<BWD_TPR>(acc);  // full-D row-dot across the group
    if (gid == 0) {
      sDrow[row] = acc;
      sLse[row] = (gm < m_len) ? lse_bh[gm] : 0.f;
    }
  }
  __syncthreads();

  const int my_m = m_block + row;
  float dq_acc[COLS];
#pragma unroll
  for (int i = 0; i < COLS; ++i) dq_acc[i] = 0.f;
  const float my_lse = sLse[row];
  const float my_D = sDrow[row];
  const float my_qsc = sQsc[row];

  const int n_end = IS_CAUSAL ? min(n_len, m_block + BWD_BM) : n_len;
  for (int n_block = 0; n_block < n_end; n_block += BWD_BN) {
    for (int idx = tid; idx < BWD_BN * D; idx += BWD_THREADS) {
      const int r = idx / D, c = idx % D;
      const int gn = n_block + r;
      sK[r][c] = (gn < n_len) ? k_bh[(int64_t)gn * D + c] : __half(0.f);
      sV[r][c] = (gn < n_len) ? v_bh[(int64_t)gn * D + c] : __half(0.f);
    }
    __syncthreads();
    // int8-quantize this K tile for the S dp4a; sV stays fp16 for dP.
    quant_pack_tile_rowwise<BWD_BN, D>(sK, sK8, sKsc, tid);

    float ds_row[BWD_HALF];
#pragma unroll
    for (int j = 0; j < BWD_HALF; ++j) {
      const int col = gid * BWD_HALF + j;
      const int gn = n_block + col;
      const bool valid = (gn < n_len) && (!IS_CAUSAL || gn <= my_m);
      if (!valid) { ds_row[j] = 0.f; continue; }
      int32_t sacc = 0;  // S = Q.K^T via int8 dp4a (contract over d)
#pragma unroll
      for (int c = 0; c < D4; ++c) sacc = __dp4a(sQ8[row][c], sK8[col][c], sacc);
      const float s = (float)sacc * my_qsc * sKsc[col];
      float dp = 0.f;  // dP = dO.V^T stays fp (the sensitive matmul)
      for (int d = 0; d < D; ++d)
        dp += __half2float(sdO[row][d]) * __half2float(sV[col][d]);
      const float p = exp2f((s * scale - my_lse) * BWD_LOG2E);
      // dS' = dS * sKsc[col]: fold the per-key K scale in so dQ = dS'.Kq
      // (raw int8 K) factors cleanly for the int8 dp4a below.
      ds_row[j] = p * (dp - my_D) * sKsc[col];
    }

    // ---- int8 dQ = dS'.K via dp4a (dS' per-row int8, K transposed+packed) ----
    // stage Kt8: K transposed+packed over keys, quantized with sKsc (=raw int8 Kq)
    for (int idx = tid; idx < D * NPACK; idx += BWD_THREADS) {
      const int d = idx / NPACK, kp = idx % NPACK;
      int32_t packed = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int nn = kp * 4 + b;
        int qi = __float2int_rn(__half2float(sK[nn][d]) / sKsc[nn]);
        qi = max(-127, min(127, qi));
        packed |= (qi & 0xff) << (b * 8);
      }
      sKt8[d][kp] = packed;
    }
    // per-row dS' scale (rowmax over all 64 keys via the lane pair)
    float pmax = 0.f;
#pragma unroll
    for (int j = 0; j < BWD_HALF; ++j) pmax = fmaxf(pmax, fabsf(ds_row[j]));
    pmax = bwd_group_max<BWD_TPR>(pmax);   // rowmax over all keys (the group)
    const float sds = fmaxf(pmax * (1.f / 127.f), 2e-38f);
    const float inv_sds = 1.f / sds;
    // quantize my BN/TPR dS' -> MYPACK int32 packs of my keys
    int32_t my_pk[MYPACK];
#pragma unroll
    for (int pk = 0; pk < MYPACK; ++pk) {
      int32_t packed = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        int qi = __float2int_rn(ds_row[pk * 4 + b] * inv_sds);
        qi = max(-127, min(127, qi));
        packed |= (qi & 0xff) << (b * 8);
      }
      my_pk[pk] = packed;
    }
    // gather all NPACK key-packs from the TPR group (lane sg owns packs
    // [sg*MYPACK .. +MYPACK), covering keys [sg*BWD_HALF ..) in order).
    int32_t packs[NPACK];
#pragma unroll
    for (int sg = 0; sg < BWD_TPR; ++sg)
#pragma unroll
      for (int pk = 0; pk < MYPACK; ++pk)
        packs[sg * MYPACK + pk] = __shfl_sync(FULL, my_pk[pk], gbase + sg);
    __syncthreads();  // sKt8 ready for all lanes
    const int d0 = gid * COLS;
#pragma unroll
    for (int i = 0; i < COLS; ++i) {
      const int d = d0 + i;
      int32_t acc = 0;
#pragma unroll
      for (int pk = 0; pk < NPACK; ++pk) acc = __dp4a(packs[pk], sKt8[d][pk], acc);
      dq_acc[i] += sds * (float)acc;   // dequant per tile; epilogue applies *scale
    }
    __syncthreads();
  }

  if (my_m < m_len) {
    __half* dq_row = dq + bh * (int64_t)m_len * D + (int64_t)my_m * D;
#pragma unroll
    for (int i = 0; i < COLS; ++i)
      dq_row[gid * COLS + i] = __float2half(dq_acc[i] * scale);
  }
}

// ---------------------------------------------------------------------------
// dK/dV pass: grid (ceil(N/BN), B*H). Lane pair owns one key row; each lane owns
// D/2 of dK and dV. dV[n,d]=sum_m P[m,n] dO[m,d]; dK[n,d]=scale*sum_m dS[m,n] Q[m,d].
// ---------------------------------------------------------------------------
template <int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(BWD_THREADS)
bwd_dkv_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
               const __half* __restrict__ v, const __half* __restrict__ o,
               const __half* __restrict__ do_, const float* __restrict__ lse,
               __half* __restrict__ dk, __half* __restrict__ dv,
               float scale, int m_len, int n_len, int h_q, int gqa_group) {
  constexpr int COLS = D / BWD_TPR;   // output dims per lane
  constexpr int D4 = D / 4;

  extern __shared__ char smem[];
  auto sK = reinterpret_cast<__half(*)[D]>(smem);
  auto sV = reinterpret_cast<__half(*)[D]>(smem + BWD_BN * D * sizeof(__half));
  auto sQ = reinterpret_cast<__half(*)[D]>(smem + 2 * BWD_BN * D * sizeof(__half));
  auto sdO = reinterpret_cast<__half(*)[D]>(smem + (2 * BWD_BN + BWD_BM) * D * sizeof(__half));
  auto sDrow = reinterpret_cast<float*>(smem + (2 * BWD_BN + 2 * BWD_BM) * D * sizeof(__half));
  auto sLse = sDrow + BWD_BM;
  char* i8base = smem + (2 * BWD_BN + 2 * BWD_BM) * D * sizeof(__half) +
                 2 * BWD_BM * sizeof(float);
  auto sK8 = reinterpret_cast<int32_t(*)[D4]>(i8base);
  auto sQ8 = reinterpret_cast<int32_t(*)[D4]>(i8base + BWD_BN * D4 * sizeof(int32_t));
  auto sKsc = reinterpret_cast<float*>(i8base + (BWD_BN + BWD_BM) * D4 * sizeof(int32_t));
  auto sQsc = sKsc + BWD_BN;
  // dOt8: dO transposed+packed over m, [D][BM/4] int32 — the int8 dV operand.
  // Reuses the region the dq pass calls Kt8 (mutually exclusive). sRed follows.
  auto sdOt8 = reinterpret_cast<int32_t(*)[BWD_BM / 4]>(
      reinterpret_cast<char*>(sQsc + BWD_BM));
  auto sRed = reinterpret_cast<float*>(
      reinterpret_cast<char*>(sdOt8) + D * (BWD_BM / 4) * sizeof(int32_t));

  constexpr int MPACK = BWD_BM / 4;   // query-packs for one Q tile (16)

  const int tid = threadIdx.x;
  const int krow = tid / BWD_TPR;    // key row owned by this lane-group
  const int gid = tid % BWD_TPR;     // lane index within the group
  const int n_block = blockIdx.x * BWD_BN;
  const int64_t bh_kv = blockIdx.y;                  // over B * H_kv
  const int h_kv = h_q / gqa_group;
  const int64_t bat = bh_kv / h_kv;                  // batch index
  const int hkv = (int)(bh_kv % h_kv);

  // K/V (and dK/dV output) belong to this KV head.
  const __half* k_bh = k + bh_kv * (int64_t)n_len * D;
  const __half* v_bh = v + bh_kv * (int64_t)n_len * D;

  for (int idx = tid; idx < BWD_BN * D; idx += BWD_THREADS) {
    const int r = idx / D, c = idx % D;
    const int gn = n_block + r;
    sK[r][c] = (gn < n_len) ? k_bh[(int64_t)gn * D + c] : __half(0.f);
    sV[r][c] = (gn < n_len) ? v_bh[(int64_t)gn * D + c] : __half(0.f);
  }
  __syncthreads();
  // int8-quantize the owned K tile once for the S=Q.K^T dp4a recompute.
  quant_pack_tile_rowwise<BWD_BN, D>(sK, sK8, sKsc, tid);

  const int my_n = n_block + krow;
  float dk_acc[COLS], dv_acc[COLS];
#pragma unroll
  for (int i = 0; i < COLS; ++i) { dk_acc[i] = 0.f; dv_acc[i] = 0.f; }
  const float my_ksc = sKsc[krow];

  // GQA: sum dK/dV over the gqa_group query heads sharing this K/V head.
  for (int qg = 0; qg < gqa_group; ++qg) {
    const int64_t bh_q = bat * h_q + (int64_t)hkv * gqa_group + qg;
    const __half* q_bh = q + bh_q * (int64_t)m_len * D;
    const __half* o_bh = o + bh_q * (int64_t)m_len * D;
    const __half* do_bh = do_ + bh_q * (int64_t)m_len * D;
    const float* lse_bh = lse + bh_q * m_len;

  const int m_start = IS_CAUSAL ? (n_block / BWD_BM) * BWD_BM : 0;
  for (int m_block = m_start; m_block < m_len; m_block += BWD_BM) {
    for (int idx = tid; idx < BWD_BM * D; idx += BWD_THREADS) {
      const int r = idx / D, c = idx % D;
      const int gm = m_block + r;
      sQ[r][c] = (gm < m_len) ? q_bh[(int64_t)gm * D + c] : __half(0.f);
      sdO[r][c] = (gm < m_len) ? do_bh[(int64_t)gm * D + c] : __half(0.f);
    }
    for (int r = tid; r < BWD_BM; r += BWD_THREADS) {
      const int gm = m_block + r;
      float acc = 0.f;
      if (gm < m_len)
        for (int d = 0; d < D; ++d)
          acc += __half2float(sdO[r][d]) * __half2float(o_bh[(int64_t)gm * D + d]);
      sDrow[r] = acc;
      sLse[r] = (gm < m_len) ? lse_bh[gm] : 0.f;
    }
    __syncthreads();
    // int8-quantize this Q tile for the S dp4a; sQ stays fp16 for dK=dS^T.Q (fp).
    quant_pack_tile_rowwise<BWD_BM, D>(sQ, sQ8, sQsc, tid);

    // Per-tile dO absmax -> scale sdO (block reduction), then stage dOt8
    // (dO transposed+packed over m) for the int8 dV = P^T.dO dp4a.
    {
      float loc = 0.f;
      for (int idx = tid; idx < BWD_BM * D; idx += BWD_THREADS)
        loc = fmaxf(loc, fabsf(__half2float(sdO[idx / D][idx % D])));
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) loc = fmaxf(loc, __shfl_xor_sync(0xffffffffu, loc, o));
      if ((tid & 31) == 0) sRed[tid >> 5] = loc;
      __syncthreads();
      if (tid == 0) {
        float m = 0.f;
        for (int w = 0; w < BWD_THREADS / 32; ++w) m = fmaxf(m, sRed[w]);
        sRed[0] = m;
      }
      __syncthreads();
    }
    const float sdO_scale = fmaxf(sRed[0] * (1.f / 127.f), 2e-38f);
    const float inv_sdO = 1.f / sdO_scale;
    for (int idx = tid; idx < D * MPACK; idx += BWD_THREADS) {
      const int d = idx / MPACK, mp = idx % MPACK;
      int32_t packed = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        int qi = __float2int_rn(__half2float(sdO[mp * 4 + b][d]) * inv_sdO);
        qi = max(-127, min(127, qi));
        packed |= (qi & 0xff) << (b * 8);
      }
      sdOt8[d][mp] = packed;
    }
    __syncthreads();

    // Pass 1 (cheap: int8 S only, no dP) — per-key P scale = max_m P[m,krow]/127.
    // A fixed 1/127 scale would zero the softmax tail (most P ~ 1/N); the per-key
    // max preserves it. P depends only on S + LSE, so dP is NOT needed here.
    // The TPR lanes of a key SPLIT the S/dP dots over D (not redundant recompute)
    // and group-reduce — so raising TPR raises occupancy at ~constant work.
    constexpr int D4_PER = D4 / BWD_TPR;   // int8 S dp4a terms per lane
    constexpr int D_PER = D / BWD_TPR;     // fp dP dot terms per lane (== COLS)
    const int d0 = gid * COLS;
    const int c0 = gid * D4_PER;

    float pmax = 0.f;
    for (int mm = 0; mm < BWD_BM; ++mm) {
      const int gm = m_block + mm;
      if (!((gm < m_len) && (my_n < n_len) && (!IS_CAUSAL || my_n <= gm))) continue;
      int32_t sacc = 0;  // my D-slice of S = Q.K^T
#pragma unroll
      for (int c = 0; c < D4_PER; ++c) sacc = __dp4a(sQ8[mm][c0 + c], sK8[krow][c0 + c], sacc);
      sacc = bwd_group_sum_i32<BWD_TPR>(sacc);   // full S across the group
      const float s = (float)sacc * sQsc[mm] * my_ksc;
      pmax = fmaxf(pmax, exp2f((s * scale - sLse[mm]) * BWD_LOG2E));
    }
    const float sP = fmaxf(pmax * (1.f / 127.f), 2e-38f);
    const float inv_sP = 1.f / sP;
    const float dv_scale = sP * sdO_scale;   // dV = sP*sdO * dp4a(Pq, dOq)

    // Pass 2: stream query-packs of 4. dV int8 dp4a; dK stays fp (next step).
    for (int mp = 0; mp < MPACK; ++mp) {
      int32_t p_pack = 0;
      float ds4[4];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int mm = mp * 4 + b;
        const int gm = m_block + mm;
        // validity is group-uniform (same krow/mm) -> group-reduces stay in sync
        const bool valid = (gm < m_len) && (my_n < n_len) && (!IS_CAUSAL || my_n <= gm);
        float p = 0.f, ds = 0.f;
        if (valid) {
          int32_t sacc = 0;  // my D-slice of S
#pragma unroll
          for (int c = 0; c < D4_PER; ++c) sacc = __dp4a(sQ8[mm][c0 + c], sK8[krow][c0 + c], sacc);
          sacc = bwd_group_sum_i32<BWD_TPR>(sacc);
          const float s = (float)sacc * sQsc[mm] * my_ksc;
          float dp = 0.f;  // my D-slice of dP = dO.V^T (stays fp)
          for (int d = 0; d < D_PER; ++d)
            dp += __half2float(sdO[mm][d0 + d]) * __half2float(sV[krow][d0 + d]);
          dp = bwd_group_sum<BWD_TPR>(dp);
          p = exp2f((s * scale - sLse[mm]) * BWD_LOG2E);
          ds = p * (dp - sDrow[mm]);
        }
        int pq = max(0, min(127, __float2int_rn(p * inv_sP)));
        p_pack |= (pq & 0xff) << (b * 8);
        ds4[b] = ds;
      }
      // dV += (sdO/127) * dp4a(Pq_pack, dOt8[d][mp]); dK += ds*Q (fp scalar)
#pragma unroll
      for (int i = 0; i < COLS; ++i)
        dv_acc[i] += dv_scale * (float)__dp4a(p_pack, sdOt8[d0 + i][mp], 0);
#pragma unroll
      for (int b = 0; b < 4; ++b)
        if (ds4[b] != 0.f)
          for (int i = 0; i < COLS; ++i)
            dk_acc[i] += ds4[b] * __half2float(sQ[mp * 4 + b][d0 + i]);
    }
    __syncthreads();
  }  // m-loop
  }  // qg-loop (GQA: accumulated dK/dV over the group's query heads)

  if (my_n < n_len) {
    __half* dk_row = dk + bh_kv * (int64_t)n_len * D + (int64_t)my_n * D;
    __half* dv_row = dv + bh_kv * (int64_t)n_len * D + (int64_t)my_n * D;
#pragma unroll
    for (int i = 0; i < COLS; ++i) {
      dk_row[gid * COLS + i] = __float2half(dk_acc[i] * scale);
      dv_row[gid * COLS + i] = __float2half(dv_acc[i]);
    }
  }
}

}  // namespace fni8
