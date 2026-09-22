// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Native GGUF Q3_K ROW-GATHER dequant (token embeddings) for sm_70.
//
//   out[i, :] = dequant(w[ids[i]])     w RESIDENT in native Q3_K super-blocks
//
// An embedding table is a pure GATHER, never a GEMM (the LM head is a separate
// tensor), so there is no dp4a to fuse into -- the only thing that matters is
// that the table stays in its native bytes. Dequantizing + requantizing
// `token_embd.weight` to a per_row_i8 table just to index it costs ~2.3x the
// native footprint RESIDENT (Qwen3.8-27B-UD-IQ3_S: 0.5088 GiB of Q3_K
// token_embd -> 1.1850 GiB i8, a +0.6762 GiB expansion on a 16 GiB card).
// This kernel indexes the Q3_K bytes directly, so that expansion never happens.
//
// Block layout (110 B): hmask[32] qs[64] scales[12] d  (ggml-common.h:396-402).
// 16 sub-blocks of 16 values; code = (qs 2-bit) | (hmask bit << 2), centered -4
// -> [-4,3]; signed 6-bit per-16 scale from the 12 packed `scales` bytes;
// y = d * sc * (q - 4). SYMMETRIC, no min. The sub-block/bit addressing below is
// the SAME arithmetic as the tile `gemm_q3k_kernel` (gemm_q4k_dp4a.cuh:509-566)
// and the warp-per-column `gemm_decode_q3k_kernel` (gemm_decode_dp4a.cuh:689),
// and it reuses their `q3k_scale` helper verbatim -- one decode, three kernels.
//
// Shape: one thread block per gathered row; 128 threads, each owning a PAIR of
// consecutive weights, so a block covers one 256-weight super-block per step.
// The pair is the point: both halves live in the same 16-value sub-block (same
// scale, same qs shift, adjacent bytes) and land on adjacent outputs, so the
// epilogue is one packed `pair_store` -- the healthy half2 CUDA-core pipe
// (~27 TFLOP/s, a SEPARATE pipe from this fleet's firmware-gimped tensor cores,
// AGENTS.md:16-29), and half the stores. No wmma/cp.async/ldmatrix anywhere.
// qs/hmask are read as scalar bytes: a row is ~2 KB and this op is bandwidth-
// trivial next to the GEMMs, so correct-and-simple wins over a packed unpack.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "compute_dtype.cuh"        // pair_store / zero_val for the fp16 epilogue
#include "gemm_q4k_dp4a.cuh"        // q3k_scale + Q3K_TYPE_SIZE (the SAME decode)

namespace fni8 {

constexpr int GATHER_Q3K_QK = 256;                    // weights per super-block
constexpr int GATHER_Q3K_PAIR = 2;                    // weights per thread (half2 store)
constexpr int GATHER_Q3K_THREADS = GATHER_Q3K_QK / GATHER_Q3K_PAIR;   // 128

// ids [n] int64, w [N, num_sb*110] uint8 native Q3_K, out [n, K] OutT.
// grid = (n), block = 128. An id outside [0,N) yields a ZERO row -- the table is
// half a gigabyte, so a bad id must never become an out-of-bounds read, and
// raising would cost a host sync on the decode path.
template <typename OutT>
__global__ void __launch_bounds__(GATHER_Q3K_THREADS)
gather_q3k_kernel(const int64_t* __restrict__ ids, const uint8_t* __restrict__ w,
                  OutT* __restrict__ out, int64_t N, int K, int num_sb) {
  const int64_t row = blockIdx.x;
  OutT* __restrict__ orow = out + row * (int64_t)K;
  const int64_t id = ids[row];
  if (id < 0 || id >= N) {
    for (int e = threadIdx.x * GATHER_Q3K_PAIR; e < K; e += GATHER_Q3K_QK)
      pair_store<OutT>(orow + e, 0.f, 0.f);
    return;
  }
  const uint8_t* __restrict__ wrow = w + id * (int64_t)num_sb * Q3K_TYPE_SIZE;

  // Linear position within a super-block is `is*16 + i16`, so thread t owns
  // sub-block is = t/8 at the even position i16 = 2*(t%8), plus its neighbour
  // i16+1. is -> (qs, hmask) addressing is verbatim gemm_q3k_kernel: the 2 low
  // bits live in qs[32*g + (sig&1)*16 + i16] at shift `sig&6`, the 3rd bit in
  // hmask[(sig&1)*16 + i16] at bit g*4 + sig/2. Both halves of the pair share
  // every one of those (only i16 differs by 1), so the scale is decoded once.
  const int is = threadIdx.x >> 3, i16 = (threadIdx.x & 7) * GATHER_Q3K_PAIR;
  const int g_ = is >> 3, sig = is & 7, shift = sig & 6;
  const int m_shift = g_ * 4 + (sig >> 1);
  const int qs_off = 32 + 32 * g_ + (sig & 1) * 16 + i16;   // qs[64] starts at byte 32
  const int hm_off = (sig & 1) * 16 + i16;                  // hmask[32] starts at byte 0
  const int out_off = is * 16 + i16;                        // position within the super-block

  for (int sb = 0; sb < num_sb; ++sb) {
    const uint8_t* __restrict__ blk = wrow + (int64_t)sb * Q3K_TYPE_SIZE;
    // Row bytes and 110 are both even, so the fp16 `d` at +108 stays 2-B aligned.
    const float dsc = __half2float(*reinterpret_cast<const __half*>(blk + 108))
                    * (float)q3k_scale(blk + 96, is);
    const int lo0 = (blk[qs_off] >> shift) & 3, hb0 = (blk[hm_off] >> m_shift) & 1;
    const int lo1 = (blk[qs_off + 1] >> shift) & 3, hb1 = (blk[hm_off + 1] >> m_shift) & 1;
    pair_store<OutT>(orow + sb * GATHER_Q3K_QK + out_off,
                     dsc * (float)((lo0 | (hb0 << 2)) - 4),
                     dsc * (float)((lo1 | (hb1 << 2)) - 4));
  }
}

}  // namespace fni8
