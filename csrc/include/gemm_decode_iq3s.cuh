// SPDX-License-Identifier: BSD-3-Clause
// ===========================================================================
// IQ3_S decode — warp-per-column MMVQ over ggml's 512-entry codebook grid.
// Block (110 B / 256 weights, 3.4375 bpw):
//     +0   d        fp16 super-block scale
//     +2   qs[64]   low 8 bits of the grid index (one byte per 4 weights)
//     +66  qh[8]    one byte per 32-weight sub-block; bit p is the 9th index bit
//     +74  signs[32] one bit per weight (4 bytes per sub-block)
//     +106 scales[4] two 4-bit fields per byte; sub-block scale is 1 + 2*s
// For position p in 0..7 of sub-block ib32:
//     idx  = qs[8*ib32 + p] | ((qh[ib32] << (8 - p)) & 256)
//     sign = signs[4*ib32 + p/2] bit ((p%2)*4 + j) flips magnitude j
// Grid entries are 4 packed uint8 magnitudes (max 15), so the sign-flipped bytes
// stay in int8 range and feed __dp4a directly. Oracle: fni8.quant.iq3s (fni8#317).
// Adapted (MIT->BSD-3) from llama.cpp vec_dot_iq3_s_q8_1 + dequantize_row_iq3_s.
// ===========================================================================
#pragma once

#include "gemm_decode_dp4a.cuh"

namespace fni8 {

constexpr int IQ3S_TYPE_SIZE = 110;

// ggml-common.h iq3s_grid (512 x uint32 = 4 packed uint8 magnitudes each).
// NOT __constant__: divergent lanes serialize on constant memory. IQ3_S is 30.9%
// of Qwen3.8-27B-UD-IQ3_S's weights, so its 1.61x is the largest absolute win
// despite being the smallest ratio (fni8#334).
__device__ uint32_t kIq3sGrid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_iq3s_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                        const uint8_t* __restrict__ w, OutT* __restrict__ out,
                        int M, int N, int K, int num_sb) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;

  // Stage the 512-entry (2 KiB) grid in shared memory. Post-widening, LG throttle is
  // still the top stall (12.9 cycles, est 56.4%) and the grid gathers are most of the
  // remaining global-memory instructions; shared reads issue on LDS, not the LG queue.
  // Static __shared__ on purpose: DECODE_KQUANT_LAUNCH passes 0 dynamic smem for all
  // 13 k-quant kernels, so this needs no launcher change.
  // ORDER MATTERS: stage and sync BEFORE the `n >= N` early return -- tail blocks have
  // warps with n >= N (N=70 in tests is not a multiple of DEC_WARPS_PER_BLOCK=4), and
  // a __syncthreads() those warps never reach is a hang, not a slowdown.
  __shared__ uint32_t s_iq3s_grid[512];
  for (int t = threadIdx.x; t < 512; t += DEC_THREADS) s_iq3s_grid[t] = kIq3sGrid[t];
  __syncthreads();

  if (n >= N) return;
  const int nsub = K / 32;                       // 32-weight sub-blocks along K
  const int64_t row_bytes = (int64_t)num_sb * IQ3S_TYPE_SIZE;
  const uint8_t* wrow = w + (int64_t)n * row_bytes;

  float f_acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) f_acc[m] = 0.f;

  for (int i = lane; i < nsub; i += DEC_WARP) {
    const int sb = i >> 3, ib32 = i & 7;         // super-block, sub-block within it
    const uint8_t* blk = wrow + (int64_t)sb * IQ3S_TYPE_SIZE;
    const uint8_t* qs = blk + 2 + 8 * ib32;
    const int qh = blk[66 + ib32];
    const uint8_t* sg = blk + 74 + 4 * ib32;
    const int sc = (blk[106 + (ib32 >> 1)] >> (4 * (ib32 & 1))) & 0xF;
    const float dsc = __half2float(*reinterpret_cast<const __half*>(blk)) * (float)(1 + 2 * sc);

    // LG THROTTLE (67.3% of stall cycles, fni8#334): the per-element scalar gather
    // below issued 8 qs byte loads + 4 sg byte loads per sub-block. qs[0..7] and
    // sg[0..3] are each contiguous, so fold them into three 32-bit loads. Same fix
    // Q4_K/Q6_K already carry -- see dec_ld_u32's use there, whose comment records
    // that the scalar gather made Q6_K decode ~2x slower than Q3_K/Q5_K.
    // Alignment: qs = blk+2+8*ib32 and sg = blk+74+4*ib32 are both even, so the
    // uint16-pair reads inside dec_ld_u32 are safe.
    const uint32_t qs_lo = dec_ld_u32(qs);       // qs[0..3]
    const uint32_t qs_hi = dec_ld_u32(qs + 4);   // qs[4..7]
    const uint32_t sg_w  = dec_ld_u32(sg);       // sg[0..3]

    int32_t q[8];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint32_t qs_p = ((p < 4 ? qs_lo : qs_hi) >> (8 * (p & 3))) & 0xFFu;
      const uint32_t g = s_iq3s_grid[qs_p | ((qh << (8 - p)) & 256)];
      const int mb = (int)((sg_w >> (8 * (p >> 1))) & 0xFFu) >> ((p & 1) * 4);
      const uint32_t sbits = (uint32_t)(mb & 1) | ((uint32_t)((mb >> 1) & 1) << 8) |
                             ((uint32_t)((mb >> 2) & 1) << 16) | ((uint32_t)((mb >> 3) & 1) << 24);
      const int neg = __vcmpne4(sbits, 0x00000000u);        // 0xFF per byte to negate
      q[p] = __vsub4((int)(g ^ (uint32_t)neg), neg);        // (b^0xFF) - (-1) = -b
    }
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m >= M) break;
      const int4* xr = reinterpret_cast<const int4*>(x + (int64_t)m * K);
      const int4 xv0 = xr[i * 2], xv1 = xr[i * 2 + 1];      // 32 int8 = 8 dp4a words
      int32_t si = 0;
      si = __dp4a(xv0.x, q[0], si);
      si = __dp4a(xv0.y, q[1], si);
      si = __dp4a(xv0.z, q[2], si);
      si = __dp4a(xv0.w, q[3], si);
      si = __dp4a(xv1.x, q[4], si);
      si = __dp4a(xv1.y, q[5], si);
      si = __dp4a(xv1.z, q[6], si);
      si = __dp4a(xv1.w, q[7], si);
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

}  // namespace fni8
