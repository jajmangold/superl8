// SPDX-License-Identifier: BSD-3-Clause
// ===========================================================================
// IQ3_XXS decode — warp-per-column MMVQ over ggml's 256-entry grid.
// Block (98 B / 256 weights, 3.0625 bpw):
//     +0   d        fp16 super-block scale
//     +2   qs[64]   8-bit grid indices, 8 per 32-weight sub-block
//     +66  scales_and_signs: ONE uint32 per sub-block (8 x 4 B)
// Per sub-block ib32 with aux = scales_and_signs[ib32]:
//     db    = d * (0.5f + (aux >> 28)) * 0.5f      // top-nibble, half-offset scale
//     signs = ksigns_iq2xs[(aux >> 7*l) & 127]     // l = p/2, a 128-entry LUT
//     value j of entry p flips when (signs & kmask[(p%2)*4 + j])
// Differs from IQ3_S three ways: top-nibble half-offset scale (not 1+2s), a plain
// 8-bit grid index (no qh 9th bit), and sign bits via a lookup instead of stored
// bytes. Grid magnitudes are small (<= 0x3e), so sign-flipped bytes stay in int8
// range and feed __dp4a directly. Oracle: fni8.quant.iq3xxs (fni8#317).
// Adapted (MIT->BSD-3) from llama.cpp vec_dot_iq3_xxs_q8_1 + dequantize_row_iq3_xxs.
// ===========================================================================
#pragma once

#include "gemm_decode_dp4a.cuh"

namespace fni8 {

constexpr int IQ3XXS_TYPE_SIZE = 98;

// NOT __constant__: divergent lanes serialize on constant memory (fni8#334).
__device__ uint32_t kIq3xxsGrid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

// NOT __constant__: only 128 B, but constant memory serializes on distinct
// ADDRESSES, not cache lines, and this index is data-dependent (fni8#334).
__device__ uint8_t kSignsIq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_iq3xxs_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                          const uint8_t* __restrict__ w, OutT* __restrict__ out,
                          int M, int N, int K, int num_sb) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;
  if (n >= N) return;
  const int nsub = K / 32;
  const int64_t row_bytes = (int64_t)num_sb * IQ3XXS_TYPE_SIZE;
  const uint8_t* wrow = w + (int64_t)n * row_bytes;

  float f_acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) f_acc[m] = 0.f;

  for (int i = lane; i < nsub; i += DEC_WARP) {
    const int sb = i >> 3, ib32 = i & 7;
    const uint8_t* blk = wrow + (int64_t)sb * IQ3XXS_TYPE_SIZE;
    const uint8_t* qs = blk + 2 + 8 * ib32;
    const uint32_t aux = dec_ld_u32(blk + 66 + 4 * ib32);
    const float db = __half2float(*reinterpret_cast<const __half*>(blk)) *
                     (0.5f + (float)(aux >> 28)) * 0.5f;

    int32_t q[8];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint32_t g = __ldg(&kIq3xxsGrid[qs[p]]);
      const int sgn = __ldg(&kSignsIq2xs[(aux >> (7 * (p >> 1))) & 127]);
      const int mb = sgn >> ((p & 1) * 4);             // 4 sign bits for this entry
      const uint32_t sbits = (uint32_t)(mb & 1) | ((uint32_t)((mb >> 1) & 1) << 8) |
                             ((uint32_t)((mb >> 2) & 1) << 16) | ((uint32_t)((mb >> 3) & 1) << 24);
      const int neg = __vcmpne4(sbits, 0x00000000u);
      q[p] = __vsub4((int)(g ^ (uint32_t)neg), neg);
    }
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m >= M) break;
      const int4* xr = reinterpret_cast<const int4*>(x + (int64_t)m * K);
      const int4 xv0 = xr[i * 2], xv1 = xr[i * 2 + 1];
      int32_t si = 0;
      si = __dp4a(xv0.x, q[0], si);
      si = __dp4a(xv0.y, q[1], si);
      si = __dp4a(xv0.z, q[2], si);
      si = __dp4a(xv0.w, q[3], si);
      si = __dp4a(xv1.x, q[4], si);
      si = __dp4a(xv1.y, q[5], si);
      si = __dp4a(xv1.z, q[6], si);
      si = __dp4a(xv1.w, q[7], si);
      f_acc[m] += db * (float)si;
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
