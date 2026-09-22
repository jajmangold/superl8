// SPDX-License-Identifier: BSD-3-Clause
// ===========================================================================
// IQ2_XXS decode -- warp-per-column MMVQ over ggml's 256-entry grid.
// Block (66 B / 256 weights, 2.0625 bpw):
//     +0   d        fp16 super-block scale
//     +2   qs[64]   EIGHT sub-blocks x TWO uint32 (aux0, aux1)
// Per 32-weight sub-block ib32, with aux0/aux1 the pair at +2 + 8*ib32:
//     aux0  = FOUR 8-bit grid indices, one per byte; each entry is 8 values
//     db    = d * (0.5f + (aux1 >> 28)) * 0.25f     // top-nibble, quarter-offset
//     signs = ksigns_iq2xs[(aux1 >> 7*l) & 127]     // l = entry 0..3
//     the sign byte's 8 bits apply to that entry's 8 values
// Unlike IQ3_XXS (8 entries x 4 values), IQ2_XXS packs 4 entries x 8 values per
// sub-block, so each grid entry spans TWO dp4a words: sign bits 0-3 drive word 0
// and bits 4-7 drive word 1. Grid magnitudes are {8, 25, 43}, so sign-flipped
// bytes stay well inside int8 and feed __dp4a directly.
// The 66-B block puts every uint32 at an offset == 2 (mod 4), so the aux words are
// 2-byte aligned ONLY -- they must go through dec_ld_u32, never a u32 load.
// Oracle: fni8.quant.iq2xxs (fni8#317). ksigns_iq2xs is REUSED from the IQ3_XXS
// header rather than duplicated (duplicated tables have caused six bugs today).
// Adapted (MIT->BSD-3) from llama.cpp vec_dot_iq2_xxs_q8_1.
// ===========================================================================
#pragma once

#include "gemm_decode_dp4a.cuh"
#include "gemm_decode_iq3xxs.cuh"   // kSignsIq2xs[128], shared with IQ3_XXS

namespace fni8 {

constexpr int IQ2XXS_TYPE_SIZE = 66;

// 256 entries x 8 int8 magnitudes, packed as two dp4a words (values 0-3, 4-7).
// Generated from fni8.quant.iq2xxs._GRID; 256 * 2 * 4 B = 2048 B of constant memory.
// NOT __constant__: divergent lanes serialize on constant memory (fni8#334).
__device__ uint32_t kIq2xxsGrid[256][2] = {
    {0x08080808, 0x08080808}, {0x0808082b, 0x08080808}, {0x08081919, 0x08080808}, {0x08082b08, 0x08080808},
    {0x08082b2b, 0x08080808}, {0x08190819, 0x08080808}, {0x08191908, 0x08080808}, {0x082b0808, 0x08080808},
    {0x082b082b, 0x08080808}, {0x082b2b08, 0x08080808}, {0x082b2b2b, 0x08080808}, {0x19080819, 0x08080808},
    {0x19081908, 0x08080808}, {0x19190808, 0x08080808}, {0x19192b08, 0x08080808}, {0x192b0819, 0x08080808},
    {0x192b1908, 0x08080808}, {0x2b080808, 0x08080808}, {0x2b08082b, 0x08080808}, {0x2b082b2b, 0x08080808},
    {0x2b2b082b, 0x08080808}, {0x08080819, 0x08080819}, {0x08081908, 0x08080819}, {0x08190808, 0x08080819},
    {0x08191919, 0x08080819}, {0x19080808, 0x08080819}, {0x2b081908, 0x08080819}, {0x2b192b08, 0x08080819},
    {0x08080808, 0x0808082b}, {0x0808082b, 0x0808082b}, {0x082b082b, 0x0808082b}, {0x2b08082b, 0x0808082b},
    {0x08080819, 0x08081908}, {0x08081908, 0x08081908}, {0x08190808, 0x08081908}, {0x082b0819, 0x08081908},
    {0x082b1908, 0x08081908}, {0x19080808, 0x08081908}, {0x1908082b, 0x08081908}, {0x19082b08, 0x08081908},
    {0x192b0808, 0x08081908}, {0x2b080819, 0x08081908}, {0x2b081908, 0x08081908}, {0x2b190808, 0x08081908},
    {0x2b2b1908, 0x08081908}, {0x08080808, 0x08081919}, {0x0808082b, 0x08081919}, {0x08082b08, 0x08081919},
    {0x082b0808, 0x08081919}, {0x1908192b, 0x08081919}, {0x192b2b19, 0x08081919}, {0x2b080808, 0x08081919},
    {0x2b190819, 0x08081919}, {0x08082b19, 0x0808192b}, {0x08190808, 0x0808192b}, {0x19080808, 0x0808192b},
    {0x2b081908, 0x0808192b}, {0x2b2b1908, 0x0808192b}, {0x08080808, 0x08082b08}, {0x08081919, 0x08082b08},
    {0x08082b08, 0x08082b08}, {0x08191908, 0x08082b08}, {0x082b2b08, 0x08082b08}, {0x19080819, 0x08082b08},
    {0x19081908, 0x08082b08}, {0x19190808, 0x08082b08}, {0x1919082b, 0x08082b08}, {0x2b082b08, 0x08082b08},
    {0x08081908, 0x08082b19}, {0x19080808, 0x08082b19}, {0x0808082b, 0x08082b2b}, {0x08191908, 0x08082b2b},
    {0x08080819, 0x08190808}, {0x08081908, 0x08190808}, {0x08190808, 0x08190808}, {0x082b0819, 0x08190808},
    {0x19080808, 0x08190808}, {0x192b0808, 0x08190808}, {0x2b081908, 0x08190808}, {0x2b190808, 0x08190808},
    {0x2b191919, 0x08190808}, {0x08080808, 0x08190819}, {0x08082b08, 0x08190819}, {0x082b0808, 0x08190819},
    {0x19190808, 0x08190819}, {0x19192b2b, 0x08190819}, {0x2b080808, 0x08190819}, {0x082b1908, 0x0819082b},
    {0x19081919, 0x0819082b}, {0x08080808, 0x08191908}, {0x08082b08, 0x08191908}, {0x082b0808, 0x08191908},
    {0x082b1919, 0x08191908}, {0x19082b19, 0x08191908}, {0x2b080808, 0x08191908}, {0x08192b08, 0x08191919},
    {0x192b082b, 0x08191919}, {0x08080808, 0x0819192b}, {0x0819192b, 0x0819192b}, {0x08080819, 0x08192b08},
    {0x08081908, 0x08192b08}, {0x08190808, 0x08192b08}, {0x19080808, 0x08192b08}, {0x2b080819, 0x08192b08},
    {0x08080808, 0x08192b19}, {0x08081919, 0x08192b19}, {0x2b2b0808, 0x08192b19}, {0x19190819, 0x08192b2b},
    {0x08080808, 0x082b0808}, {0x0808082b, 0x082b0808}, {0x08082b2b, 0x082b0808}, {0x19081908, 0x082b0808},
    {0x192b0819, 0x082b0808}, {0x2b080808, 0x082b0808}, {0x2b08082b, 0x082b0808}, {0x082b2b19, 0x082b0819},
    {0x19082b08, 0x082b0819}, {0x08080808, 0x082b082b}, {0x0808082b, 0x082b082b}, {0x08080819, 0x082b1908},
    {0x08081908, 0x082b1908}, {0x08190808, 0x082b1908}, {0x19080808, 0x082b1908}, {0x1919192b, 0x082b1908},
    {0x08080808, 0x082b1919}, {0x19080819, 0x082b1919}, {0x192b1908, 0x082b1919}, {0x2b190808, 0x082b192b},
    {0x08082b08, 0x082b2b08}, {0x082b0808, 0x082b2b08}, {0x2b191908, 0x082b2b08}, {0x19081908, 0x082b2b2b},
    {0x08080819, 0x19080808}, {0x08081908, 0x19080808}, {0x08190808, 0x19080808}, {0x08192b08, 0x19080808},
    {0x082b0819, 0x19080808}, {0x082b1908, 0x19080808}, {0x19080808, 0x19080808}, {0x19082b08, 0x19080808},
    {0x1919192b, 0x19080808}, {0x192b0808, 0x19080808}, {0x2b080819, 0x19080808}, {0x2b081908, 0x19080808},
    {0x2b190808, 0x19080808}, {0x08080808, 0x19080819}, {0x082b0808, 0x19080819}, {0x192b0819, 0x19080819},
    {0x2b080808, 0x19080819}, {0x2b081919, 0x19080819}, {0x08080819, 0x1908082b}, {0x08190808, 0x1908082b},
    {0x19082b08, 0x1908082b}, {0x1919192b, 0x1908082b}, {0x192b2b08, 0x1908082b}, {0x08080808, 0x19081908},
    {0x08082b08, 0x19081908}, {0x082b0808, 0x19081908}, {0x2b080808, 0x19081908}, {0x2b192b19, 0x19081908},
    {0x0819082b, 0x19081919}, {0x082b1908, 0x19081919}, {0x08080808, 0x1908192b}, {0x08080819, 0x19082b08},
    {0x08081908, 0x19082b08}, {0x08190808, 0x19082b08}, {0x19080808, 0x19082b08}, {0x19081919, 0x19082b08},
    {0x08080808, 0x19082b19}, {0x19192b08, 0x19082b19}, {0x192b0819, 0x19082b19}, {0x2b08082b, 0x19082b19},
    {0x19081919, 0x19082b2b}, {0x2b190808, 0x19082b2b}, {0x08080808, 0x19190808}, {0x08082b08, 0x19190808},
    {0x08190819, 0x19190808}, {0x08192b19, 0x19190808}, {0x082b0808, 0x19190808}, {0x2b080808, 0x19190808},
    {0x2b082b08, 0x19190808}, {0x08081908, 0x19190819}, {0x1908082b, 0x19190819}, {0x2b2b1908, 0x19190819},
    {0x2b190819, 0x1919082b}, {0x2b190808, 0x19191908}, {0x2b19082b, 0x19191908}, {0x08082b2b, 0x19191919},
    {0x08080819, 0x1919192b}, {0x19191908, 0x1919192b}, {0x08080808, 0x19192b08}, {0x08190819, 0x19192b08},
    {0x08192b19, 0x19192b08}, {0x192b1908, 0x19192b08}, {0x19080808, 0x19192b19}, {0x08082b08, 0x19192b2b},
    {0x08081908, 0x192b0808}, {0x08190808, 0x192b0808}, {0x19080808, 0x192b0808}, {0x192b2b08, 0x192b0808},
    {0x08080808, 0x192b0819}, {0x19191919, 0x192b0819}, {0x08192b08, 0x192b082b}, {0x192b0808, 0x192b082b},
    {0x08080808, 0x192b1908}, {0x08081919, 0x192b1908}, {0x08190808, 0x192b1919}, {0x0819082b, 0x192b1919},
    {0x2b081908, 0x192b1919}, {0x1908082b, 0x192b2b08}, {0x08080808, 0x2b080808}, {0x0808082b, 0x2b080808},
    {0x08082b2b, 0x2b080808}, {0x19080819, 0x2b080808}, {0x2b08082b, 0x2b080808}, {0x08081908, 0x2b080819},
    {0x08192b08, 0x2b080819}, {0x19080808, 0x2b080819}, {0x08190819, 0x2b08082b}, {0x08080819, 0x2b081908},
    {0x08081908, 0x2b081908}, {0x08190808, 0x2b081908}, {0x08191919, 0x2b081908}, {0x19080808, 0x2b081908},
    {0x192b0808, 0x2b081908}, {0x08080808, 0x2b081919}, {0x1908192b, 0x2b081919}, {0x2b191908, 0x2b081919},
    {0x08082b19, 0x2b08192b}, {0x19080808, 0x2b08192b}, {0x192b0808, 0x2b08192b}, {0x0808082b, 0x2b082b08},
    {0x08081908, 0x2b082b19}, {0x08190819, 0x2b082b2b}, {0x08081908, 0x2b190808}, {0x08190808, 0x2b190808},
    {0x082b1908, 0x2b190808}, {0x19080808, 0x2b190808}, {0x2b2b0819, 0x2b190808}, {0x0819192b, 0x2b190819},
    {0x2b080808, 0x2b190819}, {0x19081919, 0x2b19082b}, {0x08080808, 0x2b191908}, {0x082b082b, 0x2b191908},
    {0x19081908, 0x2b191908}, {0x19190819, 0x2b191919}, {0x2b080819, 0x2b192b08}, {0x082b0808, 0x2b192b19},
    {0x0808082b, 0x2b2b0808}, {0x19190808, 0x2b2b0808}, {0x2b081919, 0x2b2b0808}, {0x08082b19, 0x2b2b0819},
    {0x08080808, 0x2b2b082b}, {0x08192b08, 0x2b2b1908}, {0x19190808, 0x2b2b2b08}, {0x08081908, 0x2b2b2b19},
};

template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_iq2xxs_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                          const uint8_t* __restrict__ w, OutT* __restrict__ out,
                          int M, int N, int K, int num_sb) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;
  if (n >= N) return;
  const int nsub = K / 32;
  const int64_t row_bytes = (int64_t)num_sb * IQ2XXS_TYPE_SIZE;
  const uint8_t* wrow = w + (int64_t)n * row_bytes;

  float f_acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) f_acc[m] = 0.f;

  for (int i = lane; i < nsub; i += DEC_WARP) {
    const int sb = i >> 3, ib32 = i & 7;
    const uint8_t* blk = wrow + (int64_t)sb * IQ2XXS_TYPE_SIZE;
    const uint8_t* pair = blk + 2 + 8 * ib32;
    const uint32_t aux0 = dec_ld_u32(pair);        // four 8-bit grid indices
    const uint32_t aux1 = dec_ld_u32(pair + 4);    // scale nibble + four 7-bit sign idx
    const float db = __half2float(*reinterpret_cast<const __half*>(blk)) *
                     (0.5f + (float)(aux1 >> 28)) * 0.25f;

    int32_t q[8];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint32_t idx = (aux0 >> (8 * l)) & 0xFFu;
      const int sgn = __ldg(&kSignsIq2xs[(aux1 >> (7 * l)) & 127]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {                // h=0 -> values 0-3, h=1 -> 4-7
        const uint32_t g = __ldg(&kIq2xxsGrid[idx][h]);
        const int mb = sgn >> (4 * h);             // 4 sign bits for these 4 values
        const uint32_t sbits = (uint32_t)(mb & 1) | ((uint32_t)((mb >> 1) & 1) << 8) |
                               ((uint32_t)((mb >> 2) & 1) << 16) |
                               ((uint32_t)((mb >> 3) & 1) << 24);
        const int neg = __vcmpne4(sbits, 0x00000000u);
        q[2 * l + h] = __vsub4((int)(g ^ (uint32_t)neg), neg);
      }
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
