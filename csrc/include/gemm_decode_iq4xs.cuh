// SPDX-License-Identifier: BSD-3-Clause
// ===========================================================================
// IQ4_XS decode — warp-per-column MMVQ over ggml's 16-entry signed codebook.
// Block (136 B / 256 weights, 4.25 bpw):
//     +0   d          fp16 super-block scale
//     +2   scales_h   uint16: 2 high bits of each of the 8 sub-block scales
//     +4   scales_l[4] two 4-bit low halves per byte
//     +8   qs[128]    4-bit codebook indices, 16 bytes per 32-weight sub-block
// Per 32-weight sub-block ib:
//     ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xF) | (((scales_h >> 2*ib) & 3) << 4)
//     dl = d * (ls - 32)        SIGNED offset scale (IQ3_S uses 1 + 2*s)
// The two halves are INTERLEAVED BY NIBBLE: weight j is the low nibble of qs[j],
// weight j+16 the high nibble. So a sub-block is two 16-lane groups, and the
// activation is read as two separate int4 loads (x[0:16] and x[16:32]) rather than
// the single shifted 32-byte run Q4_K uses.
// Codebook values are signed int8 (-127..113), so unpacked bytes feed __dp4a
// directly — no grid, no sign bits. Oracle: fni8.quant.iq4xs (fni8#317).
// Adapted (MIT->BSD-3) from llama.cpp vec_dot_iq4_xs_q8_1 + dequantize_row_iq4_xs.
// ===========================================================================
#pragma once

#include "gemm_decode_dp4a.cuh"

namespace fni8 {

constexpr int IQ4XS_TYPE_SIZE = 136;

// ggml-common.h kvalues_iq4nl, packed 4 per word so a nibble maps to a byte lane.
// NOT __constant__ despite being only 16 entries: a warp still issues up to 16
// distinct addresses and constant memory serializes on ADDRESSES, not cache lines.
// Measured 3.28x (55.7 -> 182.8 GB/s divergent vs uniform) -- higher than IQ2_XS,
// and IQ4_XS is 21.6% of Qwen3.8-27B-UD-IQ3_S's weights (fni8#334).
__device__ int8_t kIq4nlValues[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// Unpack 4 nibbles (one per byte of `nib`) into 4 packed signed codebook bytes.
__device__ __forceinline__ int iq4xs_lookup4(uint32_t nib) {
  int out = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int v = (int)(uint8_t)__ldg(&kIq4nlValues[(nib >> (8 * k)) & 0xF]);
    out |= (v & 0xFF) << (8 * k);
  }
  return out;
}

template <typename OutT, int MAX_M>
__global__ void __launch_bounds__(DEC_THREADS)
gemm_decode_iq4xs_kernel(const int8_t* __restrict__ x, const float* __restrict__ x_scale,
                         const uint8_t* __restrict__ w, OutT* __restrict__ out,
                         int M, int N, int K, int num_sb) {
  const int warp_id = threadIdx.x / DEC_WARP;
  const int lane = threadIdx.x % DEC_WARP;
  const int n = blockIdx.x * DEC_WARPS_PER_BLOCK + warp_id;
  if (n >= N) return;
  const int nsub = K / 32;                        // 32-weight sub-blocks along K
  const int64_t row_bytes = (int64_t)num_sb * IQ4XS_TYPE_SIZE;
  const uint8_t* wrow = w + (int64_t)n * row_bytes;

  float f_acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) f_acc[m] = 0.f;

  for (int i = lane; i < nsub; i += DEC_WARP) {
    const int sb = i >> 3, ib = i & 7;            // super-block, sub-block within it
    const uint8_t* blk = wrow + (int64_t)sb * IQ4XS_TYPE_SIZE;
    const int scales_h = (int)blk[2] | ((int)blk[3] << 8);
    const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) |
                   (((scales_h >> (2 * ib)) & 3) << 4);
    const float dl = __half2float(*reinterpret_cast<const __half*>(blk)) * (float)(ls - 32);

    const uint8_t* qs = blk + 8 + 16 * ib;        // 16 bytes = 32 weights
    int32_t qlo[4], qhi[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint32_t nib = dec_ld_u32(qs + 4 * c);
      qlo[c] = iq4xs_lookup4(nib & 0x0F0F0F0Fu);          // weights  0..15
      qhi[c] = iq4xs_lookup4((nib >> 4) & 0x0F0F0F0Fu);   // weights 16..31
    }
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m >= M) break;
      const int4* xr = reinterpret_cast<const int4*>(x + (int64_t)m * K);
      const int4 xv0 = xr[i * 2], xv1 = xr[i * 2 + 1];  // x[0:16] and x[16:32]
      int32_t si = 0;
      si = __dp4a(xv0.x, qlo[0], si);
      si = __dp4a(xv0.y, qlo[1], si);
      si = __dp4a(xv0.z, qlo[2], si);
      si = __dp4a(xv0.w, qlo[3], si);
      si = __dp4a(xv1.x, qhi[0], si);
      si = __dp4a(xv1.y, qhi[1], si);
      si = __dp4a(xv1.z, qhi[2], si);
      si = __dp4a(xv1.w, qhi[3], si);
      f_acc[m] += dl * (float)si;
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
