// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: launcher + torch wrapper for the fused per-row int8
// activation quantizer. Replaces the eager `quantize_int8_rowwise` prologue with
// one launch — see csrc/include/quant_rowwise.cuh and tests/test_quant_rowwise.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>

#include "quant_rowwise.cuh"

namespace fni8 {
namespace {

// Half2-vectorized fp16→int8 quantization. Launched ONLY for even K (launcher
// guarantees it): then every row offset is 4-byte aligned and K splits into K/2
// pairs with no tail. The abs-max reduction and quantize step both use fp32 math
// (same rintf(__fdiv_rn) as the scalar template) → bit-exact int8 + scale.
// Half2 loads give memory-BW win vs scalar; zero arithmetic rounding difference.
__global__ void quantize_i8_rowwise_half2_kernel(
    const __half* __restrict__ x,   // [M,K]
    int8_t* __restrict__ q,         // [M,K]
    float* __restrict__ scale,      // [M]
    int M, int K) {
  const int row = blockIdx.x;
  if (row >= M) return;
  const __half* __restrict__ xr = x + (int64_t)row * K;
  int8_t* __restrict__ qr = q + (int64_t)row * K;
  const int tid = threadIdx.x;
  const int K2 = K >> 1;  // K is even (launcher guarantees it; K >= 2)

  // Pass 1: abs-max reduction using half2 loads + __habs2.
  float amax = 0.0f;
  const __half2* __restrict__ xr2 = reinterpret_cast<const __half2*>(xr);
  for (int j = tid; j < K2; j += QUANT_ROW_THREADS) {
    __half2 v = xr2[j];
    __half2 av = __habs2(v);
    amax = fmaxf(amax, __half2float(av.x));
    amax = fmaxf(amax, __half2float(av.y));
  }
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  __shared__ float warp_amax[QUANT_ROW_THREADS / 32];
  const int lane = tid & 31, warp_id = tid >> 5;
  if (lane == 0) warp_amax[warp_id] = amax;
  __syncthreads();
  __shared__ float row_safe;
  if (warp_id == 0) {
    float v = (lane < (QUANT_ROW_THREADS / 32)) ? warp_amax[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if (lane == 0) {
      const float s = v * FNI8_Q_MAX_INV;
      row_safe = (s == 0.0f) ? 1.0f : s;
      scale[row] = row_safe;
    }
  }
  __syncthreads();

  // Pass 2: quantize in fp32 (mirror the scalar template exactly: half2 loads
  // → float, then __fdiv_rn + rintf + clamp). Bit-exact int8 output for all K.
  const float inv = row_safe;
  for (int j = tid; j < K2; j += QUANT_ROW_THREADS) {
    __half2 v = xr2[j];
    float2 vf = __half22float2(v);
    vf.x = rintf(__fdiv_rn(vf.x, inv));
    vf.y = rintf(__fdiv_rn(vf.y, inv));
    vf.x = fminf(fmaxf(vf.x, -FNI8_Q_MAX), FNI8_Q_MAX);
    vf.y = fminf(fmaxf(vf.y, -FNI8_Q_MAX), FNI8_Q_MAX);
    const int i2 = j << 1;
    qr[i2]     = (int8_t)vf.x;
    qr[i2 + 1] = (int8_t)vf.y;
  }
}

}  // anonymous namespace

// x: [M,K] fp16/bf16/fp32 (contiguous). Returns (q int8 [M,K], scale fp32 [M]).
std::tuple<at::Tensor, at::Tensor> quantize_i8_rowwise(at::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "fni8: quantize_i8_rowwise expects a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "fni8: expect x[M,K], got dim=", x.dim());
  auto xc = x.contiguous();
  const auto st = xc.scalar_type();
  TORCH_CHECK(st == at::kHalf || st == at::kBFloat16 || st == at::kFloat,
              "fni8: quantize_i8_rowwise input must be float16/bfloat16/float32");

  const int64_t M = xc.size(0), K = xc.size(1);
  const at::cuda::CUDAGuard guard(xc.device());
  auto q = at::empty({M, K}, xc.options().dtype(at::kChar));
  auto scale = at::empty({M}, xc.options().dtype(at::kFloat));
  if (M == 0 || K == 0) return {q, scale};

  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)M);
  if (st == at::kHalf && (K % 2 == 0)) {
    // Even K: half2-vectorized kernel for aligned row access + __habs2/fp32 quant.
    quantize_i8_rowwise_half2_kernel<<<grid, QUANT_ROW_THREADS, 0, stream>>>(
        reinterpret_cast<const __half*>(xc.data_ptr()), q.data_ptr<int8_t>(),
        scale.data_ptr<float>(), (int)M, (int)K);
  } else if (st == at::kHalf) {
    // Odd K: fall back to the scalar template (alignment-safe).
    quantize_i8_rowwise_kernel<__half><<<grid, QUANT_ROW_THREADS, 0, stream>>>(
        reinterpret_cast<const __half*>(xc.data_ptr()), q.data_ptr<int8_t>(),
        scale.data_ptr<float>(), (int)M, (int)K);
  } else if (st == at::kBFloat16) {
    quantize_i8_rowwise_kernel<__nv_bfloat16><<<grid, QUANT_ROW_THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()), q.data_ptr<int8_t>(),
        scale.data_ptr<float>(), (int)M, (int)K);
  } else {
    quantize_i8_rowwise_kernel<float><<<grid, QUANT_ROW_THREADS, 0, stream>>>(
        xc.data_ptr<float>(), q.data_ptr<int8_t>(),
        scale.data_ptr<float>(), (int)M, (int)K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, scale};
}

}  // namespace fni8
