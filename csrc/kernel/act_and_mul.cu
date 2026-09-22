// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: launcher + torch wrapper for the fused gated activation.
// See csrc/include/act_and_mul.cuh and tests/test_act_and_mul.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <string>

#include "act_and_mul.cuh"

namespace fni8 {

namespace {

template <typename T>
void launch_act(const T* x, T* out, int64_t M, int64_t I, ActKind kind, cudaStream_t s) {
  const int threads = 256;
  const int64_t total = M * I;
  const int blocks = (int)std::min<int64_t>((total + threads - 1) / threads, 65535);
#define LAUNCH(K) act_and_mul_kernel<T, K><<<blocks, threads, 0, s>>>(x, out, M, I)
  if (kind == ActKind::Silu) LAUNCH(ActKind::Silu); else LAUNCH(ActKind::GeluTanh);
#undef LAUNCH
}

void launch_act_half2(const __half* x, __half* out, int64_t M, int64_t I,
                      ActKind kind, cudaStream_t s) {
  const int threads = 256;
  const int64_t total_pairs = M * (I >> 1);
  const int blocks = total_pairs > 0
    ? (int)std::min<int64_t>((total_pairs + threads - 1) / threads, 65535) : 1;
#define LAUNCH(K) act_and_mul_kernel_half2<K><<<blocks, threads, 0, s>>>(x, out, M, I)
  if (kind == ActKind::Silu) LAUNCH(ActKind::Silu); else LAUNCH(ActKind::GeluTanh);
#undef LAUNCH
}

}  // namespace

// x[..., 2I] (fp16/bf16) -> out[..., I] = act(gate) * up. `kind`: "silu" |
// "gelu_tanh" (Gemma GeGLU).
at::Tensor act_and_mul(at::Tensor x, const std::string& kind) {
  TORCH_CHECK(x.is_cuda(), "fni8: act_and_mul expects a CUDA tensor");
  auto xc = x.contiguous();
  const auto st = xc.scalar_type();
  TORCH_CHECK(st == at::kHalf || st == at::kBFloat16, "fni8: act_and_mul needs fp16/bf16");
  const int64_t twoI = xc.size(-1);
  TORCH_CHECK(twoI % 2 == 0, "fni8: last dim must be even (gate|up), got ", twoI);
  const int64_t I = twoI / 2, M = xc.numel() / twoI;
  ActKind ak;
  if (kind == "silu") ak = ActKind::Silu;
  else if (kind == "gelu_tanh" || kind == "gelu") ak = ActKind::GeluTanh;
  else TORCH_CHECK(false, "fni8: act_and_mul kind must be 'silu' or 'gelu_tanh', got ", kind);

  auto shape = xc.sizes().vec();
  shape.back() = I;
  const at::cuda::CUDAGuard guard(xc.device());
  auto out = at::empty(shape, xc.options());
  if (M == 0 || I == 0) return out;

  auto stream = at::cuda::getCurrentCUDAStream();
  if (st == at::kHalf && (I % 2 == 0))
    launch_act_half2(reinterpret_cast<const __half*>(xc.data_ptr()),
                     reinterpret_cast<__half*>(out.data_ptr()), M, I, ak, stream);
  else if (st == at::kHalf)
    launch_act<__half>(reinterpret_cast<const __half*>(xc.data_ptr()),
                       reinterpret_cast<__half*>(out.data_ptr()), M, I, ak, stream);
  else
    launch_act<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()),
                              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, I, ak, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace fni8
