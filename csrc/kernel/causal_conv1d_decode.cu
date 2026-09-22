// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Launcher + torch wrapper for the fused causal-conv1d(kernel=K)+SiLU decode
// token shift. See csrc/include/causal_conv1d_decode.cuh and
// tests/test_deltanet_fused_ops.py. No dynamic shared memory ->
// CUDA-graph-capturable.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <tuple>

#include "causal_conv1d_decode.cuh"

namespace fni8 {

std::tuple<at::Tensor, at::Tensor> causal_conv1d_silu_decode(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& tail) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && tail.is_cuda(),
              "fni8.causal_conv1d_silu_decode: inputs must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2, "fni8.causal_conv1d_silu_decode: x must be [B, Wc]");
  TORCH_CHECK(weight.dim() == 2, "fni8.causal_conv1d_silu_decode: weight must be [Wc, K]");
  TORCH_CHECK(tail.dim() == 3, "fni8.causal_conv1d_silu_decode: tail must be [B, K-1, Wc]");
  const int64_t B = x.size(0), Wc = x.size(1);
  const int64_t K = weight.size(1);
  TORCH_CHECK(weight.size(0) == Wc,
              "fni8.causal_conv1d_silu_decode: weight rows must equal Wc");
  TORCH_CHECK(tail.size(0) == B && tail.size(1) == K - 1 && tail.size(2) == Wc,
              "fni8.causal_conv1d_silu_decode: tail must be [B, K-1, Wc]");
  TORCH_CHECK(K >= 1 && K <= CONV_MAX_K,
              "fni8.causal_conv1d_silu_decode: supports 1 <= K <= ", CONV_MAX_K,
              " (kernel width), got ", K);
  const auto st = x.scalar_type();
  TORCH_CHECK(weight.scalar_type() == st && tail.scalar_type() == st,
              "fni8.causal_conv1d_silu_decode: x/weight/tail must share dtype");
  TORCH_CHECK(st == at::kHalf || st == at::kBFloat16 || st == at::kFloat,
              "fni8.causal_conv1d_silu_decode: dtype must be fp16/bf16/fp32");

  const at::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous();
  auto wc = weight.contiguous();
  auto tc = tail.contiguous();
  auto out = at::empty({B, Wc}, xc.options());
  auto new_tail = at::empty({B, K - 1, Wc}, xc.options());
  const int64_t total = B * Wc;
  if (total == 0) return {out, new_tail};

  const int threads = 256;
  const unsigned blocks = (unsigned)std::min<int64_t>(
      (total + threads - 1) / threads, 2147483647LL);
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH(T)                                                             \
  causal_conv1d_silu_decode_kernel<T><<<blocks, threads, 0, stream>>>(        \
      reinterpret_cast<const T*>(xc.data_ptr()),                             \
      reinterpret_cast<const T*>(tc.data_ptr()),                             \
      reinterpret_cast<const T*>(wc.data_ptr()),                             \
      reinterpret_cast<T*>(out.data_ptr()),                                  \
      reinterpret_cast<T*>(new_tail.data_ptr()), (int)B, (int)Wc, (int)K)
  if (st == at::kHalf) LAUNCH(__half);
  else if (st == at::kBFloat16) LAUNCH(__nv_bfloat16);
  else LAUNCH(float);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, new_tail};
}

}  // namespace fni8
