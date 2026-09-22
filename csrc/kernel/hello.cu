// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: launcher + torch wrapper for the hello_add kernel.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include "hello.cuh"

namespace fni8 {

// C = A + B, fp16, same shape. Proves the sm_70 build/bind/dispatch loop.
at::Tensor hello_add(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "fni8.hello_add: inputs must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf,
              "fni8.hello_add: inputs must be float16");
  TORCH_CHECK(a.sizes() == b.sizes(), "fni8.hello_add: shape mismatch");

  const at::cuda::CUDAGuard guard(a.device());
  auto ac = a.contiguous();
  auto bc = b.contiguous();
  auto c = at::empty_like(ac);

  const int64_t n = ac.numel();
  if (n == 0) return c;

  const int threads = 256;
  const int max_blocks = 4096;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  if (blocks > max_blocks) blocks = max_blocks;

  auto stream = at::cuda::getCurrentCUDAStream();
  hello_add_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __half*>(ac.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(bc.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

}  // namespace fni8
