// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: launcher + torch wrapper for the fused RoPE kernel.
// See csrc/include/rope.cuh and tests/test_rope.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <tuple>

#include "rope.cuh"

namespace fni8 {

namespace {

void rope_one(at::Tensor& x, const at::Tensor& positions, const at::Tensor& cos,
              const at::Tensor& sin, int rotary_dim, cudaStream_t stream) {
  // x: [..., H, D] -> flatten leading dims to N. positions broadcasts over H.
  const int64_t D = x.size(-1), H = x.size(-2), N = x.numel() / (H * D);
  const dim3 grid((unsigned)N, (unsigned)H);
  const int threads = std::min<int>(rotary_dim / 2, 256);
  if (x.scalar_type() == at::kHalf) {
    rope_half2_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<__half*>(x.data_ptr()), positions.data_ptr<int64_t>(),
        cos.data_ptr<float>(), sin.data_ptr<float>(), (int)N, (int)H, (int)D, rotary_dim);
  } else {
    rope_kernel<__nv_bfloat16><<<grid, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), positions.data_ptr<int64_t>(),
        cos.data_ptr<float>(), sin.data_ptr<float>(), (int)N, (int)H, (int)D, rotary_dim);
  }
}

}  // namespace

// q [...,Hq,D], k [...,Hk,D] (fp16/bf16), positions [...] int64, cos/sin
// [max_pos, rotary_dim] fp32. Rotates q and k IN PLACE and returns them.
std::tuple<at::Tensor, at::Tensor> rope(at::Tensor positions, at::Tensor q, at::Tensor k,
                                        at::Tensor cos, at::Tensor sin, int64_t rotary_dim) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda(), "fni8: rope expects CUDA tensors");
  const auto st = q.scalar_type();
  TORCH_CHECK(st == at::kHalf || st == at::kBFloat16, "fni8: rope needs fp16/bf16 q/k");
  TORCH_CHECK(k.scalar_type() == st, "fni8: q/k dtype mismatch");
  TORCH_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
              "fni8: cos/sin tables must be float32");
  TORCH_CHECK(rotary_dim % 2 == 0 && rotary_dim <= q.size(-1),
              "fni8: rotary_dim must be even and <= head_dim");
  TORCH_CHECK(cos.size(-1) == rotary_dim && sin.size(-1) == rotary_dim,
              "fni8: cos/sin last dim must equal rotary_dim");
  auto qc = q.contiguous(), kc = k.contiguous();
  auto pos = positions.to(at::kLong).contiguous();
  auto cosc = cos.contiguous(), sinc = sin.contiguous();
  TORCH_CHECK(pos.numel() == qc.numel() / (qc.size(-1) * qc.size(-2)),
              "fni8: positions count must match q's token count");

  const at::cuda::CUDAGuard guard(qc.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  rope_one(qc, pos, cosc, sinc, (int)rotary_dim, stream);
  rope_one(kc, pos, cosc, sinc, (int)rotary_dim, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {qc.view_as(q), kc.view_as(k)};
}

}  // namespace fni8
