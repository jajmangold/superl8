// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Launcher + torch wrapper for the v3 int8 dp4a MLA absorb-path decode kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "mla_attn_int8.cuh"

namespace fni8 {

at::Tensor mla_decode_int8(const at::Tensor& q_abs, const at::Tensor& q_rope,
                           const at::Tensor& c_kv_cache, const at::Tensor& k_rope_cache,
                           double scale) {
  TORCH_CHECK(q_abs.is_cuda() && q_rope.is_cuda() && c_kv_cache.is_cuda() &&
                  k_rope_cache.is_cuda(),
              "fni8 mla_decode_int8: all inputs must be CUDA tensors");
  TORCH_CHECK(q_abs.device() == q_rope.device() && q_abs.device() == c_kv_cache.device() &&
                  q_abs.device() == k_rope_cache.device(),
              "fni8 mla_decode_int8: all inputs must be on the same device");
  TORCH_CHECK(q_abs.scalar_type() == at::kHalf && q_rope.scalar_type() == at::kHalf &&
                  c_kv_cache.scalar_type() == at::kHalf &&
                  k_rope_cache.scalar_type() == at::kHalf,
              "fni8 mla_decode_int8: v3 is fp16-input (int8 dp4a QK/PV internal); "
              "softmax/PV accumulation stays fp32 per AGENTS.md");
  TORCH_CHECK(q_abs.dim() == 3 && q_rope.dim() == 3 && c_kv_cache.dim() == 3 &&
                  k_rope_cache.dim() == 3,
              "fni8 mla_decode_int8: expect q_abs/q_rope [B,H,*], "
              "c_kv_cache/k_rope_cache [B,N,*]");
  TORCH_CHECK(q_abs.is_contiguous() && q_rope.is_contiguous() &&
                  c_kv_cache.is_contiguous() && k_rope_cache.is_contiguous(),
              "fni8 mla_decode_int8: inputs must be contiguous");

  const auto B = q_abs.size(0), H = q_abs.size(1), Dc = q_abs.size(2);
  const auto Dr = q_rope.size(2);
  const auto N = c_kv_cache.size(1);
  TORCH_CHECK(q_rope.size(0) == B && q_rope.size(1) == H,
              "fni8 mla_decode_int8: q_rope batch/head mismatch");
  TORCH_CHECK(c_kv_cache.size(0) == B && c_kv_cache.size(2) == Dc,
              "fni8 mla_decode_int8: c_kv_cache shape mismatch");
  TORCH_CHECK(k_rope_cache.size(0) == B && k_rope_cache.size(1) == N &&
                  k_rope_cache.size(2) == Dr,
              "fni8 mla_decode_int8: k_rope_cache shape mismatch");
  TORCH_CHECK(Dc > 0 && Dr > 0 && H > 0 && B > 0,
              "fni8 mla_decode_int8: degenerate shape");
  TORCH_CHECK(Dc % 4 == 0,
              "fni8 mla_decode_int8: d_c must be a multiple of 4 for int32-packing");

  const at::cuda::CUDAGuard guard(q_abs.device());
  auto out = at::empty({B, H, Dc}, q_abs.options());
  if (N == 0) return out.zero_();

  const int64_t bh = B * H;
  const int d_c4 = (int)Dc / 4;
  // smem layout (bytes):
  //   4*Dc     q_abs_s  [Dc] float
  //   4*Dr     q_rope_s [Dr] float
  //   4*Dc     o_s      [Dc] float
  //   4*d_c4   q_int8_p [d_c4] int32 (== Dc bytes int8)
  //   2*Dc     ckv_f16  [Dc] half
  //   4*d_c4   ckv_int8_p [d_c4] int32 (== Dc bytes int8)
  const size_t smem_bytes =
      4 * Dc + 4 * Dr + 4 * Dc + 4 * d_c4 + 2 * Dc + 4 * d_c4;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(mla_decode_int8_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);
  }
  mla_decode_int8_kernel<<<(unsigned)bh, MLA_INT8_WARP, smem_bytes, stream>>>(
      reinterpret_cast<const half*>(q_abs.data_ptr()),
      reinterpret_cast<const half*>(q_rope.data_ptr()),
      reinterpret_cast<const half*>(c_kv_cache.data_ptr()),
      reinterpret_cast<const half*>(k_rope_cache.data_ptr()),
      reinterpret_cast<half*>(out.data_ptr()),
      (int)H, (int)N, (int)Dc, (int)Dr, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace fni8
