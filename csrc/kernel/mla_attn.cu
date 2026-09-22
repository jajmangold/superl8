// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Launcher + torch wrapper for the v1 fp32 MLA absorb-path decode kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "mla_attn.cuh"

namespace fni8 {

// q_abs [B,H,d_c] fp32 (q_nope . W_UK^T, per head — the absorb-folded query),
// q_rope [B,H,d_r] fp32 (RoPE'd), c_kv_cache [B,N,d_c] fp32 (RMSNorm'd latent
// cache, includes the current decode step's own row), k_rope_cache [B,N,d_r]
// fp32 (shared decoupled-RoPE key cache). Returns out_abs [B,H,d_c] fp32 --
// the weighted-latent output BEFORE the W_UV.W_O absorb projection (applied
// in fni8/ops.py, a small per-token GEMM -- linear projections stay out of
// the attention kernel here, same convention as gemm_w8a8 vs attn_int8_fwd).
// Decode-only (M=1): the query is the cache's last row, so every cached key
// is attended unconditionally -- no causal mask.
at::Tensor mla_decode(const at::Tensor& q_abs, const at::Tensor& q_rope,
                      const at::Tensor& c_kv_cache, const at::Tensor& k_rope_cache,
                      double scale) {
  TORCH_CHECK(q_abs.is_cuda() && q_rope.is_cuda() && c_kv_cache.is_cuda() &&
                  k_rope_cache.is_cuda(),
              "fni8 mla_decode: all inputs must be CUDA tensors");
  TORCH_CHECK(q_abs.device() == q_rope.device() && q_abs.device() == c_kv_cache.device() &&
                  q_abs.device() == k_rope_cache.device(),
              "fni8 mla_decode: all inputs must be on the same device");
  TORCH_CHECK(q_abs.scalar_type() == at::kFloat && q_rope.scalar_type() == at::kFloat &&
                  c_kv_cache.scalar_type() == at::kFloat &&
                  k_rope_cache.scalar_type() == at::kFloat,
              "fni8 mla_decode: v1 is fp32-only (softmax/PV over the latent is "
              "numerically load-bearing per AGENTS.md; int8 dp4a lands in a follow-up)");
  TORCH_CHECK(q_abs.dim() == 3 && q_rope.dim() == 3 && c_kv_cache.dim() == 3 &&
                  k_rope_cache.dim() == 3,
              "fni8 mla_decode: expect q_abs/q_rope [B,H,*], c_kv_cache/k_rope_cache [B,N,*]");
  TORCH_CHECK(q_abs.is_contiguous() && q_rope.is_contiguous() && c_kv_cache.is_contiguous() &&
                  k_rope_cache.is_contiguous(),
              "fni8 mla_decode: inputs must be contiguous");

  const auto B = q_abs.size(0), H = q_abs.size(1), Dc = q_abs.size(2);
  const auto Dr = q_rope.size(2);
  const auto N = c_kv_cache.size(1);
  TORCH_CHECK(q_rope.size(0) == B && q_rope.size(1) == H,
              "fni8 mla_decode: q_rope batch/head mismatch");
  TORCH_CHECK(c_kv_cache.size(0) == B && c_kv_cache.size(2) == Dc,
              "fni8 mla_decode: c_kv_cache shape mismatch");
  TORCH_CHECK(k_rope_cache.size(0) == B && k_rope_cache.size(1) == N && k_rope_cache.size(2) == Dr,
              "fni8 mla_decode: k_rope_cache shape mismatch");
  TORCH_CHECK(Dc > 0 && Dr > 0 && H > 0 && B > 0, "fni8 mla_decode: degenerate shape");

  const at::cuda::CUDAGuard guard(q_abs.device());
  auto out = at::empty({B, H, Dc}, q_abs.options());
  if (N == 0) return out.zero_();

  const int64_t bh = B * H;
  const size_t smem_bytes = (size_t)(2 * Dc + Dr) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(mla_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);
  }
  mla_decode_kernel<<<(unsigned)bh, MLA_WARP, smem_bytes, stream>>>(
      q_abs.data_ptr<float>(), q_rope.data_ptr<float>(), c_kv_cache.data_ptr<float>(),
      k_rope_cache.data_ptr<float>(), out.data_ptr<float>(), (int)H, (int)N, (int)Dc, (int)Dr,
      (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace fni8
