// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: launcher + torch wrapper for full-W8A8 forward.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "attn_w8a8_fwd.cuh"

namespace fni8 {

namespace {

template <int D, bool PER_WARP>
void launch(const at::Tensor& q, const at::Tensor& q_scale, const at::Tensor& k,
            const at::Tensor& k_scale, const at::Tensor& v, const at::Tensor& v_scale,
            at::Tensor& out, bool causal, int m_len, int n_len, int64_t bh,
            int h_q, int gqa_group, int causal_diag, cudaStream_t stream) {
  const dim3 grid((m_len + BLOCK_M - 1) / BLOCK_M, (unsigned)bh);
  const dim3 block(THREADS);
  auto* qp = q.data_ptr<int8_t>();
  auto* qsp = q_scale.data_ptr<float>();
  auto* kp = k.data_ptr<int8_t>();
  auto* ksp = k_scale.data_ptr<float>();
  auto* vp = v.data_ptr<int8_t>();
  auto* vsp = v_scale.data_ptr<float>();
  auto* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  if (causal) {
    attn_w8a8_fwd_kernel<D, true, PER_WARP><<<grid, block, 0, stream>>>(
        qp, qsp, kp, ksp, vp, vsp, op, m_len, n_len, h_q, gqa_group, causal_diag);
  } else {
    attn_w8a8_fwd_kernel<D, false, PER_WARP><<<grid, block, 0, stream>>>(
        qp, qsp, kp, ksp, vp, vsp, op, m_len, n_len, h_q, gqa_group, causal_diag);
  }
}

// Dynamic-smem launch for head dims whose W8A8 tiles exceed the 48 KB static
// cap (D=256: ~48.25 KB). Opts the kernel into >48 KB dynamic smem via
// cudaFuncSetAttribute (Volta caps at 96 KB), then launches with that smem.
template <int D, bool PER_WARP>
void launch_dyn(const at::Tensor& q, const at::Tensor& q_scale, const at::Tensor& k,
                const at::Tensor& k_scale, const at::Tensor& v, const at::Tensor& v_scale,
                at::Tensor& out, bool causal, int m_len, int n_len, int64_t bh,
                int h_q, int gqa_group, int causal_diag, cudaStream_t stream) {
  const dim3 grid((m_len + BLOCK_M - 1) / BLOCK_M, (unsigned)bh);
  const dim3 block(THREADS);
  auto* qp = q.data_ptr<int8_t>();
  auto* qsp = q_scale.data_ptr<float>();
  auto* kp = k.data_ptr<int8_t>();
  auto* ksp = k_scale.data_ptr<float>();
  auto* vp = v.data_ptr<int8_t>();
  auto* vsp = v_scale.data_ptr<float>();
  auto* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  constexpr int smem = w8a8_dyn_smem_bytes<D, PER_WARP>();
  static_assert(smem <= 98304, "W8A8 dyn tiles exceed the 96 KB Volta smem cap");
  if (causal) {
    auto* fn = attn_w8a8_fwd_dyn_kernel<D, true, PER_WARP>;
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fn<<<grid, block, smem, stream>>>(qp, qsp, kp, ksp, vp, vsp, op, m_len, n_len, h_q,
                                      gqa_group, causal_diag);
  } else {
    auto* fn = attn_w8a8_fwd_dyn_kernel<D, false, PER_WARP>;
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fn<<<grid, block, smem, stream>>>(qp, qsp, kp, ksp, vp, vsp, op, m_len, n_len, h_q,
                                      gqa_group, causal_diag);
  }
}

}  // namespace

// Full-W8A8: q/k/v int8, v_scale [B,H,D] per-channel; out fp16.
at::Tensor attn_w8a8_fwd(const at::Tensor& q, const at::Tensor& q_scale,
                         const at::Tensor& k, const at::Tensor& k_scale,
                         const at::Tensor& v, const at::Tensor& v_scale, bool causal,
                         int64_t causal_diag, bool per_warp_quant) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && q_scale.is_cuda() &&
                  k_scale.is_cuda() && v_scale.is_cuda(),
              "fni8: all inputs must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                  q.device() == q_scale.device() && q.device() == k_scale.device() &&
                  q.device() == v_scale.device(),
              "fni8: all inputs must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kChar && k.scalar_type() == at::kChar &&
                  v.scalar_type() == at::kChar,
              "fni8: q/k/v must be int8");
  TORCH_CHECK(q_scale.scalar_type() == at::kFloat && k_scale.scalar_type() == at::kFloat &&
                  v_scale.scalar_type() == at::kFloat,
              "fni8: scales must be float32");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "fni8: expect [B,H,S,D]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  q_scale.is_contiguous() && k_scale.is_contiguous() &&
                  v_scale.is_contiguous(),
              "fni8: inputs must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(k.data_ptr<int8_t>()) % 4 == 0,
              "fni8: q/k must be 4-byte aligned for dp4a int32 loads");

  const auto B = q.size(0), H = q.size(1), M = q.size(2), D = q.size(3);
  const auto N = k.size(2);
  const auto H_KV = k.size(1);   // GQA/MQA
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == H_KV,
              "fni8: batch/head mismatch");
  TORCH_CHECK(H_KV > 0 && H % H_KV == 0, "fni8: H_q must be divisible by H_kv (GQA)");
  const int gqa_group = (int)(H / H_KV);
  TORCH_CHECK(k.size(3) == D && v.size(3) == D && v.size(2) == N, "fni8: K/V shape mismatch");
  TORCH_CHECK(q_scale.numel() == B * H * M && k_scale.numel() == B * H_KV * N &&
                  v_scale.numel() == B * H_KV * D,
              "fni8: scale shape mismatch");
  TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
              "fni8: head dim must be 32/64/128/256, got ", D);

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({B, H, M, D}, v.options().dtype(at::kHalf));
  if (M == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();

  const int cdiag = (int)causal_diag;
  // Dispatch on D first, then per_warp_quant (both compile-time template params).
  // This gives 3 D x 2 PER_WARP = 6 kernel variants per causal/non-causal.
  if (per_warp_quant) {
    switch (D) {
      case 32:
        launch<32, true>(q, q_scale, k, k_scale, v, v_scale, out, causal, (int)M, (int)N,
                         B * H, (int)H, gqa_group, cdiag, stream);
        break;
      case 64:
        launch<64, true>(q, q_scale, k, k_scale, v, v_scale, out, causal, (int)M, (int)N,
                         B * H, (int)H, gqa_group, cdiag, stream);
        break;
      case 128:
        launch<128, true>(q, q_scale, k, k_scale, v, v_scale, out, causal, (int)M, (int)N,
                          B * H, (int)H, gqa_group, cdiag, stream);
        break;
      case 256:
        launch_dyn<256, true>(q, q_scale, k, k_scale, v, v_scale, out, causal, (int)M, (int)N,
                              B * H, (int)H, gqa_group, cdiag, stream);
        break;
    }
  } else {
    switch (D) {
      case 32:
        launch<32, false>(q, q_scale, k, k_scale, v, v_scale, out, causal, (int)M, (int)N,
                          B * H, (int)H, gqa_group, cdiag, stream);
        break;
      case 64:
        launch<64, false>(q, q_scale, k, k_scale, v, v_scale, out, causal, (int)M, (int)N,
                          B * H, (int)H, gqa_group, cdiag, stream);
        break;
      case 128:
        launch<128, false>(q, q_scale, k, k_scale, v, v_scale, out, causal, (int)M, (int)N,
                           B * H, (int)H, gqa_group, cdiag, stream);
        break;
      case 256:
        launch_dyn<256, false>(q, q_scale, k, k_scale, v, v_scale, out, causal, (int)M, (int)N,
                               B * H, (int)H, gqa_group, cdiag, stream);
        break;
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace fni8
