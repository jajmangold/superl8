// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Launcher + torch wrapper for the fused FA2 backward (PR5b-1: fp accumulate).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <tuple>

#include "attn_bwd.cuh"

namespace fni8 {

namespace {

template <int D>
void launch_bwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                const at::Tensor& o, const at::Tensor& do_, const at::Tensor& lse,
                at::Tensor& dq, at::Tensor& dk, at::Tensor& dv, float scale,
                bool causal, int m_len, int n_len, int64_t bh_q, int64_t bh_kv,
                int h_q, int gqa_group, cudaStream_t stream) {
  auto* qp = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
  auto* kp = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
  auto* vp = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
  auto* op = reinterpret_cast<const __half*>(o.data_ptr<at::Half>());
  auto* dop = reinterpret_cast<const __half*>(do_.data_ptr<at::Half>());
  auto* lsep = lse.data_ptr<float>();
  auto* dqp = reinterpret_cast<__half*>(dq.data_ptr<at::Half>());
  auto* dkp = reinterpret_cast<__half*>(dk.data_ptr<at::Half>());
  auto* dvp = reinterpret_cast<__half*>(dv.data_ptr<at::Half>());

  constexpr int smem = bwd_smem_bytes<D>();
  const dim3 block(BWD_THREADS);
  const dim3 grid_dq((m_len + BWD_BM - 1) / BWD_BM, (unsigned)bh_q);    // per Q head
  const dim3 grid_dkv((n_len + BWD_BN - 1) / BWD_BN, (unsigned)bh_kv);  // per K/V head

#define LAUNCH_BWD(CAUSAL)                                                              \
  do {                                                                                  \
    cudaFuncSetAttribute(bwd_dq_kernel<D, CAUSAL>,                                       \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);             \
    cudaFuncSetAttribute(bwd_dkv_kernel<D, CAUSAL>,                                      \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);             \
    bwd_dq_kernel<D, CAUSAL><<<grid_dq, block, smem, stream>>>(                          \
        qp, kp, vp, op, dop, lsep, dqp, scale, m_len, n_len, h_q, gqa_group);           \
    bwd_dkv_kernel<D, CAUSAL><<<grid_dkv, block, smem, stream>>>(                        \
        qp, kp, vp, op, dop, lsep, dkp, dvp, scale, m_len, n_len, h_q, gqa_group);      \
  } while (0)

  if (causal) LAUNCH_BWD(true);
  else LAUNCH_BWD(false);
#undef LAUNCH_BWD
}

}  // namespace

// Returns (dq, dk, dv) fp16. q,k,v,o,do fp16 [B,H,S,D]; lse fp32 [B,H,M].
std::tuple<at::Tensor, at::Tensor, at::Tensor> attn_bwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& o,
    const at::Tensor& do_, const at::Tensor& lse, double scale, bool causal) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && o.is_cuda() && do_.is_cuda() &&
                  lse.is_cuda(),
              "fni8 bwd: all inputs must be CUDA");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == o.device() &&
                  q.device() == do_.device() && q.device() == lse.device(),
              "fni8 bwd: all inputs must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
                  v.scalar_type() == at::kHalf && o.scalar_type() == at::kHalf &&
                  do_.scalar_type() == at::kHalf,
              "fni8 bwd: q/k/v/o/do must be float16");
  TORCH_CHECK(lse.scalar_type() == at::kFloat, "fni8 bwd: lse must be float32");
  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
  auto oc = o.contiguous(), doc = do_.contiguous(), lc = lse.contiguous();

  const auto B = qc.size(0), H = qc.size(1), M = qc.size(2), D = qc.size(3);
  const auto N = kc.size(2);
  const auto H_KV = kc.size(1);   // GQA/MQA: dK/dV summed over the group's Q heads
  TORCH_CHECK(H_KV > 0 && H % H_KV == 0, "fni8 bwd: H_q must be divisible by H_kv (GQA)");
  const int gqa_group = (int)(H / H_KV);
  TORCH_CHECK(D == 32 || D == 64 || D == 128, "fni8 bwd: head dim must be 32/64/128, got ", D);

  const at::cuda::CUDAGuard guard(qc.device());
  auto dq = at::empty_like(qc);
  auto dk = at::empty_like(kc);   // [B, H_kv, N, D]
  auto dv = at::empty_like(vc);
  if (M == 0 || N == 0) return {dq.zero_(), dk.zero_(), dv.zero_()};
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (D) {
    case 32:
      launch_bwd<32>(qc, kc, vc, oc, doc, lc, dq, dk, dv, (float)scale, causal,
                     (int)M, (int)N, B * H, B * H_KV, (int)H, gqa_group, stream);
      break;
    case 64:
      launch_bwd<64>(qc, kc, vc, oc, doc, lc, dq, dk, dv, (float)scale, causal,
                     (int)M, (int)N, B * H, B * H_KV, (int)H, gqa_group, stream);
      break;
    case 128:
      launch_bwd<128>(qc, kc, vc, oc, doc, lc, dq, dk, dv, (float)scale, causal,
                      (int)M, (int)N, B * H, B * H_KV, (int)H, gqa_group, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv};
}

}  // namespace fni8
