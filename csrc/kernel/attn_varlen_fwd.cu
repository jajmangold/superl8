// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Launcher + torch wrapper for the varlen (cu_seqlens) int8-QK forward.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "attn_varlen_fwd.cuh"

namespace fni8 {

namespace {

template <int D>
void launch_varlen(const at::Tensor& q, const at::Tensor& q_scale, const at::Tensor& k,
                   const at::Tensor& k_scale, const at::Tensor& v, at::Tensor& out,
                   const at::Tensor& cu_q, const at::Tensor& cu_k, int batch,
                   int max_seqlen_q, int h_q, int h_kv, int gqa_group, bool causal,
                   cudaStream_t stream) {
  const dim3 grid((max_seqlen_q + BLOCK_M - 1) / BLOCK_M, (unsigned)batch, (unsigned)h_q);
  const dim3 block(THREADS);
  auto* qp = q.data_ptr<int8_t>();
  auto* qsp = q_scale.data_ptr<float>();
  auto* kp = k.data_ptr<int8_t>();
  auto* ksp = k_scale.data_ptr<float>();
  auto* vp = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
  auto* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  auto* cqp = cu_q.data_ptr<int>();
  auto* ckp = cu_k.data_ptr<int>();
  if (causal) {
    attn_int8_varlen_fwd_kernel<D, true><<<grid, block, 0, stream>>>(
        qp, qsp, kp, ksp, vp, op, cqp, ckp, h_q, h_kv, gqa_group);
  } else {
    attn_int8_varlen_fwd_kernel<D, false><<<grid, block, 0, stream>>>(
        qp, qsp, kp, ksp, vp, op, cqp, ckp, h_q, h_kv, gqa_group);
  }
}

}  // namespace

// Packed [total, H, D] layout. q/k int8, q_scale [total_q,H_q] & k_scale
// [total_k,H_kv] fp32 (softmax_scale*log2e folded into q_scale), v fp16.
// cu_seqlens int32 [batch+1]. Returns out fp16 [total_q, H_q, D].
at::Tensor attn_int8_varlen(const at::Tensor& q, const at::Tensor& q_scale,
                            const at::Tensor& k, const at::Tensor& k_scale,
                            const at::Tensor& v, const at::Tensor& cu_seqlens_q,
                            const at::Tensor& cu_seqlens_k, int64_t max_seqlen_q,
                            bool causal) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && q_scale.is_cuda() &&
                  k_scale.is_cuda() && cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(),
              "fni8 varlen: all inputs must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                  q.device() == q_scale.device() && q.device() == k_scale.device() &&
                  q.device() == cu_seqlens_q.device() && q.device() == cu_seqlens_k.device(),
              "fni8 varlen: all inputs must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kChar && k.scalar_type() == at::kChar,
              "fni8 varlen: q/k must be int8");
  TORCH_CHECK(q_scale.scalar_type() == at::kFloat && k_scale.scalar_type() == at::kFloat,
              "fni8 varlen: scales must be float32");
  TORCH_CHECK(v.scalar_type() == at::kHalf, "fni8 varlen: v must be float16");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_k.scalar_type() == at::kInt,
              "fni8 varlen: cu_seqlens must be int32");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "fni8 varlen: expect [total,H,D]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  q_scale.is_contiguous() && k_scale.is_contiguous() &&
                  cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
              "fni8 varlen: inputs must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(k.data_ptr<int8_t>()) % 4 == 0,
              "fni8 varlen: q/k must be 4-byte aligned for dp4a int32 loads");

  const auto H_Q = q.size(1), D = q.size(2);
  const auto H_KV = k.size(1);
  const int batch = (int)cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(batch >= 1 && cu_seqlens_k.size(0) == batch + 1, "fni8 varlen: cu_seqlens mismatch");
  TORCH_CHECK(H_KV > 0 && H_Q % H_KV == 0, "fni8 varlen: H_q must be divisible by H_kv (GQA)");
  TORCH_CHECK(k.size(2) == D && v.size(2) == D && v.size(1) == H_KV, "fni8 varlen: K/V shape");
  TORCH_CHECK(q_scale.numel() == q.size(0) * H_Q && k_scale.numel() == k.size(0) * H_KV,
              "fni8 varlen: scale shape mismatch");
  TORCH_CHECK(D == 32 || D == 64 || D == 128, "fni8 varlen: head dim must be 32/64/128, got ", D);
  const int gqa_group = (int)(H_Q / H_KV);

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({q.size(0), H_Q, D}, v.options());
  if (q.size(0) == 0 || max_seqlen_q == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (D) {
    case 32:
      launch_varlen<32>(q, q_scale, k, k_scale, v, out, cu_seqlens_q, cu_seqlens_k, batch,
                        (int)max_seqlen_q, (int)H_Q, (int)H_KV, gqa_group, causal, stream);
      break;
    case 64:
      launch_varlen<64>(q, q_scale, k, k_scale, v, out, cu_seqlens_q, cu_seqlens_k, batch,
                        (int)max_seqlen_q, (int)H_Q, (int)H_KV, gqa_group, causal, stream);
      break;
    case 128:
      launch_varlen<128>(q, q_scale, k, k_scale, v, out, cu_seqlens_q, cu_seqlens_k, batch,
                         (int)max_seqlen_q, (int)H_Q, (int)H_KV, gqa_group, causal, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace fni8
