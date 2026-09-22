// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Launcher + torch wrapper for the tree-attention verify (EAGLE spec-decode).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "attn_tree_fwd.cuh"

namespace fni8 {

namespace {

template <int D>
void launch(const at::Tensor& q, const at::Tensor& q_scale, const at::Tensor& k,
            const at::Tensor& k_scale, const at::Tensor& v, at::Tensor& out,
            const at::Tensor& tree_mask, int tree_offset, int t_len,
            int mask_batch_stride, int m_len, int n_len, int64_t bh, int h_q,
            int gqa_group, cudaStream_t stream) {
  const dim3 grid((m_len + BLOCK_M - 1) / BLOCK_M, (unsigned)bh);
  const dim3 block(THREADS);
  attn_int8_tree_kernel<D><<<grid, block, 0, stream>>>(
      q.data_ptr<int8_t>(), q_scale.data_ptr<float>(), k.data_ptr<int8_t>(),
      k_scale.data_ptr<float>(), reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), tree_mask.data_ptr<int8_t>(),
      tree_offset, t_len, mask_batch_stride, m_len, n_len, h_q, gqa_group);
}

}  // namespace

// Tree-verify: q [B,H,T,D] int8 drafts, q_scale [B,H,T] fp32 folded, k [B,H_kv,N,D]
// int8, k_scale [B,H_kv,N], v [B,H_kv,N,D] fp16. tree_mask [T,T] int8 (row qi = which
// nodes qi attends among the T tree keys). N = prefix + T; tree_offset = N - T.
at::Tensor attn_int8_tree(const at::Tensor& q, const at::Tensor& q_scale,
                          const at::Tensor& k, const at::Tensor& k_scale,
                          const at::Tensor& v, const at::Tensor& tree_mask) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && q_scale.is_cuda() &&
                  k_scale.is_cuda() && tree_mask.is_cuda(),
              "fni8 tree: all inputs must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                  q.device() == q_scale.device() && q.device() == k_scale.device() &&
                  q.device() == tree_mask.device(),
              "fni8 tree: all inputs must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kChar && k.scalar_type() == at::kChar,
              "fni8 tree: q/k must be int8");
  TORCH_CHECK(q_scale.scalar_type() == at::kFloat && k_scale.scalar_type() == at::kFloat,
              "fni8 tree: scales must be float32");
  TORCH_CHECK(v.scalar_type() == at::kHalf, "fni8 tree: v must be float16");
  TORCH_CHECK(tree_mask.scalar_type() == at::kChar, "fni8 tree: tree_mask must be int8");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "fni8 tree: expect [B,H,S,D]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  q_scale.is_contiguous() && k_scale.is_contiguous() &&
                  tree_mask.is_contiguous(),
              "fni8 tree: inputs must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(k.data_ptr<int8_t>()) % 4 == 0,
              "fni8 tree: q/k must be 4-byte aligned for dp4a int32 loads");

  const auto B = q.size(0), H = q.size(1), T = q.size(2), D = q.size(3);
  const auto N = k.size(2);
  const auto H_KV = k.size(1);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == H_KV,
              "fni8 tree: batch/head mismatch");
  TORCH_CHECK(H_KV > 0 && H % H_KV == 0, "fni8 tree: H_q must be divisible by H_kv (GQA)");
  TORCH_CHECK(k.size(3) == D && v.size(3) == D && v.size(2) == N, "fni8 tree: K/V shape");
  TORCH_CHECK(N >= T, "fni8 tree: cache length N must be >= tree size T");
  // tree_mask is [T,T] (one shared tree) or [B,T,T] (per-request trees).
  const bool per_req = tree_mask.dim() == 3;
  TORCH_CHECK((tree_mask.dim() == 2 && tree_mask.size(0) == T && tree_mask.size(1) == T) ||
                  (per_req && tree_mask.size(0) == B && tree_mask.size(1) == T &&
                   tree_mask.size(2) == T),
              "fni8 tree: tree_mask must be [T,T] (shared) or [B,T,T] (per-request)");
  const int mask_batch_stride = per_req ? (int)(T * T) : 0;
  TORCH_CHECK(q_scale.numel() == B * H * T && k_scale.numel() == B * H_KV * N,
              "fni8 tree: scale shape mismatch");
  TORCH_CHECK(D == 32 || D == 64 || D == 128, "fni8 tree: head dim must be 32/64/128, got ", D);
  const int gqa_group = (int)(H / H_KV);
  const int tree_offset = (int)(N - T);

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({B, H, T, D}, v.options());
  if (T == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (D) {
    case 32:
      launch<32>(q, q_scale, k, k_scale, v, out, tree_mask, tree_offset, (int)T, mask_batch_stride, (int)T,
                 (int)N, B * H, (int)H, gqa_group, stream);
      break;
    case 64:
      launch<64>(q, q_scale, k, k_scale, v, out, tree_mask, tree_offset, (int)T, mask_batch_stride, (int)T,
                 (int)N, B * H, (int)H, gqa_group, stream);
      break;
    case 128:
      launch<128>(q, q_scale, k, k_scale, v, out, tree_mask, tree_offset, (int)T, mask_batch_stride, (int)T,
                  (int)N, B * H, (int)H, gqa_group, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace fni8
