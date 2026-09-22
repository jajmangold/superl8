// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: launcher + torch wrapper for attn_int8_fwd.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <tuple>

#include "attn_int8_fwd.cuh"

namespace fni8 {

namespace {

// `VT` is V's storage type (__half or __nv_bfloat16); the output tensor is
// allocated with v.options() (see run_fwd), so out shares VT too.
template <int D, typename VT>
void launch(const at::Tensor& q, const at::Tensor& q_scale, const at::Tensor& k,
            const at::Tensor& k_scale, const at::Tensor& v, at::Tensor& out,
            float* lse_ptr, bool causal, int m_len, int n_len, int64_t bh,
            int h_q, int gqa_group, int window_left, cudaStream_t stream) {
  const dim3 grid((m_len + BLOCK_M - 1) / BLOCK_M, (unsigned)bh);
  const dim3 block(THREADS);
  auto* qp = q.data_ptr<int8_t>();
  auto* qsp = q_scale.data_ptr<float>();
  auto* kp = k.data_ptr<int8_t>();
  auto* ksp = k_scale.data_ptr<float>();
  auto* vp = reinterpret_cast<const VT*>(v.data_ptr());
  auto* op = reinterpret_cast<VT*>(out.data_ptr());
  if (causal) {
    attn_int8_fwd_kernel<D, true, VT><<<grid, block, 0, stream>>>(
        qp, qsp, kp, ksp, vp, op, lse_ptr, m_len, n_len, h_q, gqa_group, window_left);
  } else {
    attn_int8_fwd_kernel<D, false, VT><<<grid, block, 0, stream>>>(
        qp, qsp, kp, ksp, vp, op, lse_ptr, m_len, n_len, h_q, gqa_group, window_left);
  }
}

// Dynamic-smem launch (D whose tiles exceed the 48 KB static cap, e.g. D=256).
template <int D, typename VT>
void launch_dyn(const at::Tensor& q, const at::Tensor& q_scale, const at::Tensor& k,
                const at::Tensor& k_scale, const at::Tensor& v, at::Tensor& out,
                float* lse_ptr, bool causal, int m_len, int n_len, int64_t bh,
                int h_q, int gqa_group, int window_left, cudaStream_t stream) {
  const dim3 grid((m_len + BLOCK_M - 1) / BLOCK_M, (unsigned)bh);
  const dim3 block(THREADS);
  constexpr int smem = fwd_dyn_smem_bytes<D, VT>();
  auto* qp = q.data_ptr<int8_t>();
  auto* qsp = q_scale.data_ptr<float>();
  auto* kp = k.data_ptr<int8_t>();
  auto* ksp = k_scale.data_ptr<float>();
  auto* vp = reinterpret_cast<const VT*>(v.data_ptr());
  auto* op = reinterpret_cast<VT*>(out.data_ptr());
#define LAUNCH_DYN(CAUSAL)                                                          \
  do {                                                                             \
    cudaFuncSetAttribute(attn_int8_fwd_dyn_kernel<D, CAUSAL, VT>,                   \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);        \
    attn_int8_fwd_dyn_kernel<D, CAUSAL, VT><<<grid, block, smem, stream>>>(         \
        qp, qsp, kp, ksp, vp, op, lse_ptr, m_len, n_len, h_q, gqa_group, window_left);          \
  } while (0)
  if (causal) LAUNCH_DYN(true);
  else LAUNCH_DYN(false);
#undef LAUNCH_DYN
}

// Dispatch D (compile-time template arg) x VT (v's runtime dtype: fp16/bf16).
template <int D>
void launch_typed(const at::Tensor& q, const at::Tensor& q_scale, const at::Tensor& k,
                  const at::Tensor& k_scale, const at::Tensor& v, at::Tensor& out,
                  float* lse_ptr, bool causal, int m_len, int n_len, int64_t bh,
                  int h_q, int gqa_group, int window_left, cudaStream_t stream) {
  if (v.scalar_type() == at::kBFloat16) {
    launch<D, __nv_bfloat16>(q, q_scale, k, k_scale, v, out, lse_ptr, causal, m_len, n_len,
                             bh, h_q, gqa_group, window_left, stream);
  } else {
    launch<D, __half>(q, q_scale, k, k_scale, v, out, lse_ptr, causal, m_len, n_len, bh,
                      h_q, gqa_group, window_left, stream);
  }
}

template <int D>
void launch_dyn_typed(const at::Tensor& q, const at::Tensor& q_scale, const at::Tensor& k,
                      const at::Tensor& k_scale, const at::Tensor& v, at::Tensor& out,
                      float* lse_ptr, bool causal, int m_len, int n_len, int64_t bh,
                      int h_q, int gqa_group, int window_left, cudaStream_t stream) {
  if (v.scalar_type() == at::kBFloat16) {
    launch_dyn<D, __nv_bfloat16>(q, q_scale, k, k_scale, v, out, lse_ptr, causal, m_len,
                                 n_len, bh, h_q, gqa_group, window_left, stream);
  } else {
    launch_dyn<D, __half>(q, q_scale, k, k_scale, v, out, lse_ptr, causal, m_len, n_len,
                          bh, h_q, gqa_group, window_left, stream);
  }
}

// Shared impl; returns (out, lse). lse is empty unless want_lse.
std::tuple<at::Tensor, at::Tensor> run_fwd(const at::Tensor& q, const at::Tensor& q_scale,
                                           const at::Tensor& k, const at::Tensor& k_scale,
                                           const at::Tensor& v, bool causal, bool want_lse,
                                           int window_left) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && q_scale.is_cuda() &&
                  k_scale.is_cuda(),
              "fni8: all inputs (incl. scales) must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                  q.device() == q_scale.device() && q.device() == k_scale.device(),
              "fni8: all inputs must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kChar && k.scalar_type() == at::kChar,
              "fni8: q/k must be int8");
  TORCH_CHECK(q_scale.scalar_type() == at::kFloat && k_scale.scalar_type() == at::kFloat,
              "fni8: scales must be float32");
  TORCH_CHECK(v.scalar_type() == at::kHalf || v.scalar_type() == at::kBFloat16,
              "fni8: v must be float16 or bfloat16");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "fni8: expect [B,H,S,D]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  q_scale.is_contiguous() && k_scale.is_contiguous(),
              "fni8: inputs must be contiguous");
  // The kernel reads int8 rows as int32x (dp4a packing); rows start at a
  // multiple of D bytes and torch aligns allocations >=16B, so 4B alignment
  // holds — assert it rather than trust it silently.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(k.data_ptr<int8_t>()) % 4 == 0,
              "fni8: q/k must be 4-byte aligned for dp4a int32 loads");

  const auto B = q.size(0), H = q.size(1), M = q.size(2), D = q.size(3);
  const auto N = k.size(2);
  const auto H_KV = k.size(1);   // GQA/MQA: K/V may have fewer heads than Q
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == H_KV,
              "fni8: batch/head mismatch");
  TORCH_CHECK(H_KV > 0 && H % H_KV == 0, "fni8: H_q must be divisible by H_kv (GQA)");
  const int gqa_group = (int)(H / H_KV);
  TORCH_CHECK(k.size(3) == D && v.size(3) == D && v.size(2) == N, "fni8: K/V shape mismatch");
  TORCH_CHECK(q_scale.numel() == B * H * M && k_scale.numel() == B * H_KV * N,
              "fni8: scale shape mismatch");
  TORCH_CHECK(D == 32 || D == 64 || D == 72 || D == 80 || D == 128 || D == 256,
              "fni8: head dim must be 32/64/72/80/128/256, got ", D);

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({B, H, M, D}, v.options());
  auto lse = want_lse ? at::empty({B, H, M}, q.options().dtype(at::kFloat))
                      : at::empty({0}, q.options().dtype(at::kFloat));
  if (M == 0) return {out, lse};
  float* lse_ptr = want_lse ? lse.data_ptr<float>() : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (D) {
    case 32:
      launch_typed<32>(q, q_scale, k, k_scale, v, out, lse_ptr, causal, (int)M, (int)N, B * H,
                       (int)H, gqa_group, window_left, stream);
      break;
    case 64:
      launch_typed<64>(q, q_scale, k, k_scale, v, out, lse_ptr, causal, (int)M, (int)N, B * H,
                       (int)H, gqa_group, window_left, stream);
      break;
    case 72:   // odd head dims (D multiple of 4 -> dp4a packs; static smem fits)
      launch_typed<72>(q, q_scale, k, k_scale, v, out, lse_ptr, causal, (int)M, (int)N, B * H,
                       (int)H, gqa_group, window_left, stream);
      break;
    case 80:
      launch_typed<80>(q, q_scale, k, k_scale, v, out, lse_ptr, causal, (int)M, (int)N, B * H,
                       (int)H, gqa_group, window_left, stream);
      break;
    case 128:
      launch_typed<128>(q, q_scale, k, k_scale, v, out, lse_ptr, causal, (int)M, (int)N, B * H,
                        (int)H, gqa_group, window_left, stream);
      break;
    case 256:  // exceeds 48KB static smem -> dynamic-smem kernel
      launch_dyn_typed<256>(q, q_scale, k, k_scale, v, out, lse_ptr, causal, (int)M, (int)N,
                            B * H, (int)H, gqa_group, window_left, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}

}  // namespace

// q [B,H,M,D] int8, q_scale [B,H,M] fp32 (softmax_scale*log2e folded),
// k [B,H,N,D] int8 (pre-smoothed), k_scale [B,H,N] fp32, v [B,H,N,D] fp16 or
// bf16 -> out shares v's dtype (bf16 avoids the fp16 overflow that black-images
// bf16-native models, e.g. Gemma / diffusion DiTs — issue #11).
at::Tensor attn_int8_fwd(const at::Tensor& q, const at::Tensor& q_scale,
                         const at::Tensor& k, const at::Tensor& k_scale,
                         const at::Tensor& v, bool causal, int64_t window_left) {
  return std::get<0>(run_fwd(q, q_scale, k, k_scale, v, causal, /*want_lse=*/false,
                             (int)window_left));
}

// Same forward, also returns the per-row log-sum-exp (natural log) needed by
// the backward pass. Returns (out [B,H,M,D] sharing v's dtype, lse fp32 [B,H,M]).
std::tuple<at::Tensor, at::Tensor> attn_int8_fwd_train(const at::Tensor& q,
                                                       const at::Tensor& q_scale,
                                                       const at::Tensor& k,
                                                       const at::Tensor& k_scale,
                                                       const at::Tensor& v, bool causal) {
  return run_fwd(q, q_scale, k, k_scale, v, causal, /*want_lse=*/true, /*window_left=*/-1);
}

}  // namespace fni8
