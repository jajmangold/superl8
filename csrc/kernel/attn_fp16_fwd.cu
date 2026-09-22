// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: launcher + torch wrapper for attn_fp16_fwd — the
// fp16/bf16 half2 FlashAttention-2 prefill kernel (the O(N)-memory fp16 sibling
// of attn_int8_fwd, for video-DiT attention that OOMs torch SDPA on Volta).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <tuple>

#include "attn_fp16_fwd.cuh"

namespace fni8 {

namespace {

constexpr float LOG2E_F = 1.4426950408889634f;  // log2(e)

// Additive-mask strides (element units; 0 = broadcast over that axis).
struct MaskStrides {
  int64_t sb, sh, sm, sn;
};

// Static-smem launch (D whose tiles fit the 48 KB cap: <= 128).
template <int D, typename T>
void fp16_launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                 at::Tensor& out, float* lse_ptr, const T* mask, MaskStrides ms,
                 float scale, bool causal, int m_len, int n_len, int64_t bh, int h_q,
                 int gqa_group, int window_left, cudaStream_t stream) {
  const dim3 grid((m_len + FP16_BLOCK_M - 1) / FP16_BLOCK_M, (unsigned)bh);
  const dim3 block(FP16_THREADS);
  auto* qp = reinterpret_cast<const T*>(q.data_ptr());
  auto* kp = reinterpret_cast<const T*>(k.data_ptr());
  auto* vp = reinterpret_cast<const T*>(v.data_ptr());
  auto* op = reinterpret_cast<T*>(out.data_ptr());
  if (causal) {
    attn_fp16_fwd_kernel<D, true, T><<<grid, block, 0, stream>>>(
        qp, kp, vp, op, lse_ptr, mask, ms.sb, ms.sh, ms.sm, ms.sn, scale, m_len, n_len,
        h_q, gqa_group, window_left);
  } else {
    attn_fp16_fwd_kernel<D, false, T><<<grid, block, 0, stream>>>(
        qp, kp, vp, op, lse_ptr, mask, ms.sb, ms.sh, ms.sm, ms.sn, scale, m_len, n_len,
        h_q, gqa_group, window_left);
  }
}

// Dynamic-smem launch (D whose tiles exceed the 48 KB static cap, e.g. D=256).
template <int D, typename T>
void fp16_launch_dyn(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                     at::Tensor& out, float* lse_ptr, const T* mask, MaskStrides ms,
                     float scale, bool causal, int m_len, int n_len, int64_t bh, int h_q,
                     int gqa_group, int window_left, cudaStream_t stream) {
  const dim3 grid((m_len + FP16_BLOCK_M - 1) / FP16_BLOCK_M, (unsigned)bh);
  const dim3 block(FP16_THREADS);
  constexpr int smem = fp16_fwd_dyn_smem_bytes<D, T>();
  auto* qp = reinterpret_cast<const T*>(q.data_ptr());
  auto* kp = reinterpret_cast<const T*>(k.data_ptr());
  auto* vp = reinterpret_cast<const T*>(v.data_ptr());
  auto* op = reinterpret_cast<T*>(out.data_ptr());
#define FP16_LAUNCH_DYN(CAUSAL)                                                       \
  do {                                                                               \
    cudaFuncSetAttribute(attn_fp16_fwd_dyn_kernel<D, CAUSAL, T>,                      \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);          \
    attn_fp16_fwd_dyn_kernel<D, CAUSAL, T><<<grid, block, smem, stream>>>(            \
        qp, kp, vp, op, lse_ptr, mask, ms.sb, ms.sh, ms.sm, ms.sn, scale, m_len,      \
        n_len, h_q, gqa_group, window_left);                                         \
  } while (0)
  if (causal) FP16_LAUNCH_DYN(true);
  else FP16_LAUNCH_DYN(false);
#undef FP16_LAUNCH_DYN
}

// Dispatch D (compile-time) x T (runtime dtype: fp16/bf16 — q/k/v share it).
template <typename T>
void fp16_dispatch_d(int D, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                     at::Tensor& out, float* lse_ptr, const T* mask, MaskStrides ms,
                     float scale, bool causal, int m_len, int n_len, int64_t bh, int h_q,
                     int gqa_group, int window_left, cudaStream_t stream) {
#define FP16_CASE_STATIC(DD)                                                              \
  case DD:                                                                                \
    fp16_launch<DD, T>(q, k, v, out, lse_ptr, mask, ms, scale, causal, m_len, n_len, bh,  \
                       h_q, gqa_group, window_left, stream);                              \
    break
  switch (D) {
    FP16_CASE_STATIC(32);
    FP16_CASE_STATIC(64);
    FP16_CASE_STATIC(72);
    FP16_CASE_STATIC(80);
    FP16_CASE_STATIC(128);
    case 256:  // tiles exceed 48 KB static smem -> dynamic-smem kernel
      fp16_launch_dyn<256, T>(q, k, v, out, lse_ptr, mask, ms, scale, causal, m_len, n_len,
                              bh, h_q, gqa_group, window_left, stream);
      break;
  }
#undef FP16_CASE_STATIC
}

// Shared impl; returns (out, lse). lse is empty unless want_lse.
std::tuple<at::Tensor, at::Tensor> run_fp16_fwd(const at::Tensor& q, const at::Tensor& k,
                                                const at::Tensor& v,
                                                const c10::optional<at::Tensor>& mask_opt,
                                                double scale_arg, bool has_scale,
                                                bool causal, bool want_lse,
                                                int window_left) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "fni8: q/k/v must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
              "fni8: q/k/v must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
              "fni8: q/k/v must be float16 or bfloat16");
  TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
              "fni8: q/k/v must share one dtype (fp16 or bf16)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "fni8: expect [B,H,S,D]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "fni8: q/k/v must be contiguous");

  const auto B = q.size(0), H = q.size(1), M = q.size(2), D = q.size(3);
  const auto N = k.size(2);
  const auto H_KV = k.size(1);   // GQA/MQA: K/V may have fewer heads than Q
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == H_KV,
              "fni8: batch/head mismatch");
  TORCH_CHECK(H_KV > 0 && H % H_KV == 0, "fni8: H_q must be divisible by H_kv (GQA)");
  const int gqa_group = (int)(H / H_KV);
  TORCH_CHECK(k.size(3) == D && v.size(3) == D && v.size(2) == N, "fni8: K/V shape mismatch");
  TORCH_CHECK(D == 32 || D == 64 || D == 72 || D == 80 || D == 128 || D == 256,
              "fni8: head dim must be 32/64/72/80/128/256, got ", D);

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({B, H, M, D}, v.options());
  auto lse = want_lse ? at::empty({B, H, M}, q.options().dtype(at::kFloat))
                      : at::empty({0}, q.options().dtype(at::kFloat));
  if (M == 0) return {out, lse};
  float* lse_ptr = want_lse ? lse.data_ptr<float>() : nullptr;

  const float softmax_scale =
      has_scale ? (float)scale_arg : (float)(1.0 / std::sqrt((double)D));
  const float scale = softmax_scale * LOG2E_F;  // fold log2(e) for exp2 softmax
  auto stream = at::cuda::getCurrentCUDAStream();

  // Optional additive mask [*, M, N] (natural-log bias), broadcastable over the
  // leading (B, H_q) axes. We pass element strides (0 = broadcast) so the caller's
  // own tensor is indexed in place — no O(N^2) expansion that would defeat the
  // memory win. Mask shares q/k/v's dtype and is indexed by the Q head.
  MaskStrides ms{0, 0, 0, 0};
  const void* mask_ptr = nullptr;
  if (mask_opt.has_value()) {
    const at::Tensor& mk = *mask_opt;
    TORCH_CHECK(mk.is_cuda() && mk.device() == q.device(),
                "fni8: mask must be a CUDA tensor on q's device");
    TORCH_CHECK(mk.scalar_type() == q.scalar_type(),
                "fni8: mask dtype must match q/k/v");
    TORCH_CHECK(mk.dim() == 4, "fni8: mask must be 4-D [B|1, H|1, M, N]");
    TORCH_CHECK(mk.size(2) == M && mk.size(3) == N, "fni8: mask trailing dims must be [M,N]");
    TORCH_CHECK((mk.size(0) == B || mk.size(0) == 1) && (mk.size(1) == H || mk.size(1) == 1),
                "fni8: mask leading dims must broadcast to [B,H]");
    // Zero the stride on any broadcast (size-1) axis; use the tensor's own strides
    // otherwise (works for a non-contiguous broadcast view too).
    ms.sb = (mk.size(0) == 1) ? 0 : mk.stride(0);
    ms.sh = (mk.size(1) == 1) ? 0 : mk.stride(1);
    ms.sm = mk.stride(2);
    ms.sn = mk.stride(3);
    mask_ptr = mk.data_ptr();
  }

  if (q.scalar_type() == at::kBFloat16) {
    fp16_dispatch_d<__nv_bfloat16>(D, q, k, v, out, lse_ptr,
                                   reinterpret_cast<const __nv_bfloat16*>(mask_ptr), ms,
                                   scale, causal, (int)M, (int)N, B * H, (int)H, gqa_group,
                                   window_left, stream);
  } else {
    fp16_dispatch_d<__half>(D, q, k, v, out, lse_ptr,
                            reinterpret_cast<const __half*>(mask_ptr), ms, scale, causal,
                            (int)M, (int)N, B * H, (int)H, gqa_group, window_left, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}

}  // namespace

// q/k/v fp16 or bf16 [B,H,S,D] (all one dtype) -> out shares that dtype. Tiled
// FA-2 prefill: half2 CUDA-core QK^T, fp32 online-softmax, fp16/bf16 PV. O(N)
// memory (streams K/V tiles) where torch SDPA materializes O(N^2) and OOMs on
// long video-DiT attention. `window_left >= 0` restricts key range (Mistral
// sliding window; only meaningful with causal); -1 = none.
at::Tensor attn_fp16_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                         bool causal, c10::optional<double> scale, int64_t window_left,
                         const c10::optional<at::Tensor>& mask) {
  return std::get<0>(run_fp16_fwd(q, k, v, mask, scale.value_or(0.0), scale.has_value(),
                                  causal, /*want_lse=*/false, (int)window_left));
}

// Same forward, also returns the per-row log-sum-exp (natural log) for the
// backward pass. Returns (out [B,H,M,D] sharing v's dtype, lse fp32 [B,H,M]).
std::tuple<at::Tensor, at::Tensor> attn_fp16_fwd_train(const at::Tensor& q,
                                                       const at::Tensor& k,
                                                       const at::Tensor& v, bool causal,
                                                       c10::optional<double> scale) {
  return run_fp16_fwd(q, k, v, /*mask=*/c10::nullopt, scale.value_or(0.0),
                      scale.has_value(), causal, /*want_lse=*/true, /*window_left=*/-1);
}

}  // namespace fni8
