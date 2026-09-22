// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Launcher + torch wrapper for the Lightning attention kernel.
//
// v1 — naive sequential fp32 Lightning recurrence (ground truth).
// v2 (next PR) — int8 dp4a variant.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdint>
#include <tuple>

#include "lightning_attn.cuh"

namespace fni8 {

namespace {
constexpr int64_t LIGHTNING_V1_MAX_DIM = 128;
constexpr size_t  LIGHTNING_MAX_SMEM   = 98304;
}  // namespace

std::tuple<at::Tensor, at::Tensor> lightning_attn_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& initial_state) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "fni8.lightning_attn_fwd: inputs must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                  v.scalar_type() == at::kFloat,
              "fni8.lightning_attn_fwd: v1 is fp32-only (state recurrence is "
              "numerically load-bearing, per AGENTS.md)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "fni8.lightning_attn_fwd: q,k,v must be [B,H,T,D]");
  const int64_t B = q.size(0), H = q.size(1), T = q.size(2), Dk = q.size(3);
  const int64_t Dv = v.size(3);
  TORCH_CHECK(k.sizes() == q.sizes(), "fni8.lightning_attn_fwd: k shape must match q");
  TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == T,
              "fni8.lightning_attn_fwd: v shape mismatch");
  TORCH_CHECK(Dk <= LIGHTNING_V1_MAX_DIM && Dv <= LIGHTNING_V1_MAX_DIM,
              "fni8.lightning_attn_fwd: v1 supports Dk,Dv <= 128 (untiled shared-mem "
              "state); larger head dims need the v2 chunked/tiled kernel");

  const at::cuda::CUDAGuard guard(q.device());
  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();

  const float* init_ptr = nullptr;
  at::Tensor init_c;
  if (initial_state.has_value() && initial_state->defined()) {
    init_c = initial_state->contiguous();
    TORCH_CHECK(init_c.scalar_type() == at::kFloat, "fni8.lightning_attn_fwd: "
                "initial_state must be float32");
    TORCH_CHECK(init_c.dim() == 4 && init_c.size(0) == B && init_c.size(1) == H &&
                    init_c.size(2) == Dv && init_c.size(3) == Dk,
                "fni8.lightning_attn_fwd: initial_state must be [B,H,Dv,Dk]");
    init_ptr = init_c.data_ptr<float>();
  }

  auto out = at::empty({B, H, T, Dv}, qc.options());
  auto final_state = at::empty({B, H, Dv, Dk}, qc.options());
  if (B * H == 0) return {out, final_state};

  const size_t smem_bytes =
      (size_t)(Dv * Dk + 2 * Dk + Dv) * sizeof(float);
  TORCH_CHECK(smem_bytes <= LIGHTNING_MAX_SMEM,
              "fni8.lightning_attn_fwd: state (Dv*Dk) exceeds the Volta shared-mem cap");
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(lightning_attn_fwd_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  lightning_attn_fwd_kernel<<<(unsigned)(B * H), LIGHTNING_V1_THREADS, smem_bytes, stream>>>(
      qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(), init_ptr,
      out.data_ptr<float>(), final_state.data_ptr<float>(),
      (int)H, (int)T, (int)Dk, (int)Dv);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, final_state};
}

// ===========================================================================
// v2 (issue #42): int8 dp4a chunked Lightning attention
// ===========================================================================

namespace {
int compute_chunk_size_v2(int Dk, int Dv, int Dkp) {
  constexpr int max_c = LIGHTNING_V2_CHUNK_SIZE;
  const int64_t state_bytes = (int64_t)Dv * Dk * (int64_t)sizeof(float);
  const int64_t smem_avail = (int64_t)LIGHTNING_MAX_SMEM - state_bytes;
  if (smem_avail <= 0) return 1;
  const int64_t per_token =
      ((int64_t)Dk + 3) * (int64_t)sizeof(float) + 2 * (int64_t)Dkp;
  int c = (int)(smem_avail / per_token);
  if (c > max_c) c = max_c;
  if (c < 1) c = 1;
  return c;
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> lightning_attn_int8_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& initial_state) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "fni8.lightning_attn_int8_fwd: inputs must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                  v.scalar_type() == at::kFloat,
              "fni8.lightning_attn_int8_fwd: v2 takes fp32 q/k/v (state/V "
              "are numerically load-bearing, per AGENTS.md); only the k·q dot "
              "products are quantized to int8 internally");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "fni8.lightning_attn_int8_fwd: q,k,v must be [B,H,T,D]");
  const int64_t B = q.size(0), H = q.size(1), T = q.size(2), Dk = q.size(3);
  const int64_t Dv = v.size(3);
  TORCH_CHECK(k.sizes() == q.sizes(), "fni8.lightning_attn_int8_fwd: k shape must match q");
  TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == T,
              "fni8.lightning_attn_int8_fwd: v shape mismatch");
  TORCH_CHECK(Dk <= LIGHTNING_V1_MAX_DIM && Dv <= LIGHTNING_V1_MAX_DIM,
              "fni8.lightning_attn_int8_fwd: v2 supports Dk,Dv <= 128");

  const at::cuda::CUDAGuard guard(q.device());
  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();

  const float* init_ptr = nullptr;
  at::Tensor init_c;
  if (initial_state.has_value() && initial_state->defined()) {
    init_c = initial_state->contiguous();
    TORCH_CHECK(init_c.scalar_type() == at::kFloat, "fni8.lightning_attn_int8_fwd: "
                "initial_state must be float32");
    TORCH_CHECK(init_c.dim() == 4 && init_c.size(0) == B && init_c.size(1) == H &&
                    init_c.size(2) == Dv && init_c.size(3) == Dk,
                "fni8.lightning_attn_int8_fwd: initial_state must be [B,H,Dv,Dk]");
    init_ptr = init_c.data_ptr<float>();
  }

  auto out = at::empty({B, H, T, Dv}, qc.options());
  auto final_state = at::empty({B, H, Dv, Dk}, qc.options());
  if (B * H == 0) return {out, final_state};

  const int Dkp = (int)((Dk + 3) & ~3LL);
  const int C = compute_chunk_size_v2((int)Dk, (int)Dv, Dkp);
  const size_t smem_bytes =
      (size_t)(Dv * Dk + C * Dk + 3 * C) * sizeof(float) +
      (size_t)(2 * C * Dkp);
  TORCH_CHECK(smem_bytes <= LIGHTNING_MAX_SMEM,
              "fni8.lightning_attn_int8_fwd: shared memory required exceeds the Volta cap");
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(lightning_attn_int8_fwd_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  lightning_attn_int8_fwd_kernel<<<(unsigned)(B * H), LIGHTNING_V2_THREADS,
                                    smem_bytes, stream>>>(
      qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(), init_ptr,
      out.data_ptr<float>(), final_state.data_ptr<float>(),
      (int)H, (int)T, (int)Dk, (int)Dv, C, Dkp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, final_state};
}

}  // namespace fni8
