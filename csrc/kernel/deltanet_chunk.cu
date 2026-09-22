// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Launchers + torch wrappers for the DeltaNet kernels.
//
//   v1 — naive sequential fp32 Gated-DeltaNet recurrence (ground truth)
//   v2 — ungated chunked WY/UT parallel form (issue #40)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdint>
#include <tuple>

#include "deltanet_chunk.cuh"

namespace fni8 {

namespace {
constexpr int64_t DELTANET_V1_MAX_DIM = 128;      // v1's supported Dv/Dk range
constexpr int64_t DELTANET_V2_MAX_DIM = 128;      // v2 supports Dv,Dk <= 128 with dynamic C
constexpr int64_t DELTANET_V3_MAX_DIM = 128;      // v3 same as v2 + extra smem for alpha/beta/log_gamma
constexpr size_t  DELTANET_MAX_SMEM     = 98304;  // Volta opt-in shared-mem cap

int compute_chunk_size(int Dk, int Dv) {
  // state(Dv*Dk) + k_chunk(C*Dk) + r_buf(Dv*C) + gram_row(C)  must fit in 98304 B.
  constexpr int max_c = 64;
  const int64_t smem_avail = (int64_t)DELTANET_MAX_SMEM - (int64_t)Dv * Dk * (int64_t)sizeof(float);
  if (smem_avail <= 0) return 1;  // should not happen; launcher validates before calling
  const int64_t per_token = ((int64_t)Dk + Dv + 1) * (int64_t)sizeof(float);
  int c = (int)(smem_avail / per_token);
  if (c > max_c) c = max_c;
  if (c < 1) c = 1;
  return c;
}

int compute_chunk_size_v3(int Dk, int Dv) {
  // Like compute_chunk_size but with extra smem for beta_chunk(C) + log_gamma(C+1).
  constexpr int max_c = 64;
  const int64_t state_bytes = (int64_t)Dv * Dk * (int64_t)sizeof(float);
  const int64_t smem_avail = (int64_t)DELTANET_MAX_SMEM - state_bytes;
  if (smem_avail <= 0) return 1;
  const int64_t per_token = ((int64_t)Dk + Dv + 1 + 1 + 1) * (int64_t)sizeof(float);  // k+r+gram+beta+logγ
  int c = (int)(smem_avail / per_token);
  if (c > max_c) c = max_c;
  if (c < 1) c = 1;
  return c;
}

int compute_chunk_size_v3_padded(int Dk, int Dv, int nthreads) {
  // v3's shared-memory rows are padded for bank-conflict-free group access
  // (deltanet_v3_row_stride / _rbuf_stride), and r_buf's stride depends on C
  // itself, so search downwards from the cap instead of solving for C.
  for (int c = DELTANET_V3_CHUNK_SIZE; c > 1; --c) {
    if (deltanet_v3_smem_floats(Dk, Dv, c, nthreads) * (int64_t)sizeof(float) <=
        (int64_t)DELTANET_MAX_SMEM)
      return c;
  }
  return 1;
}

int compute_chunk_size_v4(int Dk, int Dv, int Dkp) {
  // fp32: state(Dv*Dk) + k_chunk(C*Dk) + r_buf(Dv*C) + gram_row(C) + ks(C) + qs(C)
  // int8: k_i8(C*Dkp) + q_i8(C*Dkp) bytes.
  constexpr int max_c = 64;
  const int64_t state_bytes = (int64_t)Dv * Dk * (int64_t)sizeof(float);
  const int64_t smem_avail = (int64_t)DELTANET_MAX_SMEM - state_bytes;
  if (smem_avail <= 0) return 1;
  const int64_t per_token =
      ((int64_t)Dk + Dv + 3) * (int64_t)sizeof(float) + 2 * (int64_t)Dkp;  // k+r+gram+ks+qs + 2*int8
  int c = (int)(smem_avail / per_token);
  if (c > max_c) c = max_c;
  if (c < 1) c = 1;
  return c;
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> deltanet_recurrent_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& alpha, const at::Tensor& beta,
    const c10::optional<at::Tensor>& initial_state) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && alpha.is_cuda() && beta.is_cuda(),
              "fni8.deltanet_recurrent_fwd: inputs must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                  v.scalar_type() == at::kFloat && alpha.scalar_type() == at::kFloat &&
                  beta.scalar_type() == at::kFloat,
              "fni8.deltanet_recurrent_fwd: v1 is fp32-only (state recurrence is "
              "numerically load-bearing, per AGENTS.md)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "fni8.deltanet_recurrent_fwd: q,k,v must be [B,H,T,D]");
  const int64_t B = q.size(0), H = q.size(1), T = q.size(2), Dk = q.size(3);
  const int64_t Dv = v.size(3);
  TORCH_CHECK(k.sizes() == q.sizes(), "fni8.deltanet_recurrent_fwd: k shape must match q");
  TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == T,
              "fni8.deltanet_recurrent_fwd: v shape mismatch");
  TORCH_CHECK(alpha.sizes() == beta.sizes() && alpha.size(0) == B && alpha.size(1) == H &&
                  alpha.size(2) == T,
              "fni8.deltanet_recurrent_fwd: alpha/beta must be [B,H,T]");
  TORCH_CHECK(Dk <= DELTANET_V1_MAX_DIM && Dv <= DELTANET_V1_MAX_DIM,
              "fni8.deltanet_recurrent_fwd: v1 supports Dk,Dv <= 128 (untiled shared-mem "
              "state); larger head dims need the v2 chunked/tiled kernel");

  const at::cuda::CUDAGuard guard(q.device());
  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();
  auto ac = alpha.contiguous();
  auto bc = beta.contiguous();

  const float* init_ptr = nullptr;
  at::Tensor init_c;
  if (initial_state.has_value() && initial_state->defined()) {
    init_c = initial_state->contiguous();
    TORCH_CHECK(init_c.scalar_type() == at::kFloat, "fni8.deltanet_recurrent_fwd: "
                "initial_state must be float32");
    TORCH_CHECK(init_c.dim() == 4 && init_c.size(0) == B && init_c.size(1) == H &&
                    init_c.size(2) == Dv && init_c.size(3) == Dk,
                "fni8.deltanet_recurrent_fwd: initial_state must be [B,H,Dv,Dk]");
    init_ptr = init_c.data_ptr<float>();
  }

  auto out = at::empty({B, H, T, Dv}, qc.options());
  auto final_state = at::empty({B, H, Dv, Dk}, qc.options());
  if (B * H == 0) return {out, final_state};

  const size_t smem_bytes =
      (size_t)(Dv * Dk + 3 * Dk + Dv) * sizeof(float);
  TORCH_CHECK(smem_bytes <= DELTANET_MAX_SMEM,
              "fni8.deltanet_recurrent_fwd: state (Dv*Dk) exceeds the Volta shared-mem cap");
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(deltanet_recurrent_fwd_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  deltanet_recurrent_fwd_kernel<<<(unsigned)(B * H), DELTANET_V1_THREADS, smem_bytes, stream>>>(
      qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(), ac.data_ptr<float>(),
      bc.data_ptr<float>(), init_ptr, out.data_ptr<float>(), final_state.data_ptr<float>(),
      (int)H, (int)T, (int)Dk, (int)Dv);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, final_state};
}

// ===========================================================================
// v2 (issue #40): ungated chunked WY/UT parallel DeltaNet (fp32)
// ===========================================================================

std::tuple<at::Tensor, at::Tensor> deltanet_chunk_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& initial_state) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "fni8.deltanet_chunk_fwd: inputs must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                  v.scalar_type() == at::kFloat,
              "fni8.deltanet_chunk_fwd: v2 is fp32-only (state recurrence is "
              "numerically load-bearing, per AGENTS.md)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "fni8.deltanet_chunk_fwd: q,k,v must be [B,H,T,D]");
  const int64_t B = q.size(0), H = q.size(1), T = q.size(2), Dk = q.size(3);
  const int64_t Dv = v.size(3);
  TORCH_CHECK(k.sizes() == q.sizes(), "fni8.deltanet_chunk_fwd: k shape must match q");
  TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == T,
              "fni8.deltanet_chunk_fwd: v shape mismatch");
  TORCH_CHECK(Dk <= DELTANET_V2_MAX_DIM && Dv <= DELTANET_V2_MAX_DIM,
              "fni8.deltanet_chunk_fwd: v2 supports Dk,Dv <= 128");

  const at::cuda::CUDAGuard guard(q.device());
  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();

  const float* init_ptr = nullptr;
  at::Tensor init_c;
  if (initial_state.has_value() && initial_state->defined()) {
    init_c = initial_state->contiguous();
    TORCH_CHECK(init_c.scalar_type() == at::kFloat, "fni8.deltanet_chunk_fwd: "
                "initial_state must be float32");
    TORCH_CHECK(init_c.dim() == 4 && init_c.size(0) == B && init_c.size(1) == H &&
                    init_c.size(2) == Dv && init_c.size(3) == Dk,
                "fni8.deltanet_chunk_fwd: initial_state must be [B,H,Dv,Dk]");
    init_ptr = init_c.data_ptr<float>();
  }

  auto out = at::empty({B, H, T, Dv}, qc.options());
  auto final_state = at::empty({B, H, Dv, Dk}, qc.options());
  if (B * H == 0) return {out, final_state};

  const int C = compute_chunk_size((int)Dk, (int)Dv);
  const size_t smem_bytes =
      (size_t)(Dv * Dk + C * Dk + Dv * C + C) * sizeof(float);
  TORCH_CHECK(smem_bytes <= DELTANET_MAX_SMEM,
              "fni8.deltanet_chunk_fwd: shared memory required exceeds the Volta cap");
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(deltanet_chunk_fwd_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  deltanet_chunk_fwd_kernel<<<(unsigned)(B * H), DELTANET_V2_THREADS, smem_bytes, stream>>>(
      qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(), init_ptr,
      out.data_ptr<float>(), final_state.data_ptr<float>(),
      (int)H, (int)T, (int)Dk, (int)Dv, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, final_state};
}

// ===========================================================================
// v3 (issue #56): gated chunked WY/UT parallel DeltaNet (log-space γ-cumprod)
// ===========================================================================

std::tuple<at::Tensor, at::Tensor> deltanet_gated_chunk_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& alpha, const at::Tensor& beta,
    const c10::optional<at::Tensor>& initial_state) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && alpha.is_cuda() && beta.is_cuda(),
              "fni8.deltanet_gated_chunk_fwd: inputs must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                  v.scalar_type() == at::kFloat && alpha.scalar_type() == at::kFloat &&
                  beta.scalar_type() == at::kFloat,
              "fni8.deltanet_gated_chunk_fwd: v3 is fp32-only (state recurrence is "
              "numerically load-bearing, per AGENTS.md)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "fni8.deltanet_gated_chunk_fwd: q,k,v must be [B,H,T,D]");
  const int64_t B = q.size(0), H = q.size(1), T = q.size(2), Dk = q.size(3);
  const int64_t Dv = v.size(3);
  TORCH_CHECK(k.sizes() == q.sizes(), "fni8.deltanet_gated_chunk_fwd: k shape must match q");
  TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == T,
              "fni8.deltanet_gated_chunk_fwd: v shape mismatch");
  TORCH_CHECK(alpha.sizes() == beta.sizes() && alpha.size(0) == B && alpha.size(1) == H &&
                  alpha.size(2) == T,
              "fni8.deltanet_gated_chunk_fwd: alpha/beta must be [B,H,T]");
  TORCH_CHECK(Dk <= DELTANET_V3_MAX_DIM && Dv <= DELTANET_V3_MAX_DIM,
              "fni8.deltanet_gated_chunk_fwd: v3 supports Dk,Dv <= 128");

  const at::cuda::CUDAGuard guard(q.device());
  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();
  auto ac = alpha.contiguous();
  auto bc = beta.contiguous();

  const float* init_ptr = nullptr;
  at::Tensor init_c;
  if (initial_state.has_value() && initial_state->defined()) {
    init_c = initial_state->contiguous();
    TORCH_CHECK(init_c.scalar_type() == at::kFloat, "fni8.deltanet_gated_chunk_fwd: "
                "initial_state must be float32");
    TORCH_CHECK(init_c.dim() == 4 && init_c.size(0) == B && init_c.size(1) == H &&
                    init_c.size(2) == Dv && init_c.size(3) == Dk,
                "fni8.deltanet_gated_chunk_fwd: initial_state must be [B,H,Dv,Dk]");
    init_ptr = init_c.data_ptr<float>();
  }

  auto out = at::empty({B, H, T, Dv}, qc.options());
  auto final_state = at::empty({B, H, Dv, Dk}, qc.options());
  if (B * H == 0) return {out, final_state};

  const int C = compute_chunk_size_v3_padded((int)Dk, (int)Dv, DELTANET_V3_THREADS);
  // smem: state(Dv*SS) + k_chunk(C*SS) + r_buf(Dv*RS) + gram_row(C) +
  //       beta_chunk(C) + gexp(C) + log_gamma(C+1), with the padded strides
  const size_t smem_bytes = (size_t)(deltanet_v3_smem_floats(
      (int)Dk, (int)Dv, C, DELTANET_V3_THREADS) * (int64_t)sizeof(float));
  TORCH_CHECK(smem_bytes <= DELTANET_MAX_SMEM,
              "fni8.deltanet_gated_chunk_fwd: shared memory required exceeds the Volta cap");
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(deltanet_gated_chunk_fwd_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  deltanet_gated_chunk_fwd_kernel<<<(unsigned)(B * H), DELTANET_V3_THREADS, smem_bytes, stream>>>(
      qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
      ac.data_ptr<float>(), bc.data_ptr<float>(), init_ptr,
      out.data_ptr<float>(), final_state.data_ptr<float>(),
      (int)H, (int)T, (int)Dk, (int)Dv, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, final_state};
}

// ===========================================================================
// v3.5 (issue #122): half2 (FP16x2) CUDA-core gated chunked DeltaNet
// ===========================================================================

std::tuple<at::Tensor, at::Tensor> deltanet_gated_chunk_h2_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& alpha, const at::Tensor& beta,
    const c10::optional<at::Tensor>& initial_state) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && alpha.is_cuda() && beta.is_cuda(),
              "fni8.deltanet_gated_chunk_h2_fwd: inputs must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                  v.scalar_type() == at::kFloat && alpha.scalar_type() == at::kFloat &&
                  beta.scalar_type() == at::kFloat,
              "fni8.deltanet_gated_chunk_h2_fwd: v3.5 takes fp32 inputs (half2 dot products "
              "are internal only; state recurrence is numerically load-bearing per AGENTS.md)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "fni8.deltanet_gated_chunk_h2_fwd: q,k,v must be [B,H,T,D]");
  const int64_t B = q.size(0), H = q.size(1), T = q.size(2), Dk = q.size(3);
  const int64_t Dv = v.size(3);
  TORCH_CHECK(k.sizes() == q.sizes(), "fni8.deltanet_gated_chunk_h2_fwd: k shape must match q");
  TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == T,
              "fni8.deltanet_gated_chunk_h2_fwd: v shape mismatch");
  TORCH_CHECK(alpha.sizes() == beta.sizes() && alpha.size(0) == B && alpha.size(1) == H &&
                  alpha.size(2) == T,
              "fni8.deltanet_gated_chunk_h2_fwd: alpha/beta must be [B,H,T]");
  TORCH_CHECK(Dk <= DELTANET_V3_MAX_DIM && Dv <= DELTANET_V3_MAX_DIM,
              "fni8.deltanet_gated_chunk_h2_fwd: v3.5 supports Dk,Dv <= 128");

  const at::cuda::CUDAGuard guard(q.device());
  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();
  auto ac = alpha.contiguous();
  auto bc = beta.contiguous();

  const float* init_ptr = nullptr;
  at::Tensor init_c;
  if (initial_state.has_value() && initial_state->defined()) {
    init_c = initial_state->contiguous();
    TORCH_CHECK(init_c.scalar_type() == at::kFloat, "fni8.deltanet_gated_chunk_h2_fwd: "
                "initial_state must be float32");
    TORCH_CHECK(init_c.dim() == 4 && init_c.size(0) == B && init_c.size(1) == H &&
                    init_c.size(2) == Dv && init_c.size(3) == Dk,
                "fni8.deltanet_gated_chunk_h2_fwd: initial_state must be [B,H,Dv,Dk]");
    init_ptr = init_c.data_ptr<float>();
  }

  auto out = at::empty({B, H, T, Dv}, qc.options());
  auto final_state = at::empty({B, H, Dv, Dk}, qc.options());
  if (B * H == 0) return {out, final_state};

  const int C = compute_chunk_size_v3((int)Dk, (int)Dv);
  const size_t smem_bytes =
      (size_t)(Dv * Dk + C * Dk + Dv * C + C + C + (C + 1)) * sizeof(float);
  TORCH_CHECK(smem_bytes <= DELTANET_MAX_SMEM,
              "fni8.deltanet_gated_chunk_h2_fwd: shared memory required exceeds the Volta cap");
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(deltanet_gated_chunk_h2_fwd_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  deltanet_gated_chunk_h2_fwd_kernel<<<(unsigned)(B * H), DELTANET_V35_THREADS, smem_bytes, stream>>>(
      qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(),
      ac.data_ptr<float>(), bc.data_ptr<float>(), init_ptr,
      out.data_ptr<float>(), final_state.data_ptr<float>(),
      (int)H, (int)T, (int)Dk, (int)Dv, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, final_state};
}

// ===========================================================================
// v4 (issue #83): int8 dp4a ungated chunked DeltaNet (dp4a K-Gram + Q·K)
// ===========================================================================

std::tuple<at::Tensor, at::Tensor> deltanet_chunk_int8_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& initial_state) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "fni8.deltanet_chunk_int8_fwd: inputs must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                  v.scalar_type() == at::kFloat,
              "fni8.deltanet_chunk_int8_fwd: v4 takes fp32 q/k/v (state/V are "
              "numerically load-bearing, per AGENTS.md); only the q·k / k·k dot "
              "products are quantized to int8 internally");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "fni8.deltanet_chunk_int8_fwd: q,k,v must be [B,H,T,D]");
  const int64_t B = q.size(0), H = q.size(1), T = q.size(2), Dk = q.size(3);
  const int64_t Dv = v.size(3);
  TORCH_CHECK(k.sizes() == q.sizes(), "fni8.deltanet_chunk_int8_fwd: k shape must match q");
  TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == T,
              "fni8.deltanet_chunk_int8_fwd: v shape mismatch");
  TORCH_CHECK(Dk <= DELTANET_V2_MAX_DIM && Dv <= DELTANET_V2_MAX_DIM,
              "fni8.deltanet_chunk_int8_fwd: v4 supports Dk,Dv <= 128");

  const at::cuda::CUDAGuard guard(q.device());
  auto qc = q.contiguous();
  auto kc = k.contiguous();
  auto vc = v.contiguous();

  const float* init_ptr = nullptr;
  at::Tensor init_c;
  if (initial_state.has_value() && initial_state->defined()) {
    init_c = initial_state->contiguous();
    TORCH_CHECK(init_c.scalar_type() == at::kFloat, "fni8.deltanet_chunk_int8_fwd: "
                "initial_state must be float32");
    TORCH_CHECK(init_c.dim() == 4 && init_c.size(0) == B && init_c.size(1) == H &&
                    init_c.size(2) == Dv && init_c.size(3) == Dk,
                "fni8.deltanet_chunk_int8_fwd: initial_state must be [B,H,Dv,Dk]");
    init_ptr = init_c.data_ptr<float>();
  }

  auto out = at::empty({B, H, T, Dv}, qc.options());
  auto final_state = at::empty({B, H, Dv, Dk}, qc.options());
  if (B * H == 0) return {out, final_state};

  const int Dkp = (int)((Dk + 3) & ~3LL);   // Dk padded to a multiple of 4 for dp4a
  const int C = compute_chunk_size_v4((int)Dk, (int)Dv, Dkp);
  // smem: state(Dv*Dk) + k_chunk(C*Dk) + r_buf(Dv*C) + gram_row(C) + ks(C) +
  //       qs(C) fp32, then k_i8(C*Dkp) + q_i8(C*Dkp) int8.
  const size_t smem_bytes =
      (size_t)(Dv * Dk + C * Dk + Dv * C + 3 * C) * sizeof(float) +
      (size_t)(2 * C * Dkp);
  TORCH_CHECK(smem_bytes <= DELTANET_MAX_SMEM,
              "fni8.deltanet_chunk_int8_fwd: shared memory required exceeds the Volta cap");
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(deltanet_chunk_int8_fwd_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  deltanet_chunk_int8_fwd_kernel<<<(unsigned)(B * H), DELTANET_V4_THREADS, smem_bytes, stream>>>(
      qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(), init_ptr,
      out.data_ptr<float>(), final_state.data_ptr<float>(),
      (int)H, (int)T, (int)Dk, (int)Dv, C, Dkp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, final_state};
}

}  // namespace fni8
