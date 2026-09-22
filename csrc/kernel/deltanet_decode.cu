// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Launcher + torch wrapper for the decode-specialized (warp-per-v-row,
// register-state, no-smem, CUDA-graph-friendly) Gated-DeltaNet recurrence.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <tuple>

#include "deltanet_decode.cuh"
#include "deltanet_fused_decode.cuh"

namespace fni8 {

namespace {
constexpr int64_t DND_MAX_DIM = 128;  // ceil(128/32)=4 register slots per lane
}  // namespace

std::tuple<at::Tensor, at::Tensor> deltanet_recurrent_decode(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& alpha, const at::Tensor& beta,
    const c10::optional<at::Tensor>& initial_state) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && alpha.is_cuda() && beta.is_cuda(),
              "fni8.deltanet_recurrent_decode: inputs must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                  v.scalar_type() == at::kFloat && alpha.scalar_type() == at::kFloat &&
                  beta.scalar_type() == at::kFloat,
              "fni8.deltanet_recurrent_decode: fp32-only (state recurrence is "
              "numerically load-bearing, per AGENTS.md)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "fni8.deltanet_recurrent_decode: q,k,v must be [B,H,T,D]");
  const int64_t B = q.size(0), H = q.size(1), T = q.size(2), Dk = q.size(3);
  const int64_t Dv = v.size(3);
  TORCH_CHECK(k.sizes() == q.sizes(), "fni8.deltanet_recurrent_decode: k shape must match q");
  TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == T,
              "fni8.deltanet_recurrent_decode: v shape mismatch");
  TORCH_CHECK(alpha.sizes() == beta.sizes() && alpha.size(0) == B && alpha.size(1) == H &&
                  alpha.size(2) == T,
              "fni8.deltanet_recurrent_decode: alpha/beta must be [B,H,T]");
  TORCH_CHECK(Dk <= DND_MAX_DIM && Dv <= DND_MAX_DIM,
              "fni8.deltanet_recurrent_decode: supports Dk,Dv <= 128 (register state)");

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
    TORCH_CHECK(init_c.scalar_type() == at::kFloat,
                "fni8.deltanet_recurrent_decode: initial_state must be float32");
    TORCH_CHECK(init_c.dim() == 4 && init_c.size(0) == B && init_c.size(1) == H &&
                    init_c.size(2) == Dv && init_c.size(3) == Dk,
                "fni8.deltanet_recurrent_decode: initial_state must be [B,H,Dv,Dk]");
    init_ptr = init_c.data_ptr<float>();
  }

  auto out = at::empty({B, H, T, Dv}, qc.options());
  auto final_state = at::empty({B, H, Dv, Dk}, qc.options());
  const int64_t BH = B * H;
  if (BH == 0 || Dv == 0) return {out, final_state};

  // One warp per (bh, v-row); pack DND_WARPS_PER_BLOCK warps into a block. No
  // dynamic shared memory -> no cudaFuncSetAttribute -> CUDA-graph-friendly.
  const int64_t total_warps = BH * Dv;
  const unsigned blocks =
      (unsigned)((total_warps + DND_WARPS_PER_BLOCK - 1) / DND_WARPS_PER_BLOCK);
  auto stream = at::cuda::getCurrentCUDAStream();
  deltanet_recurrent_decode_kernel<<<blocks, DND_THREADS, 0, stream>>>(
      qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(), ac.data_ptr<float>(),
      bc.data_ptr<float>(), init_ptr, out.data_ptr<float>(), final_state.data_ptr<float>(),
      (int)BH, (int)T, (int)Dk, (int)Dv);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, final_state};
}

// ---------------------------------------------------------------------------
// Fused decode: L2-norm(q,k) + GQA expand + gate/beta + delta-rule recurrence +
// gated output RMSNorm, all in ONE graph-capturable launch. Dk == Dv == 128.
// See deltanet_fused_decode.cuh for the math and the qengine attribution.
// ---------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor> deltanet_fused_decode(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& dt,
    const at::Tensor& b_logit, const at::Tensor& a_log, const at::Tensor& dt_bias,
    const at::Tensor& gain, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& initial_state, double q_scale, double eps) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && dt.is_cuda() && b_logit.is_cuda() &&
                  a_log.is_cuda() && dt_bias.is_cuda() && gain.is_cuda(),
              "fni8.deltanet_fused_decode: inputs must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                  v.scalar_type() == at::kFloat && dt.scalar_type() == at::kFloat &&
                  b_logit.scalar_type() == at::kFloat && a_log.scalar_type() == at::kFloat &&
                  dt_bias.scalar_type() == at::kFloat && gain.scalar_type() == at::kFloat,
              "fni8.deltanet_fused_decode: fp32-only (state/recurrence/norm are "
              "numerically load-bearing, per AGENTS.md)");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "fni8.deltanet_fused_decode: q,k,v must be [B,H,T,D] with T==1");
  const int64_t B = q.size(0), nk = q.size(1), T = q.size(2), Dk = q.size(3);
  const int64_t nv = v.size(1), Dv = v.size(3);
  TORCH_CHECK(T == 1 && v.size(2) == 1, "fni8.deltanet_fused_decode: decode only (T==1)");
  TORCH_CHECK(Dk == DFD_DIM && Dv == DFD_DIM,
              "fni8.deltanet_fused_decode: Dk==Dv==128 specialization (other "
              "shapes fall back to the eager op sequence)");
  TORCH_CHECK(k.size(0) == B && k.size(1) == nk && k.size(3) == Dk,
              "fni8.deltanet_fused_decode: k shape must match q");
  TORCH_CHECK(v.size(0) == B, "fni8.deltanet_fused_decode: v batch mismatch");
  TORCH_CHECK(nv % nk == 0, "fni8.deltanet_fused_decode: nv must be a multiple of nk (GQA)");
  const int64_t rep = nv / nk;
  TORCH_CHECK(dt.dim() == 3 && dt.size(0) == B && dt.size(1) == nv && dt.size(2) == 1,
              "fni8.deltanet_fused_decode: dt must be [B,nv,1]");
  TORCH_CHECK(b_logit.sizes() == dt.sizes(),
              "fni8.deltanet_fused_decode: b_logit must be [B,nv,1] like dt");
  TORCH_CHECK(a_log.dim() == 1 && a_log.size(0) == nv && dt_bias.dim() == 1 &&
                  dt_bias.size(0) == nv,
              "fni8.deltanet_fused_decode: a_log/dt_bias must be [nv]");
  TORCH_CHECK(gain.dim() == 1 && gain.size(0) == Dv,
              "fni8.deltanet_fused_decode: gain must be [Dv]");

  const at::cuda::CUDAGuard guard(q.device());
  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
  auto dtc = dt.contiguous(), bc = b_logit.contiguous();
  auto ac = a_log.contiguous(), dbc = dt_bias.contiguous(), gc = gain.contiguous();

  const float* z_ptr = nullptr;
  at::Tensor zc;
  if (z.has_value() && z->defined()) {
    zc = z->contiguous();
    TORCH_CHECK(zc.scalar_type() == at::kFloat, "fni8.deltanet_fused_decode: z must be float32");
    TORCH_CHECK(zc.numel() == B * nv * Dv, "fni8.deltanet_fused_decode: z must be [B,nv,1,Dv]");
    z_ptr = zc.data_ptr<float>();
  }
  const float* init_ptr = nullptr;
  at::Tensor init_c;
  if (initial_state.has_value() && initial_state->defined()) {
    init_c = initial_state->contiguous();
    TORCH_CHECK(init_c.scalar_type() == at::kFloat,
                "fni8.deltanet_fused_decode: initial_state must be float32");
    TORCH_CHECK(init_c.dim() == 4 && init_c.size(0) == B && init_c.size(1) == nv &&
                    init_c.size(2) == Dv && init_c.size(3) == Dk,
                "fni8.deltanet_fused_decode: initial_state must be [B,nv,Dv,Dk]");
    init_ptr = init_c.data_ptr<float>();
  }

  auto out = at::empty({B, nv, 1, Dv}, vc.options());
  auto final_state = at::empty({B, nv, Dv, Dk}, qc.options());
  const int64_t BH = B * nv;
  if (BH == 0) return {out, final_state};

  auto stream = at::cuda::getCurrentCUDAStream();
  deltanet_fused_decode_kernel<DFD_DIM, DFD_DIM><<<(unsigned)BH, DFD_THREADS, 0, stream>>>(
      qc.data_ptr<float>(), kc.data_ptr<float>(), vc.data_ptr<float>(), dtc.data_ptr<float>(),
      bc.data_ptr<float>(), ac.data_ptr<float>(), dbc.data_ptr<float>(), gc.data_ptr<float>(),
      z_ptr, init_ptr, out.data_ptr<float>(), final_state.data_ptr<float>(), (int)nk, (int)nv,
      (int)rep, (float)q_scale, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, final_state};
}

}  // namespace fni8
