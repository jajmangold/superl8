// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Launcher + torch wrapper for the fused gated output RMSNorm decode kernel
// (warp-per-(batch,head)-row, register state, no smem -> CUDA-graph-capturable).
// See csrc/include/gated_rmsnorm_decode.cuh and tests/test_deltanet_fused_ops.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

#include "gated_rmsnorm_decode.cuh"

namespace fni8 {

namespace {
constexpr int64_t GRN_MAX_VD = 128;  // ceil(128/32)=4 register slots per lane
}  // namespace

// o:    [..., vd] fp32 (last dim is head_v_dim; leading dims are B, nv, ...)
// gain: [vd]      fp32
// z:    same shape as o, fp32, or None (no z gate)
// Returns out with o's shape.
at::Tensor gated_rmsnorm_decode(const at::Tensor& o, const at::Tensor& gain,
                                const c10::optional<at::Tensor>& z, double eps) {
  TORCH_CHECK(o.is_cuda() && gain.is_cuda(),
              "fni8.gated_rmsnorm_decode: inputs must be CUDA tensors");
  TORCH_CHECK(o.scalar_type() == at::kFloat && gain.scalar_type() == at::kFloat,
              "fni8.gated_rmsnorm_decode: fp32-only (the gated norm is numerically "
              "load-bearing, per AGENTS.md)");
  TORCH_CHECK(o.dim() >= 1, "fni8.gated_rmsnorm_decode: o must have a trailing vd dim");
  const int64_t vd = o.size(-1);
  TORCH_CHECK(gain.dim() == 1 && gain.size(0) == vd,
              "fni8.gated_rmsnorm_decode: gain must be [vd] matching o's last dim");
  TORCH_CHECK(vd <= GRN_MAX_VD,
              "fni8.gated_rmsnorm_decode: supports vd <= 128 (register state), got ", vd);
  const int64_t N = o.numel() / (vd == 0 ? 1 : vd);

  const at::cuda::CUDAGuard guard(o.device());
  auto oc = o.contiguous();
  auto gc = gain.contiguous();
  auto out = at::empty_like(oc);

  const float* z_ptr = nullptr;
  at::Tensor zc;
  if (z.has_value() && z->defined()) {
    zc = z->contiguous();
    TORCH_CHECK(zc.scalar_type() == at::kFloat,
                "fni8.gated_rmsnorm_decode: z gate must be float32");
    TORCH_CHECK(zc.sizes() == oc.sizes(),
                "fni8.gated_rmsnorm_decode: z must match o's shape");
    z_ptr = zc.data_ptr<float>();
  }

  if (N == 0 || vd == 0) return out;

  const unsigned blocks =
      (unsigned)((N + GRN_WARPS_PER_BLOCK - 1) / GRN_WARPS_PER_BLOCK);
  auto stream = at::cuda::getCurrentCUDAStream();
  gated_rmsnorm_decode_kernel<<<blocks, GRN_THREADS, 0, stream>>>(
      oc.data_ptr<float>(), gc.data_ptr<float>(), z_ptr, out.data_ptr<float>(),
      (int)N, (int)vd, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace fni8
