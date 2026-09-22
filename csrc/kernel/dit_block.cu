// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: launcher + torch wrapper for the fused DiT block kernel.
// See csrc/include/dit_block.cuh and tests/test_dit_block.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "dit_block.cuh"

namespace fni8 {

namespace {

template <typename T>
void launch_dit_block(const at::Tensor& x, const at::Tensor& rms_weight,
                      const at::Tensor& scale, const at::Tensor& shift,
                      const c10::optional<at::Tensor>& gate,
                      at::Tensor& out, float eps,
                      const c10::optional<at::Tensor>& positions,
                      const c10::optional<at::Tensor>& cos,
                      const c10::optional<at::Tensor>& sin,
                      int64_t rotary_dim,
                      int M, int D, cudaStream_t stream) {
  const dim3 grid((unsigned)M);
  const int smem_bytes = D * (int)sizeof(float);
  // Volta default dynamic smem cap = 48 KB; we need D*4 bytes.
  // For D <= 12288 (48KB / 4) this stays within the default; beyond that we'd
  // need cudaFuncSetAttribute (opt-in to 96 KB max).  DiT dims are <= 4096.
  TORCH_CHECK(smem_bytes <= 48 * 1024,
              "fni8: dit_block D=", D, " would need ", smem_bytes,
              " bytes dynamic smem; max default is 49152");

  const T* xp = reinterpret_cast<const T*>(x.data_ptr());
  const T* wp = reinterpret_cast<const T*>(rms_weight.data_ptr());
  const T* sp = reinterpret_cast<const T*>(scale.data_ptr());
  const T* shp = reinterpret_cast<const T*>(shift.data_ptr());
  T* op = reinterpret_cast<T*>(out.data_ptr());

  const bool has_gate = gate.has_value();
  const T* gp = has_gate ? reinterpret_cast<const T*>(gate->data_ptr()) : nullptr;

  const bool has_rope = positions.has_value() && cos.has_value() && sin.has_value() && rotary_dim > 0;
  const int64_t* pp = has_rope ? positions->data_ptr<int64_t>() : nullptr;
  const float* cp = has_rope ? cos->data_ptr<float>() : nullptr;
  const float* rsp = has_rope ? sin->data_ptr<float>() : nullptr;

#define LAUNCH(HAS_GATE, HAS_ROPE)                                                   \
  do {                                                                               \
    if (D % 2 == 0) {                                                                \
      dit_block_kernel_vec2<T, HAS_GATE, HAS_ROPE><<<grid, DIT_BLOCK_THREADS, smem_bytes, stream>>>( \
          xp, wp, sp, shp, gp, op, eps, pp, cp, rsp, (int)rotary_dim, M, D);        \
    } else {                                                                         \
      dit_block_kernel<T, HAS_GATE, HAS_ROPE><<<grid, DIT_BLOCK_THREADS, smem_bytes, stream>>>( \
          xp, wp, sp, shp, gp, op, eps, pp, cp, rsp, (int)rotary_dim, M, D);        \
    }                                                                                \
  } while (0)

  if (has_gate) {
    if (has_rope) LAUNCH(true, true);
    else          LAUNCH(true, false);
  } else {
    if (has_rope) LAUNCH(false, true);
    else          LAUNCH(false, false);
  }

#undef LAUNCH
}

}  // namespace

// x[M,D], rms_weight[D], scale[M,D], shift[M,D] (fp16/bf16).
// gate[M,D] (optional).  positions[M] int64, cos/sin[max_pos,rotary_dim] fp32
// (optional, for RoPE).  Returns out[M,D].
at::Tensor dit_block(at::Tensor x, at::Tensor rms_weight, at::Tensor scale,
                     at::Tensor shift, double eps,
                     c10::optional<at::Tensor> gate,
                     c10::optional<at::Tensor> positions,
                     c10::optional<at::Tensor> cos,
                     c10::optional<at::Tensor> sin,
                     int64_t rotary_dim) {
  TORCH_CHECK(x.is_cuda(), "fni8: dit_block expects CUDA tensors");
  auto xc = x.contiguous();
  const auto st = xc.scalar_type();
  TORCH_CHECK(st == at::kHalf || st == at::kBFloat16, "fni8: dit_block needs fp16/bf16");
  TORCH_CHECK(rms_weight.scalar_type() == st, "fni8: rms_weight dtype must match x");
  TORCH_CHECK(scale.scalar_type() == st, "fni8: scale dtype must match x");
  TORCH_CHECK(shift.scalar_type() == st, "fni8: shift dtype must match x");

  const int64_t D = xc.size(-1);
  int64_t M = 1;
  for (int64_t dim = 0; dim + 1 < xc.dim(); ++dim) M *= xc.size(dim);
  TORCH_CHECK(rms_weight.numel() == D, "fni8: rms_weight length must be D=", D);
  TORCH_CHECK(scale.size(-1) == D, "fni8: scale last dim must be D=", D);
  TORCH_CHECK(shift.size(-1) == D, "fni8: shift last dim must be D=", D);
  TORCH_CHECK(scale.numel() == M * D, "fni8: scale must be [M, D]");
  TORCH_CHECK(shift.numel() == M * D, "fni8: shift must be [M, D]");

  auto rwc = rms_weight.contiguous();
  auto sc = scale.contiguous();
  auto shc = shift.contiguous();

  if (gate.has_value()) {
    auto g = gate->contiguous();
    TORCH_CHECK(g.scalar_type() == st, "fni8: gate dtype must match x");
    TORCH_CHECK(g.size(-1) == D, "fni8: gate last dim must be D=", D);
    TORCH_CHECK(g.numel() == M * D, "fni8: gate must be [M, D]");
    gate = g;
  }

  if (positions.has_value()) {
    auto pos = positions->contiguous();
    TORCH_CHECK(pos.scalar_type() == at::kLong, "fni8: positions must be int64");
    TORCH_CHECK(pos.numel() == M, "fni8: positions count must match M=", M);
    TORCH_CHECK(cos.has_value() && sin.has_value(), "fni8: RoPE requires cos and sin tables");
    TORCH_CHECK(rotary_dim % 2 == 0 && rotary_dim <= D,
                "fni8: rotary_dim must be even and <= D");
    auto cc = cos->contiguous();
    auto ss = sin->contiguous();
    TORCH_CHECK(cc.scalar_type() == at::kFloat && ss.scalar_type() == at::kFloat,
                "fni8: cos/sin tables must be float32");
    TORCH_CHECK(cc.size(-1) == rotary_dim && ss.size(-1) == rotary_dim,
                "fni8: cos/sin last dim must equal rotary_dim");
    positions = pos;
    cos = cc;
    sin = ss;
  }

  const at::cuda::CUDAGuard guard(xc.device());
  auto out = at::empty_like(xc);
  if (M == 0 || D == 0) return out;

  auto stream = at::cuda::getCurrentCUDAStream();
  if (st == at::kHalf)
    launch_dit_block<__half>(xc, rwc, sc, shc, gate, out, (float)eps,
                             positions, cos, sin, rotary_dim, M, D, stream);
  else
    launch_dit_block<__nv_bfloat16>(xc, rwc, sc, shc, gate, out, (float)eps,
                                    positions, cos, sin, rotary_dim, M, D, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.view_as(x);
}

}  // namespace fni8
