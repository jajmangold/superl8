// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: launcher + torch wrapper for the fused RMSNorm kernel.
// See csrc/include/rmsnorm.cuh and tests/test_rmsnorm.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>

#include "rmsnorm.cuh"

namespace fni8 {

namespace {

template <typename T>
void launch_rmsnorm(const at::Tensor& x, const c10::optional<at::Tensor>& res,
                    const at::Tensor& weight, at::Tensor& out, at::Tensor& xr,
                    bool unit_offset, float eps, int M, int D, cudaStream_t stream) {
  const dim3 grid((unsigned)M);
  const T* xp = reinterpret_cast<const T*>(x.data_ptr());
  const T* wp = reinterpret_cast<const T*>(weight.data_ptr());
  T* op = reinterpret_cast<T*>(out.data_ptr());
  const bool has_res = res.has_value();
  const T* rp = has_res ? reinterpret_cast<const T*>(res->data_ptr()) : nullptr;
  T* xrp = has_res ? reinterpret_cast<T*>(xr.data_ptr()) : nullptr;

  // Even D -> half2-vectorized kernel (issue #126): rows stay 4-byte aligned and D splits
  // into D/2 pairs with no tail. Odd D -> the scalar kernel (misaligned half2 rows otherwise).
  const bool vec2 = (D % 2 == 0);
#define LAUNCH(KERNEL, HAS_RES, UNIT)                                               \
  KERNEL<T, HAS_RES, UNIT><<<grid, RMSNORM_THREADS, 0, stream>>>(                   \
      xp, rp, wp, op, xrp, eps, M, D)
#define DISPATCH(HAS_RES, UNIT)                                                     \
  do {                                                                              \
    if (vec2) LAUNCH(rmsnorm_kernel_vec2, HAS_RES, UNIT);                           \
    else      LAUNCH(rmsnorm_kernel, HAS_RES, UNIT);                                \
  } while (0)
  if (has_res) {
    if (unit_offset) DISPATCH(true, true); else DISPATCH(true, false);
  } else {
    if (unit_offset) DISPATCH(false, true); else DISPATCH(false, false);
  }
#undef DISPATCH
#undef LAUNCH
}

}  // namespace

// x[M,D], weight[D] (fp16/bf16). Returns (normed[M,D], xr[M,D]); xr is x+residual
// when `residual` is given, else an undefined tensor (Python drops it).
std::tuple<at::Tensor, at::Tensor> rmsnorm(at::Tensor x, at::Tensor weight, double eps,
                                           c10::optional<at::Tensor> residual,
                                           bool unit_offset) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "fni8: rmsnorm expects CUDA tensors");
  auto xc = x.contiguous();
  const auto st = xc.scalar_type();
  TORCH_CHECK(st == at::kHalf || st == at::kBFloat16, "fni8: rmsnorm needs fp16/bf16");
  TORCH_CHECK(weight.scalar_type() == st, "fni8: weight dtype must match x");
  const int64_t M = xc.numel() / xc.size(-1), D = xc.size(-1);
  TORCH_CHECK(weight.numel() == D, "fni8: weight length must be D=", D);

  c10::optional<at::Tensor> res;
  if (residual.has_value()) {
    res = residual->contiguous();
    TORCH_CHECK(res->sizes() == xc.sizes() && res->scalar_type() == st,
                "fni8: residual must match x shape/dtype");
  }

  const at::cuda::CUDAGuard guard(xc.device());
  auto out = at::empty_like(xc);
  auto xr = res.has_value() ? at::empty_like(xc) : at::Tensor();
  if (M == 0 || D == 0) return {out, xr};

  auto stream = at::cuda::getCurrentCUDAStream();
  if (st == at::kHalf)
    launch_rmsnorm<__half>(xc, res, weight, out, xr, unit_offset, (float)eps, M, D, stream);
  else
    launch_rmsnorm<__nv_bfloat16>(xc, res, weight, out, xr, unit_offset, (float)eps, M, D, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out.view_as(x), res.has_value() ? xr.view_as(x) : xr};
}

}  // namespace fni8
