// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: launcher + torch wrapper for the MoE grouped/batched
// int8 dp4a GEMM.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "gemm_grouped_dp4a.cuh"

namespace fni8 {

// x [total_M,K] int8 (rows CONTIGUOUS per expert, e.g. via a stable argsort by
// expert id), x_scale [total_M] fp32, w [E,N,K] int8 (stacked per-row-quantized
// expert weights), w_scale [E,N] fp32, group_sizes [E] int64 (token count per
// expert, must sum to total_M) -> out [total_M,N] fp16, one kernel launch for
// every active expert. Y[m,n] = (sum_k x[m,k]*w[e(m),n,k]) * x_scale[m] * w_scale[e(m),n].
at::Tensor gemm_grouped_w8a8(const at::Tensor& x, const at::Tensor& x_scale,
                             const at::Tensor& w, const at::Tensor& w_scale,
                             const at::Tensor& group_sizes) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x_scale.is_cuda() && w_scale.is_cuda(),
              "fni8: x/x_scale/w/w_scale must be CUDA tensors");
  TORCH_CHECK(x.device() == w.device() && x.device() == x_scale.device() &&
                  x.device() == w_scale.device(),
              "fni8: x/x_scale/w/w_scale must be on the same device");
  TORCH_CHECK(x.scalar_type() == at::kChar && w.scalar_type() == at::kChar,
              "fni8: x/w must be int8");
  TORCH_CHECK(x_scale.scalar_type() == at::kFloat && w_scale.scalar_type() == at::kFloat,
              "fni8: scales must be float32");
  TORCH_CHECK(x.dim() == 2, "fni8: expect x[total_M,K]");
  TORCH_CHECK(w.dim() == 3, "fni8: expect w[E,N,K]");
  TORCH_CHECK(group_sizes.dim() == 1 && group_sizes.scalar_type() == at::kLong,
              "fni8: group_sizes must be a 1-D int64 tensor [E]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && x_scale.is_contiguous() &&
                  w_scale.is_contiguous(),
              "fni8: inputs must be contiguous");

  const auto total_M = x.size(0), K = x.size(1);
  const auto E = w.size(0), N = w.size(1);
  TORCH_CHECK(w.size(2) == K, "fni8: x/w contraction mismatch (x K=", K, ", w K=", w.size(2), ")");
  TORCH_CHECK(K % 4 == 0, "fni8: contraction dim K must be %4==0 for dp4a int32 loads, got ", K);
  TORCH_CHECK(x_scale.numel() == total_M, "fni8: x_scale shape mismatch");
  TORCH_CHECK(w_scale.numel() == E * N, "fni8: w_scale shape mismatch, expect [E,N]");
  TORCH_CHECK(group_sizes.numel() == E, "fni8: group_sizes must have E=", E, " entries");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr<int8_t>()) % 4 == 0,
              "fni8: x/w must be 4-byte aligned for dp4a int32 loads");

  // ---- build the (expert, row_start, row_end) M-tile table on the host. Tiny
  // (E is at most a few hundred for real MoE configs), so a device->host sync
  // here is a first-correctness-pass tradeoff, not a hot-path cost; building
  // the table on-device to skip the sync is a follow-up (PR7+ territory). ----
  auto gs_cpu = group_sizes.to(at::kCPU).contiguous();
  const int64_t* gs = gs_cpu.data_ptr<int64_t>();

  std::vector<int32_t> tile_expert, tile_row_start, tile_row_end;
  int64_t offset = 0;
  for (int64_t e = 0; e < E; ++e) {
    const int64_t cnt = gs[e];
    TORCH_CHECK(cnt >= 0, "fni8: group_sizes must be non-negative, got ", cnt, " at expert ", e);
    const int64_t n_tiles = (cnt + GEMM_BM - 1) / GEMM_BM;
    for (int64_t t = 0; t < n_tiles; ++t) {
      tile_expert.push_back((int32_t)e);
      tile_row_start.push_back((int32_t)(offset + t * GEMM_BM));
      tile_row_end.push_back((int32_t)(offset + std::min<int64_t>(cnt, (t + 1) * GEMM_BM)));
    }
    offset += cnt;
  }
  TORCH_CHECK(offset == total_M,
              "fni8: group_sizes must sum to x.size(0) (sum=", offset, ", x.size(0)=", total_M, ")");

  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({total_M, N}, x.options().dtype(at::kHalf));
  const int64_t n_m_tiles = (int64_t)tile_expert.size();
  if (n_m_tiles == 0 || N == 0) return out;

  auto to_device_i32 = [&](const std::vector<int32_t>& v) {
    return at::from_blob((void*)v.data(), {(int64_t)v.size()}, at::kInt).clone().to(x.device());
  };
  auto d_tile_expert = to_device_i32(tile_expert);
  auto d_row_start = to_device_i32(tile_row_start);
  auto d_row_end = to_device_i32(tile_row_end);

  const dim3 grid((unsigned)n_m_tiles, (unsigned)((N + GEMM_BN - 1) / GEMM_BN));
  const dim3 block(GEMM_THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();
  gemm_grouped_w8a8_kernel<<<grid, block, 0, stream>>>(
      x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<int8_t>(),
      w_scale.data_ptr<float>(), d_tile_expert.data_ptr<int32_t>(),
      d_row_start.data_ptr<int32_t>(), d_row_end.data_ptr<int32_t>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (int)N, (int)K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace fni8
