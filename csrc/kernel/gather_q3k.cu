// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: launcher + torch wrapper for the native Q3_K row-gather
// dequant (token embeddings). See csrc/include/gather_q3k.cuh and
// tests/test_gather_q3k.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "gather_q3k.cuh"

namespace fni8 {

// ids [...] int32/int64 (flattened here), w [N,(K/256)*110] uint8 native Q3_K
// super-blocks -- the SAME byte contract gemm_q3k/gemm_decode_q3k take -> out
// [n_ids, K] in `out_dtype`. K is implied by the row width, so the caller never
// passes it. The Python wrapper (fni8.gather_q3k) restores the ids' shape.
at::Tensor gather_q3k(const at::Tensor& ids, const at::Tensor& w, at::ScalarType out_dtype) {
  TORCH_CHECK(ids.is_cuda() && w.is_cuda(), "fni8: gather_q3k expects CUDA tensors");
  TORCH_CHECK(ids.device() == w.device(), "fni8: ids and w must be on the same device");
  TORCH_CHECK(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong,
              "fni8: gather_q3k ids must be int32 or int64, got ", ids.scalar_type());
  TORCH_CHECK(w.scalar_type() == at::kByte, "fni8: w must be uint8 (native Q3_K bytes)");
  TORCH_CHECK(w.dim() == 2, "fni8: expect w[N,(K/256)*", Q3K_TYPE_SIZE, "], got ", w.dim(),
              " dims");
  TORCH_CHECK(w.is_contiguous(), "fni8: w must be contiguous");
  TORCH_CHECK(out_dtype == at::kHalf || out_dtype == at::kBFloat16,
              "fni8: out_dtype must be float16 or bfloat16, got ", out_dtype);
  const auto N = w.size(0), row_bytes = w.size(1);
  TORCH_CHECK(row_bytes > 0 && row_bytes % Q3K_TYPE_SIZE == 0,
              "fni8: q3k weight row must be (K/256)*", Q3K_TYPE_SIZE, " bytes, got ",
              row_bytes);
  const int num_sb = (int)(row_bytes / Q3K_TYPE_SIZE);
  const int K = num_sb * GATHER_Q3K_QK;

  // int32 ids are widened once here (as rope does with positions) so the kernel
  // stays single-typed; the copy is n*8 B, nothing next to the table read.
  auto idx = ids.reshape({-1}).to(at::kLong).contiguous();
  const int64_t n = idx.numel();

  const at::cuda::CUDAGuard guard(w.device());
  auto out = at::empty({n, (int64_t)K}, w.options().dtype(out_dtype));
  if (n == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (out_dtype == at::kBFloat16) {
    gather_q3k_kernel<__nv_bfloat16><<<(unsigned)n, GATHER_Q3K_THREADS, 0, stream>>>(
        idx.data_ptr<int64_t>(), w.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), N, K, num_sb);
  } else {
    gather_q3k_kernel<__half><<<(unsigned)n, GATHER_Q3K_THREADS, 0, stream>>>(
        idx.data_ptr<int64_t>(), w.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(out.data_ptr()), N, K, num_sb);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace fni8
