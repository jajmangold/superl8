// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: launcher + torch wrapper for the int8 dp4a GEMM.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>

#include "gemm_dp4a.cuh"
#include "gemm_q4k_dp4a.cuh"
#include "gemm_iq3s_dp4a.cuh"
#include "gemm_iq3xxs_dp4a.cuh"
#include "gemm_iq4xs_dp4a.cuh"
#include "gemm_iq2s_dp4a.cuh"
#include "gemm_iq2xs_dp4a.cuh"
#include "gemm_iq2xxs_dp4a.cuh"
#include "gemm_iq1s_dp4a.cuh"
#include "gemm_tq34s_dp4a.cuh"

namespace fni8 {

namespace {
void check_out_dtype(at::ScalarType out_dtype) {
  TORCH_CHECK(out_dtype == at::kHalf || out_dtype == at::kBFloat16,
              "fni8: out_dtype must be float16 or bfloat16, got ", out_dtype);
}
}  // namespace

// x [M,K] int8, x_scale [M] fp32, w [N,K] int8, w_scale [N] fp32 -> out [M,N]
// `out_dtype` (fp16 or bf16; fp16 default). Y[m,n] = (sum_k x[m,k]*w[n,k]) *
// x_scale[m] * w_scale[n]. bf16 avoids the fp16 overflow (max 65504) that
// black-images bf16-native models (Gemma, most diffusion DiTs) — see issue #11.
at::Tensor gemm_w8a8(const at::Tensor& x, const at::Tensor& x_scale,
                     const at::Tensor& w, const at::Tensor& w_scale,
                     at::ScalarType out_dtype) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x_scale.is_cuda() && w_scale.is_cuda(),
              "fni8: all inputs (incl. scales) must be CUDA tensors");
  TORCH_CHECK(x.device() == w.device() && x.device() == x_scale.device() &&
                  x.device() == w_scale.device(),
              "fni8: all inputs must be on the same device");
  TORCH_CHECK(x.scalar_type() == at::kChar && w.scalar_type() == at::kChar,
              "fni8: x/w must be int8");
  TORCH_CHECK(x_scale.scalar_type() == at::kFloat && w_scale.scalar_type() == at::kFloat,
              "fni8: scales must be float32");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "fni8: expect x[M,K] and w[N,K]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && x_scale.is_contiguous() &&
                  w_scale.is_contiguous(),
              "fni8: inputs must be contiguous");
  check_out_dtype(out_dtype);

  const auto M = x.size(0), K = x.size(1);
  const auto N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "fni8: x/w contraction mismatch (x K=", K, ", w K=", w.size(1), ")");
  TORCH_CHECK(K % 4 == 0, "fni8: contraction dim K must be %4==0 for dp4a int32 loads, got ", K);
  TORCH_CHECK(x_scale.numel() == M && w_scale.numel() == N, "fni8: scale shape mismatch");
  // dp4a reads int8 rows as int32; rows start at K-byte multiples and torch aligns
  // allocations >=16B, so 4B alignment holds — assert it rather than trust it.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr<int8_t>()) % 4 == 0,
              "fni8: x/w must be 4-byte aligned for dp4a int32 loads");

  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0 || N == 0) return out;

  const dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM);
  const dim3 block(GEMM_THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (out_dtype == at::kBFloat16) {
    gemm_w8a8_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<int8_t>(),
        w_scale.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        (int)M, (int)N, (int)K);
  } else {
    gemm_w8a8_kernel<__half><<<grid, block, 0, stream>>>(
        x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<int8_t>(),
        w_scale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr()),
        (int)M, (int)N, (int)K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// W4A8: x [M,K] int8, x_scale [M] fp32, w_packed [N,K/2] uint8 (2 nibbles/byte,
// signed int4), w_scale [N,K/G] fp32 (per output channel, per group), group_size
// G (%32==0). -> out [M,N] `out_dtype` (fp16 or bf16; fp16 default). Weights are
// the .fni8 `per_group_i4` (int4) blob.
at::Tensor gemm_w4a8(const at::Tensor& x, const at::Tensor& x_scale,
                     const at::Tensor& w, const at::Tensor& w_scale,
                     int64_t group_size, at::ScalarType out_dtype) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x_scale.is_cuda() && w_scale.is_cuda(),
              "fni8: all inputs (incl. scales) must be CUDA tensors");
  TORCH_CHECK(x.device() == w.device() && x.device() == x_scale.device() &&
                  x.device() == w_scale.device(),
              "fni8: all inputs must be on the same device");
  TORCH_CHECK(x.scalar_type() == at::kChar, "fni8: x must be int8");
  TORCH_CHECK(w.scalar_type() == at::kByte, "fni8: w must be uint8 (packed int4 nibbles)");
  TORCH_CHECK(x_scale.scalar_type() == at::kFloat && w_scale.scalar_type() == at::kFloat,
              "fni8: scales must be float32");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "fni8: expect x[M,K] and w[N,K/2]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && x_scale.is_contiguous() &&
                  w_scale.is_contiguous(),
              "fni8: inputs must be contiguous");
  check_out_dtype(out_dtype);

  const auto M = x.size(0), K = x.size(1);
  const auto N = w.size(0);
  const int G = (int)group_size;
  TORCH_CHECK(w.size(1) == K / 2, "fni8: packed w must be [N, K/2], got K=", K, " w=", w.size(1));
  TORCH_CHECK(K % 4 == 0, "fni8: contraction dim K must be %4==0 for dp4a, got ", K);
  TORCH_CHECK(G > 0 && G % 32 == 0, "fni8: group_size must be a positive multiple of 32, got ", G);
  TORCH_CHECK(K % G == 0, "fni8: K (", K, ") must be divisible by group_size (", G, ")");
  TORCH_CHECK(x_scale.numel() == M && w_scale.numel() == N * (K / G),
              "fni8: scale shape mismatch (x_scale [M], w_scale [N, K/G])");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr<uint8_t>()) % 4 == 0,
              "fni8: x/w must be 4-byte aligned for dp4a int32 loads");

  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0 || N == 0) return out;

  const dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM);
  const dim3 block(GEMM_THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (out_dtype == at::kBFloat16) {
    gemm_w4a8_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
        w_scale.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        (int)M, (int)N, (int)K, G);
  } else {
    gemm_w4a8_kernel<__half><<<grid, block, 0, stream>>>(
        x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
        w_scale.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr()),
        (int)M, (int)N, (int)K, G);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Q4_K fused: x [M,K] int8 (K%256==0), x_scale [M] fp32, w [N, (K/256)*144] uint8
// native GGUF Q4_K bytes -> out [M,N] `out_dtype`. The weight stays in its native
// Q4_K super-block layout; the kernel unpacks each 32-elem sub-block to int8 and
// dp4a's it, honoring the 6-bit per-sub-block scale/min exactly. K is inferred
// from x (Q4_K tensors always have in%256==0). See gemm_q4k_dp4a.cuh.
at::Tensor gemm_q4k(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x_scale.is_cuda(),
              "fni8: all inputs (incl. scale) must be CUDA tensors");
  TORCH_CHECK(x.device() == w.device() && x.device() == x_scale.device(),
              "fni8: all inputs must be on the same device");
  TORCH_CHECK(x.scalar_type() == at::kChar, "fni8: x must be int8");
  TORCH_CHECK(w.scalar_type() == at::kByte, "fni8: w must be uint8 (native Q4_K bytes)");
  TORCH_CHECK(x_scale.scalar_type() == at::kFloat, "fni8: x_scale must be float32");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "fni8: expect x[M,K] and w[N, (K/256)*144]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && x_scale.is_contiguous(),
              "fni8: inputs must be contiguous");
  check_out_dtype(out_dtype);

  const auto M = x.size(0), K = x.size(1);
  const auto N = w.size(0);
  TORCH_CHECK(K % Q4K_QK == 0,
              "fni8: Q4_K contraction dim K must be %256==0 (a Q4_K tensor always is), got ", K);
  const int num_sb = (int)(K / Q4K_QK);
  TORCH_CHECK(w.size(1) == (int64_t)num_sb * Q4K_TYPE_SIZE,
              "fni8: Q4_K weight must be [N, (K/256)*144], got [", N, ",", w.size(1),
              "] for K=", K, " (expected ", (int64_t)num_sb * Q4K_TYPE_SIZE, ")");
  TORCH_CHECK(x_scale.numel() == M, "fni8: x_scale must be [M]");

  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0 || N == 0) return out;

  const dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM);
  const dim3 block(GEMM_THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (out_dtype == at::kBFloat16) {
    gemm_q4k_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)M, (int)N, (int)K, num_sb);
  } else {
    gemm_q4k_kernel<__half><<<grid, block, 0, stream>>>(
        x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(out.data_ptr()), (int)M, (int)N, (int)K, num_sb);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Shared checks + launch for a k-quant fused GEMM (Q5_K/Q6_K mirror Q4_K; the
// only differences are the native type_size and the kernel). `type_size` is the
// bytes per 256-weight super-block (176 Q5_K, 210 Q6_K).
namespace {
template <typename Kernel>
at::Tensor launch_kquant(const at::Tensor& x, const at::Tensor& x_scale,
                         const at::Tensor& w, at::ScalarType out_dtype,
                         int type_size, const char* tag, Kernel kern) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x_scale.is_cuda(),
              "fni8: all inputs (incl. scale) must be CUDA tensors");
  TORCH_CHECK(x.device() == w.device() && x.device() == x_scale.device(),
              "fni8: all inputs must be on the same device");
  TORCH_CHECK(x.scalar_type() == at::kChar, "fni8: x must be int8");
  TORCH_CHECK(w.scalar_type() == at::kByte, "fni8: w must be uint8 (native k-quant bytes)");
  TORCH_CHECK(x_scale.scalar_type() == at::kFloat, "fni8: x_scale must be float32");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "fni8: expect x[M,K] and w[N, (K/256)*type_size]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && x_scale.is_contiguous(),
              "fni8: inputs must be contiguous");
  check_out_dtype(out_dtype);
  const auto M = x.size(0), K = x.size(1);
  const auto N = w.size(0);
  TORCH_CHECK(K % Q4K_QK == 0, "fni8: ", tag,
              " contraction dim K must be %256==0 (a k-quant tensor always is), got ", K);
  const int num_sb = (int)(K / Q4K_QK);
  TORCH_CHECK(w.size(1) == (int64_t)num_sb * type_size, "fni8: ", tag,
              " weight must be [N, (K/256)*", type_size, "], got [", N, ",", w.size(1),
              "] for K=", K);
  TORCH_CHECK(x_scale.numel() == M, "fni8: x_scale must be [M]");
  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0 || N == 0) return out;
  const dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM);
  const dim3 block(GEMM_THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();
  kern(grid, block, stream, x, x_scale, w, out, (int)M, (int)N, (int)K, num_sb, out_dtype);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}  // namespace

// Q5_K fused: w [N, (K/256)*176] uint8 native GGUF Q5_K. See gemm_q4k_dp4a.cuh.
at::Tensor gemm_q5k(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype) {
  return launch_kquant(
      x, x_scale, w, out_dtype, Q5K_TYPE_SIZE, "Q5_K",
      [](dim3 grid, dim3 block, cudaStream_t stream, const at::Tensor& x,
         const at::Tensor& x_scale, const at::Tensor& w, at::Tensor& out, int M, int N,
         int K, int num_sb, at::ScalarType od) {
        if (od == at::kBFloat16)
          gemm_q5k_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, num_sb);
        else
          gemm_q5k_kernel<__half><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__half*>(out.data_ptr()), M, N, K, num_sb);
      });
}

// Q6_K fused: w [N, (K/256)*210] uint8 native GGUF Q6_K. See gemm_q4k_dp4a.cuh.
at::Tensor gemm_q6k(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype) {
  return launch_kquant(
      x, x_scale, w, out_dtype, Q6K_TYPE_SIZE, "Q6_K",
      [](dim3 grid, dim3 block, cudaStream_t stream, const at::Tensor& x,
         const at::Tensor& x_scale, const at::Tensor& w, at::Tensor& out, int M, int N,
         int K, int num_sb, at::ScalarType od) {
        if (od == at::kBFloat16)
          gemm_q6k_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, num_sb);
        else
          gemm_q6k_kernel<__half><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__half*>(out.data_ptr()), M, N, K, num_sb);
      });
}

// Q3_K fused: w [N, (K/256)*110] uint8 native GGUF Q3_K (the 27B-Q3_K_S type).
at::Tensor gemm_q3k(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype) {
  return launch_kquant(
      x, x_scale, w, out_dtype, Q3K_TYPE_SIZE, "Q3_K",
      [](dim3 grid, dim3 block, cudaStream_t stream, const at::Tensor& x,
         const at::Tensor& x_scale, const at::Tensor& w, at::Tensor& out, int M, int N,
         int K, int num_sb, at::ScalarType od) {
        if (od == at::kBFloat16)
          gemm_q3k_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, num_sb);
        else
          gemm_q3k_kernel<__half><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__half*>(out.data_ptr()), M, N, K, num_sb);
      });
}

at::Tensor gemm_iq3s(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype) {
  return launch_kquant(
      x, x_scale, w, out_dtype, IQ3S_TYPE_SIZE, "IQ3_S",
      [](dim3 grid, dim3 block, cudaStream_t stream, const at::Tensor& x,
         const at::Tensor& x_scale, const at::Tensor& w, at::Tensor& out, int M, int N,
         int K, int num_sb, at::ScalarType od) {
        if (od == at::kBFloat16)
          gemm_iq3s_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, num_sb);
        else
          gemm_iq3s_kernel<__half><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__half*>(out.data_ptr()), M, N, K, num_sb);
      });
}

at::Tensor gemm_iq3xxs(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype) {
  return launch_kquant(
      x, x_scale, w, out_dtype, IQ3XXS_TYPE_SIZE, "IQ3_XXS",
      [](dim3 grid, dim3 block, cudaStream_t stream, const at::Tensor& x,
         const at::Tensor& x_scale, const at::Tensor& w, at::Tensor& out, int M, int N,
         int K, int num_sb, at::ScalarType od) {
        if (od == at::kBFloat16)
          gemm_iq3xxs_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, num_sb);
        else
          gemm_iq3xxs_kernel<__half><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__half*>(out.data_ptr()), M, N, K, num_sb);
      });
}

at::Tensor gemm_iq4xs(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype) {
  return launch_kquant(
      x, x_scale, w, out_dtype, IQ4XS_TYPE_SIZE, "IQ4_XS",
      [](dim3 grid, dim3 block, cudaStream_t stream, const at::Tensor& x,
         const at::Tensor& x_scale, const at::Tensor& w, at::Tensor& out, int M, int N,
         int K, int num_sb, at::ScalarType od) {
        if (od == at::kBFloat16)
          gemm_iq4xs_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, num_sb);
        else
          gemm_iq4xs_kernel<__half><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__half*>(out.data_ptr()), M, N, K, num_sb);
      });
}

at::Tensor gemm_iq2s(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype) {
  return launch_kquant(
      x, x_scale, w, out_dtype, IQ2S_TYPE_SIZE, "IQ2_S",
      [](dim3 grid, dim3 block, cudaStream_t stream, const at::Tensor& x,
         const at::Tensor& x_scale, const at::Tensor& w, at::Tensor& out, int M, int N,
         int K, int num_sb, at::ScalarType od) {
        if (od == at::kBFloat16)
          gemm_iq2s_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, num_sb);
        else
          gemm_iq2s_kernel<__half><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__half*>(out.data_ptr()), M, N, K, num_sb);
      });
}

at::Tensor gemm_iq2xs(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype) {
  return launch_kquant(
      x, x_scale, w, out_dtype, IQ2XS_TYPE_SIZE, "IQ2_XS",
      [](dim3 grid, dim3 block, cudaStream_t stream, const at::Tensor& x,
         const at::Tensor& x_scale, const at::Tensor& w, at::Tensor& out, int M, int N,
         int K, int num_sb, at::ScalarType od) {
        if (od == at::kBFloat16)
          gemm_iq2xs_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, num_sb);
        else
          gemm_iq2xs_kernel<__half><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__half*>(out.data_ptr()), M, N, K, num_sb);
      });
}

at::Tensor gemm_iq2xxs(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype) {
  return launch_kquant(
      x, x_scale, w, out_dtype, IQ2XXS_TYPE_SIZE, "IQ2_XXS",
      [](dim3 grid, dim3 block, cudaStream_t stream, const at::Tensor& x,
         const at::Tensor& x_scale, const at::Tensor& w, at::Tensor& out, int M, int N,
         int K, int num_sb, at::ScalarType od) {
        if (od == at::kBFloat16)
          gemm_iq2xxs_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, num_sb);
        else
          gemm_iq2xxs_kernel<__half><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__half*>(out.data_ptr()), M, N, K, num_sb);
      });
}

at::Tensor gemm_iq1s(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype) {
  return launch_kquant(
      x, x_scale, w, out_dtype, IQ1S_TYPE_SIZE, "IQ1_S",
      [](dim3 grid, dim3 block, cudaStream_t stream, const at::Tensor& x,
         const at::Tensor& x_scale, const at::Tensor& w, at::Tensor& out, int M, int N,
         int K, int num_sb, at::ScalarType od) {
        if (od == at::kBFloat16)
          gemm_iq1s_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, num_sb);
        else
          gemm_iq1s_kernel<__half><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__half*>(out.data_ptr()), M, N, K, num_sb);
      });
}

// Q2_K fused: w [N, (K/256)*84] uint8 native GGUF Q2_K (aggressive UD-Q2).
at::Tensor gemm_q2k(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype) {
  return launch_kquant(
      x, x_scale, w, out_dtype, Q2K_TYPE_SIZE, "Q2_K",
      [](dim3 grid, dim3 block, cudaStream_t stream, const at::Tensor& x,
         const at::Tensor& x_scale, const at::Tensor& w, at::Tensor& out, int M, int N,
         int K, int num_sb, at::ScalarType od) {
        if (od == at::kBFloat16)
          gemm_q2k_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K, num_sb);
        else
          gemm_q2k_kernel<__half><<<grid, block, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
              reinterpret_cast<__half*>(out.data_ptr()), M, N, K, num_sb);
      });
}

// TQ3_4S fused tile: x [M,K] fp16/bf16 (K%32==0), w [N, (K/32)*16] uint8 native
// GGUF TQ3_4S bytes -> out [M,N] `out_dtype`. The activation is rotated with the
// forward RHT per 32-block and per-32 q8_1-quantized ONCE by
// `tq34s_rht_prepass_kernel` into an int8 workspace (fni8#281 perf pass); the
// tile kernel then dp4a's against the CORRECTED int8 centroid levels and flushes
// each per-8 E3M5 scale into the fp32 accumulator (see gemm_tq34s_dp4a.cuh).
// sm_70-only: __dp4a + __shfl_xor_sync, no mma/wmma/cp.async/ldmatrix.
template <bool CommonScale, bool NarrowAccumulators = false>
static at::Tensor gemm_tq34s_impl(const at::Tensor& x, const at::Tensor& w,
                                  int64_t in_features, at::ScalarType out_dtype) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(),
              "fni8: all inputs must be CUDA tensors");
  TORCH_CHECK(x.device() == w.device(),
              "fni8: all inputs must be on the same device");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "fni8: gemm_tq34s x must be fp16 or bf16, got ", x.scalar_type());
  TORCH_CHECK(w.scalar_type() == at::kByte,
              "fni8: w must be uint8 (native TQ3_4S bytes)");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "fni8: expect x[M,K] and w[N,(K/32)*16]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
              "fni8: inputs must be contiguous");
  check_out_dtype(out_dtype);

  const auto M = x.size(0), K = x.size(1);
  const auto N = w.size(0);
  TORCH_CHECK(K == in_features, "fni8: in_features (", in_features, ") != x K (", K, ")");
  TORCH_CHECK(K % TQ3_QK == 0, "fni8: TQ3_4S contraction K must be %32==0, got ", K);
  const int nblk = (int)(K / TQ3_QK);
  TORCH_CHECK(w.size(1) == (int64_t)nblk * TQ3_TYPE_SIZE,
              "fni8: TQ3_4S weight must be [N,(K/32)*16], got [", N, ",", w.size(1),
              "] for K=", K);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr<uint8_t>()) % 16 == 0,
              "fni8: w must be 16-byte aligned for the vectorized uint4 block loads");

  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0 || N == 0) return out;

  // Pre-rotate + quantize the activation ONCE (workspace: [M,K] int8 + [M,nblk] fp32).
  auto xq = at::empty({M, K}, x.options().dtype(at::kChar));
  auto xs = at::empty({M, nblk}, x.options().dtype(at::kFloat));

  const dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM);
  constexpr int tile_threads = NarrowAccumulators ? 256 : GEMM_THREADS;
  constexpr int tile_tm = 4;
  constexpr int tile_tn = NarrowAccumulators ? 4 : GEMM_TN;
  const dim3 block(tile_threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool bf16 = (x.scalar_type() == at::kBFloat16);
  const auto* xp = x.data_ptr();
  const auto* wp = w.data_ptr<uint8_t>();
  {
    constexpr int threads = 256;
    const int total = (int)(M * nblk);
    const int blocks = std::max(
        1, std::min(512, (total + threads / 32 - 1) / (threads / 32)));
    if (bf16)
      tq34s_rht_prepass_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(xp), xq.data_ptr<int8_t>(),
          xs.data_ptr<float>(), (int)M, (int)K, nblk);
    else
      tq34s_rht_prepass_kernel<__half><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __half*>(xp), xq.data_ptr<int8_t>(),
          xs.data_ptr<float>(), (int)M, (int)K, nblk);
  }
  const auto* xqp = xq.data_ptr<int8_t>();
  const auto* xsp = xs.data_ptr<float>();
  if (out_dtype == at::kBFloat16) {
    gemm_tq34s_kernel<__nv_bfloat16, CommonScale, tile_threads, tile_tm, tile_tn>
        <<<grid, block, 0, stream>>>(
        xqp, xsp, wp, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        (int)M, (int)N, (int)K, nblk);
  } else {
    gemm_tq34s_kernel<__half, CommonScale, tile_threads, tile_tm, tile_tn>
        <<<grid, block, 0, stream>>>(
        xqp, xsp, wp, reinterpret_cast<__half*>(out.data_ptr()),
        (int)M, (int)N, (int)K, nblk);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor gemm_tq34s(const at::Tensor& x, const at::Tensor& w, int64_t in_features,
                      at::ScalarType out_dtype) {
  return gemm_tq34s_impl<false>(x, w, in_features, out_dtype);
}

at::Tensor gemm_tq34s_common_scale(const at::Tensor& x, const at::Tensor& w,
                                   int64_t in_features, at::ScalarType out_dtype) {
  return gemm_tq34s_impl<true>(x, w, in_features, out_dtype);
}

at::Tensor gemm_tq34s_common_scale_tm4tn4(const at::Tensor& x, const at::Tensor& w,
                                          int64_t in_features,
                                          at::ScalarType out_dtype) {
  return gemm_tq34s_impl<true, true>(x, w, in_features, out_dtype);
}

}  // namespace fni8
