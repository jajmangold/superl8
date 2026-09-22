// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Thin translation unit: split-K sizing + launcher + torch wrapper for the
// decode-specialized int8 dp4a GEMM (issue #27).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <tuple>

#include "gemm_decode_dp4a.cuh"
#include "gemm_decode_iq3s.cuh"
#include "gemm_decode_iq4xs.cuh"
#include "gemm_decode_iq3xxs.cuh"
#include "gemm_decode_iq2s.cuh"
#include "gemm_decode_iq2xs.cuh"
#include "gemm_decode_iq2xxs.cuh"
#include "gemm_decode_iq1s.cuh"
#include "gemm_tq34s_dp4a.cuh"
#include "gemm_decode_tq34s_dp4a.cuh"

namespace fni8 {

namespace {

void check_out_dtype(at::ScalarType out_dtype) {
  TORCH_CHECK(out_dtype == at::kHalf || out_dtype == at::kBFloat16,
              "fni8: out_dtype must be float16 or bfloat16, got ", out_dtype);
}

// ~4x the real V100's 80 SMs (AGENTS.md) worth of (32-thread) blocks is
// plenty to hide launch/scheduling overhead without over-splitting K.
constexpr int DEC_TARGET_BLOCKS = 320;
constexpr int DEC_MIN_K4_PER_SPLIT = 32;   // don't split so fine atomics dominate
constexpr int DEC_MAX_SPLIT_K = 64;

// Split K just enough to keep all SMs busy when N alone is too small; N by
// itself already gives N blocks (vs the tile GEMM's ceil(N/64)), so most
// decode shapes (N in the hundreds-to-thousands) need split_k == 1.
int compute_split_k(int64_t N, int64_t K4) {
  int split_k = (N < DEC_TARGET_BLOCKS) ? (int)((DEC_TARGET_BLOCKS + N - 1) / N) : 1;
  split_k = std::min<int64_t>(split_k, std::max<int64_t>(1, K4 / DEC_MIN_K4_PER_SPLIT));
  split_k = std::min(split_k, DEC_MAX_SPLIT_K);
  return std::max(1, split_k);
}

// Runtime M -> smallest compile-time MAX_M template that covers it (1, 2, 4,
// 8, 16). The accumulator array (and therefore register footprint) scales
// with MAX_M, so small decode batches (M=1 or M=8, the common cases) get a
// much lighter kernel instantiation than always paying for DEC_MAX_M=16.
#define DEC_DISPATCH_MAX_M(M, CALL)   \
  do {                                \
    if ((M) <= 1) {                   \
      constexpr int MAX_M = 1;        \
      CALL;                           \
    } else if ((M) <= 2) {            \
      constexpr int MAX_M = 2;        \
      CALL;                           \
    } else if ((M) <= 4) {            \
      constexpr int MAX_M = 4;        \
      CALL;                           \
    } else if ((M) <= 8) {            \
      constexpr int MAX_M = 8;        \
      CALL;                           \
    } else {                          \
      constexpr int MAX_M = DEC_MAX_M; \
      CALL;                           \
    }                                 \
  } while (0)

// Fused fp16-in launch helper — a single parenthesizable call so the DEC_DISPATCH_MAX_M
// macro doesn't see the kernel-launch commas as macro-arg separators (the <<<...>>> and
// cudaFuncSetAttribute commas are not protected by braces, only by the wrapping parens at
// the call site). Sets the >48KB dynamic-smem attribute per instantiation, then launches.
template <typename OutT, int MAX_M>
void launch_decode_fp16in(dim3 grid, size_t smem_bytes, cudaStream_t stream,
                          const __half* xp, const int8_t* wp, const float* wsp,
                          OutT* op, int M, int N, int K) {
  auto kern = gemm_decode_single_fp16in_kernel<OutT, MAX_M>;
  if (smem_bytes > 48 * 1024)
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
  kern<<<grid, DEC_THREADS, smem_bytes, stream>>>(xp, wp, wsp, op, M, N, K);
}

}  // namespace

// x [M,K] int8 (M<=DEC_MAX_M -- decode shapes; use gemm_w8a8 for prefill),
// x_scale [M] fp32, w [N,K] int8, w_scale [N] fp32 -> out [M,N] `out_dtype`
// (fp16 or bf16; fp16 default). Same math/contract as gemm_w8a8, split-K grid.
at::Tensor gemm_decode_w8a8(const at::Tensor& x, const at::Tensor& x_scale,
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
  TORCH_CHECK(M <= DEC_MAX_M,
              "fni8: gemm_decode_w8a8 supports decode shapes M<=", DEC_MAX_M,
              ", got M=", M, " -- use gemm_w8a8 for prefill-size M");
  TORCH_CHECK(w.size(1) == K, "fni8: x/w contraction mismatch (x K=", K, ", w K=", w.size(1), ")");
  TORCH_CHECK(K % 4 == 0, "fni8: contraction dim K must be %4==0 for dp4a int32 loads, got ", K);
  TORCH_CHECK(x_scale.numel() == M && w_scale.numel() == N, "fni8: scale shape mismatch");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr<int8_t>()) % 4 == 0,
              "fni8: x/w must be 4-byte aligned for dp4a int32 loads");

  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0 || N == 0) return out;

  const int split_k = compute_split_k(N, K / 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  const unsigned n_blocks = (unsigned)((N + DEC_WARPS_PER_BLOCK - 1) / DEC_WARPS_PER_BLOCK);

  // Common case: N alone already gives enough warps to fill the GPU -- do the
  // whole K reduction in one warp per column, ONE kernel launch, no atomics.
  if (split_k == 1) {
    const dim3 grid(n_blocks);
    if (out_dtype == at::kBFloat16) {
      auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
      DEC_DISPATCH_MAX_M((int)M,
          (gemm_decode_single_kernel<__nv_bfloat16, MAX_M><<<grid, DEC_THREADS, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<int8_t>(),
              w_scale.data_ptr<float>(), op, (int)M, (int)N, (int)K)));
    } else {
      auto* op = reinterpret_cast<__half*>(out.data_ptr());
      DEC_DISPATCH_MAX_M((int)M,
          (gemm_decode_single_kernel<__half, MAX_M><<<grid, DEC_THREADS, 0, stream>>>(
              x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<int8_t>(),
              w_scale.data_ptr<float>(), op, (int)M, (int)N, (int)K)));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }

  // N itself is too small to fill the GPU -- split K across blocks, reduce
  // partials through an int32 workspace + a tiny epilogue kernel.
  auto acc32 = at::zeros({M, N}, x.options().dtype(at::kInt));
  const dim3 grid1(n_blocks, (unsigned)split_k);
  DEC_DISPATCH_MAX_M((int)M,
      (gemm_decode_accum_kernel<MAX_M><<<grid1, DEC_THREADS, 0, stream>>>(
          x.data_ptr<int8_t>(), w.data_ptr<int8_t>(), acc32.data_ptr<int32_t>(),
          (int)M, (int)N, (int)K, split_k)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int total = (int)(M * N);
  const int threads2 = 256;
  const int blocks2 = (total + threads2 - 1) / threads2;
  if (out_dtype == at::kBFloat16) {
    gemm_decode_epilogue_kernel<__nv_bfloat16><<<blocks2, threads2, 0, stream>>>(
        acc32.data_ptr<int32_t>(), x_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)M, (int)N);
  } else {
    gemm_decode_epilogue_kernel<__half><<<blocks2, threads2, 0, stream>>>(
        acc32.data_ptr<int32_t>(), x_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr()), (int)M, (int)N);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// W4A8 decode: x [M,K] int8 (M<=DEC_MAX_M), x_scale [M] fp32, w_packed [N,K/2]
// uint8 (2 signed nibbles/byte), w_scale [N,K/G] fp32 (per-channel, per-group)
// -> out [M,N] `out_dtype`. Warp-per-column, split_k==1 (large-N decode);
// same int math as the tile gemm_w4a8. Callers use gemm_w4a8 for prefill/tiny-N.
at::Tensor gemm_decode_w4a8(const at::Tensor& x, const at::Tensor& x_scale,
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
  TORCH_CHECK(M <= DEC_MAX_M,
              "fni8: gemm_decode_w4a8 supports decode shapes M<=", DEC_MAX_M,
              ", got M=", M, " -- use gemm_w4a8 for prefill-size M");
  TORCH_CHECK(w.size(1) == K / 2, "fni8: w must be [N,K/2] packed int4 (K=", K, ")");
  TORCH_CHECK(G > 0 && G % 32 == 0, "fni8: group_size must be a positive multiple of 32, got ", G);
  TORCH_CHECK(K % G == 0, "fni8: K (", K, ") must be divisible by group_size (", G, ")");
  TORCH_CHECK(K % 4 == 0, "fni8: contraction dim K must be %4==0 for dp4a int32 loads, got ", K);
  TORCH_CHECK(x_scale.numel() == M && w_scale.numel() == N * (K / G),
              "fni8: scale shape mismatch (x_scale ", x_scale.numel(), ", w_scale ",
              w_scale.numel(), " expected ", N * (K / G), ")");
  TORCH_CHECK(K % 32 == 0, "fni8: gemm_decode_w4a8 needs K %32==0 for 128-bit loads, got ", K);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr<uint8_t>()) % 16 == 0,
              "fni8: x/w must be 16-byte aligned for the vectorized uint4/int4 loads "
              "(contiguous .fni8 weights + fresh rowwise-quant activations satisfy this)");

  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0 || N == 0) return out;

  auto stream = at::cuda::getCurrentCUDAStream();
  const unsigned n_blocks = (unsigned)((N + DEC_WARPS_PER_BLOCK - 1) / DEC_WARPS_PER_BLOCK);
  const dim3 grid(n_blocks);
  if (out_dtype == at::kBFloat16) {
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    DEC_DISPATCH_MAX_M((int)M,
        (gemm_decode_w4a8_single_kernel<__nv_bfloat16, MAX_M><<<grid, DEC_THREADS, 0, stream>>>(
            x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
            w_scale.data_ptr<float>(), op, (int)M, (int)N, (int)K, G)));
  } else {
    auto* op = reinterpret_cast<__half*>(out.data_ptr());
    DEC_DISPATCH_MAX_M((int)M,
        (gemm_decode_w4a8_single_kernel<__half, MAX_M><<<grid, DEC_THREADS, 0, stream>>>(
            x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(),
            w_scale.data_ptr<float>(), op, (int)M, (int)N, (int)K, G)));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// W3A8 decode: x [M,K] int8 (M<=DEC_MAX_M), x_scale [M] fp32, w [N,(K/32)*3]
// int32 bit-planes (uniform 3-bit, Q3_K-style; passed as int32, reinterpreted to
// uint32 in-kernel), w_scale [N,K/G] fp32 -> out [M,N] `out_dtype`. Warp-per-
// column, split_k==1 (large-N MLP decode). VRAM/context lever: 3.0 bpw, decode
// PARITY with W4A8 (unpack ALU contends with dp4a), not a speedup (issue #181).
at::Tensor gemm_decode_w3a8(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, const at::Tensor& w_scale,
                            int64_t group_size, at::ScalarType out_dtype) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x_scale.is_cuda() && w_scale.is_cuda(),
              "fni8: all inputs (incl. scales) must be CUDA tensors");
  TORCH_CHECK(x.device() == w.device() && x.device() == x_scale.device() &&
                  x.device() == w_scale.device(),
              "fni8: all inputs must be on the same device");
  TORCH_CHECK(x.scalar_type() == at::kChar, "fni8: x must be int8");
  TORCH_CHECK(w.scalar_type() == at::kInt, "fni8: w must be int32 (packed 3-bit bit-planes)");
  TORCH_CHECK(x_scale.scalar_type() == at::kFloat && w_scale.scalar_type() == at::kFloat,
              "fni8: scales must be float32");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "fni8: expect x[M,K] and w[N,(K/32)*3]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && x_scale.is_contiguous() &&
                  w_scale.is_contiguous(),
              "fni8: inputs must be contiguous");
  check_out_dtype(out_dtype);

  const auto M = x.size(0), K = x.size(1);
  const auto N = w.size(0);
  const int G = (int)group_size;
  TORCH_CHECK(M <= DEC_MAX_M,
              "fni8: gemm_decode_w3a8 supports decode shapes M<=", DEC_MAX_M,
              ", got M=", M, " -- use a tile GEMM for prefill-size M");
  TORCH_CHECK(K % 32 == 0, "fni8: gemm_decode_w3a8 needs K %32==0 for the 32-value bit-plane "
              "group and 128-bit loads, got ", K);
  TORCH_CHECK(w.size(1) == (K / 32) * 3, "fni8: w must be [N,(K/32)*3] int32 bit-planes (K=", K, ")");
  TORCH_CHECK(G > 0 && G % 32 == 0, "fni8: group_size must be a positive multiple of 32, got ", G);
  TORCH_CHECK(K % G == 0, "fni8: K (", K, ") must be divisible by group_size (", G, ")");
  TORCH_CHECK(x_scale.numel() == M && w_scale.numel() == N * (K / G),
              "fni8: scale shape mismatch (x_scale ", x_scale.numel(), ", w_scale ",
              w_scale.numel(), " expected ", N * (K / G), ")");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 16 == 0,
              "fni8: x must be 16-byte aligned for the vectorized int4 activation loads "
              "(fresh rowwise-quant activations satisfy this)");

  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0 || N == 0) return out;

  auto stream = at::cuda::getCurrentCUDAStream();
  const unsigned n_blocks = (unsigned)((N + DEC_WARPS_PER_BLOCK - 1) / DEC_WARPS_PER_BLOCK);
  const dim3 grid(n_blocks);
  const auto* wp = reinterpret_cast<const uint32_t*>(w.data_ptr<int32_t>());
  if (out_dtype == at::kBFloat16) {
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    DEC_DISPATCH_MAX_M((int)M,
        (gemm_decode_w3a8_single_kernel<__nv_bfloat16, MAX_M><<<grid, DEC_THREADS, 0, stream>>>(
            x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), wp,
            w_scale.data_ptr<float>(), op, (int)M, (int)N, (int)K, G)));
  } else {
    auto* op = reinterpret_cast<__half*>(out.data_ptr());
    DEC_DISPATCH_MAX_M((int)M,
        (gemm_decode_w3a8_single_kernel<__half, MAX_M><<<grid, DEC_THREADS, 0, stream>>>(
            x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), wp,
            w_scale.data_ptr<float>(), op, (int)M, (int)N, (int)K, G)));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Fused fp16-input decode GEMV (issue #130 rung-1). x [M,K] fp16, w [N,K] int8,
// w_scale [N] fp32 -> out [M,N] `out_dtype`. Quantizes x -> int8 in smem in a
// prologue (no standalone quant kernel, no HBM round-trip of the int8 x). Only
// the fused-eligible case (split_k==1 AND M*K int8 fits in smem); the Python op
// falls back to quantize_int8_rowwise + gemm_decode_w8a8 otherwise.
constexpr size_t DEC_SMEM_CAP = 98304;   // 96 KB Volta dynamic-smem cap

at::Tensor gemm_decode_w8a8_fp16in(const at::Tensor& x, const at::Tensor& w,
                                   const at::Tensor& w_scale, at::ScalarType out_dtype) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && w_scale.is_cuda(),
              "fni8: all inputs must be CUDA tensors");
  TORCH_CHECK(x.device() == w.device() && x.device() == w_scale.device(),
              "fni8: all inputs must be on the same device");
  TORCH_CHECK(x.scalar_type() == at::kHalf, "fni8: gemm_decode_w8a8_fp16in x must be fp16");
  TORCH_CHECK(w.scalar_type() == at::kChar, "fni8: w must be int8");
  TORCH_CHECK(w_scale.scalar_type() == at::kFloat, "fni8: w_scale must be float32");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "fni8: expect x[M,K] and w[N,K]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && w_scale.is_contiguous(),
              "fni8: inputs must be contiguous");
  check_out_dtype(out_dtype);

  const auto M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M <= DEC_MAX_M, "fni8: gemm_decode_w8a8_fp16in supports decode shapes M<=",
              DEC_MAX_M, ", got M=", M);
  TORCH_CHECK(w.size(1) == K, "fni8: x/w contraction mismatch");
  TORCH_CHECK(K % 4 == 0, "fni8: K must be %4==0 for dp4a int32 loads, got ", K);
  TORCH_CHECK(w_scale.numel() == N, "fni8: w_scale shape mismatch");

  const int split_k = compute_split_k(N, K / 4);
  const size_t smem_bytes = (size_t)(((M * K + 15) & ~15)) + (size_t)M * sizeof(float);
  TORCH_CHECK(split_k == 1 && smem_bytes <= DEC_SMEM_CAP,
              "fni8: gemm_decode_w8a8_fp16in fused path needs split_k==1 and M*K int8 within "
              "smem (", DEC_SMEM_CAP, "B); got split_k=", split_k, ", smem=", smem_bytes,
              "B -- caller should fall back to quantize_int8_rowwise + gemm_decode_w8a8");

  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0 || N == 0) return out;

  auto stream = at::cuda::getCurrentCUDAStream();
  const unsigned n_blocks = (unsigned)((N + DEC_WARPS_PER_BLOCK - 1) / DEC_WARPS_PER_BLOCK);
  const dim3 grid(n_blocks);
  const __half* xp = reinterpret_cast<const __half*>(x.data_ptr());

  if (out_dtype == at::kBFloat16) {
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    DEC_DISPATCH_MAX_M((int)M,
        (launch_decode_fp16in<__nv_bfloat16, MAX_M>(grid, smem_bytes, stream, xp,
            w.data_ptr<int8_t>(), w_scale.data_ptr<float>(), op, (int)M, (int)N, (int)K)));
  } else {
    auto* op = reinterpret_cast<__half*>(out.data_ptr());
    DEC_DISPATCH_MAX_M((int)M,
        (launch_decode_fp16in<__half, MAX_M>(grid, smem_bytes, stream, xp,
            w.data_ptr<int8_t>(), w_scale.data_ptr<float>(), op, (int)M, (int)N, (int)K)));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ── Native GGUF k-quant DECODE launchers (warp-per-column MMVQ) ──────────────
// Shared checks + launch for Q4_K/Q5_K/Q6_K decode. x [M,K] int8 (M<=DEC_MAX_M,
// K%256==0), x_scale [M] fp32, w [N,(K/256)*type_size] uint8 native k-quant ->
// out [M,N] out_dtype. Warp-per-column; same per-sub-block math as the tile
// gemm_q{4,5,6}k. Callers use the tile kernel for prefill-size M.
namespace {
template <typename Launch>
at::Tensor decode_kquant(const at::Tensor& x, const at::Tensor& x_scale, const at::Tensor& w,
                         at::ScalarType out_dtype, int type_size, const char* tag, Launch launch) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x_scale.is_cuda(),
              "fni8: all inputs (incl. scale) must be CUDA tensors");
  TORCH_CHECK(x.device() == w.device() && x.device() == x_scale.device(),
              "fni8: all inputs must be on the same device");
  TORCH_CHECK(x.scalar_type() == at::kChar, "fni8: x must be int8");
  TORCH_CHECK(w.scalar_type() == at::kByte, "fni8: w must be uint8 (native k-quant bytes)");
  TORCH_CHECK(x_scale.scalar_type() == at::kFloat, "fni8: x_scale must be float32");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "fni8: expect x[M,K] and w[N,(K/256)*type_size]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && x_scale.is_contiguous(),
              "fni8: inputs must be contiguous");
  check_out_dtype(out_dtype);
  const auto M = x.size(0), K = x.size(1);
  const auto N = w.size(0);
  TORCH_CHECK(M <= DEC_MAX_M, "fni8: gemm_decode_", tag, " supports M<=", DEC_MAX_M,
              ", got M=", M, " -- use the tile gemm_", tag, " for prefill");
  TORCH_CHECK(K % 256 == 0, "fni8: ", tag, " needs K%256==0, got ", K);
  const int num_sb = (int)(K / 256);
  TORCH_CHECK(w.size(1) == (int64_t)num_sb * type_size, "fni8: ", tag,
              " weight must be [N,(K/256)*", type_size, "]");
  TORCH_CHECK(x_scale.numel() == M, "fni8: x_scale must be [M]");
  // 16-byte alignment: torch allocs are >=16B; native row_bytes = num_sb*ts, and
  // 144/176/210 are all %16-friendly for the uint4 qs/qh loads within a block
  // (ts%16: 144->0, 176->0, 210->2 but Q6_K reads int4 x only + scalar weight).
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 16 == 0,
              "fni8: x must be 16-byte aligned for the vectorized int4 loads");
  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0 || N == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)((N + DEC_WARPS_PER_BLOCK - 1) / DEC_WARPS_PER_BLOCK));
  launch(grid, stream, x, x_scale, w, out, (int)M, (int)N, (int)K, num_sb, out_dtype);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}  // namespace

#define DECODE_KQUANT_LAUNCH(KERN)                                                          \
  [](dim3 grid, cudaStream_t stream, const at::Tensor& x, const at::Tensor& x_scale,        \
     const at::Tensor& w, at::Tensor& out, int M, int N, int K, int num_sb,                 \
     at::ScalarType od) {                                                                    \
    if (od == at::kBFloat16) {                                                               \
      auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());                           \
      DEC_DISPATCH_MAX_M(M, (KERN<__nv_bfloat16, MAX_M><<<grid, DEC_THREADS, 0, stream>>>(   \
          x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(), op,        \
          M, N, K, num_sb)));                                                                \
    } else {                                                                                 \
      auto* op = reinterpret_cast<__half*>(out.data_ptr());                                  \
      DEC_DISPATCH_MAX_M(M, (KERN<__half, MAX_M><<<grid, DEC_THREADS, 0, stream>>>(          \
          x.data_ptr<int8_t>(), x_scale.data_ptr<float>(), w.data_ptr<uint8_t>(), op,        \
          M, N, K, num_sb)));                                                                \
    }                                                                                        \
  }

at::Tensor gemm_decode_q4k(const at::Tensor& x, const at::Tensor& x_scale,
                           const at::Tensor& w, at::ScalarType out_dtype) {
  return decode_kquant(x, x_scale, w, out_dtype, 144, "q4k",
                       DECODE_KQUANT_LAUNCH(gemm_decode_q4k_kernel));
}
at::Tensor gemm_decode_q5k(const at::Tensor& x, const at::Tensor& x_scale,
                           const at::Tensor& w, at::ScalarType out_dtype) {
  return decode_kquant(x, x_scale, w, out_dtype, 176, "q5k",
                       DECODE_KQUANT_LAUNCH(gemm_decode_q5k_kernel));
}
at::Tensor gemm_decode_q6k(const at::Tensor& x, const at::Tensor& x_scale,
                           const at::Tensor& w, at::ScalarType out_dtype) {
  return decode_kquant(x, x_scale, w, out_dtype, 210, "q6k",
                       DECODE_KQUANT_LAUNCH(gemm_decode_q6k_kernel));
}
at::Tensor gemm_decode_q3k(const at::Tensor& x, const at::Tensor& x_scale,
                           const at::Tensor& w, at::ScalarType out_dtype) {
  return decode_kquant(x, x_scale, w, out_dtype, 110, "q3k",
                       DECODE_KQUANT_LAUNCH(gemm_decode_q3k_kernel));
}
at::Tensor gemm_decode_iq3s(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, at::ScalarType out_dtype) {
  return decode_kquant(x, x_scale, w, out_dtype, IQ3S_TYPE_SIZE, "iq3s",
                       DECODE_KQUANT_LAUNCH(gemm_decode_iq3s_kernel));
}
at::Tensor gemm_decode_iq4xs(const at::Tensor& x, const at::Tensor& x_scale,
                             const at::Tensor& w, at::ScalarType out_dtype) {
  return decode_kquant(x, x_scale, w, out_dtype, IQ4XS_TYPE_SIZE, "iq4xs",
                       DECODE_KQUANT_LAUNCH(gemm_decode_iq4xs_kernel));
}
at::Tensor gemm_decode_iq3xxs(const at::Tensor& x, const at::Tensor& x_scale,
                              const at::Tensor& w, at::ScalarType out_dtype) {
  return decode_kquant(x, x_scale, w, out_dtype, IQ3XXS_TYPE_SIZE, "iq3xxs",
                       DECODE_KQUANT_LAUNCH(gemm_decode_iq3xxs_kernel));
}

at::Tensor gemm_decode_iq2s(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, at::ScalarType out_dtype) {
  return decode_kquant(x, x_scale, w, out_dtype, IQ2S_TYPE_SIZE, "iq2s",
                       DECODE_KQUANT_LAUNCH(gemm_decode_iq2s_kernel));
}

at::Tensor gemm_decode_iq2xs(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, at::ScalarType out_dtype) {
  return decode_kquant(x, x_scale, w, out_dtype, IQ2XS_TYPE_SIZE, "iq2xs",
                       DECODE_KQUANT_LAUNCH(gemm_decode_iq2xs_kernel));
}

at::Tensor gemm_decode_iq2xxs(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, at::ScalarType out_dtype) {
  return decode_kquant(x, x_scale, w, out_dtype, IQ2XXS_TYPE_SIZE, "iq2xxs",
                       DECODE_KQUANT_LAUNCH(gemm_decode_iq2xxs_kernel));
}

at::Tensor gemm_decode_iq1s(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, at::ScalarType out_dtype) {
  return decode_kquant(x, x_scale, w, out_dtype, IQ1S_TYPE_SIZE, "iq1s",
                       DECODE_KQUANT_LAUNCH(gemm_decode_iq1s_kernel));
}
at::Tensor gemm_decode_q2k(const at::Tensor& x, const at::Tensor& x_scale,
                           const at::Tensor& w, at::ScalarType out_dtype) {
  return decode_kquant(x, x_scale, w, out_dtype, 84, "q2k",
                       DECODE_KQUANT_LAUNCH(gemm_decode_q2k_kernel));
}

// ── TQ3_4S decode (warp-per-column MMVQ) ─────────────────────────────────────
// x [M,K] fp16/bf16 (M<=DEC_MAX_M, K%32==0), w [N,(K/32)*16] uint8 native TQ3_4S
// -> out [M,N] `out_dtype`. Same math as the tile gemm_tq34s; the forward RHT +
// per-32 q8_1 quant of the activation runs ONCE in `tq34s_rht_prepass_kernel`
// (fni8#281 perf pass — it was 1280x redundant per block at M=1), then the
// warp-per-column kernel stages the int8 activation into smem and streams each
// output column's native blocks. Guarded to smem <= 98304 (the Python wrapper
// routes larger M*K to the tile).
constexpr size_t TQ3_DEC_SMEM_CAP = 98304;

namespace {
template <typename OutT, int MAX_M>
void launch_decode_tq34s(dim3 grid, size_t smem_bytes, cudaStream_t stream,
                         const int8_t* xqp, const float* xsp, const uint8_t* wp,
                         OutT* op, int M, int N, int K, int nblk) {
  auto kern = gemm_decode_tq34s_kernel<OutT, MAX_M>;
  if (smem_bytes > 48 * 1024)
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
  kern<<<grid, DEC_THREADS, smem_bytes, stream>>>(xqp, xsp, wp, op, M, N, K, nblk);
}
}  // namespace

at::Tensor gemm_decode_tq34s(const at::Tensor& x, const at::Tensor& w,
                             at::ScalarType out_dtype) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(),
              "fni8: all inputs must be CUDA tensors");
  TORCH_CHECK(x.device() == w.device(),
              "fni8: all inputs must be on the same device");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "fni8: gemm_decode_tq34s x must be fp16 or bf16, got ", x.scalar_type());
  TORCH_CHECK(w.scalar_type() == at::kByte,
              "fni8: w must be uint8 (native TQ3_4S bytes)");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "fni8: expect x[M,K] and w[N,(K/32)*16]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
              "fni8: inputs must be contiguous");
  check_out_dtype(out_dtype);

  const auto M = x.size(0), K = x.size(1);
  const auto N = w.size(0);
  TORCH_CHECK(M <= DEC_MAX_M, "fni8: gemm_decode_tq34s supports M<=", DEC_MAX_M,
              ", got M=", M, " -- use the tile gemm_tq34s for prefill");
  TORCH_CHECK(K % TQ3_QK == 0, "fni8: TQ3_4S K must be %32==0, got ", K);
  const int nblk = (int)(K / TQ3_QK);
  TORCH_CHECK(w.size(1) == (int64_t)nblk * TQ3_TYPE_SIZE,
              "fni8: TQ3_4S weight must be [N,(K/32)*16], got [", N, ",", w.size(1),
              "] for K=", K);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr<uint8_t>()) % 16 == 0,
              "fni8: w must be 16-byte aligned for the vectorized uint4 block loads");
  const size_t smem_bytes = (size_t)(((M * K + 15) & ~15)) + (size_t)(M * nblk) * sizeof(float)
                          + 64 * sizeof(uint16_t);  // + the 128-B uint16 pair LUT
  TORCH_CHECK(smem_bytes <= TQ3_DEC_SMEM_CAP,
              "fni8: gemm_decode_tq34s fused activation smem ", smem_bytes,
              "B > cap ", TQ3_DEC_SMEM_CAP, "B (M*K too large) -- caller should "
              "route to the tile gemm_tq34s");

  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({M, N}, x.options().dtype(out_dtype));
  if (M == 0 || N == 0) return out;

  // Pre-rotate + quantize the activation ONCE (workspace: [M,K] int8 + [M,nblk] fp32).
  auto xq = at::empty({M, K}, x.options().dtype(at::kChar));
  auto xs = at::empty({M, nblk}, x.options().dtype(at::kFloat));

  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)((N + DEC_WARPS_PER_BLOCK - 1) / DEC_WARPS_PER_BLOCK));
  const bool bf16 = (x.scalar_type() == at::kBFloat16);
  const auto* wp = w.data_ptr<uint8_t>();
  {
    constexpr int threads = 256;
    const int total = (int)(M * nblk);
    const int blocks = std::max(
        1, std::min(512, (total + threads / 32 - 1) / (threads / 32)));
    if (bf16)
      tq34s_rht_prepass_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), xq.data_ptr<int8_t>(),
          xs.data_ptr<float>(), (int)M, (int)K, nblk);
    else
      tq34s_rht_prepass_kernel<__half><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __half*>(x.data_ptr()), xq.data_ptr<int8_t>(),
          xs.data_ptr<float>(), (int)M, (int)K, nblk);
  }
  const auto* xqp = xq.data_ptr<int8_t>();
  const auto* xsp = xs.data_ptr<float>();
  if (out_dtype == at::kBFloat16) {
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    DEC_DISPATCH_MAX_M((int)M,
        (launch_decode_tq34s<__nv_bfloat16, MAX_M>(grid, smem_bytes, stream, xqp, xsp, wp,
            op, (int)M, (int)N, (int)K, nblk)));
  } else {
    auto* op = reinterpret_cast<__half*>(out.data_ptr());
    DEC_DISPATCH_MAX_M((int)M,
        (launch_decode_tq34s<__half, MAX_M>(grid, smem_bytes, stream, xqp, xsp, wp,
            op, (int)M, (int)N, (int)K, nblk)));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ── Fused requant epilogue decode (issue #118 / PR #155, fresh attempt) ─────
// x [M,K] int8 (M<=DEC_MAX_M), x_scale [M] fp32, w [N,K] int8, w_scale [N] fp32
// -> (out_i8 [M,N] int8, out_scale [M] fp32).  Reuses the existing accum kernel
// for the int32 dot products, then a new requant epilogue kernel computes
// per-row absmax and requantizes — output feeds directly into the next int8 op
// without a standalone quantize kernel or an fp16-to-HBM round-trip.
std::tuple<at::Tensor, at::Tensor> gemm_decode_w8a8_requant(
    const at::Tensor& x, const at::Tensor& x_scale,
    const at::Tensor& w, const at::Tensor& w_scale) {
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

  const auto M = x.size(0), K = x.size(1);
  const auto N = w.size(0);
  TORCH_CHECK(M <= DEC_MAX_M,
              "fni8: gemm_decode_w8a8_requant supports decode shapes M<=", DEC_MAX_M,
              ", got M=", M);
  TORCH_CHECK(w.size(1) == K, "fni8: x/w contraction mismatch (x K=", K, ", w K=", w.size(1), ")");
  TORCH_CHECK(K % 4 == 0, "fni8: contraction dim K must be %4==0 for dp4a int32 loads, got ", K);
  TORCH_CHECK(x_scale.numel() == M && w_scale.numel() == N, "fni8: scale shape mismatch");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr<int8_t>()) % 4 == 0,
              "fni8: x/w must be 4-byte aligned for dp4a int32 loads");

  const at::cuda::CUDAGuard guard(x.device());
  auto out_i8 = at::empty({M, N}, x.options().dtype(at::kChar));
  auto out_scale = at::empty({M}, x.options().dtype(at::kFloat));
  if (M == 0 || N == 0) return std::make_tuple(out_i8, out_scale);

  const int split_k = compute_split_k(N, K / 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  const unsigned n_blocks = (unsigned)((N + DEC_WARPS_PER_BLOCK - 1) / DEC_WARPS_PER_BLOCK);

  auto acc32 = at::zeros({M, N}, x.options().dtype(at::kInt));
  const dim3 grid1(n_blocks, (unsigned)split_k);
  DEC_DISPATCH_MAX_M((int)M,
      (gemm_decode_accum_kernel<MAX_M><<<grid1, DEC_THREADS, 0, stream>>>(
          x.data_ptr<int8_t>(), w.data_ptr<int8_t>(), acc32.data_ptr<int32_t>(),
          (int)M, (int)N, (int)K, split_k)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 grid2((unsigned)M);
  DEC_DISPATCH_MAX_M((int)M,
      (gemm_decode_requant_epilogue_kernel<MAX_M><<<grid2, DEC_THREADS, 0, stream>>>(
          acc32.data_ptr<int32_t>(), x_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
          out_i8.data_ptr<int8_t>(), out_scale.data_ptr<float>(), (int)M, (int)N)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(out_i8, out_scale);
}

}  // namespace fni8
