// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Launcher + torch wrapper for the decode (M=1) split-KV attention.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "attn_decode.cuh"

namespace fni8 {

namespace {

// Split sizing for flash-decoding (fill the SMs, especially at batch≈1).
//
// The split grid is (n_splits, bhq) with ONE warp/block, so total resident warps
// ≈ n_splits*bhq (capped by the SMs). Decode is latency-bound (utils/docs/
// decode-profiling.md): it needs enough independent warps to hide the per-key
// load→dp4a→warp-reduce→exp2f chain. At high batch (bhq in the hundreds) the
// keys-per-split bound alone already yields plenty of warps. But the single-card
// autoregressive case — Qwen3.6-27B full attention is B=1, Hq=24 → bhq=24, and
// at D=256 each key does 2x the work — leaves the grid THIN: the old target of
// 256 blocks gave only ~3 warps/SM (~5-8% occupancy), so decode ran 1.5-1.8x
// slower than a well-split launch (measured, real V100 idx-14). We therefore
// target enough blocks to actually fill the 80 SMs, but clamp splits into a
// keys-per-split band so we neither leave huge serial per-split loops (MAX) nor
// over-split into a combine-dominated tail (MIN). At high batch s_occ is tiny and
// the MAX-keys floor governs — i.e. the high-batch behavior is unchanged.
constexpr int DEC_MAX_KEYS_PER_SPLIT = 256;  // cap serial work per split (lower split bound)
constexpr int DEC_MIN_KEYS_PER_SPLIT = 48;   // don't over-split into a combine-bound tail
constexpr int DEC_MAX_SPLITS = 256;          // long-context low-batch cap
constexpr int DEC_TARGET_BLOCKS = 2048;      // ~26 blocks/SM on 80 SMs to hide decode latency

int choose_n_splits(int n_len, int bhq) {
  if (n_len <= 0 || bhq <= 0) return 1;
  // Occupancy driver: open enough splits (blocks = n_splits*bhq) to fill the SMs.
  const int s_occ = (DEC_TARGET_BLOCKS + bhq - 1) / bhq;
  // Lower bound: each split does at most DEC_MAX_KEYS_PER_SPLIT keys (bounded
  // serial reduction) — this is the original keys-per-split target.
  const int s_lo = (n_len + DEC_MAX_KEYS_PER_SPLIT - 1) / DEC_MAX_KEYS_PER_SPLIT;
  // Upper bound: each split does at least DEC_MIN_KEYS_PER_SPLIT keys, so we do
  // not over-split short caches into a per-split-tail/combine-bound regime.
  const int s_hi = std::max(1, (n_len + DEC_MIN_KEYS_PER_SPLIT - 1) / DEC_MIN_KEYS_PER_SPLIT);
  // Aim for the occupancy target, but stay inside [s_lo, s_hi].
  int s = std::min(std::max(s_occ, s_lo), s_hi);
  if (s < 1) s = 1;
  if (s > DEC_MAX_SPLITS) s = DEC_MAX_SPLITS;
  return s;
}

// Split sizing for the VERIFY variant (M = k drafts). The grid is
// (n_splits, B*H_q*M), one warp/block, so total resident warps ≈ n_splits*bhm.
// The dense-tile W8A8 verify kernel launched only B*H_q blocks at the tiny-M /
// large-N MTP verify shape (B=1, H_q=16, k=2 → 16 blocks, ~6% occupancy,
// latency-bound), which masked the int8 bandwidth win. We open enough splits to
// fill the 80 SMs, clamped into a keys-per-split band so we neither leave huge
// serial per-split loops (MAX) nor over-split into a combine-bound tail (MIN).
// At long context (N≈4k) the shape fills to the target; at very short context
// with k=2 there simply aren't enough keys to reach it (documented, honest).
constexpr int VER_TARGET_BLOCKS = 2048;        // ~26 blocks/SM on 80 SMs
constexpr int VER_MAX_KEYS_PER_SPLIT = 256;    // cap serial work per split (lower split bound)
constexpr int VER_MIN_KEYS_PER_SPLIT = 48;     // don't over-split into a combine-bound tail

int choose_n_splits_verify(int n_len, int bhm) {
  if (n_len <= 0 || bhm <= 0) return 1;
  const int s_occ = (VER_TARGET_BLOCKS + bhm - 1) / bhm;
  const int s_lo = (n_len + VER_MAX_KEYS_PER_SPLIT - 1) / VER_MAX_KEYS_PER_SPLIT;
  const int s_hi = std::max(1, (n_len + VER_MIN_KEYS_PER_SPLIT - 1) / VER_MIN_KEYS_PER_SPLIT);
  int s = std::min(std::max(s_occ, s_lo), s_hi);
  if (s < 1) s = 1;
  if (s > DEC_MAX_SPLITS) s = DEC_MAX_SPLITS;
  return s;
}

template <int D>
void launch(const at::Tensor& q, const at::Tensor& q_scale, const at::Tensor& k,
            const at::Tensor& k_scale, const at::Tensor& v, at::Tensor& out,
            at::Tensor& o_part, at::Tensor& m_part, at::Tensor& l_part,
            int n_len, int n_splits, int64_t bhq, int h_q, int gqa_group,
            cudaStream_t stream) {
  // n_splits is pre-computed by the caller (choose_n_splits or user override).
  const dim3 split_grid((unsigned)n_splits, (unsigned)bhq);
  const dim3 combine_grid((unsigned)bhq);
  const dim3 block(DEC_WARP);
  decode_split_kernel<D><<<split_grid, block, 0, stream>>>(
      q.data_ptr<int8_t>(), q_scale.data_ptr<float>(), k.data_ptr<int8_t>(),
      k_scale.data_ptr<float>(), reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
      o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      n_len, n_splits, h_q, gqa_group);
  decode_combine_kernel<D><<<combine_grid, block, 0, stream>>>(
      o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n_splits);
}

template <int D>
void launch_i8v(const at::Tensor& q, const at::Tensor& q_scale, const at::Tensor& k,
                const at::Tensor& k_scale, const at::Tensor& v, const at::Tensor& v_scale,
                at::Tensor& out, at::Tensor& o_part, at::Tensor& m_part, at::Tensor& l_part,
                int n_len, int n_splits, int64_t bhq, int h_q, int gqa_group,
                cudaStream_t stream) {
  const dim3 split_grid((unsigned)n_splits, (unsigned)bhq);
  const dim3 combine_grid((unsigned)bhq);
  const dim3 block(DEC_WARP);
  decode_split_i8v_kernel<D><<<split_grid, block, 0, stream>>>(
      q.data_ptr<int8_t>(), q_scale.data_ptr<float>(), k.data_ptr<int8_t>(),
      k_scale.data_ptr<float>(), v.data_ptr<int8_t>(), v_scale.data_ptr<float>(),
      o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      n_len, n_splits, h_q, gqa_group);
  decode_combine_kernel<D><<<combine_grid, block, 0, stream>>>(
      o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n_splits);
}

// VERIFY (M = k drafts) launcher: split kernel over (n_splits, B*H_q*M) then the
// shared combine over (B*H_q*M). causal_diag = prefix = N - M (end-aligned mask).
template <int D>
void launch_i8v_verify(const at::Tensor& q, const at::Tensor& q_scale, const at::Tensor& k,
                       const at::Tensor& k_scale, const at::Tensor& v, const at::Tensor& v_scale,
                       at::Tensor& out, at::Tensor& o_part, at::Tensor& m_part, at::Tensor& l_part,
                       int n_len, int n_splits, int m_len, int64_t bhm, int h_q, int gqa_group,
                       int causal_diag, cudaStream_t stream) {
  const dim3 split_grid((unsigned)n_splits, (unsigned)bhm);
  const dim3 combine_grid((unsigned)bhm);
  const dim3 block(DEC_WARP);
  decode_split_i8v_verify_kernel<D><<<split_grid, block, 0, stream>>>(
      q.data_ptr<int8_t>(), q_scale.data_ptr<float>(), k.data_ptr<int8_t>(),
      k_scale.data_ptr<float>(), v.data_ptr<int8_t>(), v_scale.data_ptr<float>(),
      o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      n_len, n_splits, m_len, h_q, gqa_group, causal_diag);
  decode_combine_kernel<D><<<combine_grid, block, 0, stream>>>(
      o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n_splits);
}

template <int D>
void launch_i4v(const at::Tensor& q, const at::Tensor& q_scale, const at::Tensor& k,
                const at::Tensor& k_scale, const at::Tensor& v, const at::Tensor& v_scale,
                at::Tensor& out, at::Tensor& o_part, at::Tensor& m_part, at::Tensor& l_part,
                int n_len, int n_splits, int nblocks, int64_t bhq, int h_q,
                int gqa_group, cudaStream_t stream) {
  const dim3 split_grid((unsigned)n_splits, (unsigned)bhq);
  const dim3 block(DEC_WARP);
  decode_split_i4v_kernel<D><<<split_grid, block, 0, stream>>>(
      q.data_ptr<int8_t>(), q_scale.data_ptr<float>(), k.data_ptr<int8_t>(),
      k_scale.data_ptr<float>(), v.data_ptr<int8_t>(), v_scale.data_ptr<float>(),
      o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      n_len, n_splits, nblocks, h_q, gqa_group);
  decode_combine_kernel<D><<<dim3((unsigned)bhq), block, 0, stream>>>(
      o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n_splits);
}

}  // namespace

// q [B,H_q,1,D] int8, q_scale [B,H_q,1] fp32 (folded), k [B,H_kv,N,D] int8,
// k_scale [B,H_kv,N] fp32, v [B,H_kv,N,D] fp16. num_splits: -1 = auto (default),
// otherwise override the split count (for long-context tuning).
// Returns out [B,H_q,1,D] fp16.
at::Tensor attn_int8_decode(const at::Tensor& q, const at::Tensor& q_scale,
                            const at::Tensor& k, const at::Tensor& k_scale,
                            const at::Tensor& v, int64_t num_splits) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && q_scale.is_cuda() &&
                  k_scale.is_cuda(),
              "fni8 decode: all inputs must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                  q.device() == q_scale.device() && q.device() == k_scale.device(),
              "fni8 decode: all inputs must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kChar && k.scalar_type() == at::kChar,
              "fni8 decode: q/k must be int8");
  TORCH_CHECK(q_scale.scalar_type() == at::kFloat && k_scale.scalar_type() == at::kFloat,
              "fni8 decode: scales must be float32");
  TORCH_CHECK(v.scalar_type() == at::kHalf, "fni8 decode: v must be float16");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "fni8 decode: expect [B,H,S,D]");
  TORCH_CHECK(q.size(2) == 1, "fni8 decode: query length M must be 1 (decode)");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  q_scale.is_contiguous() && k_scale.is_contiguous(),
              "fni8 decode: inputs must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(k.data_ptr<int8_t>()) % 4 == 0,
              "fni8 decode: q/k must be 4-byte aligned for dp4a int32 loads");

  const auto B = q.size(0), H = q.size(1), D = q.size(3);
  const auto N = k.size(2);
  const auto H_KV = k.size(1);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == H_KV,
              "fni8 decode: batch/head mismatch");
  TORCH_CHECK(H_KV > 0 && H % H_KV == 0, "fni8 decode: H_q must be divisible by H_kv (GQA)");
  TORCH_CHECK(k.size(3) == D && v.size(3) == D && v.size(2) == N, "fni8 decode: K/V shape mismatch");
  TORCH_CHECK(q_scale.numel() == B * H && k_scale.numel() == B * H_KV * N,
              "fni8 decode: scale shape mismatch");
  TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
              "fni8 decode: head dim must be 32/64/128/256, got ", D);
  const int gqa_group = (int)(H / H_KV);

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({B, H, 1, D}, v.options());
  if (N == 0) return out.zero_();

  const int64_t bhq = B * H;
  int n_splits;
  if (num_splits >= 1) {
    n_splits = (int)num_splits;
  } else if (num_splits == 0) {
    TORCH_CHECK(false, "fni8 decode: num_splits=0 is invalid; use -1 for auto or 1..", DEC_MAX_SPLITS);
  } else {  // num_splits < 0 (None sentinel) → auto
    n_splits = choose_n_splits((int)N, (int)bhq);
  }
  TORCH_CHECK(n_splits >= 1 && n_splits <= DEC_MAX_SPLITS,
              "fni8 decode: num_splits must be 1..", DEC_MAX_SPLITS, ", got ", n_splits);
  auto fopt = q.options().dtype(at::kFloat);
  auto o_part = at::empty({n_splits, bhq, D}, fopt);
  auto m_part = at::empty({n_splits, bhq}, fopt);
  auto l_part = at::empty({n_splits, bhq}, fopt);
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (D) {
    case 32:
      launch<32>(q, q_scale, k, k_scale, v, out, o_part, m_part, l_part, (int)N, n_splits,
                 bhq, (int)H, gqa_group, stream);
      break;
    case 64:
      launch<64>(q, q_scale, k, k_scale, v, out, o_part, m_part, l_part, (int)N, n_splits,
                 bhq, (int)H, gqa_group, stream);
      break;
    case 128:
      launch<128>(q, q_scale, k, k_scale, v, out, o_part, m_part, l_part, (int)N, n_splits,
                  bhq, (int)H, gqa_group, stream);
      break;
    case 256:
      launch<256>(q, q_scale, k, k_scale, v, out, o_part, m_part, l_part, (int)N, n_splits,
                  bhq, (int)H, gqa_group, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// INT8 KV-cache decode: K and V both int8 (persistent int8 cache). v is int8
// [B,H_kv,N,D] with per-channel v_scale [B,H_kv,D]. Halves KV-cache memory and
// the V read bandwidth. num_splits: -1 = auto (default). Returns out [B,H_q,1,D] fp16.
at::Tensor attn_int8_decode_kv8(const at::Tensor& q, const at::Tensor& q_scale,
                                const at::Tensor& k, const at::Tensor& k_scale,
                                const at::Tensor& v, const at::Tensor& v_scale,
                                int64_t num_splits) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && q_scale.is_cuda() &&
                  k_scale.is_cuda() && v_scale.is_cuda(),
              "fni8 decode_kv8: all inputs must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                  q.device() == q_scale.device() && q.device() == k_scale.device() &&
                  q.device() == v_scale.device(),
              "fni8 decode_kv8: all inputs must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kChar && k.scalar_type() == at::kChar &&
                  v.scalar_type() == at::kChar,
              "fni8 decode_kv8: q/k/v must be int8");
  TORCH_CHECK(q_scale.scalar_type() == at::kFloat && k_scale.scalar_type() == at::kFloat &&
                  v_scale.scalar_type() == at::kFloat,
              "fni8 decode_kv8: scales must be float32");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "fni8 decode_kv8: expect [B,H,S,D]");
  TORCH_CHECK(q.size(2) == 1, "fni8 decode_kv8: query length M must be 1 (decode)");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  q_scale.is_contiguous() && k_scale.is_contiguous() &&
                  v_scale.is_contiguous(),
              "fni8 decode_kv8: inputs must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(k.data_ptr<int8_t>()) % 4 == 0,
              "fni8 decode_kv8: q/k must be 4-byte aligned for dp4a int32 loads");

  const auto B = q.size(0), H = q.size(1), D = q.size(3);
  const auto N = k.size(2);
  const auto H_KV = k.size(1);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == H_KV,
              "fni8 decode_kv8: batch/head mismatch");
  TORCH_CHECK(H_KV > 0 && H % H_KV == 0, "fni8 decode_kv8: H_q must be divisible by H_kv (GQA)");
  TORCH_CHECK(k.size(3) == D && v.size(3) == D && v.size(2) == N,
              "fni8 decode_kv8: K/V shape mismatch");
  TORCH_CHECK(q_scale.numel() == B * H && k_scale.numel() == B * H_KV * N &&
                  v_scale.numel() == B * H_KV * D,
              "fni8 decode_kv8: scale shape mismatch");
  TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
              "fni8 decode_kv8: head dim must be 32/64/128/256, got ", D);
  const int gqa_group = (int)(H / H_KV);

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({B, H, 1, D}, q.options().dtype(at::kHalf));
  if (N == 0) return out.zero_();

  const int64_t bhq = B * H;
  int n_splits;
  if (num_splits >= 1) {
    n_splits = (int)num_splits;
  } else if (num_splits == 0) {
    TORCH_CHECK(false, "fni8 decode_kv8: num_splits=0 is invalid; use -1 for auto or 1..", DEC_MAX_SPLITS);
  } else {
    n_splits = choose_n_splits((int)N, (int)bhq);
  }
  TORCH_CHECK(n_splits >= 1 && n_splits <= DEC_MAX_SPLITS,
              "fni8 decode_kv8: num_splits must be 1..", DEC_MAX_SPLITS, ", got ", n_splits);
  auto fopt = q.options().dtype(at::kFloat);
  auto o_part = at::empty({n_splits, bhq, D}, fopt);
  auto m_part = at::empty({n_splits, bhq}, fopt);
  auto l_part = at::empty({n_splits, bhq}, fopt);
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (D) {
    case 32:
      launch_i8v<32>(q, q_scale, k, k_scale, v, v_scale, out, o_part, m_part, l_part,
                     (int)N, n_splits, bhq, (int)H, gqa_group, stream);
      break;
    case 64:
      launch_i8v<64>(q, q_scale, k, k_scale, v, v_scale, out, o_part, m_part, l_part,
                     (int)N, n_splits, bhq, (int)H, gqa_group, stream);
      break;
    case 128:
      launch_i8v<128>(q, q_scale, k, k_scale, v, v_scale, out, o_part, m_part, l_part,
                      (int)N, n_splits, bhq, (int)H, gqa_group, stream);
      break;
    case 256:
      launch_i8v<256>(q, q_scale, k, k_scale, v, v_scale, out, o_part, m_part, l_part,
                      (int)N, n_splits, bhq, (int)H, gqa_group, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// INT4-V KV-cache decode: K int8, V int4 (packed 2 channels/byte -> [B,H_kv,N,D/2]),
// per-channel v_scale [B,H_kv,D]. Halves the V cache again vs int8. Returns fp16.
// num_splits: -1 = auto (default).
at::Tensor attn_int4v_decode(const at::Tensor& q, const at::Tensor& q_scale,
                             const at::Tensor& k, const at::Tensor& k_scale,
                             const at::Tensor& v, const at::Tensor& v_scale,
                             int64_t num_splits) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && q_scale.is_cuda() &&
                  k_scale.is_cuda() && v_scale.is_cuda(),
              "fni8 decode_i4v: all inputs must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                  q.device() == q_scale.device() && q.device() == k_scale.device() &&
                  q.device() == v_scale.device(),
              "fni8 decode_i4v: all inputs must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kChar && k.scalar_type() == at::kChar &&
                  v.scalar_type() == at::kChar,
              "fni8 decode_i4v: q/k/v(packed) must be int8");
  TORCH_CHECK(q_scale.scalar_type() == at::kFloat && k_scale.scalar_type() == at::kFloat &&
                  v_scale.scalar_type() == at::kFloat,
              "fni8 decode_i4v: scales must be float32");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "fni8 decode_i4v: expect [B,H,S,*]");
  TORCH_CHECK(q.size(2) == 1, "fni8 decode_i4v: query length M must be 1 (decode)");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  q_scale.is_contiguous() && k_scale.is_contiguous() &&
                  v_scale.is_contiguous(),
              "fni8 decode_i4v: inputs must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(k.data_ptr<int8_t>()) % 4 == 0,
              "fni8 decode_i4v: q/k must be 4-byte aligned for dp4a int32 loads");

  const auto B = q.size(0), H = q.size(1), D = q.size(3);
  const auto N = k.size(2);
  const auto H_KV = k.size(1);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == H_KV,
              "fni8 decode_i4v: batch/head mismatch");
  TORCH_CHECK(H_KV > 0 && H % H_KV == 0, "fni8 decode_i4v: H_q must be divisible by H_kv (GQA)");
  TORCH_CHECK(k.size(3) == D && v.size(2) == N && v.size(3) == D / 2,
              "fni8 decode_i4v: V must be [B,H_kv,N,D/2] int4-packed");
  const int nblocks = (int)((N + DEC_I4V_GROUP - 1) / DEC_I4V_GROUP);
  TORCH_CHECK(q_scale.numel() == B * H && k_scale.numel() == B * H_KV * N &&
                  v_scale.numel() == B * H_KV * (int64_t)nblocks * D,
              "fni8 decode_i4v: scale shape mismatch (v_scale must be [B,H_kv,nblocks,D])");
  TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
              "fni8 decode_i4v: head dim must be 32/64/128/256, got ", D);
  const int gqa_group = (int)(H / H_KV);

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({B, H, 1, D}, q.options().dtype(at::kHalf));
  if (N == 0) return out.zero_();

  const int64_t bhq = B * H;
  int n_splits;
  if (num_splits >= 1) {
    n_splits = (int)num_splits;
  } else if (num_splits == 0) {
    TORCH_CHECK(false, "fni8 decode_i4v: num_splits=0 is invalid; use -1 for auto or 1..", DEC_MAX_SPLITS);
  } else {
    n_splits = choose_n_splits((int)N, (int)bhq);
  }
  TORCH_CHECK(n_splits >= 1 && n_splits <= DEC_MAX_SPLITS,
              "fni8 decode_i4v: num_splits must be 1..", DEC_MAX_SPLITS, ", got ", n_splits);
  auto fopt = q.options().dtype(at::kFloat);
  auto o_part = at::empty({n_splits, bhq, D}, fopt);
  auto m_part = at::empty({n_splits, bhq}, fopt);
  auto l_part = at::empty({n_splits, bhq}, fopt);
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (D) {
    case 32:
      launch_i4v<32>(q, q_scale, k, k_scale, v, v_scale, out, o_part, m_part, l_part,
                     (int)N, n_splits, nblocks, bhq, (int)H, gqa_group, stream);
      break;
    case 64:
      launch_i4v<64>(q, q_scale, k, k_scale, v, v_scale, out, o_part, m_part, l_part,
                     (int)N, n_splits, nblocks, bhq, (int)H, gqa_group, stream);
      break;
    case 128:
      launch_i4v<128>(q, q_scale, k, k_scale, v, v_scale, out, o_part, m_part, l_part,
                      (int)N, n_splits, nblocks, bhq, (int)H, gqa_group, stream);
      break;
    case 256:
      launch_i4v<256>(q, q_scale, k, k_scale, v, v_scale, out, o_part, m_part, l_part,
                      (int)N, n_splits, nblocks, bhq, (int)H, gqa_group, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// VERIFY (M = k drafts) against a persistent INT8 KV cache via flash-decoding.
// q [B,H_q,M,D] int8, q_scale [B,H_q,M] fp32 (folded softmax_scale*log2e), k
// [B,H_kv,N,D] int8, k_scale [B,H_kv,N] fp32, v [B,H_kv,N,D] int8, v_scale
// [B,H_kv,D] per-channel fp32. causal_diag = prefix = N - M (end-aligned causal:
// draft i attends keys [0, prefix+i]). num_splits: -1 = auto (default), otherwise
// override. Splits the KEY dim across blocks (grid n_splits*B*H_q*M) to fill the
// SMs at the tiny-M verify shape, then LSE-combines (split-invariant → numerically
// the same as the dense-tile attn_w8a8_fwd verify). Returns out [B,H_q,M,D] fp16.
at::Tensor attn_int8_verify_split(const at::Tensor& q, const at::Tensor& q_scale,
                                  const at::Tensor& k, const at::Tensor& k_scale,
                                  const at::Tensor& v, const at::Tensor& v_scale,
                                  int64_t causal_diag, int64_t num_splits) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && q_scale.is_cuda() &&
                  k_scale.is_cuda() && v_scale.is_cuda(),
              "fni8 verify_split: all inputs must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                  q.device() == q_scale.device() && q.device() == k_scale.device() &&
                  q.device() == v_scale.device(),
              "fni8 verify_split: all inputs must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kChar && k.scalar_type() == at::kChar &&
                  v.scalar_type() == at::kChar,
              "fni8 verify_split: q/k/v must be int8");
  TORCH_CHECK(q_scale.scalar_type() == at::kFloat && k_scale.scalar_type() == at::kFloat &&
                  v_scale.scalar_type() == at::kFloat,
              "fni8 verify_split: scales must be float32");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "fni8 verify_split: expect [B,H,S,D]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  q_scale.is_contiguous() && k_scale.is_contiguous() &&
                  v_scale.is_contiguous(),
              "fni8 verify_split: inputs must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr<int8_t>()) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(k.data_ptr<int8_t>()) % 4 == 0,
              "fni8 verify_split: q/k must be 4-byte aligned for dp4a int32 loads");

  const auto B = q.size(0), H = q.size(1), M = q.size(2), D = q.size(3);
  const auto N = k.size(2);
  const auto H_KV = k.size(1);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == H_KV,
              "fni8 verify_split: batch/head mismatch");
  TORCH_CHECK(H_KV > 0 && H % H_KV == 0, "fni8 verify_split: H_q must be divisible by H_kv (GQA)");
  TORCH_CHECK(k.size(3) == D && v.size(3) == D && v.size(2) == N,
              "fni8 verify_split: K/V shape mismatch");
  TORCH_CHECK(q_scale.numel() == B * H * M && k_scale.numel() == B * H_KV * N &&
                  v_scale.numel() == B * H_KV * D,
              "fni8 verify_split: scale shape mismatch");
  TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
              "fni8 verify_split: head dim must be 32/64/128/256, got ", D);
  TORCH_CHECK(causal_diag >= 0 && causal_diag + M == N,
              "fni8 verify_split: causal_diag (prefix) must equal N - M");
  const int gqa_group = (int)(H / H_KV);

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({B, H, M, D}, q.options().dtype(at::kHalf));
  if (M == 0) return out;
  if (N == 0) return out.zero_();

  const int64_t bhm = B * H * M;
  int n_splits;
  if (num_splits >= 1) {
    n_splits = (int)num_splits;
  } else if (num_splits == 0) {
    TORCH_CHECK(false, "fni8 verify_split: num_splits=0 is invalid; use -1 for auto or 1..",
                DEC_MAX_SPLITS);
  } else {
    n_splits = choose_n_splits_verify((int)N, (int)bhm);
  }
  TORCH_CHECK(n_splits >= 1 && n_splits <= DEC_MAX_SPLITS,
              "fni8 verify_split: num_splits must be 1..", DEC_MAX_SPLITS, ", got ", n_splits);
  auto fopt = q.options().dtype(at::kFloat);
  auto o_part = at::empty({n_splits, bhm, D}, fopt);
  auto m_part = at::empty({n_splits, bhm}, fopt);
  auto l_part = at::empty({n_splits, bhm}, fopt);
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (D) {
    case 32:
      launch_i8v_verify<32>(q, q_scale, k, k_scale, v, v_scale, out, o_part, m_part, l_part,
                            (int)N, n_splits, (int)M, bhm, (int)H, gqa_group, (int)causal_diag, stream);
      break;
    case 64:
      launch_i8v_verify<64>(q, q_scale, k, k_scale, v, v_scale, out, o_part, m_part, l_part,
                            (int)N, n_splits, (int)M, bhm, (int)H, gqa_group, (int)causal_diag, stream);
      break;
    case 128:
      launch_i8v_verify<128>(q, q_scale, k, k_scale, v, v_scale, out, o_part, m_part, l_part,
                             (int)N, n_splits, (int)M, bhm, (int)H, gqa_group, (int)causal_diag, stream);
      break;
    case 256:
      launch_i8v_verify<256>(q, q_scale, k, k_scale, v, v_scale, out, o_part, m_part, l_part,
                             (int)N, n_splits, (int)M, bhm, (int)H, gqa_group, (int)causal_diag, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace fni8
