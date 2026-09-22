// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Launcher + torch wrappers for the paged-KV block-table decode and the
// quantize-on-write paged KV store.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "attn_paged_decode.cuh"

namespace fni8 {

namespace {

// Same low-batch retune as attn_decode.cu's choose_n_splits — the split grid is
// (n_splits, bhq) with one warp/block, so a thin B=1 batch (Qwen3.6-27B decode:
// Hq=24, D=256) needs many splits to fill the 80 SMs and hide the latency-bound
// per-key chain. Target enough blocks to fill the SMs, clamped into a keys-per-
// split band. At high batch s_occ is tiny and the MAX-keys floor governs, so
// high-batch behavior is unchanged. See utils/docs/decode-profiling.md.
constexpr int PDEC_MAX_KEYS_PER_SPLIT = 256;  // cap serial work per split
constexpr int PDEC_MIN_KEYS_PER_SPLIT = 48;   // don't over-split into a combine tail
constexpr int PDEC_MAX_SPLITS = 256;
constexpr int PDEC_TARGET_BLOCKS = 2048;  // ~26 blocks/SM on 80 SMs to hide decode latency

int choose_n_splits(int n_len_max, int bhq) {
  if (n_len_max <= 0 || bhq <= 0) return 1;
  const int s_occ = (PDEC_TARGET_BLOCKS + bhq - 1) / bhq;
  const int s_lo = (n_len_max + PDEC_MAX_KEYS_PER_SPLIT - 1) / PDEC_MAX_KEYS_PER_SPLIT;
  const int s_hi =
      std::max(1, (n_len_max + PDEC_MIN_KEYS_PER_SPLIT - 1) / PDEC_MIN_KEYS_PER_SPLIT);
  int s = std::min(std::max(s_occ, s_lo), s_hi);
  if (s < 1) s = 1;
  if (s > PDEC_MAX_SPLITS) s = PDEC_MAX_SPLITS;
  return s;
}

template <int D>
void launch_paged_decode(const at::Tensor& q, const at::Tensor& q_scale,
                         const at::Tensor& k_cache, const at::Tensor& k_scale,
                         const at::Tensor& v_cache, const at::Tensor& v_scale,
                         const at::Tensor& block_table, const at::Tensor& context_lens,
                         at::Tensor& out, at::Tensor& o_part, at::Tensor& m_part,
                         at::Tensor& l_part, int n_len_max, int n_splits,
                         int max_blocks_per_seq, int block_size, int h_kv,
                         int64_t bhq, int h_q, int gqa_group, cudaStream_t stream) {
  const dim3 split_grid((unsigned)n_splits, (unsigned)bhq);
  const dim3 combine_grid((unsigned)bhq);
  const dim3 block(DEC_WARP);
  paged_decode_split_kernel<D><<<split_grid, block, 0, stream>>>(
      q.data_ptr<int8_t>(), q_scale.data_ptr<float>(), k_cache.data_ptr<int8_t>(),
      k_scale.data_ptr<float>(), v_cache.data_ptr<int8_t>(), v_scale.data_ptr<float>(),
      nullptr, nullptr, nullptr,
      block_table.data_ptr<int32_t>(), context_lens.data_ptr<int32_t>(),
      o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      n_len_max, n_splits, max_blocks_per_seq, block_size, h_kv, h_q, gqa_group);
  decode_combine_kernel<D><<<combine_grid, block, 0, stream>>>(
      o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n_splits);
}

template <int D>
void launch_paged_decode_k8v3(const at::Tensor& q, const at::Tensor& q_scale,
                              const at::Tensor& k_cache, const at::Tensor& k_scale,
                              const at::Tensor& v_packed, const at::Tensor& v_norm,
                              const at::Tensor& v_codebook,
                              const at::Tensor& block_table,
                              const at::Tensor& context_lens,
                              at::Tensor& out, at::Tensor& o_part, at::Tensor& m_part,
                              at::Tensor& l_part, int n_len_max, int n_splits,
                              int max_blocks_per_seq, int block_size, int h_kv,
                              int64_t bhq, int h_q, int gqa_group, cudaStream_t stream) {
  const dim3 split_grid((unsigned)n_splits, (unsigned)bhq);
  const dim3 combine_grid((unsigned)bhq);
  const dim3 block(DEC_WARP);
  // The int8 V args are unused in the LowbitV instantiation (nullptr); the
  // packed V store is consumed directly in-kernel.
  paged_decode_split_kernel<D, true><<<split_grid, block, 0, stream>>>(
      q.data_ptr<int8_t>(), q_scale.data_ptr<float>(), k_cache.data_ptr<int8_t>(),
      k_scale.data_ptr<float>(), nullptr, nullptr, v_packed.data_ptr<int32_t>(),
      v_norm.data_ptr<float>(), v_codebook.data_ptr<float>(),
      block_table.data_ptr<int32_t>(), context_lens.data_ptr<int32_t>(),
      o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      n_len_max, n_splits, max_blocks_per_seq, block_size, h_kv, h_q, gqa_group);
  decode_combine_kernel<D><<<combine_grid, block, 0, stream>>>(
      o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), n_splits);
}

template <int D>
void launch_kv_write(const at::Tensor& k_new, const at::Tensor& v_new,
                     const at::Tensor& slot_mapping, at::Tensor& k_cache,
                     at::Tensor& k_scale, at::Tensor& v_cache, at::Tensor& v_scale,
                     int64_t bh, int h_kv, int block_size, cudaStream_t stream) {
  const dim3 grid((unsigned)bh);
  const dim3 block(DEC_WARP);
  kv_write_paged_kernel<D><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(k_new.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(v_new.data_ptr<at::Half>()),
      slot_mapping.data_ptr<int32_t>(), k_cache.data_ptr<int8_t>(),
      k_scale.data_ptr<float>(), v_cache.data_ptr<int8_t>(), v_scale.data_ptr<float>(),
      h_kv, block_size);
}

}  // namespace

// Paged decode (M=1) against an int8 K/V cache addressed via a block table.
// q [B,H_q,1,D] int8, q_scale [B,H_q,1] fp32 (folded), k_cache/v_cache
// [num_blocks,H_kv,block_size,D] int8, k_scale/v_scale
// [num_blocks,H_kv,block_size] fp32, block_table [B,max_blocks_per_seq] int32,
// context_lens [B] int32. num_splits: -1 = auto (default).
// Returns out [B,H_q,1,D] fp16.
at::Tensor attn_paged_decode(const at::Tensor& q, const at::Tensor& q_scale,
                             const at::Tensor& k_cache, const at::Tensor& k_scale,
                             const at::Tensor& v_cache, const at::Tensor& v_scale,
                             const at::Tensor& block_table, const at::Tensor& context_lens,
                             int64_t block_size, int64_t max_context_len,
                             int64_t num_splits) {
  TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda() && q_scale.is_cuda() &&
                  k_scale.is_cuda() && v_scale.is_cuda() && block_table.is_cuda() &&
                  context_lens.is_cuda(),
              "fni8 paged_decode: all inputs must be CUDA tensors");
  TORCH_CHECK(q.device() == k_cache.device() && q.device() == v_cache.device() &&
                  q.device() == block_table.device() && q.device() == context_lens.device(),
              "fni8 paged_decode: all inputs must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kChar && k_cache.scalar_type() == at::kChar &&
                  v_cache.scalar_type() == at::kChar,
              "fni8 paged_decode: q/k_cache/v_cache must be int8");
  TORCH_CHECK(q_scale.scalar_type() == at::kFloat && k_scale.scalar_type() == at::kFloat &&
                  v_scale.scalar_type() == at::kFloat,
              "fni8 paged_decode: scales must be float32");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && context_lens.scalar_type() == at::kInt,
              "fni8 paged_decode: block_table/context_lens must be int32");
  TORCH_CHECK(q.dim() == 4 && q.size(2) == 1, "fni8 paged_decode: q must be [B,H_q,1,D]");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
              "fni8 paged_decode: k_cache/v_cache must be [num_blocks,H_kv,block_size,D]");
  TORCH_CHECK(q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous() &&
                  q_scale.is_contiguous() && k_scale.is_contiguous() &&
                  v_scale.is_contiguous() && block_table.is_contiguous() &&
                  context_lens.is_contiguous(),
              "fni8 paged_decode: inputs must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr<int8_t>()) % 4 == 0,
              "fni8 paged_decode: q must be 4-byte aligned for dp4a int32 loads");

  const auto B = q.size(0), H = q.size(1), D = q.size(3);
  const auto H_KV = k_cache.size(1);
  const auto bs = k_cache.size(2);
  TORCH_CHECK(bs == block_size, "fni8 paged_decode: block_size mismatch vs k_cache.size(2)");
  TORCH_CHECK(k_cache.size(3) == D && v_cache.size(3) == D && v_cache.size(1) == H_KV &&
                  v_cache.size(2) == bs,
              "fni8 paged_decode: k_cache/v_cache shape mismatch");
  TORCH_CHECK(H_KV > 0 && H % H_KV == 0, "fni8 paged_decode: H_q must be divisible by H_kv (GQA)");
  TORCH_CHECK(k_cache.numel() == k_scale.numel() * D && v_cache.numel() == v_scale.numel() * D,
              "fni8 paged_decode: scale shape mismatch vs cache");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == B,
              "fni8 paged_decode: block_table must be [B, max_blocks_per_seq]");
  TORCH_CHECK(context_lens.dim() == 1 && context_lens.size(0) == B,
              "fni8 paged_decode: context_lens must be [B]");
  TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
              "fni8 paged_decode: head dim must be 32/64/128/256, got ", D);
  const int max_blocks_per_seq = (int)block_table.size(1);
  TORCH_CHECK(max_context_len >= 0 && max_context_len <= max_blocks_per_seq * bs,
              "fni8 paged_decode: max_context_len exceeds block_table capacity");

  const int gqa_group = (int)(H / H_KV);
  // max_context_len is the true upper bound on context_lens (max(context_lens),
  // passed by the caller) -- NOT block_table's allocated capacity, which a
  // serving engine typically over-provisions (e.g. to a model's max sequence
  // length) well beyond what any sequence in this launch actually holds.
  // Sizing splits off the allocation instead of the real max would waste most
  // of them on ranges no sequence reaches.
  const int n_len_max = (int)max_context_len;

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({B, H, 1, D}, q.options().dtype(at::kHalf));

  const int64_t bhq = B * H;
  int n_splits;
  if (num_splits >= 1) {
    n_splits = (int)num_splits;
  } else if (num_splits == 0) {
    TORCH_CHECK(false, "fni8 paged_decode: num_splits=0 is invalid; use -1 for auto or 1..", PDEC_MAX_SPLITS);
  } else {
    n_splits = choose_n_splits(n_len_max, (int)bhq);
  }
  TORCH_CHECK(n_splits >= 1 && n_splits <= PDEC_MAX_SPLITS,
              "fni8 paged_decode: num_splits must be 1..", PDEC_MAX_SPLITS,
              ", got ", n_splits);
  auto fopt = q.options().dtype(at::kFloat);
  auto o_part = at::empty({n_splits, bhq, D}, fopt);
  auto m_part = at::empty({n_splits, bhq}, fopt);
  auto l_part = at::empty({n_splits, bhq}, fopt);
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (D) {
    case 32:
      launch_paged_decode<32>(q, q_scale, k_cache, k_scale, v_cache, v_scale, block_table,
                              context_lens, out, o_part, m_part, l_part, n_len_max, n_splits,
                              max_blocks_per_seq, (int)bs, (int)H_KV, bhq, (int)H, gqa_group,
                              stream);
      break;
    case 64:
      launch_paged_decode<64>(q, q_scale, k_cache, k_scale, v_cache, v_scale, block_table,
                              context_lens, out, o_part, m_part, l_part, n_len_max, n_splits,
                              max_blocks_per_seq, (int)bs, (int)H_KV, bhq, (int)H, gqa_group,
                              stream);
      break;
    case 128:
      launch_paged_decode<128>(q, q_scale, k_cache, k_scale, v_cache, v_scale, block_table,
                               context_lens, out, o_part, m_part, l_part, n_len_max, n_splits,
                               max_blocks_per_seq, (int)bs, (int)H_KV, bhq, (int)H, gqa_group,
                               stream);
      break;
    case 256:
      launch_paged_decode<256>(q, q_scale, k_cache, k_scale, v_cache, v_scale, block_table,
                               context_lens, out, o_part, m_part, l_part, n_len_max, n_splits,
                               max_blocks_per_seq, (int)bs, (int)H_KV, bhq, (int)H, gqa_group,
                               stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// K8V3 fused paged decode (M=1, fni8#295): the int8 rotated K path of
// `attn_paged_decode` unchanged, but V is consumed as the LloydMax3 packed
// store instead of int8 rows — v_packed [num_blocks,H_kv,block_size,D*3/32]
// int32 (3-bit codes, LSB-first), v_norm [num_blocks,H_kv,block_size,D/128]
// fp32 (per-128-block L2 norms), v_codebook [8] fp32 (fixed Gaussian, the
// layer's row). Same block_table/context_lens/num_splits contract; one batched
// paged launch with no Python per-slot loop, no fp16 V materialization and no
// K re-quantization.
at::Tensor attn_paged_decode_k8v3(const at::Tensor& q, const at::Tensor& q_scale,
                                  const at::Tensor& k_cache, const at::Tensor& k_scale,
                                  const at::Tensor& v_packed, const at::Tensor& v_norm,
                                  const at::Tensor& v_codebook,
                                  const at::Tensor& block_table,
                                  const at::Tensor& context_lens,
                                  int64_t block_size, int64_t max_context_len,
                                  int64_t num_splits) {
  TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_packed.is_cuda() &&
                  v_norm.is_cuda() && v_codebook.is_cuda() && q_scale.is_cuda() &&
                  k_scale.is_cuda() && block_table.is_cuda() && context_lens.is_cuda(),
              "fni8 paged_decode_k8v3: all inputs must be CUDA tensors");
  TORCH_CHECK(q.device() == q_scale.device() && q.device() == k_cache.device() &&
                  q.device() == k_scale.device() && q.device() == v_packed.device() &&
                  q.device() == v_norm.device() && q.device() == v_codebook.device() &&
                  q.device() == block_table.device() && q.device() == context_lens.device(),
              "fni8 paged_decode_k8v3: all inputs must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kChar && k_cache.scalar_type() == at::kChar,
              "fni8 paged_decode_k8v3: q/k_cache must be int8");
  TORCH_CHECK(v_packed.scalar_type() == at::kInt,
              "fni8 paged_decode_k8v3: v_packed must be int32");
  TORCH_CHECK(q_scale.scalar_type() == at::kFloat && k_scale.scalar_type() == at::kFloat &&
                  v_norm.scalar_type() == at::kFloat && v_codebook.scalar_type() == at::kFloat,
              "fni8 paged_decode_k8v3: scales/norms/codebook must be float32");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && context_lens.scalar_type() == at::kInt,
              "fni8 paged_decode_k8v3: block_table/context_lens must be int32");
  TORCH_CHECK(q.dim() == 4 && q.size(2) == 1, "fni8 paged_decode_k8v3: q must be [B,H_q,1,D]");
  TORCH_CHECK(k_cache.dim() == 4 && v_packed.dim() == 4 && v_norm.dim() == 4,
              "fni8 paged_decode_k8v3: caches must be [num_blocks,H_kv,block_size,...]");
  TORCH_CHECK(q.is_contiguous() && k_cache.is_contiguous() && v_packed.is_contiguous() &&
                  v_norm.is_contiguous() && v_codebook.is_contiguous() &&
                  q_scale.is_contiguous() && k_scale.is_contiguous() &&
                  block_table.is_contiguous() && context_lens.is_contiguous(),
              "fni8 paged_decode_k8v3: inputs must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr<int8_t>()) % 4 == 0,
              "fni8 paged_decode_k8v3: q must be 4-byte aligned for dp4a int32 loads");

  const auto B = q.size(0), H = q.size(1), D = q.size(3);
  const auto H_KV = k_cache.size(1);
  const auto bs = k_cache.size(2);
  TORCH_CHECK(bs == block_size, "fni8 paged_decode_k8v3: block_size mismatch vs k_cache.size(2)");
  TORCH_CHECK(k_cache.size(3) == D && v_packed.size(0) == k_cache.size(0) &&
                  v_packed.size(1) == H_KV && v_packed.size(2) == bs &&
                  v_norm.size(0) == k_cache.size(0) && v_norm.size(1) == H_KV &&
                  v_norm.size(2) == bs,
              "fni8 paged_decode_k8v3: k_cache/v_packed/v_norm shape mismatch");
  TORCH_CHECK(H_KV > 0 && H % H_KV == 0,
              "fni8 paged_decode_k8v3: H_q must be divisible by H_kv (GQA)");
  TORCH_CHECK((D == 128 || D == 256) && D % 128 == 0,
              "fni8 paged_decode_k8v3: lloydmax3 requires head dim 128 or 256, got ", D);
  TORCH_CHECK(k_cache.numel() == k_scale.numel() * D,
              "fni8 paged_decode_k8v3: k_scale shape mismatch vs k_cache");
  TORCH_CHECK(v_packed.size(3) == (int64_t)((D * 3) / 32),
              "fni8 paged_decode_k8v3: v_packed last dim must be D*3/32, got ",
              v_packed.size(3));
  TORCH_CHECK(v_norm.size(3) == (int64_t)(D / 128),
              "fni8 paged_decode_k8v3: v_norm last dim must be D/128, got ", v_norm.size(3));
  TORCH_CHECK(v_codebook.dim() == 1 && v_codebook.numel() == 8,
              "fni8 paged_decode_k8v3: v_codebook must be [8] fp32");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == B,
              "fni8 paged_decode_k8v3: block_table must be [B, max_blocks_per_seq]");
  TORCH_CHECK(context_lens.dim() == 1 && context_lens.size(0) == B,
              "fni8 paged_decode_k8v3: context_lens must be [B]");
  const int max_blocks_per_seq = (int)block_table.size(1);
  TORCH_CHECK(max_context_len >= 0 && max_context_len <= max_blocks_per_seq * bs,
              "fni8 paged_decode_k8v3: max_context_len exceeds block_table capacity");

  const int gqa_group = (int)(H / H_KV);
  const int n_len_max = (int)max_context_len;

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({B, H, 1, D}, q.options().dtype(at::kHalf));

  const int64_t bhq = B * H;
  int n_splits;
  if (num_splits >= 1) {
    n_splits = (int)num_splits;
  } else if (num_splits == 0) {
    TORCH_CHECK(false, "fni8 paged_decode_k8v3: num_splits=0 is invalid; use -1 for auto or 1..",
                PDEC_MAX_SPLITS);
  } else {
    n_splits = choose_n_splits(n_len_max, (int)bhq);
  }
  TORCH_CHECK(n_splits >= 1 && n_splits <= PDEC_MAX_SPLITS,
              "fni8 paged_decode_k8v3: num_splits must be 1..", PDEC_MAX_SPLITS,
              ", got ", n_splits);
  auto fopt = q.options().dtype(at::kFloat);
  auto o_part = at::empty({n_splits, bhq, D}, fopt);
  auto m_part = at::empty({n_splits, bhq}, fopt);
  auto l_part = at::empty({n_splits, bhq}, fopt);
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (D) {
    case 128:
      launch_paged_decode_k8v3<128>(q, q_scale, k_cache, k_scale, v_packed, v_norm,
                                    v_codebook, block_table, context_lens, out, o_part,
                                    m_part, l_part, n_len_max, n_splits,
                                    max_blocks_per_seq, (int)bs, (int)H_KV, bhq, (int)H,
                                    gqa_group, stream);
      break;
    case 256:
      launch_paged_decode_k8v3<256>(q, q_scale, k_cache, k_scale, v_packed, v_norm,
                                    v_codebook, block_table, context_lens, out, o_part,
                                    m_part, l_part, n_len_max, n_splits,
                                    max_blocks_per_seq, (int)bs, (int)H_KV, bhq, (int)H,
                                    gqa_group, stream);
      break;
    default:
      TORCH_CHECK(false, "fni8 paged_decode_k8v3: lloydmax3 requires head dim 128 or 256, got ",
                  D);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Quantize-on-write: commits ONE new token's K/V for every (batch, H_kv) row
// into the paged int8 cache in-place (k_cache/k_scale/v_cache/v_scale are
// mutated directly — the persistent serving-engine KV pool). k_new/v_new
// [B,H_kv,D] fp16 (K pre-rotated by the caller if using the Hadamard
// incoherence rotation); slot_mapping [B] int32 flat slot =
// block_id*block_size + offset, computed by the caller from its block table +
// per-sequence write position.
void kv_write_paged(const at::Tensor& k_new, const at::Tensor& v_new,
                    const at::Tensor& slot_mapping, at::Tensor& k_cache,
                    at::Tensor& k_scale, at::Tensor& v_cache, at::Tensor& v_scale) {
  TORCH_CHECK(k_new.is_cuda() && v_new.is_cuda() && slot_mapping.is_cuda() &&
                  k_cache.is_cuda() && v_cache.is_cuda(),
              "fni8 kv_write_paged: all inputs must be CUDA tensors");
  TORCH_CHECK(k_new.scalar_type() == at::kHalf && v_new.scalar_type() == at::kHalf,
              "fni8 kv_write_paged: k_new/v_new must be float16");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kInt,
              "fni8 kv_write_paged: slot_mapping must be int32");
  TORCH_CHECK(k_cache.scalar_type() == at::kChar && v_cache.scalar_type() == at::kChar,
              "fni8 kv_write_paged: k_cache/v_cache must be int8");
  TORCH_CHECK(k_new.dim() == 3 && v_new.dim() == 3,
              "fni8 kv_write_paged: k_new/v_new must be [B,H_kv,D]");
  TORCH_CHECK(k_new.is_contiguous() && v_new.is_contiguous() && slot_mapping.is_contiguous() &&
                  k_cache.is_contiguous() && v_cache.is_contiguous(),
              "fni8 kv_write_paged: inputs must be contiguous");

  const auto B = k_new.size(0), H_KV = k_new.size(1), D = k_new.size(2);
  TORCH_CHECK(v_new.size(0) == B && v_new.size(1) == H_KV && v_new.size(2) == D,
              "fni8 kv_write_paged: k_new/v_new shape mismatch");
  TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.size(0) == B,
              "fni8 kv_write_paged: slot_mapping must be [B]");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(1) == H_KV && k_cache.size(3) == D,
              "fni8 kv_write_paged: k_cache must be [num_blocks,H_kv,block_size,D]");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes(),
              "fni8 kv_write_paged: v_cache shape must match k_cache");
  TORCH_CHECK(k_scale.is_contiguous() && v_scale.is_contiguous(),
              "fni8 kv_write_paged: k_scale/v_scale must be contiguous");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat && v_scale.scalar_type() == at::kFloat,
              "fni8 kv_write_paged: k_scale/v_scale must be float32");
  TORCH_CHECK(k_scale.numel() == k_cache.numel() / D && v_scale.numel() == v_cache.numel() / D,
              "fni8 kv_write_paged: scale shape mismatch vs cache");
  TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
              "fni8 kv_write_paged: head dim must be 32/64/128/256, got ", D);

  const int block_size = (int)k_cache.size(2);
  const at::cuda::CUDAGuard guard(k_new.device());
  const int64_t bh = B * H_KV;
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (D) {
    case 32:
      launch_kv_write<32>(k_new, v_new, slot_mapping, k_cache, k_scale, v_cache, v_scale, bh,
                          (int)H_KV, block_size, stream);
      break;
    case 64:
      launch_kv_write<64>(k_new, v_new, slot_mapping, k_cache, k_scale, v_cache, v_scale, bh,
                          (int)H_KV, block_size, stream);
      break;
    case 128:
      launch_kv_write<128>(k_new, v_new, slot_mapping, k_cache, k_scale, v_cache, v_scale, bh,
                           (int)H_KV, block_size, stream);
      break;
    case 256:
      launch_kv_write<256>(k_new, v_new, slot_mapping, k_cache, k_scale, v_cache, v_scale, bh,
                           (int)H_KV, block_size, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace fni8
