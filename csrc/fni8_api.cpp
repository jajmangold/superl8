// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Python binding surface for the fni8 CUDA extension (`fni8._C`).
// As ops stabilize, promote them to a TORCH_LIBRARY schema for torch.compile.
#include <torch/extension.h>
#include <tuple>

namespace fni8 {
at::Tensor hello_add(const at::Tensor& a, const at::Tensor& b);
at::Tensor attn_int8_fwd(const at::Tensor& q, const at::Tensor& q_scale,
                         const at::Tensor& k, const at::Tensor& k_scale,
                         const at::Tensor& v, bool causal, int64_t window_left);
std::tuple<at::Tensor, at::Tensor> attn_int8_fwd_train(const at::Tensor& q,
                                                       const at::Tensor& q_scale,
                                                       const at::Tensor& k,
                                                       const at::Tensor& k_scale,
                                                       const at::Tensor& v, bool causal);
at::Tensor attn_fp16_fwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                         bool causal, c10::optional<double> scale, int64_t window_left,
                         const c10::optional<at::Tensor>& mask);
std::tuple<at::Tensor, at::Tensor> attn_fp16_fwd_train(const at::Tensor& q,
                                                       const at::Tensor& k,
                                                       const at::Tensor& v, bool causal,
                                                       c10::optional<double> scale);
std::tuple<at::Tensor, at::Tensor, at::Tensor> attn_bwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& o,
    const at::Tensor& do_, const at::Tensor& lse, double scale, bool causal);
at::Tensor attn_w8a8_fwd(const at::Tensor& q, const at::Tensor& q_scale,
                         const at::Tensor& k, const at::Tensor& k_scale,
                         const at::Tensor& v, const at::Tensor& v_scale, bool causal,
                         int64_t causal_diag, bool per_warp_quant);
at::Tensor attn_int8_decode(const at::Tensor& q, const at::Tensor& q_scale,
                            const at::Tensor& k, const at::Tensor& k_scale,
                            const at::Tensor& v, int64_t num_splits = -1);
at::Tensor attn_int8_decode_kv8(const at::Tensor& q, const at::Tensor& q_scale,
                                const at::Tensor& k, const at::Tensor& k_scale,
                                const at::Tensor& v, const at::Tensor& v_scale,
                                int64_t num_splits = -1);
at::Tensor attn_int4v_decode(const at::Tensor& q, const at::Tensor& q_scale,
                             const at::Tensor& k, const at::Tensor& k_scale,
                             const at::Tensor& v, const at::Tensor& v_scale,
                             int64_t num_splits = -1);
at::Tensor attn_int8_verify_split(const at::Tensor& q, const at::Tensor& q_scale,
                                  const at::Tensor& k, const at::Tensor& k_scale,
                                  const at::Tensor& v, const at::Tensor& v_scale,
                                  int64_t causal_diag, int64_t num_splits = -1);
at::Tensor attn_int8_varlen(const at::Tensor& q, const at::Tensor& q_scale,
                            const at::Tensor& k, const at::Tensor& k_scale,
                            const at::Tensor& v, const at::Tensor& cu_seqlens_q,
                            const at::Tensor& cu_seqlens_k, int64_t max_seqlen_q, bool causal);
at::Tensor attn_int8_tree(const at::Tensor& q, const at::Tensor& q_scale,
                          const at::Tensor& k, const at::Tensor& k_scale,
                          const at::Tensor& v, const at::Tensor& tree_mask);
at::Tensor gemm_w8a8(const at::Tensor& x, const at::Tensor& x_scale,
                     const at::Tensor& w, const at::Tensor& w_scale,
                     at::ScalarType out_dtype);
at::Tensor gemm_w4a8(const at::Tensor& x, const at::Tensor& x_scale,
                     const at::Tensor& w, const at::Tensor& w_scale, int64_t group_size,
                     at::ScalarType out_dtype);
at::Tensor gemm_q4k(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_q5k(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_q6k(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_decode_q4k(const at::Tensor& x, const at::Tensor& x_scale,
                           const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_decode_q5k(const at::Tensor& x, const at::Tensor& x_scale,
                           const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_decode_q6k(const at::Tensor& x, const at::Tensor& x_scale,
                           const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_q3k(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_iq3s(const at::Tensor& x, const at::Tensor& x_scale,
                     const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_iq3xxs(const at::Tensor& x, const at::Tensor& x_scale,
                       const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_iq4xs(const at::Tensor& x, const at::Tensor& x_scale,
                      const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_iq2s(const at::Tensor& x, const at::Tensor& x_scale,
                     const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_iq2xs(const at::Tensor& x, const at::Tensor& x_scale,
                      const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_iq2xxs(const at::Tensor& x, const at::Tensor& x_scale,
                       const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_iq1s(const at::Tensor& x, const at::Tensor& x_scale,
                     const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_q2k(const at::Tensor& x, const at::Tensor& x_scale,
                    const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_decode_q3k(const at::Tensor& x, const at::Tensor& x_scale,
                           const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gather_q3k(const at::Tensor& ids, const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_decode_iq3s(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_decode_iq4xs(const at::Tensor& x, const at::Tensor& x_scale,
                             const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_decode_iq3xxs(const at::Tensor& x, const at::Tensor& x_scale,
                              const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_decode_iq2s(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_decode_iq2xs(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_decode_iq2xxs(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_decode_iq1s(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_decode_q2k(const at::Tensor& x, const at::Tensor& x_scale,
                           const at::Tensor& w, at::ScalarType out_dtype);
at::Tensor gemm_tq34s(const at::Tensor& x, const at::Tensor& w, int64_t in_features,
                      at::ScalarType out_dtype);
at::Tensor gemm_tq34s_common_scale(const at::Tensor& x, const at::Tensor& w,
                                   int64_t in_features, at::ScalarType out_dtype);
at::Tensor gemm_tq34s_common_scale_tm4tn4(const at::Tensor& x, const at::Tensor& w,
                                          int64_t in_features,
                                          at::ScalarType out_dtype);
at::Tensor gemm_decode_tq34s(const at::Tensor& x, const at::Tensor& w,
                             at::ScalarType out_dtype);
at::Tensor gemm_decode_w8a8(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, const at::Tensor& w_scale,
                            at::ScalarType out_dtype);
std::tuple<at::Tensor, at::Tensor> gemm_decode_w8a8_requant(const at::Tensor& x,
                                                             const at::Tensor& x_scale,
                                                             const at::Tensor& w,
                                                             const at::Tensor& w_scale);
at::Tensor gemm_decode_w8a8_fp16in(const at::Tensor& x, const at::Tensor& w,
                                   const at::Tensor& w_scale, at::ScalarType out_dtype);
at::Tensor gemm_decode_w4a8(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, const at::Tensor& w_scale,
                            int64_t group_size, at::ScalarType out_dtype);
at::Tensor gemm_decode_w3a8(const at::Tensor& x, const at::Tensor& x_scale,
                            const at::Tensor& w, const at::Tensor& w_scale,
                            int64_t group_size, at::ScalarType out_dtype);
at::Tensor gemm_grouped_w8a8(const at::Tensor& x, const at::Tensor& x_scale,
                             const at::Tensor& w, const at::Tensor& w_scale,
                             const at::Tensor& group_sizes);
at::Tensor attn_paged_decode(const at::Tensor& q, const at::Tensor& q_scale,
                             const at::Tensor& k_cache, const at::Tensor& k_scale,
                             const at::Tensor& v_cache, const at::Tensor& v_scale,
                             const at::Tensor& block_table, const at::Tensor& context_lens,
                             int64_t block_size, int64_t max_context_len,
                             int64_t num_splits = -1);
at::Tensor attn_paged_decode_k8v3(const at::Tensor& q, const at::Tensor& q_scale,
                                  const at::Tensor& k_cache, const at::Tensor& k_scale,
                                  const at::Tensor& v_packed, const at::Tensor& v_norm,
                                  const at::Tensor& v_codebook,
                                  const at::Tensor& block_table,
                                  const at::Tensor& context_lens,
                                  int64_t block_size, int64_t max_context_len,
                                  int64_t num_splits = -1);
void kv_write_paged(const at::Tensor& k_new, const at::Tensor& v_new,
                    const at::Tensor& slot_mapping, at::Tensor& k_cache,
                    at::Tensor& k_scale, at::Tensor& v_cache, at::Tensor& v_scale);
std::tuple<at::Tensor, at::Tensor> lightning_attn_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& initial_state);
std::tuple<at::Tensor, at::Tensor> lightning_attn_int8_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& initial_state);
std::tuple<at::Tensor, at::Tensor> deltanet_recurrent_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& alpha,
    const at::Tensor& beta, const c10::optional<at::Tensor>& initial_state);
std::tuple<at::Tensor, at::Tensor> deltanet_recurrent_decode(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& alpha,
    const at::Tensor& beta, const c10::optional<at::Tensor>& initial_state);
std::tuple<at::Tensor, at::Tensor> deltanet_fused_decode(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& dt,
    const at::Tensor& b_logit, const at::Tensor& a_log, const at::Tensor& dt_bias,
    const at::Tensor& gain, const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& initial_state, double q_scale, double eps);
std::tuple<at::Tensor, at::Tensor> causal_conv1d_silu_decode(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& tail);
at::Tensor gated_rmsnorm_decode(const at::Tensor& o, const at::Tensor& gain,
                                const c10::optional<at::Tensor>& z, double eps);
std::tuple<at::Tensor, at::Tensor> deltanet_chunk_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& initial_state);
std::tuple<at::Tensor, at::Tensor> deltanet_gated_chunk_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& alpha,
    const at::Tensor& beta, const c10::optional<at::Tensor>& initial_state);
std::tuple<at::Tensor, at::Tensor> deltanet_gated_chunk_h2_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& alpha,
    const at::Tensor& beta, const c10::optional<at::Tensor>& initial_state);
std::tuple<at::Tensor, at::Tensor> deltanet_chunk_int8_fwd(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& initial_state);
at::Tensor mla_decode(const at::Tensor& q_abs, const at::Tensor& q_rope,
                      const at::Tensor& c_kv_cache, const at::Tensor& k_rope_cache,
                      double scale);
at::Tensor mla_decode_fp16(const at::Tensor& q_abs, const at::Tensor& q_rope,
                            const at::Tensor& c_kv_cache, const at::Tensor& k_rope_cache,
                            double scale);
at::Tensor mla_decode_int8(const at::Tensor& q_abs, const at::Tensor& q_rope,
                           const at::Tensor& c_kv_cache, const at::Tensor& k_rope_cache,
                           double scale);
std::tuple<at::Tensor, at::Tensor> quantize_i8_rowwise(at::Tensor x);
std::tuple<at::Tensor, at::Tensor> rmsnorm(at::Tensor x, at::Tensor weight, double eps,
                                           c10::optional<at::Tensor> residual,
                                           bool unit_offset);
std::tuple<at::Tensor, at::Tensor> rope(at::Tensor positions, at::Tensor q, at::Tensor k,
                                        at::Tensor cos, at::Tensor sin, int64_t rotary_dim);
at::Tensor act_and_mul(at::Tensor x, const std::string& kind);
at::Tensor dit_block(at::Tensor x, at::Tensor rms_weight, at::Tensor scale,
                     at::Tensor shift, double eps,
                     c10::optional<at::Tensor> gate,
                     c10::optional<at::Tensor> positions,
                     c10::optional<at::Tensor> cos,
                     c10::optional<at::Tensor> sin,
                     int64_t rotary_dim);
}  // namespace fni8

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_q3k", &fni8::gather_q3k,
        "Native GGUF Q3_K ROW-GATHER dequant (token embeddings): ids[...] "
        "int32/int64 index the SAME native Q3_K super-block bytes gemm_q3k takes "
        "(uint8 [N,(K/256)*110]) -> [n_ids,K] out_dtype, row i = the full dequant "
        "of w[ids[i]]. An embedding is a pure gather (never a GEMM), so the win is "
        "keeping the table in native bytes instead of a ~2.3x per_row_i8 expansion. "
        "An id outside [0,N) yields a zero row",
        py::arg("ids"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.doc() = "fni8: W8A8 DP4A FlashAttention-2 for Volta (sm_70)";
  m.def("hello_add", &fni8::hello_add, "Elementwise fp16 add (toolchain smoke op)",
        py::arg("a"), py::arg("b"));
  m.def("attn_int8_fwd", &fni8::attn_int8_fwd,
        "int8 dp4a QK^T FlashAttention-2 forward (fp16/bf16 PV; out dtype "
        "follows v)", py::arg("q"), py::arg("q_scale"), py::arg("k"),
        py::arg("k_scale"), py::arg("v"), py::arg("causal") = false,
        py::arg("window_left") = -1);
  m.def("attn_int8_fwd_train", &fni8::attn_int8_fwd_train,
        "int8 QK^T forward returning (out, lse) for the backward pass",
        py::arg("q"), py::arg("q_scale"), py::arg("k"), py::arg("k_scale"),
        py::arg("v"), py::arg("causal") = false);
  m.def("attn_fp16_fwd", &fni8::attn_fp16_fwd,
        "tiled fp16/bf16 half2 FlashAttention-2 prefill forward (sm_70): "
        "half2 CUDA-core QK^T, fp32 online-softmax, fp16/bf16 PV; O(N) memory "
        "(streams K/V) where torch SDPA materializes O(N^2) and OOMs. q/k/v one "
        "dtype (fp16 or bf16), out follows it; scale defaults to 1/sqrt(D); "
        "window_left>=0 = Mistral sliding window (causal only); mask = optional "
        "additive [B|1,H|1,M,N] natural-log bias (broadcast, indexed in place -> "
        "no O(N^2) expansion), same dtype as q/k/v",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false,
        py::arg("scale") = c10::nullopt, py::arg("window_left") = -1,
        py::arg("mask") = c10::nullopt);
  m.def("attn_fp16_fwd_train", &fni8::attn_fp16_fwd_train,
        "fp16/bf16 half2 QK^T forward returning (out, lse) for the backward pass",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false,
        py::arg("scale") = c10::nullopt);
  m.def("attn_w8a8_fwd", &fni8::attn_w8a8_fwd,
        "full-W8A8 dp4a FlashAttention-2 forward (int8 QK + int8 PV); causal_diag "
        "aligns the causal diagonal to the sequence end (spec-decode/MTP verify); "
        "per_warp_quant enables finer per-lane P requantization (SageAttention2)",
        py::arg("q"), py::arg("q_scale"), py::arg("k"), py::arg("k_scale"), py::arg("v"),
        py::arg("v_scale"), py::arg("causal") = false, py::arg("causal_diag") = 0,
        py::arg("per_warp_quant") = true);
  m.def("attn_bwd", &fni8::attn_bwd,
        "fused FA2 backward -> (dq, dk, dv)", py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("o"), py::arg("do_"), py::arg("lse"), py::arg("scale"),
        py::arg("causal") = false);
  m.def("attn_int8_decode", &fni8::attn_int8_decode,
        "decode (M=1) split-KV int8 dp4a attention (fp16 PV)", py::arg("q"),
        py::arg("q_scale"), py::arg("k"), py::arg("k_scale"), py::arg("v"),
        py::arg("num_splits") = -1);
  m.def("attn_int8_decode_kv8", &fni8::attn_int8_decode_kv8,
        "decode (M=1) split-KV with an INT8 KV cache (int8 K + int8 V)", py::arg("q"),
        py::arg("q_scale"), py::arg("k"), py::arg("k_scale"), py::arg("v"),
        py::arg("v_scale"), py::arg("num_splits") = -1);
  m.def("attn_int4v_decode", &fni8::attn_int4v_decode,
        "decode (M=1) with an int8-K / INT4-V cache (V packed 2 channels/byte)",
        py::arg("q"), py::arg("q_scale"), py::arg("k"), py::arg("k_scale"),
        py::arg("v"), py::arg("v_scale"), py::arg("num_splits") = -1);
  m.def("attn_int8_verify_split", &fni8::attn_int8_verify_split,
        "MTP/spec-decode VERIFY (M=k drafts) split-KV against an INT8 KV cache; "
        "end-aligned causal (draft i attends keys <= causal_diag+i)", py::arg("q"),
        py::arg("q_scale"), py::arg("k"), py::arg("k_scale"), py::arg("v"),
        py::arg("v_scale"), py::arg("causal_diag"), py::arg("num_splits") = -1);
  m.def("attn_int8_varlen", &fni8::attn_int8_varlen,
        "varlen (cu_seqlens-packed) int8 dp4a prefill forward (fp16 PV)", py::arg("q"),
        py::arg("q_scale"), py::arg("k"), py::arg("k_scale"), py::arg("v"),
        py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k"), py::arg("max_seqlen_q"),
        py::arg("causal") = false);
  m.def("attn_int8_tree", &fni8::attn_int8_tree,
        "tree-attention verify (EAGLE spec-decode): drafts attend prefix + tree "
        "ancestors via a custom [T,T] mask", py::arg("q"), py::arg("q_scale"),
        py::arg("k"), py::arg("k_scale"), py::arg("v"), py::arg("tree_mask"));
  m.def("quantize_i8_rowwise", &fni8::quantize_i8_rowwise,
        "fused per-row symmetric-RTN int8 activation quantizer: x[M,K] "
        "(fp16/bf16/fp32) -> (q int8 [M,K], scale fp32 [M]) in ONE launch, "
        "byte-identical to the eager quantize_int8_rowwise prologue (Q_MAX=127, "
        "rintf round-half-to-even). Feeds gemm_w8a8/gemm_decode_w8a8",
        py::arg("x"));
  m.def("rmsnorm", &fni8::rmsnorm,
        "fused RMSNorm: x[...,D], weight[D] (fp16/bf16) -> (normed, xr) in one "
        "launch. residual (optional) is added first and returned as xr=x+residual; "
        "unit_offset uses (1+w) (Gemma). fp32 reduction internally, never quantized",
        py::arg("x"), py::arg("weight"), py::arg("eps"),
        py::arg("residual") = c10::nullopt, py::arg("unit_offset") = false);
  m.def("rope", &fni8::rope,
        "fused RoPE: rotates q[...,Hq,D] and k[...,Hk,D] (fp16/bf16) IN PLACE using "
        "fp32 cos/sin[max_pos,rotary_dim] tables gathered by positions[...] (int64); "
        "dims >= rotary_dim pass through (partial rotary). Returns (q, k)",
        py::arg("positions"), py::arg("q"), py::arg("k"), py::arg("cos"), py::arg("sin"),
        py::arg("rotary_dim"));
  m.def("act_and_mul", &fni8::act_and_mul,
        "fused gated activation on a merged gate_up output: x[...,2I] -> [...,I] = "
        "act(gate)*up in one launch. kind: 'silu' (SwiGLU) or 'gelu_tanh' (Gemma "
        "GeGLU). Activation computed in fp32",
        py::arg("x"), py::arg("kind"));
  m.def("dit_block", &fni8::dit_block,
        "fused DiT post-attention block: RMSNorm (fp32) -> adaLN scale/shift/gate -> "
        "residual add -> RoPE, all in fp32 registers, bf16 storage, one HBM round-trip. "
        "x[M,D], rms_weight[D], scale[M,D], shift[M,D] (fp16/bf16). gate[M,D] (optional). "
        "positions[M] int64, cos/sin[max_pos,rotary_dim] fp32 (optional RoPE). "
        "Returns out[M,D]",
        py::arg("x"), py::arg("rms_weight"), py::arg("scale"), py::arg("shift"),
        py::arg("eps"), py::arg("gate") = c10::nullopt,
        py::arg("positions") = c10::nullopt, py::arg("cos") = c10::nullopt,
        py::arg("sin") = c10::nullopt, py::arg("rotary_dim") = int64_t(0));
  m.def("gemm_w8a8", &fni8::gemm_w8a8,
        "int8 dp4a GEMM: (x_i8[M,K] . w_i8[N,K]^T) * x_scale[m] * w_scale[n] -> "
        "[M,N] in out_dtype (float16 or bfloat16; float16 default) (linear layers: "
        "QKV/O projections, FFN, MoE experts)",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("w_scale"),
        py::arg("out_dtype") = at::kHalf);
  m.def("gemm_w4a8", &fni8::gemm_w4a8,
        "W4A8 int8 dp4a GEMM: 4-bit weights (packed 2 nibbles/byte, per-group fp32 "
        "scale) unpacked to int8 for dp4a -> [M,N] in out_dtype (float16 or bfloat16; "
        "float16 default). 2x weight footprint/bandwidth",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("w_scale"),
        py::arg("group_size"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_q4k", &fni8::gemm_q4k,
        "Fused GGUF Q4_K dp4a GEMM (native-GGUF-on-the-fly): weights stay resident "
        "in native Q4_K super-blocks (uint8 [N,(K/256)*144]); the kernel unpacks each "
        "32-elem sub-block to int8 + dp4a's it, honoring the 6-bit per-sub-block "
        "scale/min exactly -> [M,N] in out_dtype (float16 or bfloat16; float16 "
        "default). Batched tiled GEMM: serves both prefill (M>1) and decode (M=1)",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_q5k", &fni8::gemm_q5k,
        "Fused GGUF Q5_K dp4a GEMM (native): 5-bit weights (Q4_K affine + a 5th bit "
        "from qh) unpacked to int8 in-kernel, native 6-bit sub-scales/mins honored -> "
        "[M,N] out_dtype. See gemm_q4k for the shared spine",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_q6k", &fni8::gemm_q6k,
        "Fused GGUF Q6_K dp4a GEMM (native): SYMMETRIC 6-bit weights (centered -32, "
        "signed int8 per-16 scales, no min term) unpacked to int8 in-kernel -> [M,N] "
        "out_dtype. The outlier-tensor precision in a Q4_K_M mix (e.g. LTX to_v)",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_q4k", &fni8::gemm_decode_q4k,
        "Decode-specialized (warp-per-column MMVQ) fused Q4_K dp4a GEMV for M<=16: "
        "saturates the GPU at M=1 where the tile gemm_q4k under-fills it. Same "
        "per-sub-block math -> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_q5k", &fni8::gemm_decode_q5k,
        "Decode-specialized (warp-per-column) fused Q5_K dp4a GEMV for M<=16. Same "
        "math as tile gemm_q5k -> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_q6k", &fni8::gemm_decode_q6k,
        "Decode-specialized (warp-per-column) fused Q6_K dp4a GEMV for M<=16. Same "
        "math as tile gemm_q6k -> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_q3k", &fni8::gemm_q3k,
        "Fused GGUF Q3_K dp4a GEMM (native): SYMMETRIC 3-bit weights (centered -4, "
        "signed 6-bit per-16 scales) unpacked to int8 in-kernel -> [M,N] out_dtype. "
        "THE type for Qwen3.6-27B-Q3_K_S single-card residency",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_iq3s", &fni8::gemm_iq3s,
        "Fused GGUF IQ3_S dp4a tile GEMM (native, PREFILL): same 512-entry grid + "
        "sign-bit + per-32 (1+2s) scale math as gemm_decode_iq3s, blocked for M>16 "
        "-> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_iq3xxs", &fni8::gemm_iq3xxs,
        "Fused GGUF IQ3_XXS dp4a tile GEMM (native, PREFILL): 256-entry grid, "
        "ksigns_iq2xs sign LUT and the half-offset d*(0.5+s)*0.5 scale, blocked for "
        "M>16 -> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_iq4xs", &fni8::gemm_iq4xs,
        "Fused GGUF IQ4_XS dp4a tile GEMM (native, PREFILL): nibble-interleaved halves "
        "through kvalues_iq4nl with the split 6-bit scale, blocked for M>16 -> [M,N]",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_iq2s", &fni8::gemm_iq2s,
        "Fused GGUF IQ2_S dp4a tile GEMM (native, PREFILL): 1024-entry grid, 10-bit "
        "index (8 qs + 2 qh), EXPLICIT per-weight sign bits, 4-bit scale per SIXTEEN "
        "weights d*(0.5+s)*0.25, blocked for M>16 -> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_iq2xs", &fni8::gemm_iq2xs,
        "Fused GGUF IQ2_XS dp4a tile GEMM (native, PREFILL): 512-entry grid, per uint16 "
        "a 9-bit index plus a 7-bit ksigns_iq2xs index, 4-bit scale per SIXTEEN weights "
        "d*(0.5+s)*0.25, blocked for M>16 -> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_iq2xxs", &fni8::gemm_iq2xxs,
        "Fused GGUF IQ2_XXS dp4a tile GEMM (native, PREFILL): 256-entry grid, four "
        "8-bit indices and four 7-bit ksigns_iq2xs indices per 32 weights, top-nibble "
        "scale d*(0.5+s)*0.25, blocked for M>16 -> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_iq1s", &fni8::gemm_iq1s,
        "Fused GGUF IQ1_S dp4a tile GEMM (native, PREFILL): 2048-entry TERNARY grid, "
        "11-bit index (8 qs + 3 qh), no sign bits -- codes stored as 8*g so 8*(g+delta) "
        "is the integer 8g+-1, blocked for M>16 -> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_q2k", &fni8::gemm_q2k,
        "Fused GGUF Q2_K dp4a GEMM (native): affine 2-bit weights (4-bit per-16 "
        "scale+min) unpacked to int8 in-kernel -> [M,N] out_dtype (aggressive UD-Q2)",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_q3k", &fni8::gemm_decode_q3k,
        "Decode-specialized (warp-per-column) fused Q3_K dp4a GEMV for M<=16. Same "
        "math as tile gemm_q3k -> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_iq3s", &fni8::gemm_decode_iq3s,
        "IQ3_S decode dp4a (M<=16): warp-per-column MMVQ over ggml's 512-entry grid "
        "(9-bit index + per-weight sign bit, per-32 scale 1+2s), native 110-B blocks "
        "-> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype"));
  m.def("gemm_decode_iq4xs", &fni8::gemm_decode_iq4xs,
        "IQ4_XS decode dp4a (M<=16): warp-per-column MMVQ over the 16-entry signed "
        "kvalues_iq4nl codebook, split 6-bit per-32 scale d*(ls-32), nibble-interleaved "
        "halves, native 136-B blocks -> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_iq3xxs", &fni8::gemm_decode_iq3xxs,
        "IQ3_XXS decode dp4a (M<=16): warp-per-column MMVQ over the 256-entry grid, "
        "top-nibble half-offset scale d*(0.5+s)*0.5, sign bits via ksigns_iq2xs, "
        "native 98-B blocks -> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_iq2s", &fni8::gemm_decode_iq2s,
        "IQ2_S decode dp4a (M<=16): 1024-entry grid, 10-bit index (8 qs + 2 qh), EXPLICIT per-weight sign bits, 4-bit scale per SIXTEEN weights d*(0.5+s)*0.25 "
        "-> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_iq2xs", &fni8::gemm_decode_iq2xs,
        "IQ2_XS decode dp4a (M<=16): 512-entry grid, per uint16 a 9-bit index plus a 7-bit ksigns_iq2xs index; 4-bit scale per SIXTEEN weights d*(0.5+s)*0.25 "
        "-> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_iq2xxs", &fni8::gemm_decode_iq2xxs,
        "IQ2_XXS decode dp4a (M<=16): 256-entry grid, four 8-bit indices and four 7-bit ksigns_iq2xs indices per 32 weights, top-nibble scale d*(0.5+s)*0.25 "
        "-> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_iq1s", &fni8::gemm_decode_iq1s,
        "IQ1_S decode dp4a (M<=16): 2048-entry TERNARY grid, 11-bit index (8 qs + 3 qh), no sign bits -- codes stored as 8*g so 8*(g+delta) is the integer 8g+-1 "
        "-> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_q2k", &fni8::gemm_decode_q2k,
        "Decode-specialized (warp-per-column) fused Q2_K dp4a GEMV for M<=16. Same "
        "math as tile gemm_q2k -> [M,N] out_dtype",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_tq34s", &fni8::gemm_tq34s,
        "Fused GGUF TQ3_4S dp4a GEMM (native TurboQuant type 46): the kernel "
        "rotates the fp activation with the forward RHT per 32-block BEFORE the "
        "per-32 q8_1 quant, unpacks 3-bit codes to the corrected int8 centroid "
        "levels, and flushes each per-8 E3M5 scale -> [M,N] out_dtype. x[M,K] "
        "fp16/bf16 (K%32==0), w[N,(K/32)*16] uint8 native bytes. sm_70 dp4a",
        py::arg("x"), py::arg("w"), py::arg("in_features"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_tq34s_common_scale", &fni8::gemm_tq34s_common_scale,
        "Experimental llama.cpp-style TQ3_4S GEMM: requantize each 32-weight block "
        "to one shared Q8 scale before dp4a, then flush once per block",
        py::arg("x"), py::arg("w"), py::arg("in_features"),
        py::arg("out_dtype") = at::kHalf);
  m.def("gemm_tq34s_common_scale_tm4tn4", &fni8::gemm_tq34s_common_scale_tm4tn4,
        "Experimental 256-thread TQ3 common-scale GEMM with TM4xTN4 accumulators",
        py::arg("x"), py::arg("w"), py::arg("in_features"),
        py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_tq34s", &fni8::gemm_decode_tq34s,
        "Decode-specialized (warp-per-column MMVQ) fused TQ3_4S dp4a GEMV for "
        "M<=16: same math as the tile gemm_tq34s (fused RHT + per-32 quant in "
        "smem), GPU-saturating at M=1 -> [M,N] out_dtype",
        py::arg("x"), py::arg("w"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_w4a8", &fni8::gemm_decode_w4a8,
        "decode-specialized W4A8 dp4a GEMM: 4-bit weights (packed 2 nibbles/byte, "
        "per-group fp32 scale) at small M (<=16), one warp per output column so "
        "decode fills the GPU -> [M,N] out_dtype. Same int math as gemm_w4a8; use "
        "gemm_w4a8 for prefill-size M",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("w_scale"),
        py::arg("group_size"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_w3a8", &fni8::gemm_decode_w3a8,
        "decode-specialized W3A8 dp4a GEMM: uniform 3-bit weights (Q3_K-style "
        "bit-planes [N,(K/32)*3] int32, per-group fp32 scale) unpacked to int8 for "
        "dp4a at small M (<=16), one warp per output column -> [M,N] out_dtype. "
        "VRAM/context lever: 3.0 bpw, decode PARITY with W4A8 (not faster) — buys "
        "longer KV context / more batch slots before OOM (issue #181)",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("w_scale"),
        py::arg("group_size"), py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_w8a8_requant", &fni8::gemm_decode_w8a8_requant,
        "fused-requant decode GEMV (issue #118): same int8 dp4a as gemm_decode_w8a8, "
        "but the epilogue computes per-row absmax and requantizes to int8 in [−127,127] "
        "with a per-row fp32 out_scale — output feeds DIRECTLY into the next int8 op "
        "without a standalone quantize kernel or an fp16 HBM round-trip. "
        "x[M,K] int8, x_scale[M] fp32, w[N,K] int8, w_scale[N] fp32 -> "
        "(out_i8[M,N] int8, out_scale[M] fp32)",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("w_scale"));
  m.def("gemm_decode_w8a8", &fni8::gemm_decode_w8a8,
        "decode-specialized int8 dp4a GEMM (split-K, issue #27): small M (<=16) "
        "-> [M,N] in out_dtype (float16 or bfloat16; float16 default). Grid = "
        "(N, split_k), one warp per (output column, K-slice) so decode's small "
        "M/N still fills the GPU; use gemm_w8a8 for prefill-size M",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("w_scale"),
        py::arg("out_dtype") = at::kHalf);
  m.def("gemm_decode_w8a8_fp16in", &fni8::gemm_decode_w8a8_fp16in,
        "fused fp16-input decode GEMV (issue #130): quantizes the fp16 activation "
        "to int8 in the GEMV prologue (shared memory) -> no standalone quant kernel, "
        "no HBM round-trip of int8 x. x[M,K] fp16, w[N,K] int8, w_scale[N] fp32 -> "
        "[M,N] out_dtype. Fused path needs split_k==1 and M*K int8 within smem; "
        "caller falls back to quantize_int8_rowwise + gemm_decode_w8a8 otherwise",
        py::arg("x"), py::arg("w"), py::arg("w_scale"),
        py::arg("out_dtype") = at::kHalf);
  m.def("gemm_grouped_w8a8", &fni8::gemm_grouped_w8a8,
        "MoE grouped/batched int8 dp4a GEMM: every active expert's GEMM for a "
        "pre-sorted token batch (rows contiguous per expert) in ONE launch -> "
        "fp16 [total_M,N]. x[total_M,K] int8, x_scale[total_M] fp32, "
        "w[E,N,K] int8, w_scale[E,N] fp32, group_sizes[E] int64 (sums to total_M)",
        py::arg("x"), py::arg("x_scale"), py::arg("w"), py::arg("w_scale"),
        py::arg("group_sizes"));
  m.def("attn_paged_decode", &fni8::attn_paged_decode,
        "paged-KV decode (M=1): int8 K/V cache addressed via a per-sequence "
        "block table + context length, so one launch serves a batch of "
        "mixed-length sequences", py::arg("q"), py::arg("q_scale"),
        py::arg("k_cache"), py::arg("k_scale"), py::arg("v_cache"), py::arg("v_scale"),
        py::arg("block_table"), py::arg("context_lens"), py::arg("block_size"),
        py::arg("max_context_len"), py::arg("num_splits") = -1);
  m.def("attn_paged_decode_k8v3", &fni8::attn_paged_decode_k8v3,
        "paged-KV decode (M=1) fused for the K8V3 store (fni8#295): int8 "
        "rotated K + 3-bit Lloyd-Max packed V consumed in-kernel (no fp16 V "
        "materialization, no K re-quantization, one batched launch)",
        py::arg("q"), py::arg("q_scale"), py::arg("k_cache"), py::arg("k_scale"),
        py::arg("v_packed"), py::arg("v_norm"), py::arg("v_codebook"),
        py::arg("block_table"), py::arg("context_lens"), py::arg("block_size"),
        py::arg("max_context_len"), py::arg("num_splits") = -1);
  m.def("kv_write_paged", &fni8::kv_write_paged,
        "quantize-on-write: commit one new token's K/V into the paged int8 "
        "KV cache in-place (per-token RTN scale)", py::arg("k_new"), py::arg("v_new"),
        py::arg("slot_mapping"), py::arg("k_cache"), py::arg("k_scale"),
        py::arg("v_cache"), py::arg("v_scale"));
  m.def("lightning_attn_fwd", &fni8::lightning_attn_fwd,
        "Track-2 (issue #42) v1: naive sequential fp32 MiniMax Lightning "
        "(un-gated linear) attention recurrence, one CUDA block per "
        "(batch,head) -> (out, final_state). The on-device ground truth "
        "the int8 variant (next PR) is validated against",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("initial_state") = c10::nullopt);
  m.def("lightning_attn_int8_fwd", &fni8::lightning_attn_int8_fwd,
        "Track-2 (issue #42) v2: int8 dp4a chunked Lightning attention. "
        "Same un-gated linear-attention recurrence as v1, but intra-chunk "
        "k·q dot products use the __dp4a int8x4 CUDA-core intrinsic (sm_70 "
        "has no int8 tensor cores). Keys/queries quantized per-row symmetric "
        "RTN; state/V/output base stay fp32. -> (out, final_state). "
        "Matches fp32 oracle at int8 SQNR/cos tolerance (not allclose).",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("initial_state") = c10::nullopt);
  m.def("deltanet_recurrent_fwd", &fni8::deltanet_recurrent_fwd,
        "Track-2 (issue #6) v1: naive sequential fp32 Gated-DeltaNet recurrence, "
        "one CUDA block per (batch,head) -> (out, final_state). The on-device "
        "ground truth v2 (chunked)/v3 (gated)/v4 (int8 dp4a) are validated against",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("alpha"), py::arg("beta"),
        py::arg("initial_state") = c10::nullopt);
  m.def("deltanet_recurrent_decode", &fni8::deltanet_recurrent_decode,
        "Decode-specialized fp32 Gated-DeltaNet recurrence: one WARP per "
        "(batch,head,v-row) with state in registers (no shared memory, so no "
        "per-call cudaFuncSetAttribute -> CUDA-graph-capturable). Same math as "
        "deltanet_recurrent_fwd (validated against the same oracle); the fast "
        "L==1 decode path fni8serve replays inside its graphed decode. Dk,Dv<=128",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("alpha"), py::arg("beta"),
        py::arg("initial_state") = c10::nullopt);
  m.def("deltanet_fused_decode", &fni8::deltanet_fused_decode,
        "Fused Gated-DeltaNet decode: L2-norm(q,k) + GQA expand + sigmoid(beta) + "
        "g=-softplus(dt+dt_bias)*exp(A_log) + delta-rule recurrence + gated output "
        "RMSNorm (silu(z)*o) in ONE graph-capturable launch (block per (batch,"
        "value-head), fp32 register state, static smem -> no cudaFuncSetAttribute). "
        "Collapses the ~15-19 tiny glue ops fni8serve runs per linear-attn layer at "
        "decode. Adapted from qengine (Apache-2.0). Dk==Dv==128, T==1. "
        "-> (out[B,nv,1,Dv] post-norm, final_state[B,nv,Dv,Dk]).",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dt"), py::arg("b_logit"),
        py::arg("a_log"), py::arg("dt_bias"), py::arg("gain"), py::arg("z") = c10::nullopt,
        py::arg("initial_state") = c10::nullopt, py::arg("q_scale") = 1.0, py::arg("eps") = 1e-6);
  m.def("causal_conv1d_silu_decode", &fni8::causal_conv1d_silu_decode,
        "Fused causal depthwise conv1d(kernel=K) + SiLU for the L==1 decode "
        "token shift (vLLM causal_conv1d_update analogue). x[B,Wc], weight[Wc,K], "
        "tail[B,K-1,Wc] (previous raw window) -> (out[B,Wc]=silu(conv), "
        "new_tail[B,K-1,Wc]=rolled history). One thread per (batch,channel), fp32 "
        "math / dtype-follows-input store (fp16/bf16/fp32), no smem -> "
        "CUDA-graph-capturable. K<=8",
        py::arg("x"), py::arg("weight"), py::arg("tail"));
  m.def("gated_rmsnorm_decode", &fni8::gated_rmsnorm_decode,
        "Fused gated output RMSNorm decode (HF Qwen3_5RMSNormGated, norm BEFORE "
        "gate): per-head RMS over vd, x gain, x silu(z). o[...,vd] fp32, gain[vd] "
        "fp32, z (optional, same shape) fp32 -> out[...,vd] fp32. One warp per "
        "(batch,head) row, register state, no smem -> CUDA-graph-capturable. "
        "fp32 throughout (load-bearing, never quantized). vd<=128",
        py::arg("o"), py::arg("gain"), py::arg("z") = c10::nullopt, py::arg("eps"));
  m.def("deltanet_chunk_fwd", &fni8::deltanet_chunk_fwd,
        "Track-2 (issue #40) v2: ungated chunked WY/UT parallel DeltaNet (fp32). "
        "alpha=beta=1; L2-normalised k; chunk size C computed to fit shared mem. "
        "-> (out, final_state). Matches ungated_delta_rule_oracle at fp32 "
        "reassociation tolerance.",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("initial_state") = c10::nullopt);
  m.def("deltanet_gated_chunk_fwd", &fni8::deltanet_gated_chunk_fwd,
        "Track-2 (issue #56) v3: gated chunked WY/UT parallel DeltaNet (fp32) "
        "with log-space γ-cumprod decay stabilisation. Implements the full "
        "gated-delta-rule recurrence (α decay, β write-strength) via the WY/UT "
        "decomposition.  Matches gated_delta_rule_oracle at fp32 reassociation "
        "tolerance.",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("alpha"), py::arg("beta"),
        py::arg("initial_state") = c10::nullopt);
  m.def("deltanet_gated_chunk_h2_fwd", &fni8::deltanet_gated_chunk_h2_fwd,
        "Track-2 (issue #122) v3.5: half2 (FP16x2) CUDA-core gated chunked "
        "DeltaNet.  Recasts the WY/UT chunk-matmul inner dot products (S@k, "
        "k·k Gram, q·k) onto __hfma2 half2 packed math with fp32 accumulation. "
        "Everything numerically load-bearing (state, γ-cumprod, L2-norm, α/β "
        "gates, state update AXPY) stays fp32.  This is the healthy half2 "
        "CUDA-core pipe (~27 TFLOP/s), NOT the firmware-dead tensor cores. "
        "Inputs/outputs are fp32; half2 is internal only.  Matches "
        "gated_delta_rule_oracle at fp16 tolerance.",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("alpha"), py::arg("beta"),
        py::arg("initial_state") = c10::nullopt);
  m.def("deltanet_chunk_int8_fwd", &fni8::deltanet_chunk_int8_fwd,
        "Track-2 (issue #83) v4: int8 dp4a ungated chunked DeltaNet. Same chunked "
        "delta-rule algebra as v2, but the intra-chunk K-Gram (k·k) and Q·K (q·k) "
        "dot products are computed with the __dp4a int8×4 CUDA-core intrinsic "
        "(sm_70 has no int8 tensor cores). Keys L2-normalised before symmetric "
        "per-row int8 quant; state/V/residuals stay fp32. -> (out, final_state). "
        "Matches ungated_delta_rule_oracle at int8 SQNR/cos tolerance (not allclose).",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("initial_state") = c10::nullopt);
  m.def("mla_decode", &fni8::mla_decode,
        "MLA (DeepSeek-V2/V3) absorb-path decode (v1, fp32): MQA-style QK/softmax/PV "
        "against the shared d_c-dim latent cache, decoupled-RoPE score folded in",
        py::arg("q_abs"), py::arg("q_rope"), py::arg("c_kv_cache"), py::arg("k_rope_cache"),
        py::arg("scale"));
  m.def("mla_decode_fp16", &fni8::mla_decode_fp16,
        "MLA absorb-path decode (v2, fp16): same algorithm as v1 but fp16 cache I/O "
        "for half the HBM bandwidth; internal softmax accumulation stays fp32",
        py::arg("q_abs"), py::arg("q_rope"), py::arg("c_kv_cache"), py::arg("k_rope_cache"),
        py::arg("scale"));
  m.def("mla_decode_int8", &fni8::mla_decode_int8,
        "MLA absorb-path decode (v3, int8 dp4a): QK dot product via int8 dp4a; "
        "per-token C_KV quantized on-the-fly (symmetric per-row RTN, fp32 scales); "
        "softmax/PV accumulation stays fp32",
        py::arg("q_abs"), py::arg("q_rope"), py::arg("c_kv_cache"), py::arg("k_rope_cache"),
        py::arg("scale"));
}
