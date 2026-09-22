# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Reference (oracle) implementation for Multi-head Latent Attention (MLA,
DeepSeek-V2/V3) — Track-2 (issue #7): int8 dp4a MLA absorb-path decode.

Staged like the softmax-attention harness in `reference.py` and the Gated-
DeltaNet harness in `reference_linear_attn.py`:
  v1 (this file) — fp64/fp32 einsum-based math, no kernel, no quantization.
      Two algebraically-equivalent formulations are implemented and cross-
      checked against each other in `test_mla_absorb.py`:
        - `mla_decompress` — the "textbook" path: materialize per-head
          K_nope/V by decompressing the shared latent through W_UK/W_UV,
          then run ordinary per-head (MHA-shaped) attention. This is what
          prefill uses (compute-bound; materializing is cheap and lets a
          normal FA2-shaped kernel run).
        - `mla_absorb` + `absorb_qk_equiv` — the decode-time reformulation:
          fold W_UK into the query side (`q'_h = W_UK_hᵀ q_nope_h`) and
          W_UV·W_O offline, so the QK and PV matmuls run **MQA-style against
          the single shared d_c-dim latent** instead of per-head K/V. This is
          the shape the issue's dp4a kernel must reproduce: quantize the
          latent once per token, share it across heads.
  v2+ — the int8 dp4a kernel (quantized latent QK/PV, fp32/fp16 softmax +
      decoupled RoPE + RMSNorm per AGENTS.md) — not yet implemented; tracked
      in a follow-up `kernel`-labeled issue linked from #7 (touches csrc/,
      needs plan mode per CLAUDE.md).

Architecture (DeepSeek-V2 "DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model", section 2.1; DeepSeek-V3 uses the same
MLA design). Per token hidden state h_t in R^{d_model}:

    c_Q  = h_t @ W_DQ                      down-project (Q LoRA)
    c_Q  = RMSNorm(c_Q) * q_norm_w
    q_nope_h = c_Q @ W_UQ_h                per-head [d_h], h = 1..H
    q_rope_h = RoPE(c_Q @ W_QR_h)          per-head [d_r], decoupled RoPE

    c_KV = h_t @ W_DKV                     down-project (KV LoRA), shared
    c_KV = RMSNorm(c_KV) * kv_norm_w       across ALL heads -> this is the
                                            "latent" that gets cached/quantized
    k_rope = RoPE(h_t @ W_KR)              shared across heads (NOT per-head,
                                            NOT derived from c_KV)

  Decompress path (per head h):
    k_nope_h = c_KV @ W_UK_h               [N, d_h]
    v_h      = c_KV @ W_UV_h               [N, d_v]
    k_h      = concat(k_nope_h, k_rope)    [N, d_h + d_r]
    q_h      = concat(q_nope_h, q_rope_h)  [T, d_h + d_r]
    scores_h = q_h @ k_hᵀ * scale
    o_h      = softmax(scores_h) @ v_h
    out      = concat_h(o_h) @ W_O

  Absorb path (algebraically identical, decode-optimized): since
  q_nope_h @ k_nope_hᵀ = q_nope_h @ (c_KV @ W_UK_h)ᵀ
                        = (q_nope_h @ W_UK_hᵀ) @ c_KVᵀ,
  define q'_h = q_nope_h @ W_UK_hᵀ (`absorb_qk_equiv`) and score against the
  SHARED c_KV directly (MQA over the latent — no per-head K ever
  materialized). Symmetrically, o_h = softmax(...) @ v_h = (P_h @ c_KV) @
  W_UV_h, and out = concat_h(o_h) @ W_O = sum_h (P_h @ c_KV) @ (W_UV_h @
  W_O_h) — so W_UV and W_O fold into one [d_c, d_model] matrix per head,
  offline, and V is likewise never materialized.

NOTE ON THE ORACLE: issue #7 names `superl8serve/layers/mla_attn.py::MLAAttention`
(decompress path) + `absorb_qk_equiv` (in the sibling `fni8-serve` repo) as
the oracle the eventual kernel must match. That file is not reachable from
this repo's checkout/sandbox (same situation as issue #6's Gated-DeltaNet
oracle) — this is the standard published DeepSeek-V2/V3 MLA formulation
rather than a byte-for-byte port. Before wiring the fni8-serve integration,
diff the two formulations (weight layout, RMSNorm placement, RoPE convention
— this file uses the common "rotate_half"/NeoX-style RoPE; DeepSeek's actual
implementation should be checked token-for-token) — do not assume they agree
without that check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class MLAWeights:
    """All MLA projection weights, fp32/fp64. Shapes use:
    d_model, d_q_lora, d_c (kv_lora_rank), H (n_heads), d_h (qk_nope_head_dim),
    d_r (qk_rope_head_dim), d_v (v_head_dim).
    """

    w_dq: torch.Tensor  # [d_model, d_q_lora]
    q_norm_w: torch.Tensor  # [d_q_lora]
    w_uq: torch.Tensor  # [d_q_lora, H*d_h]
    w_qr: torch.Tensor  # [d_q_lora, H*d_r]
    w_dkv: torch.Tensor  # [d_model, d_c]
    kv_norm_w: torch.Tensor  # [d_c]
    w_uk: torch.Tensor  # [H, d_c, d_h]
    w_kr: torch.Tensor  # [d_model, d_r]  (shared across heads)
    w_uv: torch.Tensor  # [H, d_c, d_v]
    w_o: torch.Tensor  # [H*d_v, d_model]

    @property
    def h(self) -> int:
        return self.w_uk.shape[0]

    @property
    def d_h(self) -> int:
        return self.w_uk.shape[2]

    @property
    def d_r(self) -> int:
        return self.w_kr.shape[1]

    @property
    def d_v(self) -> int:
        return self.w_uv.shape[2]

    @property
    def d_c(self) -> int:
        return self.w_uk.shape[1]


def random_mla_weights(
    d_model: int, d_q_lora: int, d_c: int, h: int, d_h: int, d_r: int, d_v: int,
    *, device=None, dtype=torch.float64, generator=None,
) -> MLAWeights:
    """Small random weight set for tests. fp64 by default (tight equivalence checks)."""

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    return MLAWeights(
        w_dq=randn(d_model, d_q_lora) * 0.05,
        q_norm_w=torch.ones(d_q_lora, device=device, dtype=dtype),
        w_uq=randn(d_q_lora, h * d_h) * 0.05,
        w_qr=randn(d_q_lora, h * d_r) * 0.05,
        w_dkv=randn(d_model, d_c) * 0.05,
        kv_norm_w=torch.ones(d_c, device=device, dtype=dtype),
        w_uk=randn(h, d_c, d_h) * 0.05,
        w_kr=randn(d_model, d_r) * 0.05,
        w_uv=randn(h, d_c, d_v) * 0.05,
        w_o=randn(h * d_v, d_model) * 0.05,
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard RMSNorm, computed in the input's own precision."""
    var = x.pow(2).mean(dim=-1, keepdim=True)
    xn = x * torch.rsqrt(var + eps)
    return xn * weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def precompute_rope(
    seq_len: int, dim: int, *, base: float = 10000.0, offset: int = 0,
    device=None, dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for positions [offset, offset+seq_len), shape [seq_len, dim]."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(offset, offset + seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # [T, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, dim]
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [..., T, dim]; cos, sin: [T, dim] (broadcasts over leading dims)."""
    return x * cos + rotate_half(x) * sin


def project_q(
    h_t: torch.Tensor, w: MLAWeights, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """h_t: [B,T,d_model] -> (q_nope [B,H,T,d_h], q_rope [B,H,T,d_r])."""
    b, t, _ = h_t.shape
    h, d_h, d_r = w.h, w.d_h, w.d_r
    c_q = rms_norm(h_t @ w.w_dq, w.q_norm_w)
    q_nope = (c_q @ w.w_uq).view(b, t, h, d_h).transpose(1, 2)
    q_rope = (c_q @ w.w_qr).view(b, t, h, d_r).transpose(1, 2)
    q_rope = apply_rope(q_rope, rope_cos, rope_sin)
    return q_nope, q_rope


def project_kv_latent(
    h_t: torch.Tensor, w: MLAWeights, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """h_t: [B,N,d_model] -> (c_kv_n [B,N,d_c] normalized latent, k_rope [B,N,d_r]).

    Both outputs are exactly what gets cached/quantized for decode: the
    latent is shared across ALL heads (MQA-shaped), and the decoupled RoPE
    key is likewise a single per-token vector, not per-head.
    """
    c_kv = h_t @ w.w_dkv
    c_kv_n = rms_norm(c_kv, w.kv_norm_w)
    k_rope = apply_rope(h_t @ w.w_kr, rope_cos, rope_sin)
    return c_kv_n, k_rope


def _causal_mask(t_q: int, n_k: int, offset: int, device, dtype) -> torch.Tensor:
    """Query t (0-indexed, absolute position = offset+t) may attend key n iff n <= offset+t."""
    row = torch.arange(t_q, device=device).view(-1, 1) + offset
    col = torch.arange(n_k, device=device).view(1, -1)
    keep = col <= row
    return torch.zeros(t_q, n_k, device=device, dtype=dtype).masked_fill(~keep, float("-inf"))


def mla_decompress(
    h_q: torch.Tensor, h_kv: torch.Tensor, w: MLAWeights,
    *, causal: bool = True, kv_offset: int = 0,
) -> torch.Tensor:
    """The textbook / prefill path: decompress per-head K_nope, V and run
    ordinary MHA. h_q: [B,Tq,d_model] queries; h_kv: [B,N,d_model] the tokens
    whose K/V are attended (== h_q for self-attention prefill; the cache's
    source tokens for decode). ``kv_offset`` is h_kv's absolute start
    position (for RoPE and the causal mask) — h_q is assumed to occupy
    absolute positions [kv_offset + N - Tq, kv_offset + N) i.e. its own RoPE
    offset is inferred so the last query aligns with the last key (decode-
    step convention). Returns out [B,Tq,d_model].
    """
    b, t_q, _ = h_q.shape
    n_k = h_kv.shape[1]
    q_offset = kv_offset + n_k - t_q
    dtype, device = h_q.dtype, h_q.device
    d_h, d_r = w.d_h, w.d_r

    q_cos, q_sin = precompute_rope(t_q, d_r, offset=q_offset, device=device, dtype=dtype)
    k_cos, k_sin = precompute_rope(n_k, d_r, offset=kv_offset, device=device, dtype=dtype)
    q_nope, q_rope = project_q(h_q, w, q_cos, q_sin)
    c_kv_n, k_rope = project_kv_latent(h_kv, w, k_cos, k_sin)

    k_nope = torch.einsum("bnc,hcd->bhnd", c_kv_n, w.w_uk)  # [B,H,N,d_h]
    v = torch.einsum("bnc,hcd->bhnd", c_kv_n, w.w_uv)  # [B,H,N,d_v]
    k_rope_b = k_rope.unsqueeze(1).expand(-1, w.h, -1, -1)  # shared across heads

    q = torch.cat([q_nope, q_rope], dim=-1)  # [B,H,Tq,d_h+d_r]
    k = torch.cat([k_nope, k_rope_b], dim=-1)  # [B,H,N,d_h+d_r]
    scale = 1.0 / math.sqrt(d_h + d_r)
    scores = torch.einsum("bhtd,bhnd->bhtn", q, k) * scale
    if causal:
        scores = scores + _causal_mask(t_q, n_k, q_offset - kv_offset, device, dtype)
    p = torch.softmax(scores, dim=-1)
    o = torch.einsum("bhtn,bhnd->bhtd", p, v)  # [B,H,Tq,d_v]
    o = o.transpose(1, 2).reshape(b, t_q, w.h * w.d_v)
    return o @ w.w_o


def absorb_qk_equiv(w: MLAWeights) -> torch.Tensor:
    """Fold W_UK into the query side for the absorb path.

    q_nope_h @ k_nope_hᵀ = q_nope_h @ (c_KV @ W_UK_h)ᵀ = (q_nope_h @ W_UK_hᵀ) @ c_KVᵀ,
    so define q'_h = q_nope_h @ W_UK_hᵀ. Returns [H, d_h, d_c] such that
    ``q_abs = einsum('bhtd,hdc->bhtc', q_nope, absorb_qk_equiv(w))``.
    """
    return w.w_uk.transpose(-1, -2).contiguous()  # [H, d_h, d_c]


def absorb_ov_equiv(w: MLAWeights) -> torch.Tensor:
    """Fold W_UV and W_O into one [H, d_c, d_model] matrix for the absorb
    path's output side (V is never materialized).

    o_h = (P_h @ c_KV) @ W_UV_h; out = sum_h o_h @ W_O_h = sum_h (P_h @ c_KV)
    @ (W_UV_h @ W_O_h). Returns W_UV_h @ W_O_h per head.
    """
    h, d_v = w.h, w.d_v
    w_o_h = w.w_o.view(h, d_v, -1)  # [H, d_v, d_model]
    return torch.einsum("hcv,hvm->hcm", w.w_uv, w_o_h)  # [H, d_c, d_model]


def mla_absorb(
    h_q: torch.Tensor, c_kv_n_cache: torch.Tensor, k_rope_cache: torch.Tensor, w: MLAWeights,
    *, causal: bool = True, kv_offset: int = 0,
) -> torch.Tensor:
    """The decode-optimized absorb path: MQA against the shared latent cache,
    no per-head K/V ever materialized.

    h_q: [B,Tq,d_model] queries. ``c_kv_n_cache`` [B,N,d_c] and
    ``k_rope_cache`` [B,N,d_r] are the PRE-COMPUTED (already RMSNorm'd /
    RoPE'd) cached latent + decoupled-RoPE key for the N attended tokens —
    exactly what :func:`project_kv_latent` produces and what a real cache
    would store/quantize. ``kv_offset`` is the cache's absolute start
    position; h_q is assumed to occupy the last ``Tq`` positions of
    ``[kv_offset, kv_offset+N)`` (see :func:`mla_decompress`). Returns
    out [B,Tq,d_model], algebraically identical to :func:`mla_decompress`
    given the same cache contents.
    """
    b, t_q, _ = h_q.shape
    n_k = c_kv_n_cache.shape[1]
    q_offset = kv_offset + n_k - t_q
    dtype, device = h_q.dtype, h_q.device
    d_h, d_r = w.d_h, w.d_r

    q_cos, q_sin = precompute_rope(t_q, d_r, offset=q_offset, device=device, dtype=dtype)
    q_nope, q_rope = project_q(h_q, w, q_cos, q_sin)

    w_qabs = absorb_qk_equiv(w)  # [H, d_h, d_c]
    q_abs = torch.einsum("bhtd,hdc->bhtc", q_nope, w_qabs)  # [B,H,Tq,d_c] == q'_h

    scores_c = torch.einsum("bhtc,bnc->bhtn", q_abs, c_kv_n_cache)  # shared latent (MQA)
    scores_r = torch.einsum("bhtd,bnd->bhtn", q_rope, k_rope_cache)  # shared rope key
    scale = 1.0 / math.sqrt(d_h + d_r)
    scores = (scores_c + scores_r) * scale
    if causal:
        scores = scores + _causal_mask(t_q, n_k, q_offset - kv_offset, device, dtype)
    p = torch.softmax(scores, dim=-1)

    o_abs = torch.einsum("bhtn,bnc->bhtc", p, c_kv_n_cache)  # [B,H,Tq,d_c], still MQA-shared
    w_ovabs = absorb_ov_equiv(w)  # [H, d_c, d_model]
    return torch.einsum("bhtc,hcm->btm", o_abs, w_ovabs)
