# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Reference attention implementations — the correctness authorities.

Three tiers (AGENTS.md numerics conventions):
  1. `attention_fp32_oracle` — naive einsum attention computed entirely in fp32
     (upcast inputs). The ground truth every kernel is measured against.
  2. `sdpa_fp16` — PyTorch SDPA in native fp16: the "what you'd get today"
     baseline whose error vs the oracle sets the RELATIVE tolerance bar
     (fwd <= 2x + 1e-5, bwd <= 3x + 1e-4, ai-bond style).
  3. `flash_attn_v100` (optional import) — external hand-written fp16 FA2 for
     sm_70; the perf/quality baseline our dp4a kernel is honestly compared to.

Layout convention everywhere: [B, H, M, D] (batch, heads, seq, head_dim).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def attention_fp32_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Naive attention, fully fp32, [B,H,M,D]. Slow and memory-hungry — small shapes only.

    Supports GQA/MQA: if k/v have fewer heads than q (H_q % H_kv == 0), each K/V
    head is shared by H_q/H_kv consecutive query heads (repeat_interleave).
    """
    qf, kf, vf = q.float(), k.float(), v.float()
    h_q, h_kv = qf.shape[1], kf.shape[1]
    if h_q != h_kv:
        assert h_q % h_kv == 0, "GQA requires H_q divisible by H_kv"
        rep = h_q // h_kv
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.einsum("bhmd,bhnd->bhmn", qf, kf) * scale
    if causal:
        m_len, n_len = s.shape[-2], s.shape[-1]
        mask = torch.ones(m_len, n_len, device=s.device, dtype=torch.bool).tril(n_len - m_len)
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", p, vf)


def attention_fp32_window_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window_left: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Causal sliding-window attention, fp32. Query i attends keys in
    (i - window_left, i] (Mistral-style local attention). GQA-aware."""
    qf, kf, vf = q.float(), k.float(), v.float()
    h_q, h_kv = qf.shape[1], kf.shape[1]
    if h_q != h_kv:
        assert h_q % h_kv == 0, "GQA requires H_q divisible by H_kv"
        rep = h_q // h_kv
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.einsum("bhmd,bhnd->bhmn", qf, kf) * scale
    m_len, n_len = s.shape[-2], s.shape[-1]
    row = torch.arange(m_len, device=s.device).view(-1, 1)
    col = torch.arange(n_len, device=s.device).view(1, -1)
    # causal + left window: 0 <= (row - col) < window_left (n aligned to m tail).
    # The lower endpoint is excluded: keys are in (position - window_left, position].
    diff = (row + (n_len - m_len)) - col
    mask = (diff >= 0) & (diff < window_left)
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", p, vf)


def sdpa_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """PyTorch SDPA in native fp16 — the error-bar-setting baseline."""
    assert q.dtype == torch.float16
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


def flash_v100_available() -> bool:
    try:
        import flash_attn_v100  # noqa: F401

        return True
    except ImportError:
        return False


def flash_v100_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """External ai-bond fp16 FA2 baseline. Input [B,H,M,D] -> its native [B,M,H,D] and back."""
    from flash_attn_v100 import flash_attn_func

    out = flash_attn_func(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        softmax_scale=scale,
        causal=causal,
    )
    return out.permute(0, 2, 1, 3)


def attention_fp32_varlen_oracle(
    q, k, v, cu_seqlens_q, cu_seqlens_k, *, causal: bool = False, scale=None
):
    """Varlen (cu_seqlens-packed) attention oracle, fp32.

    Packed layout [total_tokens, H, D] (token-major, heads interleaved) — the
    serving-framework convention. Each sequence b attends only within its own
    [cu_q[b]:cu_q[b+1]] queries x [cu_k[b]:cu_k[b+1]] keys. Loops sequences and
    reuses `attention_fp32_oracle` per sequence; GQA-aware. Returns [total_q,H,D].
    """
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    out = torch.empty_like(q, dtype=torch.float32)
    for b in range(len(cu_q) - 1):
        q_b = q[cu_q[b]:cu_q[b + 1]].permute(1, 0, 2).unsqueeze(0)   # [1,Hq,sq,D]
        k_b = k[cu_k[b]:cu_k[b + 1]].permute(1, 0, 2).unsqueeze(0)   # [1,Hkv,sk,D]
        v_b = v[cu_k[b]:cu_k[b + 1]].permute(1, 0, 2).unsqueeze(0)
        o_b = attention_fp32_oracle(q_b, k_b, v_b, causal=causal, scale=scale)
        out[cu_q[b]:cu_q[b + 1]] = o_b.squeeze(0).permute(1, 0, 2)   # back to [sq,Hq,D]
    return out


def attention_fp32_paged_oracle(q, k_dense, v_dense, context_lens, *, scale=None):
    """fp32 oracle for paged/ragged-batch decode. ``q`` [B,H_q,1,D]; ``k_dense``/
    ``v_dense`` [B,H_kv,N_max,D] hold each sequence b's tokens left-aligned in
    ``[0, context_lens[b])`` (the paged cache stores exactly these tokens, just
    addressed through a block table instead of a contiguous slice — the oracle
    only needs the logical tokens, not the physical layout). Loops sequences
    (each may have a different length) through :func:`attention_fp32_oracle`,
    which is GQA-aware. Returns [B,H_q,1,D]."""
    outs = []
    for b in range(q.shape[0]):
        n = int(context_lens[b])
        outs.append(
            attention_fp32_oracle(q[b:b + 1], k_dense[b:b + 1, :, :n], v_dense[b:b + 1, :, :n],
                                  scale=scale)
        )
    return torch.cat(outs, dim=0)


def attention_fp32_tree_oracle(q, k, v, tree_mask, *, scale=None):
    """Tree-attention oracle, fp32. q [B,H,T,D]; k,v [B,H_kv,N,D] = [prefix + T tree
    nodes]; tree_mask [T,T] bool (qi attends the marked tree nodes). Each query attends
    the full prefix + its tree ancestors. GQA-aware. Returns [B,H,T,D]."""
    qf, kf, vf = q.float(), k.float(), v.float()
    h_q, h_kv = qf.shape[1], kf.shape[1]
    if h_q != h_kv:
        rep = h_q // h_kv
        kf, vf = kf.repeat_interleave(rep, 1), vf.repeat_interleave(rep, 1)
    t, n = qf.shape[2], kf.shape[2]
    n_p = n - t
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.einsum("bhtd,bhnd->bhtn", qf, kf) * scale       # [B,H,T,N]
    mask = torch.ones(t, n, dtype=torch.bool, device=q.device)
    mask[:, n_p:] = tree_mask.bool()                          # tree keys per the mask
    s = s.masked_fill(~mask.view(1, 1, t, n), float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhtn,bhnd->bhtd", p, vf)
