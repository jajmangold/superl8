# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Autograd integration for superl8 attention.

The forward is the quantized int8 dp4a kernel; the backward implements the
EXACT FlashAttention-2 gradient (quantization treated as straight-through).
Because the quantized forward is non-differentiable (round), we do NOT gradcheck
it directly — instead `attn_ref` wraps the SAME analytic backward around a
non-quantized forward so the gradient MATH can be gradcheck'd, and the quantized
`attn` is graded against the fp32 oracle with relative bounds (see tests).

PR5a: the backward is computed in PyTorch (fp32 recompute) — correct and
gradient-checked, delivering usable training. PR5b swaps in the int8 dp4a
backward kernel behind these same gates.
"""

from __future__ import annotations

import math

import torch


def _softmax_scale(d: int, scale: float | None) -> float:
    return (1.0 / math.sqrt(d)) if scale is None else scale


def _attn_backward(q, k, v, out, d_out, causal, scale):
    """Exact FA2 backward (fp32). All tensors [B,H,S,D]. Returns dq,dk,dv."""
    qf, kf, vf, of, dof = (t.float() for t in (q, k, v, out, d_out))
    s = torch.einsum("bhmd,bhnd->bhmn", qf, kf) * scale
    if causal:
        m, n = s.shape[-2], s.shape[-1]
        mask = torch.ones(m, n, device=s.device, dtype=torch.bool).tril(n - m)
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)  # [B,H,M,N]
    dv = torch.einsum("bhmn,bhmd->bhnd", p, dof)  # P^T @ dO
    dp = torch.einsum("bhmd,bhnd->bhmn", dof, vf)  # dO @ V^T
    drow = (dof * of).sum(dim=-1, keepdim=True)  # rowsum(dO.O)
    ds = p * (dp - drow)  # dS
    dq = torch.einsum("bhmn,bhnd->bhmd", ds, kf) * scale
    dk = torch.einsum("bhmn,bhmd->bhnd", ds, qf) * scale
    return dq, dk, dv


class _AttnInt8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale):
        from . import _C
        from .quant import quantize_qk

        scale = _softmax_scale(q.shape[-1], scale)
        q_i8, q_scale, k_i8, k_scale, _ = quantize_qk(q, k, softmax_scale=scale)
        out, lse = _C.attn_int8_fwd_train(
            q_i8.contiguous(),
            q_scale.squeeze(-1).contiguous(),
            k_i8.contiguous(),
            k_scale.squeeze(-1).contiguous(),
            v.contiguous(),
            causal,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal, ctx.scale = causal, scale
        return out

    @staticmethod
    def backward(ctx, d_out):
        from .ops import backward_cuda

        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = backward_cuda(q, k, v, out, lse, d_out, causal=ctx.causal, scale=ctx.scale)
        return dq, dk, dv, None, None


class _AttnRef(torch.autograd.Function):
    """Non-quantized forward + the SAME analytic backward, for gradcheck."""

    @staticmethod
    def forward(ctx, q, k, v, causal, scale):
        scale = _softmax_scale(q.shape[-1], scale)
        s = torch.einsum("bhmd,bhnd->bhmn", q, k) * scale
        if causal:
            m, n = s.shape[-2], s.shape[-1]
            mask = torch.ones(m, n, device=s.device, dtype=torch.bool).tril(n - m)
            s = s.masked_fill(~mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        out = torch.einsum("bhmn,bhnd->bhmd", p, v)
        ctx.save_for_backward(q, k, v, out)
        ctx.causal, ctx.scale = causal, scale
        return out

    @staticmethod
    def backward(ctx, d_out):
        q, k, v, out = ctx.saved_tensors
        dq, dk, dv = _attn_backward(q, k, v, out, d_out, ctx.causal, ctx.scale)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def attn(q, k, v, *, causal: bool = False, scale: float | None = None):
    """Differentiable int8 dp4a attention (forward kernel + exact FA2 backward)."""
    return _AttnInt8.apply(q, k, v, causal, scale)


def attn_ref(q, k, v, *, causal: bool = False, scale: float | None = None):
    """Non-quantized reference attention sharing superl8's analytic backward."""
    return _AttnRef.apply(q, k, v, causal, scale)
