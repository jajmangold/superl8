# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Reference (oracle) for MiniMax Lightning (un-gated linear) attention.

Recurrence (standard linear attention, no k-norm, no delta rule):

    S_t = S_{t-1} + v_t (x) k_t    ((x) = outer product, R^{Dv x Dk})
    o_t = S_t @ q_t

This is the ground truth the CUDA kernel must match exactly (fp32 tolerance).
The int8 kernel (next PR) is validated against this oracle before it's trusted.
"""

from __future__ import annotations

import torch


def lightning_attn_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive sequential fp32 Lightning/MiniMax linear attention recurrence.

    S_t = S_{t-1} + v_t (x) k_t
    o_t = S_t @ q_t

    q, k: [B, H, T, Dk] fp32; v: [B, H, T, Dv] fp32.
    ``initial_state``, if given: [B, H, Dv, Dk] fp32 carry-in state.

    Returns ``(o [B, H, T, Dv] fp32, final_state [B, H, Dv, Dk] fp32)``.
    """
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    state = (
        torch.zeros(b, h, dv, dk, device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty(b, h, t, dv, device=q.device, dtype=torch.float32)
    for ti in range(t):
        k_t = kf[:, :, ti, :]                              # [B,H,Dk]
        v_t = vf[:, :, ti, :]                              # [B,H,Dv]
        q_t = qf[:, :, ti, :]                              # [B,H,Dk]
        state = state + torch.einsum("bhv,bhk->bhvk", v_t, k_t)
        out[:, :, ti, :] = torch.einsum("bhvk,bhk->bhv", state, q_t)
    return out, state


def lightning_attn_quadratic_closed_form(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """O(T^2) closed-form cross-check for the Lightning recurrence.

    The cumulative state up to position t is:
        S_t = S_0 + sum_{j < t} v_j (x) k_j

    So the output at t is:
        o_t = S_0 @ q_t + sum_{j < t} (v_j (x) k_j) @ q_t
            = S_0 @ q_t + sum_{j < t} v_j * (k_j . q_t)

    This computes pairwise contributions explicitly, providing an independent
    check of the sequential oracle's recurrence algebra.  fp32 tolerance only.

    q, k: [B, H, T, Dk] fp32; v: [B, H, T, Dv] fp32.
    ``initial_state``, if given: [B, H, Dv, Dk] fp32.

    Returns ``(o [B, H, T, Dv] fp32, final_state [B, H, Dv, Dk] fp32)``.
    """
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    s0 = (
        torch.zeros(b, h, dv, dk, device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float()
    )
    out = torch.empty(b, h, t, dv, device=q.device, dtype=torch.float32)
    for ti in range(t):
        q_t = qf[:, :, ti, :]                                    # [B,H,Dk]
        base = torch.einsum("bhvk,bhk->bhv", s0, q_t)            # S_0 @ q_t
        acc = torch.zeros(b, h, dv, device=q.device, dtype=torch.float32)
        for j in range(ti + 1):
            k_j = kf[:, :, j, :]                                 # [B,H,Dk]
            v_j = vf[:, :, j, :]                                 # [B,H,Dv]
            kq = torch.einsum("bhk,bhk->bh", k_j, q_t)           # k_j . q_t
            acc = acc + v_j * kq.unsqueeze(-1)
        out[:, :, ti, :] = base + acc
    final_state = s0.clone()
    for ti in range(t):
        final_state = final_state + torch.einsum(
            "bhv,bhk->bhvk", vf[:, :, ti, :], kf[:, :, ti, :]
        )
    return out, final_state
