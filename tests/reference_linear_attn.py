# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Reference (oracle) implementation for the Gated-DeltaNet linear-attention
recurrence — Track-2 (issue #6): int8 dp4a chunked Gated-DeltaNet.

Staged like the softmax-attention harness in `reference.py`:
  v1 (this file) — a naive SEQUENTIAL fp32 recurrence, one token at a time.
      Unambiguous ground truth; O(T) python loop, small shapes only. Every
      later parallel/chunked form (v2 ungated, v3 gated, v4 int8 dp4a) must
      be validated against this before it is trusted.
  v2+ — chunked WY/UT parallel forms (chunk C=64) — not yet implemented;
      tracked in follow-up `kernel`-labeled issues linked from #6.

Recurrence (Gated DeltaNet — Yang, Kautz, Hatamizadeh 2024; the formulation
used by Qwen3-Next / MiniMax "lightning attention" hybrid layers). Per
(batch, head), state S in R^{Dv x Dk}:

    k_t <- k_t / ||k_t||_2                     (L2-norm; keeps the
                                                  Householder-style write
                                                  well-conditioned)
    S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    o_t = S_t @ q_t

  Implemented here as the algebraically-equivalent decayed-read form (matches
  the fla-org / Qwen3-Next reference convention):
    sk_t = alpha_t * (S_{t-1} @ k_t)
    S_t  = alpha_t * S_{t-1} + beta_t * (v_t - sk_t) (x) k_t     ((x) = outer)

  alpha_t in (0, 1] is a per-step scalar decay gate — the "gamma" of the
  issue's "gamma-cumprod": a chunked parallel form needs the product of
  consecutive alpha_t folded in LOG space (it can underflow fp32 in linear
  space over a long chunk), but this naive sequential form applies alpha_t
  one step at a time, so it needs no such stabilization itself. beta_t in
  (0, 1) is the write-strength gate. When beta_t == 1 and k_t is unit-norm,
  the update is an exact interpolation: S_t @ k_t == v_t (the delta-rule
  invariant `test_delta_rule_write_invariant` in test_gated_delta_rule.py
  checks this directly).

NOTE ON THE ORACLE: issue #6 names
`superl8serve/layers/linear_attn.py::recurrent_gated_delta_rule` (in the sibling
`fni8-serve` repo) as the oracle the eventual kernel must match. That file is
not reachable from this repo's checkout/sandbox, so this is the standard
published Gated DeltaNet recurrence rather than a byte-for-byte port. Before
wiring the fni8-serve integration, diff the two formulations (state layout,
gate parameterization, L2-norm placement) token-for-token — do not assume
they agree without that check.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gates_from_logits(
    beta_logit: torch.Tensor, decay_logit: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw (unconstrained) logits to the gate ranges the recurrence needs.

    ``beta = sigmoid(beta_logit)`` in (0, 1) — the write-strength gate.
    ``alpha = exp(-softplus(decay_logit))`` in (0, 1] — the per-step decay
    gate (1 at ``decay_logit -> -inf``, i.e. "don't forget"; -> 0 as
    ``decay_logit -> +inf``, i.e. "forget everything").
    """
    beta = torch.sigmoid(beta_logit.float())
    alpha = torch.exp(-F.softplus(decay_logit.float()))
    return alpha, beta


def gated_delta_rule_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    normalize_k: bool = True,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive sequential Gated-DeltaNet recurrence, fully fp32. Slow (O(T)
    python loop) — small shapes only. The ground truth every later
    chunked/int8 form is measured against.

    q, k: [B,H,T,Dk]; v: [B,H,T,Dv]; alpha, beta: [B,H,T] (gate ranges are
    the caller's responsibility — see :func:`gates_from_logits`).
    ``initial_state``, if given: [B,H,Dv,Dk] fp32, the carry-in state (lets
    a long sequence be chunked into successive calls; see
    ``test_compose_in_loop_stability``).

    Returns ``(o [B,H,T,Dv] fp32, final_state [B,H,Dv,Dk] fp32)``.
    """
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    af, bf = alpha.float(), beta.float()
    if normalize_k:
        kf = kf / kf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    state = (
        torch.zeros(b, h, dv, dk, device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty(b, h, t, dv, device=q.device, dtype=torch.float32)
    for ti in range(t):
        k_t = kf[:, :, ti, :]  # [B,H,Dk]
        v_t = vf[:, :, ti, :]  # [B,H,Dv]
        q_t = qf[:, :, ti, :]  # [B,H,Dk]
        a_t = af[:, :, ti].view(b, h, 1, 1)  # [B,H,1,1], state-shaped
        a_t_v = af[:, :, ti].view(b, h, 1)  # [B,H,1], Dv-shaped
        beta_t = bf[:, :, ti].view(b, h, 1)  # [B,H,1]
        sk = a_t_v * torch.einsum("bhvk,bhk->bhv", state, k_t)  # alpha_t * S_{t-1} k_t
        write = beta_t * (v_t - sk)  # [B,H,Dv] delta-rule write
        state = a_t * state + torch.einsum("bhv,bhk->bhvk", write, k_t)
        out[:, :, ti, :] = torch.einsum("bhvk,bhk->bhv", state, q_t)
    return out, state


def ungated_delta_rule_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    normalize_k: bool = True,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ungated (alpha=1, beta=1) DeltaNet recurrence -- the target the v2
    chunked WY/UT kernel must match (fp32 reassociation tolerance).

    Mathematically identical to the plain delta rule::

        k_t <- k_t / ||k_t||_2
        S_t = S_{t-1} + (v_t - S_{t-1} @ k_t) (x) k_t
        o_t = S_t @ q_t

    where (x) is the outer product.  This is the same recurrence as
    :func:`gated_delta_rule_oracle` with alpha=1, beta=1, but the
    implementation avoids unnecessary multiplies so the summation order
    is closer to what the chunked kernel actually does.

    q, k: [B,H,T,Dk] fp32; v: [B,H,T,Dv] fp32.
    ``initial_state``, if given: [B,H,Dv,Dk] fp32 carry-in state.

    Returns ``(o [B,H,T,Dv] fp32, final_state [B,H,Dv,Dk] fp32)``.
    """
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    if normalize_k:
        kf = kf / kf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    state = (
        torch.zeros(b, h, dv, dk, device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty(b, h, t, dv, device=q.device, dtype=torch.float32)
    for ti in range(t):
        k_t = kf[:, :, ti, :]                                 # [B,H,Dk]
        v_t = vf[:, :, ti, :]                                 # [B,H,Dv]
        q_t = qf[:, :, ti, :]                                 # [B,H,Dk]
        sk = torch.einsum("bhvk,bhk->bhv", state, k_t)        # S_{t-1} @ k_t
        r = v_t - sk                                           # residual
        state = state + torch.einsum("bhv,bhk->bhvk", r, k_t)  # rank-1 update
        out[:, :, ti, :] = torch.einsum("bhvk,bhk->bhv", state, q_t)
    return out, state


def _quantize_rowwise_i8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row RTN int8 over the last dim. Returns (q float in
    [-127,127], scale fp32 [...,1]) — the exact arithmetic the v4 CUDA kernel
    performs per q/k row (values kept as float so the int32 dp4a accumulate is
    reproduced with an ordinary matmul)."""
    Q_MAX = 127.0
    scale = x.abs().amax(dim=-1, keepdim=True) / Q_MAX
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(x / safe).clamp_(-Q_MAX, Q_MAX)
    return q, safe


def ungated_delta_rule_int8_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    normalize_k: bool = True,
    initial_state: torch.Tensor | None = None,
    C: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """int8-dp4a chunked ungated delta-rule — the Python sim of the v4 CUDA
    kernel (``superl8.deltanet_chunk_int8_fwd``).

    Identical chunked algebra to :func:`ungated_delta_rule_oracle`, but the two
    Dk-contraction score matrices are quantized to int8 and dotted as integers
    (dequantized with the two per-row scales), exactly as the kernel's ``__dp4a``
    path does:

      * K-Gram   ``G[i,j] = k̂ᵢ·k̂ⱼ``   (feeds forward substitution)
      * Q·K read ``A[i,j] = qᵢ·k̂ⱼ``     (feeds the output)

    where ``k̂`` is the L2-normalised key. Everything else — state S, V,
    residuals r, ``W = V − S@Kᵀ``, the output base ``S@q`` and the state update
    ``r@K`` — stays fp32 (never quantized). K is L2-normalised BEFORE quant; that
    normalisation is the delta-rule analogue of K-smoothing.

    Exists so the int8 kernel has a Python arithmetic twin to cross-check against
    (kernel SQNR-vs-oracle should be no worse than this reference's), and so the
    inherent quantization quality is measurable independent of the CUDA code.

    Returns ``(o [B,H,T,Dv] fp32, final_state [B,H,Dv,Dk] fp32)``.
    """
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    if normalize_k:
        kf = kf / kf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    state = (
        torch.zeros(b, h, dv, dk, device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty(b, h, t, dv, device=q.device, dtype=torch.float32)

    num_chunks = (t + C - 1) // C
    for ci in range(num_chunks):
        lo = ci * C
        hi = min(lo + C, t)
        Kc = kf[:, :, lo:hi, :]  # [B,H,L,Dk] (already L2-normalised)
        Qc = qf[:, :, lo:hi, :]
        Vc = vf[:, :, lo:hi, :]
        L = hi - lo

        # int8 quant (one scheme per row, shared by Gram + Q·K, matching kernel)
        k_i8, k_s = _quantize_rowwise_i8(Kc)      # [B,H,L,Dk], [B,H,L,1]
        q_i8, q_s = _quantize_rowwise_i8(Qc)
        ks_row = k_s.squeeze(-1)                   # [B,H,L]
        qs_row = q_s.squeeze(-1)

        # int8 K-Gram: G[i,j] = (k_i8·k_j8) * ks[i] * ks[j]
        G = torch.einsum("bhik,bhjk->bhij", k_i8, k_i8)
        G = G * ks_row[:, :, :, None] * ks_row[:, :, None, :]
        # int8 Q·K: A[i,j] = (q_i8·k_j8) * qs[i] * ks[j]
        A = torch.einsum("bhik,bhjk->bhij", q_i8, k_i8)
        A = A * qs_row[:, :, :, None] * ks_row[:, :, None, :]

        # W = V - S@K^T (fp32)
        w = Vc - torch.einsum("bhvk,bhlk->bhlv", state, Kc)
        # forward substitution (fp32): r[i] = w[i] - Σ_{j<i} G[j,i] r[j]
        r = w.clone()
        for i in range(1, L):
            corr = torch.zeros(b, h, dv, device=q.device, dtype=torch.float32)
            for j in range(i):
                corr = corr + G[:, :, j, i, None] * r[:, :, j, :]
            r[:, :, i, :] = r[:, :, i, :] - corr
        # output: o_i = S@q_i + Σ_{j≤i} A[i,j] r[j]
        base = torch.einsum("bhvk,bhlk->bhlv", state, Qc)  # [B,H,L,Dv]
        for i in range(L):
            acc = base[:, :, i, :].clone()
            for j in range(i + 1):
                acc = acc + A[:, :, i, j, None] * r[:, :, j, :]
            out[:, :, lo + i, :] = acc
        # state update (fp32): S += r^T @ K
        state = state + torch.einsum("bhlv,bhlk->bhvk", r, Kc)

    return out, state


def gated_chunked_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    normalize_k: bool = True,
    initial_state: torch.Tensor | None = None,
    C: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaNet computed in chunks using the log-space γ-cumprod
    decomposition — the Python reference for the v3 CUDA kernel.

    Mathematically identical to :func:`gated_delta_rule_oracle`; exists to
    validate the chunked algebra independently before trusting the CUDA
    kernel.  Chunk size *C* mirrors the kernel's dynamic chunk-size
    heuristic and does not affect the result (within fp32 reassociation).

    Algorithm per chunk of length L ≤ C:
    1.  L2-normalise K rows.
    2.  Compute γ[i] = ∏_{j=0}^{i-1} α_j in log-space (i=0..L).
    3.  w[i] = v[i] - γ[i+1] · S₀ @ k[i]   (initial residuals).
    4.  Forward-substitute r[i] from w[i] with (γ[i+1]/γ[j+1])·β[j]·(kⱼ·kᵢ) coefficients.
    5.  o[i] = γ[i+1]·S₀@q[i] + Σ_{j≤i} (γ[i+1]/γ[j+1])·β[j]·(kⱼ·q[i])·r[j].
    6.  S_L = γ[L]·S₀ + Σ_{j<L} (γ[L]/γ[j+1])·β[j]·r[j]⊗k[j].

    q, k: [B,H,T,Dk] fp32; v: [B,H,T,Dv] fp32; alpha, beta: [B,H,T] fp32.
    ``initial_state``, if given: [B,H,Dv,Dk] fp32 carry-in state.
    C: chunk size (default 64, same as kernel's ``compute_chunk_size`` cap).

    Returns ``(o [B,H,T,Dv] fp32, final_state [B,H,Dv,Dk] fp32)``.
    """
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    af, bf = alpha.float(), beta.float()
    if normalize_k:
        kf = kf / kf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    state = (
        torch.zeros(b, h, dv, dk, device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty(b, h, t, dv, device=q.device, dtype=torch.float32)

    num_chunks = (t + C - 1) // C
    for ci in range(num_chunks):
        lo = ci * C
        hi = min(lo + C, t)
        L = hi - lo

        Qc = qf[:, :, lo:hi, :]  # [B,H,L,Dk]
        Kc = kf[:, :, lo:hi, :]  # [B,H,L,Dk]
        Vc = vf[:, :, lo:hi, :]  # [B,H,L,Dv]
        Ac = af[:, :, lo:hi]     # [B,H,L]
        Bc = bf[:, :, lo:hi]     # [B,H,L]

        # log-space γ-cumprod: γ[i] = exp(Σ_{j=0}^{i-1} log α_j), γ[0]=1
        log_ac = torch.log(Ac.clamp_min(1e-12))          # [B,H,L]
        log_gamma = torch.cat(
            [torch.zeros(b, h, 1, device=q.device, dtype=torch.float32),
             log_ac.cumsum(dim=-1)],
            dim=-1,
        )                                                # [B,H,L+1]
        gamma = torch.exp(log_gamma)                     # [B,H,L+1]

        # ---- w[i] = v[i] - γ[i+1]·S₀@k[i] -----------------------------------
        w = Vc.clone()  # [B,H,L,Dv], mutated in-place
        for i in range(L):
            sk = torch.einsum("bhvk,bhk->bhv", state, Kc[:, :, i, :])
            w[:, :, i, :] = w[:, :, i, :] - gamma[:, :, i + 1, None] * sk

        # ---- K_gram[i,j] = k_i · k_j  [B,H,L,L] -----------------------------
        K_gram = torch.einsum("bhik,bhjk->bhij", Kc, Kc)

        # ---- Forward substitution: r from w ---------------------------------
        r = w.clone()  # [B,H,L,Dv]
        for i in range(1, L):
            correction = torch.zeros(b, h, dv, device=q.device, dtype=torch.float32)
            for j in range(i):
                decay = gamma[:, :, i + 1, None] / gamma[:, :, j + 1, None]    # [B,H,1]
                gated_dot = decay * Bc[:, :, j, None] * K_gram[:, :, j, i, None]  # [B,H,1]
                correction = correction + gated_dot * r[:, :, j, :]
            r[:, :, i, :] = r[:, :, i, :] - correction

        # ---- Output o[i] ----------------------------------------------------
        for i in range(L):
            acc = gamma[:, :, i + 1, None] * torch.einsum(
                "bhvk,bhk->bhv", state, Qc[:, :, i, :]
            )
            for j in range(i + 1):
                coeff = gamma[:, :, i + 1, None] / gamma[:, :, j + 1, None]
                coeff = coeff * Bc[:, :, j, None]
                qkj = torch.einsum("bhk,bhk->bh", Qc[:, :, i, :], Kc[:, :, j, :])
                acc = acc + coeff * qkj.unsqueeze(-1) * r[:, :, j, :]
            out[:, :, lo + i, :] = acc

        # ---- State update S_L -----------------------------------------------
        state = gamma[:, :, L, None, None] * state
        for j in range(L):
            coeff = (gamma[:, :, L] / gamma[:, :, j + 1])[:, :, None, None]
            coeff = coeff * Bc[:, :, j, None, None]
            update = torch.einsum("bhv,bhk->bhvk", r[:, :, j, :], Kc[:, :, j, :])
            state = state + coeff * update

    return out, state
