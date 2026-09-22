# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Track-2 (issue #6) v1: the naive scalar-fp32 Gated-DeltaNet recurrence is
under test. This is the harness every later stage (v2 ungated chunked, v3
gated, v4 int8 dp4a) must be validated against — it must be trustworthy
first, same principle as PR1's `test_reference.py` for softmax attention.
"""

import pytest
import torch

from tests.reference_linear_attn import gated_delta_rule_oracle, gates_from_logits

# (B, H, T, Dk, Dv) — small; the naive O(T) loop is slow. Includes T values
# that are NOT multiples of the issue's chunk size (C=64): 1, 5, 63, 130.
SHAPES = [
    (1, 2, 1, 32, 32),
    (2, 2, 5, 16, 32),
    (1, 1, 63, 32, 32),
    (1, 2, 130, 64, 64),
]


def make_inputs(shape, device, *, beta_val=None, decay_val=None):
    b, h, t, dk, dv = shape
    q = torch.randn(b, h, t, dk, device=device, dtype=torch.float32)
    k = torch.randn(b, h, t, dk, device=device, dtype=torch.float32)
    v = torch.randn(b, h, t, dv, device=device, dtype=torch.float32)
    beta_logit = (
        torch.randn(b, h, t, device=device)
        if beta_val is None
        else torch.full((b, h, t), beta_val, device=device)
    )
    decay_logit = (
        torch.randn(b, h, t, device=device)
        if decay_val is None
        else torch.full((b, h, t), decay_val, device=device)
    )
    return q, k, v, beta_logit, decay_logit


@pytest.mark.correctness
def test_gates_from_logits_ranges(device):
    beta_logit = torch.randn(4, 8, 100, device=device) * 5
    decay_logit = torch.randn(4, 8, 100, device=device) * 5
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    assert torch.isfinite(alpha).all() and torch.isfinite(beta).all()
    assert (alpha > 0).all() and (alpha <= 1).all()
    assert (beta > 0).all() and (beta < 1).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_smoke_finite_shape_dtype(device, shape):
    """Smoke: runs, returns fp32, correct shapes, no NaN/Inf."""
    b, h, t, dk, dv = shape
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    out, state = gated_delta_rule_oracle(q, k, v, alpha, beta)
    assert out.shape == (b, h, t, dv)
    assert state.shape == (b, h, dv, dk)
    assert out.dtype == torch.float32 and state.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.isfinite(state).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_delta_rule_write_invariant(device, shape):
    """beta=1 (full write) + L2-normalized k: the delta rule guarantees an
    EXACT interpolation, S_t @ k_t == v_t, for every t and regardless of the
    decay gate. This is the defining algebraic property of the delta rule
    (not softmax attention) and the strongest test available without the
    (unreachable) fni8-serve oracle to diff against."""
    b, h, t, dk, dv = shape
    q, k, v, _, decay_logit = make_inputs(shape, device)
    alpha, _ = gates_from_logits(torch.zeros(b, h, t, device=device), decay_logit)
    beta = torch.ones(b, h, t, device=device)
    _, final_state = gated_delta_rule_oracle(q, k, v, alpha, beta, normalize_k=True)

    # Replay to capture the state right after each write (not just the final one).
    state = torch.zeros(b, h, dv, dk, device=device)
    k_n = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    for ti in range(t):
        _, state = gated_delta_rule_oracle(
            q[:, :, ti : ti + 1],
            k[:, :, ti : ti + 1],
            v[:, :, ti : ti + 1],
            alpha[:, :, ti : ti + 1],
            beta[:, :, ti : ti + 1],
            normalize_k=True,
            initial_state=state,
        )
        readback = torch.einsum("bhvk,bhk->bhv", state, k_n[:, :, ti, :])
        torch.testing.assert_close(readback, v[:, :, ti, :], rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_pure_decay_no_write_matches_closed_form(device, shape):
    """beta=0: no writes happen, so S_t = (prod_{s<=t} alpha_s) * S_0 — check
    the loop implementation against that closed form."""
    b, h, t, dk, dv = shape
    q, k, v, _, decay_logit = make_inputs(shape, device)
    beta = torch.zeros(b, h, t, device=device)
    alpha, _ = gates_from_logits(torch.zeros(b, h, t, device=device), decay_logit)
    s0 = torch.randn(b, h, dv, dk, device=device)

    out, final_state = gated_delta_rule_oracle(q, k, v, alpha, beta, initial_state=s0)

    cumdecay = alpha.cumprod(dim=-1)  # [B,H,T]
    expected_final = cumdecay[:, :, -1].view(b, h, 1, 1) * s0
    torch.testing.assert_close(final_state, expected_final, rtol=1e-4, atol=1e-5)

    state = s0.clone()
    for ti in range(t):
        state = alpha[:, :, ti].view(b, h, 1, 1) * state
        expected_t = torch.einsum("bhvk,bhk->bhv", state, q[:, :, ti, :])
        torch.testing.assert_close(out[:, :, ti, :], expected_t, rtol=1e-4, atol=1e-5)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_determinism(device, shape):
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    out1, state1 = gated_delta_rule_oracle(q, k, v, alpha, beta)
    out2, state2 = gated_delta_rule_oracle(q, k, v, alpha, beta)
    out3, state3 = gated_delta_rule_oracle(q, k, v, alpha, beta)
    assert torch.equal(out1, out2) and torch.equal(out2, out3)
    assert torch.equal(state1, state2) and torch.equal(state2, state3)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("split", [1, 2, 3])
def test_compose_in_loop_stability(device, shape, split):
    """Splitting a sequence into successive calls that hand off `initial_state`
    must reproduce one full sequential pass bit-close. This is the exact
    associativity property the future chunked (v2) parallel form depends on
    -- derisking it is the point of staging v1 before v2."""
    b, h, t, dk, dv = shape
    if t < split:
        pytest.skip("sequence shorter than requested split count")
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)

    out_full, state_full = gated_delta_rule_oracle(q, k, v, alpha, beta)

    bounds = sorted(set(torch.linspace(0, t, split + 1).round().long().tolist()))
    out_chunks = []
    state = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi == lo:
            continue
        o_c, state = gated_delta_rule_oracle(
            q[:, :, lo:hi],
            k[:, :, lo:hi],
            v[:, :, lo:hi],
            alpha[:, :, lo:hi],
            beta[:, :, lo:hi],
            initial_state=state,
        )
        out_chunks.append(o_c)
    out_chunked = torch.cat(out_chunks, dim=2)

    torch.testing.assert_close(out_chunked, out_full, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state, state_full, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
def test_alpha_one_beta_one_is_plain_delta_rule(device):
    """alpha==1 (no decay) recovers the ungated (plain) delta rule -- the v2
    milestone's target. Sanity-checks that the gates are additive dials, not
    entangled with the base recurrence."""
    shape = (1, 2, 40, 32, 32)
    b, h, t, dk, dv = shape
    q, k, v, _, _ = make_inputs(shape, device)
    alpha = torch.ones(b, h, t, device=device)
    beta = torch.full((b, h, t), 0.3, device=device)
    out, state = gated_delta_rule_oracle(q, k, v, alpha, beta)

    k_n = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    s = torch.zeros(b, h, dv, dk, device=device)
    for ti in range(t):
        sk = torch.einsum("bhvk,bhk->bhv", s, k_n[:, :, ti, :])
        write = 0.3 * (v[:, :, ti, :] - sk)
        s = s + torch.einsum("bhv,bhk->bhvk", write, k_n[:, :, ti, :])
        expected_t = torch.einsum("bhvk,bhk->bhv", s, q[:, :, ti, :])
        torch.testing.assert_close(out[:, :, ti, :], expected_t, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(state, s, rtol=1e-4, atol=1e-5)
