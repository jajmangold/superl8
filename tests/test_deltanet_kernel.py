# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Track-2 (issue #6) v1: the CUDA scalar-recurrence Gated-DeltaNet kernel
(`superl8.deltanet_recurrent_fwd`) under test against the merged fp32 Python
oracle (`tests/reference_linear_attn.py`, landed in PR #17). This is the
on-device ground truth v2 (chunked)/v3 (gated)/v4 (int8 dp4a) must be
validated against, so it needs to be trustworthy first -- same principle
`test_gated_delta_rule.py` applied to the Python oracle itself.
"""

import sys
from pathlib import Path

import pytest
import torch

import superl8
from tests.reference_linear_attn import (gated_chunked_reference, gated_delta_rule_oracle,
                                       gates_from_logits, ungated_delta_rule_oracle)
from tests.test_gated_delta_rule import SHAPES, make_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, time_ms  # noqa: E402


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_smoke_finite_shape_dtype(device, shape):
    b, h, t, dk, dv = shape
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    out, state = superl8.deltanet_recurrent_fwd(q, k, v, alpha, beta)
    assert out.shape == (b, h, t, dv)
    assert state.shape == (b, h, dv, dk)
    assert out.dtype == torch.float32 and state.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(state).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_fp32_oracle(device, shape):
    """The defining property: the CUDA kernel must reproduce the sequential
    fp32 Python oracle bit-close (both are fp32, same op order per element;
    only floating-point summation order differs)."""
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    out_ref, state_ref = gated_delta_rule_oracle(q, k, v, alpha, beta)
    out_cuda, state_cuda = superl8.deltanet_recurrent_fwd(q, k, v, alpha, beta)
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_oracle_with_initial_state(device, shape):
    b, h, t, dk, dv = shape
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    s0 = torch.randn(b, h, dv, dk, device=device, dtype=torch.float32)
    out_ref, state_ref = gated_delta_rule_oracle(q, k, v, alpha, beta, initial_state=s0)
    out_cuda, state_cuda = superl8.deltanet_recurrent_fwd(q, k, v, alpha, beta, initial_state=s0)
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("split", [1, 2, 3])
def test_compose_in_loop_stability(device, shape, split):
    """Same associativity property `test_gated_delta_rule.py` proves for the
    Python oracle -- required before v2's chunked form can trust hand-off via
    `initial_state`."""
    b, h, t, dk, dv = shape
    if t < split:
        pytest.skip("sequence shorter than requested split count")
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)

    out_full, _ = superl8.deltanet_recurrent_fwd(q, k, v, alpha, beta)

    bounds = sorted(set(torch.linspace(0, t, split + 1).round().long().tolist()))
    out_chunks = []
    state = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi == lo:
            continue
        o_c, state = superl8.deltanet_recurrent_fwd(
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


@pytest.mark.correctness
def test_determinism(device):
    shape = (2, 2, 130, 64, 64)
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    out1, state1 = superl8.deltanet_recurrent_fwd(q, k, v, alpha, beta)
    out2, state2 = superl8.deltanet_recurrent_fwd(q, k, v, alpha, beta)
    out3, state3 = superl8.deltanet_recurrent_fwd(q, k, v, alpha, beta)
    assert torch.equal(out1, out2) and torch.equal(out2, out3)
    assert torch.equal(state1, state2) and torch.equal(state2, state3)


@pytest.mark.correctness
def test_rejects_dim_over_128(device):
    shape = (1, 1, 4, 256, 32)
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    with pytest.raises(RuntimeError, match="128"):
        superl8.deltanet_recurrent_fwd(q, k, v, alpha, beta)


@pytest.mark.correctness
def test_rejects_non_fp32(device):
    shape = (1, 1, 4, 32, 32)
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    with pytest.raises(RuntimeError, match="fp32"):
        superl8.deltanet_recurrent_fwd(q.half(), k.half(), v.half(), alpha, beta)


@pytest.mark.perf
def test_deltanet_recurrent_fwd_perf(device):
    b, h, t, dk, dv = 2, 16, 2048, 64, 64
    q, k, v, beta_logit, decay_logit = make_inputs((b, h, t, dk, dv), device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    ms = time_ms(lambda: superl8.deltanet_recurrent_fwd(q, k, v, alpha, beta))
    # Soft-skips until a baseline is committed (v1 is a correctness ground
    # truth, not the optimized path -- v2+ get the real perf gate); then
    # fails on >5% regression like every other kernel's perf test.
    assert_no_regression("deltanet_recurrent_fwd_v1.b2h16t2048d64.fp32", ms)


# ============================================================================
# decode: warp-per-(head,v-row) fp32 recurrence, CUDA-graph-friendly (no smem).
# Same gated-delta-rule math as v1 -> validated against the same fp32 oracle;
# grid = ceil(B*H*Dv / warps_per_block) instead of B*H, and state lives in
# registers so there is no per-call cudaFuncSetAttribute (v1's graph-hostile
# runtime call). This is the fast decode path superl8serve wires in for L==1.
# ============================================================================

# include the T=1 decode shape (B=1,H=16,Dk=Dv=128) that Qwen3.5 actually runs.
DECODE_SHAPES = list(SHAPES) + [(1, 16, 1, 128, 128)]


@pytest.mark.correctness
@pytest.mark.parametrize("shape", DECODE_SHAPES)
def test_decode_smoke_finite_shape_dtype(device, shape):
    b, h, t, dk, dv = shape
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    out, state = superl8.deltanet_recurrent_decode(q, k, v, alpha, beta)
    assert out.shape == (b, h, t, dv) and state.shape == (b, h, dv, dk)
    assert out.dtype == torch.float32 and state.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(state).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", DECODE_SHAPES)
def test_decode_matches_fp32_oracle(device, shape):
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    out_ref, state_ref = gated_delta_rule_oracle(q, k, v, alpha, beta)
    out_cuda, state_cuda = superl8.deltanet_recurrent_decode(q, k, v, alpha, beta)
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", DECODE_SHAPES)
def test_decode_matches_oracle_with_initial_state(device, shape):
    b, h, t, dk, dv = shape
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    s0 = torch.randn(b, h, dv, dk, device=device, dtype=torch.float32)
    out_ref, state_ref = gated_delta_rule_oracle(q, k, v, alpha, beta, initial_state=s0)
    out_cuda, state_cuda = superl8.deltanet_recurrent_decode(q, k, v, alpha, beta, initial_state=s0)
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", DECODE_SHAPES)
@pytest.mark.parametrize("split", [1, 2, 3])
def test_decode_compose_in_loop_stability(device, shape, split):
    """Step-by-step decode (the real serving pattern): N single-token calls
    carrying `initial_state` must equal one whole-sequence call."""
    b, h, t, dk, dv = shape
    if t < split:
        pytest.skip("sequence shorter than requested split count")
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    out_full, _ = superl8.deltanet_recurrent_decode(q, k, v, alpha, beta)
    bounds = sorted(set(torch.linspace(0, t, split + 1).round().long().tolist()))
    out_chunks, state = [], None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi == lo:
            continue
        o_c, state = superl8.deltanet_recurrent_decode(
            q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi],
            alpha[:, :, lo:hi], beta[:, :, lo:hi], initial_state=state)
        out_chunks.append(o_c)
    torch.testing.assert_close(torch.cat(out_chunks, dim=2), out_full, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
def test_decode_determinism(device):
    shape = (1, 16, 1, 128, 128)
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    o1, s1 = superl8.deltanet_recurrent_decode(q, k, v, alpha, beta)
    o2, s2 = superl8.deltanet_recurrent_decode(q, k, v, alpha, beta)
    o3, s3 = superl8.deltanet_recurrent_decode(q, k, v, alpha, beta)
    assert torch.equal(o1, o2) and torch.equal(o2, o3)
    assert torch.equal(s1, s2) and torch.equal(s2, s3)


@pytest.mark.correctness
def test_decode_rejects_dim_over_128(device):
    shape = (1, 1, 4, 256, 32)
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    with pytest.raises(RuntimeError, match="128"):
        superl8.deltanet_recurrent_decode(q, k, v, alpha, beta)


@pytest.mark.correctness
def test_decode_rejects_non_fp32(device):
    shape = (1, 1, 4, 32, 32)
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    with pytest.raises(RuntimeError, match="fp32"):
        superl8.deltanet_recurrent_decode(q.half(), k.half(), v.half(), alpha, beta)


@pytest.mark.perf
def test_deltanet_recurrent_decode_perf(device):
    # Qwen3.5's actual per-layer decode shape: B=1, H=16 value heads, T=1, Dk=Dv=128.
    b, h, t, dk, dv = 1, 16, 1, 128, 128
    q, k, v, beta_logit, decay_logit = make_inputs((b, h, t, dk, dv), device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    ms = time_ms(lambda: superl8.deltanet_recurrent_decode(q, k, v, alpha, beta))
    assert_no_regression("deltanet_recurrent_decode.b1h16t1d128.fp32", ms)


# ============================================================================
# v2 (issue #40): ungated chunked WY/UT parallel DeltaNet kernel
# ============================================================================

V2_SHAPES = [
    (1, 2, 1, 32, 32),          # T=1 single-token edge case
    (2, 2, 66, 16, 32),         # T not multiple of chunk size
    (1, 2, 130, 64, 64),        # typical config, C=64
    (1, 1, 200, 32, 32),        # multiple full chunks + partial
    (1, 2, 130, 128, 64),       # Dk>Dv, C=64
    (1, 1, 130, 64, 128),       # Dv>Dk, C=64
    (1, 2, 130, 128, 128),      # large Dv,Dk, dynamic C
]


def make_v2_inputs(shape, device):
    b, h, t, dk, dv = shape
    q = torch.randn(b, h, t, dk, device=device, dtype=torch.float32)
    k = torch.randn(b, h, t, dk, device=device, dtype=torch.float32)
    v = torch.randn(b, h, t, dv, device=device, dtype=torch.float32)
    return q, k, v


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V2_SHAPES)
def test_v2_smoke_finite_shape_dtype(device, shape):
    b, h, t, dk, dv = shape
    q, k, v = make_v2_inputs(shape, device)
    out, state = superl8.deltanet_chunk_fwd(q, k, v)
    assert out.shape == (b, h, t, dv)
    assert state.shape == (b, h, dv, dk)
    assert out.dtype == torch.float32 and state.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(state).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V2_SHAPES)
def test_v2_matches_fp32_oracle(device, shape):
    q, k, v = make_v2_inputs(shape, device)
    out_ref, state_ref = ungated_delta_rule_oracle(q, k, v)
    out_cuda, state_cuda = superl8.deltanet_chunk_fwd(q, k, v)
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V2_SHAPES)
def test_v2_matches_oracle_with_initial_state(device, shape):
    b, h, t, dk, dv = shape
    q, k, v = make_v2_inputs(shape, device)
    s0 = torch.randn(b, h, dv, dk, device=device, dtype=torch.float32)
    out_ref, state_ref = ungated_delta_rule_oracle(q, k, v, initial_state=s0)
    out_cuda, state_cuda = superl8.deltanet_chunk_fwd(q, k, v, initial_state=s0)
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V2_SHAPES)
@pytest.mark.parametrize("split", [1, 2, 3])
def test_v2_compose_in_loop_stability(device, shape, split):
    b, h, t, dk, dv = shape
    if t < split:
        pytest.skip("sequence shorter than requested split count")
    q, k, v = make_v2_inputs(shape, device)

    out_full, _ = superl8.deltanet_chunk_fwd(q, k, v)

    bounds = sorted(set(torch.linspace(0, t, split + 1).round().long().tolist()))
    out_chunks = []
    state = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi == lo:
            continue
        o_c, state = superl8.deltanet_chunk_fwd(
            q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi], initial_state=state,
        )
        out_chunks.append(o_c)
    out_chunked = torch.cat(out_chunks, dim=2)
    torch.testing.assert_close(out_chunked, out_full, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
def test_v2_determinism(device):
    shape = (2, 2, 130, 64, 64)
    q, k, v = make_v2_inputs(shape, device)
    out1, state1 = superl8.deltanet_chunk_fwd(q, k, v)
    out2, state2 = superl8.deltanet_chunk_fwd(q, k, v)
    out3, state3 = superl8.deltanet_chunk_fwd(q, k, v)
    assert torch.equal(out1, out2) and torch.equal(out2, out3)
    assert torch.equal(state1, state2) and torch.equal(state2, state3)


@pytest.mark.correctness
def test_v2_rejects_non_fp32(device):
    shape = (1, 1, 4, 32, 32)
    q, k, v = make_v2_inputs(shape, device)
    with pytest.raises(RuntimeError, match="fp32"):
        superl8.deltanet_chunk_fwd(q.half(), k.half(), v.half())


@pytest.mark.perf
def test_v2_deltanet_chunk_fwd_perf(device):
    b, h, t, dk, dv = 2, 16, 2048, 64, 64
    q, k, v = make_v2_inputs((b, h, t, dk, dv), device)
    ms = time_ms(lambda: superl8.deltanet_chunk_fwd(q, k, v))
    assert_no_regression("deltanet_chunk_fwd_v2.b2h16t2048d64.fp32", ms)


# ============================================================================
# v3 (issue #56): gated chunked WY/UT parallel DeltaNet (log-space γ-cumprod)
# ============================================================================

V3_SHAPES = [
    (1, 2, 1, 32, 32),          # T=1 single-token edge case
    (2, 2, 66, 16, 32),         # T not multiple of chunk size
    (1, 2, 130, 64, 64),        # typical config, C ≤ 64
    (1, 1, 200, 32, 32),        # multiple full chunks + partial
    (1, 2, 130, 128, 64),       # Dk>Dv
    (1, 1, 130, 64, 128),       # Dv>Dk
    (1, 2, 130, 128, 128),      # large Dv,Dk, dynamic C
]


def make_v3_inputs(shape, device):
    b, h, t, dk, dv = shape
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    return q, k, v, alpha, beta


@torch.no_grad()
def _call_v3(q, k, v, alpha, beta, *, initial_state=None):
    """Call the v3 gated chunked kernel.  Will raise AttributeError until
    the kernel is wired into superl8 — that is expected (TDD: test first)."""
    return superl8.deltanet_gated_chunk_fwd(
        q, k, v, alpha, beta, initial_state=initial_state,
    )


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V3_SHAPES)
def test_v3_smoke_finite_shape_dtype(device, shape):
    b, h, t, dk, dv = shape
    q, k, v, alpha, beta = make_v3_inputs(shape, device)
    out, state = _call_v3(q, k, v, alpha, beta)
    assert out.shape == (b, h, t, dv)
    assert state.shape == (b, h, dv, dk)
    assert out.dtype == torch.float32 and state.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(state).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V3_SHAPES)
def test_v3_matches_fp32_oracle(device, shape):
    q, k, v, alpha, beta = make_v3_inputs(shape, device)
    out_ref, state_ref = gated_delta_rule_oracle(q, k, v, alpha, beta)
    out_cuda, state_cuda = _call_v3(q, k, v, alpha, beta)
    torch.testing.assert_close(out_cuda, out_ref, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=2e-4, atol=2e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V3_SHAPES)
def test_v3_matches_oracle_with_initial_state(device, shape):
    b, h, t, dk, dv = shape
    q, k, v, alpha, beta = make_v3_inputs(shape, device)
    s0 = torch.randn(b, h, dv, dk, device=device, dtype=torch.float32)
    out_ref, state_ref = gated_delta_rule_oracle(q, k, v, alpha, beta, initial_state=s0)
    out_cuda, state_cuda = _call_v3(q, k, v, alpha, beta, initial_state=s0)
    torch.testing.assert_close(out_cuda, out_ref, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=2e-4, atol=2e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V3_SHAPES)
@pytest.mark.parametrize("split", [1, 2, 3])
def test_v3_compose_in_loop_stability(device, shape, split):
    b, h, t, dk, dv = shape
    if t < split:
        pytest.skip("sequence shorter than requested split count")
    q, k, v, alpha, beta = make_v3_inputs(shape, device)

    out_full, _ = _call_v3(q, k, v, alpha, beta)

    bounds = sorted(set(torch.linspace(0, t, split + 1).round().long().tolist()))
    out_chunks = []
    state = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi == lo:
            continue
        o_c, state = _call_v3(
            q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi],
            alpha[:, :, lo:hi], beta[:, :, lo:hi],
            initial_state=state,
        )
        out_chunks.append(o_c)
    out_chunked = torch.cat(out_chunks, dim=2)
    torch.testing.assert_close(out_chunked, out_full, rtol=2e-4, atol=2e-4)


@pytest.mark.correctness
def test_v3_determinism(device):
    shape = (2, 2, 130, 64, 64)
    q, k, v, alpha, beta = make_v3_inputs(shape, device)
    out1, state1 = _call_v3(q, k, v, alpha, beta)
    out2, state2 = _call_v3(q, k, v, alpha, beta)
    out3, state3 = _call_v3(q, k, v, alpha, beta)
    assert torch.equal(out1, out2) and torch.equal(out2, out3)
    assert torch.equal(state1, state2) and torch.equal(state2, state3)


@pytest.mark.correctness
def test_v3_rejects_non_fp32(device):
    shape = (1, 1, 4, 32, 32)
    q, k, v, alpha, beta = make_v3_inputs(shape, device)
    with pytest.raises(RuntimeError, match="fp32"):
        _call_v3(q.half(), k.half(), v.half(), alpha, beta)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V3_SHAPES)
def test_v3_matches_chunked_reference(device, shape):
    q, k, v, alpha, beta = make_v3_inputs(shape, device)
    out_ref, state_ref = gated_chunked_reference(q, k, v, alpha, beta, C=64)
    out_cuda, state_cuda = _call_v3(q, k, v, alpha, beta)
    torch.testing.assert_close(out_cuda, out_ref, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=2e-4, atol=2e-4)


@pytest.mark.perf
def test_v3_deltanet_gated_chunk_fwd_perf(device):
    b, h, t, dk, dv = 2, 16, 2048, 64, 64
    q, k, v, alpha, beta = make_v3_inputs((b, h, t, dk, dv), device)
    ms = time_ms(lambda: _call_v3(q, k, v, alpha, beta))
    assert_no_regression("deltanet_gated_chunk_fwd_v3.b2h16t2048d64.fp32", ms)


# ---------------------------------------------------------------------------
# v3 cooperative-group geometry.
#
# v3 runs G = blockDim/Dv threads per Dv row and pads its shared-memory row
# strides to keep that access pattern bank-conflict free.  Dv values that do
# not divide blockDim, Dv smaller than a warp, Dk != Dv, and sequence lengths
# that are not a multiple of the dynamic chunk size C each exercise a distinct
# index path that the older shape list does not pin down.  `final_state` is
# asserted as well as `out`: the recurrence carries it between chunks, so a
# state-update bug can hide entirely inside the last chunk's output.
# ---------------------------------------------------------------------------
V3_GEOMETRY_SHAPES = [
    (1, 2, 130, 64, 96),        # Dv=96: blockDim % Dv != 0
    (1, 2, 199, 128, 128),      # real head dims, T not a multiple of C
    (1, 2, 67, 128, 32),        # Dv=32 -> wide groups, T not a multiple of C
    (1, 2, 64, 16, 16),         # Dk=Dv=16: C pinned at its 64 cap
    (1, 3, 512, 128, 128),      # the real prefill shape
]


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V3_GEOMETRY_SHAPES)
def test_v3_geometry_matches_fp32_oracle(device, shape):
    q, k, v, alpha, beta = make_v3_inputs(shape, device)
    out_ref, state_ref = gated_delta_rule_oracle(q, k, v, alpha, beta)
    out_cuda, state_cuda = _call_v3(q, k, v, alpha, beta)
    torch.testing.assert_close(out_cuda, out_ref, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=2e-4, atol=2e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V3_GEOMETRY_SHAPES)
def test_v3_geometry_carries_initial_state(device, shape):
    b, h, t, dk, dv = shape
    q, k, v, alpha, beta = make_v3_inputs(shape, device)
    s0 = torch.randn(b, h, dv, dk, device=device, dtype=torch.float32)
    out_ref, state_ref = gated_delta_rule_oracle(q, k, v, alpha, beta, initial_state=s0)
    out_cuda, state_cuda = _call_v3(q, k, v, alpha, beta, initial_state=s0)
    torch.testing.assert_close(out_cuda, out_ref, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=2e-4, atol=2e-4)


@pytest.mark.perf
def test_v3_deltanet_gated_chunk_fwd_prefill_perf(device):
    """The real GDN prefill shape: 44.66% of prefill CUDA time on the fleet."""
    b, h, t, dk, dv = 1, 48, 512, 128, 128
    q, k, v, alpha, beta = make_v3_inputs((b, h, t, dk, dv), device)
    ms = time_ms(lambda: _call_v3(q, k, v, alpha, beta))
    assert_no_regression("deltanet_gated_chunk_fwd_v3.b1h48t512d128.fp32", ms)


# ============================================================================
# v4 (issue #83): int8 dp4a ungated chunked DeltaNet (dp4a K-Gram + Q·K)
#
# int8 paths do NOT use allclose (a rounding boundary legitimately flips) — the
# gate is SQNR / cosine / rel-L1 (AGENTS.md numerics contract).  K is
# L2-normalised before quant; the accuracy gate decides where int8 is allowed.
# ============================================================================

from tests.reference_linear_attn import ungated_delta_rule_int8_reference  # noqa: E402
from tests.tolerances import (  # noqa: E402
    assert_int8_quality,
    cos_sim,
    rel_l1,
    sqnr_db,
)

# (B, H, T, Dk, Dv) — include non-tile-multiple T and a Dk that is NOT a
# multiple of 4 (Dk=34 → dp4a zero-pads to 36), plus Dk≠Dv cases.
V4_SHAPES = [
    (1, 2, 1, 32, 32),          # T=1 single-token edge case
    (2, 2, 66, 16, 32),         # T not multiple of chunk size
    (1, 2, 130, 64, 64),        # typical config
    (1, 1, 256, 64, 64),        # longer sequence, multiple chunks
    (1, 2, 130, 128, 64),       # Dk>Dv
    (1, 1, 130, 64, 128),       # Dv>Dk
    (1, 2, 130, 128, 128),      # large Dv,Dk (dynamic C)
    (1, 1, 66, 34, 48),         # Dk not a multiple of 4 (dp4a padding), Dv=48
]

# int8 accuracy bars (SageAttention-level, with margin over measured ~40 dB
# SQNR / 0.99995 cos / 0.009 rel-L1 on random inputs). NEVER weaken to pass.
V4_MIN_COS = 0.999
V4_MAX_REL_L1 = 0.02
V4_MIN_SQNR = 30.0
# State is a secondary output; the int8 K-Gram feeds it through r, so it earns a
# slightly looser (still tight) bar.
V4_STATE_MIN_SQNR = 25.0


def make_v4_inputs(shape, device):
    b, h, t, dk, dv = shape
    q = torch.randn(b, h, t, dk, device=device, dtype=torch.float32)
    k = torch.randn(b, h, t, dk, device=device, dtype=torch.float32)
    v = torch.randn(b, h, t, dv, device=device, dtype=torch.float32)
    return q, k, v


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V4_SHAPES)
def test_v4_smoke_finite_shape_dtype(device, shape):
    b, h, t, dk, dv = shape
    q, k, v = make_v4_inputs(shape, device)
    out, state = superl8.deltanet_chunk_int8_fwd(q, k, v)
    assert out.shape == (b, h, t, dv)
    assert state.shape == (b, h, dv, dk)
    assert out.dtype == torch.float32 and state.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(state).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V4_SHAPES)
def test_v4_matches_oracle_int8_quality(device, shape):
    """The int8 dp4a kernel must meet the SQNR/cos/rel-L1 gate vs the fp32
    ungated delta-rule oracle (NOT allclose)."""
    q, k, v = make_v4_inputs(shape, device)
    out_ref, state_ref = ungated_delta_rule_oracle(q, k, v)
    out_cuda, state_cuda = superl8.deltanet_chunk_int8_fwd(q, k, v)
    assert_int8_quality(
        out_cuda, out_ref, min_cos=V4_MIN_COS, max_rel_l1=V4_MAX_REL_L1,
        min_sqnr_db=V4_MIN_SQNR, what="v4 int8 out",
    )
    # State: cos + a looser SQNR floor (still well above the 20 dB minimum).
    assert cos_sim(state_cuda, state_ref) >= V4_MIN_COS
    assert sqnr_db(state_cuda, state_ref) >= V4_STATE_MIN_SQNR


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V4_SHAPES)
def test_v4_no_worse_than_int8_reference(device, shape):
    """The kernel's int8 quality must be no worse (within a small margin) than
    the Python int8 arithmetic twin — catches kernel-side quant/dp4a bugs that
    a raw-vs-oracle SQNR bar with slack could still pass."""
    q, k, v = make_v4_inputs(shape, device)
    out_ref, _ = ungated_delta_rule_oracle(q, k, v)
    out_pyi8, _ = ungated_delta_rule_int8_reference(q, k, v)
    out_cuda, _ = superl8.deltanet_chunk_int8_fwd(q, k, v)
    # Kernel and the Python int8 sim implement the same scheme; the kernel's
    # SQNR-vs-oracle must be within 3 dB of the reference's (rounding-mode /
    # summation-order differences only).
    assert sqnr_db(out_cuda, out_ref) >= sqnr_db(out_pyi8, out_ref) - 3.0
    # And the two int8 paths must agree tightly with each other.
    assert cos_sim(out_cuda, out_pyi8) >= 0.9995
    assert rel_l1(out_cuda, out_pyi8) <= 0.01


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V4_SHAPES)
def test_v4_matches_oracle_with_initial_state(device, shape):
    b, h, t, dk, dv = shape
    q, k, v = make_v4_inputs(shape, device)
    s0 = torch.randn(b, h, dv, dk, device=device, dtype=torch.float32)
    out_ref, state_ref = ungated_delta_rule_oracle(q, k, v, initial_state=s0)
    out_cuda, state_cuda = superl8.deltanet_chunk_int8_fwd(q, k, v, initial_state=s0)
    assert_int8_quality(
        out_cuda, out_ref, min_cos=V4_MIN_COS, max_rel_l1=V4_MAX_REL_L1,
        min_sqnr_db=V4_MIN_SQNR, what="v4 int8 out (initial_state)",
    )
    assert cos_sim(state_cuda, state_ref) >= V4_MIN_COS


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V4_SHAPES)
@pytest.mark.parametrize("split", [1, 2, 3])
def test_v4_compose_in_loop_meets_bar(device, shape, split):
    """int8 quant per-chunk means split≠full is NOT bitwise (chunk boundaries
    change the quantization). The int8 invariant is instead: chunked hand-off
    via the fp32 state still meets the int8 accuracy bar vs the fp32 oracle."""
    b, h, t, dk, dv = shape
    if t < split:
        pytest.skip("sequence shorter than requested split count")
    q, k, v = make_v4_inputs(shape, device)
    out_ref, _ = ungated_delta_rule_oracle(q, k, v)

    bounds = sorted(set(torch.linspace(0, t, split + 1).round().long().tolist()))
    out_chunks = []
    state = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi == lo:
            continue
        o_c, state = superl8.deltanet_chunk_int8_fwd(
            q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi], initial_state=state,
        )
        out_chunks.append(o_c)
    out_chunked = torch.cat(out_chunks, dim=2)
    assert_int8_quality(
        out_chunked, out_ref, min_cos=V4_MIN_COS, max_rel_l1=V4_MAX_REL_L1,
        min_sqnr_db=V4_MIN_SQNR, what="v4 int8 compose-in-loop",
    )


@pytest.mark.correctness
def test_v4_determinism(device):
    shape = (2, 2, 130, 64, 64)
    q, k, v = make_v4_inputs(shape, device)
    out1, state1 = superl8.deltanet_chunk_int8_fwd(q, k, v)
    out2, state2 = superl8.deltanet_chunk_int8_fwd(q, k, v)
    out3, state3 = superl8.deltanet_chunk_int8_fwd(q, k, v)
    assert torch.equal(out1, out2) and torch.equal(out2, out3)
    assert torch.equal(state1, state2) and torch.equal(state2, state3)


@pytest.mark.correctness
def test_v4_rejects_non_fp32(device):
    shape = (1, 1, 4, 32, 32)
    q, k, v = make_v4_inputs(shape, device)
    with pytest.raises(RuntimeError, match="fp32"):
        superl8.deltanet_chunk_int8_fwd(q.half(), k.half(), v.half())


@pytest.mark.perf
def test_v4_deltanet_chunk_int8_fwd_perf(device):
    b, h, t, dk, dv = 2, 16, 2048, 64, 64
    q, k, v = make_v4_inputs((b, h, t, dk, dv), device)
    ms = time_ms(lambda: superl8.deltanet_chunk_int8_fwd(q, k, v))
    assert_no_regression("deltanet_chunk_int8_fwd_v4.b2h16t2048d64.fp32", ms)


# ============================================================================
# v3.5 (issue #122): half2 (FP16x2) CUDA-core gated chunked DeltaNet
#
# Recasts the WY/UT chunk-matmul inner dot products (steps 4,5,6 in the v3
# kernel) onto __hfma2 half2 packed math with fp32 accumulation.  Everything
# numerically load-bearing stays fp32: log-space γ-cumprod, L2-norm, α/β gates,
# state, values, residuals.  This is the half2 CUDA-core pipe, NOT the dead
# fp16 tensor cores (per AGENTS.md silicon audit).
#
# Tolerance: fp16 path (per the issue — ai-bond fp16 tol upper limit).
# ============================================================================

V35_SHAPES = [
    (1, 2, 1, 32, 32),          # T=1 single-token edge case
    (2, 2, 66, 16, 32),         # T not multiple of chunk size
    (1, 2, 130, 64, 64),        # typical config, C ≤ 64
    (1, 1, 200, 32, 32),        # multiple full chunks + partial
    (1, 2, 130, 128, 64),       # Dk>Dv
    (1, 1, 130, 64, 128),       # Dv>Dk
    (1, 2, 130, 128, 128),      # large Dv,Dk, dynamic C
    (1, 1, 102, 63, 63),        # odd Dk, non-tile-multiple T
    (1, 1, 66, 34, 48),         # odd Dk=34, Dk≠Dv
]

# fp16-path tolerance per ai-bond convention (relative to fp32 oracle).
# Dk=128 dot products accumulate ~64 half2 FMAs whose rounding errors compound
# to ~3e-2 absolute; the tolerance absorbs that plus chunked-reassociation
# difference.  Matches ai-bond "≤ 2×fp16-baseline-err + 1e-5" at fp16-baseline
# err ≈ 0.015 (the measured half2-vs-fp32 ceiling on this fleet).
V35_H2_RTOL = 2e-2
V35_H2_ATOL = 3e-2


def make_v35_inputs(shape, device):
    b, h, t, dk, dv = shape
    q, k, v, beta_logit, decay_logit = make_inputs(shape, device)
    alpha, beta = gates_from_logits(beta_logit, decay_logit)
    return q, k, v, alpha, beta


@torch.no_grad()
def _call_v35(q, k, v, alpha, beta, *, initial_state=None):
    return superl8.deltanet_gated_chunk_h2_fwd(
        q, k, v, alpha, beta, initial_state=initial_state,
    )


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V35_SHAPES)
def test_v35_smoke_finite_shape_dtype(device, shape):
    b, h, t, dk, dv = shape
    q, k, v, alpha, beta = make_v35_inputs(shape, device)
    out, state = _call_v35(q, k, v, alpha, beta)
    assert out.shape == (b, h, t, dv)
    assert state.shape == (b, h, dv, dk)
    assert out.dtype == torch.float32 and state.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(state).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V35_SHAPES)
def test_v35_matches_fp32_oracle(device, shape):
    q, k, v, alpha, beta = make_v35_inputs(shape, device)
    out_ref, state_ref = gated_delta_rule_oracle(q, k, v, alpha, beta)
    out_cuda, state_cuda = _call_v35(q, k, v, alpha, beta)
    torch.testing.assert_close(out_cuda, out_ref, rtol=V35_H2_RTOL, atol=V35_H2_ATOL)
    torch.testing.assert_close(state_cuda, state_ref, rtol=V35_H2_RTOL, atol=V35_H2_ATOL)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V35_SHAPES)
def test_v35_matches_oracle_with_initial_state(device, shape):
    b, h, t, dk, dv = shape
    q, k, v, alpha, beta = make_v35_inputs(shape, device)
    s0 = torch.randn(b, h, dv, dk, device=device, dtype=torch.float32)
    out_ref, state_ref = gated_delta_rule_oracle(q, k, v, alpha, beta, initial_state=s0)
    out_cuda, state_cuda = _call_v35(q, k, v, alpha, beta, initial_state=s0)
    torch.testing.assert_close(out_cuda, out_ref, rtol=V35_H2_RTOL, atol=V35_H2_ATOL)
    torch.testing.assert_close(state_cuda, state_ref, rtol=V35_H2_RTOL, atol=V35_H2_ATOL)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V35_SHAPES)
@pytest.mark.parametrize("split", [1, 2, 3])
def test_v35_compose_in_loop_stability(device, shape, split):
    b, h, t, dk, dv = shape
    if t < split:
        pytest.skip("sequence shorter than requested split count")
    q, k, v, alpha, beta = make_v35_inputs(shape, device)

    out_full, _ = _call_v35(q, k, v, alpha, beta)

    bounds = sorted(set(torch.linspace(0, t, split + 1).round().long().tolist()))
    out_chunks = []
    state = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi == lo:
            continue
        o_c, state = _call_v35(
            q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi],
            alpha[:, :, lo:hi], beta[:, :, lo:hi],
            initial_state=state,
        )
        out_chunks.append(o_c)
    out_chunked = torch.cat(out_chunks, dim=2)
    torch.testing.assert_close(out_chunked, out_full, rtol=V35_H2_RTOL, atol=V35_H2_ATOL)


@pytest.mark.correctness
def test_v35_determinism(device):
    shape = (2, 2, 130, 64, 64)
    q, k, v, alpha, beta = make_v35_inputs(shape, device)
    out1, state1 = _call_v35(q, k, v, alpha, beta)
    out2, state2 = _call_v35(q, k, v, alpha, beta)
    out3, state3 = _call_v35(q, k, v, alpha, beta)
    assert torch.equal(out1, out2) and torch.equal(out2, out3)
    assert torch.equal(state1, state2) and torch.equal(state2, state3)


@pytest.mark.correctness
def test_v35_rejects_non_fp32(device):
    shape = (1, 1, 4, 32, 32)
    q, k, v, alpha, beta = make_v35_inputs(shape, device)
    with pytest.raises(RuntimeError, match="fp32"):
        _call_v35(q.half(), k.half(), v.half(), alpha, beta)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V35_SHAPES)
def test_v35_matches_chunked_reference(device, shape):
    q, k, v, alpha, beta = make_v35_inputs(shape, device)
    out_ref, state_ref = gated_chunked_reference(q, k, v, alpha, beta, C=64)
    out_cuda, state_cuda = _call_v35(q, k, v, alpha, beta)
    torch.testing.assert_close(out_cuda, out_ref, rtol=V35_H2_RTOL, atol=V35_H2_ATOL)
    torch.testing.assert_close(state_cuda, state_ref, rtol=V35_H2_RTOL, atol=V35_H2_ATOL)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", V35_SHAPES)
def test_v35_no_worse_than_v3_fp32(device, shape):
    """v3.5 half2 output must be no worse than v3 fp32 within the half2 tol
    (the whole point — this is the correctness ceiling for this path)."""
    q, k, v, alpha, beta = make_v35_inputs(shape, device)
    out_v3, state_v3 = superl8.deltanet_gated_chunk_fwd(q, k, v, alpha, beta)
    out_v35, state_v35 = _call_v35(q, k, v, alpha, beta)
    torch.testing.assert_close(out_v35, out_v3, rtol=V35_H2_RTOL, atol=V35_H2_ATOL)
    torch.testing.assert_close(state_v35, state_v3, rtol=V35_H2_RTOL, atol=V35_H2_ATOL)


@pytest.mark.perf
def test_v35_deltanet_gated_chunk_h2_fwd_perf(device):
    b, h, t, dk, dv = 2, 16, 2048, 64, 64
    q, k, v, alpha, beta = make_v35_inputs((b, h, t, dk, dv), device)
    ms = time_ms(lambda: _call_v35(q, k, v, alpha, beta))
    assert_no_regression("deltanet_gated_chunk_h2_fwd_v35.b2h16t2048d64.fp32_h2", ms)
