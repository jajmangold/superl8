# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Track-2 (issue #42): MiniMax Lightning (un-gated linear) attention — fp32
reference oracle + fp CUDA kernel.

Staged like the DeltaNet kernel test suite:
  v1 (this file) — CUDA port of the sequential fp32 Lightning recurrence
    (one block per (batch,head), state in shared memory).
  v2 (next PR)   — int8 dp4a variant.

The oracle lives in tests/reference_lightning.py.  Both the oracle and the
CUDA kernel are validated here.
"""

import sys
from pathlib import Path

import pytest
import torch

import superl8
from tests.reference_lightning import lightning_attn_oracle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, time_ms  # noqa: E402


def _sqnr(ref, test):
    noise = (test.float() - ref.float()).pow(2).mean()
    signal = ref.float().pow(2).mean()
    return float(10.0 * torch.log10(signal / (noise + 1e-12)))


def _cosine_sim(ref, test):
    return float(torch.nn.functional.cosine_similarity(
        test.float().flatten(-2, -1), ref.float().flatten(-2, -1), dim=-1
    ).mean())


def _rel_l1(ref, test):
    return float(
        (test.float() - ref.float()).abs().sum() / ref.float().abs().sum().clamp_min(1e-12)
    )

# (B, H, T, Dk, Dv) — the naive CUDA kernel is O(T) sequential, so keep
# T moderate. Includes values NOT tile multiples of 64.
SHAPES = [
    (1, 2, 1, 32, 32),       # T=1 single-token edge case
    (2, 2, 5, 16, 32),       # small, Dk != Dv
    (1, 1, 63, 32, 32),      # T not multiple of chunk sizes
    (1, 2, 130, 64, 64),     # typical head dim, T not tile-multiple
    (1, 1, 256, 64, 64),     # longer sequence
    (1, 2, 130, 128, 64),    # Dk > Dv
    (1, 1, 130, 64, 128),    # Dv > Dk
    (2, 2, 130, 128, 128),   # large Dv,Dk; batch=2 heads=2
    (1, 2, 130, 32, 32),     # small Dk,Dv with longer T
]


def make_inputs(shape, device):
    b, h, t, dk, dv = shape
    q = torch.randn(b, h, t, dk, device=device, dtype=torch.float32)
    k = torch.randn(b, h, t, dk, device=device, dtype=torch.float32)
    v = torch.randn(b, h, t, dv, device=device, dtype=torch.float32)
    return q, k, v


# ============================================================================
# Oracle self-consistency
# ============================================================================

@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_oracle_smoke_finite_shape_dtype(device, shape):
    b, h, t, dk, dv = shape
    q, k, v = make_inputs(shape, device)
    out, state = lightning_attn_oracle(q, k, v)
    assert out.shape == (b, h, t, dv)
    assert state.shape == (b, h, dv, dk)
    assert out.dtype == torch.float32 and state.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(state).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_oracle_smoke_with_initial_state(device, shape):
    b, h, t, dk, dv = shape
    q, k, v = make_inputs(shape, device)
    s0 = torch.randn(b, h, dv, dk, device=device, dtype=torch.float32)
    out, state = lightning_attn_oracle(q, k, v, initial_state=s0)
    assert out.shape == (b, h, t, dv)
    assert state.shape == (b, h, dv, dk)
    assert torch.isfinite(out).all() and torch.isfinite(state).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_oracle_matches_quadratic_closed_form(device, shape):
    """The recurrent oracle must match the O(T^2) closed form at fp32
    reassociation tolerance (independent check)."""
    from tests.reference_lightning import lightning_attn_quadratic_closed_form

    q, k, v = make_inputs(shape, device)
    out_rec, state_rec = lightning_attn_oracle(q, k, v)
    out_quad, state_quad = lightning_attn_quadratic_closed_form(q, k, v)
    torch.testing.assert_close(out_rec, out_quad, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(state_rec, state_quad, rtol=2e-4, atol=2e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_oracle_matches_quadratic_with_initial_state(device, shape):
    from tests.reference_lightning import lightning_attn_quadratic_closed_form

    b, h, t, dk, dv = shape
    q, k, v = make_inputs(shape, device)
    s0 = torch.randn(b, h, dv, dk, device=device, dtype=torch.float32)
    out_rec, state_rec = lightning_attn_oracle(q, k, v, initial_state=s0)
    out_quad, state_quad = lightning_attn_quadratic_closed_form(q, k, v, initial_state=s0)
    torch.testing.assert_close(out_rec, out_quad, rtol=2e-4, atol=2e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("split", [1, 2, 3])
def test_oracle_compose_in_loop_stability(device, shape, split):
    """Splitting a sequence and feeding the state across calls must produce
    the same output as a single call (associativity of the recurrence)."""
    b, h, t, dk, dv = shape
    if t < split:
        pytest.skip("sequence shorter than requested split count")
    q, k, v = make_inputs(shape, device)

    out_full, _ = lightning_attn_oracle(q, k, v)

    bounds = sorted(set(torch.linspace(0, t, split + 1).round().long().tolist()))
    out_chunks = []
    state = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi == lo:
            continue
        o_c, state = lightning_attn_oracle(
            q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi], initial_state=state,
        )
        out_chunks.append(o_c)
    out_chunked = torch.cat(out_chunks, dim=2)
    torch.testing.assert_close(out_chunked, out_full, rtol=1e-4, atol=1e-4)


# ============================================================================
# CUDA kernel tests  (v1 naive sequential kernel via superl8.lightning_attn_fwd)
# ============================================================================

def _call_kernel(q, k, v, *, initial_state=None):
    """Call the CUDA kernel.  Will raise AttributeError until wired into
    superl8 — that is expected (TDD: test first)."""
    return superl8.lightning_attn_fwd(q, k, v, initial_state=initial_state)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_kernel_smoke_finite_shape_dtype(device, shape):
    b, h, t, dk, dv = shape
    q, k, v = make_inputs(shape, device)
    out, state = _call_kernel(q, k, v)
    assert out.shape == (b, h, t, dv)
    assert state.shape == (b, h, dv, dk)
    assert out.dtype == torch.float32 and state.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(state).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_kernel_matches_fp32_oracle(device, shape):
    q, k, v = make_inputs(shape, device)
    out_ref, state_ref = lightning_attn_oracle(q, k, v)
    out_cuda, state_cuda = _call_kernel(q, k, v)
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_kernel_matches_oracle_with_initial_state(device, shape):
    b, h, t, dk, dv = shape
    q, k, v = make_inputs(shape, device)
    s0 = torch.randn(b, h, dv, dk, device=device, dtype=torch.float32)
    out_ref, state_ref = lightning_attn_oracle(q, k, v, initial_state=s0)
    out_cuda, state_cuda = _call_kernel(q, k, v, initial_state=s0)
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_cuda, state_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("split", [1, 2, 3])
def test_kernel_compose_in_loop_stability(device, shape, split):
    b, h, t, dk, dv = shape
    if t < split:
        pytest.skip("sequence shorter than requested split count")
    q, k, v = make_inputs(shape, device)

    out_full, _ = _call_kernel(q, k, v)

    bounds = sorted(set(torch.linspace(0, t, split + 1).round().long().tolist()))
    out_chunks = []
    state = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi == lo:
            continue
        o_c, state = _call_kernel(
            q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi], initial_state=state,
        )
        out_chunks.append(o_c)
    out_chunked = torch.cat(out_chunks, dim=2)
    torch.testing.assert_close(out_chunked, out_full, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
def test_kernel_determinism(device):
    shape = (2, 2, 130, 64, 64)
    q, k, v = make_inputs(shape, device)
    out1, state1 = _call_kernel(q, k, v)
    out2, state2 = _call_kernel(q, k, v)
    out3, state3 = _call_kernel(q, k, v)
    assert torch.equal(out1, out2) and torch.equal(out2, out3)
    assert torch.equal(state1, state2) and torch.equal(state2, state3)


@pytest.mark.correctness
def test_kernel_rejects_non_fp32(device):
    shape = (1, 1, 4, 32, 32)
    q, k, v = make_inputs(shape, device)
    with pytest.raises(RuntimeError, match="fp32"):
        _call_kernel(q.half(), k.half(), v.half())


@pytest.mark.correctness
def test_kernel_rejects_dim_over_128(device):
    shape = (1, 1, 4, 256, 32)
    q, k, v = make_inputs(shape, device)
    with pytest.raises(RuntimeError, match="128"):
        _call_kernel(q, k, v)


@pytest.mark.perf
def test_lightning_attn_fwd_perf(device):
    """Perf gate (soft-skip until baseline committed)."""
    b, h, t, dk, dv = 2, 16, 2048, 64, 64
    q, k, v = make_inputs((b, h, t, dk, dv), device)
    ms = time_ms(lambda: _call_kernel(q, k, v))
    assert_no_regression("lightning_attn_fwd.b2h16t2048d64.fp32", ms)


# ============================================================================
# v2 (issue #42): int8 dp4a Lightning kernel tests
# ============================================================================

INT8_SHAPES = SHAPES  # same shape sweep as the fp kernel


def _call_int8_kernel(q, k, v, *, initial_state=None):
    """Call the int8 CUDA kernel — fails until the binding is wired."""
    return superl8.lightning_attn_int8_fwd(q, k, v, initial_state=initial_state)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", INT8_SHAPES)
def test_lightning_int8_smoke_finite_shape_dtype(device, shape):
    b, h, t, dk, dv = shape
    q, k, v = make_inputs(shape, device)
    out, state = _call_int8_kernel(q, k, v)
    assert out.shape == (b, h, t, dv)
    assert state.shape == (b, h, dv, dk)
    assert out.dtype == torch.float32 and state.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(state).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", INT8_SHAPES)
def test_lightning_int8_matches_fp32_oracle(device, shape):
    """int8 kernel must match the fp32 oracle at int8 tolerance
    (cos >= 0.999, rel-L1 <= 0.02, SQNR >= 30 dB)."""
    q, k, v = make_inputs(shape, device)
    out_ref, state_ref = lightning_attn_oracle(q, k, v)
    out_i8, state_i8 = _call_int8_kernel(q, k, v)

    cos_out = _cosine_sim(out_ref, out_i8)
    cos_st = _cosine_sim(state_ref, state_i8)
    rl1_out = _rel_l1(out_ref, out_i8)
    rl1_st = _rel_l1(state_ref, state_i8)
    sqnr_out = _sqnr(out_ref, out_i8)

    assert cos_out >= 0.999, f"out cos={cos_out:.6f} < 0.999"
    assert cos_st >= 0.999, f"state cos={cos_st:.6f} < 0.999"
    assert rl1_out <= 0.02, f"out rel-L1={rl1_out:.6f} > 0.02"
    assert rl1_st <= 0.02, f"state rel-L1={rl1_st:.6f} > 0.02"
    assert sqnr_out >= 30.0, f"out SQNR={sqnr_out:.2f} dB < 30 dB"


@pytest.mark.correctness
@pytest.mark.parametrize("shape", INT8_SHAPES)
def test_lightning_int8_matches_fp32_kernel(device, shape):
    """int8 kernel must match the fp kernel at int8 tolerance."""
    q, k, v = make_inputs(shape, device)
    out_fp, state_fp = _call_kernel(q, k, v)
    out_i8, state_i8 = _call_int8_kernel(q, k, v)

    cos_out = _cosine_sim(out_fp, out_i8)
    rl1_out = _rel_l1(out_fp, out_i8)
    sqnr_out = _sqnr(out_fp, out_i8)

    assert cos_out >= 0.999, f"out cos={cos_out:.6f} < 0.999 vs fp kernel"
    assert sqnr_out >= 30.0, f"out SQNR={sqnr_out:.2f} dB < 30 dB vs fp kernel"


@pytest.mark.correctness
@pytest.mark.parametrize("shape", INT8_SHAPES)
def test_lightning_int8_with_initial_state(device, shape):
    b, h, t, dk, dv = shape
    q, k, v = make_inputs(shape, device)
    s0 = torch.randn(b, h, dv, dk, device=device, dtype=torch.float32)
    out_ref, state_ref = lightning_attn_oracle(q, k, v, initial_state=s0)
    out_i8, state_i8 = _call_int8_kernel(q, k, v, initial_state=s0)

    cos_out = _cosine_sim(out_ref, out_i8)
    cos_st = _cosine_sim(state_ref, state_i8)
    sqnr_out = _sqnr(out_ref, out_i8)

    assert cos_out >= 0.999, f"out cos={cos_out:.6f} < 0.999"
    assert cos_st >= 0.999, f"state cos={cos_st:.6f} < 0.999"
    assert sqnr_out >= 30.0, f"out SQNR={sqnr_out:.2f} dB < 30 dB"


@pytest.mark.correctness
def test_lightning_int8_determinism(device):
    shape = (2, 2, 130, 64, 64)
    q, k, v = make_inputs(shape, device)
    out1, state1 = _call_int8_kernel(q, k, v)
    out2, state2 = _call_int8_kernel(q, k, v)
    out3, state3 = _call_int8_kernel(q, k, v)
    assert torch.equal(out1, out2) and torch.equal(out2, out3)
    assert torch.equal(state1, state2) and torch.equal(state2, state3)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", INT8_SHAPES)
@pytest.mark.parametrize("split", [1, 2, 3])
def test_lightning_int8_compose_in_loop_stability(device, shape, split):
    b, h, t, dk, dv = shape
    if t < split:
        pytest.skip("sequence shorter than requested split count")
    q, k, v = make_inputs(shape, device)

    out_full, _ = _call_int8_kernel(q, k, v)

    bounds = sorted(set(torch.linspace(0, t, split + 1).round().long().tolist()))
    out_chunks = []
    state = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi == lo:
            continue
        o_c, state = _call_int8_kernel(
            q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi], initial_state=state,
        )
        out_chunks.append(o_c)
    out_chunked = torch.cat(out_chunks, dim=2)

    cos = _cosine_sim(out_full, out_chunked)
    sqnr_val = _sqnr(out_full, out_chunked)
    assert cos >= 0.999, f"composed cos={cos:.6f} < 0.999"
    assert sqnr_val >= 30.0, f"composed SQNR={sqnr_val:.2f} dB < 30 dB"


@pytest.mark.correctness
def test_lightning_int8_rejects_non_fp32(device):
    shape = (1, 1, 4, 32, 32)
    q, k, v = make_inputs(shape, device)
    with pytest.raises(RuntimeError, match="fp32"):
        _call_int8_kernel(q.half(), k.half(), v.half())


@pytest.mark.perf
def test_lightning_attn_int8_fwd_perf(device):
    """Perf gate for the int8 dp4a Lightning kernel."""
    b, h, t, dk, dv = 2, 16, 2048, 64, 64
    q, k, v = make_inputs((b, h, t, dk, dv), device)
    ms = time_ms(lambda: _call_int8_kernel(q, k, v))
    assert_no_regression("lightning_attn_int8_fwd.b2h16t2048d64", ms)
