# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Fused Gated-DeltaNet DECODE kernel (`superl8.deltanet_fused_decode`).

One launch that absorbs the per-layer DeltaNet *glue* — `_l2norm(q)`,
`_l2norm(k)`, GQA `repeat_interleave`, `beta = sigmoid(b_proj)`,
`g = -softplus(dt + dt_bias) * exp(A_log)`, the delta-rule recurrence, and the
gated output RMSNorm (`silu(z) * o`) — into the already-good register-state,
CUDA-graph-capturable decode recurrence. It exists to collapse the ~15-19 tiny
PyTorch op launches per linear-attn layer at decode into ~1.

The oracle is the *committed op sequence itself* (`_l2norm` + gate +
`deltanet_recurrent_decode` + `gated_rmsnorm_decode`), all fp32 — so the fused
kernel must reproduce it to fp32-reassociation tolerance (cos >= 0.9999,
rel-L1 <~ 1e-3), NOT merely "close". Both do identical fp32 math; only the
floating-point summation order differs.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import superl8
from tests.tolerances import cos_sim, rel_l1

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, time_ms  # noqa: E402


# (B, nk, nv, Dk, Dv). Decode is T==1. Qwen3.x runs nk=4 key heads / nv=16
# value heads (GQA rep=4), Dk=Dv=128. B=1 (single stream) and B=4 (batched).
FUSED_SHAPES = [
    (1, 4, 16, 128, 128),
    (4, 4, 16, 128, 128),
    (1, 2, 16, 128, 128),   # rep=8
    (2, 16, 16, 128, 128),  # rep=1 (no GQA expand)
]


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    # Mirror superl8serve.layers.linear_attn._l2norm exactly (clamp the NORM at 1e-6).
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def make_fused_inputs(shape, device, *, with_state=False):
    b, nk, nv, dk, dv = shape
    g = torch.Generator(device=device).manual_seed(1234)
    q = torch.randn(b, nk, 1, dk, device=device, dtype=torch.float32, generator=g)
    k = torch.randn(b, nk, 1, dk, device=device, dtype=torch.float32, generator=g)
    v = torch.randn(b, nv, 1, dv, device=device, dtype=torch.float32, generator=g)
    dt = torch.randn(b, nv, 1, device=device, dtype=torch.float32, generator=g)
    b_logit = torch.randn(b, nv, 1, device=device, dtype=torch.float32, generator=g)
    # A_log is the raw log-decay parameter; HF stores it so exp(A_log) is O(1).
    a_log = torch.randn(nv, device=device, dtype=torch.float32, generator=g) * 0.5
    dt_bias = torch.randn(nv, device=device, dtype=torch.float32, generator=g)
    gain = torch.randn(dv, device=device, dtype=torch.float32, generator=g)
    z = torch.randn(b, nv, 1, dv, device=device, dtype=torch.float32, generator=g)
    state = None
    if with_state:
        state = torch.randn(b, nv, dv, dk, device=device, dtype=torch.float32, generator=g)
    return q, k, v, dt, b_logit, a_log, dt_bias, gain, z, state


def reference_fused(q, k, v, dt, b_logit, a_log, dt_bias, gain, z, state, eps, q_scale=1.0):
    """The committed op sequence = fp32 oracle for the fused kernel."""
    b, nk, _, dk = q.shape
    nv, dv = v.shape[1], v.shape[3]
    rep = nv // nk
    # q_scale lands AFTER the L2-norm (HF gated-delta-rule readout scale).
    qn = (_l2norm(q.view(b, nk, dk)) * q_scale).view(b, nk, 1, dk)
    kn = _l2norm(k.view(b, nk, dk)).view(b, nk, 1, dk)
    qn = qn.repeat_interleave(rep, dim=1)
    kn = kn.repeat_interleave(rep, dim=1)
    beta = torch.sigmoid(b_logit.float())               # [B,nv,1]
    g = -F.softplus(dt.float() + dt_bias.view(1, -1, 1)) * a_log.exp().view(1, -1, 1)
    alpha = g.exp()                                      # [B,nv,1]
    o, final_state = superl8.deltanet_recurrent_decode(
        qn, kn, v, alpha.reshape(b, nv, 1), beta.reshape(b, nv, 1), initial_state=state
    )
    # gated RMSNorm per (b, head) row over the vd axis.
    orows = o.reshape(b * nv, dv)
    zrows = z.reshape(b * nv, dv) if z is not None else None
    normed = superl8.gated_rmsnorm_decode(orows, gain, zrows, eps).view(b, nv, 1, dv)
    return normed, final_state


def call_fused(q, k, v, dt, b_logit, a_log, dt_bias, gain, z, state, eps, q_scale=1.0):
    return superl8.deltanet_fused_decode(
        q, k, v, dt, b_logit, a_log, dt_bias, gain,
        z=z, initial_state=state, q_scale=q_scale, eps=eps,
    )


EPS = 1e-6


@pytest.mark.correctness
@pytest.mark.parametrize("shape", FUSED_SHAPES)
def test_fused_smoke_finite_shape_dtype(device, shape):
    b, nk, nv, dk, dv = shape
    inp = make_fused_inputs(shape, device)
    out, state = call_fused(*inp, EPS)
    assert out.shape == (b, nv, 1, dv)
    assert state.shape == (b, nv, dv, dk)
    assert out.dtype == torch.float32 and state.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(state).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", FUSED_SHAPES)
@pytest.mark.parametrize("with_z", [True, False])
@pytest.mark.parametrize("q_scale", [1.0, 128 ** -0.5])  # 1.0 and the HF readout scale
def test_fused_matches_op_sequence(device, shape, with_z, q_scale):
    q, k, v, dt, b_logit, a_log, dt_bias, gain, z, state = make_fused_inputs(shape, device)
    zz = z if with_z else None
    out_ref, st_ref = reference_fused(
        q, k, v, dt, b_logit, a_log, dt_bias, gain, zz, state, EPS, q_scale=q_scale
    )
    out_cuda, st_cuda = call_fused(
        q, k, v, dt, b_logit, a_log, dt_bias, gain, zz, state, EPS, q_scale=q_scale
    )
    assert cos_sim(out_cuda, out_ref) >= 0.9999
    assert rel_l1(out_cuda, out_ref) <= 1e-3
    # state is the load-bearing carry; hold it to the same fp32 bar.
    assert cos_sim(st_cuda, st_ref) >= 0.9999
    assert rel_l1(st_cuda, st_ref) <= 1e-3


@pytest.mark.correctness
@pytest.mark.parametrize("shape", FUSED_SHAPES)
def test_fused_matches_op_sequence_with_initial_state(device, shape):
    q, k, v, dt, b_logit, a_log, dt_bias, gain, z, state = make_fused_inputs(
        shape, device, with_state=True
    )
    out_ref, st_ref = reference_fused(q, k, v, dt, b_logit, a_log, dt_bias, gain, z, state, EPS)
    out_cuda, st_cuda = call_fused(q, k, v, dt, b_logit, a_log, dt_bias, gain, z, state, EPS)
    assert cos_sim(out_cuda, out_ref) >= 0.9999
    assert rel_l1(out_cuda, out_ref) <= 1e-3
    assert cos_sim(st_cuda, st_ref) >= 0.9999
    assert rel_l1(st_cuda, st_ref) <= 1e-3


@pytest.mark.correctness
def test_fused_determinism(device):
    shape = (1, 4, 16, 128, 128)
    inp = make_fused_inputs(shape, device, with_state=True)
    o1, s1 = call_fused(*inp, EPS)
    o2, s2 = call_fused(*inp, EPS)
    o3, s3 = call_fused(*inp, EPS)
    assert torch.equal(o1, o2) and torch.equal(o2, o3)
    assert torch.equal(s1, s2) and torch.equal(s2, s3)


@pytest.mark.correctness
def test_fused_rejects_non_128(device):
    # Fused decode is the Dk=Dv=128 specialization; other dims fall back to the
    # eager op sequence in the caller, so the kernel itself must reject them.
    shape = (1, 4, 16, 64, 64)
    inp = make_fused_inputs(shape, device)
    with pytest.raises(RuntimeError, match="128"):
        call_fused(*inp, EPS)


@pytest.mark.correctness
def test_fused_rejects_non_fp32(device):
    shape = (1, 4, 16, 128, 128)
    q, k, v, dt, b_logit, a_log, dt_bias, gain, z, state = make_fused_inputs(shape, device)
    with pytest.raises(RuntimeError, match="fp32"):
        call_fused(q.half(), k, v, dt, b_logit, a_log, dt_bias, gain, z, state, EPS)


@pytest.mark.perf
def test_fused_decode_perf(device):
    # Qwen3.x per-layer decode shape: B=1, nk=4, nv=16, Dk=Dv=128.
    shape = (1, 4, 16, 128, 128)
    inp = make_fused_inputs(shape, device)
    ms = time_ms(lambda: call_fused(*inp, EPS))
    assert_no_regression("deltanet_fused_decode.b1nk4nv16d128.fp32", ms)
