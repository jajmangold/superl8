# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR5: autograd integration + backward — tests written FIRST.

Contract:
  superl8.attn(q, k, v, causal=False, scale=None) -> out    # differentiable
    a torch.autograd.Function; forward = the int8 dp4a kernel (+ saved LSE),
    backward returns dq,dk,dv. Backward implements the EXACT-attention gradient
    (quantization is straight-through), so it is graded against the fp32 oracle
    with relative-to-fp16 bounds — NOT gradcheck'd on the quantized forward.
Numerics (AGENTS.md): grads err <= 3x the fp16-baseline grad err + 1e-4; int8
paths use SQNR/cos/rel-L1 bars because grads are noisier under quantization.
"""

import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle
from tests.tolerances import cos_sim, rel_l1, sqnr_db

SHAPES = [(1, 2, 128, 64), (2, 4, 256, 64), (1, 2, 257, 64), (1, 2, 192, 128)]

# gradients are noisier than activations under int8 — looser than the fwd bars
GRAD_BARS = dict(min_cos=0.99, max_rel_l1=0.09, min_sqnr_db=14.0)


def grads_of(fn, q, k, v, causal):
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    out = fn(q, k, v, causal=causal)
    g = torch.randn_like(out)
    out.backward(g)
    return q.grad, k.grad, v.grad, g


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_attn_backward_matches_oracle(device, shape, causal):
    q, k, v = (torch.randn(shape, device=device, dtype=torch.float16) for _ in range(3))
    dq, dk, dv, g = grads_of(superl8.attn, q, k, v, causal)

    # reference grads from the fp32 oracle with the SAME upstream grad g
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    ref = attention_fp32_oracle(qf, kf, vf, causal=causal)
    ref.backward(g.float())

    for name, got, exp in [("dq", dq, qf.grad), ("dk", dk, kf.grad), ("dv", dv, vf.grad)]:
        assert got is not None and got.shape == exp.shape
        c, l1, s = cos_sim(got, exp), rel_l1(got, exp), sqnr_db(got, exp)
        assert c >= GRAD_BARS["min_cos"], f"{name} {shape} causal={causal}: cos {c:.4f}"
        assert l1 <= GRAD_BARS["max_rel_l1"], f"{name} {shape} causal={causal}: rel-L1 {l1:.4f}"
        assert s >= GRAD_BARS["min_sqnr_db"], f"{name} {shape} causal={causal}: SQNR {s:.1f}dB"


@pytest.mark.correctness
def test_attn_backward_gradcheck_reference(device):
    """Gradcheck the backward MATH on the non-quantized fp64 reference path
    (superl8.attn_ref) — validates the analytic gradient independent of int8 noise."""
    torch.manual_seed(0)
    b, h, m, d = 1, 1, 16, 16
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float64, requires_grad=True)
    k = torch.randn(b, h, m, d, device=device, dtype=torch.float64, requires_grad=True)
    v = torch.randn(b, h, m, d, device=device, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda q, k, v: superl8.attn_ref(q, k, v), (q, k, v), atol=1e-4, rtol=1e-3
    )


@pytest.mark.correctness
def test_attn_backward_uses_saved_lse_no_einsum(device):
    """Backward must use saved LSE, NOT recompute via dense [B,H,M,N] einsum.
    When LSE is saved from the forward and threaded through, the backward
    should never call torch.einsum to materialize a dense attention score."""
    shape = (1, 1, 256, 64)
    q, k, v = (
        torch.randn(shape, device=device, dtype=torch.float16, requires_grad=True) for _ in range(3)
    )
    out = superl8.attn(q, k, v)
    g = torch.randn_like(out)

    real_einsum = torch.einsum

    def _fail_einsum(*args, **kwargs):
        raise AssertionError(
            f"torch.einsum called during backward — "
            f"LSE was not saved/used properly, dense [B,H,M,N] recompute triggered"
        )

    torch.einsum = _fail_einsum
    try:
        out.backward(g)
    finally:
        torch.einsum = real_einsum

    for t, name in [(q, "q"), (k, "k"), (v, "v")]:
        assert t.grad is not None, f"{name}.grad is None"


@pytest.mark.correctness
def test_attn_is_autograd_function(device):
    q, k, v = (
        torch.randn(1, 2, 128, 64, device=device, dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    )
    out = superl8.attn(q, k, v)
    assert out.requires_grad and out.grad_fn is not None
    out.sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
