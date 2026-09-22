# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Fused RMSNorm (+ optional residual add, + Gemma unit-offset).

RMSNorm runs twice per decoder layer (pre-attn / pre-MLP) plus per-head QK-norm,
and in eager torch each call is float()/pow/mean(reduce)/rsqrt/mul/mul/cast (+ a
residual add) — the `reduce_kernel` cluster was ~a fifth of decode GPU time. This
kernel does the whole thing in one launch: fp32 sum-of-squares reduction
(`__shfl_xor_sync` + shared cross-warp), `rsqrt`, scale by the (optionally
1+w) weight, cast back. The reduction stays fp32 (AGENTS.md: numerically
load-bearing) — never quantized.

Gate: int8 conventions don't apply (this is fp16/bf16 in/out), but the fp32 mean
is an order-dependent float sum so it is NOT bit-exact vs torch's reduction — use
cos ≈ 1 / low rel-L1 vs the fp32 oracle (the difference is last-ULP reduction
order, so the bar is tight).
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import time_ms  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(1, 256), (8, 896), (16, 1024), (4096, 896), (2, 17), (5, 4097)]


def _ref(x, w, eps, residual=None, unit_offset=False):
    """Pure-torch reference == superl8serve RMSNorm._norm (superl8serve/layers/norm.py)."""
    if residual is not None:
        x = x + residual
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    wf = (1.0 + w.float()) if unit_offset else w.float()
    normed = (xf * wf).to(dt)
    return (normed, x) if residual is not None else normed


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _rl1(a, b):
    return (a.float() - b.float()).abs().sum().item() / b.float().abs().sum().clamp_min(1e-9).sum().item()


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M,D", SHAPES)
def test_rmsnorm_matches_reference(M, D, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype) * 0.2
    out = superl8.rmsnorm(x, w, 1e-6)
    ref = _ref(x, w, 1e-6)
    assert out.shape == (M, D) and out.dtype == dtype
    assert _cos(out, ref) > 0.9999, f"cos={_cos(out, ref)}"
    assert _rl1(out, ref) < 2e-3, f"relL1={_rl1(out, ref)}"


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("unit_offset", [False, True])
def test_rmsnorm_residual_and_unit_offset(unit_offset):
    torch.manual_seed(1)
    x = torch.randn(8, 896, device="cuda", dtype=torch.float16)
    r = torch.randn(8, 896, device="cuda", dtype=torch.float16)
    w = torch.randn(896, device="cuda", dtype=torch.float16) * 0.2
    normed, xr = superl8.rmsnorm(x, w, 1e-6, residual=r, unit_offset=unit_offset)
    ref_n, ref_xr = _ref(x, w, 1e-6, residual=r, unit_offset=unit_offset)
    assert torch.equal(xr, ref_xr), "residual sum (x+residual) must be exact"
    assert _cos(normed, ref_n) > 0.9999 and _rl1(normed, ref_n) < 2e-3


@CUDA
@pytest.mark.correctness
def test_rmsnorm_deterministic():
    x = torch.randn(8, 896, device="cuda", dtype=torch.float16)
    w = torch.randn(896, device="cuda", dtype=torch.float16) * 0.2
    outs = [superl8.rmsnorm(x, w, 1e-6) for _ in range(3)]
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("D", [128, 127, 130, 129, 1])
def test_rmsnorm_vec2_even_odd_boundary(D, dtype):
    """Issue #126: even D takes the half2-vectorized kernel, odd D the scalar fallback.
    Pin both sides of that dispatch boundary (D and D+1) plus D=1 (all-tail), and confirm
    the vectorized (even-D) path is still bitwise-deterministic."""
    torch.manual_seed(3)
    x = torch.randn(6, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype) * 0.2
    out = superl8.rmsnorm(x, w, 1e-6)
    ref = _ref(x, w, 1e-6)
    assert out.shape == (6, D) and out.dtype == dtype
    assert _cos(out, ref) > 0.9999, f"D={D} cos={_cos(out, ref)}"
    assert _rl1(out, ref) < 2e-3, f"D={D} relL1={_rl1(out, ref)}"
    if D % 2 == 0:  # vec2 path must stay deterministic
        again = superl8.rmsnorm(x, w, 1e-6)
        assert torch.equal(out, again), f"vec2 path non-deterministic at D={D}"


@CUDA
@pytest.mark.perf
@pytest.mark.parametrize("M,D", [(8, 896), (4096, 896)])
def test_rmsnorm_faster_than_eager(M, D):
    x = torch.randn(M, D, device="cuda", dtype=torch.float16)
    w = torch.randn(D, device="cuda", dtype=torch.float16) * 0.2
    fused = time_ms(lambda: superl8.rmsnorm(x, w, 1e-6))
    eager = time_ms(lambda: _ref(x, w, 1e-6))
    assert fused <= eager * 1.2, f"fused {fused:.4f}ms vs eager {eager:.4f}ms"
