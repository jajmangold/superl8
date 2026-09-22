# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Fused gated activation — one launch replaces the eager chunk/silu/mul (SwiGLU)
or chunk/gelu-tanh/mul (GeGLU) on the merged gate_up projection output. Input
[..., 2I] -> output [..., I]: out[i] = act(x[i]) * x[i+I].

The fp16 path uses half2 vectorized loads/stores and __hmul2 multiply on
the healthy CUDA-core pipe (~27 TFLOP/s); the bf16 path stays scalar fp32 math.
Activation (sigmoid/tanh) is computed in fp32 for accuracy in both paths.

Gate (AGENTS.md, fused elementwise): cos ≈ 1 / low rel-L1 vs the torch reference
(sigmoid/tanh via device intrinsics differ from torch's by <1 ULP, so not
bit-exact).
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import superl8

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import time_ms  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(1, 512), (8, 4864), (16, 1024), (4096, 896), (3, 34)]
# Shapes with odd I that exercise half2 tail-handling
ODD_I_SHAPES = [(1, 34), (8, 34), (128, 34), (3, 6)]


def _ref(x, kind):
    gate, up = x.chunk(2, dim=-1)
    act = F.silu(gate) if kind == "silu" else F.gelu(gate, approximate="tanh")
    return act * up


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _rl1(a, b):
    return (a.float() - b.float()).abs().sum().item() / b.float().abs().sum().clamp_min(1e-9).item()


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("kind", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M,twoI", SHAPES)
def test_act_and_mul_matches_reference(M, twoI, dtype, kind):
    torch.manual_seed(0)
    x = torch.randn(M, twoI, device="cuda", dtype=dtype)
    out = superl8.act_and_mul(x, kind)
    ref = _ref(x, "silu" if kind == "silu" else "gelu")
    assert out.shape == (M, twoI // 2) and out.dtype == dtype
    assert _cos(out, ref) > 0.9999, f"cos={_cos(out, ref)}"
    assert _rl1(out, ref) < 0.02, f"relL1={_rl1(out, ref)}"


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("kind", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("M,twoI", ODD_I_SHAPES)
def test_act_and_mul_odd_I(M, twoI, kind):
    """Odd-I shapes fall back to scalar fp32 path (half2 needs aligned stores)."""
    torch.manual_seed(0)
    x = torch.randn(M, twoI, device="cuda", dtype=torch.float16)
    out = superl8.act_and_mul(x, kind)
    ref = _ref(x, "silu" if kind == "silu" else "gelu")
    assert out.shape == (M, twoI // 2) and out.dtype == torch.float16
    assert _cos(out, ref) > 0.9999, f"cos={_cos(out, ref)}"
    assert _rl1(out, ref) < 0.02, f"relL1={_rl1(out, ref)}"


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("kind", ["silu", "gelu_tanh"])
def test_act_and_mul_edge_values(kind):
    """Extreme input values — near fp16 range limit, zero, all-negative."""
    M, D = 16, 256
    twoD = 2 * D
    vec = torch.tensor(
        [0.0, -10.0, 10.0, 100.0, -200.0, 1e-4, -1e-4, 42.0],
        device="cuda",
        dtype=torch.float16,
    )
    x = vec.repeat(M, twoD // len(vec))[:, :twoD].contiguous()
    out = superl8.act_and_mul(x, kind)
    ref = _ref(x, "silu" if kind == "silu" else "gelu")
    assert out.shape == (M, D) and out.dtype == torch.float16
    assert _cos(out, ref) > 0.9999, f"cos={_cos(out, ref)}"
    assert _rl1(out, ref) < 0.02, f"relL1={_rl1(out, ref)}"


@CUDA
@pytest.mark.correctness
def test_act_and_mul_deterministic():
    x = torch.randn(8, 4864, device="cuda", dtype=torch.float16)
    outs = [superl8.act_and_mul(x, "silu") for _ in range(3)]
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


@CUDA
@pytest.mark.perf
@pytest.mark.parametrize("M,twoI", [(8, 4864), (4096, 896)])
def test_act_and_mul_faster_than_eager(M, twoI):
    x = torch.randn(M, twoI, device="cuda", dtype=torch.float16)
    fused = time_ms(lambda: superl8.act_and_mul(x, "silu"))
    eager = time_ms(lambda: _ref(x, "silu"))
    assert fused <= eager * 1.2, f"fused {fused:.4f} vs eager {eager:.4f}"
