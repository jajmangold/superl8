# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Fused per-row symmetric-RTN int8 activation quantizer.

`quantize_int8_rowwise` (the pure-torch prologue in `superl8/quant/core.py`) is
called once per int8 linear — ~200x/decode step — and expands to ~11 eager aten
ops + 3 dtype conversions each (`abs`, `amax`, `div`, `where`, `round`, `clamp`,
two `.float()`, a `.to(int8)`). Profiling a real Qwen3-0.6B decode step
(batch=8, real V100 idx-4) attributed the bulk of the ~3200 kernel launches/step
and the `aten::to`/`copy_` flood to this prologue; the GPU was 86% idle waiting on
CPU dispatch. `_C.quantize_i8_rowwise` collapses the whole prologue into ONE
kernel launch.

Gate (AGENTS.md): int8 paths never use `allclose`. This kernel is byte-identical
to the torch reference by construction — symmetric RTN with `Q_MAX=127`, and CUDA
`rintf` matches torch's round-half-to-even — so the correctness bar is EXACT
integer equality of the int8 output and exact equality of the fp32 scale, not a
tolerance. (`quantize_int8_rowwise` stays the CPU fallback + reference oracle.)
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8
from superl8.quant.core import quantize_int8_rowwise

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import time_ms

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
# Shapes incl. non-vec-multiple K (17, 4097) and M in {1,3,8,4096}; K=896 is a
# real Qwen3-0.6B projection width.
SHAPES = [(1, 896), (3, 896), (8, 896), (16, 4864), (4096, 896), (2, 17), (5, 4097)]


def _ref(x):
    """Independent eager oracle, scale squeezed to [M] like the fused op.

    Do not call ``quantize_int8_rowwise`` here: CUDA tensors now dispatch through
    the fused implementation under test, which would make this comparison
    circular.
    """
    scale = x.float().abs().amax(dim=-1, keepdim=True) / 127.0
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(x.float() / safe).clamp_(-127.0, 127.0).to(torch.int8)
    return q, safe.squeeze(-1)


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M,K", SHAPES)
def test_fused_quant_vs_reference(M, K, dtype):
    """All dtypes: bit-exact (same fp32 math across scalar and half2-vectorized paths)."""
    torch.manual_seed(0)
    x = (torch.randn(M, K, device="cuda", dtype=dtype) * 3.0)
    q, scale = superl8.quantize_i8_rowwise(x)
    ref_q, ref_scale = _ref(x)

    assert q.dtype == torch.int8 and q.shape == (M, K)
    assert scale.dtype == torch.float32 and scale.shape == (M,)

    assert torch.equal(q, ref_q), (
        f"int8 mismatch: {(q != ref_q).sum().item()} / {M * K} elems differ"
    )
    assert torch.equal(scale, ref_scale), "fp32 scale must be bit-exact vs reference"


@CUDA
@pytest.mark.correctness
def test_fused_quant_zero_row_and_outlier():
    # A zero row -> scale forced to 1.0, q all zeros (no div-by-zero / NaN).
    # An outlier row -> scale set by the single large magnitude.
    x = torch.zeros(3, 128, device="cuda", dtype=torch.float16)
    x[1, 40] = 50.0                 # single outlier in row 1
    q, scale = superl8.quantize_i8_rowwise(x)
    ref_q, ref_scale = _ref(x)
    assert scale[0].item() == 1.0 and torch.all(q[0] == 0)
    assert torch.equal(q, ref_q), f"int8 mismatch: {(q != ref_q).sum().item()} / 384 elems differ"
    assert torch.equal(scale, ref_scale), "fp32 scale must be bit-exact"


@CUDA
@pytest.mark.correctness
def test_fused_quant_deterministic():
    x = torch.randn(8, 896, device="cuda", dtype=torch.float16) * 4.0
    outs = [superl8.quantize_i8_rowwise(x) for _ in range(3)]
    for q, s in outs[1:]:
        assert torch.equal(q, outs[0][0]) and torch.equal(s, outs[0][1])


@CUDA
@pytest.mark.perf
@pytest.mark.parametrize("M,K", [(8, 896), (4096, 896)])
def test_fused_quant_faster_than_eager(M, K):
    x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 3.0
    fused = time_ms(lambda: superl8.quantize_i8_rowwise(x))
    eager = time_ms(lambda: quantize_int8_rowwise(x))
    # The win is launch/dispatch count (11 ops -> 1); at these sizes the fused
    # kernel must not be slower than the eager chain.
    assert fused <= eager * 1.5, f"fused {fused:.4f}ms vs eager {eager:.4f}ms"
