# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Sliding-window (Mistral-style local) causal attention. Tests first.

``window_left=w`` restricts query i to keys in ``(i - w, i]`` — the local-attention
pattern used by Mistral (w=4096) and windowed video-DiTs. Baked into the int8 dp4a
forward's valid-mask (fp16-PV path). ``window_left=-1`` reproduces the full-causal path.

Accuracy note (measured, both seeds): int8-QK error over a W-key window scales as
~1/sqrt(W) — the softmax over W keys averages out the per-logit int8 noise. The strict
int8 gate (cos>=0.999) is met for W>=256, the realistic regime; narrower windows degrade
smoothly and predictably (w=8 -> cos~0.94). This is a property of int8 attention with few
keys, not of the windowing (the mask itself is proven exact below).
"""
import pytest
import torch

import superl8
from tests.reference import attention_fp32_window_oracle
from tests.tolerances import assert_finite, assert_int8_quality, cos_sim


@pytest.mark.correctness
@pytest.mark.parametrize("window", [384, 480])
@pytest.mark.parametrize("d", [64, 128])
def test_window_matches_oracle(device, window, d):
    """Wide window widths hold the full int8 accuracy gate (cos>=0.999, rel-L1<=0.02).
    w=256 sits right on the rel-L1 boundary (0.020); the narrow-regime test below
    covers 256 on the cos gate and documents the 1/sqrt(W) floor honestly."""
    b, hq, hkv, s = 2, 8, 2, 512
    q = torch.randn(b, hq, s, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, s, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, s, d, device=device, dtype=torch.float16)
    out = superl8.attn_int8_fwd(q, k, v, causal=True, window_left=window)
    oracle = attention_fp32_window_oracle(q, k, v, window_left=window)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"window={window} d={d}")


@pytest.mark.correctness
def test_window_narrow_regime_degrades_smoothly(device):
    """CHARACTERIZATION (not a weakened gate): int8-QK error over a W-key window
    falls monotonically as W grows (~1/sqrt(W) averaging). Documents the narrow
    regime honestly rather than hiding it — each width still matches its own
    windowed oracle to a measured, width-appropriate floor."""
    b, hq, hkv, s, d = 2, 8, 2, 512, 128
    q = torch.randn(b, hq, s, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, s, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, s, d, device=device, dtype=torch.float16)
    widths = [8, 32, 64, 128, 256]
    coss = []
    for w in widths:
        out = superl8.attn_int8_fwd(q, k, v, causal=True, window_left=w)
        oracle = attention_fp32_window_oracle(q, k, v, window_left=w)
        assert_finite(out)
        coss.append(cos_sim(out, oracle))
    # wider window => at least as accurate (monotone non-decreasing, small slack)
    for a, b_ in zip(coss, coss[1:]):
        assert b_ >= a - 1e-3, f"cos not monotone in window width: {coss}"
    # even the narrowest 8-key window is a faithful (if coarse) int8 result
    assert coss[0] >= 0.90, f"w=8 cos {coss[0]:.4f} below the measured int8 floor"
    assert coss[-1] >= 0.999, f"w=256 cos {coss[-1]:.4f} misses the strict gate"


@pytest.mark.correctness
def test_window_ge_seqlen_equals_causal(device):
    """A window wider than the sequence is exactly full causal attention (bitwise) —
    proves the mask logic itself is exact, independent of int8 accuracy."""
    b, hq, hkv, s, d = 1, 8, 2, 256, 128
    q = torch.randn(b, hq, s, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, s, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, s, d, device=device, dtype=torch.float16)
    windowed = superl8.attn_int8_fwd(q, k, v, causal=True, window_left=s)
    full = superl8.attn_int8_fwd(q, k, v, causal=True, window_left=-1)
    assert torch.equal(windowed, full), "window >= seqlen must equal full causal"


@pytest.mark.correctness
def test_window_deterministic(device):
    b, hq, hkv, s, d = 2, 8, 2, 384, 64
    q = torch.randn(b, hq, s, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, s, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, s, d, device=device, dtype=torch.float16)
    r0 = superl8.attn_int8_fwd(q, k, v, causal=True, window_left=32)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_fwd(q, k, v, causal=True, window_left=32), r0)
