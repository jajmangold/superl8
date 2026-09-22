# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR6: Stress-test W8A8 per-row activation quantization under massive outliers.

Reference: issue #112 — residual-stream activations on large models have
massive-activation channels (a few dims at 100-1000x) that set the per-row
quant scale for the whole token and crush the other channels into a few
int8 levels. K-smoothing (mean subtraction) handles K's channel outliers
(softmax-invariant), but Q is quantized per-row over the full hidden dim —
one Q spike can dominate the token's scale.

This test generates activations with extreme per-channel outliers, runs the
full int8 dp4a W8A8 path (int8 QK^T + int8 PV) with K-smoothing, and gates
on SQNR / cosine-sim / rel-L1 vs an fp32 PyTorch-SDPA oracle — NEVER allclose.

If the kernel FAILS the SQNR bar that is a REAL finding — report it. Do NOT
weaken the tolerance (AGENTS.md: "correctness before performance", "int8 paths
do NOT use allclose").
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle
from tests.tolerances import assert_int8_quality, cos_sim, rel_l1

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, time_ms  # noqa: E402

# int8 PV bars (from test_attn_fwd_w8a8.py: P carries ~7 bits).
PV_BARS = dict(min_cos=0.998, max_rel_l1=0.03, min_sqnr_db=18.0)

# Shapes: M in {1, 8, 16, non-tile-multiple}, D in {64, 128}.
# D must be in {32, 64, 128} per C++ kernel constraint (attn_w8a8_fwd.cu:81).
SHAPES = [
    (1, 2, 1, 64),       # M=1 decode
    (1, 2, 1, 128),      # M=1, D=128
    (1, 2, 8, 64),       # M=8, tile multiple
    (1, 2, 8, 128),      # M=8
    (1, 2, 16, 64),      # M=16, tile multiple
    (1, 2, 16, 128),     # M=16
    (1, 2, 15, 64),      # M=15 non-tile
    (2, 4, 33, 128),     # M=33 non-tile
    (1, 2, 257, 64),     # ragged M
    (1, 8, 2048, 64),    # larger prefill
    (1, 2, 1024, 128),   # D=128 prefill
]

# Outlier factors — spanning the 100-1000x range from issue #112.
OUTLIER_FACTORS = [50.0, 200.0, 1000.0]


def make_qkv(shape, device, outlier_factor=100.0):
    """Generate Q/K/V with massive per-channel outliers.

    Q has one outlier channel at ``outlier_factor`` x the background std.
    K also has one outlier channel (K-smoothing handles K's outliers).
    V is standard random.
    """
    b, h, m, d = shape
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16) * 0.3
    k = torch.randn(b, h, m, d, device=device, dtype=torch.float16) * 0.3
    v = torch.randn(b, h, m, d, device=device, dtype=torch.float16) * 0.3
    # Massive outlier channel in Q (the stress target — per-row quant over d).
    q[..., d // 2] *= outlier_factor
    # K channel outlier (K-smoothing should absorb this).
    k[..., d // 3] *= outlier_factor
    return q, k, v


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("outlier_factor", OUTLIER_FACTORS)
def test_w8a8_massive_outlier_quality(device, shape, outlier_factor):
    """Massive per-channel Q outliers must stay within int8 quality bars."""
    q, k, v = make_qkv(shape, device, outlier_factor)
    out = superl8.attn_int8_fwd(q, k, v, int8_pv=True)
    oracle = attention_fp32_oracle(q, k, v)
    assert_int8_quality(
        out, oracle,
        what=f"massive-outlier W8A8 {shape} outlier={outlier_factor}x",
        **PV_BARS,
    )


@pytest.mark.correctness
@pytest.mark.parametrize("shape", [(1, 2, 15, 64), (1, 2, 33, 128)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("outlier_factor", [200.0, 1000.0])
def test_w8a8_massive_outlier_causal(device, shape, causal, outlier_factor):
    """Causal masking with massive outliers — additional stress on near-tie rows."""
    q, k, v = make_qkv(shape, device, outlier_factor)
    out = superl8.attn_int8_fwd(q, k, v, causal=causal, int8_pv=True)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    assert_int8_quality(
        out, oracle,
        what=f"massive-outlier W8A8 {shape} causal={causal} outlier={outlier_factor}x",
        **PV_BARS,
    )


@pytest.mark.correctness
def test_w8a8_massive_outlier_multiple_spikes(device):
    """Multiple massive channels (e.g. top-4 of 64) — worst-case per-row crush."""
    b, h, m, d = 1, 2, 16, 64
    q, k, v = make_qkv((b, h, m, d), device, outlier_factor=200.0)
    # Spike 4 channels in Q (not just 1).
    for ch in [5, 13, 31, 50]:
        q[..., ch] *= 5.0
    out = superl8.attn_int8_fwd(q, k, v, int8_pv=True)
    oracle = attention_fp32_oracle(q, k, v)
    assert_int8_quality(out, oracle, what="massive-outlier multi-spike W8A8", **PV_BARS)


@pytest.mark.correctness
def test_w8a8_massive_outlier_rotation_measurement(device):
    """MEASUREMENT (not a gate): verify that Hadamard rotation collapses int8
    fidelity on massive-activation Q, per the issue #112 finding.
    Ref: sota-research-2026-07.md:395-405: int8 cos 0.99994 (no-rot) -> 0.948
    (+Hadamard) on real decode/boundary activations."""
    shape = (2, 4, 16, 64)
    q, k, v = make_qkv(shape, device, outlier_factor=500.0)
    out_no_rot = superl8.attn_int8_fwd(q, k, v, int8_pv=True, rotate=False)
    out_rot = superl8.attn_int8_fwd(q, k, v, int8_pv=True, rotate=True)
    oracle = attention_fp32_oracle(q, k, v)
    c_no = cos_sim(out_no_rot, oracle)
    c_rot = cos_sim(out_rot, oracle)
    l1_no = rel_l1(out_no_rot, oracle)
    l1_rot = rel_l1(out_rot, oracle)
    print(
        f"\n  Rotation effect (outlier=500x):"
        f"\n    no-rot: cos={c_no:.6f}  rel-L1={l1_no:.4f}"
        f"\n   +rot:   cos={c_rot:.6f}  rel-L1={l1_rot:.4f}"
    )
    # The no-rot cos must be strictly better than rot cos (verifying the claim).
    # This is a structural assertion that should always hold — if it fails the
    # issue's premise is wrong, which is also a valid finding.
    assert c_no >= c_rot, (
        f"EXPECTED rotation to hurt (smear spike across channels), but "
        f"no-rot cos={c_no:.6f} <= rot cos={c_rot:.6f}. This contradicts "
        f"issue #112 — investigate."
    )


@pytest.mark.correctness
def test_w8a8_massive_outlier_deterministic(device):
    """Same massive-outlier input x3 must produce bitwise-identical output."""
    q, k, v = make_qkv((2, 4, 16, 64), device, outlier_factor=500.0)
    r0 = superl8.attn_int8_fwd(q, k, v, int8_pv=True)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_fwd(q, k, v, int8_pv=True), r0)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", [(1, 4, 16, 64), (2, 8, 33, 128)])  # H_q > H_kv
def test_w8a8_outlier_gqa_fallback(device, shape):
    """GQA + outlier-gate fallback must not crash. When Q outlier domination
    trips the accuracy gate, attn_int8_fwd falls back to fp16 PyTorch SDPA; that
    SDPA call must pass ``enable_gqa`` when H_q != H_kv, or torch raises a
    head-dim size mismatch (regression surfaced by serve's GQA prefix-cache
    tests: 'size of tensor a (H_q) must match tensor b (H_kv)')."""
    from superl8.quant.core import detect_q_outlier_domination

    b, hq, m, d = shape
    hkv = hq // 2  # GQA: half as many KV heads
    q = torch.randn(b, hq, m, d, device=device, dtype=torch.float16) * 0.3
    k = torch.randn(b, hkv, m, d, device=device, dtype=torch.float16) * 0.3
    v = torch.randn(b, hkv, m, d, device=device, dtype=torch.float16) * 0.3
    q[..., d // 2] *= 1000.0  # massive outlier — must trip the gate
    assert detect_q_outlier_domination(q), "test setup: gate must fire for a real fallback"

    out = superl8.attn_int8_fwd(q, k, v)  # would crash pre-fix on the SDPA fallback
    oracle = attention_fp32_oracle(q, k, v)  # oracle handles GQA (repeat_interleave)
    assert out.shape == (b, hq, m, d)
    assert_int8_quality(out, oracle, what=f"GQA outlier-fallback {shape}", **PV_BARS)


@pytest.mark.perf
def test_w8a8_massive_outlier_perf(device):
    """Perf gate: massive outliers must not trigger a slow path or regression."""
    shape = (1, 8, 2048, 64)
    q, k, v = make_qkv(shape, device, outlier_factor=200.0)
    ms = time_ms(lambda: superl8.attn_int8_fwd(q, k, v, int8_pv=True))
    name = f"attn_w8a8_fwd_massive_outlier.b1h8m2048d64"
    assert_no_regression(name, ms)
