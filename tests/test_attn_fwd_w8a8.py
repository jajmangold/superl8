# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR4: full-W8A8 forward — int8 dp4a PV as well — tests written FIRST.

Contract:
  superl8.attn_int8_fwd(..., int8_pv=True) -> out
    V quantized PER-CHANNEL (scale must factor out of the key-dim sum — per-key
    scales cannot), P re-quantized per-row to int8 IN-LOOP (SDNQ recipe:
    p_scale = rowmax/127 with a tiny floor). Slightly looser bars than the
    QK-only path (P has 7 bits here), still SageAttention-grade.
Fallback rule (AGENTS.md): if this path cannot hold its accuracy gate, the op
falls back to fp16 PV — the gate decides, not ideology.
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle, sdpa_fp16
from tests.tolerances import assert_finite, assert_int8_quality

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, compare_report, time_ms  # noqa: E402

SHAPES = [
    (1, 2, 128, 64),
    (2, 4, 257, 64),
    (1, 2, 333, 128),
    (1, 8, 2048, 64),
]

# int8 PV budget: P carries ~7 bits -> allow modestly looser rel-L1 than QK-only.
PV_BARS = dict(min_cos=0.998, max_rel_l1=0.03, min_sqnr_db=18.0)


def make_qkv(shape, device):
    b, h, m, d = shape
    return (torch.randn(b, h, m, d, device=device, dtype=torch.float16) for _ in range(3))


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_w8a8_fwd_quality(device, shape, causal):
    q, k, v = make_qkv(shape, device)
    out = superl8.attn_int8_fwd(q, k, v, causal=causal, int8_pv=True)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    assert_int8_quality(out, oracle, what=f"w8a8 {shape} causal={causal}", **PV_BARS)


@pytest.mark.correctness
def test_w8a8_fwd_deterministic(device):
    q, k, v = make_qkv((2, 4, 512, 64), device)
    r0 = superl8.attn_int8_fwd(q, k, v, int8_pv=True)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_fwd(q, k, v, int8_pv=True), r0)


@pytest.mark.correctness
def test_w8a8_no_worse_than_bars_vs_fp16pv(device):
    """int8 PV may cost accuracy vs fp16 PV but must stay within its own bars;
    and the fp16-PV path must remain strictly better (sanity of the ladder)."""
    from tests.tolerances import rel_l1

    q, k, v = make_qkv((1, 8, 1024, 64), device)
    oracle = attention_fp32_oracle(q, k, v)
    out_fp16pv = superl8.attn_int8_fwd(q, k, v)
    out_w8a8 = superl8.attn_int8_fwd(q, k, v, int8_pv=True)
    assert rel_l1(out_fp16pv, oracle) <= rel_l1(out_w8a8, oracle) + 1e-4
    assert_int8_quality(out_w8a8, oracle, what="w8a8 ladder", **PV_BARS)


@pytest.mark.perf
@pytest.mark.parametrize("shape", [(2, 16, 2048, 64), (2, 16, 2048, 128)])
def test_w8a8_fwd_perf(device, shape):
    """int8-PV vs fp16-PV vs SDPA — honest report + regression gate."""
    b, h, m, d = shape
    q, k, v = make_qkv(shape, device)
    ours = time_ms(lambda: superl8.attn_int8_fwd(q, k, v, int8_pv=True))
    fp16pv = time_ms(lambda: superl8.attn_int8_fwd(q, k, v))
    sdpa = time_ms(lambda: sdpa_fp16(q, k, v))
    name = f"attn_w8a8_fwd.b{b}h{h}m{m}d{d}"
    print("\n" + compare_report(name, ours, {"fp16_PV": fp16pv, "sdpa_fp16": sdpa}))
    assert_no_regression(name, ours)
