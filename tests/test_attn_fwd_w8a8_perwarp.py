# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR7: per-warp int8 quant granularity on the dp4a W8A8 attention forward.

SageAttention2 (superl8 issue #103) shows per-warp scales buy ~+1.4 cosine-sim
points over per-block on the P matrix. On Volta dp4a the per-warp granularity
maps to per-lane scales within each lane pair: each lane computes its own
p_max over its 32 key positions and uses that as its requantization scale,
halving the P-quant tile size from 64 keys to 32 keys with zero cross-lane
communication for the scale itself.

Contract:
  superl8.attn_int8_fwd(q, k, v, int8_pv=True, per_warp_quant=True) -> out
    Per-warp P requant: each lane keeps its own p_scale for 32 keys.
    PV dequant splits into two dp4a accumulators (my half, partner half),
    each dequantised by its own scale before the v_scale multiply.

  superl8.attn_int8_fwd(q, k, v, int8_pv=True, per_warp_quant=False) -> out
    Existing per-row (64-key) P requant — the baseline for A/B comparison.

Accuracy gate: per-warp must NEVER be weaker than per-row (finer granularity
cannot hurt). If it ever regresses vs per-row, that is a bug in the split-dp4a
dequant logic, not a tolerance to accept.
"""

import sys
from pathlib import Path

import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle
from tests.tolerances import assert_finite, assert_int8_quality, cos_sim, rel_l1, sqnr_db

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, compare_report, time_ms  # noqa: E402

SHAPES = [
    (1, 2, 128, 64),
    (2, 4, 257, 64),
    (1, 2, 333, 128),
    (1, 8, 2048, 64),
    (1, 2, 129, 64),   # non-tile-multiple
    (2, 1, 70, 128),   # non-tile-multiple
]

PV_BARS = dict(min_cos=0.998, max_rel_l1=0.03, min_sqnr_db=18.0)


def make_qkv(shape, device):
    b, h, m, d = shape
    return (torch.randn(b, h, m, d, device=device, dtype=torch.float16) for _ in range(3))


# ---------------------------------------------------------------------------
# Correctness: per-warp must meet the same quality bars as per-row
# ---------------------------------------------------------------------------

@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_perwarp_fwd_quality(device, shape, causal):
    """Per-warp quant meets the W8A8 accuracy bars (same as per-row)."""
    q, k, v = make_qkv(shape, device)
    out = superl8.attn_int8_fwd(q, k, v, causal=causal, int8_pv=True, per_warp_quant=True)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    assert_int8_quality(
        out, oracle,
        what=f"perwarp {shape} causal={causal}",
        **PV_BARS,
    )


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_perwarp_no_worse_than_perrow(device, shape, causal):
    """Per-warp must NOT be less accurate than per-row on every metric.

    Finer granularity (32-key segments vs 64-key full row) cannot reduce
    accuracy — this is a structural invariant, not an empirical hope.
    """
    q, k, v = make_qkv(shape, device)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    out_row = superl8.attn_int8_fwd(q, k, v, causal=causal, int8_pv=True, per_warp_quant=False)
    out_warp = superl8.attn_int8_fwd(q, k, v, causal=causal, int8_pv=True, per_warp_quant=True)

    cos_row, l1_row, sqnr_row = cos_sim(out_row, oracle), rel_l1(out_row, oracle), sqnr_db(out_row, oracle)
    cos_warp, l1_warp, sqnr_warp = cos_sim(out_warp, oracle), rel_l1(out_warp, oracle), sqnr_db(out_warp, oracle)

    assert cos_warp >= cos_row - 1e-6, (
        f"per-warp cos {cos_warp:.6f} < per-row cos {cos_row:.6f} "
        f"(shape={shape} causal={causal})"
    )
    assert l1_warp <= l1_row + 1e-6, (
        f"per-warp rel-L1 {l1_warp:.6f} > per-row rel-L1 {l1_row:.6f} "
        f"(shape={shape} causal={causal})"
    )
    assert sqnr_warp >= sqnr_row - 0.1, (
        f"per-warp SQNR {sqnr_warp:.1f} dB < per-row SQNR {sqnr_row:.1f} dB "
        f"(shape={shape} causal={causal})"
    )


@pytest.mark.correctness
def test_perwarp_deterministic(device):
    """Same input x3 -> bitwise identical output (per-warp path)."""
    q, k, v = make_qkv((2, 4, 512, 64), device)
    r0 = superl8.attn_int8_fwd(q, k, v, int8_pv=True, per_warp_quant=True)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_fwd(q, k, v, int8_pv=True, per_warp_quant=True), r0)


@pytest.mark.correctness
def test_perwarp_no_worse_than_bars_vs_fp16pv(device):
    """Same accuracy ladder as W8A8: int8 PV (even per-warp) may cost vs
    fp16 PV but must stay within its own bars; fp16 PV path must remain
    strictly better (sanity of the ladder comparison)."""
    from tests.tolerances import rel_l1

    q, k, v = make_qkv((1, 8, 1024, 64), device)
    oracle = attention_fp32_oracle(q, k, v)
    out_fp16pv = superl8.attn_int8_fwd(q, k, v)
    out_warp = superl8.attn_int8_fwd(q, k, v, int8_pv=True, per_warp_quant=True)
    assert rel_l1(out_fp16pv, oracle) <= rel_l1(out_warp, oracle) + 1e-4
    assert_int8_quality(out_warp, oracle, what="perwarp ladder", **PV_BARS)


# ---------------------------------------------------------------------------
# Accuracy measurement: report deltas for CI visibility
# ---------------------------------------------------------------------------

@pytest.mark.correctness
def test_perwarp_accuracy_report(device):
    """Report per-row vs per-warp metrics for human review in CI logs."""
    import math

    b, h, m, d = 2, 8, 1024, 64
    q, k, v = make_qkv((b, h, m, d), device)
    oracle = attention_fp32_oracle(q, k, v)

    out_row = superl8.attn_int8_fwd(q, k, v, int8_pv=True, per_warp_quant=False)
    out_warp = superl8.attn_int8_fwd(q, k, v, int8_pv=True, per_warp_quant=True)

    def _metrics(o):
        return cos_sim(o, oracle), rel_l1(o, oracle), sqnr_db(o, oracle)

    cr, lr, sr = _metrics(out_row)
    cw, lw, sw = _metrics(out_warp)

    print(f"\n  Per-row quant:  cos={cr:.6f}  rel-L1={lr:.6f}  SQNR={sr:.2f} dB")
    print(f"  Per-warp quant: cos={cw:.6f}  rel-L1={lw:.6f}  SQNR={sw:.2f} dB")
    print(f"  Delta (warp-row): cos={cw-cr:+.6f}  rel-L1={lw-lr:+.6f}  SQNR={sw-sr:+.2f} dB")


# ---------------------------------------------------------------------------
# Perf: measure overhead of the split-dp4a PV vs per-row
# ---------------------------------------------------------------------------

@pytest.mark.perf
@pytest.mark.parametrize("shape", [(2, 16, 2048, 64), (2, 16, 2048, 128)])
def test_perwarp_fwd_perf(device, shape):
    """Per-warp vs per-row vs fp16-PV vs SDPA — honest report + regression gate."""
    b, h, m, d = shape
    q, k, v = make_qkv(shape, device)

    ours = time_ms(lambda: superl8.attn_int8_fwd(q, k, v, int8_pv=True, per_warp_quant=True))
    pw_row = time_ms(lambda: superl8.attn_int8_fwd(q, k, v, int8_pv=True, per_warp_quant=False))
    fp16pv = time_ms(lambda: superl8.attn_int8_fwd(q, k, v))
    sdpa = time_ms(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

    name = f"attn_w8a8_perwarp.b{b}h{h}m{m}d{d}"
    print("\n" + compare_report(name, ours, {
        "per_row": pw_row,
        "fp16_PV": fp16pv,
        "sdpa_fp16": sdpa,
    }))
    assert_no_regression(name, ours)
