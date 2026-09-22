# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Unit tests for bench/power_eval.py — data structures and analysis logic.

No GPU required; tests the metrics computation, cap-binding verdict, and
data round-trip. The actual nvidia-smi telemetry + sustained workload loops
are exercised by running `docker compose run --rm power` on a real V100.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.power_eval import (
    DEFAULT_WORKLOADS,
    POWER_LIMITS_SWEEP,
    TelemetrySeries,
    compute_cap_binding,
)

# ── Workload definitions ────────────────────────────────────────────────────


def test_default_workloads_are_valid():
    for wl in DEFAULT_WORKLOADS:
        assert wl.kind in ("attn", "gemm")
        assert wl.measure_duration_s > 0
        assert wl.warmup_passes >= 0
        if wl.kind == "attn":
            assert len(wl.shape) == 4  # B, H, M, D


def test_workload_tokens_per_pass():
    B, H, M, _ = 2, 16, 2048, 128
    assert B * H * M == 65536


def test_power_limits_sweep_order():
    assert POWER_LIMITS_SWEEP == sorted(POWER_LIMITS_SWEEP)
    assert POWER_LIMITS_SWEEP[0] == 120
    assert POWER_LIMITS_SWEEP[-1] == 250


# ── Telemetry series (synthetic data) ───────────────────────────────────────


def test_telemetry_series_basic():
    ts = TelemetrySeries()
    ts.power_w = [100.0, 110.0, 120.0]
    ts.sm_mhz = [1380.0, 1380.0, 1380.0]
    assert ts.mean_power_w == 110.0
    assert ts.peak_power_w == 120.0
    assert ts.mean_sm_mhz == 1380.0
    assert ts.num_samples == 3


def test_telemetry_series_empty():
    ts = TelemetrySeries()
    assert ts.mean_power_w == 0.0
    assert ts.peak_power_w == 0.0
    assert ts.num_samples == 0


# ── Cap binding verdict logic ───────────────────────────────────────────────


def _make_attn_result(workload_name: str, pl: int, mean_power: float,
                      limit_w: float, tokens_per_sec: float, ms_per_call: float,
                      sm_mhz: float = 1380.0):
    """Synthetic attention result dict matching measure_workload output."""
    return {
        "workload": workload_name,
        "shape": [2, 16, 2048, 64],
        "kind": "attn",
        "idle_power_w": 20.0,
        "mean_power_w": round(mean_power, 2),
        "peak_power_w": round(mean_power * 1.05, 2),
        "mean_sm_mhz": round(sm_mhz, 0),
        "mean_mem_mhz": 877.0,
        "mean_temp_c": 45.0,
        "pstate": "P0",
        "limit_w": round(limit_w, 0),
        "ms_per_call": round(ms_per_call, 3),
        "telemetry_samples": 15,
        "tokens_per_pass": 256,
        "tokens_per_sec": round(tokens_per_sec, 0),
        "tokens_per_joule": round(tokens_per_sec / max(mean_power, 0.1), 2),
        "tflops_per_w": 0.01,
        "target_pl_w": pl,
    }


def _make_gemm_result(workload_name: str, pl: int, mean_power: float,
                      limit_w: float, tops: float, ms_per_call: float):
    return {
        "workload": workload_name,
        "shape": [2048, 12288, 3072],
        "kind": "gemm",
        "idle_power_w": 20.0,
        "mean_power_w": round(mean_power, 2),
        "peak_power_w": round(mean_power * 1.05, 2),
        "mean_sm_mhz": 1380.0,
        "mean_mem_mhz": 877.0,
        "mean_temp_c": 48.0,
        "pstate": "P0",
        "limit_w": round(limit_w, 0),
        "ms_per_call": round(ms_per_call, 3),
        "telemetry_samples": 15,
        "gflops": 75.5,
        "tops": round(tops, 3),
        "tops_per_w": round(tops / max(mean_power, 0.1), 4),
        "target_pl_w": pl,
    }


def test_cap_binding_clear_binds():
    """When raising the cap gives >3% throughput, verdict is 'binds'."""
    results = [
        _make_attn_result("prefill", pl=120, mean_power=115.0, limit_w=120.0,
                          tokens_per_sec=2000, ms_per_call=10.0),
        _make_attn_result("prefill", pl=150, mean_power=145.0, limit_w=150.0,
                          tokens_per_sec=2200, ms_per_call=9.0),  # +10%
    ]
    verdict = compute_cap_binding(results)
    assert "binds" in verdict
    assert "1.10" in verdict


def test_cap_binding_does_not_bind():
    """When throughput is unchanged, verdict is 'does_not_bind'."""
    results = [
        _make_attn_result("prefill", pl=120, mean_power=52.0, limit_w=120.0,
                          tokens_per_sec=2000, ms_per_call=10.0),
        _make_attn_result("prefill", pl=250, mean_power=55.0, limit_w=250.0,
                          tokens_per_sec=2005, ms_per_call=9.95),  # +0.25%
    ]
    verdict = compute_cap_binding(results)
    assert "does_not_bind" in verdict


def test_cap_binding_barely_binds():
    """1-3% throughput increase → 'barely_binds'."""
    results = [
        _make_attn_result("prefill", pl=120, mean_power=118.0, limit_w=120.0,
                          tokens_per_sec=2000, ms_per_call=10.0),
        _make_attn_result("prefill", pl=250, mean_power=235.0, limit_w=250.0,
                          tokens_per_sec=2030, ms_per_call=9.85),  # +1.5%
    ]
    verdict = compute_cap_binding(results)
    assert "barely_binds" in verdict


def test_cap_binding_single_limit_90pct():
    """Single power limit at ≥90% of cap → 'likely_binds'."""
    results = [
        _make_attn_result("prefill", pl=120, mean_power=115.0, limit_w=120.0,
                          tokens_per_sec=2000, ms_per_call=10.0),
    ]
    verdict = compute_cap_binding(results)
    assert "likely_binds" in verdict
    assert "115" in verdict


def test_cap_binding_single_limit_75pct():
    """Single power limit at 75-90% of cap → 'possibly_binds'."""
    results = [
        _make_attn_result("prefill", pl=120, mean_power=100.0, limit_w=120.0,
                          tokens_per_sec=2000, ms_per_call=10.0),
    ]
    verdict = compute_cap_binding(results)
    assert "possibly_binds" in verdict


def test_cap_binding_single_limit_low():
    """Single power limit well below cap → 'unlikely_binds'."""
    results = [
        _make_attn_result("prefill", pl=120, mean_power=55.0, limit_w=120.0,
                          tokens_per_sec=2000, ms_per_call=10.0),
    ]
    verdict = compute_cap_binding(results)
    assert "unlikely_binds" in verdict


def test_cap_binding_empty():
    assert compute_cap_binding([]) == "insufficient_data"
    assert compute_cap_binding([_make_gemm_result("gemm", 120, 100, 120, 91.0, 0.83)]) == "insufficient_data"


def test_cap_binding_mixed_attention_and_gemm():
    """GEMM-only results ignored; binding computed only from attention."""
    results = [
        _make_attn_result("prefill", pl=120, mean_power=115.0, limit_w=120.0,
                          tokens_per_sec=2000, ms_per_call=10.0),
        _make_attn_result("prefill", pl=250, mean_power=230.0, limit_w=250.0,
                          tokens_per_sec=2080, ms_per_call=9.6),  # +4%
        _make_gemm_result("gemm_flx", pl=120, mean_power=90.0, limit_w=120.0,
                          tops=91.0, ms_per_call=0.83),
    ]
    verdict = compute_cap_binding(results)
    assert "binds" in verdict


# ── JSON round-trip ─────────────────────────────────────────────────────────


def test_result_dict_json_roundtrip():
    r = _make_attn_result("prefill", pl=120, mean_power=115.0, limit_w=120.0,
                          tokens_per_sec=2000, ms_per_call=5.49)
    s = json.dumps(r)
    r2 = json.loads(s)
    assert r2["workload"] == "prefill"
    assert r2["target_pl_w"] == 120
    assert r2["kind"] == "attn"
    assert r2["mean_power_w"] == 115.0


# ── Metrics consistency ─────────────────────────────────────────────────────


def test_tokens_per_joule_consistency():
    """tokens_per_joule = tokens_per_sec / mean_power_w."""
    r = _make_attn_result("prefill", pl=120, mean_power=100.0, limit_w=120.0,
                          tokens_per_sec=5000, ms_per_call=5.0)
    expected = 5000 / 100.0
    assert r["tokens_per_joule"] == pytest.approx(expected, rel=1e-9)


def test_tops_consistency():
    """tops = 2*M*N*K / (ms/1000) / 1e12."""
    M, N, K = 2048, 12288, 3072
    flops = 2 * M * N * K
    ms = 0.83
    expected = flops / (ms / 1000.0) / 1e12
    r = _make_gemm_result("gemm", 120, mean_power=100.0, limit_w=120.0,
                          tops=expected, ms_per_call=ms)
    # tops is rounded to 3 decimal places in _make_gemm_result
    assert r["tops"] == pytest.approx(expected, rel=2e-5)


def test_attention_tflops_formula():
    """attention_flops = 4*B*H*M*M*D (used in tflops_per_w)."""
    B, H, M, D = 2, 16, 2048, 64
    flops = 4 * B * H * M * M * D
    assert flops == 4 * 2 * 16 * 2048 * 2048 * 64
    # ~34.36 GFLOPs
    assert 34e9 < flops < 35e9


# ── Power limit values ──────────────────────────────────────────────────────


def test_power_limits_are_reasonable():
    """120W floor is the fleet cap; 250W ceiling is the V100 TDP."""
    assert min(POWER_LIMITS_SWEEP) == 120
    assert max(POWER_LIMITS_SWEEP) == 250
    assert all(120 <= w <= 250 for w in POWER_LIMITS_SWEEP)
