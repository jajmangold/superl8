# SPDX-License-Identifier: BSD-3-Clause
"""Offline contract tests for the kernel-evolution evaluator."""

import math
from inspect import signature

import numpy as np
import pytest

from evalpool.data_gen import generate_test_data, quantize_int8_rowwise
from evalpool import evaluator
from evalpool.gates import FidelityGate, score_candidate
from evalpool.server import MAX_SOURCE_BYTES, run_server


def test_fidelity_gate_requires_every_numeric_bar():
    gate = FidelityGate(min_cosine=0.999, min_sqnr_db=40.0, max_rel_l1=0.02)

    assert gate.accepts(cosine=0.9995, sqnr_db=41.0, rel_l1=0.01)
    assert not gate.accepts(cosine=0.9989, sqnr_db=80.0, rel_l1=0.001)
    assert not gate.accepts(cosine=1.0, sqnr_db=39.9, rel_l1=0.001)
    assert not gate.accepts(cosine=1.0, sqnr_db=80.0, rel_l1=0.021)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_fidelity_gate_rejects_nonfinite_metrics(value):
    gate = FidelityGate()
    assert not gate.accepts(cosine=value, sqnr_db=80.0, rel_l1=0.001)
    assert not gate.accepts(cosine=1.0, sqnr_db=value, rel_l1=0.001)
    assert not gate.accepts(cosine=1.0, sqnr_db=80.0, rel_l1=value)


@pytest.mark.parametrize("candidate_ms", [0.0, -1.0, float("nan"), float("inf")])
def test_score_rejects_invalid_timings(candidate_ms):
    gate = FidelityGate()
    score = score_candidate(
        baseline_ms=7.0,
        candidate_ms=candidate_ms,
        cosine=1.0,
        sqnr_db=80.0,
        rel_l1=0.001,
        gate=gate,
    )
    assert score == 0.0


def test_score_is_speedup_only_after_fidelity_passes():
    gate = FidelityGate()
    assert score_candidate(8.0, 4.0, 1.0, 80.0, 0.001, gate) == 2.0
    assert score_candidate(8.0, 4.0, 0.9, 80.0, 0.001, gate) == 0.0


def test_rowwise_quantization_is_finite_for_zero_rows():
    q, scale = quantize_int8_rowwise(np.zeros((2, 256), dtype=np.float32))
    assert q.dtype == np.int8
    assert np.count_nonzero(q) == 0
    assert np.isfinite(scale).all()
    assert (scale > 0).all()


def test_q5k_data_generation_is_deterministic_and_well_shaped():
    first = generate_test_data(2, 3, 256, seed=7)
    second = generate_test_data(2, 3, 256, seed=7)

    assert first["x_i8"].shape == (2, 256)
    assert first["x_scale"].shape == (2,)
    assert first["w_bytes"].shape == (3, 176)
    assert first["y_ref"].shape == (2, 3)
    assert first["num_sb"] == 1
    for key in ("x_i8", "x_scale", "w_bytes", "y_ref"):
        np.testing.assert_array_equal(first[key], second[key])
    assert math.isfinite(float(first["y_ref"].sum()))


class _FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def evaluate(self, *args, **kwargs):
        self.calls += 1
        return next(self.responses)


def _result(*, sqnr=80.0, rel_l1=0.001, ms=1.0):
    return {
        "compile_ok": True,
        "correct": True,
        "cos": 1.0,
        "sqnr": sqnr,
        "rel_l1": rel_l1,
        "ms": ms,
    }


def test_openevolve_adapter_rejects_bad_small_shape_sqnr(monkeypatch):
    client = _FakeClient([_result(sqnr=10.0)])
    monkeypatch.setattr(evaluator, "EvalPoolClient", lambda **kwargs: client)

    result = evaluator.evaluate('extern "C" __global__ void candidate() {}')

    assert result.metrics["combined_score"] == 0.0
    assert result.metrics["correct"] == 0.0
    assert client.calls == 1


def test_openevolve_adapter_rejects_bad_production_rel_l1(monkeypatch):
    client = _FakeClient([_result(), _result(rel_l1=0.5, ms=4.0)])
    monkeypatch.setattr(evaluator, "EvalPoolClient", lambda **kwargs: client)

    result = evaluator.evaluate('extern "C" __global__ void candidate() {}')

    assert result.metrics["combined_score"] == 0.0
    assert result.metrics["correct"] == 0.0
    assert client.calls == 2


def test_eval_server_defaults_to_loopback_and_bounds_source():
    assert signature(run_server).parameters["host"].default == "127.0.0.1"
    assert MAX_SOURCE_BYTES == 1 << 20
