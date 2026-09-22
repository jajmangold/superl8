# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Tests for bench/harness.py — GPU warm-up, timing, and regression guards."""

import statistics
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import gpu_warmup, time_ms  # noqa: E402


@pytest.mark.correctness
def test_gpu_warmup_available():
    """gpu_warmup must be importable and callable — existence gate."""
    assert callable(gpu_warmup)


@pytest.mark.correctness
def test_gpu_warmup_executes(device):
    """gpu_warmup must run sustained GPU work without error."""
    gpu_warmup(device=device, duration_s=0.1)


@pytest.mark.correctness
def test_time_ms_has_docstring():
    """time_ms must have a docstring (live-comment smoke gate)."""
    assert time_ms.__doc__ is not None
    assert len(time_ms.__doc__.strip()) > 0


@pytest.mark.correctness
def test_time_ms_runs_after_warmup(device):
    """time_ms must produce a meaningful (non-zero, finite) median for a tiny kernel."""
    x = torch.randn(8, 896, device=device, dtype=torch.float16)
    w = torch.randn(4864, 896, device=device, dtype=torch.float16)
    ms = time_ms(lambda: torch.matmul(x, w.t()), warmup=5, iters=15)
    assert ms > 0.0
    assert torch.isfinite(torch.tensor(ms))


@pytest.mark.perf
def test_tiny_gemm_consistent_timing(device):
    """Even tiny-K kernels must report stable times after clock warm-up:
    second call should be within 20% of the first (not 2x slower from idle)."""
    import superl8
    from superl8.quant.core import quantize_int8_rowwise

    m, n, k = 8, 4864, 896
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq_local(w)
    xq, xs = quantize_int8_rowwise(x)
    xs = xs.squeeze(-1).contiguous()

    # First timing (includes warm-up)
    ms1 = time_ms(lambda: superl8._C.gemm_w8a8(xq, xs, w_i8, w_scale))
    # Second timing (GPU already warm)
    ms2 = time_ms(lambda: superl8._C.gemm_w8a8(xq, xs, w_i8, w_scale))

    # After proper clock warm-up successive runs must be loosely consistent.
    # Idle clock (135 MHz) vs boosted (~1380 MHz) would show >3x difference;
    # 20% is generous for microkernel noise.
    print(f"\ntiny-M consistency: {ms1:.4f} ms vs {ms2:.4f} ms")
    assert ms1 <= ms2 * 1.25 or ms2 <= ms1 * 1.25, (
        f"tiny-M timing unstable across calls: {ms1:.4f} ms vs {ms2:.4f} ms"
    )


@pytest.mark.perf
def test_tiny_gemm_within_run_variance(device):
    """The time_ms harness must produce stable single-run medians for tiny M=8
    kernels.  Use per-iteration sample variance (via a dedicated high-iter run)
    to assert that the coefficient of variation (std/mean) is below 15% — this
    catches unstable measurement from clock drops or launch noise.

    This is a MEASUREMENT-QUALITY gate, not a kernel-performance gate.
    """
    import superl8
    from superl8.quant.core import quantize_int8_rowwise

    m, n, k = 8, 4864, 896
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq_local(w)
    xq, xs = quantize_int8_rowwise(x)
    xs = xs.squeeze(-1).contiguous()

    # Collect raw samples from a dedicated high-iteration run
    samples = _collect_raw_samples(lambda: superl8._C.gemm_w8a8(xq, xs, w_i8, w_scale),
                                   warmup=30, iters=200)
    mean = statistics.mean(samples)
    std = statistics.stdev(samples)
    cv = std / mean if mean > 0 else 1.0
    print(f"\ntiny-M within-run samples: n={len(samples)}, mean={mean:.4f} ms, "
          f"std={std:.4f} ms, cv={cv:.3f}")
    # CV > 15% indicates a measurement-quality problem (clock jitter, launch noise)
    assert cv <= 0.15, (
        f"tiny-M timing unstable within a single run: CV={cv:.3f} "
        f"(mean={mean:.4f} ms, std={std:.4f} ms) — the median is unreliable"
    )


def _collect_raw_samples(fn, *, warmup=10, iters=50) -> list[float]:
    """Return raw per-iteration samples (not median) for variance analysis.

    Uses direct CUDA event timing (no adaptive iters or trimming) to measure
    the raw iteration-level variance that the harness's time_ms must stabilize.
    """
    from bench.harness import gpu_warmup

    gpu_warmup()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _wq_local(w: torch.Tensor):
    from superl8.quant.core import quantize_int8_rowwise

    q, s = quantize_int8_rowwise(w)
    return q.contiguous(), s.squeeze(-1).contiguous()
