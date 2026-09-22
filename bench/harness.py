# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""GPU timing + baseline regression helpers.

GPU timing needs CUDA events + explicit sync + warmup; wall-clock measures async
launch, not execution. We take the MEDIAN over N iters. Baselines live in
bench/baseline.json and are only mutated in a dedicated "update baseline" PR.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Callable

import torch

BASELINE_PATH = Path(__file__).parent / "baseline.json"

# Latency may exceed baseline by this fraction before the perf gate fails.
REGRESSION_TOLERANCE = 0.05

# ---------------------------------------------------------------------------
# GPU clock warm-up: on CMP 100-210 / V100 cards the idle P8 state is ~135 MHz.
# Micro-kernels (e.g. m=8 GEMM) complete before the clock ramps up, so the
# harness measures idle-clock latency instead of steady-state performance.
# We issue a sustained GPU workload once per process to force the SM clock
# out of idle before any timing call.
# ---------------------------------------------------------------------------

_WARMUP_SIZE = 4096
_WARMUP_DURATION_S = 0.5

_gpu_clock_warmed: bool = False
_gpu_warmup_time: float = 0.0  # monotonic time of last warmup
_WARMUP_COOLDOWN_S: float = 10.0  # re-warm after this many idle seconds


def gpu_warmup(
    *, device: torch.device | None = None, duration_s: float = _WARMUP_DURATION_S,
    force: bool = False,
) -> None:
    """Run sustained GPU matmul work for `duration_s` seconds to force the SM
    clock out of the P8 idle state (~135 MHz) into a boost P-state.

    Idempotent: no-op if the clock has already been warmed this process (unless
    *force=True* or more than *WARMUP_COOLDOWN_S* seconds have elapsed since the
    last warmup).  Call before the first microbenchmark if using the harness
    outside of `time_ms` (which calls this automatically).
    """
    global _gpu_clock_warmed, _gpu_warmup_time
    if not force and _gpu_clock_warmed:
        return
    if not torch.cuda.is_available():
        _gpu_clock_warmed = True
        _gpu_warmup_time = 0.0
        return
    dev = device or torch.cuda.current_device()
    with torch.cuda.device(dev):
        a = torch.randn(_WARMUP_SIZE, _WARMUP_SIZE, dtype=torch.float16, device=dev)
        b = torch.randn(_WARMUP_SIZE, _WARMUP_SIZE, dtype=torch.float16, device=dev)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        one_ms = start.elapsed_time(end)
        if one_ms <= 0.1:
            one_ms = 0.5
        n_iters = max(1, int(duration_s * 1e3 / one_ms))
        for _ in range(n_iters):
            _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    _gpu_clock_warmed = True
    _gpu_warmup_time = _monotonic_sec()


def _monotonic_sec() -> float:
    """Monotonic time in seconds (wall clock, for warmup-cooldown tracking)."""
    import time
    return time.monotonic()


def gpu_class() -> str:
    """Coarse GPU key for baselines. sm_70 == Volta here (V100 / CMP)."""
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


# ---------------------------------------------------------------------------
# Clock-keep: a tiny sustained workload to prevent the GPU SM clock from
# dropping back to idle between timed iterations of micro-kernels.  A ~256x256
# fp16 matmul is ~2-3 µs on V100 — negligible overhead vs a 0.1 ms kernel, but
# keeps the SM in the boost P-state between timed calls.
# ---------------------------------------------------------------------------

_KA_SIZE = 256
_KA_A: torch.Tensor | None = None
_KA_B: torch.Tensor | None = None


def _ensure_keep_alive(device: torch.device) -> None:
    global _KA_A, _KA_B
    if _KA_A is not None and _KA_A.device == device:
        return
    _KA_A = torch.randn(_KA_SIZE, _KA_SIZE, dtype=torch.float16, device=device)
    _KA_B = torch.randn(_KA_SIZE, _KA_SIZE, dtype=torch.float16, device=device)


def _keep_alive_tick() -> None:
    """Run one tiny matmul on the keep-alive buffers to sustain the SM clock."""
    if _KA_A is not None:
        _ = torch.matmul(_KA_A, _KA_B)


# micro-kernel threshold: kernels below this duration (ms) get more iters + keep-alive
_MICRO_KERNEL_THRESHOLD_MS = 0.5
_TARGET_MEASUREMENT_MS = 100.0  # target total measurement time for micro-kernels
_MAX_ADAPTIVE_ITERS = 3000
_TRIM_FRACTION = 0.1  # discard fastest/slowest this fraction before median


def time_ms(fn: Callable[[], object], *, warmup: int = 10, iters: int = 50) -> float:
    """Median kernel time in milliseconds (CUDA events, synced).

    On first call, runs a sustained GPU warm-up (gpu_warmup) to force the SM
    clock out of idle P8 (~135 MHz) into a boost P-state.  Micro-kernels
    (e.g. m=8 GEMM) complete before the idle-to-boost clock ramp, so without
    this the harness measures idle-clock latency instead of steady-state perf.

    For micro-kernels (< 0.5 ms) the number of iterations is adaptively
    increased to reach ~100 ms total measurement time (capped at 3000 iters),
    and a tiny keep-alive matmul is run between iterations to prevent the SM
    clock from dropping during gaps.  The fastest and slowest 10% of samples
    are trimmed before the median to reject outliers from clock jitter or
    launch-noise spikes.
    """
    _gpu_warmup_or_rewarm()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # ---- Probe: measure a few samples to estimate kernel duration ----
    probe = _probe_kernel(fn, n=5)
    probe_median = statistics.median(probe)
    is_micro = probe_median < _MICRO_KERNEL_THRESHOLD_MS

    # ---- Adaptive iters for micro-kernels ----
    if is_micro:
        target_iters = max(iters, int(_TARGET_MEASUREMENT_MS / probe_median))
        target_iters = min(target_iters, _MAX_ADAPTIVE_ITERS)
    else:
        target_iters = iters

    # ---- Main measurement ----
    dev = torch.cuda.current_device()
    samples: list[float] = []
    if is_micro:
        _ensure_keep_alive(dev)

    for i in range(target_iters):
        # Keep-alive tick between iterations to sustain SM clock for micro-kernels
        if is_micro and i > 0 and i % 5 == 0:
            _keep_alive_tick()
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    # ---- Trim outliers ----
    trimmed = _trim_samples(samples, _TRIM_FRACTION)
    return statistics.median(trimmed)


def _gpu_warmup_or_rewarm() -> None:
    """Warm the GPU clock, re-warming if the clock may have cooled."""
    if not _gpu_clock_warmed:
        gpu_warmup()
        return
    elapsed = _monotonic_sec() - _gpu_warmup_time
    if elapsed > _WARMUP_COOLDOWN_S:
        gpu_warmup(force=True)


def _probe_kernel(fn: Callable[[], object], n: int = 5) -> list[float]:
    """Quick probe: time `n` iterations of `fn`, return raw sample list."""
    samples: list[float] = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _trim_samples(samples: list[float], frac: float = 0.1) -> list[float]:
    """Discard the fastest and slowest `frac` of samples before taking median."""
    if not samples:
        return samples
    n = len(samples)
    k = max(1, int(n * frac))
    if 2 * k >= n:
        return samples  # too few samples to trim meaningfully
    sorted_s = sorted(samples)
    return sorted_s[k:-k]


def attention_flops(b: int, h: int, m: int, n: int, d: int, *, causal: bool = False) -> int:
    """FLOPs for one attention forward: QK^T (2bhmnd) + PV (2bhmnd). Halved if causal."""
    total = 4 * b * h * m * n * d
    return total // 2 if causal else total


def tflops(flops: int, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


def compare_report(name: str, ours_ms: float, baselines_ms: dict[str, float]) -> str:
    """The honest dp4a-vs-fp16 report line. A ratio < 1.0 means we are SLOWER —
    that is recorded, not hidden (AGENTS.md: the harness measures, it doesn't assume)."""
    parts = [f"{name}: {ours_ms:.3f} ms"]
    for base_name, base_ms in baselines_ms.items():
        parts.append(f"{base_ms / ours_ms:.2f}x vs {base_name} ({base_ms:.3f} ms)")
    return " | ".join(parts)


def load_baseline() -> dict:
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text())
    return {}


def baseline_get(name: str) -> float | None:
    """Committed median-ms baseline for `name` on the current GPU class, or None."""
    return load_baseline().get(gpu_class(), {}).get(name, {}).get("median_ms")


def assert_no_regression(name: str, measured_ms: float, *, tolerance: float = REGRESSION_TOLERANCE):
    """Fail if `measured_ms` regresses past the committed baseline.

    A missing baseline is a soft skip (recorded), NOT a pass — the first run of a
    kernel records its baseline in a dedicated PR before the gate becomes active.
    """
    base = baseline_get(name)
    if base is None:
        import pytest

        pytest.skip(f"no committed baseline for {name!r} on {gpu_class()} — record it first")
    limit = base * (1.0 + tolerance)
    assert measured_ms <= limit, (
        f"perf regression: {name} {measured_ms:.4f} ms > {limit:.4f} ms "
        f"(baseline {base:.4f} ms + {tolerance:.0%})"
    )
