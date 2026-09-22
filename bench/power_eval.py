# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Power-perf telemetry on a real V100 (host GPU idx 4).

Measures sustained clock, power draw, and tokens/joule under real superl8 workloads
at the fleet's 120W cap and at raised limits, producing the data for the
perf/watt scoreboard (#59) and answering the sota-research-2026-07.md question:
does the 120W power cap bind sustained serving throughput?

Pin to one real V100 (idx 4/7/9/11/14). NEVER run on the shared CMP cards.

Usage (inside the bench container, real V100 only):
  docker compose run --rm bench python3 bench/power_eval.py
  docker compose run --rm bench python3 bench/power_eval.py --skip-power-cap  # measurement-only, no -pl
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

import superl8
from bench.harness import gpu_warmup


# ── workload config ─────────────────────────────────────────────────────────

@dataclass
class Workload:
    name: str
    shape: tuple  # (B, H, M, D) for attention; (M, N, K) for GEMM
    kind: str     # "attn" or "gemm"
    causal: bool = False
    warmup_passes: int = 20
    measure_duration_s: float = 15.0

# Prefill attention (the heaviest sustained workload — stresses dp4a + FP + HBM).
PREMILL = Workload("prefill_2048_d128", (2, 16, 2048, 128), "attn")
# Medium prefill
PREMILL_MED = Workload("prefill_2048_d64", (2, 16, 2048, 64), "attn")
# GEMM — MLP fc1 (Flux-class), stresses the INT pipe.
GEMM_LARGE = Workload("gemm_flx_fc1", (2048, 12288, 3072), "gemm")
# GEMM — prefill FFN up-proj (Qwen2-ish).
GEMM_PREMILL = Workload("gemm_prefill", (2048, 4864, 896), "gemm")

DEFAULT_WORKLOADS = [PREMILL, PREMILL_MED, GEMM_LARGE, GEMM_PREMILL]

# Power limits to sweep (W). 120 = fleet default. Ordered low→high so the
# default is always first (and therefore also last — we restore to it).
POWER_LIMITS_SWEEP = [120, 150, 175, 200, 225, 250]

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "utils" / "docs" / "power-characterization.json"


# ── nvidia-smi telemetry ────────────────────────────────────────────────────

def _smi_query(fields: str, gpu_index: int = 4) -> str:
    """Query `nvidia-smi` for one snapshot."""
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits",
             "-i", str(gpu_index)],
            text=True, timeout=10,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return ""

def smi_power(gpu_index: int = 4) -> float | None:
    v = _smi_query("power.draw", gpu_index)
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def smi_clock(gpu_index: int = 4) -> float | None:
    v = _smi_query("clocks.sm", gpu_index)
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def smi_power_limit(gpu_index: int = 4) -> float | None:
    v = _smi_query("power.limit", gpu_index)
    try:
        return float(v.split(".")[0])
    except (ValueError, TypeError):
        return None

def smi_snapshot(gpu_index: int = 4) -> dict:
    fields = "power.draw,temperature.gpu,clocks.sm,clocks.mem,pstate,power.limit"
    raw = _smi_query(fields, gpu_index)
    keys = ["power_w", "temp_c", "sm_mhz", "mem_mhz", "pstate", "limit_w"]
    out = {}
    if not raw:
        return out
    parts = [p.strip() for p in raw.split(",")]
    for i, k in enumerate(keys):
        if i < len(parts):
            try:
                out[k] = float(parts[i]) if k not in ("pstate",) else parts[i]
            except (ValueError, TypeError):
                out[k] = parts[i]
    return out

def set_power_limit(watts: int, gpu_index: int = 4) -> bool:
    """Set the GPU power limit. Requires root + persistence mode or admin caps."""
    try:
        subprocess.check_call(
            ["nvidia-smi", "-i", str(gpu_index), "-pl", str(watts)],
            timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)
        actual = smi_power_limit(gpu_index)
        if actual is not None and abs(actual - watts) <= 5:
            return True
        return False
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── telemetry collector (background thread) ─────────────────────────────────

@dataclass
class TelemetrySeries:
    """Collected per-second snapshots from nvidia-smi during a workload run."""
    power_w: list[float] = field(default_factory=list)
    sm_mhz: list[float] = field(default_factory=list)
    mem_mhz: list[float] = field(default_factory=list)
    temp_c: list[float] = field(default_factory=list)
    pstate: list[str] = field(default_factory=list)
    limit_w: list[float] = field(default_factory=list)

    @property
    def mean_power_w(self) -> float:
        return statistics.mean(self.power_w) if self.power_w else 0.0
    @property
    def peak_power_w(self) -> float:
        return max(self.power_w) if self.power_w else 0.0
    @property
    def mean_sm_mhz(self) -> float:
        return statistics.mean(self.sm_mhz) if self.sm_mhz else 0.0
    @property
    def mean_mem_mhz(self) -> float:
        return statistics.mean(self.mem_mhz) if self.mem_mhz else 0.0
    @property
    def mean_temp_c(self) -> float:
        return statistics.mean(self.temp_c) if self.temp_c else 0.0
    @property
    def num_samples(self) -> int:
        return len(self.power_w)


def _telemetry_thread(gpu_index: int, series: TelemetrySeries, stop: threading.Event,
                      interval_s: float = 1.0):
    """Poll nvidia-smi every interval_s seconds; run until stop is set."""
    while not stop.is_set():
        snap = smi_snapshot(gpu_index)
        if snap:
            series.power_w.append(snap.get("power_w", 0.0))
            series.sm_mhz.append(snap.get("sm_mhz", 0.0))
            series.mem_mhz.append(snap.get("mem_mhz", 0.0))
            series.temp_c.append(snap.get("temp_c", 0.0))
            series.pstate.append(snap.get("pstate", ""))
            series.limit_w.append(snap.get("limit_w", 0.0))
        # aim for exactly interval_s, but nvidia-smi call itself takes non-zero time
        time.sleep(max(0.05, interval_s - 0.1))

def collect_telemetry(gpu_index: int, target_fn, duration_s: float,
                      poll_interval_s: float = 1.0) -> TelemetrySeries:
    """Run `target_fn` continuously while polling nvidia-smi in a background thread."""
    series = TelemetrySeries()
    stop = threading.Event()
    t = threading.Thread(target=_telemetry_thread, args=(gpu_index, series, stop, poll_interval_s),
                         daemon=True)
    t.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        target_fn()
    stop.set()
    t.join(timeout=5.0)
    return series


# ── workload runners ─────────────────────────────────────────────────────────

def _time_per_iter(fn, n_warmup: int = 10, n_iters: int = 50) -> float:
    """Median ms per call (CUDA events)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(n_iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return statistics.median(samples)

def measure_workload(workload: Workload, gpu_index: int = 4) -> dict:
    """Run one workload under sustained-measurement, return metrics."""
    device = torch.device(f"cuda:{gpu_index}")
    with torch.cuda.device(device):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        if workload.kind == "attn":
            B, H, M, D = workload.shape
            q = torch.randn(B, H, M, D, dtype=torch.float16, device=device)
            k = torch.randn(B, H, M, D, dtype=torch.float16, device=device)
            v = torch.randn(B, H, M, D, dtype=torch.float16, device=device)

            def _attn_fn():
                superl8.attn_int8_fwd(q, k, v, causal=workload.causal)
            fn = _attn_fn
        elif workload.kind == "gemm":
            Msz, Nsz, Ksz = workload.shape
            a = torch.randn(Msz, Ksz, dtype=torch.float16, device=device)
            b = torch.randn(Ksz, Nsz, dtype=torch.float16, device=device)

            def _gemm_fn():
                superl8.gemm_w8a8(a, b)
            fn = _gemm_fn
        else:
            raise ValueError(f"unknown workload kind: {workload.kind}")

        # GPU clock warm-up from idle P8
        gpu_warmup(device=device, force=True)

        # Idle power baseline (after warmup, before workload)
        torch.cuda.synchronize()
        time.sleep(0.5)
        idle_power = smi_power(gpu_index)

        # Sustained measurement
        series = collect_telemetry(gpu_index, fn, workload.measure_duration_s)

        # Single-pass timing (for tokens/sec)
        ms_per_call = _time_per_iter(fn, n_warmup=workload.warmup_passes, n_iters=50)

        result = {
            "workload": workload.name,
            "shape": list(workload.shape),
            "kind": workload.kind,
            "idle_power_w": idle_power,
            "mean_power_w": round(series.mean_power_w, 2),
            "peak_power_w": round(series.peak_power_w, 2),
            "mean_sm_mhz": round(series.mean_sm_mhz, 0),
            "mean_mem_mhz": round(series.mean_mem_mhz, 0),
            "mean_temp_c": round(series.mean_temp_c, 1),
            "pstate": series.pstate[-1] if series.pstate else "",
            "limit_w": round(series.limit_w[-1], 0) if series.limit_w else 0,
            "ms_per_call": round(ms_per_call, 3),
            "telemetry_samples": series.num_samples,
        }

        # Tokens processed per forward pass
        if workload.kind == "attn":
            B, H, M, D = workload.shape
            tokens_per_pass = B * H * M
            result["tokens_per_pass"] = tokens_per_pass
            result["tokens_per_sec"] = round(tokens_per_pass / (ms_per_call / 1000.0), 0)
            result["tokens_per_joule"] = round(
                result["tokens_per_sec"] / max(series.mean_power_w, 0.1), 2
            )
            flops = 4 * B * H * M * M * D  # QK^T + PV
            result["tflops_per_w"] = round(
                (flops / (ms_per_call / 1000.0) / 1e12) / max(series.mean_power_w, 0.1), 4
            )
        elif workload.kind == "gemm":
            Msz, Nsz, Ksz = workload.shape
            flops = 2 * Msz * Nsz * Ksz
            result["gflops"] = round(flops / 1e9, 1)
            result["tops"] = round(flops / (ms_per_call / 1000.0) / 1e12, 3)
            result["tops_per_w"] = round(
                (flops / (ms_per_call / 1000.0) / 1e12) / max(series.mean_power_w, 0.1), 5
            )

        return result


# ── main sweep ──────────────────────────────────────────────────────────────

def sweep(workloads: list[Workload], gpu_index: int = 4,
          limits: list[int] | None = None, skip_set_pl: bool = False) -> list[dict]:
    """Run workloads at each power limit. Returns list of per-run results."""
    if limits is None:
        limits = POWER_LIMITS_SWEEP

    original_limit = smi_power_limit(gpu_index)
    print(f"≡≡≡ superl8 power telemetry sweep  GPU idx={gpu_index}  "
          f"original power limit={original_limit}W\n")

    # Device check — refuse to run if not a real V100 (compute capability 7.0)
    if not torch.cuda.is_available():
        print("[FATAL] CUDA unavailable", file=sys.stderr)
        sys.exit(1)
    dev = torch.cuda.get_device_properties(gpu_index)
    if dev.major != 7:
        print(f"[FATAL] GPU idx {gpu_index} is sm_{dev.major}{dev.minor}, not sm_70 (Volta). "
              f"Must use a real Tesla V100 (host idx 4/7/9/11/14).", file=sys.stderr)
        sys.exit(1)
    print(f"[device] {dev.name}  sm_{dev.major}{dev.minor}  "
          f"{dev.multi_processor_count} SM  {dev.total_memory // (1024**3)} GB\n")

    can_set_pl = not skip_set_pl
    if can_set_pl and not skip_set_pl:
        if not set_power_limit(limits[0], gpu_index):
            print("[WARN] Cannot set power limit (permissions or persistence mode).\n"
                  "       Running measurement-only at the existing power cap.\n"
                  "       Retry with --skip-power-cap or run as root.\n")
            can_set_pl = False
            limits = [int(original_limit or 120)]

    results = []
    for watts in limits:
        if can_set_pl:
            ok = set_power_limit(watts, gpu_index)
            if not ok:
                print(f"  skip {watts}W — could not set power limit\n")
                continue
            actual = smi_power_limit(gpu_index)
            print(f"[pl] set {watts}W  actual limit={actual}W")
        else:
            actual = smi_power_limit(gpu_index)
            print(f"[pl] existing cap {actual}W (no -pl attempted)")

        for wl in workloads:
            res = measure_workload(wl, gpu_index)
            res["target_pl_w"] = watts
            results.append(res)
            if wl.kind == "attn":
                print(f"  {wl.name:28s}  {res['mean_power_w']:5.1f}W  "
                      f"{res['mean_sm_mhz']:5.0f} MHz  "
                      f"{res['tokens_per_sec']:9.0f} tok/s  "
                      f"{res['tokens_per_joule']:8.1f} tok/J  "
                      f"{res['ms_per_call']:.3f} ms/call")
            else:
                print(f"  {wl.name:28s}  {res['mean_power_w']:5.1f}W  "
                      f"{res['mean_sm_mhz']:5.0f} MHz  "
                      f"{res['tops']:5.3f} TOP/s  "
                      f"{res['gflops']:6.0f} GFLOPs  "
                      f"{res['ms_per_call']:.3f} ms/call")
        print()
        torch.cuda.synchronize()
        time.sleep(1.0)  # let power settle between limits

    # Restore original power limit
    if can_set_pl and original_limit is not None:
        set_power_limit(int(original_limit), gpu_index)
        print(f"[pl] restored {int(original_limit)}W")

    return results


def compute_cap_binding(results: list[dict]) -> str:
    """Determine whether the power cap binds sustained serving throughput."""
    attn_results = [r for r in results if r["kind"] == "attn"]

    if not attn_results:
        return "insufficient_data"

    by_limit = {}
    for r in attn_results:
        pl = r.get("target_pl_w", r.get("limit_w", 0))
        by_limit.setdefault(pl, []).append(r)

    # At 120W vs highest measured: do clock and throughput increase?
    limits_sorted = sorted(by_limit.keys())
    if len(limits_sorted) < 2:
        # Only one power limit — check if power hits the cap
        r = attn_results[0]
        mean_power = r["mean_power_w"]
        limit_w = r.get("limit_w", 120)
        if mean_power >= 0.90 * limit_w:
            return f"likely_binds: mean power {mean_power:.1f}W close to cap {limit_w:.0f}W (≥90%)"
        elif mean_power >= 0.75 * limit_w:
            return f"possibly_binds: mean power {mean_power:.1f}W is 75-90% of cap {limit_w:.0f}W"
        else:
            return f"unlikely_binds: mean power {mean_power:.1f}W well below cap {limit_w:.0f}W"

    lo = limits_sorted[0]
    hi = limits_sorted[-1]
    lo_avgs = [r["tokens_per_sec"] for r in by_limit.get(lo, [])]
    hi_avgs = [r["tokens_per_sec"] for r in by_limit.get(hi, [])]
    if lo_avgs and hi_avgs:
        lo_tok = statistics.mean(lo_avgs)
        hi_tok = statistics.mean(hi_avgs)
        speedup = hi_tok / lo_tok if lo_tok > 0 else 1.0
        if speedup >= 1.03:
            return (f"binds: raising cap {lo}W→{hi}W gives {speedup:.2f}x throughput "
                    f"({lo_tok:.0f}→{hi_tok:.0f} tok/s)")
        elif speedup >= 1.01:
            return (f"barely_binds: {lo}W→{hi}W gives {speedup:.2f}x ({lo_tok:.0f}→{hi_tok:.0f} tok/s)")
        else:
            return f"does_not_bind: raising cap {lo}W→{hi}W gives {speedup:.2f}x (no throughput gain)"

    return "insufficient_data"


def print_scoreboard(results: list[dict]) -> None:
    """Print a perf/watt scoreboard table."""
    print("\n" + "=" * 110)
    print("PERF/WATT SCOREBOARD — superl8 int8 dp4a on Tesla V100 (sm_70)")
    print("=" * 110)
    header = (f"{'Workload':<28s} {'PL':>6s} {'Power':>7s} {'SM MHz':>7s} "
              f"{'Temp':>5s} {'tok/s':>10s} {'tok/J':>8s} {'TOP/s':>7s} {'ms':>8s} {'TOP/W':>8s}")
    print(header)
    print("-" * 110)
    for r in sorted(results, key=lambda x: (x["kind"], x["workload"], x.get("target_pl_w", 0))):
        pl = r.get("target_pl_w", r.get("limit_w", 0))
        tok_s = f"{r.get('tokens_per_sec', 0):.0f}" if r["kind"] == "attn" else "—"
        tok_j = f"{r.get('tokens_per_joule', 0):.1f}" if r["kind"] == "attn" else "—"
        tops = f"{r.get('tops', 0):.3f}" if r["kind"] == "gemm" else "—"
        tops_w = (f"{r.get('tflops_per_w', r.get('tops_per_w', 0)):.4f}"
                  if r["kind"] == "attn" else f"{r.get('tops_per_w', 0):.5f}")
        print(f"{r['workload']:<28s} {pl:4.0f}W  {r['mean_power_w']:5.1f}W  "
              f"{r['mean_sm_mhz']:5.0f}  {r['mean_temp_c']:3.0f}C  "
              f"{tok_s:>10s}  {tok_j:>8s}  {tops:>7s}  {r['ms_per_call']:6.3f}  {tops_w}")
    print("-" * 110)

    cap = compute_cap_binding(results)
    print(f"\nPower cap binding verdict: {cap}")
    print("=" * 110)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="superl8 power-perf telemetry (real V100 only)")
    parser.add_argument("--gpu-index", type=int, default=4,
                        help="CUDA device index of the real V100 (default: 4)")
    parser.add_argument("--skip-power-cap", action="store_true",
                        help="Only measure at existing power cap; do not attempt -pl")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Sustained measurement duration per workload (s)")
    parser.add_argument("--workload", choices=["all", "attn", "gemm", "prefill", "prefill_heavy"],
                        default="all", help="Which workloads to run")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE),
                        help="JSON output path")
    parser.add_argument("--limits", type=str, default="",
                        help="Comma-separated power limits to sweep (default: 120,150,175,200,225,250)")
    args = parser.parse_args()

    gpu_idx = args.gpu_index
    if gpu_idx not in (4, 7, 9, 11, 14):
        print(f"[WARN] GPU idx {gpu_idx} is not in the known real-V100 list [4,7,9,11,14]. "
              f"Continuing anyway, but verify this is a real Tesla V100.", file=sys.stderr)

    # Select workloads
    if args.workload == "all":
        workloads = DEFAULT_WORKLOADS
    elif args.workload == "attn":
        workloads = [PREMILL, PREMILL_MED]
    elif args.workload == "gemm":
        workloads = [GEMM_LARGE, GEMM_PREMILL]
    elif args.workload == "prefill":
        workloads = [PREMILL]
    elif args.workload == "prefill_heavy":
        workloads = [PREMILL]
    else:
        workloads = DEFAULT_WORKLOADS

    # Patch duration
    for wl in workloads:
        wl.measure_duration_s = args.duration

    # Parse limits
    limits = None
    if args.limits:
        limits = [int(x.strip()) for x in args.limits.split(",") if x.strip()]

    results = sweep(workloads, gpu_idx, limits=limits, skip_set_pl=args.skip_power_cap)
    print_scoreboard(results)

    # Write JSON output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output_path}")

    return results


if __name__ == "__main__":
    main()
