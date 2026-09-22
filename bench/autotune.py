# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR7 backward autotuner (Optuna).

Sweeps the backward tile config (compile-time, via setup.py FNI8_BWD_* env
defines), rebuilding the extension per trial and running a FRESH worker
subprocess that gates correctness (hard constraint) and times the backward.
Optuna TPE minimizes total backward latency; invalid configs are pruned.

Run in the build/test container (needs a GPU + nvcc):

    docker compose run --rm test python3 -m bench.autotune --trials 6

Emits bench/bwd_autotune.json (all trials + best). Bake the winner by setting
its BM/BN as the FNI8_BWD_* defaults (or the constexpr defaults in attn_bwd.cuh)
in a dedicated commit; the perf regression gate then protects it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "bwd_autotune.json"

# Joint {tile, tpr} search. tile divisible by 4 (int8 pack); THREADS=tile*tpr<=1024;
# tpr a power of 2 dividing 32. tpr is the occupancy lever (12%->48% at tpr=8).
TILE_CHOICES = [32, 64]
TPR_CHOICES = [2, 4, 8]


def rebuild(tile: int, tpr: int = 2) -> bool:
    """Force a clean recompile with the given tile/tpr config. Returns True on success."""
    env = dict(os.environ, FNI8_BWD_BM=str(tile), FNI8_BWD_BN=str(tile), FNI8_BWD_TPR=str(tpr))
    # ninja caches on source mtime, not flag changes -> nuke build artifacts.
    for pat in ("build", "*.egg-info"):
        subprocess.run(f"rm -rf {ROOT}/{pat}", shell=True)
    subprocess.run(f"rm -f {ROOT}/superl8/_C*.so", shell=True)
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "-q"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  [tile={tile}] build FAILED: {r.stderr.strip()[-200:]}")
    return r.returncode == 0


def evaluate(tile: int, tpr: int) -> dict:
    if not rebuild(tile, tpr):
        return {"tile": tile, "tpr": tpr, "ok": False, "reason": "build failed"}
    r = subprocess.run(
        [sys.executable, "-m", "bench.autotune_worker"],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True,
        timeout=300,
    )
    line = (r.stdout.strip().splitlines() or ["{}"])[-1]
    try:
        res = json.loads(line)
    except json.JSONDecodeError:
        res = {"ok": False, "reason": f"worker crash: {r.stderr.strip()[-160:]}"}
    res["tile"], res["tpr"] = tile, tpr
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=len(TILE_CHOICES))
    args = ap.parse_args()

    try:
        import optuna
    except ModuleNotFoundError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna"], check=True)
        import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials: list[dict] = []

    def objective(trial):
        tile = trial.suggest_categorical("tile", TILE_CHOICES)
        tpr = trial.suggest_categorical("tpr", TPR_CHOICES)
        for t in trials:  # dedup (rebuilds are expensive)
            if t["tile"] == tile and t["tpr"] == tpr:
                return t["objective"] if t["ok"] else 1e9
        res = evaluate(tile, tpr)
        trials.append(res)
        status = f"ok obj={res.get('objective')}" if res["ok"] else f"INVALID ({res.get('reason')})"
        print(f"trial tile={tile} tpr={tpr}: {status}")
        return res["objective"] if res["ok"] else 1e9

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    # Enqueue the full {tile, tpr} grid so the search is COMPLETE.
    for tile in TILE_CHOICES:
        for tpr in TPR_CHOICES:
            study.enqueue_trial({"tile": tile, "tpr": tpr})
    study.optimize(objective, n_trials=max(args.trials, len(TILE_CHOICES) * len(TPR_CHOICES)))

    valid = [t for t in trials if t["ok"]]
    best = min(valid, key=lambda t: t["objective"]) if valid else None
    OUT.write_text(json.dumps({"trials": trials, "best": best}, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    if best:
        print(f"BEST: tile={best['tile']} tpr={best['tpr']}  ms={best['ms']}  obj={best['objective']}")
    else:
        print("no valid config found")


if __name__ == "__main__":
    main()
