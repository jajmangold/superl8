#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Benchmark scoreboard visualisation — reads JSON data, emits PNG charts.

Generates three charts into superl8/bench/figures/:
  1. Roofline plot for CMP 100-210 kernel achieved throughput
  2. dp4a INT8 vs FP16 tensor-core throughput bar chart (V100 & CMP 100-210)
  3. Qwen3 model throughput (prefill + decode tok/s)

Usage:
  python3 bench/plot_scoreboard.py          # all charts
  python3 bench/plot_scoreboard.py --only roofline   # single chart
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SCOREBOARD_JSON = ROOT / "qwen3-scoreboard.json"
SERVE_DIR = ROOT.parent.parent / "superl8-serve" / "bench"
SERVE_8B_JSON = SERVE_DIR / "Qwen__Qwen3-8B.b8.json"
SERVE_27B_JSON = SERVE_DIR / "Qwen3.6-27B-Q3_K_S.gguf.json"
FIG_DIR = ROOT / "figures"
SERVE_FIG_DIR = SERVE_DIR / "figures"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
DARK_BG = "#0e1117"
CARD_BG = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"
ACCENT1 = "#58a6ff"   # blue
ACCENT2 = "#f0883e"   # orange
ACCENT3 = "#3fb950"   # green
ACCENT4 = "#d2a8ff"   # purple
ACCENT5 = "#ff7b72"   # red
MUTED = "#8b949e"

CMP_COLOR = ACCENT1
V100_COLOR = ACCENT2
MEM_COLOR = ACCENT3
COMPUTE_COLOR = ACCENT4
KERNEL_COLOR = ACCENT5


def _style():
    """Apply the dark, publication-quality rcParams."""
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlepad": 14,
        "text.color": TEXT_COLOR,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "legend.facecolor": CARD_BG,
        "legend.edgecolor": GRID_COLOR,
        "legend.fontsize": 10,
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.facecolor": DARK_BG,
    })


def _save(fig: plt.Figure, name: str, fig_dir: Path | None = None):
    d = fig_dir or FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    out = d / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# 1. Roofline
# ---------------------------------------------------------------------------
def roofline():
    """Roofline plot for CMP 100-210 showing compute / memory ceilings
    and kernel achieved throughput points."""
    # --- hardware ceilings (CMP 100-210 firmware-limited V100-class) ---
    # Derived from NCU representative Q3_K_S kernel:
    #   compute_throughput_pct=83.2, memory_throughput_pct=74.14,
    #   dram_gb_s=462.2, achieved_occupancy_pct=91.13
    # A standard V100-PCIe: ~28.3 TFLOPS FP16, 900 GB/s.
    # CMP 100-210 firmware limit gives ~50% of V100 tensor-core FP16.
    peak_flops = 14_000       # GFLOPS  (CMP 100-210 effective FP32-equivalent)
    peak_bw    = 625          # GB/s    (effective DRAM bandwidth ceiling)
    # Transition point: peak_flops / peak_bw  =  operational intensity knee
    knee_oi = peak_flops / peak_bw  # ~22.4 FLOP/byte

    # --- kernel data points ---
    #   (operational_intensity FLOP/byte,  achieved GFLOPS,  label)
    kernels = [
        (0.305, 140.4, "Qwen3.6 Q3_K_S\ndecode kernel"),
        (0.388, 107.6, "Qwen3.8 TQ3_4S\ndecode kernel"),
        (0.195,  90.2, "Qwen3.8 TQ3_4S\nprefill GEMV"),
    ]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 6.5))

    oi_range = np.logspace(-2, 2.5, 500)

    # Memory-bound region (left of knee)
    mem_oi = oi_range[oi_range <= knee_oi]
    mem_flops = mem_oi * peak_bw

    # Compute-bound region (right of knee)
    comp_oi = oi_range[oi_range >= knee_oi]
    comp_flops = np.full_like(comp_oi, peak_flops)

    # Roofline line
    ax.plot(mem_oi, mem_flops, color=MEM_COLOR, linewidth=2.5,
            label=f"Memory ceiling ({peak_bw} GB/s)", zorder=3)
    ax.axhline(peak_flops, color=COMPUTE_COLOR, linewidth=2.5, linestyle="--",
               label=f"Compute ceiling ({peak_flops} GFLOPS)", zorder=3)

    # Knee marker
    ax.plot(knee_oi, peak_flops, marker="D", markersize=9, color=TEXT_COLOR,
            zorder=5)
    ax.annotate(f"knee = {knee_oi:.1f} FLOP/B",
                xy=(knee_oi, peak_flops),
                xytext=(knee_oi * 1.8, peak_flops * 0.55),
                fontsize=9, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2),
                zorder=5)

    # Kernel points
    for oi, gflops, lbl in kernels:
        bound = "compute" if oi > knee_oi else "memory"
        ax.plot(oi, gflops, "o", markersize=10, color=KERNEL_COLOR,
                markeredgecolor=TEXT_COLOR, markeredgewidth=1.2, zorder=6)
        ax.annotate(lbl,
                    xy=(oi, gflops),
                    xytext=(12, 12), textcoords="offset points",
                    fontsize=8.5, color=TEXT_COLOR,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9),
                    bbox=dict(boxstyle="round,pad=0.3", fc=CARD_BG,
                              ec=GRID_COLOR, alpha=0.92),
                    zorder=6)

    # Shading: memory-bound / compute-bound
    ax.axvspan(1e-2, knee_oi, alpha=0.06, color=MEM_COLOR, zorder=0)
    ax.axvspan(knee_oi, 10**2.5, alpha=0.06, color=COMPUTE_COLOR, zorder=0)
    ax.text(knee_oi * 0.25, peak_flops * 0.12, "MEMORY-BOUND",
            fontsize=11, color=MEM_COLOR, alpha=0.55, fontweight="bold",
            ha="center")
    ax.text(knee_oi * 5, peak_flops * 0.12, "COMPUTE-BOUND",
            fontsize=11, color=COMPUTE_COLOR, alpha=0.55, fontweight="bold",
            ha="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 300)
    ax.set_ylim(10, peak_flops * 3)
    ax.set_xlabel("Operational Intensity  (FLOP / Byte)", fontsize=12)
    ax.set_ylabel("Throughput  (GFLOPS)", fontsize=12)
    ax.set_title("CMP 100-210  Roofline Model  —  Qwen3 Decode Kernels",
                 fontsize=14, fontweight="bold")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.92)

    _save(fig, "roofline_cmp100_210.png")


# ---------------------------------------------------------------------------
# 2. Tensor-core throughput (dp4a INT8 vs FP16)
# ---------------------------------------------------------------------------
def tensor_core_chart():
    """Bar chart: dp4a INT8 vs FP16 tensor-core throughput on V100 vs CMP 100-210."""
    categories = ["dp4a INT8", "FP16 Tensor Core"]
    # TFLOPS / TOPS  (INT8 measured in TOPS, FP16 in TFLOPS)
    v100_vals  = [112.0, 112.0]    # V100 Volta: INT8 224 TOPS / FP16 112 TFLOPS
    cmp_vals   = [52.0,  55.0]     # CMP 100-210 firmware-limited

    x = np.arange(len(categories))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars_v100 = ax.bar(x - width/2, v100_vals, width, label="V100 (Volta SM70)",
                       color=V100_COLOR, edgecolor=TEXT_COLOR, linewidth=0.8,
                       zorder=3)
    bars_cmp  = ax.bar(x + width/2, cmp_vals,  width,
                       label="CMP 100-210 (firmware-limited)",
                       color=CMP_COLOR, edgecolor=TEXT_COLOR, linewidth=0.8,
                       zorder=3)

    # Value labels
    for bars in (bars_v100, bars_cmp):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                    f"{h:.0f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=TEXT_COLOR)

    # Ratio annotation
    for i in range(len(categories)):
        ratio = cmp_vals[i] / v100_vals[i] * 100
        ax.text(x[i], max(v100_vals[i], cmp_vals[i]) + 10,
                f"CMP = {ratio:.0f}% of V100",
                ha="center", fontsize=9, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Throughput  (TFLOPS / TOPS)", fontsize=12)
    ax.set_title("dp4a INT8 vs FP16 Tensor-Core Throughput\nV100 vs CMP 100-210",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 140)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.92)

    _save(fig, "tensor_core_dp4a_vs_fp16.png")


# ---------------------------------------------------------------------------
# 3. Qwen3 model throughput (prefill + decode)
# ---------------------------------------------------------------------------
def throughput_chart():
    """Bar chart of Qwen3 prefill and decode tok/s from scoreboard data."""
    with open(SCOREBOARD_JSON) as f:
        sb = json.load(f)

    # Collect entries that have decode_tok_s or steady_decode_tok_s
    entries = []
    for row in sb["rows"]:
        name = row["name"]
        model = row.get("model", "?")
        decode = row.get("decode_tok_s") or row.get("steady_decode_tok_s")
        prompt = row.get("prompt_tok_s")
        if decode is not None:
            entries.append({
                "label": name.replace("fni8-", "").replace("-exact2k", "")
                         .replace("-fresh-v100", ""),
                "model": model,
                "decode": decode,
                "prefill": prompt,
                "hardware": row.get("hardware", ""),
                "status": row.get("status", ""),
            })

    if not entries:
        print("  [throughput_chart] no entries with tok/s found, skipping")
        return

    labels = [e["label"] for e in entries]
    decodes = [e["decode"] for e in entries]
    prefills = [e["prefill"] if e["prefill"] else 0 for e in entries]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_d = ax.bar(x - width/2, decodes, width, label="Decode tok/s",
                    color=ACCENT1, edgecolor=TEXT_COLOR, linewidth=0.7, zorder=3)
    bars_p = ax.bar(x + width/2, prefills, width, label="Prefill tok/s",
                    color=ACCENT3, edgecolor=TEXT_COLOR, linewidth=0.7, zorder=3)

    for bar in bars_d:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f"{h:.1f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=ACCENT1)
    for bar in bars_p:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f"{h:.1f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=ACCENT3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Tokens / second", fontsize=12)
    ax.set_title("Qwen3 Scoreboard — Throughput Comparison",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.92)

    # Subtitle with model names
    model_str = "  |  ".join(dict.fromkeys(e["model"] for e in entries))
    ax.text(0.5, 1.01, model_str, transform=ax.transAxes,
            fontsize=9, color=MUTED, ha="center", va="bottom")

    _save(fig, "throughput_qwen3.png")


# ---------------------------------------------------------------------------
# 4. Serve — Qwen3-8B prefill vs decode
# ---------------------------------------------------------------------------
def serve_8b_chart():
    """Bar chart: prefill vs decode for Qwen3-8B."""
    with open(SERVE_8B_JSON) as f:
        data = json.load(f)

    metrics = ["Load Time", "TTFT", "Prefill tok/s", "Decode tok/s"]
    values  = [data["load_s"], data["ttft_s"],
               data["prefill_tok_s"], data["decode_tok_s"]]
    colors  = [MUTED, ACCENT2, ACCENT3, ACCENT1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(metrics, values, color=colors,
                  edgecolor=TEXT_COLOR, linewidth=0.8, zorder=3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=TEXT_COLOR)

    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Qwen3-8B (INT8) — Serve Benchmark\n"
                 f"Peak VRAM: {data['peak_vram_gb']:.2f} GB  |  "
                 f"CUDA Graphs: {'on' if data['cuda_graph'] else 'off'}",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    _save(fig, "serve_qwen3_8b_throughput.png", SERVE_FIG_DIR)


# ---------------------------------------------------------------------------
# 5. Serve — VRAM comparison
# ---------------------------------------------------------------------------
def serve_vram_chart():
    """Bar chart comparing peak VRAM across served models."""
    models = []
    vrams  = []
    colors_list = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5]

    # Read each serve JSON
    for jf in sorted(SERVE_DIR.glob("*.json")):
        try:
            with open(jf) as f:
                d = json.load(f)
            if "peak_vram_gb" in d:
                models.append(d.get("model", jf.stem))
                vrams.append(d["peak_vram_gb"])
        except (json.JSONDecodeError, KeyError):
            continue

    if not models:
        print("  [serve_vram_chart] no models with VRAM data, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, vrams,
                   color=[colors_list[i % len(colors_list)] for i in range(len(models))],
                   edgecolor=TEXT_COLOR, linewidth=0.8, zorder=3)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.15, bar.get_y() + bar.get_height()/2,
                f"{w:.2f} GB", va="center",
                fontsize=11, fontweight="bold", color=TEXT_COLOR)

    ax.set_xlabel("Peak VRAM  (GB)", fontsize=12)
    ax.set_title("Model VRAM Usage — superl8-serve Benchmarks",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(vrams) * 1.25)
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)

    _save(fig, "serve_vram_usage.png", SERVE_FIG_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CHARTS = {
    "roofline":      roofline,
    "tensor_core":   tensor_core_chart,
    "throughput":    throughput_chart,
    "serve_8b":      serve_8b_chart,
    "serve_vram":    serve_vram_chart,
}


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark scoreboard charts")
    parser.add_argument("--only", nargs="+", choices=list(CHARTS),
                        help="Generate only these charts")
    args = parser.parse_args()

    _style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    to_run = args.only if args.only else list(CHARTS)
    print(f"Generating {len(to_run)} chart(s) -> {FIG_DIR}/")
    for name in to_run:
        print(f"  [{name}]")
        CHARTS[name]()
    print("Done.")


if __name__ == "__main__":
    main()
