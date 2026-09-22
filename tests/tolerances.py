# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Tolerance helpers implementing the AGENTS.md numerics contract.

fp16 kernels: RELATIVE-to-fp32 bounds (ai-bond style) — the kernel's max abs
error vs the fp32 oracle must not exceed a multiple of the error PyTorch's own
fp16 baseline makes on the same inputs:
    forward:  err(kernel) <= 2 * err(pt_fp16) + 1e-5
    backward: err(kernel) <= 3 * err(pt_fp16) + 1e-4

int8 kernels: never `allclose` — a single rounding boundary legitimately flips.
Use SQNR / cosine similarity / relative-L1 (SageAttention reports cos ~= 1.0,
rel-L1 ~= 0.02 for int8 QK^T attention).
"""
from __future__ import annotations

import torch

FWD_MULT, FWD_ABS = 2.0, 1e-5
BWD_MULT, BWD_ABS = 3.0, 1e-4


def max_abs_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return (out.float() - ref.float()).abs().max().item()


def assert_relative_to_fp32(
    out: torch.Tensor,
    baseline_fp16: torch.Tensor,
    oracle_fp32: torch.Tensor,
    *,
    mult: float = FWD_MULT,
    abs_slack: float = FWD_ABS,
    what: str = "forward",
):
    """out must not err vs the fp32 oracle more than `mult`x the fp16 baseline's err."""
    err_out = max_abs_err(out, oracle_fp32)
    err_base = max_abs_err(baseline_fp16, oracle_fp32)
    limit = mult * err_base + abs_slack
    assert err_out <= limit, (
        f"{what}: kernel err {err_out:.3e} > {limit:.3e} "
        f"(= {mult}x fp16-baseline err {err_base:.3e} + {abs_slack:g})"
    )


def assert_finite(t: torch.Tensor, what: str = "output"):
    assert torch.isfinite(t.float()).all(), f"{what} contains NaN/Inf"


# ---- int8 metrics (gates for the quantized paths, PR3+) ----

def sqnr_db(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB (higher is better)."""
    ref_f, out_f = ref.float(), out.float()
    noise = (out_f - ref_f).pow(2).mean()
    signal = ref_f.pow(2).mean()
    if noise == 0:
        return float("inf")
    return (10.0 * torch.log10(signal / noise)).item()


def cos_sim(out: torch.Tensor, ref: torch.Tensor) -> float:
    a, b = out.float().flatten(), ref.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def rel_l1(out: torch.Tensor, ref: torch.Tensor) -> float:
    ref_f = ref.float()
    return ((out.float() - ref_f).abs().sum() / ref_f.abs().sum().clamp_min(1e-12)).item()


def assert_int8_quality(
    out: torch.Tensor,
    ref: torch.Tensor,
    *,
    min_cos: float = 0.999,
    max_rel_l1: float = 0.02,
    min_sqnr_db: float = 20.0,
    what: str = "int8 output",
):
    """The int8 accuracy gate (SageAttention-level bars). NEVER weaken to pass."""
    assert_finite(out, what)
    c, l1, s = cos_sim(out, ref), rel_l1(out, ref), sqnr_db(out, ref)
    assert c >= min_cos, f"{what}: cos-sim {c:.6f} < {min_cos}"
    assert l1 <= max_rel_l1, f"{what}: rel-L1 {l1:.4f} > {max_rel_l1}"
    assert s >= min_sqnr_db, f"{what}: SQNR {s:.1f} dB < {min_sqnr_db} dB"
