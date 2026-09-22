# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Diffusion applicability — does int8 attention error COMPOUND over denoising steps?

The LLM validation (bench/real_model_eval.py) proved superl8 holds for a SINGLE forward
(+0.23% perplexity). Diffusion is different: it applies attention 20-50 times in an
iterative denoising loop, each step feeding the next — so a per-step error could in
principle accumulate. This test probes that mechanism directly.

It runs a stable, denoising-style damped iteration (x += alpha*(attn(x) - x), which
is contractive like real denoising) with fixed random Q/K/V projections, comparing
the fp16-attention trajectory to the superl8-int8-attention trajectory over many steps.
Diffusion attention is bidirectional, so this is NON-CAUSAL. If int8 error compounds,
the trajectories diverge; if it stays bounded (the expected, stable regime), they
track. This is a MECHANISTIC check on synthetic dynamics — a real-DiT sampling run is
the confirmation (see utils/docs/diffusion-applicability.md); the published precedent
is SageAttention, which superl8 follows, validated on real video diffusion.
"""
import pytest
import torch

import superl8
from tests.reference import sdpa_fp16
from tests.tolerances import cos_sim, rel_l1


@pytest.mark.correctness
@pytest.mark.parametrize("d", [64, 128])
def test_int8_attn_drift_bounded_over_denoising_steps(device, d):
    torch.manual_seed(0)
    b, hq, hkv, s, steps, alpha = 1, 8, 2, 512, 32, 0.15
    x = torch.randn(b, hq, s, d, device=device, dtype=torch.float16)
    # fixed random per-head projections -> realistic activation structure emerges.
    wq = torch.randn(hq, d, d, device=device, dtype=torch.float16) / (d ** 0.5)
    wk = torch.randn(hkv, d, d, device=device, dtype=torch.float16) / (d ** 0.5)
    wv = torch.randn(hkv, d, d, device=device, dtype=torch.float16) / (d ** 0.5)

    def proj(x, w):  # x [b,h,s,d] @ w [h,d,d] -> [b,h,s,d]
        return torch.einsum("bhsd,hde->bhse", x, w)

    def step(x, attn_fn):
        q = proj(x, wq)
        k = proj(x[:, :hkv], wk)  # GQA: kv from first hkv heads
        v = proj(x[:, :hkv], wv)
        a = attn_fn(q, k, v)      # non-causal (diffusion is bidirectional)
        return (x + alpha * (a - x)).half()

    x_fp, x_i8 = x.clone(), x.clone()
    for _ in range(steps):
        x_fp = step(x_fp, lambda q, k, v: sdpa_fp16(q, k.repeat_interleave(hq // hkv, 1),
                                                    v.repeat_interleave(hq // hkv, 1)))
        x_i8 = step(x_i8, lambda q, k, v: superl8.attn_int8_fwd(q, k, v, causal=False))

    c, l1 = cos_sim(x_i8, x_fp), rel_l1(x_i8, x_fp)
    print(f"\nD={d}: after {steps} denoising steps  cos={c:.5f}  rel_l1={l1:.4f}")
    # int8 attention error must stay BOUNDED (not compound to divergence) across the
    # denoising loop — the property diffusion needs.
    assert c >= 0.99, f"int8 attention drift compounded over {steps} steps: cos={c:.5f}"
    assert l1 <= 0.05, f"drift too large: rel_l1={l1:.4f}"
