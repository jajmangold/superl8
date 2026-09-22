# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Track-2 (issue #7) v1: the MLA decompress-path oracle and its absorb-path
reformulation are under test. This is the harness the future int8 dp4a
absorb-decode kernel must be validated against — it must be trustworthy
first, same principle as PR1's `test_reference.py` for softmax attention and
issue #6's `test_gated_delta_rule.py` for Gated-DeltaNet.

The core claim under test: `mla_decompress` (materializes per-head K/V) and
`mla_absorb` (MQA against the shared latent, via `absorb_qk_equiv` /
`absorb_ov_equiv`) are the SAME function, just reassociated. No quantization
or kernel is involved yet, so equivalence should hold to near machine
precision (fp64) and to ordinary floating-point-reassociation noise (fp32).
"""

import pytest
import torch

from tests.reference_mla import (
    MLAWeights,
    absorb_ov_equiv,
    absorb_qk_equiv,
    mla_absorb,
    mla_decompress,
    precompute_rope,
    project_kv_latent,
    random_mla_weights,
    rms_norm,
)

# (d_model, d_q_lora, d_c, H, d_h, d_r, d_v) — small; kept tiny relative to
# real DeepSeek-V3 (d_model=7168, d_c=512, H=128, d_h=128, d_r=64, d_v=128)
# but structurally identical. T values include ones that are NOT tile
# multiples of any plausible future kernel block size (1, 5, 17, 63).
CONFIGS = [
    (32, 24, 16, 2, 8, 4, 8),
    (48, 32, 24, 3, 16, 8, 12),
]
T_VALUES = [1, 5, 17, 63]


def make_weights(cfg, device, dtype=torch.float64):
    d_model, d_q_lora, d_c, h, d_h, d_r, d_v = cfg
    return random_mla_weights(d_model, d_q_lora, d_c, h, d_h, d_r, d_v, device=device, dtype=dtype)


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("t", T_VALUES)
def test_decompress_vs_absorb_equivalence_prefill(device, cfg, t):
    """Self-attention (h_q == h_kv), causal, fp64 — tightest equivalence check."""
    d_model = cfg[0]
    w = make_weights(cfg, device)
    b = 2
    h_t = torch.randn(b, t, d_model, device=device, dtype=torch.float64) * 0.1

    out_decompress = mla_decompress(h_t, h_t, w, causal=True)

    k_cos, k_sin = precompute_rope(t, w.d_r, offset=0, device=device, dtype=torch.float64)
    c_kv_n, k_rope = project_kv_latent(h_t, w, k_cos, k_sin)
    out_absorb = mla_absorb(h_t, c_kv_n, k_rope, w, causal=True)

    torch.testing.assert_close(out_decompress, out_absorb, rtol=1e-10, atol=1e-10)


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
def test_decompress_vs_absorb_equivalence_fp32(device, cfg):
    """Same equivalence in fp32 — the precision the real softmax/PV path runs
    in per AGENTS.md ("Softmax, LSE... stay fp32/fp16 -- never quantize
    them"). Looser tolerance: fp32 matmul reassociation noise, not a
    modeling error."""
    d_model = cfg[0]
    w = make_weights(cfg, device, dtype=torch.float32)
    b, t = 2, 17
    h_t = torch.randn(b, t, d_model, device=device, dtype=torch.float32) * 0.1

    out_decompress = mla_decompress(h_t, h_t, w, causal=True)
    k_cos, k_sin = precompute_rope(t, w.d_r, offset=0, device=device, dtype=torch.float32)
    c_kv_n, k_rope = project_kv_latent(h_t, w, k_cos, k_sin)
    out_absorb = mla_absorb(h_t, c_kv_n, k_rope, w, causal=True)

    torch.testing.assert_close(out_decompress, out_absorb, rtol=2e-3, atol=2e-3)


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
def test_decompress_vs_absorb_equivalence_decode_step(device, cfg):
    """The actual target shape: ONE new query token (Tq=1) against a cache of
    N prior tokens (Tq << N) — the MLA absorb path's raison d'etre."""
    d_model = cfg[0]
    w = make_weights(cfg, device)
    b, n_prior = 2, 23
    h_kv = torch.randn(b, n_prior, d_model, device=device, dtype=torch.float64) * 0.1
    h_q = torch.randn(b, 1, d_model, device=device, dtype=torch.float64) * 0.1
    h_full = torch.cat([h_kv, h_q], dim=1)  # decode token occupies position n_prior

    out_decompress = mla_decompress(h_q, h_full, w, causal=True)

    k_cos, k_sin = precompute_rope(n_prior + 1, w.d_r, offset=0, device=device, dtype=torch.float64)
    c_kv_n, k_rope = project_kv_latent(h_full, w, k_cos, k_sin)
    out_absorb = mla_absorb(h_q, c_kv_n, k_rope, w, causal=True)

    torch.testing.assert_close(out_decompress, out_absorb, rtol=1e-10, atol=1e-10)


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
def test_compose_in_loop_stability(device, cfg):
    """Grow the latent cache ONE decode step at a time (the real serving
    loop: append this step's (c_kv_n, k_rope) row, run mla_absorb for the new
    query) and check the final output matches a single batched decompress
    call over the whole sequence. This is the property a chunked/streaming
    kernel implementation depends on -- the cache is append-only and each
    step's cached latent row must never be recomputed."""
    d_model = cfg[0]
    w = make_weights(cfg, device)
    b, n_steps = 2, 9
    h_all = torch.randn(b, n_steps, d_model, device=device, dtype=torch.float64) * 0.1

    k_cos, k_sin = precompute_rope(n_steps, w.d_r, offset=0, device=device, dtype=torch.float64)
    c_kv_n_full, k_rope_full = project_kv_latent(h_all, w, k_cos, k_sin)

    outs = []
    for step in range(n_steps):
        h_q = h_all[:, step : step + 1, :]
        c_kv_n_cache = c_kv_n_full[:, : step + 1, :]  # append-only prefix
        k_rope_cache = k_rope_full[:, : step + 1, :]
        outs.append(mla_absorb(h_q, c_kv_n_cache, k_rope_cache, w, causal=True))
    out_incremental = torch.cat(outs, dim=1)

    out_batched = mla_decompress(h_all, h_all, w, causal=True)
    torch.testing.assert_close(out_incremental, out_batched, rtol=1e-10, atol=1e-10)


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
def test_determinism(device, cfg):
    """Same input x3 -> bitwise-equal output, both paths."""
    d_model = cfg[0]
    w = make_weights(cfg, device)
    b, t = 2, 11
    h_t = torch.randn(b, t, d_model, device=device, dtype=torch.float64) * 0.1

    out1 = mla_decompress(h_t, h_t, w, causal=True)
    out2 = mla_decompress(h_t, h_t, w, causal=True)
    out3 = mla_decompress(h_t, h_t, w, causal=True)
    assert torch.equal(out1, out2) and torch.equal(out2, out3)

    k_cos, k_sin = precompute_rope(t, w.d_r, offset=0, device=device, dtype=torch.float64)
    c_kv_n, k_rope = project_kv_latent(h_t, w, k_cos, k_sin)
    a1 = mla_absorb(h_t, c_kv_n, k_rope, w, causal=True)
    a2 = mla_absorb(h_t, c_kv_n, k_rope, w, causal=True)
    assert torch.equal(a1, a2)


@pytest.mark.correctness
def test_rms_norm_unit_scale_matches_torch(device):
    x = torch.randn(4, 16, device=device, dtype=torch.float64)
    w = torch.ones(16, device=device, dtype=torch.float64)
    got = rms_norm(x, w)
    want = torch.nn.functional.rms_norm(x, (16,), weight=w, eps=1e-6)
    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


@pytest.mark.correctness
def test_absorb_qk_equiv_matches_direct_decompression(device):
    """absorb_qk_equiv's folded weight must reproduce q_nope . k_nope exactly
    (this is the algebraic identity the whole absorb path rests on)."""
    cfg = CONFIGS[0]
    w = make_weights(cfg, device)
    n, t = 13, 7
    c_kv_n = torch.randn(2, n, w.d_c, device=device, dtype=torch.float64) * 0.1
    q_nope = torch.randn(2, w.h, t, w.d_h, device=device, dtype=torch.float64) * 0.1

    k_nope = torch.einsum("bnc,hcd->bhnd", c_kv_n, w.w_uk)
    direct = torch.einsum("bhtd,bhnd->bhtn", q_nope, k_nope)

    q_abs = torch.einsum("bhtd,hdc->bhtc", q_nope, absorb_qk_equiv(w))
    folded = torch.einsum("bhtc,bnc->bhtn", q_abs, c_kv_n)

    torch.testing.assert_close(direct, folded, rtol=1e-10, atol=1e-10)


@pytest.mark.correctness
def test_absorb_ov_equiv_matches_direct_decompression(device):
    """absorb_ov_equiv's folded weight must reproduce (P@V)@W_O exactly."""
    cfg = CONFIGS[0]
    w = make_weights(cfg, device)
    n, t = 13, 7
    c_kv_n = torch.randn(2, n, w.d_c, device=device, dtype=torch.float64) * 0.1
    p = torch.softmax(torch.randn(2, w.h, t, n, device=device, dtype=torch.float64), dim=-1)

    v = torch.einsum("bnc,hcd->bhnd", c_kv_n, w.w_uv)
    o = torch.einsum("bhtn,bhnd->bhtd", p, v)
    direct = o.transpose(1, 2).reshape(2, t, w.h * w.d_v) @ w.w_o

    o_abs = torch.einsum("bhtn,bnc->bhtc", p, c_kv_n)
    folded = torch.einsum("bhtc,hcm->btm", o_abs, absorb_ov_equiv(w))

    torch.testing.assert_close(direct, folded, rtol=1e-10, atol=1e-10)
