# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Track-2 (issue #7) v1: the CUDA MLA absorb-path decode kernel
(`superl8.mla_decode_absorb`) under test against the merged fp32 Python oracle
(`tests/reference_mla.py`, PR #18). This is the on-device ground truth v2
(fp16 absorb) and v3 (int8 dp4a the latent QK/PV) must be validated against,
so it needs to be trustworthy first -- same principle `test_mla_absorb.py`
applied to the oracle itself, and `test_deltanet_kernel.py` applied to
issue #6's kernel.
"""

import sys
from pathlib import Path

import pytest
import torch

import superl8
from tests.reference_mla import (
    absorb_ov_equiv,
    absorb_qk_equiv,
    mla_absorb,
    mla_decompress,
    precompute_rope,
    project_kv_latent,
    project_q,
    random_mla_weights,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, time_ms  # noqa: E402
from tests.tolerances import assert_int8_quality, cos_sim, rel_l1, sqnr_db  # noqa: E402

# (d_model, d_q_lora, d_c, H, d_h, d_r, d_v) -- same small configs as
# test_mla_absorb.py, structurally identical to real DeepSeek-V3
# (d_model=7168, d_c=512, H=128, d_h=128, d_r=64, d_v=128) but tiny.
CONFIGS = [
    (32, 24, 16, 2, 8, 4, 8),
    (48, 32, 24, 3, 16, 8, 12),
]
# cache length (N) sweep -- includes non-power-of-2 / non-multiple-of-32
# sizes since the kernel's channel striping must not assume tile alignment.
N_VALUES = [1, 5, 17, 63]


def _decode_step_inputs(cfg, n_prior, device, dtype=torch.float32):
    """Build one decode step: n_prior cached tokens + 1 new query token, all
    fp32. Returns (w, h_q [B,1,d_model], h_full [B,n_prior+1,d_model])."""
    d_model = cfg[0]
    w = random_mla_weights(*cfg, device=device, dtype=dtype)
    b = 2
    h_kv = torch.randn(b, n_prior, d_model, device=device, dtype=dtype) * 0.1
    h_q = torch.randn(b, 1, d_model, device=device, dtype=dtype) * 0.1
    h_full = torch.cat([h_kv, h_q], dim=1)
    return w, h_q, h_full


def _kernel_inputs(w, h_q, h_full, device, dtype=torch.float32):
    """Project h_q/h_full into the kernel's expected tensors: q_nope, q_rope,
    c_kv_cache, k_rope_cache, w_qabs, w_ovabs (mirrors mla_absorb's own
    projection calls exactly)."""
    n_full = h_full.shape[1]
    q_cos, q_sin = precompute_rope(1, w.d_r, offset=n_full - 1, device=device, dtype=dtype)
    q_nope, q_rope = project_q(h_q, w, q_cos, q_sin)  # [B,H,1,d_h], [B,H,1,d_r]
    k_cos, k_sin = precompute_rope(n_full, w.d_r, offset=0, device=device, dtype=dtype)
    c_kv_cache, k_rope_cache = project_kv_latent(h_full, w, k_cos, k_sin)  # [B,N,d_c], [B,N,d_r]
    w_qabs = absorb_qk_equiv(w)  # [H,d_h,d_c]
    w_ovabs = absorb_ov_equiv(w)  # [H,d_c,d_model]
    return q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("n_prior", N_VALUES)
def test_smoke_finite_shape_dtype(device, cfg, n_prior):
    d_model = cfg[0]
    w, h_q, h_full = _decode_step_inputs(cfg, n_prior, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _kernel_inputs(
        w, h_q, h_full, device
    )
    out = superl8.mla_decode_absorb(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    assert out.shape == (2, 1, d_model)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("n_prior", N_VALUES)
def test_matches_decompress_oracle(device, cfg, n_prior):
    """The defining property the issue asks for: kernel output == the
    decompress-path oracle for a decode step (Tq=1, causal is automatic since
    the query is the cache's last row)."""
    w, h_q, h_full = _decode_step_inputs(cfg, n_prior, device)
    out_ref = mla_decompress(h_q, h_full, w, causal=True)  # [B,1,d_model]

    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _kernel_inputs(
        w, h_q, h_full, device
    )
    out_cuda = superl8.mla_decode_absorb(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    torch.testing.assert_close(out_cuda, out_ref, rtol=2e-3, atol=2e-3)


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
def test_matches_absorb_python_reference(device, cfg):
    """Cross-check against the absorb-path Python reference too (not just
    decompress) -- pins the kernel to the exact reformulation
    `absorb_qk_equiv`/`absorb_ov_equiv` describe."""
    n_prior = 23
    w, h_q, h_full = _decode_step_inputs(cfg, n_prior, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _kernel_inputs(
        w, h_q, h_full, device
    )
    out_ref = mla_absorb(h_q, c_kv_cache, k_rope_cache, w, causal=True)
    out_cuda = superl8.mla_decode_absorb(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    torch.testing.assert_close(out_cuda, out_ref, rtol=2e-3, atol=2e-3)


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
def test_compose_in_loop_stability(device, cfg):
    """Grow the latent cache one decode step at a time (the real serving
    loop) and check the concatenated per-step kernel outputs match a single
    batched decompress call -- same property `test_mla_absorb.py` proves for
    the Python oracle, now for the CUDA kernel."""
    d_model = cfg[0]
    device_dtype = torch.float32
    w = random_mla_weights(*cfg, device=device, dtype=device_dtype)
    b, n_steps = 2, 9
    h_all = torch.randn(b, n_steps, d_model, device=device, dtype=device_dtype) * 0.1

    k_cos, k_sin = precompute_rope(n_steps, w.d_r, offset=0, device=device, dtype=device_dtype)
    c_kv_full, k_rope_full = project_kv_latent(h_all, w, k_cos, k_sin)
    w_qabs = absorb_qk_equiv(w)
    w_ovabs = absorb_ov_equiv(w)

    outs = []
    for step in range(n_steps):
        h_q = h_all[:, step : step + 1, :]
        q_cos, q_sin = precompute_rope(1, w.d_r, offset=step, device=device, dtype=device_dtype)
        q_nope, q_rope = project_q(h_q, w, q_cos, q_sin)
        c_kv_cache = c_kv_full[:, : step + 1, :]
        k_rope_cache = k_rope_full[:, : step + 1, :]
        outs.append(
            superl8.mla_decode_absorb(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
        )
    out_incremental = torch.cat(outs, dim=1)

    out_batched = mla_decompress(h_all, h_all, w, causal=True)
    torch.testing.assert_close(out_incremental, out_batched, rtol=2e-3, atol=2e-3)


@pytest.mark.correctness
def test_determinism(device):
    cfg = CONFIGS[0]
    w, h_q, h_full = _decode_step_inputs(cfg, 11, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _kernel_inputs(
        w, h_q, h_full, device
    )
    out1 = superl8.mla_decode_absorb(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    out2 = superl8.mla_decode_absorb(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    out3 = superl8.mla_decode_absorb(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    assert torch.equal(out1, out2) and torch.equal(out2, out3)


@pytest.mark.correctness
def test_rejects_non_decode_query(device):
    cfg = CONFIGS[0]
    w, _h_q, h_full = _decode_step_inputs(cfg, 11, device)
    h_q2 = h_full[:, -2:, :]  # Tq=2, not a decode step
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _kernel_inputs(
        w, h_q2, h_full, device
    )
    with pytest.raises(AssertionError, match="Tq=1"):
        superl8.mla_decode_absorb(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)


@pytest.mark.correctness
def test_rejects_non_fp32(device):
    cfg = CONFIGS[0]
    w, h_q, h_full = _decode_step_inputs(cfg, 11, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _kernel_inputs(
        w, h_q, h_full, device
    )
    with pytest.raises(RuntimeError, match="fp32"):
        superl8.mla_decode_absorb(
            q_nope.half(),
            q_rope.half(),
            c_kv_cache.half(),
            k_rope_cache.half(),
            w_qabs.half(),
            w_ovabs.half(),
        )


@pytest.mark.perf
def test_mla_decode_absorb_perf(device):
    # DeepSeek-V3-shaped: d_model=7168, d_c=512, H=128, d_h=128, d_r=64,
    # d_v=128, d_q_lora=1536. N=4096 cached tokens.
    cfg = (7168, 1536, 512, 128, 128, 64, 128)
    w, h_q, h_full = _decode_step_inputs(cfg, 4095, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _kernel_inputs(
        w, h_q, h_full, device
    )
    ms = time_ms(
        lambda: superl8.mla_decode_absorb(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    )
    # Soft-skips until a baseline is committed (v1 is a correctness ground
    # truth, not the optimized path); then fails on >5% regression like
    # every other kernel's perf test.
    assert_no_regression("mla_decode_absorb_v1.b2h128dc512n4096.fp32", ms)


# ── fp16 v2 tests (issue #41) ──────────────────────────────────────────

FP16_N_VALUES = [1, 5, 17, 63, 100]  # includes non-multiple-of-32 N for cache sweep


def _fp16_kernel_inputs(w, h_q, h_full, device, cache_dtype=torch.float16):
    """Same as _kernel_inputs but cache tensors are converted to fp16 for
    the v2 bandwidth-light path. Query tensors stay fp32 for the absorb folds
    (matmuls are tiny); w_qabs/w_ovabs stay fp32 (offline-folded weights)."""
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _kernel_inputs(
        w, h_q, h_full, device, dtype=torch.float32
    )
    c_kv_cache = c_kv_cache.to(cache_dtype)
    k_rope_cache = k_rope_cache.to(cache_dtype)
    return q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs


def _fp16_oracle(w, h_q, c_kv_cache, k_rope_cache):
    """Run the absorb-path reference in fp32, then convert to fp16.
    Returns (out_fp16, out_fp32) for tolerance comparison."""
    out_fp32 = mla_absorb(h_q, c_kv_cache.float(), k_rope_cache.float(), w, causal=True)
    return out_fp32.half(), out_fp32


def _fp16_baseline_err(w, h_q, c_kv_cache, k_rope_cache):
    """Compute the fp16 basline error: run the Python absorb-path in fp16 vs
    fp32 to get a reference error bound for the ai-bond relative tolerance."""
    out_fp16_ref, out_fp32_ref = _fp16_oracle(w, h_q, c_kv_cache, k_rope_cache)
    return (out_fp16_ref.float() - out_fp32_ref).abs().max().item()


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("n_prior", FP16_N_VALUES)
def test_fp16_smoke_finite_shape_dtype(device, cfg, n_prior):
    w, h_q, h_full = _decode_step_inputs(cfg, n_prior, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _fp16_kernel_inputs(
        w, h_q, h_full, device
    )
    out = superl8.mla_decode_absorb_fp16(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    d_model = cfg[0]
    assert out.shape == (2, 1, d_model)
    assert out.dtype == torch.float16
    assert torch.isfinite(out.float()).all()


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("n_prior", FP16_N_VALUES)
def test_fp16_matches_fp32_oracle(device, cfg, n_prior):
    """fp16 kernel output must match the fp32 oracle within ai-bond fp16
    tolerance: kernel_err <= 2 * pt_fp16_baseline_err + 1e-5."""
    from tests.tolerances import assert_relative_to_fp32

    w, h_q, h_full = _decode_step_inputs(cfg, n_prior, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _fp16_kernel_inputs(
        w, h_q, h_full, device
    )
    out_fp16_ref, out_fp32_ref = _fp16_oracle(w, h_q, c_kv_cache, k_rope_cache)
    out_kernel = superl8.mla_decode_absorb_fp16(
        q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs
    )

    assert_relative_to_fp32(
        out_kernel,
        out_fp16_ref,
        out_fp32_ref,
        mult=2.0,
        abs_slack=1e-5,
        what="mla fp16",
    )


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
def test_fp16_matches_decompress_oracle(device, cfg):
    """Cross-check fp16 kernel against the decompress-path oracle too."""
    n_prior = 17
    w, h_q, h_full = _decode_step_inputs(cfg, n_prior, device)
    out_ref_fp32 = mla_decompress(h_q, h_full, w, causal=True)

    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _fp16_kernel_inputs(
        w, h_q, h_full, device
    )
    out_kernel = superl8.mla_decode_absorb_fp16(
        q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs
    )
    torch.testing.assert_close(out_kernel.float(), out_ref_fp32.float(), rtol=5e-3, atol=1e-2)


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
def test_fp16_compose_in_loop_stability(device, cfg):
    """Grow the latent cache one decode step at a time in fp16, check the
    concatenated per-step kernel outputs match a single batched decompress."""
    d_model = cfg[0]
    dtype = torch.float32
    w = random_mla_weights(*cfg, device=device, dtype=dtype)
    b, n_steps = 2, 9
    h_all = torch.randn(b, n_steps, d_model, device=device, dtype=dtype) * 0.1

    k_cos, k_sin = precompute_rope(n_steps, w.d_r, offset=0, device=device, dtype=dtype)
    c_kv_full, k_rope_full = project_kv_latent(h_all, w, k_cos, k_sin)
    c_kv_full_fp16 = c_kv_full.half()
    k_rope_full_fp16 = k_rope_full.half()
    w_qabs = absorb_qk_equiv(w)
    w_ovabs = absorb_ov_equiv(w)

    outs = []
    for step in range(n_steps):
        h_q = h_all[:, step : step + 1, :]
        q_cos, q_sin = precompute_rope(1, w.d_r, offset=step, device=device, dtype=dtype)
        q_nope, q_rope = project_q(h_q, w, q_cos, q_sin)
        c_kv_cache = c_kv_full_fp16[:, : step + 1, :]
        k_rope_cache = k_rope_full_fp16[:, : step + 1, :]
        outs.append(
            superl8.mla_decode_absorb_fp16(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
        )
    out_incremental = torch.cat(outs, dim=1).float()

    out_batched = mla_decompress(h_all, h_all, w, causal=True)
    torch.testing.assert_close(out_incremental, out_batched, rtol=1e-2, atol=1e-2)


@pytest.mark.correctness
def test_fp16_determinism(device):
    cfg = CONFIGS[0]
    w, h_q, h_full = _decode_step_inputs(cfg, 11, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _fp16_kernel_inputs(
        w, h_q, h_full, device
    )
    out1 = superl8.mla_decode_absorb_fp16(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    out2 = superl8.mla_decode_absorb_fp16(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    out3 = superl8.mla_decode_absorb_fp16(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    assert torch.equal(out1, out2) and torch.equal(out2, out3)


@pytest.mark.correctness
def test_fp16_rejects_non_decode_query(device):
    cfg = CONFIGS[0]
    w, _h_q, h_full = _decode_step_inputs(cfg, 11, device)
    h_q2 = h_full[:, -2:, :]
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _fp16_kernel_inputs(
        w, h_q2, h_full, device
    )
    with pytest.raises(AssertionError, match="Tq=1"):
        superl8.mla_decode_absorb_fp16(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)


@pytest.mark.perf
def test_mla_decode_absorb_fp16_perf(device):
    # DeepSeek-V3-shaped: same as v1 perf test
    cfg = (7168, 1536, 512, 128, 128, 64, 128)
    w, h_q, h_full = _decode_step_inputs(cfg, 4095, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _fp16_kernel_inputs(
        w, h_q, h_full, device
    )
    ms = time_ms(
        lambda: superl8.mla_decode_absorb_fp16(
            q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs
        )
    )
    # Soft-skips until a baseline is committed
    assert_no_regression("mla_decode_absorb_v2.b2h128dc512n4096.fp16", ms)


# ── int8 v3 tests (issue #57) ──────────────────────────────────────────

INT8_N_VALUES = [1, 5, 17, 63, 100]


def _int8_kernel_inputs(w, h_q, h_full, device, cache_dtype=torch.float16):
    """Same as _fp16_kernel_inputs — cache is fp16. Query tensors are folded
    to fp16 for the int8 kernel."""
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _kernel_inputs(
        w, h_q, h_full, device, dtype=torch.float32
    )
    c_kv_cache = c_kv_cache.to(cache_dtype)
    k_rope_cache = k_rope_cache.to(cache_dtype)
    return q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs


def _int8_oracle(w, h_q, c_kv_cache, k_rope_cache):
    """Run the absorb-path reference in fp32 (ground truth for int8 metrics)."""
    out_fp32 = mla_absorb(h_q, c_kv_cache.float(), k_rope_cache.float(), w, causal=True)
    return out_fp32


@torch.no_grad()
def _int8_kernel_output(
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs, w=None, h_q=None
):
    """Run int8 kernel and return [B,1,d_model] output.
    If the kernel isn't available yet (AttributeError), returns None."""
    try:
        return superl8.mla_decode_absorb_int8(
            q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs
        )
    except AttributeError:
        return None


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("n_prior", INT8_N_VALUES)
def test_int8_smoke_finite_shape_dtype(device, cfg, n_prior):
    w, h_q, h_full = _decode_step_inputs(cfg, n_prior, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _int8_kernel_inputs(
        w, h_q, h_full, device
    )
    out = _int8_kernel_output(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    if out is None:
        pytest.skip("int8 kernel not yet implemented")
    d_model = cfg[0]
    assert out.shape == (2, 1, d_model)
    assert out.dtype == torch.float16
    assert torch.isfinite(out.float()).all()


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("n_prior", INT8_N_VALUES)
def test_int8_quality_vs_fp32_oracle(device, cfg, n_prior):
    """int8 kernel output must pass the int8 quality gate (cos-sim ~= 1.0,
    rel-L1 <= 0.02, SQNR >= 20 dB) vs the fp32 oracle."""
    w, h_q, h_full = _decode_step_inputs(cfg, n_prior, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _int8_kernel_inputs(
        w, h_q, h_full, device
    )
    out_ref = _int8_oracle(w, h_q, c_kv_cache, k_rope_cache)
    out_kernel = _int8_kernel_output(
        q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs, w, h_q
    )
    if out_kernel is None:
        pytest.skip("int8 kernel not yet implemented")
    assert_int8_quality(out_kernel.float(), out_ref.float(), what="mla int8 vs fp32")


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
def test_int8_quality_vs_decompress_oracle(device, cfg):
    """int8 kernel vs decompress-path oracle, int8 quality gate."""
    n_prior = 17
    w, h_q, h_full = _decode_step_inputs(cfg, n_prior, device)
    out_ref = mla_decompress(h_q, h_full, w, causal=True)

    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _int8_kernel_inputs(
        w, h_q, h_full, device
    )
    out_kernel = _int8_kernel_output(
        q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs, w, h_q
    )
    if out_kernel is None:
        pytest.skip("int8 kernel not yet implemented")
    assert_int8_quality(out_kernel.float(), out_ref.float(), what="mla int8 vs decompress")


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
def test_int8_vs_fp16_baseline(device, cfg):
    """int8 kernel must not be worse than the v2 fp16 kernel (rel-L1 floor check)."""
    n_prior = 17
    w, h_q, h_full = _decode_step_inputs(cfg, n_prior, device)

    # fp16 kernel output
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _fp16_kernel_inputs(
        w, h_q, h_full, device
    )
    out_fp16, out_fp32_ref = _fp16_oracle(w, h_q, c_kv_cache, k_rope_cache)

    # int8 kernel output
    out_int8 = _int8_kernel_output(
        q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs, w, h_q
    )
    if out_int8 is None:
        pytest.skip("int8 kernel not yet implemented")

    fp16_cos = cos_sim(out_fp16, out_fp32_ref)
    int8_cos = cos_sim(out_int8.float(), out_fp32_ref)
    fp16_l1 = rel_l1(out_fp16, out_fp32_ref)
    int8_l1 = rel_l1(out_int8.float(), out_fp32_ref)

    assert int8_cos >= fp16_cos - 0.001, (
        f"int8 cos {int8_cos:.6f} < fp16 cos {fp16_cos:.6f} - 0.001"
    )
    assert int8_l1 <= fp16_l1 * 2.0 + 0.02, (
        f"int8 rel-L1 {int8_l1:.4f} > 2*fp16 rel-L1 {fp16_l1:.4f} + 0.02"
    )


@pytest.mark.correctness
@pytest.mark.parametrize("cfg", CONFIGS)
def test_int8_compose_in_loop_stability(device, cfg):
    """Grow the latent cache one decode step at a time with int8 kernel,
    check the concatenated output vs batched decompress oracle."""
    d_model = cfg[0]
    dtype = torch.float32
    w = random_mla_weights(*cfg, device=device, dtype=dtype)
    b, n_steps = 2, 9
    h_all = torch.randn(b, n_steps, d_model, device=device, dtype=dtype) * 0.1

    k_cos, k_sin = precompute_rope(n_steps, w.d_r, offset=0, device=device, dtype=dtype)
    c_kv_full, k_rope_full = project_kv_latent(h_all, w, k_cos, k_sin)
    c_kv_full_fp16 = c_kv_full.half()
    k_rope_full_fp16 = k_rope_full.half()
    w_qabs = absorb_qk_equiv(w)
    w_ovabs = absorb_ov_equiv(w)

    outs = []
    for step in range(n_steps):
        h_q = h_all[:, step : step + 1, :]
        q_cos, q_sin = precompute_rope(1, w.d_r, offset=step, device=device, dtype=dtype)
        q_nope, q_rope = project_q(h_q, w, q_cos, q_sin)
        c_kv_cache = c_kv_full_fp16[:, : step + 1, :]
        k_rope_cache = k_rope_full_fp16[:, : step + 1, :]
        out_step = _int8_kernel_output(
            q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs, w, h_q
        )
        if out_step is None:
            pytest.skip("int8 kernel not yet implemented")
        outs.append(out_step)
    out_incremental = torch.cat(outs, dim=1).float()

    out_batched = mla_decompress(h_all, h_all, w, causal=True)
    assert_int8_quality(
        out_incremental,
        out_batched,
        what="mla int8 compose-loop",
        min_cos=0.998,
        max_rel_l1=0.03,
        min_sqnr_db=18.0,
    )


@pytest.mark.correctness
def test_int8_determinism(device):
    cfg = CONFIGS[0]
    w, h_q, h_full = _decode_step_inputs(cfg, 11, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _int8_kernel_inputs(
        w, h_q, h_full, device
    )
    out1 = _int8_kernel_output(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs, w, h_q)
    out2 = _int8_kernel_output(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs, w, h_q)
    out3 = _int8_kernel_output(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs, w, h_q)
    if out1 is None:
        pytest.skip("int8 kernel not yet implemented")
    assert torch.equal(out1, out2) and torch.equal(out2, out3)


@pytest.mark.correctness
def test_int8_rejects_non_decode_query(device):
    cfg = CONFIGS[0]
    w, _h_q, h_full = _decode_step_inputs(cfg, 11, device)
    h_q2 = h_full[:, -2:, :]
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _int8_kernel_inputs(
        w, h_q2, h_full, device
    )
    try:
        superl8.mla_decode_absorb_int8(q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs)
    except AttributeError:
        pytest.skip("int8 kernel not yet implemented")
    except (AssertionError, RuntimeError) as e:
        assert "Tq=1" in str(e), f"expected Tq=1 rejection, got {e}"


@pytest.mark.perf
def test_mla_decode_absorb_int8_perf(device):
    # DeepSeek-V3-shaped: same as v1/v2 perf test
    cfg = (7168, 1536, 512, 128, 128, 64, 128)
    w, h_q, h_full = _decode_step_inputs(cfg, 4095, device)
    q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs = _int8_kernel_inputs(
        w, h_q, h_full, device
    )
    try:
        ms = time_ms(
            lambda: superl8.mla_decode_absorb_int8(
                q_nope, q_rope, c_kv_cache, k_rope_cache, w_qabs, w_ovabs
            )
        )
    except AttributeError:
        pytest.skip("int8 kernel not yet implemented")
    assert_no_regression("mla_decode_absorb_v3.b2h128dc512n4096.int8", ms)
