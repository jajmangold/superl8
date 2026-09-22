# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Fused per-token DECODE elementwise kernels for the Gated-DeltaNet block:

  1. `superl8.causal_conv1d_silu_decode` — the causal depthwise conv1d(k)+SiLU
     token-shift (vLLM's `causal_conv1d_update` analogue). One launch replaces
     the eager cat/conv1d/slice/silu (~4 ops) in `GatedDeltaNetAttention._conv`.
  2. `superl8.gated_rmsnorm_decode` — the gated output RMSNorm (HF
     `Qwen3_5RMSNormGated`, "norm before gate"): per-head RMS over head_v_dim,
     ×gain, ×silu(z). One launch replaces ~6 eager fp32 reduce/rsqrt/mul/silu
     ops after the recurrence.

Both are CUDA-graph-capturable the same way the recurrence decode kernel is:
register/warp-reduction state, ZERO dynamic shared memory, no per-call
`cudaFuncSetAttribute`. Oracles are the exact eager math computed in fp32.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import superl8

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, time_ms  # noqa: E402
from tests.tolerances import cos_sim, rel_l1  # noqa: E402


# ============================================================================
# 1. causal depthwise conv1d(kernel=K) + SiLU, decode (L==1) token shift.
# ============================================================================
# (B, Wc, K) — Wc=6144 is Qwen3.5-0.8B's merged qkv width; include K=3 and K=4,
# B>1, a Wc that is not a multiple of 32, and a small Wc edge case.
CONV_SHAPES = [
    (1, 6144, 4),   # Qwen3.5 decode shape
    (1, 6144, 3),   # kernel=3 (LFM2-style short conv)
    (2, 6144, 4),   # batch > 1
    (1, 100, 4),    # tiny Wc, not a warp multiple
    (3, 257, 4),    # Wc not a multiple of 32, B=3
    (1, 4096, 4),   # power-of-two Wc
]

CONV_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def conv_silu_oracle(x, weight, tail):
    """Eager reference for the L==1 fused conv (matches
    `GatedDeltaNetAttention._conv` for a single token). x:[B,Wc], weight:[Wc,K],
    tail:[B,K-1,Wc]. Computed in fp32. Returns (out[B,Wc], new_tail[B,K-1,Wc])."""
    xf = x.float()
    wf = weight.float()
    tf = tail.float()
    window = torch.cat([tf, xf.unsqueeze(1)], dim=1)          # [B,K,Wc]
    conv = (window * wf.t().unsqueeze(0)).sum(1)              # [B,Wc]
    out = F.silu(conv)
    new_tail = torch.cat([tail, x.unsqueeze(1)], dim=1)[:, 1:]  # [B,K-1,Wc], no math
    return out, new_tail


def make_conv_inputs(shape, device, dtype):
    b, wc, k = shape
    x = torch.randn(b, wc, device=device, dtype=dtype)
    weight = torch.randn(wc, k, device=device, dtype=dtype) * 0.3
    tail = torch.randn(b, k - 1, wc, device=device, dtype=dtype)
    return x, weight, tail


@pytest.mark.correctness
@pytest.mark.parametrize("shape", CONV_SHAPES)
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv_smoke_shape_dtype(device, shape, dtype):
    b, wc, k = shape
    x, weight, tail = make_conv_inputs(shape, device, dtype)
    out, new_tail = superl8.causal_conv1d_silu_decode(x, weight, tail)
    assert out.shape == (b, wc) and new_tail.shape == (b, k - 1, wc)
    assert out.dtype == dtype and new_tail.dtype == dtype
    assert torch.isfinite(out.float()).all() and torch.isfinite(new_tail.float()).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", CONV_SHAPES)
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv_matches_oracle(device, shape, dtype):
    x, weight, tail = make_conv_inputs(shape, device, dtype)
    out_ref, tail_ref = conv_silu_oracle(x, weight, tail)
    out_cuda, tail_cuda = superl8.causal_conv1d_silu_decode(x, weight, tail)
    # new_tail is a pure copy/shift of the inputs (no arithmetic) -> bit-exact.
    assert torch.equal(tail_cuda, tail_ref)
    if dtype == torch.float32:
        torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4)
    else:
        # fp16/bf16 rounds; compare in fp32 with cosine + rel-L1 (AGENTS numerics).
        assert cos_sim(out_cuda.float(), out_ref) >= 0.999
        assert rel_l1(out_cuda.float(), out_ref) <= 0.02


@pytest.mark.correctness
@pytest.mark.parametrize("dtype", CONV_DTYPES)
def test_conv_zero_tail_prefill_start(device, dtype):
    """First decode step (no history): a zero tail must match plain causal conv."""
    b, wc, k = 1, 6144, 4
    x = torch.randn(b, wc, device=device, dtype=dtype)
    weight = torch.randn(wc, k, device=device, dtype=dtype) * 0.3
    tail = torch.zeros(b, k - 1, wc, device=device, dtype=dtype)
    out_ref, tail_ref = conv_silu_oracle(x, weight, tail)
    out_cuda, tail_cuda = superl8.causal_conv1d_silu_decode(x, weight, tail)
    assert torch.equal(tail_cuda, tail_ref)
    if dtype == torch.float32:
        torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4)
    else:
        assert cos_sim(out_cuda.float(), out_ref) >= 0.999


@pytest.mark.correctness
def test_conv_determinism(device):
    x, weight, tail = make_conv_inputs((1, 6144, 4), device, torch.float16)
    o1, t1 = superl8.causal_conv1d_silu_decode(x, weight, tail)
    o2, t2 = superl8.causal_conv1d_silu_decode(x, weight, tail)
    o3, t3 = superl8.causal_conv1d_silu_decode(x, weight, tail)
    assert torch.equal(o1, o2) and torch.equal(o2, o3)
    assert torch.equal(t1, t2) and torch.equal(t2, t3)


@pytest.mark.correctness
def test_conv_roll_matches_stepwise(device):
    """Two decode steps in sequence: feeding step-1's new_tail into step-2 must
    equal the eager conv carrying the same history (the real serving pattern)."""
    b, wc, k = 1, 6144, 4
    dtype = torch.float32
    weight = torch.randn(wc, k, device=device, dtype=dtype) * 0.3
    tail0 = torch.randn(b, k - 1, wc, device=device, dtype=dtype)
    x1 = torch.randn(b, wc, device=device, dtype=dtype)
    x2 = torch.randn(b, wc, device=device, dtype=dtype)
    _, tail1 = superl8.causal_conv1d_silu_decode(x1, weight, tail0)
    out2, _ = superl8.causal_conv1d_silu_decode(x2, weight, tail1)
    # Oracle: tail1 = [tail0[1:], x1]; step 2 over [tail1, x2].
    o2_ref, _ = conv_silu_oracle(x2, weight, tail1)
    torch.testing.assert_close(out2, o2_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
def test_conv_rejects_kernel_too_large(device):
    b, wc, k = 1, 64, 16  # K=16 > CONV_MAX_K
    x = torch.randn(b, wc, device=device, dtype=torch.float16)
    weight = torch.randn(wc, k, device=device, dtype=torch.float16)
    tail = torch.zeros(b, k - 1, wc, device=device, dtype=torch.float16)
    with pytest.raises(RuntimeError):
        superl8.causal_conv1d_silu_decode(x, weight, tail)


@pytest.mark.perf
def test_conv_decode_perf(device):
    x, weight, tail = make_conv_inputs((1, 6144, 4), device, torch.float16)
    ms = time_ms(lambda: superl8.causal_conv1d_silu_decode(x, weight, tail))
    assert_no_regression("causal_conv1d_silu_decode.b1w6144k4.fp16", ms)


# ============================================================================
# 2. gated output RMSNorm (HF Qwen3_5RMSNormGated: norm BEFORE gate), decode.
# ============================================================================
# (B, nv, vd) — Qwen3.5 decode is B=1, nv=16 value heads, vd=128; include a vd
# that is not a multiple of 32 (48), B>1, and a small edge case.
GN_SHAPES = [
    (1, 16, 128),   # Qwen3.5 decode shape
    (2, 16, 128),   # batch > 1
    (1, 32, 64),    # more heads, smaller vd
    (1, 4, 48),     # vd not a multiple of 32
    (4, 8, 96),     # vd not a warp multiple, B=4
    (1, 1, 128),    # single head
]


def gated_rmsnorm_oracle(o, gain, z, eps):
    """Eager reference matching `GatedDeltaNetAttention.forward` (lines 185-190):
    per-head RMS over vd, ×gain, then ×silu(z). fp32 throughout."""
    of = o.float()
    r = torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    out = of * r * gain.float()
    if z is not None:
        out = out * F.silu(z.float())
    return out


def make_gn_inputs(shape, device, with_z=True):
    b, nv, vd = shape
    o = torch.randn(b, nv, vd, device=device, dtype=torch.float32)
    gain = torch.randn(vd, device=device, dtype=torch.float32)
    z = torch.randn(b, nv, vd, device=device, dtype=torch.float32) if with_z else None
    return o, gain, z


@pytest.mark.correctness
@pytest.mark.parametrize("shape", GN_SHAPES)
@pytest.mark.parametrize("with_z", [True, False])
def test_gn_smoke_shape_dtype(device, shape, with_z):
    b, nv, vd = shape
    o, gain, z = make_gn_inputs(shape, device, with_z)
    out = superl8.gated_rmsnorm_decode(o, gain, z, 1e-6)
    assert out.shape == (b, nv, vd) and out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.correctness
@pytest.mark.parametrize("shape", GN_SHAPES)
@pytest.mark.parametrize("with_z", [True, False])
def test_gn_matches_oracle(device, shape, with_z):
    o, gain, z = make_gn_inputs(shape, device, with_z)
    eps = 1e-6
    out_ref = gated_rmsnorm_oracle(o, gain, z, eps)
    out_cuda = superl8.gated_rmsnorm_decode(o, gain, z, eps)
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.correctness
def test_gn_determinism(device):
    o, gain, z = make_gn_inputs((1, 16, 128), device, with_z=True)
    o1 = superl8.gated_rmsnorm_decode(o, gain, z, 1e-6)
    o2 = superl8.gated_rmsnorm_decode(o, gain, z, 1e-6)
    o3 = superl8.gated_rmsnorm_decode(o, gain, z, 1e-6)
    assert torch.equal(o1, o2) and torch.equal(o2, o3)


@pytest.mark.correctness
def test_gn_rejects_non_fp32(device):
    o, gain, z = make_gn_inputs((1, 16, 128), device, with_z=True)
    with pytest.raises(RuntimeError, match="fp32|float32|float"):
        superl8.gated_rmsnorm_decode(o.half(), gain, z, 1e-6)


@pytest.mark.correctness
def test_gn_rejects_vd_over_128(device):
    o, gain, z = make_gn_inputs((1, 2, 256), device, with_z=True)
    with pytest.raises(RuntimeError, match="128"):
        superl8.gated_rmsnorm_decode(o, gain, z, 1e-6)


@pytest.mark.perf
def test_gn_decode_perf(device):
    o, gain, z = make_gn_inputs((1, 16, 128), device, with_z=True)
    ms = time_ms(lambda: superl8.gated_rmsnorm_decode(o, gain, z, 1e-6))
    assert_no_regression("gated_rmsnorm_decode.b1nv16vd128.fp32", ms)
