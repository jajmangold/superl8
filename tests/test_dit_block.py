# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Fused DiT post-attention block kernel: RMSNorm → adaLN scale/shift/gate →
residual add → RoPE, all in fp32 registers, bf16 storage, one HBM round-trip.

Collapses N bandwidth-bound passes (RMSNorm → adaLN mod → residual → RoPE) into
1 — the highest-leverage bf16 fusion per the roofline analysis (sota-research-2026-07.md).

Gate (AGENTS.md, fused elementwise): cos ≈ 1 / low rel-L1 vs the fp32 oracle
(the reduction order of the sum-of-squares means it is NOT bit-exact vs torch's
rmsnorm).
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import superl8

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import time_ms  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [
    (1, 256),
    (8, 896),
    (16, 1024),
    (4096, 896),
    (2, 17),
    (5, 4097),
]


def _rope_tables(max_pos, rotary_dim, base=1e6):
    inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().cuda(), emb.sin().cuda()


def _ref(
    x, rms_weight, scale, shift, eps, gate=None, positions=None, cos=None, sin=None, rotary_dim=None
):
    """Pure-torch reference for the fused DiT block.

    Steps (all fp32, cast back to x.dtype at the end):
      1. RMSNorm: normed = x * rsqrt(mean(x^2) + eps) * rms_weight
      2. adaLN modulation: mod = normed * (1 + scale) + shift
      3. Gate (optional): mod = mod * sigmoid(gate)
      4. Residual: h = mod + x
      5. RoPE (optional): rotate first `rotary_dim` dims of h
    """
    dt = x.dtype
    xf = x.float()
    # 1. RMSNorm
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    normed = xf * rms * rms_weight.float()
    # 2. adaLN modulation
    modulated = normed * (1.0 + scale.float()) + shift.float()
    # 3. Gate
    if gate is not None:
        modulated = modulated * torch.sigmoid(gate.float())
    # 4. Residual
    h = modulated + xf
    # 5. RoPE — rotate first `rotary_dim` dims (kernel expects 2D [M,D] input)
    if (
        positions is not None
        and cos is not None
        and sin is not None
        and rotary_dim is not None
        and rotary_dim > 0
    ):
        rd = rotary_dim
        half = rd // 2
        c = cos[positions].to(torch.float32)  # [M, rd]
        s = sin[positions].to(torch.float32)  # [M, rd]
        hr, hp = h[:, :rd], h[:, rd:]
        hr_rot = torch.empty_like(hr)
        hr_rot[:, :half] = hr[:, :half] * c[:, :half] - hr[:, half:] * s[:, :half]
        hr_rot[:, half:] = hr[:, half:] * c[:, half:] + hr[:, :half] * s[:, half:]
        h = torch.cat((hr_rot, hp), dim=-1) if hp.shape[-1] != 0 else hr_rot
    return h.to(dt)


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _rl1(a, b):
    ref_f = b.float()
    return (a.float() - b.float()).abs().sum().item() / ref_f.abs().sum().clamp_min(1e-9).item()


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M,D", SHAPES)
def test_dit_block_matches_reference(M, D, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype) * 0.2
    s = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    sh = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    out = superl8.dit_block(x, w, s, sh, 1e-6)
    ref = _ref(x, w, s, sh, 1e-6)
    assert out.shape == (M, D) and out.dtype == dtype
    assert _cos(out, ref) > 0.9999, f"cos={_cos(out, ref)}"
    assert _rl1(out, ref) < 2e-3, f"relL1={_rl1(out, ref)}"


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
def test_dit_block_with_gate(dtype):
    torch.manual_seed(1)
    M, D = 8, 896
    x = torch.randn(M, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype) * 0.2
    s = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    sh = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    g = torch.randn(M, D, device="cuda", dtype=dtype) * 0.5
    out = superl8.dit_block(x, w, s, sh, 1e-6, gate=g)
    ref = _ref(x, w, s, sh, 1e-6, gate=g)
    assert _cos(out, ref) > 0.9999, f"cos gate={_cos(out, ref)}"
    assert _rl1(out, ref) < 2e-3, f"relL1 gate={_rl1(out, ref)}"


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
def test_dit_block_with_rope(dtype):
    torch.manual_seed(2)
    M, D, nh = 4, 128, 8
    rd = 64
    max_pos = 64
    cos, sin = _rope_tables(max_pos, rd)
    positions = torch.randint(0, max_pos, (M,), device="cuda")
    x = torch.randn(M, nh, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype) * 0.2
    s = torch.randn(D, device="cuda", dtype=dtype) * 0.1
    sh = torch.randn(D, device="cuda", dtype=dtype) * 0.1
    flat_x = x.reshape(-1, D)
    flat_s = s.unsqueeze(0).expand(M * nh, -1)
    flat_sh = sh.unsqueeze(0).expand(M * nh, -1)
    out = superl8.dit_block(
        flat_x,
        w,
        flat_s,
        flat_sh,
        1e-6,
        positions=positions.repeat(nh),
        cos=cos,
        sin=sin,
        rotary_dim=rd,
    )
    ref = _ref(
        flat_x,
        w,
        flat_s,
        flat_sh,
        1e-6,
        positions=positions.repeat(nh),
        cos=cos,
        sin=sin,
        rotary_dim=rd,
    )
    assert _cos(out, ref) > 0.9999, f"cos rope={_cos(out, ref)}"
    assert _rl1(out, ref) < 2e-3, f"relL1 rope={_rl1(out, ref)}"


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
def test_dit_block_deterministic(dtype):
    M, D = 8, 896
    x = torch.randn(M, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype) * 0.2
    s = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    sh = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    outs = [superl8.dit_block(x, w, s, sh, 1e-6) for _ in range(3)]
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
def test_dit_block_deterministic_with_gate_and_rope(dtype):
    M, D, rd = 4, 128, 64
    max_pos = 32
    cos, sin = _rope_tables(max_pos, rd)
    positions = torch.randint(0, max_pos, (M,), device="cuda")
    x = torch.randn(M, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype) * 0.2
    s = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    sh = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    g = torch.randn(M, D, device="cuda", dtype=dtype) * 0.5
    outs = [
        superl8.dit_block(
            x, w, s, sh, 1e-6, gate=g, positions=positions, cos=cos, sin=sin, rotary_dim=rd
        )
        for _ in range(3)
    ]
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
def test_dit_block_edge_empty(dtype):
    """M=0 or D=0 returns an empty tensor (kernel exits early)."""
    M, D = 0, 256
    x = torch.randn(M, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype) * 0.2
    s = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    sh = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    out = superl8.dit_block(x, w, s, sh, 1e-6)
    assert out.shape == (M, D) and out.numel() == 0

    M, D = 8, 0
    x = torch.randn(M, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype)
    s = torch.randn(M, D, device="cuda", dtype=dtype)
    sh = torch.randn(M, D, device="cuda", dtype=dtype)
    out = superl8.dit_block(x, w, s, sh, 1e-6)
    assert out.shape == (M, D) and out.numel() == 0


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
def test_dit_block_full_rope(dtype):
    """Rotary_dim == D (full RoPE, no pass-through dims)."""
    M, D, rd = 4, 64, 64
    max_pos = 32
    cos, sin = _rope_tables(max_pos, rd)
    positions = torch.randint(0, max_pos, (M,), device="cuda")
    x = torch.randn(M, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype) * 0.2
    s = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    sh = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    out = superl8.dit_block(x, w, s, sh, 1e-6, positions=positions, cos=cos, sin=sin, rotary_dim=rd)
    ref = _ref(x, w, s, sh, 1e-6, positions=positions, cos=cos, sin=sin, rotary_dim=rd)
    assert _cos(out, ref) > 0.9999, f"cos full_rope={_cos(out, ref)}"
    assert _rl1(out, ref) < 2e-3, f"relL1 full_rope={_rl1(out, ref)}"


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
def test_dit_block_odd_rotary_dim(dtype):
    """Non-tile-multiple rotary_dim (rd=34, half=17 — odd half)."""
    M, D, rd = 4, 128, 34
    max_pos = 32
    cos, sin = _rope_tables(max_pos, rd)
    positions = torch.randint(0, max_pos, (M,), device="cuda")
    x = torch.randn(M, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype) * 0.2
    s = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    sh = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    out = superl8.dit_block(x, w, s, sh, 1e-6, positions=positions, cos=cos, sin=sin, rotary_dim=rd)
    ref = _ref(x, w, s, sh, 1e-6, positions=positions, cos=cos, sin=sin, rotary_dim=rd)
    assert _cos(out, ref) > 0.9999, f"cos odd_rd={_cos(out, ref)}"
    assert _rl1(out, ref) < 2e-3, f"relL1 odd_rd={_rl1(out, ref)}"


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
def test_dit_block_compose_in_loop_stability(dtype):
    """Verify the kernel is stable when applied repeatedly in a loop
    (no divergent accumulation of error vs the eager reference composed
    the same number of times)."""
    M, D = 4, 128
    torch.manual_seed(42)
    x = torch.randn(M, D, device="cuda", dtype=dtype)
    w = torch.randn(D, device="cuda", dtype=dtype) * 0.2
    s = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1
    sh = torch.randn(M, D, device="cuda", dtype=dtype) * 0.1

    out = x
    ref = x.float()
    for _ in range(10):
        out = superl8.dit_block(out, w, s, sh, 1e-6)
        ref = _ref(ref.to(dtype), w, s, sh, 1e-6)
    assert out.shape == (M, D) and out.dtype == dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert _cos(out, ref.to(dtype)) > 0.999, f"cos compose={_cos(out, ref.to(dtype))}"
    assert _rl1(out, ref.to(dtype)) < 0.02, f"relL1 compose={_rl1(out, ref.to(dtype))}"


@CUDA
@pytest.mark.perf
@pytest.mark.parametrize("M,D", [(8, 896), (4096, 896)])
@pytest.mark.parametrize("variant", ["base", "gate", "rope", "gate_rope"])
def test_dit_block_faster_than_eager(M, D, variant):
    x = torch.randn(M, D, device="cuda", dtype=torch.float16)
    w = torch.randn(D, device="cuda", dtype=torch.float16) * 0.2
    s = torch.randn(M, D, device="cuda", dtype=torch.float16) * 0.1
    sh = torch.randn(M, D, device="cuda", dtype=torch.float16) * 0.1
    kwargs = {}
    if "gate" in variant:
        kwargs["gate"] = torch.randn(M, D, device="cuda", dtype=torch.float16) * 0.5
    if "rope" in variant:
        rd = 64
        cos, sin = _rope_tables(64, rd)
        kwargs.update(
            positions=torch.randint(0, 64, (M,), device="cuda"),
            cos=cos,
            sin=sin,
            rotary_dim=rd,
        )
    fused = time_ms(lambda: superl8.dit_block(x, w, s, sh, 1e-6, **kwargs))
    eager = time_ms(lambda: _ref(x, w, s, sh, 1e-6, **kwargs))
    assert fused <= eager * 1.2, f"{variant}: fused {fused:.4f}ms vs eager {eager:.4f}ms"
