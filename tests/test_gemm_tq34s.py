# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Fused GGUF TQ3_4S -> dp4a GEMM (native TurboQuant type 46, superl8#272).

The fused kernel rotates the fp activation with the FORWARD RHT per 32-block
(signs -> WHT butterfly -> 1/sqrt(32)) BEFORE the per-32 q8_1 int8 quant,
unpacks the 3-bit codes to the corrected int8 centroid levels
``{-127,-82,-47,-16,15,46,81,127}``, and flushes each per-8 E3M5 scale into
the fp32 accumulator::

    out[m,n] = sum_b xs_b * sum_g E3M5(g)·(maxc/127) * dp4a(xhat_b, levels_g)

where the identity ``x^T·RHT_inv(v) = (RHT_fwd(x))^T·v`` (F = H·diag(SIGNS)/
sqrt(32)) moves the rotation to the activation side. See
csrc/docs/gguf-fused-kquant-dp4a.md + superl8/quant/tq34s.py.

Gates (SQNR/cos/rel-L1, never allclose — AGENTS.md):
  1. **FIDELITY** — fused output vs the fp32 CPU dequant oracle
     (:func:`superl8.quant.tq34s.reference_linear`) on the SAME synthesized bytes,
     SQNR >= 40 dB / cos >= 0.999 / rel-L1 <= 0.02.
  2. **ORACLE** — a real fp weight quantized to TQ3_4S bytes vs the fp32 matmul,
     at TQ3's intrinsic (encoder) bar.
  3. ragged M/N, bitwise determinism x3, an E3M5==0 (zero-scale) case,
     decode-vs-tile parity, and the M<=16 -> decode routing.
Perf marker: Qwen3.8-27B linear shapes (hidden 5120 / gate-up 17408).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import superl8
from superl8.quant import tq34s as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.tolerances import assert_int8_quality, cos_sim

pytestmark = pytest.mark.correctness

_C_mod = getattr(superl8, "_C", None)
if _C_mod is not None and type(_C_mod).__name__ != "_MissingC":
    _TILE = getattr(_C_mod, "gemm_tq34s", None)
    _DECODE = getattr(_C_mod, "gemm_decode_tq34s", None)
else:
    _TILE = _DECODE = None

_needs_fused = pytest.mark.skipif(
    _TILE is None, reason="fused TQ3_4S dp4a kernel not built (superl8#272)"
)


# ---------------------------------------------------------------------------
# TQ3_4S encoder for the ORACLE gate (produces real native bytes from an fp
# weight, then the exact fork dequant — the kernel's fidelity oracle).
# ---------------------------------------------------------------------------
def tq34s_quantize(w: torch.Tensor):
    """Quantize fp ``[N,K]`` (K%32==0) to native TQ3_4S bytes + fp32 dequant.

    Encoding is the inverse of the fork's dequant: ``v = RHT_fwd(w)`` per
    32-block, then per-8 group an E3M5 scale = ``amax / max|centroid|`` (rounded
    to the nearest E3M5 byte) and the nearest 8-centroid 3-bit codes.
    Reconstruction uses :func:`superl8.quant.tq34s.dequantize_tq34s_bytes` — the
    exact fork semantics the fused kernel must match.
    """
    N, K = w.shape
    assert K % T.QK_TQ3 == 0, "TQ3_4S needs K % 32 == 0"
    nblk = K // T.QK_TQ3
    wf = w.detach().float().cpu().numpy().reshape(N, nblk, T.QK_TQ3)
    v = T.rht_forward(wf)  # [N,nblk,32]
    vg = v.reshape(N, nblk, 4, 8)  # 4 per-8 groups
    amax = np.abs(vg).max(-1)  # [N,nblk,4]
    kmax = float(np.abs(T.TQ3_CENTROIDS).max())
    scale_target = np.where(amax == 0, 0.0, amax / kmax)
    table = T.decode_e3m5(np.arange(256, dtype=np.uint8))  # [256]
    sb = np.abs(table[None, None, None, :] - scale_target[..., None]).argmin(-1)
    sb = sb.astype(np.uint8)  # [N,nblk,4]
    scale = table[sb]  # exact E3M5 value
    scale_safe = np.where(scale == 0, 1.0, scale)
    vn = (vg / scale_safe[..., None])[..., None]  # [N,nblk,4,8,1]
    codes = np.abs(vn - T.TQ3_CENTROIDS[None, None, None, None, :]).argmin(-1)
    codes = codes.astype(np.uint8)  # [N,nblk,4,8]

    blk = np.zeros((N, nblk, 16), dtype=np.uint8)
    blk[:, :, 0:4] = sb
    for g in range(4):  # fork 3-byte pack loop
        idx = codes[:, :, g, :]  # [N,nblk,8]
        blk[:, :, 4 + 3 * g + 0] = (idx[..., 0] | (idx[..., 1] << 3) | (idx[..., 2] << 6)).astype(
            np.uint8
        )
        blk[:, :, 4 + 3 * g + 1] = (
            (idx[..., 2] >> 2) | (idx[..., 3] << 1) | (idx[..., 4] << 4) | (idx[..., 5] << 7)
        ).astype(np.uint8)
        blk[:, :, 4 + 3 * g + 2] = (
            (idx[..., 5] >> 1) | (idx[..., 6] << 2) | (idx[..., 7] << 5)
        ).astype(np.uint8)
    deq = T.dequantize_tq34s_bytes(blk.reshape(N, nblk * 16), K)
    return (
        torch.from_numpy(blk.reshape(N, nblk * 16)).contiguous(),
        torch.from_numpy(deq).contiguous(),
    )


def _tq3_bytes(n, k, rng, zero_half_blocks=False):
    """Synthesize native TQ3_4S bytes [n,(k//32)*16]; optionally zero every other
    32-block's 4 scale bytes (E3M5==0 zero-scale case)."""
    u8 = rng.integers(0, 256, (n, (k // T.QK_TQ3) * T.TQ3_TYPE_SIZE), dtype=np.uint8)
    if zero_half_blocks:
        u8 = u8.reshape(n, k // 32, 16)
        u8[:, 0::2, 0:4] = 0
        u8 = u8.reshape(n, (k // 32) * 16)
    return torch.from_numpy(u8).contiguous()


# ---------------------------------------------------------------------------
# Encoder sanity (no kernel — validates the ORACLE path itself).
# ---------------------------------------------------------------------------
def test_tq34s_encoder_reconstructs_weight():
    """The encoder's bytes dequant to a faithful reconstruction of w (the oracle
    gate's ground truth). TQ3 is 3-bit + WHT, so the bar is the intrinsic one."""
    torch.manual_seed(0)
    w = torch.randn(16, 512) * 0.1
    blk, deq = tq34s_quantize(w)
    assert blk.shape == (16, (512 // 32) * 16)
    assert deq.shape == (16, 512)
    assert torch.isfinite(deq).all()
    # 3-bit codes + E3M5 scales on random structureless gaussians (worst case,
    # no imatrix) — document the encoder's intrinsic quality floor.
    c = cos_sim(deq, w)
    assert c > 0.85, f"encoder reconstruction cos {c:.4f} too low"
    # round-trip through the reference oracle path
    x = torch.randn(2, 512, dtype=torch.float16)
    y = T.reference_linear(x, blk, 512)
    ref = x.float() @ deq.t()
    assert torch.equal(y, ref)


# ---------------------------------------------------------------------------
# THE FIDELITY gate: fused kernel == fp32 dequant oracle of the SAME bytes.
# ---------------------------------------------------------------------------
# M<=16 routes to the warp-per-column decode kernel, M>16 to the prefill tile —
# the sweep exercises BOTH. Ragged M/N are non-tile-multiples.
TQ3_SHAPES = [
    (1, 5120, 5120),  # M=1 decode, real attn shape
    (7, 6144, 5120),  # batched decode, ragged M
    (33, 8192, 5120),  # prefill tile, ragged M (>16)
    (17, 513, 1024),  # ragged N (not %64)
    (65, 130, 512),  # ragged M and N
    (200, 96, 512),  # ragged N below tile
]


@_needs_fused
@pytest.mark.parametrize("m,n,k", TQ3_SHAPES)
def test_linear_tq34s_reproduces_dequant(device, m, n, k):
    """Fused TQ3_4S dp4a == fp32 dequant matmul of the SAME bytes. The gate is
    vs the FULL-precision reference (only the int8 quant of the ROTATED
    activation separates them) — SQNR>=40 dB / cos>=0.999 per superl8#270."""
    rng = np.random.default_rng(m * 31 + n)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    blk = _tq3_bytes(n, k, rng).to(device)
    ref = T.reference_linear(x, blk, k)
    y = superl8.linear_tq34s(x, blk, k)
    assert y.shape == (m, n) and y.dtype == torch.float16
    assert_int8_quality(
        y,
        ref,
        min_cos=0.999,
        max_rel_l1=0.02,
        min_sqnr_db=40.0,
        what=f"linear_tq34s dequant-exact {m}x{n}x{k}",
    )


@_needs_fused
def test_gemm_tq34s_tile_reproduces_dequant(device):
    """Raw tile op (M>16) vs the oracle — the q4k-spine path in isolation."""
    m, n, k = 80, 6144, 5120
    rng = np.random.default_rng(11)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    blk = _tq3_bytes(n, k, rng).to(device)
    ref = T.reference_linear(x, blk, k)
    y = _TILE(x, blk, k, torch.float16)
    assert_int8_quality(
        y,
        ref,
        min_cos=0.999,
        max_rel_l1=0.02,
        min_sqnr_db=40.0,
        what=f"gemm_tq34s tile {m}x{n}x{k}",
    )


@_needs_fused
def test_gemm_tq34s_common_scale_candidate_reproduces_dequant(device):
    """llama.cpp-style per-32 common-Q8 scale candidate stays above the same
    fidelity bar as the exact per-8-flush production kernel before any perf A/B."""
    m, n, k = 80, 130, 512
    rng = np.random.default_rng(289)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    blk = _tq3_bytes(n, k, rng).to(device)
    ref = T.reference_linear(x, blk, k)
    y = _C_mod.gemm_tq34s_common_scale(x, blk, k, torch.float16)
    assert_int8_quality(
        y,
        ref,
        min_cos=0.999,
        max_rel_l1=0.02,
        min_sqnr_db=40.0,
        what=f"gemm_tq34s common-scale candidate {m}x{n}x{k}",
    )


@_needs_fused
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_tq34s_tm4tn4_candidate_matches_common_scale(device, dtype):
    """The 256-thread/TM4xTN4 register candidate changes ownership only, not
    the common-scale accumulation order or output bits."""
    m, n, k = 80, 130, 512
    x = torch.randn(m, k, device=device, dtype=dtype)
    blk = _tq3_bytes(n, k, np.random.default_rng(293)).to(device)
    ref = _C_mod.gemm_tq34s_common_scale(x, blk, k, dtype)
    y = _C_mod.gemm_tq34s_common_scale_tm4tn4(x, blk, k, dtype)
    assert torch.equal(y, ref)


@_needs_fused
def test_gemm_decode_tq34s_reproduces_dequant(device):
    """Raw warp-per-column decode op (M<=16) vs the oracle."""
    m, n, k = 1, 2048, 1024
    rng = np.random.default_rng(12)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    blk = _tq3_bytes(n, k, rng).to(device)
    ref = T.reference_linear(x, blk, k)
    y = _DECODE(x, blk, torch.float16)
    assert_int8_quality(
        y,
        ref,
        min_cos=0.999,
        max_rel_l1=0.02,
        min_sqnr_db=40.0,
        what="gemm_decode_tq34s dequant-exact",
    )


# ---------------------------------------------------------------------------
# ORACLE gate: a REAL fp weight quantized to TQ3_4S vs the fp32 matmul, at
# TQ3's intrinsic bar (the ENCODER's error — the kernel is proven exact at
# 40 dB separately by the fidelity gate above).
# ---------------------------------------------------------------------------
@_needs_fused
def test_linear_tq34s_matches_fp32_oracle(device):
    m, n, k = 128, 512, 1024
    torch.manual_seed(2)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k) * 0.1
    blk, _ = tq34s_quantize(w)
    y = superl8.linear_tq34s(x, blk.to(device), k)
    ref = x.float() @ w.float().to(device).t()
    # The oracle bar tracks the ENCODER's measured quality (see the sanity test);
    # TQ3 is 3-bit so the intrinsic SQNR on structureless gaussians is modest.
    assert_int8_quality(
        y, ref, min_cos=0.95, max_rel_l1=0.35, min_sqnr_db=8.0, what="linear_tq34s fp32-oracle"
    )


# ---------------------------------------------------------------------------
# Determinism x3 (bitwise), E3M5==0 zero-scale, bf16 output.
# ---------------------------------------------------------------------------
@_needs_fused
def test_linear_tq34s_deterministic(device):
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    x = torch.randn(64, 512, device=device, dtype=torch.float16)
    blk = _tq3_bytes(256, 512, rng).to(device)
    r0 = superl8.linear_tq34s(x, blk, 512)
    for _ in range(3):
        assert torch.equal(superl8.linear_tq34s(x, blk, 512), r0)


@_needs_fused
def test_linear_tq34s_zero_scale(device):
    """E3M5==0 (zero scale) case: every other 32-block's 4 scale bytes are 0, so
    those weight columns vanish. The fused kernel must track the oracle (the
    activation quant is the only error source), and an ALL-zero weight must
    produce an all-zero output."""
    rng = np.random.default_rng(4)
    m, n, k = 8, 512, 1024
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    blk = _tq3_bytes(n, k, rng, zero_half_blocks=True).to(device)
    ref = T.reference_linear(x, blk, k)
    y = superl8.linear_tq34s(x, blk, k)
    assert_int8_quality(
        y, ref, min_cos=0.999, max_rel_l1=0.02, min_sqnr_db=40.0, what="linear_tq34s zero-scale"
    )
    zeros = torch.zeros(n, (k // T.QK_TQ3) * T.TQ3_TYPE_SIZE, device=device, dtype=torch.uint8)
    yz = superl8.linear_tq34s(x, zeros, k)
    assert torch.equal(yz, torch.zeros(m, n, device=device, dtype=torch.float16))


@_needs_fused
def test_linear_tq34s_bf16_output(device):
    rng = np.random.default_rng(5)
    m, n, k = 16, 512, 512
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    blk = _tq3_bytes(n, k, rng).to(device)
    ref = T.reference_linear(x, blk, k)
    y = superl8.linear_tq34s(x, blk, k, out_dtype=torch.bfloat16)
    assert y.dtype == torch.bfloat16
    assert_int8_quality(
        y.float(), ref, min_cos=0.999, max_rel_l1=0.02, min_sqnr_db=38.0, what="linear_tq34s bf16"
    )


@_needs_fused
def test_linear_tq34s_bias(device):
    rng = np.random.default_rng(6)
    m, n, k = 4, 64, 128
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    blk = _tq3_bytes(n, k, rng).to(device)
    bias = torch.randn(n, device=device)
    y = superl8.linear_tq34s(x, blk, k, bias=bias)
    y0 = superl8.linear_tq34s(x, blk, k)
    # The wrapper adds bias in fp32 then casts back to out_dtype — bit-equal to
    # the same rounding on the bias-free kernel output.
    assert torch.equal(y, (y0 + bias).to(torch.float16))


# ---------------------------------------------------------------------------
# Decode (warp-per-column) vs tile (prefill) — same int math, different launch.
# ---------------------------------------------------------------------------
_DECODE_SHAPES = [(1, 4096, 3072), (4, 4096, 4096), (8, 512, 768), (16, 4096, 3072)]


@_needs_fused
@pytest.mark.parametrize("m,n,k", _DECODE_SHAPES)
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_gemm_decode_tq34s_matches_tile(device, m, n, k, dt):
    """Decode MMVQ kernel must equal the tile kernel (identical activation
    rotation + quant, identical int8 levels; only fp32 accumulation order
    differs)."""
    torch.manual_seed(m * 13 + n)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    rng = np.random.default_rng(m + n)
    blk = _tq3_bytes(n, k, rng).to(device)
    ref = _TILE(x, blk, k, dt)
    y = _DECODE(x, blk, dt)
    assert y.shape == (m, n) and y.dtype == dt
    assert_int8_quality(
        y,
        ref.float(),
        min_cos=0.9999,
        max_rel_l1=1e-3,
        min_sqnr_db=40.0,
        what=f"gemm_decode_tq34s vs tile {m}x{n}x{k} {dt}",
    )


@_needs_fused
def test_linear_tq34s_routes_decode_at_small_m(device):
    """linear_tq34s must route M<=16 to the warp-per-column decode kernel and
    match the tile kernel's math."""
    x = torch.randn(1, 1024, device=device, dtype=torch.float16)
    rng = np.random.default_rng(7)
    blk = _tq3_bytes(4096, 1024, rng).to(device)
    called = {}
    orig = _DECODE

    def spy(*a, **k):
        called["decode"] = True
        return orig(*a, **k)

    superl8._C.gemm_decode_tq34s = spy
    try:
        y = superl8.linear_tq34s(x, blk, 1024)
    finally:
        superl8._C.gemm_decode_tq34s = orig
    assert called.get("decode"), "linear_tq34s did not route M=1 to the decode kernel"
    ref = _TILE(x, blk, 1024, torch.float16)
    assert_int8_quality(
        y,
        ref.float(),
        min_cos=0.9999,
        max_rel_l1=1e-3,
        min_sqnr_db=40.0,
        what="linear_tq34s decode-route",
    )


@_needs_fused
def test_linear_tq34s_routes_prefill_to_common_scale_candidate(device, monkeypatch):
    """M>16 uses the quality-gated common-scale tile while decode remains on
    the exact warp-per-column kernel."""
    x = torch.randn(17, 512, device=device, dtype=torch.float16)
    blk = _tq3_bytes(1024, 512, np.random.default_rng(289)).to(device)
    called = {}
    common = superl8._C.gemm_tq34s_common_scale

    def spy(*args, **kwargs):
        called["common"] = True
        return common(*args, **kwargs)

    monkeypatch.setattr(superl8._C, "gemm_tq34s_common_scale", spy)
    y = superl8.linear_tq34s(x, blk, 512)

    assert called.get("common"), "linear_tq34s did not route prefill to common-scale tile"
    assert y.shape == (17, 1024)


@_needs_fused
def test_linear_tq34s_routes_k_heavy_prefill_to_narrow_accumulators(device, monkeypatch):
    """The 256-thread kernel is selected only for its measured winning region K>=N."""
    x = torch.randn(17, 512, device=device, dtype=torch.float16)
    blk = _tq3_bytes(130, 512, np.random.default_rng(293)).to(device)
    called = {}
    narrow = superl8._C.gemm_tq34s_common_scale_tm4tn4

    def spy(*args, **kwargs):
        called["narrow"] = True
        return narrow(*args, **kwargs)

    monkeypatch.setattr(superl8._C, "gemm_tq34s_common_scale_tm4tn4", spy)
    y = superl8.linear_tq34s(x, blk, 512)

    assert called.get("narrow"), "K>=N prefill did not route to narrow accumulators"
    assert y.shape == (17, 130)


@_needs_fused
def test_linear_tq34s_narrow_accumulator_rollback_uses_common(device, monkeypatch):
    x = torch.randn(17, 512, device=device, dtype=torch.float16)
    blk = _tq3_bytes(130, 512, np.random.default_rng(294)).to(device)
    called = {}
    common = superl8._C.gemm_tq34s_common_scale

    def spy(*args, **kwargs):
        called["common"] = True
        return common(*args, **kwargs)

    monkeypatch.setenv("FNI8_TQ34S_NARROW_ACC", "0")
    monkeypatch.setattr(superl8._C, "gemm_tq34s_common_scale", spy)
    y = superl8.linear_tq34s(x, blk, 512)

    assert called.get("common"), "narrow rollback did not restore common-scale tile"
    assert y.shape == (17, 130)


@_needs_fused
def test_linear_tq34s_common_scale_rollback_routes_exact_tile(device, monkeypatch):
    x = torch.randn(17, 512, device=device, dtype=torch.float16)
    blk = _tq3_bytes(130, 512, np.random.default_rng(290)).to(device)
    called = {}
    exact = superl8._C.gemm_tq34s

    def spy(*args, **kwargs):
        called["exact"] = True
        return exact(*args, **kwargs)

    monkeypatch.setenv("FNI8_TQ34S_COMMON_SCALE", "0")
    monkeypatch.setattr(superl8._C, "gemm_tq34s", spy)
    y = superl8.linear_tq34s(x, blk, 512)

    assert called.get("exact"), "rollback did not restore the exact per-8 tile"
    assert y.shape == (17, 130)


# ---------------------------------------------------------------------------
# superl8.linear dispatch keeps routing gguf_kquant/tq3_4s to linear_tq34s.
# ---------------------------------------------------------------------------
@_needs_fused
def test_linear_dispatch_tq34s_fused(device):
    from superl8.format import QTensor

    rng = np.random.default_rng(8)
    u8 = rng.integers(0, 256, (6, (128 // 32) * 16), dtype=np.uint8)
    qt = QTensor(
        torch.from_numpy(u8.copy()).to(device),
        None,
        scheme="gguf_kquant",
        group_size=32,
        codebook="tq3_4s",
    )
    x = torch.randn(3, 128, device=device, dtype=torch.float16)
    y = superl8.linear(x, qt)
    y2 = superl8.linear_tq34s(x, qt.data, 128)
    assert y.shape == (3, 6)
    assert torch.equal(y, y2)


# ---------------------------------------------------------------------------
# Perf marker — Qwen3.8-27B linear shapes (hidden 5120, gate/up 17408).
# Decode (M=1) is the warp-per-column MMVQ; prefill is the tile. Identifies
# the exact shapes in the report.
# ---------------------------------------------------------------------------
@pytest.mark.perf
@_needs_fused
def test_gemm_tq34s_decode_perf(device):
    import sys as _sys
    from pathlib import Path as _P

    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from bench.harness import assert_no_regression, compare_report, time_ms

    rng = np.random.default_rng(0)
    for tag, m, n, k in [
        ("gemm_decode_tq34s.attn.m1n5120k5120", 1, 5120, 5120),
        ("gemm_decode_tq34s.gate_up.m1n17408k5120", 1, 17408, 5120),
        ("gemm_decode_tq34s.down.m1n5120k17408", 1, 5120, 17408),
    ]:
        x = torch.randn(m, k, device=device, dtype=torch.float16)
        blk = _tq3_bytes(n, k, rng).to(device)
        dec_ms = time_ms(lambda x=x, blk=blk: _DECODE(x, blk, torch.float16))
        tile_ms = time_ms(lambda x=x, blk=blk, k=k: _TILE(x, blk, k, torch.float16))
        tops = 2.0 * m * n * k / (dec_ms * 1e-3) / 1e12
        print(
            "\n"
            + compare_report(tag, dec_ms, {"gemm_tq34s.tile": tile_ms})
            + f" | {tops:.2f} int8-TOP/s"
        )
        assert_no_regression(tag, dec_ms)


@pytest.mark.perf
@_needs_fused
def test_gemm_tq34s_prefill_perf(device):
    import sys as _sys
    from pathlib import Path as _P

    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from bench.harness import assert_no_regression, compare_report, time_ms

    rng = np.random.default_rng(1)
    for tag, m, n, k in [
        ("gemm_tq34s.prefill.attn.m2048n5120k5120", 2048, 5120, 5120),
        ("gemm_tq34s.prefill.gate_up.m2048n17408k5120", 2048, 17408, 5120),
    ]:
        x = torch.randn(m, k, device=device, dtype=torch.float16)
        blk = _tq3_bytes(n, k, rng).to(device)
        ms = time_ms(lambda x=x, blk=blk, k=k: _TILE(x, blk, k, torch.float16))
        tops = 2.0 * m * n * k / (ms * 1e-3) / 1e12
        print("\n" + compare_report(tag, ms, {}) + f" | {tops:.1f} int8-TOP/s")
        assert_no_regression(tag, ms)
