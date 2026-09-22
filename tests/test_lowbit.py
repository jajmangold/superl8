# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Sub-INT8 (2/3/4-bit) KV-cache feasibility — the accuracy gate decides.

sm_70 has no int4/int2 dp4a, so low-bit is a STORAGE codec (pack low-bit, unpack
to int8 for the matmul). This module measures the *attention-output* degradation
of each bit-width x role x rotation config on REALISTIC (channel-outlier) K, so we
only build packed kernels for configs that actually hold their accuracy.

Quant granularity mirrors the int8 recipe: K per-row (grouped over D), V
per-channel (over keys). Q is kept full here to isolate the KV-cache bit effect
(q is separately int8 in the real pipeline; that error is common to all rows).
"""
import pytest
import torch

import superl8  # noqa: F401  (ensures package import path)
from superl8.quant import smooth_k
from superl8.quant.lowbit import (
    fake_quant_lowbit,
    pack_rows_lowbit,
    quantize_lowbit,
    unpack_rows_lowbit,
)
from superl8.quant.rotation import rotate_last
from tests.reference import attention_fp32_oracle
from tests.tolerances import cos_sim, rel_l1, sqnr_db


def _attn_lowbit(q, k, v, *, bits_k, bits_v, rotate, gk, gv):
    """fp32 attention with K/V fake-quantized to low bits (rotation-consistent)."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    k_s, _ = smooth_k(k)  # mean-subtract (softmax-invariant), like the int8 path
    qr = rotate_last(q) if rotate else q
    kr = rotate_last(k_s) if rotate else k_s
    kq = fake_quant_lowbit(kr, bits_k, dim=-1, group_size=gk).float()   # K per-row/group over D
    vq = fake_quant_lowbit(v, bits_v, dim=-2, group_size=gv).float()    # V per-channel over keys
    s = torch.einsum("bhmd,bhnd->bhmn", qr.float(), kq) * scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", p, vq)


def _outlier_qkv(device, d=128):
    torch.manual_seed(0)
    q = torch.randn(1, 8, 512, d, device=device, dtype=torch.float16)
    k = torch.randn(1, 8, 512, d, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, 512, d, device=device, dtype=torch.float16)
    k[..., [0, 1, 7]] *= 40.0  # per-channel outliers (real activations look like this)
    return q, k, v


@pytest.mark.correctness
@pytest.mark.parametrize("bits", [3, 4, 5, 6, 8])
@pytest.mark.parametrize("d", [32, 64, 128])
def test_lowbit_packing_aligned_and_lossless(device, bits, d):
    """A full row packs to EXACTLY D*bits/32 int32 words (no waste) and unpacks
    losslessly — even for the non-byte-aligned 3/5/6-bit widths. This is the
    'align the tile with the packing period' point: D is a whole number of periods.
    """
    x = torch.randn(4, 16, d, device=device)
    codes, _ = quantize_lowbit(x, bits, dim=-1, group_size=16)
    packed = pack_rows_lowbit(codes, bits)
    # exact word count = D*bits/32 (int32), i.e. D*bits/8 bytes vs D bytes at int8.
    assert packed.shape[-1] == (d * bits) // 32, \
        f"expected {(d*bits)//32} words, got {packed.shape[-1]}"
    back = unpack_rows_lowbit(packed, bits, d)
    assert torch.equal(back, codes), f"pack/unpack not lossless at bits={bits} d={d}"


@pytest.mark.correctness
def test_lowbit_roundtrip_monotone(device):
    """Codec sanity: round-trip SQNR increases strictly with bit width."""
    x = torch.randn(4, 1024, device=device)
    sqnrs = [sqnr_db(fake_quant_lowbit(x, b, dim=-1, group_size=32), x) for b in (2, 3, 4, 8)]
    assert all(a < b for a, b in zip(sqnrs, sqnrs[1:])), f"SQNR not monotone in bits: {sqnrs}"
    assert sqnrs[-1] > 30.0, f"8-bit round-trip should be clean, got {sqnrs[-1]:.1f} dB"


@pytest.mark.correctness
def test_lowbit_feasibility_sweep(device):
    """Print the attention-output quality of each KV bit config vs the fp32
    oracle, and gate the ones we'd actually ship."""
    q, k, v = _outlier_qkv(device)
    oracle = attention_fp32_oracle(q, k, v)

    # Isolate K-bits (V at int8) and V-bits (K at int8) to see what each side
    # can bear, and sweep the group size (finer groups = less low-bit clipping).
    # (label, bits_k, bits_v, rotate, group_k, group_v)
    configs = [
        ("ref   K8V8 +rot ", 8, 8, True, None, None),
        # --- isolate K bit-width (V kept int8): 6/5/4/3/2 to find the K knee ---
        ("K-only K6V8 g16+r", 6, 8, True, 16, None),
        ("K-only K6V8 g16  ", 6, 8, False, 16, None),
        ("K-only K5V8 g16+r", 5, 8, True, 16, None),
        ("K-only K4V8 g32+r", 4, 8, True, 32, None),
        ("K-only K4V8 g16+r", 4, 8, True, 16, None),
        ("K-only K3V8 g16+r", 3, 8, True, 16, None),
        ("K-only K2V8 g16+r", 2, 8, True, 16, None),
        # --- isolate V bit-width (K kept int8); V is unrotated per-channel ---
        ("V-only K8V4 g128 ", 8, 4, True, None, 128),
        ("V-only K8V4 g32  ", 8, 4, True, None, 32),
        ("V-only K8V4 g16  ", 8, 4, True, None, 16),
        ("V-only K8V2 g16  ", 8, 2, True, None, 16),
        # --- combined caches, fine groups ---
        ("both  K6V6 g16+r ", 6, 6, True, 16, 16),
        ("both  K5V5 g16+r ", 5, 5, True, 16, 16),
        ("both  K4V4 g16+r ", 4, 4, True, 16, 16),
        # --- asymmetric K-high / V-4bit (K sensitive, V compressible) ---
        ("asym  K6V4 g16+r ", 6, 4, True, 16, 16),
        ("asym  K5V4 g16+r ", 5, 4, True, 16, 16),
    ]
    print(f"\n{'config':18s}  {'cos':>8s}  {'rel_l1':>8s}  {'sqnr_dB':>8s}")
    results = {}
    for label, bk, bv, rot, gk, gv in configs:
        out = _attn_lowbit(q, k, v, bits_k=bk, bits_v=bv, rotate=rot, gk=gk, gv=gv)
        c, l1, sq = cos_sim(out, oracle), rel_l1(out, oracle), sqnr_db(out, oracle)
        results[label.strip()] = (c, l1, sq)
        print(f"{label:18s}  {c:8.5f}  {l1:8.4f}  {sq:8.2f}")

    # --- honest, data-backed gates (assert the MEASURED truths) ---
    cos_k4 = results["K-only K4V8 g16+r"][0]   # 4-bit K (V int8)
    cos_v4 = results["V-only K8V4 g16"][0]     # 4-bit V (K int8)
    # (1) V is the compressible side: 4-bit V tolerates it (near int8 grade) while
    #     4-bit K does NOT — confirms TurboQuant's asymmetric-KV, and pins K as the
    #     sensitive side (it feeds softmax; error is amplified exponentially).
    assert cos_v4 >= 0.995, f"4-bit V should be tolerable, got cos={cos_v4:.5f}"
    assert cos_k4 < 0.99, f"4-bit K unexpectedly clean ({cos_k4:.5f}) — recheck the wall"
    assert cos_v4 > cos_k4 + 0.03, "expected 4-bit V >> 4-bit K (asymmetric KV)"
    # (2) finer V groups strictly help low-bit V (less clipping/rounding).
    assert results["V-only K8V4 g16"][1] < results["V-only K8V4 g128"][1]
    # (3) the wall: 2-bit (symmetric => ternary) collapses on BOTH sides.
    assert results["K-only K2V8 g16+r"][0] < 0.9
    assert results["V-only K8V2 g16"][0] < cos_v4
    # (4) 5/6-bit K: the knee. 6-bit K should recover most of what 4-bit lost and
    #     approach int8 grade; 5-bit lands in between.
    cos_k5 = results["K-only K5V8 g16+r"][0]
    cos_k6 = results["K-only K6V8 g16+r"][0]
    assert cos_k6 > cos_k5 > cos_k4, f"K bits not monotone: 6={cos_k6:.4f} 5={cos_k5:.4f} 4={cos_k4:.4f}"
    # 6-bit K reaches NEAR int8 grade even under these harsh 40x outliers (cleaner
    # on benign data); 5-bit is the knee (usable but visibly degraded here).
    assert cos_k6 >= 0.997, f"6-bit K should be ~int8 grade, got cos={cos_k6:.5f}"
    assert 0.98 <= cos_k5 < cos_k6, f"5-bit K knee off: cos={cos_k5:.5f}"
    # (5) the rotation SIGN FLIPS with grid fineness: on the coarse 4-bit K grid
    #     rotation hurt; on the finer 6-bit grid it helps again (like int8).
    assert results["K-only K6V8 g16+r"][1] < results["K-only K6V8 g16"][1]
    # (6) asymmetric K-high / V-4bit: K and V errors COMPOUND (~add in 1-cos), they do
    #     NOT just floor. K6V4 (0.994) < K8V4 (0.996) — 6-bit K is not free on top of a
    #     4-bit V. Still a sensible SMALLER point (0.625x int8); 5-bit K adds more (knee).
    #     Both K (period-aligned 6b) and V (nibble 4b) pack cleanly.
    cos_k8v4 = results["V-only K8V4 g16"][0]                 # == K8V4 (0.75x int8)
    cos_k6v4 = results["asym  K6V4 g16+r".strip()][0]        # 0.625x int8
    cos_k5v4 = results["asym  K5V4 g16+r".strip()][0]        # 0.5625x int8
    assert cos_k8v4 > cos_k6v4 > cos_k5v4, "K bits should order K8V4 > K6V4 > K5V4 (errors compound)"
    assert cos_k6v4 >= 0.99, f"K6V4 should stay usable, got cos={cos_k6v4:.5f}"
    # => KV size/quality ladder (harsh outliers; cleaner on benign): int8 1.0x .9998 |
    #    K6V6 .75x .997 | K8V4 .75x .996 | K6V4 .625x .994 | K5V4 .5625x .986 | K4V4 .5x
    #    .950(fail). Pick by accuracy budget; every width packs a row to exactly D*b/32
    #    int32 (period-aligned, lossless).
