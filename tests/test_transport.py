# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Transport compression codec for the PCIe-1.0-x1 fleet. Tests first.

The fleet's interconnect is PCIe 1.0 x1 (~250 MB/s), ~3300x slower than HBM, so
multi-GPU collectives are brutally wire-bound. `superl8.compress_activation` reuses
the int8/int4/NF4 quant kernels as a transport codec: quantize+pack before the
wire, reconstruct on arrival. These tests pin the round-trip QUALITY per scheme
(so callers can choose against a bar), the losslessness of the packing, and that
the compressed payload actually shrinks the effective transfer.
"""
import pytest
import torch

import superl8
from superl8.transport import PCIE1_X1_BYTES_PER_S, code_entropy_bits


def _activation(shape, device, outlier=False):
    """A realistic-ish activation: Gaussian with optional per-channel outliers
    (real transformer activations are outlier-heavy — the int-quant stress case)."""
    x = torch.randn(*shape, device=device, dtype=torch.float16)
    if outlier:
        d = shape[-1]
        chan = torch.zeros(d, device=device, dtype=torch.float16)
        chan[:: max(1, d // 8)] = 12.0        # a few large channels
        x = x + chan
    return x


LOSSY = ["int8", "int4", "int4-had", "nf4"]


@pytest.mark.correctness
def test_fp16_scheme_is_lossless(device):
    x = _activation((4, 128, 256), device)
    c = superl8.compress_activation(x, scheme="fp16")
    assert torch.equal(superl8.decompress_activation(c), x)
    assert c.on_wire_bytes == x.numel() * 2


@pytest.mark.correctness
@pytest.mark.parametrize("scheme", LOSSY)
def test_roundtrip_quality(device, scheme):
    """Each lossy scheme reconstructs the activation to a scheme-appropriate bar
    and reports its true on-wire ratio. int8 ~2x (near-perfect), 4-bit ~4x."""
    x = _activation((8, 64, 256), device)
    gs = 64 if scheme != "int8" else None
    c = superl8.compress_activation(x, scheme=scheme, group_size=gs)
    rep = superl8.reconstruction_report(x, c)
    assert torch.isfinite(superl8.decompress_activation(c)).all()
    if scheme == "int8":
        assert rep["cos"] >= 0.9995 and rep["rel_l1"] <= 0.02, rep
        assert rep["ratio"] >= 1.9, rep
    else:  # 4-bit family
        assert rep["cos"] >= 0.98, rep
        assert rep["ratio"] >= 3.0, rep      # ~4x minus per-group scale overhead


@pytest.mark.correctness
def test_int4_packing_is_lossless_vs_fakequant(device):
    """compress->decompress must equal the underlying fake-quant (pack/unpack adds
    no error of its own — it only moves the already-quantized codes)."""
    from superl8.quant.lowbit import fake_quant_lowbit

    x = _activation((2, 32, 128), device)
    c = superl8.compress_activation(x, scheme="int4", group_size=128)
    got = superl8.decompress_activation(c).float()
    want = fake_quant_lowbit(x, 4, group_size=128).float()
    assert torch.allclose(got, want, atol=1e-3), (got - want).abs().max().item()


@pytest.mark.correctness
def test_hadamard_helps_on_outliers(device):
    """int4-had (rotate then quantize, un-rotate on decompress) must beat plain
    int4 when the activation has channel outliers — the incoherence rotation
    spreads them. On clean Gaussian it need not help; on outliers it must."""
    x = _activation((4, 64, 128), device, outlier=True)
    plain = superl8.reconstruction_report(x, superl8.compress_activation(x, scheme="int4", group_size=128))
    had = superl8.reconstruction_report(x, superl8.compress_activation(x, scheme="int4-had", group_size=128))
    assert had["rel_l1"] < plain["rel_l1"], (plain["rel_l1"], had["rel_l1"])


@pytest.mark.correctness
def test_nf4_beats_uniform_int4_on_gaussian(device):
    """NF4's non-uniform levels fit a Gaussian better than uniform int4 at equal
    bits/ratio — same 4-bit payload, lower error."""
    x = _activation((8, 64, 256), device)  # clean Gaussian
    u = superl8.reconstruction_report(x, superl8.compress_activation(x, scheme="int4", group_size=64))
    n = superl8.reconstruction_report(x, superl8.compress_activation(x, scheme="nf4", group_size=64))
    assert n["cos"] >= u["cos"], (u["cos"], n["cos"])


@pytest.mark.correctness
@pytest.mark.parametrize("scheme", LOSSY)
def test_effective_link_speedup(device, scheme):
    """Over the 250 MB/s link, the compressed payload transfers faster than raw
    fp16 — and because codec compute (~50 GB/s+ on-GPU) is orders of magnitude
    faster than the 250 MB/s wire, the effective speedup ~= the compression ratio.
    Compute cost is modeled from a CONSERVATIVE 50 GB/s codec throughput (the real
    thing is HBM-bound, far faster); the point is it's negligible vs the wire."""
    x = _activation((16, 128, 256), device)
    gs = 64 if scheme != "int8" else None
    c = superl8.compress_activation(x, scheme=scheme, group_size=gs)
    CODEC_BPS = 50e9  # conservative; HBM is 829 GB/s
    compress_ms = (x.numel() * 2) / CODEC_BPS * 1e3        # reads the fp16 source
    decompress_ms = c.on_wire_bytes / CODEC_BPS * 1e3      # reads the packed payload
    sp = superl8.link_speedup(x, c, compress_ms=compress_ms, decompress_ms=decompress_ms)
    rep = superl8.reconstruction_report(x, c)
    # speedup lands within a few % of the pure ratio -> compute is negligible.
    assert sp >= 0.95 * rep["ratio"], f"{scheme}: speedup {sp:.2f} << ratio {rep['ratio']:.2f}"
    floor = 1.8 if scheme == "int8" else 3.0
    assert sp > floor, f"{scheme}: link speedup {sp:.2f}x below {floor}x"


@pytest.mark.correctness
def test_entropy_bound_is_sane(device):
    """The code-histogram entropy must be <= the fixed width (an entropy coder can
    only help), and > 0 for a non-degenerate activation."""
    x = _activation((8, 64, 256), device)
    for scheme, w in (("int8", 8), ("int4", 4), ("nf4", 4)):
        gs = 64 if scheme != "int8" else None
        c = superl8.compress_activation(x, scheme=scheme, group_size=gs)
        h = code_entropy_bits(c)
        assert 0.0 < h <= w + 1e-6, f"{scheme}: entropy {h} out of (0, {w}]"


@pytest.mark.correctness
@pytest.mark.parametrize("scheme", LOSSY)
def test_deterministic(device, scheme):
    x = _activation((4, 32, 256), device)
    gs = 64 if scheme != "int8" else None
    c0 = superl8.compress_activation(x, scheme=scheme, group_size=gs)
    r0 = superl8.decompress_activation(c0)
    for _ in range(3):
        c = superl8.compress_activation(x, scheme=scheme, group_size=gs)
        assert torch.equal(c.payload, c0.payload)
        assert torch.equal(superl8.decompress_activation(c), r0)


@pytest.mark.correctness
def test_pcie1_x1_constant():
    # 2.5 GT/s * 8b/10b / 8 = 250 MB/s — the fleet's per-direction wire budget.
    assert PCIE1_X1_BYTES_PER_S == 250e6
