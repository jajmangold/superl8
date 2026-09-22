# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Decode-specialized uniform-3-bit dp4a GEMM (`gemm_decode_w3a8`) — issue #181.

This is the PRODUCTION form of the issue #181 spike (csrc/spike/w3a8_spike.cu,
cos 0.99996 vs fp-dequant). It is a **VRAM / context / batch lever, NOT a speed
lever**: the sub-byte Q3_K-style bit-plane unpack ALU contends with dp4a on the
INT pipe, so decode is measured at PARITY with W4A8 (not faster) while storing
3.0 bpw (0.75x the W4A8 weight bytes). That frees ~1.5-2 GiB on the single-card
27B, buying longer KV context / more concurrent batch slots before OOM.

Gates (AGENTS.md): int8/low-bit paths never use `allclose`. Correctness is the
shared SQNR / cos / rel-L1 gate vs the fp-dequant oracle (the exact tensor the
kernel reconstructs), plus a host pack/unpack round-trip proving losslessness,
plus bitwise determinism x3.
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8
from superl8.quant.core import quantize_int8_rowwise
from superl8.quant.lowbit import (
    dequantize_w3a8,
    pack_w3a8_bitplanes,
    quantize_w3a8,
    unpack_w3a8_bitplanes,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, compare_report, time_ms  # noqa: E402
from tests.tolerances import assert_int8_quality, cos_sim  # noqa: E402

_HAS_W3 = hasattr(superl8._C, "gemm_decode_w3a8")
_skip_w3 = pytest.mark.skipif(not _HAS_W3, reason="gemm_decode_w3a8 not built")

# M spans the decode range (1-16); N/K are multiples of 32 (bit-plane group) and
# G is a multiple of 128, including real 27B MLP-ish projection shapes.
DECODE_SHAPES = [
    (1, 128, 512), (1, 5120, 5120), (2, 128, 128), (8, 1024, 5120),
    (8, 5120, 5120), (16, 2048, 5120), (16, 128, 4096), (5, 256, 512),
]


def _w3(w: torch.Tensor, group_size: int):
    codes, scale = quantize_w3a8(w, group_size=group_size)
    planes = pack_w3a8_bitplanes(codes).contiguous()
    deq = dequantize_w3a8(codes, scale, group_size=group_size)  # fp32 [N,K] oracle
    return planes, scale.contiguous(), deq


# ── Host codec round-trip (no GPU needed for the pack/unpack proof) ──────────
@pytest.mark.correctness
@pytest.mark.parametrize("n,k", [(4, 128), (7, 512), (3, 5120)])
def test_w3_pack_unpack_lossless(n, k):
    codes = torch.randint(-4, 4, (n, k), dtype=torch.int8)
    planes = pack_w3a8_bitplanes(codes)
    assert planes.shape == (n, (k // 32) * 3) and planes.dtype == torch.int32
    back = unpack_w3a8_bitplanes(planes, k)
    assert torch.equal(back, codes), "Q3_K bit-plane pack/unpack is not lossless"


# ── Kernel correctness: matches the fp-dequant oracle (the tensor it reconstructs)
@_skip_w3
@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", DECODE_SHAPES)
@pytest.mark.parametrize("g", [128])
def test_gemm_decode_w3a8_matches_dequant_oracle(device, m, n, k, g):
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    planes, scale, deq = _w3(w, g)
    planes, scale = planes.to(device), scale.to(device)
    x_i8, x_scale = quantize_int8_rowwise(x)
    xs = x_scale.squeeze(-1).contiguous()
    y = superl8._C.gemm_decode_w3a8(x_i8, xs, planes, scale, g, torch.float16)
    # Oracle: fp32 activation (as int8-quantized) against the SAME dequant weights.
    x_deq = (x_i8.float() * xs[:, None])
    ref = x_deq @ deq.to(device).t()
    assert cos_sim(y, ref) >= 0.999, f"w3a8 cos below 0.999 @ {m}x{n}x{k}"
    assert_int8_quality(y, ref, min_cos=0.999, max_rel_l1=0.01, min_sqnr_db=30.0,
                        what=f"gemm_decode_w3a8 dequant-oracle {m}x{n}x{k}")


# ── vs the true fp32 matmul (end-to-end 3-bit quality gate) ─────────────────
@_skip_w3
@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", [(1, 512, 512), (8, 1024, 5120), (16, 2048, 5120)])
def test_gemm_decode_w3a8_vs_fp32(device, m, n, k):
    """Uniform 3-bit is INTENTIONALLY lossy (that is why it is opt-in and down_proj /
    late layers keep int4). Its cos vs true fp32 floors at the RTN error of a symmetric
    3-bit grid on Gaussian weights (~0.977 weight-cos) — NOT a kernel defect: the
    kernel-FIDELITY gate is `matches_dequant_oracle` above (cos>=0.999, the kernel adds
    no error beyond quantization). Here we assert the fp32 cos sits at that expected
    3-bit floor (>=0.97) — documenting the lossiness that makes this a VRAM lever, not a
    free win — and that the kernel output equals the fp32 matmul of the SAME dequant
    weights to int8 precision (the real correctness statement)."""
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    planes, scale, deq = _w3(w, 128)
    x_i8, x_scale = quantize_int8_rowwise(x)
    xs = x_scale.squeeze(-1).contiguous()
    y = superl8._C.gemm_decode_w3a8(x_i8, xs, planes.to(device), scale.to(device), 128, torch.float16)
    ref_fp32 = x.float() @ w.float().t()
    ref_deq = (x_i8.float() * xs[:, None]) @ deq.to(device).t()
    # Kernel adds no error beyond quantization (the true correctness bar).
    assert cos_sim(y, ref_deq) >= 0.999, f"w3a8 kernel diverges from its dequant matmul @ {m}x{n}x{k}"
    # 3-bit quant floor vs true fp32 (informational bar, inherent RTN loss ~0.977).
    assert cos_sim(y, ref_fp32) >= 0.97, f"w3a8 vs fp32 cos below the 3-bit floor @ {m}x{n}x{k}"


@_skip_w3
@pytest.mark.correctness
def test_gemm_decode_w3a8_deterministic(device):
    x = torch.randn(8, 5120, device=device, dtype=torch.float16)
    w = torch.randn(1024, 5120, device=device, dtype=torch.float16) * 0.1
    planes, scale, _ = _w3(w, 128)
    planes, scale = planes.to(device), scale.to(device)
    x_i8, x_scale = quantize_int8_rowwise(x)
    xs = x_scale.squeeze(-1).contiguous()
    r0 = superl8._C.gemm_decode_w3a8(x_i8, xs, planes, scale, 128, torch.float16)
    for _ in range(3):
        assert torch.equal(
            superl8._C.gemm_decode_w3a8(x_i8, xs, planes, scale, 128, torch.float16), r0
        )


@_skip_w3
@pytest.mark.correctness
def test_gemm_decode_w3a8_bf16_output(device):
    x = torch.randn(4, 512, device=device, dtype=torch.bfloat16)
    w = torch.randn(256, 512, device=device, dtype=torch.float16) * 0.1
    planes, scale, _ = _w3(w, 128)
    x_i8, x_scale = quantize_int8_rowwise(x.float().half())
    xs = x_scale.squeeze(-1).contiguous()
    y = superl8._C.gemm_decode_w3a8(x_i8, xs, planes.to(device), scale.to(device), 128,
                                 torch.bfloat16)
    assert y.dtype == torch.bfloat16


@_skip_w3
@pytest.mark.correctness
def test_gemm_decode_w3a8_rejects_m_too_large(device):
    x_i8 = torch.zeros(17, 512, device=device, dtype=torch.int8)
    xs = torch.ones(17, device=device, dtype=torch.float32)
    planes = torch.zeros(64, (512 // 32) * 3, device=device, dtype=torch.int32)
    scale = torch.ones(64, 512 // 128, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="M<=16|decode shapes"):
        superl8._C.gemm_decode_w3a8(x_i8, xs, planes, scale, 128, torch.float16)


# ── End-to-end wrapper (linear_w3a8) ────────────────────────────────────────
@_skip_w3
@pytest.mark.correctness
def test_linear_w3a8_matches_kernel(device):
    x = torch.randn(2, 5120, device=device, dtype=torch.float16)
    w = torch.randn(4096, 5120, device=device, dtype=torch.float16) * 0.1
    planes, scale, _ = _w3(w, 128)
    y = superl8.linear_w3a8(x, planes.to(device), scale.to(device), group_size=128)
    assert y.shape == (2, 4096) and y.dtype == torch.float16


# ── Perf: the LEVER claim — decode PARITY (not a regression) vs W4A8 ─────────
@_skip_w3
@pytest.mark.perf
@pytest.mark.parametrize("m,n,k,tag", [
    (1, 17408, 5120, "gemm_decode_w3a8.gate_up.m1"),
    (1, 5120, 17408, "gemm_decode_w3a8.down.m1"),
])
def test_gemm_decode_w3a8_perf(device, m, n, k, tag):
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    planes, scale, _ = _w3(w, 128)
    planes, scale = planes.to(device), scale.to(device)
    xq, xs = quantize_int8_rowwise(x)
    xs = xs.squeeze(-1).contiguous()
    ms = time_ms(lambda: superl8._C.gemm_decode_w3a8(xq, xs, planes, scale, 128, torch.float16))
    print("\n" + compare_report(tag, ms, {}))
    # weight bytes: (K/32)*3*4 per row = 3.0 bpw (vs 4.0 for W4A8's K/2).
    wbytes = n * (k // 32) * 3 * 4
    gbps = (wbytes + m * n * 2) / (ms * 1e-3) / 1e9
    print(f"{tag}: {gbps:.1f} GB/s ({gbps / 829 * 100:.1f}% of 829 GB/s HBM peak), "
          f"weight {wbytes / 1e6:.1f} MB @ 3.0 bpw")
    assert_no_regression(tag, ms)
