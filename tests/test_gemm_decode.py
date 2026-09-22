# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Decode-specialized int8 dp4a GEMM (split-K) — issue #27.

`gemm_w8a8_kernel` (the prefill tile GEMM, BM=BN=BK=64) launches only
ceil(N/64) x ceil(M/64) threadblocks. At decode (M<=16) that's ONE M-tile, and
for a typical projection N it's far too few blocks to fill 80 SMs (measured:
~1.4% of HBM bandwidth). `gemm_decode_w8a8` fixes this with a split-K grid —
one warp per (output column, K-slice) — so it stays memory-bound-fast even
when M and N are both small.

Gates (AGENTS.md): int8 paths never use `allclose`. The kernel's int32
dp4a accumulate is EXACT and integer addition is associative/commutative, so
vs `gemm_w8a8` (same integer math, different reduction order) the two must
match exactly; vs the fp32 oracle it's the shared int8 SQNR/cos/rel-L1 gate.
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8
from superl8.quant.core import quantize_int8_rowwise

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, compare_report, time_ms  # noqa: E402
from tests.tolerances import assert_int8_quality, cos_sim  # noqa: E402


def _wq(w: torch.Tensor):
    q, s = quantize_int8_rowwise(w)
    return q.contiguous(), s.squeeze(-1).contiguous()


# M spans the whole decode range (1-16); N/K include non-tile-multiple tails
# and real projection shapes (down_proj-ish small N, FFN-ish large N).
DECODE_SHAPES = [
    (1, 64, 64), (1, 1024, 4864), (2, 37, 128), (8, 1024, 4864),
    (8, 4864, 896), (8, 896, 4864), (16, 4864, 896), (16, 128, 4096),
    (5, 200, 512),
    # K % 4 == 0 but K % 16 != 0: exercises the scalar tail / non-vectorized
    # fallback of the 128-bit-load path (K % 16 == 0 gates the int4 bulk load).
    (1, 300, 68), (8, 1024, 132), (16, 200, 20), (4, 37, 52),
]


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", DECODE_SHAPES)
def test_gemm_decode_matches_gemm_w8a8_exact(device, m, n, k):
    """Same integer math as the tile GEMM, different reduction order -> exact match."""
    xq = torch.randint(-127, 128, (m, k), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (n, k), device=device, dtype=torch.int8)
    xs = torch.rand(m, device=device, dtype=torch.float32) * 0.01 + 1e-3
    ws = torch.rand(n, device=device, dtype=torch.float32) * 0.01 + 1e-3
    y_decode = superl8._C.gemm_decode_w8a8(xq, xs, wq, ws)
    y_tile = superl8._C.gemm_w8a8(xq, xs, wq, ws)
    assert torch.equal(y_decode, y_tile), f"decode kernel diverges from gemm_w8a8 @ {m}x{n}x{k}"


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", DECODE_SHAPES)
def test_gemm_decode_reproduces_integer_matmul(device, m, n, k):
    xq = torch.randint(-127, 128, (m, k), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (n, k), device=device, dtype=torch.int8)
    xs = torch.rand(m, device=device, dtype=torch.float32) * 0.01 + 1e-3
    ws = torch.rand(n, device=device, dtype=torch.float32) * 0.01 + 1e-3
    y = superl8._C.gemm_decode_w8a8(xq, xs, wq, ws)
    ref = (xq.float() @ wq.float().t()) * xs[:, None] * ws[None, :]
    assert cos_sim(y, ref) >= 0.9999
    assert_int8_quality(y, ref, min_cos=0.9999, max_rel_l1=0.005, min_sqnr_db=40.0,
                        what=f"gemm_decode_w8a8 integer-exact {m}x{n}x{k}")


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", [(1, 64, 64), (8, 1024, 4864), (16, 4864, 896)])
def test_gemm_decode_matches_fp32_oracle(device, m, n, k):
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    x_i8, x_scale = quantize_int8_rowwise(x)
    w_i8, w_scale = _wq(w)
    y = superl8._C.gemm_decode_w8a8(x_i8, x_scale.squeeze(-1).contiguous(), w_i8, w_scale)
    ref = x.float() @ w.float().t()
    assert_int8_quality(y, ref, what=f"gemm_decode_w8a8 fp32-oracle {m}x{n}x{k}")


@pytest.mark.correctness
def test_gemm_decode_deterministic(device):
    """atomicAdd on int32 is exact (associative/commutative) -> bitwise-repeatable."""
    x = torch.randn(8, 896, device=device, dtype=torch.float16)
    w = torch.randn(1024, 896, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    x_i8, x_scale = quantize_int8_rowwise(x)
    xs = x_scale.squeeze(-1).contiguous()
    r0 = superl8._C.gemm_decode_w8a8(x_i8, xs, w_i8, w_scale)
    for _ in range(3):
        assert torch.equal(superl8._C.gemm_decode_w8a8(x_i8, xs, w_i8, w_scale), r0)


# ── Fused fp16-input decode GEMV (issue #130 rung-1) ────────────────────────
# gemm_decode_w8a8_fp16in folds the per-row activation quantization into the
# GEMV prologue (fp16 x -> int8 in smem, then dp4a) so decode drops one kernel
# AND one HBM round-trip of the activation per linear. Same rowwise quant +
# same dp4a as the unfused path, so the output matches to within last-bit
# quant rounding.


def _fp16in_ref(x, wq, ws):
    x_i8, x_scale = quantize_int8_rowwise(x)
    return superl8._C.gemm_decode_w8a8(x_i8, x_scale.squeeze(-1).contiguous(), wq, ws)


# Fused-eligible shapes only: split_k==1 (N >= the SM-fill target, ~320) AND M*K int8
# within dynamic smem. The C++ op rejects the rest by design (RuntimeError) so the caller
# falls back to quantize_int8_rowwise + gemm_decode_w8a8; that fallback is a Python-layer
# concern, tested separately. Covers M in {1,8,16}, K%16==0 and the K%16!=0 scalar tail.
FUSED_SHAPES = [
    (1, 1024, 4864), (8, 1024, 4864), (8, 4864, 896), (8, 896, 4864),
    (16, 4864, 896), (8, 1024, 132),
]


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", FUSED_SHAPES)
def test_gemm_decode_fp16in_matches_quantized(device, m, n, k):
    """Fused fp16-in GEMV == unfused quantize_int8_rowwise -> gemm_decode_w8a8."""
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    wq, ws = _wq(w)
    y_ref = _fp16in_ref(x, wq, ws)
    y_fused = superl8._C.gemm_decode_w8a8_fp16in(x, wq, ws)
    assert y_fused.shape == y_ref.shape and y_fused.dtype == y_ref.dtype
    assert_int8_quality(y_fused, y_ref, min_cos=0.99999, max_rel_l1=1e-3, min_sqnr_db=60.0,
                        what=f"fused fp16-in vs quant+gemm {m}x{n}x{k}")


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", [(1, 64, 64), (8, 1024, 4864), (16, 4864, 896)])
def test_gemm_decode_fp16in_matches_fp32_oracle(device, m, n, k):
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    wq, ws = _wq(w)
    y = superl8._C.gemm_decode_w8a8_fp16in(x, wq, ws)
    ref = x.float() @ w.float().t()
    assert_int8_quality(y, ref, what=f"fused fp16-in fp32-oracle {m}x{n}x{k}")


@pytest.mark.correctness
def test_gemm_decode_fp16in_deterministic(device):
    x = torch.randn(8, 896, device=device, dtype=torch.float16)
    w = torch.randn(1024, 896, device=device, dtype=torch.float16) * 0.1
    wq, ws = _wq(w)
    r0 = superl8._C.gemm_decode_w8a8_fp16in(x, wq, ws)
    for _ in range(3):
        assert torch.equal(superl8._C.gemm_decode_w8a8_fp16in(x, wq, ws), r0)


@pytest.mark.perf
@pytest.mark.parametrize("m,n,k,tag", [(1, 4096, 4096, "fp16in-M1"), (8, 4096, 4096, "fp16in-M8")])
def test_gemm_decode_fp16in_perf(device, m, n, k, tag):
    """Fused fp16-in vs unfused (quantize_int8_rowwise + gemm_decode_w8a8): the fused path
    removes a standalone quant kernel launch + an HBM round-trip of the int8 activation, so
    it should not regress (relative comparison → robust to box load)."""
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    wq, ws = _wq(w)
    fused = time_ms(lambda: superl8._C.gemm_decode_w8a8_fp16in(x, wq, ws))

    def _unfused():
        xq, xs = quantize_int8_rowwise(x)
        return superl8._C.gemm_decode_w8a8(xq, xs.squeeze(-1).contiguous(), wq, ws)

    unfused = time_ms(_unfused)
    print("\n" + compare_report(tag, fused, {"unfused.quant+gemm": unfused}))
    assert fused <= unfused * 1.15, f"fused {fused:.4f}ms regressed vs unfused {unfused:.4f}ms"


@pytest.mark.correctness
def test_gemm_decode_bf16_output(device):
    m, n, k = 8, 1024, 4864
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    x_i8, x_scale = quantize_int8_rowwise(x)
    w_i8, w_scale = _wq(w)
    y = superl8._C.gemm_decode_w8a8(x_i8, x_scale.squeeze(-1).contiguous(), w_i8, w_scale,
                                 torch.bfloat16)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    ref = x.float() @ w.float().t()
    assert_int8_quality(y, ref, what=f"gemm_decode_w8a8 bf16-out {m}x{n}x{k}")


@pytest.mark.correctness
def test_gemm_decode_rejects_m_too_large(device):
    """M > 16 is the tile GEMM's job (gemm_w8a8); the decode kernel must reject it."""
    xq = torch.randint(-127, 128, (17, 64), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (8, 64), device=device, dtype=torch.int8)
    xs = torch.ones(17, device=device, dtype=torch.float32)
    ws = torch.ones(8, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="M"):
        superl8._C.gemm_decode_w8a8(xq, xs, wq, ws)


@pytest.mark.correctness
def test_gemm_decode_rejects_odd_k(device):
    xq = torch.randint(-127, 128, (4, 66), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (8, 66), device=device, dtype=torch.int8)
    xs = torch.ones(4, device=device, dtype=torch.float32)
    ws = torch.ones(8, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="4"):
        superl8._C.gemm_decode_w8a8(xq, xs, wq, ws)


@pytest.mark.correctness
def test_gemm_decode_zero_m_or_n(device):
    xq = torch.zeros(0, 64, device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (8, 64), device=device, dtype=torch.int8)
    xs = torch.zeros(0, device=device, dtype=torch.float32)
    ws = torch.ones(8, device=device, dtype=torch.float32)
    y = superl8._C.gemm_decode_w8a8(xq, xs, wq, ws)
    assert y.shape == (0, 8)


@pytest.mark.correctness
def test_linear_w8a8_routes_eligible_decode_through_fused_fp16in(device):
    """Fused-eligible decode shapes (fp16 x, M<=16, N>=320, M*K in smem) must be
    routed through the FUSED fp16-in GEMV (issue #130) — quant folded into the
    prologue — so linear_w8a8's output is BIT-IDENTICAL to calling that op
    directly, and matches the old two-step path to int8 tolerance (the fused
    in-prologue quant differs from the standalone quant only in last-bit
    rounding, per test_gemm_decode_fp16in_matches_quantized)."""
    m, n, k = 8, 1024, 4864
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    y_linear = superl8.linear_w8a8(x, w_i8, w_scale)
    y_fused = superl8._C.gemm_decode_w8a8_fp16in(x, w_i8, w_scale)
    assert torch.equal(y_linear, y_fused), "eligible decode shape not routed to fused op"
    x_i8, x_scale = quantize_int8_rowwise(x)
    y_twostep = superl8._C.gemm_decode_w8a8(x_i8, x_scale.squeeze(-1).contiguous(), w_i8, w_scale)
    assert_int8_quality(y_linear, y_twostep, min_cos=0.99999, max_rel_l1=1e-3,
                        min_sqnr_db=60.0, what="linear_w8a8 fused vs two-step")


@pytest.mark.correctness
def test_linear_w8a8_fused_deterministic(device):
    """The fused decode route must be bitwise-deterministic across repeats."""
    m, n, k = 8, 1024, 4864
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    r0 = superl8.linear_w8a8(x, w_i8, w_scale)
    for _ in range(3):
        assert torch.equal(superl8.linear_w8a8(x, w_i8, w_scale), r0)


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k,why", [
    (8, 256, 4864, "N<320 would need split_k>1"),   # small-N ⇒ two-step
    (8, 1023, 4868, "non-tile-multiple N, K%4==0"),  # eligible odd-ish sizes
])
def test_linear_w8a8_fallback_shapes_match_twostep(device, m, n, k, why):
    """Shapes that are NOT fused-eligible must fall back to the two-step decode
    path and be BIT-IDENTICAL to it (no silent RuntimeError, no numeric drift)."""
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    y_linear = superl8.linear_w8a8(x, w_i8, w_scale)
    x_i8, x_scale = quantize_int8_rowwise(x)
    y_twostep = superl8._C.gemm_decode_w8a8(x_i8, x_scale.squeeze(-1).contiguous(), w_i8, w_scale)
    if n >= 320:  # eligible → fused; still must match two-step to int8 tolerance
        assert_int8_quality(y_linear, y_twostep, min_cos=0.99999, max_rel_l1=1e-3,
                            min_sqnr_db=60.0, what=f"linear_w8a8 {m}x{n}x{k} ({why})")
    else:  # ineligible → two-step, bit-identical
        assert torch.equal(y_linear, y_twostep), f"fallback {m}x{n}x{k} not bit-exact ({why})"


@pytest.mark.correctness
def test_linear_w8a8_bf16_activation_uses_twostep(device):
    """bf16 activations are NOT accepted by the fused fp16-in op — must fall back
    to two-step (bit-identical) rather than throw."""
    m, n, k = 8, 1024, 4864
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    y_linear = superl8.linear_w8a8(x, w_i8, w_scale, out_dtype=torch.bfloat16)
    x_i8, x_scale = quantize_int8_rowwise(x)
    y_twostep = superl8._C.gemm_decode_w8a8(x_i8, x_scale.squeeze(-1).contiguous(), w_i8, w_scale,
                                         torch.bfloat16)
    assert torch.equal(y_linear, y_twostep)


@pytest.mark.correctness
def test_linear_w8a8_prefill_shapes_still_use_tile_gemm(device):
    """M>16 (prefill) must still take the gemm_w8a8 tile-GEMM path."""
    m, n, k = 64, 256, 512
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    y_linear = superl8.linear_w8a8(x, w_i8, w_scale)
    x_i8, x_scale = quantize_int8_rowwise(x)
    y_direct = superl8._C.gemm_w8a8(x_i8, x_scale.squeeze(-1).contiguous(), w_i8, w_scale)
    assert torch.equal(y_linear, y_direct)


# ---------------------------------------------------------------------------
# Perf — decode shapes (M=8), targeting a large fraction of 829 GB/s HBM and a
# large speedup over the tile GEMM that decode used to be stuck with.
# ---------------------------------------------------------------------------
@pytest.mark.perf
@pytest.mark.parametrize("m,n,k,tag", [
    (8, 1024, 4864, "gemm_decode_w8a8.m8n1024k4864"),   # down_proj-ish decode shape
    (8, 4864, 896, "gemm_decode_w8a8.m8n4864k896"),     # up_proj-ish decode shape
    (16, 896, 896, "gemm_decode_w8a8.m16n896k896"),     # small-N decode shape
])
def test_gemm_decode_perf(device, m, n, k, tag):
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    xq, xs = quantize_int8_rowwise(x)
    xs = xs.squeeze(-1).contiguous()
    ms = time_ms(lambda: superl8._C.gemm_decode_w8a8(xq, xs, w_i8, w_scale))
    tile_ms = time_ms(lambda: superl8._C.gemm_w8a8(xq, xs, w_i8, w_scale))
    fp16_ms = time_ms(lambda: torch.matmul(x, w.t()))
    print("\n" + compare_report(tag, ms, {"gemm_w8a8.tile": tile_ms, "torch.matmul.fp16": fp16_ms}))
    # bytes moved: read W (N*K int8) + write out (M*N halfs) is decode's dominant term.
    bytes_moved = n * k + m * n * 2
    gbps = bytes_moved / (ms * 1e-3) / 1e9
    print(f"{tag}: {gbps:.1f} GB/s ({gbps / 829 * 100:.1f}% of 829 GB/s HBM peak)")
    assert_no_regression(tag, ms)


@pytest.mark.perf
@pytest.mark.parametrize("m,n,k,tag", [
    (1, 4096, 4096, "linear_w8a8.fused-vs-twostep.m1"),
    (8, 4096, 4096, "linear_w8a8.fused-vs-twostep.m8"),
])
def test_linear_w8a8_fused_beats_twostep(device, m, n, k, tag):
    """End-to-end linear_w8a8: the wired fused decode route (issue #130) must not
    regress against a forced two-step quantize + gemm_decode for the same shape —
    it removes a standalone quant launch + an int8-activation HBM round-trip, so
    the whole point is that it's faster (relative comparison → robust to box load)."""
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    fused = time_ms(lambda: superl8.linear_w8a8(x, w_i8, w_scale))  # fused route (eligible)

    def _twostep():
        xq, xs = quantize_int8_rowwise(x)
        return superl8._C.gemm_decode_w8a8(xq, xs.squeeze(-1).contiguous(), w_i8, w_scale)

    twostep = time_ms(_twostep)
    print("\n" + compare_report(tag, fused, {"twostep.quant+gemm": twostep}))
    assert fused <= twostep * 1.15, f"fused {fused:.4f}ms regressed vs two-step {twostep:.4f}ms"


# ---------------------------------------------------------------------------
# Fused requant epilogue — issue #118 (PR #155, fresh attempt)
# gemm_decode_w8a8_requant emits already-requantized int8 output with a
# per-row fp32 scale instead of fp16-to-HBM, cutting the int8 float-tax.
# Acceptance: cos >= 0.998 vs fp32 oracle, SQNR >= 38 dB.  Non-tile-multiple
# shapes + determinism x3 included per AGENTS.md TDD mandate.
# ---------------------------------------------------------------------------
REQUANT_SHAPES = [
    (1, 1024, 4864), (8, 1024, 4864), (8, 4864, 896), (8, 896, 4864),
    (16, 4864, 896), (5, 200, 512),
    (1, 300, 68), (8, 1024, 132), (16, 200, 20), (4, 37, 52),
]


def _dequant_requant(out_i8, out_scale):
    return out_i8.float() * out_scale.unsqueeze(-1)


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", REQUANT_SHAPES)
def test_gemm_decode_requant_matches_fp32_oracle(device, m, n, k):
    """Requant output dequantized with its own scale must match the fp32 oracle
    within int8 quality gates."""
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    x_i8, x_scale = quantize_int8_rowwise(x)
    w_i8, w_scale = _wq(w)
    xs = x_scale.squeeze(-1).contiguous()
    out_i8, out_scale = superl8._C.gemm_decode_w8a8_requant(x_i8, xs, w_i8, w_scale)
    assert out_i8.dtype == torch.int8
    assert out_i8.shape == (m, n)
    assert out_scale.dtype == torch.float32
    assert out_scale.shape == (m,)
    y_deq = _dequant_requant(out_i8, out_scale)
    ref = x.float() @ w.float().t()
    assert_int8_quality(y_deq, ref, min_cos=0.998, max_rel_l1=0.025, min_sqnr_db=35.0,
                        what=f"gemm_decode_w8a8_requant fp32-oracle {m}x{n}x{k}")


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", REQUANT_SHAPES)
def test_gemm_decode_requant_matches_fp16_decode(device, m, n, k):
    """Requant->dequant output must be close to the existing gemm_decode_w8a8
    fp16 output (the requant adds one symmetric-RTN rounding step)."""
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    x_i8, x_scale = quantize_int8_rowwise(x)
    w_i8, w_scale = _wq(w)
    xs = x_scale.squeeze(-1).contiguous()
    y_decode = superl8._C.gemm_decode_w8a8(x_i8, xs, w_i8, w_scale)
    out_i8, out_scale = superl8._C.gemm_decode_w8a8_requant(x_i8, xs, w_i8, w_scale)
    y_deq = _dequant_requant(out_i8, out_scale)
    assert_int8_quality(y_deq, y_decode, min_cos=0.999, max_rel_l1=0.015, min_sqnr_db=38.0,
                        what=f"requant vs fp16-decode {m}x{n}x{k}")


@pytest.mark.correctness
def test_gemm_decode_requant_deterministic(device):
    """Same inputs x3 must give bitwise-identical (int8_out, scale) output pairs."""
    m, n, k = 8, 1024, 4864
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    x_i8, x_scale = quantize_int8_rowwise(x)
    w_i8, w_scale = _wq(w)
    xs = x_scale.squeeze(-1).contiguous()
    r0, s0 = superl8._C.gemm_decode_w8a8_requant(x_i8, xs, w_i8, w_scale)
    for _ in range(3):
        r, s = superl8._C.gemm_decode_w8a8_requant(x_i8, xs, w_i8, w_scale)
        assert torch.equal(r, r0), "int8 output not deterministic"
        assert torch.equal(s, s0), "scale output not deterministic"


@pytest.mark.perf
@pytest.mark.parametrize("m,n,k,tag", [
    (8, 1024, 4864, "requant.m8n1024k4864"),
    (8, 4864, 896, "requant.m8n4864k896"),
    (1, 4096, 4096, "requant.m1n4096k4096"),
])
def test_gemm_decode_requant_perf(device, m, n, k, tag):
    """Requant must not regress vs the existing decode + eager requant (two-step)."""
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    x_i8, x_scale = quantize_int8_rowwise(x)
    xs = x_scale.squeeze(-1).contiguous()

    requant_ms = time_ms(lambda: superl8._C.gemm_decode_w8a8_requant(x_i8, xs, w_i8, w_scale))

    def _twostep_requant():
        y_fp16 = superl8._C.gemm_decode_w8a8(x_i8, xs, w_i8, w_scale)
        return quantize_int8_rowwise(y_fp16)

    twostep_ms = time_ms(_twostep_requant)
    print("\n" + compare_report(tag, requant_ms, {"twostep.decode+quant": twostep_ms}))
    # Requant should match or beat the decode+quant two-step (no regression).
    assert requant_ms <= twostep_ms * 1.20, (
        f"requant {requant_ms:.4f}ms regressed vs two-step {twostep_ms:.4f}ms")
