# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""int8 dp4a GEMM (W8A8) — the linear-layer kernel that unblocks fni8-serve.

Y[M,N] = (X_i8[M,K] . W_i8[N,K]^T) * x_scale[m] * w_scale[n], int32 accumulate,
fp16 out. This mirrors how the .superl8 `per_row_i8` weight is stored (W [out, in]
row-major, in%4==0, one fp32 scale per output channel) so the kernel consumes it
byte-identically.

Gates (AGENTS.md): int8 paths use SQNR / cos-sim / rel-L1, never allclose. The
kernel's integer matmul is EXACT, so vs the int8-dequant reference it is tight to
fp-rounding; vs the fp32 oracle it carries only quantization error.
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
    """Per-row (per-output-channel) int8 quant of a weight -> (w_i8 [N,K], w_scale [N])."""
    q, s = quantize_int8_rowwise(w)      # q [N,K] int8, s [N,1] fp32
    return q.contiguous(), s.squeeze(-1).contiguous()


# M includes 1 (decode) and non-tile-multiple tails; N/K span real projection dims.
SHAPES = [
    (1, 64, 64), (7, 64, 128), (64, 64, 64), (257, 128, 256),
    (2048, 896, 4096), (16, 4864, 896), (33, 4096, 896), (128, 11008, 4096),
]


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_gemm_w8a8_matches_fp32_oracle(device, m, n, k):
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    y = superl8.linear_w8a8(x, w_i8, w_scale)
    assert y.shape == (m, n) and y.dtype == torch.float16
    ref = x.float() @ w.float().t()      # fp32 full-precision oracle
    assert_int8_quality(y, ref, what=f"gemm_w8a8 {m}x{n}x{k}")


@pytest.mark.correctness
def test_gemm_w8a8_reproduces_integer_matmul(device):
    """The kernel's int32 accumulate must equal the exact integer matmul (the only
    slack is the single fp32 dequant multiply)."""
    m, n, k = 130, 200, 512
    xq = torch.randint(-127, 128, (m, k), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (n, k), device=device, dtype=torch.int8)
    xs = torch.rand(m, device=device, dtype=torch.float32) * 0.01 + 1e-3
    ws = torch.rand(n, device=device, dtype=torch.float32) * 0.01 + 1e-3
    y = superl8._C.gemm_w8a8(xq, xs, wq, ws)
    ref = (xq.float() @ wq.float().t()) * xs[:, None] * ws[None, :]
    # Exact integer part; only fp16 store rounding differs -> very high SQNR.
    assert cos_sim(y, ref) >= 0.9999
    assert_int8_quality(y, ref, min_cos=0.9999, max_rel_l1=0.005, min_sqnr_db=40.0,
                        what="gemm_w8a8 integer-exact")


@pytest.mark.correctness
def test_gemm_w8a8_deterministic(device):
    x = torch.randn(64, 512, device=device, dtype=torch.float16)
    w = torch.randn(256, 512, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    r0 = superl8.linear_w8a8(x, w_i8, w_scale)
    for _ in range(3):
        assert torch.equal(superl8.linear_w8a8(x, w_i8, w_scale), r0)


@pytest.mark.correctness
def test_gemm_w8a8_bias(device):
    x = torch.randn(8, 128, device=device, dtype=torch.float16)
    w = torch.randn(64, 128, device=device, dtype=torch.float16) * 0.1
    b = torch.randn(64, device=device, dtype=torch.float16)
    w_i8, w_scale = _wq(w)
    y = superl8.linear_w8a8(x, w_i8, w_scale, bias=b)
    y0 = superl8.linear_w8a8(x, w_i8, w_scale)
    torch.testing.assert_close(y, (y0.float() + b.float()).half(), rtol=1e-3, atol=1e-3)


@pytest.mark.correctness
def test_gemm_w8a8_leading_dims(device):
    """Wrapper flattens leading dims: [B, T, K] -> [B, T, N]."""
    x = torch.randn(2, 5, 128, device=device, dtype=torch.float16)
    w = torch.randn(64, 128, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    y = superl8.linear_w8a8(x, w_i8, w_scale)
    assert y.shape == (2, 5, 64)
    ref = superl8.linear_w8a8(x.reshape(10, 128), w_i8, w_scale).reshape(2, 5, 64)
    assert torch.equal(y, ref)


@pytest.mark.correctness
def test_gemm_w8a8_rejects_odd_k(device):
    xq = torch.randint(-127, 128, (4, 66), device=device, dtype=torch.int8)  # K=66, %4!=0
    wq = torch.randint(-127, 128, (8, 66), device=device, dtype=torch.int8)
    xs = torch.ones(4, device=device, dtype=torch.float32)
    ws = torch.ones(8, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="4"):
        superl8._C.gemm_w8a8(xq, xs, wq, ws)


@pytest.mark.correctness
def test_gemm_w8a8_rejects_bad_dtype(device):
    xq = torch.randint(-127, 128, (4, 64), device=device, dtype=torch.int8)
    wf = torch.randn(8, 64, device=device, dtype=torch.float16)  # weight not int8
    xs = torch.ones(4, device=device, dtype=torch.float32)
    ws = torch.ones(8, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="int8"):
        superl8._C.gemm_w8a8(xq, xs, wf, ws)


@pytest.mark.correctness
def test_gemm_w8a8_shape_mismatch(device):
    xq = torch.randint(-127, 128, (4, 64), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (8, 128), device=device, dtype=torch.int8)  # K mismatch
    xs = torch.ones(4, device=device, dtype=torch.float32)
    ws = torch.ones(8, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        superl8._C.gemm_w8a8(xq, xs, wq, ws)


# ---------------------------------------------------------------------------
# bf16 output — the black-image fix (issue #11). bf16-native models (Gemma,
# most diffusion DiTs) carry activations that overflow fp16 (max 65504) ->
# inf/NaN -> black output. int32 dp4a accumulate -> fp32 dequant is unaffected;
# only the final store dtype matters, so this is opt-in via `out_dtype`.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", [(1, 64, 64), (7, 64, 128), (257, 128, 256)])
def test_gemm_w8a8_bf16_output(device, m, n, k):
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    y = superl8.linear_w8a8(x, w_i8, w_scale, out_dtype=torch.bfloat16)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    ref = x.float() @ w.float().t()
    assert_int8_quality(y, ref, what=f"gemm_w8a8 bf16-out {m}x{n}x{k}")


@pytest.mark.correctness
def test_gemm_w8a8_bf16_output_avoids_fp16_overflow(device):
    """Fixed large-magnitude fixture that DOES overflow fp16 (max 65504); bf16
    (max ~3.4e38) must stay finite and numerically match the integer matmul."""
    m, n, k = 4, 4, 64
    xq = torch.full((m, k), 100, device=device, dtype=torch.int8)
    wq = torch.full((n, k), 100, device=device, dtype=torch.int8)
    xs = torch.ones(m, device=device, dtype=torch.float32)
    ws = torch.ones(n, device=device, dtype=torch.float32)
    y_fp16 = superl8._C.gemm_w8a8(xq, xs, wq, ws)
    y_bf16 = superl8._C.gemm_w8a8(xq, xs, wq, ws, torch.bfloat16)
    assert not torch.isfinite(y_fp16.float()).all(), "fixture must overflow fp16"
    assert torch.isfinite(y_bf16.float()).all()
    ref = (xq.float() @ wq.float().t()) * xs[:, None] * ws[None, :]
    assert_int8_quality(y_bf16, ref, min_cos=0.9999, max_rel_l1=0.005, min_sqnr_db=40.0,
                        what="gemm_w8a8 bf16 overflow-avoidance")


@pytest.mark.correctness
def test_gemm_w8a8_bf16_output_deterministic(device):
    x = torch.randn(64, 512, device=device, dtype=torch.float16)
    w = torch.randn(256, 512, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    r0 = superl8.linear_w8a8(x, w_i8, w_scale, out_dtype=torch.bfloat16)
    for _ in range(3):
        assert torch.equal(superl8.linear_w8a8(x, w_i8, w_scale, out_dtype=torch.bfloat16), r0)


@pytest.mark.correctness
def test_gemm_w8a8_rejects_bad_out_dtype(device):
    xq = torch.randint(-127, 128, (4, 64), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (8, 64), device=device, dtype=torch.int8)
    xs = torch.ones(4, device=device, dtype=torch.float32)
    ws = torch.ones(8, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="float16|bfloat16"):
        superl8._C.gemm_w8a8(xq, xs, wq, ws, torch.float32)


# ---------------------------------------------------------------------------
# smem XOR bank-swizzle (perf/gemm-dp4a-smem-swizzle): the dp4a inner loop reads
# smem columns, which alias 32 banks up to 8-way with the plain [row][BK4]
# layout. gemm_swz_col permutes the physical column by (row>>2) -> conflict-free.
# It is a per-row column bijection, so numerics MUST be byte-identical to before.
# These shapes stress the swizzle's ragged store/load paths: M/N/K that are NOT
# tile multiples, and K spanning many BK(=64)-blocks so the store swizzle repeats
# across k-blocks. Guards against a swizzle-indexing regression.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", [
    (191, 133, 320),      # all three ragged inside the tile
    (65, 65, 64),         # one row/col past a tile edge
    (2048, 3072, 3072),   # DiT square (Flux-class self-attn proj)
    (300, 256, 1600),     # 25 BK-blocks: store swizzle repeats every k-block
])
def test_gemm_w8a8_swizzle_matches_integer_matmul(device, m, n, k):
    """Swizzle is layout-only: the int32 accumulate must still equal the exact
    integer matmul (only the fp store rounds)."""
    xq = torch.randint(-127, 128, (m, k), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (n, k), device=device, dtype=torch.int8)
    xs = torch.rand(m, device=device, dtype=torch.float32) * 0.01 + 1e-3
    ws = torch.rand(n, device=device, dtype=torch.float32) * 0.01 + 1e-3
    y = superl8._C.gemm_w8a8(xq, xs, wq, ws)
    ref = (xq.float() @ wq.float().t()) * xs[:, None] * ws[None, :]
    assert_int8_quality(y, ref, min_cos=0.9999, max_rel_l1=0.005, min_sqnr_db=40.0,
                        what=f"gemm_w8a8 swizzle {m}x{n}x{k}")


@pytest.mark.perf
@pytest.mark.parametrize("m,n,k,tag", [
    # DiT denoising-step linear shapes (Flux-class H=3072). These are large-M,
    # compute-region GEMMs — the bank-conflict lever this branch targets. No
    # committed baseline yet (soft-skips), but compare_report prints the honest
    # dp4a TOP/s so the rung-1 delta is recorded in the PR.
    (2048, 3072, 3072, "gemm_w8a8.m2048n3072k3072"),   # attn qkv/o proj
    (2048, 12288, 3072, "gemm_w8a8.m2048n12288k3072"), # mlp fc1
])
def test_gemm_w8a8_perf_dit(device, m, n, k, tag):
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    xq, xs = quantize_int8_rowwise(x)
    xs = xs.squeeze(-1).contiguous()
    ms = time_ms(lambda: superl8._C.gemm_w8a8(xq, xs, w_i8, w_scale, torch.bfloat16))
    tops = 2.0 * m * n * k / (ms * 1e-3) / 1e12
    fp16_ms = time_ms(lambda: torch.matmul(x, w.t()))
    print("\n" + compare_report(tag, ms, {"torch.matmul.fp16": fp16_ms})
          + f" | {tops:.1f} int8-TOP/s")
    assert_no_regression(tag, ms)


@pytest.mark.perf
@pytest.mark.parametrize("m,n,k,tag", [
    (8, 4864, 896, "gemm_w8a8.m8n4864k896"),        # decode FFN up-proj (Qwen2-ish)
    (2048, 4864, 896, "gemm_w8a8.m2048n4864k896"),  # prefill FFN up-proj
])
def test_gemm_w8a8_perf(device, m, n, k, tag):
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    xq, xs = quantize_int8_rowwise(x)
    xs = xs.squeeze(-1).contiguous()
    ms = time_ms(lambda: superl8._C.gemm_w8a8(xq, xs, w_i8, w_scale))
    # First-class honest report: dp4a vs a torch fp16 matmul on THIS fleet.
    fp16_ms = time_ms(lambda: torch.matmul(x, w.t()))
    print("\n" + compare_report(tag, ms, {"torch.matmul.fp16": fp16_ms}))
    assert_no_regression(tag, ms)


@pytest.mark.perf
def test_gemm_w8a8_bf16_output_perf(device):
    """bf16 store vs fp16 store: same int32-accumulate/fp32-dequant path, only
    the epilogue write differs. No baseline yet (soft-skips until recorded)."""
    m, n, k = 2048, 4864, 896
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    xq, xs = quantize_int8_rowwise(x)
    xs = xs.squeeze(-1).contiguous()
    ms = time_ms(lambda: superl8._C.gemm_w8a8(xq, xs, w_i8, w_scale, torch.bfloat16))
    fp16_ms = time_ms(lambda: superl8._C.gemm_w8a8(xq, xs, w_i8, w_scale))
    tag = "gemm_w8a8.m2048n4864k896.bf16out"
    print("\n" + compare_report(tag, ms, {"gemm_w8a8.fp16out": fp16_ms}))
    assert_no_regression(tag, ms)


# ---------------------------------------------------------------------------
# PR-G2 — W4A8: 4-bit weights (storage-only int4), unpacked to int8 for dp4a.
# ---------------------------------------------------------------------------
from superl8.quant.lowbit import dequantize_lowbit, quantize_lowbit  # noqa: E402


def _w4(w: torch.Tensor, g: int):
    """Uniform int4 (signed [-7,7]) per-group weight -> (packed uint8 [N,K//2],
    scale fp32 [N,K//g], dequant-fp32 reference [N,K]). Matches .superl8 per_group_i4:
    even col -> low nibble, odd col -> high nibble."""
    codes, scale = quantize_lowbit(w, 4, dim=-1, group_size=g)   # [N,K] int8, [N,K//g]
    c = codes.to(torch.int64)
    lo = (c[:, 0::2] & 0xF).to(torch.int32)
    hi = (c[:, 1::2] & 0xF).to(torch.int32)
    packed = (lo | (hi << 4)).to(torch.uint8).contiguous()
    deq = dequantize_lowbit(codes, scale, group_size=g)          # fp32 [N,K]
    return packed, scale.contiguous(), deq


W4_SHAPES = [
    (1, 64, 64, 32), (7, 128, 128, 32), (64, 64, 128, 64), (257, 256, 256, 128),
    (16, 4864, 896, 32), (2048, 896, 4096, 128), (33, 512, 896, 32),
]


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k,g", W4_SHAPES)
def test_gemm_w4a8_matches_fp32_oracle(device, m, n, k, g):
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    packed, scale, _ = _w4(w, g)
    y = superl8.linear_w4a8(x, packed, scale, group_size=g)
    assert y.shape == (m, n) and y.dtype == torch.float16
    ref = x.float() @ w.float().t()
    # W4 gets its OWN documented bar (not a weakened shared gate). These are the
    # MEASURED int4-on-random-Gaussian numbers (cos 0.993-0.997, SQNR 18.5-22 dB,
    # rel-L1 up to ~0.12 at coarse groups) — the intrinsic error of uniform 4-bit
    # weights with NO structure and NO baked Hadamard, i.e. the worst case. cos +
    # SQNR are the meaningful gates; rel-L1 is loose here because the reference is
    # near-zero-mean noise. Kernel FIDELITY (vs what int4 can represent) is proven
    # separately and tightly by test_gemm_w4a8_reproduces_grouped_dequant (34 dB).
    assert_int8_quality(y, ref, min_cos=0.99, max_rel_l1=0.13, min_sqnr_db=17.0,
                        what=f"gemm_w4a8 {m}x{n}x{k} g{g}")


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k,g", [(7, 128, 128, 32), (64, 64, 128, 64)])
def test_gemm_w4a8_bf16_output(device, m, n, k, g):
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    packed, scale, _ = _w4(w, g)
    y = superl8.linear_w4a8(x, packed, scale, group_size=g, out_dtype=torch.bfloat16)
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    ref = x.float() @ w.float().t()
    assert_int8_quality(y, ref, min_cos=0.99, max_rel_l1=0.13, min_sqnr_db=17.0,
                        what=f"gemm_w4a8 bf16-out {m}x{n}x{k} g{g}")


@pytest.mark.correctness
@pytest.mark.parametrize("g", [32, 64, 128])
def test_gemm_w4a8_reproduces_grouped_dequant(device, g):
    """Kernel must equal the exact grouped int-matmul: sum_g w_scale[n,g] *
    (x_i8 . codes_group^T), times x_scale[m]."""
    m, n, k = 40, 96, 256
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    packed, scale, deq = _w4(w, g)
    x_i8, x_scale = quantize_int8_rowwise(x)
    ref = (x_i8.float() @ deq.t()) * x_scale       # [M,N], x_scale [M,1]
    y = superl8._C.gemm_w4a8(x_i8, x_scale.squeeze(-1).contiguous(), packed, scale, g)
    assert_int8_quality(y, ref, min_cos=0.9999, max_rel_l1=0.01, min_sqnr_db=34.0,
                        what=f"gemm_w4a8 grouped-exact g{g}")


@pytest.mark.correctness
def test_gemm_w4a8_deterministic(device):
    x = torch.randn(32, 512, device=device, dtype=torch.float16)
    w = torch.randn(128, 512, device=device, dtype=torch.float16) * 0.1
    packed, scale, _ = _w4(w, 64)
    r0 = superl8.linear_w4a8(x, packed, scale, group_size=64)
    for _ in range(3):
        assert torch.equal(superl8.linear_w4a8(x, packed, scale, group_size=64), r0)


@pytest.mark.correctness
def test_gemm_w4a8_rejects_group_not_mult_32(device):
    x_i8 = torch.randint(-127, 128, (4, 128), device=device, dtype=torch.int8)
    xs = torch.ones(4, device=device, dtype=torch.float32)
    packed = torch.zeros(8, 64, device=device, dtype=torch.uint8)
    scale = torch.ones(8, 8, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="32"):
        superl8._C.gemm_w4a8(x_i8, xs, packed, scale, 16)


# ── W4A8 DECODE kernel (gemm_decode_w4a8, issue #173) ────────────────────────
# Decode shapes (M<=16). Same int math as the tile gemm_w4a8, so the two must
# agree tightly; the decode kernel is the fast warp-per-column path.
W4_DECODE_SHAPES = [
    (1, 64, 128, 64), (1, 3072, 4096, 128), (1, 4864, 896, 32),
    (4, 320, 3072, 128), (8, 512, 896, 32), (16, 4096, 4096, 128),
    (1, 257, 256, 128),  # non-tile-multiple N
]


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k,g", W4_DECODE_SHAPES)
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_gemm_decode_w4a8_matches_tile(device, m, n, k, g, dt):
    """gemm_decode_w4a8 must equal the tile gemm_w4a8 (same grouped int math)."""
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    packed, scale, _ = _w4(w, g)
    x_i8, x_scale = quantize_int8_rowwise(x)
    xs1 = x_scale.squeeze(-1).contiguous()
    ref = superl8._C.gemm_w4a8(x_i8, xs1, packed, scale, g, dt)          # tile
    y = superl8._C.gemm_decode_w4a8(x_i8, xs1, packed, scale, g, dt)     # decode
    assert y.shape == (m, n) and y.dtype == dt
    # identical int32 group sums; only fp32 accumulation order differs -> ~bit-equal.
    assert_int8_quality(y, ref.float(), min_cos=0.9999, max_rel_l1=1e-3, min_sqnr_db=40.0,
                        what=f"gemm_decode_w4a8 vs tile {m}x{n}x{k} g{g} {dt}")


@pytest.mark.correctness
def test_gemm_decode_w4a8_deterministic(device):
    x = torch.randn(1, 4096, device=device, dtype=torch.float16)
    w = torch.randn(3072, 4096, device=device, dtype=torch.float16) * 0.1
    packed, scale, _ = _w4(w, 128)
    x_i8, x_scale = quantize_int8_rowwise(x)
    xs = x_scale.squeeze(-1).contiguous()
    r0 = superl8._C.gemm_decode_w4a8(x_i8, xs, packed, scale, 128)
    for _ in range(3):
        assert torch.equal(superl8._C.gemm_decode_w4a8(x_i8, xs, packed, scale, 128), r0)


@pytest.mark.correctness
def test_linear_w4a8_routes_decode_at_small_m(device):
    """linear_w4a8 must use the decode kernel for M<=16 and match the tile path."""
    x = torch.randn(1, 4096, device=device, dtype=torch.float16)
    w = torch.randn(3072, 4096, device=device, dtype=torch.float16) * 0.1
    packed, scale, _ = _w4(w, 128)
    called = {}
    orig = superl8._C.gemm_decode_w4a8

    def spy(*a, **k):
        called["decode"] = True
        return orig(*a, **k)

    superl8._C.gemm_decode_w4a8 = spy
    try:
        y = superl8.linear_w4a8(x, packed, scale, group_size=128)
    finally:
        superl8._C.gemm_decode_w4a8 = orig
    assert called.get("decode"), "linear_w4a8 did not route M=1 to the decode kernel"
    x_i8, x_scale = quantize_int8_rowwise(x)
    ref = superl8._C.gemm_w4a8(x_i8, x_scale.squeeze(-1).contiguous(), packed, scale, 128)
    assert_int8_quality(y, ref.float(), min_cos=0.9999, max_rel_l1=1e-3, min_sqnr_db=40.0,
                        what="linear_w4a8 decode-route")


@pytest.mark.perf
def test_gemm_decode_w4a8_perf(device):
    from bench.harness import assert_no_regression, time_ms

    x = torch.randn(1, 896, device=device, dtype=torch.float16)
    w = torch.randn(4864, 896, device=device, dtype=torch.float16) * 0.1
    packed, scale, _ = _w4(w, 32)
    x_i8, x_scale = quantize_int8_rowwise(x)
    xs = x_scale.squeeze(-1).contiguous()
    ms = time_ms(lambda: superl8._C.gemm_decode_w4a8(x_i8, xs, packed, scale, 32))
    assert_no_regression("gemm_decode_w4a8.m1n4864k896", ms)


@pytest.mark.correctness
def test_gemm_w4a8_nf4_rejected(device):
    """NF4 values are non-integer -> cannot enter dp4a; the QTensor path must reject."""
    from superl8.format import QTensor
    packed = torch.zeros(8, 64, dtype=torch.uint8)
    scale = torch.ones(8, 4, dtype=torch.float32)
    qt = QTensor(packed, scale, scheme="per_group_i4", group_size=32, codebook="nf4")
    x = torch.randn(4, 128, device=device, dtype=torch.float16)
    with pytest.raises((ValueError, RuntimeError), match="nf4|int4|codebook"):
        superl8.linear(x, qt)


@pytest.mark.correctness
def test_linear_dispatches_qtensor(device):
    """superl8.linear(x, qt) routes per_row_i8 -> w8a8 and per_group_i4/int4 -> w4a8."""
    from superl8.format import QTensor
    x = torch.randn(6, 128, device=device, dtype=torch.float16)
    # per_row_i8
    w = torch.randn(64, 128, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq(w)
    qt8 = QTensor(w_i8, w_scale, scheme="per_row_i8")
    y8 = superl8.linear(x, qt8)
    assert torch.equal(y8, superl8.linear_w8a8(x, w_i8, w_scale))
    # per_group_i4
    packed, scale, _ = _w4(w, 32)
    qt4 = QTensor(packed, scale, scheme="per_group_i4", group_size=32, codebook="int4")
    y4 = superl8.linear(x, qt4)
    assert torch.equal(y4, superl8.linear_w4a8(x, packed, scale, group_size=32))


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k,g", [
    (8, 4864, 896, 128),               # the perf-gate shape (M=8 decode)
    (7, 133, 128, 32),                 # non-tile-multiple M/N
    (33, 4096, 896, 32),               # M not % BM
    (65, 4864, 896, 64),               # M just past one tile
    (191, 131, 320, 32),               # all three ragged
])
def test_gemm_w4a8_edge_correctness(device, m, n, k, g):
    """W4A8 correctness at the perf-gate shape and non-tile-multiple M/N/K.
    Kernel must match the fp32 oracle within int4 quantization bars.
    """
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    packed, scale, _ = _w4(w, g)
    y = superl8.linear_w4a8(x, packed, scale, group_size=g)
    assert y.shape == (m, n) and y.dtype == torch.float16
    ref = x.float() @ w.float().t()
    assert_int8_quality(y, ref, min_cos=0.99, max_rel_l1=0.13, min_sqnr_db=17.0,
                        what=f"gemm_w4a8 edge {m}x{n}x{k} g{g}")


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k,g", [
    (8, 4864, 896, 128),
    (7, 133, 128, 32),
    (33, 4096, 896, 32),
])
def test_gemm_w4a8_edge_deterministic(device, m, n, k, g):
    """Determinism x3 at non-tile-multiple shapes."""
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    packed, scale, _ = _w4(w, g)
    r0 = superl8.linear_w4a8(x, packed, scale, group_size=g)
    for _ in range(3):
        assert torch.equal(superl8.linear_w4a8(x, packed, scale, group_size=g), r0)


@pytest.mark.perf
def test_gemm_w4a8_perf(device):
    m, n, k, g = 8, 4864, 896, 128
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, device=device, dtype=torch.float16) * 0.1
    packed, scale, _ = _w4(w, g)
    x_i8, x_scale = quantize_int8_rowwise(x)
    x_scale = x_scale.squeeze(-1).contiguous()
    ms = time_ms(lambda: superl8._C.gemm_w4a8(x_i8, x_scale, packed, scale, g))
    fp16_ms = time_ms(lambda: torch.matmul(x, w.t()))
    print("\n" + compare_report("gemm_w4a8.m8n4864k896", ms, {"torch.matmul.fp16": fp16_ms}))
    assert_no_regression("gemm_w4a8.m8n4864k896", ms)
