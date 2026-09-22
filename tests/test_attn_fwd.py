# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR3: int8 dp4a QK^T forward (fp16 PV) — tests written FIRST.

Contract:
  superl8.attn_int8_fwd(q, k, v, causal=False, scale=None) -> out
    q,k,v fp16 [B,H,M,D], D in {64,128}; quantization (per-row int8 Q/K +
    K-smoothing) happens inside the op; softmax/LSE fp32; PV fp16.
Gates: int8 quality (SQNR/cos/rel-L1) vs the fp32 oracle — NEVER allclose;
determinism; finite; non-tile-multiple shapes; perf recorded vs fp16 ladder.
"""

import sys
from pathlib import Path

import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle, sdpa_fp16
from tests.tolerances import assert_finite, assert_int8_quality

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, compare_report, time_ms  # noqa: E402

# (B, H, M, D) — MVP head dims {64, 128}; seqlens include non-tile-multiples.
SHAPES = [
    (1, 2, 128, 64),
    (2, 4, 512, 64),
    (2, 4, 257, 64),  # ragged tail
    (1, 2, 333, 128),  # ragged tail, D=128
    (1, 8, 2048, 64),
    (1, 8, 2048, 128),
]


def make_qkv(shape, device):
    b, h, m, d = shape
    return (torch.randn(b, h, m, d, device=device, dtype=torch.float16) for _ in range(3))


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_attn_int8_fwd_quality(device, shape, causal):
    q, k, v = make_qkv(shape, device)
    out = superl8.attn_int8_fwd(q, k, v, causal=causal)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    assert out.shape == oracle.shape and out.dtype == torch.float16
    assert_int8_quality(out, oracle, what=f"attn_int8_fwd {shape} causal={causal}")


@pytest.mark.correctness
def test_attn_int8_fwd_outlier_k(device):
    """Channel outliers in K are THE int8 failure mode — K-smoothing must hold the gate."""
    q, k, v = make_qkv((2, 4, 512, 64), device)
    k = k.clone()
    k[..., 13] += 8.0
    out = superl8.attn_int8_fwd(q, k, v)
    oracle = attention_fp32_oracle(q, k, v)
    assert_int8_quality(out, oracle, what="attn_int8_fwd outlier-K")


@pytest.mark.correctness
def test_attn_int8_fwd_external_quality_gate_bypasses_internal_detector(device, monkeypatch):
    """A caller with its own output-SQNR gate can request the actual DP4A result.

    The default remains conservative, but ``internal_accuracy_gate=False`` must not
    run the global any-row detector: on long video attention that detector becomes
    sequence-length-dependent and can demote an otherwise accurate call because of
    one statistically rare row. The caller compares this output to the fp reference
    and owns demotion.
    """
    import superl8.quant

    q, k, v = make_qkv((1, 4, 257, 128), device)

    def detector_must_not_run(_q):
        raise AssertionError("internal detector ran despite external quality gate")

    monkeypatch.setattr(superl8.quant, "detect_q_outlier_domination", detector_must_not_run)
    out = superl8.attn_int8_fwd(q, k, v, rotate=True, internal_accuracy_gate=False)
    oracle = attention_fp32_oracle(q, k, v)
    assert_int8_quality(out, oracle, what="external-gated attn_int8_fwd")


@pytest.mark.correctness
def test_attn_int8_fwd_peaked_softmax(device):
    """Adversarial routing case (turboquant 'MSE is a broken proxy' warning):
    a near-tie between two dominant keys is where int8 can bucket-flip which key
    wins. Gate the OUTPUT directly, not just intermediate SQNR."""
    b, h, m, d = 1, 4, 256, 64
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, m, d, device=device, dtype=torch.float16) * 0.1
    v = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    # Two near-tie-dominant keys -> sharp, tie-sensitive softmax where an int8
    # rounding error could flip which key wins.
    k[:, :, 0] += 4.0
    k[:, :, 1] += 3.98
    out = superl8.attn_int8_fwd(q, k, v)
    oracle = attention_fp32_oracle(q, k, v)
    assert_int8_quality(out, oracle, what="peaked/near-tie softmax")


@pytest.mark.correctness
def test_attn_int8_fwd_deterministic(device):
    q, k, v = make_qkv((2, 4, 512, 64), device)
    r0 = superl8.attn_int8_fwd(q, k, v)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_fwd(q, k, v), r0)


@pytest.mark.correctness
def test_attn_int8_fwd_rejects_bad_headdim(device):
    q, k, v = make_qkv((1, 2, 128, 96), device)  # D=96 unsupported in MVP
    with pytest.raises(RuntimeError):
        superl8.attn_int8_fwd(q, k, v)


# ---------------------------------------------------------------------------
# half2 epilogue (PR7): the kernel's output cast uses packed __half2 stores
# (__float22half2_rn + st.v2.f16) on the healthy CUDA-core half2 pipe when the
# output dtype is fp16. This is a transparent optimization — the rounding is
# IEEE 754 RN, same as scalar __float2half; output must be bit-identical to
# the fp32 oracle within the usual int8 quality bars. The D=256 variant uses
# a separate dynamic-smem kernel with its own epilogue, so both are tested.
# ---------------------------------------------------------------------------
_HALF2_SHAPES = [
    (1, 2, 128, 64),
    (2, 4, 257, 64),
    (1, 2, 333, 128),
    (1, 2, 256, 256),  # D=256 dynamic-smem kernel
    (2, 4, 257, 256),  # ragged tail + dynamic smem
]


@pytest.mark.correctness
@pytest.mark.parametrize("shape", _HALF2_SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_attn_int8_fwd_half2_epilogue(device, shape, causal):
    """FP16 V/output path: the half2-packed output-store optimization
    (pair_store<__half> → __float22half2_rn) must match the fp32 oracle within
    the standard int8 quality bars. Covers D=256 (dynamic smem kernel) whose
    epilogue is a separate code path."""
    q, k, v = make_qkv(shape, device)
    out = superl8.attn_int8_fwd(q, k, v, causal=causal)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    assert_int8_quality(out, oracle, what=f"half2-epilogue-fp16 {shape} causal={causal}")


@pytest.mark.correctness
def test_attn_int8_fwd_half2_epilogue_deterministic(device):
    """Determinism: the half2 store path must produce bitwise-identical output
    across repeated calls (same input). Guards against any latent alignment or
    race from pairing fp32→half conversions."""
    q, k, v = make_qkv((2, 4, 512, 64), device)
    r0 = superl8.attn_int8_fwd(q, k, v)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_fwd(q, k, v), r0)


@pytest.mark.correctness
def test_attn_int8_fwd_half2_epilogue_d256_deterministic(device):
    """Determinism on D=256 (dynamic smem kernel) — the half2 store path in
    the dynamic-smem epilogue is a separate code instantiation."""
    q, k, v = make_qkv((1, 2, 256, 256), device)
    r0 = superl8.attn_int8_fwd(q, k, v)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_fwd(q, k, v), r0)


# ---------------------------------------------------------------------------
# bf16 V / output — the black-image fix (issue #11). bf16-native DiT/Gemma
# activations overflow fp16 (max 65504); today's kernel hardcodes __half for
# V-load and the epilogue store, forcing callers to downcast V to fp16 BEFORE
# the op runs (where the overflow actually happens). Output dtype follows V's
# dtype (matches the existing `out = at::empty(..., v.options())` contract).
# ---------------------------------------------------------------------------
@pytest.mark.correctness
@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (2, 4, 257, 64), (1, 2, 333, 128)])
@pytest.mark.parametrize("causal", [False, True])
def test_attn_int8_fwd_bf16_v_and_output(device, shape, causal):
    b, h, m, d = shape
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h, m, d, device=device, dtype=torch.bfloat16)
    out = superl8.attn_int8_fwd(q, k, v, causal=causal)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    assert out.shape == oracle.shape and out.dtype == torch.bfloat16
    assert_int8_quality(out, oracle, what=f"attn_int8_fwd bf16-V {shape} causal={causal}")


@pytest.mark.correctness
def test_attn_int8_fwd_bf16_avoids_fp16_overflow(device):
    """bf16-native V can carry magnitudes that overflow fp16 (max 65504); with
    V/out in bf16 (max ~3.4e38) the output must stay finite."""
    b, h, m, d = 1, 2, 128, 64
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, m, d, device=device, dtype=torch.float16) * 0.1
    v_bf16 = (torch.randn(b, h, m, d, device=device) * 1e5).to(torch.bfloat16)
    assert not torch.isfinite(v_bf16.float().half()).all(), "fixture must overflow fp16"
    out = superl8.attn_int8_fwd(q, k, v_bf16, causal=False)
    assert torch.isfinite(out.float()).all()
    oracle = attention_fp32_oracle(q, k, v_bf16, causal=False)
    assert_int8_quality(out, oracle, what="attn_int8_fwd bf16 overflow-avoidance")


@pytest.mark.correctness
def test_attn_int8_fwd_bf16_deterministic(device):
    q, k, v = make_qkv((2, 4, 512, 64), device)
    v = v.to(torch.bfloat16)
    r0 = superl8.attn_int8_fwd(q, k, v)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_fwd(q, k, v), r0)


@pytest.mark.correctness
def test_attn_int8_fwd_rejects_bad_v_dtype(device):
    q, k, v = make_qkv((1, 2, 128, 64), device)
    with pytest.raises(RuntimeError, match="float16|bfloat16"):
        superl8.attn_int8_fwd(q, k, v.float())


@pytest.mark.perf
@pytest.mark.parametrize("shape", [(2, 16, 2048, 64), (2, 16, 2048, 128)])
def test_attn_int8_fwd_perf(device, shape):
    """Regression gate vs own baseline + the HONEST dp4a-vs-fp16 report."""
    b, h, m, d = shape
    q, k, v = make_qkv(shape, device)
    ours = time_ms(lambda: superl8.attn_int8_fwd(q, k, v))
    sdpa = time_ms(lambda: sdpa_fp16(q, k, v))
    name = f"attn_int8_fwd.b{b}h{h}m{m}d{d}"
    print("\n" + compare_report(name, ours, {"sdpa_fp16": sdpa}))
    assert_no_regression(name, ours)


@pytest.mark.perf
def test_attn_int8_fwd_bf16_perf(device):
    """bf16 V/out vs fp16 V/out: same int8 dp4a QK + fp32-softmax path, only
    V-load/epilogue-store dtype differs. No baseline yet (soft-skips until recorded)."""
    shape = (2, 16, 2048, 64)
    b, h, m, d = shape
    q, k, v = make_qkv(shape, device)
    v_bf16 = v.to(torch.bfloat16)
    ours = time_ms(lambda: superl8.attn_int8_fwd(q, k, v_bf16))
    fp16 = time_ms(lambda: superl8.attn_int8_fwd(q, k, v))
    name = f"attn_int8_fwd.b{b}h{h}m{m}d{d}.bf16v"
    print("\n" + compare_report(name, ours, {"attn_int8_fwd.fp16v": fp16}))
    assert_no_regression(name, ours)


# ===========================================================================
# attn_fp16_fwd — tiled fp16/bf16 FlashAttention-2 PREFILL kernel (sm_70).
#
# The fp16 sibling of attn_int8_fwd: SAME tiling (BLOCK_M=BLOCK_N=64, lane-pair
# per row, online exp2 softmax, __shfl_xor pair reductions, causal/GQA,
# dynamic-smem D=256), but the QK^T inner loop is a half2 __hfma2 CUDA-core dot
# (the healthy Volta pipe — NEVER wmma/HMMA tensor cores) instead of int8 dp4a,
# with fp32-accumulated softmax/LSE. No quant prologue, no scales: inputs are
# fp16/bf16 and out follows v's dtype.
#
# Gate = RELATIVE-to-fp32 (assert_relative_to_fp32, <= 2x fp16-baseline err +
# 1e-5), NOT the int8 gate — this is a full-precision QK path.
#
# The point: O(N) memory. torch SDPA has no flash/mem-efficient backend on
# Volta, so it materializes the O(N^2) score matrix and OOMs on the ~10k-17k
# token video-DiT attention (LTX). This kernel streams the K/V tiles.
# ===========================================================================
import torch.nn.functional as F  # noqa: E402

from tests.tolerances import assert_relative_to_fp32  # noqa: E402

# (B, H, M, D) — head dims {64,128,256}; seqlens include non-tile-multiples.
_FP16_SHAPES = [
    (1, 2, 128, 64),
    (2, 4, 512, 64),
    (2, 4, 257, 64),  # ragged tail
    (1, 2, 333, 128),  # ragged tail, D=128
    (1, 2, 200, 256),  # D=256 (dynamic-smem kernel), ragged
    (1, 8, 2048, 64),
]


def _sdpa_native_baseline(q, k, v, causal):
    """SDPA in the inputs' native dtype (fp16/bf16), GQA-aware — the error-bar
    setter for assert_relative_to_fp32 (sdpa_fp16 asserts fp16 + no GQA)."""
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=causal, enable_gqa=q.shape[1] != k.shape[1]
    )


@pytest.mark.correctness
@pytest.mark.parametrize("shape", _FP16_SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_attn_fp16_fwd_matches_oracle(device, shape, causal):
    q, k, v = make_qkv(shape, device)
    out = superl8.attn_fp16_fwd(q, k, v, causal=causal)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    base = _sdpa_native_baseline(q, k, v, causal)
    assert out.shape == oracle.shape and out.dtype == torch.float16
    assert_finite(out)
    assert_relative_to_fp32(out, base, oracle, what=f"attn_fp16_fwd {shape} causal={causal}")


@pytest.mark.correctness
@pytest.mark.parametrize("causal", [False, True])
def test_attn_fp16_fwd_gqa(device, causal):
    """GQA/MQA: fewer K/V heads than Q heads (H_q % H_kv == 0)."""
    b, hq, hkv, m, d = 2, 8, 2, 320, 64
    q = torch.randn(b, hq, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, m, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, m, d, device=device, dtype=torch.float16)
    out = superl8.attn_fp16_fwd(q, k, v, causal=causal)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    base = _sdpa_native_baseline(q, k, v, causal)
    assert_relative_to_fp32(out, base, oracle, what=f"attn_fp16_fwd GQA causal={causal}")


@pytest.mark.correctness
@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (2, 4, 257, 64), (1, 2, 333, 128)])
@pytest.mark.parametrize("causal", [False, True])
def test_attn_fp16_fwd_bf16(device, shape, causal):
    """bf16 in/out — the video-DiT dtype (LTX/Wan/Z-Image). bf16 QK dot upcasts
    to fp32 (sm_70 has no bf162 arithmetic); out follows v's dtype."""
    b, h, m, d = shape
    q = torch.randn(b, h, m, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(b, h, m, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(b, h, m, d, device=device, dtype=torch.bfloat16)
    out = superl8.attn_fp16_fwd(q, k, v, causal=causal)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    base = _sdpa_native_baseline(q, k, v, causal)
    assert out.dtype == torch.bfloat16
    assert_relative_to_fp32(out, base, oracle, what=f"attn_fp16_fwd bf16 {shape} causal={causal}")


@pytest.mark.correctness
def test_attn_fp16_fwd_deterministic(device):
    q, k, v = make_qkv((2, 4, 512, 64), device)
    r0 = superl8.attn_fp16_fwd(q, k, v)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_fp16_fwd(q, k, v), r0)


@pytest.mark.correctness
def test_attn_fp16_fwd_long_seq_is_O_N_memory(device):
    """THE POINT: at LTX-scale seqlen (9792) the kernel must RUN in O(N) memory.
    torch SDPA on Volta has no flash backend -> materializes the O(N^2) score
    matrix (~750 MB here at B*H=2, GBs at model scale) and OOMs. This kernel
    streams K/V tiles, so its extra allocation is ~the output only."""
    b, h, m, d = 1, 2, 9792, 64
    q, k, v = make_qkv((b, h, m, d), device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device)
    out = superl8.attn_fp16_fwd(q, k, v, causal=False)
    torch.cuda.synchronize(device)
    extra_peak = torch.cuda.max_memory_allocated(device) - base_alloc
    assert_finite(out)
    n2_bytes = 2 * b * h * m * m  # the fp16 score matrix SDPA would materialize
    assert extra_peak < n2_bytes // 4, (
        f"peak extra {extra_peak / 1e6:.1f} MB not O(N) (N^2 scores = {n2_bytes / 1e6:.0f} MB)"
    )
    # Correctness at long seq (fp32 oracle is feasible here at B*H=2).
    oracle = attention_fp32_oracle(q, k, v, causal=False)
    base = _sdpa_native_baseline(q, k, v, False)
    assert_relative_to_fp32(out, base, oracle, what="attn_fp16_fwd long-seq 9792")


@pytest.mark.correctness
def test_attn_fp16_fwd_rejects_bad_headdim(device):
    q, k, v = make_qkv((1, 2, 128, 96), device)  # D=96 unsupported
    with pytest.raises(RuntimeError):
        superl8.attn_fp16_fwd(q, k, v)


@pytest.mark.perf
def test_attn_fp16_fwd_long_perf(device):
    """Half2 QK^T rate + the O(N)-memory win vs torch SDPA at long seq. SDPA may
    OOM (the whole point); report it honestly. No baseline edited here (per the
    contract — a dedicated baseline PR records it); assert_no_regression
    soft-skips until then."""
    b, h, m, d = 1, 8, 4096, 64
    q, k, v = make_qkv((b, h, m, d), device)
    ours = time_ms(lambda: superl8.attn_fp16_fwd(q, k, v))
    others = {}
    try:
        sdpa = time_ms(lambda: _sdpa_native_baseline(q, k, v, False))
        others["sdpa_fp16"] = sdpa
    except RuntimeError as e:  # OOM at model scale is the motivation
        print(f"\n[attn_fp16_fwd perf] torch SDPA OOM/failed at {(b, h, m, d)}: {e}")
    name = f"attn_fp16_fwd.b{b}h{h}m{m}d{d}"
    print("\n" + compare_report(name, ours, others))
    assert_no_regression(name, ours)


# ===========================================================================
# attn_fp16_fwd additive mask (PR b) — the guided/reference video-DiT case.
# LTX passes a real additive `self_attention_mask` (a load-bearing log-space
# bias). The mask is [B|1, H|1, M, N] in NATURAL log space, added to the logits
# before softmax, broadcastable and indexed in place (no O(N^2) expansion).
# ===========================================================================
def _oracle_with_mask(q, k, v, mask, causal=False):
    """fp32 attention with an additive natural-log mask [.,M,N] (broadcast)."""
    import math as _m

    qf, kf, vf = q.float(), k.float(), v.float()
    hq, hkv = qf.shape[1], kf.shape[1]
    if hq != hkv:
        rep = hq // hkv
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)
    s = torch.einsum("bhmd,bhnd->bhmn", qf, kf) * (1.0 / _m.sqrt(q.shape[-1]))
    s = s + mask.float()  # broadcasts [B|1,H|1,M,N]
    if causal:
        ml, nl = s.shape[-2], s.shape[-1]
        cm = torch.ones(ml, nl, device=s.device, dtype=torch.bool).tril(nl - ml)
        s = s.masked_fill(~cm, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", p, vf)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", [(2, 4, 128, 64), (1, 2, 200, 128), (1, 2, 200, 256)])
@pytest.mark.parametrize("bcast", ["full", "head", "batch"])
def test_attn_fp16_fwd_additive_mask(device, shape, bcast):
    """Additive mask matches the fp32 oracle, for full [B,H,M,N] and broadcast
    ([B,1,M,N], [1,1,M,N]) masks — indexed in place, not expanded."""
    b, h, m, d = shape
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    mb = {"full": b, "head": b, "batch": 1}[bcast]
    mh = {"full": h, "head": 1, "batch": 1}[bcast]
    # A structured log-space bias incl. hard -inf (fully-masked keys) — the
    # load-bearing case (a key a query must not attend).
    mask = torch.randn(mb, mh, m, m, device=device, dtype=torch.float16) * 2.0
    mask[..., 0::7] = float("-inf")
    out = superl8.attn_fp16_fwd(q, k, v, mask=mask)
    oracle = _oracle_with_mask(q, k, v, mask)
    # baseline = SDPA with the same additive mask (fp16), GQA off (h==hkv here)
    base = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert_finite(out)
    assert_relative_to_fp32(out, base, oracle, what=f"fp16 mask {shape} {bcast}")


@pytest.mark.correctness
def test_attn_fp16_fwd_mask_plus_causal(device):
    """Additive mask composes with the causal triangle."""
    b, h, m, d = 1, 4, 192, 64
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    mask = torch.randn(b, 1, m, m, device=device, dtype=torch.float16)
    out = superl8.attn_fp16_fwd(q, k, v, causal=True, mask=mask)
    oracle = _oracle_with_mask(q, k, v, mask, causal=True)
    base = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    # (SDPA can't take both is_causal and a mask; the mask above is finite so the
    # causal triangle is applied by the oracle — compare kernel vs oracle directly
    # and use the finite-mask SDPA as a loose error-bar reference.)
    del base
    from tests.tolerances import cos_sim, rel_l1

    assert_finite(out)
    assert cos_sim(out, oracle) >= 0.999, f"cos {cos_sim(out, oracle):.6f}"
    assert rel_l1(out, oracle) <= 0.01, f"relL1 {rel_l1(out, oracle):.4f}"


@pytest.mark.correctness
def test_attn_fp16_fwd_mask_is_O_N_memory(device):
    """A broadcast [B,1,N,N] mask must NOT be expanded to [B,H,N,N]: the kernel
    indexes it via strides, so peak stays O(N·output), not O(H·N^2)."""
    b, h, m, d = 1, 8, 4096, 64
    q, k, v = make_qkv((b, h, m, d), device)
    mask = torch.randn(b, 1, m, m, device=device, dtype=torch.float16)  # [B,1,N,N]
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device)
    out = superl8.attn_fp16_fwd(q, k, v, mask=mask)
    torch.cuda.synchronize(device)
    extra_peak = torch.cuda.max_memory_allocated(device) - base_alloc
    assert_finite(out)
    # Expanding to [B,H,N,N] would add (H-1)*N^2*2 bytes; assert we're far under it.
    expand_bytes = (h - 1) * m * m * 2
    assert extra_peak < expand_bytes // 8, (
        f"mask peak {extra_peak / 1e6:.1f} MB suggests expansion (H·N^2 = {expand_bytes / 1e6:.0f} MB)"
    )
