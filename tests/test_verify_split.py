# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""KV-split (flash-decoding) VERIFY variant. Tests first.

The dense-tile W8A8 verify kernel (`attn_int8_verify(..., use_split=False)`) is
CORRECT at head_dim 256 but launches only `B·H_q` blocks at the tiny-M / large-N
MTP verify shape (B=1, H_q=16, k=2 → 16 blocks, ~6% occupancy) — latency-bound,
masking the int8 bandwidth win. The split-KV variant
(`attn_int8_verify(..., use_split=True)`, the default) maps the k drafts onto the
proven `decode_split_i8v` machinery as k staggered end-aligned-causal decode
queries, launching `n_splits·B·H_q·k` blocks + an LSE combine.

Correctness FIRST (AGENTS.md TDD): the split kernel must match the non-split
kernel (the LSE combine is split-invariant → numerically the same) and the fp32
oracle across the #179 sweep, be deterministic, and not regress D≤128. The perf
test is the deliverable: split must beat the non-split dense-tile kernel at the
verify shape (the under-launch fix), reported alongside the fp16 dense fallback.
"""

import pytest
import torch

import superl8
from bench.harness import assert_no_regression, compare_report, time_ms
from tests.reference import attention_fp32_oracle
from tests.test_verify import _cache_and_drafts, _fp16_dense_verify
from tests.tolerances import assert_finite, assert_int8_quality, cos_sim


@pytest.mark.correctness
@pytest.mark.parametrize("k", [2, 4, 8])
@pytest.mark.parametrize("prefix", [16, 512, 4096, 4095])
@pytest.mark.parametrize("hq,hkv", [(16, 16), (16, 4), (16, 1)])
def test_verify_split_matches_nonsplit_d256(device, k, prefix, hq, hkv):
    """Split-KV verify must track the #179 dense-tile kernel closely AND clear the
    same int8 quality bars vs the fp32 oracle, across the full #179 D=256 sweep:
    k∈{2,4,8}, prefix incl. non-tile-multiple 4095, GQA 16/16, GQA 16/4, MQA 16/1.

    Note: the two kernels use different (both valid) int8 PV recipes — the split
    path reuses the decode machinery's fp32-P × int8-V PV, the #179 dense kernel
    re-quantizes P to int8 for a dp4a PV — so they agree closely but not bitwise.
    The exact split-count invariance is checked separately below."""
    b, d = 2, 256
    q, kk, vv = _cache_and_drafts(b, hq, hkv, prefix, k, d, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(kk, vv)
    split = superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale)  # use_split=True (default)
    nonsplit = superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale, use_split=False)
    oracle = attention_fp32_oracle(q, kk, vv, causal=True)  # end-aligned tril(N-k)
    assert split.shape == q.shape
    assert_finite(split)
    # Two int8 PV recipes on the same data → track closely (both are good vs oracle).
    c = cos_sim(split, nonsplit)
    assert c >= 0.998, f"split vs non-split cos {c:.6f} (prefix={prefix} k={k} gqa={hq}/{hkv})"
    # And the split path clears the same int8 quality bars vs the fp32 oracle.
    assert_int8_quality(
        split,
        oracle,
        what=f"verify-split-d256 prefix={prefix} k={k} gqa={hq}/{hkv}",
        min_cos=0.998,
        max_rel_l1=0.03,
        min_sqnr_db=18.0,
    )


@pytest.mark.correctness
@pytest.mark.parametrize("prefix,k", [(500, 4), (2000, 8), (60, 2), (33, 1)])
@pytest.mark.parametrize("d", [64, 128])
def test_verify_split_no_regression_d_le_128(device, prefix, k, d):
    """No regression to the D≤128 verify: the split path must stay correct vs the
    non-split kernel and the oracle at head_dim 64 and 128."""
    b, hq, hkv = 2, 16, 4
    q, kk, vv = _cache_and_drafts(b, hq, hkv, prefix, k, d, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(kk, vv)
    split = superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale)
    nonsplit = superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale, use_split=False)
    oracle = attention_fp32_oracle(q, kk, vv, causal=True)
    assert_finite(split)
    assert cos_sim(split, nonsplit) >= 0.998, f"cos {cos_sim(split, nonsplit):.6f}"
    assert_int8_quality(
        split, oracle, what=f"verify-split d={d} prefix={prefix} k={k}",
        min_cos=0.998, max_rel_l1=0.03, min_sqnr_db=18.0,
    )


@pytest.mark.correctness
@pytest.mark.parametrize("num_splits", [1, 4, 32, 128])
def test_verify_split_invariant_to_split_count(device, num_splits):
    """The number of KV splits must NOT change the result (LSE combine is
    split-count-invariant) — a hard check that the combine is correct."""
    b, hq, hkv, prefix, k, d = 2, 16, 4, 2000, 4, 256
    q, kk, vv = _cache_and_drafts(b, hq, hkv, prefix, k, d, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(kk, vv)
    ref = superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale, num_splits=1)
    got = superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale, num_splits=num_splits)
    assert_finite(got)
    c = cos_sim(got, ref)
    assert c >= 0.99995, f"num_splits={num_splits} cos vs 1-split {c:.6f}"


@pytest.mark.correctness
def test_verify_split_deterministic(device):
    q, kk, vv = _cache_and_drafts(2, 16, 4, 300, 4, 256, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(kk, vv)
    r0 = superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale), r0)


@pytest.mark.perf
@pytest.mark.parametrize("prefix,k", [(1024, 2), (4096, 4)])
def test_verify_split_perf(device, prefix, k):
    """DELIVERABLE: the split-KV verify must beat the #179 dense-tile (non-split)
    kernel at the verify shape — the under-launch fix. Reports split vs non-split
    vs the fp16 dense fallback.

    The split-vs-non-split win is a real, card-independent occupancy fix (the
    non-split kernel launches only B·H_q blocks; split launches n_splits·B·H_q·k).
    The int8-vs-fp16 ratio is FLEET-SPECIFIC (AGENTS.md): on the CMP deployment
    fleet the fp16 tensor cores are firmware-gimped (~6% of a real V100) so int8
    dp4a wins and flips MTP to a net decode gain; on the real-V100 bench card the
    fp16 tensor cores are full-speed — so we assert only the split-vs-non-split
    fix here (not faster-than-fp16, which would contradict the hardware truth on
    the ncu card). int8 also reads HALF the KV-cache bytes regardless of card."""
    b, hq, hkv, d = 1, 16, 4, 256
    q, kk, vv = _cache_and_drafts(b, hq, hkv, prefix, k, d, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(kk, vv)  # cache: one-time

    split = time_ms(lambda: superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale))
    nonsplit = time_ms(
        lambda: superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale, use_split=False)
    )
    fp16 = time_ms(lambda: _fp16_dense_verify(q, kk, vv, prefix))
    name = f"attn_w8a8_verify_split.b{b}h{hq}p{prefix}k{k}d{d}"
    print("\n" + compare_report(name, split, {"nonsplit_int8": nonsplit, "fp16_dense": fp16}))

    # The deliverable: the split launch fixes the under-occupied non-split kernel.
    assert split < nonsplit, (
        f"split-KV verify ({split:.3f} ms) must beat the under-launched non-split "
        f"kernel ({nonsplit:.3f} ms) at prefix={prefix} k={k}"
    )
    assert_no_regression(name, split)  # soft-skip until a baseline is recorded
