# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Speculative-decode / MTP VERIFY against the persistent INT8 KV cache. Tests first.

Chain/sequential speculative decoding (and Qwen3-style MTP) verify k draft tokens
in ONE forward pass against the existing K/V cache, causal with the diagonal
aligned to the sequence end. `attn_int8_verify` does this directly on the int8
cache (from `quantize_kv_cache`) — only the k drafts are quantized per call; the
cache is not re-quantized. This is the M=k analogue of `attn_decode_cached`.
"""

import pytest
import torch

import superl8
from bench.harness import assert_no_regression, compare_report, time_ms
from tests.reference import attention_fp32_oracle
from tests.tolerances import assert_finite, assert_int8_quality, cos_sim


def _fp16_dense_verify(q, kk, vv, prefix):
    """The fp16 fallback the serve MTP verify uses today: dense SDPA of the k
    drafts against the fp16 [prefix+k] cache, end-aligned causal, GQA-aware."""
    import torch.nn.functional as F

    b, hq, k, d = q.shape
    n = kk.shape[2]
    row = torch.arange(k, device=q.device).view(-1, 1) + prefix  # global draft pos
    col = torch.arange(n, device=q.device).view(1, -1)
    mask = torch.where(col <= row, 0.0, float("-inf")).to(torch.float16)  # [k,N]
    return F.scaled_dot_product_attention(
        q, kk, vv, attn_mask=mask, scale=1.0 / (d**0.5), enable_gqa=hq != kk.shape[1]
    )


def _cache_and_drafts(b, hq, hkv, prefix, k, d, device):
    # the cache = prefix + the drafts' own K/V (length N = prefix + k).
    n = prefix + k
    q = torch.randn(b, hq, k, d, device=device, dtype=torch.float16)  # k drafts
    kk = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)  # cache
    vv = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    return q, kk, vv


@pytest.mark.correctness
@pytest.mark.parametrize("prefix,k", [(500, 4), (2000, 8), (60, 2), (129, 1)])
@pytest.mark.parametrize("d", [64, 128])
def test_verify_int8_cache(device, prefix, k, d):
    b, hq, hkv = 2, 16, 4
    q, kk, vv = _cache_and_drafts(b, hq, hkv, prefix, k, d, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(kk, vv)
    out = superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale)
    oracle = attention_fp32_oracle(q, kk, vv, causal=True)  # tril(N-k) = end-aligned
    assert out.shape == q.shape
    assert_finite(out)
    # int8 QK + int8 PV -> the looser W8A8 bars.
    assert_int8_quality(
        out,
        oracle,
        what=f"verify prefix={prefix} k={k} d={d}",
        min_cos=0.998,
        max_rel_l1=0.03,
        min_sqnr_db=18.0,
    )


@pytest.mark.correctness
@pytest.mark.parametrize("k", [2, 4, 8])
@pytest.mark.parametrize("prefix", [16, 512, 4096, 4095])
@pytest.mark.parametrize("hq,hkv", [(16, 16), (16, 4), (16, 1)])
def test_verify_int8_cache_d256(device, k, prefix, hq, hkv):
    """head_dim 256 int8 verify (Gemma-family / MTP 9B & 27B). D=256 exceeds the
    48 KB static-smem cap, so this exercises the dynamic-smem W8A8 kernel path.
    Sweeps GQA/MQA ratios and a non-tile-multiple prefix (4095)."""
    b, d = 2, 256
    q, kk, vv = _cache_and_drafts(b, hq, hkv, prefix, k, d, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(kk, vv)
    out = superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale)
    oracle = attention_fp32_oracle(q, kk, vv, causal=True)  # end-aligned tril(N-k)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(
        out,
        oracle,
        what=f"verify-d256 prefix={prefix} k={k} gqa={hq}/{hkv}",
        min_cos=0.998,
        max_rel_l1=0.03,
        min_sqnr_db=18.0,
    )


@pytest.mark.correctness
def test_verify_d256_deterministic(device):
    q, kk, vv = _cache_and_drafts(2, 16, 4, 300, 4, 256, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(kk, vv)
    r0 = superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale), r0)


@pytest.mark.perf
@pytest.mark.parametrize("prefix,k", [(1024, 2), (4096, 4)])
def test_verify_d256_perf(device, prefix, k):
    """D=256 int8 verify vs the fp16 dense fallback the serve MTP path uses today.

    Honest report + int8-self regression gate (mirrors test_w8a8_fwd_perf). The
    int8-vs-fp16 ratio is FLEET-SPECIFIC: on the CMP deployment fleet the fp16
    tensor cores are firmware-gimped (~6% of a real V100) so int8 dp4a wins large
    and unblocks the MTP net-speedup; on a real V100 (the ncu-capable bench card)
    the fp16 tensor cores are full-speed and beat dp4a — so we do NOT assert
    faster-than-fp16 here (that would contradict the AGENTS.md hardware truth).
    int8 also reads HALF the KV-cache bytes regardless of card."""
    b, hq, hkv, d = 1, 16, 4, 256
    q, kk, vv = _cache_and_drafts(b, hq, hkv, prefix, k, d, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(kk, vv)  # cache: one-time
    ours = time_ms(lambda: superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale))
    fp16 = time_ms(lambda: _fp16_dense_verify(q, kk, vv, prefix))
    name = f"attn_w8a8_verify.b{b}h{hq}p{prefix}k{k}d{d}"
    print("\n" + compare_report(name, ours, {"fp16_dense": fp16}))
    assert_no_regression(name, ours)  # soft-skip until a baseline is recorded


@pytest.mark.correctness
def test_verify_matches_full_w8a8(device):
    """Verify of k drafts must equal the last k rows of a full W8A8 forward over
    the same [prefix+drafts] sequence (the parallel-verify identity)."""
    b, hq, hkv, prefix, k, d = 1, 8, 2, 256, 4, 128
    n = prefix + k
    qk = torch.randn(b, hq, n, d, device=device, dtype=torch.float16)  # full Q (self-attn)
    kk = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    vv = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    full = superl8.attn_int8_fwd(qk, kk, vv, causal=True, int8_pv=True)  # [b,hq,n,d]
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(kk, vv)
    verify = superl8.attn_int8_verify(qk[:, :, -k:], k_i8, k_scale, v_i8, v_scale)
    # both are int8 paths on the same data; the last k rows should track closely.
    assert cos_sim(verify, full[:, :, -k:]) >= 0.998, f"cos {cos_sim(verify, full[:, :, -k:]):.5f}"


@pytest.mark.correctness
def test_verify_deterministic(device):
    q, kk, vv = _cache_and_drafts(2, 16, 4, 300, 4, 128, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(kk, vv)
    r0 = superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_verify(q, k_i8, k_scale, v_i8, v_scale), r0)
