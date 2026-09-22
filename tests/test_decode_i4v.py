# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""INT4-V KV-cache decode — the byte-aligned low-bit lever. Tests written FIRST.

V is the compressible side (the low-bit study measured K8V4 cos ~0.996). Storing
V as int4 (nibbles) halves the V cache again vs int8: int8-K/int4-V = 0.75x the
int8 cache, 0.375x fp16 -> ~2.7x more context in 16 GB. The decode kernel unpacks
a signed nibble per channel in the PV accumulation.
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle
from tests.tolerances import assert_finite, assert_int8_quality, cos_sim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import time_ms  # noqa: E402

I4V_SHAPES = [
    (1, 8, 8, 512, 64),
    (4, 8, 8, 1024, 128),
    (1, 16, 4, 2000, 128),
    (2, 8, 1, 777, 64),
    (1, 4, 4, 300, 32),
]


def _qkv(shape, device):
    b, hq, hkv, n, d = shape
    q = torch.randn(b, hq, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    return q, k, v


@pytest.mark.correctness
@pytest.mark.parametrize("shape", I4V_SHAPES)
def test_i4v_quality(device, shape):
    q, k, v = _qkv(shape, device)
    k_i8, k_scale, v_i4, v_scale = superl8.quantize_kv_cache_i4v(k, v)
    assert v_i4.shape[-1] == shape[-1] // 2  # V packed to D/2 bytes
    out = superl8.attn_decode_cached_i4v(q, k_i8, k_scale, v_i4, v_scale)
    oracle = attention_fp32_oracle(q, k, v)
    assert out.shape == q.shape
    assert_finite(out)
    # NF4 (non-uniform 4-bit) + group-32 recovers most of the int4->int8 gap
    # (uniform g128 rel-L1 0.12 -> NF4 g32 ~0.088). Still a CAPACITY lever with a
    # real (smaller) magnitude cost vs int8-V; bars reflect that honestly.
    assert_int8_quality(out, oracle, what=f"i4v {shape}",
                        min_cos=0.995, max_rel_l1=0.11, min_sqnr_db=13.0)


@pytest.mark.correctness
def test_i4v_d256_quality(device):
    """int4-V cache must also generalize to D=256 (Gemma-4B head_dim)."""
    q, k, v = _qkv((1, 8, 2, 640, 256), device)
    k_i8, k_scale, v_i4, v_scale = superl8.quantize_kv_cache_i4v(k, v)
    assert v_i4.shape[-1] == 256 // 2
    out = superl8.attn_decode_cached_i4v(q, k_i8, k_scale, v_i4, v_scale)
    oracle = attention_fp32_oracle(q, k, v)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(
        out, oracle, what="i4v d256", min_cos=0.995, max_rel_l1=0.11, min_sqnr_db=13.0
    )


@pytest.mark.correctness
def test_i4v_tracks_int8v(device):
    """int4-V must track the int8-V decode (only the extra V-quant error apart)."""
    q, k, v = _qkv((2, 8, 2, 1536, 128), device)
    k_i8, k_scale, v_i4, v_scale = superl8.quantize_kv_cache_i4v(k, v)
    out_i4 = superl8.attn_decode_cached_i4v(q, k_i8, k_scale, v_i4, v_scale)
    k8, ks8, v8, vs8 = superl8.quantize_kv_cache(k, v)
    out_i8 = superl8.attn_decode_cached(q, k8, ks8, v8, vs8)
    assert cos_sim(out_i4, out_i8) >= 0.985, f"cos {cos_sim(out_i4, out_i8):.5f}"


@pytest.mark.correctness
def test_i4v_deterministic(device):
    q, k, v = _qkv((2, 8, 8, 1024, 64), device)
    k_i8, k_scale, v_i4, v_scale = superl8.quantize_kv_cache_i4v(k, v)
    r0 = superl8.attn_decode_cached_i4v(q, k_i8, k_scale, v_i4, v_scale)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_decode_cached_i4v(q, k_i8, k_scale, v_i4, v_scale), r0)


@pytest.mark.perf
def test_i4v_perf(device):
    """int4-V vs int8-V decode (honest — decode is latency-bound, so the win is
    CAPACITY; report the speed delta too)."""
    b, hq, hkv, n, d = 8, 32, 8, 4096, 128
    q, k, v = _qkv((b, hq, hkv, n, d), device)
    k_i8, k_scale, v_i4, v_scale = superl8.quantize_kv_cache_i4v(k, v)
    k8, ks8, v8, vs8 = superl8.quantize_kv_cache(k, v)
    i4 = time_ms(lambda: superl8.attn_decode_cached_i4v(q, k_i8, k_scale, v_i4, v_scale))
    i8 = time_ms(lambda: superl8.attn_decode_cached(q, k8, ks8, v8, vs8))
    v_bytes_i8 = b * hkv * n * d
    print(f"\nint4-V decode {i4:.3f} ms | int8-V {i8:.3f} ms | {i8 / i4:.2f}x | "
          f"V cache {v_bytes_i8 // 2}B vs {v_bytes_i8}B (half)")
    assert i4 > 0  # runs; the guaranteed win is the halved V cache (capacity)
