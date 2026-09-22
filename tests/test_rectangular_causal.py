# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Bottom-right causal attention for cached multi-token continuation.

FlashAttention aligns a non-square causal mask to the bottom-right: query row
``i`` attends through key ``i + N - M``.  Chunked prefill therefore passes only
the current ``M`` queries while retaining all ``N`` cached keys and values.
"""

import sys
from pathlib import Path

import pytest
import torch

import superl8

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import time_ms
from tests.reference import attention_fp32_oracle, attention_fp32_window_oracle
from tests.tolerances import assert_finite, assert_int8_quality


@pytest.mark.correctness
@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 2, 37, 101, 64),  # static-smem, GQA, ragged M/N
        (1, 4, 1, 17, 83, 256),  # dynamic-smem, MQA, ragged M/N
    ],
)
def test_rectangular_causal_matches_bottom_right_oracle(device, shape):
    b, hq, hkv, m, n, d = shape
    q = torch.randn(b, hq, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)

    out = superl8.attn_int8_fwd(q, k, v, causal=True, internal_accuracy_gate=False)
    oracle = attention_fp32_oracle(q, k, v, causal=True)

    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"rectangular causal {shape}")


@pytest.mark.correctness
def test_rectangular_causal_window_uses_shifted_query_positions(device):
    b, hq, hkv, m, n, d, window = 1, 8, 2, 33, 97, 128, 64
    q = torch.randn(b, hq, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)

    out = superl8.attn_int8_fwd(
        q,
        k,
        v,
        causal=True,
        window_left=window,
        internal_accuracy_gate=False,
    )
    oracle = attention_fp32_window_oracle(q, k, v, window_left=window)

    assert_finite(out)
    assert_int8_quality(out, oracle, what="rectangular shifted causal window")


@pytest.mark.correctness
def test_rectangular_causal_is_deterministic(device):
    q = torch.randn(1, 4, 31, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 2, 159, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 2, 159, 64, device=device, dtype=torch.float16)
    kwargs = {"causal": True, "internal_accuracy_gate": False}
    first = superl8.attn_int8_fwd(q, k, v, **kwargs)
    for _ in range(3):
        assert torch.equal(first, superl8.attn_int8_fwd(q, k, v, **kwargs))


@pytest.mark.correctness
def test_rectangular_causal_fp16_accuracy_fallback_preserves_mask(device, monkeypatch):
    import superl8.quant

    q = torch.randn(1, 4, 29, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 2, 157, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 2, 157, 64, device=device, dtype=torch.float16)
    monkeypatch.setattr(superl8.quant, "detect_q_outlier_domination", lambda _q: True)

    out = superl8.attn_int8_fwd(q, k, v, causal=True, window_left=64)
    oracle = attention_fp32_window_oracle(q, k, v, window_left=64)

    assert_finite(out)
    # This is the full-precision half2 fallback, so compare more tightly than
    # the int8 gate while allowing normal fp16 accumulation error.
    torch.testing.assert_close(out.float(), oracle, rtol=2e-2, atol=2e-2)


@pytest.mark.correctness
def test_rectangular_causal_w8a8_uses_bottom_right_diagonal(device):
    q = torch.randn(1, 4, 64, 128, device=device, dtype=torch.float16)
    k = torch.randn(1, 2, 320, 128, device=device, dtype=torch.float16)
    v = torch.randn(1, 2, 320, 128, device=device, dtype=torch.float16)

    out = superl8.attn_int8_fwd(q, k, v, causal=True, int8_pv=True, internal_accuracy_gate=False)
    oracle = attention_fp32_oracle(q, k, v, causal=True)

    assert_finite(out)
    assert_int8_quality(out, oracle, what="rectangular causal W8A8")


@pytest.mark.perf
def test_rectangular_chunk_avoids_padded_prefix_query_work(device):
    """A current chunk must be materially faster than recomputing prefix Q rows."""
    b, hq, hkv, m, n, d = 1, 8, 2, 256, 8192, 128
    q = torch.randn(b, hq, m, d, device=device, dtype=torch.float16)
    q_pad = torch.zeros(b, hq, n, d, device=device, dtype=torch.float16)
    q_pad[:, :, -m:] = q
    k = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    kwargs = {"causal": True, "internal_accuracy_gate": False}

    rectangular_ms = time_ms(lambda: superl8.attn_int8_fwd(q, k, v, **kwargs))
    padded_ms = time_ms(lambda: superl8.attn_int8_fwd(q_pad, k, v, **kwargs))

    # Pinned CMP measurement is 6.09 ms vs 21.79 ms (3.58x). Keep a wide
    # cross-host margin while requiring a material long-context win.
    assert rectangular_ms < padded_ms * 0.5, (
        f"rectangular={rectangular_ms:.3f}ms padded={padded_ms:.3f}ms"
    )
