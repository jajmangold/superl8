# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR1: the harness itself is under test.

The reference oracle, the relative-tolerance contract, and the int8 metrics
must be trustworthy before any kernel is judged by them.
"""
import pytest
import torch

from tests.reference import (
    attention_fp32_oracle,
    flash_v100_available,
    flash_v100_fp16,
    sdpa_fp16,
)
from tests.tolerances import (
    assert_int8_quality,
    assert_relative_to_fp32,
    cos_sim,
    max_abs_err,
    rel_l1,
    sqnr_db,
)

# (B, H, M, N==M, D) — small enough for the naive oracle, incl. non-tile-multiples.
SHAPES = [(1, 2, 128, 64), (2, 4, 257, 64), (1, 2, 333, 128), (1, 1, 64, 32)]


def make_qkv(shape, device, dtype=torch.float16):
    b, h, m, d = shape
    return (torch.randn(b, h, m, d, device=device, dtype=dtype) for _ in range(3))


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_oracle_matches_sdpa_fp32(device, shape, causal):
    """The naive fp32 oracle and SDPA-in-fp32 must agree tightly (two independent paths)."""
    q, k, v = make_qkv(shape, device, torch.float32)
    ours = attention_fp32_oracle(q, k, v, causal=causal)
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(ours, sdpa, rtol=1e-4, atol=1e-5)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_sdpa_fp16_passes_its_own_bar(device, shape, causal):
    """The fp16 baseline trivially satisfies the relative contract (mult >= 1)."""
    q, k, v = make_qkv(shape, device)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    base = sdpa_fp16(q, k, v, causal=causal)
    assert max_abs_err(base, oracle) < 0.05  # sanity: fp16 attention err is small
    assert_relative_to_fp32(base, base, oracle)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_v100_baseline_within_bar(device, shape, causal):
    """External ai-bond fp16 FA2 must pass the same relative bar our kernel will face."""
    if not flash_v100_available():
        pytest.skip("flash_attn_v100 not installed in this image")
    b, h, m, d = shape
    if d not in (16, 32, 64, 128, 256):
        pytest.skip(f"flash_attn_v100 does not support D={d}")
    q, k, v = make_qkv(shape, device)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    base = sdpa_fp16(q, k, v, causal=causal)
    out = flash_v100_fp16(q, k, v, causal=causal)
    assert_relative_to_fp32(out, base, oracle, what=f"flash_v100 {shape} causal={causal}")


@pytest.mark.correctness
def test_int8_metrics_calibrated(device):
    """int8 metrics must accept a faithful per-row int8 round-trip and reject garbage."""
    x = torch.randn(64, 128, device=device)
    scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
    x_q = (x / scale).round().clamp(-128, 127)
    x_dq = x_q * scale  # faithful symmetric per-row RTN round-trip
    assert sqnr_db(x_dq, x) > 30.0
    assert cos_sim(x_dq, x) > 0.999
    assert rel_l1(x_dq, x) < 0.01
    assert_int8_quality(x_dq, x, what="rtn round-trip")
    with pytest.raises(AssertionError):
        assert_int8_quality(torch.randn_like(x), x, what="garbage")
