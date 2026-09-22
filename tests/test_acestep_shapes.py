# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Attention shapes read from the production ACE-Step XL SFT Q6 GGUF."""
import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle
from tests.tolerances import assert_finite, assert_int8_quality


@pytest.mark.correctness
@pytest.mark.parametrize(
    ("m", "n"),
    [
        (1125, 1125),  # 90 s XL SFT self-attention at 25 Hz with patch size 2
        (1125, 257),   # XL SFT latent queries attending a ragged text condition
    ],
)
def test_acestep_xl_sft_attention(device, m, n):
    """The production model uses non-causal GQA 32/8 with head dimension 128."""
    q = torch.randn(1, 32, m, 128, device=device, dtype=torch.float16)
    k = torch.randn(1, 8, n, 128, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, n, 128, device=device, dtype=torch.float16)
    out = superl8.attn_int8_fwd(q, k, v, causal=False)
    oracle = attention_fp32_oracle(q, k, v, causal=False)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"ACE-Step XL SFT m={m} n={n}")
