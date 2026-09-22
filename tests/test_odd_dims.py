# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Odd head dims (D=72, 80) — broadens diffusion/model coverage beyond {32,64,128,256}.
D is a multiple of 4 so dp4a int32 packing works, and the tiles fit static smem."""
import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle
from tests.tolerances import assert_finite, assert_int8_quality


@pytest.mark.correctness
@pytest.mark.parametrize("d", [72, 80])
@pytest.mark.parametrize("causal", [False, True])
def test_odd_head_dim(device, d, causal):
    q = torch.randn(2, 8, 257, d, device=device, dtype=torch.float16)   # ragged M
    k = torch.randn(2, 2, 257, d, device=device, dtype=torch.float16)   # GQA
    v = torch.randn(2, 2, 257, d, device=device, dtype=torch.float16)
    out = superl8.attn_int8_fwd(q, k, v, causal=causal)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"odd-dim D={d} causal={causal}")


@pytest.mark.correctness
@pytest.mark.parametrize("d", [72, 80])
def test_odd_head_dim_deterministic(device, d):
    q, k, v = (torch.randn(1, 4, 200, d, device=device, dtype=torch.float16) for _ in range(3))
    r0 = superl8.attn_int8_fwd(q, k, v)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_fwd(q, k, v), r0)
