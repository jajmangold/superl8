# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR6 coverage matrix — GQA/MQA and extended head dims. Tests written FIRST.

GQA (grouped-query attention): K/V have H_kv heads, Q has H_q = g*H_kv heads;
each K/V head is shared by g consecutive query heads. Every Qwen/Llama model
uses it, so the kernel is unusable on real models without it.
"""
import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle
from tests.tolerances import assert_int8_quality

# (B, H_q, H_kv, M, N, D). H_q % H_kv == 0. Includes MQA (H_kv=1).
GQA_SHAPES = [
    (1, 8, 2, 256, 256, 64),    # GQA group=4
    (2, 8, 1, 128, 128, 64),    # MQA
    (1, 16, 4, 257, 257, 128),  # GQA group=4, D=128, ragged
    (1, 4, 2, 192, 192, 128),
]


def _qkv_gqa(shape, device):
    b, hq, hkv, m, n, d = shape
    q = torch.randn(b, hq, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    return q, k, v


@pytest.mark.correctness
@pytest.mark.parametrize("shape", GQA_SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_gqa_forward(device, shape, causal):
    q, k, v = _qkv_gqa(shape, device)
    out = superl8.attn_int8_fwd(q, k, v, causal=causal)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    assert out.shape == q.shape
    assert_int8_quality(out, oracle, what=f"gqa fwd {shape} causal={causal}")


@pytest.mark.correctness
@pytest.mark.parametrize("shape", GQA_SHAPES)
def test_gqa_w8a8_forward(device, shape):
    q, k, v = _qkv_gqa(shape, device)
    out = superl8.attn_int8_fwd(q, k, v, int8_pv=True)
    oracle = attention_fp32_oracle(q, k, v)
    assert_int8_quality(out, oracle, what=f"gqa w8a8 {shape}",
                        min_cos=0.998, max_rel_l1=0.03, min_sqnr_db=18.0)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", GQA_SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_gqa_backward(device, shape, causal):
    """GQA backward: dQ per Q head, dK/dV summed over the group's Q heads."""
    from tests.tolerances import cos_sim, rel_l1

    q, k, v = _qkv_gqa(shape, device)
    out = superl8.attn_int8_fwd(q, k, v, causal=causal)
    d_out = torch.randn_like(out)
    dq, dk, dv = superl8.backward_cuda(q, k, v, out, None, d_out, causal=causal)
    assert dk.shape == k.shape and dv.shape == v.shape and dq.shape == q.shape

    qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    ref = attention_fp32_oracle(qf, kf, vf, causal=causal)
    ref.backward(d_out.float())
    for name, got, exp in [("dq", dq, qf.grad), ("dk", dk, kf.grad), ("dv", dv, vf.grad)]:
        assert cos_sim(got, exp) >= 0.99, f"{name} {shape} c={causal}: cos {cos_sim(got, exp):.4f}"
        assert rel_l1(got, exp) <= 0.06, f"{name} {shape} c={causal}: relL1 {rel_l1(got, exp):.4f}"


@pytest.mark.correctness
@pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("causal", [False, True])
def test_extra_head_dim(device, d, causal):
    """Head dim beyond the {64,128} MVP set (D=32 fits static smem)."""
    q, k, v = (torch.randn(2, 4, 256, d, device=device, dtype=torch.float16) for _ in range(3))
    out = superl8.attn_int8_fwd(q, k, v, causal=causal)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    assert_int8_quality(out, oracle, what=f"D={d} causal={causal}")


@pytest.mark.correctness
def test_gqa_matches_repeated_kv(device):
    """GQA output must equal the dense output with K/V repeat-interleaved."""
    q, k, v = _qkv_gqa((1, 8, 2, 256, 256, 64), device)
    out_gqa = superl8.attn_int8_fwd(q, k, v)
    k_rep = k.repeat_interleave(4, dim=1)
    v_rep = v.repeat_interleave(4, dim=1)
    out_dense = superl8.attn_int8_fwd(q, k_rep, v_rep)
    torch.testing.assert_close(out_gqa, out_dense, rtol=0, atol=0)
