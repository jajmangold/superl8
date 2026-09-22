# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Tree-attention verify — the EAGLE / tree-speculative-decoding path. Tests first.

EAGLE drafts a TREE of candidate tokens and verifies the whole tree in one forward:
each node attends the shared prefix (KV cache) + its ANCESTORS in the tree, not
siblings. `attn_int8_tree_verify` bakes that custom mask into the int8 dp4a kernel
(the pattern that "precludes FlashAttention" in eager tree implementations).
"""
import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle, attention_fp32_tree_oracle
from tests.tolerances import assert_finite, assert_int8_quality, cos_sim


def _tree_mask(parents, device):
    """parents[i] = parent of node i (-1 for root). Returns [T,T] int8:
    mask[i][j] = 1 iff j is an ancestor-or-self of i (the nodes i attends)."""
    t = len(parents)
    m = torch.zeros(t, t, dtype=torch.int8, device=device)
    for i in range(t):
        j = i
        while j != -1:
            m[i, j] = 1
            j = parents[j]
    return m


# A branching draft tree: root -> {1,2}; 1 -> {3,4}; 2 -> {5,6}; 3 -> {7}.
TREE_PARENTS = [-1, 0, 0, 1, 1, 2, 2, 3]


@pytest.mark.correctness
@pytest.mark.parametrize("prefix", [0, 300, 2000])
@pytest.mark.parametrize("d", [64, 128])
def test_tree_verify(device, prefix, d):
    b, hq, hkv = 2, 16, 4
    parents = TREE_PARENTS
    t = len(parents)
    n = prefix + t
    q = torch.randn(b, hq, t, d, device=device, dtype=torch.float16)     # tree nodes
    k = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)    # prefix + tree
    v = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    tm = _tree_mask(parents, device)
    out = superl8.attn_int8_tree_verify(q, k, v, tm)
    oracle = attention_fp32_tree_oracle(q, k, v, tm)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"tree prefix={prefix} d={d}")


@pytest.mark.correctness
def test_tree_chain_equals_causal(device):
    """A CHAIN tree (each node's parent is the previous) is exactly causal —
    the tree verify must then equal a normal causal forward over [prefix+chain]."""
    b, hq, hkv, prefix, t, d = 1, 8, 2, 128, 8, 128
    n = prefix + t
    parents = [i - 1 for i in range(t)]  # 0<-root, chain
    q = torch.randn(b, hq, t, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    tm = _tree_mask(parents, device)
    tree = superl8.attn_int8_tree_verify(q, k, v, tm)
    # causal reference: the last t rows of a causal forward over the full [prefix+chain].
    q_full = torch.cat([torch.zeros(b, hq, prefix, d, device=device, dtype=torch.float16), q], 2)
    causal = attention_fp32_oracle(q_full, k, v, causal=True)[:, :, -t:]
    assert cos_sim(tree, causal) >= 0.998, f"chain-tree != causal: cos {cos_sim(tree, causal):.5f}"


@pytest.mark.correctness
def test_tree_deterministic(device):
    b, hq, hkv, prefix, d = 2, 16, 4, 400, 128
    t = len(TREE_PARENTS)
    q = torch.randn(b, hq, t, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, prefix + t, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, prefix + t, d, device=device, dtype=torch.float16)
    tm = _tree_mask(TREE_PARENTS, device)
    r0 = superl8.attn_int8_tree_verify(q, k, v, tm)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_tree_verify(q, k, v, tm), r0)


@pytest.mark.correctness
def test_tree_per_request(device):
    """Per-request trees: a different [T,T] mask per batch item ([B,T,T])."""
    b, hq, hkv, prefix, d = 2, 8, 2, 200, 128
    pa = [-1, 0, 0, 1, 1, 2, 2, 3]         # batch 0's tree
    pb = [-1, 0, 1, 1, 0, 4, 4, 5]         # batch 1's tree (different)
    t = len(pa)
    n = prefix + t
    q = torch.randn(b, hq, t, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    ma, mb = _tree_mask(pa, device), _tree_mask(pb, device)
    tm = torch.stack([ma, mb], 0)          # [B,T,T]
    out = superl8.attn_int8_tree_verify(q, k, v, tm)
    for i, m in enumerate((ma, mb)):       # each batch matches ITS tree's oracle
        o_i = attention_fp32_tree_oracle(q[i:i+1], k[i:i+1], v[i:i+1], m)
        assert_int8_quality(out[i:i+1], o_i, what=f"tree per-request batch {i}")
