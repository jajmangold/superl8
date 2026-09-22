# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""N-gram draft store + tree-verify integration (superl8#282). Tests first.

The store is a match table over the context/generated token stream producing
draft sequences (community prompt-lookup / llama.cpp ngram-map-k approach). A
linear n-gram draft is verified as a CHAIN tree through ``attn_int8_tree_verify``
— the built tree-verify path — and a chain tree is exactly a causal forward
(``test_tree_chain_equals_causal``), so the verify gate is the tree kernel's own.
"""

import pytest
import torch

import superl8
from superl8.ngram import NgramDraftStore, chain_tree_mask
from tests.reference import attention_fp32_tree_oracle
from tests.tolerances import assert_finite, assert_int8_quality, cos_sim


# ---------------------------------------------------------------------------
# Draft store semantics (CPU, no GPU needed).
# ---------------------------------------------------------------------------


def test_propose_returns_continuation_of_recent_match():
    # The pattern (11,12) recurs at the tail; its earlier occurrence (start 1)
    # was followed by 13,99,... — the continuation is proposed (the run rides).
    store = NgramDraftStore(min_n=2, max_n=3, max_k=8)
    store.add([10, 11, 12, 13, 99, 11, 12])
    assert store.propose(k=4) == [13, 99, 11, 12]


def test_longest_match_wins():
    # "7 8 9" recurs; the most specific (n=3) match drives the continuation.
    store = NgramDraftStore(min_n=1, max_n=4, max_k=8)
    store.add([7, 8, 9, 42, 43, 1, 2, 7, 8, 9])
    assert store.propose(k=3) == [42, 43, 1]


def test_miss_returns_empty():
    # Last "4 5" never occurred earlier with a continuation -> empty draft.
    store = NgramDraftStore(min_n=2, max_n=3)
    store.add([1, 2, 3, 4, 5])
    assert store.propose(k=4) == []


def test_respects_k_and_max_k():
    store = NgramDraftStore(min_n=2, max_n=2, max_k=2)
    store.add([1, 2, 3, 4, 5, 1, 2])  # "1 2" recurs; continuation "3 4 5..." capped
    assert store.propose(k=8) == [3, 4]


def test_structured_json_run():
    # A repeated structured key run: after the first `KEY = [ 40 41 42 ]` the
    # same `KEY =` recurs, so the store proposes the bracketed value run for free.
    KEY, EQ = 200, 201
    store = NgramDraftStore(min_n=2, max_n=4, max_k=8)
    store.add([KEY, EQ, 40, 41, 42, 99, KEY, EQ])
    assert store.propose(k=4) == [40, 41, 42, 99]


def test_store_is_incremental_match_table():
    # add() twice == add() once (the table grows with the stream).
    a = NgramDraftStore(min_n=2, max_n=3, max_k=8)
    a.add([10, 11, 12, 13, 99])
    a.add([11, 12])
    b = NgramDraftStore(min_n=2, max_n=3, max_k=8)
    b.add([10, 11, 12, 13, 99, 11, 12])
    assert a.propose(k=4) == b.propose(k=4) == [13, 99, 11, 12]


def test_could_match_after():
    store = NgramDraftStore(min_n=2, max_n=3, max_k=8)
    # (5,6) appeared earlier with a token after it -> some next token can match.
    store.add([5, 6, 7, 8, 9, 5])
    assert store.could_match_after()
    # No earlier (n-1) suffix ever had a successor -> no possible match.
    store = NgramDraftStore(min_n=2, max_n=3, max_k=8)
    store.add([1, 2, 3, 4, 5])
    assert not store.could_match_after()


def test_chain_tree_mask():
    """A linear draft is a CHAIN tree: node i attends nodes 0..i (tril)."""
    t = 5
    m = chain_tree_mask(t)
    assert m.shape == (t, t) and m.dtype == torch.int8
    assert torch.equal(m, torch.tril(torch.ones(t, t, dtype=torch.int8)))
    # sanity: row i marks exactly i+1 ancestors-or-self.
    for i in range(t):
        assert int(m[i].sum()) == i + 1


# ---------------------------------------------------------------------------
# Tree-verify integration (GPU): a chain draft flows through attn_tree_fwd.
# ---------------------------------------------------------------------------


def _chain_mask(t, device):
    return chain_tree_mask(t).to(device)


def _draft_qkv(b, hq, hkv, prefix, t, d, device):
    """q [B,H,T,D] = the draft's own queries; k/v [B,H,N,D] = [prefix + T nodes]."""
    n = prefix + t
    q = torch.randn(b, hq, t, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    return q, k, v


@pytest.mark.correctness
@pytest.mark.parametrize("prefix,t", [(300, 4), (2000, 8), (60, 2)])
@pytest.mark.parametrize("d", [64, 128])
def test_chain_draft_verify_matches_tree_oracle(device, prefix, t, d):
    """The chain-tree verify (what a linear n-gram draft becomes) equals the
    fp32 tree oracle — the same gate the EAGLE path already holds."""
    b, hq, hkv = 2, 16, 4
    q, k, v = _draft_qkv(b, hq, hkv, prefix, t, d, device)
    tm = _chain_mask(t, device)
    out = superl8.attn_int8_tree_verify(q, k, v, tm)
    oracle = attention_fp32_tree_oracle(q, k, v, tm)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"chain draft prefix={prefix} t={t} d={d}")


@pytest.mark.correctness
def test_chain_draft_equals_causal():
    """A linear draft verified as a chain tree == a causal forward over
    [prefix + draft] — the identity the acceptance harness relies on."""
    b, hq, hkv, prefix, t, d = 1, 8, 2, 128, 8, 128
    q, k, v = _draft_qkv(b, hq, hkv, prefix, t, d, torch.device("cuda"))
    tm = _chain_mask(t, torch.device("cuda"))
    tree = superl8.attn_int8_tree_verify(q, k, v, tm)
    from tests.reference import attention_fp32_oracle

    q_full = torch.cat(
        [torch.zeros(b, hq, prefix, d, device=q.device, dtype=torch.float16), q], 2
    )
    causal = attention_fp32_oracle(q_full, k, v, causal=True)[:, :, -t:]
    assert cos_sim(tree, causal) >= 0.998, f"chain-draft != causal: cos {cos_sim(tree, causal):.5f}"


@pytest.mark.correctness
def test_chain_draft_verify_deterministic(device):
    b, hq, hkv, prefix, t, d = 2, 16, 4, 400, 4, 128
    q, k, v = _draft_qkv(b, hq, hkv, prefix, t, d, device)
    tm = _chain_mask(t, device)
    r0 = superl8.attn_int8_tree_verify(q, k, v, tm)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_tree_verify(q, k, v, tm), r0)
