# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""N-gram speculative draft store + chain-tree verify integration (superl8#282).

The design doc's N-gram write-up (qwen38-tq3 docs/superl8-tq34s-fusion.md) quantifies
n-gram as the highest-value untapped lever for the agentic JSON/code workload:
long repetitive tails, and each verified draft costs ONE weight pass. The verify
machinery (``attn_int8_tree_verify``, chain trees) is already built and merged;
this module supplies the missing draft store and wires drafts into that path.

The store is a prompt-lookup / n-gram MAP match table over the context/generated
token stream (Saxena 2023; llama.cpp ``ngram-map-k`` / ``ngram-simple``):

  * :class:`NgramDraftStore` — a hash table from each n-gram (n in [min_n,max_n])
    to its most recent start positions in the stream. ``add`` grows the stream
    and the table incrementally; ``propose`` looks up the current tail n-gram,
    takes the most recent EARLIER occurrence with a continuation, and returns up
    to ``k`` continuation tokens as the draft. Longest/most-specific match wins
    (n swept high -> low), the same contract as the serving-side
    ``NgramDrafter`` in fni8-serve (one logical place for the store; the serving
    cascade can swap it in).
  * :func:`chain_tree_mask` — a LINEAR n-gram draft is a CHAIN tree (node i's
    parent is node i-1); this builds the ``[T,T]`` int8 tree mask that
    ``attn_int8_tree_verify`` consumes. A chain tree is exactly a causal forward
    (``tests/test_tree.py::test_tree_chain_equals_causal``), so verification is
    the existing tree kernel — no new CUDA.

The acceptance harness (``bench/ngram_acceptance.py``) measures how many drafts
survive verification on real agentic vs novel-prose streams; tok/s with n-gram
on = base decode tok/s x (accepted tokens per pass).
"""
from __future__ import annotations

from collections import deque

import torch

# How many recent start positions to keep per n-gram in the match table. Enough
# recency to ride repeated runs; bounded so a long stream can't grow the table
# without limit. (The table never holds the whole history, only the positions of
# the n-grams that actually repeat near the tail.)
MAX_RECENT = 8


class NgramDraftStore:
    """Match table over a token stream producing n-gram draft sequences.

    ``propose`` finds the most recent earlier occurrence of the current tail
    n-gram and returns what followed it as the draft. Longest match wins (n
    swept ``max_n`` -> ``min_n``). Pure Python — CPU-side, no CUDA; it feeds the
    tree-verify path (:func:`chain_tree_mask` + ``attn_int8_tree_verify``).
    """

    def __init__(self, min_n: int = 2, max_n: int = 3, max_k: int = 8):
        # min_n: shortest pattern we trust (n==1 matches far too loosely and
        # drafts noise); max_n: longest pattern we bother trying (diminishing
        # returns). max_k: cap on proposed continuation length (the verify cost
        # ceiling).
        self.min_n = max(1, min_n)
        self.max_n = max(self.min_n, max_n)
        self.max_k = max_k
        self._stream: list[int] = []
        # table[n][ngram_tuple] = deque of start positions (most recent LAST).
        self._table: dict[int, dict[tuple, deque]] = {}

    # -- stream / table maintenance ----------------------------------------

    def reset(self) -> None:
        self._stream = []
        self._table = {}

    def add(self, tokens: list[int]) -> None:
        """Append ``tokens`` to the stream and update the match table.

        The table also keeps the ``(n-1)``-grams (down to ``min_n - 1``): they are
        what :meth:`could_match_after` needs to decide whether a not-yet-sampled
        token could complete a match. ``propose`` only consults ``[min_n, max_n]``."""
        start = len(self._stream)
        self._stream.extend(tokens)
        for n in range(max(1, self.min_n - 1), self.max_n + 1):
            if n > len(self._stream):
                break
            tbl = self._table.setdefault(n, {})
            for p in range(start, len(self._stream)):
                s = p - n + 1
                if s < 0:
                    continue
                key = tuple(self._stream[s:p + 1])
                dq = tbl.setdefault(key, deque(maxlen=MAX_RECENT))
                if not dq or dq[-1] != s:
                    dq.append(s)

    # -- drafting ----------------------------------------------------------

    def propose(self, k: int | None = None) -> list[int]:
        """Draft up to ``min(k, max_k)`` continuation tokens from the stream.

        Sweeps pattern length ``n`` from ``max_n`` down to ``min_n``; for the
        current tail n-gram, takes the most recent EARLIER occurrence (strictly
        before the tail, with room for a continuation) and returns the up-to-k
        tokens that followed it. Empty list on a miss. Longest match first =
        most context = best drafts."""
        k = min(k if k is not None else self.max_k, self.max_k)
        if k <= 0:
            return []
        L = len(self._stream)
        for n in range(min(self.max_n, L - 1), self.min_n - 1, -1):
            key = tuple(self._stream[L - n:])
            dq = self._table.get(n, {}).get(key)
            if not dq:
                continue
            for s in reversed(dq):
                if s + n < L:  # strictly before the tail, continuation exists
                    return self._stream[s + n: s + n + k]
        return []

    def could_match_after(self) -> bool:
        """Whether *some* next token could complete a usable n-gram match.

        The target model has not produced that token yet, but the preceding
        ``n-1`` suffix is already known. If it never appeared earlier with a
        token after it, no possible target token can make :meth:`propose`
        succeed — a caller can stay on plain decode without an eager probe.
        """
        L = len(self._stream)
        for n in range(self.min_n, min(self.max_n, L + 1) + 1):
            prefix_len = n - 1
            suffix = tuple(self._stream[L - prefix_len:]) if prefix_len else ()
            dq = self._table.get(prefix_len, {}).get(suffix) if prefix_len else None
            if dq:
                for s in dq:
                    if s + prefix_len < L:  # had a successor in the stream
                        return True
        return False

    def __len__(self) -> int:
        return len(self._stream)


def chain_tree_mask(t: int) -> torch.Tensor:
    """``[T,T]`` int8 tree mask for a LINEAR draft (a chain tree).

    Node i's parent is node i-1, so node i attends nodes ``0..i`` (ancestors-or-
    self) — exactly ``tril``. ``attn_int8_tree_verify`` consumes this mask; a
    chain tree degenerates to a causal forward, which ``test_tree_chain_equals_causal``
    already proves the kernel handles correctly.
    """
    return torch.tril(torch.ones(t, t, dtype=torch.int8))
