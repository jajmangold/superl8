# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR (issue #8): paged-KV block-table decode + quantize-on-write KV store —
tests written FIRST (TDD, AGENTS.md).

Two coupled ops:
  `superl8.quantize_kv_write_paged` — commit ONE new token's K/V into an int8
    paged cache (per-token RTN scale for both K and V: quantize-on-write only
    ever sees one token, so it cannot use `quantize_kv_cache`'s per-channel V
    scale, which needs the whole cache's amax up front).
  `superl8.attn_paged_decode_cached` — decode (M=1) against that cache via a
    per-sequence block table + context length, so ONE launch serves a batch of
    mixed-length sequences (today `attn_decode_cached` needs every sequence in
    the launch to share the same contiguous N).

Cache layout: k_cache/v_cache [num_blocks, H_kv, block_size, D] int8,
k_scale/v_scale [num_blocks, H_kv, block_size] fp32. block_table
[B, max_blocks_per_seq] int32 maps a sequence's logical block index to a
PHYSICAL block id that need not be contiguous or sorted — the whole point of a
block table is that physical blocks are scattered/reused across sequences, so
every test below allocates a randomly shuffled block_table to make sure the
kernel actually follows it instead of assuming block_id == logical index.
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8
from tests.reference import attention_fp32_paged_oracle
from tests.tolerances import assert_finite, assert_int8_quality

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import time_ms  # noqa: E402

# (H_q, H_kv, block_size, context_lens, D). context_lens are per-sequence key
# counts, deliberately NOT multiples of block_size (AGENTS.md: include sizes
# that are not tile multiples), and deliberately ragged within one batch —
# the scenario a paged, block-table launch exists to serve in one shot.
PAGED_SHAPES = [
    (8, 8, 16, [37, 100, 5, 63, 16, 1, 200, 48], 64),   # MHA, ragged lens, D=64
    (16, 4, 32, [70, 33, 129, 32], 128),                # GQA group=4, D=128
    (8, 1, 16, [9, 500, 47], 32),                       # MQA, short D=32
    (4, 4, 16, [1, 1000], 256),                         # D=256, extreme ragged
]


def _make_paged_cache(h_kv, context_lens, block_size, d, device, *, shuffle=True):
    """Allocate an int8 paged pool sized for the batch. Each sequence's
    logical blocks are assigned a physical block id via a (by default)
    shuffled permutation, so block ids are neither contiguous nor in
    sequence order."""
    blocks_per_seq = [(n + block_size - 1) // block_size for n in context_lens]
    max_blocks = max(blocks_per_seq)
    num_blocks = sum(blocks_per_seq) + 3  # a few spare, never-referenced blocks
    order = torch.randperm(num_blocks) if shuffle else torch.arange(num_blocks)
    block_table = torch.zeros((len(context_lens), max_blocks), dtype=torch.int32)
    cursor = 0
    for i, nb in enumerate(blocks_per_seq):
        block_table[i, :nb] = order[cursor:cursor + nb].to(torch.int32)
        cursor += nb
    k_cache = torch.zeros(num_blocks, h_kv, block_size, d, dtype=torch.int8, device=device)
    k_scale = torch.ones(num_blocks, h_kv, block_size, dtype=torch.float32, device=device)
    v_cache = torch.zeros(num_blocks, h_kv, block_size, d, dtype=torch.int8, device=device)
    v_scale = torch.ones(num_blocks, h_kv, block_size, dtype=torch.float32, device=device)
    return block_table.to(device), k_cache, k_scale, v_cache, v_scale


def _fill_paged_cache(k_dense, v_dense, context_lens, block_table, block_size,
                      k_cache, k_scale, v_cache, v_scale):
    """Commit every token of every sequence ONE AT A TIME via
    `quantize_kv_write_paged` — the real decode-loop usage pattern (append the
    newest token, then decode against the whole cache)."""
    device = k_dense.device
    for t in range(max(context_lens)):
        active = [b for b, n in enumerate(context_lens) if t < n]
        if not active:
            continue
        idx = torch.tensor(active, device=device)
        slots = [int(block_table[b, t // block_size]) * block_size + (t % block_size)
                 for b in active]
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device=device)
        superl8.quantize_kv_write_paged(
            k_dense[idx, :, t, :].contiguous(), v_dense[idx, :, t, :].contiguous(),
            k_cache, k_scale, v_cache, v_scale, slot_mapping,
        )


def _paged_setup(shape, device, *, shuffle=True):
    hq, hkv, block_size, context_lens, d = shape
    b = len(context_lens)
    n_max = max(context_lens)
    q = torch.randn(b, hq, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n_max, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n_max, d, device=device, dtype=torch.float16)
    block_table, k_cache, k_scale, v_cache, v_scale = _make_paged_cache(
        hkv, context_lens, block_size, d, device, shuffle=shuffle
    )
    _fill_paged_cache(k, v, context_lens, block_table, block_size,
                      k_cache, k_scale, v_cache, v_scale)
    cl = torch.tensor(context_lens, dtype=torch.int32, device=device)
    return q, k, v, block_table, cl, k_cache, k_scale, v_cache, v_scale


@pytest.mark.correctness
@pytest.mark.parametrize("shape", PAGED_SHAPES)
def test_paged_decode_quality(device, shape):
    _, _, block_size, context_lens, _ = shape
    q, k, v, block_table, cl, k_cache, k_scale, v_cache, v_scale = _paged_setup(shape, device)
    out = superl8.attn_paged_decode_cached(
        q, k_cache, k_scale, v_cache, v_scale, block_table, cl, block_size
    )
    oracle = attention_fp32_paged_oracle(q, k, v, context_lens)
    assert out.shape == q.shape
    assert_finite(out)
    # int8 V (per-token, no channel-scale averaging) adds a bit more error
    # than fp16 V -> same looser bar as the contiguous int8-KV decode test.
    assert_int8_quality(out, oracle, what=f"paged decode {shape}",
                        min_cos=0.998, max_rel_l1=0.03, min_sqnr_db=18.0)


@pytest.mark.correctness
def test_paged_decode_deterministic(device):
    shape = (8, 8, 16, [37, 100, 5, 63, 16, 1, 200, 48], 64)
    _, _, block_size, _, _ = shape
    q, k, v, block_table, cl, k_cache, k_scale, v_cache, v_scale = _paged_setup(shape, device)
    r0 = superl8.attn_paged_decode_cached(
        q, k_cache, k_scale, v_cache, v_scale, block_table, cl, block_size
    )
    assert_finite(r0)
    for _ in range(3):
        r = superl8.attn_paged_decode_cached(
            q, k_cache, k_scale, v_cache, v_scale, block_table, cl, block_size
        )
        assert torch.equal(r, r0)


@pytest.mark.correctness
def test_paged_decode_block_table_indirection(device):
    """A shuffled (scattered, non-contiguous) block_table must give the exact
    same result as an identity-order one holding the same logical tokens —
    proves the kernel dereferences block_table rather than assuming
    block_id == logical block index."""
    torch.manual_seed(1)
    shape = (4, 2, 16, [20, 45, 100, 8], 64)
    hq, hkv, block_size, context_lens, d = shape
    b = len(context_lens)
    n_max = max(context_lens)
    q = torch.randn(b, hq, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n_max, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n_max, d, device=device, dtype=torch.float16)
    cl = torch.tensor(context_lens, dtype=torch.int32, device=device)

    outs = []
    for shuffle in (False, True):
        block_table, k_cache, k_scale, v_cache, v_scale = _make_paged_cache(
            hkv, context_lens, block_size, d, device, shuffle=shuffle
        )
        _fill_paged_cache(k, v, context_lens, block_table, block_size,
                          k_cache, k_scale, v_cache, v_scale)
        outs.append(superl8.attn_paged_decode_cached(
            q, k_cache, k_scale, v_cache, v_scale, block_table, cl, block_size
        ))
    assert torch.equal(outs[0], outs[1]), "block-table indirection changed the result"


@pytest.mark.correctness
def test_kv_write_paged_only_touches_its_own_slot(device):
    """Writing one token must not perturb any other row of the cache."""
    hkv, block_size, d = 2, 16, 64
    context_lens = [50, 33]
    block_table, k_cache, k_scale, v_cache, v_scale = _make_paged_cache(
        hkv, context_lens, block_size, d, device
    )
    b, n_max = len(context_lens), max(context_lens)
    k = torch.randn(b, hkv, n_max, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n_max, d, device=device, dtype=torch.float16)
    _fill_paged_cache(k, v, context_lens, block_table, block_size,
                      k_cache, k_scale, v_cache, v_scale)
    k_cache_before = k_cache.clone()
    v_cache_before = v_cache.clone()

    slot = int(block_table[0, 0]) * block_size + block_size - 1  # overwrite one already-written row
    slot_mapping = torch.tensor([slot], dtype=torch.int32, device=device)
    k_new = torch.randn(1, hkv, d, device=device, dtype=torch.float16)
    v_new = torch.randn(1, hkv, d, device=device, dtype=torch.float16)
    superl8.quantize_kv_write_paged(k_new, v_new, k_cache, k_scale, v_cache, v_scale, slot_mapping,
                                 rotate=False)

    # Every (block, h, offset) row must be unchanged except the written slot.
    for h in range(hkv):
        for blk in range(k_cache.shape[0]):
            for off in range(block_size):
                if blk == int(block_table[0, 0]) and off == block_size - 1:
                    continue
                assert torch.equal(k_cache[blk, h, off], k_cache_before[blk, h, off])
                assert torch.equal(v_cache[blk, h, off], v_cache_before[blk, h, off])


@pytest.mark.perf
def test_paged_decode_beats_looped_contiguous_decode(device):
    """The whole point of a block table: ONE launch over a ragged batch must
    beat looping the contiguous `attn_decode_cached` once per sequence
    (today's only option for mixed-length batched decode)."""
    hq, hkv, block_size, d = 8, 8, 16, 128
    context_lens = [512, 1024, 2048, 4096, 700, 1500, 3000, 256]
    shape = (hq, hkv, block_size, context_lens, d)
    q, k, v, block_table, cl, k_cache, k_scale, v_cache, v_scale = _paged_setup(shape, device)
    max_context_len = max(context_lens)  # a real engine has this as a plain int already

    def paged():
        return superl8.attn_paged_decode_cached(
            q, k_cache, k_scale, v_cache, v_scale, block_table, cl, block_size,
            max_context_len=max_context_len,
        )

    def looped():
        outs = []
        for b, n in enumerate(context_lens):
            k_i8, k_sc, v_i8, v_sc = superl8.quantize_kv_cache(k[b:b + 1, :, :n], v[b:b + 1, :, :n])
            outs.append(superl8.attn_decode_cached(q[b:b + 1], k_i8, k_sc, v_i8, v_sc))
        return torch.cat(outs, dim=0)

    paged_ms = time_ms(paged)
    looped_ms = time_ms(looped)
    print(f"\npaged_decode {paged_ms:.3f} ms | looped_contiguous_decode {looped_ms:.3f} ms "
          f"| {looped_ms / paged_ms:.2f}x")
    assert paged_ms < looped_ms, (
        f"one paged-decode launch ({paged_ms:.3f} ms) must beat looping the contiguous "
        f"decode once per sequence ({looped_ms:.3f} ms)"
    )


@pytest.mark.perf
def test_kv_write_paged_beats_full_cache_requantize(device):
    """Quantize-on-write commits ONE token in O(1); the naive alternative —
    re-running the batch quantizer over the whole growing cache every decode
    step — is O(N) and must lose, badly, once N is large."""
    b, hkv, d, n = 8, 8, 128, 4096
    k_dense = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    v_dense = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    block_size = 16
    context_lens = [n] * b
    block_table, k_cache, k_scale, v_cache, v_scale = _make_paged_cache(
        hkv, context_lens, block_size, d, device
    )
    logical_block, offset = (n - 1) // block_size, (n - 1) % block_size
    slots = [int(block_table[bi, logical_block]) * block_size + offset for bi in range(b)]
    slot_mapping = torch.tensor(slots, dtype=torch.int32, device=device)
    k_new = k_dense[:, :, -1, :].contiguous()
    v_new = v_dense[:, :, -1, :].contiguous()

    write = lambda: superl8.quantize_kv_write_paged(  # noqa: E731
        k_new, v_new, k_cache, k_scale, v_cache, v_scale, slot_mapping
    )
    requantize_full = lambda: superl8.quantize_kv_cache(k_dense, v_dense)  # noqa: E731

    write_ms = time_ms(write)
    full_ms = time_ms(requantize_full)
    print(f"\nkv_write_paged {write_ms:.4f} ms | requantize_full_cache {full_ms:.4f} ms "
          f"| {full_ms / write_ms:.1f}x")
    assert write_ms < full_ms, "quantize-on-write must beat re-quantizing the whole cache"
