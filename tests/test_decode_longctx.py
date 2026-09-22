# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR: long-context split-KV decode tuning — tests written FIRST (TDD).

For batch≈1 / long context, DEC_MAX_SPLITS=64 may not produce enough blocks
to fill all SMs (especially with MQA where H_q=1). This suite validates
correctness at long context and that more splits preserve output quality.
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle, attention_fp32_paged_oracle
from tests.tolerances import assert_finite, assert_int8_quality, cos_sim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import time_ms  # noqa: E402

# Long-context decode shapes: batch≈1, long sequences, MQA/GQA.
# Include non-tile-multiple N values that test split boundary logic.
LONGCTX_SHAPES = [
    (1, 1, 1, 16384, 128),     # MQA, 16K context, worst-case SM fill
    (1, 4, 2, 7000, 64),       # GQA, non-tile-multiple N
    (1, 1, 1, 32000, 128),     # MQA, 32K context
    (1, 8, 4, 31000, 128),     # GQA group=2, non-tile-multiple N
    (1, 8, 8, 64000, 64),      # MHA, 64K context
]

LONGCTX_D256_SHAPES = [
    (1, 1, 1, 16384, 256),     # D=256, MQA
    (1, 4, 1, 32000, 256),     # D=256, MQA, non-tile N
]


def _qkv_decode(shape, device):
    b, hq, hkv, n, d = shape
    q = torch.randn(b, hq, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    return q, k, v


@pytest.mark.correctness
@pytest.mark.parametrize("shape", LONGCTX_SHAPES)
def test_decode_longctx_quality(device, shape):
    """Long-context split-KV decode must hold the int8 quality bars vs fp32 oracle."""
    q, k, v = _qkv_decode(shape, device)
    out = superl8.attn_int8_decode(q, k, v)
    oracle = attention_fp32_oracle(q, k, v)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"decode_longctx {shape}")


@pytest.mark.correctness
@pytest.mark.parametrize("shape", LONGCTX_D256_SHAPES)
def test_decode_longctx_d256_quality(device, shape):
    """D=256 long-context must also hold the int8 quality bars."""
    q, k, v = _qkv_decode(shape, device)
    out = superl8.attn_int8_decode(q, k, v)
    oracle = attention_fp32_oracle(q, k, v)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"decode_longctx d256 {shape}")


@pytest.mark.correctness
def test_decode_longctx_deterministic(device):
    """Long-context decode must be bitwise-identical on repeated calls."""
    q, k, v = _qkv_decode((1, 1, 1, 32768, 128), device)
    r0 = superl8.attn_int8_decode(q, k, v)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_decode(q, k, v), r0)


@pytest.mark.correctness
def test_decode_longctx_cached_quality(device):
    """INT8 KV-cache decode for long context must hold the int8 accuracy bars."""
    b, hq, hkv, n, d = 1, 1, 1, 32768, 128
    q, k, v = _qkv_decode((b, hq, hkv, n, d), device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v)
    out = superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale)
    oracle = attention_fp32_oracle(q, k, v)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what="decode_longctx cached",
                        min_cos=0.998, max_rel_l1=0.03, min_sqnr_db=18.0)


@pytest.mark.correctness
def test_decode_longctx_cached_deterministic(device):
    """INT8 KV-cache decode for long context must be deterministic."""
    b, hq, hkv, n, d = 1, 1, 1, 32768, 128
    q, k, v = _qkv_decode((b, hq, hkv, n, d), device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v)
    r0 = superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale), r0)


@pytest.mark.correctness
def test_decode_longctx_matches_prefill(device):
    """At long context, decode must match the prefill kernel at M=1."""
    q, k, v = _qkv_decode((1, 4, 2, 16384, 128), device)
    out_dec = superl8.attn_int8_decode(q, k, v)
    out_pre = superl8.attn_int8_fwd(q, k, v)
    assert cos_sim(out_dec, out_pre) >= 0.999, f"cos {cos_sim(out_dec, out_pre):.5f}"


@pytest.mark.correctness
def test_decode_longctx_tiny_batch_full_coverage(device):
    """Batch=1, single-head GQA (H_kv=1, H_q=1) covers worst-case SM underfill."""
    b, hq, hkv, n, d = 1, 2, 1, 32768, 128
    q, k, v = _qkv_decode((b, hq, hkv, n, d), device)
    out = superl8.attn_int8_decode(q, k, v)
    oracle = attention_fp32_oracle(q, k, v)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"decode_longctx tiny_batch {b}x{hq}x{hkv} N{n} D{d}")


@pytest.mark.correctness
def test_decode_num_splits_identity(device):
    """Different num_splits values must produce the same result (split-KV is
    mathematically split-invariant via the LSE combine trick)."""
    q, k, v = _qkv_decode((1, 4, 2, 16384, 128), device)
    ref = superl8.attn_int8_decode(q, k, v)
    for ns in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        out = superl8.attn_int8_decode(q, k, v, num_splits=ns)
        assert cos_sim(out, ref) >= 0.99999, f"num_splits={ns} cos {cos_sim(out, ref):.8f}"


@pytest.mark.correctness
def test_decode_num_splits_deterministic(device):
    """Explicit num_splits must be bitwise-identical on repeated calls."""
    q, k, v = _qkv_decode((1, 1, 1, 32768, 128), device)
    r0 = superl8.attn_int8_decode(q, k, v, num_splits=128)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_decode(q, k, v, num_splits=128), r0)


@pytest.mark.correctness
def test_decode_num_splits_bounds(device):
    """Invalid num_splits must be rejected."""
    q, k, v = _qkv_decode((1, 1, 1, 1024, 128), device)
    with pytest.raises(RuntimeError, match="num_splits"):
        superl8.attn_int8_decode(q, k, v, num_splits=0)
    with pytest.raises(RuntimeError, match="num_splits"):
        superl8.attn_int8_decode(q, k, v, num_splits=257)


@pytest.mark.correctness
def test_paged_decode_longctx_quality(device):
    """Paged decode at long context must hold the int8 accuracy bars vs fp32 oracle."""
    hq, hkv, block_size, d = 4, 2, 16, 128
    context_lens = [8192, 4096]
    b = len(context_lens)
    n_max = max(context_lens)
    q = torch.randn(b, hq, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n_max, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n_max, d, device=device, dtype=torch.float16)

    blocks_per_seq = [(n + block_size - 1) // block_size for n in context_lens]
    max_blocks = max(blocks_per_seq)
    num_blocks = sum(blocks_per_seq)
    block_table = torch.zeros(b, max_blocks, dtype=torch.int32, device=device)
    k_cache = torch.zeros(num_blocks, hkv, block_size, d, dtype=torch.int8, device=device)
    k_scale = torch.ones(num_blocks, hkv, block_size, dtype=torch.float32, device=device)
    v_cache = torch.zeros(num_blocks, hkv, block_size, d, dtype=torch.int8, device=device)
    v_scale = torch.ones(num_blocks, hkv, block_size, dtype=torch.float32, device=device)
    cursor = 0
    for i, nb in enumerate(blocks_per_seq):
        block_table[i, :nb] = torch.arange(cursor, cursor + nb, dtype=torch.int32)
        cursor += nb

    for t in range(n_max):
        active = [bi for bi, n in enumerate(context_lens) if t < n]
        if not active:
            continue
        idx = torch.tensor(active, device=device)
        slots = [int(block_table[bi, t // block_size]) * block_size + (t % block_size)
                 for bi in active]
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device=device)
        superl8.quantize_kv_write_paged(
            k[idx, :, t, :].contiguous(), v[idx, :, t, :].contiguous(),
            k_cache, k_scale, v_cache, v_scale, slot_mapping,
        )

    cl = torch.tensor(context_lens, dtype=torch.int32, device=device)
    out = superl8.attn_paged_decode_cached(
        q, k_cache, k_scale, v_cache, v_scale, block_table, cl, block_size,
        max_context_len=n_max,
    )
    oracle = attention_fp32_paged_oracle(q, k, v, cl)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"paged_decode_longctx {context_lens}",
                        min_cos=0.998, max_rel_l1=0.03, min_sqnr_db=18.0)


@pytest.mark.correctness
def test_paged_decode_longctx_deterministic(device):
    """Paged decode at long context must be deterministic."""
    hq, hkv, block_size, d = 2, 2, 16, 64
    context_lens = [4096]
    n_max = context_lens[0]
    q = torch.randn(1, hq, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(1, hkv, n_max, d, device=device, dtype=torch.float16)
    v = torch.randn(1, hkv, n_max, d, device=device, dtype=torch.float16)

    nb = (n_max + block_size - 1) // block_size
    block_table = torch.zeros(1, nb, dtype=torch.int32, device=device)
    block_table[0] = torch.arange(nb, dtype=torch.int32)
    k_cache = torch.zeros(nb, hkv, block_size, d, dtype=torch.int8, device=device)
    k_scale = torch.ones(nb, hkv, block_size, dtype=torch.float32, device=device)
    v_cache = torch.zeros(nb, hkv, block_size, d, dtype=torch.int8, device=device)
    v_scale = torch.ones(nb, hkv, block_size, dtype=torch.float32, device=device)

    for t in range(n_max):
        slot = int(block_table[0, t // block_size]) * block_size + (t % block_size)
        slot_mapping = torch.tensor([slot], dtype=torch.int32, device=device)
        superl8.quantize_kv_write_paged(
            k[:, :, t, :].contiguous(), v[:, :, t, :].contiguous(),
            k_cache, k_scale, v_cache, v_scale, slot_mapping,
        )

    cl = torch.tensor(context_lens, dtype=torch.int32, device=device)
    r0 = superl8.attn_paged_decode_cached(
        q, k_cache, k_scale, v_cache, v_scale, block_table, cl, block_size,
        max_context_len=n_max,
    )
    assert_finite(r0)
    for _ in range(3):
        r = superl8.attn_paged_decode_cached(
            q, k_cache, k_scale, v_cache, v_scale, block_table, cl, block_size,
            max_context_len=n_max,
        )
        assert torch.equal(r, r0)


@pytest.mark.correctness
def test_paged_decode_longctx_num_splits(device):
    """Paged decode with explicit num_splits must match auto."""
    hq, hkv, block_size, d = 2, 1, 16, 128
    context_lens = [8192]
    n_max = context_lens[0]
    q = torch.randn(1, hq, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(1, hkv, n_max, d, device=device, dtype=torch.float16)
    v = torch.randn(1, hkv, n_max, d, device=device, dtype=torch.float16)

    nb = (n_max + block_size - 1) // block_size
    block_table = torch.zeros(1, nb, dtype=torch.int32, device=device)
    block_table[0] = torch.arange(nb, dtype=torch.int32)
    k_cache = torch.zeros(nb, hkv, block_size, d, dtype=torch.int8, device=device)
    k_scale = torch.ones(nb, hkv, block_size, dtype=torch.float32, device=device)
    v_cache = torch.zeros(nb, hkv, block_size, d, dtype=torch.int8, device=device)
    v_scale = torch.ones(nb, hkv, block_size, dtype=torch.float32, device=device)

    for t in range(n_max):
        slot = int(block_table[0, t // block_size]) * block_size + (t % block_size)
        slot_mapping = torch.tensor([slot], dtype=torch.int32, device=device)
        superl8.quantize_kv_write_paged(
            k[:, :, t, :].contiguous(), v[:, :, t, :].contiguous(),
            k_cache, k_scale, v_cache, v_scale, slot_mapping,
        )

    cl = torch.tensor(context_lens, dtype=torch.int32, device=device)
    kwargs = dict(k_cache=k_cache, k_scale=k_scale, v_cache=v_cache, v_scale=v_scale,
                  block_table=block_table, context_lens=cl, block_size=block_size,
                  max_context_len=n_max)
    ref = superl8.attn_paged_decode_cached(q, **kwargs)
    for ns in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        out = superl8.attn_paged_decode_cached(q, **kwargs, num_splits=ns)
        assert cos_sim(out, ref) >= 0.9999, f"paged num_splits={ns} cos {cos_sim(out, ref):.8f}"


@pytest.mark.perf
def test_decode_longctx_speed_scaling(device):
    """Characterise long-context decode latency vs N at batch=1 (MQA worst-case).
    The int8 KV cache reads half the V bytes — at high N it must be faster than
    the fp16-V decode path on this memory-bound kernel."""
    b, hq, hkv, d = 1, 1, 1, 128
    print(f"\n long-context decode (B{b} Hq{hq} Hkv{hkv} D{d}):")
    print(f" {'N':>6} {'fp16V ms':>9} {'int8KV ms':>10}")
    prev_cached = None
    for n in (8192, 16384, 32768, 65536):
        q, k, v = _qkv_decode((b, hq, hkv, n, d), device)
        k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v)
        fp16v = time_ms(lambda: superl8.attn_int8_decode(q, k, v))
        cached = time_ms(lambda: superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale))
        print(f" {n:>6} {fp16v:>9.3f} {cached:>10.3f}")
        assert fp16v > 0 and cached > 0
        # At long context, the int8 KV cache reads half the V bytes and must win.
        assert cached < fp16v, f"N={n}: int8 KV ({cached:.3f}ms) must beat fp16V ({fp16v:.3f}ms)"
        if prev_cached is not None:
            assert cached > prev_cached, f"latency must increase with N ({prev_cached:.3f} -> {cached:.3f})"
        prev_cached = cached
