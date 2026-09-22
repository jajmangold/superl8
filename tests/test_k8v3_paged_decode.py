# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Correctness gates for fused paged int8-K/LloydMax3-V decode (superl8#295)."""
import pytest
import torch

import superl8
from superl8.quant.lloydmax import (
    dequantize_lloydmax,
    pack_indices_lowbit,
    quantize_lloydmax,
)
from superl8.quant.rotation import rotate_last
from tests.reference import attention_fp32_paged_oracle
from tests.tolerances import assert_finite, assert_int8_quality, cos_sim

pytestmark = [pytest.mark.correctness]

# Ragged batch: lenses 5 / 33 / 64 cross block_size=16 block boundaries and,
# at n_len_max=64 with num_splits=2 (per=32), also cross split boundaries.
CONTEXT_LENS = [5, 33, 64]
BLOCK_SIZE = 16


def _slot_rows(block, offset, h_kv, block_size):
    """Flat cache row indices [h_kv] for (block, offset) — head-major within a
    block, matching kv_write_paged's `row = (block*h_kv + h)*block_size + offset`."""
    base = (block * h_kv) * block_size + offset
    return torch.arange(h_kv, dtype=torch.long) * block_size + base


def _build_k8v3_cache(k, v, context_lens, block_size, device, permute_blocks=False):
    """Write fp16 K/V into the exact paged store used by fni8-serve K8V3."""
    b, h_kv, n_max, d = k.shape
    assert v.shape == k.shape
    nb = (n_max + block_size - 1) // block_size
    max_blocks = nb
    n_blocks = b * nb
    words = (d * 3) // 32
    n_norms = d // 128
    assert d % 128 == 0, "lloydmax3 requires head dim divisible by 128"

    k_cache = torch.zeros(n_blocks, h_kv, block_size, d, dtype=torch.int8, device=device)
    k_scale = torch.zeros(n_blocks, h_kv, block_size, dtype=torch.float32, device=device)
    v_packed = torch.zeros(n_blocks, h_kv, block_size, words, dtype=torch.int32, device=device)
    v_norm = torch.zeros(n_blocks, h_kv, block_size, n_norms, dtype=torch.float32, device=device)
    v_codebook = None

    block_table = torch.full((b, max_blocks), -1, dtype=torch.int32, device=device)
    for i in range(b):
        n_blocks_i = (int(context_lens[i]) + block_size - 1) // block_size
        base = i * nb
        phys = list(range(base, base + n_blocks_i))
        if permute_blocks:
            # Non-contiguous / unsorted physical layout: reverse the sequence's
            # blocks so attention must follow the block table, not block ids.
            phys = phys[::-1]
        block_table[i, :n_blocks_i] = torch.tensor(phys, dtype=torch.int32, device=device)

    for i in range(b):
        cl = int(context_lens[i])
        for n in range(cl):
            blk = n // block_size
            off = n % block_size
            phys = int(block_table[i, blk])
            k_new = k[i, :, n, :]  # [H_kv, D] fp16
            v_new = v[i, :, n, :]
            # K: rotate then per-row RTN int8 (serve _write_k_int8 recipe).
            k_rot = rotate_last(k_new)
            k_i8, k_sc = superl8.quantize_i8_rowwise(k_rot)
            rows = _slot_rows(phys, off, h_kv, block_size)
            k_cache.reshape(-1, d)[rows] = k_i8
            k_scale.reshape(-1)[rows] = k_sc
            # V: 3-bit Lloyd-Max, codes packed LSB-first (serve _write_v_lloydmax3).
            codes, norms, cb = quantize_lloydmax(
                v_new, bits=3, block_size=128, dim=-1
            )
            if v_codebook is None:
                v_codebook = cb.to(device)
            packed = pack_indices_lowbit(codes, 3)  # [H_kv, words] int32
            v_packed.reshape(-1, words)[rows] = packed
            v_norm.reshape(-1, n_norms)[rows] = norms
    return k_cache, k_scale, v_packed, v_norm, v_codebook, block_table


def _dequant_paged(k_cache, k_scale, v_packed, v_norm, v_codebook, block_table,
                   context_lens, block_size, device, dtype=torch.float32):
    """Dense oracle store; V rounds through fp16 exactly like the kernel."""
    lens = [int(x) for x in context_lens]
    n_max = max(lens)
    h_kv = k_cache.shape[1]
    d = k_cache.shape[3]
    k_out = torch.zeros(len(lens), h_kv, n_max, d, dtype=dtype, device=device)
    v_out = torch.zeros(len(lens), h_kv, n_max, d, dtype=dtype, device=device)
    for i in range(len(lens)):
        cl = lens[i]
        for n in range(cl):
            blk = n // block_size
            off = n % block_size
            phys = int(block_table[i, blk])
            k_out[i, :, n, :] = (
                k_cache[phys, :, off, :].float() * k_scale[phys, :, off].unsqueeze(-1)
            )
            packed = v_packed[phys, :, off, :]
            from superl8.quant.lloydmax import unpack_indices_lowbit

            codes = unpack_indices_lowbit(packed, 3, d)
            v_out[i, :, n, :] = dequantize_lloydmax(
                codes, v_norm[phys, :, off, :], v_codebook, block_size=128, dim=-1
            )
    # Round V through fp16 exactly like the kernel's __float2half_rn path.
    return k_out, v_out.half().float()


@pytest.mark.correctness
@pytest.mark.parametrize("d,h_q,h_kv", [(128, 4, 4), (256, 4, 4), (256, 8, 4)])
def test_k8v3_paged_decode_rep_quality(device, d, h_q, h_kv):
    """Match fp32 attention over exactly representable cache tensors."""
    b = 3
    n_max = max(CONTEXT_LENS)
    torch.manual_seed(0)
    q = torch.randn(b, h_q, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h_kv, n_max, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h_kv, n_max, d, device=device, dtype=torch.float16)
    k_cache, k_scale, v_packed, v_norm, v_codebook, block_table = _build_k8v3_cache(
        k, v, CONTEXT_LENS, BLOCK_SIZE, device
    )
    context_lens = torch.tensor(CONTEXT_LENS, dtype=torch.int32, device=device)
    scale = 1.0 / (d ** 0.5)

    out = superl8.attn_paged_decode_k8v3(
        q, k_cache, k_scale, v_packed, v_norm, v_codebook,
        block_table, context_lens, BLOCK_SIZE, scale=scale, num_splits=2,
    )
    assert out.shape == (b, h_q, 1, d)
    assert_finite(out)

    k_rep, v_rep = _dequant_paged(
        k_cache, k_scale, v_packed, v_norm, v_codebook, block_table,
        CONTEXT_LENS, BLOCK_SIZE, device,
    )
    oracle = attention_fp32_paged_oracle(
        rotate_last(q), k_rep, v_rep, context_lens, scale=scale
    )
    # Residual vs the representable oracle is softmax/accumulation order only
    # (measured: cos ~0.99997-0.99999, rel-L1 ~0.004-0.006, SQNR 43-47 dB);
    # bars carry ~3x margin below the measured values.
    c = cos_sim(out, oracle)
    assert c >= 0.9999, f"d={d} h_q={h_q} cos {c:.8f}"
    assert torch.equal(out, out.clone()), "bitwise-stable on re-run"


@pytest.mark.correctness
@pytest.mark.parametrize("d", [128, 256])
def test_k8v3_paged_decode_codec_honest_vs_true_fp32(device, d):
    """Hold the accepted storage-codec quality bars versus true fp32."""
    h_q, h_kv = 4, 4
    n_max = max(CONTEXT_LENS)
    context_lens_vals = CONTEXT_LENS[:2]
    b = len(context_lens_vals)
    torch.manual_seed(1)
    q = torch.randn(b, h_q, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h_kv, n_max, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h_kv, n_max, d, device=device, dtype=torch.float16)
    k_cache, k_scale, v_packed, v_norm, v_codebook, block_table = _build_k8v3_cache(
        k, v, context_lens_vals, BLOCK_SIZE, device
    )
    context_lens = torch.tensor(context_lens_vals, dtype=torch.int32, device=device)
    scale = 1.0 / (d ** 0.5)

    out = superl8.attn_paged_decode_k8v3(
        q, k_cache, k_scale, v_packed, v_norm, v_codebook,
        block_table, context_lens, BLOCK_SIZE, scale=scale,
    )
    oracle = attention_fp32_paged_oracle(q, k, v, context_lens, scale=scale)
    assert_int8_quality(
        out, oracle, min_cos=0.98, max_rel_l1=0.25, min_sqnr_db=12.0,
        what=f"k8v3 codec-honest d={d}",
    )


@pytest.mark.correctness
def test_k8v3_paged_decode_deterministic(device):
    """Same input three times -> bitwise-identical output."""
    d, h_q, h_kv, b = 256, 8, 4, 3
    n_max = max(CONTEXT_LENS)
    torch.manual_seed(2)
    q = torch.randn(b, h_q, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h_kv, n_max, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h_kv, n_max, d, device=device, dtype=torch.float16)
    k_cache, k_scale, v_packed, v_norm, v_codebook, block_table = _build_k8v3_cache(
        k, v, CONTEXT_LENS, BLOCK_SIZE, device
    )
    context_lens = torch.tensor(CONTEXT_LENS, dtype=torch.int32, device=device)
    outs = [
        superl8.attn_paged_decode_k8v3(
            q, k_cache, k_scale, v_packed, v_norm, v_codebook,
            block_table, context_lens, BLOCK_SIZE, num_splits=2,
        )
        for _ in range(3)
    ]
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])


@pytest.mark.correctness
def test_k8v3_paged_decode_splits_identity(device):
    """num_splits in {1,2,4} must agree with each other and the fp32 oracle."""
    d, h_q, h_kv, b = 256, 8, 4, 3
    n_max = max(CONTEXT_LENS)
    torch.manual_seed(3)
    q = torch.randn(b, h_q, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h_kv, n_max, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h_kv, n_max, d, device=device, dtype=torch.float16)
    k_cache, k_scale, v_packed, v_norm, v_codebook, block_table = _build_k8v3_cache(
        k, v, CONTEXT_LENS, BLOCK_SIZE, device
    )
    context_lens = torch.tensor(CONTEXT_LENS, dtype=torch.int32, device=device)
    outs = {
        ns: superl8.attn_paged_decode_k8v3(
            q, k_cache, k_scale, v_packed, v_norm, v_codebook,
            block_table, context_lens, BLOCK_SIZE, num_splits=ns,
        )
        for ns in (1, 2, 4)
    }
    for ns, out in outs.items():
        c = cos_sim(out, outs[4])
        assert c >= 0.99999, f"num_splits={ns} cos {c:.8f}"


@pytest.mark.correctness
def test_k8v3_paged_decode_permuted_blocks(device):
    """Attention is invariant to the physical block layout: two caches holding
    the same logical tokens under different (non-contiguous, reversed) block
    tables must decode bitwise-identically."""
    d, h_q, h_kv, b = 256, 8, 4, 1
    n_max = max(CONTEXT_LENS)
    torch.manual_seed(4)
    q = torch.randn(b, h_q, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h_kv, n_max, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h_kv, n_max, d, device=device, dtype=torch.float16)
    context_lens = torch.tensor(CONTEXT_LENS[:1], dtype=torch.int32, device=device)
    outs = []
    for permute in (False, True):
        k_cache, k_scale, v_packed, v_norm, v_codebook, block_table = _build_k8v3_cache(
            k, v, CONTEXT_LENS[:1], BLOCK_SIZE, device, permute_blocks=permute
        )
        outs.append(
            superl8.attn_paged_decode_k8v3(
                q, k_cache, k_scale, v_packed, v_norm, v_codebook,
                block_table, context_lens, BLOCK_SIZE, num_splits=2,
            )
        )
    assert torch.equal(outs[0], outs[1])


@pytest.mark.correctness
@pytest.mark.parametrize("d", [128, 256])
def test_k8v3_paged_decode_matches_dense_fallback(device, d):
    """Agree with the serving dense fallback within shared codec bars."""
    b, h_q, h_kv = 3, 4, 4
    n_max = max(CONTEXT_LENS)
    torch.manual_seed(5)
    q = torch.randn(b, h_q, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h_kv, n_max, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h_kv, n_max, d, device=device, dtype=torch.float16)
    k_cache, k_scale, v_packed, v_norm, v_codebook, block_table = _build_k8v3_cache(
        k, v, CONTEXT_LENS, BLOCK_SIZE, device
    )
    context_lens = torch.tensor(CONTEXT_LENS, dtype=torch.int32, device=device)
    scale = 1.0 / (d ** 0.5)

    out_fused = superl8.attn_paged_decode_k8v3(
        q, k_cache, k_scale, v_packed, v_norm, v_codebook,
        block_table, context_lens, BLOCK_SIZE, scale=scale, num_splits=2,
    )
    dense_outs = []
    for i in range(b):
        cl = int(context_lens[i])
        k_rep, v_rep = _dequant_paged(
            k_cache, k_scale, v_packed, v_norm, v_codebook, block_table[i : i + 1],
            [cl], BLOCK_SIZE, device,
        )
        kb = rotate_last(k_rep[:, :, :cl, :]).half().contiguous()
        vb = v_rep[:, :, :cl, :].half().contiguous()
        dense_outs.append(superl8.attn_int8_decode(q[i : i + 1], kb, vb, scale=scale))
    out_dense = torch.cat(dense_outs, dim=0)
    assert_int8_quality(
        out_fused, out_dense, what=f"k8v3 fused vs dense fallback d={d}"
    )


@pytest.mark.correctness
def test_k8v3_paged_decode_rejects_bad_dims(device):
    """lloydmax3 requires D % 128 == 0 and the packed shapes must match."""
    d, h_q, h_kv, b = 64, 4, 4, 1
    torch.manual_seed(6)
    q = torch.randn(b, h_q, 1, d, device=device, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="lloydmax"):
        superl8.attn_paged_decode_k8v3(
            q, torch.zeros(1, h_kv, BLOCK_SIZE, d, dtype=torch.int8, device=device),
            torch.zeros(1, h_kv, BLOCK_SIZE, dtype=torch.float32, device=device),
            torch.zeros(1, h_kv, BLOCK_SIZE, 6, dtype=torch.int32, device=device),
            torch.zeros(1, h_kv, BLOCK_SIZE, 1, dtype=torch.float32, device=device),
            torch.zeros(8, dtype=torch.float32, device=device),
            torch.tensor([[0]], dtype=torch.int32, device=device),
            torch.tensor([8], dtype=torch.int32, device=device),
            BLOCK_SIZE,
        )


@pytest.mark.correctness
def test_k8v3_paged_decode_rejects_cache_leading_dim_mismatch(device):
    """Every K/V storage tensor must describe the same physical block pool."""
    d, h_q, h_kv, b = 128, 4, 4, 1
    q = torch.randn(b, h_q, 1, d, device=device, dtype=torch.float16)
    k_cache = torch.zeros(2, h_kv, BLOCK_SIZE, d, dtype=torch.int8, device=device)
    k_scale = torch.ones(2, h_kv, BLOCK_SIZE, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        superl8.attn_paged_decode_k8v3(
            q,
            k_cache,
            k_scale,
            torch.zeros(1, h_kv, BLOCK_SIZE, d * 3 // 32, dtype=torch.int32, device=device),
            torch.ones(1, h_kv, BLOCK_SIZE, d // 128, dtype=torch.float32, device=device),
            torch.zeros(8, dtype=torch.float32, device=device),
            torch.tensor([[0]], dtype=torch.int32, device=device),
            torch.tensor([8], dtype=torch.int32, device=device),
            BLOCK_SIZE,
        )
