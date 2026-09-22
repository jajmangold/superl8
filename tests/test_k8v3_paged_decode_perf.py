# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Pinned-V100 fused K8V3 decode gates against dense and int8 paged paths."""
import math

import pytest
import torch

import superl8
from bench.harness import time_ms
from superl8.quant.lloydmax import dequantize_lloydmax, gaussian_codebook, unpack_indices_lowbit
from superl8.quant.rotation import rotate_last
from tests.test_k8v3_paged_decode import BLOCK_SIZE

pytestmark = [pytest.mark.perf]

D = 256
H_Q = 8
H_KV = 4
DENSE_CHUNK = 1024  # serve read_dense chunk (fni8-serve#431)
PERF_LENS = [2048, 8192, 32768]


def _choose_splits(n_len_max, bhq):
    """Mirror of attn_paged_decode.cu choose_n_splits (for reporting)."""
    s_occ = (2048 + bhq - 1) // bhq
    s_lo = max(1, (n_len_max + 256 - 1) // 256)
    s_hi = max(1, (n_len_max + 48 - 1) // 48)
    return max(1, min(max(s_occ, s_lo), s_hi, 256))


def _perf_cache(n, b, block_size, device):
    """Build a vectorized valid-shaped K8V3 store for timing only."""
    words = (D * 3) // 32
    n_blk = (n + block_size - 1) // block_size
    n_blocks = b * n_blk
    k_cache = torch.randint(-127, 128, (n_blocks, H_KV, block_size, D),
                            dtype=torch.int8, device=device)
    k_scale = 0.5 + torch.rand(n_blocks, H_KV, block_size, device=device)
    v_packed = torch.randint(-(2**31), 2**31 - 1,
                             (n_blocks, H_KV, block_size, words),
                             dtype=torch.int32, device=device)
    v_norm = 1.0 + torch.rand(n_blocks, H_KV, block_size, D // 128, device=device)
    v_codebook = gaussian_codebook(3, 128, device).contiguous()
    block_table = torch.arange(n_blk, dtype=torch.int32, device=device).expand(b, -1).clone()
    return k_cache, k_scale, v_packed, v_norm, v_codebook, block_table


def _dense_fallback(q, k_cache, k_scale, v_packed, v_norm, v_codebook,
                    block_table, context_lens, scale, device):
    """Mirror the serving fallback: chunked dense dequant then attention."""
    outs = []
    b = q.shape[0]
    h_kv = k_cache.shape[1]
    for i in range(b):
        cl = int(context_lens[i])
        nb = (cl + BLOCK_SIZE - 1) // BLOCK_SIZE
        blocks = [int(x) for x in block_table[i, :nb]]
        k_out = torch.zeros(1, h_kv, cl, D, dtype=torch.float16, device=device)
        v_out = torch.zeros(1, h_kv, cl, D, dtype=torch.float16, device=device)
        for lo in range(0, cl, DENSE_CHUNK):
            hi = min(lo + DENSE_CHUNK, cl)
            blk = torch.tensor([blocks[p // BLOCK_SIZE] for p in range(lo, hi)],
                               device=device, dtype=torch.long)
            off = torch.arange(lo, hi, device=device, dtype=torch.long) % BLOCK_SIZE
            k = (k_cache[blk, :, off, :].float()
                 * k_scale[blk, :, off].unsqueeze(-1)).half()
            k_out[0, :, lo:hi, :] = rotate_last(k).permute(1, 0, 2)
            packed = v_packed[blk, :, off, :]
            codes = unpack_indices_lowbit(packed, 3, D)
            v = dequantize_lloydmax(
                codes, v_norm[blk, :, off, :], v_codebook, block_size=128, dim=-1
            ).half()
            v_out[0, :, lo:hi, :] = v.permute(1, 0, 2)
        outs.append(superl8.attn_int8_decode(q[i : i + 1], k_out, v_out, scale=scale))
    return torch.cat(outs, dim=0)


@pytest.mark.perf
@pytest.mark.parametrize("n", PERF_LENS)
@pytest.mark.parametrize("b", [1, 4])
def test_k8v3_paged_decode_vs_dense_fallback(device, n, b):
    """Fused K8V3 decode must beat the dense fallback at long context, and the
    ratio is reported at every length (issue acceptance: >=15% at 32k)."""
    torch.manual_seed(11)
    q = torch.randn(b, H_Q, 1, D, device=device, dtype=torch.float16)
    k_cache, k_scale, v_packed, v_norm, v_codebook, block_table = _perf_cache(
        n, b, BLOCK_SIZE, device
    )
    context_lens = torch.tensor([n] * b, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(D)
    bhq = b * H_Q
    splits = _choose_splits(n, bhq)

    def fused():
        return superl8.attn_paged_decode_k8v3(
            q, k_cache, k_scale, v_packed, v_norm, v_codebook,
            block_table, context_lens, BLOCK_SIZE, scale=scale,
        )

    def dense():
        return _dense_fallback(q, k_cache, k_scale, v_packed, v_norm, v_codebook,
                               block_table, context_lens, scale, device)

    t_fused = time_ms(fused, warmup=5, iters=20)
    t_dense = time_ms(dense, warmup=3, iters=15)
    ratio = t_dense / t_fused
    print(
        f"\nK8V3 decode A/B n={n} B={b} splits={splits}: "
        f"fused {t_fused:.3f} ms ({1000/t_fused:.1f} tok/s), "
        f"dense {t_dense:.3f} ms ({1000/t_dense:.1f} tok/s), "
        f"speedup {ratio:.3f}x"
    )
    assert t_fused < t_dense, (
        f"fused decode ({t_fused:.3f} ms) not faster than dense fallback "
        f"({t_dense:.3f} ms) at n={n}"
    )


@pytest.mark.perf
def test_k8v3_paged_decode_short_ctx_vs_int8(device):
    """Short-context regression vs the int8 paged decode path must stay <=3%
    (the fused kernel shares the K/dp4a/softmax path; only the V source differs)."""
    n, b = 2048, 1
    torch.manual_seed(12)
    q = torch.randn(b, H_Q, 1, D, device=device, dtype=torch.float16)
    k_cache, k_scale, v_packed, v_norm, v_codebook, block_table = _perf_cache(
        n, b, BLOCK_SIZE, device
    )
    context_lens = torch.tensor([n], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(D)
    # Direct int8 V store (valid-shaped; the int8 write path is tested elsewhere).
    v_cache = torch.randint(-127, 128, k_cache.shape, dtype=torch.int8, device=device)
    v_scale = 0.5 + torch.rand(k_scale.shape, device=device)

    def fused():
        return superl8.attn_paged_decode_k8v3(
            q, k_cache, k_scale, v_packed, v_norm, v_codebook,
            block_table, context_lens, BLOCK_SIZE, scale=scale,
        )

    def int8_ref():
        return superl8.attn_paged_decode_cached(
            q, k_cache, k_scale, v_cache, v_scale,
            block_table, context_lens, BLOCK_SIZE, scale=scale, rotate=True,
        )

    t_fused = time_ms(fused, warmup=5, iters=30)
    t_int8 = time_ms(int8_ref, warmup=5, iters=30)
    ratio = t_fused / t_int8
    print(
        f"\nK8V3 short-ctx n={n} B={b}: fused {t_fused:.3f} ms "
        f"({1000/t_fused:.1f} tok/s), int8-paged {t_int8:.3f} ms "
        f"({1000/t_int8:.1f} tok/s), ratio {ratio:.3f}x"
    )
    assert ratio <= 1.03, (
        f"fused K8V3 decode is {ratio:.3f}x the int8 paged decode at n={n} "
        f"(>3% short-context regression)"
    )
