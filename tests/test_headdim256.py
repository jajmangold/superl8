# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Head dim 256 forward — unblocks Gemma-family (head_dim 256) and some video
DiTs. D=256 exceeds the 48KB static-smem cap, so this path uses dynamic shared
memory (opted in via cudaFuncSetAttribute, like the backward). Tests first.

Diffusion attention is non-causal (bidirectional denoising), so the non-causal
path is the one that matters here; causal is covered too for completeness.
"""
import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle
from tests.tolerances import assert_finite, assert_int8_quality

# (B, H_q, H_kv, M, N), D=256. Square shapes take causal+non-causal (causal =
# self-attention, M==N — the dense forward's contract). M!=N shapes are the
# cross-attention case (DiT image-Q / text-KV), which is non-causal.
D256_SQUARE = [
    (1, 4, 4, 128, 128),      # MHA square
    (2, 8, 2, 256, 256),      # GQA group 4
    (1, 8, 2, 320, 320),      # GQA, non-tile-multiple
]
D256_CROSS = [
    (1, 4, 1, 333, 200),      # MQA, ragged, M != N
    (1, 2, 2, 64, 512),       # short Q, long K (image Q attends text K)
]


@pytest.mark.correctness
@pytest.mark.parametrize("shape", D256_SQUARE)
@pytest.mark.parametrize("causal", [False, True])
def test_d256_selfattn(device, shape, causal):
    b, hq, hkv, m, n = shape
    q = torch.randn(b, hq, m, 256, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n, 256, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, 256, device=device, dtype=torch.float16)
    out = superl8.attn_int8_fwd(q, k, v, causal=causal)
    oracle = attention_fp32_oracle(q, k, v, causal=causal)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"d256 self {shape} causal={causal}")


@pytest.mark.correctness
@pytest.mark.parametrize("shape", D256_CROSS)
def test_d256_crossattn(device, shape):
    """Non-causal M!=N — the DiT cross-attention shape diffusion needs."""
    b, hq, hkv, m, n = shape
    q = torch.randn(b, hq, m, 256, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n, 256, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, 256, device=device, dtype=torch.float16)
    out = superl8.attn_int8_fwd(q, k, v, causal=False)
    oracle = attention_fp32_oracle(q, k, v, causal=False)
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"d256 cross {shape}")


@pytest.mark.correctness
def test_d256_bf16_v_and_output(device):
    """D=256 uses the dynamic-smem kernel (a separate code path from D<=128) —
    the bf16 V/out template must cover it too (Gemma's head_dim is 256)."""
    b, hq, hkv, m = 1, 4, 4, 128
    q = torch.randn(b, hq, m, 256, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, m, 256, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, m, 256, device=device, dtype=torch.bfloat16)
    out = superl8.attn_int8_fwd(q, k, v, causal=False)
    oracle = attention_fp32_oracle(q, k, v, causal=False)
    assert out.dtype == torch.bfloat16
    assert_finite(out)
    assert_int8_quality(out, oracle, what="d256 bf16-V")


@pytest.mark.correctness
def test_d256_deterministic(device):
    q, k, v = (torch.randn(1, 4, 256, 256, device=device, dtype=torch.float16) for _ in range(3))
    r0 = superl8.attn_int8_fwd(q, k, v)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_fwd(q, k, v), r0)


@pytest.mark.perf
def test_d256_perf(device):
    """D=256 is correctness-first (dynamic smem, lane-pair -> low occupancy at
    COLS=128); measure honestly vs SDPA. A TPR treatment is the perf follow-up."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench.harness import time_ms
    from tests.reference import sdpa_fp16

    q, k, v = (torch.randn(2, 8, 1024, 256, device=device, dtype=torch.float16) for _ in range(3))
    ours = time_ms(lambda: superl8.attn_int8_fwd(q, k, v))
    sdpa = time_ms(lambda: sdpa_fp16(q, k, v))
    print(f"\nD=256 superl8 {ours:.3f} ms | sdpa_fp16 {sdpa:.3f} ms | {sdpa / ours:.2f}x")
    assert ours == ours and ours > 0  # runs, finite; perf is a documented follow-up
