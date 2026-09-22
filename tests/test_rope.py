# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Fused RoPE — one launch per tensor replaces the eager gather/unsqueeze/cast +
`_rotate_half` (chunk/neg/**cat**) + mul/mul/add chain applied to q and k every
layer (the `CatArray` x56 + index-gather cluster in the decode profile).

For each rotary pair (j, j+half) with half = rotary_dim/2:
    out[j]      = x[j]*cos[j]      - x[j+half]*sin[j]
    out[j+half] = x[j+half]*cos[j] + x[j]*sin[j]
dims >= rotary_dim pass through unchanged (partial rotary: GLM, etc.).

Gate: fp16/bf16 out; cos/sin computed in fp32 internally (more accurate than the
reference's fp16-cast table) so cos ≈ 1 / low rel-L1 vs the reference, not
bit-exact.
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import time_ms  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DTYPES = [torch.float16, torch.bfloat16]


def _tables(max_pos, rotary_dim, base=1e6):
    inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().cuda(), emb.sin().cuda()          # [max_pos, rotary_dim] fp32


def _ref(positions, x, cos, sin, rotary_dim):
    """== superl8serve RotaryEmbedding._rotate (superl8serve/layers/rotary.py)."""
    c = cos[positions].unsqueeze(-2).to(x.dtype)       # [...,1,rotary_dim]
    s = sin[positions].unsqueeze(-2).to(x.dtype)
    xr, xp = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = xr.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    out = xr * c + rot * s
    return torch.cat((out, xp), dim=-1) if xp.shape[-1] else out


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


@CUDA
@pytest.mark.correctness
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B,S,nh,nkv,hd,rd", [
    (1, 1, 16, 8, 64, 64),      # decode, full rotary (Qwen3-0.6B shape)
    (2, 5, 8, 2, 128, 128),     # prefill-ish
    (1, 1, 4, 4, 128, 64),      # partial rotary (rd < hd)
    (3, 1, 16, 8, 80, 80),      # non-pow2 head dim
    (1, 2, 4, 2, 72, 72),       # non-tile-multiple (36 pairs)
    (1, 1, 4, 2, 65, 64),       # partial rotary, odd head dim
])
def test_rope_matches_reference(B, S, nh, nkv, hd, rd, dtype):
    torch.manual_seed(0)
    max_pos = 128
    cos, sin = _tables(max_pos, rd)
    positions = torch.randint(0, max_pos, (B, S), device="cuda")
    q = torch.randn(B, S, nh, hd, device="cuda", dtype=dtype)
    k = torch.randn(B, S, nkv, hd, device="cuda", dtype=dtype)
    qo, ko = superl8.rope(positions, q.clone(), k.clone(), cos, sin, rd)
    # Gate (AGENTS.md, fused elementwise): cos ≈ 1 + low rel-L1 vs the fp32 oracle.
    # The kernel uses fp32 cos/sin so it's *closer* to the oracle than the serve
    # reference's own fp16/bf16-cast table; remaining error is just storage ULP.
    q_oracle = _ref(positions, q.float(), cos, sin, rd)
    k_oracle = _ref(positions, k.float(), cos, sin, rd)
    assert qo.shape == q.shape and ko.shape == k.shape
    assert _cos(qo, q_oracle) > 0.9999 and _cos(ko, k_oracle) > 0.9999
    def _rl1(a, b):
        return (a.float() - b).abs().sum().item() / b.abs().sum().clamp_min(1e-9).item()
    assert _rl1(qo, q_oracle) < 0.02 and _rl1(ko, k_oracle) < 0.02


@CUDA
@pytest.mark.correctness
def test_rope_deterministic():
    cos, sin = _tables(64, 64)
    positions = torch.randint(0, 64, (2, 3), device="cuda")
    q = torch.randn(2, 3, 16, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 3, 8, 64, device="cuda", dtype=torch.float16)
    outs = [superl8.rope(positions, q.clone(), k.clone(), cos, sin, 64) for _ in range(3)]
    for qo, ko in outs[1:]:
        assert torch.equal(qo, outs[0][0]) and torch.equal(ko, outs[0][1])


@CUDA
@pytest.mark.correctness
def test_rope_half2_deterministic():
    """Determinism x3: same fp16 input -> bitwise-equal output."""
    cos, sin = _tables(128, 80)
    positions = torch.randint(0, 128, (4, 7), device="cuda")
    q = torch.randn(4, 7, 16, 80, device="cuda", dtype=torch.float16)
    k = torch.randn(4, 7, 8, 80, device="cuda", dtype=torch.float16)
    outs = [superl8.rope(positions, q.clone(), k.clone(), cos, sin, 80) for _ in range(3)]
    for qo, ko in outs[1:]:
        assert torch.equal(qo, outs[0][0]) and torch.equal(ko, outs[0][1])


@CUDA
@pytest.mark.correctness
def test_rope_half2_non_tile_multiples():
    """fp16 with non-tile-multiple rotary dims (36 pairs, odd head_dim partial rotary)."""
    torch.manual_seed(42)
    for hd, rd in [(72, 72), (65, 64), (81, 80)]:
        cos, sin = _tables(64, rd)
        positions = torch.randint(0, 64, (3, 4), device="cuda")
        q = torch.randn(3, 4, 8, hd, device="cuda", dtype=torch.float16)
        k = torch.randn(3, 4, 4, hd, device="cuda", dtype=torch.float16)
        qo, ko = superl8.rope(positions, q.clone(), k.clone(), cos, sin, rd)
        q_ref = _ref(positions, q.float(), cos, sin, rd)
        k_ref = _ref(positions, k.float(), cos, sin, rd)
        assert qo.shape == q.shape and ko.shape == k.shape
        assert _cos(qo, q_ref) > 0.9999 and _cos(ko, k_ref) > 0.9999
        def _rl1(a, b):
            return (a.float() - b).abs().sum().item() / b.abs().sum().clamp_min(1e-9).item()
        assert _rl1(qo, q_ref) < 0.02 and _rl1(ko, k_ref) < 0.02


@CUDA
@pytest.mark.perf
def test_rope_faster_than_eager():
    cos, sin = _tables(128, 64)
    positions = torch.zeros(8, 1, dtype=torch.long, device="cuda")
    q = torch.randn(8, 1, 16, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(8, 1, 8, 64, device="cuda", dtype=torch.float16)
    fused = time_ms(lambda: superl8.rope(positions, q.clone(), k.clone(), cos, sin, 64))
    eager = time_ms(lambda: (_ref(positions, q, cos, sin, 64), _ref(positions, k, cos, sin, 64)))
    assert fused <= eager * 1.2, f"fused {fused:.4f} vs eager {eager:.4f}"
