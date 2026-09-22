# SPDX-License-Identifier: BSD-3-Clause
"""Fused GGUF IQ1_S dp4a GEMV (superl8#317).

Gate the kernel against the fp32 reference dequant of the SAME bytes
(`superl8.quant.iq1s`, itself bit-exact vs gguf-py). Every 50-byte pattern is a valid
IQ1_S block, so random bytes cover an 11-bit index into a 2048-entry TERNARY grid, no sign bits, and a per-block
+/-0.125 delta folded into int8 codes as 8*g +/- 1
without needing a model file.
"""
import numpy as np
import pytest
import torch

import superl8
from superl8.quant.iq1s import BLOCK_BYTES, QK_K, dequantize_iq1_s_tensor

pytestmark = pytest.mark.correctness

CUDA = torch.cuda.is_available()


def _weights(n_out: int, in_features: int, seed: int = 0):
    """Random native IQ1_S bytes + their fp32 dequant."""
    rng = np.random.default_rng(seed)
    nsb = in_features // QK_K
    w = rng.integers(0, 256, size=(n_out, nsb * BLOCK_BYTES), dtype=np.uint8)
    d = rng.uniform(0.005, 0.05, size=(n_out, nsb)).astype(np.float16)
    for b in range(nsb):
        w[:, b * BLOCK_BYTES : b * BLOCK_BYTES + 2] = (
            d[:, b : b + 1].view(np.uint8).reshape(n_out, 2))
    return w, dequantize_iq1_s_tensor(w, in_features)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


@pytest.mark.skipif(not CUDA, reason="dp4a decode kernel needs a Volta GPU")
@pytest.mark.parametrize("M", [1, 2, 8, 16])
def test_decode_matches_the_reference(M):
    torch.manual_seed(M)
    N, K = 96, 512
    w_np, w_ref = _weights(N, K, seed=M)
    w = torch.from_numpy(w_np).cuda()
    x = (torch.randn(M, K, device="cuda") * 0.5).half()

    got = superl8.linear_iq1s(x, w).float()
    want = x.float() @ torch.from_numpy(w_ref).cuda().t()

    # int8 activation quantization is the only approximation the kernel adds.
    assert _cos(got, want) >= 0.999, f"cos={_cos(got, want)}"
    rel = (got - want).abs().sum() / (want.abs().sum() + 1e-12)
    assert rel <= 0.05, f"rel_l1={rel.item()}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("M,N,K", [(17, 64, 256), (32, 96, 512), (64, 64, 256), (100, 70, 768)])
def test_tile_matches_the_reference(M, N, K):
    """M > 16 routes to the blocked tile kernel (superl8#332). Ragged M/N (100, 70)
    deliberately do not divide GEMM_BM/GEMM_BN=64, so the out-of-range guards on both
    stages are exercised."""
    torch.manual_seed(M)
    w_np, w_ref = _weights(N, K, seed=M)
    x = (torch.randn(M, K, device="cuda") * 0.5).half()

    got = superl8.linear_iq1s(x, torch.from_numpy(w_np).cuda()).float()
    want = x.float() @ torch.from_numpy(w_ref).cuda().t()

    assert _cos(got, want) >= 0.9999, f"cos={_cos(got, want)}"
    rel = (got - want).abs().sum() / (want.abs().sum() + 1e-12)
    assert rel <= 0.02, f"rel_l1={rel.item()}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_tile_and_decode_agree_on_the_same_row():
    """The two kernels are one op to callers, so they must not disagree at the M=16/17
    boundary: same row through the decode path and the tile path, same answer.

    NOTE this FAILS while M > 16 still falls back to the fp32 CPU reference: that
    fallback is EXACT, so it differs from the int8-activation decode kernel by far more
    than 1e-3. That is deliberate -- it is the gate that goes green when the tile kernel
    is actually wired, rather than a test that passes either way."""
    torch.manual_seed(23)
    N, K = 128, 512
    w = torch.from_numpy(_weights(N, K, seed=13)[0]).cuda()
    x = (torch.randn(24, K, device="cuda") * 0.5).half()

    decode = superl8.linear_iq1s(x[:1], w).float()      # M=1  -> MMVQ
    tile = superl8.linear_iq1s(x, w).float()[:1]        # M=24 -> tile
    # Identical dp4a math on identical int8 rows; only accumulation ORDER differs.
    torch.testing.assert_close(tile, decode, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_decode_is_deterministic():
    torch.manual_seed(11)
    N, K = 128, 768
    w = torch.from_numpy(_weights(N, K, seed=5)[0]).cuda()
    x = (torch.randn(4, K, device="cuda") * 0.5).half()
    a = superl8.linear_iq1s(x, w)
    for _ in range(2):
        assert torch.equal(superl8.linear_iq1s(x, w), a)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_bias_is_applied():
    torch.manual_seed(3)
    N, K = 64, 256
    w = torch.from_numpy(_weights(N, K, seed=9)[0]).cuda()
    x = (torch.randn(2, K, device="cuda") * 0.5).half()
    bias = torch.randn(N, device="cuda")
    base = superl8.linear_iq1s(x, w).float()
    with_bias = superl8.linear_iq1s(x, w, bias=bias).float()
    torch.testing.assert_close(with_bias, base + bias, rtol=2e-2, atol=2e-2)
