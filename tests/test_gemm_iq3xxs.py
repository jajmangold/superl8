# SPDX-License-Identifier: BSD-3-Clause
"""Fused GGUF IQ3_XXS dp4a GEMV (superl8#317 work item 2d).

Gate the kernel against the fp32 reference dequant of the SAME bytes (`superl8.quant.iq3xxs`,
itself bit-exact vs gguf-py). Random 98-byte blocks cover the codebook, both nibble
halves and the split 6-bit scales without a model file.
"""
import numpy as np
import pytest
import torch

import superl8
from superl8.quant.iq3xxs import BLOCK_BYTES, QK_K, dequantize_iq3_xxs_tensor

pytestmark = pytest.mark.correctness

CUDA = torch.cuda.is_available()


def _weights(n_out: int, in_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    nsb = in_features // QK_K
    w = rng.integers(0, 256, size=(n_out, nsb * BLOCK_BYTES), dtype=np.uint8)
    d = rng.uniform(0.005, 0.05, size=(n_out, nsb)).astype(np.float16)
    for b in range(nsb):
        w[:, b * BLOCK_BYTES : b * BLOCK_BYTES + 2] = d[:, b : b + 1].view(np.uint8).reshape(n_out, 2)
    return w, dequantize_iq3_xxs_tensor(w, in_features)


def _cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


@pytest.mark.skipif(not CUDA, reason="dp4a decode kernel needs a Volta GPU")
@pytest.mark.parametrize("M", [1, 2, 8, 16])
def test_decode_matches_the_reference(M):
    torch.manual_seed(M)
    N, K = 96, 512
    w_np, w_ref = _weights(N, K, seed=M)
    x = (torch.randn(M, K, device="cuda") * 0.5).half()
    got = superl8.linear_iq3xxs(x, torch.from_numpy(w_np).cuda()).float()
    want = x.float() @ torch.from_numpy(w_ref).cuda().t()
    assert _cos(got, want) >= 0.999, f"cos={_cos(got, want)}"
    rel = (got - want).abs().sum() / (want.abs().sum() + 1e-12)
    assert rel <= 0.05, f"rel_l1={rel.item()}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_sign_lut_and_scale_are_applied():
    """The sign LUT and the top-nibble half-offset scale are the two places IQ3_XXS can
    silently diverge; compare elementwise on a sparse row, not just by cosine."""
    torch.manual_seed(5)
    N, K = 64, 256
    w_np, w_ref = _weights(N, K, seed=21)
    x = torch.zeros(1, K, device="cuda", dtype=torch.float16)
    x[0, 0] = 1.0        # picks out column 0 of every row = weight 0 of each sub-block
    x[0, 16] = 2.0       # weight 16 — a different grid entry in the same sub-block
    got = superl8.linear_iq3xxs(x, torch.from_numpy(w_np).cuda()).float()
    want = x.float() @ torch.from_numpy(w_ref).cuda().t()
    torch.testing.assert_close(got, want, rtol=0.05, atol=0.5)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("M,N,K", [(17, 64, 256), (32, 96, 512), (64, 64, 256), (100, 70, 768)])
def test_tile_matches_the_reference(M, N, K):
    """M > 16 routes to the blocked tile kernel. Ragged M/N (100, 70) deliberately do not
    divide GEMM_BM/GEMM_BN=64, so the out-of-range guards on both stages are exercised."""
    torch.manual_seed(M)
    w_np, w_ref = _weights(N, K, seed=M)
    x = (torch.randn(M, K, device="cuda") * 0.5).half()

    got = superl8.linear_iq3xxs(x, torch.from_numpy(w_np).cuda()).float()
    want = x.float() @ torch.from_numpy(w_ref).cuda().t()

    assert _cos(got, want) >= 0.999, f"cos={_cos(got, want)}"
    rel = (got - want).abs().sum() / (want.abs().sum() + 1e-12)
    assert rel <= 0.05, f"rel_l1={rel.item()}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_tile_and_decode_agree_on_the_same_row():
    """The two kernels are one op to callers, so they must not disagree at the M=16/17
    boundary: same row through the decode path and the tile path, same answer."""
    torch.manual_seed(23)
    N, K = 128, 512
    w = torch.from_numpy(_weights(N, K, seed=13)[0]).cuda()
    x = (torch.randn(24, K, device="cuda") * 0.5).half()

    decode = superl8.linear_iq3xxs(x[:1], w).float()      # M=1  -> MMVQ
    tile = superl8.linear_iq3xxs(x, w).float()[:1]        # M=24 -> tile
    torch.testing.assert_close(tile, decode, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_decode_is_deterministic():
    torch.manual_seed(11)
    N, K = 128, 768
    w = torch.from_numpy(_weights(N, K, seed=5)[0]).cuda()
    x = (torch.randn(4, K, device="cuda") * 0.5).half()
    a = superl8.linear_iq3xxs(x, w)
    for _ in range(2):
        assert torch.equal(superl8.linear_iq3xxs(x, w), a)
