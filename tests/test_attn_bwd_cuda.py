# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR5b: fused CUDA int8 dp4a backward kernel — tests written FIRST.

Contract:
  superl8.backward_cuda(q, k, v, out, lse, d_out, *, causal, scale) -> (dq, dk, dv)
    a fused FlashAttention-2 backward (recompute S/P in smem tiles, never
    materialize [M,N]); dQ pass is atomic-free (Q-outer), dK/dV pass is KV-outer
    (ai-bond grid.y split). Matmul precision (int8-dp4a vs fp-on-CUDA-core) is
    decided per-matmul by the accuracy gate; dO.V^T is the sensitive one.
Grades: the CUDA backward must match PR5a's reference backward tightly and the
fp32 oracle within relative bars (grads are noisier under int8).
"""
import pytest
import torch

import superl8
from superl8.autograd import _attn_backward
from tests.reference import attention_fp32_oracle
from tests.tolerances import cos_sim, rel_l1, sqnr_db

SHAPES = [(1, 2, 128, 64), (2, 4, 256, 64), (1, 2, 257, 64), (1, 2, 192, 128)]
GRAD_BARS = dict(min_cos=0.99, max_rel_l1=0.06, min_sqnr_db=14.0)


def _fwd_and_dout(shape, device, causal=False):
    q, k, v = (torch.randn(shape, device=device, dtype=torch.float16) for _ in range(3))
    out = superl8.attn_int8_fwd(q, k, v, causal=causal)  # O must match the bwd's causal flag
    d_out = torch.randn_like(out)
    return q, k, v, out, d_out


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_backward_cuda_matches_oracle(device, shape, causal):
    q, k, v, out, d_out = _fwd_and_dout(shape, device, causal=causal)
    dq, dk, dv = superl8.backward_cuda(q, k, v, out, None, d_out, causal=causal, scale=None)

    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    ref = attention_fp32_oracle(qf, kf, vf, causal=causal)
    ref.backward(d_out.float())

    for name, got, exp in [("dq", dq, qf.grad), ("dk", dk, kf.grad), ("dv", dv, vf.grad)]:
        assert got.shape == exp.shape and got.dtype == torch.float16
        c, l1, s = cos_sim(got, exp), rel_l1(got, exp), sqnr_db(got, exp)
        assert c >= GRAD_BARS["min_cos"], f"{name} {shape} c={causal}: cos {c:.4f}"
        assert l1 <= GRAD_BARS["max_rel_l1"], f"{name} {shape} c={causal}: rel-L1 {l1:.4f}"
        assert s >= GRAD_BARS["min_sqnr_db"], f"{name} {shape} c={causal}: SQNR {s:.1f}dB"


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_backward_cuda_matches_reference_backward(device, shape):
    """CUDA and PR5a PyTorch backward run the same math -> should agree closely."""
    q, k, v, out, d_out = _fwd_and_dout(shape, device)
    dq_c, dk_c, dv_c = superl8.backward_cuda(q, k, v, out, None, d_out, causal=False, scale=None)
    dq_r, dk_r, dv_r = _attn_backward(q, k, v, out, d_out, False, 1.0 / (q.shape[-1] ** 0.5))
    for got, exp in [(dq_c, dq_r), (dk_c, dk_r), (dv_c, dv_r)]:
        assert cos_sim(got, exp) >= 0.995  # same math, only int8-requant differs


@pytest.mark.perf
@pytest.mark.parametrize("shape", [(2, 16, 2048, 64), (2, 16, 2048, 128)])
def test_backward_cuda_perf(device, shape):
    """Honest report vs the PyTorch (cuBLAS) backward + regression gate. PR5b-1
    is fp-scalar (no dp4a) so it is EXPECTED to trail cuBLAS here; the win is
    memory (fused, no [M,N] materialization). dp4a speedup lands in PR5b-2."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench.harness import assert_no_regression, compare_report, time_ms

    b, h, m, d = shape
    q, k, v, out, d_out = _fwd_and_dout(shape, device)
    ours = time_ms(lambda: superl8.backward_cuda(q, k, v, out, None, d_out))
    torch_bwd = time_ms(lambda: _attn_backward(q, k, v, out, d_out, False, 1.0 / (d ** 0.5)))
    name = f"attn_bwd.b{b}h{h}m{m}d{d}"
    print("\n" + compare_report(name, ours, {"pytorch_bwd": torch_bwd}))
    assert_no_regression(name, ours)


@pytest.mark.correctness
def test_backward_cuda_deterministic(device):
    q, k, v, out, d_out = _fwd_and_dout((2, 4, 256, 64), device)
    r0 = superl8.backward_cuda(q, k, v, out, None, d_out, causal=False, scale=None)
    for _ in range(3):
        r = superl8.backward_cuda(q, k, v, out, None, d_out, causal=False, scale=None)
        for a, b in zip(r, r0):
            assert torch.equal(a, b)  # atomic-free -> bitwise reproducible
