# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""MoE grouped/batched int8 dp4a GEMM (`gemm_grouped_w8a8`) — every active
expert's GEMM for a token batch in ONE launch instead of a Python loop over
`gemm_w8a8` per expert (gather by expert, segment the M dimension).

Gates (AGENTS.md): int8 paths use SQNR / cos-sim / rel-L1, never allclose,
EXCEPT where compared against the kernel's own exact integer matmul (torch.equal
holds there, same as gemm_w8a8's existing tests).
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8
from superl8.quant.core import quantize_int8_rowwise

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, compare_report, time_ms  # noqa: E402
from tests.tolerances import assert_int8_quality  # noqa: E402


def _wq_stack(w: torch.Tensor):
    """[E,N,K] fp16 weights -> per-row int8 stack (w_i8 [E,N,K], w_scale [E,N])."""
    e, n, k = w.shape
    q, s = quantize_int8_rowwise(w.reshape(e * n, k))
    return q.reshape(e, n, k).contiguous(), s.reshape(e, n).contiguous()


def _expert_ids(group_sizes, device):
    return torch.cat([
        torch.full((c,), i, device=device, dtype=torch.int64) for i, c in enumerate(group_sizes)
    ])


# (group_sizes, n, k) -- includes tile-multiple, ragged (non-tile-multiple),
# EMPTY experts (0 tokens, must cost 0 blocks not a wasted launch), a single
# expert, and leading empty experts (offset math must still land right).
GROUPED_SHAPES = [
    ([64, 64, 64, 64], 64, 64),
    ([1, 7, 0, 130], 128, 256),
    ([300], 256, 512),
    ([0, 0, 33], 64, 128),
    ([5, 0, 5, 0, 5], 96, 64),
]


@pytest.mark.correctness
@pytest.mark.parametrize("group_sizes,n,k", GROUPED_SHAPES)
def test_grouped_gemm_matches_looped_linear(device, group_sizes, n, k):
    e = len(group_sizes)
    total_m = sum(group_sizes)
    x = torch.randn(total_m, k, device=device, dtype=torch.float16)
    w = torch.randn(e, n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq_stack(w)
    expert_ids = _expert_ids(group_sizes, device)

    y = superl8.grouped_linear_w8a8(x, expert_ids, w_i8, w_scale)
    assert y.shape == (total_m, n) and y.dtype == torch.float16

    # Reference: today's approach -- loop the existing (already-tested)
    # per-expert gemm_w8a8 kernel and stitch results into the ORIGINAL order.
    ref = torch.empty(total_m, n, device=device, dtype=torch.float16)
    for i in range(e):
        mask = expert_ids == i
        if not mask.any():
            continue
        ref[mask] = superl8.linear_w8a8(x[mask], w_i8[i], w_scale[i])
    assert torch.equal(y, ref)

    ref_fp32 = torch.bmm(x.float().unsqueeze(1), w[expert_ids].float().transpose(1, 2)).squeeze(1)
    assert_int8_quality(y, ref_fp32, what=f"grouped_gemm E{e} n{n} k{k}")


@pytest.mark.correctness
def test_grouped_gemm_reproduces_integer_matmul(device):
    """The kernel's int32 accumulate must equal the exact per-expert integer
    matmul (the only slack is the single fp32 dequant multiply)."""
    group_sizes = [37, 0, 128, 5]
    e, n, k = len(group_sizes), 96, 256
    total_m = sum(group_sizes)
    xq = torch.randint(-127, 128, (total_m, k), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (e, n, k), device=device, dtype=torch.int8)
    xs = torch.rand(total_m, device=device, dtype=torch.float32) * 0.01 + 1e-3
    ws = torch.rand(e, n, device=device, dtype=torch.float32) * 0.01 + 1e-3
    gs = torch.tensor(group_sizes, dtype=torch.int64)

    y = superl8._C.gemm_grouped_w8a8(xq, xs, wq, ws, gs)

    expert_ids = _expert_ids(group_sizes, device)
    ref = torch.bmm(xq.float().unsqueeze(1), wq[expert_ids].float().transpose(1, 2)).squeeze(1)
    ref = ref * xs[:, None] * ws[expert_ids]
    assert_int8_quality(y, ref, min_cos=0.9999, max_rel_l1=0.005, min_sqnr_db=40.0,
                        what="gemm_grouped_w8a8 integer-exact")


@pytest.mark.correctness
def test_grouped_gemm_deterministic(device):
    group_sizes = [10, 54, 3]
    e, n, k = len(group_sizes), 64, 128
    total_m = sum(group_sizes)
    x = torch.randn(total_m, k, device=device, dtype=torch.float16)
    w = torch.randn(e, n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq_stack(w)
    expert_ids = _expert_ids(group_sizes, device)

    r0 = superl8.grouped_linear_w8a8(x, expert_ids, w_i8, w_scale)
    for _ in range(3):
        assert torch.equal(superl8.grouped_linear_w8a8(x, expert_ids, w_i8, w_scale), r0)


@pytest.mark.correctness
def test_grouped_gemm_bias(device):
    group_sizes = [8, 12]
    e, n, k = len(group_sizes), 32, 64
    total_m = sum(group_sizes)
    x = torch.randn(total_m, k, device=device, dtype=torch.float16)
    w = torch.randn(e, n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq_stack(w)
    bias = torch.randn(e, n, device=device, dtype=torch.float16)
    expert_ids = _expert_ids(group_sizes, device)

    y = superl8.grouped_linear_w8a8(x, expert_ids, w_i8, w_scale, bias=bias)
    y0 = superl8.grouped_linear_w8a8(x, expert_ids, w_i8, w_scale)
    torch.testing.assert_close(
        y, (y0.float() + bias[expert_ids].float()).half(), rtol=1e-3, atol=1e-3
    )


@pytest.mark.correctness
def test_grouped_gemm_rejects_group_sizes_sum_mismatch(device):
    xq = torch.randint(-127, 128, (10, 64), device=device, dtype=torch.int8)
    wq = torch.randint(-127, 128, (2, 8, 64), device=device, dtype=torch.int8)
    xs = torch.ones(10, device=device, dtype=torch.float32)
    ws = torch.ones(2, 8, device=device, dtype=torch.float32)
    gs = torch.tensor([3, 3], dtype=torch.int64)  # sums to 6, not 10
    with pytest.raises(RuntimeError, match="group_sizes"):
        superl8._C.gemm_grouped_w8a8(xq, xs, wq, ws, gs)


@pytest.mark.correctness
def test_grouped_gemm_rejects_odd_k(device):
    xq = torch.randint(-127, 128, (4, 66), device=device, dtype=torch.int8)  # K=66, %4!=0
    wq = torch.randint(-127, 128, (1, 8, 66), device=device, dtype=torch.int8)
    xs = torch.ones(4, device=device, dtype=torch.float32)
    ws = torch.ones(1, 8, device=device, dtype=torch.float32)
    gs = torch.tensor([4], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="4"):
        superl8._C.gemm_grouped_w8a8(xq, xs, wq, ws, gs)


@pytest.mark.perf
def test_grouped_gemm_perf_vs_looped(device):
    # 32 active experts, uniform batch -- the launch-overhead-dominated regime
    # the issue calls out (Qwen3-MoE / Qwen3-Next scale expert counts).
    group_sizes = [64] * 32
    e, n, k = len(group_sizes), 4864, 896
    total_m = sum(group_sizes)
    x = torch.randn(total_m, k, device=device, dtype=torch.float16)
    w = torch.randn(e, n, k, device=device, dtype=torch.float16) * 0.1
    w_i8, w_scale = _wq_stack(w)
    expert_ids = _expert_ids(group_sizes, device)

    ms_grouped = time_ms(lambda: superl8.grouped_linear_w8a8(x, expert_ids, w_i8, w_scale))

    def looped():
        for i in range(e):
            superl8.linear_w8a8(x[i * 64:(i + 1) * 64], w_i8[i], w_scale[i])

    ms_looped = time_ms(looped)
    tag = "gemm_grouped_w8a8.e32m64n4864k896"
    print("\n" + compare_report(tag, ms_grouped, {"looped.gemm_w8a8": ms_looped}))
    assert_no_regression(tag, ms_grouped)
