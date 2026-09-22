# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR0 toolchain smoke tests for the hello_add op.

These prove the full loop (nvcc sm_70 -> .so -> pybind -> torch -> gate) works
end-to-end, and demonstrate the correctness + perf gate shapes every real kernel
PR must follow. Not a numerical stress test.
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8

# Make the bench harness importable (repo-root/bench).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import assert_no_regression, time_ms  # noqa: E402

SHAPES = [(1,), (255,), (256,), (257,), (1024, 1024), (13, 71)]  # incl. non-multiples


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_hello_add_matches_reference(device, shape):
    a = torch.randn(shape, device=device, dtype=torch.float16)
    b = torch.randn(shape, device=device, dtype=torch.float16)
    out = superl8.hello_add(a, b)
    ref = a + b  # torch fp16 reference
    assert out.shape == a.shape and out.dtype == torch.float16
    torch.testing.assert_close(out, ref, rtol=0, atol=0)  # exact for a single add


@pytest.mark.correctness
def test_hello_add_deterministic(device):
    a = torch.randn(4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, device=device, dtype=torch.float16)
    r0 = superl8.hello_add(a, b)
    for _ in range(3):
        assert torch.equal(superl8.hello_add(a, b), r0)  # bitwise identical -> no races


@pytest.mark.correctness
def test_hello_add_rejects_bad_dtype(device):
    a = torch.randn(16, device=device, dtype=torch.float32)
    b = torch.randn(16, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="float16"):
        superl8.hello_add(a, b)


@pytest.mark.perf
def test_hello_add_perf(device):
    a = torch.randn(1 << 22, device=device, dtype=torch.float16)
    b = torch.randn(1 << 22, device=device, dtype=torch.float16)
    ms = time_ms(lambda: superl8.hello_add(a, b))
    # Soft-skips until a baseline is committed; then fails on >5% regression.
    assert_no_regression("hello_add.4Mi.fp16", ms)
