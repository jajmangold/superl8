"""w8a8 decode-kernel routing crossover (fni8-serve#479): per-row decode GEMV up to M=8 rows,
tile GEMM above. CPU-only: drives the real routing in linear_w8a8 with the CUDA kernels stubbed."""
from unittest import mock

import pytest
import torch

import superl8.ops as ops

pytestmark = pytest.mark.cpu


def _route(m, n=4096, k=4096):
    calls = []

    def kernel(name):
        def fn(x, *rest):
            calls.append(name)
            w_i8 = rest[-3]
            return torch.zeros(x.shape[0], w_i8.shape[0], dtype=torch.float16)
        return fn

    fake_c = mock.MagicMock()
    for name in ("gemm_decode_w8a8_fp16in", "gemm_decode_w8a8", "gemm_w8a8"):
        setattr(fake_c, name, kernel(name))
    x = torch.randn(m, k, dtype=torch.float16)
    w = torch.zeros(n, k, dtype=torch.int8)
    ws = torch.ones(n, dtype=torch.float32)
    with mock.patch.object(ops, "_C", fake_c), \
         mock.patch.object(torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True), \
         mock.patch.object(ops, "quantize_i8_rowwise", lambda t: (t.to(torch.int8), torch.ones(t.shape[0]))):
        ops.linear_w8a8(x, w, ws, out_dtype=torch.float16)
    return calls


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_small_batches_use_the_decode_gemv(m):
    calls = _route(m)
    assert len(calls) == 1 and calls[0] in ("gemm_decode_w8a8_fp16in", "gemm_decode_w8a8"), calls


@pytest.mark.parametrize("m", [16, 32])
def test_batch_16_and_up_use_the_tile_gemm(m):
    assert _route(m) == ["gemm_w8a8"]


def test_crossover_never_exceeds_the_cxx_decode_cap():
    assert ops._W8A8_DECODE_MAX_M <= ops._DECODE_MAX_M
