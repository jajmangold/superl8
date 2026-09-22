# SPDX-License-Identifier: BSD-3-Clause
"""IQ3_S reference dequant is the oracle for the fused dp4a kernel (superl8#317 work item 2).

Gate it against upstream gguf-py, which ships ggml's own implementation. Every 110-byte
pattern is a valid IQ3_S block (qs/qh/signs/scales are all free bits), so random bytes are
a complete test — no model file needed.
"""
import numpy as np
import pytest

from superl8.quant.iq3s import BLOCK_BYTES, QK_K, dequantize_iq3_s, dequantize_iq3_s_tensor

gguf = pytest.importorskip("gguf", reason="gguf-py provides the ggml oracle")


def _oracle(blocks: np.ndarray) -> np.ndarray:
    """gguf-py's own ggml implementation. It takes the raw uint8 ARRAY (it reads
    `.shape` to infer the block count) — handing it `bytes` raises AttributeError."""
    from gguf import GGMLQuantizationType, dequantize

    out = dequantize(np.ascontiguousarray(blocks, dtype=np.uint8), GGMLQuantizationType.IQ3_S)
    return out.astype(np.float32).reshape(blocks.shape[0], -1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_gguf_py_on_random_blocks(seed):
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(37, BLOCK_BYTES), dtype=np.uint8)
    # keep d finite: the fp16 scale must not be NaN/Inf
    blocks[:, 0:2] = np.frombuffer(
        rng.uniform(-2, 2, 37).astype(np.float16).tobytes(), dtype=np.uint8
    ).reshape(37, 2)
    got = dequantize_iq3_s(blocks)
    want = _oracle(blocks).reshape(37, QK_K)
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)


def test_zero_scale_block_is_all_zero():
    blocks = np.zeros((2, BLOCK_BYTES), dtype=np.uint8)
    assert not dequantize_iq3_s(blocks).any()


def test_tensor_shape_round_trip():
    rng = np.random.default_rng(7)
    in_features, n_out = 512, 3
    nsb = in_features // QK_K
    w = rng.integers(0, 256, size=(n_out, nsb * BLOCK_BYTES), dtype=np.uint8)
    w[:, 0:2] = 0x3C  # ~1.0 in fp16 for both bytes' worth of blocks
    out = dequantize_iq3_s_tensor(w, in_features)
    assert out.shape == (n_out, in_features)
    want = _oracle(w.reshape(-1, BLOCK_BYTES)).reshape(n_out, in_features)
    np.testing.assert_array_equal(out, want)


def test_block_geometry_is_the_ggml_one():
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

    n, nbytes = GGML_QUANT_SIZES[GGMLQuantizationType.IQ3_S]
    assert (n, nbytes) == (QK_K, BLOCK_BYTES)
