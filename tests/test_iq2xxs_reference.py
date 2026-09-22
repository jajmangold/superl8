# SPDX-License-Identifier: BSD-3-Clause
"""IQ2_XXS reference dequant, gated against gguf-py (ggml's own implementation).

Every 66-byte pattern is a valid IQ2_XXS block, so random bytes exercise the grid/codebook,
the sign bits and the scales without a model file (superl8#317).
"""
import numpy as np
import pytest

from superl8.quant.iq2xxs import BLOCK_BYTES, QK_K, dequantize_iq2_xxs, dequantize_iq2_xxs_tensor

gguf = pytest.importorskip("gguf", reason="gguf-py provides the ggml oracle")


def _oracle(blocks: np.ndarray) -> np.ndarray:
    from gguf import GGMLQuantizationType, dequantize

    out = dequantize(np.ascontiguousarray(blocks, dtype=np.uint8), GGMLQuantizationType.IQ2_XXS)
    return out.astype(np.float32).reshape(blocks.shape[0], -1)


def _blocks(n, seed):
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, size=(n, BLOCK_BYTES), dtype=np.uint8)
    b[:, 0:2] = np.frombuffer(rng.uniform(-2, 2, n).astype(np.float16).tobytes(),
                              dtype=np.uint8).reshape(n, 2)
    return b


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_gguf_py_on_random_blocks(seed):
    b = _blocks(31, seed)
    np.testing.assert_array_equal(dequantize_iq2_xxs(b), _oracle(b).reshape(31, QK_K))


def test_zero_block_is_all_zero():
    assert not dequantize_iq2_xxs(np.zeros((2, BLOCK_BYTES), dtype=np.uint8)).any()


def test_tensor_round_trip():
    in_features, n_out = 512, 3
    w = _blocks(n_out * (in_features // QK_K), 7).reshape(n_out, -1)
    out = dequantize_iq2_xxs_tensor(w, in_features)
    assert out.shape == (n_out, in_features)
    want = _oracle(w.reshape(-1, BLOCK_BYTES)).reshape(n_out, in_features)
    np.testing.assert_array_equal(out, want)


def test_block_geometry_is_the_ggml_one():
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

    assert GGML_QUANT_SIZES[GGMLQuantizationType.IQ2_XXS] == (QK_K, BLOCK_BYTES)
    assert BLOCK_BYTES == 66
