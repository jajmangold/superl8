# SPDX-License-Identifier: BSD-3-Clause
"""IQ4_XS reference dequant, gated against gguf-py (ggml's own implementation).

Every 136-byte pattern is a valid IQ4_XS block, so random bytes exercise the codebook,
both nibble halves and the split 6-bit scales without a model file.
"""
import numpy as np
import pytest

from superl8.quant.iq4xs import BLOCK_BYTES, QK_K, dequantize_iq4_xs, dequantize_iq4_xs_tensor

gguf = pytest.importorskip("gguf", reason="gguf-py provides the ggml oracle")


def _oracle(blocks: np.ndarray) -> np.ndarray:
    from gguf import GGMLQuantizationType, dequantize

    out = dequantize(np.ascontiguousarray(blocks, dtype=np.uint8), GGMLQuantizationType.IQ4_XS)
    return out.astype(np.float32).reshape(blocks.shape[0], -1)


def _blocks(n, seed):
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, size=(n, BLOCK_BYTES), dtype=np.uint8)
    b[:, 0:2] = np.frombuffer(rng.uniform(-2, 2, n).astype(np.float16).tobytes(),
                              dtype=np.uint8).reshape(n, 2)
    return b


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_gguf_py_on_random_blocks(seed):
    b = _blocks(29, seed)
    np.testing.assert_array_equal(dequantize_iq4_xs(b), _oracle(b).reshape(29, QK_K))


def test_nibble_halves_are_interleaved_not_sequential():
    """Weight j uses the LOW nibble of qs[j]; weight j+16 the HIGH nibble. Getting this
    backwards still produces plausible values, so pin it against the oracle explicitly."""
    b = np.zeros((1, BLOCK_BYTES), dtype=np.uint8)
    b[0, 0:2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    # Sub-block 0 scale is SPLIT: low 4 bits from scales_l[0], high 2 bits from
    # scales_h bits 0-1. ls = 33 = 1 | (2 << 4) -> dl = d * (ls - 32) = 1.0.
    b[0, 4] = 1               # scales_l[0] low nibble = 1
    b[0, 2] = 2               # scales_h bits 0-1 = 2
    b[0, 8] = 0x0F            # qs[0]: low nibble 15, high nibble 0
    got = dequantize_iq4_xs(b)[0]
    assert got[0] == pytest.approx(113.0)   # kvalues_iq4nl[15]
    assert got[16] == pytest.approx(-127.0)  # kvalues_iq4nl[0]
    np.testing.assert_array_equal(dequantize_iq4_xs(b), _oracle(b).reshape(1, QK_K))


def test_tensor_round_trip():
    in_features, n_out = 512, 3
    w = _blocks(n_out * (in_features // QK_K), 7).reshape(n_out, -1)
    out = dequantize_iq4_xs_tensor(w, in_features)
    assert out.shape == (n_out, in_features)
    want = _oracle(w.reshape(-1, BLOCK_BYTES)).reshape(n_out, in_features)
    np.testing.assert_array_equal(out, want)


def test_block_geometry_is_the_ggml_one():
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

    assert GGML_QUANT_SIZES[GGMLQuantizationType.IQ4_XS] == (QK_K, BLOCK_BYTES)
