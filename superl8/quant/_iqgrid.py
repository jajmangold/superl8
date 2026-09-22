# SPDX-License-Identifier: BSD-3-Clause
"""Shared codebook-grid derivation for the i-quant references.

ggml's i-quant codebooks are big (IQ1_S is 2048x8 float values) but they are NOT
stored as tables: gguf-py keeps a compact ASCII-hex blob plus a 3-entry value map and
expands it. We do the same, for two reasons: a transcribed table is ~87 KB of source
for IQ1_S alone, and a mistyped entry fails *plausibly* -- right shape, wrong values,
which a tolerance-based check will not catch.

Verified to reproduce `gguf.quants.<TYPE>.grid` exactly (0.0 error) for IQ2_XXS,
IQ2_XS, IQ2_S and IQ1_S.
"""
import numpy as np

# ggml's ksigns_iq2xs: 128 sign patterns, one byte each (IQ2_XXS and IQ2_XS).
KSIGNS = b'\x00\x81\x82\x03\x84\x05\x06\x87\x88\t\n\x8b\x0c\x8d\x8e\x0f\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f\xa0!"\xa3$\xa5\xa6\'(\xa9\xaa+\xac-.\xaf0\xb1\xb23\xb456\xb7\xb89:\xbb<\xbd\xbe?\xc0AB\xc3D\xc5\xc6GH\xc9\xcaK\xccMN\xcfP\xd1\xd2S\xd4UV\xd7\xd8YZ\xdb\\\xdd\xde_`\xe1\xe2c\xe4ef\xe7\xe8ij\xebl\xed\xeeo\xf0qr\xf3t\xf5\xf6wx\xf9\xfa{\xfc}~\xff'


def derive_grid(grid_hex: bytes, grid_map: tuple, grid_shape: tuple) -> np.ndarray:
    """Expand a `grid_hex` blob into the [1, 1, n_entries, 8] float32 codebook.

    Mirrors gguf-py's `__init_grid`: decode ASCII-hex pairs to bytes, unpack
    `log2(len(grid_map))`-bit indices little-endian, then map through `grid_map`.
    The leading (1, 1) axes are what `np.take_along_axis(grid, idx, axis=-2)` wants.
    """
    bits = int(np.ceil(np.log2(len(grid_map))))
    per_byte = 8 // bits
    g = np.frombuffer(grid_hex, dtype=np.uint8).reshape((-1, 2))
    g = (np.where(g > 0x40, g + 9, g) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
    g = g[..., 0] | g[..., 1]
    g = g.reshape((-1, 1)) >> np.array(
        [i for i in range(0, 8, 8 // per_byte)], dtype=np.uint8
    ).reshape((1, per_byte))
    g = (g & ((1 << bits) - 1)).reshape((-1, 1))
    gm = np.array(grid_map, dtype=np.float32).reshape((1, -1))
    return np.take_along_axis(gm, g, axis=-1).reshape((1, 1, *grid_shape))
