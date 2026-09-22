# SPDX-License-Identifier: BSD-3-Clause
"""IQ3_XXS reference dequantizer — the fp32 oracle for the fused dp4a kernel (superl8#317).

Direct transcription of ggml's ``dequantize_row_iq3_xxs``. Block = 98 B per 256 weights
(3.0625 bpw):

    +0   d       fp16 super-block scale
    +2   qs[64]  8-bit grid indices, 8 per 32-weight sub-block
    +66  qs[64:96] = scales_and_signs, ONE uint32 per sub-block (8 x 4 B)

Per 32-weight sub-block ``ib32`` with ``aux32 = scales_and_signs[ib32]``::

    db    = d * (0.5 + (aux32 >> 28)) * 0.5      # 4-bit scale in the TOP nibble, half-offset
    signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]   # l = 0..3, a 128-entry sign LUT
    y[j]   = db * grid[qs[2l  ]][j] * (signs & kmask[j]   ? -1 : +1)
    y[j+4] = db * grid[qs[2l+1]][j] * (signs & kmask[j+4] ? -1 : +1)

Three things differ from IQ3_S: the scale is a top-nibble half-offset (not ``1 + 2s``), the
grid index is a plain byte (no ``qh`` 9th bit), and the sign bits come through a lookup
rather than being stored directly.
"""
from __future__ import annotations

import numpy as np

QK_K = 256
BLOCK_BYTES = 98

# ggml-common.h iq3xxs_grid (256 x uint32 = 4 packed uint8 magnitudes each).
_IQ3XXS_GRID = np.array([
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
], dtype=np.uint32)

# ggml-common.h ksigns_iq2xs: 7 index bits -> 8 sign bits.
_KSIGNS = np.array([
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
], dtype=np.uint8)

# ggml-common.h kmask_iq2xs
_KMASK = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

_GRID_U8 = _IQ3XXS_GRID.view(np.uint8).reshape(256, 4)


def dequantize_iq3_xxs(blocks: np.ndarray) -> np.ndarray:
    """``blocks`` uint8 ``[nb, 98]`` -> fp32 ``[nb, 256]``, bit-exact vs ggml."""
    b = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
    nb = b.shape[0]
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    qs = b[:, 2:66].reshape(nb, 8, 8).astype(np.int32)          # [nb, ib32, position]
    aux = b[:, 66:98].copy().view(np.uint32).reshape(nb, 8)     # one per sub-block

    db = (d * (0.5 + (aux >> 28).astype(np.float32)) * 0.5)     # [nb, 8]

    mags = _GRID_U8[qs].astype(np.float32)                      # [nb, 8, 8, 4]

    l = np.arange(8, dtype=np.uint32) // 2                      # position p -> l = p//2
    signs = _KSIGNS[((aux[:, :, None] >> (7 * l)[None, None, :]) & 127)]   # [nb, 8, 8]
    half = (np.arange(8) % 2) * 4                               # even p -> bits 0..3, odd -> 4..7
    mask = _KMASK[half[None, None, :, None] + np.arange(4)[None, None, None, :]]
    mags = np.where((signs[:, :, :, None] & mask) != 0, -mags, mags)

    return (db[:, :, None, None] * mags).reshape(nb, QK_K)


def dequantize_iq3_xxs_tensor(w_bytes: np.ndarray, in_features: int) -> np.ndarray:
    """``w_bytes`` uint8 ``[N, (in//256)*98]`` -> fp32 ``[N, in]``."""
    n_out = w_bytes.shape[0]
    nsb = in_features // QK_K
    assert w_bytes.shape[1] == nsb * BLOCK_BYTES, (w_bytes.shape, nsb * BLOCK_BYTES)
    return dequantize_iq3_xxs(w_bytes.reshape(-1, BLOCK_BYTES)).reshape(n_out, in_features)
