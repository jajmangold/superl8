# SPDX-License-Identifier: BSD-3-Clause
"""IQ2_XXS reference dequantizer -- the CPU oracle for the fused kernel (superl8#317).

2.06 bpw: a 256-entry codebook, a top-nibble scale `d*(0.5+s)*0.25`, and four
7-bit sign-LUT indices packed into the second uint32 of each pair.

A faithful transcription of ggml's own dequant (via gguf-py), operation for operation:
the gate is `np.testing.assert_array_equal` against `gguf.dequantize`, so the dtypes and
the ORDER of the float ops matter, not just the algebra.

The codebook is derived from the compact `grid_hex` blob rather than written out as a
table -- see `_iqgrid.derive_grid`.
"""
import numpy as np

from ._iqgrid import KSIGNS, derive_grid  # noqa: F401  (KSIGNS unused by some types)

QK_K = 256
BLOCK_BYTES = 66

_GRID_MAP = (8, 25, 43)
_GRID_SHAPE = (256, 8)
_GRID_HEX = b'00000200050008000a00110014002000220028002a00410044005000580061006400800082008a00a20001010401100115014001840198010002020222028202010404041004210424044004420448046004810484049004a404000502050805200546056905800591050906100640068406a406000805080808140828084108440850085208880804094009020a140a01100410101021104010601084109010951000110811201150115a1180112412451200140814201425144914801418156215001616160118041810184018811800190519a019511a002002200a2044206120802082202921482100220222012404241024402456240025412564259026082820289428442a0140044010401840214024404040484056406040814084409040004120416141804185410142104248425642684200440844204480449944124524450046014804481048404845480049584961498249454a904a005008501150195020508050885004514251a4519152905492540a550156545600581158195864584059085a04601060406068600061556118626062006405641065126584654268008002800a8041808280048118814081118201840484108415844084608400854685948509864086608602880489118a0490109024904090a19016918091459200942294449451958198209902a050a085a009a100a218a450a804a9'
_GRID = derive_grid(_GRID_HEX, _GRID_MAP, _GRID_SHAPE)


def dequantize_iq2_xxs(blocks: np.ndarray) -> np.ndarray:
    """Dequantize raw IQ2_XXS blocks `[n_blocks, 66]` -> float32 `[n_blocks, 256]`."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8)
    if blocks.ndim != 2 or blocks.shape[1] != BLOCK_BYTES:
        raise ValueError(
            f"IQ2_XXS blocks must be [n, {BLOCK_BYTES}], got {tuple(blocks.shape)}")
    n_blocks = blocks.shape[0]
    d, qs = np.hsplit(blocks, [2])
    d = d.view(np.float16).astype(np.float32)
    qs = qs.view(np.uint32).reshape(n_blocks, -1, 2)

    # Scale rides in the top nibble of the second word: d * (0.5 + s) * 0.25.
    db = d * (np.float32(0.5) + (qs[..., 1] >> 28).astype(np.float32)) * np.float32(0.25)
    db = db.reshape((n_blocks, -1, 1, 1))

    # Four 7-bit sign indices per word, each selecting one of 128 sign patterns.
    signs = qs[..., 1].reshape((n_blocks, -1, 1)) >> np.array(
        [0, 7, 14, 21], dtype=np.uint32).reshape((1, 1, 4))
    ksigns = np.frombuffer(KSIGNS, dtype=np.uint8).reshape((1, 1, 1, 128))
    signs = (signs & np.uint32(0x7F)).reshape((n_blocks, -1, 4, 1))
    signs = np.take_along_axis(ksigns, signs, axis=-1)
    signs = signs.reshape((n_blocks, -1, 4, 1)) >> np.array(
        [i for i in range(8)], dtype=np.uint8).reshape((1, 1, 1, 8))
    signs = signs & np.uint8(0x01)
    signs = np.where(signs == 0, np.float32(1), np.float32(-1))
    signs = signs.reshape((n_blocks, -1, 4, 8))

    grid = np.take_along_axis(
        _GRID, qs[..., 0].copy().view(np.uint8).reshape((n_blocks, -1, 1, 1)), axis=-2)
    grid = grid.reshape((n_blocks, -1, 4, 8))
    return (db * grid * signs).reshape((n_blocks, -1))


def dequantize_iq2_xxs_tensor(w: np.ndarray, in_features: int) -> np.ndarray:
    """Dequantize a weight row-block `[n_out, (in/256)*66]` -> float32 `[n_out, in]`."""
    w = np.ascontiguousarray(w, dtype=np.uint8)
    if in_features % QK_K:
        raise ValueError(f"IQ2_XXS needs in%256==0, got {in_features}")
    n_out = w.shape[0]
    n_sb = in_features // QK_K
    if w.shape[1] != n_sb * BLOCK_BYTES:
        raise ValueError(
            f"IQ2_XXS row must be {n_sb * BLOCK_BYTES} bytes, got {w.shape[1]}")
    out = dequantize_iq2_xxs(w.reshape(n_out * n_sb, BLOCK_BYTES))
    return out.reshape(n_out, in_features)
