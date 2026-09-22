# SPDX-License-Identifier: BSD-3-Clause
"""IQ2_XS reference dequantizer -- the CPU oracle for the fused kernel (superl8#317).

2.31 bpw: each uint16 carries a 9-bit grid index (512-entry codebook) plus a
7-bit index into the 128-entry `ksigns_iq2xs` LUT; 4-bit per-32 scales.

A faithful transcription of ggml's own dequant (via gguf-py), operation for operation:
the gate is `np.testing.assert_array_equal` against `gguf.dequantize`, so the dtypes and
the ORDER of the float ops matter, not just the algebra.

The codebook is derived from the compact `grid_hex` blob rather than written out as a
table -- see `_iqgrid.derive_grid`.
"""
import numpy as np

from ._iqgrid import KSIGNS, derive_grid  # noqa: F401  (KSIGNS unused by some types)

QK_K = 256
BLOCK_BYTES = 74

_GRID_MAP = (8, 25, 43)
_GRID_SHAPE = (512, 8)
_GRID_HEX = b'00000200050008000a001100140016001900200022002500280041004400460049005000520055005800610064008000820085008800910094009900a000010104010601090110011201150118011a0121012401400142014501480151015401600168018101840190010002020205020802110214022002410244025002550280028a02010404040604090410041204150418042104240440044204450448045104540456046004810484049004000502050505080511051405200541054405500561058005010604061006260640064206840600080208050808080a08110814082008250841084408500858088008a008aa08010904091009400981098909000a200a280a960aa00a011004100610091010101210151018102110241040104210451048105110541060106a108110841090100011021105110811111114112011411144115011801194119611011204120612101240126012001402140514081411141414201441144414491450146414801401150415101540150016141649160118041810181218401854188618001905196619511aa91a00200220052008200a201120142020204120442050208020a020012104211021402148216521002222228022a82201240424102429244024002541255225992501261a26a626002808280a28202855288828a22868299029082a202a822a882a8a2a014004400640094010401240154018402140244040404240454048404a405140544060406540814084409040004102410541084111411441204141414441504180418541a2410142044210421242294240420044024405440844114414441944204441444444504480449444014504451045244540459a4500460a4644465046014804481048404845485448624800491149444950496949044a00500250055008501150145020502850415044505050805001510451105115514051425100524452aa520154045410542154405460548154a154005508558055885521566856a156005814584158505899581a5940594259855a0160046010604060546062608660a960006124624a62926200641664106540654565a46501686a682569066a546a626a00800280058008801180148020802a8041804480508080808280a880aa8001810481068110814081518159810082208280828282a082a8820184048410841284158440846084898400854485a58518866a860088088825885a8880888288a8880689228a808a888a968aa88a019004901090409056908490009122916491569289920094059444945094589429959095929541965198a6984999159a609a00a002a008a00aa020a02aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a495a465a698a60aa820a822a828a8a0a8a8a804a984a986a928aa2aaa91aaaaaa'
_GRID = derive_grid(_GRID_HEX, _GRID_MAP, _GRID_SHAPE)


def dequantize_iq2_xs(blocks: np.ndarray) -> np.ndarray:
    """Dequantize raw IQ2_XS blocks `[n_blocks, 74]` -> float32 `[n_blocks, 256]`."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8)
    if blocks.ndim != 2 or blocks.shape[1] != BLOCK_BYTES:
        raise ValueError(
            f"IQ2_XS blocks must be [n, {BLOCK_BYTES}], got {tuple(blocks.shape)}")
    n_blocks = blocks.shape[0]
    d, rest = np.hsplit(blocks, [2])
    qs, scales = np.hsplit(rest, [2 * QK_K // 8])
    d = d.view(np.float16).astype(np.float32)
    qs = qs.view(np.uint16)

    scales = scales.reshape((n_blocks, -1, 1)) >> np.array(
        [0, 4], dtype=np.uint8).reshape((1, 1, 2))
    scales = (scales & 0x0F).reshape((n_blocks, -1))
    db = d * (np.float32(0.5) + scales) * np.float32(0.25)
    db = db.reshape((n_blocks, -1, 1, 1))

    # Top 7 bits of each uint16 index the 128-entry sign LUT; low 9 bits are the grid.
    signs = np.frombuffer(KSIGNS, dtype=np.uint8).reshape(1, 1, 128)
    signs = np.take_along_axis(signs, (qs >> 9).reshape((n_blocks, -1, 1)), axis=-1)
    signs = signs.reshape((n_blocks, -1, 1)) >> np.array(
        [i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8))
    signs = signs & np.uint8(0x01)
    signs = np.where(signs == 0, np.float32(1), np.float32(-1))
    signs = signs.reshape((n_blocks, -1, 2, 8))

    grid = np.take_along_axis(
        _GRID, (qs & np.uint16(511)).reshape((n_blocks, -1, 1, 1)), axis=-2)
    grid = grid.reshape((n_blocks, -1, 2, 8))
    return (db * grid * signs).reshape((n_blocks, -1))


def dequantize_iq2_xs_tensor(w: np.ndarray, in_features: int) -> np.ndarray:
    """Dequantize a weight row-block `[n_out, (in/256)*74]` -> float32 `[n_out, in]`."""
    w = np.ascontiguousarray(w, dtype=np.uint8)
    if in_features % QK_K:
        raise ValueError(f"IQ2_XS needs in%256==0, got {in_features}")
    n_out = w.shape[0]
    n_sb = in_features // QK_K
    if w.shape[1] != n_sb * BLOCK_BYTES:
        raise ValueError(
            f"IQ2_XS row must be {n_sb * BLOCK_BYTES} bytes, got {w.shape[1]}")
    out = dequantize_iq2_xs(w.reshape(n_out * n_sb, BLOCK_BYTES))
    return out.reshape(n_out, in_features)
