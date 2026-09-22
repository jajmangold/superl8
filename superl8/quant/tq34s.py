# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""TurboQuant ``TQ3_4S`` reference (type-46) — pure torch/numpy oracle.

Mirrors the turbo-tan/llama.cpp-tq3 fork's CPU semantics exactly
(`ggml-quants.c:2706-2840`, `ggml-cuda/convert.cu:805-845`):

    1. unpack 3-bit indices (4 groups x 8 per 32-value block),
    2. ``v_j = TQ3_CENTROIDS[code_j] * scale_g(j)``,  ``scale`` = E3M5 u8
       per-8 (0 => 0.0; else ``2^(exp-9) * (1 + mant/32)``),
    3. ``w = RHT_inv(v)``,  RHT_inv = ``diag(SIGNS) . H / sqrt(32)``
       (Walsh-Hadamard butterfly + sign pattern + 1/sqrt(32)).

Block = 4 u8 scales + 12 code bytes = 16 B / 32 wt, ``GGML_TYPE_TQ3_4S = 46``,
rows ``[out, (in/32)*16]``, ``in % 32 == 0``.

The **linear-algebra identity** that makes this dp4a-fusable (and is the kernel's
charter, superl8#270):

    x_act^T . w = x_act^T . RHT_inv(v) = (RHT_fwd(x_act))^T . v
              RHT_fwd = H . diag(SIGNS) / sqrt(32)   (symmetric pair, F . F^T = I)

So the activation is rotated per 32-block (signs, then WHT butterfly, then 1/sqrt(32))
and the weight side is a plain dot against the raw centroid-codes. The exact dequant
here is the correctness oracle; `tq34s_levels()` gives the int8 centroid levels the
fused dp4a kernel uses (superl8 levels, with the corrected max-centroid constant).
"""

from __future__ import annotations

import numpy as np
import torch

QK_TQ3 = 32          # values per block
TQ3_TYPE_SIZE = 16   # 4 scale bytes + 12 code bytes
TQ3_GGML_TYPE = 46   # GGML_TYPE_TQ3_4S

# Fixed codebook + sign pattern (fork ggml-quants.c:2360-2390).
TQ3_CENTROIDS = np.array(
    [-1.996684, -1.291398, -0.740341, -0.247508,
      0.230106,  0.725222,  1.277503,  1.988943], dtype=np.float32)

TQ3_SIGNS = np.array(
    [+1, -1, +1, -1, +1, +1, -1, +1,
     -1, -1, +1, -1, +1, +1, -1, +1,
     -1, -1, +1, -1, +1, -1, -1, +1,
     -1, +1, +1, -1, +1, -1, -1, +1], dtype=np.float32)


def decode_e3m5(byte: np.ndarray) -> np.ndarray:
    """E3M5 mini-float scale decode, exact fp32 (fork `tq3_4s_ratio4s`).

    ``byte == 0`` -> 0.0; else ``2^(exp-9) * (1 + mant/32)``, exp = byte>>5,
    mant = byte&31. Built directly from the fp32 bit pattern:
    ``((byte>>5)+118)<<23 | (byte&31)<<18`` (no ldexpf in the hot loop).
    """
    b = byte.astype(np.uint32)
    bits = (((b >> 5) + 118) << 23) | ((b & 31) << 18)
    scale = bits.view(np.float32)
    scale = np.where(b == 0, np.float32(0.0), scale)
    return scale


def unpack_3bit(qs: np.ndarray) -> np.ndarray:
    """Unpack the 12 code bytes (4 groups x 3) into 32 3-bit indices.

    ``qs`` shape ``[..., 12]`` -> indices ``[..., 32]`` in the fork's group
    bit order (ggml-quants.c dequantize_row_tq3_4s idx[] lines: 8 values from
    each 3-byte packed group, groups stored contiguously: byte 3g+r of qs
    belongs to group g).
    """
    qs = qs.astype(np.uint32)
    grp = qs.reshape(qs.shape[:-1] + (4, 3))           # [..., group, byte]
    qp0, qp1, qp2 = grp[..., 0], grp[..., 1], grp[..., 2]
    idx = np.empty(qs.shape[:-1] + (4, 8), dtype=np.uint32)
    idx[..., 0] = qp0 & 7
    idx[..., 1] = (qp0 >> 3) & 7
    idx[..., 2] = ((qp0 >> 6) | (qp1 << 2)) & 7
    idx[..., 3] = (qp1 >> 1) & 7
    idx[..., 4] = (qp1 >> 4) & 7
    idx[..., 5] = ((qp1 >> 7) | (qp2 << 1)) & 7
    idx[..., 6] = (qp2 >> 2) & 7
    idx[..., 7] = (qp2 >> 5) & 7
    return idx.reshape(qs.shape[:-1] + (QK_TQ3,))


def _wht(x: np.ndarray) -> np.ndarray:
    """In-place Walsh-Hadamard butterfly over the last axis (n=32).

    Mirrors the fork's ``tq3_0_rht_forward/inverse`` inner loop exactly: for
    each block of ``2*step`` values, EVERY offset ``j`` in ``[0, step)`` pairs
    with ``j+step`` (out[j]=a+b, out[j+step]=a-b). Slicing ``0::2*step`` alone
    only hits the offset-0 pairs — group each block as ``[2, step]`` so the
    whole first half butterflies against the second half.
    """
    x = x.astype(np.float32).copy()
    shape = x.shape
    y = x.reshape(-1, QK_TQ3)
    step = 1
    while step < QK_TQ3:
        g = y.reshape(-1, QK_TQ3 // (2 * step), 2, step)
        s = g[..., 0, :] + g[..., 1, :]
        d = g[..., 0, :] - g[..., 1, :]
        g[..., 0, :] = s
        g[..., 1, :] = d
        step <<= 1
    return y.reshape(shape)


_RHT_NORM = np.float32(1.0 / np.sqrt(QK_TQ3))  # fp32 to match the fork's float math


def rht_forward(x: np.ndarray) -> np.ndarray:
    """Forward RHT: ``out = H . diag(SIGNS) . x / sqrt(32)`` (signs then WHT)."""
    return _wht(x * TQ3_SIGNS) * _RHT_NORM


def rht_inverse(x: np.ndarray) -> np.ndarray:
    """Inverse RHT: ``out = diag(SIGNS) . H . x / sqrt(32)`` (WHT then signs)."""
    return _wht(x) * (TQ3_SIGNS * _RHT_NORM)


def dequantize_tq34s_bytes(u8: np.ndarray, in_features: int) -> np.ndarray:
    """Dequant raw TQ3_4S bytes ``[out, (in/32)*16]`` -> fp32 ``[out, in]``.

    Exact fork semantics: codes -> centroids * per-8 E3M5 scale, then the
    inverse RHT. This is the ORACLE the fused kernel must match (SQNR gate).
    """
    u8 = np.asarray(u8)
    out = u8.shape[0]
    assert u8.shape[1] == (in_features // QK_TQ3) * TQ3_TYPE_SIZE, (
        f"tq3_4s bytes {u8.shape} != [out, {in_features // QK_TQ3 * TQ3_TYPE_SIZE}]")
    blk = u8.reshape(out, in_features // QK_TQ3, TQ3_TYPE_SIZE)
    scales = decode_e3m5(blk[..., 0:4])                       # [out, nb, 4]
    idx = unpack_3bit(blk[..., 4:16]).reshape(out, -1, 4, 8)  # [out, nb, 4, 8]
    v = TQ3_CENTROIDS[idx] * scales[..., None]                # [out, nb, 4, 8]
    v = v.reshape(out, -1, QK_TQ3)
    w = rht_inverse(v)
    return w.reshape(out, in_features)


def tq34s_levels() -> np.ndarray:
    """int8 centroid levels for the fused dp4a kernel.

    ``l_c = round(centroid[c] * K)`` with ``K = 127 / max|centroid|`` so the
    largest centroid maps to exactly ±127 and the per-block fp scale factor in
    the epilogue is ``max|centroid|/127`` (0.0157227...). Fork uses a stale
    constant (2.1519) that does not match its own levels; superl8 uses the exact
    max centroid (1.996684) so ``l_c / K == centroid[c]`` up to 0.5-level rounding.
    """
    kmax = float(np.abs(TQ3_CENTROIDS).max())
    lv = np.rint(TQ3_CENTROIDS * (127.0 / kmax)).astype(np.int8)
    assert lv[0] == -127 and lv[-1] == 127
    return lv


def reference_linear(x: torch.Tensor, u8: torch.Tensor, in_features: int) -> torch.Tensor:
    """Dequant-then-matmul reference in fp32 — the kernel's correctness oracle.

    ``x``: [M, in] fp32/bf16 activations.  ``u8``: [out, (in/32)*16] uint8 bytes.
    Returns [M, out]. Matches the exact fork dequant semantics.
    """
    w = dequantize_tq34s_bytes(u8.detach().cpu().numpy(), in_features)
    w = torch.from_numpy(w).to(x.device, torch.float32)
    return x.float() @ w.t()
