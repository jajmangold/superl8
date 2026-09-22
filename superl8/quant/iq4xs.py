# SPDX-License-Identifier: BSD-3-Clause
"""IQ4_XS reference dequantizer — the fp32 oracle for the fused dp4a kernel (superl8#317).

Direct transcription of ggml's ``dequantize_row_iq4_xs``. Block = 136 B per 256 weights
(4.25 bpw):

    +0   d          fp16 super-block scale
    +2   scales_h   uint16; 2 high bits of each of the 8 sub-block scales
    +4   scales_l[4] two 4-bit low halves per byte
    +8   qs[128]    4-bit codebook indices, 16 bytes per 32-weight sub-block

Per 32-weight sub-block ``ib``::

    ls = ((scales_l[ib//2] >> 4*(ib%2)) & 0xF) | (((scales_h >> 2*ib) & 3) << 4)
    dl = d * (ls - 32)                      # SIGNED offset scale, unlike IQ3_S's 1+2s

Within a sub-block the two halves are INTERLEAVED BY NIBBLE, not sequential: for
``j`` in 0..15, weight ``j`` takes the low nibble of ``qs[j]`` and weight ``j+16`` the
high nibble. Values come from the 16-entry signed codebook ``kvalues_iq4nl``.
"""
from __future__ import annotations

import numpy as np

QK_K = 256
BLOCK_BYTES = 136

# ggml-common.h kvalues_iq4nl
_KVALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                     1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int8)


def dequantize_iq4_xs(blocks: np.ndarray) -> np.ndarray:
    """``blocks`` uint8 ``[nb, 136]`` -> fp32 ``[nb, 256]``, bit-exact vs ggml."""
    b = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
    nb = b.shape[0]
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    scales_h = b[:, 2:4].copy().view(np.uint16).astype(np.int32).reshape(nb, 1)
    scales_l = b[:, 4:8].astype(np.int32)                       # [nb, 4]
    qs = b[:, 8:136].reshape(nb, 8, 16).astype(np.int32)        # [nb, ib, j]

    ib = np.arange(8, dtype=np.int32)
    lo = (scales_l[:, ib // 2] >> (4 * (ib % 2))[None, :]) & 0xF
    hi = ((scales_h >> (2 * ib)[None, :]) & 3) << 4
    dl = d * ((lo | hi) - 32).astype(np.float32)                # [nb, 8]

    low = _KVALUES[qs & 0xF].astype(np.float32)                 # weights  0..15
    high = _KVALUES[qs >> 4].astype(np.float32)                 # weights 16..31
    out = np.concatenate([low, high], axis=2)                   # [nb, 8, 32]
    return (out * dl[:, :, None]).reshape(nb, QK_K)


def dequantize_iq4_xs_tensor(w_bytes: np.ndarray, in_features: int) -> np.ndarray:
    """``w_bytes`` uint8 ``[N, (in//256)*136]`` -> fp32 ``[N, in]``."""
    n_out = w_bytes.shape[0]
    nsb = in_features // QK_K
    assert w_bytes.shape[1] == nsb * BLOCK_BYTES, (w_bytes.shape, nsb * BLOCK_BYTES)
    return dequantize_iq4_xs(w_bytes.reshape(-1, BLOCK_BYTES)).reshape(n_out, in_features)
