# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Hadamard incoherence rotation for tighter INT8 quantization.

Attention is invariant to an orthogonal rotation of the head dim: for any
orthogonal M, (Q M)(K M)^T = Q M M^T K^T = Q K^T. So rotating Q and K by the
SAME normalized Hadamard M before int8 quantization leaves the logits exactly
unchanged, while the rotated vectors quantize far more accurately — the
transform spreads per-channel outliers (the dominant int8 error source in K)
across all D coordinates toward ~N(0, 1/d). This is the incoherence-processing
step from QuaRot / TurboQuant+ (Apache-2.0, studied for approach only).

Rotate Q/K only: V "compresses free" (TurboQuant's asymmetric-KV finding) and
rotating it would require un-rotating the output — not worth it for int8. The
rotation touches NO CUDA kernel; it is purely a quant-prologue change.

M = H_d / sqrt(d) is the normalized Sylvester Hadamard (orthogonal, symmetric,
deterministic — no seed, so determinism tests hold). d must be a power of two
(our head dims {32, 64, 128} all are).
"""
import functools

import torch


@functools.lru_cache(maxsize=8)
def hadamard_matrix(d: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Normalized Sylvester-Hadamard M [d, d], orthogonal (M @ M.T == I). Cached."""
    assert d > 0 and (d & (d - 1)) == 0, f"Hadamard needs a power-of-two dim, got {d}"
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < d:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    m = (h / (d ** 0.5)).to(device=device, dtype=dtype)
    return m


def rotate_last(x: torch.Tensor) -> torch.Tensor:
    """Apply the Hadamard rotation over the last (head) dim: returns x @ M.

    Orthogonal, so it preserves every Q·K inner product; rotate Q and K with
    this same function and the attention logits are unchanged.
    """
    # Include the CUDA index in the cache key. ``device.type`` collapses every
    # GPU to the same literal ``"cuda"`` entry, so a matrix first materialized
    # on cuda:0 is incorrectly reused for an activation on cuda:1.
    m = hadamard_matrix(x.shape[-1], device=str(x.device), dtype=torch.float32)
    return (x.float() @ m).to(x.dtype)
