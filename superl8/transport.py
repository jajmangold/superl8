# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Real-time TRANSPORT compression for the PCIe-1.0-x1 fleet.

Motivation (see utils/docs/transport-compression.md). Every card here is a
CMP 100-210 on **PCIe 1.0 x1 ~= 250 MB/s** — ~3300x slower than the 829 GB/s HBM.
So any multi-GPU collective (PP activation handoff, MoE all-to-all, gradient
all-reduce) is wire-bound by a colossal margin: on-GPU compute is ~1000-3000x
cheaper than the link, so you should compress the wire AGGRESSIVELY (maximize
ratio, not throughput). The breakeven throughput to beat the wire at ratio r is
only ~250 MB/s / (1 - 1/r); the GPU clears that by three orders of magnitude.

This module reuses superl8's existing quant kernels as a transport CODEC: quantize a
tensor to int8 / int4 / NF4 (+ optional Hadamard incoherence rotation), bit-pack
it, ship the payload + fp32 scales, and reconstruct on arrival. Unlike the
attention path (where Q.K^T is rotation-invariant so we never un-rotate), here we
reconstruct the ACTIVATION itself, so a rotated scheme un-rotates on decompress
(the normalized Hadamard is self-inverse).

It is a lossy codec; `reconstruction_report` gives the cos / rel-L1 so the caller
picks a scheme against a quality bar. `code_entropy_bits` estimates the *extra*
ratio a rANS/range entropy coder would add on top (cheaply, from the code
histogram) so we can bound that headroom without building an arithmetic coder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .quant.lowbit import NF4_CODEBOOK, lowbit_levels, pack_rows_lowbit, unpack_rows_lowbit
from .quant.rotation import hadamard_matrix

# PCIe 1.0 x1: 2.5 GT/s * 1 lane * 8b/10b = 2.0 Gbit/s = 250 MB/s per direction.
PCIE1_X1_BYTES_PER_S = 250e6

_SCHEMES = ("fp16", "int8", "int4", "int4-had", "nf4")


@dataclass
class Compressed:
    """A compressed activation payload ready for the wire.

    `payload`/`scales` are the bytes actually sent; `on_wire_bytes` is their sum.
    Everything needed to reconstruct rides along (shape, scheme, group_size).
    """
    scheme: str
    shape: tuple
    dtype: torch.dtype
    group_size: int | None
    payload: torch.Tensor          # packed codes (int32 words / uint8 nibbles / int8)
    scales: torch.Tensor           # fp32 per-group scales
    d: int                         # last-dim length (for unpack)

    @property
    def on_wire_bytes(self) -> int:
        return self.payload.numel() * self.payload.element_size() + self.scales.numel() * 4


def _rot_block(d: int, group_size: int | None) -> int:
    """Power-of-two Hadamard block size for a last dim `d`. Prefers `group_size`
    (so rotation and quant grouping align); falls back to the largest pow2 <=128
    that divides `d`. Returns 1 (rotation is a no-op) if `d` has no pow2 factor."""
    b = group_size if (group_size and (group_size & (group_size - 1)) == 0) else 128
    while b > 1 and d % b:
        b //= 2
    return b


def _block_hadamard(x: torch.Tensor, blk: int) -> torch.Tensor:
    """Block-diagonal normalized Hadamard over the last dim (QuaRot-style): rotate
    within each `blk`-wide sub-block. Handles non-power-of-two `d` (blk | d). The
    normalized Hadamard is symmetric+orthogonal, so applying it twice is identity —
    decompress un-rotates by calling this again."""
    if blk <= 1:
        return x
    d = x.shape[-1]
    m = hadamard_matrix(blk, x.device.type, torch.float32)
    xr = x.float().reshape(*x.shape[:-1], d // blk, blk)
    return (xr @ m).reshape(x.shape).to(x.dtype)


def _pack_nibbles(idx: torch.Tensor) -> torch.Tensor:
    """Pack an even-length last dim of unsigned 4-bit indices (0..15) -> uint8."""
    lo = idx[..., 0::2].to(torch.int32)
    hi = idx[..., 1::2].to(torch.int32)
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor, d: int) -> torch.Tensor:
    p = packed.to(torch.int32)
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], d)


def _grouped_absmax(xf: torch.Tensor, g: int, levels: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group amax scale over the last dim. Returns (scale[..., ng], xg[..., ng, g])."""
    d = xf.shape[-1]
    xg = xf.reshape(*xf.shape[:-1], d // g, g)
    scale = xg.abs().amax(dim=-1, keepdim=True) / levels
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    return safe, xg


def compress_activation(
    x: torch.Tensor, *, scheme: str = "int8", group_size: int | None = None
) -> Compressed:
    """Compress a tensor for transport over the slow link. `x`: any shape, the
    LAST dim is the quantized axis. Returns a `Compressed` payload.

    Schemes: ``fp16`` (baseline, no loss), ``int8`` (2x), ``int4`` (uniform
    symmetric, ~4x), ``int4-had`` (Hadamard-rotated int4 — spreads outliers, needs
    a power-of-two last dim), ``nf4`` (non-uniform 4-bit, best for Gaussian-ish
    activations). ``group_size`` trades quality (smaller = tighter) against scale
    overhead; ``None`` = one scale per row.
    """
    assert scheme in _SCHEMES, f"unknown scheme {scheme!r}, pick from {_SCHEMES}"
    d = x.shape[-1]
    g = d if group_size is None else group_size
    assert d % g == 0, f"last dim {d} not divisible by group_size {g}"

    if scheme == "fp16":
        payload = x.to(torch.float16)
        empty = torch.empty(0, dtype=torch.float32, device=x.device)
        return Compressed(scheme, tuple(x.shape), x.dtype, group_size, payload, empty, d)

    if scheme == "int8":
        safe, xg = _grouped_absmax(x.float(), g, 127.0)
        codes = torch.round(xg / safe).clamp_(-127, 127).reshape(x.shape).to(torch.int8)
        return Compressed(scheme, tuple(x.shape), x.dtype, group_size, codes,
                          safe.squeeze(-1).float(), d)

    if scheme in ("int4", "int4-had"):
        xf = _block_hadamard(x, _rot_block(d, group_size)).float() if scheme == "int4-had" \
            else x.float()
        levels = lowbit_levels(4)  # ±7
        safe, xg = _grouped_absmax(xf, g, levels)
        codes = torch.round(xg / safe).clamp_(-levels, levels).reshape(xf.shape).to(torch.int8)
        packed = pack_rows_lowbit(codes, 4)  # signed 4-bit -> int32 words
        return Compressed(scheme, tuple(x.shape), x.dtype, group_size, packed,
                          safe.squeeze(-1).float(), d)

    # nf4: nearest non-uniform codebook index per element, per-group absmax scale.
    cb = torch.tensor(NF4_CODEBOOK, device=x.device, dtype=torch.float32)  # [16], in [-1,1]
    safe, xg = _grouped_absmax(x.float(), g, 1.0)  # scale so |x/scale| <= 1
    norm = (xg / safe).clamp_(-1.0, 1.0)
    idx = (norm.unsqueeze(-1) - cb).abs().argmin(dim=-1).reshape(x.shape).to(torch.uint8)
    packed = _pack_nibbles(idx)
    return Compressed(scheme, tuple(x.shape), x.dtype, group_size, packed,
                      safe.squeeze(-1).float(), d)


def decompress_activation(c: Compressed) -> torch.Tensor:
    """Reconstruct the tensor from a `Compressed` payload (returns `c.dtype`)."""
    if c.scheme == "fp16":
        return c.payload.to(c.dtype)

    ng = c.scales.shape[-1] if c.scales.dim() >= 1 else 1
    g = c.d // ng
    scale = c.scales.reshape(*c.scales.shape[:-1], ng, 1)

    if c.scheme == "int8":
        cg = c.payload.float().reshape(*c.shape[:-1], ng, g)
        out = (cg * scale).reshape(c.shape)
        return out.to(c.dtype)

    if c.scheme in ("int4", "int4-had"):
        codes = unpack_rows_lowbit(c.payload, 4, c.d)      # int8 signed [..., d]
        cg = codes.float().reshape(*c.shape[:-1], ng, g)
        out = (cg * scale).reshape(c.shape)
        if c.scheme == "int4-had":
            out = _block_hadamard(out, _rot_block(c.d, c.group_size))  # self-inverse
        return out.to(c.dtype)

    # nf4
    cb = torch.tensor(NF4_CODEBOOK, device=c.payload.device, dtype=torch.float32)
    idx = _unpack_nibbles(c.payload, c.d).long()
    vals = cb[idx].reshape(*c.shape[:-1], ng, g)
    out = (vals * scale).reshape(c.shape)
    return out.to(c.dtype)


def code_entropy_bits(c: Compressed) -> float:
    """Shannon entropy (bits/element) of the quantized code stream — the floor a
    rANS/range coder approaches. `raw_bits / entropy` is the EXTRA ratio an entropy
    coder would add on top of the fixed-width packing (cheap bound, no coder built).
    Returns 16.0 for fp16 (no gain modeled)."""
    if c.scheme == "fp16":
        return 16.0
    if c.scheme in ("int4", "int4-had"):
        codes = unpack_rows_lowbit(c.payload, 4, c.d).long().flatten()
    elif c.scheme == "nf4":
        codes = _unpack_nibbles(c.payload, c.d).long().flatten()
    else:  # int8
        codes = c.payload.long().flatten()
    counts = torch.bincount(codes - codes.min())
    p = counts.float() / counts.sum()
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def raw_bits(scheme: str) -> int:
    return {"fp16": 16, "int8": 8, "int4": 4, "int4-had": 4, "nf4": 4}[scheme]


def reconstruction_report(x: torch.Tensor, c: Compressed) -> dict:
    """cos / rel-L1 of the reconstruction vs the original, plus the ratio metrics."""
    xr = decompress_activation(c).float()
    xf = x.float()
    cos = torch.nn.functional.cosine_similarity(xr.flatten(), xf.flatten(), dim=0).item()
    rel_l1 = (xr - xf).abs().sum().item() / xf.abs().sum().item()
    fp16_bytes = x.numel() * 2
    ratio = fp16_bytes / c.on_wire_bytes
    ent = code_entropy_bits(c)
    entropy_extra = raw_bits(c.scheme) / ent if ent > 0 else 1.0
    return {
        "scheme": c.scheme,
        "group_size": c.group_size,
        "ratio": ratio,                       # fixed-width, vs fp16
        "entropy_bits": ent,                  # bits/element after packing
        "entropy_extra_ratio": entropy_extra,  # what rANS would add on top
        "ratio_with_entropy": ratio * entropy_extra,
        "cos": cos,
        "rel_l1": rel_l1,
        "on_wire_bytes": c.on_wire_bytes,
    }


def effective_transfer_ms(
    nbytes: int, *, link_bytes_per_s: float = PCIE1_X1_BYTES_PER_S
) -> float:
    """Wire time (ms) to move `nbytes` over the link (pure transport, no compute)."""
    return nbytes / link_bytes_per_s * 1e3


def link_speedup(
    x: torch.Tensor, c: Compressed, *, compress_ms: float = 0.0, decompress_ms: float = 0.0,
    link_bytes_per_s: float = PCIE1_X1_BYTES_PER_S,
) -> float:
    """End-to-end speedup vs shipping raw fp16: t_raw / (compress + wire + decompress).
    With compute ~1000x cheaper than the wire, this ~= the compression ratio."""
    t_raw = effective_transfer_ms(x.numel() * 2, link_bytes_per_s=link_bytes_per_s)
    t_comp = compress_ms + effective_transfer_ms(c.on_wire_bytes,
                                                 link_bytes_per_s=link_bytes_per_s) + decompress_ms
    return t_raw / t_comp if t_comp > 0 else math.inf
