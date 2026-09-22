# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Sub-INT8 (2/3/4-bit) symmetric quantization for the KV cache.

sm_70 has no int4/int2 dp4a — the ONLY int8 matmul primitive is `__dp4a`. So
low-bit here is a STORAGE codec: pack K/V at 2/3/4 bits (memory + bandwidth win),
and unpack to int8 for the dp4a. This module is the numeric core (symmetric
per-group RTN); real bit-packing and the packed-cache kernel build on it.

Symmetric b-bit: levels = 2^(b-1) - 1 (4-bit -> ±7, 3-bit -> ±3, 2-bit -> ±1,
i.e. ternary). Per-GROUP scales (a group of `group_size` consecutive elements
along `dim` shares one fp32 scale) — finer grouping buys accuracy at low bits,
which is what makes 4-bit K viable alongside the Hadamard rotation.
"""
import torch

# QLoRA NF4 codebook: 16 non-uniform 4-bit levels at the quantiles of a standard
# normal, in [-1, 1]. Gaussian activations (V) quantize far better on this than a
# uniform grid. Index order matches the kernel's DEC_NF4 table.
NF4_CODEBOOK = (
    -1.0, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343,
    -0.09105004, 0.0, 0.07958030, 0.16093020, 0.24611230, 0.33791524,
    0.44070983, 0.562617, 0.72295684, 1.0,
)


def lowbit_levels(bits: int) -> int:
    """Max magnitude of a symmetric b-bit code (4->7, 3->3, 2->1)."""
    assert 2 <= bits <= 8, f"bits must be in [2, 8], got {bits}"
    return (1 << (bits - 1)) - 1


def quantize_lowbit(
    x: torch.Tensor, bits: int, *, dim: int = -1, group_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group b-bit quantize along `dim`.

    Returns (codes int8 in [-levels, +levels] (same shape as x), scales fp32
    with the `dim` axis reduced to num_groups = dim_len // group_size).
    `group_size=None` means one scale for the whole `dim` axis.
    """
    levels = lowbit_levels(bits)
    xf = x.float().movedim(dim, -1)
    d = xf.shape[-1]
    g = d if group_size is None else group_size
    assert d % g == 0, f"dim {d} not divisible by group_size {g}"
    xg = xf.reshape(*xf.shape[:-1], d // g, g)
    scale = xg.abs().amax(dim=-1, keepdim=True) / levels          # [..., d//g, 1]
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.round(xg / safe).clamp_(-levels, levels).reshape(xf.shape)
    codes = codes.movedim(-1, dim).to(torch.int8)
    scale_out = safe.squeeze(-1).movedim(-1, dim)                 # groups axis -> dim
    return codes, scale_out.float()


def dequantize_lowbit(
    codes: torch.Tensor, scale: torch.Tensor, *, dim: int = -1, group_size: int | None = None
) -> torch.Tensor:
    """Inverse of quantize_lowbit -> fp32."""
    cf = codes.float().movedim(dim, -1)
    d = cf.shape[-1]
    g = d if group_size is None else group_size
    ng = d // g
    sf = scale.movedim(dim, -1).reshape(*cf.shape[:-1], ng, 1)     # [..., ng, 1]
    out = (cf.reshape(*cf.shape[:-1], ng, g) * sf).reshape(cf.shape)
    return out.movedim(-1, dim).to(codes.device)


def pack_rows_lowbit(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-pack the last dim (D signed b-bit codes) into int32 words.

    For D a multiple of 32/gcd(b,32) — true for all our head dims {32,64,128},
    where D*bits is always a multiple of 32 — a full row packs into EXACTLY
    D*bits/32 int32 words with zero waste and no cross-row raggedness. That is
    the "align the tile with the packing period" point: 6-bit -> 16 values/3
    words, 5-bit -> 32/5, 3-bit -> 32/3; a row is a whole number of periods.
    Codes are two's-complement b-bit (LSB-first within the stream). Reference
    implementation (the CUDA kernel does the same shifts); proves losslessness.
    """
    d = codes.shape[-1]
    assert (d * bits) % 32 == 0, f"D*bits ({d}*{bits}) must be a multiple of 32"
    u = (codes.to(torch.int64) & ((1 << bits) - 1))                    # b-bit pattern
    bit = (u.unsqueeze(-1) >> torch.arange(bits, device=codes.device)) & 1  # [...,D,b] LSB-first
    stream = bit.reshape(*codes.shape[:-1], d * bits)                  # [..., D*b]
    words = stream.reshape(*codes.shape[:-1], (d * bits) // 32, 32)
    wt = (torch.arange(32, device=codes.device, dtype=torch.int64)).exp2().to(torch.int64)
    packed = (words.to(torch.int64) * wt).sum(-1)                      # [..., D*b/32]
    return (packed - (1 << 32) * (packed >= (1 << 31))).to(torch.int32)


def unpack_rows_lowbit(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Inverse of pack_rows_lowbit -> int8 codes [..., d] (sign-extended)."""
    p = packed.to(torch.int64) & 0xFFFFFFFF
    bit = (p.unsqueeze(-1) >> torch.arange(32, device=packed.device)) & 1  # [...,W,32]
    stream = bit.reshape(*packed.shape[:-1], d * bits).reshape(*packed.shape[:-1], d, bits)
    u = (stream << torch.arange(bits, device=packed.device)).sum(-1)   # [..., d], 0..2^b-1
    v = u - (1 << bits) * (u >= (1 << (bits - 1)))                      # sign-extend
    return v.to(torch.int8)


# ── W3A8 decode weight codec (uniform 3-bit, Q3_K-style bit-planes) ──────────
# The gemm_decode_w3a8 kernel consumes weights as [N, (K//32)*3] int32 bit-planes
# with per-group fp32 scales. Unlike the KV codec above (symmetric ±3), the MLP
# weight path uses the FULL asymmetric 3-bit range [-4, 3] (scale = max|w|/4),
# matching the validated issue #181 spike (cos 0.99996). Per 32-value packing
# group: (qs0, qs1, hp) laid out byte-lane-aligned for the kernel's 3-op unpack
#   vil=(qs>>2i)&0x03030303; vih=((hp>>i)<<2)&0x04040404; vi=__vsubss4(vil,vih).


def quantize_w3a8(
    w: torch.Tensor, *, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric uniform 3-bit weight quantize, per output-channel per-group.

    ``w`` [N, K] fp -> (codes int8 [N, K] in [-4, 3], scale fp32 [N, K//group_size]).
    ``scale = max|w_group| / 4`` (full 3-bit two's-complement range), matching the
    ``gemm_decode_w3a8`` kernel's ``(high<<2|low2)-4`` unpack.
    """
    N, K = w.shape
    g = group_size
    assert K % g == 0, f"K {K} not divisible by group_size {g}"
    wf = w.float().reshape(N, K // g, g)
    scale = wf.abs().amax(dim=-1, keepdim=True) / 4.0                 # [N, K//g, 1]
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.round(wf / safe).clamp_(-4, 3).reshape(N, K).to(torch.int8)
    return codes, safe.squeeze(-1).float()


def dequantize_w3a8(
    codes: torch.Tensor, scale: torch.Tensor, *, group_size: int
) -> torch.Tensor:
    """Inverse of :func:`quantize_w3a8` -> fp32 [N, K]."""
    N, K = codes.shape
    g = group_size
    cf = codes.float().reshape(N, K // g, g)
    sf = scale.float().reshape(N, K // g, 1)
    return (cf * sf).reshape(N, K)


def pack_w3a8_bitplanes(codes: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit codes [N, K] (in [-4, 3]) into [N, (K//32)*3] int32 bit-planes.

    Mirrors the kernel's per-group (qs0, qs1, hp) layout exactly (llama.cpp Q3_K
    bit-plane split, MIT — adapted for uniform 3-bit; study/attribution only). For
    value j in a 32-group: byte lane b = j&3, sub-index s = j>>2; u = code+4 in
    [0,7]; low2 = u&3 -> qs0/qs1, inverted high bit -> hp.
    """
    N, K = codes.shape
    assert K % 32 == 0, f"K {K} must be a multiple of 32"
    ng = K // 32
    c = codes.to(torch.int64).reshape(N, ng, 32)
    u = c + 4                                                          # [0,7]
    low2 = u & 3
    hpbit = 1 - ((u >> 2) & 1)                                         # inverted high bit
    qs0 = torch.zeros(N, ng, dtype=torch.int64, device=codes.device)
    qs1 = torch.zeros(N, ng, dtype=torch.int64, device=codes.device)
    hp = torch.zeros(N, ng, dtype=torch.int64, device=codes.device)
    for j in range(32):
        b, s = j & 3, j >> 2
        if s < 4:
            qs0 |= low2[:, :, j] << (8 * b + 2 * s)
        else:
            qs1 |= low2[:, :, j] << (8 * b + 2 * (s - 4))
        hp |= hpbit[:, :, j] << (8 * b + s)
    packed = torch.stack([qs0, qs1, hp], dim=-1).reshape(N, ng * 3)   # [N,(K/32)*3]
    packed = packed & 0xFFFFFFFF
    packed = packed - (1 << 32) * (packed >= (1 << 31))               # -> signed int32 bits
    return packed.to(torch.int32)


def unpack_w3a8_bitplanes(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Inverse of :func:`pack_w3a8_bitplanes` -> int8 codes [N, K] in [-4, 3].

    Reference (host) unpack proving the packing is lossless and matches the
    kernel's ``(high<<2|low2)-4`` decode; used by the prefill/CPU fallback and
    the correctness tests.
    """
    N = packed.shape[0]
    ng = K // 32
    p = packed.to(torch.int64).reshape(N, ng, 3) & 0xFFFFFFFF
    qs0, qs1, hp = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    codes = torch.empty(N, ng, 32, dtype=torch.int8, device=packed.device)
    for j in range(32):
        b, s = j & 3, j >> 2
        vl = qs0 if s < 4 else qs1
        ss = s if s < 4 else s - 4
        low2 = (vl >> (8 * b + 2 * ss)) & 3
        hpbit = (hp >> (8 * b + s)) & 1
        high = 1 - hpbit                                   # invert back
        codes[:, :, j] = ((high << 2) | low2).to(torch.int8) - 4
    return codes.reshape(N, K)


def fake_quant_lowbit(
    x: torch.Tensor, bits: int, *, dim: int = -1, group_size: int | None = None
) -> torch.Tensor:
    """Quantize then dequantize (round-trip) — the exact tensor the packed cache
    reconstructs at int8 dequant. Self-contained (no external-scale layout
    dependence); used for accuracy measurement and the low-bit gate."""
    levels = lowbit_levels(bits)
    xf = x.float().movedim(dim, -1)
    d = xf.shape[-1]
    g = d if group_size is None else group_size
    assert d % g == 0, f"dim {d} not divisible by group_size {g}"
    xg = xf.reshape(*xf.shape[:-1], d // g, g)
    scale = xg.abs().amax(dim=-1, keepdim=True) / levels
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    deq = (torch.round(xg / safe).clamp_(-levels, levels) * safe).reshape(xf.shape)
    return deq.movedim(-1, dim).to(x.dtype)
