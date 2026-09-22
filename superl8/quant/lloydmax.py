# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
#
# The 3-bit Lloyd-Max KV-cache quantizer ("TQ3") is adapted from qengine
# (https://github.com/Haru-neo/qengine, Apache-2.0, (c) Haru-neo) — specifically
# `src/turboquant.cuh`'s block-norm + fixed-Gaussian-codebook recipe. We adapt
# the numeric core to sit ON TOP of the KV path's EXISTING Hadamard rotation
# (superl8/quant/rotation.py); the rotation is NOT reimplemented here. See NOTICE.
# ============================================================================
"""3-bit Lloyd-Max (non-uniform scalar) quantizer for the rotated KV cache.

sm_70 has no int4/int2/int3 dp4a — the only int8 matmul primitive is `__dp4a`.
So, like `lowbit.py`, this is a STORAGE codec: pack the *already-Hadamard-rotated*
K/V at 3 bits (memory + bandwidth win), dequant on read. What this adds over the
uniform `lowbit.py` grid is **non-uniform Lloyd-Max levels** fit to the rotated
coordinate distribution, plus qengine's **per-block L2-norm scale** instead of a
per-group abs-max — this is what makes 3-bit viable where the uniform 3-bit grid
collapses.

Recipe (qengine TQ3, adapted — rotation already applied upstream):
  1. Split the last dim into blocks of `block_size` (128 by default).
  2. Per block: `norm = ||x||_2`; normalize to the unit sphere `xn = x / norm`.
     After the Hadamard rotation each coordinate of a unit block is ~N(0, 1/B),
     so a *fixed* Gaussian Lloyd-Max codebook quantizes it near-optimally and the
     only per-block scalar we store is `norm` (fp32).
  3. Quantize each `xn` coord to the nearest of 2^bits non-uniform levels.
  4. Store 3-bit codes (indices) + the fp32 norm. Dequant: `centroid[code]*norm`.

Storage (block_size=128, 3-bit): 128 codes -> 48 B packed + 4 B norm = 52 B, vs
128*2 B fp16 = 256 B  ->  **4.92x** compression (norm in fp16 -> 5.12x).

`fit=True` runs the Lloyd-Max iteration on the actual normalized data to refine
the codebook (stored per-tensor, 8 fp32 = 32 B, negligible) — use it when the
rotated distribution departs from Gaussian; otherwise the fixed codebook is free.
"""
import functools

import torch

# Standard-normal 8-level (3-bit) Lloyd-Max centroids (unit-variance Gaussian
# MSE-optimal reproduction levels). qengine bakes these * (1/sqrt(128)); we keep
# them in unit-variance form and scale by the block's per-coord sigma = 1/sqrt(B),
# so the same table serves any power-of-two block size (32/64/128).
STDNORMAL_LM8 = (
    -2.152135, -1.343909, -0.756096, -0.245098,
     0.245098,  0.756096,  1.343909,  2.152135,
)


def lloydmax_bits_levels(bits: int) -> int:
    """Number of reproduction levels for a `bits`-bit code (3 -> 8)."""
    assert 2 <= bits <= 8, f"bits must be in [2, 8], got {bits}"
    return 1 << bits


@functools.lru_cache(maxsize=16)
def gaussian_codebook(bits: int, block_size: int, device: str = "cpu"):
    """Fixed Lloyd-Max codebook (ascending) for a unit-L2 block of `block_size`
    coords whose entries are ~N(0, 1/block_size). Cached. 3-bit only for now
    (matches qengine's baked TQ3 table); other widths derive from `STDNORMAL_LM8`
    only at 3-bit."""
    assert bits == 3, "fixed Gaussian codebook is calibrated for 3-bit (TQ3); use fit=True otherwise"
    sigma = block_size ** -0.5
    return torch.tensor([c * sigma for c in STDNORMAL_LM8], dtype=torch.float32, device=device)


def lloyd_max_fit(
    samples: torch.Tensor, levels: int, *, iters: int = 40, tol: float = 1e-6
) -> torch.Tensor:
    """Fit `levels` MSE-optimal reproduction levels to a 1-D `samples` vector via
    the Lloyd-Max iteration (1-D k-means: nearest-centroid assign -> conditional-
    mean update). Returns centroids sorted ascending [levels] fp32.

    Deterministic: initialised from equal-mass quantiles of the data (no RNG), so
    determinism tests hold. Empty clusters are re-seeded from the global quantiles.
    """
    # The fit is a one-time (offline) calibration, not a per-token op, so run its
    # reductions on CPU in float64: CUDA `cumsum`/`scatter_add` reassociate their
    # float reduction order and are RUN-TO-RUN NONDETERMINISTIC, which would break
    # the determinism gate. CPU prefix sums are deterministic. The per-block
    # `quantize_lloydmax` hot path uses only `bucketize` (deterministic on CUDA).
    x = samples.detach().flatten().double().cpu()
    n = x.numel()
    if n == 0:
        return torch.zeros(levels, dtype=torch.float32, device=samples.device)
    qs = torch.linspace(0.5 / levels, 1.0 - 0.5 / levels, levels, dtype=torch.float64)
    qinit = torch.quantile(x, qs)                       # quantile init (ascending)
    xs, _ = torch.sort(x)
    csum = torch.cat([xs.new_zeros(1), xs.cumsum(0)])   # [n+1] prefix sums (fp64, CPU)
    c = qinit.clone()
    prev = None
    for _ in range(iters):
        bounds = 0.5 * (c[1:] + c[:-1])                 # [levels-1] decision boundaries
        e = torch.searchsorted(xs, bounds, right=True)  # [levels-1] split points in xs
        edges = torch.cat([e.new_zeros(1), e, e.new_full((1,), n)])  # [levels+1]
        seg_sum = csum[edges[1:]] - csum[edges[:-1]]    # [levels]
        seg_cnt = (edges[1:] - edges[:-1]).double()
        c_new = torch.where(seg_cnt > 0, seg_sum / seg_cnt.clamp_min(1.0), qinit)
        c_new, _ = torch.sort(c_new)
        if prev is not None and (c_new - prev).abs().max() < tol:
            c = c_new
            break
        prev, c = c, c_new
    return c.to(device=samples.device, dtype=torch.float32)


def _blockify(x: torch.Tensor, block_size: int, dim: int):
    xf = x.float().movedim(dim, -1)
    d = xf.shape[-1]
    assert d % block_size == 0, f"last dim {d} not divisible by block_size {block_size}"
    return xf, d


def quantize_lloydmax(
    x: torch.Tensor,
    *,
    bits: int = 3,
    block_size: int = 128,
    dim: int = -1,
    codebook: torch.Tensor | None = None,
):
    """Block-norm + Lloyd-Max quantize along `dim`.

    Returns ``(codes uint8 indices [same shape as x, in 0..2^bits-1], norm fp32
    [dim reduced to num_blocks], codebook fp32 [2^bits])`` on ``x.device``. If
    ``codebook`` is None the fixed Gaussian codebook is used (qengine TQ3).
    """
    lloydmax_bits_levels(bits)  # validate bit width
    xf, d = _blockify(x, block_size, dim)
    nb = d // block_size
    xg = xf.reshape(*xf.shape[:-1], nb, block_size)                      # [..., nb, B]
    norm = xg.pow(2).sum(-1, keepdim=True).sqrt()                        # [..., nb, 1]
    safe = torch.where(norm == 0, torch.ones_like(norm), norm)
    xn = xg / safe                                                      # unit-L2 block
    cb = (gaussian_codebook(bits, block_size, x.device.type) if codebook is None
          else codebook.to(device=x.device, dtype=torch.float32))
    bounds = 0.5 * (cb[1:] + cb[:-1])
    idx = torch.bucketize(xn.reshape(*xn.shape[:-2], -1).contiguous(), bounds)        # [..., nb*B]
    codes = idx.reshape(xf.shape).movedim(-1, dim).to(torch.uint8)
    norm_out = safe.squeeze(-1).movedim(-1, dim).float()                 # blocks -> dim
    return codes, norm_out, cb


def dequantize_lloydmax(
    codes: torch.Tensor,
    norm: torch.Tensor,
    codebook: torch.Tensor,
    *,
    block_size: int = 128,
    dim: int = -1,
) -> torch.Tensor:
    """Inverse of :func:`quantize_lloydmax` -> fp32."""
    cb = codebook.to(device=codes.device, dtype=torch.float32)
    cf = codes.long().movedim(dim, -1)
    d = cf.shape[-1]
    nb = d // block_size
    vals = cb[cf].reshape(*cf.shape[:-1], nb, block_size)               # centroid lookup
    nf = norm.movedim(dim, -1).float().reshape(*cf.shape[:-1], nb, 1)
    out = (vals * nf).reshape(cf.shape)
    return out.movedim(-1, dim)


def fake_quant_lloydmax(
    x: torch.Tensor,
    *,
    bits: int = 3,
    block_size: int = 128,
    dim: int = -1,
    fit: bool = False,
    fit_iters: int = 40,
) -> torch.Tensor:
    """Quantize then dequantize (round-trip) — the exact tensor the packed cache
    reconstructs. If ``fit`` fit a Lloyd-Max codebook to the unit-normalized data
    (over the whole tensor) instead of the fixed Gaussian one. Used for accuracy
    measurement and the low-bit quality gate."""
    codebook = None
    if fit:
        xf, d = _blockify(x, block_size, dim)
        nb = d // block_size
        xg = xf.reshape(*xf.shape[:-1], nb, block_size)
        norm = xg.pow(2).sum(-1, keepdim=True).sqrt()
        xn = xg / torch.where(norm == 0, torch.ones_like(norm), norm)
        codebook = lloyd_max_fit(xn, lloydmax_bits_levels(bits), iters=fit_iters)
    codes, norm, cb = quantize_lloydmax(
        x, bits=bits, block_size=block_size, dim=dim, codebook=codebook
    )
    deq = dequantize_lloydmax(codes, norm, cb, block_size=block_size, dim=dim)
    return deq.to(x.dtype)


# --------------------------------------------------------------------------
# Bit-packing (unsigned 3-bit indices) — matches qengine `pack_3bit_8`:
# 8 indices (0..7) -> 3 bytes, LSB-first; a `block_size`-block packs to exactly
# block_size*bits/8 bytes with zero waste (proves the 4.92x storage claim).
# --------------------------------------------------------------------------

def pack_indices_lowbit(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned `bits`-bit indices along the last dim into int32 words.
    Requires (D*bits) % 32 == 0 (true for D in {32,64,128} at 3-bit)."""
    d = indices.shape[-1]
    assert (d * bits) % 32 == 0, f"D*bits ({d}*{bits}) must be a multiple of 32"
    u = indices.to(torch.int64) & ((1 << bits) - 1)
    bit = (u.unsqueeze(-1) >> torch.arange(bits, device=indices.device)) & 1
    stream = bit.reshape(*indices.shape[:-1], d * bits)
    words = stream.reshape(*indices.shape[:-1], (d * bits) // 32, 32)
    wt = torch.arange(32, device=indices.device, dtype=torch.int64).exp2().to(torch.int64)
    packed = (words.to(torch.int64) * wt).sum(-1)
    return (packed - (1 << 32) * (packed >= (1 << 31))).to(torch.int32)


def unpack_indices_lowbit(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Inverse of :func:`pack_indices_lowbit` -> uint8 indices [..., d] (NO sign
    extension — these are unsigned codebook indices, unlike lowbit's signed codes)."""
    p = packed.to(torch.int64) & 0xFFFFFFFF
    bit = (p.unsqueeze(-1) >> torch.arange(32, device=packed.device)) & 1
    stream = bit.reshape(*packed.shape[:-1], d * bits).reshape(*packed.shape[:-1], d, bits)
    u = (stream << torch.arange(bits, device=packed.device)).sum(-1)     # 0..2^b-1
    return u.to(torch.uint8)
