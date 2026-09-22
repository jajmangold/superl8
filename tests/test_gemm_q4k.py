# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Fused GGUF Q4_K -> dp4a GEMM (native-GGUF-on-the-fly): the weights stay in
their native Q4_K super-block layout and the kernel unpacks each sub-block to
int8 in-kernel + runs __dp4a, honoring the 6-bit per-sub-block scales/mins
exactly. See csrc/docs/gguf-fused-kquant-dp4a.md.

Reference Q4_K enc/dec (this file, pure-torch/numpy) matches llama.cpp's block
layout (ggml-common.h:408-419, get_scale_min_k4 convert.cu:195-202, and the
qs low/high-nibble interleave of dequantize_row_q4_K). The kernel FIDELITY gate
compares the fused output to a dequant->matmul of the SAME bytes at SQNR>=40dB
(only fp-store rounding separates them); the ORACLE gate compares a Q4_K quant
of a real fp weight to the fp32 matmul at Q4_K's intrinsic bar. int8 gates use
SQNR/cos/rel-L1, never allclose (AGENTS.md).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import superl8
from superl8.quant.core import quantize_int8_rowwise

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.tolerances import assert_int8_quality, cos_sim  # noqa: E402

QK_K = 256
Q4K_TYPE_SIZE = 144  # 2(d)+2(dmin)+12(scales)+128(qs)
Q5K_TYPE_SIZE = 176  # 2+2+12+32(qh)+128(qs)
Q6K_TYPE_SIZE = 210  # 128(ql)+64(qh)+16(scales int8)+2(d)
Q3K_TYPE_SIZE = 110  # 32(hmask)+64(qs)+12(scales)+2(d)
Q2K_TYPE_SIZE = 84   # 16(scales)+64(qs)+2(d)+2(dmin)
_LTX = os.environ.get(
    "FNI8_LTX_GGUF",
    "",
)
_Q27B = os.environ.get(
    "FNI8_Q27B_GGUF",
    "",
)


# ---------------------------------------------------------------------------
# Pure-torch Q4_K encoder + decoder (reference oracle). Produces the exact
# native GGUF byte layout the kernel consumes, plus its fp32 dequant.
# ---------------------------------------------------------------------------
def _pack_scales_min_k4(sc, m):
    """Inverse of get_scale_min_k4: 8x 6-bit scales `sc` + 8x 6-bit mins `m`
    (each [...,8] uint8 in [0,63]) -> 12 packed bytes [...,12] uint8."""
    sc = sc.astype(np.uint16)
    m = m.astype(np.uint16)
    q = np.zeros(sc.shape[:-1] + (12,), dtype=np.uint16)
    # j<4: low6 of q[0..3]=sc[0..3], low6 of q[4..7]=m[0..3];
    #      bits6-7 of q[0..3]=high2 of sc[4..7], bits6-7 of q[4..7]=high2 of m[4..7]
    q[..., 0:4] = (sc[..., 0:4] & 0x3F) | (((sc[..., 4:8] >> 4) & 0x3) << 6)
    q[..., 4:8] = (m[..., 0:4] & 0x3F) | (((m[..., 4:8] >> 4) & 0x3) << 6)
    # j>=4: q[8..11] low nibble = sc[4..7]&0xF, high nibble = m[4..7]&0xF
    q[..., 8:12] = (sc[..., 4:8] & 0xF) | ((m[..., 4:8] & 0xF) << 4)
    return q.astype(np.uint8)


def _unpack_scales_min_k4(q):
    """get_scale_min_k4 (convert.cu:195-202), vectorized: 12 bytes -> (sc,m) [...,8]."""
    q = q.astype(np.uint16)
    sc = np.zeros(q.shape[:-1] + (8,), dtype=np.uint16)
    m = np.zeros(q.shape[:-1] + (8,), dtype=np.uint16)
    sc[..., 0:4] = q[..., 0:4] & 0x3F
    m[..., 0:4] = q[..., 4:8] & 0x3F
    sc[..., 4:8] = (q[..., 8:12] & 0xF) | ((q[..., 0:4] >> 6) << 4)
    m[..., 4:8] = (q[..., 8:12] >> 4) | ((q[..., 4:8] >> 6) << 4)
    return sc.astype(np.uint8), m.astype(np.uint8)


def q4k_quantize(w: torch.Tensor):
    """Quantize an fp weight [N,K] (K%256==0) to native Q4_K bytes.

    Returns (bytes uint8 [N, (K//256)*144], deq fp32 [N,K]). Asymmetric per-32
    sub-block RTN (x ~= d*sc*q - dmin*m, q in [0,15]) with 6-bit super-block
    quantization of the sub-scales/mins, matching llama.cpp's make_qkx2 form.
    """
    N, K = w.shape
    assert K % QK_K == 0, "Q4_K needs K % 256 == 0"
    nsb = K // QK_K
    x = w.detach().float().cpu().numpy().reshape(N, nsb, 8, 32)  # [N,nsb,sub,32]

    xmax = x.max(axis=-1)
    xmin_nonneg = np.maximum(0.0, -x.min(axis=-1))  # non-negative "min" offset
    scale = (xmax + xmin_nonneg) / 15.0             # per sub-block [N,nsb,8]
    scale = np.where(scale == 0, 1.0, scale)
    q = np.rint((x + xmin_nonneg[..., None]) / scale[..., None])
    q = np.clip(q, 0, 15).astype(np.int64)          # 4-bit quants [N,nsb,8,32]

    # 6-bit super-quant of the 8 per-sub scales and mins.
    d = scale.max(axis=-1) / 63.0                   # [N,nsb]
    dmin = xmin_nonneg.max(axis=-1) / 63.0
    d = np.where(d == 0, 1.0, d)
    dmin = np.where(dmin == 0, 1.0, dmin)
    sc6 = np.clip(np.rint(scale / d[..., None]), 0, 63).astype(np.uint8)   # [N,nsb,8]
    m6 = np.clip(np.rint(xmin_nonneg / dmin[..., None]), 0, 63).astype(np.uint8)
    d16 = d.astype(np.float16)
    dmin16 = dmin.astype(np.float16)

    # ---- assemble native bytes ----
    blk = np.zeros((N, nsb, Q4K_TYPE_SIZE), dtype=np.uint8)
    blk[:, :, 0:2] = np.frombuffer(np.ascontiguousarray(d16).tobytes(), dtype=np.uint8).reshape(N, nsb, 2)
    blk[:, :, 2:4] = np.frombuffer(np.ascontiguousarray(dmin16).tobytes(), dtype=np.uint8).reshape(N, nsb, 2)
    blk[:, :, 4:16] = _pack_scales_min_k4(sc6, m6)
    # qs interleave: byte[g*32+l].low = q[2g][l], .high = q[2g+1][l], g=0..3, l=0..31
    qs = np.zeros((N, nsb, 128), dtype=np.uint8)
    for g in range(4):
        lo = q[:, :, 2 * g, :].astype(np.uint8) & 0xF        # [N,nsb,32]
        hi = q[:, :, 2 * g + 1, :].astype(np.uint8) & 0xF
        qs[:, :, g * 32:(g + 1) * 32] = lo | (hi << 4)
    blk[:, :, 16:144] = qs

    # ---- dequant (exact reconstruction from the stored bytes) ----
    dsc = d16.astype(np.float32)[..., None] * sc6.astype(np.float32)   # [N,nsb,8]
    dm = dmin16.astype(np.float32)[..., None] * m6.astype(np.float32)
    deq = dsc[..., None] * q.astype(np.float32) - dm[..., None]        # [N,nsb,8,32]
    deq = deq.reshape(N, K)
    return (
        torch.from_numpy(blk.reshape(N, nsb * Q4K_TYPE_SIZE)).contiguous(),
        torch.from_numpy(deq).contiguous(),
    )


def q4k_dequantize_bytes(blk_bytes: np.ndarray, N: int, K: int) -> np.ndarray:
    """Dequant native Q4_K bytes [N, nsb*144] -> fp32 [N,K] (city96/llama math).
    Used for the real-LTX gate where we don't own the encoder."""
    nsb = K // QK_K
    b = blk_bytes.reshape(N, nsb, Q4K_TYPE_SIZE)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(N, nsb)
    dmin = b[:, :, 2:4].copy().view(np.float16).astype(np.float32).reshape(N, nsb)
    sc6, m6 = _unpack_scales_min_k4(b[:, :, 4:16])
    qs = b[:, :, 16:144]
    q = np.zeros((N, nsb, 8, 32), dtype=np.float32)
    for g in range(4):
        seg = qs[:, :, g * 32:(g + 1) * 32]
        q[:, :, 2 * g, :] = (seg & 0xF).astype(np.float32)
        q[:, :, 2 * g + 1, :] = (seg >> 4).astype(np.float32)
    dsc = d[..., None] * sc6.astype(np.float32)
    dm = dmin[..., None] * m6.astype(np.float32)
    return (dsc[..., None] * q - dm[..., None]).reshape(N, K)


def q5k_quantize(w: torch.Tensor):
    """Quantize fp [N,K] (K%256==0) to native Q5_K bytes -> (bytes uint8
    [N,(K//256)*176], deq fp32 [N,K]). Same 6-bit sub-scale/min affine as Q4_K
    (x ~= d*sc*q - dmin*m) but q is 5-bit: low 4 in qs (Q4_K interleave), 5th bit
    in qh[32] (bit j of qh[l] = high bit of sub-block j, position l)."""
    N, K = w.shape
    assert K % QK_K == 0
    nsb = K // QK_K
    x = w.detach().float().cpu().numpy().reshape(N, nsb, 8, 32)
    xmax = x.max(axis=-1)
    xmin_nn = np.maximum(0.0, -x.min(axis=-1))
    scale = (xmax + xmin_nn) / 31.0
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.rint((x + xmin_nn[..., None]) / scale[..., None]), 0, 31).astype(np.int64)
    d = scale.max(axis=-1) / 63.0
    dmin = xmin_nn.max(axis=-1) / 63.0
    d = np.where(d == 0, 1.0, d)
    dmin = np.where(dmin == 0, 1.0, dmin)
    sc6 = np.clip(np.rint(scale / d[..., None]), 0, 63).astype(np.uint8)
    m6 = np.clip(np.rint(xmin_nn / dmin[..., None]), 0, 63).astype(np.uint8)
    d16, dmin16 = d.astype(np.float16), dmin.astype(np.float16)

    blk = np.zeros((N, nsb, Q5K_TYPE_SIZE), dtype=np.uint8)
    blk[:, :, 0:2] = np.frombuffer(np.ascontiguousarray(d16).tobytes(), np.uint8).reshape(N, nsb, 2)
    blk[:, :, 2:4] = np.frombuffer(np.ascontiguousarray(dmin16).tobytes(), np.uint8).reshape(N, nsb, 2)
    blk[:, :, 4:16] = _pack_scales_min_k4(sc6, m6)
    low4 = (q & 0xF).astype(np.uint8)          # [N,nsb,8,32]
    high1 = ((q >> 4) & 1).astype(np.uint8)
    qh = np.zeros((N, nsb, 32), dtype=np.uint8)  # qh[l] bit j = high1[j,l]
    for j in range(8):
        qh |= (high1[:, :, j, :] << j)
    blk[:, :, 16:48] = qh
    qs = np.zeros((N, nsb, 128), dtype=np.uint8)
    for g in range(4):
        qs[:, :, g * 32:(g + 1) * 32] = low4[:, :, 2 * g, :] | (low4[:, :, 2 * g + 1, :] << 4)
    blk[:, :, 48:176] = qs

    dsc = d16.astype(np.float32)[..., None] * sc6.astype(np.float32)
    dm = dmin16.astype(np.float32)[..., None] * m6.astype(np.float32)
    deq = (dsc[..., None] * q.astype(np.float32) - dm[..., None]).reshape(N, K)
    return (torch.from_numpy(blk.reshape(N, nsb * Q5K_TYPE_SIZE)).contiguous(),
            torch.from_numpy(deq).contiguous())


def q6k_quantize(w: torch.Tensor):
    """Quantize fp [N,K] (K%256==0) to native Q6_K bytes -> (bytes uint8
    [N,(K//256)*210], deq fp32 [N,K]). SYMMETRIC per-16 sub-block:
    y = d*sc_i*(q-32), q in [0,63], sc_i signed int8. Packs the ql/qh quadrant
    interleave of dequantize_row_q6_K (natural per-16 sub-block i == scales[i])."""
    N, K = w.shape
    assert K % QK_K == 0
    nsb = K // QK_K
    x = w.detach().float().cpu().numpy().reshape(N, nsb, 16, 16)  # 16 sub-blocks of 16
    a = np.abs(x).max(axis=-1)                    # per-16 absmax [N,nsb,16]
    scale_f = np.where(a == 0, 1.0, a / 32.0)     # so (q-32) spans the range
    d = scale_f.max(axis=-1) / 127.0              # super-block fp16 scale [N,nsb]
    d = np.where(d == 0, 1.0, d)
    sc = np.clip(np.rint(scale_f / d[..., None]), -128, 127).astype(np.int8)  # [N,nsb,16]
    d16 = d.astype(np.float16)
    recon_scale = d16.astype(np.float32)[..., None] * sc.astype(np.float32)   # [N,nsb,16]
    recon_scale_safe = np.where(recon_scale == 0, 1.0, recon_scale)
    q = np.clip(np.rint(x / recon_scale_safe[..., None]) + 32, 0, 63).astype(np.int64)  # [N,nsb,16,16]

    deq = (recon_scale[..., None] * (q.astype(np.float32) - 32.0)).reshape(N, K)

    # pack: natural element e = g6*128 + qd*32 + l -> reshape [N,nsb,2(g6),4(qd),32(l)]
    qc = q.reshape(N, nsb, 2, 4, 32).astype(np.uint8)
    ql = np.zeros((N, nsb, 128), dtype=np.uint8)
    qh = np.zeros((N, nsb, 64), dtype=np.uint8)
    for g6 in range(2):
        lo0, lo1 = qc[:, :, g6, 0, :] & 0xF, qc[:, :, g6, 1, :] & 0xF
        hi2, hi3 = qc[:, :, g6, 2, :] & 0xF, qc[:, :, g6, 3, :] & 0xF
        ql[:, :, g6 * 64 + 0:g6 * 64 + 32] = lo0 | (hi2 << 4)   # ql[l]: qd0 low, qd2 high
        ql[:, :, g6 * 64 + 32:g6 * 64 + 64] = lo1 | (hi3 << 4)  # ql[l+32]: qd1 low, qd3 high
        qh[:, :, g6 * 32:g6 * 32 + 32] = (
            ((qc[:, :, g6, 0, :] >> 4) & 3)
            | (((qc[:, :, g6, 1, :] >> 4) & 3) << 2)
            | (((qc[:, :, g6, 2, :] >> 4) & 3) << 4)
            | (((qc[:, :, g6, 3, :] >> 4) & 3) << 6)
        )
    blk = np.zeros((N, nsb, Q6K_TYPE_SIZE), dtype=np.uint8)
    blk[:, :, 0:128] = ql
    blk[:, :, 128:192] = qh
    blk[:, :, 192:208] = sc.view(np.uint8).reshape(N, nsb, 16)
    blk[:, :, 208:210] = np.frombuffer(np.ascontiguousarray(d16).tobytes(), np.uint8).reshape(N, nsb, 2)
    return (torch.from_numpy(blk.reshape(N, nsb * Q6K_TYPE_SIZE)).contiguous(),
            torch.from_numpy(deq).contiguous())


def q2k_quantize(w: torch.Tensor):
    """Quantize fp [N,K] (K%256==0) to native Q2_K bytes -> (bytes uint8
    [N,(K//256)*84], deq fp32 [N,K]). Affine per-16 sub-block: y = d*sc*q - dmin*m,
    q in [0,3], sc/m 4-bit. Block: scales[16] qs[64] d dmin (ggml-common.h:379-390)."""
    N, K = w.shape
    assert K % QK_K == 0
    nsb = K // QK_K
    x = w.detach().float().cpu().numpy().reshape(N, nsb, 16, 16)  # 16 sub-blocks of 16
    xmax = x.max(-1)
    xmin = np.maximum(0.0, -x.min(-1))
    scale = np.where((xmax + xmin) == 0, 1.0, (xmax + xmin) / 3.0)
    q = np.clip(np.rint((x + xmin[..., None]) / scale[..., None]), 0, 3).astype(np.int64)
    d = np.where(scale.max(-1) == 0, 1.0, scale.max(-1) / 15.0)
    dmin = np.where(xmin.max(-1) == 0, 1.0, xmin.max(-1) / 15.0)
    sc4 = np.clip(np.rint(scale / d[..., None]), 0, 15).astype(np.uint8)     # [N,nsb,16]
    m4 = np.clip(np.rint(xmin / dmin[..., None]), 0, 15).astype(np.uint8)
    d16, dmin16 = d.astype(np.float16), dmin.astype(np.float16)
    blk = np.zeros((N, nsb, 84), np.uint8)
    blk[:, :, 0:16] = (sc4 | (m4 << 4))                                       # scales
    qs = np.zeros((N, nsb, 64), np.uint8)
    for g in range(2):
        for half in range(2):
            for s4 in range(4):
                isb = g * 8 + 2 * s4 + half
                off = 32 * g + half * 16
                qs[:, :, off:off + 16] |= (q[:, :, isb, :].astype(np.uint8) << (2 * s4))
    blk[:, :, 16:80] = qs
    blk[:, :, 80:82] = np.frombuffer(np.ascontiguousarray(d16).tobytes(), np.uint8).reshape(N, nsb, 2)
    blk[:, :, 82:84] = np.frombuffer(np.ascontiguousarray(dmin16).tobytes(), np.uint8).reshape(N, nsb, 2)
    dsc = d16.astype(np.float32)[..., None] * sc4.astype(np.float32)
    dm = dmin16.astype(np.float32)[..., None] * m4.astype(np.float32)
    deq = (dsc[..., None] * q.astype(np.float32) - dm[..., None]).reshape(N, K)
    return (torch.from_numpy(blk.reshape(N, nsb * 84)).contiguous(),
            torch.from_numpy(deq).contiguous())


def q3k_quantize(w: torch.Tensor):
    """Quantize fp [N,K] (K%256==0) to native Q3_K bytes -> (bytes uint8
    [N,(K//256)*110], deq fp32 [N,K]). SYMMETRIC per-16: y = d*scale*(q-4), q in
    [0,7], scale signed 6-bit. Block: hmask[32] qs[64] scales[12] d
    (ggml-common.h:396-402). Non-negative scale subset (stored 6-bit in [32,63])."""
    N, K = w.shape
    assert K % QK_K == 0
    nsb = K // QK_K
    x = w.detach().float().cpu().numpy().reshape(N, nsb, 16, 16)
    amax = np.abs(x).max(-1)
    scale_f = np.where(amax == 0, 1.0, amax / 4.0)             # (q-4) spans [-4,3]
    d = np.where(scale_f.max(-1) == 0, 1.0, scale_f.max(-1) / 31.0)
    sc6 = np.clip(np.rint(scale_f / d[..., None]), 0, 31).astype(np.int64)    # signed scale [0,31]
    d16 = d.astype(np.float16)
    recon = d16.astype(np.float32)[..., None] * sc6.astype(np.float32)        # [N,nsb,16]
    recon_safe = np.where(recon == 0, 1.0, recon)
    q = np.clip(np.rint(x / recon_safe[..., None]) + 4, 0, 7).astype(np.int64)  # [N,nsb,16,16]
    deq = (recon[..., None] * (q.astype(np.float32) - 4.0)).reshape(N, K)

    stored = (sc6 + 32).astype(np.uint16)                     # 6-bit stored value [32,63]
    scales = np.zeros((N, nsb, 12), np.uint16)
    for isb in range(16):
        scales[:, :, isb % 8] |= (stored[:, :, isb] & 0xF) << (4 * (isb // 8))
        scales[:, :, 8 + isb % 4] |= ((stored[:, :, isb] >> 4) & 3) << (2 * (isb // 4))
    hmask = np.zeros((N, nsb, 32), np.uint16)
    qs = np.zeros((N, nsb, 64), np.uint16)
    low2 = (q & 3).astype(np.uint16)
    hbit = ((q >> 2) & 1).astype(np.uint16)
    for g in range(2):
        for half in range(2):
            for s4 in range(4):
                isb = g * 8 + 2 * s4 + half
                m_shift = g * 4 + s4
                qs[:, :, 32 * g + half * 16:32 * g + half * 16 + 16] |= (low2[:, :, isb, :] << (2 * s4))
                hmask[:, :, half * 16:half * 16 + 16] |= (hbit[:, :, isb, :] << m_shift)
    blk = np.zeros((N, nsb, 110), np.uint8)
    blk[:, :, 0:32] = hmask.astype(np.uint8)
    blk[:, :, 32:96] = qs.astype(np.uint8)
    blk[:, :, 96:108] = scales.astype(np.uint8)
    blk[:, :, 108:110] = np.frombuffer(np.ascontiguousarray(d16).tobytes(), np.uint8).reshape(N, nsb, 2)
    return (torch.from_numpy(blk.reshape(N, nsb * 110)).contiguous(),
            torch.from_numpy(deq).contiguous())


# ---------------------------------------------------------------------------
# Round-trip sanity: our encoder/decoder self-consistency (no kernel).
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_q4k_reference_roundtrip_selfconsistent():
    torch.manual_seed(0)
    w = torch.randn(8, 512) * 0.1
    blk, deq = q4k_quantize(w)
    deq2 = q4k_dequantize_bytes(blk.numpy(), 8, 512)
    np.testing.assert_allclose(deq.numpy(), deq2, rtol=0, atol=1e-3)


# ---------------------------------------------------------------------------
# THE gate: fused kernel must equal the dequant->matmul of the SAME bytes.
# ---------------------------------------------------------------------------
Q4K_SHAPES = [
    (1, 64, 256), (7, 128, 256), (64, 64, 512), (257, 128, 768),
    (16, 4864, 896 + (256 - 896 % 256)),  # K padded to %256
    (2048, 896, 4096), (33, 512, 1024),
    (5, 130, 256),   # ragged N (not tile multiple)
    (200, 96, 512),  # ragged N below tile
]


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", Q4K_SHAPES)
def test_gemm_q4k_reproduces_dequant(device, m, n, k):
    """Fused Q4_K dp4a == exact int-affine dequant matmul (only fp store rounds)."""
    torch.manual_seed(m * 31 + n)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    blk, deq = q4k_quantize(w)
    blk = blk.to(device)
    deq = deq.to(device)
    x_i8, x_scale = quantize_int8_rowwise(x)
    ref = (x_i8.float() @ deq.t()) * x_scale        # [M,N], x_scale [M,1]
    y = superl8._C.gemm_q4k(x_i8, x_scale.squeeze(-1).contiguous(), blk, torch.float16)
    assert y.shape == (m, n) and y.dtype == torch.float16
    assert_int8_quality(y, ref, min_cos=0.999, max_rel_l1=0.01, min_sqnr_db=40.0,
                        what=f"gemm_q4k dequant-exact {m}x{n}x{k}")


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", [(7, 128, 256), (64, 64, 512)])
def test_gemm_q4k_bf16_output(device, m, n, k):
    torch.manual_seed(1)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    blk, deq = q4k_quantize(w)
    blk, deq = blk.to(device), deq.to(device)
    x_i8, x_scale = quantize_int8_rowwise(x)
    ref = (x_i8.float() @ deq.t()) * x_scale
    y = superl8._C.gemm_q4k(x_i8, x_scale.squeeze(-1).contiguous(), blk, torch.bfloat16)
    assert y.dtype == torch.bfloat16
    assert_int8_quality(y, ref, min_cos=0.999, max_rel_l1=0.01, min_sqnr_db=38.0,
                        what=f"gemm_q4k bf16 {m}x{n}x{k}")


@pytest.mark.correctness
def test_gemm_q4k_matches_fp32_oracle(device):
    """Q4_K of a real fp weight vs the fp32 matmul — Q4_K's intrinsic error bar."""
    torch.manual_seed(2)
    m, n, k = 128, 512, 1024
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    blk, _ = q4k_quantize(w)
    y = superl8.linear_q4k(x, blk.to(device))
    ref = x.float() @ w.float().to(device).t()
    # Q4_K (4.5 bpw, per-32 affine sub-scales) beats uniform int4; cos+SQNR gate.
    assert_int8_quality(y, ref, min_cos=0.99, max_rel_l1=0.12, min_sqnr_db=20.0,
                        what="gemm_q4k fp32-oracle")


@pytest.mark.correctness
def test_gemm_q4k_deterministic(device):
    torch.manual_seed(3)
    x = torch.randn(64, 512, device=device, dtype=torch.float16)
    w = torch.randn(256, 512, dtype=torch.float16) * 0.1
    blk, _ = q4k_quantize(w)
    blk = blk.to(device)
    r0 = superl8.linear_q4k(x, blk)
    for _ in range(3):
        assert torch.equal(superl8.linear_q4k(x, blk), r0)


@pytest.mark.correctness
def test_gemm_q4k_rejects_bad_k(device):
    x_i8 = torch.randint(-127, 128, (4, 128), device=device, dtype=torch.int8)  # K=128, %256!=0
    xs = torch.ones(4, device=device, dtype=torch.float32)
    blk = torch.zeros(8, Q4K_TYPE_SIZE, device=device, dtype=torch.uint8)  # implies 1 superblock->K=256
    with pytest.raises(RuntimeError):
        superl8._C.gemm_q4k(x_i8, xs, blk, torch.float16)


@pytest.mark.correctness
def test_linear_dispatches_gguf_kquant(device):
    """superl8.linear(x, qt) routes a gguf_kquant/q4_k QTensor to the fused kernel."""
    from superl8.format import QTensor

    torch.manual_seed(4)
    x = torch.randn(6, 512, device=device, dtype=torch.float16)
    w = torch.randn(64, 512, dtype=torch.float16) * 0.1
    blk, _ = q4k_quantize(w)
    blk = blk.to(device)
    qt = QTensor(blk, None, scheme="gguf_kquant", group_size=256, codebook="q4_k")
    y = superl8.linear(x, qt)
    assert torch.equal(y, superl8.linear_q4k(x, blk))


# ===========================================================================
# Q5_K and Q6_K — the other 37% of the LTX Q4_K_M mix. Same failing-first
# fidelity gate: fused kernel == dequant->matmul of the SAME native bytes.
# ===========================================================================
KQUANT_SHAPES = [
    (1, 64, 256), (7, 128, 256), (64, 64, 512), (257, 128, 768),
    (2048, 896, 4096), (33, 512, 1024), (5, 130, 256), (200, 96, 512),
]


@pytest.mark.correctness
def test_q5k_q6k_reference_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(8, 512) * 0.1
    for enc in (q5k_quantize, q6k_quantize):
        blk, deq = enc(w)
        assert torch.isfinite(deq).all()
        # deq must be a faithful low-bit reconstruction of w (sanity, not exact).
        assert cos_sim(deq, w) > 0.99


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", KQUANT_SHAPES)
def test_gemm_q5k_reproduces_dequant(device, m, n, k):
    torch.manual_seed(m * 17 + n)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    blk, deq = q5k_quantize(w)
    blk, deq = blk.to(device), deq.to(device)
    x_i8, x_scale = quantize_int8_rowwise(x)
    ref = (x_i8.float() @ deq.t()) * x_scale
    y = superl8._C.gemm_q5k(x_i8, x_scale.squeeze(-1).contiguous(), blk, torch.float16)
    assert y.shape == (m, n)
    assert_int8_quality(y, ref, min_cos=0.999, max_rel_l1=0.01, min_sqnr_db=40.0,
                        what=f"gemm_q5k dequant-exact {m}x{n}x{k}")


@pytest.mark.correctness
@pytest.mark.parametrize("m,n,k", KQUANT_SHAPES)
def test_gemm_q6k_reproduces_dequant(device, m, n, k):
    torch.manual_seed(m * 19 + n)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    blk, deq = q6k_quantize(w)
    blk, deq = blk.to(device), deq.to(device)
    x_i8, x_scale = quantize_int8_rowwise(x)
    ref = (x_i8.float() @ deq.t()) * x_scale
    y = superl8._C.gemm_q6k(x_i8, x_scale.squeeze(-1).contiguous(), blk, torch.float16)
    assert y.shape == (m, n)
    assert_int8_quality(y, ref, min_cos=0.999, max_rel_l1=0.01, min_sqnr_db=40.0,
                        what=f"gemm_q6k dequant-exact {m}x{n}x{k}")


@pytest.mark.correctness
def test_gemm_q5k_q6k_matches_fp32_oracle(device):
    """Q5_K/Q6_K of a real fp weight vs fp32 matmul — Q6_K (6-bit) is tighter
    than Q5_K (5-bit) is tighter than Q4_K (4-bit); assert the ladder."""
    torch.manual_seed(2)
    m, n, k = 128, 512, 1024
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    ref = x.float() @ w.float().to(device).t()
    y5 = superl8.linear_q5k(x, q5k_quantize(w)[0].to(device))
    y6 = superl8.linear_q6k(x, q6k_quantize(w)[0].to(device))
    assert_int8_quality(y5, ref, min_cos=0.995, max_rel_l1=0.09, min_sqnr_db=24.0,
                        what="gemm_q5k fp32-oracle")
    assert_int8_quality(y6, ref, min_cos=0.999, max_rel_l1=0.05, min_sqnr_db=30.0,
                        what="gemm_q6k fp32-oracle")


@pytest.mark.correctness
@pytest.mark.parametrize("tag,enc,lin", [
    ("q5_k", q5k_quantize, "linear_q5k"), ("q6_k", q6k_quantize, "linear_q6k"),
])
def test_linear_dispatches_q5k_q6k(device, tag, enc, lin):
    from superl8.format import QTensor

    torch.manual_seed(4)
    x = torch.randn(6, 512, device=device, dtype=torch.float16)
    w = torch.randn(64, 512, dtype=torch.float16) * 0.1
    blk = enc(w)[0].to(device)
    qt = QTensor(blk, None, scheme="gguf_kquant", group_size=256, codebook=tag)
    y = superl8.linear(x, qt)
    assert torch.equal(y, getattr(superl8, lin)(x, blk))


@pytest.mark.correctness
def test_gemm_q5k_q6k_deterministic(device):
    torch.manual_seed(3)
    x = torch.randn(64, 512, device=device, dtype=torch.float16)
    w = torch.randn(256, 512, dtype=torch.float16) * 0.1
    for lin, enc in [(superl8.linear_q5k, q5k_quantize), (superl8.linear_q6k, q6k_quantize)]:
        blk = enc(w)[0].to(device)
        r0 = lin(x, blk)
        for _ in range(3):
            assert torch.equal(lin(x, blk), r0)


def _real_kquant_tensor(reader, gguf, want_type, qk):
    """Find a 2-D tensor of ggml `want_type` with in%256==0, return (raw_u8[n,-1],
    n, k, deq_oracle) where deq_oracle is the gguf package's OWN dequant (the
    independent reference — NOT our encoder)."""
    for t in reader.tensors:
        if int(t.tensor_type) != int(want_type) or len(t.shape) != 2:
            continue
        n, k = int(t.shape[1]), int(t.shape[0])
        if k % qk == 0 and 256 <= n <= 8192 and 256 <= k <= 8192:
            raw = np.ascontiguousarray(t.data).view(np.uint8).reshape(n, -1)
            deq = gguf.quants.dequantize(t.data, want_type).astype(np.float32).reshape(n, k)
            return raw, n, k, deq
    return None


@pytest.mark.correctness
@pytest.mark.skipif(not os.path.exists(_LTX), reason="LTX Q4_K_M gguf not present")
@pytest.mark.parametrize("tag,ggml_name,op", [
    ("q5_k", "Q5_K", "gemm_q5k"), ("q6_k", "Q6_K", "gemm_q6k"),
])
def test_gemm_q5k_q6k_real_ltx(device, tag, ggml_name, op):
    """The gold check: fused kernel vs the gguf PACKAGE's own dequant on REAL
    llama.cpp Q5_K/Q6_K bytes from the LTX file (independent oracle)."""
    gguf = pytest.importorskip("gguf")
    reader = gguf.GGUFReader(_LTX)
    got = _real_kquant_tensor(reader, gguf, getattr(gguf.GGMLQuantizationType, ggml_name), QK_K)
    if got is None:
        pytest.skip(f"no suitable 2-D {ggml_name} tensor in the LTX file")
    raw, n, k, deq = got
    blk = torch.from_numpy(raw.copy()).to(device)
    deq_t = torch.from_numpy(deq).to(device)
    x = torch.randn(16, k, device=device, dtype=torch.float16)
    x_i8, x_scale = quantize_int8_rowwise(x)
    ref = (x_i8.float() @ deq_t.t()) * x_scale
    y = getattr(superl8._C, op)(x_i8, x_scale.squeeze(-1).contiguous(), blk, torch.float16)
    assert_int8_quality(y, ref, min_cos=0.999, max_rel_l1=0.01, min_sqnr_db=38.0,
                        what=f"{op} real-LTX {n}x{k}")


# ---------------------------------------------------------------------------
# Real-LTX Q4_K tensors (gated on gguf package + file). The payoff proof: native
# GGUF bytes straight into the fused kernel, no conversion.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
@pytest.mark.skipif(not os.path.exists(_LTX), reason="LTX Q4_K_M gguf not present")
def test_gemm_q4k_real_ltx_tensor(device):
    gguf = pytest.importorskip("gguf")
    reader = gguf.GGUFReader(_LTX)
    got = _real_kquant_tensor(reader, gguf, gguf.GGMLQuantizationType.Q4_K, QK_K)
    if got is None:
        pytest.skip("no suitable 2-D Q4_K tensor found")
    raw, n, k, deq_np = got
    blk = torch.from_numpy(raw.copy()).to(device)
    # Independent oracle = the gguf package's OWN dequant (city96/llama math),
    # cross-checked against our own byte-decoder for belt-and-suspenders.
    np.testing.assert_allclose(q4k_dequantize_bytes(raw, n, k), deq_np, rtol=0, atol=2e-3)
    deq = torch.from_numpy(deq_np).to(device)
    x = torch.randn(16, k, device=device, dtype=torch.float16)
    x_i8, x_scale = quantize_int8_rowwise(x)
    ref = (x_i8.float() @ deq.t()) * x_scale
    y = superl8._C.gemm_q4k(x_i8, x_scale.squeeze(-1).contiguous(), blk, torch.float16)
    assert_int8_quality(y, ref, min_cos=0.999, max_rel_l1=0.01, min_sqnr_db=38.0,
                        what=f"gemm_q4k real-LTX {n}x{k}")


# ===========================================================================
# Q3_K (Qwen3.6-27B-Q3_K_S bulk) and Q2_K (aggressive UD-Q2) — native fused.
# ===========================================================================
_Q32 = [("q3_k", q3k_quantize, "gemm_q3k"), ("q2_k", q2k_quantize, "gemm_q2k")]


@pytest.mark.correctness
@pytest.mark.parametrize("tag,enc,tile", _Q32)
def test_q3k_q2k_encoder_matches_gguf_dequant(tag, enc, tile):
    """Our numpy Q3_K/Q2_K encoder's bytes must dequant to the SAME weight via the
    gguf package's OWN dequant (independent oracle) — validates the byte packing."""
    gguf = pytest.importorskip("gguf")
    torch.manual_seed(0)
    w = torch.randn(4, 512) * 0.1
    blk, deq = enc(w)
    ggml = {"q3_k": "Q3_K", "q2_k": "Q2_K"}[tag]
    raw = blk.numpy().reshape(4, -1)
    ref = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType[ggml]).astype(np.float32).reshape(4, 512)
    np.testing.assert_allclose(deq.numpy(), ref, rtol=0, atol=2e-3)


@pytest.mark.correctness
@pytest.mark.parametrize("tag,enc,tile", _Q32)
@pytest.mark.parametrize("m,n,k", KQUANT_SHAPES)
def test_gemm_q3k_q2k_reproduces_dequant(device, tag, enc, tile, m, n, k):
    """Fused Q3_K/Q2_K tile == exact dequant->matmul (SQNR>=40 dB)."""
    torch.manual_seed(m * 23 + n + hash(tag) % 5)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    blk, deq = enc(w)
    blk, deq = blk.to(device), deq.to(device)
    x_i8, x_scale = quantize_int8_rowwise(x)
    ref = (x_i8.float() @ deq.t()) * x_scale
    y = getattr(superl8._C, tile)(x_i8, x_scale.squeeze(-1).contiguous(), blk, torch.float16)
    assert y.shape == (m, n)
    assert_int8_quality(y, ref, min_cos=0.999, max_rel_l1=0.01, min_sqnr_db=40.0,
                        what=f"{tile} dequant-exact {m}x{n}x{k}")


@pytest.mark.correctness
def test_gemm_q3k_q2k_fp32_oracle(device):
    """Q3_K/Q2_K of a real fp weight vs fp32 matmul — 3-bit tighter than 2-bit."""
    torch.manual_seed(2)
    m, n, k = 128, 512, 1024
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    ref = x.float() @ w.float().to(device).t()
    y3 = superl8.linear_q3k(x, q3k_quantize(w)[0].to(device))
    y2 = superl8.linear_q2k(x, q2k_quantize(w)[0].to(device))
    # These bars document the ENCODER's error, not the kernel's: our test
    # q3k/q2k_quantize is a simple positive-scale RTN (llama.cpp's signed-scale
    # make_q3_quants quantizes better), and the weights are structureless random
    # Gaussians (no imatrix) — the worst case. The KERNEL is proven byte-exact
    # separately: test_gemm_q3k_q2k_reproduces_dequant (SQNR>=40 dB vs dequant of
    # the same bytes) and test_gemm_q3k_real_27b (SQNR>=38 dB vs the gguf package's
    # own dequant on REAL Qwen3.6-27B Q3_K tensors). So these are loose, never
    # tightened to hide a kernel bug — cos is the meaningful gate here.
    assert_int8_quality(y3, ref, min_cos=0.97, max_rel_l1=0.24, min_sqnr_db=12.0,
                        what="gemm_q3k fp32-oracle")
    assert_int8_quality(y2, ref, min_cos=0.93, max_rel_l1=0.40, min_sqnr_db=6.0,
                        what="gemm_q2k fp32-oracle")


@pytest.mark.correctness
@pytest.mark.parametrize("tag,enc,tile", _Q32)
def test_linear_dispatches_q3k_q2k(device, tag, enc, tile):
    from superl8.format import QTensor

    torch.manual_seed(4)
    x = torch.randn(6, 512, device=device, dtype=torch.float16)
    w = torch.randn(64, 512, dtype=torch.float16) * 0.1
    blk = enc(w)[0].to(device)
    qt = QTensor(blk, None, scheme="gguf_kquant", group_size=256, codebook=tag)
    lin = {"q3_k": superl8.linear_q3k, "q2_k": superl8.linear_q2k}[tag]
    assert torch.equal(superl8.linear(x, qt), lin(x, blk))


@pytest.mark.correctness
@pytest.mark.skipif(not os.path.exists(_Q27B), reason="Qwen3.6-27B-Q3_K_S gguf not present")
def test_gemm_q3k_real_27b(device):
    """The flagship gate: fused Q3_K vs the gguf package's own dequant on REAL Q3_K
    tensors from Qwen3.6-27B-Q3_K_S.gguf (353 Q3_K tensors), SQNR>=38 dB."""
    gguf = pytest.importorskip("gguf")
    reader = gguf.GGUFReader(_Q27B)
    got = _real_kquant_tensor(reader, gguf, gguf.GGMLQuantizationType.Q3_K, QK_K)
    if got is None:
        pytest.skip("no suitable 2-D Q3_K tensor in the 27B file")
    raw, n, k, deq = got
    blk = torch.from_numpy(raw.copy()).to(device)
    deq_t = torch.from_numpy(deq).to(device)
    x = torch.randn(16, k, device=device, dtype=torch.float16)
    x_i8, x_scale = quantize_int8_rowwise(x)
    ref = (x_i8.float() @ deq_t.t()) * x_scale
    xs = x_scale.squeeze(-1).contiguous()
    for op in ("gemm_q3k", "gemm_decode_q3k"):
        y = getattr(superl8._C, op)(x_i8, xs, blk, torch.float16)
        assert_int8_quality(y, ref, min_cos=0.999, max_rel_l1=0.01, min_sqnr_db=38.0,
                            what=f"{op} real-27B {n}x{k}")


# ===========================================================================
# Warp-per-column DECODE kernel (MMVQ) — gemm_decode_q{4,5,6}k. Must equal the
# tile kernel (same per-sub-block math, different launch) at decode shapes M<=16,
# and match dequant->matmul at SQNR>=40 dB. This is the M=1 GPU-saturating path.
# ===========================================================================
_DECODE_KQ = [
    ("q4_k", q4k_quantize, "gemm_q4k", "gemm_decode_q4k"),
    ("q5_k", q5k_quantize, "gemm_q5k", "gemm_decode_q5k"),
    ("q6_k", q6k_quantize, "gemm_q6k", "gemm_decode_q6k"),
    ("q3_k", q3k_quantize, "gemm_q3k", "gemm_decode_q3k"),
    ("q2_k", q2k_quantize, "gemm_q2k", "gemm_decode_q2k"),
]
# Decode shapes: M in {1,4,8,16}, real projection N/K (K%256==0).
_DECODE_SHAPES = [
    (1, 4096, 3072), (1, 1024, 4096), (4, 4096, 4096), (8, 512, 768),
    (16, 4096, 3072), (1, 130, 256), (1, 4864, 1024),
]


@pytest.mark.correctness
@pytest.mark.parametrize("tag,enc,tile,decode", _DECODE_KQ)
@pytest.mark.parametrize("m,n,k", _DECODE_SHAPES)
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_gemm_decode_kquant_matches_tile(device, tag, enc, tile, decode, m, n, k, dt):
    """Decode MMVQ kernel must equal the tile kernel (same int math)."""
    torch.manual_seed(m * 13 + n + hash(tag) % 7)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    blk = enc(w)[0].to(device)
    x_i8, x_scale = quantize_int8_rowwise(x)
    xs = x_scale.squeeze(-1).contiguous()
    ref = getattr(superl8._C, tile)(x_i8, xs, blk, dt)          # prefill tile
    y = getattr(superl8._C, decode)(x_i8, xs, blk, dt)          # warp-per-column
    assert y.shape == (m, n) and y.dtype == dt
    # identical int32 sub-block sums; only fp32 accumulation ORDER differs -> ~bit-equal.
    assert_int8_quality(y, ref.float(), min_cos=0.9999, max_rel_l1=1e-3, min_sqnr_db=40.0,
                        what=f"{decode} vs tile {m}x{n}x{k} {dt}")


@pytest.mark.correctness
@pytest.mark.parametrize("tag,enc,tile,decode", _DECODE_KQ)
def test_gemm_decode_kquant_reproduces_dequant(device, tag, enc, tile, decode):
    """Decode kernel vs exact dequant->matmul at SQNR>=40 dB (the fidelity gate)."""
    m, n, k = 1, 2048, 1024
    torch.manual_seed(7)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    blk, deq = enc(w)
    blk, deq = blk.to(device), deq.to(device)
    x_i8, x_scale = quantize_int8_rowwise(x)
    ref = (x_i8.float() @ deq.t()) * x_scale
    y = getattr(superl8._C, decode)(x_i8, x_scale.squeeze(-1).contiguous(), blk, torch.float16)
    assert_int8_quality(y, ref, min_cos=0.999, max_rel_l1=0.01, min_sqnr_db=40.0,
                        what=f"{decode} dequant-exact")


@pytest.mark.correctness
@pytest.mark.parametrize("tag,enc,tile,decode", _DECODE_KQ)
def test_linear_kquant_routes_decode_at_small_m(device, tag, enc, tile, decode):
    """linear_q{4,5,6}k must route M<=16 to the decode kernel and match it."""
    lin = {"q4_k": superl8.linear_q4k, "q5_k": superl8.linear_q5k, "q6_k": superl8.linear_q6k,
           "q3_k": superl8.linear_q3k, "q2_k": superl8.linear_q2k}[tag]
    x = torch.randn(1, 1024, device=device, dtype=torch.float16)
    w = torch.randn(4096, 1024, dtype=torch.float16) * 0.1
    blk = enc(w)[0].to(device)
    called = {}
    orig = getattr(superl8._C, decode)

    def spy(*a, **k):
        called["decode"] = True
        return orig(*a, **k)

    setattr(superl8._C, decode, spy)
    try:
        y = lin(x, blk)
    finally:
        setattr(superl8._C, decode, orig)
    assert called.get("decode"), f"{tag} linear did not route M=1 to the decode kernel"
    x_i8, x_scale = quantize_int8_rowwise(x)
    ref = getattr(superl8._C, tile)(x_i8, x_scale.squeeze(-1).contiguous(), blk, torch.float16)
    assert_int8_quality(y, ref.float(), min_cos=0.9999, max_rel_l1=1e-3, min_sqnr_db=40.0,
                        what=f"{tag} decode-route")


@pytest.mark.perf
@pytest.mark.parametrize("tag,enc,tile,decode", _DECODE_KQ)
def test_gemm_decode_kquant_perf(device, tag, enc, tile, decode):
    """M=1 decode: warp-per-column MMVQ vs the prefill tile (the 6.7x-slower path)."""
    import sys as _sys
    from pathlib import Path as _P

    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from bench.harness import assert_no_regression, compare_report, time_ms

    m, n, k = 1, 4096, 3072
    torch.manual_seed(0)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    blk = enc(w)[0].to(device)
    x_i8, x_scale = quantize_int8_rowwise(x)
    xs = x_scale.squeeze(-1).contiguous()
    dec_ms = time_ms(lambda: getattr(superl8._C, decode)(x_i8, xs, blk, torch.float16))
    tile_ms = time_ms(lambda: getattr(superl8._C, tile)(x_i8, xs, blk, torch.float16))
    tag2 = f"{decode}.m1n4096k3072"
    print("\n" + compare_report(tag2, dec_ms, {f"{tile}.tile": tile_ms}))
    assert_no_regression(tag2, dec_ms)


# ---------------------------------------------------------------------------
# Perf: fused native-Q4_K dp4a vs the per-forward fp32-dequant->matmul path
# (what the 143 s/step GGUF path does every forward). Fused keeps the k-quant
# resident (O(native-bytes) VRAM) AND fuses the dequant into the matmul.
# ---------------------------------------------------------------------------
@pytest.mark.perf
@pytest.mark.parametrize("op,enc,short", [
    ("gemm_q4k", q4k_quantize, "q4k"),
    ("gemm_q5k", q5k_quantize, "q5k"),
    ("gemm_q6k", q6k_quantize, "q6k"),
])
@pytest.mark.parametrize("m", [1, 2048])   # LTX DiT decode (M=1) and prefill/batched
def test_gemm_kquant_perf_vs_dequant(device, op, enc, short, m):
    import sys as _sys
    from pathlib import Path as _P

    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from bench.harness import assert_no_regression, compare_report, time_ms

    n, k = 4096, 3072
    tag = f"{op}.m{m}n4096k3072"
    torch.manual_seed(0)
    x = torch.randn(m, k, device=device, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    blk, deq = enc(w)
    blk, deq = blk.to(device), deq.half().to(device)
    x_i8, x_scale = quantize_int8_rowwise(x)
    xs = x_scale.squeeze(-1).contiguous()
    fn = getattr(superl8._C, op)

    ms = time_ms(lambda: fn(x_i8, xs, blk, torch.float16))
    # The path superl8 replaces: the per-forward fp16-dequant->matmul. Note this
    # baseline uses a PRE-materialized deq, so it UNDER-counts the real GGUF path
    # (which must also unpack the k-quant to fp every forward). Even so the fused
    # dp4a additionally saves the whole dequant materialization + ~1.78x VRAM.
    dequant_matmul_ms = time_ms(lambda: torch.matmul(x, deq.t()))
    tops = 2.0 * m * n * k / (ms * 1e-3) / 1e12
    print("\n" + compare_report(tag, ms, {"dequant->fp16.matmul": dequant_matmul_ms})
          + f" | {tops:.1f} int8-TOP/s")
    assert_no_regression(tag, ms)
