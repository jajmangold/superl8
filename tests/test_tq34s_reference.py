# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""TQ3_4S reference dequant — the CPU oracle for the fused dp4a kernel (superl8#272).

Golden vectors are hand-derived from the turbo-tan/llama.cpp-tq3 fork's CPU
dequant (``ggml-quants.c``: ``dequantize_row_tq3_4s`` ~2746,
``tq3_4s_decode_scale`` ~2686, ``tq3_0_rht_forward/inverse`` ~2395-2446,
``TQ3_0_CENTROIDS`` ~2360, ``TQ3_0_SIGNS`` ~2366). The reference must match the
fork's fp32 dequant semantics BITWISE — never `allclose` for quantized paths
(superl8 AGENTS.md).

CPU-only: no ``superl8._C`` / CUDA device required — the oracle is pure torch/numpy.
"""

import mmap
import os
import struct

import numpy as np
import pytest
import torch

import superl8
from superl8.quant import tq34s as T

pytestmark = [pytest.mark.correctness, pytest.mark.cpu]

# Real TQ3_4S artifact (gates the real-tensor sanity test; skipped if absent).
_TQ3_GGUF = os.environ.get("FNI8_TQ3_GGUF", "")

# ---------------------------------------------------------------------------
# Fork transcriptions (independent scalars, NOT the vectorized reference)
# ---------------------------------------------------------------------------


def _fork_pack(indices: np.ndarray) -> np.ndarray:
    """Transcription of the fork's ``quantize_row_tq3_4s_ref`` 3-byte pack loop."""
    qp = []
    for g in range(4):
        idx = indices[g * 8:(g + 1) * 8]
        qp.append(np.uint8(idx[0] | (idx[1] << 3) | (idx[2] << 6)))
        qp.append(np.uint8((idx[2] >> 2) | (idx[3] << 1) | (idx[4] << 4) | (idx[5] << 7)))
        qp.append(np.uint8((idx[5] >> 1) | (idx[6] << 2) | (idx[7] << 5)))
    return np.array(qp, dtype=np.uint8)


def _fork_dequant_block(blk: np.ndarray) -> np.ndarray:
    """Direct fp32 transcription of ``dequantize_row_tq3_4s`` for one 16-byte block."""
    scale = np.float32(1.0 / np.sqrt(32.0))

    def dsc(b: int) -> np.float32:
        if b == 0:
            return np.float32(0.0)
        return np.float32(1.0 + (b & 31) / 32.0) * np.float32(2.0 ** ((b >> 5) - 9))

    rotated = np.zeros(32, dtype=np.float32)
    for g in range(4):
        d = dsc(int(blk[g]))
        qp = [int(blk[4 + g * 3 + k]) for k in range(3)]
        idx = np.zeros(8, dtype=np.int64)
        idx[0] = qp[0] & 7
        idx[1] = (qp[0] >> 3) & 7
        idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7
        idx[3] = (qp[1] >> 1) & 7
        idx[4] = (qp[1] >> 4) & 7
        idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7
        idx[6] = (qp[2] >> 2) & 7
        idx[7] = (qp[2] >> 5) & 7
        for j in range(8):
            rotated[g * 8 + j] = T.TQ3_CENTROIDS[idx[j]] * d
    out = rotated.copy()
    step = 1
    while step < 32:
        for i in range(0, 32, step * 2):
            for j in range(i, i + step):
                a = out[j]
                b = out[j + step]
                out[j] = np.float32(a + b)
                out[j + step] = np.float32(a - b)
        step <<= 1
    return out * (T.TQ3_SIGNS * scale)


# ---------------------------------------------------------------------------
# E3M5 scale decode
# ---------------------------------------------------------------------------


def _e3m5_golden(byte: int) -> float:
    """Fork ``tq3_4s_decode_scale``: 0 => 0.0; else 2^(exp-9)*(1+mant/32)."""
    if byte == 0:
        return 0.0
    e = (byte >> 5) - 9
    return (1.0 + (byte & 31) / 32.0) * (2.0 ** e)


def test_e3m5_golden_table():
    """All 256 E3M5 scale bytes decode per the fork's ratio semantics (exact fp32)."""
    b = np.arange(256, dtype=np.uint8)
    got = T.decode_e3m5(b)
    assert got.dtype == np.float32
    ref = np.array([_e3m5_golden(int(x)) for x in b], dtype=np.float32)
    bad = np.flatnonzero(got != ref)
    assert len(bad) == 0, f"E3M5 mismatch at bytes {bad.tolist()[:5]}"
    # hand spot-checks
    for byte, val in [
        (0x00, 0.0),                       # zero scale -> 0.0
        (0x20, 2.0 ** -8),                 # exp=1, mant=0
        (0x40, 2.0 ** -7),                 # exp=2, mant=0
        (0x7F, (1.0 + 31 / 32) * 2.0 ** -6),
        (0xFF, (1.0 + 31 / 32) * 2.0 ** -2),
    ]:
        assert float(T.decode_e3m5(np.array([byte], dtype=np.uint8))[0]) == pytest.approx(
            val, rel=1e-6)


# ---------------------------------------------------------------------------
# 3-bit unpack
# ---------------------------------------------------------------------------


def test_unpack_3bit_golden():
    """Hand-derived golden from the fork's dequant idx[] lines.

    qp0=qp2=0xAA, qp1=0x55 (repeated over the 4 groups) unpacks to
    [2,5,6,2,5,4,2,5] per group: idx0=qp0&7=2, idx1=(qp0>>3)&7=5,
    idx2=((qp0>>6)|(qp1<<2))&7=6, idx3=(qp1>>1)&7=2, idx4=(qp1>>4)&7=5,
    idx5=((qp1>>7)|(qp2<<1))&7=4, idx6=(qp2>>2)&7=2, idx7=(qp2>>5)&7=5.
    """
    qs = np.array([0xAA, 0x55, 0xAA] * 4, dtype=np.uint8)
    got = T.unpack_3bit(qs)
    expect = np.array([2, 5, 6, 2, 5, 4, 2, 5] * 4, dtype=np.uint32)
    assert got.shape == (32,)
    assert np.array_equal(got, expect)


def test_unpack_3bit_pack_roundtrip():
    """Pack (fork transcription) -> unpack (reference) round-trips 32 random codes."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = rng.integers(0, 8, 32)
        qs = _fork_pack(idx)
        assert np.array_equal(T.unpack_3bit(qs), idx)
        # batch shape [..., 12] -> [..., 32]
        batch = np.stack([qs, _fork_pack(rng.integers(0, 8, 32))])
        assert T.unpack_3bit(batch).shape == (2, 32)


# ---------------------------------------------------------------------------
# Randomized Hadamard transform
# ---------------------------------------------------------------------------


def test_rht_orthogonal():
    """F . F^T = I for F = H . diag(SIGNS)/sqrt(32) — the dp4a-fusion identity
    (x^T . RHT_inv(v) = (RHT_fwd(x))^T . v)."""
    basis = np.eye(32, dtype=np.float32)
    F = np.stack([T.rht_forward(basis[:, i]) for i in range(32)], axis=1)  # cols = F e_i
    G = F @ F.T
    assert np.abs(G - np.eye(32)).max() < 1e-4, "F F^T != I"
    # round-trip: inverse is the transpose up to fp32 rounding
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.standard_normal(32).astype(np.float32)
        assert np.abs(T.rht_inverse(T.rht_forward(x)) - x).max() < 1e-4


# ---------------------------------------------------------------------------
# Full dequant: fork-bitwise + synthesized round-trip
# ---------------------------------------------------------------------------


def test_dequant_matches_fork_bitwise():
    """Vectorized dequant == a direct transcription of the fork's
    ``dequantize_row_tq3_4s`` on random blocks — BITWISE in fp32."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        blk = rng.integers(0, 256, 16).astype(np.uint8)
        ref = _fork_dequant_block(blk)
        got = T.dequantize_tq34s_bytes(blk.reshape(1, 16), 32).reshape(32)
        assert np.array_equal(got, ref), (
            f"fork mismatch (max abs diff {np.abs(got - ref).max():.3e})")


def test_dequant_synthesized_roundtrip():
    """Random bytes -> dequant -> finite, right shape, plausible magnitude; a
    zero-scale (E3M5==0) block collapses to zeros (acceptance-gate case)."""
    rng = np.random.default_rng(2)
    u8 = rng.integers(0, 256, (5, (128 // 32) * 16), dtype=np.uint8)
    w = T.dequantize_tq34s_bytes(u8, 128)
    assert w.shape == (5, 128)
    assert np.isfinite(w).all()
    # |v| <= max|centroid| * max_scale ~= 0.98; WHT then /sqrt(32) -> |w| <= ~5.6
    assert float(np.abs(w).max()) < 10.0
    # E3M5==0 for all four scales -> all centroids vanish -> all-zero block
    zero = np.zeros((1, 16), dtype=np.uint8)
    assert np.array_equal(T.dequantize_tq34s_bytes(zero, 32), np.zeros((1, 32)))


# ---------------------------------------------------------------------------
# Level table (the fused kernel's int8 centroid levels)
# ---------------------------------------------------------------------------


def test_tq34s_levels_spec():
    """Corrected superl8 levels: K = 127/max|centroid| = 127/1.996684 — NOT the
    fork's stale 2.1519 constant. round(centroid*K) recovers the centroid."""
    kmax = float(np.abs(T.TQ3_CENTROIDS).max())
    K = 127.0 / kmax
    # TQ3_CENTROIDS is stored fp32 (like the fork's float table), so max|centroid|
    # is 1.9966840744018555 — K matches 127/1.996684 to ~2.6e-8 relative.
    assert K == pytest.approx(127.0 / 1.996684, rel=1e-6)
    lv = T.tq34s_levels()
    assert lv.tolist() == [-127, -82, -47, -16, 15, 46, 81, 127]
    implied = lv.astype(np.float64) / K
    assert np.abs(implied - T.TQ3_CENTROIDS).max() <= 0.5 / K, "level rounding drift"


# ---------------------------------------------------------------------------
# Reference linear + superl8.linear dispatch
# ---------------------------------------------------------------------------


def test_reference_linear_matches_dequant_matmul():
    """reference_linear == fp32 matmul with the dequantized weight (bitwise)."""
    rng = np.random.default_rng(3)
    u8 = rng.integers(0, 256, (6, (64 // 32) * 16), dtype=np.uint8)
    u8t = torch.from_numpy(u8.copy())
    x = torch.randn(2, 64, dtype=torch.float16)
    y = T.reference_linear(x, u8t, 64)
    w = torch.from_numpy(T.dequantize_tq34s_bytes(u8, 64))
    assert torch.equal(y, x.float() @ w.t())
    # op wrapper agrees (bias + out_dtype path)
    y3 = superl8.linear_tq34s(x, u8t, 64)
    assert torch.allclose(y3.float(), x.float() @ w.t(), atol=1e-3)
    yb = superl8.linear_tq34s(x, u8t, 64, bias=torch.randn(6))
    assert yb.shape == (2, 6) and torch.isfinite(yb).all()


def test_qtensor_validate_accepts_tq34s():
    """gguf_kquant QTensor validation accepts tq3_4s (block 16 B, group 32)."""
    from superl8.format import QTensor

    rng = np.random.default_rng(5)
    u8 = torch.from_numpy(rng.integers(0, 256, (2, 32), dtype=np.uint8))  # [2, (64//32)*16]
    QTensor(u8, None, scheme="gguf_kquant", group_size=32, codebook="tq3_4s").validate()
    # QK_TQ3_0=32 (NOT QK_K=256): group_size 256 must be rejected
    with pytest.raises(AssertionError):
        QTensor(u8, None, scheme="gguf_kquant", group_size=256, codebook="tq3_4s").validate()
    # unknown codebook still rejected
    with pytest.raises(AssertionError):
        QTensor(u8, None, scheme="gguf_kquant", group_size=32, codebook="q8_0").validate()


def test_linear_dispatch_tq34s():
    """superl8.linear routes gguf_kquant codebook tq3_4s to linear_tq34s."""
    from superl8.format import QTensor

    rng = np.random.default_rng(4)
    u8 = rng.integers(0, 256, (6, (128 // 32) * 16), dtype=np.uint8)
    qt = QTensor(torch.from_numpy(u8.copy()), None, scheme="gguf_kquant",
                 group_size=32, codebook="tq3_4s")
    x = torch.randn(3, 128, dtype=torch.float16)
    y = superl8.linear(x, qt)
    y2 = superl8.linear_tq34s(x, qt.data, 128)
    assert y.shape == (3, 6)
    assert torch.allclose(y.float(), y2.float(), atol=1e-3)


def test_gguf_loader_plumbing_tq34s():
    """_NATIVE_FUSED / _KQUANT_TYPE_SIZE / kquant_qtensor handle tq3_4s (group 32)."""
    from superl8 import gguf as fgguf

    assert fgguf._NATIVE_FUSED["TQ3_4S"] == "tq3_4s"
    assert fgguf._KQUANT_TYPE_SIZE["tq3_4s"] == 16
    assert fgguf._NATIVE_FUSED_BY_TYPE[46] == "tq3_4s"
    cov = fgguf.gguf_type_coverage()
    assert cov["TQ3_4S"] == "native_fused"
    rng = np.random.default_rng(6)
    u8 = rng.integers(0, 256, (4, 16), dtype=np.uint8)  # [4, (32//32)*16]
    qt = fgguf.kquant_qtensor(u8, 4, 32, "tq3_4s")
    assert qt.group_size == 32 and qt.codebook == "tq3_4s"
    assert tuple(qt.data.shape) == (4, 16)
    # k-quants remain QK_K=256 gated (in%256==0)
    with pytest.raises(AssertionError):
        fgguf.kquant_qtensor(u8, 4, 32, "q4_k")


# ---------------------------------------------------------------------------
# Real-tensor sanity (gated on the artifact being on disk)
# ---------------------------------------------------------------------------


_GGUF_VALUE_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class _T3Tensor:
    __slots__ = ("dims", "ggml_type", "name", "offset")

    def __init__(self, name, dims, ggml_type, offset):
        self.name = name
        self.dims = dims
        self.ggml_type = ggml_type
        self.offset = offset


def _read_gguf_tensor_infos(path: str) -> tuple[list, int]:
    """Minimal GGUF reader (header + metadata + tensor infos) + data offset.

    The stock ``gguf`` package cannot open TQ3_4S files: type 46 is a
    turbo-tan fork type unknown to every released gguf-py (the fork's own
    constants stop at Q1_0), so GGUFReader raises on the raw dtype. The GGUF
    framing itself is simple and stable, so the real-tensor test parses it
    directly. Only the metadata byte layout is walked (no tensor data touched).
    """
    alignment = 32
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        assert mm[0:4] == b"GGUF", "bad GGUF magic"
        off = 4

        def rd(fmt):
            nonlocal off
            v = struct.unpack_from(fmt, mm, off)
            off += struct.calcsize(fmt)
            return v[0]

        def rd_bytes(n):
            nonlocal off
            b = mm[off:off + n]
            off += n
            return b

        def rd_str():
            return rd_bytes(rd("<Q")).decode("utf-8", "replace")

        rd("<I")                       # version
        tensor_count = rd("<Q")
        kv_count = rd("<Q")
        for _ in range(kv_count):
            key = rd_str()
            vt = rd("<I")
            if key == "general.alignment" and vt == 4:
                alignment = rd("<I")
                continue
            if vt == 8:                # string
                rd_str()
            elif vt == 9:              # array
                et = rd("<I")
                cnt = rd("<Q")
                for _ in range(cnt):
                    if et == 8:
                        rd_str()
                    else:
                        off += _GGUF_VALUE_SIZE[et]
            else:
                off += _GGUF_VALUE_SIZE[vt]
        infos = []
        for _ in range(tensor_count):
            name = rd_str()
            n_dims = rd("<I")
            dims = [rd("<Q") for _ in range(n_dims)]
            ggml_type = rd("<I")
            toff = rd("<Q")
            infos.append(_T3Tensor(name, dims, ggml_type, toff))
        data_start = (off + alignment - 1) // alignment * alignment
        return infos, data_start


def _tensor_bytes(path: str, t3: "_T3Tensor", data_start: int) -> np.ndarray:
    """Extract a 2-D TQ3_4S tensor's raw bytes -> uint8 [out, (in//32)*16]."""
    in_f, out_f = int(t3.dims[0]), int(t3.dims[1])
    nbytes = (in_f // T.QK_TQ3) * T.TQ3_TYPE_SIZE * out_f
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        buf = mm[data_start + t3.offset:data_start + t3.offset + nbytes]
        return np.frombuffer(buf, dtype=np.uint8).reshape(
            out_f, (in_f // T.QK_TQ3) * T.TQ3_TYPE_SIZE)


skip_no_tq3_gguf = pytest.mark.skipif(
    not os.path.exists(_TQ3_GGUF), reason=f"TQ3_4S GGUF not on disk ({_TQ3_GGUF})")


@skip_no_tq3_gguf
def test_real_tensor_dequant():
    """Dequant real TQ3_4S tensors from the Qwen3.8-27B artifact (type 46) —
    finite, plausible magnitude. Sanity against actual fork-quantized bytes."""
    infos, data_start = _read_gguf_tensor_infos(_TQ3_GGUF)
    t46 = sorted(
        [t for t in infos if t.ggml_type == 46 and len(t.dims) == 2],
        key=lambda t: (t.dims[0] // T.QK_TQ3) * T.TQ3_TYPE_SIZE * t.dims[1],
    )
    assert len(t46) >= 3, f"expected >=3 TQ3_4S tensors, found {len(t46)}"
    for t3 in t46[:3]:
        raw = _tensor_bytes(_TQ3_GGUF, t3, data_start)
        w = T.dequantize_tq34s_bytes(raw, int(t3.dims[0]))
        assert w.shape == (int(t3.dims[1]), int(t3.dims[0])), t3.name
        assert np.isfinite(w).all(), t3.name
        assert float(np.abs(w).max()) < 10.0, t3.name
        assert float(np.abs(w).mean()) < 0.5, t3.name


# ---------------------------------------------------------------------------
# Fused-kernel fidelity gate lives in tests/test_gemm_tq34s.py (superl8#272): the
# full gate (SQNR>=40 dB / cos>=0.999 / ragged M/N / determinism x3 / E3M5==0)
# needs CUDA + the `device` fixture, so it moved out of this CPU-only file when
# the fused kernel landed. This file stays the CPU oracle.
# ---------------------------------------------------------------------------
