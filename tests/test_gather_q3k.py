# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Native GGUF Q3_K ROW-GATHER dequant (`superl8.gather_q3k`) — the token-embedding
path.

`token_embd.weight` is only ever GATHERED (the LM head is a separate tensor), so
there is no dp4a to fuse into; the win is that the table stays RESIDENT in its
native Q3_K bytes instead of being dequantized+requantized into a `per_row_i8`
table (Qwen3.8-27B-UD-IQ3_S: 0.5088 GiB native vs 1.1850 GiB i8).

TOLERANCE: this op has NO activation quantization — it is an exact dequant of
stored bytes — so the int8 SQNR/cos convention does NOT apply here. Kernel and
reference perform the SAME two fp32 multiplies in the SAME order (d*sc, then
*(q-4)); there is no add, so no FMA contraction can reorder them, and the fp16
store rounds identically. The gate is therefore BITWISE equality against the
repo's own Q3_K reference dequant. Only the `gguf`-package oracle (an
independent implementation, free to associate differently) gets a 1-ulp bound.

Reference: `q3k_quantize` from tests/test_gemm_q4k.py, whose byte packing is
itself validated against the gguf package's dequant by
`test_q3k_q2k_encoder_matches_gguf_dequant`.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import superl8

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_gemm_q4k import QK_K, _Q27B, _real_kquant_tensor, q3k_quantize  # noqa: E402

Q3K_TYPE_SIZE = 110  # 32(hmask)+64(qs)+12(scales)+2(d)
# One fp16 ulp is ~1e-3 relative. Used ONLY against the independent gguf oracle.
_ULP_RTOL, _ULP_ATOL = 1e-3, 1e-6


def _table(n: int, k: int, seed: int = 0):
    """Random fp16 weights -> (native Q3_K bytes [n,(k//256)*110], fp32 dequant [n,k])."""
    torch.manual_seed(seed)
    w = torch.randn(n, k, dtype=torch.float16) * 0.1
    return q3k_quantize(w)


# ---------------------------------------------------------------------------
# Stage 0 (runs with NO GPU): a numpy mirror of the KERNEL'S OWN loop — the
# thread/pair split and the sub-block/shift/mask expressions copied out of
# gather_q3k.cuh, not re-derived from the spec — must reproduce the reference
# dequant exactly. This is the gate on the bit-mapping itself: a wrong shift,
# hmask bit, scale nibble, qs offset or pair split fails here on CPU, before any
# device ever runs. One loop iteration == one CUDA thread (which owns a pair of
# adjacent weights and stores them with one `pair_store`).
# ---------------------------------------------------------------------------
def _kernel_mirror_dequant(blk_bytes: np.ndarray, n: int, k: int) -> np.ndarray:
    nsb = k // QK_K
    b = blk_bytes.reshape(n, nsb, Q3K_TYPE_SIZE)
    out = np.zeros((n, k), np.float32)
    for sb in range(nsb):
        d = np.ascontiguousarray(b[:, sb, 108:110]).view(np.float16).astype(np.float32)
        d = d.reshape(n)
        sc_bytes = b[:, sb, 96:108].astype(np.int32)
        for t in range(QK_K // 2):                  # one THREAD per iteration
            is_ = t >> 3                            # sub-block owned by this thread
            i16 = (t & 7) * 2                       # even position; pair is i16, i16+1
            g_, sig = is_ >> 3, is_ & 7
            shift = sig & 6
            m_shift = g_ * 4 + (sig >> 1)
            qs_off = 32 + 32 * g_ + (sig & 1) * 16 + i16
            hm_off = (sig & 1) * 16 + i16
            # q3k_scale(blk+96, is): low 4 bits from scales[0..7], high 2 from
            # scales[8..11], minus 32.
            lo = (sc_bytes[:, is_ % 8] >> (4 * (is_ // 8))) & 0xF
            hi = ((sc_bytes[:, 8 + is_ % 4] >> (2 * (is_ // 4))) & 3) << 4
            sc = ((lo | hi) - 32).astype(np.float32)
            dsc = d * sc                            # kernel's order: (d*sc) then *code
            for half in range(2):                   # the two halves of the pair_store
                low2 = (b[:, sb, qs_off + half].astype(np.int32) >> shift) & 3
                hbit = (b[:, sb, hm_off + half].astype(np.int32) >> m_shift) & 1
                code = ((low2 | (hbit << 2)) - 4).astype(np.float32)
                out[:, sb * QK_K + is_ * 16 + i16 + half] = dsc * code
    return out


@pytest.mark.cpu
@pytest.mark.correctness
@pytest.mark.parametrize("n,k", [(4, 256), (7, 512), (3, 1280)])
def test_kernel_index_arithmetic_matches_reference_dequant(n, k):
    """The kernel's bit-mapping, mirrored in numpy, == the reference dequant EXACTLY."""
    blk, deq = _table(n, k, seed=n * 31 + k)
    np.testing.assert_array_equal(_kernel_mirror_dequant(blk.numpy(), n, k), deq.numpy())


# ---------------------------------------------------------------------------
# Stage 1 — smoke: the op runs, shape/dtype are right.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_gather_q3k_smoke(device):
    blk, _ = _table(32, 256, seed=1)
    y = superl8.gather_q3k(torch.zeros(4, device=device, dtype=torch.int64), blk.to(device))
    assert y.shape == (4, 256) and y.dtype == torch.float16
    assert torch.isfinite(y.float()).all()


# ---------------------------------------------------------------------------
# Stage 2 — shape sweep (INCLUDING non-tile-multiple N and K) x dtype, and
# Stage 3 — exact numeric match vs the reference dequant, with DUPLICATE ids.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
@pytest.mark.parametrize("n,k", [
    (64, 256),      # one super-block per row
    (130, 512),     # ragged N (not a tile multiple)
    (37, 1024),     # ragged N, odd
    (200, 768),     # ragged N below a tile
    (3, 1280),      # 5 super-blocks, tiny N
    (256, 5120),    # the real Qwen3.8-27B hidden size (20 super-blocks/row)
])
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_gather_q3k_matches_reference_dequant(device, n, k, dt):
    blk, deq = _table(n, k, seed=n + k)
    blk, deq = blk.to(device), deq.to(device)
    ids = torch.randint(0, n, (33,), device=device, dtype=torch.int64)
    ids[1] = ids[0]          # duplicate ids must both decode
    ids[2] = ids[0]
    ids[0] = 0               # first row
    ids[-1] = n - 1          # last row
    y = superl8.gather_q3k(ids, blk, out_dtype=dt)
    assert y.shape == (33, k) and y.dtype == dt
    # Same fp32 arithmetic, same order, same store rounding -> bitwise equal.
    assert torch.equal(y, deq[ids].to(dt))


@pytest.mark.correctness
def test_gather_q3k_duplicate_ids_are_identical(device):
    """The same id gathered many times must give bitwise-identical rows."""
    blk, _ = _table(64, 1024, seed=11)
    blk = blk.to(device)
    y = superl8.gather_q3k(torch.full((16,), 7, device=device, dtype=torch.int64), blk)
    assert torch.equal(y, y[0].expand_as(y))


# ---------------------------------------------------------------------------
# Ragged / arbitrary ids shapes: out is [..., K] for ANY ids shape.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
@pytest.mark.parametrize("shape", [(), (1,), (5,), (3, 5, 2), (2, 1, 7), (17, 3)])
def test_gather_q3k_ragged_ids_shape(device, shape):
    n, k = 96, 768
    blk, deq = _table(n, k, seed=5)
    blk, deq = blk.to(device), deq.to(device)
    ids = torch.randint(0, n, shape, device=device, dtype=torch.int64)
    y = superl8.gather_q3k(ids, blk)
    assert y.shape == (*shape, k)
    assert torch.equal(y, deq[ids].to(torch.float16))


@pytest.mark.correctness
def test_gather_q3k_empty_ids(device):
    blk, _ = _table(32, 256, seed=6)
    y = superl8.gather_q3k(torch.empty(0, device=device, dtype=torch.int64), blk.to(device))
    assert y.shape == (0, 256)


@pytest.mark.correctness
def test_gather_q3k_int32_ids_match_int64(device):
    n, k = 80, 512
    blk, _ = _table(n, k, seed=9)
    blk = blk.to(device)
    ids64 = torch.randint(0, n, (21,), device=device, dtype=torch.int64)
    assert torch.equal(superl8.gather_q3k(ids64.to(torch.int32), blk),
                       superl8.gather_q3k(ids64, blk))


# ---------------------------------------------------------------------------
# Stage 4 — determinism: same input three times, bitwise equal.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_gather_q3k_deterministic(device):
    n, k = 64, 1024
    blk, _ = _table(n, k, seed=10)
    blk = blk.to(device)
    ids = torch.randint(0, n, (9,), device=device, dtype=torch.int64)
    y0 = superl8.gather_q3k(ids, blk)
    for _ in range(3):
        assert torch.equal(superl8.gather_q3k(ids, blk), y0)


# ---------------------------------------------------------------------------
# Out-of-range ids: a bad id must never read past a half-GiB table. The kernel
# zero-fills that row (documented contract) instead of faulting or reading OOB.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_gather_q3k_out_of_range_ids_zero_fill(device):
    n, k = 32, 256
    blk, deq = _table(n, k, seed=12)
    blk, deq = blk.to(device), deq.to(device)
    ids = torch.tensor([0, n, -1, n + 1000, n - 1], device=device, dtype=torch.int64)
    y = superl8.gather_q3k(ids, blk)
    zero = torch.zeros(k, device=device, dtype=torch.float16)
    assert torch.equal(y[1], zero) and torch.equal(y[2], zero) and torch.equal(y[3], zero)
    assert torch.equal(y[0], deq[0].to(torch.float16))
    assert torch.equal(y[4], deq[n - 1].to(torch.float16))


# ---------------------------------------------------------------------------
# Shape/dtype validation — clear errors, like the other k-quant ops.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
def test_gather_q3k_rejects_bad_inputs(device):
    blk, _ = _table(32, 512, seed=13)
    blk = blk.to(device)
    ids = torch.zeros(4, device=device, dtype=torch.int64)

    with pytest.raises(RuntimeError, match="must be int32 or int64"):
        superl8.gather_q3k(ids.to(torch.float32), blk)
    with pytest.raises(RuntimeError, match="uint8"):
        superl8.gather_q3k(ids, blk.to(torch.int8))
    with pytest.raises(RuntimeError, match=r"\(K/256\)\*110"):
        superl8.gather_q3k(ids, blk[:, :-1].contiguous())     # row width not a multiple
    with pytest.raises(RuntimeError, match=r"\[N,\(K/256\)\*110\]"):
        superl8.gather_q3k(ids, blk.reshape(-1))              # not 2-D
    with pytest.raises(RuntimeError, match="float16 or bfloat16"):
        superl8.gather_q3k(ids, blk, out_dtype=torch.float32)
    with pytest.raises(RuntimeError, match="CUDA"):
        superl8.gather_q3k(ids.cpu(), blk.cpu())


# ---------------------------------------------------------------------------
# Real Qwen3.6-27B Q3_K bytes vs the gguf package's OWN dequant (independent
# oracle — not our encoder, so 1 fp16 ulp, not bitwise). Mirrors
# test_gemm_q3k_real_27b.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
@pytest.mark.skipif(not os.path.exists(_Q27B), reason="Qwen3.6-27B-Q3_K_S gguf not present")
def test_gather_q3k_real_27b(device):
    gguf = pytest.importorskip("gguf")
    reader = gguf.GGUFReader(_Q27B)
    got = _real_kquant_tensor(reader, gguf, gguf.GGMLQuantizationType.Q3_K, QK_K)
    if got is None:
        pytest.skip("no suitable 2-D Q3_K tensor in the 27B file")
    raw, n, k, deq = got
    blk = torch.from_numpy(raw.copy()).to(device)
    deq_t = torch.from_numpy(deq).to(device)
    ids = torch.randint(0, n, (64,), device=device, dtype=torch.int64)
    ids[1] = ids[0]
    y = superl8.gather_q3k(ids, blk)
    ref = deq_t[ids].to(torch.float16)
    torch.testing.assert_close(y.float(), ref.float(), rtol=_ULP_RTOL, atol=_ULP_ATOL)


# ---------------------------------------------------------------------------
# Perf gate. The claim this op makes is a RESIDENCY claim, not a latency one:
# the alternative is not "a slower gather", it is "the same gather off a
# dequantized+requantized table that costs ~2.3x the bytes on the card". So the
# assertion is the byte win (exact, computable) and the latency is reported.
# NOT YET RUN ON DEVICE — no measured number from this op is committed anywhere,
# and per AGENTS.md any perf CLAIM needs an ncu artifact behind it.
# ---------------------------------------------------------------------------
@pytest.mark.perf
def test_gather_q3k_resident_bytes_win(device):
    n, k = 8192, 5120
    blk, _ = _table(n, k, seed=14)
    blk = blk.to(device)
    native_bytes = blk.numel()
    # per_row_i8 table: one int8 per weight + one fp32 scale per row.
    i8_bytes = n * k + n * 4
    assert native_bytes < i8_bytes / 2, "Q3_K native must beat a per_row_i8 table 2:1"
    print(f"gather_q3k table {n}x{k}: native Q3_K {native_bytes / 2**30:.4f} GiB vs "
          f"per_row_i8 {i8_bytes / 2**30:.4f} GiB "
          f"(+{(i8_bytes - native_bytes) / 2**30:.4f} GiB avoided)")

    ids = torch.randint(0, n, (4096,), device=device, dtype=torch.int64)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(5):
        superl8.gather_q3k(ids, blk)
    torch.cuda.synchronize()
    start.record()
    for _ in range(50):
        superl8.gather_q3k(ids, blk)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 50
    print(f"gather_q3k {ids.numel()} rows x {k}: {ms:.3f} ms/call")
    assert np.isfinite(ms) and ms > 0
