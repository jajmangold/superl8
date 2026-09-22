# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Bounds validation for FQReader: a corrupt header must fail loudly at open time.

The validation in FQReader.__post_init__ checks every tensor entry:
  1. offset >= 0
  2. offset + nbytes fits inside the mmap
  3. shape * itemsize == nbytes (self-consistent)

A valid file must still open byte-identically.
"""
import json
import struct

import numpy as np
import pytest
import torch

from superl8 import FQReader, QTensor, save_superl8
from superl8.format import ALIGN, FOOTER_MAGIC, MAGIC

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pad(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _i8_weight(out, in_):
    w = torch.randn(out, in_)
    scale = w.abs().amax(-1, keepdim=True) / 127.0
    q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
    return QTensor(q, scale.squeeze(-1).float(), scheme="per_row_i8")


def _rebuild_with_corrupt_header(path, modify_fn):
    """Read a valid .superl8, apply *modify_fn* to the header dict, and write
    the corrupted copy back to *path*.  Handles both header-front
    (``save_superl8``) and streaming/footer (``FQWriter``) formats."""
    with open(path, "rb") as f:
        raw = f.read()

    hlen = struct.unpack("<Q", raw[8:16])[0]

    if hlen == 0:
        # ---- streaming / footer format ------------------------------------
        # Layout:  MAGIC (8) | hlen=0 (8) | pad → blob | header JSON |
        #          header_off (8) | header_len (8) | FOOTER_MAGIC (8)
        if raw[-8:] != FOOTER_MAGIC:
            raise RuntimeError("not a valid streaming .superl8")
        header_off, header_len = struct.unpack("<QQ", raw[-24:-8])
        header = json.loads(raw[header_off:header_off + header_len])
        modify_fn(header)
        new_hbytes = json.dumps(header, default=str).encode("utf-8")
        new_header_len = len(new_hbytes)
        blob_end = header_off                     # everything before the header
        with open(path, "wb") as f:
            f.write(raw[:blob_end])
            f.write(new_hbytes)
            f.write(struct.pack("<QQ", header_off, new_header_len))
            f.write(FOOTER_MAGIC)
    else:
        # ---- header-front format (save_superl8) ------------------------------
        header = json.loads(raw[16:16 + hlen])
        modify_fn(header)
        new_hbytes = json.dumps(header, default=str).encode("utf-8")
        new_hlen = len(new_hbytes)
        blob_start = _pad(16 + new_hlen)
        original_blob_start = _pad(16 + hlen)
        blob_data = raw[original_blob_start:]
        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<Q", new_hlen))
            f.write(new_hbytes)
            pad = blob_start - (16 + new_hlen)
            if pad > 0:
                f.write(b"\x00" * pad)
            f.write(blob_data)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.correctness
def test_valid_file_opens_and_get_works(tmp_path):
    """A well-formed .superl8 must load normally — validation is purely additive."""
    qt = _i8_weight(64, 128)
    path = str(tmp_path / "good.superl8")
    save_superl8(path, {"w": qt})
    with FQReader(path) as r:
        got = r.get_qtensor("w", "cpu")
    assert torch.equal(got.data, qt.data)
    assert torch.equal(got.scale, qt.scale)


@pytest.mark.correctness
def test_negative_offset_rejected(tmp_path):
    qt = _i8_weight(64, 128)
    path = str(tmp_path / "bad_off.superl8")
    save_superl8(path, {"w": qt})

    def _corrupt(h):
        h["tensors"]["w"]["offset"] = -128

    _rebuild_with_corrupt_header(path, _corrupt)
    with pytest.raises(ValueError, match="superl8 format"):
        FQReader(path)


@pytest.mark.correctness
def test_offset_nbytes_exceeds_mmap_rejected(tmp_path):
    qt = _i8_weight(64, 128)
    path = str(tmp_path / "bad_nbytes.superl8")
    save_superl8(path, {"w": qt})

    def _corrupt(h):
        h["tensors"]["w"]["nbytes"] = 999_999_999_999

    _rebuild_with_corrupt_header(path, _corrupt)
    with pytest.raises(ValueError, match="superl8 format"):
        FQReader(path)


@pytest.mark.correctness
def test_shape_dtype_mismatch_nbytes_rejected(tmp_path):
    qt = _i8_weight(64, 128)
    path = str(tmp_path / "bad_shape.superl8")
    save_superl8(path, {"w": qt})

    def _corrupt(h):
        e = h["tensors"]["w"]
        e["shape"] = [e["shape"][0], e["shape"][1] * 2]   # double last dim

    _rebuild_with_corrupt_header(path, _corrupt)
    with pytest.raises(ValueError, match="superl8 format"):
        FQReader(path)
