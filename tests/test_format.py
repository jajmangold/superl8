# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""`.superl8` weight container — bytes on disk ARE the resident VRAM layout. Tests first.

The point of the format is a ZERO-transform load: mmap + copy, no dequant/repack.
So the load-back must be BYTE-IDENTICAL to what was saved, tensors must be 128B
aligned, int8/int4 weights must round-trip with their fp32 scales + baked flags
(rotated / smoothed / group_size / codebook), a shard must load without touching
other tensors, and a corrupt byte must be caught by the CRC.
"""
import os
import struct

import pytest
import torch

from superl8 import FQReader, FQWriter, QTensor, save_superl8
from superl8.format import (
    ALIGN,
    FOOTER_MAGIC,
    KERNEL_ABI_VERSION,
    LAYOUT,
    MAGIC,
    SUPPORTED_ABIS,
    SUPPORTED_LAYOUTS,
    SUPPORTED_VERSIONS,
)


def _i8_weight(out, in_, device):
    w = torch.randn(out, in_, device=device)
    scale = w.abs().amax(-1, keepdim=True) / 127.0
    q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
    return QTensor(q, scale.squeeze(-1).float(), scheme="per_row_i8", rotated=True, hadamard_dim=in_)


def _i4_weight(out, in_, group, device, codebook="int4"):
    """Uniform signed int4 weight, per-group scale, nibble-packed (2/byte)."""
    w = torch.randn(out, in_, device=device)
    wg = w.reshape(out, in_ // group, group)
    scale = wg.abs().amax(-1, keepdim=True) / 7.0
    codes = torch.round(wg / scale).clamp_(-7, 7).reshape(out, in_).to(torch.int64)
    u = (codes & 0xF).to(torch.int32)                       # 4-bit two's complement
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).to(torch.uint8)  # [out, in//2]
    return QTensor(packed, scale.squeeze(-1).float(), scheme="per_group_i4",
                   group_size=group, codebook=codebook, rotated=True, hadamard_dim=in_)


@pytest.mark.correctness
def test_int8_weight_roundtrip_byte_identical(device, tmp_path):
    qt = _i8_weight(256, 512, device)
    path = str(tmp_path / "w.superl8")
    save_superl8(path, {"layer.q_proj.weight": qt})
    with FQReader(path) as r:
        got = r.get_qtensor("layer.q_proj.weight", device)
    assert torch.equal(got.data, qt.data)           # byte-identical int8
    assert torch.equal(got.scale, qt.scale)         # fp32 scale preserved
    assert got.scheme == "per_row_i8" and got.rotated and got.hadamard_dim == 512


@pytest.mark.correctness
def test_int4_weight_roundtrip_and_halves_bytes(device, tmp_path):
    i8 = _i8_weight(256, 512, device)
    i4 = _i4_weight(256, 512, 128, device)
    path = str(tmp_path / "w.superl8")
    save_superl8(path, {"w8": i8, "w4": i4})
    with FQReader(path) as r:
        g4 = r.get_qtensor("w4", device)
        assert torch.equal(g4.data, i4.data)        # packed nibbles byte-identical
        assert torch.equal(g4.scale, i4.scale)
        assert g4.scheme == "per_group_i4" and g4.group_size == 128 and g4.codebook == "int4"
        # 4-bit weight payload is half the int8 payload for the same [out,in].
        assert r.info("w4")["nbytes"] * 2 == r.info("w8")["nbytes"]


@pytest.mark.correctness
def test_nf4_codebook_flag_preserved(device, tmp_path):
    i4 = _i4_weight(128, 256, 64, device, codebook="nf4")
    path = str(tmp_path / "w.superl8")
    save_superl8(path, {"w": i4})
    with FQReader(path) as r:
        assert r.get_qtensor("w", device).codebook == "nf4"


@pytest.mark.correctness
def test_tensors_are_128B_aligned(device, tmp_path):
    tensors = {f"t{i}": _i8_weight(64 + i, 128, device) for i in range(4)}
    path = str(tmp_path / "w.superl8")
    save_superl8(path, tensors)
    with FQReader(path) as r:
        for name in r.header["tensors"]:
            assert r.info(name)["offset"] % ALIGN == 0, f"{name} not {ALIGN}B aligned"


@pytest.mark.correctness
def test_raw_tensor_and_dtypes(device, tmp_path):
    norm = QTensor(torch.randn(512, device=device, dtype=torch.float16), scheme="raw")
    emb = QTensor(torch.randn(1000, 512, device=device, dtype=torch.float32), scheme="raw")
    path = str(tmp_path / "w.superl8")
    save_superl8(path, {"norm": norm, "emb": emb})
    with FQReader(path) as r:
        assert r.get("norm", device).dtype == torch.float16
        assert torch.equal(r.get("emb", device), emb.data)


@pytest.mark.correctness
def test_raw_bfloat16_roundtrips_byte_exact_in_both_writers(device, tmp_path):
    """BF16 has no NumPy scalar dtype in the pinned stack; writers must preserve
    its raw uint16 payload instead of converting it through Tensor.numpy()."""
    raw = torch.tensor(
        [[1.0, -2.5, float("inf")], [float("nan"), 0.0, 3.25]],
        device=device,
        dtype=torch.bfloat16,
    )
    tensor = QTensor(raw, scheme="raw")

    front = str(tmp_path / "bf16-front.superl8")
    save_superl8(front, {"raw_bf16": tensor})

    stream = str(tmp_path / "bf16-stream.superl8")
    with FQWriter(stream) as writer:
        writer.add("raw_bf16", tensor)
        writer.finalize()

    for path in (front, stream):
        with FQReader(path) as reader:
            loaded = reader.get("raw_bf16", device, verify=True)
            assert reader.info("raw_bf16")["dtype"] == "bfloat16"
            assert loaded.dtype == torch.bfloat16
            assert torch.equal(loaded.view(torch.uint16), raw.view(torch.uint16))


@pytest.mark.correctness
def test_shard_partial_load(device, tmp_path):
    """A PP-stage / expert shard loads only its tensors — the shard index names them
    and the mmap reader returns exactly those without needing the rest."""
    tensors = {f"model.layers.{i}.mlp.weight": _i8_weight(128, 256, device) for i in range(6)}
    shards = {"pp_stages": [[f"model.layers.{i}.mlp.weight" for i in s]
                            for s in ([0, 1, 2], [3, 4, 5])]}
    path = str(tmp_path / "w.superl8")
    save_superl8(path, tensors, shards=shards)
    with FQReader(path) as r:
        stage1 = r.shards["pp_stages"][1]
        loaded = r.load_many(stage1, device)
        assert set(loaded) == set(stage1)
        assert "model.layers.0.mlp.weight" not in loaded   # stage 0 untouched
        assert torch.equal(loaded[stage1[0]].data, tensors[stage1[0]].data)


@pytest.mark.correctness
def test_crc_catches_corruption(device, tmp_path):
    path = str(tmp_path / "w.superl8")
    save_superl8(path, {"w": _i8_weight(64, 128, device)})
    with FQReader(path) as r:
        assert r.verify()                                  # clean file verifies
        off = r._blob_start + r.info("w")["offset"]
    with open(path, "r+b") as f:                           # flip one payload byte
        f.seek(off)
        b = f.read(1)
        f.seek(off)
        f.write(bytes([b[0] ^ 0xFF]))
    with FQReader(path) as r:
        with pytest.raises(ValueError, match="crc"):
            r.get("w", device, verify=True)


@pytest.mark.correctness
def test_bad_magic_and_arch_rejected(tmp_path):
    good = str(tmp_path / "w.superl8")
    save_superl8(good, {"w": _i8_weight(64, 128, "cpu")})
    bad = str(tmp_path / "bad.superl8")
    with open(good, "rb") as f:
        blob = f.read()
    with open(bad, "wb") as f:
        f.write(b"NOTFNI8_" + blob[8:])
    with pytest.raises(ValueError, match="magic"):
        FQReader(bad)
    assert MAGIC == b"FNI8QCK1" and struct.calcsize("<Q") == 8


@pytest.mark.correctness
def test_streaming_writer_byte_identical_to_save_superl8(device, tmp_path):
    """FQWriter (streaming, footer-format) and save_superl8 (header-front) must produce
    the SAME loaded tensors — the converter uses FQWriter to avoid holding the whole
    quantized model in RAM, so a footer file must read back identically."""
    tensors = {
        "layer.q_proj.weight": _i8_weight(256, 512, device),
        "layer.mlp.w4": _i4_weight(128, 256, 64, device),
        "norm": QTensor(torch.randn(512, device=device, dtype=torch.float16), scheme="raw"),
    }
    front = str(tmp_path / "front.superl8")
    save_superl8(front, tensors, meta={"arch": "test", "weight_bits": 8})

    stream = str(tmp_path / "stream.superl8")
    with FQWriter(stream) as w:
        for name, qt in tensors.items():
            w.add(name, qt)
        w.finalize(meta={"arch": "test", "weight_bits": 8})

    with FQReader(stream) as rs, FQReader(front) as rf:
        assert set(rs.names) == set(rf.names)
        assert rs.header["__meta__"] == rf.header["__meta__"]
        for name in tensors:
            gs, gf = rs.get_qtensor(name, device), rf.get_qtensor(name, device)
            assert torch.equal(gs.data, gf.data), name
            if gf.scale is not None:
                assert torch.equal(gs.scale, gf.scale), name
            assert gs.scheme == gf.scheme
            assert rs.info(name)["offset"] % ALIGN == 0     # streaming stays aligned
        assert rs.verify()                                  # footer-format CRCs check


@pytest.mark.correctness
def test_nonserializable_meta_object_does_not_truncate(device, tmp_path):
    """A VLM-wrapped checkpoint (Qwen3.5/3.6 ship as *ForConditionalGeneration) carries
    a nested HF `VisionConfig` in its meta — a `PretrainedConfig`, NOT a dict or
    dataclass. `finalize()`'s json.dumps must serialize it (via `.to_dict()`) instead
    of raising, which previously left a headerless, unreadable `.superl8` ('bad footer
    magic'). Regression: the Qwen3.6-27B 4-bit convert crashed exactly here."""

    class _FakeVisionConfig:            # mimics HF PretrainedConfig: has to_dict(), not JSON-native
        def __init__(self, d):
            self._d = d

        def to_dict(self):
            return dict(self._d)

    vc = {"depth": 32, "hidden_size": 1152}
    meta = {"arch": "qwen3_6_text", "weight_bits": 4, "vision_config": _FakeVisionConfig(vc)}
    qt = _i8_weight(64, 128, device)

    # streaming (FQWriter) path — the exact writer that crashed at footer time
    stream = str(tmp_path / "vl_stream.superl8")
    with FQWriter(stream) as w:
        w.add("layer.q_proj.weight", qt)
        w.finalize(meta=meta)                               # must NOT raise
    with FQReader(stream) as r:
        assert r.verify()                                   # footer present + CRCs ok
        assert r.header["__meta__"]["vision_config"] == vc  # serialized via to_dict()
        assert r.header["__meta__"]["arch"] == "qwen3_6_text"

    # header-front (save_superl8) path — same guarantee
    front = str(tmp_path / "vl_front.superl8")
    save_superl8(front, {"layer.q_proj.weight": qt}, meta=meta)
    with FQReader(front) as r:
        assert r.header["__meta__"]["vision_config"] == vc


@pytest.mark.correctness
def test_streaming_writer_footer_sentinel(device, tmp_path):
    """A streaming file starts with the hlen==0 sentinel (header lives in the footer);
    a save_superl8 file does not. The reader dispatches on it."""
    stream = str(tmp_path / "s.superl8")
    with FQWriter(stream) as w:
        w.add("w", _i8_weight(64, 128, device))
        w.finalize()
    with open(stream, "rb") as f:
        assert f.read(8) == MAGIC
        (hlen,) = struct.unpack("<Q", f.read(8))
        assert hlen == 0                                    # footer-format sentinel


@pytest.mark.correctness
def test_unsupported_format_version_rejected(device, tmp_path, monkeypatch):
    """A file written by a NEWER .superl8 format than this build supports must fail loudly
    with a version error (upgrade hint), not be silently read under v1 assumptions."""
    import superl8.format as fmt
    future = max(fmt.SUPPORTED_VERSIONS) + 1
    monkeypatch.setattr(fmt, "FORMAT_VERSION", future)          # pretend a newer writer
    path = str(tmp_path / "w.superl8")
    save_superl8(path, {"w": _i8_weight(64, 128, device)})
    monkeypatch.undo()                                          # reader = current build
    with pytest.raises(ValueError, match="version"):
        FQReader(path)
    assert future not in SUPPORTED_VERSIONS


@pytest.mark.correctness
def test_bad_footer_magic_rejected(device, tmp_path):
    """A streaming (footer-format) file with a corrupt trailing FOOTER_MAGIC is a
    truncated/garbage checkpoint — the reader must reject it, not index random bytes."""
    stream = str(tmp_path / "s.superl8")
    with FQWriter(stream) as w:
        w.add("w", _i8_weight(64, 128, device))
        w.finalize()
    assert FOOTER_MAGIC == b"FNI8FOOT"
    with open(stream, "r+b") as f:                             # clobber the 8B footer magic
        f.seek(-len(FOOTER_MAGIC), os.SEEK_END)
        f.write(b"XXXXXXXX")
    with pytest.raises(ValueError, match="footer"):
        FQReader(stream)


@pytest.mark.correctness
def test_writer_provenance_stamped(device, tmp_path):
    """Every .superl8 carries deterministic writer provenance (tool + format + superl8 version)
    for reproducibility/debugging — and both writers stamp it identically."""
    front = str(tmp_path / "front.superl8")
    save_superl8(front, {"w": _i8_weight(64, 128, device)})
    stream = str(tmp_path / "stream.superl8")
    with FQWriter(stream) as w:
        w.add("w", _i8_weight(64, 128, device))
        w.finalize()
    for path in (front, stream):
        with FQReader(path) as r:
            prov = r.header["provenance"]
            assert prov["tool"] == "superl8.format"
            assert prov["format_version"] in SUPPORTED_VERSIONS
            assert isinstance(prov["superl8_version"], str) and prov["superl8_version"]


@pytest.mark.correctness
def test_provenance_is_deterministic(device, tmp_path):
    """Provenance is timestamp-free on purpose: identical inputs -> byte-identical files,
    so a convert stays reproducible."""
    a, b = str(tmp_path / "a.superl8"), str(tmp_path / "b.superl8")
    qt = _i8_weight(64, 128, device)
    save_superl8(a, {"w": qt})
    save_superl8(b, {"w": qt})
    with FQReader(a) as ra, FQReader(b) as rb:
        assert ra.header["provenance"] == rb.header["provenance"]
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read() == fb.read()                          # fully byte-identical


@pytest.mark.correctness
def test_int8_weight_rejects_odd_contraction(device):
    w = torch.zeros(16, 30, dtype=torch.int8, device=device)   # 30 % 4 != 0
    with pytest.raises(AssertionError, match="%4"):
        QTensor(w, torch.ones(16, device=device), scheme="per_row_i8").validate()


# ============================================================================
# Phase 1-2 platform follow-ons (#100): meaningful converter provenance,
# kernel-ABI + backend-layout tags, per-tensor recipe IDs, cross-repo compat.
# All ADDITIVE: a legacy v1 file lacking every one of these fields must still load.
# ============================================================================


def _write_legacy_v1(path, name, qt):
    """Emit a genuinely OLD-STYLE (#100-era) header-front .superl8 that predates the
    provenance/ABI/layout/recipe follow-ons — no kernel_abi_version, no layout, no
    per-tensor recipe. Proves the reader treats those absent fields as legacy v1."""
    import json as _json
    import zlib as _zlib

    def _pad(n):
        return (n + ALIGN - 1) // ALIGN * ALIGN

    raw = qt.data.detach().contiguous().cpu().numpy().tobytes()
    sraw = qt.scale.detach().contiguous().cpu().numpy().tobytes()
    entries = {
        name: {"dtype": "int8", "shape": list(qt.data.shape), "offset": 0,
               "nbytes": len(raw), "crc32": _zlib.crc32(raw) & 0xFFFFFFFF,
               "scheme": "per_row_i8", "scale": f"{name}.scale",
               "rotated": qt.rotated, "hadamard_dim": qt.hadamard_dim,
               "smoothed": qt.smoothed, "group_size": 0, "codebook": ""},
        f"{name}.scale": {"dtype": "float32", "shape": list(qt.scale.shape),
                          "offset": _pad(len(raw)), "nbytes": len(sraw),
                          "crc32": _zlib.crc32(sraw) & 0xFFFFFFFF, "scheme": "raw",
                          "scale": None, "rotated": False, "hadamard_dim": 0,
                          "smoothed": False, "group_size": 0, "codebook": ""},
    }
    header = {"arch": "sm70", "quant": "dp4a_w8a8", "version": 1, "align": ALIGN,
              "provenance": {"tool": "superl8.format", "format_version": 1, "superl8_version": "0.0.0"},
              "tensors": entries, "shards": {}, "__meta__": {}}
    hbytes = _json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        blob_start = _pad(f.tell())
        f.write(b"\x00" * (blob_start - f.tell()))
        f.write(raw)
        f.write(b"\x00" * (_pad(len(raw)) - len(raw)))
        f.write(sraw)


@pytest.mark.correctness
def test_kernel_abi_and_layout_stamped(device, tmp_path):
    """Every .superl8 stamps the resident dp4a byte-layout ABI version and the backend
    layout tag (header + provenance), from BOTH writers — so a checkpoint says exactly
    what ABI/backend it targets."""
    front = str(tmp_path / "front.superl8")
    save_superl8(front, {"w": _i8_weight(64, 128, device)})
    stream = str(tmp_path / "stream.superl8")
    with FQWriter(stream) as w:
        w.add("w", _i8_weight(64, 128, device))
        w.finalize()
    for path in (front, stream):
        with FQReader(path) as r:
            assert r.header["kernel_abi_version"] == KERNEL_ABI_VERSION
            assert r.header["layout"] == LAYOUT == "cuda-dp4a"
            prov = r.header["provenance"]
            assert prov["kernel_abi_version"] == KERNEL_ABI_VERSION
            assert prov["layout"] == "cuda-dp4a"


@pytest.mark.correctness
def test_converter_provenance_fields_roundtrip(device, tmp_path):
    """Converter provenance is meaningful: a real superl8 package version (never 0.0.0),
    plus source-model id, a weight hash, and a calibration fingerprint round-trip in
    the header — deterministically (caller-supplied hashes, no timestamps)."""
    front = str(tmp_path / "front.superl8")
    save_superl8(front, {"w": _i8_weight(64, 128, device)},
              source_model="Qwen/Qwen3.5-4B",
              weight_hash="sha256:deadbeef",
              calibration_fingerprint="pileval-512:cafef00d")
    with FQReader(front) as r:
        prov = r.header["provenance"]
        assert prov["superl8_version"] != "0.0.0" and prov["superl8_version"]
        assert prov["source_model"] == "Qwen/Qwen3.5-4B"
        assert prov["weight_hash"] == "sha256:deadbeef"
        assert prov["calibration_fingerprint"] == "pileval-512:cafef00d"


@pytest.mark.correctness
def test_auto_weight_hash_is_deterministic_and_content_addressed(device, tmp_path):
    """When the caller gives no weight hash, the writer derives one deterministically
    from the tensor blobs (content-addressed) so identical models get identical files
    and a changed weight changes the hash."""
    a, b = str(tmp_path / "a.superl8"), str(tmp_path / "b.superl8")
    qt = _i8_weight(64, 128, device)
    save_superl8(a, {"w": qt})
    save_superl8(b, {"w": qt})
    with FQReader(a) as ra, FQReader(b) as rb:
        ha = ra.header["provenance"]["weight_hash"]
        assert ha and ha == rb.header["provenance"]["weight_hash"]
    diff = str(tmp_path / "c.superl8")
    save_superl8(diff, {"w": _i8_weight(64, 128, device)})   # different random weight
    with FQReader(diff) as rc:
        assert rc.header["provenance"]["weight_hash"] != ha


@pytest.mark.correctness
def test_per_tensor_recipe_ids(device, tmp_path):
    """Each tensor is self-describing: its recipe records logical dtype, per-row
    symmetry, group size, and applied transforms (K-smoothing / Hadamard) so a backend
    can refuse a recipe it doesn't implement."""
    i8 = _i8_weight(256, 512, device)          # rotated=True, hadamard_dim=512
    i4 = _i4_weight(128, 256, 64, device)      # w4a8, group 64, rotated
    path = str(tmp_path / "w.superl8")
    save_superl8(path, {"w8": i8, "w4": i4})
    with FQReader(path) as r:
        r8 = r.recipe("w8")
        assert r8["dtype"] == "int8" and r8["per_row_symmetric"] is True
        assert "hadamard" in r8["transforms"]
        assert r8["recipe_id"]
        r4 = r.recipe("w4")
        assert r4["dtype"] == "w4a8" and r4["group_size"] == 64
        assert "hadamard" in r4["transforms"]


@pytest.mark.correctness
def test_unsupported_layout_rejected(device, tmp_path, monkeypatch):
    """A checkpoint whose blobs are in another backend's byte-layout (e.g. flint8's
    vulkan-dp4a) must fail LOUDLY with an upgrade hint — this is what stops superl8 and
    flint8 silently misreading each other's files."""
    import superl8.format as fmt
    monkeypatch.setattr(fmt, "LAYOUT", "vulkan-dp4a")           # pretend a flint8 writer
    path = str(tmp_path / "w.superl8")
    save_superl8(path, {"w": _i8_weight(64, 128, device)})
    monkeypatch.undo()                                          # reader = this (cuda) build
    with pytest.raises(ValueError, match="layout"):
        FQReader(path)
    assert "vulkan-dp4a" not in SUPPORTED_LAYOUTS


@pytest.mark.correctness
def test_unsupported_kernel_abi_rejected(device, tmp_path, monkeypatch):
    """A file written against a NEWER dp4a resident-layout ABI than this build knows
    must fail loudly (upgrade hint), not be copied into VRAM under stale byte-layout
    assumptions."""
    import superl8.format as fmt
    future = max(fmt.SUPPORTED_ABIS) + 1
    monkeypatch.setattr(fmt, "KERNEL_ABI_VERSION", future)
    path = str(tmp_path / "w.superl8")
    save_superl8(path, {"w": _i8_weight(64, 128, device)})
    monkeypatch.undo()
    with pytest.raises(ValueError, match="ABI"):
        FQReader(path)
    assert future not in SUPPORTED_ABIS


@pytest.mark.correctness
def test_legacy_v1_file_without_new_fields_still_loads(device, tmp_path):
    """BACK-COMPAT: a pre-follow-on v1 file (no kernel_abi_version, no layout, no
    per-tensor recipe) must load unchanged — absent fields == legacy cuda-dp4a v1."""
    qt = _i8_weight(64, 128, "cpu")
    path = str(tmp_path / "legacy.superl8")
    _write_legacy_v1(path, "layer.q_proj.weight", qt)
    with FQReader(path) as r:
        assert "kernel_abi_version" not in r.header       # genuinely legacy
        assert "layout" not in r.header
        got = r.get_qtensor("layer.q_proj.weight", device)
        assert torch.equal(got.data, qt.data.to(device))
        assert r.verify()
        # absent per-tensor recipe still yields a sane self-describing recipe
        assert r.recipe("layer.q_proj.weight")["dtype"] == "int8"
