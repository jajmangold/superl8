# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""`.superl8` — a quantized-checkpoint container whose on-disk bytes ARE the resident
VRAM layout. Load = mmap + cudaMemcpy per tensor; NO dequant / repack / transpose.

Why not GGUF: GGUF is weight-only WnA16 block quant (fp16 activations), so it must
be dequantized every forward — which throws away the dp4a advantage on this fleet
(gimped fp16 tensor cores). Our runtime is W8A8: weights are per-row symmetric int8
laid out row-major with `in % 4 == 0`, which is byte-identical to how the dp4a
kernel reads them (4 consecutive int8 -> one int32, little-endian). So the resident
layout serializes with zero transform.

The container is intentionally safetensors-shaped (magic, JSON header, contiguous
blob, mmap, bounds-checked, no code execution) but adds what a quant runtime needs:
  * 128-byte tensor alignment (coalesced copies / matches the alignas(128) convention)
    vs safetensors' 8B;
  * explicit int8 <-> fp32-scale pairing (fp32 scales — fp16 overflows, per AGENTS.md);
  * BAKED transforms: `rotated` (Hadamard folded into Q/K weights offline -> no runtime
    rotation) and `smoothed` (calibrated K-mean folded in) so load is zero-compute;
  * a shard index (PP stages / MoE experts) for rank-local PARTIAL loads — mmap only
    faults in the pages a rank actually touches, so a stage/expert loads without
    reading the whole file (critical on the 250 MB/s fleet);
  * kernel-compat assertions (arch=sm70, dp4a_w8a8) that fail loud on a mismatch.
"""
from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import struct
import zlib
import dataclasses
from dataclasses import dataclass, field

import numpy as np
import torch


def _json_default(o):
    """Fallback serializer for the JSON header so a stray non-JSON object in
    ``__meta__``/config can never truncate the footer. A VLM-wrapped checkpoint
    (e.g. Qwen3.5/3.6 shipping as ``*ForConditionalGeneration`` with a nested HF
    ``VisionConfig``) carries a ``PretrainedConfig`` object that is neither a dict
    nor a dataclass; without this, ``json.dumps`` raised mid-``finalize()`` and
    left a headerless, unreadable ``.superl8`` ('bad footer magic'). Prefer a
    structured dict (HF ``.to_dict()`` / dataclass ``asdict``); last resort is
    ``str()`` so ``json.dumps`` is guaranteed never to raise on metadata."""
    to_dict = getattr(o, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)

MAGIC = b"FNI8QCK1"
ALIGN = 128                      # tensor byte alignment (coalesced loads; alignas(128))
ARCH = "sm70"
QUANT = "dp4a_w8a8"
# On-disk format version. Bump when the layout/semantics change; add the old number
# to SUPPORTED_VERSIONS only while this build can still read that layout. The reader
# NEGOTIATES this (below) so a newer file fails loudly with an upgrade hint instead of
# being silently mis-parsed under stale assumptions.
FORMAT_VERSION = 1
SUPPORTED_VERSIONS = frozenset({1})

# Kernel-ABI / resident-layout version: the dp4a on-disk BYTE layout the blobs are in
# (row-major int8, in%4==0, 4 int8 -> one little-endian int32). This is versioned
# INDEPENDENTLY of FORMAT_VERSION (the container framing) — bump it only when the
# resident byte layout the kernel copies into VRAM changes, so a reader can refuse a
# blob layout it can't dp4a. Absent in a header == legacy ABI 1 (all v1 files were 1).
KERNEL_ABI_VERSION = 1
SUPPORTED_ABIS = frozenset({1})

# Backend / resident-layout tag: which backend's byte layout the blobs are in. THIS
# build (superl8, CUDA sm_70) writes and reads "cuda-dp4a". The Vulkan backend (flint8)
# defines "vulkan-dp4a"; the two are NOT interchangeable, so a reader refuses a foreign
# layout loudly instead of copying alien bytes into VRAM. Absent == legacy "cuda-dp4a".
LAYOUT = "cuda-dp4a"
SUPPORTED_LAYOUTS = frozenset({"cuda-dp4a"})


def _provenance(
    *,
    source_model: str | None = None,
    weight_hash: str | None = None,
    calibration_fingerprint: str | None = None,
) -> dict:
    """Deterministic writer provenance stamped into every `.superl8` header — which tool,
    format, package version, kernel-ABI and backend layout wrote it, plus (when the
    converter supplies them) the source-model id, a content weight hash, and a
    calibration fingerprint. Kept timestamp-free / random-free ON PURPOSE so the same
    inputs still produce byte-identical files (the caller passes the hashes)."""
    try:                                             # lazy: __init__ imports us first
        from . import __version__ as v
    except Exception:
        v = "unknown"
    prov = {
        "tool": "superl8.format",
        "format_version": FORMAT_VERSION,
        "superl8_version": v,
        "kernel_abi_version": KERNEL_ABI_VERSION,
        "layout": LAYOUT,
        "source_model": source_model,
        "weight_hash": weight_hash,
        "calibration_fingerprint": calibration_fingerprint,
    }
    return prov


def _weight_hash(entries: dict) -> str:
    """Deterministic content hash of the checkpoint's tensor blobs, derived from the
    per-tensor CRC32s the writer already computes (no re-reading of blobs). Identical
    weights -> identical hash (content-addressed); a single changed byte flips a CRC and
    therefore the digest. Used when the converter doesn't supply a source weight hash."""
    parts = [f"{n}:{e['crc32']:08x}:{e['nbytes']}" for n, e in sorted(entries.items())]
    return "superl8-crc:" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


# Map a stored scheme -> the logical quant recipe a backend dispatches on. per_row_i8 is
# per-row symmetric int8; per_chan_i8 is per-channel; the grouped schemes are the low-bit
# weight recipes (unpacked to int8 for dp4a). `raw` reports its stored dtype (fp16/fp32).
_RECIPE_DTYPE = {
    "per_row_i8": "int8", "per_chan_i8": "int8",
    "per_group_i4": "w4a8", "per_group_w3a8": "w3a8",
}


def _recipe(entry: dict) -> dict:
    """Self-describing per-tensor quant recipe: logical dtype, per-row symmetry, group
    size, and the transforms baked in (K-smoothing / Hadamard). Derived purely from an
    entry's stored fields, so it works for legacy files that predate the `recipe` field."""
    scheme = entry.get("scheme", "raw")
    if scheme == "raw":
        dtype = {"float16": "fp16", "float32": "fp32"}.get(entry.get("dtype", ""), entry.get("dtype", "raw"))
    else:
        dtype = _RECIPE_DTYPE.get(scheme, scheme)
    transforms = []
    if entry.get("smoothed"):
        transforms.append("k-smoothing")
    if entry.get("rotated"):
        transforms.append("hadamard")
    gs = int(entry.get("group_size", 0) or 0)
    rid = dtype
    if scheme == "per_row_i8":
        rid += "-perrow-sym"
    elif scheme == "per_chan_i8":
        rid += "-perchan-sym"
    if gs:
        rid += f"-g{gs}"
    if entry.get("codebook"):
        rid += f"-{entry['codebook']}"
    if entry.get("rotated"):
        rid += "-had"
    if entry.get("smoothed"):
        rid += "-ksmooth"
    return {
        "dtype": dtype,
        "per_row_symmetric": scheme == "per_row_i8",
        "group_size": gs,
        "transforms": transforms,
        "recipe_id": rid,
    }
_DTYPE_NP = {"int8": np.int8, "uint8": np.uint8, "int32": np.int32,
             "float32": np.float32, "float16": np.float16,
             # NumPy in the pinned torch 2.10 image has no bf16 scalar dtype.
             # Read its byte-identical uint16 payload and restore the torch view in get().
             "bfloat16": np.uint16}
_DTYPE_TORCH = {"int8": torch.int8, "uint8": torch.uint8, "int32": torch.int32,
                "float32": torch.float32, "float16": torch.float16,
                "bfloat16": torch.bfloat16}


def _dtype_name(t: torch.Tensor) -> str:
    for n, dt in _DTYPE_TORCH.items():
        if t.dtype == dt:
            return n
    raise ValueError(
        f"superl8 format: unsupported dtype {t.dtype} "
        "(use int8/uint8/int32/float16/bfloat16/float32)"
    )


def _tensor_bytes(t: torch.Tensor) -> bytes:
    """Return a tensor's native byte payload without a dtype conversion.

    Viewing a flattened CPU tensor as uint8 works for every supported torch dtype,
    including bfloat16 (which ``Tensor.numpy()`` cannot represent in the pinned
    environment). It also keeps all pre-existing little-endian payloads unchanged.
    """
    cpu = t.detach().contiguous().cpu()
    return cpu.reshape(-1).view(torch.uint8).numpy().tobytes()


# GGUF quantized block geometry, by our `gguf_kquant` codebook tag: bytes per block.
# k-quants (ggml QK_K=256 super-blocks) + TQ3_4S (fork type 46, QK_TQ3_0=32) + the
# i-quant family (codebook/grid types; 256-element super-blocks except IQ4_NL's 32).
# A tag being listed here means the FORMAT can carry its raw bytes — NOT that a fused
# kernel exists. `superl8.gguf._NATIVE_FUSED` is the kernel gate; a type without one loads
# through dequant -> per_row_i8 exactly as before (superl8#317).
_GGUF_KQUANT_BLOCK = {
    "q2_k": 84, "q3_k": 110, "q4_k": 144, "q5_k": 176, "q6_k": 210,
    "tq3_4s": 16,
    "iq1_s": 50, "iq2_xxs": 66, "iq2_xs": 74, "iq2_s": 82,
    "iq3_xxs": 98, "iq3_s": 110, "iq4_xs": 136, "iq4_nl": 18,
}
# Values per block where it is not QK_K=256.
_GGUF_KQUANT_GROUP = {"tq3_4s": 32, "iq4_nl": 32}


@dataclass
class QTensor:
    """A weight to store. Schemes:
      'per_row_i8'   int8 `data` [out, in] (in%4==0 for dp4a) + fp32 `scale` [out];
      'per_chan_i8'  int8 + fp32 `scale` [.., 1, d];
      'per_group_i4' 4-bit WEIGHTS packed 2 nibbles/byte -> uint8 `data` [out, in//2]
                     + fp32 per-group `scale` [out, in//group_size]; `codebook` is
                     'int4' (uniform, signed [-7,7]) or 'nf4' (non-uniform). Halves the
                     weight footprint + read bandwidth; the kernel UNPACKS to int8 for
                     dp4a (sm_70 has no int4 matmul), so compute stays W8A8 rate.
      'per_group_w3a8' uniform 3-bit WEIGHTS in a Q3_K-style bit-plane split -> int32
                     `data` [out, (in//32)*3] (3.0 bit/wt exactly) + fp32 per-group
                     `scale` [out, in//group_size]. The `gemm_decode_w3a8` kernel unpacks
                     each 32-value group -> int8 for dp4a. A VRAM/context lever (0.75x
                     the per_group_i4 bytes), decode PARITY not faster (issue #181).
      'gguf_kquant'  RAW GGUF k-quant bytes kept RESIDENT (native-GGUF-on-the-fly):
                     uint8 `data` [out, n_superblocks*type_size]; `codebook` is the
                     GGML type tag ('q2_k'/'q3_k'/'q4_k'/'q5_k'/'q6_k' with
                     `group_size`=256 (QK_K), or 'tq3_4s' with `group_size`=32
                     (QK_TQ3_0 — the TurboQuant 3-bit block is 32 values, NOT 256)).
                     No paired scale blob — the per-32/16 sub-block scales+mins live
                     INSIDE the bytes. The fused dp4a kernel unpacks each sub-block to
                     int8 in-kernel and honors the native scales exactly (Q4_K live;
                     Q5_K/Q6_K next; TQ3_4S via the reference in superl8.linear_tq34s
                     until the fused kernel, superl8#272). Byte-identical to the GGUF
                     file, so a .gguf tensor mmaps straight into VRAM with zero
                     transcode.
      'raw'          store `data` as-is (fp16/fp32 norms, embeddings), no scale.
    """
    data: torch.Tensor
    scale: torch.Tensor | None = None
    scheme: str = "per_row_i8"
    rotated: bool = False
    hadamard_dim: int = 0
    smoothed: bool = False
    group_size: int = 0
    codebook: str = ""

    def validate(self) -> None:
        if self.scheme in ("per_row_i8", "per_chan_i8"):
            assert self.data.dtype == torch.int8, f"{self.scheme} needs int8 data"
            assert self.scale is not None and self.scale.dtype == torch.float32, \
                f"{self.scheme} needs an fp32 scale (fp16 scales overflow)"
            assert self.data.shape[-1] % 4 == 0, \
                f"dp4a needs the contraction dim %4==0, got {tuple(self.data.shape)}"
        elif self.scheme == "per_group_i4":
            assert self.data.dtype == torch.uint8, "per_group_i4 needs uint8 (packed nibbles)"
            assert self.scale is not None and self.scale.dtype == torch.float32, \
                "per_group_i4 needs an fp32 per-group scale"
            in_dim = self.data.shape[-1] * 2                 # 2 nibbles/byte
            assert self.group_size > 0 and in_dim % self.group_size == 0, \
                f"in ({in_dim}) not divisible by group_size {self.group_size}"
            assert in_dim % 4 == 0, "dp4a needs in%4==0 after unpack"
            assert self.codebook in ("int4", "nf4"), f"codebook {self.codebook!r}"
        elif self.scheme == "per_group_w3a8":
            assert self.data.dtype == torch.int32, "per_group_w3a8 needs int32 bit-planes"
            assert self.scale is not None and self.scale.dtype == torch.float32, \
                "per_group_w3a8 needs an fp32 per-group scale"
            assert self.data.shape[-1] % 3 == 0, \
                "per_group_w3a8 data must be [out, (in//32)*3]"
            in_dim = (self.data.shape[-1] // 3) * 32          # 3 int32 per 32-value group
            assert self.group_size > 0 and self.group_size % 32 == 0 and \
                in_dim % self.group_size == 0, \
                f"in ({in_dim}) not divisible by group_size {self.group_size} (must be %32)"
        elif self.scheme == "gguf_kquant":
            assert self.data.dtype == torch.uint8, "gguf_kquant needs raw uint8 GGUF bytes"
            assert self.scale is None, "gguf_kquant carries no separate scale (sub-scales are in the bytes)"
            assert self.codebook in _GGUF_KQUANT_BLOCK, \
                f"gguf_kquant type tag {self.codebook!r} " \
                f"(expect one of {', '.join(sorted(_GGUF_KQUANT_BLOCK))})"
            _TS = _GGUF_KQUANT_BLOCK[self.codebook]
            # Most types are QK_K=256 super-blocks; tq3_4s groups QK_TQ3_0=32 values
            # (16 B/block) and iq4_nl QK4_NL=32 (18 B). group_size carries which
            # tile-staging applies.
            expect_gs = _GGUF_KQUANT_GROUP.get(self.codebook, 256)
            assert self.group_size == expect_gs, (
                f"gguf_kquant {self.codebook} needs group_size {expect_gs} "
                f"(QK_TQ3_0=32, NOT QK_K=256), got {self.group_size}"
            )
            assert self.data.dim() == 2 and self.data.shape[-1] % _TS == 0, \
                f"gguf_kquant {self.codebook} data must be [out, n_superblocks*{_TS}]"
        elif self.scheme == "raw":
            assert self.scale is None, "raw tensors carry no scale"
        else:
            raise ValueError(f"unknown scheme {self.scheme!r}")


def _pad(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def save_superl8(
    path: str,
    tensors: dict[str, QTensor],
    *,
    shards: dict | None = None,
    meta: dict | None = None,
    source_model: str | None = None,
    weight_hash: str | None = None,
    calibration_fingerprint: str | None = None,
) -> None:
    """Write a `.superl8` checkpoint. Each QTensor emits its `data` blob and (if any) a
    paired `<name>.scale` blob; every blob starts 128B-aligned and is byte-identical
    to its resident layout. `shards` is an opaque partial-load index (e.g.
    {"pp_stages": [[names...]], "experts": {"L.E": [names...]}}).

    Converter provenance (all optional, all deterministic — no timestamps): `source_model`
    is the upstream HF id; `weight_hash` a caller-supplied source digest (defaults to a
    content hash of the blobs); `calibration_fingerprint` identifies the calibration
    set/recipe. They ride in the header `provenance` so a checkpoint says what made it."""
    # Two-pass, streaming: build the header (offsets/crc) in pass 1 -- materializing
    # each blob only long enough to crc it, then freeing it -- and stream the blobs
    # to disk one at a time in pass 2. Peak RAM is the input `tensors` dict plus a
    # SINGLE blob, never a second full copy of the model (the old `blobs` list held
    # the whole quantized model in RAM again, OOM-ing 160GB+ giants). Output bytes
    # are identical.
    entries: dict[str, dict] = {}
    plan: list[tuple[str, "torch.Tensor"]] = []   # (name, tensor) in write order
    cursor = 0

    def _add(name: str, t: torch.Tensor, extra: dict) -> None:
        nonlocal cursor
        raw = _tensor_bytes(t)
        cursor = _pad(cursor)
        entries[name] = {
            "dtype": _dtype_name(t), "shape": list(t.shape),
            "offset": cursor, "nbytes": len(raw), "crc32": zlib.crc32(raw) & 0xFFFFFFFF,
            **extra,
        }
        cursor += len(raw)
        plan.append((name, t))
        del raw

    for name, qt in tensors.items():
        qt.validate()
        _add(name, qt.data, {
            "scheme": qt.scheme,
            "scale": f"{name}.scale" if qt.scale is not None else None,
            "rotated": qt.rotated, "hadamard_dim": qt.hadamard_dim, "smoothed": qt.smoothed,
            "group_size": qt.group_size, "codebook": qt.codebook,
        })
        if qt.scale is not None:
            _add(f"{name}.scale", qt.scale, {"scheme": "raw", "scale": None,
                                             "rotated": False, "hadamard_dim": 0,
                                             "smoothed": False, "group_size": 0,
                                             "codebook": ""})

    for e in entries.values():
        e["recipe"] = _recipe(e)
    header = {
        "arch": ARCH, "quant": QUANT, "version": FORMAT_VERSION,
        "kernel_abi_version": KERNEL_ABI_VERSION, "layout": LAYOUT,
        "provenance": _provenance(
            source_model=source_model,
            weight_hash=weight_hash if weight_hash is not None else _weight_hash(entries),
            calibration_fingerprint=calibration_fingerprint),
        "align": ALIGN, "tensors": entries,
        "shards": shards or {}, "__meta__": meta or {},
    }
    hbytes = json.dumps(header, default=_json_default).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        blob_start = _pad(f.tell())
        f.write(b"\x00" * (blob_start - f.tell()))
        for name, t in plan:
            off = blob_start + entries[name]["offset"]
            f.write(b"\x00" * (off - f.tell()))
            f.write(_tensor_bytes(t))


FOOTER_MAGIC = b"FNI8FOOT"


class FQWriter:
    """Streaming `.superl8` writer: write each tensor's blob to disk AS IT'S PRODUCED
    (and freed), then a header + footer at the end -- so peak RAM is a single
    tensor, not the whole quantized model. `save_superl8` needs the full dict resident;
    a 160GB+ model then OOMs the converter even with plenty of disk. Byte-compatible
    blob layout; a footer format so the header (which needs every tensor's offset)
    can be written last:

        MAGIC, hlen=0 sentinel, pad, [blobs...], header, footer(header_off,
        header_len, FOOTER_MAGIC).

    FQReader detects the hlen==0 sentinel and reads the footer. Existing
    header-front files are unchanged."""

    def __init__(self, path: str):
        self._f = open(path, "wb")
        self._f.write(MAGIC)
        self._f.write(struct.pack("<Q", 0))          # hlen=0 sentinel -> footer format
        self._data_start = _pad(self._f.tell())      # blobs begin here (== 16)
        self._f.write(b"\x00" * (self._data_start - self._f.tell()))
        self._entries: dict[str, dict] = {}

    def _add_blob(self, name: str, t: torch.Tensor, extra: dict) -> None:
        raw = _tensor_bytes(t)
        pos = _pad(self._f.tell())
        self._f.write(b"\x00" * (pos - self._f.tell()))
        self._entries[name] = {
            "dtype": _dtype_name(t), "shape": list(t.shape),
            "offset": pos - self._data_start, "nbytes": len(raw),
            "crc32": zlib.crc32(raw) & 0xFFFFFFFF, **extra,
        }
        self._f.write(raw)
        del raw

    def add(self, name: str, qt: "QTensor") -> None:
        qt.validate()
        self._add_blob(name, qt.data, {
            "scheme": qt.scheme,
            "scale": f"{name}.scale" if qt.scale is not None else None,
            "rotated": qt.rotated, "hadamard_dim": qt.hadamard_dim, "smoothed": qt.smoothed,
            "group_size": qt.group_size, "codebook": qt.codebook,
        })
        if qt.scale is not None:
            self._add_blob(f"{name}.scale", qt.scale, {
                "scheme": "raw", "scale": None, "rotated": False, "hadamard_dim": 0,
                "smoothed": False, "group_size": 0, "codebook": "",
            })

    def finalize(
        self,
        *,
        shards: dict | None = None,
        meta: dict | None = None,
        source_model: str | None = None,
        weight_hash: str | None = None,
        calibration_fingerprint: str | None = None,
    ) -> None:
        for e in self._entries.values():
            e["recipe"] = _recipe(e)
        header = {
            "arch": ARCH, "quant": QUANT, "version": FORMAT_VERSION, "align": ALIGN,
            "kernel_abi_version": KERNEL_ABI_VERSION, "layout": LAYOUT,
            "provenance": _provenance(
                source_model=source_model,
                weight_hash=weight_hash if weight_hash is not None else _weight_hash(self._entries),
                calibration_fingerprint=calibration_fingerprint),
            "tensors": self._entries, "shards": shards or {}, "__meta__": meta or {},
        }
        hbytes = json.dumps(header, default=_json_default).encode("utf-8")
        header_off = self._f.tell()
        self._f.write(hbytes)
        self._f.write(struct.pack("<QQ", header_off, len(hbytes)))
        self._f.write(FOOTER_MAGIC)
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *a):
        if not self._f.closed:
            self._f.close()


@dataclass
class FQReader:
    """mmap reader for a `.superl8` checkpoint. Lazy: only tensors you `get()` fault in,
    so a PP stage / MoE expert loads without reading the whole file."""
    path: str
    header: dict = field(init=False)
    _blob_start: int = field(init=False)
    _mm: mmap.mmap = field(init=False)

    def __post_init__(self) -> None:
        self._f = open(self.path, "rb")
        magic = self._f.read(8)
        if magic != MAGIC:
            raise ValueError(f"superl8 format: bad magic {magic!r} (not a .superl8 checkpoint)")
        (hlen,) = struct.unpack("<Q", self._f.read(8))
        if hlen == 0:
            # Streaming (footer) format: header + (header_off, header_len, FOOTER_MAGIC)
            # live at the end; blobs start right after the 16B preamble.
            self._f.seek(-24, os.SEEK_END)
            header_off, header_len = struct.unpack("<QQ", self._f.read(16))
            if self._f.read(8) != FOOTER_MAGIC:
                raise ValueError("superl8 format: bad footer magic (truncated streaming .superl8?)")
            self._f.seek(header_off)
            self.header = json.loads(self._f.read(header_len))
            self._blob_start = _pad(16)
        else:
            self.header = json.loads(self._f.read(hlen))
            self._blob_start = _pad(16 + hlen)
        # Version negotiation: reject a file written by a newer/unknown format with a
        # clear upgrade hint, rather than silently reading it under v1 assumptions. A
        # missing field means a legacy v1 file (all writers have always emitted it).
        ver = self.header.get("version", 1)
        if ver not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"superl8 format: unsupported format version {ver} "
                f"(this build reads {sorted(SUPPORTED_VERSIONS)}) — upgrade superl8 to read this file")
        if self.header.get("arch") != ARCH or self.header.get("quant") != QUANT:
            raise ValueError(
                f"superl8 format: arch/quant mismatch "
                f"({self.header.get('arch')}/{self.header.get('quant')} != {ARCH}/{QUANT})")
        # Backend-layout + kernel-ABI negotiation (mirrors the version gate above). A
        # MISSING field means a legacy pre-follow-on v1 file, which was always cuda-dp4a
        # ABI 1 — so default to this build's values and it reads unchanged (back-compat).
        # A PRESENT-but-unsupported value fails loudly: a foreign backend layout (flint8's
        # vulkan-dp4a) or a newer dp4a byte-layout ABI must not be copied into VRAM under
        # this build's stale assumptions.
        layout = self.header.get("layout", LAYOUT)
        if layout not in SUPPORTED_LAYOUTS:
            raise ValueError(
                f"superl8 format: unsupported backend layout {layout!r} "
                f"(this build reads {sorted(SUPPORTED_LAYOUTS)}) — a {layout!r} checkpoint "
                f"needs the matching backend (e.g. flint8 for 'vulkan-dp4a')")
        abi = self.header.get("kernel_abi_version", KERNEL_ABI_VERSION)
        if abi not in SUPPORTED_ABIS:
            raise ValueError(
                f"superl8 format: unsupported kernel-ABI / resident-layout version {abi} "
                f"(this build reads {sorted(SUPPORTED_ABIS)}) — upgrade superl8 to read this file")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        # Bounds-check every tensor entry so a malformed header fails LOUDLY at
        # open time instead of silently reading wrong pages inside _raw()/get().
        for _name, _e in self.header["tensors"].items():
            if _e["offset"] < 0:
                raise ValueError(
                    f"superl8 format: negative offset {_e['offset']} for tensor {_name!r}")
            blob_end = self._blob_start + _e["offset"] + _e["nbytes"]
            if blob_end > len(self._mm):
                raise ValueError(
                    f"superl8 format: tensor {_name!r} offset+{_e['nbytes']}B "
                    f"exceeds file size ({blob_end} > {len(self._mm)})")
            _itemsize = np.dtype(_DTYPE_NP[_e["dtype"]]).itemsize
            _expected = math.prod(_e["shape"]) * _itemsize
            if _expected != _e["nbytes"]:
                raise ValueError(
                    f"superl8 format: tensor {_name!r} shape {_e['shape']} * {_e['dtype']} "
                    f"= {_expected}B, but header declares {_e['nbytes']}B")

    # -- introspection -------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return [n for n in self.header["tensors"] if not n.endswith(".scale")]

    @property
    def shards(self) -> dict:
        return self.header["shards"]

    def info(self, name: str) -> dict:
        return self.header["tensors"][name]

    @property
    def provenance(self) -> dict:
        """Writer provenance: tool, format/kernel-ABI versions, backend layout, package
        version, and (when the converter stamped them) source model + weight hash +
        calibration fingerprint. Empty-ish for a legacy file that predates the fields."""
        return self.header.get("provenance", {})

    def recipe(self, name: str) -> dict:
        """Per-tensor quant recipe (dtype / per-row symmetry / group size / transforms /
        recipe_id). Prefers the stored `recipe`; falls back to deriving it from the
        entry's fields so legacy files that predate the field are still self-describing.
        A backend can inspect this to refuse a recipe it doesn't implement."""
        e = self.header["tensors"][name]
        return e.get("recipe") or _recipe(e)

    # -- loads ---------------------------------------------------------------
    def _raw(self, name: str) -> np.ndarray:
        e = self.header["tensors"][name]
        base = self._blob_start + e["offset"]
        buf = self._mm[base:base + e["nbytes"]]          # faults in only these pages
        arr = np.frombuffer(buf, dtype=_DTYPE_NP[e["dtype"]])
        return arr.reshape(e["shape"])

    def get(self, name: str, device: str = "cpu", *, verify: bool = False) -> torch.Tensor:
        """Return one tensor on `device`. mmap->host->(single H2D) copy, no transcode."""
        e = self.header["tensors"][name]
        arr = self._raw(name)
        if verify and (zlib.crc32(arr.tobytes()) & 0xFFFFFFFF) != e["crc32"]:
            raise ValueError(f"superl8 format: crc mismatch on {name!r} (corrupt file)")
        tensor = torch.from_numpy(arr.copy())
        if e["dtype"] == "bfloat16":
            tensor = tensor.view(torch.bfloat16)
        return tensor.to(device)

    def get_qtensor(self, name: str, device: str = "cpu") -> QTensor:
        """Return a weight together with its paired scale + baked-transform flags."""
        e = self.header["tensors"][name]
        scale = self.get(e["scale"], device) if e.get("scale") else None
        return QTensor(self.get(name, device), scale, e["scheme"],
                       e.get("rotated", False), e.get("hadamard_dim", 0),
                       e.get("smoothed", False), e.get("group_size", 0),
                       e.get("codebook", ""))

    def load_many(self, names: list[str], device: str = "cpu") -> dict[str, QTensor]:
        return {n: self.get_qtensor(n, device) for n in names}

    def verify(self) -> bool:
        """CRC every tensor (corruption check). Touches the whole file."""
        for name in self.header["tensors"]:
            self.get(name, verify=True)
        return True

    def close(self) -> None:
        self._mm.close()
        self._f.close()

    def __enter__(self) -> "FQReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
