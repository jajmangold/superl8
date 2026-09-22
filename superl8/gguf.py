# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Load ANY GGUF (incl. Unsloth UD Dynamic mixed-quant) for the sm_70 dp4a path.

A GGUF file is a per-tensor menu of ggml quant types. Unsloth "UD" (Dynamic)
files deliberately MIX types in one file — protected tensors at Q6_K/Q8_0, bulk
at Q4_K/Q3_K/Q2_K/IQ*. So the loader dispatches PER TENSOR by type. Three paths:

  1. NATIVE-FUSED (resident in native GGUF bytes, unpacked in-kernel by __dp4a):
     Q2_K / Q3_K / Q4_K / Q5_K / Q6_K -> ``gguf_kquant`` QTensor -> superl8.linear_q{2,3,4,5,6}k.
     All five k-quant types stay resident (VRAM parity, single card) and the
     dequant is FUSED into the matmul. See csrc/docs/gguf-fused-kquant-dp4a.md.
     TQ3_4S (type 46, the TurboQuant fork) also stays native-resident; it routes to
     the CPU reference (``superl8.linear_tq34s``) until its fused kernel lands (superl8#272).
  2. REQUANT-I8 (dequant once at load -> per-row symmetric int8 -> gemm_w8a8 dp4a):
     the universal fast fallback for every other quantizable type (Q8_0,
     legacy Q4_0/Q4_1/Q5_0/Q5_1, and the IQ codebook types). Loads
     CORRECTLY on the fast dp4a path; quality is bounded by the source quant (int8
     re-quant of an already-quantized weight adds negligible error since int8 has
     more precision than any of these sources). Resident as int8 (1 B/weight) — a
     bit more VRAM than the native sub-4-bit layout, acceptable for the minority of
     UD tensors that use these types.
  3. RAW (fp16): float tensors (F32/F16/BF16) and non-2D weights (norms,
     embeddings, biases) -> ``raw`` QTensor, used as-is.

This makes "load any gguf" TRUE today: all five k-quant types take the
native fused path; everything else loads via requant-i8; nothing is unsupported.

Requires the ``gguf`` package (llama.cpp's python reader + dequantizer).
"""

from __future__ import annotations

import numpy as np
import torch

# --- dispatch classes by ggml type NAME (stable across gguf-py versions) -----
# NATIVE-FUSED: kept in native GGUF super-block bytes, unpacked in-kernel. All 5
# k-quants have fused dp4a tile+decode kernels (Q3_K unblocks 27B-Q3_K_S single card);
# TQ3_4S (type 46) is native-resident too, served via the CPU reference
# (superl8.linear_tq34s) until its fused kernel (superl8#272). Note QK_TQ3_0=32 — the
# TQ3_4S block is 16 B / 32 values, NOT the QK_K=256 super-block.
_NATIVE_FUSED = {"Q2_K": "q2_k", "Q3_K": "q3_k", "Q4_K": "q4_k", "Q5_K": "q5_k", "Q6_K": "q6_k",
                 "TQ3_4S": "tq3_4s",
                 # i-quant (superl8#317): all seven now fused. IQ3_S/IQ4_XS/IQ3_XXS have
                 # decode AND tile kernels; the other four are decode-only so far.
                 "IQ3_S": "iq3_s", "IQ4_XS": "iq4_xs", "IQ3_XXS": "iq3_xxs",
                 "IQ2_S": "iq2_s", "IQ2_XS": "iq2_xs",
                 "IQ2_XXS": "iq2_xxs", "IQ1_S": "iq1_s"}
# The released `gguf` package stops at Q1_0 and cannot name TQ3_4S (type 46 is a
# turbo-tan fork type), so fall back to the numeric ggml_type for native dispatch.
_NATIVE_FUSED_BY_TYPE = {46: "tq3_4s"}
# Block byte size per native tag. Sourced from superl8.format so the loader and the QTensor
# validator cannot disagree; includes the i-quant tags whose kernels are still pending
# (superl8#317) — being here is NOT a kernel gate, `_NATIVE_FUSED` is.
from .format import _GGUF_KQUANT_BLOCK as _KQUANT_TYPE_SIZE  # noqa: E402
# RAW: float / non-quantized types stored as-is (fp16).
_RAW_FLOAT = {"F32", "F16", "BF16", "F64"}
# REQUANT-I8: dequant->per_row_i8. Everything else quantizable (legacy block
# formats + the IQ codebook family — no native fused kernel).
_REQUANT_I8 = {
    "Q8_0",
    "Q8_1",
    "Q8_K",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "TQ1_0",
    "TQ2_0",
    "IQ1_M",
    "IQ4_NL",
}


def gguf_type_coverage() -> dict:
    """The ggml_type -> dispatch-path coverage matrix (name -> path str).

    Paths: ``"native_fused"`` (resident native bytes, fused dp4a),
    ``"requant_i8"`` (dequant->per_row_i8 dp4a), ``"raw"`` (fp16 as-is)."""
    cov = {}
    for n in _NATIVE_FUSED:
        cov[n] = "native_fused"
    for n in _RAW_FLOAT:
        cov[n] = "raw"
    for n in _REQUANT_I8:
        cov[n] = "requant_i8"
    return cov


def _type_name(gguf, ttype: int) -> str:
    try:
        return gguf.GGMLQuantizationType(ttype).name
    except Exception:
        return f"ggml_type_{ttype}"


def _native_tag(gguf, ttype: int) -> str | None:
    """Native-fused tag for a ggml type: by name when the package knows it, else by
    numeric type (the released ``gguf`` package cannot name the fork's TQ3_4S=46)."""
    return _NATIVE_FUSED.get(_type_name(gguf, ttype)) or _NATIVE_FUSED_BY_TYPE.get(ttype)


def kquant_qtensor(raw_bytes: np.ndarray, out_features: int, in_features: int, tag: str):
    """Wrap raw native GGUF k-quant bytes (Q2_K..Q6_K / TQ3_4S) as a ``gguf_kquant``
    QTensor -> ``[out, n_superblocks*type_size]`` uint8, byte-identical to the file.

    The k-quants are QK_K=256 super-blocks (in%256==0); ``tq3_4s`` is QK_TQ3_0=32
    (in%32==0, 16 B/block) — the group_size rides into the QTensor for tile staging."""
    from .format import QTensor

    assert tag in _KQUANT_TYPE_SIZE, f"unsupported native-fused tag {tag!r}"
    gs = 32 if tag == "tq3_4s" else 256
    assert in_features % gs == 0, f"{tag} needs in%{gs}==0, got {in_features}"
    ts = _KQUANT_TYPE_SIZE[tag]
    row_bytes = (in_features // gs) * ts
    u8 = np.ascontiguousarray(raw_bytes).view(np.uint8).reshape(-1)
    assert u8.size == out_features * row_bytes, (
        f"gguf {tag}: {u8.size} bytes != {out_features}*{row_bytes}"
    )
    data = torch.from_numpy(u8.reshape(out_features, row_bytes).copy())
    return QTensor(data, None, scheme="gguf_kquant", group_size=gs, codebook=tag)


def _requant_i8_qtensor(deq_fp32: np.ndarray):
    """dequant fp32 weight [out,in] -> per-row symmetric int8 ``per_row_i8`` QTensor.
    in must be %4==0 for dp4a (always true for gguf 2-D weights). Falls back to raw
    fp16 if the row dim isn't dp4a-aligned."""
    from .format import QTensor
    from .quant.core import quantize_int8_rowwise

    w = torch.from_numpy(np.ascontiguousarray(deq_fp32))
    if w.shape[-1] % 4 != 0:
        return QTensor(w.half(), None, scheme="raw")
    q, s = quantize_int8_rowwise(w)  # int8 [out,in], fp32 [out,1]
    return QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")


def tensor_to_qtensor(gguf, t):
    """Dispatch one GGUFReader tensor to a QTensor + path label.

    Returns ``(qtensor, path, type_name)`` where path is ``native_fused`` /
    ``requant_i8`` / ``raw``. 2-D weights quantize; anything else stays raw fp16.
    """
    from .format import QTensor

    tname = _type_name(gguf, int(t.tensor_type))
    is_2d = len(t.shape) == 2
    # gguf shape is reversed vs [out,in]: shape == [in, out].
    in_f, out_f = (int(t.shape[0]), int(t.shape[1])) if is_2d else (0, 0)

    tag = _native_tag(gguf, int(t.tensor_type))
    if is_2d and tag and in_f % (32 if tag == "tq3_4s" else 256) == 0:
        try:
            return kquant_qtensor(t.data, out_f, in_f, tag), "native_fused", tname
        except AssertionError:
            pass  # fall through to requant
    if is_2d and tname not in _RAW_FLOAT:
        # REQUANT-I8: dequant (gguf pkg / city96 math) then per-row int8.
        try:
            deq = gguf.quants.dequantize(t.data, gguf.GGMLQuantizationType(int(t.tensor_type)))
            deq = np.ascontiguousarray(deq).astype(np.float32).reshape(out_f, in_f)
            return _requant_i8_qtensor(deq), "requant_i8", tname
        except Exception:
            pass  # unknown/undequantizable -> raw
    # RAW: float weights, biases, norms, embeddings, or anything we couldn't quantize.
    try:
        arr = gguf.quants.dequantize(t.data, gguf.GGMLQuantizationType(int(t.tensor_type)))
    except Exception:
        arr = np.ascontiguousarray(t.data)
    w = torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32))
    shape = tuple(int(s) for s in reversed(t.shape))
    if w.numel() == int(np.prod(shape)) and shape:
        w = w.reshape(shape)
    return QTensor(w.half(), None, scheme="raw"), "raw", tname


def load_gguf_tensors(path: str):
    """Yield ``(name, qtensor, path, type_name)`` for every tensor in a GGUF file,
    each dispatched to its best dp4a path (native_fused / requant_i8 / raw). Makes
    ANY gguf — incl. Unsloth UD mixed-quant — loadable. Requires ``gguf``."""
    import gguf

    reader = gguf.GGUFReader(path)
    for t in reader.tensors:
        qt, path_label, tname = tensor_to_qtensor(gguf, t)
        yield t.name, qt, path_label, tname


def gguf_type_histogram(path: str) -> dict:
    """Histogram of ggml_type NAME -> count over a GGUF file's tensors, plus each
    type's dispatch path. Returns ``{type_name: {"count": n, "path": path}}``.
    Shows exactly which types a (UD) file uses and how each is served."""
    import gguf

    reader = gguf.GGUFReader(path)
    cov = gguf_type_coverage()
    hist: dict = {}
    for t in reader.tensors:
        tname = _type_name(gguf, int(t.tensor_type))
        tag = _native_tag(gguf, int(t.tensor_type))
        is_2d = len(t.shape) == 2
        if tname in _RAW_FLOAT or not is_2d:
            path_label = "raw"
        elif tag is not None:
            path_label = "native_fused"
        else:
            path_label = cov.get(tname, "requant_i8")
        e = hist.setdefault(tname, {"count": 0, "path": path_label})
        e["count"] += 1
    return hist


# ---- back-compat: the original native-only loader (Q4_K/Q5_K/Q6_K) -----------
def load_gguf_kquant_tensors(path: str, *, only_fused: bool = True):
    """Yield ``(name, qtensor_or_none, type_name)`` for native-fused k-quant
    tensors only (Q4_K/Q5_K/Q6_K -> gguf_kquant); non-fused tensors yield
    ``(name, None, type_name)``. Prefer :func:`load_gguf_tensors` for full
    coverage."""
    import gguf

    reader = gguf.GGUFReader(path)
    for t in reader.tensors:
        tname = _type_name(gguf, int(t.tensor_type))
        tag = _native_tag(gguf, int(t.tensor_type))
        if len(t.shape) == 2 and tag and int(t.shape[0]) % (32 if tag == "tq3_4s" else 256) == 0:
            try:
                qt = kquant_qtensor(t.data, int(t.shape[1]), int(t.shape[0]), tag)
                yield t.name, qt, tname
                continue
            except AssertionError:
                pass
        yield t.name, None, tname
