# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Full GGUF type-coverage: load ANY gguf (incl. Unsloth UD mixed-quant) by
dispatching each tensor to its best sm_70 dp4a path. See superl8/gguf.py.

  native_fused  Q2_K/Q3_K/Q4_K/Q5_K/Q6_K  -> gguf_kquant, unpacked in-kernel (fused dp4a)
  requant_i8    everything else quantizable -> dequant->per_row_i8 (dp4a)
  raw           float/1-D tensors -> fp16

The UD test enumerates a REAL Unsloth UD file's tensors, histograms the ggml
types it uses, asserts every type dispatches to the right path, and gates the
native-fused types on the real tensors at SQNR>=40 dB (int8 gate, not allclose).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import superl8
from superl8 import gguf as fgguf
from superl8.quant.core import quantize_int8_rowwise

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.tolerances import assert_int8_quality  # noqa: E402

# Real Unsloth UD (Dynamic mixed-quant) files on disk.
_UD_Q5 = os.environ.get("FNI8_UD_Q5_GGUF", "")
_UD_Q6 = os.environ.get("FNI8_UD_Q6_GGUF", "")
_UD = next((p for p in (_UD_Q5, _UD_Q6) if os.path.exists(p)), None)


@pytest.mark.correctness
def test_coverage_matrix_paths():
    """Every declared type maps to exactly one dp4a path; all five k-quants
    (Q2_K–Q6_K) are native-fused, Q8_0/legacy are requant-i8, floats are raw.

    The i-quants are SPLIT and the split moves as kernels land (superl8#317): IQ3_S,
    IQ4_XS and IQ3_XXS are fused (decode + tile); IQ2_S/IQ2_XS/IQ2_XXS/IQ1_S still
    requantize to int8, which is what keeps Qwen3.8-27B-UD-IQ3_S from fitting on a
    16 GiB card. Move a type here when its kernel lands, not before."""
    cov = fgguf.gguf_type_coverage()
    assert cov["Q2_K"] == cov["Q3_K"] == cov["Q4_K"] == cov["Q5_K"] == cov["Q6_K"] == "native_fused"
    for t in ("IQ3_S", "IQ4_XS", "IQ3_XXS", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_S"):
        assert cov[t] == "native_fused", f"{t} has a fused kernel and must not requantize"
    for t in ("Q8_0", "Q4_0", "Q5_1", "IQ1_M", "IQ4_NL"):
        assert cov[t] == "requant_i8", f"{t} should be requant_i8"
    for t in ("F32", "F16", "BF16"):
        assert cov[t] == "raw"


@pytest.mark.correctness
def test_requant_i8_fallback_loads_and_dp4as(device):
    """A dequant->per_row_i8 QTensor must load and run gemm_w8a8 (the universal
    fallback path for Q8_0/legacy/IQ). Uses a fp weight as the 'dequant'."""
    torch.manual_seed(0)
    w = (torch.randn(128, 512) * 0.1).float().numpy()
    qt = fgguf._requant_i8_qtensor(w)
    assert qt.scheme == "per_row_i8" and qt.data.dtype == torch.int8
    x = torch.randn(4, 512, device=device, dtype=torch.float16)
    y = superl8.linear(
        x, superl8.format.QTensor(qt.data.to(device), qt.scale.to(device), scheme="per_row_i8")
    )
    ref = x.float() @ torch.from_numpy(w).to(device).t()
    # int8 requant of a full-precision weight: standard W8A8 quality.
    assert_int8_quality(
        y, ref, min_cos=0.999, max_rel_l1=0.03, min_sqnr_db=30.0, what="requant_i8 fallback"
    )


# ---------------------------------------------------------------------------
# Real Unsloth UD file: type histogram + per-tensor dispatch + fused fidelity.
# ---------------------------------------------------------------------------
@pytest.mark.correctness
@pytest.mark.skipif(_UD is None, reason="no Unsloth UD gguf on disk")
def test_ud_type_histogram_and_dispatch():
    gguf = pytest.importorskip("gguf")
    hist = fgguf.gguf_type_histogram(_UD)
    print(f"\n=== UD file: {os.path.basename(_UD)} — ggml_type histogram ===")
    for tname, e in sorted(hist.items(), key=lambda kv: -kv[1]["count"]):
        print(f"  {tname:10s} x{e['count']:4d} -> {e['path']}")
    # It's a genuine MIXED file: more than one quant type present.
    quant_types = [t for t in hist if t not in ("F32", "F16", "BF16")]
    assert len(quant_types) >= 2, f"UD file should mix types, got {quant_types}"
    # Every type resolves to a known path (nothing unsupported).
    for tname, e in hist.items():
        assert e["path"] in ("native_fused", "requant_i8", "raw"), (tname, e)


@pytest.mark.correctness
@pytest.mark.skipif(_UD is None, reason="no Unsloth UD gguf on disk")
def test_ud_native_fused_fidelity(device):
    """For each native-fused type present in the UD file, gate ONE real tensor:
    fused kernel vs the gguf package's own dequant->matmul at SQNR>=40 dB."""
    gguf = pytest.importorskip("gguf")
    reader = gguf.GGUFReader(_UD)
    seen = set()
    checked = 0
    for t in reader.tensors:
        tname = fgguf._type_name(gguf, int(t.tensor_type))
        if tname not in fgguf._NATIVE_FUSED or len(t.shape) != 2 or tname in seen:
            continue
        in_f, out_f = int(t.shape[0]), int(t.shape[1])
        if in_f % 256 != 0 or not (256 <= out_f <= 8192 and 256 <= in_f <= 8192):
            continue
        seen.add(tname)
        checked += 1
        qt = fgguf.kquant_qtensor(t.data, out_f, in_f, fgguf._NATIVE_FUSED[tname])
        deq = (
            gguf.quants.dequantize(t.data, gguf.GGMLQuantizationType(int(t.tensor_type)))
            .astype(np.float32)
            .reshape(out_f, in_f)
        )
        x = torch.randn(16, in_f, device=device, dtype=torch.float16)
        x_i8, x_scale = quantize_int8_rowwise(x)
        ref = (x_i8.float() @ torch.from_numpy(deq).to(device).t()) * x_scale
        op = {"q4_k": superl8._C.gemm_q4k, "q5_k": superl8._C.gemm_q5k, "q6_k": superl8._C.gemm_q6k}[
            fgguf._NATIVE_FUSED[tname]
        ]
        y = op(x_i8, x_scale.squeeze(-1).contiguous(), qt.data.to(device), torch.float16)
        assert_int8_quality(
            y,
            ref,
            min_cos=0.999,
            max_rel_l1=0.01,
            min_sqnr_db=40.0,
            what=f"UD {tname} real-tensor {out_f}x{in_f}",
        )
    if checked == 0:
        pytest.skip("no native-fused 2-D tensor found in the UD file")
