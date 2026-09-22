# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""SDNQ-informed INT8 quantization layer."""

from .core import (
    LOG2E,
    dequantize_int8_rowwise,
    detect_q_outlier_domination,
    quantize_int8_rowwise,
    quantize_qk,
    quantize_v_perchannel,
    quantize_v_rowwise,
    smooth_k,
)
from .lloydmax import (
    dequantize_lloydmax,
    fake_quant_lloydmax,
    gaussian_codebook,
    lloyd_max_fit,
    quantize_lloydmax,
)
from .lowbit import (
    dequantize_w3a8,
    pack_w3a8_bitplanes,
    quantize_w3a8,
    unpack_w3a8_bitplanes,
)
from .rotation import hadamard_matrix, rotate_last

__all__ = [
    "LOG2E",
    "dequantize_w3a8",
    "pack_w3a8_bitplanes",
    "quantize_w3a8",
    "unpack_w3a8_bitplanes",
    "dequantize_int8_rowwise",
    "dequantize_lloydmax",
    "detect_q_outlier_domination",
    "fake_quant_lloydmax",
    "gaussian_codebook",
    "hadamard_matrix",
    "lloyd_max_fit",
    "quantize_int8_rowwise",
    "quantize_lloydmax",
    "quantize_qk",
    "quantize_v_perchannel",
    "quantize_v_rowwise",
    "rotate_last",
    "smooth_k",
]
