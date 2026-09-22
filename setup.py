# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Build the superl8 CUDA extension (sm_70 / Volta only).

Follows the ai-bond/flash-attention-v100 build conventions: a single
torch CUDAExtension, header-only device code under csrc/include, thin .cu
translation units, and the exact nvcc flag set proven on Volta.
"""

import os
from pathlib import Path

from setuptools import setup

# setuptools requires /-separated paths RELATIVE to setup.py — never absolute.
CSRC = Path("csrc")

# Import torch lazily with a clear message — the pinned Volta wheel is required.
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "torch is required to build superl8. Install the pinned Volta wheel first:\n"
        "  pip install torch==2.10.0+cu129 --index-url https://download.pytorch.org/whl/cu129"
    ) from e

from packaging.version import parse

if torch.version.cuda is None or parse(torch.version.cuda) < parse("12.9"):
    raise RuntimeError(
        f"CUDA {torch.version.cuda} < 12.9 is not supported. 12.9 is the last "
        "Volta-capable toolkit; CUDA 13 drops sm_70."
    )

NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-gencode",
    "arch=compute_70,code=sm_70",  # Volta only. Do NOT add other archs.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-Wno-deprecated-gpu-targets",
]
if os.environ.get("SUPERL8_DEBUG"):
    NVCC_FLAGS += ["-g", "-lineinfo", "-Xptxas", "-v"]

# PR7 autotuning: the backward tile/thread config is overridable at compile time
# via env vars so bench/autotune.py can sweep configs without editing source.
# Defaults (unset) preserve the committed kernel exactly.
for _var in ("FNI8_BWD_BM", "FNI8_BWD_BN", "FNI8_BWD_TPR"):
    _val = os.environ.get(_var)
    if _val:
        NVCC_FLAGS.append(f"-D{_var}={int(_val)}")

ext = CUDAExtension(
    name="superl8._C",
    sources=[
        str(CSRC / "fni8_api.cpp"),
        str(CSRC / "kernel" / "hello.cu"),
        str(CSRC / "kernel" / "gather_q3k.cu"),
        str(CSRC / "kernel" / "quant_rowwise.cu"),
        str(CSRC / "kernel" / "rmsnorm.cu"),
        str(CSRC / "kernel" / "rope.cu"),
        str(CSRC / "kernel" / "act_and_mul.cu"),
        str(CSRC / "kernel" / "dit_block.cu"),
        str(CSRC / "kernel" / "attn_int8_fwd.cu"),
        str(CSRC / "kernel" / "attn_fp16_fwd.cu"),
        str(CSRC / "kernel" / "attn_w8a8_fwd.cu"),
        str(CSRC / "kernel" / "attn_bwd.cu"),
        str(CSRC / "kernel" / "attn_decode.cu"),
        str(CSRC / "kernel" / "attn_paged_decode.cu"),
        str(CSRC / "kernel" / "attn_varlen_fwd.cu"),
        str(CSRC / "kernel" / "attn_tree_fwd.cu"),
        str(CSRC / "kernel" / "gemm_dp4a.cu"),
        str(CSRC / "kernel" / "gemm_decode_dp4a.cu"),
        str(CSRC / "kernel" / "gemm_grouped_dp4a.cu"),
        str(CSRC / "kernel" / "deltanet_chunk.cu"),
        str(CSRC / "kernel" / "deltanet_decode.cu"),
        str(CSRC / "kernel" / "causal_conv1d_decode.cu"),
        str(CSRC / "kernel" / "gated_rmsnorm_decode.cu"),
        str(CSRC / "kernel" / "lightning_attn.cu"),
        str(CSRC / "kernel" / "mla_attn.cu"),
        str(CSRC / "kernel" / "mla_attn_fp16.cu"),
        str(CSRC / "kernel" / "mla_attn_int8.cu"),
    ],
    include_dirs=[str((Path(__file__).parent / "csrc" / "include").resolve())],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": NVCC_FLAGS,
    },
)

setup(
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
