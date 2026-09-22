# ============================================================================
# Copyright (c) 2026, fni8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
# fni8 dev/build/test/bench/profile image. sm_70 (Volta) only.
# CUDA 12.9 is the LAST Volta-capable toolkit (CUDA 13 drops sm_70). Do not bump.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    TORCH_CUDA_ARCH_LIST=7.0 \
    PYTHONUNBUFFERED=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID

# Ubuntu 24.04 ships Python 3.12, matching the cp312 Volta torch wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ninja-build ca-certificates ccache && \
    rm -rf /var/lib/apt/lists/*

# Pinned Volta torch wheel + build/test deps. Bumping torch/CUDA breaks sm_70.
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu129 \
        torch==2.10.0+cu129 && \
    python3 -m pip install \
        numpy psutil "setuptools>=70" wheel packaging \
        pytest pytest-benchmark ruff optuna \
        gguf

# External fp16 baseline: ai-bond's FA2 for sm_70 (BSD-3), installed from a
# locally prebuilt wheel (compiling it inside `docker build` fails: no GPU +
# BuildKit RAM limits). Rebuild the wheel with tools/build_fa_v100_wheel.sh.
# Tests soft-skip if the wheel is absent, so a fresh checkout still builds.
COPY docker/wheels/ /tmp/wheels/
RUN if ls /tmp/wheels/flash_attn_v100*.whl >/dev/null 2>&1; then \
        python3 -m pip install --no-deps /tmp/wheels/flash_attn_v100*.whl; \
    else \
        echo "NOTE: no flash_attn_v100 wheel — external baseline tests will skip"; \
    fi && rm -rf /tmp/wheels

# Nsight is best-effort here (only the `profile` service uses it); PR8 hardens it.
RUN apt-get update && \
    (apt-get install -y --no-install-recommends cuda-nsight-compute-12-9 cuda-nsight-systems-12-9 \
      || echo "WARN: nsight not installed via apt; verify ncu/nsys before profiling") && \
    rm -rf /var/lib/apt/lists/*

# ccache config lives LAST — it's the only frequently-tweaked layer. Everything above
# (cuda, torch, the fa wheel, the ~3 GB nsight download) is the stable "baked base" and
# stays in Docker's layer cache; editing ccache rebuilds only this thin top layer in
# seconds instead of re-downloading nsight. (Ordering IS the base-image split.)
#
# A kernel PR usually touches ONE .cu, but torch recompiles all ~18 objects from cold
# every CI run (~30 min). ccache keys on file CONTENT (not path/mtime), so it hits even
# across CI's fresh git checkouts → ~17 cache hits + 1 recompile (~3 min). Wrap nvcc so
# torch's absolute-path invocation is cached; the /usr/lib/ccache masquerade caches the
# host-compiler side. The cache lives in a persistent volume (compose) shared by every
# build on the host, so it stays warm across all PRs and runners.
RUN mv /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc.real && \
    printf '#!/bin/sh\nexec ccache /usr/local/cuda/bin/nvcc.real "$@"\n' > /usr/local/cuda/bin/nvcc && \
    chmod +x /usr/local/cuda/bin/nvcc
# Use ccache's masquerade symlinks (/usr/lib/ccache/{gcc,g++,cc,c++} → ccache) via PATH,
# NOT CC="ccache gcc" — the space in CC breaks torch's cpp_extension ninja generation
# (TypeError: not enough arguments for format string). nvcc is handled by the wrapper
# above. Caches both C++ and CUDA objects without confusing torch.
ENV PATH="/usr/lib/ccache:${PATH}" \
    CCACHE_DIR=/ccache \
    CCACHE_MAXSIZE=20G \
    CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime,pch_defines

WORKDIR /workspace
