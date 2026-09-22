// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Shared dp4a GEMM tile geometry — the ONE source of truth for the
// register/smem footprint used by both the dense kernel (gemm_dp4a.cuh) and the
// MoE grouped kernel (gemm_grouped_dp4a.cuh). Split out so the two entry kernels
// live in separate translation units without a `multiple definition` clash
// (their non-inline __global__ symbols must not be shared via a common header),
// while still guaranteeing an identical BM/BN/BK/thread layout — a divergence
// here would corrupt the shared-memory staging of one of them.
#pragma once

namespace fni8 {

constexpr int GEMM_BM = 64;   // output rows per block
constexpr int GEMM_BN = 64;   // output cols per block
constexpr int GEMM_BK = 64;   // contraction chunk (== 16 int32 words)
constexpr int GEMM_THREADS = 128;
constexpr int GEMM_TM = 4;    // rows per thread
constexpr int GEMM_TN = 8;    // cols per thread
constexpr int GEMM_BK4 = GEMM_BK / 4;   // int32 words per k-block

}  // namespace fni8
