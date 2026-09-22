// ============================================================================
// Copyright (c) 2026, fni8 authors
// SPDX-License-Identifier: BSD-3-Clause
// ============================================================================
// Trivial elementwise-add device code. Its only purpose is to exercise the
// full toolchain loop end-to-end (nvcc sm_70 -> .so -> pybind -> torch op ->
// correctness + perf gate) before any real kernel lands. One concern per file.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace fni8 {

// Elementwise C = A + B over `n` half elements, grid-stride loop.
__global__ void hello_add_kernel(const __half* __restrict__ a,
                                 const __half* __restrict__ b,
                                 __half* __restrict__ c,
                                 int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    c[i] = __hadd(a[i], b[i]);
  }
}

}  // namespace fni8
