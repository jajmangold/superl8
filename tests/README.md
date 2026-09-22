# superl8 test suite

## Test categories

Tests are organized by pytest markers:

| Marker | Purpose |
|---|---|
| `correctness` | Numerical accuracy tests — kernel output matches fp32 oracle within tolerance. |
| `regression` | Guards against known past failures (e.g. fp16 overflow, device mismatch). |
| `perf` | Performance regression tests — fail if latency regresses >5% vs committed baseline. |
| `cpu` | Pure-torch CPU-only tests — no CUDA required. Skipped by default when CUDA is absent. |

Uncategorized tests (no marker) require CUDA and test core functionality.

## Running tests

```bash
# Full suite (requires CUDA GPU)
pytest tests/ -ra

# Correctness only
pytest tests/ -m correctness -ra

# Performance regressions
pytest tests/ -m perf -ra

# CPU-only tests (no GPU needed)
pytest tests/ -m cpu -ra

# Everything except performance
pytest tests/ -m "not perf" -ra
```

## Tolerance system

All tolerances live in `tests/tolerances.py`.

**fp16 kernels** — relative-to-fp32 bounds (ai-bond style). The kernel's max absolute error vs the fp32 oracle must not exceed a multiple of PyTorch's own fp16 baseline error on the same inputs:

- Forward: `err(kernel) <= 2 * err(pt_fp16) + 1e-5`
- Backward: `err(kernel) <= 3 * err(pt_fp16) + 1e-4`

**int8 kernels** — never `allclose`. A single rounding boundary legitimately flips. Instead use:

- SQNR (signal-to-quantization-noise ratio, dB) — minimum 20 dB
- Cosine similarity — minimum 0.999
- Relative L1 — maximum 0.02

These are the SageAttention-level accuracy bars. Never weaken to pass.

## Hardware requirements

- **CPU tests** (`@pytest.mark.cpu`): No GPU needed. Pure-torch implementations.
- **All other tests**: Require a CUDA GPU with sm_70 support (Volta or later). The `conftest.py` session fixture asserts CUDA availability.

## How to add a new test

1. **Choose the right marker** — `correctness` for numerical validation, `perf` for latency checks, `cpu` for pure-torch tests.
2. **Use the `device` fixture** — injected by `conftest.py`, provides `torch.device("cuda")`.
3. **For fp16 kernels** — use `assert_relative_to_fp32()` from `tolerances.py`. Provide the kernel output, a PyTorch fp16 baseline, and the fp32 oracle.
4. **For int8 kernels** — use `assert_int8_quality()` from `tolerances.py`. Provide kernel output and fp32 reference.
5. **For perf tests** — use `time_ms()` and `assert_no_regression()` from `bench/harness.py`. The harness auto-baselines on first run and fails on >5% regression after that.
6. **Seed is automatic** — `conftest.py` calls `torch.manual_seed(0)` before every test. No manual seeding needed.
