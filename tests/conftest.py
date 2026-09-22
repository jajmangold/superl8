# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Shared test fixtures: CUDA/sm_70 gating and a fixed RNG seed."""
import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Skip everything if CUDA is unavailable — except CPU-only tests."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cpu" in item.keywords:
            continue
        item.add_marker(skip)


@pytest.fixture(scope="session")
def device() -> torch.device:
    assert torch.cuda.is_available(), "superl8 tests require a CUDA device"
    return torch.device("cuda")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield
