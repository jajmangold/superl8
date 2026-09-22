"""CUDA-graph capture guard on the int8 attention accuracy gate (superl8#310).

``detect_q_outlier_domination`` ends in a data-dependent ``.any()`` -- a device->host
sync, which raises inside ``torch.cuda.graph()`` capture. While capturing, the gate must
be skipped (return False: take the int8 path). Outside capture it must still run.

Capture is simulated by patching ``torch.cuda.is_current_stream_capturing``, so these run
on CPU; the input is outlier-dominated, so without the guard the gate returns True.
"""

from unittest import mock

import pytest
import torch

from superl8.quant import detect_q_outlier_domination

pytestmark = pytest.mark.cpu


def _outlier_dominated_q() -> torch.Tensor:
    # [B, H, L, D]: one channel 100x the rest, so max/median per row is 100 > 12.
    q = torch.ones(1, 2, 4, 16, dtype=torch.float16)
    q[..., 3] = 100.0
    return q


def _capturing(value: bool):
    return mock.patch("torch.cuda.is_current_stream_capturing", return_value=value)


def test_gate_fires_outside_capture():
    with _capturing(False):
        assert detect_q_outlier_domination(_outlier_dominated_q()) is True


def test_gate_is_skipped_during_capture():
    with _capturing(True):
        assert detect_q_outlier_domination(_outlier_dominated_q()) is False


def test_capture_path_does_no_tensor_work():
    """During capture the gate must not touch the tensor at all: any op on it would
    either sync (``.any()``) or bake a stale decision into the graph."""

    class Untouchable(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            raise AssertionError(f"gate touched q during capture: {func}")

    q = _outlier_dominated_q().as_subclass(Untouchable)
    with _capturing(True):
        assert detect_q_outlier_domination(q) is False


def test_well_conditioned_q_passes_outside_capture():
    q = torch.ones(1, 2, 4, 16, dtype=torch.float16)
    with _capturing(False):
        assert detect_q_outlier_domination(q) is False
