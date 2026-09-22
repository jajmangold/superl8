# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Test that Lloyd-Max quantization handles non-contiguous inputs without warnings and produces bit-identical outputs.

This test addresses the GitHub issue where torch.bucketize was being called with
a non-contiguous tensor, causing a performance warning on every K8V3 engine run.

The fix ensures the tensor passed to bucketize is contiguous by calling .contiguous()
on the reshape result before bucketize, avoiding the warning while preserving bit-identical outputs.
"""

import warnings

import pytest
import torch

from superl8.quant import lloydmax


def _reference_quantize_lloydmax(
    x: torch.Tensor,
    *,
    bits: int = 3,
    block_size: int = 128,
    dim: int = -1,
    codebook: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation of quantize_lloydmax body without any .contiguous() calls.
    
    This is the master version (before fix) - used to verify bit-identical outputs.
    """
    lloydmax_bits_levels = lambda b: 1 << b
    lloydmax_bits_levels(bits)  # validate bit width
    
    xf = x.float().movedim(dim, -1)
    d = xf.shape[-1]
    assert d % block_size == 0, f"last dim {d} not divisible by block_size {block_size}"
    nb = d // block_size
    
    # NO .contiguous() - master version
    xg = xf.reshape(*xf.shape[:-1], nb, block_size)
    
    norm = xg.pow(2).sum(-1, keepdim=True).sqrt()
    safe = torch.where(norm == 0, torch.ones_like(norm), norm)
    xn = xg / safe
    
    cb = lloydmax.gaussian_codebook(bits, block_size, x.device.type)
    bounds = 0.5 * (cb[1:] + cb[:-1])
    
    # NO .contiguous() - master version
    idx = torch.bucketize(xn.reshape(*xn.shape[:-2], -1), bounds)
    
    codes = idx.reshape(xf.shape).movedim(-1, dim).to(torch.uint8)
    norm_out = safe.squeeze(-1).movedim(-1, dim).float()
    return codes, norm_out, cb


def _is_contiguous_after_call(f, *args, **kwargs) -> tuple:
    """Check if a call to f would produce a non-contiguous intermediate that triggers warnings.
    
    This is used for the test to verify the fix is working.
    """
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f(*args, **kwargs)
            return any("non-contiguous" in str(warning.message).lower() for warning in w)
    except Exception:
        return False


@pytest.mark.cpu
@pytest.mark.parametrize("shape", [(4, 128, 8)])
def test_lloydmax_no_searchsorted_warning_transpose_noncontiguous(shape):
    """Test that transposed non-contiguous inputs don't trigger torch.searchsorted warnings.
    
    Creates a genuinely non-contiguous tensor by transposing and reshaping.
    """
    x = torch.randn(4, 128, 8).transpose(1, 2)
    assert not x.is_contiguous(), "Test requires non-contiguous input"
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        codes, norm, cb = lloydmax.quantize_lloydmax(
            x, bits=3, block_size=128, dim=-1
        )
        
        bad_warnings = [
            warning for warning in w
            if "searchsorted" in str(warning.message).lower() or
               "non-contiguous" in str(warning.message).lower()
        ]
        
        assert len(bad_warnings) == 0, (
            f"Unexpected searchsorted/non-contiguous warnings found: "
            f"{[str(w.message) for w in bad_warnings]}"
        )
        
        assert codes.shape == x.shape
        assert norm.shape[0] == x.shape[0]
    
    # Verify bit-identical to reference
    ref_codes, ref_norm, ref_cb = _reference_quantize_lloydmax(x, bits=3, block_size=128, dim=-1)
    assert torch.equal(codes, ref_codes), "Codes should be bit-identical to reference"
    assert torch.equal(norm, ref_norm), "Norms should be bit-identical to reference"
    assert torch.equal(cb, ref_cb), "Codebooks should be identical"


@pytest.mark.cpu
@pytest.mark.parametrize("shape", [(2, 16, 4, 256)])
def test_lloydmax_no_searchsorted_warning_permute_noncontiguous(shape):
    """Test that permuted non-contiguous inputs don't trigger torch.searchsorted warnings.
    
    Creates a genuinely non-contiguous tensor by permuting.
    """
    x = torch.randn(2, 16, 4, 256).permute(0, 2, 1, 3)
    assert not x.is_contiguous(), "Test requires non-contiguous input"
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        codes, norm, cb = lloydmax.quantize_lloydmax(
            x, bits=3, block_size=128, dim=-1
        )
        
        bad_warnings = [
            warning for warning in w
            if "searchsorted" in str(warning.message).lower() or
               "non-contiguous" in str(warning.message).lower()
        ]
        
        assert len(bad_warnings) == 0, (
            f"Unexpected searchsorted/non-contiguous warnings found: "
            f"{[str(w.message) for w in bad_warnings]}"
        )
        
        assert codes.shape == x.shape
    
    # Verify bit-identical to reference
    ref_codes, ref_norm, ref_cb = _reference_quantize_lloydmax(x, bits=3, block_size=128, dim=-1)
    assert torch.equal(codes, ref_codes), "Codes should be bit-identical to reference"
    assert torch.equal(norm, ref_norm), "Norms should be bit-identical to reference"
    assert torch.equal(cb, ref_cb), "Codebooks should be identical"


@pytest.mark.cpu
@pytest.mark.parametrize("shape", [(4, 128, 8), (4, 16, 4, 128)])
def test_lloydmax_fake_quant_no_searchsorted_warning(shape):
    """Test that fake_quant_lloydmax doesn't trigger warnings with non-contiguous inputs."""
    if shape == (4, 8, 128):
        x = torch.randn(4, 128, 8).transpose(1, 2)
    else:
        x = torch.randn(2, 16, 4, 128).permute(0, 2, 1, 3)
    
    assert not x.is_contiguous(), f"Test requires non-contiguous input for shape {shape}"
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = lloydmax.fake_quant_lloydmax(x, bits=3, block_size=128, dim=-1, fit=False)
        
        bad_warnings = [
            warning for warning in w
            if "searchsorted" in str(warning.message).lower() or
               "non-contiguous" in str(warning.message).lower()
        ]
        
        assert len(bad_warnings) == 0, (
            f"Unexpected searchsorted/non-contiguous warnings: {bad_warnings}"
        )
