# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Hadamard incoherence rotation (QuaRot / TurboQuant+ style) — tests first.

Two properties to prove:
  1. LOGIT-INVARIANCE: rotating Q/K by a shared orthogonal Hadamard leaves the
     attention output unchanged (up to int8 noise) — it is a free correctness
     no-op on already-incoherent (randn) data.
  2. THE PAYOFF: on K with per-channel OUTLIERS (what real-model activations
     look like, unlike randn), plain per-row int8 crushes the non-outlier
     channels, but the rotation spreads the outlier energy and recovers accuracy.
     `rotate=True` must be strictly better and clear a bar that plain int8 fails.
"""
import pytest
import torch

import superl8
from superl8.quant import hadamard_matrix, rotate_last
from tests.reference import attention_fp32_oracle
from tests.tolerances import assert_int8_quality, cos_sim, rel_l1

DIMS = [32, 64, 128]


@pytest.mark.correctness
@pytest.mark.parametrize("d", DIMS)
def test_hadamard_orthogonal(device, d):
    m = hadamard_matrix(d, device="cuda", dtype=torch.float32)
    eye = m @ m.t()
    assert torch.allclose(eye, torch.eye(d, device=eye.device), atol=1e-5), \
        f"Hadamard not orthogonal at d={d}"


@pytest.mark.correctness
@pytest.mark.parametrize("d", DIMS)
def test_rotation_preserves_logits(device, d):
    """QK^T is invariant: rotate(Q)·rotate(K)^T == Q·K^T."""
    q = torch.randn(2, 4, 128, d, device=device, dtype=torch.float16)
    k = torch.randn(2, 4, 128, d, device=device, dtype=torch.float16)
    s = torch.einsum("bhmd,bhnd->bhmn", q.float(), k.float())
    s_rot = torch.einsum("bhmd,bhnd->bhmn", rotate_last(q).float(), rotate_last(k).float())
    # Invariance is exact in math; the only gap is fp16 STORAGE of the rotated
    # Q/K (which we then int8-quantize anyway, far coarser). So the honest bound
    # is magnitude-relative, not a fixed atol on O(10-50) logits.
    rel = (s - s_rot).abs().max() / s.abs().max()
    assert rel < 5e-3, f"logits shifted at d={d}: rel={rel:.2e}"


@pytest.mark.correctness
@pytest.mark.parametrize("d", [64, 128])
def test_rotation_noop_on_clean_data(device, d):
    """On randn (no outliers), rotate on/off both hold the int8 bars."""
    q, k, v = (torch.randn(1, 8, 512, d, device=device, dtype=torch.float16) for _ in range(3))
    oracle = attention_fp32_oracle(q, k, v)
    for rot in (False, True):
        out = superl8.attn_int8_fwd(q, k, v, rotate=rot)
        assert_int8_quality(out, oracle, what=f"clean rotate={rot} d={d}")


@pytest.mark.correctness
@pytest.mark.parametrize("d", [64, 128])
def test_rotation_rescues_channel_outliers(device, d):
    """The payoff: K with per-channel outliers. Plain int8 degrades; rotation
    recovers it and clears a bar the plain path fails."""
    torch.manual_seed(0)
    q = torch.randn(1, 8, 512, d, device=device, dtype=torch.float16)
    k = torch.randn(1, 8, 512, d, device=device, dtype=torch.float16)
    v = torch.randn(1, 8, 512, d, device=device, dtype=torch.float16)
    # inject a few massive channel outliers into K (real activations look like this)
    k[..., [0, 1, 7]] *= 40.0
    oracle = attention_fp32_oracle(q, k, v)

    out_plain = superl8.attn_int8_fwd(q, k, v, rotate=False)
    out_rot = superl8.attn_int8_fwd(q, k, v, rotate=True)
    l1_plain, l1_rot = rel_l1(out_plain, oracle), rel_l1(out_rot, oracle)
    cos_plain, cos_rot = cos_sim(out_plain, oracle), cos_sim(out_rot, oracle)
    print(f"\nd={d} outliers: plain rel_l1={l1_plain:.4f} cos={cos_plain:.5f} | "
          f"rot rel_l1={l1_rot:.4f} cos={cos_rot:.5f}")

    # rotation is strictly, materially better under outliers (here ~3-4x lower
    # rel-L1) ...
    assert l1_rot < 0.5 * l1_plain, f"rotation barely helped: {l1_rot:.4f} vs {l1_plain:.4f}"
    # ... and clears the int8-PV stress bar under these deliberately extreme 40x
    # outliers (where even the rotated path carries ~2% residual), which the plain
    # path misses badly.
    assert_int8_quality(out_rot, oracle, what=f"rotated outlier d={d}",
                        min_cos=0.998, max_rel_l1=0.03, min_sqnr_db=18.0)
    assert l1_plain > 0.03, f"outliers not severe enough to matter (plain rel_l1={l1_plain:.4f})"


@pytest.mark.correctness
def test_cached_decode_rotation_matches(device):
    """INT8 KV-cache decode with matched rotate flags stays correct + deterministic."""
    q = torch.randn(2, 8, 1, 128, device=device, dtype=torch.float16)
    k = torch.randn(2, 2, 1024, 128, device=device, dtype=torch.float16)
    v = torch.randn(2, 2, 1024, 128, device=device, dtype=torch.float16)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v, rotate=True)
    out = superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale, rotate=True)
    oracle = attention_fp32_oracle(q, k, v)
    assert_int8_quality(out, oracle, what="cached decode rotated",
                        min_cos=0.998, max_rel_l1=0.03, min_sqnr_db=18.0)
    assert torch.equal(out, superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale, rotate=True))


@pytest.mark.correctness
def test_rotated_attention_cache_is_scoped_to_exact_cuda_device():
    """Priming cuda:0 must not strand the cached matrix away from cuda:1.

    H3 runs one contiguous transformer stage per GPU. A cache keyed only by
    ``device.type`` reuses cuda:0's matrix on cuda:1, raises a device mismatch,
    and makes the caller silently rescue every second-stage block through the
    much slower fp16 attention kernel.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required for the cache-device regression")

    hadamard_matrix.cache_clear()
    seed = torch.Generator(device="cpu").manual_seed(235)
    tensors = [
        torch.randn(1, 4, 128, 128, generator=seed, dtype=torch.float16).to("cuda:1")
        for _ in range(3)
    ]

    # Populate the same (D, dtype) cache entry from the first H3 stage.
    rotate_last(torch.zeros(1, 4, 8, 128, device="cuda:0", dtype=torch.float16))

    q, k, v = tensors
    got = superl8.attn_int8_fwd(
        q, k, v, rotate=True, causal=False, internal_accuracy_gate=False
    )
    repeat = superl8.attn_int8_fwd(
        q, k, v, rotate=True, causal=False, internal_accuracy_gate=False
    )
    oracle = attention_fp32_oracle(q, k, v)

    assert got.device == q.device
    assert torch.equal(got, repeat)
    assert_int8_quality(got, oracle, what="rotated attention on cuda:1")
