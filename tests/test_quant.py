# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR2: quantization layer tests (written FIRST — the implementation follows).

Contract (AGENTS.md numerics conventions, SDNQ-informed):
  - symmetric per-row RTN int8: scale = amax(|x|, dim=-1) / 127, fp32 scales
  - quantize_qk folds the softmax scale and log2(e) into the Q scale
  - K-smoothing subtracts K's per-channel mean (softmax row-shift invariant)
  - round-trip quality gated by SQNR/cos-sim, never allclose
"""

import math

import pytest
import torch

from tests.tolerances import assert_int8_quality, sqnr_db

# Deliberately imported before it exists — TDD red first.
from superl8.quant import quantize_int8_rowwise, dequantize_int8_rowwise, smooth_k, quantize_qk

SHAPES = [(64, 64), (128, 128), (2, 4, 257, 64), (1, 2, 333, 128)]


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_rowwise_roundtrip_quality(device, shape):
    x = torch.randn(shape, device=device, dtype=torch.float16)
    q, scale = quantize_int8_rowwise(x)
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float32  # fp16 scales overflow — SDNQ note
    assert scale.shape == (*shape[:-1], 1)
    x_dq = dequantize_int8_rowwise(q, scale)
    assert_int8_quality(x_dq, x, what=f"roundtrip {shape}")
    assert sqnr_db(x_dq, x) > 30.0


@pytest.mark.correctness
def test_rowwise_handles_zero_rows(device):
    """A zero row must not divide by zero; it round-trips to exactly zero."""
    x = torch.randn(8, 64, device=device, dtype=torch.float16)
    x[3] = 0
    q, scale = quantize_int8_rowwise(x)
    assert torch.isfinite(scale).all()
    x_dq = dequantize_int8_rowwise(q, scale)
    assert torch.all(x_dq[3] == 0)


@pytest.mark.correctness
def test_rowwise_saturates_at_127(device):
    x = torch.randn(16, 64, device=device, dtype=torch.float16)
    q, _ = quantize_int8_rowwise(x)
    assert q.max() <= 127 and q.min() >= -127  # symmetric: -127..127, never -128
    # each row's absmax element must hit exactly +-127 (scale defined by amax)
    assert (q.abs().amax(dim=-1) == 127).all()


@pytest.mark.correctness
def test_rowwise_long_context_uses_bounded_cuda_memory(device):
    """CUDA rowwise quantization writes INT8 directly, without a full FP32 input cast."""
    x = torch.randn(1, 4, 32768, 256, device=device, dtype=torch.float16)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device)

    q, scale = quantize_int8_rowwise(x)
    torch.cuda.synchronize(device)
    extra_peak = torch.cuda.max_memory_allocated(device) - base_alloc

    assert q.shape == x.shape and q.dtype == torch.int8
    assert scale.shape == (*x.shape[:-1], 1) and scale.dtype == torch.float32
    max_extra = x.numel() + 8 * 2**20
    assert extra_peak <= max_extra, (
        f"rowwise quantization extra peak {extra_peak / 2**20:.1f} MiB exceeds the "
        f"{max_extra / 2**20:.1f} MiB direct-to-int8 budget"
    )


@pytest.mark.correctness
def test_smooth_k_preserves_attention(device):
    """Softmax is row-shift invariant: smoothing K must not change attention output."""
    from tests.reference import attention_fp32_oracle

    b, h, m, d = 1, 2, 128, 64
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    # inject a strong channel outlier — the case smoothing exists for
    k[..., 7] += 8.0
    v = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k_s, k_mean = smooth_k(k)
    assert torch.allclose(k_s.float().mean(dim=-2), torch.zeros_like(k_mean.squeeze(-2)), atol=2e-3)
    out_ref = attention_fp32_oracle(q, k, v)
    out_smooth = attention_fp32_oracle(q, k_s, v)  # QK^T shifts per-row; softmax cancels it...
    # NOTE: softmax invariance requires the SAME shift across a row of S, which
    # q @ k_mean^T provides only if k_mean is constant across keys — it is (per-channel).
    torch.testing.assert_close(out_smooth, out_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.correctness
def test_smooth_k_improves_outlier_sqnr(device):
    """With a channel outlier, int8(K - mean) must round-trip better than int8(K)."""
    k = torch.randn(4, 8, 512, 64, device=device, dtype=torch.float16)
    k[..., 11] += 10.0  # channel outlier
    k_s, _ = smooth_k(k)
    q_raw, s_raw = quantize_int8_rowwise(k)
    q_sm, s_sm = quantize_int8_rowwise(k_s)
    raw_rt = dequantize_int8_rowwise(q_raw, s_raw)
    sm_rt = dequantize_int8_rowwise(q_sm, s_sm) + k.mean(dim=-2, keepdim=True)
    assert sqnr_db(sm_rt, k) > sqnr_db(raw_rt, k) + 3.0  # >=3 dB better


@pytest.mark.correctness
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_smooth_k_matches_explicit_fp32_reference(device, dtype):
    """The low-peak path must preserve the existing fp32 smoothing result exactly."""
    k = torch.randn(1, 4, 257, 128, device=device, dtype=dtype)
    ref_mean = k.float().mean(dim=-2, keepdim=True)
    ref = (k.float() - ref_mean).to(dtype)

    got, got_mean = smooth_k(k)

    assert torch.equal(got_mean, ref_mean)
    assert torch.equal(got, ref)


@pytest.mark.correctness
def test_smooth_k_does_not_materialize_full_fp32_inputs(device):
    """Long-context smoothing keeps the reduction below the 32k serving headroom."""
    # Use the production failure shape so PyTorch's fixed reduction workspace
    # does not dominate the ratio.
    k = torch.randn(1, 4, 32768, 256, device=device, dtype=torch.float16)
    ref_mean = k.float().mean(dim=-2, keepdim=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device)

    got, mean = smooth_k(k)
    torch.cuda.synchronize(device)
    extra_peak = torch.cuda.max_memory_allocated(device) - base_alloc

    assert torch.isfinite(got).all() and torch.isfinite(mean).all()
    torch.testing.assert_close(mean, ref_mean, rtol=1e-6, atol=1e-7)
    max_extra = k.nbytes + 8 * 2**20
    assert extra_peak <= max_extra, (
        f"smooth_k extra peak {extra_peak / 2**20:.1f} MiB exceeds the "
        f"{max_extra / 2**20:.1f} MiB 32k serving budget"
    )


@pytest.mark.correctness
def test_smooth_k_preserves_autograd(device):
    k = torch.randn(1, 2, 33, 16, device=device, dtype=torch.float32, requires_grad=True)
    ref_mean = k.mean(dim=-2, keepdim=True)
    ref = k - ref_mean
    grad = torch.randn_like(ref)
    ref.backward(grad, retain_graph=True)
    ref_grad = k.grad.detach().clone()
    k.grad = None

    got, got_mean = smooth_k(k)
    got.backward(grad)

    assert torch.equal(got_mean, ref_mean)
    assert torch.equal(got, ref)
    assert torch.equal(k.grad, ref_grad)


@pytest.mark.correctness
def test_quantize_qk_folds_scales(device):
    """int8 QK^T dequantized with the folded scales must match scaled fp32 QK^T."""
    b, h, m, d = 1, 2, 64, 64
    q = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, m, d, device=device, dtype=torch.float16)
    qq, q_scale, kq, k_scale, k_mean = quantize_qk(q, k)
    # int32 QK^T then dequant: (per-row q_scale) x (per-row k_scale)
    s_int = torch.einsum("bhmd,bhnd->bhmn", qq.float(), kq.float())
    s_dq = s_int * q_scale * k_scale.transpose(-1, -2)
    # reference: log2(e) * softmax_scale * (Q @ (K - mean)^T), exp2-ready
    softmax_scale = 1.0 / math.sqrt(d)
    s_ref = torch.einsum(
        "bhmd,bhnd->bhmn", q.float(), (k.float() - k.float().mean(dim=-2, keepdim=True))
    )
    s_ref = s_ref * softmax_scale * math.log2(math.e)
    assert sqnr_db(s_dq, s_ref) > 25.0
    assert torch.isfinite(k_mean).all()


# -----------------------------------------------------------------------
# KIVI-style V-per-token quantization tests (issue #99)
# -----------------------------------------------------------------------


@pytest.mark.correctness
@pytest.mark.parametrize("shape", SHAPES)
def test_v_rowwise_roundtrip_quality(device, shape):
    """V-per-token roundtrip SQNR must meet the int8 accuracy gate."""
    from superl8.quant import quantize_v_rowwise, dequantize_int8_rowwise

    v = torch.randn(shape, device=device, dtype=torch.float16)
    q, scale = quantize_v_rowwise(v)
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert scale.shape == (*shape[:-1], 1)  # per-token: one scale per row
    v_dq = dequantize_int8_rowwise(q, scale)
    assert_int8_quality(v_dq, v, what=f"V-per-token roundtrip {shape}")
    assert sqnr_db(v_dq, v) > 30.0


@pytest.mark.correctness
def test_v_per_token_beats_per_channel_on_token_outliers(device):
    """V-per-token gives better SQNR than V-per-channel when one token is an
    outlier (KIVI's core observation: V outliers are per-token, not
    per-channel). Inject a single token with large magnitude across all
    channels — per-channel scale dominates and clips all other tokens,
    while per-token scale bounds each token individually."""
    from superl8.quant import quantize_v_perchannel, quantize_v_rowwise

    b, h, n, d = 2, 4, 256, 64
    v = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    # Make token 100 a global outlier — large in every channel
    v[0, 0, 100] += 15.0
    v[0, 1, 100] += 12.0

    q_pc, s_pc = quantize_v_perchannel(v)
    q_pt, s_pt = quantize_v_rowwise(v)
    rt_pc = q_pc.float() * s_pc  # [B,H,N,D] per-channel dequant
    rt_pt = q_pt.float() * s_pt  # [B,H,N,D] per-token dequant

    sqnr_pc = sqnr_db(rt_pc, v)
    sqnr_pt = sqnr_db(rt_pt, v)
    assert sqnr_pt > sqnr_pc, (
        f"per-token SQNR {sqnr_pt:.1f} dB must beat per-channel {sqnr_pc:.1f} dB "
        f"on token-outlier inputs (KIVI claim)"
    )


@pytest.mark.correctness
def test_quantize_kv_cache_v_per_token_shapes(device):
    """quantize_kv_cache(v_quant='per_token') returns v_scale [B,H,N] (per-token)
    instead of [B,H,D] (per-channel). v_i8 shape is unchanged."""
    import superl8

    b, h, n, d = 2, 4, 128, 64
    k = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h, n, d, device=device, dtype=torch.float16)

    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v, v_quant="per_token")
    assert k_i8.shape == (b, h, n, d), k_i8.shape
    assert k_i8.dtype == torch.int8
    assert k_scale.shape == (b, h, n), k_scale.shape
    assert k_scale.dtype == torch.float32
    assert v_i8.shape == (b, h, n, d), v_i8.shape
    assert v_i8.dtype == torch.int8
    assert v_scale.shape == (b, h, n), v_scale.shape  # per-token: [B,H,N]
    assert v_scale.dtype == torch.float32

    # Default (per_channel) must still work
    k_i8_c, k_scale_c, v_i8_c, v_scale_c = superl8.quantize_kv_cache(k, v)
    assert v_scale_c.shape == (b, h, d), v_scale_c.shape  # per-channel: [B,H,D]


@pytest.mark.correctness
def test_quantize_kv_cache_v_per_token_vs_per_channel_outputs(device):
    """Both quant granularities must pass the accuracy gate independently."""
    import superl8

    b, h, n, d = 2, 4, 200, 64
    k = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    k_s, _k_mean = smooth_k(k)  # K is smoothed inside quantize_kv_cache

    for vq in ("per_channel", "per_token"):
        k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v, v_quant=vq)
        # V round-trip: per-channel scale [B,H,D] broadcasts over N;
        # per-token scale [B,H,N] broadcasts over D.
        if vq == "per_channel":
            v_dq = v_i8.float() * v_scale.unsqueeze(-2)
        else:
            v_dq = v_i8.float() * v_scale.unsqueeze(-1)
        assert_int8_quality(v_dq, v, what=f"V-{vq} roundtrip")
        assert sqnr_db(v_dq, v) >= 20.0

        # K round-trip: K-scale [B,H,N] always per-token
        k_dq = k_i8.float() * k_scale.unsqueeze(-1)
        assert_int8_quality(k_dq, k_s, what=f"K roundtrip ({vq})")


@pytest.mark.correctness
def test_v_per_token_decode_via_paged_path(device):
    """V-per-token quantized cache, decoded through the paged-decode path,
    must match the fp32 oracle within int8 accuracy bars."""
    import superl8
    from tests.reference import attention_fp32_paged_oracle

    b, hq, hkv, n, d = 4, 8, 4, 256, 64
    block_size = 16
    context_lens = [200, 100, 256, 50]
    q = torch.randn(b, hq, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)

    # Quantize V per-token
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v, v_quant="per_token")
    # k_scale: [B,H,N] per-token; v_scale: [B,H,N] per-token

    # Repack into a paged cache with an identity block table (logical==physical)
    blocks_per_seq = [(cl + block_size - 1) // block_size for cl in context_lens]
    max_blocks = max(blocks_per_seq)
    num_blocks = sum(blocks_per_seq)
    block_table = torch.zeros((b, max_blocks), dtype=torch.int32, device=device)
    cursor = 0
    for bi, nb in enumerate(blocks_per_seq):
        block_table[bi, :nb] = torch.arange(cursor, cursor + nb, dtype=torch.int32)
        cursor += nb

    k_cache = torch.zeros(num_blocks, hkv, block_size, d, dtype=torch.int8, device=device)
    ks_cache = torch.ones(num_blocks, hkv, block_size, dtype=torch.float32, device=device)
    v_cache = torch.zeros(num_blocks, hkv, block_size, d, dtype=torch.int8, device=device)
    vs_cache = torch.ones(num_blocks, hkv, block_size, dtype=torch.float32, device=device)

    # Scatter the contiguous quantized K/V into the paged layout
    for bi in range(b):
        for t in range(context_lens[bi]):
            blk = t // block_size
            off = t % block_size
            phys = int(block_table[bi, blk].item())
            k_cache[phys, :, off] = k_i8[bi, :, t]
            ks_cache[phys, :, off] = k_scale[bi, :, t]
            v_cache[phys, :, off] = v_i8[bi, :, t]
            vs_cache[phys, :, off] = v_scale[bi, :, t]

    cl = torch.tensor(context_lens, dtype=torch.int32, device=device)
    out = superl8.attn_paged_decode_cached(
        q,
        k_cache,
        ks_cache,
        v_cache,
        vs_cache,
        block_table,
        cl,
        block_size,
        rotate=False,  # K was smoothed (not rotated) by quantize_kv_cache
    )
    oracle = attention_fp32_paged_oracle(q, k, v, context_lens)
    assert out.shape == q.shape
    assert_int8_quality(
        out,
        oracle,
        what="V-per-token paged decode",
        min_cos=0.998,
        max_rel_l1=0.03,
        min_sqnr_db=18.0,
    )


@pytest.mark.correctness
def test_attn_decode_cached_rejects_v_per_token(device):
    """The contiguous decode kernel only supports per-channel V scales.
    Passing per-token scales must raise a clear RuntimeError, not silently
    compute garbage."""
    import superl8

    b, h, n, d = 2, 4, 64, 64
    k = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h, n, d, device=device, dtype=torch.float16)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v, v_quant="per_token")
    q = torch.randn(b, h, 1, d, device=device, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="per-token V scale"):
        superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale)
