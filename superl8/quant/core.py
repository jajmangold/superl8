# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""SDNQ-informed symmetric per-row INT8 quantization (torch implementation).

Recipe (AGENTS.md): scale = amax(|x|, dim=-1) / 127 in fp32 (fp16 scales
overflow), RTN to [-127, 127] (symmetric — never -128), int32 accumulation
downstream, single dequant multiply. K-smoothing subtracts K's per-channel
mean before quantization: softmax is row-shift invariant, so attention output
is unchanged while channel outliers (the dominant int8 error source) vanish.

A fused Triton prologue may replace these on the hot path later; this torch
version stays as the reference/fallback. Algorithms follow Disty0/sdnq
(GPL-3.0) as a *reference only* — this is an independent implementation.
"""

from __future__ import annotations

import math

import torch

LOG2E = math.log2(math.e)
Q_MAX = 127.0
_K_MEAN_CHUNK = 1024


def _mean_k_fp32_bounded(k: torch.Tensor) -> torch.Tensor:
    """FP32 mean over keys without TensorIterator's long-reduction workspace."""
    n = k.shape[-2]
    if n <= _K_MEAN_CHUNK:
        return torch.mean(k, dim=-2, keepdim=True, dtype=torch.float32)
    mean = torch.zeros((*k.shape[:-2], 1, k.shape[-1]), device=k.device, dtype=torch.float32)
    for start in range(0, n, _K_MEAN_CHUNK):
        mean.add_(
            torch.sum(
                k[..., start : start + _K_MEAN_CHUNK, :],
                dim=-2,
                keepdim=True,
                dtype=torch.float32,
            )
        )
    return mean.div_(n)


def quantize_int8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row RTN int8. Returns (q int8 [..., d], scale fp32 [..., 1])."""
    if x.is_cuda:
        # The compiled operator reads fp16/bf16/fp32 and writes INT8 + FP32
        # scales directly. Besides being one launch, this avoids the eager
        # path's full-size x.float()/abs()/divide temporaries at long context.
        from superl8.ops import quantize_i8_rowwise

        shape = x.shape
        q, scale = quantize_i8_rowwise(x.reshape(-1, shape[-1]))
        return q.reshape(shape), scale.reshape(*shape[:-1], 1)
    scale = x.float().abs().amax(dim=-1, keepdim=True) / Q_MAX
    # Zero rows: keep scale finite; the row quantizes to exactly 0.
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(x.float() / safe).clamp_(-Q_MAX, Q_MAX).to(torch.int8)
    return q, safe


def dequantize_int8_rowwise(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_int8_rowwise (fp32 out — callers cast as needed)."""
    return q.float() * scale


def smooth_k(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract K's per-channel mean (over keys). Returns (k_smoothed, k_mean).

    k: [..., n, d]; k_mean: [..., 1, d]. Attention output is invariant because
    Q @ mean^T shifts every score in a row equally and softmax cancels it.
    """
    training = torch.is_grad_enabled() and k.requires_grad
    # Direct reduction preserves the functional autograd graph. In inference,
    # long reductions are split into bounded FP32 partial sums: PyTorch's direct
    # dtype-promoting mean requests an 80 MiB workspace at Qwen3.8's 32k shape,
    # while 1024-token partials peak at 8 MiB on the deployment fleet.
    k_mean = (
        torch.mean(k, dim=-2, keepdim=True, dtype=torch.float32)
        if training
        else _mean_k_fp32_bounded(k)
    )
    if training:
        # ``out=`` operations do not participate in autograd. Preserve the
        # functional training path for the library's forward/backward contract.
        k_s = (k.float() - k_mean).to(k.dtype)
    else:
        # TensorIterator computes the promoted fp32 subtraction and writes the
        # cast result directly to fp16/bf16 output, avoiding another full fp32 K.
        k_s = torch.empty_like(k)
        torch.sub(k, k_mean, out=k_s)
    return k_s, k_mean


def quantize_v_perchannel(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-CHANNEL int8 for V (scale over the key dim). Returns
    (v_int8 [...,n,d], v_scale [...,1,d]).

    Per-channel (not per-key) is required for int8 PV: in
    out[m,d] = sum_n P[m,n] V[n,d], a per-d scale factors cleanly out of the
    key-sum, whereas a per-key scale would sit inside it. (Review of
    TheTom/turboquant_plus: V compresses nearly free; K is what matters.)
    """
    scale = v.float().abs().amax(dim=-2, keepdim=True) / Q_MAX
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(v.float() / safe).clamp_(-Q_MAX, Q_MAX).to(torch.int8)
    return q, safe


def detect_q_outlier_domination(q: torch.Tensor, *, ratio_threshold: float = 12.0) -> bool:
    """Check if any Q row has a channel that dominates its per-row quant scale.

    A row is outlier-dominated when one channel's magnitude sets ``scale =
    max(|q|)/127``, crushing all other channels into a few int8 levels. This
    is detected by comparing ``max(|q|)`` to ``median(|q|)`` within each row
    — a row where max >> median has a dominant outlier. The original Q is
    checked (before any Hadamard rotation that might spread the spike).

    ``q``: fp16/bf16 [B, H, M, D] — the original Q tensor (before quantize_qk).
    ``ratio_threshold``: rows with ``max(|q|) / median(|q|) > threshold`` are
    considered outlier-dominated. Default 12.0 (safe margin above Gaussian
    expectation ~6, catches all 50x+ outlier factors).
    Returns True if *any* row exceeds the threshold.

    **CUDA graph capture note:** During ``torch.cuda.graph(...)`` capture, this
    function skips the gate to avoid device\u2192host sync (``.any()``). Callers that
    invoke kernels from graph capture must independently verify int8 quality
    per-model.
    """
    # Skip the data-dependent gate during CUDA graph capture to avoid
    # device\u2192host sync which raises "operation not permitted when stream is capturing"
    try:
        _capturing = hasattr(torch.cuda, 'is_current_stream_capturing') and torch.cuda.is_current_stream_capturing()
    except Exception:
        # If CUDA context unavailable or capture check raises, assume not capturing
        _capturing = False
    if _capturing:
        return False
    q_abs = q.float().abs()
    row_max = q_abs.amax(dim=-1)
    row_med = q_abs.median(dim=-1).values
    return bool(((row_max / (row_med + 1e-10)) > ratio_threshold).any())


def quantize_v_rowwise(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-TOKEN int8 for V (scale per key, KIVI-style). Returns
    (v_int8 [...,n,d], v_scale [...,n,1]).

    KIVI (arXiv:2402.02750) shows V outliers are per-token, not per-channel:
    some tokens have globally large magnitudes in all channels. A per-token
    scale captures each token's full range, giving better SQNR on outlier
    tokens than per-channel (which averages over tokens and clips outliers).
    The trade-off: per-token V scale cannot factor out of the key-sum in a
    simple int8 PV matmul, so the decode kernel must either multiply the
    scale inside the PV accumulation loop (paged-decode does this) or fall
    back to an fp16 dequant+matmul.
    """
    return quantize_int8_rowwise(v)


def quantize_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    rotate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize Q and (smoothed) K for the int8 dp4a QK^T.

    Folds `softmax_scale * log2(e)` into the Q scale so the kernel can dequant
    with one multiply and feed `exp2` directly:
        S_f = int32(QK^T) * q_scale * k_scale^T   (exp2-ready logits)

    With ``rotate=True`` a Hadamard incoherence rotation is applied to Q and the
    smoothed K before quantization. Q·K^T is invariant under a shared orthogonal
    rotation, so the logits are unchanged, but per-channel outliers in K (the
    dominant int8 error source) are spread out — materially tighter int8 on
    real-model activations. The kernel is unaffected (prologue-only).

    Returns (q_int8, q_scale[..., m, 1], k_int8, k_scale[..., n, 1], k_mean[..., 1, d]).
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    k_s, k_mean = smooth_k(k)
    if rotate:
        from .rotation import rotate_last

        q = rotate_last(q)
        k_s = rotate_last(k_s)
    q_i8, q_scale = quantize_int8_rowwise(q)
    k_i8, k_scale = quantize_int8_rowwise(k_s)
    q_scale = q_scale * (softmax_scale * LOG2E)
    return q_i8, q_scale, k_i8, k_scale, k_mean
