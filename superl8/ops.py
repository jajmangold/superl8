# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Python-facing op wrappers over the compiled `superl8._C` extension."""

import os

import torch


class _MissingC:
    """Stand-in for the CUDA extension when it isn't built (CPU-only installs, e.g.
    running the pure-torch quantization path on an HF Job). Importing `superl8` and using
    `superl8.format` / `superl8.quant` works fine; only *calling* a kernel raises, clearly."""

    def __init__(self, err):
        self._err = err

    def __getattr__(self, name):
        raise RuntimeError(
            f"superl8._C (the sm_70 CUDA extension) is not available, so kernel op "
            f"'{name}' can't run. Build superl8 on a Volta GPU (CUDA 12.9), or use only "
            f"the CPU quantization path (superl8.format / superl8.quant). Import error: {self._err}"
        )


try:
    from . import _C  # compiled CUDA extension
except ImportError as _e:  # CPU-only: keep import working; kernels error on use
    _C = _MissingC(_e)


def hello_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise fp16 add. Toolchain smoke op — proves build/bind/dispatch."""
    return _C.hello_add(a, b)


def backward_cuda(q, k, v, out, lse, d_out, *, causal: bool = False, scale=None):
    """Fused FA2 backward (CUDA) -> (dq, dk, dv), all fp16 [B,H,S,D].

    lse [B,H,M] fp32 (natural log) from the forward; recomputed here if None.
    PR5b-1 runs fp32 accumulation on CUDA cores; PR5b-2 int8-ifies S/dV/dQ/dK.
    """
    import math

    scale = (1.0 / math.sqrt(q.shape[-1])) if scale is None else scale
    if lse is None:
        kf = k.float()
        if q.shape[1] != k.shape[1]:  # GQA: expand K heads to match Q for the LSE recompute
            kf = kf.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        s = torch.einsum("bhmd,bhnd->bhmn", q.float(), kf) * scale
        if causal:
            m, n = s.shape[-2], s.shape[-1]
            mask = torch.ones(m, n, device=s.device, dtype=torch.bool).tril(n - m)
            s = s.masked_fill(~mask, float("-inf"))
        lse = torch.logsumexp(s, dim=-1).to(torch.float32).contiguous()
    return _C.attn_bwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        out.contiguous(),
        d_out.contiguous(),
        lse.contiguous(),
        float(scale),
        causal,
    )


def attn_int8_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    int8_pv: bool = False,
    rotate: bool = False,
    window_left: int = -1,
    per_warp_quant: bool = True,
    internal_accuracy_gate: bool = True,
) -> torch.Tensor:
    """int8 dp4a FlashAttention-2 forward for sm_70.

    Default: INT8 QK^T with fp32-accumulated **fp16-or-bf16 PV** — ``v``'s dtype
    (``float16`` or ``bfloat16``) selects the PV/output store dtype; the fp32
    softmax/dequant math is unaffected. bf16 avoids the fp16 overflow (max
    65504) that black-images bf16-native models (Gemma, most diffusion DiTs).
    With ``int8_pv=True`` the PV matmul is ALSO int8 dp4a (full W8A8): V is
    quantized per-channel and P is re-quantized per-row to int8 in-loop (this
    path is fp16-out only). q, k, v: fp16/bf16 [B, H, S, D], D in {32, 64, 72,
    80, 128, 256}. QK quantization (per-row symmetric int8 + K-smoothing,
    exp2-ready folded scales) happens in the prologue; softmax runs fp32.
    ``rotate=True`` adds a Hadamard incoherence rotation to Q/K for tighter int8
    on outlier-heavy (real-model) activations — logit-invariant, no kernel change.
    Rectangular causal attention aligns the diagonal to the bottom-right, so
    query ``i`` attends through key ``i + N - M``. ``window_left >= 0`` applies
    the left window around that shifted position; ``-1`` = no window. It applies
    to the fp16-PV path only and is meaningful only with ``causal=True``.

    ``internal_accuracy_gate=False`` is for callers that independently compare the
    returned DP4A output against a full-precision reference and demote on output SQNR.
    It bypasses the conservative any-row Q detector so such a caller can evaluate the
    actual quantized result. The default stays ``True`` for all standalone callers.

    **Accuracy gate (SageAttention discipline):** when any Q row has a dominant
    outlier channel (max(|q|) / median(|q|) > 12, indicating the per-row int8
    scale is set by that channel and all others are crushed), this function
    transparently falls back to full fp16 via PyTorch SDPA. The gate decides
    where int8 is allowed — never weaken the bar.
    """
    from .quant import detect_q_outlier_domination, quantize_qk, quantize_v_perchannel
    import math as _math

    # Accuracy gate (SageAttention discipline): detect Q outlier domination on
    # the ORIGINAL Q (before any Hadamard rotation). If int8 QK cannot meet the
    # quality bar (one channel dominates the per-row scale, crushing all others),
    # fall back entirely to fp16. The gate decides where int8 is allowed — never
    # weaken the bar to fit.
    if internal_accuracy_gate and detect_q_outlier_domination(q):
        # Memory-efficient fp16 fallback (issue #97): the tiled half2 FA-2 prefill
        # kernel is O(N) memory, unlike torch SDPA which has no flash backend on
        # Volta and materializes the O(N^2) score matrix (OOMs on the long
        # video-DiT attention this fallback is most likely to fire on). Same
        # fp16/bf16 math the gate demands, GQA-aware.
        s = (1.0 / _math.sqrt(q.shape[-1])) if scale is None else scale
        if v.dtype == q.dtype and k.dtype == q.dtype and q.shape[-1] in (32, 64, 72, 80, 128, 256):
            return attn_fp16_fwd(
                q, k, v, causal=causal, scale=s, window_left=window_left
            )
        # Mixed dtypes / unsupported head dim -> torch SDPA (attn_fp16_fwd needs
        # one shared dtype and a supported D).
        import torch.nn.functional as _F

        attn_mask = None
        if causal and q.shape[-2] != k.shape[-2]:
            m, n = q.shape[-2], k.shape[-2]
            attn_mask = torch.ones(m, n, device=q.device, dtype=torch.bool).tril(n - m)
        return _F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
            scale=s,
            enable_gqa=q.shape[1] != k.shape[1],
        )

    q_i8, q_scale, k_i8, k_scale, _k_mean = quantize_qk(q, k, softmax_scale=scale, rotate=rotate)
    q_scale_folded = q_scale.squeeze(-1).contiguous()
    k_scale = k_scale.squeeze(-1).contiguous()
    if int8_pv:
        if window_left >= 0:
            raise ValueError("window_left is only supported on the fp16-PV path (int8_pv=False)")
        v_i8, v_scale = quantize_v_perchannel(v)  # [...,1,d]
        return _C.attn_w8a8_fwd(
            q_i8.contiguous(),
            q_scale_folded,
            k_i8.contiguous(),
            k_scale,
            v_i8.contiguous(),
            v_scale.squeeze(-2).contiguous(),
            causal,
            causal_diag=k.shape[-2] - q.shape[-2] if causal else 0,
            per_warp_quant=per_warp_quant,
        )
    return _C.attn_int8_fwd(
        q_i8.contiguous(),
        q_scale_folded,
        k_i8.contiguous(),
        k_scale,
        v.contiguous(),
        causal,
        window_left,
    )


def attn_fp16_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
    window_left: int = -1,
) -> torch.Tensor:
    """Tiled fp16/bf16 half2 FlashAttention-2 **prefill** forward for sm_70.

    The full-precision, O(N)-memory sibling of :func:`attn_int8_fwd`: same FA-2
    tiling (streams the K/V tiles, online fp32 softmax) but the QK^T inner loop
    is a **half2 CUDA-core dot** (the healthy Volta pipe, ~27 TFLOP/s — a
    SEPARATE pipe from the firmware-dead tensor cores; NEVER wmma/HMMA) instead
    of int8 dp4a. No quant prologue, no scales: ``q``, ``k``, ``v`` are plain
    fp16/bf16 ``[B, H, S, D]`` (all one dtype), ``D`` in {32, 64, 72, 80, 128,
    256}, and the output follows ``v``'s dtype. Softmax/LSE stay fp32.

    This exists because torch SDPA has **no flash / memory-efficient backend on
    Volta**: its math backend materializes the O(N^2) score matrix and OOMs on
    the ~10k-17k-token self-attention of video DiTs (LTX/Wan/Hunyuan-Video). This
    kernel is O(N) memory, so it runs where SDPA cannot — and it makes the int8
    accuracy-gate fp16 fallback memory-efficient. GQA/MQA (``H_q % H_kv == 0``)
    is supported. Rectangular causal attention is bottom-right aligned, matching
    FlashAttention. ``window_left >= 0`` applies a shifted causal left window;
    ``-1`` disables it. ``scale`` defaults to ``1/sqrt(D)``.

    ``mask`` (optional): an **additive** ``[B|1, H|1, M, N]`` bias in **natural
    log** space (broadcastable over batch/heads, same dtype as ``q``), added to
    the logits before softmax — the guided/reference video-DiT case (LTX's
    ``self_attention_mask``). It is indexed in place via strides (0 on broadcast
    axes), so a ``[B,1,N,N]`` mask is NOT expanded to ``[B,H,N,N]`` — the O(N^2)
    memory win is preserved.
    """
    # NB: do NOT force the mask contiguous — a broadcast/expanded [B,1,N,N] mask
    # would then be materialized to [B,H,N,N], defeating the O(N^2) memory win.
    # The kernel indexes it via strides (0 on broadcast axes). It must have a
    # unit-stride last dim and share q's dtype (both checked in the C++ op).
    out = _C.attn_fp16_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        causal,
        scale,
        window_left,
        mask,
    )
    return out


def attn_int8_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    rotate: bool = False,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Decode (M=1) int8 dp4a FlashAttention via split-KV (flash-decoding).

    For autoregressive decode: ONE query row (M=1) against a long K/V cache. The
    key dim is split across blocks so the GPU stays busy, then partials merge via
    the log-sum-exp trick. The single query attends to ALL N keys (decode = full
    attention over the cache). q: fp16 [B, H_q, 1, D]; k, v: fp16 [B, H_kv, N, D],
    D in {32, 64, 128, 256}. QK go int8 dp4a (same prologue as the prefill path); V
    stays fp16. Returns fp16 [B, H_q, 1, D].

    ``num_splits``: override the number of KV splits (flash-decoding parallelism).
    Larger values produce more blocks to fill SMs at low batch, at the cost of a
    heavier merge step. ``None`` (default) auto-tunes from the context length and
    batch×head count. Use explicit split counts to experiment with the trade-off.
    """
    from .quant import quantize_qk

    assert q.shape[-2] == 1, f"decode expects M=1, got M={q.shape[-2]}"
    q_i8, q_scale, k_i8, k_scale, _k_mean = quantize_qk(q, k, softmax_scale=scale, rotate=rotate)
    q_scale = q_scale.reshape(q.shape[0], q.shape[1], 1).contiguous()
    k_scale = k_scale.squeeze(-1).contiguous()
    ns = -1 if num_splits is None else int(num_splits)
    return _C.attn_int8_decode(
        q_i8.contiguous(),
        q_scale,
        k_i8.contiguous(),
        k_scale,
        v.contiguous(),
        ns,
    )


def _varlen_smooth_k(
    k: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_k: int
) -> torch.Tensor:
    """Per-sequence K-smoothing over packed [T, H, D] with cu_seqlens_k [B+1].

    Returns k_s [T, H, D] fp16 = k - each token's own sequence's per-(head,channel)
    K mean — the same value serial `smooth_k` produces per sequence (bit-exact when
    computed the same way). Equal-length batches are detected WITHOUT reading CUDA
    data (no .item()/sync): every segment is <= ``max_seqlen_k``, so ``total ==
    B * max_seqlen_k`` iff all segments are that length. That path is a vectorized
    reshape + ``mean(dim=1, keepdim=True)`` + broadcast subtract (the production
    fast path — no per-row gather, no repeat_interleave). Ragged batches copy the
    bounds to a CPU list ONCE, then per-slice ``mean`` over each contiguous segment
    (bit-exact to serial) and concatenate. ``torch.segment_reduce`` is NOT used:
    its mean is not bit-exact to serial (measured ~1.2e-7) and varlen parity is
    bit-exact.
    """
    kf = k.float()
    B = len(cu_seqlens_k) - 1
    if kf.shape[0] == B * max_seqlen_k and max_seqlen_k > 0:
        L = max_seqlen_k
        kf3 = kf.reshape(B, L, *kf.shape[1:])                 # view
        k_mean = kf3.mean(dim=1, keepdim=True)                # [B,1,H,D]
        return (kf3 - k_mean).reshape(-1, *kf.shape[1:]).to(k.dtype)
    # Ragged: partition the packed sequences into maximal CONSECUTIVE runs that
    # share one length. Each run is contiguous and reshapes to [run_B, L, H, D], so
    # it uses the same vectorized mean(dim=1)+broadcast as the equal-length path —
    # identical per-sequence reduction order, but a handful of vectorized runs
    # instead of one GPU launch per segment. Zero-length segments contribute no
    # tokens and are skipped. (torch.segment_reduce is NOT used: its mean is not
    # bit-exact to serial, measured ~1.2e-7.)
    bounds = cu_seqlens_k.cpu().tolist()
    parts = []
    i = 0
    while i < B:
        L = bounds[i + 1] - bounds[i]
        j = i
        while j + 1 < B and (bounds[j + 2] - bounds[j + 1]) == L:
            j += 1
        if L > 0:
            run_B = j - i + 1
            kf3 = kf[bounds[i]:bounds[j + 1]].reshape(run_B, L, *kf.shape[1:])
            k_mean = kf3.mean(dim=1, keepdim=True)            # [run_B,1,H,D]
            parts.append((kf3 - k_mean).reshape(-1, *kf.shape[1:]).to(k.dtype))
        i = j + 1
    return torch.cat(parts, 0) if parts else kf.to(k.dtype)


def attn_int8_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Varlen (cu_seqlens-packed) int8 dp4a prefill forward — the serving path.

    q, k, v are packed [total_tokens, H, D] fp16 (token-major, heads interleaved,
    the flash_attn_varlen convention); cu_seqlens_{q,k} are int32 [batch+1]
    cumulative offsets. Each sequence attends only within itself; ``causal``
    aligns the diagonal to the sequence end (KV-cache-prefix compatible). QK go
    int8 dp4a; V stays fp16. Returns fp16 [total_q, H, D].
    """
    import math

    from .quant import LOG2E, quantize_int8_rowwise

    softmax_scale = (1.0 / math.sqrt(q.shape[-1])) if scale is None else scale
    # K-smoothing: subtract the per-(head,channel) mean of EACH sequence's OWN keys.
    # A single GLOBAL mean over all packed keys is NOT softmax-invariant once the
    # per-row int8 quantization is applied — for non-power-of-two / ragged / differing
    # prompts the packed-wide mean differs (even in fp32 accumulation order) from the
    # per-sequence mean the serial `smooth_k` path uses, flipping int8 QK codes and
    # changing the output (superl8#241, breaks fni8-serve#351 non-power varlen parity).
    k_s = _varlen_smooth_k(k, cu_seqlens_k, int(max_seqlen_k))
    q_i8, q_scale = quantize_int8_rowwise(q)  # [total_q,H,D], [total_q,H,1]
    k_i8, k_scale = quantize_int8_rowwise(k_s)
    q_scale = (q_scale * (softmax_scale * LOG2E)).squeeze(-1).contiguous()  # [total_q,H]
    k_scale = k_scale.squeeze(-1).contiguous()  # [total_k,H_kv]
    return _C.attn_int8_varlen(
        q_i8.contiguous(),
        q_scale,
        k_i8.contiguous(),
        k_scale,
        v.contiguous(),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        int(max_seqlen_q),
        causal,
    )


def quantize_kv_cache_i4v(k: torch.Tensor, v: torch.Tensor):
    """Quantize a K/V cache with int8 K and INT4 V — the byte-aligned low-bit lever.

    K is smoothed + per-row int8 (as usual). V is per-channel symmetric int4 (scale
    = amax/7 over keys) and packed 2 channels/byte -> ``v_i4 [B,H_kv,N,D/2]`` int8.
    Halves the V cache vs int8 (int8-K/int4-V = 0.75x the int8 cache, 0.375x fp16 ->
    ~2.7x more context in 16 GB). Returns (k_i8 [B,H_kv,N,D], k_scale [B,H_kv,N],
    v_i4 [B,H_kv,N,D/2], v_scale [B,H_kv,D]). Feed to :func:`attn_decode_cached_i4v`.
    """
    from .quant import quantize_int8_rowwise, smooth_k

    from .quant.lowbit import NF4_CODEBOOK

    k_s, _k_mean = smooth_k(k)
    k_i8, k_scale = quantize_int8_rowwise(k_s)
    # V: per-(channel, key-block of GROUP=32) NF4 (non-uniform 16 levels at the
    # normal's quantiles). NF4 + fine grouping recovers most of the int4->int8 gap
    # (uniform g128 rel-L1 0.117 -> NF4 g32 0.088). GROUP matches DEC_I4V_GROUP.
    GROUP = 32
    nf4 = torch.tensor(NF4_CODEBOOK, device=v.device, dtype=torch.float32)
    vf = v.float()
    b, h, n, d = vf.shape
    nb = (n + GROUP - 1) // GROUP
    pad = nb * GROUP - n
    vp = torch.nn.functional.pad(vf, (0, 0, 0, pad)) if pad else vf
    vg = vp.reshape(b, h, nb, GROUP, d)
    v_scale = vg.abs().amax(dim=3)  # [B,H_kv,nb,D] absmax
    v_scale = torch.where(v_scale == 0, torch.ones_like(v_scale), v_scale)
    vn = (vg / v_scale.unsqueeze(3)).clamp_(-1, 1)  # normalise to [-1,1]
    idx = (vn.unsqueeze(-1) - nf4).abs().argmin(-1)  # nearest NF4 index 0..15
    codes = idx.reshape(b, h, nb * GROUP, d)[:, :, :n, :].to(torch.int16)  # unpad
    lo = codes[..., 0::2] & 0xF
    hi = codes[..., 1::2] & 0xF
    v_i4 = (lo | (hi << 4)).to(torch.int8)  # [B,H_kv,N,D/2] NF4 indices
    return (
        k_i8.contiguous(),
        k_scale.squeeze(-1).contiguous(),
        v_i4.contiguous(),
        v_scale.contiguous(),
    )


def attn_decode_cached_i4v(q, k_i8, k_scale, v_i4, v_scale, *, scale=None, num_splits=None):
    """Decode (M=1) against an int8-K / INT4-V cache (from
    :func:`quantize_kv_cache_i4v`). Same as :func:`attn_decode_cached` but the V
    cache is int4 (unpacked in-kernel). Returns fp16 [B, H_q, 1, D].

    ``num_splits``: override the number of KV splits; ``None`` = auto."""
    import math

    from .quant import LOG2E, quantize_int8_rowwise

    softmax_scale = (1.0 / math.sqrt(q.shape[-1])) if scale is None else scale
    q_i8, q_scale = quantize_int8_rowwise(q)
    q_scale = (q_scale * (softmax_scale * LOG2E)).reshape(q.shape[0], q.shape[1], 1).contiguous()
    ns = -1 if num_splits is None else int(num_splits)
    return _C.attn_int4v_decode(
        q_i8.contiguous(),
        q_scale,
        k_i8.contiguous(),
        k_scale.contiguous(),
        v_i4.contiguous(),
        v_scale.contiguous(),
        ns,
    )


def attn_int8_verify(
    q: torch.Tensor,
    k_i8: torch.Tensor,
    k_scale: torch.Tensor,
    v_i8: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    scale: float | None = None,
    use_split: bool = True,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Speculative-decode / MTP VERIFY against a persistent INT8 KV cache.

    ``q``: fp16 [B, H_q, k, D] — the k draft tokens to verify in ONE forward.
    ``k_i8, k_scale, v_i8, v_scale``: the int8 cache from :func:`quantize_kv_cache`,
    where the cache = the prefix + the drafts' own K/V (length N = prefix + k).
    Runs int8 dp4a QK + int8 dp4a PV, causal with the diagonal aligned to the
    sequence END — draft i attends the cache prefix + itself + preceding drafts
    (the exact mask a chain/sequential spec-decode step needs). Only the query is
    quantized per call; the cache is not re-quantized. Returns fp16 [B, H_q, k, D].

    ``use_split`` (default True): route through the flash-decoding **split-KV**
    verify kernel — it maps the k drafts onto the decode split machinery as k
    staggered-causal decode queries, launching ``n_splits·B·H_q·k`` blocks so the
    tiny-M / large-N verify shape actually fills the SMs (the dense-tile kernel
    launched only ``B·H_q`` blocks → ~6% occupancy, latency-bound). The LSE combine
    is split-invariant, so this is numerically the same as the ``use_split=False``
    dense-tile ``attn_w8a8_fwd`` path. ``num_splits``: override the KV-split count
    (``None`` = auto-tune from context length and B·H_q·k); only used when
    ``use_split``.
    """
    import math

    from .quant import LOG2E, quantize_int8_rowwise

    _, _, k, d = q.shape
    n = k_i8.shape[2]
    softmax_scale = (1.0 / math.sqrt(d)) if scale is None else scale
    q_i8, q_scale = quantize_int8_rowwise(q)
    q_scale = (q_scale * (softmax_scale * LOG2E)).squeeze(-1).contiguous()
    if use_split:
        ns = -1 if num_splits is None else int(num_splits)
        return _C.attn_int8_verify_split(
            q_i8.contiguous(),
            q_scale,
            k_i8.contiguous(),
            k_scale.contiguous(),
            v_i8.contiguous(),
            v_scale.contiguous(),
            n - k,  # causal_diag = prefix
            ns,
        )
    return _C.attn_w8a8_fwd(
        q_i8.contiguous(),
        q_scale,
        k_i8.contiguous(),
        k_scale.contiguous(),
        v_i8.contiguous(),
        v_scale.contiguous(),
        True,
        n - k,
    )


def attn_int8_tree_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tree_mask: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """TREE-attention verify — the EAGLE / tree-speculative-decoding path.

    Verify a whole draft TREE in one forward. ``q``: fp16 [B, H_q, T, D] the T tree
    nodes; ``k, v``: fp16 [B, H_kv, N, D] = [prefix + T tree nodes] (N = N_p + T);
    ``tree_mask``: [T, T] bool/int8 where row qi marks which of the T tree nodes qi
    attends (its ancestors-or-self). Each node attends the full prefix + its tree
    ancestors (NOT siblings) — the custom mask EAGLE needs. int8 dp4a QK + fp16 PV.
    Returns fp16 [B, H_q, T, D]. (Single shared tree across the batch/heads.)
    """
    from .quant import quantize_qk

    q_i8, q_scale, k_i8, k_scale, _k_mean = quantize_qk(q, k, softmax_scale=scale)
    q_scale = q_scale.squeeze(-1).contiguous()
    k_scale = k_scale.squeeze(-1).contiguous()
    tm = tree_mask.to(torch.int8).contiguous()
    return _C.attn_int8_tree(
        q_i8.contiguous(),
        q_scale,
        k_i8.contiguous(),
        k_scale,
        v.contiguous(),
        tm,
    )


def quantize_kv_cache(
    k: torch.Tensor, v: torch.Tensor, *, rotate: bool = False, v_quant: str = "per_channel"
):
    """Quantize a K/V cache to INT8 for persistent storage (quantize once).

    K is smoothed (per-channel mean subtracted — softmax-invariant) then
    per-row symmetric int8.

    V quant granularity is gated by ``v_quant`` (KIVI-style, issue #99):
      * ``"per_channel"`` (default) — one scale per head×channel over all keys.
        v_scale ``[B,H_kv,D]``. The per-channel scale factors cleanly out of
        the key-sum in int8 PV, so the decode kernel stays simple. This is the
        current path, good for most workloads.
      * ``"per_token"`` — one scale per key (KIVI V quant: arXiv:2402.02750).
        v_scale ``[B,H_kv,N]``. Better SQNR on outlier tokens (some tokens have
        globally large V magnitudes in all channels) and buys memory for
        long-context regimes at the cost of a heavier PV accumulation loop
        in the decode kernel. This path is experimental — the contiguous
        decode kernel (:func:`attn_decode_cached`) does not support it yet;
        use the paged decode path (:func:`attn_paged_decode_cached`) or
        repack into a paged cache.

    Returns ``(k_i8 [B,H_kv,N,D], k_scale [B,H_kv,N], v_i8 [B,H_kv,N,D],
    v_scale [B,H_kv,D] | [B,H_kv,N])`` — half the HBM of the fp16 cache.
    Feed to :func:`attn_decode_cached` (per_channel V) or
    :func:`attn_paged_decode_cached` (per_token V) each step; no per-step
    K/V re-quantization.

    ``rotate=True`` applies a Hadamard incoherence rotation to K before quant
    (Q·K^T-invariant, spreads K channel outliers -> tighter int8 on real-model
    caches). **The query MUST be decoded with the same flag** —
    ``attn_decode_cached(..., rotate=True)`` — or the logits will not match.

    The ``v_quant`` choice is the only per-model gate; store it alongside the
    cache in ``.superl8`` ``__meta__`` as ``"v_quant_granularity"``."""
    from .quant import quantize_int8_rowwise, quantize_v_perchannel, quantize_v_rowwise, smooth_k

    if v_quant not in ("per_channel", "per_token"):
        raise ValueError(f"v_quant must be 'per_channel' or 'per_token', got {v_quant!r}")

    k_s, _k_mean = smooth_k(k)
    if rotate:
        from .quant.rotation import rotate_last

        k_s = rotate_last(k_s)
    k_i8, k_scale = quantize_int8_rowwise(k_s)
    if v_quant == "per_channel":
        v_i8, v_scale = quantize_v_perchannel(v)
        v_scale = v_scale.squeeze(-2)  # [B,H_kv,D]
    else:
        v_i8, v_scale = quantize_v_rowwise(v)
        v_scale = v_scale.squeeze(-1)  # [B,H_kv,N]
    return (
        k_i8.contiguous(),
        k_scale.squeeze(-1).contiguous(),
        v_i8.contiguous(),
        v_scale.contiguous(),
    )


def quantize_kv_cache_3bit(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rotate: bool = True,
    fit: bool = True,
    block_size: int = 128,
):
    """OPT-IN 3-bit Lloyd-Max ("TQ3", qengine-adapted) KV STORAGE codec — trades
    quality for ~4.9x smaller KV to free VRAM / extend context.

    This is a storage codec only: it packs the (smoothed + Hadamard-rotated) K and
    the raw V at 3 bits. sm_70 has no int3 dp4a, so there is no fused 3-bit decode
    kernel — read back with :func:`dequantize_kv_cache_3bit` (to fp16), then run the
    normal decode. The Hadamard rotation is the EXISTING KV-path rotation
    (:func:`superl8.quant.rotation.rotate_last`); this only adds the quantizer on top.

    NOT default-on: measured attention-output quality is ~cos 0.98 / rel-L1 ~0.18
    (SQNR ~14.5 dB) — below the int8 gate — so it stays opt-in and falls back to
    int8/fp16 for quality-critical serving. ``fit=True`` (recommended) fits the
    Lloyd-Max codebook to the rotated distribution (much better on outlier-heavy K).

    Returns a dict with packed int32 K/V codes, per-block fp32 norms, the fp32
    codebooks, and the K channel-mean (needed to reconstruct un-smoothed K if ever
    required). Decode with ``rotate=`` matching the ``rotate`` used here."""
    from .quant import smooth_k
    from .quant.lloydmax import (
        lloyd_max_fit,
        lloydmax_bits_levels,
        pack_indices_lowbit,
        quantize_lloydmax,
    )
    from .quant.rotation import rotate_last

    k_s, k_mean = smooth_k(k)
    if rotate:
        k_s = rotate_last(k_s)

    def _enc(x):
        cb = None
        if fit:
            xf = x.float()
            nb = xf.shape[-1] // block_size
            xg = xf.reshape(*xf.shape[:-1], nb, block_size)
            norm = xg.pow(2).sum(-1, keepdim=True).sqrt()
            xn = xg / torch.where(norm == 0, torch.ones_like(norm), norm)
            cb = lloyd_max_fit(xn, lloydmax_bits_levels(3))
        codes, norm, codebook = quantize_lloydmax(x, bits=3, block_size=block_size, codebook=cb)
        return pack_indices_lowbit(codes, 3).contiguous(), norm.contiguous(), codebook

    k_packed, k_norm, k_cb = _enc(k_s)
    v_packed, v_norm, v_cb = _enc(v)
    return {
        "k_packed": k_packed,
        "k_norm": k_norm,
        "k_codebook": k_cb,
        "v_packed": v_packed,
        "v_norm": v_norm,
        "v_codebook": v_cb,
        "k_mean": k_mean,
        "rotate": rotate,
        "block_size": block_size,
        "bits": 3,
    }


def dequantize_kv_cache_3bit(packed: dict, d: int):
    """Inverse of :func:`quantize_kv_cache_3bit` -> ``(k, v)`` fp16 (K still
    smoothed+rotated, matching the ``rotate=`` decode contract). ``d`` is the head
    dim (needed to unpack the last axis)."""
    from .quant.lloydmax import dequantize_lloydmax, unpack_indices_lowbit

    bs = packed["block_size"]
    k_codes = unpack_indices_lowbit(packed["k_packed"], 3, d)
    v_codes = unpack_indices_lowbit(packed["v_packed"], 3, d)
    k = dequantize_lloydmax(k_codes, packed["k_norm"], packed["k_codebook"], block_size=bs)
    v = dequantize_lloydmax(v_codes, packed["v_norm"], packed["v_codebook"], block_size=bs)
    return k.half(), v.half()


def attn_decode_cached(
    q: torch.Tensor,
    k_i8: torch.Tensor,
    k_scale: torch.Tensor,
    v_i8: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    scale: float | None = None,
    rotate: bool = False,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Decode (M=1) against a pre-quantized INT8 K/V cache (from
    :func:`quantize_kv_cache`).

    Only the single query row is quantized per call (cheap); K and V are already
    int8, so this halves the V read bandwidth (the dominant cost of memory-bound
    decode) and avoids re-quantizing the cache every step. q: fp16 [B, H_q, 1, D].
    Returns fp16 [B, H_q, 1, D]. ``rotate`` MUST match the flag the cache was
    built with in :func:`quantize_kv_cache` (rotates Q to match rotated K).

    ``num_splits``: override the number of KV splits (flash-decoding parallelism).
    ``None`` (default) auto-tunes from the context length and batch×head count.
    """
    from .quant import LOG2E, quantize_int8_rowwise
    from .quant.rotation import rotate_last

    assert q.shape[-2] == 1, f"decode expects M=1, got M={q.shape[-2]}"
    if v_scale.shape[-1] == k_i8.shape[2]:
        raise RuntimeError(
            "attn_decode_cached received a per-token V scale [B,H,N] — the "
            "contiguous decode kernel (attn_int8_decode_kv8) only supports "
            "per-channel V scales [B,H,D]. For V-per-token decode use "
            "attn_paged_decode_cached (repack into a paged cache with an "
            "identity block table, or use quantize_kv_write_paged incrementally)."
        )
    softmax_scale = (1.0 / (q.shape[-1] ** 0.5)) if scale is None else scale
    if rotate:
        q = rotate_last(q)
    # Fuse the eager per-row Q quant (~11 aten kernels) into the single fused kernel
    # (issue #130 Lever 1). q is [total_q, H, D]; the fused op takes [M,K] -> flatten
    # the (token, head) rows, quantize, reshape back. Byte-identical to the eager path.
    q_shape = q.shape  # [total_q, H, 1, D] (decode M=1)
    q_i8, q_scale = quantize_i8_rowwise(q.reshape(-1, q_shape[-1]).contiguous())
    q_i8 = q_i8.reshape(q_shape)
    q_scale = (
        (q_scale.float() * (softmax_scale * LOG2E)).reshape(q_shape[0], q_shape[1], 1).contiguous()
    )
    ns = -1 if num_splits is None else int(num_splits)
    return _C.attn_int8_decode_kv8(
        q_i8.contiguous(),
        q_scale,
        k_i8.contiguous(),
        k_scale.contiguous(),
        v_i8.contiguous(),
        v_scale.contiguous(),
        ns,
    )


def quantize_kv_write_paged(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_cache: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    rotate: bool = True,
) -> None:
    """Quantize-on-write: commit ONE new token's K/V into a paged int8 KV
    cache, in-place, without touching any other row.

    ``k_new, v_new``: fp16 ``[B, H_kv, D]`` — the newest token for every
    sequence in the batch. ``k_cache, v_cache``: int8
    ``[num_blocks, H_kv, block_size, D]``; ``k_scale, v_scale``: fp32
    ``[num_blocks, H_kv, block_size]`` — mutated in place (the persistent
    serving-engine KV pool). ``slot_mapping``: int32 ``[B]``, the flat slot
    ``block_id * block_size + offset`` each sequence's new token lands in
    (computed by the caller from its block table + write position).

    Both K and V use a per-TOKEN symmetric RTN scale here (unlike
    :func:`quantize_kv_cache`'s per-channel V scale, which needs the whole
    cache's amax up front and so cannot be computed one new token at a time).
    ``rotate=True`` (default) applies the Hadamard incoherence rotation to K
    before quantizing — the per-channel-MEAN K-smoothing used elsewhere
    (:func:`~superl8.quant.smooth_k`) also needs the whole cache, so it is not
    available for a true one-token-at-a-time write; the rotation is a fixed
    per-token linear map, so it is the smoothing option compatible with
    quantize-on-write. The query must be rotated identically at decode time
    — see :func:`attn_paged_decode_cached`.
    """
    from .quant.rotation import rotate_last

    if rotate:
        k_new = rotate_last(k_new)
    _C.kv_write_paged(
        k_new.contiguous(),
        v_new.contiguous(),
        slot_mapping.contiguous(),
        k_cache,
        k_scale,
        v_cache,
        v_scale,
    )


def attn_paged_decode_cached(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_cache: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    *,
    max_context_len: int | None = None,
    scale: float | None = None,
    rotate: bool = True,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Decode (M=1) against a paged int8 KV cache, addressed via a per-
    sequence block table instead of a contiguous ``[B,H_kv,N,D]`` slice.

    Lets ONE launch serve a whole batch of sequences with different context
    lengths (:func:`attn_decode_cached` needs the same ``N`` for every
    sequence, so a serving engine batching mixed lengths would otherwise loop
    it once per distinct length). Feed it the cache built incrementally by
    :func:`quantize_kv_write_paged`.

    ``q``: fp16 ``[B, H_q, 1, D]``. ``k_cache, v_cache``: int8
    ``[num_blocks, H_kv, block_size, D]``; ``k_scale, v_scale``: fp32
    ``[num_blocks, H_kv, block_size]`` (per-token scale — see
    :func:`quantize_kv_write_paged`). ``block_table``: int32
    ``[B, max_blocks_per_seq]``, ``block_table[b, i]`` is the physical block
    holding logical positions ``[i*block_size, (i+1)*block_size)`` of sequence
    ``b`` (need not be contiguous or sorted). ``context_lens``: int32 ``[B]``,
    the valid key count per sequence. ``max_context_len`` upper-bounds the
    split sizing and should be ``max(context_lens)`` — a serving engine already
    has this as a plain int before uploading tensors, and passing it avoids a
    device -> host sync; if omitted it is computed here (forces a sync,
    correctness-path fallback only). ``rotate`` MUST match the flag
    :func:`quantize_kv_write_paged` built the cache with. ``num_splits``:
    override the number of KV splits; ``None`` = auto. Returns fp16
    ``[B, H_q, 1, D]``.
    """
    from .quant import LOG2E, quantize_int8_rowwise
    from .quant.rotation import rotate_last

    assert q.shape[-2] == 1, f"paged decode expects M=1, got M={q.shape[-2]}"
    if max_context_len is None:
        max_context_len = int(context_lens.max().item())
    softmax_scale = (1.0 / (q.shape[-1] ** 0.5)) if scale is None else scale
    if rotate:
        q = rotate_last(q)
    # Fuse the eager per-row Q quant (~11 aten kernels) into the single fused kernel
    # (issue #130 Lever 1). q is [total_q, H, D]; the fused op takes [M,K] -> flatten
    # the (token, head) rows, quantize, reshape back. Byte-identical to the eager path.
    q_shape = q.shape  # [total_q, H, 1, D] (decode M=1)
    q_i8, q_scale = quantize_i8_rowwise(q.reshape(-1, q_shape[-1]).contiguous())
    q_i8 = q_i8.reshape(q_shape)
    q_scale = (
        (q_scale.float() * (softmax_scale * LOG2E)).reshape(q_shape[0], q_shape[1], 1).contiguous()
    )
    ns = -1 if num_splits is None else int(num_splits)
    return _C.attn_paged_decode(
        q_i8.contiguous(),
        q_scale,
        k_cache.contiguous(),
        k_scale.contiguous(),
        v_cache.contiguous(),
        v_scale.contiguous(),
        block_table.contiguous(),
        context_lens.contiguous(),
        int(block_size),
        int(max_context_len),
        ns,
    )


def attn_paged_decode_k8v3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_packed: torch.Tensor,
    v_norm: torch.Tensor,
    v_codebook: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    *,
    max_context_len: int | None = None,
    scale: float | None = None,
    rotate: bool = True,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Decode (M=1) against a paged **K8V3** cache (superl8#295): int8 rotated K
    with per-token scale + 3-bit Lloyd-Max packed V, both consumed inside the
    paged split-KV kernel — no fp16 V materialization, no K re-quantization,
    one batched launch for a whole batch of mixed-length sequences.

    This is the fused replacement for the K8V3 dense fallback (fni8-serve
    ``_decode_lloydmax3``: per-slot Python loop + chunked dequant + fp16 V).
    Consumes the store fni8-serve's ``_write_k_int8``/``_write_v_lloydmax3``
    write: ``k_cache`` int8 ``[num_blocks, H_kv, block_size, D]``, ``k_scale``
    fp32 ``[num_blocks, H_kv, block_size]`` (per-token), ``v_packed`` int32
    ``[num_blocks, H_kv, block_size, D*3//32]`` (3-bit codes, LSB-first),
    ``v_norm`` fp32 ``[num_blocks, H_kv, block_size, D//128]`` (per-128-block
    L2 norms), ``v_codebook`` fp32 ``[8]`` (the layer's fixed Gaussian
    codebook row). Head dim must be 128 or 256.

    ``q``: fp16 ``[B, H_q, 1, D]``. ``block_table``: int32
    ``[B, max_blocks_per_seq]``; ``context_lens``: int32 ``[B]``; contract and
    ``max_context_len``/``num_splits`` semantics as in
    :func:`attn_paged_decode_cached`. ``rotate`` MUST match the flag the K
    cache was written with. Returns fp16 ``[B, H_q, 1, D]``.
    """
    from .quant import LOG2E
    from .quant.rotation import rotate_last

    assert q.shape[-2] == 1, f"paged decode expects M=1, got M={q.shape[-2]}"
    if max_context_len is None:
        max_context_len = int(context_lens.max().item())
    softmax_scale = (1.0 / (q.shape[-1] ** 0.5)) if scale is None else scale
    if rotate:
        q = rotate_last(q)
    q_shape = q.shape  # [total_q, H, 1, D] (decode M=1)
    q_i8, q_scale = quantize_i8_rowwise(q.reshape(-1, q_shape[-1]).contiguous())
    q_i8 = q_i8.reshape(q_shape)
    q_scale = (
        (q_scale.float() * (softmax_scale * LOG2E)).reshape(q_shape[0], q_shape[1], 1).contiguous()
    )
    ns = -1 if num_splits is None else int(num_splits)
    return _C.attn_paged_decode_k8v3(
        q_i8.contiguous(),
        q_scale,
        k_cache.contiguous(),
        k_scale.contiguous(),
        v_packed.contiguous(),
        v_norm.contiguous(),
        v_codebook.contiguous(),
        block_table.contiguous(),
        context_lens.contiguous(),
        int(block_size),
        int(max_context_len),
        ns,
    )


# Decode-shape M cap routed to `gemm_decode_w8a8` (split-K, issue #27) instead
# of the prefill tile GEMM (`gemm_w8a8`). MUST match DEC_MAX_M in
# csrc/include/gemm_decode_dp4a.cuh — gemm_decode_w8a8 rejects M above this.
_DECODE_MAX_M = 16

# w8a8 crossover between the per-row decode GEMV and the tile GEMM (fni8-serve#479).
# Measured on a V100 with Qwen3.5-9B decode (graphed decode pads batches to 1/2/4/8/16 rows):
# at M=8 the decode GEMV is faster (75.5 vs 65.4 tok/s aggregate), at M=16 the tile GEMM is
# 2.1x faster (123.1 vs 57.6 tok/s; the decode GEMV's cost grows per row). Both compute
# byte-identical integer math. Must stay <= _DECODE_MAX_M.
_W8A8_DECODE_MAX_M = 8

# Fused fp16-in decode GEMV (issue #130): quantizes the fp16 activation to int8
# in the GEMV prologue (shared memory), dropping the standalone
# `quantize_i8_rowwise` launch AND the int8-activation HBM round-trip. The C++
# op (`gemm_decode_w8a8_fp16in`, csrc/kernel/gemm_decode_dp4a.cu) only takes the
# fused path when split_k==1 and the per-row int8 fits dynamic smem; these two
# Python-side constants MUST mirror its C++ guards so we can decide eligibility
# WITHOUT provoking a RuntimeError (a throw is fatal under CUDA-graph capture,
# the graphed-decode hot path). See DEC_TARGET_BLOCKS / DEC_SMEM_CAP there.
_DECODE_SPLITK1_MIN_N = 320  # DEC_TARGET_BLOCKS: N>=this ⇒ compute_split_k()==1
_DECODE_SMEM_CAP = 98304  # DEC_SMEM_CAP: 96 KB Volta dynamic-smem cap

# Kill-switch for the fused decode quant (issue #130). Default on; set
# FNI8_FUSE_DECODE_QUANT=0 to force the classic two-step quantize + GEMV (for
# A/B measurement or as a rollback). Read once at import — the decode hot path
# (and CUDA-graph capture) must not touch os.environ per call.
_FUSE_DECODE_QUANT = os.environ.get("FNI8_FUSE_DECODE_QUANT", "1") != "0"


def _fused_fp16in_eligible(m: int, n: int, k: int, dtype: torch.dtype) -> bool:
    """Whether the decode linear may use the fused fp16-in GEMV instead of the
    two-step ``quantize_i8_rowwise`` + ``gemm_decode_w8a8``.

    Mirrors the C++ op's accept conditions exactly so an ineligible shape takes
    the fallback path silently (no exception) — required because the graphed
    decode step cannot tolerate a throw mid-capture. Ineligible ⇒ the classic
    two-step path, which is numerically identical.
    """
    if not _FUSE_DECODE_QUANT:  # kill-switch (FNI8_FUSE_DECODE_QUANT=0)
        return False
    if dtype != torch.float16:  # the C++ op requires fp16 x (bf16 stays two-step)
        return False
    if m <= 0 or m > _W8A8_DECODE_MAX_M:
        return False
    if k % 4 != 0:  # dp4a int32 loads need K%4==0
        return False
    if n < _DECODE_SPLITK1_MIN_N:  # guarantees compute_split_k()==1
        return False
    # smem = round_up(M*K, 16) int8 bytes for the quantized activation + M fp32 scales
    smem_bytes = ((m * k + 15) & ~15) + m * 4
    return smem_bytes <= _DECODE_SMEM_CAP


def quantize_i8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused per-row symmetric-RTN int8 quantizer: ``x[M,K]`` (fp16/bf16/fp32) ->
    ``(q int8 [M,K], scale fp32 [M])`` in ONE kernel launch.

    Byte-identical to the eager :func:`superl8.quant.core.quantize_int8_rowwise`
    prologue (symmetric RTN, ``Q_MAX=127``, ``rintf`` round-half-to-even), but
    collapses its ~11 aten ops + 3 dtype conversions into a single launch — the
    dominant per-linear dispatch cost profiled in the decode step. On CPU (no
    kernel) it falls back to the pure-torch reference, squeezing the scale to
    ``[M]`` to match the fused op's shape. Input is made contiguous by the caller.
    """
    if x.is_cuda:
        return _C.quantize_i8_rowwise(x)
    from .quant import quantize_int8_rowwise

    q, scale = quantize_int8_rowwise(x)
    return q, scale.squeeze(-1)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    residual: torch.Tensor | None = None,
    unit_offset: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm in one launch. ``x[..., D]``, ``weight[D]`` (fp16/bf16).

    Without ``residual``: returns ``normed``. With ``residual``: adds it first and
    returns ``(normed, x + residual)`` (nano-vllm fused-residual convention). The
    fp32 sum-of-squares reduction is byte-identical in spirit to the eager
    ``float()/pow/mean/rsqrt/mul`` chain (mean is an order-dependent float sum, so
    cos ≈ 1 vs the torch reference, not bit-exact). ``unit_offset`` scales by
    ``(1 + weight)`` (Gemma). CPU / non-fp16 falls back to pure torch.
    """
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        normed, xr = _C.rmsnorm(x, weight, float(eps), residual, unit_offset)
        return (normed, xr) if residual is not None else normed
    # CPU / dtype fallback == superl8serve RMSNorm._norm
    inp = x if residual is None else x + residual
    dt = inp.dtype
    xf = inp.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    w = (1.0 + weight.float()) if unit_offset else weight.float()
    normed = (xf * w).to(dt)
    return (normed, inp) if residual is not None else normed


def rope(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RoPE in one launch per tensor: rotate ``q[...,Hq,D]`` and
    ``k[...,Hk,D]`` (fp16/bf16) in place using fp32 ``cos``/``sin[max_pos,
    rotary_dim]`` tables gathered by ``positions`` (int64). Dims ``>= rotary_dim``
    pass through (partial rotary). Returns ``(q, k)``. cos/sin computed in fp32
    internally (more accurate than an fp16-cast table)."""
    return _C.rope(positions, q, k, cos, sin, int(rotary_dim))


def act_and_mul(x: torch.Tensor, kind: str = "silu") -> torch.Tensor:
    """Fused gated activation on a merged gate_up projection output, one launch:
    ``x[..., 2I]`` (fp16/bf16) -> ``[..., I]`` = ``act(gate) * up``. ``kind`` is
    ``"silu"`` (SwiGLU, Qwen/Llama) or ``"gelu_tanh"`` (Gemma GeGLU). Activation is
    computed in fp32. CPU / non-fp16 falls back to pure torch."""
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        return _C.act_and_mul(x, kind)
    import torch.nn.functional as F

    gate, up = x.chunk(2, dim=-1)
    act = F.silu(gate) if kind == "silu" else F.gelu(gate, approximate="tanh")
    return act * up


def linear_w8a8(
    x: torch.Tensor,
    w_i8: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """W8A8 int8 dp4a linear layer: ``y = (quant(x) @ w_i8^T) * scales (+ bias)``.

    The activation ``x`` (fp16, ``[..., K]``) is quantized per-row to int8. For
    eligible **decode** shapes (fp16 ``x``, ``M <= 16``, ``N >= 320`` so the
    split-K collapses to one, and ``M*K`` int8 within dynamic smem) the quant is
    **fused into the GEMV prologue** (:func:`~superl8._C.gemm_decode_w8a8_fp16in`,
    issue #130) — one fewer kernel launch and one fewer int8-activation HBM
    round-trip per linear. Measured (real V100): ~5x on the isolated decode
    linear and ~3% on eager single-stream end-to-end, where kernel-launch cost
    dominates; under CUDA-graph decode (which already amortizes launch overhead)
    it is within run-to-run noise, since the activation round-trip it removes is
    tiny next to weight streaming. All other shapes (bf16, prefill,
    tiny-``N``) take the classic two-step ``quantize_i8_rowwise`` +
    ``gemm_decode_w8a8``/``gemm_w8a8``, which is numerically identical.
    ``w_i8`` (int8 ``[N, K]``,
    ``K % 4 == 0``) and ``w_scale`` (fp32 ``[N]``, one scale per output channel)
    are the resident `.superl8` ``per_row_i8`` weight, consumed byte-identically by
    the dp4a kernel. Leading dims of ``x`` are flattened and restored. Returns
    ``out_dtype`` (``float16`` default, or ``bfloat16``) ``[..., N]``. On this
    fleet dp4a (46 TOP/s) is the fast path — the fp16 tensor cores are
    firmware-gimped. ``out_dtype=torch.bfloat16`` avoids the fp16 overflow (max
    65504) that black-images bf16-native models (Gemma, most diffusion DiTs).

    Decode-shape rows (``M <= 16`` after flattening, e.g. one autoregressive
    step) route to :func:`~superl8._C.gemm_decode_w8a8` (split-K GEMV, issue #27)
    instead of the prefill tile GEMM: at small M the tile GEMM launches far too
    few threadblocks to fill the GPU (~20% occupancy, ~1.4% of HBM bandwidth
    measured), whereas the decode kernel puts one warp per output column (plus
    K-splitting when N itself is small) so all SMs stay busy. Both kernels
    compute byte-identical integer math, so this routing is invisible to the
    caller other than latency.
    """
    *lead, K = x.shape
    x2 = x.reshape(-1, K).contiguous()
    M, N = x2.shape[0], w_i8.shape[0]
    w_i8 = w_i8.contiguous()
    w_scale = w_scale.contiguous()
    if x2.is_cuda and _fused_fp16in_eligible(M, N, K, x2.dtype):
        # Fused decode GEMV: activation quantized in the GEMV prologue — one
        # fewer launch and one fewer int8-activation HBM round-trip per linear
        # (issue #130). Byte-identical rowwise quant + dp4a as the two-step path
        # (tests/test_gemm_decode.py::test_gemm_decode_fp16in_matches_quantized).
        y = _C.gemm_decode_w8a8_fp16in(x2, w_i8, w_scale, out_dtype)
    else:
        x_i8, x_scale = quantize_i8_rowwise(x2)  # [M,K] int8, [M] fp32 (one launch)
        gemm_fn = _C.gemm_decode_w8a8 if M <= _W8A8_DECODE_MAX_M else _C.gemm_w8a8
        y = gemm_fn(x_i8, x_scale, w_i8, w_scale, out_dtype)  # [M,N] out_dtype
    if bias is not None:
        y = (y.float() + bias.float()).to(y.dtype)
    return y.reshape(*lead, y.shape[-1])


def grouped_linear_w8a8(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    w_i8: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """MoE grouped/batched W8A8 linear: every active expert's GEMM in ONE dp4a
    kernel launch, replacing a Python loop over :func:`linear_w8a8` per expert.

    ``x``: fp16 ``[M, K]`` tokens in arbitrary (routing) order. ``expert_ids``:
    int64 ``[M]``, the expert each token was routed to (``0 <= id < E``).
    ``w_i8``: int8 ``[E, N, K]`` stacked per-row-quantized expert weights;
    ``w_scale``: fp32 ``[E, N]``. ``bias`` (optional): fp16/fp32 ``[E, N]``, one
    bias row per expert.

    Tokens are gathered into expert-contiguous order (a stable argsort, so
    ties keep their relative order), the grouped dp4a kernel runs the whole
    batch in one launch (gather by expert, segment the M dimension), and the
    result is scattered back to the input token order. Returns fp16 ``[M, N]``.
    """
    from .quant import quantize_int8_rowwise

    E = w_i8.shape[0]
    order = torch.argsort(expert_ids, stable=True)
    x_i8, x_scale = quantize_int8_rowwise(x[order].contiguous())
    group_sizes = torch.bincount(expert_ids, minlength=E).to(torch.int64)
    y_sorted = _C.gemm_grouped_w8a8(
        x_i8,
        x_scale.squeeze(-1).contiguous(),
        w_i8.contiguous(),
        w_scale.contiguous(),
        group_sizes.cpu(),
    )
    if bias is not None:
        y_sorted = (y_sorted.float() + bias[expert_ids[order]].float()).to(y_sorted.dtype)
    y = torch.empty_like(y_sorted)
    y[order] = y_sorted
    return y


def linear_w4a8(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    group_size: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """W4A8 int8 dp4a linear layer with 4-bit weights.

    ``w_packed`` (uint8 ``[N, K//2]``, 2 signed int4 nibbles/byte, even col = low
    nibble) and ``w_scale`` (fp32 ``[N, K//group_size]``, per output channel per
    group) are the resident `.superl8` ``per_group_i4`` (codebook ``int4``) blob.
    sm_70 has no int4 matmul, so the kernel unpacks nibbles -> int8 and runs the
    same dp4a as W8A8; 4-bit halves weight footprint and read bandwidth (the win
    at memory-bound decode). ``x`` fp16 ``[..., K]`` is quantized per-row in a
    torch prologue. Returns ``out_dtype`` (``float16`` default, or ``bfloat16``)
    ``[..., N]``.
    """
    from .quant import quantize_int8_rowwise

    *lead, K = x.shape
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    w_packed = w_packed.contiguous()
    w_scale = w_scale.contiguous()
    if x2.is_cuda and M <= _DECODE_MAX_M and hasattr(_C, "gemm_decode_w4a8"):
        # Decode-shape (M<=16): warp-per-column W4A8 decode kernel (issue #173) with
        # the fused rowwise quant. The tile gemm_w4a8 collapses to ceil(N/64) blocks
        # at M=1 (~1% HBM); this fills the GPU. Byte-compatible int math.
        x_i8, x_scale = quantize_i8_rowwise(x2)  # [M,K] int8, [M] fp32 (fused, one launch)
        y = _C.gemm_decode_w4a8(x_i8, x_scale, w_packed, w_scale, int(group_size), out_dtype)
    else:
        x_i8, x_scale = quantize_int8_rowwise(x2)
        y = _C.gemm_w4a8(
            x_i8,
            x_scale.squeeze(-1).contiguous(),
            w_packed,
            w_scale,
            int(group_size),
            out_dtype,
        )
    if bias is not None:
        y = (y.float() + bias.float()).to(y.dtype)
    return y.reshape(*lead, y.shape[-1])


def linear_w3a8(
    x: torch.Tensor,
    w_planes: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    group_size: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """W3A8 int8 dp4a linear with uniform 3-bit weights (VRAM/context lever).

    ``w_planes`` (int32 ``[N, (K//32)*3]`` Q3_K-style bit-planes, see
    :func:`superl8.quant.pack_w3a8_bitplanes`) and ``w_scale`` (fp32
    ``[N, K//group_size]``, per output channel per group) are the resident 3-bit
    MLP weight blob. sm_70 has no int3 matmul, so the kernel unpacks each 32-value
    group to int8 and runs the same dp4a as W8A8/W4A8. 3.0 bpw frees ~0.75x the
    W4A8 weight bytes — this is a **VRAM lever** (longer KV context / more batch
    slots before OOM), decode is measured at **PARITY** with W4A8, not faster
    (the sub-byte unpack ALU contends with dp4a on the INT pipe; issue #181).

    ``x`` fp16/bf16 ``[..., K]`` is quantized per-row (max/127). Decode shapes
    (``M<=16``) run the ``gemm_decode_w3a8`` warp-per-column kernel; larger ``M``
    (prefill) falls back to a dequantize-to-fp matmul (no 3-bit tile GEMM exists —
    3-bit is a decode-time VRAM option, not a prefill compute path).
    """
    *lead, K = x.shape
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    w_planes = w_planes.contiguous()
    w_scale = w_scale.contiguous()
    if x2.is_cuda and M <= _DECODE_MAX_M and hasattr(_C, "gemm_decode_w3a8"):
        # Decode-shape (M<=16): warp-per-column W3A8 decode kernel (issue #181).
        x_i8, x_scale = quantize_i8_rowwise(x2)  # [M,K] int8, [M] fp32 (fused, one launch)
        y = _C.gemm_decode_w3a8(x_i8, x_scale, w_planes, w_scale, int(group_size), out_dtype)
    else:
        # Prefill / CPU fallback: dequantize the 3-bit weights to fp and matmul.
        from .quant.lowbit import unpack_w3a8_bitplanes  # local import to avoid cycle

        N = w_planes.shape[0]
        codes = unpack_w3a8_bitplanes(w_planes, K)  # [N,K] int8 in [-4,3]
        wf = (
            codes.float().reshape(N, K // group_size, group_size)
            * w_scale.float().reshape(N, K // group_size, 1)
        ).reshape(N, K)
        y = (x2.float() @ wf.t().to(x2.device)).to(out_dtype)
    if bias is not None:
        y = (y.float() + bias.float()).to(y.dtype)
    return y.reshape(*lead, y.shape[-1])


def linear_q4k(
    x: torch.Tensor,
    w_bytes: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Fused GGUF **Q4_K** dp4a linear — native-GGUF-on-the-fly (NO conversion).

    ``w_bytes`` (uint8 ``[N, (K//256)*144]``) is the weight in its **native GGUF
    Q4_K super-block layout** (144 bytes / 256 weights: fp16 ``d``/``dmin`` +
    12 packed 6-bit sub-scale/min bytes + 128 qs nibble bytes). The kernel keeps it
    resident and unpacks each 32-element sub-block to int8 **in-kernel**, running
    ``__dp4a`` while honoring the 6-bit per-sub-block scale/min exactly — the
    ~4.5-bit k-quant stays resident (VRAM parity, fits one card) and the dequant is
    fused into the matmul, never materialized (the 143 s/step per-forward fp32
    dequant path is what this replaces). See csrc/docs/gguf-fused-kquant-dp4a.md.

    ``x`` (fp16/bf16 ``[..., K]``, ``K % 256 == 0``) is quantized per-row to int8.
    **Decode shapes (M<=16) route to the warp-per-column MMVQ decode kernel**
    (:func:`~superl8._C.gemm_decode_q4k`); larger M uses the prefill tile GEMM. At M=1
    the tile launches only ceil(N/64) blocks and leaves the GPU idle (native decode
    measured 6.7x slower than dequant->int8 in the join); the decode kernel puts one
    warp per output column so all SMs stay busy — and since the GEMV is memory-bound,
    native (~1.78x fewer resident bytes than int8) then wins. Both compute the same
    per-sub-block math. Returns ``out_dtype`` (``float16`` default, or ``bfloat16``)
    ``[..., N]``. Adapts llama.cpp's MMQ+MMVQ / vec_dot_q4_K (MIT).
    """
    return _linear_kquant(x, w_bytes, bias, out_dtype, _C.gemm_q4k,
                          getattr(_C, "gemm_decode_q4k", None))


# Decode-shape M cap: at/below this, the k-quant linear routes to the warp-per-
# column MMVQ decode kernel (fills the GPU at M=1); above it, the prefill tile.
# Mirrors DEC_MAX_M in gemm_decode_dp4a.cuh (which rejects M above it).
_KQUANT_DECODE_MAX_M = 16

# TQ3_4S decode smem cap: M*K int8 (rotated+quantized activation) + M*(K/32) fp32
# scales + 128 B (uint16 pair LUT) in dynamic smem. MUST mirror TQ3_DEC_SMEM_CAP
# in csrc/kernel/gemm_decode_dp4a.cu so Python routes ineligible shapes to the tile
# kernel instead of provoking a RuntimeError.
_TQ3_DEC_SMEM_CAP = 98304


def _tq34s_decode_eligible(m: int, k: int) -> bool:
    """Whether the fused warp-per-column TQ3_4S decode kernel can serve (M,K)."""
    if not (0 < m <= _KQUANT_DECODE_MAX_M):
        return False
    if k % 32 != 0:
        return False
    smem = m * k + m * (k // 32) * 4 + 64 * 2
    return smem <= _TQ3_DEC_SMEM_CAP


def _linear_kquant(x, w_bytes, bias, out_dtype, tile_op, decode_op):
    """Shared body for the fused GGUF k-quant linears (Q4_K/Q5_K/Q6_K): per-row
    int8-quantize the activation, run the fused native-k-quant dp4a kernel, add bias.
    Routes decode shapes (M<=16) to the warp-per-column ``decode_op`` (GPU-saturating
    at M=1) and prefill to the ``tile_op``; both are numerically identical. The
    weight stays resident in its native GGUF super-block layout."""
    *lead, K = x.shape
    x2 = x.reshape(-1, K).contiguous()
    w_bytes = w_bytes.contiguous()
    x_i8, x_scale = quantize_i8_rowwise(x2)
    M = x2.shape[0]
    if decode_op is not None and x2.is_cuda and 0 < M <= _KQUANT_DECODE_MAX_M:
        y = decode_op(x_i8, x_scale, w_bytes, out_dtype)
    else:
        y = tile_op(x_i8, x_scale, w_bytes, out_dtype)
    if bias is not None:
        y = (y.float() + bias.float()).to(y.dtype)
    return y.reshape(*lead, y.shape[-1])


def linear_q5k(x, w_bytes, *, bias=None, out_dtype: torch.dtype = torch.float16):
    """Fused GGUF **Q5_K** dp4a linear (native, no conversion). ``w_bytes`` uint8
    ``[N, (K//256)*176]`` native Q5_K super-blocks (d/dmin + 6-bit sub-scales/mins
    + qh high-bits + qs nibbles). Same affine + min-correction as Q4_K plus the 5th
    bit — ~5.5 bpw, the mid-precision tensors in a Q4_K_M mix. Decode shapes (M<=16)
    route to the warp-per-column MMVQ decode kernel. See
    csrc/docs/gguf-fused-kquant-dp4a.md."""
    return _linear_kquant(x, w_bytes, bias, out_dtype, _C.gemm_q5k,
                          getattr(_C, "gemm_decode_q5k", None))


def linear_q6k(x, w_bytes, *, bias=None, out_dtype: torch.dtype = torch.float16):
    """Fused GGUF **Q6_K** dp4a linear (native, no conversion). ``w_bytes`` uint8
    ``[N, (K//256)*210]`` native Q6_K super-blocks (ql+qh 6-bit codes, signed int8
    per-16 scales, super-block d). SYMMETRIC (centered -32, no min term) — the
    OUTLIER-tensor precision a Q4_K_M mix assigns to sensitive weights (e.g. LTX
    ``to_v``). Decode shapes (M<=16) route to the warp-per-column MMVQ decode
    kernel. See csrc/docs/gguf-fused-kquant-dp4a.md."""
    return _linear_kquant(x, w_bytes, bias, out_dtype, _C.gemm_q6k,
                          getattr(_C, "gemm_decode_q6k", None))


# codebook -> fused linear. MUST cover every codebook in `superl8.gguf._NATIVE_FUSED`
# except tq3_4s, which `linear()` special-cases (it needs in_features from TQ3_TYPE_SIZE).
# Module-level, not a local inside `linear()`, so `test_linear_dispatch_covers_native`
# can assert the coverage: this table replaced a hand-written if-chain in #328 and then
# went stale anyway when #330 added four kernels, raising
# "gguf_kquant type 'iq1_s' has no fused kernel" on a weight whose kernel shipped.
_FUSED_KQUANT_OPS = {}


def _register_fused_kquant_ops():
    """Populated after the linear_* functions are defined (they are referenced by name)."""
    _FUSED_KQUANT_OPS.update({
        "q2_k": linear_q2k, "q3_k": linear_q3k, "q4_k": linear_q4k,
        "q5_k": linear_q5k, "q6_k": linear_q6k,
        "iq3_s": linear_iq3s, "iq4_xs": linear_iq4xs, "iq3_xxs": linear_iq3xxs,
        "iq2_s": linear_iq2s, "iq2_xs": linear_iq2xs, "iq2_xxs": linear_iq2xxs,
        "iq1_s": linear_iq1s,
    })


def linear_q3k(x, w_bytes, *, bias=None, out_dtype: torch.dtype = torch.float16):
    """Fused GGUF **Q3_K** dp4a linear (native, no conversion). ``w_bytes`` uint8
    ``[N, (K//256)*110]`` native Q3_K super-blocks (hmask + qs 2-bit + 6-bit signed
    per-16 scales + d). SYMMETRIC 3-bit (centered -4). ~3.4 bpw — keeping it RESIDENT
    is what fits **Qwen3.6-27B-Q3_K_S** (353 Q3_K tensors) on one 16 GB card (the
    requant->int8 fallback would blow those to ~20 GB). Decode shapes (M<=16) route
    to the warp-per-column MMVQ kernel. See csrc/docs/gguf-fused-kquant-dp4a.md."""
    return _linear_kquant(x, w_bytes, bias, out_dtype, _C.gemm_q3k,
                          getattr(_C, "gemm_decode_q3k", None))


def linear_iq3xxs(x, w_bytes, *, bias=None, out_dtype: torch.dtype = torch.float16):
    """Fused GGUF **IQ3_XXS** dp4a linear (native, no conversion). ``w_bytes`` uint8
    ``[N, (K//256)*98]``: 8-bit indices into a 256-entry grid, a top-nibble half-offset
    scale ``d*(0.5+s)*0.5``, and sign bits fetched through ``ksigns_iq2xs``. ~3.06 bpw —
    2.80 GiB reclaimed on Qwen3.8-27B-UD-IQ3_S, the step that brings it under a 16 GiB
    card (superl8#317).

    Decode shapes (M <= 16) route to the warp-per-column MMVQ kernel; larger M to
    the blocked tile kernel. Both are numerically identical and fully on-GPU.
    """
    return _linear_kquant(x, w_bytes, bias, out_dtype, _C.gemm_iq3xxs,
                          getattr(_C, "gemm_decode_iq3xxs", None))

def linear_iq4xs(x, w_bytes, *, bias=None, out_dtype: torch.dtype = torch.float16):
    """Fused GGUF **IQ4_XS** dp4a linear (native, no conversion). ``w_bytes`` uint8
    ``[N, (K//256)*136]`` native IQ4_XS blocks: 4-bit indices into the 16-entry signed
    ``kvalues_iq4nl`` codebook, with a per-32 scale split across ``scales_l``/``scales_h``
    and applied as ``d * (ls - 32)``. ~4.25 bpw — 2.13 GiB reclaimed on
    Qwen3.8-27B-UD-IQ3_S versus the requant fallback (superl8#317).

    Decode shapes (M <= 16) route to the warp-per-column MMVQ kernel; larger M to
    the blocked tile kernel. Both are numerically identical and fully on-GPU.
    """
    return _linear_kquant(x, w_bytes, bias, out_dtype, _C.gemm_iq4xs,
                          getattr(_C, "gemm_decode_iq4xs", None))

def linear_iq2s(x, w_bytes, *, bias=None, out_dtype: torch.dtype = torch.float16):
    """Fused GGUF **IQ2_S** dp4a linear (native, no conversion). ``w_bytes`` uint8
    ``[N, (K//256)*82]``: a 10-bit index (8 bits ``qs`` + 2 of ``qh``) into ggml's
    1024-entry codebook, EXPLICIT per-weight sign bits (unlike IQ2_XS's sign LUT),
    and a 4-bit scale per SIXTEEN weights used as ``d*(0.5+s)*0.25``. ~2.5 bpw -- one of the four types whose requant to int8
    is what keeps **Qwen3.8-27B-UD-IQ3_S** off a 16 GiB card (superl8#317).

    Decode shapes (M <= 16) route to the warp-per-column MMVQ kernel; larger M to the
    blocked tile kernel (superl8#332). Both are numerically identical and fully on-GPU --
    prefill no longer dequantizes the whole weight matrix on CPU, which cost ~846 ms per
    4096x4096 call and made a 512-token prefill take 211 s on the real 27B."""
    return _linear_kquant(x, w_bytes, bias, out_dtype, _C.gemm_iq2s,
                          getattr(_C, "gemm_decode_iq2s", None))


def linear_iq2xs(x, w_bytes, *, bias=None, out_dtype: torch.dtype = torch.float16):
    """Fused GGUF **IQ2_XS** dp4a linear (native, no conversion). ``w_bytes`` uint8
    ``[N, (K//256)*74]``: each uint16 carries a 9-bit index into a 512-entry grid
    plus a 7-bit index into the 128-entry ``ksigns_iq2xs`` LUT; 4-bit scales at
    **16-weight** granularity (the 8 scale bytes are 16 nibbles). ~2.31 bpw -- one of the four types whose requant to int8
    is what keeps **Qwen3.8-27B-UD-IQ3_S** off a 16 GiB card (superl8#317).

    Decode shapes (M <= 16) route to the warp-per-column MMVQ kernel; larger M to the
    blocked tile kernel (superl8#332). Both are numerically identical and fully on-GPU --
    prefill no longer dequantizes the whole weight matrix on CPU, which cost ~846 ms per
    4096x4096 call and made a 512-token prefill take 211 s on the real 27B."""
    return _linear_kquant(x, w_bytes, bias, out_dtype, _C.gemm_iq2xs,
                          getattr(_C, "gemm_decode_iq2xs", None))


def linear_iq2xxs(x, w_bytes, *, bias=None, out_dtype: torch.dtype = torch.float16):
    """Fused GGUF **IQ2_XXS** dp4a linear (native, no conversion). ``w_bytes`` uint8
    ``[N, (K//256)*66]``: four 8-bit indices into a 256-entry grid per 32 weights,
    a top-nibble scale ``d*(0.5+s)*0.25``, and four 7-bit ``ksigns_iq2xs``
    indices. ~2.06 bpw -- one of the four types whose requant to int8
    is what keeps **Qwen3.8-27B-UD-IQ3_S** off a 16 GiB card (superl8#317).

    Decode shapes (M <= 16) route to the warp-per-column MMVQ kernel; larger M to the
    blocked tile kernel (superl8#332). Both are numerically identical and fully on-GPU --
    prefill no longer dequantizes the whole weight matrix on CPU, which cost ~846 ms per
    4096x4096 call and made a 512-token prefill take 211 s on the real 27B."""
    return _linear_kquant(x, w_bytes, bias, out_dtype, _C.gemm_iq2xxs,
                          getattr(_C, "gemm_decode_iq2xxs", None))


def linear_iq1s(x, w_bytes, *, bias=None, out_dtype: torch.dtype = torch.float16):
    """Fused GGUF **IQ1_S** dp4a linear (native, no conversion). ``w_bytes`` uint8
    ``[N, (K//256)*50]``: an 11-bit index (8 bits ``qs`` + 3 of ``qh``) into a
    2048-entry TERNARY codebook, a per-32 scale ``2*((qh>>12)&7)+1`` and a per-block
    ``+/-0.125`` delta -- there are NO sign bits. The delta makes the effective
    weight non-integer, so the kernel stores ``8*grid`` and uses the identity
    ``8*(g+delta) == 8g+/-1``, folding the 1/8 into the scale. ~1.56 bpw -- one of the four types whose requant to int8
    is what keeps **Qwen3.8-27B-UD-IQ3_S** off a 16 GiB card (superl8#317).

    Decode shapes (M <= 16) route to the warp-per-column MMVQ kernel; larger M to the
    blocked tile kernel (superl8#332). Both are numerically identical and fully on-GPU --
    prefill no longer dequantizes the whole weight matrix on CPU, which cost ~846 ms per
    4096x4096 call and made a 512-token prefill take 211 s on the real 27B."""
    return _linear_kquant(x, w_bytes, bias, out_dtype, _C.gemm_iq1s,
                          getattr(_C, "gemm_decode_iq1s", None))


def linear_iq3s(x, w_bytes, *, bias=None, out_dtype: torch.dtype = torch.float16):
    """Fused GGUF **IQ3_S** dp4a linear (native, no conversion). ``w_bytes`` uint8
    ``[N, (K//256)*110]`` native IQ3_S blocks: a 9-bit index into ggml's 512-entry
    codebook grid, one sign bit per weight, and a 4-bit per-32 scale used as 1+2s.
    ~3.44 bpw — keeping it RESIDENT is what fits **Qwen3.8-27B-UD-IQ3_S** (11.20 GiB
    native vs 23.40 GiB requantized to int8) on one 16 GiB card with room for ~100k
    context (superl8#317).

    Decode shapes (M <= 16) route to the warp-per-column MMVQ kernel; larger M to the
    blocked tile kernel. Both are numerically identical and fully on-GPU — prefill no
    longer dequantizes the whole weight matrix on CPU, which is what the ~100k-context
    target needs."""
    return _linear_kquant(x, w_bytes, bias, out_dtype, _C.gemm_iq3s,
                          getattr(_C, "gemm_decode_iq3s", None))

def linear_q2k(x, w_bytes, *, bias=None, out_dtype: torch.dtype = torch.float16):
    """Fused GGUF **Q2_K** dp4a linear (native, no conversion). ``w_bytes`` uint8
    ``[N, (K//256)*84]`` native Q2_K super-blocks (4-bit per-16 scale+min, 2-bit q).
    Affine (like Q4_K) at ~2.6 bpw — the aggressive UD-Q2 bulk. Decode shapes (M<=16)
    route to the warp-per-column MMVQ kernel. See csrc/docs/gguf-fused-kquant-dp4a.md."""
    return _linear_kquant(x, w_bytes, bias, out_dtype, _C.gemm_q2k,
                          getattr(_C, "gemm_decode_q2k", None))


def gather_q3k(ids: torch.Tensor, w_bytes: torch.Tensor, *,
               out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Native GGUF **Q3_K** row-gather dequant (token embeddings). ``w_bytes``
    uint8 ``[N, (K//256)*110]`` native Q3_K super-blocks (hmask + qs 2-bit +
    6-bit signed per-16 scales + d) — the SAME byte contract :func:`linear_q3k`
    takes. ``ids`` int32/int64 of any shape ``[...]`` -> ``[..., K]`` in
    ``out_dtype`` (fp16 default), where row ``i`` is the full dequant of
    ``w_bytes[ids[i]]``.

    An embedding table is a pure GATHER, never a GEMM (the LM head is a separate
    tensor), so there is no dp4a to fuse into and the only thing that matters is
    that the table stays RESIDENT in its native bytes: dequantizing+requantizing
    ``token_embd.weight`` to a ``per_row_i8`` table just to index it costs ~2.3x
    (Qwen3.8-27B-UD-IQ3_S: 0.5088 GiB of Q3_K token_embd -> 1.1850 GiB i8, a
    +0.6762 GiB expansion on a 16 GiB card). One thread block per gathered row;
    the same Q3_K decode as the tile and warp-per-column kernels. Ids outside
    ``[0, N)`` yield a ZERO row (checked in-kernel — never an out-of-bounds read
    of a half-GiB table, and no host sync on the decode path).
    See csrc/docs/gguf-fused-kquant-dp4a.md."""
    out = _C.gather_q3k(ids, w_bytes.contiguous(), out_dtype)
    return out.reshape(*ids.shape, out.shape[-1])


def linear_tq34s(
    x: torch.Tensor,
    u8_bytes: torch.Tensor,
    in_features: int,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """**TQ3_4S** linear — the PRODUCTION path for native TurboQuant (type 46).

    ``u8_bytes`` (uint8 ``[N, (in//32)*16]``) is the weight in its native GGUF
    TQ3_4S layout: 4 E3M5 per-8 scales + 12 packed 3-bit code bytes per 32-value
    block (QK_TQ3_0=32, 16 B/block). On CUDA this routes to the **fused dp4a
    kernel** (superl8#272/#289): the kernel rotates the fp activation with the forward
    RHT per 32-block (signs -> WHT butterfly -> 1/sqrt(32)) BEFORE the per-32
    q8_1 quant and dp4a's against the corrected int8 centroid levels, honoring
    per-8 E3M5 scales. Prefill (M>16) uses the llama.cpp-style quality-gated
    common-scale tile: four per-8 scales are requantized to one Q8 scale per
    32-weight block, reducing fp32 accumulator flushes 4x. Set
    ``FNI8_TQ34S_COMMON_SCALE=0`` to restore the exact per-8 tile. ``in_features``
    is the contraction dim.
    Decode shapes (M<=16) route to the warp-per-column MMVQ decode kernel. On
    CPU (or with no compiled extension) it falls back to the pure-torch
    reference (:func:`superl8.quant.tq34s.reference_linear`), which is the
    dequant-exactly-match-the-fork oracle — bit-identical math, not fast.
    """
    if u8_bytes.is_cuda and hasattr(_C, "gemm_tq34s"):
        *lead, K = x.shape
        x2 = x.reshape(-1, K).contiguous()
        w = u8_bytes.contiguous()
        M = x2.shape[0]
        if (
            M <= _KQUANT_DECODE_MAX_M
            and hasattr(_C, "gemm_decode_tq34s")
            and _tq34s_decode_eligible(M, K)
        ):
            y = _C.gemm_decode_tq34s(x2, w, out_dtype)
        else:
            common = getattr(_C, "gemm_tq34s_common_scale", None)
            use_common = common is not None and os.environ.get(
                "FNI8_TQ34S_COMMON_SCALE", "1"
            ) != "0"
            narrow = getattr(_C, "gemm_tq34s_common_scale_tm4tn4", None)
            use_narrow = (
                use_common
                and narrow is not None
                and K >= w.shape[0]
                and os.environ.get("FNI8_TQ34S_NARROW_ACC", "1") != "0"
            )
            kernel = narrow if use_narrow else (common if use_common else _C.gemm_tq34s)
            y = kernel(x2, w, K, out_dtype)
        if bias is not None:
            y = (y.float() + bias.float()).to(y.dtype)
        return y.reshape(*lead, y.shape[-1])
    from .quant.tq34s import reference_linear

    y = reference_linear(x, u8_bytes, in_features)
    if bias is not None:
        y = y + bias.float()
    return y.to(out_dtype)


def lightning_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track-2 (issue #42) v1: CUDA port of the naive sequential fp32
    MiniMax Lightning (un-gated linear) attention recurrence
    (:func:`tests.reference_lightning.lightning_attn_oracle`).

    S_t = S_{t-1} + v_t (x) k_t,  o_t = S_t @ q_t.

    One CUDA block per (batch, head); the block loops over T sequentially and
    keeps state ``[Dv, Dk]`` in shared memory (v1 supports ``Dv, Dk <= 128``).
    fp32 only — the recurrence is the numerically load-bearing piece, per
    AGENTS.md. This is the on-device ground truth the int8 variant (next PR)
    is validated against, not a fast path itself.

    q, k: [B,H,T,Dk] fp32; v: [B,H,T,Dv] fp32.
    ``initial_state``, if given: [B,H,Dv,Dk] fp32 carry-in state.
    Returns ``(o [B,H,T,Dv] fp32, final_state [B,H,Dv,Dk] fp32)``.
    """
    return _C.lightning_attn_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        initial_state.contiguous() if initial_state is not None else None,
    )


def lightning_attn_int8_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track-2 (issue #42) v2: int8 dp4a chunked Lightning attention.

    Same un-gated linear-attention recurrence as :func:`lightning_attn_fwd`,
    but intra-chunk ``k·q`` dot products are computed via the ``__dp4a``
    int8×4 CUDA-core intrinsic on the Volta INT pipe. Keys and queries are
    quantized per-row symmetric RTN (scale = amax/127, fp32), padded to a
    multiple of 4 for dp4a alignment. The state ``S``, values ``V``, and
    output base ``S@q`` stay fp32 — only pure ``k·q`` dot products go
    through int8.

    Matches the fp32 oracle at int8 tolerance (cos ≥ 0.999, SQNR ≥ 30 dB).

    q, k: [B,H,T,Dk] fp32; v: [B,H,T,Dv] fp32.
    ``initial_state``, if given: [B,H,Dv,Dk] fp32 carry-in state.
    Returns ``(o [B,H,T,Dv] fp32, final_state [B,H,Dv,Dk] fp32)``.
    """
    return _C.lightning_attn_int8_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        initial_state.contiguous() if initial_state is not None else None,
    )


def deltanet_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track-2 (issue #6) v1: CUDA port of the naive sequential fp32
    Gated-DeltaNet recurrence (:func:`tests.reference_linear_attn.gated_delta_rule_oracle`).

    One CUDA block per (batch, head); the block loops over T sequentially and
    keeps state ``[Dv, Dk]`` in shared memory (v1 supports ``Dv, Dk <= 128``).
    fp32 only — the recurrence is the numerically load-bearing piece, per
    AGENTS.md, so this stage does not quantize anything. This is the on-device
    ground truth the v2 (chunked WY/UT), v3 (gated), and v4 (int8 dp4a) forms
    are validated against, not a fast path itself.

    q, k: [B,H,T,Dk] fp32; v: [B,H,T,Dv] fp32; alpha, beta: [B,H,T] fp32 (gate
    ranges are the caller's responsibility, see
    :func:`tests.reference_linear_attn.gates_from_logits`). ``initial_state``,
    if given: [B,H,Dv,Dk] fp32 carry-in state. Returns ``(o [B,H,T,Dv] fp32,
    final_state [B,H,Dv,Dk] fp32)``.
    """
    return _C.deltanet_recurrent_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        alpha.contiguous(),
        beta.contiguous(),
        initial_state.contiguous() if initial_state is not None else None,
    )


def deltanet_recurrent_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-specialized Gated-DeltaNet recurrence — same math as
    :func:`deltanet_recurrent_fwd` (validated against the same fp32 oracle), but
    mapped **one warp per (batch, head, v-row)** with state in registers instead
    of one block per (batch, head) with state in shared memory. At decode B*H is
    tiny (e.g. 16), so the block-per-head launch starves the GPU; the warp-per-row
    launch fills it (B=1,H=16,Dv=128 → 2048 warps). Register state means **zero
    dynamic shared memory**, so the launcher makes no per-call
    ``cudaFuncSetAttribute`` — the reason this variant is **CUDA-graph-capturable**
    and the block-per-head one is not (superl8serve replays it inside graphed decode).

    q, k: [B,H,T,Dk] fp32; v: [B,H,T,Dv] fp32; alpha, beta: [B,H,T] fp32.
    ``initial_state``, if given: [B,H,Dv,Dk] fp32 carry-in state. Returns
    ``(o [B,H,T,Dv] fp32, final_state [B,H,Dv,Dk] fp32)``. Supports Dk,Dv ≤ 128;
    general T, but intended for the L==1 decode step.
    """
    return _C.deltanet_recurrent_decode(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        alpha.contiguous(),
        beta.contiguous(),
        initial_state.contiguous() if initial_state is not None else None,
    )


def deltanet_fused_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dt: torch.Tensor,
    b_logit: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    gain: torch.Tensor,
    *,
    z: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    q_scale: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Gated-DeltaNet **decode** step — one graph-capturable launch that
    absorbs the per-layer glue superl8serve otherwise runs as ~15-19 tiny PyTorch
    ops. In a single kernel it does, all fp32:

    ``q_scale`` (default 1.0; pass ``Dk**-0.5`` for the HF gated-delta-rule readout
    scale) multiplies the **L2-normalised** query inside the kernel — it must be
    applied after the internal L2-norm (which would divide out a pre-scale), so
    the caller cannot fold it into ``q`` beforehand.

    1. **L2-norm** ``q`` and ``k`` per head (the ``_l2norm`` clamp, norm ≥ 1e-6),
    2. **GQA expand** the ``nk`` key/query heads to the ``nv`` value heads by
       ``repeat_interleave`` index math (value head ``hv`` uses key head
       ``hv // (nv//nk)``) — no materialized copy,
    3. **gate** ``g = -softplus(dt + dt_bias) * exp(A_log)`` → ``alpha = exp(g)``
       and **beta** ``= sigmoid(b_logit)``,
    4. the **delta-rule recurrence** ``S = alpha·S + beta·(v − alpha·S·k)⊗k ;
       o = S·q`` (same math as :func:`deltanet_recurrent_decode`), and
    5. the **gated output RMSNorm** ``o = o·rsqrt(mean_v(o²)+eps)·gain·silu(z)``
       (same math as :func:`gated_rmsnorm_decode`; ``z`` optional).

    Byte-for-byte the same fp32 math as the committed ``_l2norm`` + gate +
    ``deltanet_recurrent_decode`` + ``gated_rmsnorm_decode`` op sequence (its
    numeric oracle), so it is a drop-in fast path — the win is launch-count / HBM
    reduction, **not** any precision change. State/recurrence/norm stay fp32
    (numerically load-bearing, never quantized, per AGENTS.md). Block per
    (batch, value-head), register state, static shared memory → no per-call
    ``cudaFuncSetAttribute`` → CUDA-graph-capturable. Adapted from qengine
    (Apache-2.0) ``gdn_recurrent_step_v2`` / ``gdn_fused_recur_rmsg``.

    q, k: ``[B, nk, 1, 128]`` fp32; v: ``[B, nv, 1, 128]`` fp32; dt, b_logit:
    ``[B, nv, 1]`` fp32 (raw gate/beta projection outputs); a_log, dt_bias:
    ``[nv]`` fp32; gain: ``[128]`` fp32; z: ``[B, nv, 1, 128]`` fp32 or ``None``;
    initial_state: ``[B, nv, 128, 128]`` fp32 carry-in or ``None`` (zero init).
    **Decode only (T==1), Dk==Dv==128** — other shapes must use the eager
    op sequence. Returns ``(out [B, nv, 1, 128] post-norm, final_state
    [B, nv, 128, 128])``.
    """
    return _C.deltanet_fused_decode(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        dt.contiguous(),
        b_logit.contiguous(),
        a_log.contiguous(),
        dt_bias.contiguous(),
        gain.contiguous(),
        z.contiguous() if z is not None else None,
        initial_state.contiguous() if initial_state is not None else None,
        q_scale,
        eps,
    )


def causal_conv1d_silu_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    tail: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused causal depthwise conv1d(kernel=K) + SiLU for the L==1 decode token
    shift of a Gated-DeltaNet / short-conv block (vLLM ``causal_conv1d_update``
    analogue). Replaces the eager ``cat``/``conv1d``/slice/``silu`` swarm with
    ONE launch that is also CUDA-graph-capturable (no shared memory, no per-call
    ``cudaFuncSetAttribute``).

    For channel ``w`` the causal window is ``[tail[0,w], .., tail[K-2,w], x[w]]``
    and the output is ``silu(sum_j window[j] * weight[w, j])``; the returned
    ``new_tail`` is ``window[1:]`` (the rolled history to feed the next step).

    x: ``[B, Wc]``; weight: ``[Wc, K]``; tail: ``[B, K-1, Wc]`` (all same dtype,
    fp16/bf16/fp32). Math is fp32 internally; the store dtype follows the input
    (conv is an elementwise pre/epilogue, not the load-bearing recurrence).
    Returns ``(out [B, Wc], new_tail [B, K-1, Wc])``. K <= 8.
    """
    return _C.causal_conv1d_silu_decode(x.contiguous(), weight.contiguous(), tail.contiguous())


def gated_rmsnorm_decode(
    o: torch.Tensor,
    gain: torch.Tensor,
    z: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused gated output RMSNorm for the Gated-DeltaNet decode step — HF
    ``Qwen3_5RMSNormGated`` ("norm BEFORE gate"): a per-head RMS over
    ``head_v_dim`` (the last axis), then the per-head ``gain``, then the ``z``
    gate (SiLU). Replaces the eager ``pow``/``mean``/``rsqrt``/``mul``/``silu``
    swarm (~6 ops) with ONE CUDA-graph-capturable launch (warp-per-row, register
    state, no shared memory).

    ``out = o * rsqrt(mean_d(o^2) + eps) * gain * silu(z)`` (``z`` optional).

    o: ``[..., vd]`` fp32 (leading dims are B, nv, ...); gain: ``[vd]`` fp32;
    z: same shape as ``o`` fp32 or ``None``. fp32 throughout (the gated norm is
    numerically load-bearing, never quantized, per AGENTS.md). vd <= 128.
    """
    return _C.gated_rmsnorm_decode(
        o.contiguous(),
        gain.contiguous(),
        z.contiguous() if z is not None else None,
        eps,
    )


def deltanet_chunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track-2 (issue #40) v2: ungated chunked WY/UT parallel DeltaNet (fp32).

    The ungated (alpha=1, beta=1) delta rule with L2-normalised k,
    processed in chunks using the WY/UT decomposition for intra-chunk
    parallelism.  Matches :func:`tests.reference_linear_attn.ungated_delta_rule_oracle`
    at fp32 reassociation tolerance.

    q, k: [B,H,T,Dk] fp32; v: [B,H,T,Dv] fp32.
    ``initial_state``, if given: [B,H,Dv,Dk] fp32 carry-in state.
    Returns ``(o [B,H,T,Dv] fp32, final_state [B,H,Dv,Dk] fp32)``.
    Supports Dv, Dk <= 128 (chunk size adapts to shared-mem budget).
    """
    return _C.deltanet_chunk_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        initial_state.contiguous() if initial_state is not None else None,
    )


def deltanet_gated_chunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track-2 (issue #56) v3: gated chunked WY/UT parallel DeltaNet (fp32)
    with log-space γ-cumprod decay stabilisation.

    The full gated-delta-rule recurrence with per-step α (decay) and β
    (write-strength) gates, computed in chunks using the WY/UT decomposition
    for intra-chunk parallelism.  Gates compose multiplicatively within a
    chunk; the cumulative product γ[i] = ∏ α₀…αᵢ₋₁ is tracked in log-space
    to avoid fp32 underflow.  Matches
    :func:`tests.reference_linear_attn.gated_delta_rule_oracle` at fp32
    reassociation tolerance.

    q, k: [B,H,T,Dk] fp32; v: [B,H,T,Dv] fp32; alpha, beta: [B,H,T] fp32
    (gate ranges are the caller's responsibility — see
    :func:`tests.reference_linear_attn.gates_from_logits`).
    ``initial_state``, if given: [B,H,Dv,Dk] fp32 carry-in state.
    Returns ``(o [B,H,T,Dv] fp32, final_state [B,H,Dv,Dk] fp32)``.
    Supports Dv, Dk <= 128 (chunk size adapts to shared-mem budget).
    """
    return _C.deltanet_gated_chunk_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        alpha.contiguous(),
        beta.contiguous(),
        initial_state.contiguous() if initial_state is not None else None,
    )


def deltanet_gated_chunk_h2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track-2 (issue #122) v3.5: half2 (FP16x2) CUDA-core gated chunked DeltaNet.

    Recasts the WY/UT chunk-matmul inner dot products (S@k, k·k Gram, q·k)
    onto ``__hfma2`` half2 packed math with fp32 accumulation.  Everything
    numerically load-bearing stays fp32: state, log-space γ-cumprod, L2-norm,
    α/β gates, values V, residuals, and the state update AXPY.  This is the
    healthy half2 CUDA-core pipe (~27 TFLOP/s), NOT the firmware-dead fp16
    tensor cores (per AGENTS.md silicon audit).  Inputs/outputs are fp32;
    half2 is internal only.  Matches
    :func:`tests.reference_linear_attn.gated_delta_rule_oracle` at fp16
    tolerance (rtol/atol 5e-3).

    q, k: [B,H,T,Dk] fp32; v: [B,H,T,Dv] fp32; alpha, beta: [B,H,T] fp32.
    ``initial_state``, if given: [B,H,Dv,Dk] fp32 carry-in state.
    Returns ``(o [B,H,T,Dv] fp32, final_state [B,H,Dv,Dk] fp32)``.
    Supports Dv, Dk <= 128.
    """
    return _C.deltanet_gated_chunk_h2_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        alpha.contiguous(),
        beta.contiguous(),
        initial_state.contiguous() if initial_state is not None else None,
    )


def deltanet_chunk_int8_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track-2 (issue #83) v4: int8 dp4a ungated chunked DeltaNet.

    Same chunked ungated delta-rule algebra as :func:`deltanet_chunk_fwd`, but
    the two Dk-contraction score matrices — the intra-chunk K-Gram ``kᵢ·kⱼ``
    (feeds forward substitution) and the Q·K read ``qᵢ·kⱼ`` (feeds the output) —
    are computed with the ``__dp4a`` int8×4 CUDA-core intrinsic. sm_70 has NO
    int8 tensor cores, so dp4a is the primitive, per AGENTS.md.

    Keys are L2-normalised per token BEFORE symmetric per-row int8 quantization;
    that normalisation is the delta-rule analogue of K-smoothing (the softmax
    mean-subtraction smoothing is invalid here — there is no softmax row-shift to
    cancel it). The numerically load-bearing pieces — state S, values V,
    residuals, ``W = V − S@Kᵀ``, the output base ``S@q`` and the state update
    ``r@K`` — stay fp32 and are never quantized; only the pure q/k dot products
    go through int8.

    q, k: [B,H,T,Dk] fp32; v: [B,H,T,Dv] fp32.
    ``initial_state``, if given: [B,H,Dv,Dk] fp32 carry-in state.
    Returns ``(o [B,H,T,Dv] fp32, final_state [B,H,Dv,Dk] fp32)``. Matches
    :func:`tests.reference_linear_attn.ungated_delta_rule_oracle` at int8
    SQNR/cosine tolerance (NOT allclose). Supports Dv, Dk <= 128.
    """
    return _C.deltanet_chunk_int8_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        initial_state.contiguous() if initial_state is not None else None,
    )


def mla_decode_absorb(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv_cache: torch.Tensor,
    k_rope_cache: torch.Tensor,
    w_qabs: torch.Tensor,
    w_ovabs: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Track-2 (issue #7) v1: MLA (DeepSeek-V2/V3) absorb-path decode.

    CUDA port of :func:`tests.reference_mla.mla_absorb`'s decode step (fp32
    ground truth -- the on-device oracle v2 (fp16) and v3 (int8 dp4a the
    latent QK/PV) get validated against, not a fast path itself; see
    AGENTS.md's TDD discipline).

    The per-token fold ``q'_h = q_nope_h @ W_UK_h^T`` (:func:`tests.reference_mla.absorb_qk_equiv`)
    happens here in a torch matmul -- it changes every step, so it cannot be
    precomputed offline. ``w_ovabs`` (:func:`tests.reference_mla.absorb_ov_equiv`,
    ``W_UV_h @ W_O_h``) is expected already folded OFFLINE by the caller
    (computed once from the static weights, reused every decode step). The
    CUDA kernel (``superl8._C.mla_decode``) does only the attention math: MQA
    QK/softmax/PV against the shared latent, with the decoupled-RoPE score
    folded in -- no per-head K/V and no linear-projection weight is ever
    materialized/multiplied inside the kernel, matching this repo's
    convention of keeping GEMMs (:func:`gemm_w8a8`, etc.) out of the
    attention kernels.

    ``q_nope``: [B,H,1,d_h] the un-absorbed nope query (post RMSNorm/down-up
    projection, no RoPE). ``q_rope``: [B,H,1,d_r], already RoPE'd.
    ``c_kv_cache``: [B,N,d_c] the RMSNorm'd latent cache (its last row is this
    step's own token -- decode-step convention, see
    :func:`tests.reference_mla.mla_absorb`). ``k_rope_cache``: [B,N,d_r], the
    shared decoupled-RoPE key cache. ``w_qabs``: [H,d_h,d_c]
    (:func:`tests.reference_mla.absorb_qk_equiv`). ``w_ovabs``: [H,d_c,d_model]
    (:func:`tests.reference_mla.absorb_ov_equiv`). ``scale`` defaults to
    ``1/sqrt(d_h + d_r)`` (the ORIGINAL per-head dims, not d_c -- matches the
    decompress-path score magnitude). All tensors fp32 (v1 is fp32-only).
    Returns ``out`` fp32 ``[B,1,d_model]``.
    """
    import math

    from . import _C

    assert q_nope.shape[-2] == 1, f"mla_decode_absorb expects Tq=1 (decode), got {q_nope.shape[-2]}"
    d_h, d_r = q_nope.shape[-1], q_rope.shape[-1]
    scale = (1.0 / math.sqrt(d_h + d_r)) if scale is None else scale

    b, h = q_nope.shape[0], q_nope.shape[1]
    q_nope_bh = q_nope.reshape(b, h, d_h)
    q_abs = torch.einsum("bhd,hdc->bhc", q_nope_bh, w_qabs).contiguous()
    q_rope_bh = q_rope.reshape(b, h, d_r).contiguous()

    o_abs = _C.mla_decode(
        q_abs, q_rope_bh, c_kv_cache.contiguous(), k_rope_cache.contiguous(), float(scale)
    )  # [B,H,d_c]
    return torch.einsum("bhc,hcm->bm", o_abs, w_ovabs).unsqueeze(1)  # [B,1,d_model]


def dit_block(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    *,
    gate: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    rotary_dim: int = 0,
) -> torch.Tensor:
    """Fused DiT post-attention block in ONE kernel launch.

    Does RMSNorm → adaLN scale/shift/gate apply → residual add → RoPE, all
    in fp32 registers, reading the bf16 hidden state from HBM once and
    writing bf16 once.  Collapses N bandwidth-bound elementwise passes into 1
    (the highest-leverage bf16 fusion per the roofline analysis).

    ``x``: [M, D] fp16/bf16 input.  ``rms_weight``: [D] RMSNorm weight.
    ``scale``, ``shift``: [M, D] adaLN modulation params.
    ``gate`` (optional): [M, D] gate — ``modulated *= sigmoid(gate)``.
    ``positions`` (optional): [M] int64 RoPE positions.
    ``cos``, ``sin`` (optional): [max_pos, rotary_dim] fp32 tables.
    ``rotary_dim`` (optional): number of dims to rotate (must be even).
    Returns [M, D] same dtype as x.

    CPU / dtype fallback = decomposed eager torch ops.
    """
    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        return _C.dit_block(
            x.contiguous(),
            rms_weight.contiguous(),
            scale.contiguous(),
            shift.contiguous(),
            float(eps),
            gate.contiguous() if gate is not None else None,
            positions.contiguous() if positions is not None else None,
            cos.contiguous() if cos is not None else None,
            sin.contiguous() if sin is not None else None,
            int(rotary_dim),
        )
    # CPU fallback
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    normed = xf * rms * rms_weight.float()
    modulated = normed * (1.0 + scale.float()) + shift.float()
    if gate is not None:
        modulated = modulated * torch.sigmoid(gate.float())
    h = modulated + xf
    if positions is not None and cos is not None and sin is not None and rotary_dim > 0:
        rd = rotary_dim
        half = rd // 2
        c = cos[positions].to(torch.float32)  # [M, rd]
        s = sin[positions].to(torch.float32)
        hr, hp = h[..., :rd], h[..., rd:]
        hr_rot = torch.empty_like(hr)
        hr_rot[:, :half] = hr[:, :half] * c[:, :half] - hr[:, half:] * s[:, :half]
        hr_rot[:, half:] = hr[:, half:] * c[:, half:] + hr[:, :half] * s[:, half:]
        h = torch.cat((hr_rot, hp), dim=-1) if hp.shape[-1] else hr_rot
    return h.to(x.dtype)


def linear(
    x: torch.Tensor,
    qt,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Scheme-dispatched linear over a :class:`superl8.format.QTensor` weight.

    ``per_row_i8`` -> :func:`linear_w8a8`; ``per_group_i4`` with ``codebook='int4'``
    -> :func:`linear_w4a8`; ``per_group_w3a8`` -> :func:`linear_w3a8` (uniform 3-bit
    dp4a MLP, the issue #181 VRAM/context lever). ``gguf_kquant`` -> the native fused
    k-quant linears (``tq3_4s`` routes to the :func:`linear_tq34s` reference until the
    fused kernel, superl8#272). NF4 weights are a non-integer lookup codebook and
    cannot enter dp4a, so they are rejected here (NF4 stays valid for the decode
    V-cache, which is a lookup, and for an fp16 dequant fallback). ``out_dtype``
    (``float16`` default, or ``bfloat16``) selects the store dtype for a
    bf16-native compute path (see :func:`linear_w8a8`).
    """
    scheme = qt.scheme
    if scheme == "per_row_i8":
        return linear_w8a8(x, qt.data, qt.scale, bias=bias, out_dtype=out_dtype)
    if scheme == "per_group_i4":
        if qt.codebook != "int4":
            raise ValueError(
                f"superl8.linear: per_group_i4 GEMM needs codebook 'int4', got "
                f"{qt.codebook!r} (nf4 is non-integer — not dp4a-compatible)"
            )
        return linear_w4a8(
            x, qt.data, qt.scale, group_size=qt.group_size, bias=bias, out_dtype=out_dtype
        )
    if scheme == "per_group_w3a8":
        # Uniform 3-bit dp4a MLP (issue #181 VRAM/context lever): int32 bit-planes
        # unpacked to int8 for dp4a at decode PARITY with w4a8, 0.75x the bytes.
        return linear_w3a8(
            x, qt.data, qt.scale, group_size=qt.group_size, bias=bias, out_dtype=out_dtype
        )
    if scheme == "gguf_kquant":
        # Native GGUF k-quant, matmul'd on the fly (no conversion / no resident
        # re-quant). The type tag rides in `codebook`; Q4_K is the fused dp4a path.
        # One table, not a hand-written if-chain: a chain silently rejects a type
        # whose kernel HAS shipped. The i-quant arms were missing here even after
        # their decode AND tile kernels landed, so a resident IQ4_XS weight reached
        # this line and raised "has no fused kernel" (superl8#317).
        _op = _FUSED_KQUANT_OPS.get(qt.codebook)
        if _op is not None:
            return _op(x, qt.data, bias=bias, out_dtype=out_dtype)
        if qt.codebook == "tq3_4s":
            # Native TQ3_4S (type 46): fused dp4a kernel (superl8#272) — the
            # activation is forward-RHT-rotated per 32-block before the per-32
            # q8_1 quant, 3-bit codes become corrected int8 centroid levels, and
            # each per-8 E3M5 scale flushes the fp32 accumulator. Fidelity gate
            # vs the fp32 CPU dequant oracle: SQNR>=40 dB / cos>=0.999.
            from .quant.tq34s import QK_TQ3, TQ3_TYPE_SIZE

            in_f = qt.data.shape[-1] // TQ3_TYPE_SIZE * QK_TQ3  # (row_bytes/16)*32
            return linear_tq34s(x, qt.data, in_f, bias=bias, out_dtype=out_dtype)
        raise ValueError(
            f"superl8.linear: gguf_kquant type {qt.codebook!r} has no fused kernel"
        )
    raise ValueError(f"superl8.linear: unsupported weight scheme {scheme!r}")


def mla_decode_absorb_fp16(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv_cache: torch.Tensor,
    k_rope_cache: torch.Tensor,
    w_qabs: torch.Tensor,
    w_ovabs: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Track-2 (issue #41) v2: MLA absorb-path decode in **fp16**.

    Same absorb-path reformulation as :func:`mla_decode_absorb` (v1 fp32), but
    the KV cache is fp16 — half the HBM bandwidth of the fp32 path. The CUDA
    kernel loads fp16 and accumulates online-softmax/PV in fp32 internally
    (per AGENTS.md, softmax is numerically load-bearing).

    ``q_nope``: fp32 [B,H,1,d_h] the un-absorbed nope query. ``q_rope``:
    fp32 [B,H,1,d_r], already RoPE'd. ``c_kv_cache``: fp16 [B,N,d_c] the
    RMSNorm'd latent cache. ``k_rope_cache``: fp16 [B,N,d_r]. ``w_qabs``:
    fp32 [H,d_h,d_c] (absorb_qk_equiv). ``w_ovabs``: fp32 [H,d_c,d_model]
    (absorb_ov_equiv). ``scale`` defaults to ``1/sqrt(d_h + d_r)``.

    Returns ``out`` fp16 [B,1,d_model].
    """
    import math

    from . import _C

    assert q_nope.shape[-2] == 1, (
        f"mla_decode_absorb_fp16 expects Tq=1 (decode), got {q_nope.shape[-2]}"
    )
    d_h, d_r = q_nope.shape[-1], q_rope.shape[-1]
    scale = (1.0 / math.sqrt(d_h + d_r)) if scale is None else scale

    b, h = q_nope.shape[0], q_nope.shape[1]
    q_nope_bh = q_nope.reshape(b, h, d_h)
    q_abs = torch.einsum("bhd,hdc->bhc", q_nope_bh, w_qabs).half().contiguous()
    q_rope_bh = q_rope.reshape(b, h, d_r).half().contiguous()

    o_abs = _C.mla_decode_fp16(
        q_abs,
        q_rope_bh,
        c_kv_cache.contiguous(),
        k_rope_cache.contiguous(),
        float(scale),
    )  # [B,H,d_c] fp16
    return (
        torch.einsum("bhc,hcm->bm", o_abs.float(), w_ovabs).half().unsqueeze(1)
    )  # [B,1,d_model] fp16


def mla_decode_absorb_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv_cache: torch.Tensor,
    k_rope_cache: torch.Tensor,
    w_qabs: torch.Tensor,
    w_ovabs: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Track-2 (issue #57) v3: MLA absorb-path decode in **int8 dp4a**.

    Same absorb-path reformulation as :func:`mla_decode_absorb` (v1 fp32) and
    :func:`mla_decode_absorb_fp16` (v2 fp16), but the QK dot product runs via
    int8 dp4a: Q_abs is quantized per-head and the per-token C_KV cache row
    is quantized on-the-fly (symmetric per-row RTN, fp32 scales per the SDNQ
    quant recipe). The decoupled RoPE score, softmax, and PV weighted-sum
    accumulation stay fp32 (numerically load-bearing per AGENTS.md).

    ``q_nope``: fp32 [B,H,1,d_h] the un-absorbed nope query. ``q_rope``:
    fp32 [B,H,1,d_r], already RoPE'd. ``c_kv_cache``: fp16 [B,N,d_c] the
    RMSNorm'd latent cache. ``k_rope_cache``: fp16 [B,N,d_r]. ``w_qabs``:
    fp32 [H,d_h,d_c] (absorb_qk_equiv). ``w_ovabs``: fp32 [H,d_c,d_model]
    (absorb_ov_equiv). ``scale`` defaults to ``1/sqrt(d_h + d_r)``.

    Returns ``out`` fp16 [B,1,d_model].
    """
    import math

    from . import _C

    assert q_nope.shape[-2] == 1, (
        f"mla_decode_absorb_int8 expects Tq=1 (decode), got {q_nope.shape[-2]}"
    )
    d_h, d_r = q_nope.shape[-1], q_rope.shape[-1]
    scale = (1.0 / math.sqrt(d_h + d_r)) if scale is None else scale

    b, h = q_nope.shape[0], q_nope.shape[1]
    q_nope_bh = q_nope.reshape(b, h, d_h)
    q_abs = torch.einsum("bhd,hdc->bhc", q_nope_bh, w_qabs).half().contiguous()
    q_rope_bh = q_rope.reshape(b, h, d_r).half().contiguous()

    o_abs = _C.mla_decode_int8(
        q_abs,
        q_rope_bh,
        c_kv_cache.contiguous(),
        k_rope_cache.contiguous(),
        float(scale),
    )  # [B,H,d_c] fp16
    return (
        torch.einsum("bhc,hcm->bm", o_abs.float(), w_ovabs).half().unsqueeze(1)
    )  # [B,1,d_model] fp16


_register_fused_kquant_ops()
