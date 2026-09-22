# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""3-bit Lloyd-Max KV-cache quantizer (qengine TQ3, adapted) — the quality gate
decides where it's allowed on.

The Walsh-Hadamard rotation ALREADY lives in the KV path (superl8.quant.rotation);
these tests build ONLY the 3-bit Lloyd-Max quantizer on top of the rotated KV.
They measure (a) round-trip SQNR/cos of dequant(3bit(rotated_KV)) vs rotated_KV,
(b) end-to-end attention-output quality vs the fp32 oracle, (c) that non-uniform
Lloyd-Max levels beat the uniform 3-bit grid at the same 3 bits, and (d) storage.
Low-bit paths use SQNR/cos/rel-L1 (never allclose).
"""
import pytest
import torch

import superl8  # noqa: F401
from superl8.quant import smooth_k
from superl8.quant.lloydmax import (
    dequantize_lloydmax,
    fake_quant_lloydmax,
    lloyd_max_fit,
    pack_indices_lowbit,
    quantize_lloydmax,
    unpack_indices_lowbit,
)
from superl8.quant.lowbit import fake_quant_lowbit
from superl8.quant.rotation import rotate_last
from tests.reference import attention_fp32_oracle
from tests.tolerances import cos_sim, rel_l1, sqnr_db


def _outlier_qkv(device, d=128, n=512, h=8, outlier=True):
    torch.manual_seed(0)
    q = torch.randn(1, h, n, d, device=device, dtype=torch.float16)
    k = torch.randn(1, h, n, d, device=device, dtype=torch.float16)
    v = torch.randn(1, h, n, d, device=device, dtype=torch.float16)
    if outlier:
        k[..., [0, 1, 7]] *= 40.0  # per-channel outliers (real activations look like this)
    return q, k, v


def _rotated_k(k):
    """The tensor the quantizer actually sees: smoothed + Hadamard-rotated K."""
    k_s, _ = smooth_k(k)
    return rotate_last(k_s).float()


@pytest.mark.cpu
@pytest.mark.correctness
def test_lloydmax_codebook_stays_on_cpu_for_cpu_input():
    x = torch.randn(2, 128, dtype=torch.float32)
    _, _, codebook = quantize_lloydmax(x, bits=3, block_size=128)
    assert codebook.device == x.device


@pytest.mark.correctness
def test_lloydmax_quantize_is_cuda_graph_capturable(device):
    """Quantization must not force its device codebook through host memory."""
    x = torch.randn(2, 128, device=device, dtype=torch.float32)
    explicit = torch.linspace(-0.2, 0.2, 8, device=device, dtype=torch.float32)

    # Warm lazy CUDA/runtime state before capture; the quantizer itself remains
    # fully inside the graph below.
    quantize_lloydmax(x, bits=3, block_size=128, codebook=explicit)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        codes, norm, codebook = quantize_lloydmax(
            x, bits=3, block_size=128, codebook=explicit
        )

    graph.replay()
    torch.cuda.synchronize(device)
    assert codes.device == x.device
    assert norm.device == x.device
    assert codebook.device == x.device
    assert codebook.dtype == torch.float32


# ---------------------------------------------------------------- packing ----

@pytest.mark.correctness
@pytest.mark.parametrize("d", [32, 64, 128])
def test_lloydmax_packing_aligned_and_lossless(device, d):
    """3-bit indices pack to EXACTLY d*3/32 int32 words and unpack losslessly."""
    idx = torch.randint(0, 8, (4, 16, d), device=device, dtype=torch.uint8)
    packed = pack_indices_lowbit(idx, 3)
    assert packed.shape[-1] == (d * 3) // 32, f"expected {(d*3)//32} words, got {packed.shape[-1]}"
    back = unpack_indices_lowbit(packed, 3, d)
    assert torch.equal(back.int(), idx.int()), f"pack/unpack not lossless at d={d}"


# ------------------------------------------------------------ round-trip ----

@pytest.mark.correctness
def test_lloydmax_roundtrip_beats_uniform_on_rotated_k(device):
    """On rotated K (Gaussian-ish coords), non-uniform Lloyd-Max levels beat the
    uniform 3-bit grid at the SAME 3 bits — the whole point of TQ3."""
    _, k, _ = _outlier_qkv(device)
    kr = _rotated_k(k)
    lm_fixed = fake_quant_lloydmax(kr, bits=3, block_size=128)
    lm_fit = fake_quant_lloydmax(kr, bits=3, block_size=128, fit=True)
    uni = fake_quant_lowbit(kr, 3, dim=-1, group_size=128)  # uniform 3-bit, same grouping
    sq_uni, sq_fix, sq_fit = (sqnr_db(x, kr) for x in (uni, lm_fixed, lm_fit))
    print(f"\nrotated-K 3-bit round-trip SQNR(dB): uniform={sq_uni:.2f}  "
          f"LloydMax-fixed={sq_fix:.2f}  LloydMax-fit={sq_fit:.2f}")
    assert sq_fix > sq_uni, f"Lloyd-Max (fixed) must beat uniform: {sq_fix:.2f} vs {sq_uni:.2f}"
    assert sq_fit >= sq_fix - 0.5, "fitted codebook should not be materially worse than fixed"
    assert cos_sim(lm_fit, kr) > 0.99, "3-bit rotated-K round-trip should stay high-cos"


@pytest.mark.correctness
def test_lloydmax_determinism(device):
    """Same input x3 -> bitwise-identical codes/norm/dequant (no RNG anywhere)."""
    _, k, _ = _outlier_qkv(device)
    kr = _rotated_k(k)
    outs = []
    for _ in range(3):
        codes, norm, cb = quantize_lloydmax(kr, bits=3, block_size=128)
        outs.append((codes, norm, dequantize_lloydmax(codes, norm, cb, block_size=128)))
    for i in (1, 2):
        assert torch.equal(outs[0][0], outs[i][0]), "codes not deterministic"
        assert torch.equal(outs[0][1], outs[i][1]), "norm not deterministic"
        assert torch.equal(outs[0][2], outs[i][2]), "dequant not deterministic"
    # Lloyd-Max fit is deterministic too (quantile init, no RNG)
    cb0 = lloyd_max_fit(kr / kr.norm(dim=-1, keepdim=True), 8)
    cb1 = lloyd_max_fit(kr / kr.norm(dim=-1, keepdim=True), 8)
    assert torch.equal(cb0, cb1), "Lloyd-Max fit not deterministic"


# -------------------------------------------------- attention quality gate ----

def _attn_lm_kv(q, k, v, *, k_bits, v_bits, fit):
    """fp32 attention with K/V 3-bit-Lloyd-Max fake-quantized (rotation-consistent).

    K is smoothed+rotated (quantizer sees the rotated tensor); Q is rotated to
    match. V is unrotated ("compresses free"), blocked over the head dim."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    k_s, _ = smooth_k(k)
    qr, kr = rotate_last(q).float(), rotate_last(k_s).float()
    kq = fake_quant_lloydmax(kr, bits=k_bits, block_size=128, fit=fit).float() if k_bits else kr
    vq = fake_quant_lloydmax(v, bits=v_bits, block_size=128, fit=fit).float() if v_bits else v.float()
    s = torch.einsum("bhmd,bhnd->bhmn", qr, kq) * scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", p, vq)


def _attn_uni_k(q, k, v):
    """uniform 3-bit K (V fp) baseline for the 'Lloyd-Max earns its keep' check."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    k_s, _ = smooth_k(k)
    qr, kr = rotate_last(q).float(), rotate_last(k_s).float()
    kq = fake_quant_lowbit(kr, 3, dim=-1, group_size=128).float()
    s = torch.einsum("bhmd,bhnd->bhmn", qr, kq) * scale
    return torch.einsum("bhmn,bhnd->bhmd", torch.softmax(s, dim=-1), v.float())


@pytest.mark.correctness
@pytest.mark.parametrize("outlier", [True, False], ids=["outlier40x", "benign"])
def test_lloydmax_attention_quality_gate(device, outlier):
    """Measure attention-output quality of 3-bit Lloyd-Max KV vs the fp32 oracle
    and decide default-on vs opt-in. Asserts the MEASURED truths; the hard int8
    bar (cos>=0.999, rel-L1<=0.02) is a REPORTED verdict, never silently weakened
    to pass. The measured verdict is that 3-bit KV lands ~cos 0.98 / rel-L1 ~0.18
    (SQNR ~14.5 dB) — well under the int8 bar — so it is an OPT-IN memory knob."""
    q, k, v = _outlier_qkv(device, outlier=outlier)
    oracle = attention_fp32_oracle(q, k, v)

    configs = [  # (label, k_bits, v_bits, fit)
        ("K3(fix) V8    ", 3, 0, False),
        ("K3(fit) V8    ", 3, 0, True),
        ("K8     V3(fit)", 0, 3, True),
        ("K3(fit) V3(fit)", 3, 3, True),
    ]
    print(f"\n[outlier={outlier}] {'config':16s}  {'cos':>9s}  {'rel_l1':>8s}  {'sqnr_dB':>8s}  gate")
    res = {}
    for label, kb, vb, fit in configs:
        out = _attn_lm_kv(q, k, v, k_bits=kb, v_bits=vb, fit=fit)
        c, l1, s = cos_sim(out, oracle), rel_l1(out, oracle), sqnr_db(out, oracle)
        res[label.strip()] = (c, l1, s)
        gate = "DEFAULT-ON" if (c >= 0.999 and l1 <= 0.02) else "opt-in"
        print(f"[outlier={outlier}] {label:16s}  {c:9.5f}  {l1:8.4f}  {s:8.2f}   [{gate}]")
    uni = _attn_uni_k(q, k, v)
    cu = cos_sim(uni, oracle)

    # (1) Lloyd-Max 3-bit K >= uniform 3-bit K at the attention output (fit helps
    #     most under outliers, where uniform absmax wastes codes on the tail).
    ck_lm = res["K3(fit) V8"][0]
    assert ck_lm >= cu - 1e-4, f"Lloyd-Max K3 ({ck_lm:.5f}) must not lose to uniform K3 ({cu:.5f})"
    # (2) fit >= fixed for K under outliers (non-Gaussian tails => fitted levels win).
    if outlier:
        assert res["K3(fit) V8"][0] > res["K3(fix) V8"][0] + 0.02, \
            "fitted codebook must beat fixed on outlier-heavy K"
    # (3) V is the compressible side: 3-bit V >> 3-bit K under outliers (asymmetric KV).
    if outlier:
        assert res["K8     V3(fit)"][0] > res["K3(fit) V8"][0] + 0.05, \
            "expected V3 >> K3 under outliers (TurboQuant asymmetric-KV)"
    # (4) VERDICT: no 3-bit config clears the int8 default-on bar (cos>=0.999,
    #     rel-L1<=0.02) even on benign data — it stays OPT-IN. Assert this measured
    #     ceiling so a future regression that silently promotes 3-bit to default-on
    #     (or a collapse below usable) trips here.
    best_cos = max(c for c, _, _ in res.values())
    best_l1 = min(l1 for _, l1, _ in res.values())
    assert best_cos < 0.999 or best_l1 > 0.02, \
        f"3-bit KV unexpectedly cleared the int8 bar (cos={best_cos:.5f}, l1={best_l1:.4f}) — recheck"
    # usable floor: on benign data every config stays > 0.96 cos; with fit, even the
    # 40x-outlier K path stays > 0.8 (fixed-codebook K collapses under outliers — a
    # documented reason fit is the recommended mode).
    floor = 0.96 if not outlier else 0.80
    assert res["K3(fit) V3(fit)"][0] > floor, \
        f"K3V3(fit) below usable floor {floor}: {res['K3(fit) V3(fit)'][0]:.4f}"


# ----------------------------------------------------------------- storage ----

@pytest.mark.correctness
def test_lloydmax_storage_ratio(device):
    """3-bit TQ3 block (128 coords) = 48 B codes + 4 B fp32 norm = 52 B vs 256 B
    fp16 => 4.92x. Verify the packed byte count matches and report the ratio."""
    _, k, _ = _outlier_qkv(device)
    kr = _rotated_k(k)  # [1,H,N,128]
    codes, norm, _ = quantize_lloydmax(kr, bits=3, block_size=128)
    packed = pack_indices_lowbit(codes, 3)  # [...,128*3/32 = 12] int32
    code_bytes = packed.numel() * 4
    norm_bytes = norm.numel() * 4
    fp16_bytes = kr.numel() * 2
    ratio = fp16_bytes / (code_bytes + norm_bytes)
    print(f"\nTQ3 storage: codes={code_bytes}B + norm={norm_bytes}B vs fp16={fp16_bytes}B "
          f"=> {ratio:.2f}x (norm fp16 => {fp16_bytes/(code_bytes+norm_bytes/2):.2f}x)")
    assert packed.shape[-1] == 12, "128 coords at 3-bit must pack to 12 int32 (48 B)"
    assert 4.8 <= ratio <= 5.0, f"expected ~4.92x for fp32-norm TQ3, got {ratio:.2f}x"
