# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Varlen (cu_seqlens-packed) prefill forward — the serving feature. Tests first.

Serving batches variable-length prompts packed contiguously (no padding), the
flash_attn_varlen convention: q/k/v are [total_tokens, H, D] (token-major, heads
interleaved) and cu_seqlens[b] gives each sequence's offset. Each sequence
attends only within itself. This is the prefill path of an inference server.
"""
import statistics
import sys
from pathlib import Path

import pytest
import torch

import superl8
from tests.reference import attention_fp32_varlen_oracle
from tests.tolerances import assert_finite, assert_int8_quality

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import time_ms  # noqa: E402

# (seqlens, H_q, H_kv, D). Ragged, incl. non-tile-multiple and length-1.
VARLEN_CASES = [
    ([3, 128, 1, 257, 64], 8, 2, 64),      # GQA group 4, ragged
    ([64, 64], 4, 4, 128),                  # MHA, D=128
    ([200, 37, 512], 16, 4, 128),           # GQA group 4, D=128
    ([1, 1, 1, 1], 8, 1, 64),               # MQA, all length-1 (decode-ish batch)
    ([333], 4, 2, 64),                      # single ragged sequence
]


def _packed_qkv(seqlens, hq, hkv, d, device):
    total = sum(seqlens)
    q = torch.randn(total, hq, d, device=device, dtype=torch.float16)
    k = torch.randn(total, hkv, d, device=device, dtype=torch.float16)
    v = torch.randn(total, hkv, d, device=device, dtype=torch.float16)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), dtype=torch.int32, device=device)
    return q, k, v, cu


@pytest.mark.correctness
@pytest.mark.parametrize("seqlens,hq,hkv,d", VARLEN_CASES)
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_quality(device, seqlens, hq, hkv, d, causal):
    q, k, v, cu = _packed_qkv(seqlens, hq, hkv, d, device)
    out = superl8.attn_int8_varlen(q, k, v, cu, cu, max(seqlens), max(seqlens), causal=causal)
    oracle = attention_fp32_varlen_oracle(q, k, v, cu, cu, causal=causal)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"varlen {seqlens} c={causal}")


@pytest.mark.correctness
def test_varlen_matches_dense_single_seq(device):
    """A single-sequence varlen batch must equal the dense [1,H,S,D] path."""
    from tests.tolerances import cos_sim

    s, hq, hkv, d = 256, 8, 2, 128
    q, k, v, cu = _packed_qkv([s], hq, hkv, d, device)
    out_vl = superl8.attn_int8_varlen(q, k, v, cu, cu, s, s, causal=True)
    # dense: [1, H, S, D]
    qd, kd, vd = (t.permute(1, 0, 2).unsqueeze(0).contiguous() for t in (q, k, v))
    out_dense = superl8.attn_int8_fwd(qd, kd, vd, causal=True).squeeze(0).permute(1, 0, 2)
    assert cos_sim(out_vl, out_dense) >= 0.999, f"cos {cos_sim(out_vl, out_dense):.5f}"


@pytest.mark.correctness
def test_varlen_deterministic(device):
    q, k, v, cu = _packed_qkv([128, 65, 300], 8, 2, 64, device)
    r0 = superl8.attn_int8_varlen(q, k, v, cu, cu, 300, 300, causal=True)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_varlen(q, k, v, cu, cu, 300, 300, causal=True), r0)


@pytest.mark.perf
def test_varlen_beats_padded_dense(device):
    """Varlen's whole point: a high-length-variance batch wastes most of the
    compute if padded to max_seqlen. Varlen processes only real tokens."""
    seqlens = [16, 16, 16, 16, 2048]   # one long seq -> heavy padding if dense
    hq, hkv, d = 16, 4, 128
    q, k, v, cu = _packed_qkv(seqlens, hq, hkv, d, device)
    ms = max(seqlens)
    varlen = time_ms(lambda: superl8.attn_int8_varlen(q, k, v, cu, cu, ms, ms, causal=True))

    # padded-dense equivalent: [B, H, max_S, D], the naive no-varlen path.
    b = len(seqlens)
    qd = torch.zeros(b, hq, ms, d, device=device, dtype=torch.float16)
    kd = torch.zeros(b, hkv, ms, d, device=device, dtype=torch.float16)
    vd = torch.zeros(b, hkv, ms, d, device=device, dtype=torch.float16)
    off = 0
    for i, s in enumerate(seqlens):
        qd[i, :, :s] = q[off:off + s].permute(1, 0, 2)
        kd[i, :, :s] = k[off:off + s].permute(1, 0, 2)
        vd[i, :, :s] = v[off:off + s].permute(1, 0, 2)
        off += s
    dense = time_ms(lambda: superl8.attn_int8_fwd(qd, kd, vd, causal=True))
    tokens = sum(seqlens)
    print(f"\nvarlen {tokens} tok: {varlen:.3f} ms | padded-dense {b}x{ms}: {dense:.3f} ms "
          f"| {dense / varlen:.2f}x")
    assert varlen < dense, f"varlen ({varlen:.3f}) should beat padded-dense ({dense:.3f})"


@pytest.mark.correctness
@pytest.mark.parametrize("cache,k", [(500, 4), (2000, 8), (63, 2)])
def test_varlen_speculative_verify(device, cache, k):
    """Speculative-decode / MTP VERIFY shape: k draft tokens attend to [cache + k
    drafts], causal with the diagonal aligned to the sequence END (each draft sees
    the cache prefix + itself + preceding drafts). This is the attention every
    chain/sequential spec-decode step needs — q_len (k) << k_len (cache+k)."""
    hq, hkv, d = 16, 4, 128
    q = torch.randn(k, hq, d, device=device, dtype=torch.float16)          # drafts
    kk = torch.randn(cache + k, hkv, d, device=device, dtype=torch.float16)  # cache + drafts
    vv = torch.randn(cache + k, hkv, d, device=device, dtype=torch.float16)
    cu_q = torch.tensor([0, k], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, cache + k], dtype=torch.int32, device=device)
    out = superl8.attn_int8_varlen(q, kk, vv, cu_q, cu_k, k, cache + k, causal=True)
    oracle = attention_fp32_varlen_oracle(q, kk, vv, cu_q, cu_k, causal=True)
    assert out.shape == q.shape
    assert_int8_quality(out, oracle, what=f"spec-verify cache={cache} k={k}")


def _serial_per_seq(q, k, v, cu, causal=True):
    """The serial reference: attn_int8_fwd per sequence (one at a time), returned
    packed [total_tokens, H, D]. This is the bit-exact parity oracle for varlen."""
    ref = []
    for i in range(len(cu) - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        qd = q[s:e].permute(1, 0, 2).unsqueeze(0).contiguous()
        kd = k[s:e].permute(1, 0, 2).unsqueeze(0).contiguous()
        vd = v[s:e].permute(1, 0, 2).unsqueeze(0).contiguous()
        o = superl8.attn_int8_fwd(qd, kd, vd, causal=causal)
        ref.append(o.squeeze(0).permute(1, 0, 2))
    return torch.cat(ref, 0)


@pytest.mark.correctness
@pytest.mark.parametrize("B", [31, 63, 95, 64])
def test_varlen_matches_serial_identical_prompts(device, B):
    """Equal-length IDENTICAL prompts at non-power-of-two batch sizes must be
    BIT-IDENTICAL to the serial one-sequence-at-a-time path. Regression for the
    GLOBAL K-mean: a single packed mean is only softmax-invariant before per-row
    int8 quantization, which is why B31/63/95 diverged from serial on the real
    model (fni8-serve#351) while power-of-two batches coincidentally aligned."""
    hq, hkv, d, L = 8, 2, 64, 14
    torch.manual_seed(B)
    q1 = torch.randn(L, hq, d, device=device, dtype=torch.float16)
    k1 = torch.randn(L, hkv, d, device=device, dtype=torch.float16)
    v1 = torch.randn(L, hkv, d, device=device, dtype=torch.float16)
    q = q1.repeat(B, 1, 1)
    k = k1.repeat(B, 1, 1)
    v = v1.repeat(B, 1, 1)
    cu = torch.arange(B + 1, dtype=torch.int32, device=device) * L
    out_vl = superl8.attn_int8_varlen(q, k, v, cu, cu, L, L, causal=True)
    ref = _serial_per_seq(q, k, v, cu)
    assert torch.equal(out_vl, ref), f"varlen != serial at B={B}"


@pytest.mark.correctness
@pytest.mark.parametrize("seqlens", [
    [14] * 31, [14] * 63, [14] * 95, [14] * 64,
    [3, 7, 14, 21, 9, 12, 5, 17, 8, 4],          # ragged heterogeneous
    [1, 1, 1, 1, 1, 1],                          # all length-1
    [257, 3, 64, 18, 9, 77],                     # ragged, non-tile lengths
])
def test_varlen_matches_serial_heterogeneous_ragged(device, seqlens):
    """Heterogeneous (and ragged) prompts: every varlen segment must be
    BIT-IDENTICAL to the serial per-sequence path. RED with the GLOBAL K-mean —
    each sequence must smooth K with its OWN mean, never a packed-wide one."""
    hq, hkv, d = 8, 2, 64
    total = sum(seqlens)
    q = torch.randn(total, hq, d, device=device, dtype=torch.float16)
    k = torch.randn(total, hkv, d, device=device, dtype=torch.float16)
    v = torch.randn(total, hkv, d, device=device, dtype=torch.float16)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), dtype=torch.int32, device=device)
    out_vl = superl8.attn_int8_varlen(q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True)
    ref = _serial_per_seq(q, k, v, cu)
    assert torch.equal(out_vl, ref), f"varlen != serial for seqlens={seqlens}"


@pytest.mark.perf
def test_varlen_per_sequence_mean_prologue_overhead(device):
    """The per-sequence K-mean prologue must not regress the equal-length varlen
    production path by more than 5% vs the old GLOBAL-mean prologue (same kernel),
    and the ragged per-slice path must stay within a bounded overhead. Multiple
    stable measurements are recorded; the median gate is used."""
    from bench.harness import time_ms

    from superl8 import _C, quant

    hq, hkv, d, L, B = 8, 2, 64, 14, 128
    torch.manual_seed(0)
    q = torch.randn(B * L, hq, d, device=device, dtype=torch.float16)
    k = torch.randn(B * L, hkv, d, device=device, dtype=torch.float16)
    v = torch.randn(B * L, hkv, d, device=device, dtype=torch.float16)
    cu = torch.arange(B + 1, dtype=torch.int32, device=device) * L
    maxs = L
    softmax_scale = d ** -0.5

    # production path: the REAL attn_int8_varlen with the per-sequence K-mean.
    def new_fwd():
        return superl8.attn_int8_varlen(q, k, v, cu, cu, maxs, maxs, causal=True)

    # reference: the previous GLOBAL-mean prologue + the same kernel.
    def old_fwd():
        k_s = (k.float() - k.float().mean(dim=0, keepdim=True)).to(k.dtype)
        q_i8, q_scale = quant.quantize_int8_rowwise(q)
        k_i8, k_scale = quant.quantize_int8_rowwise(k_s)
        q_s = (q_scale * (softmax_scale * quant.LOG2E)).squeeze(-1).contiguous()
        k_s_ = k_scale.squeeze(-1).contiguous()
        return _C.attn_int8_varlen(q_i8.contiguous(), q_s, k_i8.contiguous(), k_s_,
                                   v.contiguous(), cu.contiguous(), cu.contiguous(),
                                   int(maxs), True)

    # Interleave new/old measurements so clock drift cancels; gate on the median
    # ratio. The per-sequence K-mean prologue is ~15us on a ~1ms op (1.5%); the
    # alternating-pair median makes the <=5% gate robust to run-to-run noise.
    ratios = []
    new_ms, old_ms = [], []
    for _ in range(5):
        old_ms.append(time_ms(old_fwd))
        new_ms.append(time_ms(new_fwd))
        ratios.append(new_ms[-1] / old_ms[-1])
    ratio = statistics.median(ratios)
    t_new = statistics.median(new_ms)
    t_old = statistics.median(old_ms)
    print(f"\nequal-length B{B}: new={t_new:.3f} ms old(global-mean)={t_old:.3f} ms "
          f"median-ratio={ratio:.3f}")
    assert ratio <= 1.05 + 1e-9, (
        f"equal-length prologue regressed: median-ratio {ratio:.3f} > 1.05")

    # ragged per-slice path (real ragged branch: equal-length runs vectorized).
    seqlens = [14] * 127 + [7]
    total = sum(seqlens)
    qr = torch.randn(total, hq, d, device=device, dtype=torch.float16)
    kr = torch.randn(total, hkv, d, device=device, dtype=torch.float16)
    vr = torch.randn(total, hkv, d, device=device, dtype=torch.float16)
    cur = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), dtype=torch.int32, device=device)
    t_rag = statistics.median(
        time_ms(lambda: superl8.attn_int8_varlen(qr, kr, vr, cur, cur, 14, 14, causal=True))
        for _ in range(3))
    print(f"ragged B=128 (127x14 + 7): {t_rag:.3f} ms (run-vectorized prologue)")
    assert t_rag <= 4.0 * t_old + 1e-9, (
        f"ragged prologue unbounded: {t_rag:.3f} vs {t_old:.3f}")
