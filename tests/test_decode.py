# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""PR8: decode (M=1 split-KV / flash-decoding) — tests written FIRST.

Autoregressive decode issues ONE query row against a long K/V cache. The prefill
kernel launches a BLOCK_M=64 tile and masks off 63 of 64 rows, and parallelises
only over B*H — far too few blocks to fill the SMs. `attn_int8_decode` splits the
KEY dim across blocks (flash-decoding): each block reduces its K/V chunk to a
partial (O, m, l), then a combine step merges the splits via the LSE trick.

Semantics: the single query is the newest token, so it attends to ALL N cached
keys (non-causal) — decode is inherently "full" attention over the cache.
"""
import sys
from pathlib import Path

import pytest
import torch

import superl8
from tests.reference import attention_fp32_oracle, sdpa_fp16
from tests.tolerances import assert_finite, assert_int8_quality

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.harness import compare_report, time_ms  # noqa: E402

# (B, H_q, H_kv, N, D). N chosen to include non-split-multiple lengths.
DECODE_SHAPES = [
    (1, 8, 8, 512, 64),      # MHA
    (4, 8, 8, 1024, 128),    # batched decode, D=128
    (1, 16, 4, 2000, 128),   # GQA group=4, N not a nice multiple
    (2, 8, 1, 777, 64),      # MQA, ragged N
    (1, 4, 4, 300, 32),      # D=32, short cache
    (1, 32, 8, 4096, 128),   # long cache, realistic Qwen-ish
]

# D=256 (Gemma-4B head_dim). D4=64 > DEC_WARP=32, so the split kernel needs 2
# dp4a int32-chunks per lane instead of 1 -> its own shape list to isolate that.
DECODE_D256_SHAPES = [
    (1, 4, 4, 512, 256),  # MHA
    (2, 8, 2, 1024, 256),  # GQA group=4
    (1, 16, 4, 777, 256),  # GQA, ragged N (not a split-size multiple)
]


def _qkv_decode(shape, device):
    b, hq, hkv, n, d = shape
    q = torch.randn(b, hq, 1, d, device=device, dtype=torch.float16)
    k = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    v = torch.randn(b, hkv, n, d, device=device, dtype=torch.float16)
    return q, k, v


@pytest.mark.correctness
@pytest.mark.parametrize("shape", DECODE_SHAPES)
def test_decode_quality(device, shape):
    q, k, v = _qkv_decode(shape, device)
    out = superl8.attn_int8_decode(q, k, v)
    oracle = attention_fp32_oracle(q, k, v)  # non-causal, M=1
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"decode {shape}")


@pytest.mark.correctness
def test_decode_matches_prefill_kernel(device):
    """Decode over N keys must match the prefill kernel fed the same M=1 query."""
    from tests.tolerances import cos_sim

    q, k, v = _qkv_decode((2, 8, 2, 1536, 128), device)
    out_dec = superl8.attn_int8_decode(q, k, v)
    out_pre = superl8.attn_int8_fwd(q, k, v)  # M=1 prefill path, same quantization
    assert cos_sim(out_dec, out_pre) >= 0.999, f"cos {cos_sim(out_dec, out_pre):.5f}"


@pytest.mark.correctness
def test_decode_deterministic(device):
    q, k, v = _qkv_decode((2, 8, 8, 1024, 64), device)
    r0 = superl8.attn_int8_decode(q, k, v)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_decode(q, k, v), r0)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", DECODE_SHAPES)
def test_decode_cached_quality(device, shape):
    """INT8 KV-cache decode (quantize once) must hold the int8 accuracy bars."""
    q, k, v = _qkv_decode(shape, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v)
    out = superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale)
    oracle = attention_fp32_oracle(q, k, v)
    assert out.shape == q.shape
    assert_finite(out)
    # int8 V adds a little error over fp16 V -> slightly looser rel-L1 bar.
    assert_int8_quality(out, oracle, what=f"decode_cached {shape}",
                        min_cos=0.998, max_rel_l1=0.03, min_sqnr_db=18.0)


@pytest.mark.correctness
def test_decode_cached_close_to_fp16v(device):
    """int8-V cache must track the fp16-V decode closely (only quant error apart)."""
    from tests.tolerances import cos_sim

    q, k, v = _qkv_decode((2, 8, 2, 1536, 128), device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v)
    out_cached = superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale)
    out_fp16v = superl8.attn_int8_decode(q, k, v)
    assert cos_sim(out_cached, out_fp16v) >= 0.998, f"cos {cos_sim(out_cached, out_fp16v):.5f}"


@pytest.mark.correctness
def test_decode_cached_deterministic(device):
    q, k, v = _qkv_decode((2, 8, 8, 1024, 64), device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v)
    r0 = superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale), r0)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", DECODE_D256_SHAPES)
def test_decode_d256_quality(device, shape):
    q, k, v = _qkv_decode(shape, device)
    out = superl8.attn_int8_decode(q, k, v)
    oracle = attention_fp32_oracle(q, k, v)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(out, oracle, what=f"decode d256 {shape}")


@pytest.mark.correctness
def test_decode_d256_matches_prefill_kernel(device):
    """Decode over N keys must match the D=256 dynamic-smem prefill kernel fed
    the same M=1 query (same quantization prologue, different launch)."""
    from tests.tolerances import cos_sim

    q, k, v = _qkv_decode((2, 8, 2, 640, 256), device)
    out_dec = superl8.attn_int8_decode(q, k, v)
    out_pre = superl8.attn_int8_fwd(q, k, v)  # M=1 prefill path, same quantization
    assert cos_sim(out_dec, out_pre) >= 0.999, f"cos {cos_sim(out_dec, out_pre):.5f}"


@pytest.mark.correctness
def test_decode_d256_deterministic(device):
    q, k, v = _qkv_decode((2, 8, 8, 1024, 256), device)
    r0 = superl8.attn_int8_decode(q, k, v)
    assert_finite(r0)
    for _ in range(3):
        assert torch.equal(superl8.attn_int8_decode(q, k, v), r0)


@pytest.mark.correctness
@pytest.mark.parametrize("shape", DECODE_D256_SHAPES)
def test_decode_d256_cached_quality(device, shape):
    """INT8 KV-cache decode (quantize once) must also generalize to D=256."""
    q, k, v = _qkv_decode(shape, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v)
    out = superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale)
    oracle = attention_fp32_oracle(q, k, v)
    assert out.shape == q.shape
    assert_finite(out)
    assert_int8_quality(
        out,
        oracle,
        what=f"decode_cached d256 {shape}",
        min_cos=0.998,
        max_rel_l1=0.03,
        min_sqnr_db=18.0,
    )


@pytest.mark.perf
def test_decode_d256_perf(device):
    """D=256 decode is correctness-first (register pressure doubles: CH=8 output
    channels, QCH=2 QK dp4a-chunks/lane vs 1); measure honestly vs prefill-at-M=1.
    A perf-tuning pass (e.g. 2 warps/block) is a documented follow-up."""
    b, hq, hkv, n = 2, 8, 2, 4096
    q, k, v = _qkv_decode((b, hq, hkv, n, 256), device)
    dec = time_ms(lambda: superl8.attn_int8_decode(q, k, v))
    pre = time_ms(lambda: superl8.attn_int8_fwd(q, k, v))
    print(f"\nD=256 decode {dec:.3f} ms | prefill_M1 {pre:.3f} ms | {pre / dec:.2f}x")
    assert dec == dec and dec > 0  # runs, finite; perf-tuning is a documented follow-up


# Qwen3.6-27B full-attention decode shape: hidden 5120, Hq=24, Hkv=4 (GQA-6),
# head_dim=256, single-card autoregressive decode (B=1). At this THIN grid
# (bhq = B*Hq = 24, one warp/block) the split-KV kernel is latency/occupancy-
# bound, so `choose_n_splits` must open MANY splits to fill the SMs — far more
# than the high-batch regime needs. This gate asserts the AUTO split heuristic
# lands near the best split count (ratio, so it is fleet-independent), i.e. that
# the heuristic is actually tuned for the low-batch D=256 shape and not just the
# high-batch D=128 one. See utils/docs/decode-profiling.md.
@pytest.mark.perf
@pytest.mark.parametrize("shape", [(1, 24, 4, 4096, 256), (1, 24, 4, 8192, 256)])
def test_decode_d256_autosplit_near_optimal(device, shape):
    b, hq, hkv, n, d = shape
    q, k, v = _qkv_decode(shape, device)
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v)

    def run(ns):
        return time_ms(
            lambda: superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale, num_splits=ns)
        )

    auto = run(None)
    sweep = {ns: run(ns) for ns in (16, 32, 48, 64, 96, 128, 192)}
    best_ns = min(sweep, key=sweep.get)
    best = sweep[best_ns]
    print(
        f"\n27B decode {shape}: auto={auto:.4f} ms  best=splits{best_ns}({best:.4f} ms)"
        f"  auto/best={auto / best:.2f}x"
    )
    # The auto heuristic must be within 20% of the best explicit split count — it
    # was ~1.5-1.8x off before the low-batch retune of choose_n_splits.
    assert auto <= 1.20 * best, (
        f"auto num_splits is {auto / best:.2f}x the best (splits={best_ns}); "
        f"choose_n_splits under-fills the SMs at the thin B=1 D=256 grid"
    )


@pytest.mark.perf
def test_decode_speed_scaling(device):
    """Characterise decode latency vs cache length N — decode is memory-bound on
    the KV read, so latency scales ~linearly with N and the int8 cache (half the
    V bytes) is consistently faster. This is the empirical basis for projecting
    sub-int8 configs (which scale ~with cache bytes) until their kernels exist.
    """
    b, hq, hkv, d = 8, 32, 8, 128
    print(f"\n decode latency vs N (B{b} Hq{hq} Hkv{hkv} D{d}):")
    print(f" {'N':>6} {'fp16V ms':>9} {'int8KV ms':>10} {'int8 speedup':>13} {'int8KV GB/s':>12}")
    prev = None
    for n in (1024, 2048, 4096, 8192):
        q, k, v = _qkv_decode((b, hq, hkv, n, d), device)
        k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v)
        fp16v = time_ms(lambda: superl8.attn_int8_decode(q, k, v))
        i8kv = time_ms(lambda: superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale))
        # int8 KV bytes read: (K int8 + V int8) over N keys, per KV head.
        kv_bytes = 2 * b * hkv * n * d
        gbps = kv_bytes / (i8kv * 1e-3) / 1e9
        print(f" {n:>6} {fp16v:>9.3f} {i8kv:>10.3f} {fp16v / i8kv:>12.2f}x {gbps:>11.1f}")
        assert i8kv < fp16v, f"int8 KV decode must beat fp16-V at N={n}"
        prev = i8kv
    assert prev is not None


@pytest.mark.perf
@pytest.mark.parametrize("shape", [(8, 32, 8, 4096, 128), (16, 32, 8, 2048, 64)])
def test_decode_perf(device, shape):
    """Decode split-KV must beat the prefill kernel at M=1 (its whole purpose);
    the INT8 KV cache must beat the fp16-V decode (half the V read bandwidth)."""
    b, hq, hkv, n, d = shape
    q, k, v = _qkv_decode(shape, device)
    kr, vr = (t.repeat_interleave(hq // hkv, dim=1) for t in (k, v))  # GQA -> dense for SDPA
    k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v)
    dec = time_ms(lambda: superl8.attn_int8_decode(q, k, v))
    cached = time_ms(lambda: superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale))
    pre = time_ms(lambda: superl8.attn_int8_fwd(q, k, v))
    sdpa = time_ms(lambda: sdpa_fp16(q, kr, vr))
    name = f"attn_decode.b{b}h{hq}kv{hkv}n{n}d{d}"
    print("\n" + compare_report(name, dec, {"prefill_M1": pre, "sdpa_fp16": sdpa}))
    print(compare_report(name + ".kv8", cached, {"fp16v_decode": dec, "sdpa_fp16": sdpa}))
    assert dec < pre, f"decode ({dec:.3f} ms) must beat prefill-at-M=1 ({pre:.3f} ms)"
    assert cached < dec, f"int8-KV decode ({cached:.3f} ms) must beat fp16-V decode ({dec:.3f} ms)"
