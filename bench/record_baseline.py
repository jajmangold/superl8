# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Record perf baselines into bench/baseline.json.

Run inside the `bench` service (pinned, real V100) in a dedicated
"update baseline" PR only — see AGENTS.md:

    docker compose run --rm bench python3 bench/record_baseline.py
"""
import datetime
import json

import numpy as np
import torch

import superl8
from bench.harness import BASELINE_PATH, gpu_class, load_baseline

# name -> (callable-builder returning a zero-arg fn, note)
BENCHES = {}


def bench(name, note=""):
    def deco(builder):
        BENCHES[name] = (builder, note)
        return builder

    return deco


@bench("hello_add.4Mi.fp16", note="toolchain smoke op")
def _hello_add():
    a = torch.randn(1 << 22, device="cuda", dtype=torch.float16)
    b = torch.randn(1 << 22, device="cuda", dtype=torch.float16)
    return lambda: superl8.hello_add(a, b)


# Canonical attention shapes (B, H, M, D) for the comparison ladder. The dp4a
# kernel (PR3+) is reported against these fp16 numbers — honestly, win or lose.
ATTN_SHAPES = [(2, 16, 2048, 64), (2, 16, 2048, 128), (1, 32, 4096, 64)]


def _qkv(b, h, m, d):
    return [torch.randn(b, h, m, d, device="cuda", dtype=torch.float16) for _ in range(3)]


def _register_attention_benches():
    import torch.nn.functional as F

    for b, h, m, d in ATTN_SHAPES:
        for causal in (False, True):
            tag = f"b{b}h{h}m{m}d{d}{'.causal' if causal else ''}"

            @bench(f"sdpa.{tag}.fp16", note="torch SDPA fp16 baseline")
            def _sdpa(b=b, h=h, m=m, d=d, causal=causal):
                q, k, v = _qkv(b, h, m, d)
                return lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    try:
        from flash_attn_v100 import flash_attn_func

        for b, h, m, d in ATTN_SHAPES:
            for causal in (False, True):
                tag = f"b{b}h{h}m{m}d{d}{'.causal' if causal else ''}"

                @bench(f"flash_v100.{tag}.fp16", note="ai-bond fp16 FA2 baseline")
                def _fa(b=b, h=h, m=m, d=d, causal=causal):
                    # flash_attn_v100 native layout is [B, M, H, D]
                    q, k, v = [t.permute(0, 2, 1, 3).contiguous() for t in _qkv(b, h, m, d)]
                    return lambda: flash_attn_func(q, k, v, causal=causal)
    except ImportError:
        print("flash_attn_v100 not installed — skipping its baselines")


def _register_superl8_benches():
    for b, h, m, d in [(2, 16, 2048, 64), (2, 16, 2048, 128)]:

        @bench(f"attn_int8_fwd.b{b}h{h}m{m}d{d}", note="superl8 int8 dp4a QK^T, fp16 PV")
        def _superl8(b=b, h=h, m=m, d=d):
            q, k, v = _qkv(b, h, m, d)
            return lambda: superl8.attn_int8_fwd(q, k, v)

        @bench(f"attn_w8a8_fwd.b{b}h{h}m{m}d{d}", note="superl8 full W8A8 (int8 QK + int8 PV)")
        def _superl8_w8a8(b=b, h=h, m=m, d=d):
            q, k, v = _qkv(b, h, m, d)
            return lambda: superl8.attn_int8_fwd(q, k, v, int8_pv=True)

        @bench(f"attn_bwd.b{b}h{h}m{m}d{d}", note="superl8 fused fp backward (PR5b-1; dp4a=PR5b-2)")
        def _superl8_bwd(b=b, h=h, m=m, d=d):
            q, k, v = _qkv(b, h, m, d)
            o = superl8.attn_int8_fwd(q, k, v)
            do = torch.randn_like(o)
            return lambda: superl8.backward_cuda(q, k, v, o, None, do)


def _register_gemm_benches():
    from superl8.quant.core import quantize_int8_rowwise
    from superl8.quant.lowbit import quantize_lowbit

    def _w8(n, k):
        w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.1
        q, s = quantize_int8_rowwise(w)
        return q.contiguous(), s.squeeze(-1).contiguous()

    def _w4(n, k, g):
        w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.1
        codes, scale = quantize_lowbit(w, 4, dim=-1, group_size=g)
        c = codes.to(torch.int64)
        packed = ((c[:, 0::2] & 0xF) | ((c[:, 1::2] & 0xF) << 4)).to(torch.uint8).contiguous()
        return packed, scale.contiguous()

    def _xq(m, k):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        xq, xs = quantize_int8_rowwise(x)
        return xq, xs.squeeze(-1).contiguous()

    # FFN up-proj shapes: decode (M=8) and prefill (M=2048), Qwen2-ish N=4864/K=896.
    for m, tag in [(8, "m8"), (2048, "m2048")]:
        n, k = 4864, 896

        @bench(f"gemm_w8a8.{tag}n4864k896", note="superl8 int8 dp4a GEMM (linear layer)")
        def _g8(m=m, n=n, k=k):
            xq, xs = _xq(m, k)
            wq, ws = _w8(n, k)
            return lambda: superl8._C.gemm_w8a8(xq, xs, wq, ws)

    @bench("gemm_w4a8.m8n4864k896", note="superl8 W4A8 dp4a GEMM (4-bit weights, g128)")
    def _g4():
        xq, xs = _xq(8, 896)
        packed, scale = _w4(4864, 896, 128)
        return lambda: superl8._C.gemm_w4a8(xq, xs, packed, scale, 128)


def _register_kquant_benches():
    """Fused GGUF k-quant dp4a GEMMs (native, no dequant). LTX DiT decode (M=1)
    and prefill/batched (M=2048) at N=4096/K=3072. Byte values are irrelevant to
    timing, so a random valid-size uint8 blob is used."""
    from superl8.quant.core import quantize_int8_rowwise

    def _xq(m, k):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        xq, xs = quantize_int8_rowwise(x)
        return xq, xs.squeeze(-1).contiguous()

    for op, ts, short in [("gemm_q4k", 144, "q4k"), ("gemm_q5k", 176, "q5k"),
                          ("gemm_q6k", 210, "q6k"), ("gemm_q3k", 110, "q3k"),
                          ("gemm_q2k", 84, "q2k")]:
        if not hasattr(superl8._C, op):
            continue
        for m, mt in [(1, "m1"), (2048, "m2048")]:
            n, k = 4096, 3072

            @bench(f"{op}.{mt}n4096k3072", note=f"superl8 fused GGUF {short.upper()} dp4a GEMM (native)")
            def _g(op=op, ts=ts, m=m, n=n, k=k):
                xq, xs = _xq(m, k)
                w = torch.randint(0, 256, (n, (k // 256) * ts), device="cuda", dtype=torch.uint8)
                fn = getattr(superl8._C, op)
                return lambda: fn(xq, xs, w, torch.float16)

    # Warp-per-column MMVQ decode kernels (the M=1 GPU-saturating path).
    for op, ts, short in [("gemm_decode_q4k", 144, "q4k"), ("gemm_decode_q5k", 176, "q5k"),
                          ("gemm_decode_q6k", 210, "q6k"), ("gemm_decode_q3k", 110, "q3k"),
                          ("gemm_decode_q2k", 84, "q2k")]:
        if not hasattr(superl8._C, op):
            continue
        n, k = 4096, 3072

        @bench(f"{op}.m1n4096k3072", note=f"superl8 fused GGUF {short.upper()} MMVQ decode (M=1)")
        def _gd(op=op, ts=ts, n=n, k=k):
            xq, xs = _xq(1, k)
            w = torch.randint(0, 256, (n, (k // 256) * ts), device="cuda", dtype=torch.uint8)
            fn = getattr(superl8._C, op)
            return lambda: fn(xq, xs, w, torch.float16)


def _register_tq34s_benches():
    """Fused TQ3_4S kernels at the Qwen3.8-27B linear shapes (superl8#281 perf
    pass). Names MUST match the perf markers in tests/test_gemm_tq34s.py so the
    marker un-skips after recording. Decode (M=1) = warp-per-column MMVQ;
    prefill (M=2048) = the tile. Byte values are irrelevant to timing."""
    if not hasattr(superl8._C, "gemm_tq34s"):
        return

    def _blk(n, k, seed):
        rng = np.random.default_rng(seed)
        return torch.from_numpy(
            rng.integers(0, 256, (n, (k // 32) * 16), dtype=np.uint8)
        ).cuda()

    for tag, n, k in [
        ("attn.m1n5120k5120", 5120, 5120),
        ("gate_up.m1n17408k5120", 17408, 5120),
        ("down.m1n5120k17408", 5120, 17408),
    ]:

        @bench(f"gemm_decode_tq34s.{tag}", note="superl8 fused TQ3_4S MMVQ decode (M=1)")
        def _gd(tag=tag, n=n, k=k):
            x = torch.randn(1, k, device="cuda", dtype=torch.float16)
            blk = _blk(n, k, 0)
            return lambda: superl8._C.gemm_decode_tq34s(x, blk, torch.float16)

    for tag, n, k in [
        ("prefill.attn.m2048n5120k5120", 5120, 5120),
        ("prefill.gate_up.m2048n17408k5120", 17408, 5120),
    ]:

        @bench(f"gemm_tq34s.{tag}", note="superl8 fused TQ3_4S tile GEMM (M=2048)")
        def _gt(tag=tag, n=n, k=k):
            x = torch.randn(2048, k, device="cuda", dtype=torch.float16)
            blk = _blk(n, k, 1)
            return lambda: superl8._C.gemm_tq34s(x, blk, k, torch.float16)


_register_attention_benches()
_register_superl8_benches()
_register_gemm_benches()
_register_kquant_benches()
_register_tq34s_benches()


def main():
    import sys

    from bench.harness import time_ms

    # Optional argv filters: record only benches whose name contains any token
    # (keeps committed baselines untouched when adding a new kernel's numbers).
    filters = sys.argv[1:]
    data = load_baseline()
    cls = gpu_class()
    dev = torch.cuda.get_device_name()
    entry = data.setdefault(cls, {})
    for name, (builder, note) in BENCHES.items():
        if filters and not any(f in name for f in filters):
            continue
        ms = time_ms(builder())
        entry[name] = {
            "median_ms": round(ms, 4),
            "recorded": datetime.date.today().isoformat(),
            "device": dev,
            "note": note,
        }
        print(f"{cls}/{name}: {ms:.4f} ms  ({dev})")
    BASELINE_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {BASELINE_PATH}")


if __name__ == "__main__":
    main()
