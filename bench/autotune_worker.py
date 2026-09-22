# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Autotune worker — gates correctness and times the backward for ONE compiled
config, prints a JSON line. Run as a fresh subprocess per config so it imports
the just-rebuilt superl8._C. Invoked by bench/autotune.py.

    python3 -m bench.autotune_worker
"""
import json
import sys

import torch

import superl8
from bench.harness import time_ms
from tests.reference import attention_fp32_oracle
from tests.tolerances import cos_sim, rel_l1

# Correctness bars (match tests/test_attn_bwd_cuda.py). A config that misses is
# INVALID — pruned by the search, never allowed to "win" on speed.
BARS = dict(min_cos=0.99, max_rel_l1=0.06)
GATE_SHAPES = [(1, 2, 256, 64), (1, 2, 192, 128)]
TIME_SHAPES = [(2, 16, 2048, 64), (2, 16, 2048, 128)]


def _grads_ok(shape, causal):
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3))
    o = superl8.attn_int8_fwd(q, k, v, causal=causal)
    d_out = torch.randn_like(o)
    dq, dk, dv = superl8.backward_cuda(q, k, v, o, None, d_out, causal=causal)
    qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    ref = attention_fp32_oracle(qf, kf, vf, causal=causal)
    ref.backward(d_out.float())
    for got, exp in [(dq, qf.grad), (dk, kf.grad), (dv, vf.grad)]:
        if cos_sim(got, exp) < BARS["min_cos"] or rel_l1(got, exp) > BARS["max_rel_l1"]:
            return False
    return True


def main():
    try:
        for shape in GATE_SHAPES:
            for causal in (False, True):
                if not _grads_ok(shape, causal):
                    print(json.dumps({"ok": False, "reason": f"gate {shape} c={causal}"}))
                    return
        times = {}
        for b, h, m, d in TIME_SHAPES:
            q, k, v = (torch.randn(b, h, m, d, device="cuda", dtype=torch.float16) for _ in range(3))
            o = superl8.attn_int8_fwd(q, k, v)
            do = torch.randn_like(o)
            times[f"d{d}"] = round(time_ms(lambda: superl8.backward_cuda(q, k, v, o, None, do)), 4)
        # objective: sum of the two shape latencies (lower is better)
        print(json.dumps({"ok": True, "ms": times, "objective": sum(times.values())}))
    except Exception as e:  # OOM, launch failure, bad config -> invalid
        print(json.dumps({"ok": False, "reason": f"{type(e).__name__}: {str(e)[:120]}"}))


if __name__ == "__main__":
    sys.exit(main())
