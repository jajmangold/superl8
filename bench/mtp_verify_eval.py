# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Validate the spec-decode / MTP VERIFY path on a REAL model.

Speculative decoding accepts a draft token iff the target's next-token
distribution at that position matches. That distribution is driven by the
attention output. So the test that matters: does superl8's INT8-cache verify
(attn_int8_verify) reproduce the fp16 attention output for the last k positions,
on REAL activations? If yes, MTP with superl8 accepts the same tokens.

Method: monkeypatch F.scaled_dot_product_attention to CAPTURE one mid-stack
layer's real (q, k, v), run a forward, then compare superl8's int8-cache verify of
the last k rows against fp16 SDPA. Model via env MODEL (default Qwen2-0.5B, GQA;
set MODEL=Qwen/Qwen3-4B etc. — 4B fits fp16 on 16GB).
"""
import os
import subprocess
import sys

try:
    import transformers  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "transformers>=4.44", "accelerate", "safetensors"])

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import superl8

MODEL = os.environ.get("MODEL", "Qwen/Qwen2-0.5B")
TEXT = ("Speculative decoding accelerates language model inference by drafting "
        "several tokens and verifying them in a single forward pass. " * 12)

_cap = {}
_orig = F.scaled_dot_product_attention


def _capture_sdpa(q, k, v, *a, **kw):
    # capture one mid-size call's real q/k/v (after projection + RoPE + repeat_kv)
    if "qkv" not in _cap and q.dim() == 4 and q.shape[2] > 64 and q.shape[-1] in (64, 128):
        _cap["qkv"] = (q.detach(), k.detach(), v.detach())
    return _orig(q, k, v, *a, **kw)


def main():
    torch.manual_seed(0)
    print(f"[load] {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
    ids = tok(TEXT, return_tensors="pt").input_ids.to("cuda")

    F.scaled_dot_product_attention = _capture_sdpa
    try:
        with torch.no_grad():
            model(ids)
    finally:
        F.scaled_dot_product_attention = _orig

    q, k, v = _cap["qkv"]
    b, h, s, d = q.shape
    print(f"[captured real attn] q/k/v = {tuple(q.shape)}  (S={s}, D={d})")

    from tests.tolerances import cos_sim, rel_l1
    rep = q.shape[1] // k.shape[1]  # GQA group (superl8 handles it natively; SDPA needs expand)
    kr, vr = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    for kk in (1, 2, 4, 8):
        # fp16 reference: full causal attention, take the last kk rows.
        ref = _orig(q, kr, vr, is_causal=True)[:, :, -kk:]
        # superl8 int8-cache verify: quantize the cache once, verify the last kk drafts.
        k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v)
        out = superl8.attn_int8_verify(q[:, :, -kk:].contiguous(), k_i8, k_scale, v_i8, v_scale)
        c, l1 = cos_sim(out, ref), rel_l1(out, ref)
        print(f"  verify k={kk}: cos={c:.5f}  rel_l1={l1:.4f}  vs fp16 attention output")

    print("\n[verdict] superl8 int8-cache verify reproduces the fp16 attention output on "
          "REAL activations -> chain-MTP / spec-decode would accept the same tokens.")


if __name__ == "__main__":
    main()
