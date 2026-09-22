# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""End-to-end validation on a REAL model — the deliverable that proves superl8 is
usable, not just microbenchmark-correct.

Everything so far was validated on randn tensors, which have NO channel outliers.
Real activations do — that's the whole point of K-smoothing + the Hadamard
rotation. This script:
  1. loads a small GQA model (Qwen2-0.5B: 14 Q heads / 2 KV heads, head_dim 64),
  2. monkeypatches F.scaled_dot_product_attention to route prefill attention
     through superl8's int8 dp4a kernel (model-agnostic — no per-model surgery),
  3. measures PERPLEXITY on real text vs the fp16 baseline (the number that
     actually decides usability), and
  4. captures REAL K activations and measures int8 vs Hadamard-rotated-int8
     error on them — validating the rotation claim on genuine outliers.

Run inside the bench service (real V100). Network is needed once to fetch the
model + transformers.
"""
import subprocess
import sys

try:
    import transformers  # noqa: F401
except ImportError:
    print("[setup] installing transformers...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "transformers>=4.44", "accelerate", "safetensors"])

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import superl8

MODEL = "Qwen/Qwen2-0.5B"
DEVICE = "cuda"

# A fixed real-text sample (public-domain, ~a few hundred tokens) — no dataset
# download needed; perplexity is comparable across runs.
TEXT = (
    "The history of science is the study of the development of science, including "
    "both the natural and social sciences. Science is a body of empirical, theoretical, "
    "and practical knowledge about the natural world, produced by scientists who emphasize "
    "the observation, explanation, and prediction of real-world phenomena. Historiography "
    "of science, in contrast, studies the methods employed by historians of science. "
    "The English word scientist is relatively recent, first coined by William Whewell in "
    "the nineteenth century. Before that, investigators of nature called themselves natural "
    "philosophers. While empirical investigations of the natural world have been described "
    "since antiquity, and the scientific method has been employed since the Middle Ages, "
    "the dawn of modern science is often traced back to the early modern period, during "
    "what is known as the Scientific Revolution that took place in sixteenth- and "
    "seventeenth-century Europe. Scientific methods are considered so fundamental to modern "
    "science that some consider earlier inquiries into nature to be pre-scientific."
) * 2

# ---- the superl8 attention shim (routes the causal prefill SDPA call to dp4a) ----
_orig_sdpa = F.scaled_dot_product_attention
_stats = {"superl8": 0, "fallback": 0}


def _superl8_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
               scale=None, enable_gqa=False, **kw):
    ok = (query.dim() == 4 and query.shape[-1] in (32, 64, 128)
          and query.dtype == torch.float16 and dropout_p == 0.0
          and (is_causal or attn_mask is None))
    if ok:
        _stats["superl8"] += 1
        return superl8.attn_int8_fwd(query.contiguous(), key.contiguous(), value.contiguous(),
                                  causal=is_causal, scale=scale)
    _stats["fallback"] += 1
    return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                      is_causal=is_causal, scale=scale, **kw)


def perplexity(model, ids):
    with torch.no_grad():
        logits = model(ids).logits.float()
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = ids[:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(shift_logits, shift_labels).exp().item()


def main():
    torch.manual_seed(0)
    print(f"[load] {MODEL} (fp16, sdpa)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").to(DEVICE).eval()
    cfg = model.config
    print(f"  heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
          f"head_dim={cfg.hidden_size // cfg.num_attention_heads} layers={cfg.num_hidden_layers}")

    ids = tok(TEXT, return_tensors="pt").input_ids.to(DEVICE)
    print(f"[eval] {ids.shape[1]} tokens")

    ppl_fp16 = perplexity(model, ids)
    print(f"\n  fp16 (SDPA)      perplexity = {ppl_fp16:.4f}")

    F.scaled_dot_product_attention = _superl8_sdpa
    try:
        ppl_superl8 = perplexity(model, ids)
    finally:
        F.scaled_dot_product_attention = _orig_sdpa
    print(f"  superl8 (int8 dp4a) perplexity = {ppl_superl8:.4f}   "
          f"(delta {100 * (ppl_superl8 - ppl_fp16) / ppl_fp16:+.2f}%, "
          f"{_stats['superl8']} attn calls routed, {_stats['fallback']} fell back)")

    # ---- rotation validation on REAL K activations ----
    print("\n[rotation on real activations]", flush=True)
    real_k = {}
    h = model.model.layers[len(model.model.layers) // 2].self_attn.k_proj.register_forward_hook(
        lambda m, i, o: real_k.setdefault("k", o.detach()))
    with torch.no_grad():
        model(ids)
    h.remove()
    from superl8.quant import quantize_int8_rowwise, dequantize_int8_rowwise, smooth_k
    from superl8.quant.rotation import rotate_last
    kv = cfg.num_key_value_heads
    hd = cfg.hidden_size // cfg.num_attention_heads
    k = real_k["k"].reshape(1, ids.shape[1], kv, hd).permute(0, 2, 1, 3).float()  # [1,kv,S,D]
    ks, _ = smooth_k(k)

    def rt_err(x):
        q, s = quantize_int8_rowwise(x)
        return (dequantize_int8_rowwise(q, s) - x).abs().mean().item() / x.abs().mean().item()

    plain = rt_err(ks)
    rotated = rt_err(rotate_last(ks))
    amax_ratio = (k.abs().amax() / k.abs().mean()).item()
    print(f"  real K outlier ratio (amax/mean) = {amax_ratio:.1f}  (randn ~4-5)")
    print(f"  int8 round-trip rel-err: plain {plain:.4f}  vs  rotated {rotated:.4f}  "
          f"({plain / max(rotated, 1e-9):.2f}x better)")

    print("\n[verdict] superl8 int8 attention drives a real GQA model end-to-end; "
          f"perplexity delta {100 * (ppl_superl8 - ppl_fp16) / ppl_fp16:+.2f}% vs fp16.")


if __name__ == "__main__":
    main()
