# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""N-gram draft acceptance + tok/s harness (superl8#282).

Measures the workload-dependent quantity the design doc says is unmeasured: how
many n-gram draft tokens the target model would ACCEPT per verification pass on
the REAL agentic traffic mix (JSON tool-call / structured output / code) vs novel
prose. Each verified draft costs ONE weight pass, so per-stream tok/s ~= base
decode tok/s x (accepted tokens per pass).

Method (self-consistent, greedy target):
  1. Tokenize real samples with the Qwen3.5 tokenizer.
  2. Walk each stream spec-decode style: the NgramDraftStore proposes up to k
     drafts from the context+generated-so-far; a draft token is "accepted" iff it
     equals the actual continuation (what a greedy target would emit). Advance by
     1 + accepted (base token always emitted).
  3. Report accepted tokens/step, acceptance rate, and the tok/s multiplier.
  4. Base decode tok/s is measured with the superl8 attention decode kernel on a
     pinned quiet GPU (FNI8_VALIDATION_GPU_UUID identity; never the serving
     cards). tok/s with n-gram on = base x (1 + mean accepted/step).

Usage (superl8 dev image, pinned free GPU):
    FNI8_NGRAM_TOKENIZER=/mnt/24tb/qwen35-08b/tok/tokenizer.json \
      python bench/ngram_acceptance.py
Env:
  FNI8_NGRAM_TOKENIZER : Qwen3.5 tokenizer.json path (required for real tokens;
                         a simple char-fallback is used if unset, for smoke runs).
  FNI8_NGRAM_SAMPLES   : sample dir (default bench/samples).
  FNI8_NGRAM_K         : max draft length (default 8).
  FNI8_NGRAM_MIN_N/MAX_N : n-gram match window (default 2/3).
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import torch

import superl8
from bench.harness import time_ms

SAMPLES_DIR = Path(os.environ.get("FNI8_NGRAM_SAMPLES", Path(__file__).parent / "samples"))
TOKENIZER_PATH = os.environ.get("FNI8_NGRAM_TOKENIZER", "")
K = int(os.environ.get("FNI8_NGRAM_K", "8"))
MIN_N = int(os.environ.get("FNI8_NGRAM_MIN_N", "2"))
MAX_N = int(os.environ.get("FNI8_NGRAM_MAX_N", "3"))


def _tokenizer():
    """Load the Qwen3.5 tokenizer (tokenizers lib) or a char-level fallback."""
    if TOKENIZER_PATH and Path(TOKENIZER_PATH).exists():
        from tokenizers import Tokenizer

        return ("qwen3.5", Tokenizer.from_file(TOKENIZER_PATH))
    # Char-level fallback: keeps the harness runnable on a CPU-only checkout for
    # a smoke run; the real acceptance numbers need the Qwen tokenizer.
    class _CharTok:
        def encode(self, text: str):
            return type("E", (), {"ids": [ord(c) for c in text]})  # noqa: E741

        def decode(self, ids) -> str:
            return "".join(chr(i) for i in ids)

    return ("char", _CharTok())


def _samples() -> dict[str, list[str]]:
    out = {}
    for path in sorted(SAMPLES_DIR.glob("*")):
        if path.suffix not in (".json", ".py", ".md", ".txt", ".jsonl"):
            continue
        text = path.read_text()
        if path.suffix == ".json":
            try:
                text = json.dumps(json.loads(text))
            except ValueError:
                pass
        out.setdefault(_bucket(path.name), []).append(text)
    return out


def _bucket(name: str) -> str:
    if name.startswith("agentic"):
        return "agentic (repetitive)"
    return "novel prose"


def walk_acceptance(ids: list[int], *, k: int = K, min_n: int = MIN_N, max_n: int = MAX_N):
    """Spec-decode walk of a real token stream with the NgramDraftStore.

    Returns per-step stats: steps, drafts proposed, tokens drafted, tokens
    accepted (matching the greedy continuation), and mean accepted drafts/step.
    """
    from superl8.ngram import NgramDraftStore

    store = NgramDraftStore(min_n=min_n, max_n=max_n, max_k=k)
    n_steps = n_drafts = n_draft_tokens = n_accepted = 0
    pos = 0
    while pos < len(ids):
        draft = store.propose(k=k)
        if draft:
            n_drafts += 1
            n_draft_tokens += len(draft)
            accepted = 0
            for j, tok in enumerate(draft):
                if pos + j < len(ids) and ids[pos + j] == tok:
                    accepted += 1
                else:
                    break
            n_accepted += accepted
            advance = 1 + accepted
        else:
            advance = 1
        store.add(ids[pos:pos + advance])
        pos += advance
        n_steps += 1
    mean_acc = n_accepted / n_steps if n_steps else 0.0
    return {
        "steps": n_steps,
        "drafts": n_drafts,
        "draft_tokens": n_draft_tokens,
        "accepted": n_accepted,
        "accept_rate": n_accepted / n_draft_tokens if n_draft_tokens else 0.0,
        "mean_accepted_step": mean_acc,
        "multiplier": 1.0 + mean_acc,
    }


def base_decode_tok_s() -> float:
    """superl8 attention decode tok/s on the pinned GPU (quiet artifact).

    Realistic short-ctx decode shape (B=1, H_q=32, H_kv=8, head_dim 128, N=4096).
    The 13.7 GB weight pass is NOT here (that's the serving lane); this is the
    superl8 attention-kernel base, and the n-gram multiplier applies to whatever
    base the serving lane measures.
    """
    if not torch.cuda.is_available():
        return 0.0
    if not hasattr(superl8, "_C") or type(superl8._C).__name__ == "_MissingC":
        print("[warn] superl8._C not built — skipping base-decode tok/s", flush=True)
        return 0.0
    b, hq, hkv, n, d = 1, 32, 8, 4096, 128
    q = torch.randn(b, hq, 1, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, hkv, n, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, hkv, n, d, device="cuda", dtype=torch.float16)
    ms = time_ms(lambda: superl8.attn_int8_decode(q, k, v), warmup=10, iters=50)
    return 1000.0 / ms


def main():
    print(f"[superl8 n-gram acceptance] k={K} n={MIN_N}..{MAX_N} tokenizer={TOKENIZER_PATH or 'char-fallback'}")
    tok_name, tok = _tokenizer()
    samples = _samples()
    rows = []
    for bucket, texts in samples.items():
        for i, text in enumerate(texts):
            enc = tok.encode(text)
            ids = enc.ids if hasattr(enc, "ids") else list(enc)
            s = walk_acceptance(list(ids))
            rows.append((bucket, i, text, s))
    base = base_decode_tok_s()

    # The design doc's base is the full-model serving number (61 tok/s short-ctx
    # TP=1); the superl8 attention-kernel base measured here is the kernel component.
    # The n-gram MULTIPLIER is workload-dependent and applies to either base.
    serving_base = 61.0

    print("\n=== ACCEPTANCE TABLE (real traffic mix) ===")
    print(f"{'sample':<38} {'steps':>6} {'drafts':>6} {'acc':>6} {'rate':>6} "
          f"{'acc/step':>8} {'mult':>5} {'tok/s_on':>8}")
    for bucket, i, text, s in rows:
        tok_on = base * s["multiplier"] if base else 0.0
        tag = f"{bucket[:20]} #{i}"
        print(f"{tag:<38} {s['steps']:>6} {s['drafts']:>6} {s['accepted']:>6} "
              f"{s['accept_rate']:>6.2f} {s['mean_accepted_step']:>8.2f} "
              f"{s['multiplier']:>5.2f} {tok_on:>8.0f}")
    if base:
        print(f"\nbase decode tok/s (superl8 attention kernel, pinned GPU) = {base:.0f}")
        print("tok/s with n-gram ON = base x multiplier per row (one weight pass per draft).")
    else:
        print("\n[CUDA unavailable] base decode tok/s not measured (CPU smoke run).")

    # Aggregate: mean across samples per bucket.
    by_bucket = {}
    for bucket, _, _, s in rows:
        by_bucket.setdefault(bucket, []).append(s["mean_accepted_step"])
    print("\n=== SUMMARY (mean accepted drafts/step) ===")
    for bucket, vals in by_bucket.items():
        m = statistics.mean(vals)
        tok_on = base * (1 + m) if base else 0.0
        # design-doc projection: full-model serving base 61 x the same multiplier.
        serve_on = serving_base * (1 + m)
        print(f"  {bucket:<28} acc/step {m:6.2f}  multiplier {1+m:5.2f}"
              + (f"  attn tok/s on {tok_on:7.0f} / off {base:5.0f}" if base else "")
              + f"  | serving-base61 on {serve_on:6.0f} / off {serving_base:.0f}")

    # Token stream provenance (real agentic sample).
    print("\n[artifact] samples:", ", ".join(p.name for p in sorted(SAMPLES_DIR.glob('*'))),
          "| tokenizer:", tok_name)
    print("[artifact] GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")


if __name__ == "__main__":
    main()
