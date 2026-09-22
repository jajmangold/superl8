# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Transport-compression measurement — is it worth compressing the wire, and how?

The fleet is PCIe 1.0 x1 (~250 MB/s), ~3300x below the 829 GB/s HBM, so every
multi-GPU collective is wire-bound. This harness quantifies the transport codec
(superl8.transport) on REAL transformer activations (the residual-stream tensor a
pipeline-parallel boundary actually ships), producing the ratio-vs-quality-vs-
effective-bandwidth table, plus a parallelism-cost readout that shows WHY only
pipeline (and partly MoE) parallelism is viable on this link.

Run in the bench service (real V100). Network is needed once to fetch the model;
falls back to a synthetic outlier-heavy activation if the model can't load.
"""
import subprocess
import sys

try:
    import transformers  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "transformers>=4.44", "accelerate", "safetensors"])

import torch

import superl8
from bench.harness import time_ms
from superl8.transport import PCIE1_X1_BYTES_PER_S, effective_transfer_ms

MODEL = "Qwen/Qwen2-0.5B"
DEVICE = "cuda"
TEXT = ("The history of science is the study of the development of science. " * 40)

SCHEMES = [
    ("fp16", None), ("int8", None),
    ("int4", 128), ("int4", 64), ("int4-had", 128), ("nf4", 128), ("nf4", 64),
]


def capture_activation():
    """Grab a real residual-stream activation (a transformer block's input) — the
    exact tensor a PP boundary transfers. Fall back to a synthetic outlier-heavy
    activation if the model is unavailable."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").to(DEVICE).eval()
        ids = tok(TEXT, return_tensors="pt").input_ids.to(DEVICE)
        grab = {}
        layer = model.model.layers[len(model.model.layers) // 2]
        h = layer.register_forward_hook(lambda m, i, o: grab.setdefault("x", i[0].detach()))
        with torch.no_grad():
            model(ids)
        h.remove()
        x = grab["x"]  # [1, S, hidden]
        return x, f"real {MODEL} block-input {tuple(x.shape)}"
    except Exception as e:  # noqa: BLE001
        print(f"[fallback] model unavailable ({type(e).__name__}); synthetic activation")
        x = torch.randn(1, 512, 896, device=DEVICE, dtype=torch.float16)
        chan = torch.zeros(896, device=DEVICE, dtype=torch.float16)
        chan[::64] = 10.0
        return x + chan, "synthetic outlier activation (1,512,896)"


def _table(x, label):
    print(f"\n[activation] {label}")
    print(f"  shape {tuple(x.shape)}  outlier ratio (amax/mean) = "
          f"{(x.abs().amax()/x.abs().mean()).item():.1f}")
    raw_ms = effective_transfer_ms(x.numel() * 2)
    print(f"  raw fp16 wire time over PCIe-1.0-x1 (250 MB/s) = {raw_ms:.2f} ms "
          f"({x.numel()*2/1e6:.2f} MB)")
    hdr = f"{'scheme':>10} {'gsz':>4} | {'ratio':>6} {'+entropy':>8} | {'cos':>7} {'relL1':>7} | " \
          f"{'comp ms':>8} {'decomp':>7} | {'wire ms':>8} {'speedup':>8}"
    print(hdr); print("-" * len(hdr))
    for scheme, gs in SCHEMES:
        c = superl8.compress_activation(x, scheme=scheme, group_size=gs)
        rep = superl8.reconstruction_report(x, c)
        comp_ms = time_ms(lambda: superl8.compress_activation(x, scheme=scheme, group_size=gs))
        decomp_ms = time_ms(lambda: superl8.decompress_activation(c))
        wire_ms = effective_transfer_ms(c.on_wire_bytes)
        speedup = raw_ms / (comp_ms + wire_ms + decomp_ms)
        print(f"{scheme:>10} {str(gs):>4} | {rep['ratio']:6.2f} "
              f"{rep['ratio_with_entropy']:8.2f} | {rep['cos']:7.4f} {rep['rel_l1']:7.4f} | "
              f"{comp_ms:8.3f} {decomp_ms:7.3f} | {wire_ms:8.2f} {speedup:7.2f}x")


def main():
    torch.manual_seed(0)
    x, desc = capture_activation()
    _table(x, desc + " (small: codec fixed-overhead visible)")
    # A real PP-boundary-sized tensor: seq 4096 x hidden 4096 fp16 = 33.5 MB. Here the
    # codec amortizes and the measured speedup approaches the compression ratio.
    big = torch.randn(1, 4096, 4096, device=DEVICE, dtype=torch.float16)
    ch = torch.zeros(4096, device=DEVICE, dtype=torch.float16); ch[::64] = 40.0
    _table(big + ch, "synthetic PP-boundary tensor 33.5 MB (bulk: codec amortized)")

    # ---- parallelism cost readout: WHOLE-MICROBATCH wire time, per-GPU-egress basis ----
    print("\n[parallelism cost on PCIe-1.0-x1] per-microbatch wire time (per-GPU egress):")
    S, H, L = 4096, 4096, 32  # a 7B-ish block: seq(=tokens), hidden, layers
    act = S * H * 2                                   # one fp16 activation tensor
    k, G = 2, 8                                       # MoE: top-k experts, expert-parallel GPUs
    remote = 1.0 - 1.0 / G
    # MoE all-to-all (dispatch+combine), per-GPU egress: each GPU owns S/G tokens, sends
    # each to k experts, (1-1/G) of which are remote; combine returns k outputs. Every
    # FFN is MoE here (Mixtral-like). Attention layers need NO expert comm.
    moe = int(2 * (S / G) * k * H * 2 * remote * L)  # bytes / GPU / microbatch
    print(f"  model proxy: seq={S} hidden={H} layers={L}   MoE: top-k={k}, expert-parallel G={G}")
    for name, bytes_moved, note in [
        ("PP boundary (1 activation / microbatch)", act, "P2P, once per stage boundary"),
        ("MoE all-to-all (dispatch+combine / GPU)", moe, f"only routed tokens, {remote:.0%} remote"),
        ("TP all-reduce (2 / layer x L, ~2x ring)", 2 * act * 2 * L, "per-layer, every layer"),
        ("FSDP all-gather params (~2*H*H*L)", 2 * H * H * 2 * L, "per-layer param gather"),
    ]:
        raw = effective_transfer_ms(bytes_moved)
        i4 = effective_transfer_ms(bytes_moved // 4)  # ~4x codec
        print(f"  {name:<42} raw {raw/1e3:8.2f} s | int4 {i4/1e3:8.2f} s  ({note})")
    # MoE scaling: per-GPU egress ~ (S/G)(1-1/G) -> shrinks ~1/G as you ADD cards.
    print("\n  MoE per-GPU egress vs expert-parallel degree G (int4, whole microbatch):")
    for g in (2, 4, 8, 16, 32):
        m = int(2 * (S / g) * k * H * 2 * (1 - 1 / g) * L) // 4
        print(f"    G={g:>2}: {effective_transfer_ms(m)/1e3:6.2f} s/GPU  "
              f"({'more cards -> LESS per-card traffic' if g == 8 else ''})")
    print("\n  => PP and MoE are the survivors. MoE moves only routed tokens (k*H, not the")
    print("     full-activation ring), params never move, and per-card traffic DROPS ~1/G as")
    print("     you scale — the one paradigm that gets better with more cards. TP/FSDP stay dead.")
    print(f"\n[link] PCIe 1.0 x1 = {PCIE1_X1_BYTES_PER_S/1e6:.0f} MB/s per direction.")


if __name__ == "__main__":
    main()
