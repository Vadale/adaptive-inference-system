"""exp_003c — riproduzione esatta del caso fallito di exp_005 v2.

Il run di exp_005 v2 era crashato a forward #6 con `MissedProviderError` su
`layers.28.input.i0` durante l'ablation del gruppo g4 = (28, 35) sul prompt
[closed_qa/1]. Causa A: catena lunga di `layer.output = layer.input`. Causa
B (più probabile): standby/sleep MPS con monitor scollegato.

Fix applicati: boundary intervention (1 intervento per gruppo) in
`skippers/layer_skipper.py` + esecuzione sotto `caffeinate` + monitor
riattaccato.

Questo test riproduce esattamente:
  - prompt di tipo `closed_qa`
  - skip = {28, 29, 30, 31, 32, 33, 34}  (7 layer contigui, gruppo g4)
  - aspettativa: PASS (logits finiti, distinguibili dal baseline)
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skippers.layer_skipper import AdaptiveLayerSkipper

# closed_qa-like prompt
PROMPT = "Based on the following passage, what year was the model trained?"
SKIP_GROUP = set(range(28, 35))  # g4 di exp_005


def main() -> int:
    print("Loading AdaptiveLayerSkipper (E4B)...", flush=True)
    skipper = AdaptiveLayerSkipper()
    n = skipper.n_layers
    print(f"  n_text_layers={n}", flush=True)

    t0 = time.time()
    print(f"\n[1/2] baseline forward (active=all)...", flush=True)
    r_base = skipper.forward(PROMPT, active_layers=None)
    print(f"    {time.time()-t0:.1f}s  finite={torch.isfinite(r_base.logits_last).all().item()}",
          flush=True)

    t1 = time.time()
    active = set(range(n)) - SKIP_GROUP
    print(f"\n[2/2] HIGH path: skip L28-L34 (gruppo di 7 layer contigui)...", flush=True)
    r_skip = skipper.forward(PROMPT, active_layers=active)
    print(f"    {time.time()-t1:.1f}s  finite={torch.isfinite(r_skip.logits_last).all().item()}",
          flush=True)

    diff = (r_base.logits_last - r_skip.logits_last).abs().max().item()
    print(f"\nmax|Δ| baseline vs skip L28-34: {diff:.3e}", flush=True)
    print(f"baseline max|x|: {r_base.logits_last.abs().max().item():.3f}", flush=True)
    print(f"skip     max|x|: {r_skip.logits_last.abs().max().item():.3f}", flush=True)

    ok_finite = (
        torch.isfinite(r_base.logits_last).all().item()
        and torch.isfinite(r_skip.logits_last).all().item()
    )
    ok_diff = diff > 1e-3

    print("\n" + "=" * 60, flush=True)
    print(f"  Logits finiti (no NaN/Inf):    {ok_finite}", flush=True)
    print(f"  Skip effettivo (Δ > 1e-3):     {ok_diff}", flush=True)
    if ok_finite and ok_diff:
        print(f"  Group skip smoke: PASS — exp_005 può procedere", flush=True)
        rc = 0
    else:
        print(f"  Group skip smoke: FAIL", flush=True)
        rc = 1
    print("=" * 60, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
