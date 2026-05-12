"""exp_003b — HIGH path smoke: verifica che `layer.output = layer.input`
non rompa la tupla output dei layer Gemma 4 (decoder E4B).

`exp_003` ha PASSATO `verify_fallback_identity` (PASS 5/5 max|Δ|=0) ma
testava SOLO il path FALLBACK (`active_layers=None`). Il path HIGH
(`active_layers != all_layers`) attiva l'intervento `layer.output = layer.input`
in `skippers/layer_skipper.py:104`. Il review (2026-05-11) ha segnalato che
se `layer.output` è una tupla `(hidden_states, ...)` e `layer.input` è un
tensor singolo (o tupla diversa), l'assegnazione può rompere il dispatcher
dei layer successivi → logits NaN/degeneri silenziosi.

Test:
  1) Baseline forward (no skip) → logits_base
  2) Skip {0} (salto SOLO il primo layer) → logits_skip_first
  3) Skip {n-1} (salto SOLO l'ultimo layer) → logits_skip_last
  4) Verifica: tutti i logits sono finiti (no NaN/Inf), shape corretta,
     valori distinguibili dal baseline.

Se PASS: il pattern attuale funziona empiricamente.
Se FAIL: serve il fix con `_unwrap(layer.input)` + ricostruzione tupla.
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

PROMPT = "What is the capital of France?"


def main() -> int:
    print("Loading AdaptiveLayerSkipper (E4B)...", flush=True)
    skipper = AdaptiveLayerSkipper()
    n = skipper.n_layers
    print(f"  n_text_layers={n}", flush=True)

    t0 = time.time()
    print(f"\n[1/3] baseline forward (active=all)...", flush=True)
    r_base = skipper.forward(PROMPT, active_layers=None)
    print(f"    {time.time()-t0:.1f}s  logits shape={tuple(r_base.logits_last.shape)} "
          f"dtype={r_base.logits_last.dtype}  fallback={r_base.fallback}", flush=True)
    print(f"    finite={torch.isfinite(r_base.logits_last).all().item()}  "
          f"max|x|={r_base.logits_last.abs().max().item():.3f}", flush=True)

    t1 = time.time()
    print(f"\n[2/3] skip {{0}} (HIGH path, salta SOLO layer 0)...", flush=True)
    r_skip0 = skipper.forward(PROMPT, active_layers=set(range(n)) - {0})
    print(f"    {time.time()-t1:.1f}s  logits shape={tuple(r_skip0.logits_last.shape)} "
          f"fallback={r_skip0.fallback}", flush=True)
    fin0 = torch.isfinite(r_skip0.logits_last).all().item()
    print(f"    finite={fin0}  max|x|={r_skip0.logits_last.abs().max().item():.3f}", flush=True)

    t2 = time.time()
    print(f"\n[3/3] skip {{{n-1}}} (HIGH path, salta SOLO ultimo layer)...", flush=True)
    r_skip_last = skipper.forward(PROMPT, active_layers=set(range(n)) - {n - 1})
    print(f"    {time.time()-t2:.1f}s  logits shape={tuple(r_skip_last.logits_last.shape)} "
          f"fallback={r_skip_last.fallback}", flush=True)
    fin_last = torch.isfinite(r_skip_last.logits_last).all().item()
    print(f"    finite={fin_last}  max|x|={r_skip_last.logits_last.abs().max().item():.3f}",
          flush=True)

    # Confronto: skip dovrebbe produrre logits DIVERSI dal baseline (sennò
    # significa che lo skip è no-op, anche bug).
    diff0 = (r_base.logits_last - r_skip0.logits_last).abs().max().item()
    diff_last = (r_base.logits_last - r_skip_last.logits_last).abs().max().item()
    print(f"\nmax|Δ| baseline vs skip{{0}}:        {diff0:.3e}", flush=True)
    print(f"max|Δ| baseline vs skip{{{n-1}}}:       {diff_last:.3e}", flush=True)

    ok_finite = fin0 and fin_last
    ok_diff = diff0 > 1e-3 and diff_last > 1e-3  # skip deve cambiare qualcosa
    ok_baseline = torch.isfinite(r_base.logits_last).all().item()

    print("\n" + "=" * 60, flush=True)
    print(f"  baseline logits finite:       {ok_baseline}", flush=True)
    print(f"  HIGH-path logits finite:      {ok_finite}", flush=True)
    print(f"  skip realmente diverso:        {ok_diff}", flush=True)
    if ok_baseline and ok_finite and ok_diff:
        print(f"  HIGH path smoke: PASS — pattern layer.output=layer.input funziona", flush=True)
        rc = 0
    else:
        print(f"  HIGH path smoke: FAIL — serve fix _unwrap su layer.input", flush=True)
        rc = 1
    print("=" * 60, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
