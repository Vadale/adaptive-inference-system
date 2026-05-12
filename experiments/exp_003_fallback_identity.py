"""Test critico Fase 2 — verify_fallback_identity.

La garanzia AIS "FALLBACK = baseline esatto" si regge su questo test. Se
fallisce anche solo su 1 prompt: BLOCK assoluto, niente Fase 2 finché non
diagnostichi.

Procedura:
  1) Carico AdaptiveLayerSkipper (wrappa E4B).
  2) Per 20 prompt diversi (mix di lunghezze e categorie):
     a) `out_baseline` = forward diretto del modello (no skipper, no intervento)
     b) `out_fallback` = forward via skipper con `active_layers=None` (modalità
        FALLBACK: nessun layer skippato, percorso baseline puro)
  3) Verify: `torch.allclose(baseline, fallback, atol=1e-4)` AND top-5 rank
     equivalence (più robusto a futuri non-determinismi MPS — vedi P8).
  4) PASS solo se 20/20.

Vincoli: P1-P12.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skippers.layer_skipper import AdaptiveLayerSkipper

# Mix di prompt: brevi/lunghi, factual/creative/code, per stressare il test
PROMPTS = [
    "The capital of France is",
    "Write a haiku about autumn.",
    "Solve: 17 * 23 = ",
    "Translate to Italian: 'Good morning'",
    "What is the chemical symbol for gold?",
    "Explain quantum entanglement in one sentence.",
    "List three colors.",
    "What year did World War II end?",
    "Define 'photosynthesis'.",
    "Name the largest ocean.",
    "What is the speed of light?",
    "Who wrote Hamlet?",
    "Convert 100 Celsius to Fahrenheit.",
    "What is the capital of Japan?",
    "What is 2 to the power of 10?",
    "What language is spoken in Brazil?",
    "Name a famous physicist.",
    "What is the boiling point of water?",
    "Who painted the Mona Lisa?",
    "What is the largest planet in our solar system?",
]


def _baseline_forward(skipper: AdaptiveLayerSkipper, prompt: str) -> torch.Tensor:
    """Forward diretto via VLM (no skipper logic, percorso PyTorch nativo).
    Confronto contro `skipper.forward(active_layers=None)` che usa il
    percorso skipper ma SENZA interventi."""
    text = skipper._chatify(prompt)
    logits_save = None
    with torch.no_grad():
        with skipper.model.trace(text):
            logits_save = skipper.model.lm_head.output.save()
    return logits_save[0, -1, :].float().cpu()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(PROMPTS),
                    help="quanti prompt testare (default 20)")
    args = ap.parse_args()
    n_test = min(args.n, len(PROMPTS))

    print("Loading AdaptiveLayerSkipper (E4B)...", flush=True)
    skipper = AdaptiveLayerSkipper()
    print(f"  n_text_layers={skipper.n_layers}", flush=True)

    # Sanity preliminare: il path "active_layers=set(range(n))" deve dare lo
    # stesso risultato di "active_layers=None" (entrambi sono FALLBACK).
    print(f"\n[sanity] FALLBACK None == FALLBACK set(range(n))...", flush=True)
    none_logits = skipper.forward(PROMPTS[0], active_layers=None).logits_last
    full_logits = skipper.forward(
        PROMPTS[0], active_layers=set(range(skipper.n_layers))
    ).logits_last
    sanity_diff = (none_logits - full_logits).abs().max().item()
    assert sanity_diff == 0.0, (
        f"FALLBACK(None) != FALLBACK(full set): max|Δ|={sanity_diff:.3e}"
    )
    print(f"  OK max|Δ|=0.000e+00", flush=True)

    print(f"\nTesting {n_test} prompt: baseline vs FALLBACK forward", flush=True)
    print(f"  Criteri: max |Δ| < 1e-4 AND top-5 rank identica", flush=True)
    print("-" * 78, flush=True)
    n_pass_atol = 0
    n_pass_rank = 0
    max_diff_global = 0.0
    failures: list = []

    import time as _time
    t0 = _time.time()
    for i, p in enumerate(PROMPTS[:n_test]):
        t_p = _time.time()
        baseline = _baseline_forward(skipper, p)
        fallback_result = skipper.forward(p, active_layers=None)
        fallback = fallback_result.logits_last

        diff = (baseline - fallback).abs().max().item()
        max_diff_global = max(max_diff_global, diff)

        atol_ok = diff <= 1e-4  # P8/torch.allclose: <= atol, non strict <
        bt5 = baseline.topk(5).indices.tolist()
        ft5 = fallback.topk(5).indices.tolist()
        rank_ok = bt5 == ft5

        n_pass_atol += int(atol_ok)
        n_pass_rank += int(rank_ok)
        if not (atol_ok and rank_ok):
            failures.append((i, p[:40], diff, bt5, ft5))

        dt = _time.time() - t_p
        print(f"  [{i+1:2d}/{n_test}] {dt:5.1f}s  max|Δ|={diff:.2e}  "
              f"atol_1e-4={atol_ok}  top5_eq={rank_ok}", flush=True)

    print("-" * 78, flush=True)
    print(f"\nMax |Δ| global: {max_diff_global:.3e}", flush=True)
    print(f"PASS atol(1e-4):     {n_pass_atol}/{n_test}", flush=True)
    print(f"PASS top-5 rank eq:  {n_pass_rank}/{n_test}", flush=True)

    ok = (n_pass_atol == n_test) and (n_pass_rank == n_test)
    print("\n" + "=" * 60, flush=True)
    if ok:
        print(f"  verify_fallback_identity: PASS {n_test}/{n_test} — Fase 2 può procedere",
              flush=True)
        rc = 0
    else:
        print(f"  verify_fallback_identity: FAIL — BLOCK Fase 2", flush=True)
        for i, p, d, b, f in failures[:5]:
            print(f"    [{i}] {p!r}  Δ={d:.2e}  baseline-top5={b}  fallback-top5={f}",
                  flush=True)
        rc = 1
    print("=" * 60, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
