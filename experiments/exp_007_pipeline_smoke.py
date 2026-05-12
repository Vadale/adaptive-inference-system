"""exp_007 — smoke test della pipeline end-to-end AIS.

Test in 3 scenari su 2 prompt (1 dal corpus = match perfetto, 1 random):
  A) confidence_threshold=0.999 + match perfetto (cosine=1.0) → HIGH path
     attivo, qualche layer skippato.
  B) confidence_threshold=0.999 + query random → similarity bassa, FALLBACK.
  C) confidence_threshold=0.0 → forza HIGH path anche su query random (skip
     basato sull'entry più simile, qualunque sia).

Per evitare di tenere cervelletto (E2B) + cervellone (E4B) in memoria
contemporanea, uso `infer_from_embedding` con embedding pre-caricati dal
corpus Fase 1 (NPZ).

Atteso:
  - A) is_high_path=True, skipped_layers non vuoto, logits finite.
  - B) is_high_path=False, skipped_layers=[], logits = baseline esatto.
  - C) is_high_path=True (forzato), skipped_layers basati sul nearest entry.

Vincoli: P1-P14. Eseguire con `caffeinate -i python -u`.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.pipeline import AISInferencePipeline

CORPUS_NPZ = ROOT / "corpus" / "activations_gemma_e2b_n5000_L9_last.npz"
MAP_DIR = ROOT / "mappa" / "topology"


def main() -> int:
    print(f"Loading corpus from {CORPUS_NPZ}...", flush=True)
    data = np.load(CORPUS_NPZ, allow_pickle=True)
    embeddings = data["embeddings"]
    categories = data["categories"]
    prompts = data["prompts"]
    print(f"  N={len(embeddings)} hidden={embeddings.shape[1]}", flush=True)

    print(f"\nInstantiating AISInferencePipeline...", flush=True)
    t0 = time.time()
    # default threshold 0.999 → solo match quasi-perfetti vanno in HIGH
    pipe = AISInferencePipeline(MAP_DIR, confidence_threshold=0.999)
    print(f"  Loaded in {time.time()-t0:.1f}s. n_layers cervellone={pipe.cervellone.n_layers}",
          flush=True)

    # --- A) match perfetto (embedding dal corpus, prompt corrispondente) ---
    idx_perfect = 0
    prompt_A = str(prompts[idx_perfect])[:200]
    emb_A = embeddings[idx_perfect]
    cat_A = str(categories[idx_perfect])
    print(f"\n[A] Match perfetto: prompt #{idx_perfect} cat={cat_A!r}", flush=True)
    print(f"    prompt={prompt_A!r}", flush=True)
    t1 = time.time()
    trace_A = pipe.infer_from_embedding(prompt_A, emb_A)
    print(f"    {time.time()-t1:.1f}s  sim={trace_A.similarity:.4f}  "
          f"matched_domain={trace_A.matched_entry.domain if trace_A.matched_entry else None}",
          flush=True)
    print(f"    is_high_path={trace_A.is_high_path}  "
          f"skipped_layers={trace_A.skipped_layers}", flush=True)
    print(f"    logits finite: {torch.isfinite(trace_A.logits_last).all().item()}",
          flush=True)

    # --- B) random embedding (no match) ---
    print(f"\n[B] Random embedding (no match) — atteso FALLBACK", flush=True)
    rng = np.random.default_rng(99)
    emb_B = rng.normal(size=embeddings.shape[1]).astype(np.float32)
    t2 = time.time()
    trace_B = pipe.infer_from_embedding("Random query for fallback test", emb_B)
    print(f"    {time.time()-t2:.1f}s  sim={trace_B.similarity:.4f}  "
          f"matched_domain={trace_B.matched_entry.domain if trace_B.matched_entry else None}",
          flush=True)
    print(f"    is_high_path={trace_B.is_high_path}  "
          f"skipped_layers={trace_B.skipped_layers}", flush=True)

    # --- C) forzo HIGH abbassando threshold a 0 ---
    print(f"\n[C] threshold=0.0 → forza HIGH path su qualsiasi query", flush=True)
    pipe.confidence_threshold = 0.0
    t3 = time.time()
    trace_C = pipe.infer_from_embedding("Random query forced HIGH", emb_B)
    print(f"    {time.time()-t3:.1f}s  sim={trace_C.similarity:.4f}  "
          f"matched_domain={trace_C.matched_entry.domain if trace_C.matched_entry else None}",
          flush=True)
    print(f"    is_high_path={trace_C.is_high_path}  "
          f"skipped_layers={trace_C.skipped_layers}", flush=True)
    print(f"    logits finite: {torch.isfinite(trace_C.logits_last).all().item()}",
          flush=True)

    # Verdetto
    ok_A_high = trace_A.is_high_path and len(trace_A.skipped_layers) > 0
    ok_B_fallback = (not trace_B.is_high_path) and len(trace_B.skipped_layers) == 0
    ok_C_high_forced = trace_C.is_high_path and len(trace_C.skipped_layers) > 0
    ok_all_finite = all(
        torch.isfinite(t.logits_last).all().item() for t in (trace_A, trace_B, trace_C)
    )

    print("\n" + "=" * 60, flush=True)
    print(f"  [A] match perfetto → HIGH path:     {ok_A_high}", flush=True)
    print(f"  [B] random → FALLBACK:               {ok_B_fallback}", flush=True)
    print(f"  [C] threshold=0 → HIGH forced:       {ok_C_high_forced}", flush=True)
    print(f"  Tutti i logits finiti:                {ok_all_finite}", flush=True)
    rc = 0 if (ok_A_high and ok_B_fallback and ok_C_high_forced and ok_all_finite) else 1
    print(f"  Pipeline smoke: {'PASS' if rc == 0 else 'FAIL'}", flush=True)
    print("=" * 60, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
