"""exp_006 — Validazione Fase 2: HIGH path produce output ~ baseline?

La mappa AIS funziona se, sui prompt di una categoria, skippare i layer "non
critici" produce output statisticamente simile al baseline. Go criterion
Fase 2 (docs/phases.md): "≥30% layer saltati con ≤5% degrado su ≥1 categoria".

Procedura:
  1) Per ogni categoria, leggo `layer_importance` dalla mappa (popolata da
     exp_005 con group ablation).
  2) Identifico i K_skip gruppi a importance più bassa → costruisco active_layers.
  3) Su N held-out prompt (NON quelli di exp_005), confronto:
     - baseline: forward attivo su tutti i layer
     - HIGH: forward con active_layers
  4) Misure di degrado:
     a) **top-1 agreement**: same top-1 token tra baseline e HIGH (binary).
        Go criterion: ≥95% agreement.
     b) **top-5 overlap**: |top5_base ∩ top5_high| / 5.
     c) **mean KL** divergence (info aggiuntiva, range continuo).

Default: K_skip=2 (2 gruppi di 7 layer = 14/42 layer = 33% skip).
Held-out: N=3 prompt per categoria, seed diverso da exp_005.

Stima compute: 8 cat × 3 prompt × 2 forward = 48 forward × ~70s = ~56 min.

Vincoli: P1-P13. Eseguire con `caffeinate -i python -u`.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skippers.layer_skipper import AdaptiveLayerSkipper
from pipeline.topological_map import TopologicalMap

DATASET_ID = "databricks/databricks-dolly-15k"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p_log = torch.log_softmax(p_logits.float(), dim=-1)
    q_log = torch.log_softmax(q_logits.float(), dim=-1)
    return float((p_log.exp() * (p_log - q_log)).sum().item())


def _group_starts(n_layers: int, n_groups: int) -> list[tuple[int, int]]:
    """Replica la logica di group split di exp_005."""
    sizes = [n_layers // n_groups] * n_groups
    for i in range(n_layers % n_groups):
        sizes[i] += 1
    starts = []
    s = 0
    for sz in sizes:
        starts.append(s)
        s += sz
    return [(starts[i], starts[i] + sizes[i]) for i in range(n_groups)]


def _select_skip_groups(li: np.ndarray, groups: list[tuple[int, int]],
                        k_skip: int) -> set[int]:
    """Dato `layer_importance[42]`, restituisce il SET di layer da skippare:
    i k_skip gruppi con importance media più bassa."""
    group_scores = [(gi, float(li[gs:ge].mean())) for gi, (gs, ge) in enumerate(groups)]
    group_scores.sort(key=lambda x: x[1])
    skip_groups = [gi for gi, _ in group_scores[:k_skip]]
    skip_layers: set[int] = set()
    for gi in skip_groups:
        gs, ge = groups[gi]
        for L in range(gs, ge):
            skip_layers.add(L)
    return skip_layers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-per-cat", type=int, default=3, help="held-out prompt per cat")
    ap.add_argument("--k-skip", type=int, default=2, help="gruppi da skippare (default 2 di 6)")
    ap.add_argument("--n-groups", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="soft skip interp factor (0=hard skip, 1=no skip, 0.5=mid)")
    ap.add_argument("--model-id", type=str, default="google/gemma-4-E4B-it")
    ap.add_argument("--map-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1234,
                    help="seed DIVERSO da exp_005 (=42) per evitare overlap")
    ap.add_argument("--max-chars", type=int, default=300)
    args = ap.parse_args()

    is_e2b = "E2B" in args.model_id
    MAP_DIR = Path(args.map_dir) if args.map_dir else (
        ROOT / "mappa" / ("topology_e2b" if is_e2b else "topology")
    )

    print(f"Loading TopologicalMap from {MAP_DIR}...", flush=True)
    tmap = TopologicalMap.load(MAP_DIR)
    print(f"  entries={len(tmap)}  n_decoder_layers={tmap.n_decoder_layers}", flush=True)
    groups = _group_starts(tmap.n_decoder_layers, args.n_groups)
    print(f"  groups={groups}  k_skip={args.k_skip} → {args.k_skip*7}/42 layer skippati "
          f"= {args.k_skip*7/42*100:.1f}%", flush=True)

    # Per categoria: estrai layer_importance (uno per categoria — tutte le entry
    # della stessa categoria condividono lo stesso array dopo exp_005)
    cat_to_li: dict[str, np.ndarray] = {}
    for e in tmap.entries:
        if e.domain not in cat_to_li and e.layer_importance is not None:
            cat_to_li[e.domain] = e.layer_importance

    print(f"\nSkip plan per categoria:", flush=True)
    skip_plan: dict[str, set[int]] = {}
    for cat, li in sorted(cat_to_li.items()):
        skip = _select_skip_groups(li, groups, args.k_skip)
        skip_plan[cat] = skip
        skip_groups_str = sorted({next(gi for gi, (gs, ge) in enumerate(groups)
                                       if gs <= L < ge) for L in skip})
        print(f"  {cat:22s} skip groups {skip_groups_str}  "
              f"({len(skip)} layer)", flush=True)

    print(f"\nLoading held-out prompts (seed={args.seed}, distinto da exp_005 seed=42)...",
          flush=True)
    ds = load_dataset(DATASET_ID, split="train")
    rng = np.random.default_rng(args.seed)
    indices_per_cat = defaultdict(list)
    for i, row in enumerate(ds):
        indices_per_cat[row["category"]].append(i)
    held_out: dict[str, list[str]] = {}
    for cat, idxs in indices_per_cat.items():
        pick = rng.choice(idxs, size=args.k_per_cat, replace=False)
        held_out[cat] = [ds[int(i)]["instruction"][:args.max_chars] for i in pick]

    print(f"\nLoading AdaptiveLayerSkipper ({args.model_id})...", flush=True)
    if is_e2b:
        dm = {
            "model.vision_tower": "cpu", "model.audio_tower": "cpu",
            "model.embed_vision": "cpu", "model.embed_audio": "cpu",
            "model.language_model": "mps", "lm_head": "mps",
        }
        mm = None
    else:
        dm = "auto"
        mm = {"mps": "8GiB", "cpu": "30GiB"}
    skipper = AdaptiveLayerSkipper(model_id=args.model_id, device_map=dm, max_memory=mm)
    n_layers = skipper.n_layers
    assert n_layers == tmap.n_decoder_layers

    total_forward = sum(len(v) * 2 for v in held_out.values())
    print(f"\nValidation: {total_forward} forward (~{total_forward * 70 / 60:.0f} min)",
          flush=True)
    t_start = time.time()
    forward_done = 0

    results: dict[str, dict] = {}
    for cat, prompts in sorted(held_out.items()):
        skip = skip_plan[cat]
        active = set(range(n_layers)) - skip
        cat_top1_agree = []
        cat_top5_overlap = []
        cat_kl = []
        for p in prompts:
            r_base = skipper.forward(p, active_layers=None)
            forward_done += 1
            r_high = skipper.forward(p, active_layers=active, alpha=args.alpha)
            forward_done += 1
            base_logits = r_base.logits_last
            high_logits = r_high.logits_last
            t1_base = int(base_logits.argmax().item())
            t1_high = int(high_logits.argmax().item())
            t5_base = set(base_logits.topk(5).indices.tolist())
            t5_high = set(high_logits.topk(5).indices.tolist())
            kl_val = _kl(base_logits, high_logits)
            cat_top1_agree.append(t1_base == t1_high)
            cat_top5_overlap.append(len(t5_base & t5_high) / 5)
            cat_kl.append(kl_val)
            elapsed = time.time() - t_start
            rate = forward_done / elapsed
            eta_min = (total_forward - forward_done) / rate / 60
            print(f"  [{cat}] base_top1={t1_base:6d} high_top1={t1_high:6d}  "
                  f"agree={t1_base==t1_high}  top5_overlap={len(t5_base & t5_high)/5:.2f}  "
                  f"KL={kl_val:.3f}  ({forward_done}/{total_forward} "
                  f"ETA {eta_min:.1f}m)", flush=True)
        results[cat] = {
            "top1_agreement": float(np.mean(cat_top1_agree)),
            "top5_overlap_mean": float(np.mean(cat_top5_overlap)),
            "kl_mean": float(np.mean(cat_kl)),
            "n_skipped_layers": len(skip),
            "skip_layers": sorted(skip),
        }

    # Verdetto
    print("\n" + "=" * 78, flush=True)
    print(f"{'category':22s}  agree  top5o  KL    skipped", flush=True)
    print("-" * 78, flush=True)
    pass_count = 0
    for cat, r in results.items():
        marker = "[PASS]" if r["top1_agreement"] >= 0.95 else "[FAIL]"
        if r["top1_agreement"] >= 0.95:
            pass_count += 1
        print(f"  {cat:22s}  {r['top1_agreement']:.2f}   "
              f"{r['top5_overlap_mean']:.2f}   {r['kl_mean']:6.2f}  "
              f"{r['n_skipped_layers']:2d}/42  {marker}", flush=True)
    print("=" * 78, flush=True)
    print(f"\nFase 2 go criterion (top-1 agreement ≥95% su ≥1 categoria, ≥30% skip):", flush=True)
    print(f"  Categorie PASS: {pass_count}/{len(results)}", flush=True)
    rc = 0 if pass_count >= 1 else 1
    print(f"  Verdetto: {'PASS' if rc == 0 else 'FAIL'}", flush=True)

    out_npz = RESULTS_DIR / f"fase2_validation_k{args.k_per_cat}_kskip{args.k_skip}_a{args.alpha:.1f}.npz"
    np.savez(out_npz, results=np.array(results, dtype=object))
    print(f"  Saved {out_npz}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
