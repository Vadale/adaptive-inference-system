"""exp_005 — bootstrap `layer_importance` via GROUP ablation per categoria.

Per ogni categoria, prendiamo K prompt rappresentativi dolly-15k e misuriamo
quanto ciascun GRUPPO di layer del decoder E4B sia critico:
  - baseline: forward con tutti i 42 layer attivi → P_baseline = softmax(logits)
  - per ogni gruppo G di layer contigui: forward con G skippato → P_G
  - importance[G] = mean over prompt di KL(P_baseline || P_G)

Group ablation invece di single-layer riduce il costo 7× (sui 42 layer E4B, 6
gruppi da 7 layer) e fornisce informazione comunque utile per la mappa AIS
("quali REGIONI del decoder sono critiche per categoria X"). Più granulare
non vale il compute (60-90 s/forward su E4B con MPS+CPU offload).

Strategia di mapping al layer_importance per layer singolo:
  importance[L] = importance[group containing L]  (broadcast del valore di gruppo)

Heavy: 8 cat × K prompt × (1 baseline + n_groups) forward. Con K=3, n_groups=6:
  8 × 3 × 7 = 168 forward × 75 s = ~3.5 ore.

Prerequisito: `exp_003_fallback_identity.py` PASS.
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


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(P||Q) in nats. Aborta su NaN/Inf (segnale che il forward è degenerato
    — può capitare se l'intervento HIGH-path produce logits invalidi)."""
    assert torch.isfinite(p_logits).all().item(), "p_logits non finiti (baseline degenerato?)"
    assert torch.isfinite(q_logits).all().item(), "q_logits non finiti (skip ha rotto il forward)"
    p_log = torch.log_softmax(p_logits.float(), dim=-1)
    q_log = torch.log_softmax(q_logits.float(), dim=-1)
    p = p_log.exp()
    kl = float((p * (p_log - q_log)).sum().item())
    assert np.isfinite(kl), f"KL non finita: {kl}"
    return kl


def _split_into_groups(n_layers: int, n_groups: int) -> list[tuple[int, int]]:
    """Restituisce list of (start, end_excl) per i n_groups intervalli contigui."""
    sizes = [n_layers // n_groups] * n_groups
    for i in range(n_layers % n_groups):
        sizes[i] += 1
    starts: list[int] = []
    s = 0
    for sz in sizes:
        starts.append(s)
        s += sz
    return [(starts[i], starts[i] + sizes[i]) for i in range(n_groups)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-per-cat", type=int, default=3,
                    help="prompt per categoria (default 3, 1=smoke)")
    ap.add_argument("--n-groups", type=int, default=6,
                    help="quanti gruppi di layer (default 6 su 42 = 7/gruppo)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--model-id", type=str, default="google/gemma-4-E4B-it",
                    help="decoder model (E4B default, E2B per single-MPS deploy)")
    ap.add_argument("--map-dir", type=str, default=None,
                    help="dir mappa; default: topology (E4B) o topology_e2b (E2B)")
    args = ap.parse_args()

    is_e2b = "E2B" in args.model_id
    MAP_DIR = Path(args.map_dir) if args.map_dir else (
        ROOT / "mappa" / ("topology_e2b" if is_e2b else "topology")
    )
    print(f"  MAP_DIR={MAP_DIR}", flush=True)

    print(f"Loading dataset {DATASET_ID}...", flush=True)
    ds = load_dataset(DATASET_ID, split="train")
    rng = np.random.default_rng(args.seed)

    indices_per_cat: dict = defaultdict(list)
    for i, row in enumerate(ds):
        indices_per_cat[row["category"]].append(i)
    chosen: dict[str, list[str]] = {}
    for cat, idxs in indices_per_cat.items():
        pick = rng.choice(idxs, size=args.k_per_cat, replace=False)
        chosen[cat] = [ds[int(i)]["instruction"][:args.max_chars] for i in pick]
    print(f"  Categorie: {list(chosen)}  prompt/cat={args.k_per_cat}", flush=True)

    print(f"\nLoading AdaptiveLayerSkipper ({args.model_id})...", flush=True)
    # Device map per E2B (sta tutto in MPS) vs E4B (richiede auto+max_memory)
    if "E2B" in args.model_id:
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
    groups = _split_into_groups(n_layers, args.n_groups)
    print(f"  n_layers={n_layers}  groups={groups}", flush=True)

    # importance per gruppo per categoria
    imp_per_cat_per_group: dict[str, np.ndarray] = {
        c: np.zeros(args.n_groups, dtype=np.float32) for c in chosen
    }
    n_per_cat = {c: 0 for c in chosen}

    total_forward = sum(len(v) * (1 + args.n_groups) for v in chosen.values())
    print(f"\nGroup ablation: {total_forward} forward totali "
          f"(~{total_forward * 75 / 60:.0f} min stimati)", flush=True)
    t_start = time.time()
    forward_done = 0

    for cat, prompts in chosen.items():
        for p in prompts:
            # Baseline
            r_base = skipper.forward(p, active_layers=None)
            base_logits = r_base.logits_last
            forward_done += 1
            # Per ogni gruppo G
            for gi, (gs, ge) in enumerate(groups):
                active = [i for i in range(n_layers) if not (gs <= i < ge)]
                r_skip = skipper.forward(p, active_layers=active)
                kl = kl_divergence(base_logits, r_skip.logits_last)
                imp_per_cat_per_group[cat][gi] += kl
                forward_done += 1
                elapsed = time.time() - t_start
                rate = forward_done / elapsed
                eta_min = (total_forward - forward_done) / rate / 60
                print(f"  [{cat}/{n_per_cat[cat]+1}] g{gi}[L{gs:02d}-{ge-1:02d}] "
                      f"KL={kl:.4f}  ({forward_done}/{total_forward}, "
                      f"{rate:.2f} f/s, ETA {eta_min:.1f}m)", flush=True)
            n_per_cat[cat] += 1

    # Media per categoria — manteniamo sia avg (KL assoluto, base per policy
    # tipo "skippa se KL < 0.01") che norm (0-1 per visualizzazione). La
    # mappa salva norm; results NPZ salva entrambi.
    imp_norm_per_cat: dict[str, np.ndarray] = {}
    imp_avg_per_cat: dict[str, np.ndarray] = {}
    for cat, imp_sum in imp_per_cat_per_group.items():
        avg = imp_sum / max(n_per_cat[cat], 1)
        mn, mx = float(avg.min()), float(avg.max())
        norm = (avg - mn) / (mx - mn + 1e-9)
        # Espandi gruppi → layer singoli: importance[L] = norm[group(L)]
        per_layer_norm = np.zeros(n_layers, dtype=np.float32)
        per_layer_avg = np.zeros(n_layers, dtype=np.float32)
        for gi, (gs, ge) in enumerate(groups):
            per_layer_norm[gs:ge] = norm[gi]
            per_layer_avg[gs:ge] = avg[gi]
        imp_norm_per_cat[cat] = per_layer_norm
        imp_avg_per_cat[cat] = per_layer_avg
        print(f"\n[{cat}] mean-KL min={mn:.4f} max={mx:.4f}", flush=True)
        print(f"  per-group (avg_KL  norm):", flush=True)
        for gi, (gs, ge) in enumerate(groups):
            bar = "#" * int(norm[gi] * 30)
            print(f"    g{gi} L{gs:02d}-{ge-1:02d}  {avg[gi]:.4f}  {norm[gi]:.3f}  {bar}",
                  flush=True)

    # Update TopologicalMap — copy() per evitare aliasing condiviso tra
    # entries della stessa categoria (mutare uno modificherebbe tutti gli altri).
    print(f"\nUpdating TopologicalMap at {MAP_DIR}...", flush=True)
    tmap = TopologicalMap.load(MAP_DIR)
    n_updated = 0
    for entry in tmap.entries:
        if entry.domain in imp_norm_per_cat:
            entry.layer_importance = imp_norm_per_cat[entry.domain].copy()
            n_updated += 1
    tmap.save(MAP_DIR)
    print(f"  Updated {n_updated}/{len(tmap.entries)} entries.", flush=True)

    model_tag = "E2B" if "E2B" in args.model_id else "E4B"
    out_npz = RESULTS_DIR / f"layer_importance_{model_tag}_k{args.k_per_cat}_g{args.n_groups}.npz"
    save_data = {f"importance_norm_{c}": v for c, v in imp_norm_per_cat.items()}
    save_data.update({f"importance_avg_{c}": v for c, v in imp_avg_per_cat.items()})
    save_data["groups"] = np.array(groups)
    np.savez_compressed(out_npz, **save_data)
    print(f"  Saved {out_npz}", flush=True)
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
