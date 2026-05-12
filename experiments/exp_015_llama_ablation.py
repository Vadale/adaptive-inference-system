"""exp_015 — Llama ablation per category (group ablation, REAL hard skip).

Mirror exp_005 but for Llama 3.2 3B. Llama supports native hard skip via
ModuleList swap (LlamaSkipper) → forward is FAST and accurate compute saving,
no nnsight overhead.

28 layers / 4 groups = 7 layer/group (each = 25% of the model).

For each category × K prompts:
  - baseline: full 28-layer forward
  - for each group G: forward with G hard-skipped → KL(baseline || skip)

Stima compute: 8 cat × 3 prompt × (1 + 4) = 120 forward × ~0.15s = ~18 s.

Writes layer_importance per category into TopologicalMap.
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

from skippers.llama_skipper import LlamaSkipper
from pipeline.topological_map import TopologicalMap

DATASET_ID = "databricks/databricks-dolly-15k"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    assert torch.isfinite(p_logits).all().item()
    assert torch.isfinite(q_logits).all().item()
    p_log = torch.log_softmax(p_logits.float(), dim=-1)
    q_log = torch.log_softmax(q_logits.float(), dim=-1)
    return float((p_log.exp() * (p_log - q_log)).sum().item())


def _split_into_groups(n_layers: int, n_groups: int) -> list[tuple[int, int]]:
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
    ap.add_argument("--k-per-cat", type=int, default=3)
    ap.add_argument("--n-groups", type=int, default=4,
                    help="4 groups of 7 layer su 28 (default)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--map-dir", type=str, default=None,
                    help="default: mappa/topology_llama32_3b")
    args = ap.parse_args()

    map_dir = Path(args.map_dir) if args.map_dir else (
        ROOT / "mappa" / "topology_llama32_3b"
    )
    print(f"  MAP_DIR={map_dir}", flush=True)

    print(f"Loading {DATASET_ID}...", flush=True)
    ds = load_dataset(DATASET_ID, split="train")
    rng = np.random.default_rng(args.seed)
    indices_per_cat: dict = defaultdict(list)
    for i, row in enumerate(ds):
        indices_per_cat[row["category"]].append(i)
    chosen: dict[str, list[str]] = {}
    for cat, idxs in indices_per_cat.items():
        pick = rng.choice(idxs, size=args.k_per_cat, replace=False)
        chosen[cat] = [ds[int(i)]["instruction"][:args.max_chars] for i in pick]
    print(f"  cats={list(chosen)}  k/cat={args.k_per_cat}", flush=True)

    print(f"\nLoading LlamaSkipper...", flush=True)
    skipper = LlamaSkipper()
    n_layers = skipper.n_layers
    groups = _split_into_groups(n_layers, args.n_groups)
    print(f"  n_layers={n_layers}  groups={groups}", flush=True)

    imp_per_cat_per_group: dict[str, np.ndarray] = {
        c: np.zeros(args.n_groups, dtype=np.float32) for c in chosen
    }
    n_per_cat = {c: 0 for c in chosen}

    total_forward = sum(len(v) * (1 + args.n_groups) for v in chosen.values())
    print(f"\nGroup ablation: {total_forward} forward", flush=True)
    t_start = time.time()
    forward_done = 0

    for cat, prompts in chosen.items():
        for p in prompts:
            base_logits = skipper.forward(p)  # no skip → bit-identical baseline
            forward_done += 1
            for gi, (gs, ge) in enumerate(groups):
                hard = set(range(gs, ge))
                skip_logits = skipper.forward(p, hard_skip=hard)
                kl = kl_divergence(base_logits, skip_logits)
                imp_per_cat_per_group[cat][gi] += kl
                forward_done += 1
                elapsed = time.time() - t_start
                rate = forward_done / elapsed
                eta = (total_forward - forward_done) / rate
                print(f"  [{cat}/{n_per_cat[cat]+1}] g{gi}[L{gs:02d}-{ge-1:02d}] "
                      f"KL={kl:.4f}  ({forward_done}/{total_forward} "
                      f"{rate:.1f} f/s ETA {eta:.0f}s)", flush=True)
            n_per_cat[cat] += 1

    imp_norm_per_cat: dict[str, np.ndarray] = {}
    imp_avg_per_cat: dict[str, np.ndarray] = {}
    for cat, imp_sum in imp_per_cat_per_group.items():
        avg = imp_sum / max(n_per_cat[cat], 1)
        mn, mx = float(avg.min()), float(avg.max())
        norm = (avg - mn) / (mx - mn + 1e-9)
        per_layer_norm = np.zeros(n_layers, dtype=np.float32)
        per_layer_avg = np.zeros(n_layers, dtype=np.float32)
        for gi, (gs, ge) in enumerate(groups):
            per_layer_norm[gs:ge] = norm[gi]
            per_layer_avg[gs:ge] = avg[gi]
        imp_norm_per_cat[cat] = per_layer_norm
        imp_avg_per_cat[cat] = per_layer_avg
        print(f"\n[{cat}] mean-KL min={mn:.4f} max={mx:.4f}", flush=True)
        for gi, (gs, ge) in enumerate(groups):
            bar = "#" * int(norm[gi] * 30)
            print(f"  g{gi} L{gs:02d}-{ge-1:02d}  avg={avg[gi]:.4f}  "
                  f"norm={norm[gi]:.3f}  {bar}", flush=True)

    print(f"\nUpdating TopologicalMap at {map_dir}...", flush=True)
    tmap = TopologicalMap.load(map_dir)
    n_updated = 0
    for entry in tmap.entries:
        if entry.domain in imp_norm_per_cat:
            entry.layer_importance = imp_norm_per_cat[entry.domain].copy()
            n_updated += 1
    tmap.save(map_dir)
    print(f"  Updated {n_updated}/{len(tmap.entries)} entries.", flush=True)

    out_npz = RESULTS_DIR / f"layer_importance_llama32_k{args.k_per_cat}_g{args.n_groups}.npz"
    save_data = {f"importance_norm_{c}": v for c, v in imp_norm_per_cat.items()}
    save_data.update({f"importance_avg_{c}": v for c, v in imp_avg_per_cat.items()})
    save_data["groups"] = np.array(groups)
    np.savez_compressed(out_npz, **save_data)
    print(f"  Saved {out_npz}", flush=True)
    print(f"\nTotal: {(time.time()-t_start):.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
