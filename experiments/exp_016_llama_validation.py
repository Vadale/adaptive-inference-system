"""exp_016 — Llama Fase 2 validation: HIGH path produce output ~ baseline?

Mirror exp_006 but for Llama 3.2 3B with LlamaSkipper (real hard skip).
Held-out prompts (seed≠ablation), top-1 / top-5 / KL.

Hard skip: 2 group / 4 = 50% layer (14/28). Aggressive ma serve a stressare il
sistema; per soft preservation usare alpha < 1.0.
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


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p_log = torch.log_softmax(p_logits.float(), dim=-1)
    q_log = torch.log_softmax(q_logits.float(), dim=-1)
    return float((p_log.exp() * (p_log - q_log)).sum().item())


def _group_starts(n_layers: int, n_groups: int) -> list[tuple[int, int]]:
    sizes = [n_layers // n_groups] * n_groups
    for i in range(n_layers % n_groups):
        sizes[i] += 1
    starts = []
    s = 0
    for sz in sizes:
        starts.append(s)
        s += sz
    return [(starts[i], starts[i] + sizes[i]) for i in range(n_groups)]


def _select_skip_groups(li: np.ndarray, groups, k_skip: int) -> set[int]:
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
    ap.add_argument("--k-per-cat", type=int, default=3)
    ap.add_argument("--k-skip", type=int, default=1, help="groups to skip (default 1/4=25%)")
    ap.add_argument("--n-groups", type=int, default=4)
    ap.add_argument("--mode", choices=["hard", "soft"], default="hard",
                    help="hard skip or soft α-interpolation")
    ap.add_argument("--alpha", type=float, default=0.7, help="soft alpha if mode=soft")
    ap.add_argument("--map-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-chars", type=int, default=300)
    args = ap.parse_args()

    map_dir = Path(args.map_dir) if args.map_dir else (
        ROOT / "mappa" / "topology_llama32_3b"
    )
    print(f"Loading TopologicalMap from {map_dir}...", flush=True)
    tmap = TopologicalMap.load(map_dir)
    groups = _group_starts(tmap.n_decoder_layers, args.n_groups)
    print(f"  entries={len(tmap)}  n_layers={tmap.n_decoder_layers}  groups={groups}",
          flush=True)
    print(f"  k_skip={args.k_skip} → {args.k_skip*7}/{tmap.n_decoder_layers} layer "
          f"= {args.k_skip*7/tmap.n_decoder_layers*100:.1f}%  mode={args.mode}", flush=True)

    cat_to_li: dict[str, np.ndarray] = {}
    for e in tmap.entries:
        if e.domain not in cat_to_li and e.layer_importance is not None:
            cat_to_li[e.domain] = e.layer_importance

    skip_plan: dict[str, set[int]] = {}
    print(f"\nSkip plan per categoria:", flush=True)
    for cat, li in sorted(cat_to_li.items()):
        skip = _select_skip_groups(li, groups, args.k_skip)
        skip_plan[cat] = skip
        skip_groups_str = sorted({next(gi for gi, (gs, ge) in enumerate(groups)
                                       if gs <= L < ge) for L in skip})
        print(f"  {cat:22s} skip groups {skip_groups_str}  ({len(skip)} layer)", flush=True)

    print(f"\nHeld-out prompts (seed={args.seed})...", flush=True)
    ds = load_dataset(DATASET_ID, split="train")
    rng = np.random.default_rng(args.seed)
    indices_per_cat = defaultdict(list)
    for i, row in enumerate(ds):
        indices_per_cat[row["category"]].append(i)
    held_out: dict[str, list[str]] = {}
    for cat, idxs in indices_per_cat.items():
        pick = rng.choice(idxs, size=args.k_per_cat, replace=False)
        held_out[cat] = [ds[int(i)]["instruction"][:args.max_chars] for i in pick]

    print("\nLoading LlamaSkipper...", flush=True)
    skipper = LlamaSkipper()
    n_layers = skipper.n_layers
    assert n_layers == tmap.n_decoder_layers

    total = sum(len(v) * 2 for v in held_out.values())
    print(f"\nValidation: {total} forward", flush=True)
    t_start = time.time()
    done = 0
    results: dict[str, dict] = {}

    for cat, prompts in sorted(held_out.items()):
        skip = skip_plan[cat]
        cat_t1 = []
        cat_t5 = []
        cat_kl = []
        for p in prompts:
            base_logits = skipper.forward(p)
            done += 1
            if args.mode == "hard":
                high_logits = skipper.forward(p, hard_skip=skip)
            else:
                soft = {i: args.alpha for i in skip}
                high_logits = skipper.forward(p, soft_skip=soft)
            done += 1
            t1_b = int(base_logits.argmax().item())
            t1_h = int(high_logits.argmax().item())
            t5_b = set(base_logits.topk(5).indices.tolist())
            t5_h = set(high_logits.topk(5).indices.tolist())
            kl_val = _kl(base_logits, high_logits)
            cat_t1.append(t1_b == t1_h)
            cat_t5.append(len(t5_b & t5_h) / 5)
            cat_kl.append(kl_val)
            print(f"  [{cat}] base={t1_b:6d} skip={t1_h:6d}  agree={t1_b==t1_h}  "
                  f"top5o={len(t5_b & t5_h)/5:.2f}  KL={kl_val:.3f}  "
                  f"({done}/{total})", flush=True)
        results[cat] = {
            "top1_agreement": float(np.mean(cat_t1)),
            "top5_overlap_mean": float(np.mean(cat_t5)),
            "kl_mean": float(np.mean(cat_kl)),
            "n_skipped_layers": len(skip),
            "skip_layers": sorted(skip),
        }

    print("\n" + "=" * 78, flush=True)
    print(f"{'category':22s}  agree  top5o   KL     skipped", flush=True)
    print("-" * 78, flush=True)
    pass_count = 0
    for cat, r in results.items():
        marker = "[PASS]" if r["top1_agreement"] >= 0.95 else "[FAIL]"
        if r["top1_agreement"] >= 0.95:
            pass_count += 1
        print(f"  {cat:22s}  {r['top1_agreement']:.2f}   "
              f"{r['top5_overlap_mean']:.2f}   {r['kl_mean']:6.3f}  "
              f"{r['n_skipped_layers']:2d}/{n_layers}  {marker}", flush=True)
    print("=" * 78, flush=True)
    print(f"\nCategorie PASS: {pass_count}/{len(results)}", flush=True)
    rc = 0 if pass_count >= 1 else 1
    print(f"Llama Fase 2 ({args.mode}): {'PASS' if rc == 0 else 'FAIL'}", flush=True)
    print(f"Time: {(time.time()-t_start):.1f}s", flush=True)

    tag = f"hard" if args.mode == "hard" else f"soft_a{args.alpha:.1f}"
    out_npz = RESULTS_DIR / f"llama_validation_k{args.k_per_cat}_kskip{args.k_skip}_{tag}.npz"
    np.savez(out_npz, results=np.array(results, dtype=object))
    print(f"Saved {out_npz}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
