"""exp_013 — corpus embeddings on Llama 3.2 3B (router for AIS-Llama).

Mirrors exp_002 (Gemma corpus) but uses LlamaSkipper.embed() which runs a full
forward with output_hidden_states=True and extracts hidden state at mid-layer.

Default pivot = n_layers // 3 = 9 on Llama 3.2 (28 layers). Same as Gemma E2B
to keep the comparison apples-to-apples (early-mid layer last-token pooling).

Usage:
  python experiments/exp_013_llama_corpus.py --n 5000   # full ~12 min
  python experiments/exp_013_llama_corpus.py --n 200    # smoke ~30s
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import gc
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skippers.llama_skipper import LlamaSkipper

DATASET_ID = "databricks/databricks-dolly-15k"
CORPUS_DIR = ROOT / "corpus"
RESULTS_DIR = ROOT / "results"
CORPUS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def stratified_sample(ds, n_total: int, seed: int) -> list[int]:
    """Stratified indices by `category`, mirror of exp_002."""
    rng = np.random.default_rng(seed)
    by_cat: dict[str, list[int]] = {}
    for i, cat in enumerate(ds["category"]):
        by_cat.setdefault(cat, []).append(i)
    n_cat = len(by_cat)
    selected: list[int] = []
    remaining = n_total
    sorted_cats = sorted(by_cat, key=lambda c: len(by_cat[c]))
    for k, cat in enumerate(sorted_cats):
        avail = by_cat[cat]
        cats_left = n_cat - k
        target = min(remaining // cats_left, len(avail))
        idx = rng.choice(len(avail), size=target, replace=False)
        selected.extend(avail[i] for i in idx)
        remaining -= target
    rng.shuffle(selected)
    return selected


def knn_homogeneity(emb: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(norm)
    _, idx = nn.kneighbors(norm)
    same = labels[idx[:, 1:]] == labels[:, None]
    return float(same.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--pivot-layer", type=int, default=None,
                    help="default: n_layers // 3 (=9 su Llama 3.2 3B 28L)")
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=str, default="activations_llama32_3b")
    args = ap.parse_args()

    print(f"Loading dataset {DATASET_ID}...", flush=True)
    ds = load_dataset(DATASET_ID, split="train")
    print(f"  N_total={len(ds)} categories={sorted(set(ds['category']))}", flush=True)

    idx = stratified_sample(ds, args.n, args.seed)
    raw_prompts = [ds[i]["instruction"][:args.max_chars] for i in idx]
    categories = np.array([ds[i]["category"] for i in idx])
    print(f"Sampled N={len(raw_prompts)} stratified. Distrib:", flush=True)
    for c, n in sorted(Counter(categories).items()):
        print(f"  {c:25s} {n}", flush=True)

    print("\nLoading LlamaSkipper...", flush=True)
    skipper = LlamaSkipper()
    n_layers = skipper.n_layers
    pivot = args.pivot_layer if args.pivot_layer is not None else n_layers // 3
    print(f"  n_layers={n_layers}  hidden={skipper.hidden_size}  pivot=L{pivot}", flush=True)

    print(f"\nExtracting embeddings (layer {pivot}, last-token pool)...", flush=True)
    embs: list[np.ndarray] = []
    t0 = time.time()
    for i, p in enumerate(raw_prompts):
        emb = skipper.embed(p, layer_idx=pivot).numpy()
        embs.append(emb)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (len(raw_prompts) - (i + 1)) / rate / 60
            print(f"  [{i+1:5d}/{len(raw_prompts)}] {rate:.1f} prompts/s  "
                  f"ETA {eta_min:.1f} min", flush=True)
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    emb = np.stack(embs, axis=0)
    total_min = (time.time() - t0) / 60
    print(f"\nDone. shape={emb.shape} time={total_min:.1f} min", flush=True)
    assert not np.isnan(emb).any(), "NaN in embeddings"

    npz_path = CORPUS_DIR / f"{args.out_prefix}_n{args.n}_L{pivot}_last.npz"
    np.savez_compressed(
        npz_path,
        embeddings=emb,
        categories=categories,
        prompts=np.array(raw_prompts, dtype=object),
        meta=np.array({
            "model_id": "unsloth/Llama-3.2-3B-Instruct",
            "dataset_id": DATASET_ID, "pivot_layer": pivot,
            "n_layers": n_layers, "pool": "last", "use_chat": True,
            "max_chars": args.max_chars, "seed": args.seed,
        }, dtype=object),
    )
    print(f"Saved {npz_path}  ({npz_path.stat().st_size/1e6:.1f} MB)", flush=True)

    # Metrics + UMAP
    print("\nMetrics...", flush=True)
    label_ids = np.unique(categories, return_inverse=True)[1]
    sil = silhouette_score(
        emb, label_ids, metric="cosine",
        sample_size=min(2000, len(emb)), random_state=args.seed,
    )
    homog = knn_homogeneity(emb, label_ids, k=10)
    rand_baseline = 1 / len(np.unique(categories))
    print(f"  silhouette (cosine): {sil:.4f}", flush=True)
    print(f"  k-NN(k=10) homogeneity: {homog:.4f}  (random {rand_baseline:.4f})",
          flush=True)

    ok_homog = homog > 0.40
    ok_sil = sil > 0.02
    homog_strong = homog > 0.40 * 1.2
    ok_phase1 = ok_homog and (ok_sil or homog_strong)

    print("\nUMAP 2D...", flush=True)
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine",
        random_state=args.seed, n_components=2,
    )
    coords = reducer.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(10, 8))
    cats_unique = sorted(np.unique(categories))
    cmap = plt.get_cmap("tab10")
    for ci, c in enumerate(cats_unique):
        mask = categories == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6,
                   color=cmap(ci), label=f"{c} (n={mask.sum()})")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_title(
        f"Llama 3.2 3B — L{pivot} last-pool, N={len(emb)}\n"
        f"silhouette={sil:.3f}  k-NN homog={homog:.3f}  "
        f"verdict={'PASS' if ok_phase1 else 'FAIL'}"
    )
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    png_path = RESULTS_DIR / f"{args.out_prefix}_n{args.n}_L{pivot}_last_umap.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved {png_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"  PRIMARY  k-NN homog > 0.40:        {ok_homog}   [{homog:.4f}]", flush=True)
    print(f"  DIAG     silhouette > 0.02:        {ok_sil}     [{sil:.4f}]", flush=True)
    print(f"  ALT      homog > 0.48 (≥1.2× thr): {homog_strong}     [{homog:.4f}]", flush=True)
    print(f"  Fase 1 (Llama): {'PASS' if ok_phase1 else 'FAIL'}", flush=True)
    print("=" * 60, flush=True)
    return 0 if ok_phase1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
