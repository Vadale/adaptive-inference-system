"""Fase 1 — collezione attivazioni multi-categoria su Gemma 4 E2B cervelletto.

Pipeline:
  1) Carico `databricks/databricks-dolly-15k` (15 011 prompt, 8 categorie tagged).
  2) Sample stratificato: N/8 per categoria (default N=5000 → 625/cat).
  3) Format chat template Gemma → wrap istruzione in `<bos><start_of_turn>user ...`.
  4) Per ogni prompt: forward via VLM, estraggo `language_model.layers[L].output`
     a last-token position.
  5) Salvo NPZ con embeddings + categorie + prompts.
  6) UMAP 2D (metric cosine) → PNG colorato per categoria.
  7) Metriche: silhouette + k-NN homogeneity → verdetto go/no-go Fase 1.

Default scelti via `exp_002b_pivot_search.py` (N=200, chat template):
  - **L09 last-token pool** ha homog k-NN(k=10) = 0.4485 sopra threshold 0.40.
  - Chat template essenziale: raw text top layer arriva solo a 0.37.

Go criterion (vedi `docs/phases.md`): cluster visibili nel plot. Soglie:
  - **primary**: k-NN(k=10) homogeneity > 0.40 (≥40% dei vicini condividono la
    categoria, contro random baseline 1/8 = 0.125 → 3.2× sopra random).
  - **diagnostica**: silhouette (cosine). Per text embedding 1536-dim su 8
    categorie task-overlapping, valori 0.01-0.08 sono attesi anche con cluster
    chiari; il random baseline su dati high-dim è ~-0.1. Non blocco se < 0.02
    quando homogeneity passa con margine ≥ 1.2×.

Usage:
  python experiments/exp_002_activations_gemma.py --n 5000   # full run ~25-30 min
  python experiments/exp_002_activations_gemma.py --n 200    # smoke ~1 min

Vincoli: P1-P4, P9-P11. VLM + device_map split + lm_head.output (vedi P11).
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import gc
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")  # no display su server
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from nnsight import VisionLanguageModel

MODEL_ID = "google/gemma-4-E2B-it"
DATASET_ID = "databricks/databricks-dolly-15k"

DEVICE_MAP = {
    "model.vision_tower": "cpu", "model.audio_tower": "cpu",
    "model.embed_vision": "cpu", "model.embed_audio": "cpu",
    "model.language_model": "mps", "lm_head": "mps",
}

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
RESULTS_DIR = ROOT / "results"
CORPUS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def _unwrap(t):
    return t[0] if isinstance(t, tuple) else t


def stratified_sample(ds, n_total: int, seed: int) -> list[int]:
    """Restituisce N indici stratificati per `category` (uniformi quando possibile).
    Se una categoria ha meno di N/n_cat esempi, la prende tutta e ribilancia."""
    rng = np.random.default_rng(seed)
    by_cat: dict[str, list[int]] = {}
    for i, cat in enumerate(ds["category"]):
        by_cat.setdefault(cat, []).append(i)

    n_cat = len(by_cat)
    per_cat = n_total // n_cat
    selected: list[int] = []
    remaining = n_total
    sorted_cats = sorted(by_cat, key=lambda c: len(by_cat[c]))  # piccole prima
    for k, cat in enumerate(sorted_cats):
        avail = by_cat[cat]
        cats_left = n_cat - k
        target = min(remaining // cats_left, len(avail))
        idx = rng.choice(len(avail), size=target, replace=False)
        selected.extend(avail[i] for i in idx)
        remaining -= target
    rng.shuffle(selected)
    return selected


def _chatify(proc, text: str) -> str:
    msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def collect_activations(
    model, pivot_layer: int, prompts: list[str], pool: str
) -> np.ndarray:
    """Forward singolo per prompt; estrae hidden state del pivot_layer.
    pool=last → last-token; pool=mean → mean su tutti i token.
    Ritorna array [N, hidden] float32."""
    embs: list[np.ndarray] = []
    t0 = time.time()
    for i, text in enumerate(prompts):
        holder = [None]
        with torch.no_grad():
            with model.trace(text):
                holder[0] = model.model.language_model.layers[pivot_layer].output.save()
        v = _unwrap(holder[0])
        seq = v[0]  # [seq, hidden]
        if pool == "last":
            emb = seq[-1].float().cpu().numpy()
        elif pool == "mean":
            emb = seq.float().mean(dim=0).cpu().numpy()
        else:
            raise ValueError(f"pool sconosciuto: {pool}")
        embs.append(emb)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (len(prompts) - (i + 1)) / rate / 60
            print(f"  [{i+1:5d}/{len(prompts)}] {rate:.1f} prompts/s  ETA {eta_min:.1f} min")
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    return np.stack(embs, axis=0)


def knn_homogeneity(emb: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Frazione di vicini (in cosine space) di stessa categoria, mediata sui punti.
    1.0 = perfetto, 1/n_cat = caso. Per 8 cat il random baseline è ~0.125."""
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(norm)
    _, idx = nn.kneighbors(norm)
    same = labels[idx[:, 1:]] == labels[:, None]  # esclude self
    return float(same.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="numero totale di prompt")
    ap.add_argument("--pivot-layer", type=int, default=9,
                    help="layer testuale (default 9 — vedi pivot search)")
    ap.add_argument("--pool", choices=["last", "mean"], default="last")
    ap.add_argument("--no-chat", action="store_true",
                    help="NON usare il chat template (default: usalo)")
    ap.add_argument("--max-chars", type=int, default=300, help="trunc istruzione (perf)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=str, default="activations_gemma_e2b")
    ap.add_argument("--from-cache", action="store_true",
                    help="se il NPZ esiste, salta forward e fai solo analisi+plot")
    args = ap.parse_args()

    use_chat = not args.no_chat

    pivot_resolved = args.pivot_layer  # n_layers ignoto se from-cache; resolved sotto
    npz_path = CORPUS_DIR / f"{args.out_prefix}_n{args.n}_L{pivot_resolved}_{args.pool}.npz"

    if args.from_cache and npz_path.exists():
        print(f"--from-cache: carico {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        emb = data["embeddings"]
        categories = data["categories"]
        meta = data["meta"].item()
        pivot = meta["pivot_layer"]
        use_chat = meta.get("use_chat", True)
        print(f"  shape={emb.shape}  meta={meta}")
    else:
        print(f"Loading dataset {DATASET_ID}...")
        ds = load_dataset(DATASET_ID, split="train")
        print(f"  N_total={len(ds)} categories={sorted(set(ds['category']))}")

        idx = stratified_sample(ds, args.n, args.seed)
        raw_prompts = [ds[i]["instruction"] for i in idx]
        categories = np.array([ds[i]["category"] for i in idx])
        print(f"Sampled N={len(raw_prompts)} stratificati. Distrib:")
        from collections import Counter
        for c, n in sorted(Counter(categories).items()):
            print(f"  {c:25s} {n}")

        print(f"\nLoading {MODEL_ID} (VLM, device_map split)...")
        model = VisionLanguageModel(MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE_MAP)
        n_layers = len(model._model.model.language_model.layers)
        pivot = args.pivot_layer if args.pivot_layer >= 0 else n_layers + args.pivot_layer
        assert 0 <= pivot < n_layers, f"pivot {pivot} out of [0, {n_layers})"
        print(f"  n_layers={n_layers}, pivot_layer={pivot}, pool={args.pool}, use_chat={use_chat}")

        if use_chat:
            prompts = [_chatify(model.processor, p[:args.max_chars]) for p in raw_prompts]
        else:
            prompts = [p[:args.max_chars] for p in raw_prompts]

        print(f"\nCollecting activations (layer {pivot}, pool={args.pool})...")
        t0 = time.time()
        emb = collect_activations(model, pivot, prompts, args.pool)
        total_min = (time.time() - t0) / 60
        print(f"\nDone. shape={emb.shape} time={total_min:.1f} min")

        assert not np.isnan(emb).any(), "NaN nelle attivazioni"

        npz_path = CORPUS_DIR / f"{args.out_prefix}_n{args.n}_L{pivot}_{args.pool}.npz"
        np.savez_compressed(
            npz_path,
            embeddings=emb,
            categories=categories,
            prompts=np.array(raw_prompts, dtype=object),
            meta=np.array({
                "model_id": MODEL_ID, "dataset_id": DATASET_ID, "pivot_layer": pivot,
                "n_layers": n_layers, "pool": args.pool, "use_chat": use_chat,
                "max_chars": args.max_chars, "seed": args.seed,
            }, dtype=object),
        )
        print(f"Saved {npz_path}  ({npz_path.stat().st_size/1e6:.1f} MB)")

    # --- UMAP 2D ---
    print("\nUMAP 2D...")
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine",
        random_state=args.seed, n_components=2,
    )
    coords = reducer.fit_transform(emb)

    # --- Metriche ---
    label_ids = np.unique(categories, return_inverse=True)[1]
    sil = silhouette_score(emb, label_ids, metric="cosine", sample_size=min(2000, len(emb)),
                           random_state=args.seed)
    homog = knn_homogeneity(emb, label_ids, k=10)
    rand_baseline = 1 / len(np.unique(categories))
    print(f"  silhouette (cosine): {sil:.4f}")
    print(f"  k-NN(k=10) homogeneity: {homog:.4f}  (random baseline: {rand_baseline:.4f})")

    # Verdetto: primary=homog>0.40. Silhouette è diagnostica — non blocca se
    # homog supera il proprio threshold con margine ≥1.2× (homog > 0.48).
    ok_homog = homog > 0.40
    ok_sil = sil > 0.02
    homog_strong = homog > 0.40 * 1.2  # 0.48
    ok_phase1 = ok_homog and (ok_sil or homog_strong)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    cats_unique = sorted(np.unique(categories))
    cmap = plt.get_cmap("tab10")
    for ci, c in enumerate(cats_unique):
        mask = categories == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6,
                   color=cmap(ci), label=f"{c} (n={mask.sum()})")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_title(
        f"Gemma 4 E2B — L{pivot} {args.pool}-pool, "
        f"N={len(emb)}, chat={use_chat}\n"
        f"silhouette={sil:.3f}  k-NN homog={homog:.3f}  "
        f"verdict={'PASS' if ok_phase1 else 'FAIL'}"
    )
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    png_path = RESULTS_DIR / f"{args.out_prefix}_n{args.n}_L{pivot}_{args.pool}_umap.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved {png_path}")

    print("\n" + "=" * 60)
    print(f"  PRIMARY  k-NN homog > 0.40:        {ok_homog}   [{homog:.4f}]")
    print(f"  DIAG     silhouette > 0.02:        {ok_sil}     [{sil:.4f}]")
    print(f"  ALT      homog > 0.48 (≥1.2× thr): {homog_strong}     [{homog:.4f}]")
    print(f"  Fase 1 go criterion (primary AND (diag OR alt)): "
          f"{'PASS' if ok_phase1 else 'FAIL'}")
    print("=" * 60)
    return 0 if ok_phase1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
