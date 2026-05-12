"""Pivot search: trova il (layer, pool_mode) migliore per topic embedding.

Su un subsample di prompts cattura le attivazioni di TUTTI i layer testuali in
un singolo forward, calcola due pooling (last-token, mean-over-tokens), e misura
k-NN(k=10) homogeneity per categoria su ogni (layer, pool). Stampa grid + save
PNG. Il (layer, pool) migliore va usato come default per exp_002 full run.

Razionale (vedi exp_002 mini-run FAIL): gli ultimi layer del transformer
specializzano sulla next-token prediction, perdendo info semantica di alto
livello. I layer mid-late (15-28) tipicamente codificano meglio il topic.
Mean pooling di solito > last-token su prompt brevi causal-LM.
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _unwrap(t):
    return t[0] if isinstance(t, tuple) else t


def stratified_sample(ds, n_total: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    by_cat: dict[str, list[int]] = {}
    for i, cat in enumerate(ds["category"]):
        by_cat.setdefault(cat, []).append(i)
    n_cat = len(by_cat)
    selected: list[int] = []
    remaining = n_total
    for k, cat in enumerate(sorted(by_cat, key=lambda c: len(by_cat[c]))):
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


def _chatify(proc, text: str) -> str:
    msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-chat", action="store_true",
                    help="formatta i prompt col chat template Gemma (recommended)")
    args = ap.parse_args()

    print(f"Loading {DATASET_ID}...")
    ds = load_dataset(DATASET_ID, split="train")
    idx = stratified_sample(ds, args.n, args.seed)
    prompts = [ds[i]["instruction"] for i in idx]
    categories = np.array([ds[i]["category"] for i in idx])
    print(f"Sampled N={len(prompts)} su {len(set(categories))} categorie.")

    print(f"\nLoading {MODEL_ID}...")
    model = VisionLanguageModel(MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE_MAP)
    n_layers = len(model._model.model.language_model.layers)
    hidden = model._model.config.text_config.hidden_size
    print(f"  n_layers={n_layers}, hidden={hidden}, use_chat={args.use_chat}")

    if args.use_chat:
        prompts = [_chatify(model.processor, p[:args.max_chars]) for p in prompts]

    # Tensori risultato [N, n_layers, hidden] per ognuno dei due pool
    pool_last = np.empty((len(prompts), n_layers, hidden), dtype=np.float32)
    pool_mean = np.empty((len(prompts), n_layers, hidden), dtype=np.float32)

    print(f"\nForward su {len(prompts)} prompt (cattura TUTTI i {n_layers} layer)...")
    t0 = time.time()
    for i, p in enumerate(prompts):
        # quando use_chat=True, p è già formattato (non re-truncate)
        text = p if args.use_chat else p[:args.max_chars]
        holders: list = []
        with torch.no_grad():
            with model.trace(text):
                for L in range(n_layers):
                    holders.append(model.model.language_model.layers[L].output.save())
        for L in range(n_layers):
            v = _unwrap(holders[L])  # [1, seq, hidden]
            seq = v[0]                # [seq, hidden]
            pool_last[i, L] = seq[-1].float().cpu().numpy()
            pool_mean[i, L] = seq.float().mean(dim=0).cpu().numpy()
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = (len(prompts) - (i + 1)) / ((i + 1) / elapsed) / 60
            print(f"  [{i+1}/{len(prompts)}] {((i+1)/elapsed):.1f} p/s  ETA {eta:.1f}m")
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print(f"Forward done in {(time.time()-t0)/60:.1f} min.")
    assert not np.isnan(pool_last).any()
    assert not np.isnan(pool_mean).any()

    # --- Compute homogeneity per (layer, pool) ---
    label_ids = np.unique(categories, return_inverse=True)[1]
    homog_last = np.zeros(n_layers)
    homog_mean = np.zeros(n_layers)
    print("\nlayer  last_pool   mean_pool")
    for L in range(n_layers):
        homog_last[L] = knn_homogeneity(pool_last[:, L], label_ids, k=10)
        homog_mean[L] = knn_homogeneity(pool_mean[:, L], label_ids, k=10)
        print(f"  L{L:02d}   {homog_last[L]:.4f}     {homog_mean[L]:.4f}")

    # Best
    best_last = int(np.argmax(homog_last)); best_last_val = homog_last[best_last]
    best_mean = int(np.argmax(homog_mean)); best_mean_val = homog_mean[best_mean]
    rand = 1 / len(np.unique(categories))
    print(f"\nBest:")
    print(f"  last-pool: L{best_last:02d}  homog={best_last_val:.4f}  (random={rand:.4f})")
    print(f"  mean-pool: L{best_mean:02d}  homog={best_mean_val:.4f}  (random={rand:.4f})")
    overall_best = ("last", best_last, best_last_val) if best_last_val > best_mean_val else ("mean", best_mean, best_mean_val)
    print(f"  → SCELTA: pool={overall_best[0]}, layer={overall_best[1]}, homog={overall_best[2]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(homog_last, label="last-token pool", marker="o")
    ax.plot(homog_mean, label="mean-token pool", marker="s")
    ax.axhline(rand, color="grey", linestyle="--", label=f"random ({rand:.3f})")
    ax.axhline(0.40, color="green", linestyle=":", label="Fase 1 threshold (0.40)")
    ax.set_xlabel("text layer index"); ax.set_ylabel("k-NN(k=10) homogeneity")
    suffix = "chat" if args.use_chat else "raw"
    ax.set_title(f"Pivot search Gemma 4 E2B — N={len(prompts)}, "
                 f"{n_layers} layer × 2 pool ({suffix})")
    ax.legend(); ax.grid(alpha=0.3)
    out_png = RESULTS_DIR / f"pivot_search_n{args.n}_{suffix}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nSaved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
