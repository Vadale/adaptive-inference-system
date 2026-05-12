"""exp_017 — MMLU benchmark on Llama 3.2 3B AIS (router = decoder).

Mirror exp_008 but Llama-only: same model serves as encoder (embed at L9) AND
decoder (full forward with optional hard/soft skip). No model unload needed.

Setup:
  1) Sample N MMLU questions
  2) For each: embed via LlamaSkipper.embed() → mappa lookup → category match
     → skip plan from layer_importance (skip layers with importance < threshold)
  3) Run baseline forward + AIS forward (hard or soft skip)
  4) Extract answer = argmax over {' A', ' B', ' C', ' D'} token logits
  5) Aggregate accuracy + top-1 agreement + latency
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skippers.llama_skipper import LlamaSkipper
from pipeline.topological_map import TopologicalMap

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _format_mmlu(row) -> str:
    q = row["question"]
    c = row["choices"]
    return (
        f"Answer the following multiple choice question.\n\n"
        f"Question: {q}\n"
        f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\n"
        f"Answer:"
    )


def _select_skip_from_importance(li: np.ndarray, threshold: float) -> set[int]:
    return {i for i, v in enumerate(li) if v < threshold}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mode", choices=["hard", "soft"], default="hard")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--skip-thr", type=float, default=0.10,
                    help="layers con importance < thr sono skipped (default 0.10)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pivot-layer", type=int, default=9)
    ap.add_argument("--map-dir", type=str, default=None)
    args = ap.parse_args()

    map_dir = Path(args.map_dir) if args.map_dir else (
        ROOT / "mappa" / "topology_llama32_3b"
    )

    print("Loading MMLU all/test...", flush=True)
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=args.n, replace=False)
    prompts = [_format_mmlu(ds[int(i)]) for i in idx]
    answers = [int(ds[int(i)]["answer"]) for i in idx]
    subjects = [ds[int(i)]["subject"] for i in idx]
    print(f"  N={len(prompts)} subjects_distinct={len(set(subjects))}", flush=True)

    print("\nLoading LlamaSkipper (router + decoder)...", flush=True)
    skipper = LlamaSkipper()
    n_layers = skipper.n_layers
    print(f"  n_layers={n_layers}  hidden={skipper.hidden_size}", flush=True)

    print(f"\n[A] Embed {len(prompts)} prompts at L{args.pivot_layer}...", flush=True)
    embeddings = np.empty((len(prompts), skipper.hidden_size), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(prompts):
        embeddings[i] = skipper.embed(p, layer_idx=args.pivot_layer).numpy()
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(prompts)}] {(i+1)/(time.time()-t0):.1f} p/s",
                  flush=True)
    print(f"  Embed done in {(time.time()-t0):.1f}s", flush=True)

    print(f"\n[B] Mappa lookup + skip plans...", flush=True)
    tmap = TopologicalMap.load(map_dir)
    assert tmap.n_decoder_layers == n_layers
    skip_plans: list[set[int]] = []
    matched_cats: list[str] = []
    sims: list[float] = []
    for emb in embeddings:
        top = tmap.lookup(emb, k=1)
        sim, _, entry = top[0]
        sims.append(sim)
        matched_cats.append(entry.domain)
        if entry.layer_importance is not None:
            skip_plans.append(_select_skip_from_importance(entry.layer_importance,
                                                            args.skip_thr))
        else:
            skip_plans.append(set())
    print(f"  Matched cats: {sorted(set(matched_cats))}", flush=True)
    print(f"  Sim mean={np.mean(sims):.3f}  min={np.min(sims):.3f}  "
          f"max={np.max(sims):.3f}", flush=True)
    avg_skip = np.mean([len(s) for s in skip_plans])
    print(f"  Avg skip layers: {avg_skip:.1f}/{n_layers} = {avg_skip/n_layers*100:.1f}%",
          flush=True)

    # Letter token IDs
    tok = skipper.tokenizer
    letter_ids = []
    for letter in ("A", "B", "C", "D"):
        ids = tok.encode(" " + letter, add_special_tokens=False)
        letter_ids.append(ids[0])
    print(f"  letter token IDs: {dict(zip('ABCD', letter_ids))}", flush=True)

    print(f"\n[C] Forward baseline + AIS ({args.mode}, "
          f"{'α='+str(args.alpha) if args.mode=='soft' else 'hard'})...",
          flush=True)
    base_top1, ais_top1 = [], []
    base_choice, ais_choice = [], []
    tot_base_t = 0.0
    tot_ais_t = 0.0
    t_start = time.time()

    for i, (p, skip) in enumerate(zip(prompts, skip_plans)):
        t0 = time.time()
        r_base = skipper.forward(p)
        tot_base_t += time.time() - t0

        t1 = time.time()
        if args.mode == "hard":
            r_ais = skipper.forward(p, hard_skip=skip)
        else:
            soft = {i_: args.alpha for i_ in skip}
            r_ais = skipper.forward(p, soft_skip=soft)
        tot_ais_t += time.time() - t1

        b_lett = [r_base[lid].item() for lid in letter_ids]
        a_lett = [r_ais[lid].item() for lid in letter_ids]
        base_choice.append(int(np.argmax(b_lett)))
        ais_choice.append(int(np.argmax(a_lett)))
        base_top1.append(int(r_base.argmax().item()))
        ais_top1.append(int(r_ais.argmax().item()))

        b_corr = base_choice[-1] == answers[i]
        a_corr = ais_choice[-1] == answers[i]
        agree = base_top1[-1] == ais_top1[-1]
        if (i + 1) % 10 == 0 or i < 5:
            elapsed = (time.time() - t_start) / 60
            eta = (len(prompts) - i - 1) * (elapsed / (i + 1))
            print(f"  [{i+1:3d}/{len(prompts)}] cat={matched_cats[i]:18s} "
                  f"skip={len(skip):2d}  GT={'ABCD'[answers[i]]}  "
                  f"base={'ABCD'[base_choice[-1]]}{'✓' if b_corr else 'x'}  "
                  f"ais={'ABCD'[ais_choice[-1]]}{'✓' if a_corr else 'x'}  "
                  f"({elapsed:.1f}m ETA {eta:.1f}m)", flush=True)

    answers_arr = np.array(answers)
    base_choice_arr = np.array(base_choice)
    ais_choice_arr = np.array(ais_choice)
    acc_base = float((base_choice_arr == answers_arr).mean())
    acc_ais = float((ais_choice_arr == answers_arr).mean())
    top1_agree = float(np.mean(np.array(base_top1) == np.array(ais_top1)))
    mean_base = tot_base_t / len(prompts)
    mean_ais = tot_ais_t / len(prompts)
    saving = (1 - mean_ais / mean_base) * 100

    print("\n" + "=" * 70, flush=True)
    print(f"  Model: Llama 3.2 3B  N={len(prompts)}  mode={args.mode}", flush=True)
    print(f"  Accuracy baseline:  {acc_base*100:.1f}%  "
          f"({int((base_choice_arr==answers_arr).sum())}/{len(answers)})", flush=True)
    print(f"  Accuracy AIS:       {acc_ais*100:.1f}%  "
          f"({int((ais_choice_arr==answers_arr).sum())}/{len(answers)})  "
          f"(Δ={100*(acc_ais-acc_base):+.1f}pp)", flush=True)
    print(f"  Top-1 agree:        {top1_agree*100:.1f}%", flush=True)
    print(f"  Mean latency base:  {mean_base:.3f}s", flush=True)
    print(f"  Mean latency AIS:   {mean_ais:.3f}s  ({saving:+.1f}% saving)", flush=True)
    print(f"  Avg skipped layers: {avg_skip:.1f}/{n_layers}  "
          f"({avg_skip/n_layers*100:.1f}%)", flush=True)
    print("=" * 70, flush=True)

    tag = f"hard" if args.mode == "hard" else f"soft_a{args.alpha:.1f}"
    out_npz = RESULTS_DIR / f"mmlu_llama32_n{args.n}_{tag}.npz"
    np.savez(
        out_npz,
        prompts=np.array(prompts, dtype=object),
        subjects=np.array(subjects, dtype=object),
        answers=answers_arr,
        base_choice=base_choice_arr,
        ais_choice=ais_choice_arr,
        base_top1=np.array(base_top1),
        ais_top1=np.array(ais_top1),
        matched_cats=np.array(matched_cats, dtype=object),
        skip_plans=np.array([sorted(s) for s in skip_plans], dtype=object),
        sims=np.array(sims),
        meta=np.array({
            "mode": args.mode, "alpha": args.alpha, "skip_thr": args.skip_thr,
            "acc_base": acc_base, "acc_ais": acc_ais,
            "top1_agreement": top1_agree,
            "mean_base_t": mean_base, "mean_ais_t": mean_ais,
            "saving_pct": saving, "avg_skip_layers": avg_skip,
        }, dtype=object),
    )
    print(f"  Saved {out_npz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
