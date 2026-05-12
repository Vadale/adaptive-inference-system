"""exp_008 — Benchmark MMLU AIS HIGH path vs baseline.

Misura il valore aggiunto di AIS su un benchmark standardizzato (MMLU
multiple-choice). Setup:
  1) Pre-embed N=20 prompt MMLU con il cervelletto (Gemma 4 E2B).
  2) Unload cervelletto (E2B+E4B non coabitano su 16 GB unified memory).
  3) Per ogni prompt:
     a) baseline forward (cervellone E4B, tutti i 42 layer)
     b) AIS HIGH forward: confidence_threshold abbassato a 0 → forza HIGH
        path. Skip secondo la `layer_importance` della mappa per la categoria
        più simile (nearest neighbor cervelletto embedding). α=0.7 (sweet
        spot da exp_006).
  4) Per ogni prompt:
     - estraggo top-1 next token e cerco quale tra ' A', ' B', ' C', ' D'
       ha probabilità più alta (= risposta del modello)
     - confronto con ground truth (campo `answer` MMLU)
     - top-1 agreement baseline vs AIS

Metriche aggregate:
  - accuracy_baseline (= modello puro)
  - accuracy_AIS (= con soft skip α=0.7)
  - top-1 agreement medio
  - latency medio per forward

Time: 20 prompt × 2 forward = 40 forward × ~70s = ~47 min.

Prerequisiti: mappa popolata (exp_005), layer_skipper con soft skip (Step 1 PASS).
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.mappa import TopologicalMap

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MMLU_REPO = "cais/mmlu"
MMLU_SUBSET = "all"

CERVELLETTO_PIVOT = 9
CERVELLETTO_DEVICE_MAP = {
    "model.vision_tower": "cpu", "model.audio_tower": "cpu",
    "model.embed_vision": "cpu", "model.embed_audio": "cpu",
    "model.language_model": "mps", "lm_head": "mps",
}


def _format_mmlu(row) -> str:
    """Standard MMLU few-shot prompt format."""
    q = row["question"]
    choices = row["choices"]
    return (
        f"Answer the following multiple choice question.\n\n"
        f"Question: {q}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"Answer:"
    )


def _select_skip_from_importance(li: np.ndarray, threshold: float = 0.10) -> list[int]:
    return [i for i, v in enumerate(li) if v < threshold]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="numero domande MMLU")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--skip-thr", type=float, default=0.10,
                    help="layer con importance < threshold sono skippati")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--cervellone-model-id", type=str, default="google/gemma-4-E4B-it")
    ap.add_argument("--map-dir", type=str, default=None)
    args = ap.parse_args()
    is_e2b = "E2B" in args.cervellone_model_id
    MAP_DIR = Path(args.map_dir) if args.map_dir else (
        ROOT / "mappa" / ("topology_e2b" if is_e2b else "topology")
    )

    print(f"Loading MMLU {MMLU_SUBSET} split=test...", flush=True)
    ds = load_dataset(MMLU_REPO, MMLU_SUBSET, split="test")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=args.n, replace=False)
    prompts = [_format_mmlu(ds[int(i)]) for i in idx]
    answers = [int(ds[int(i)]["answer"]) for i in idx]   # 0=A, 1=B, 2=C, 3=D
    subjects = [ds[int(i)]["subject"] for i in idx]
    print(f"  Sampled N={len(prompts)} domande, {len(set(subjects))} subject distinct",
          flush=True)

    # ---------- STEP A: embed con E2B ----------
    print(f"\n[A] Loading cervelletto E2B per embedding...", flush=True)
    from nnsight import VisionLanguageModel
    enc = VisionLanguageModel(
        "google/gemma-4-E2B-it", dtype=torch.bfloat16, device_map=CERVELLETTO_DEVICE_MAP
    )
    proc = enc.processor

    def _chatify(text):
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    print(f"  Embedding {len(prompts)} prompts (~{len(prompts) * 5 / 60:.1f} min)...",
          flush=True)
    embeddings = np.empty((len(prompts), 1536), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(prompts):
        text = _chatify(p[:500])
        holder = [None]
        with torch.no_grad():
            with enc.trace(text):
                holder[0] = enc.model.language_model.layers[CERVELLETTO_PIVOT].output.save()
        v = holder[0][0] if isinstance(holder[0], tuple) else holder[0]
        embeddings[i] = v[0, -1, :].float().cpu().numpy()
        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{len(prompts)}] {(i+1)/(time.time()-t0):.1f} p/s",
                  flush=True)
    print(f"  Done embedding in {(time.time()-t0)/60:.1f} min", flush=True)

    # Unload E2B
    del enc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ---------- STEP B: mappa lookup + decide skip per prompt ----------
    print(f"\n[B] Mappa lookup per skip plan...", flush=True)
    tmap = TopologicalMap.load(MAP_DIR)
    skip_plans: list[list[int]] = []
    matched_cats: list[str] = []
    sims: list[float] = []
    for emb in embeddings:
        top = tmap.lookup(emb, k=1)
        if top and top[0][2].layer_importance is not None:
            sim, _, entry = top[0]
            sims.append(sim)
            matched_cats.append(entry.domain)
            skip_plans.append(
                _select_skip_from_importance(entry.layer_importance, args.skip_thr)
            )
        else:
            sims.append(0.0)
            matched_cats.append("?")
            skip_plans.append([])
    print(f"  Matched categorie: {sorted(set(matched_cats))}", flush=True)
    print(f"  Sim media: {np.mean(sims):.3f}  min: {np.min(sims):.3f}  "
          f"max: {np.max(sims):.3f}", flush=True)

    # ---------- STEP C: cervellone forward ----------
    print(f"\n[C] Loading cervellone {args.cervellone_model_id}...", flush=True)
    from cervellone.layer_skipper import AdaptiveLayerSkipper
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
    skipper = AdaptiveLayerSkipper(model_id=args.cervellone_model_id,
                                    device_map=dm, max_memory=mm)
    n_layers = skipper.n_layers

    print(f"\n[D] Forward baseline + AIS HIGH α={args.alpha} per {len(prompts)} prompt...",
          flush=True)
    base_top1: list[int] = []
    ais_top1: list[int] = []
    base_choice: list[int] = []  # 0-3 or -1 if non riconosciuto
    ais_choice: list[int] = []
    base_logits_all = []
    ais_logits_all = []
    tot_base_t = 0.0
    tot_ais_t = 0.0

    # Per estrarre la lettera scelta, vediamo quale token tra A/B/C/D ha
    # probabilità più alta. Pre-calcolo i token IDs.
    tok = skipper.processor.tokenizer
    letter_ids = []
    for letter in ("A", "B", "C", "D"):
        ids = tok.encode(" " + letter, add_special_tokens=False)
        # Prendi il primo token "interessante" (spazio + lettera)
        letter_ids.append(ids[0] if len(ids) >= 1 else tok.encode(letter, add_special_tokens=False)[0])
    print(f"  letter token IDs: {dict(zip('ABCD', letter_ids))}", flush=True)

    t_start = time.time()
    for i, (p, skip) in enumerate(zip(prompts, skip_plans)):
        # baseline
        t0 = time.time()
        r_base = skipper.forward(p, active_layers=None)
        tot_base_t += time.time() - t0
        # AIS HIGH
        active = set(range(n_layers)) - set(skip)
        t1 = time.time()
        r_ais = skipper.forward(p, active_layers=active, alpha=args.alpha)
        tot_ais_t += time.time() - t1

        base_logits_all.append(r_base.logits_last.numpy())
        ais_logits_all.append(r_ais.logits_last.numpy())

        # Estraggo la lettera scelta = argmax tra i 4 letter_ids
        base_letter_logits = [r_base.logits_last[lid].item() for lid in letter_ids]
        ais_letter_logits = [r_ais.logits_last[lid].item() for lid in letter_ids]
        base_choice.append(int(np.argmax(base_letter_logits)))
        ais_choice.append(int(np.argmax(ais_letter_logits)))
        base_top1.append(int(r_base.logits_last.argmax().item()))
        ais_top1.append(int(r_ais.logits_last.argmax().item()))

        elapsed = (time.time() - t_start) / 60
        eta = (len(prompts) - i - 1) * (elapsed / (i + 1))
        b_corr = base_choice[-1] == answers[i]
        a_corr = ais_choice[-1] == answers[i]
        agree = base_top1[-1] == ais_top1[-1]
        print(f"  [{i+1:2d}/{len(prompts)}] cat={matched_cats[i]:18s} skip={len(skip):2d}  "
              f"GT={'ABCD'[answers[i]]}  base={'ABCD'[base_choice[-1]]}{'✓' if b_corr else 'x'}  "
              f"ais={'ABCD'[ais_choice[-1]]}{'✓' if a_corr else 'x'}  "
              f"top1_agree={agree}  ({elapsed:.1f}m, ETA {eta:.1f}m)", flush=True)

    # ---------- STEP E: aggregate ----------
    answers_arr = np.array(answers)
    base_choice_arr = np.array(base_choice)
    ais_choice_arr = np.array(ais_choice)
    acc_base = float((base_choice_arr == answers_arr).mean())
    acc_ais = float((ais_choice_arr == answers_arr).mean())
    top1_agreement = float(np.mean(np.array(base_top1) == np.array(ais_top1)))

    print("\n" + "=" * 70, flush=True)
    print(f"  Accuracy baseline:        {acc_base*100:.1f}%  ({(base_choice_arr == answers_arr).sum()}/{len(answers)})",
          flush=True)
    print(f"  Accuracy AIS (α={args.alpha}):    {acc_ais*100:.1f}%  ({(ais_choice_arr == answers_arr).sum()}/{len(answers)})",
          flush=True)
    print(f"  Top-1 agreement AIS↔base: {top1_agreement*100:.1f}%", flush=True)
    print(f"  Mean latency baseline:    {tot_base_t/len(prompts):.1f}s", flush=True)
    print(f"  Mean latency AIS:         {tot_ais_t/len(prompts):.1f}s  "
          f"({100*(tot_ais_t/tot_base_t-1):+.1f}% vs baseline)", flush=True)
    print(f"  Layer skip media:         {np.mean([len(s) for s in skip_plans]):.1f}/{n_layers}",
          flush=True)
    print("=" * 70, flush=True)

    # Save
    out_npz = RESULTS_DIR / f"mmlu_benchmark_n{args.n}_a{args.alpha:.1f}.npz"
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
        skip_plans=np.array(skip_plans, dtype=object),
        sims=np.array(sims),
        meta=np.array({"alpha": args.alpha, "skip_thr": args.skip_thr,
                       "acc_base": acc_base, "acc_ais": acc_ais,
                       "top1_agreement": top1_agreement,
                       "tot_base_t": tot_base_t, "tot_ais_t": tot_ais_t}, dtype=object),
    )
    print(f"  Saved {out_npz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
