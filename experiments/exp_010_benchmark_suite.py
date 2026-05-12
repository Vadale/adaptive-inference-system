"""exp_010 — benchmark suite generico AIS vs baseline.

Supporta multiple datasets multiple-choice in formato unificato:
  --benchmark mmlu_classic | mmlu_pro | arc_challenge | mmmlu_it | hellaswag

Pattern unificato:
  1) Carica dataset, normalizza in (prompt_text, gt_idx, n_choices)
  2) Pre-embed via cervelletto (Gemma 4 E2B L09 last-token)
  3) Mappa lookup → categoria + skip plan
  4) Forward baseline + AIS HIGH (α=0.7) sul cervellone
  5) Accuracy A/B/C/... + top-1 agreement + latency

Stima per N=100 su E2B native: ~5-10 min/benchmark.
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

CERVELLETTO_PIVOT = 9
CERVELLETTO_DEVICE_MAP = {
    "model.vision_tower": "cpu", "model.audio_tower": "cpu",
    "model.embed_vision": "cpu", "model.embed_audio": "cpu",
    "model.language_model": "mps", "lm_head": "mps",
}


# --- loaders normalizzati ---
def load_mmlu_classic(n, seed):
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    items = []
    for i in idx:
        row = ds[int(i)]
        q = row["question"]
        c = row["choices"]
        text = (f"Answer the following multiple choice question.\n\n"
                f"Question: {q}\n"
                f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:")
        items.append((text, int(row["answer"]), 4, row["subject"]))
    return items


def load_mmlu_pro(n, seed):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    items = []
    letters = "ABCDEFGHIJ"
    for i in idx:
        row = ds[int(i)]
        opts = row["options"]
        nopt = len(opts)
        text = f"Answer the following multiple choice question.\n\nQuestion: {row['question']}\n"
        for k, opt in enumerate(opts):
            text += f"{letters[k]}. {opt}\n"
        text += "Answer:"
        items.append((text, int(row["answer_index"]), nopt, row["category"]))
    return items


def load_arc_challenge(n, seed):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    items = []
    letters = "ABCDEFGH"
    for i in idx:
        row = ds[int(i)]
        choices_text = row["choices"]["text"]
        choices_label = row["choices"]["label"]   # es. ['A','B','C','D'] o ['1','2','3','4']
        # Normalizza in lettere ABCD
        gt_label = row["answerKey"]
        # Trova idx della label corretta
        try:
            gt_idx = choices_label.index(gt_label)
        except ValueError:
            continue   # skip se mismatch
        nopt = len(choices_text)
        text = f"Answer the following multiple choice question.\n\nQuestion: {row['question']}\n"
        for k, opt in enumerate(choices_text):
            text += f"{letters[k]}. {opt}\n"
        text += "Answer:"
        items.append((text, gt_idx, nopt, "science"))
    return items


def load_mmmlu_it(n, seed):
    ds = load_dataset("openai/MMMLU", "IT_IT", split="test")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    items = []
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    for i in idx:
        row = ds[int(i)]
        text = (f"Rispondi alla seguente domanda a scelta multipla.\n\n"
                f"Domanda: {row['Question']}\n"
                f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\nRisposta:")
        items.append((text, letter_to_idx[row["Answer"]], 4, row["Subject"]))
    return items


def load_hellaswag(n, seed):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    items = []
    for i in idx:
        row = ds[int(i)]
        ctx = row["ctx"]
        endings = row["endings"]
        text = (f"Continue the sentence below by choosing the most plausible ending.\n\n"
                f"Sentence: {ctx}\n"
                f"A. {endings[0]}\nB. {endings[1]}\nC. {endings[2]}\nD. {endings[3]}\n"
                f"Answer:")
        items.append((text, int(row["label"]), 4, row["activity_label"]))
    return items


LOADERS = {
    "mmlu_classic": load_mmlu_classic,
    "mmlu_pro": load_mmlu_pro,
    "arc_challenge": load_arc_challenge,
    "mmmlu_it": load_mmmlu_it,
    "hellaswag": load_hellaswag,
}


def _select_skip(li: np.ndarray, thr: float = 0.10) -> list[int]:
    return [i for i, v in enumerate(li) if v < thr]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=list(LOADERS.keys()), required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--skip-thr", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--cervellone-model-id", type=str, default="google/gemma-4-E2B-it")
    args = ap.parse_args()

    is_e2b = "E2B" in args.cervellone_model_id
    MAP_DIR = ROOT / "mappa" / ("topology_e2b" if is_e2b else "topology")
    print(f"Benchmark: {args.benchmark}  N={args.n}  α={args.alpha}", flush=True)
    print(f"MAP_DIR={MAP_DIR}", flush=True)

    print(f"\nLoading dataset...", flush=True)
    items = LOADERS[args.benchmark](args.n, args.seed)
    print(f"  loaded {len(items)} items", flush=True)
    if len(items) < args.n:
        print(f"  WARN: dataset returned {len(items)} < {args.n} (some filtered)", flush=True)

    # --- STEP A: cervelletto embed ---
    print(f"\n[A] Loading cervelletto E2B per embedding...", flush=True)
    from nnsight import VisionLanguageModel
    enc = VisionLanguageModel(
        "google/gemma-4-E2B-it", dtype=torch.bfloat16, device_map=CERVELLETTO_DEVICE_MAP
    )
    proc_enc = enc.processor

    def _chatify(text):
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        return proc_enc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    embeddings = np.empty((len(items), 1536), dtype=np.float32)
    t0 = time.time()
    for i, (prompt, gt_idx, nopt, subj) in enumerate(items):
        text = _chatify(prompt[:600])
        holder = [None]
        with torch.no_grad():
            with enc.trace(text):
                holder[0] = enc.model.language_model.layers[CERVELLETTO_PIVOT].output.save()
        v = holder[0][0] if isinstance(holder[0], tuple) else holder[0]
        embeddings[i] = v[0, -1, :].float().cpu().numpy()
        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(items)}] {(i+1)/(time.time()-t0):.1f} p/s", flush=True)
    print(f"  done embedding in {(time.time()-t0)/60:.1f} min", flush=True)

    del enc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- STEP B: mappa lookup ---
    print(f"\n[B] Mappa lookup + skip plans...", flush=True)
    tmap = TopologicalMap.load(MAP_DIR)
    skip_plans = []
    matched_cats = []
    sims = []
    for emb in embeddings:
        top = tmap.lookup(emb, k=1)
        if top and top[0][2].layer_importance is not None:
            sim, _, entry = top[0]
            sims.append(sim)
            matched_cats.append(entry.domain)
            skip_plans.append(_select_skip(entry.layer_importance, args.skip_thr))
        else:
            sims.append(0.0); matched_cats.append("?"); skip_plans.append([])
    from collections import Counter
    print(f"  matched cat distribution: {dict(Counter(matched_cats))}", flush=True)
    print(f"  sim media={np.mean(sims):.3f}  min={np.min(sims):.3f}  max={np.max(sims):.3f}",
          flush=True)

    # --- STEP C: cervellone forward ---
    print(f"\n[C] Loading cervellone {args.cervellone_model_id}...", flush=True)
    from cervellone.layer_skipper import AdaptiveLayerSkipper
    if is_e2b:
        dm = CERVELLETTO_DEVICE_MAP
        mm = None
    else:
        dm = "auto"
        mm = {"mps": "8GiB", "cpu": "30GiB"}
    skipper = AdaptiveLayerSkipper(model_id=args.cervellone_model_id,
                                    device_map=dm, max_memory=mm)
    n_layers = skipper.n_layers

    # Letter token IDs (per gli answer A/B/C/...)
    tok = skipper.processor.tokenizer
    max_letters = max(nopt for _, _, nopt, _ in items)
    letters = "ABCDEFGHIJ"[:max_letters]
    letter_ids = []
    for letter in letters:
        ids = tok.encode(" " + letter, add_special_tokens=False)
        letter_ids.append(ids[0])
    print(f"  letter token IDs (per {len(letters)} choices): {dict(zip(letters, letter_ids))}",
          flush=True)

    base_choices = []; ais_choices = []; gt_list = []
    base_top1 = []; ais_top1 = []
    tot_base_t = 0.0; tot_ais_t = 0.0
    t_start = time.time()
    for i, ((prompt, gt_idx, nopt, subj), skip) in enumerate(zip(items, skip_plans)):
        active = set(range(n_layers)) - set(skip)
        # baseline
        t0 = time.time()
        r_base = skipper.forward(prompt, active_layers=None)
        tot_base_t += time.time() - t0
        # AIS
        t0 = time.time()
        r_ais = skipper.forward(prompt, active_layers=active, alpha=args.alpha)
        tot_ais_t += time.time() - t0

        # Estrai answer letter (argmax tra i first `nopt` letter_ids)
        bl = r_base.logits_last
        al = r_ais.logits_last
        b_letter_logits = [bl[lid].item() for lid in letter_ids[:nopt]]
        a_letter_logits = [al[lid].item() for lid in letter_ids[:nopt]]
        b_choice = int(np.argmax(b_letter_logits))
        a_choice = int(np.argmax(a_letter_logits))
        base_choices.append(b_choice); ais_choices.append(a_choice)
        gt_list.append(gt_idx)
        base_top1.append(int(bl.argmax().item()))
        ais_top1.append(int(al.argmax().item()))

        elapsed = (time.time() - t_start) / 60
        eta = (len(items) - i - 1) * (elapsed / (i + 1))
        if (i + 1) % 10 == 0 or i < 5:
            print(f"  [{i+1:3d}/{len(items)}] cat={matched_cats[i][:15]:15s} "
                  f"skip={len(skip):2d}  GT={letters[gt_idx]} "
                  f"b={letters[b_choice]}{'+' if b_choice==gt_idx else '-'} "
                  f"a={letters[a_choice]}{'+' if a_choice==gt_idx else '-'}  "
                  f"({elapsed:.1f}m, ETA {eta:.1f}m)", flush=True)

    # Aggregate
    base_acc = float(np.mean(np.array(base_choices) == np.array(gt_list)))
    ais_acc = float(np.mean(np.array(ais_choices) == np.array(gt_list)))
    top1_agree = float(np.mean(np.array(base_top1) == np.array(ais_top1)))

    print("\n" + "=" * 70, flush=True)
    print(f"  Benchmark:                 {args.benchmark}", flush=True)
    print(f"  N:                         {len(items)}", flush=True)
    print(f"  Accuracy baseline:         {base_acc*100:.1f}%  "
          f"({sum(b==g for b,g in zip(base_choices, gt_list))}/{len(items)})", flush=True)
    print(f"  Accuracy AIS (α={args.alpha}):     {ais_acc*100:.1f}%  "
          f"({sum(a==g for a,g in zip(ais_choices, gt_list))}/{len(items)})", flush=True)
    print(f"  Δ accuracy:                {(ais_acc-base_acc)*100:+.1f}%", flush=True)
    print(f"  Top-1 agreement:           {top1_agree*100:.1f}%", flush=True)
    print(f"  Mean latency baseline:     {tot_base_t/len(items):.3f}s", flush=True)
    print(f"  Mean latency AIS:          {tot_ais_t/len(items):.3f}s  "
          f"({100*(tot_ais_t/tot_base_t-1):+.1f}%)", flush=True)
    print(f"  Layer skip media:          {np.mean([len(s) for s in skip_plans]):.1f}/{n_layers}",
          flush=True)
    print("=" * 70, flush=True)

    out = RESULTS_DIR / f"bench_{args.benchmark}_n{args.n}_a{args.alpha:.1f}.npz"
    np.savez(out,
             benchmark=args.benchmark, n=len(items),
             base_acc=base_acc, ais_acc=ais_acc, top1_agree=top1_agree,
             base_choices=np.array(base_choices), ais_choices=np.array(ais_choices),
             gt=np.array(gt_list),
             matched_cats=np.array(matched_cats, dtype=object),
             sims=np.array(sims),
             tot_base_t=tot_base_t, tot_ais_t=tot_ais_t)
    print(f"  Saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
