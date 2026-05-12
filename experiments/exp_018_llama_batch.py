"""exp_018 — batch throughput Llama 3.2 3B AIS vs baseline.

Mirror exp_011 but Llama (which supports real hard skip → real RAM/compute
saving inside the inner forward, no nnsight overhead).

Measures requests/second on batches of size B in [1, 2, 4, 8].

Protocol:
  - 2 warmup forwards
  - 3 repeats per (batch_size, mode), median taken
  - hard skip ~50% layers persistent (apply_skip + restore around each mode)
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skippers.llama_skipper import LlamaSkipper

PROMPTS = [
    "What is the capital of France?",
    "Solve: 17 times 23 equals?",
    "Who wrote the play Hamlet?",
    "What is the largest planet in our solar system?",
    "What is the boiling point of water in Celsius?",
    "Translate 'good morning' to Spanish.",
    "What year did World War 2 end?",
    "What is the chemical symbol for gold?",
]

BATCH_SIZES = [1, 2, 4, 8]
N_REPEATS = 3


def _chatify(tok, text):
    msgs = [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _time_batch(skipper, prompts, batch_size, n_repeats):
    batch = [prompts[i % len(prompts)] for i in range(batch_size)]
    texts = [_chatify(skipper.tokenizer, p) for p in batch]
    times = []
    for _ in range(n_repeats):
        inputs = skipper.tokenizer(texts, return_tensors="pt", padding=True).to(
            skipper.model.device
        )
        t0 = time.time()
        with torch.no_grad():
            _ = skipper.model(**inputs)
        dt = time.time() - t0
        times.append(dt)
    return times


def main() -> int:
    print("Loading LlamaSkipper...", flush=True)
    skipper = LlamaSkipper()
    n = skipper.n_layers
    # Hard skip ~50% (mid+late layers)
    hard_skip = set(range(7, 14)) | set(range(21, 28))
    print(f"  n_layers={n}  hard_skip={len(hard_skip)}/{n} = "
          f"{len(hard_skip)/n*100:.0f}%", flush=True)

    print("\n[warmup] 2 forwards...", flush=True)
    for _ in range(2):
        _time_batch(skipper, PROMPTS, 1, 1)
    print("  done", flush=True)

    results = {}
    for B in BATCH_SIZES:
        print(f"\n=== batch_size={B} ===", flush=True)
        # Baseline
        skipper.restore()
        t_base = _time_batch(skipper, PROMPTS, B, N_REPEATS)
        m_base = statistics.median(t_base)
        rps_base = B / m_base
        print(f"  baseline:  times={[f'{t:.2f}' for t in t_base]}  "
              f"median={m_base:.2f}s  req/s={rps_base:.2f}", flush=True)

        # AIS
        skipper.apply_skip(hard_skip=hard_skip)
        t_ais = _time_batch(skipper, PROMPTS, B, N_REPEATS)
        m_ais = statistics.median(t_ais)
        rps_ais = B / m_ais
        skipper.restore()
        print(f"  AIS skip:  times={[f'{t:.2f}' for t in t_ais]}  "
              f"median={m_ais:.2f}s  req/s={rps_ais:.2f}", flush=True)

        saving = (1 - m_ais / m_base) * 100
        speedup = rps_ais / rps_base
        results[B] = {
            "base_med": m_base, "base_rps": rps_base,
            "ais_med": m_ais, "ais_rps": rps_ais,
            "saving_pct": saving, "speedup_x": speedup,
            "base_times": t_base, "ais_times": t_ais,
        }
        print(f"  → saving: {saving:+.1f}%  speedup: {speedup:.2f}x", flush=True)

    print("\n" + "=" * 76, flush=True)
    print(f"  {'B':>3} | {'base med':>10} | {'AIS med':>10} | "
          f"{'base r/s':>9} | {'AIS r/s':>8} | {'saving':>8} | {'speedup':>8}",
          flush=True)
    print("-" * 76, flush=True)
    for B in BATCH_SIZES:
        r = results[B]
        print(f"  {B:>3} | {r['base_med']:>8.2f}s | {r['ais_med']:>8.2f}s | "
              f"{r['base_rps']:>8.2f} | {r['ais_rps']:>7.2f} | "
              f"{r['saving_pct']:>+7.1f}% | {r['speedup_x']:>7.2f}x", flush=True)
    print("=" * 76, flush=True)

    out = ROOT / "results" / "batch_throughput_llama32.txt"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        f.write("  B | base_med | AIS_med | base_rps | AIS_rps | saving | speedup\n")
        for B in BATCH_SIZES:
            r = results[B]
            f.write(f"  {B} | {r['base_med']:.2f}s | {r['ais_med']:.2f}s | "
                    f"{r['base_rps']:.2f} | {r['ais_rps']:.2f} | "
                    f"{r['saving_pct']:+.1f}% | {r['speedup_x']:.2f}x\n")
    print(f"  Saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
