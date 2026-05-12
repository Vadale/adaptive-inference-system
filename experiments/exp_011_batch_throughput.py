"""exp_011 — batch throughput benchmark: AIS vs baseline.

Measures requests-per-second on parallel batches of size B in [1, 2, 4, 8].
This is the key metric for multi-user API serving: if AIS scales linearly
with batch size, then a single server can handle more users at the same
hardware cost.

Setup:
  - 8 different prompts (MMLU-style multiple choice, single-token answer)
  - For each batch size B: pad B prompts together, run single forward
  - Compare baseline (no skip) vs AIS (hard_skip persistent 33%)
  - Metric: requests/second and per-request latency

Protocol:
  - 2 warmup forwards (discarded)
  - 3 repeats per (batch_size, mode), median taken
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cervellone.native_skip import NativeLayerSkipper

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
# Hard skip 33% of layers (group g1 [L07-13] + g5 [L28-34] on E2B 35-layer)
HARD_SKIP = set(range(7, 14)) | set(range(28, 35))   # 14/35 = 40%


def _chatify(proc, text):
    msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _time_batch(native, prompts, batch_size, n_repeats):
    """Run forward on a batch of size `batch_size` (cycling through prompts),
    repeated `n_repeats` times. Returns list of wall times (in seconds)."""
    batch = [prompts[i % len(prompts)] for i in range(batch_size)]
    texts = [_chatify(native.processor, p) for p in batch]
    times = []
    for r in range(n_repeats):
        # Process batch (padding=True per allineare le lunghezze)
        inputs = native.processor(text=texts, return_tensors="pt",
                                    padding=True)
        prepared = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        t0 = time.time()
        with torch.no_grad():
            _ = native.hf_model(**prepared)
        dt = time.time() - t0
        times.append(dt)
    return times


def main() -> int:
    print("Loading NativeLayerSkipper (E2B, explicit)...", flush=True)
    # E2B device_map fisso (10GB fits in MPS 8GB+ con vision/audio off)
    dm = {
        "model.vision_tower": "cpu", "model.audio_tower": "cpu",
        "model.embed_vision": "cpu", "model.embed_audio": "cpu",
        "model.language_model": "mps", "lm_head": "mps",
    }
    native = NativeLayerSkipper(
        model_id="google/gemma-4-E2B-it", device_map=dm, max_memory=None
    )
    n = native.n_layers
    print(f"  n_layers={n}  hard_skip={len(HARD_SKIP)} layer ({len(HARD_SKIP)/n*100:.0f}%)",
          flush=True)

    # Warmup
    print("\n[warmup] 2 forward to warm MPS cache...", flush=True)
    for _ in range(2):
        _ = _time_batch(native, PROMPTS, 1, 1)
    print("  done", flush=True)

    results = {}   # results[batch_size][mode] = list of times

    for B in BATCH_SIZES:
        print(f"\n=== batch_size={B} ===", flush=True)
        # Baseline (no skip)
        native.restore()
        times_base = _time_batch(native, PROMPTS, B, N_REPEATS)
        med_base = statistics.median(times_base)
        rps_base = B / med_base
        print(f"  baseline:  times={[f'{t:.2f}' for t in times_base]}  "
              f"median={med_base:.2f}s  req/s={rps_base:.2f}", flush=True)

        # AIS (hard skip persistent)
        native.apply_skip(hard_skip=HARD_SKIP)
        times_ais = _time_batch(native, PROMPTS, B, N_REPEATS)
        med_ais = statistics.median(times_ais)
        rps_ais = B / med_ais
        print(f"  AIS skip:  times={[f'{t:.2f}' for t in times_ais]}  "
              f"median={med_ais:.2f}s  req/s={rps_ais:.2f}", flush=True)

        native.restore()
        saving = (1 - med_ais / med_base) * 100
        speedup = rps_ais / rps_base
        results[B] = {
            "baseline_times": times_base, "baseline_median": med_base, "baseline_rps": rps_base,
            "ais_times": times_ais, "ais_median": med_ais, "ais_rps": rps_ais,
            "saving_pct": saving, "speedup_x": speedup,
        }
        print(f"  → saving: {saving:+.1f}%  speedup: {speedup:.2f}x", flush=True)

    # Summary table
    print("\n" + "=" * 76, flush=True)
    print(f"  {'B':>3} | {'base median':>12} | {'AIS median':>12} | "
          f"{'base req/s':>11} | {'AIS req/s':>10} | {'saving':>8} | {'speedup':>8}", flush=True)
    print("-" * 76, flush=True)
    for B in BATCH_SIZES:
        r = results[B]
        print(f"  {B:>3} | {r['baseline_median']:>10.2f}s | {r['ais_median']:>10.2f}s | "
              f"{r['baseline_rps']:>10.2f} | {r['ais_rps']:>9.2f} | "
              f"{r['saving_pct']:>+7.1f}% | {r['speedup_x']:>7.2f}x", flush=True)
    print("=" * 76, flush=True)

    # Save
    out = ROOT / "results" / "batch_throughput_bench.txt"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        f.write(f"  B | base_median | AIS_median | base_rps | AIS_rps | saving | speedup\n")
        for B in BATCH_SIZES:
            r = results[B]
            f.write(f"  {B} | {r['baseline_median']:.2f}s | {r['ais_median']:.2f}s | "
                    f"{r['baseline_rps']:.2f} | {r['ais_rps']:.2f} | "
                    f"{r['saving_pct']:+.1f}% | {r['speedup_x']:.2f}x\n")
    print(f"  Saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
