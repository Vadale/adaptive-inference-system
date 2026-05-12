"""exp_009c — benchmark PERSISTENT skip (apply once + N forwards).

Misura il saving REALE eliminando l'overhead di re-swap. Pattern realistico
per deploy AIS: la categoria è stabile per N prompt consecutivi → skip plan
applicato UNA volta + N forward senza swap.

Usage:
  python exp_009c_persistent_skip_bench.py --model-id google/gemma-4-E4B-it
  python exp_009c_persistent_skip_bench.py --model-id google/gemma-4-E2B-it
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skippers.native_skip import NativeLayerSkipper

PROMPT = "The capital of France is"
N_REPEATS = 5


def _stats(ts):
    m = statistics.mean(ts)
    s = statistics.stdev(ts) if len(ts) > 1 else 0.0
    return m, s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="google/gemma-4-E4B-it")
    ap.add_argument("--skip-fraction", type=float, default=0.33,
                    help="frazione di layer da hard-skip (0.33 = 33%%)")
    args = ap.parse_args()

    print(f"Loading NativeLayerSkipper for {args.model_id}...", flush=True)
    # Per E2B (5B params, sta tutto in MPS) → device_map fisso. Per E4B
    # (8B, non sta tutto) → auto + max_memory.
    if "E2B" in args.model_id:
        device_map = {
            "model.vision_tower": "cpu", "model.audio_tower": "cpu",
            "model.embed_vision": "cpu", "model.embed_audio": "cpu",
            "model.language_model": "mps", "lm_head": "mps",
        }
        max_memory = None
    else:
        device_map = "auto"
        max_memory = {"mps": "8GiB", "cpu": "30GiB"}
    native = NativeLayerSkipper(model_id=args.model_id,
                                  device_map=device_map, max_memory=max_memory)
    n = native.n_layers
    n_skip = int(round(n * args.skip_fraction))
    # Skip i 2 gruppi mid: layer (n//3) e (2n//3) → spread per stress test
    skip_layers = set(range(n // 6, n // 6 + n_skip // 2)) | set(
        range(5 * n // 6 - n_skip // 2, 5 * n // 6)
    )
    # Caso edge: assicura esatto n_skip
    while len(skip_layers) < n_skip:
        skip_layers.add(max(skip_layers) + 1)
    skip_layers = set(sorted(skip_layers)[:n_skip])
    print(f"  n_layers={n}  skip={len(skip_layers)} layer = {len(skip_layers)/n*100:.1f}%",
          flush=True)

    print("\n[warmup] 2 forward scartati...", flush=True)
    for r in range(2):
        t0 = time.time()
        _ = native.forward_no_swap(PROMPT)
        print(f"    warmup {r+1}/2: {time.time()-t0:.1f}s", flush=True)

    print(f"\n[1] baseline (no skip), {N_REPEATS} run...", flush=True)
    t_base = []
    for r in range(N_REPEATS):
        t0 = time.time()
        _ = native.forward_no_swap(PROMPT)
        dt = time.time() - t0
        t_base.append(dt)
        print(f"    base {r+1}/{N_REPEATS}: {dt:.1f}s", flush=True)
    m_base, s_base = _stats(t_base)

    print(f"\n[2] apply_skip(hard={len(skip_layers)} layer) UNA VOLTA + {N_REPEATS} forward...",
          flush=True)
    native.apply_skip(hard_skip=skip_layers)
    t_skip = []
    last_skip_logits = None
    for r in range(N_REPEATS):
        t0 = time.time()
        last_skip_logits = native.forward_no_swap(PROMPT)
        dt = time.time() - t0
        t_skip.append(dt)
        print(f"    skip {r+1}/{N_REPEATS}: {dt:.1f}s", flush=True)
    m_skip, s_skip = _stats(t_skip)

    print(f"\n[3] restore() + {N_REPEATS} baseline finale...", flush=True)
    native.restore()
    t_base2 = []
    last_base2_logits = None
    for r in range(N_REPEATS):
        t0 = time.time()
        last_base2_logits = native.forward_no_swap(PROMPT)
        dt = time.time() - t0
        t_base2.append(dt)
        print(f"    base2 {r+1}/{N_REPEATS}: {dt:.1f}s", flush=True)
    m_base2, s_base2 = _stats(t_base2)

    saving = (1 - m_skip / m_base) * 100
    drift = (m_base2 / m_base - 1) * 100

    print("\n" + "=" * 70, flush=True)
    print(f"  baseline init:       {m_base:5.1f} ± {s_base:.1f}s", flush=True)
    print(f"  hard_skip 33% (persistent): {m_skip:5.1f} ± {s_skip:.1f}s  "
          f"saving={saving:+.1f}%", flush=True)
    print(f"  baseline finale:     {m_base2:5.1f} ± {s_base2:.1f}s  drift={drift:+.1f}%",
          flush=True)
    print("=" * 70, flush=True)

    # Save
    out = ROOT / "results" / "native_persistent_bench.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        f"baseline init: {m_base:.1f} ± {s_base:.1f}s\n"
        f"hard_skip 33% persistent: {m_skip:.1f} ± {s_skip:.1f}s  saving={saving:+.1f}%\n"
        f"baseline finale: {m_base2:.1f} ± {s_base2:.1f}s  drift={drift:+.1f}%\n"
    )
    print(f"  Saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
