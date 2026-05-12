"""exp_009b — benchmark latency NativeLayerSkipper (con warmup + multi-run).

Misura il saving REALE di compute via skip layer nativo. Protocollo:
  - 2 warmup forward (scartati, scaldano cache MPS/accelerate)
  - 3 baseline forward (mediati)
  - 3 hard_skip 33% forward (mediati)
  - 3 hard_skip 17% forward (mediati)
  - 3 soft_skip 33% α=0.7 forward (mediati, no saving atteso)
  - 3 baseline forward end (mediati, verify no drift)

Output: tempo medio + std dev per ogni configurazione + saving % vs baseline.

Tempo totale: 17 forward × ~50s = ~15 min.
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

from cervellone.native_skip import NativeLayerSkipper

PROMPT = "The capital of France is"
G5 = set(range(35, 42))            # 7 layer = 17%
G1_G5 = set(range(7, 14)) | G5     # 14 layer = 33%
N_REPEATS = 3


def _time_forward(native: NativeLayerSkipper, label: str,
                  hard=None, soft=None) -> list[float]:
    times = []
    for r in range(N_REPEATS):
        t0 = time.time()
        _ = native.forward(PROMPT, hard_skip=hard, soft_skip=soft)
        dt = time.time() - t0
        times.append(dt)
        print(f"    [{label}] run {r+1}/{N_REPEATS}: {dt:.1f}s", flush=True)
    return times


def main() -> int:
    print("Loading NativeLayerSkipper...", flush=True)
    native = NativeLayerSkipper()
    print(f"  n_layers={native.n_layers}", flush=True)

    # WARMUP (scartati)
    print("\n[warmup] 2 forward scartati per scaldare cache...", flush=True)
    for r in range(2):
        t0 = time.time()
        _ = native.forward(PROMPT)
        print(f"    warmup {r+1}/2: {time.time()-t0:.1f}s", flush=True)

    print("\n[1] baseline (no skip), 3 run...", flush=True)
    t_base = _time_forward(native, "base")

    print(f"\n[2] hard_skip 33% (g1+g5), 3 run...", flush=True)
    t_h33 = _time_forward(native, "h33", hard=G1_G5)

    print(f"\n[3] hard_skip 17% (g5), 3 run...", flush=True)
    t_h17 = _time_forward(native, "h17", hard=G5)

    print(f"\n[4] soft_skip 33% α=0.7, 3 run...", flush=True)
    t_s33 = _time_forward(native, "s33", soft={i: 0.7 for i in G1_G5})

    print(f"\n[5] baseline finale (no skip), 3 run...", flush=True)
    t_base2 = _time_forward(native, "base2")

    # Aggregate
    def _stats(ts: list[float]) -> tuple[float, float]:
        return statistics.mean(ts), statistics.stdev(ts) if len(ts) > 1 else 0.0

    m_base, s_base = _stats(t_base)
    m_h33, s_h33 = _stats(t_h33)
    m_h17, s_h17 = _stats(t_h17)
    m_s33, s_s33 = _stats(t_s33)
    m_base2, s_base2 = _stats(t_base2)

    def _saving(m_cfg, m_ref):
        return (1 - m_cfg / m_ref) * 100

    print("\n" + "=" * 70, flush=True)
    print(f"  {'config':25s}  mean ± std    saving vs baseline", flush=True)
    print("-" * 70, flush=True)
    print(f"  {'baseline init':25s}  {m_base:5.1f} ± {s_base:.1f}s   --", flush=True)
    print(f"  {'hard_skip 33%':25s}  {m_h33:5.1f} ± {s_h33:.1f}s   "
          f"{_saving(m_h33, m_base):+5.1f}%", flush=True)
    print(f"  {'hard_skip 17%':25s}  {m_h17:5.1f} ± {s_h17:.1f}s   "
          f"{_saving(m_h17, m_base):+5.1f}%", flush=True)
    print(f"  {'soft_skip 33% α=0.7':25s}  {m_s33:5.1f} ± {s_s33:.1f}s   "
          f"{_saving(m_s33, m_base):+5.1f}%", flush=True)
    print(f"  {'baseline finale':25s}  {m_base2:5.1f} ± {s_base2:.1f}s   "
          f"{_saving(m_base2, m_base):+5.1f}%", flush=True)
    print("=" * 70, flush=True)

    # Salva risultati
    out = ROOT / "results" / "native_latency_bench.txt"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        f.write(f"baseline init:        {m_base:.1f} ± {s_base:.1f}s\n")
        f.write(f"hard_skip 33%:        {m_h33:.1f} ± {s_h33:.1f}s   saving={_saving(m_h33,m_base):+.1f}%\n")
        f.write(f"hard_skip 17%:        {m_h17:.1f} ± {s_h17:.1f}s   saving={_saving(m_h17,m_base):+.1f}%\n")
        f.write(f"soft_skip 33% α=0.7:  {m_s33:.1f} ± {s_s33:.1f}s   saving={_saving(m_s33,m_base):+.1f}%\n")
        f.write(f"baseline finale:      {m_base2:.1f} ± {s_base2:.1f}s   delta={_saving(m_base2,m_base):+.1f}%\n")
    print(f"  Saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
