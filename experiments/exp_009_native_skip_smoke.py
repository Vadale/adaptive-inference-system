"""exp_009 — smoke test NativeLayerSkipper (standalone, no Adaptive).

Verifica:
  1) forward baseline (no skip) → logits finiti, riferimento timing.
  2) hard_skip su g5 (L35-41, 7/42 = 17%) → logits diversi dal baseline,
     LATENCY ridotta (saving reale di compute attesa).
  3) soft_skip su g5 α=0.7 → logits più vicini al baseline, NO latency saving.
  4) Restore: forward baseline DOPO skip = baseline iniziale (idempotenza).
  5) Hard skip aggressivo (g1+g5, 14 layer = 33%) → latency saving ~33%.

Run con `caffeinate -i python -u`.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cervellone.native_skip import NativeLayerSkipper

PROMPT = "The capital of France is"
G5 = set(range(35, 42))           # 7 layer (17%)
G1_G5 = set(range(7, 14)) | G5    # 14 layer (33%)


def main() -> int:
    print("Loading NativeLayerSkipper (E4B, puro HF)...", flush=True)
    native = NativeLayerSkipper()
    print(f"  n_layers={native.n_layers}", flush=True)

    print("\n[1] baseline (no skip)...", flush=True)
    t0 = time.time()
    r_base = native.forward(PROMPT)
    t_base = time.time() - t0
    print(f"    {t_base:5.1f}s  finite={torch.isfinite(r_base).all().item()}  "
          f"max|x|={r_base.abs().max().item():.2f}", flush=True)

    print(f"\n[2] hard_skip g5 (7 layer = 17%)...", flush=True)
    t0 = time.time()
    r_g5_hard = native.forward(PROMPT, hard_skip=G5)
    t_g5_hard = time.time() - t0
    diff_g5 = (r_base - r_g5_hard).abs().max().item()
    save_g5 = (1 - t_g5_hard / t_base) * 100
    print(f"    {t_g5_hard:5.1f}s  saving={save_g5:+.1f}%  max|Δ|={diff_g5:.2f}  "
          f"finite={torch.isfinite(r_g5_hard).all().item()}", flush=True)

    print(f"\n[3] soft_skip g5 α=0.7...", flush=True)
    t0 = time.time()
    r_g5_soft = native.forward(PROMPT, soft_skip={i: 0.7 for i in G5})
    t_g5_soft = time.time() - t0
    diff_g5_soft = (r_base - r_g5_soft).abs().max().item()
    save_g5_soft = (1 - t_g5_soft / t_base) * 100
    print(f"    {t_g5_soft:5.1f}s  saving={save_g5_soft:+.1f}%  max|Δ|={diff_g5_soft:.2f}  "
          f"finite={torch.isfinite(r_g5_soft).all().item()}", flush=True)

    print(f"\n[4] hard_skip aggressivo g1+g5 (14 layer = 33%)...", flush=True)
    t0 = time.time()
    r_g15 = native.forward(PROMPT, hard_skip=G1_G5)
    t_g15 = time.time() - t0
    diff_g15 = (r_base - r_g15).abs().max().item()
    save_g15 = (1 - t_g15 / t_base) * 100
    print(f"    {t_g15:5.1f}s  saving={save_g15:+.1f}%  max|Δ|={diff_g15:.2f}  "
          f"finite={torch.isfinite(r_g15).all().item()}", flush=True)

    print(f"\n[5] baseline DOPO skip (verify restore)...", flush=True)
    t0 = time.time()
    r_base2 = native.forward(PROMPT)
    t_base2 = time.time() - t0
    diff_restore = (r_base - r_base2).abs().max().item()
    print(f"    {t_base2:5.1f}s  max|Δ| vs baseline iniziale={diff_restore:.4f}",
          flush=True)

    # Verdetto
    ok_finite = all(
        torch.isfinite(x).all().item()
        for x in (r_base, r_g5_hard, r_g5_soft, r_g15, r_base2)
    )
    ok_restore = diff_restore < 0.01
    ok_save_g5 = save_g5 > 5  # 17% layer skip → atteso ~10-17% saving
    ok_save_g15 = save_g15 > 15  # 33% skip → atteso ~25-33%
    ok_soft_no_save = abs(save_g5_soft) < 10  # soft non dovrebbe salvare
    ok_soft_better_quality = diff_g5_soft < diff_g5  # soft più vicino al baseline

    print("\n" + "=" * 60, flush=True)
    print(f"  Tutti finiti:                              {ok_finite}", flush=True)
    print(f"  Restore corretto (Δ<0.01):                {ok_restore}  [{diff_restore:.4f}]",
          flush=True)
    print(f"  Saving 17% skip > 5%:                     {ok_save_g5}  [{save_g5:+.1f}%]",
          flush=True)
    print(f"  Saving 33% skip > 15%:                    {ok_save_g15}  [{save_g15:+.1f}%]",
          flush=True)
    print(f"  Soft skip no-saving (|saving|<10%):       {ok_soft_no_save}  [{save_g5_soft:+.1f}%]",
          flush=True)
    print(f"  Soft preserva quality > hard:              {ok_soft_better_quality}  "
          f"[soft Δ={diff_g5_soft:.2f} vs hard Δ={diff_g5:.2f}]", flush=True)
    rc = 0 if (ok_finite and ok_restore and ok_save_g5) else 1
    print(f"  NativeLayerSkipper smoke: {'PASS' if rc == 0 else 'FAIL'}", flush=True)
    print("=" * 60, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
