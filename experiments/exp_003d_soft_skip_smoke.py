"""exp_003d — smoke test SOFT SKIP α-interpolation.

Confronto su 2 prompt: baseline vs hard skip (α=0) vs soft skip α=0.5, α=0.7
del gruppo g3 (L21-27). Ipotesi: α=0.5 cambia meno il top-1 di α=0 e mantiene
quality più vicino al baseline.

Output: per ogni α, max|Δ|, top-1 agreement, top-5 overlap, KL divergence.
Se PASS (top-1 agreement con α=0.5 > α=0 su almeno 1 prompt), step 1 ha senso.
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

from skippers.layer_skipper import AdaptiveLayerSkipper

PROMPTS = [
    "Based on the following passage, what year was the model trained?",
    "What is the capital of France?",
]
SKIP_GROUP = set(range(21, 28))  # g3 (L21-27)
ALPHAS = [0.0, 0.3, 0.5, 0.7]


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p_log = torch.log_softmax(p_logits.float(), dim=-1)
    q_log = torch.log_softmax(q_logits.float(), dim=-1)
    return float((p_log.exp() * (p_log - q_log)).sum().item())


def main() -> int:
    print("Loading AdaptiveLayerSkipper (E4B)...", flush=True)
    skipper = AdaptiveLayerSkipper()
    n = skipper.n_layers
    print(f"  n_layers={n}  skip_group={sorted(SKIP_GROUP)}", flush=True)
    active = set(range(n)) - SKIP_GROUP

    for p_idx, prompt in enumerate(PROMPTS):
        print(f"\n=== Prompt {p_idx + 1}/{len(PROMPTS)}: {prompt!r} ===", flush=True)

        t0 = time.time()
        r_base = skipper.forward(prompt, active_layers=None)
        base = r_base.logits_last
        t_base = time.time() - t0
        t1_base = int(base.argmax().item())
        t5_base = set(base.topk(5).indices.tolist())
        print(f"  baseline:    {t_base:5.1f}s  top1={t1_base}  finite={torch.isfinite(base).all().item()}",
              flush=True)

        for alpha in ALPHAS:
            t0 = time.time()
            r = skipper.forward(prompt, active_layers=active, alpha=alpha)
            t_a = time.time() - t0
            l = r.logits_last
            t1 = int(l.argmax().item())
            t5 = set(l.topk(5).indices.tolist())
            t5_overlap = len(t5 & t5_base) / 5
            diff = (base - l).abs().max().item()
            kl = _kl(base, l)
            print(f"  α={alpha:.1f}:      {t_a:5.1f}s  top1={t1}  "
                  f"agree={t1==t1_base}  top5_overlap={t5_overlap:.2f}  "
                  f"max|Δ|={diff:.2f}  KL={kl:.3f}  "
                  f"finite={torch.isfinite(l).all().item()}", flush=True)

    print("\n[End of soft-skip sweep — guarda se top-1 agree o KL migliora a α>0]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
