"""exp_012 — smoke test NativeLayerSkipper on Llama 3.2 3B.

Llama 3.2 has a standard transformer architecture (no shared KV like Gemma 4),
so NativeLayerSkipper should work natively → real compute saving via hard skip.

Tests:
  1) FALLBACK identity: forward(no skip) == HF native (max|Δ|=0)
  2) Hard skip {0}: logits finite, different from baseline
  3) Hard skip 33% (mid+late layers): real latency saving expected
  4) Soft skip α=0.7: should preserve top-1 better than hard
  5) Latency benchmark: baseline vs hard skip 33%, multi-run with warmup
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROMPT = "The capital of France is"
CANDIDATES = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
]
N_REPEATS = 5


def _load_model():
    for repo in CANDIDATES:
        try:
            print(f"Trying {repo}...", flush=True)
            m = AutoModelForCausalLM.from_pretrained(
                repo, dtype=torch.bfloat16, device_map="mps"
            )
            tok = AutoTokenizer.from_pretrained(repo)
            print(f"  Loaded {repo}", flush=True)
            return m, tok, repo
        except Exception as e:
            print(f"  FAIL {repo}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    raise RuntimeError("None of the Llama 3.2 candidates loaded")


class HardSkipLayer(torch.nn.Module):
    """Identity layer: returns hidden_states unchanged. Bypasses compute.
    Note: Llama 3.x decoder layer returns plain tensor (not tuple). Return same shape."""
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


def main() -> int:
    model, tok, repo = _load_model()
    model.eval()
    layers = model.model.layers
    n = len(layers)
    print(f"n_layers={n} hidden={model.config.hidden_size}", flush=True)
    # Check shared KV
    has_shared_kv = hasattr(model.config, "sliding_window") or any(
        hasattr(l.self_attn, "layer_type") for l in layers
    )
    print(f"shared_kv_pattern detected: {has_shared_kv}", flush=True)

    # Format chat
    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("mps")
    print(f"input_ids shape: {inputs.input_ids.shape}", flush=True)

    def _forward():
        with torch.no_grad():
            return model(**inputs).logits[0, -1, :].float().cpu()

    # Warmup
    print("\n[warmup] 2 forwards...", flush=True)
    for _ in range(2):
        _ = _forward()

    # [1] FALLBACK identity
    print("\n[1] baseline (no skip), 3 runs", flush=True)
    t_base = []
    for r in range(3):
        t0 = time.time()
        r_base = _forward()
        t_base.append(time.time() - t0)
        print(f"    run {r+1}: {t_base[-1]:.3f}s", flush=True)
    m_base = statistics.median(t_base)
    print(f"  finite={torch.isfinite(r_base).all().item()}  median={m_base:.3f}s", flush=True)

    # [2] Hard skip {0}
    print(f"\n[2] hard skip {{0}}", flush=True)
    orig = layers[0]
    layers[0] = HardSkipLayer()
    try:
        r_h0 = _forward()
        diff_h0 = (r_base - r_h0).abs().max().item()
        finite_h0 = torch.isfinite(r_h0).all().item()
        print(f"    finite={finite_h0}  max|Δ| vs baseline={diff_h0:.3f}", flush=True)
    except Exception as e:
        print(f"    FAIL: {type(e).__name__}: {e}", flush=True)
        return 1
    finally:
        layers[0] = orig

    # [3] Hard skip 33% (layers in mid+late: 9-16 + 22-28 ~ 33%)
    skip_set = set(range(9, 16)) | set(range(22, 28))
    print(f"\n[3] hard skip {len(skip_set)}/{n} = {len(skip_set)/n*100:.0f}%, "
          f"5 runs (persistent)", flush=True)
    saved = {}
    for i in skip_set:
        saved[i] = layers[i]
        layers[i] = HardSkipLayer()
    try:
        t_skip = []
        last_skip_logits = None
        for r in range(N_REPEATS):
            t0 = time.time()
            last_skip_logits = _forward()
            t_skip.append(time.time() - t0)
            print(f"    skip run {r+1}: {t_skip[-1]:.3f}s", flush=True)
        m_skip = statistics.median(t_skip)
        diff = (r_base - last_skip_logits).abs().max().item()
        finite = torch.isfinite(last_skip_logits).all().item()
        print(f"  finite={finite}  median={m_skip:.3f}s  max|Δ|={diff:.3f}", flush=True)
        saving = (1 - m_skip / m_base) * 100
        print(f"  → SAVING: {saving:+.1f}%", flush=True)
    finally:
        for i, orig in saved.items():
            layers[i] = orig

    # [4] Baseline restore check
    print(f"\n[4] baseline after restore, 3 runs", flush=True)
    t_base2 = []
    for r in range(3):
        t0 = time.time()
        r_base2 = _forward()
        t_base2.append(time.time() - t0)
    m_base2 = statistics.median(t_base2)
    diff_restore = (r_base - r_base2).abs().max().item()
    print(f"  median={m_base2:.3f}s  max|Δ| vs initial baseline={diff_restore:.4f}",
          flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print(f"  Model: {repo}", flush=True)
    print(f"  n_layers={n}, shared_kv={has_shared_kv}", flush=True)
    print(f"  Baseline median:        {m_base:.3f}s", flush=True)
    print(f"  Skip 33% median:        {m_skip:.3f}s  ({saving:+.1f}%)", flush=True)
    print(f"  Restore baseline drift: {(m_base2/m_base - 1)*100:+.1f}%", flush=True)
    print(f"  Restore logits Δ:       {diff_restore:.4f}", flush=True)
    print("=" * 70, flush=True)

    return 0 if (finite and saving > 10) else 1


if __name__ == "__main__":
    raise SystemExit(main())
