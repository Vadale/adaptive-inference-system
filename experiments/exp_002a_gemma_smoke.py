"""Smoke test Gemma 4 E2B-it via nnsight VisionLanguageModel su MPS bf16.

Gemma 4 E2B-it è un modello **multimodale** (text+image+audio+video) registrato
come `AutoModelForImageTextToText`. nnterp `StandardizedTransformer` NON lo
accetta. Si usa `nnsight.VisionLanguageModel` con due workaround:

  1) `device_map` split: vision/audio tower su CPU (non li usiamo), language_model
     + lm_head su MPS. Riduce il warmup buffer MPS sotto il limit single-buffer
     (~7-8 GB su Mac Mini 16 GB).
  2) `os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]="0.0"` PRIMA di import torch:
     disabilita il soft cap MPS sull'allocazione totale unified memory.

Accessor:
  - `model.model.language_model.layers[i]`   — 35 layer testuali (1536 hidden)
  - `model.lm_head.output`                   — logits (1, seq, 262144)
  - `model.processor`                        — Gemma4Processor (multimodale)

Trace con stringa diretta funziona se eseguito da file .py (P1).

Vincoli: vedi `docs/conventions.md` + `docs/pitfalls.md` P1-P4, P9, P10, P11.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import gc
import torch
from transformers import AutoTokenizer
from nnsight import VisionLanguageModel

MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "The capital of France is"
MIN_TOP1_PROB = 0.01

# Senza questo split, MPS prova un caching_allocator_warmup buffer ~9.5 GiB e
# fallisce con "Invalid buffer size".
DEVICE_MAP = {
    "model.vision_tower": "cpu",
    "model.audio_tower": "cpu",
    "model.embed_vision": "cpu",
    "model.embed_audio": "cpu",
    "model.language_model": "mps",
    "lm_head": "mps",
}


def _unwrap(t):
    return t[0] if isinstance(t, tuple) else t


def main() -> int:
    print(f"Loading {MODEL_ID} (VLM, device_map split)...")
    model = VisionLanguageModel(MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE_MAP)
    n_layers = len(model._model.model.language_model.layers)
    print(f"Loaded. n_text_layers={n_layers}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    captured: list = []
    logits_save = None
    with torch.no_grad():
        with model.trace(PROMPT):
            for i in range(n_layers):
                captured.append(model.model.language_model.layers[i].output.save())
            logits_save = model.lm_head.output.save()

    hidden_dim = None
    for i in range(n_layers):
        t = _unwrap(captured[i])
        assert not torch.isnan(t).any().item(), f"NaN in layer {i}"
        if hidden_dim is None:
            hidden_dim = t.shape[-1]
        assert t.shape[-1] == hidden_dim, f"layer {i}: hidden mismatch"
    print(f"hidden_dim={hidden_dim}, all {n_layers} layers passed NaN/shape")

    for i in (0, n_layers // 2, n_layers - 1):
        t = _unwrap(captured[i])
        print(
            f"  layer {i:2d}: shape={tuple(t.shape)}, dtype={t.dtype}, "
            f"mean_abs={t.float().abs().mean().item():.4f}"
        )

    probs = torch.softmax(logits_save[0, -1, :].float(), dim=-1)
    top5 = probs.topk(5)
    print("\nTop-5 predicted next tokens (NB: -it model senza chat template):")
    for p, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  {tok.decode([idx])!r}  p={p:.4f}")
    top1 = tok.decode([top5.indices[0].item()])
    top1_prob = top5.values[0].item()

    print("\nDeterminism check (3 runs, identical input):")
    runs_last: list = []
    for r in range(3):
        with torch.no_grad():
            with model.trace(PROMPT):
                s = model.lm_head.output.save()
        l = s[0, -1, :].float().cpu()
        runs_last.append(l)
        print(f"  run {r}: top={tok.decode([l.argmax().item()])!r}  logit[0:3]={l[:3].tolist()}")
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    max_diff = max((runs_last[0] - runs_last[i]).abs().max().item() for i in (1, 2))
    print(f"  max |logit_diff| between runs: {max_diff:.3e}")

    ok_distribution = top1_prob > MIN_TOP1_PROB
    ok_determinism = max_diff == 0.0
    print("\n" + "=" * 60)
    print(f"  top-1 prob > {MIN_TOP1_PROB}: {ok_distribution}  [{top1_prob:.4f}, top1={top1!r}]")
    print(f"  MPS bf16 bit-exact 3 runs: {ok_determinism}  [{max_diff:.2e}]")
    print(f"  no NaN, hidden={hidden_dim}, all {n_layers} layers: True")
    if ok_distribution and ok_determinism:
        print("  SMOKE GEMMA E2B (VLM): PASS — pronto per exp_002 (collezione attivazioni)")
        rc = 0
    else:
        print("  SMOKE GEMMA E2B (VLM): FAIL — diagnosticare prima di Fase 1")
        rc = 1
    print("=" * 60)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
