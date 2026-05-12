"""Smoke test Gemma 4 E4B-it (decoder) via VisionLanguageModel su MPS bf16.

E4B è ~16 GB bf16 (8B totali params, 42 layer testuali). Su Mac Mini 16 GB
unified memory NON sta in MPS con `device_map={...:"mps", ...}` perché il
`caching_allocator_warmup` di transformers tenta una single-buffer allocation
di ~14 GiB > MPS single-buffer limit (~9 GiB).

Workaround: caricare TUTTO su CPU (`device_map="cpu"` → niente warmup MPS),
poi muovere a MPS manualmente solo `language_model` + `lm_head` dopo il load.

Vincoli: P1-P4, P9-P11, P12 (vedi `docs/pitfalls.md`). Pattern alternativo
rispetto a E2B (che invece passa col device_map split direttamente).
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import gc
import torch
from transformers import AutoTokenizer
from nnsight import VisionLanguageModel

MODEL_ID = "google/gemma-4-E4B-it"
PROMPT = "The capital of France is"
MIN_TOP1_PROB = 0.01


def _move_text_path_to_mps(model) -> None:
    """Sposta language_model + lm_head a MPS, lascia vision/audio su CPU.
    Da chiamare dopo `VisionLanguageModel(..., device_map="cpu")` per
    bypassare il caching_allocator_warmup MPS che fallisce su E4B."""
    inner = model._model
    print("  Spostando language_model su MPS...")
    inner.model.language_model = inner.model.language_model.to("mps")
    print("  Spostando lm_head su MPS...")
    inner.lm_head = inner.lm_head.to("mps")
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _unwrap(t):
    return t[0] if isinstance(t, tuple) else t


def main() -> int:
    print(f"Loading {MODEL_ID} (VLM, device_map=auto + max_memory MPS limit)...")
    # Limit esplicito su MPS sotto il single-buffer limit (~9 GB). Accelerate
    # auto-shard mette i moduli più piccoli/usati su MPS, il resto su CPU.
    model = VisionLanguageModel(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        max_memory={"mps": "8GiB", "cpu": "30GiB"},
    )
    n_layers = len(model._model.model.language_model.layers)
    print(f"Loaded. n_text_layers={n_layers}")
    # Stampo device per modulo per visibilità
    devmap = getattr(model._model, "hf_device_map", None)
    if devmap:
        print("  device_map effettivo:")
        for k, v in sorted(devmap.items()):
            print(f"    {k:50s} {v}")

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
    print(f"hidden_dim={hidden_dim}, all {n_layers} layers passed NaN/shape")

    for i in (0, n_layers // 2, n_layers - 1):
        t = _unwrap(captured[i])
        print(
            f"  layer {i:2d}: shape={tuple(t.shape)}, dtype={t.dtype}, "
            f"mean_abs={t.float().abs().mean().item():.4f}"
        )

    probs = torch.softmax(logits_save[0, -1, :].float(), dim=-1)
    top5 = probs.topk(5)
    print("\nTop-5 predicted next tokens (-it model senza chat template, atteso degenere):")
    for p, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  {tok.decode([idx])!r}  p={p:.4f}")
    top1 = tok.decode([top5.indices[0].item()])
    top1_prob = top5.values[0].item()

    print("\nDeterminism check (3 runs):")
    runs_last: list = []
    for r in range(3):
        with torch.no_grad():
            with model.trace(PROMPT):
                s = model.lm_head.output.save()
        l = s[0, -1, :].float().cpu()
        runs_last.append(l)
        print(f"  run {r}: top={tok.decode([l.argmax().item()])!r}")
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
        print("  SMOKE GEMMA E4B (VLM): PASS — decoder pronto per Fase 2")
        rc = 0
    else:
        print("  SMOKE GEMMA E4B (VLM): FAIL — diagnosticare prima di Fase 2")
        rc = 1
    print("=" * 60)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
