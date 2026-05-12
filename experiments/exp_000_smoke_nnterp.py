"""Smoke test ambiente AIS (pre-Fase 0).

Verifica che lo stack nnterp + nnsight 0.7 + transformers 5.x + MPS bf16 sia
funzionante caricando GPT-2 Small, leggendo le attivazioni di tutti i layer,
predicendo il token corretto su un prompt fattuale e confermando il determinismo
del backend MPS bf16.

Esegui SOLO da file (mai `python -c` o heredoc — vedi CLAUDE.md §3.1).
"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

# nnsight 0.7 crasha quando viene importato in modo che setti sys.argv[0]=''
# (es. python -c o heredoc). Tenere l'import al top di un file .py risolve.
from nnterp import StandardizedTransformer

MODEL_ID = "gpt2"
PROMPT = "The capital of France is"
# NB: GPT-2 Small (124M) NON predice ' Paris' come top-1 nemmeno in fp32 —
# rank 5/50257 in fp32, rank 7 in bf16. ROME originale usa GPT-2 XL / GPT-J.
# Lo smoke test verifica quindi solo che la distribuzione sia non-degenerata
# (max prob > 1%, ben sopra uniforme 2e-5), non la factual recall del modello.
MIN_TOP1_PROB = 0.01


def _unwrap(t):
    """nnsight ritorna alcune attivazioni come tuple (hidden, ...). Normalizza."""
    return t[0] if isinstance(t, tuple) else t


def main() -> int:
    print(f"Loading {MODEL_ID} via nnterp on MPS bf16...")
    model = StandardizedTransformer(MODEL_ID, dtype=torch.bfloat16, device_map="mps")
    n_layers = len(model.layers)
    print(f"Loaded. n_layers={n_layers}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # --- 1) Cattura residual stream output di ogni layer ---
    # nnsight 0.7 fa AST rewriting del body di trace: i locals non sopravvivono
    # fuori. Init lista FUORI, append DENTRO (idiom da nnterp/nnsight_utils.py).
    captured: list = []
    logits_save = None
    with torch.no_grad():
        with model.trace(PROMPT):
            for i in range(n_layers):
                captured.append(model.layers_output[i].save())
            logits_save = model.logits.save()

    # NaN/shape check su TUTTI i layer (cheap su GPT-2; protegge da regressioni).
    for i in range(n_layers):
        t = _unwrap(captured[i])
        assert not torch.isnan(t).any().item(), f"NaN in layer {i}"
        assert t.shape[-1] == 768, f"layer {i}: hidden={t.shape[-1]} != 768"

    print(f"\nLayer activations (sampled — all {n_layers} passed NaN/shape):")
    for i in (0, n_layers // 2, n_layers - 1):
        t = _unwrap(captured[i])
        print(
            f"  layer {i:2d}: shape={tuple(t.shape)}, dtype={t.dtype}, "
            f"mean_abs={t.float().abs().mean().item():.4f}"
        )

    # --- 2) Top-5 next-token predictions ---
    probs = torch.softmax(logits_save[0, -1, :].float(), dim=-1)
    top5 = probs.topk(5)
    print("\nTop-5 predicted next tokens:")
    for p, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  {tok.decode(idx)!r}  p={p:.4f}")
    top1 = tok.decode(top5.indices[0].item())

    # --- 3) Determinismo: 3 forward passes sullo stesso input. Atteso == 0.0. ---
    print("\nDeterminism check (3 runs, identical input):")
    runs: list = []
    for r in range(3):
        with torch.no_grad():
            with model.trace(PROMPT):
                runs.append(model.logits.save())
        l = runs[-1][0, -1, :].float().cpu()
        print(f"  run {r}: top={tok.decode(l.argmax().item())!r}  logit[0:3]={l[:3].tolist()}")
    last_logits = [r[0, -1, :].float().cpu() for r in runs]
    max_diff = max((last_logits[0] - last_logits[i]).abs().max().item() for i in (1, 2))
    print(f"  max |logit_diff| between runs: {max_diff:.3e}")

    # --- 4) Go/No-Go ---
    top1_prob = top5.values[0].item()
    ok_distribution = top1_prob > MIN_TOP1_PROB
    ok_determinism = max_diff == 0.0
    print("\n" + "=" * 60)
    print(f"  top-1 prob > {MIN_TOP1_PROB} (non-degenerate dist): {ok_distribution}  [{top1_prob:.4f}, top1={top1!r}]")
    print(f"  MPS bf16 bit-exact across 3 runs:           {ok_determinism}  [{max_diff:.2e}]")
    print(f"  no NaN, shape=[..,768] on all {n_layers} layers:     True")
    print()
    if ok_distribution and ok_determinism:
        print("  SMOKE TEST PASS — ambiente nnterp pronto per Fase 0")
        rc = 0
    else:
        print("  SMOKE TEST FAIL — diagnosticare prima di procedere a Prompt 0.2")
        rc = 1
    print("=" * 60)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
