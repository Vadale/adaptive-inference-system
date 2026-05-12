"""Causal tracing stile ROME (Meng et al. 2022) su GPT-2 Small.

Misura quali (layer, token_position) sono causalmente critici per la
predizione di ' Paris' sul prompt 'The capital of France is'.

Pipeline:
  1) Clean run     → salva hidden di ogni layer su ogni token + p(' Paris')
  2) Corrupted run → noise N(0, (3·σ_emb)²) sul residual stream di ' France'
                     all'ingresso del blocco 0 (≡ wte+wpe per GPT-2; equivalente
                     allo step ROME a meno dei pos-embed) → p(' Paris') corr
  3) Restored runs → per ogni (ℓ, t): corrotti + ripristina layers_output[ℓ][:,t,:]
                     dal clean → p(' Paris') restored
  4) Recovery      → (p_restored − p_corrupted) / (p_clean − p_corrupted)

Vincoli: file .py, `dtype=`, `StandardizedTransformer`, init liste fuori dal
`with model.trace(...)`. Vedi `docs/conventions.md` e `docs/pitfalls.md`
P1-P4, P9 (meta-tensor init), P10 (`token_embeddings` shortcut).

Go/no-go Fase 0 (criterio rilassato — vedi docs/phases.md): PASS se almeno
un (layer, pos) ha recovery > 0.30 (≥30% del gap clean−corrupted recuperato).
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from transformers import AutoTokenizer
from nnterp import StandardizedTransformer

MODEL_ID = "gpt2"
PROMPT = "The capital of France is"
SUBJECT = " France"
TARGET = " Paris"
NOISE_SCALE = 3.0  # ν = 3·σ_emb come in ROME
SEED = 42
THRESHOLD = 0.30   # criterio go/no-go rilassato (P6: GPT-2 Small non sa il fatto)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CSV_PATH = RESULTS_DIR / "exp_001_causal_tracing.csv"
LOG_PATH = RESULTS_DIR / "exp_001_causal_tracing.txt"


def _unwrap(t):
    # I blocchi transformer HF ritornano spesso `(hidden_states, presents, ...)`;
    # in alcuni backbone (Gemma) un dataclass. Per nnterp `.save()` ricade nel
    # primo caso su GPT-2 — normalizziamo qui per chiarezza.
    return t[0] if isinstance(t, tuple) else t


def _embed_weight(model):
    # API canonica: l'HF model sottostante esposto da nnsight come `_model`.
    # Su GPT-2 + nnterp 1.3.0 è l'unica strada (P10: `model.token_embeddings`
    # è già il weight tensor, non l'Envoy modulo).
    return model._model.get_input_embeddings().weight


def main() -> int:
    torch.manual_seed(SEED)
    print(f"Loading {MODEL_ID} on MPS bf16...")
    model = StandardizedTransformer(MODEL_ID, dtype=torch.bfloat16, device_map="mps")
    n_layers = len(model.layers)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    input_ids = tok(PROMPT, return_tensors="pt").input_ids[0]
    tokens_str = [tok.decode([t]) for t in input_ids.tolist()]
    print("Tokens:")
    for i, s in enumerate(tokens_str):
        print(f"  [{i}] {s!r}  (id={input_ids[i].item()})")

    # Localizza il soggetto (può essere multi-token in BPE)
    subject_ids = tok(SUBJECT, add_special_tokens=False).input_ids
    subj_positions: list[int] = []
    for start in range(len(input_ids) - len(subject_ids) + 1):
        if input_ids[start:start + len(subject_ids)].tolist() == subject_ids:
            subj_positions = list(range(start, start + len(subject_ids)))
            break
    assert subj_positions, f"Soggetto {SUBJECT!r} non trovato nei token: {tokens_str}"
    print(f"Subject positions: {subj_positions}  "
          f"(tokens: {[tokens_str[i] for i in subj_positions]!r})")

    target_ids = tok.encode(TARGET, add_special_tokens=False)
    assert len(target_ids) == 1, (
        f"TARGET {TARGET!r} non è single-token su questo tokenizer: {target_ids}. "
        "Lo script assume target single-token per misurare p(TARGET) direttamente."
    )
    target_id = target_ids[0]
    print(f"Target token: {tok.decode([target_id])!r} (id={target_id})")

    # nnsight 0.7 inizializza i pesi su meta device: vanno materializzati dal
    # primo trace prima di poter leggere .weight fuori dal with-block.
    with torch.no_grad():
        with model.trace(PROMPT):
            _ = model.logits.save()

    # σ_emb dalla matrice embedding — base ROME del noise
    emb_w = _embed_weight(model)
    sigma_emb = emb_w.float().std().item()
    noise_sigma = NOISE_SCALE * sigma_emb
    hidden = emb_w.shape[1]
    print(f"σ_emb={sigma_emb:.4f}  →  noise σ = {NOISE_SCALE}·σ_emb = {noise_sigma:.4f}  (hidden={hidden})")

    # Noise precomputato FUORI dal trace: generator CPU dedicato così rimane
    # riproducibile anche se il global RNG viene consumato dai trace successivi
    # (transformers init + dispatch_model possono toccarlo in modo opaco).
    g = torch.Generator(device="cpu").manual_seed(SEED)
    noise_per_pos = (
        (torch.randn(len(subj_positions), hidden, generator=g) * noise_sigma)
        .to(torch.bfloat16).to("mps")
    )

    # --- 1) Clean run ---
    clean_layers_saves: list = []
    clean_logits_save = None
    with torch.no_grad():
        with model.trace(PROMPT):
            for i in range(n_layers):
                clean_layers_saves.append(model.layers_output[i].save())
            clean_logits_save = model.logits.save()

    clean_hidden = [_unwrap(c).detach() for c in clean_layers_saves]
    for i, h in enumerate(clean_hidden):
        assert not torch.isnan(h).any().item(), f"clean: NaN al layer {i}"

    p_clean = torch.softmax(clean_logits_save[0, -1, :].float(), dim=-1)[target_id].item()
    top1_clean = clean_logits_save[0, -1, :].float().argmax().item()
    print(f"\n[clean]      p({TARGET!r})={p_clean:.4f}   top-1={tok.decode([top1_clean])!r}")

    # --- 2) Corrupted run ---
    # Intervento sul residual in ingresso al blocco 0 (= embed(tok) + pos_embed
    # per GPT-2; equivalente al setup ROME a meno dei pos-embed). Su altri
    # backbone (Gemma, RoPE applicato dentro l'attention) la corrispondenza
    # con "embedding output puro" va riverificata.
    # `model.token_embeddings` non è Envoy (P10) → usiamo `layers_input[0]`.
    corrupted_logits_save = None
    with torch.no_grad():
        with model.trace(PROMPT):
            for k, t in enumerate(subj_positions):
                model.layers_input[0][:, t, :] = (
                    model.layers_input[0][:, t, :] + noise_per_pos[k]
                )
            corrupted_logits_save = model.logits.save()
    assert not torch.isnan(corrupted_logits_save).any().item(), "corrupted: NaN nei logits"
    p_corr = torch.softmax(corrupted_logits_save[0, -1, :].float(), dim=-1)[target_id].item()
    top1_corr = corrupted_logits_save[0, -1, :].float().argmax().item()
    print(f"[corrupted]  p({TARGET!r})={p_corr:.4f}   top-1={tok.decode([top1_corr])!r}")

    # Hardening: se la corruzione è troppo debole, dividere per drop≈0 gonfia
    # i recovery score e produce PASS spuri. Soglia minima 0.005 (≈17% di
    # p_clean=0.029 nel run di riferimento) è già il limite del rumore.
    drop = p_clean - p_corr
    MIN_DROP = 0.005
    if drop < MIN_DROP:
        print(f"  FAIL: corruption inefficace (drop={drop:.4e} < {MIN_DROP}). "
              f"Aumentare NOISE_SCALE e ri-eseguire.")
        return 2
    denom = drop

    # --- 3) Restored runs ---
    seq_len = input_ids.shape[0]
    n_traces = n_layers * seq_len
    print(f"\nRunning {n_layers}×{seq_len} = {n_traces} restored traces...")
    # fp32/CPU per le metriche aggregate: pochi byte, non vale spendere bf16/mps.
    grid = torch.zeros(n_layers, seq_len, dtype=torch.float32)
    for layer in range(n_layers):
        for pos in range(seq_len):
            with torch.no_grad():
                with model.trace(PROMPT):
                    for k, t in enumerate(subj_positions):
                        model.layers_input[0][:, t, :] = (
                            model.layers_input[0][:, t, :] + noise_per_pos[k]
                        )
                    # Restore single point dal clean run
                    model.layers_output[layer][:, pos, :] = clean_hidden[layer][:, pos, :]
                    logits_save = model.logits.save()
            assert not torch.isnan(logits_save).any().item(), (
                f"restored: NaN logits a layer={layer} pos={pos}"
            )
            grid[layer, pos] = torch.softmax(
                logits_save[0, -1, :].float(), dim=-1
            )[target_id].item()
        # Liberare cache MPS dopo ogni layer evita drift di memory in loop lunghi
        # (preempt: pattern verrà riusato in Fase 1 con Gemma 4 ≫ GPT-2 Small).
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    rec = (grid - p_corr) / denom

    # --- 4) Output: heatmap testuale, CSV, log ---
    print(f"\np_clean={p_clean:.4f}  p_corr={p_corr:.4f}  drop={drop:.4f}")
    print("\nRecovery score (rows=layer, cols=token pos):")
    print("       " + " ".join(f"{tokens_str[p][:6]:>7}" for p in range(seq_len)))
    for L in range(n_layers):
        row = " ".join(f"{rec[L, p].item():7.3f}" for p in range(seq_len))
        print(f"  L{L:02d} {row}")

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "token_pos", "token_str", "p_restored", "recovery_score"])
        for L in range(n_layers):
            for pos in range(seq_len):
                writer.writerow([L, pos, tokens_str[pos],
                                 f"{grid[L, pos].item():.6f}",
                                 f"{rec[L, pos].item():.6f}"])
    print(f"\nWrote {CSV_PATH}")

    flat = rec.flatten()
    topk = torch.topk(flat, 3)
    hotspots = []
    for v, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        L, pos = divmod(int(idx), seq_len)
        hotspots.append((L, pos, tokens_str[pos], v))
    print("\nTop-3 hotspot (layer, pos, token, recovery):")
    for L, pos, s, v in hotspots:
        print(f"  L{L:02d}  pos={pos} ({s!r})  recovery={v:.3f}")

    max_rec = topk.values[0].item()
    ok = max_rec > THRESHOLD
    print("\n" + "=" * 60)
    print(f"  max recovery: {max_rec:.3f}   threshold: {THRESHOLD}")
    print(f"  Fase 0 go criterion: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)

    with LOG_PATH.open("w") as f:
        f.write(f"prompt={PROMPT!r} subject={SUBJECT!r} target={TARGET!r}\n")
        f.write(f"σ_emb={sigma_emb:.4f} noise_σ={noise_sigma:.4f} seed={SEED}\n")
        f.write(f"p_clean={p_clean:.4f} p_corrupted={p_corr:.4f} drop={drop:.4f}\n")
        f.write(f"top1_clean={tok.decode([top1_clean])!r} top1_corrupted={tok.decode([top1_corr])!r}\n")
        f.write(f"max_recovery={max_rec:.3f} threshold={THRESHOLD} verdict={'PASS' if ok else 'FAIL'}\n")
        f.write(f"top3_hotspots={hotspots}\n")
    print(f"Wrote {LOG_PATH}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
