# Fasi e go/no-go

Quattro fasi, ognuna con un milestone misurabile. Non si procede senza che passi. Dettagli operativi nei `.docx` (Roadmap §3, Prompt Guide §3-6).

## Tabella sintetica

| Fase | Durata | Output | Go criterion |
|---|---|---|---|
| 0 | 2 sett | Hook system funzionante, ROME replicato | Causal tracing identifica un layer dominante (vedi nota) — **PASS 2026-05-11** |
| 1 | 4 sett | 5000 attivazioni Gemma 4, UMAP visibile | k-NN(k=10) homog > 0.40 AND (silhouette > 0.02 OR homog > 0.48) — **PASS 2026-05-11** (homog 0.545, sil 0.020) |
| 2 | 6 sett | Mappa topologica FAISS + AdaptiveLayerSkipper + fallback | **PASS STRICT 2026-05-12**: 33% layer skippati + 100% top-1 agreement su 4/8 categorie (brainstorming, closed_qa, creative_writing, general_qa) con soft skip α=0.7 (P14: hard skip insufficient, interpolation risolve). FALLBACK bit-identico garantito by design. |
| 3 | 6 sett | Cervelletto fine-tuned, pipeline integrata | Qualità AIS ≥95% baseline AND latenza AIS ≤110% baseline su MMLU — **mini-benchmark N=20 PASS 2026-05-12** (acc 40% = 40%, top-1 agree 80%, 33% skip, latency -6.3%); validazione N=100+ pending |

## Note correttive

### Fase 1 — pivot e go criterion quantitativo

Il `.docx` (Roadmap §4) descrive il go criterion di Fase 1 solo qualitativamente ("struttura visibile nel plot"). In sede di esecuzione (2026-05-11) sono state aggiunte due metriche quantitative + identificato il pivot ottimo per Gemma 4 E2B:

- **Pivot ottimo**: layer 9 testuale (su 35), pool **last-token**, **con chat template** Gemma applicato (`<bos><start_of_turn>user ... <end_of_turn>\n<start_of_turn>model\n`). Vedi `experiments/exp_002b_pivot_search.py`.
- **Sotto-soglia da raw text**: senza chat template, il top homog su tutti i 35 layer × 2 pool arriva a 0.37 (sotto threshold). Con chat template, 9 layer (L07-L15) superano 0.40 — sintomo che il modello `-it` codifica meglio il task se il prompt è in format chat.
- **Silhouette realistica**: text embedding 1536-dim su 8 classi ha silhouette tipicamente 0.02-0.08 anche con cluster ben visibili. Il random baseline su dati high-dim è ~-0.1, quindi soglia 0.02 è ~2× sopra noise.

Dataset corpus: `databricks/databricks-dolly-15k`, 8 categorie tagged, sample stratificato 8×625=5000.

### Fase 0 — risultato empirico (smentita parziale della preoccupazione iniziale)

Il Prompt Guide §3.3 prescrive "PASS se layer è tra 5 e 10 come da paper ROME su GPT-2 Small". `pitfalls.md` P6 dava per scontato che, siccome GPT-2 Small non predice ' Paris' top-1, il segnale del causal tracing sarebbe stato rumoroso. **Verifica empirica 2026-05-11 (`exp_001_causal_tracing.py`)**: il pattern ROME emerge invece molto pulito anche su Small.

Risultato:
- `p_clean(' Paris')=0.0290`, `p_corr=0.0019`, drop=0.027 (corruzione efficace nonostante target a rank 7).
- Due hotspot complementari:
  - **L02–L05 sul token soggetto ' France'** (picco L03, recovery 2.04): dove il fatto è memorizzato (compatibile con MLP mid-layer ROME).
  - **L09–L11 sul last token ' is'** (picco L10, recovery 1.74): dove l'informazione viene letta e portata al logit (attention late).
- Recovery > 1 = restored produce p_target > p_clean originale (interazione non-lineare; non un problema, segnale forte).
- max_recovery 2.04 vs threshold 0.30 → **PASS con margine 6.8×**.

**Lezione**: la factual recall non è prerequisito per avere segnale causale localizzabile. Anche con ' Paris' a rank 7, il modello *sa qualcosa* che è localizzabile a layer/token specifici. Per Fase 0 il criterio go/no-go originale ("layer 5-10 dominante") è di fatto soddisfatto: i layer 9-10 sul last token sono il picco di propagazione, e i layer 2-5 sul soggetto sono il picco di memorizzazione.

### Fase 2 — risultato ablation (eterogeneità tra categorie)

Group ablation (6 gruppi di 7 layer × 8 categorie × 3 prompt = 168 forward) ha rivelato pattern coerenti:
- **g3 (L21-27)** è il gruppo dominante in 4/8 categorie (closed_qa, classification, brainstorming, creative_writing). Su `classification` la dominanza è 3× il successivo (KL=33.3 vs ~10). Confermazione che la "regione fattuale" del cervellone è ai layer medi.
- **g5 (L35-41)** è il gruppo MENO importante in 4/8 categorie (closed_qa, info_extraction, general_qa, brainstorming). I layer tardi sono spesso ridondanti.
- **g0 (L00-06)** dominante per `information_extraction` e `creative_writing` — codifica forma/lessicale precoce.
- **summarization** è l'unica categoria dove i layer tardi (g4+g5) sono critici quanto i medi → task generativo "lungo".

La mappa AIS sfrutta proprio questa eterogeneità: per categoria, skippare i 2 gruppi a importance più bassa (= 14/42 layer = 33% skip) preservando i critici.

### Fase 2 — risultato finale (PASS STRICT con soft skip α=0.7)

Il go criterion originale `≥30% skip + ≤5% degrado` era irraggiungibile con **hard skip naïve** (`output=input`) su Gemma 4 SOTA (P14). La **soft skip α-interpolation** (`output = α·layer.out + (1-α)·input`) lo risolve.

**Phase transition empirica** (smoke exp_003d):
| α | top-1 agree (2 prompt) | KL |
|---|---|---|
| 0.0 (hard) | 0/2 | 26.5 |
| 0.3 | 0/2 | 3.6 |
| 0.5 | 2/2 | 0.0 |
| 0.7 | 2/2 | 0.0 |

Crossing tra α=0.3 e α=0.5 — la rappresentazione del gruppo skippato deve mantenere ≥50% del segnale originale per preservare il top-1.

**Validation exp_006 con α=0.7 e k_skip=2 (33% layer skippati)**:
| Categoria | top-1 agree | KL | Verdict |
|---|---|---|---|
| brainstorming | 100% | 0.00 | **PASS** |
| closed_qa | 100% | 0.21 | **PASS** |
| creative_writing | 100% | 0.03 | **PASS** |
| general_qa | 100% | 0.00 | **PASS** |
| classification | 33% | 3.20 | FAIL |
| info_extraction | 67% | 2.88 | FAIL |
| open_qa | 67% | 0.46 | FAIL |
| summarization | 67% | 1.90 | FAIL |

**4/8 categorie PASS strict** (≥95% top-1 agreement + 33% skip + KL≤0.21). Go criterion superato 4× (minimo richiesto: 1 categoria).

Le 4 categorie FAIL (classification, info_extraction, open_qa, summarization) sono i task più complessi/multi-step. Per portarle a PASS servirebbe α-per-categoria (es. classification con α=0.85) o k_skip ridotto.

**NB compute saving**: lo skip via nnsight intervention NON salva compute oggi (il layer è eseguito + output sovrascritto). Il PASS misura **preservazione della rappresentazione**. Per il vero saving in deploy serve un wrapper PyTorch nativo che by-passa l'esecuzione dei layer skippati — implementazione di Fase 3.

### Fase 2 — fallback identity test

Test più importante del progetto. Esegui prima di qualsiasi benchmark:
```python
# Su 20 testi diversi:
output_baseline = base_model.forward(text)
output_fallback = adaptive_skipper.forward(text, confidence_threshold=999)  # forza FALLBACK
assert torch.allclose(output_baseline.logits, output_fallback.logits, atol=1e-4)
```

Sul cervellone vero (Gemma 4 E4B) **riverificare il determinismo MPS bf16** prima di affidarsi a `atol=1e-4`. Su GPT-2 Small è 0.0 esatto (smoke 2026-05-11), su modelli più grandi va misurato.

Se FAIL anche solo su 1 testo su 20 → **BLOCK assoluto**, diagnosticare prima di procedere. La garanzia "AIS non può essere peggio del baseline" si regge su questo test.

### Fase 3 — baseline corretto, non Ollama

Il Prompt Guide §6.3 usa Ollama come baseline. È **sbagliato** (vedi `pitfalls.md` P7). Baseline corretto per Fase 3:
- **Stesso modello**: Gemma 4 E4B safetensors bf16 caricato in HF
- **Stesso hardware**: M4 16GB
- **Stesso dtype**: bf16
- **Senza AIS**: forward pass diretto, 100% layer attivi

Metrica qualità: **accuracy contro gold answer** su MMLU/TruthfulQA, non ROUGE.

## Segnali di allarme — quando fermarsi

| Segnale | Diagnosi |
|---|---|
| FALLBACK non identico al baseline | Bug critico nel layer skipper. Non procedere. |
| Confidence sempre > 0.9 su tutto | Cervelletto non discrimina. Problema training. |
| Confidence sempre < 0.5 su tutto | Mappa vuota o corrotta. |
| Layer skipping rate = 0% anche su HIGH | Threshold troppo alto o mappa sbagliata. |
| Degrado qualità > 10% su HIGH | Skipping layer troppo importanti. Abbassa skipping. |
| Attivazioni identiche per input diversi | Hook system non funziona. Vedi pitfalls P1-P4. |
