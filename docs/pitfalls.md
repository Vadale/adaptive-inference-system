# Pitfalls verificati

Lista crescente dei pitfall scoperti empiricamente nel progetto. Ognuno include sintomo, causa, fix, e data di scoperta. Aggiungi qui quando ne trovi uno nuovo.

## Stack tecnico

### P1 — nnsight heredoc bug
- **Sintomo**: `TypeError: exec() arg 1 must be string, bytes or code object` all'`import nnsight` o `from nnterp import ...`
- **Causa**: `nnsight/__init__.py:67` — `__INTERACTIVE__ = (sys.flags.interactive or not sys.argv[0])`. `python -c` e heredoc lasciano `sys.argv[0]=''`, attivano modalità interattiva spuria.
- **Fix**: eseguire da file `.py`. Mai `python -c` o heredoc.
- **Versione**: nnsight 0.7.0
- **Scoperta**: 2026-05-11

### P2 — scope del `with model.trace(...)`
- **Sintomo**: `NameError: name 'x' is not defined` accedendo dopo il `with` a una variabile assegnata dentro.
- **Causa**: nnsight 0.7.0 fa AST rewriting del body — i locals non vivono fuori.
- **Fix**: init lista/holder FUORI, append DENTRO. Vedi `conventions.md §4`.
- **Versione**: nnsight 0.7.0
- **Scoperta**: 2026-05-11

### P3 — `torch_dtype=` deprecated
- **Sintomo**: `[transformers] torch_dtype is deprecated! Use dtype instead!`
- **Causa**: API rename in transformers 5.x.
- **Fix**: sostituire `torch_dtype=` con `dtype=` in tutti i `from_pretrained()`.
- **Versione**: transformers 5.8.0
- **Scoperta**: 2026-05-11

### P4 — `NNsightModel` non esiste
- **Sintomo**: `ImportError: cannot import name 'NNsightModel' from 'nnterp'`
- **Causa**: Prompt Guide AIS §3.2 e §4.1 citano un'API che non esiste in nnterp 1.3.0.
- **Fix**: usare `StandardizedTransformer(model_id, ...)`. Vedi `conventions.md §3`.
- **Versione**: nnterp 1.3.0
- **Scoperta**: 2026-05-11

### P9 — nnsight 0.7 init meta, materializza al primo trace
- **Sintomo**: `RuntimeError: Tensor.item() cannot be called on meta tensors` chiamando `.std()`/`.item()` su `_model.get_input_embeddings().weight` o altri parametri letti fuori dal trace, immediatamente dopo `StandardizedTransformer(...)`.
- **Causa**: nnsight 0.7 alloca i pesi su `meta` device finché un primo `with model.trace(...)` non li materializza con `dispatch_model`. Prima di quel momento i parametri non hanno storage.
- **Fix**: eseguire un trace dummy (anche `with model.trace(prompt): _ = model.logits.save()`) PRIMA di accedere a `.weight` fuori dal `with`. Dopo, i parametri sono materializzati e gli accessor funzionano.
- **Versione**: nnsight 0.7.0 + nnterp 1.3.0
- **Scoperta**: 2026-05-11 (`experiments/exp_001_causal_tracing.py`)

### P10 — `model.token_embeddings` in nnterp ritorna il weight, non l'Envoy
- **Sintomo**: `AttributeError: 'Tensor' object has no attribute 'output'` su `model.token_embeddings.output` dentro un trace.
- **Causa**: in nnterp 1.3.0 `StandardizedTransformer.token_embeddings` è un alias che ritorna direttamente il `weight` tensor del modulo embed, non l'Envoy del modulo `nn.Embedding`. Pattern asimmetrico rispetto a `model.layers` (che è Envoy lista).
- **Fix**: per interventi sul residual stream "post-embedding" usare `model.layers_input[0]` (alias standardizzato), che è il tensor in ingresso al primo blocco transformer (equivalente per causal tracing). Per leggere la matrice di pesi usare `_model.get_input_embeddings().weight` (dopo P9).
- **Versione**: nnterp 1.3.0
- **Scoperta**: 2026-05-11 (`experiments/exp_001_causal_tracing.py`)

### P11 — Gemma 4 è multimodale, non text-only; richiede VLM + device_map split
- **Sintomo (1)**: `ValueError: 'google/gemma-4-E2B-it' (gemma4) is registered with AutoModelForImageTextToText — it's a multimodal model so LanguageModel(...) can't load it`. nnterp `StandardizedTransformer` non lo carica.
- **Sintomo (2)**: con `nnsight.VisionLanguageModel(..., device_map="mps")`: `RuntimeError: Invalid buffer size: 9.51 GiB` durante `caching_allocator_warmup`. MPS ha un limit single-buffer ~7-8 GB su Mac Mini 16GB, e Gemma 4 E2B-it ha 5.5B params totali (`vision_tower` 167M + `language_model` 4628M + `audio_tower` 305M + `lm_head` 402M + embedders).
- **Sintomo (3)**: con `device_map=...` ma `model.logits.save()`: `AttributeError: ... has no attribute logits`. `VisionLanguageModel` non espone `logits` come accessor diretto.
- **Causa**: i `-it` di Gemma 4 sono `AutoModelForImageTextToText` (any-to-any modality). Hanno una struttura nested con sub-tower vision/audio/text di cui per AIS serve **solo** quella testuale.
- **Fix**:
  1. Caricare con `nnsight.VisionLanguageModel` (richiede `torchvision` come dipendenza extra).
  2. `os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]="0.0"` PRIMA di `import torch`.
  3. `device_map` esplicito che mette su CPU le componenti non testuali:
     ```python
     device_map = {
         "model.vision_tower": "cpu", "model.audio_tower": "cpu",
         "model.embed_vision": "cpu", "model.embed_audio": "cpu",
         "model.language_model": "mps", "lm_head": "mps",
     }
     ```
  4. Accessor:
     - `model.model.language_model.layers[i].output` per i 35 layer testuali (hidden 1536)
     - `model.lm_head.output` invece di `model.logits` per i logits finali
- **Verificato**: `exp_002a_gemma_smoke.py` 2026-05-11 — n_text_layers=35, hidden=1536, determinismo bit-exact MPS bf16 su 3 run.
- **Nota tuning**: il modello `-it` senza chat template produce next-token degeneri (top-1 ripete last input token con p≈1). Per usi che richiedono comportamento generativo (Fase 3) servirà `processor.apply_chat_template(...)`. Per Fase 1 (collezione attivazioni di hidden state) non è bloccante.
- **Versione**: nnsight 0.7.0, nnterp 1.3.0, transformers 5.8.0, torchvision 0.26.0
- **Scoperta**: 2026-05-11

### P12 — Gemma 4 E4B non sta in MPS 16 GB con device_map split semplice
- **Sintomo**: `RuntimeError: Invalid buffer size: 13.90 GiB` durante `caching_allocator_warmup` anche col device_map split che funzionava per E2B (vision/audio su CPU, language_model+lm_head su MPS).
- **Causa**: E4B ha 8B params totali (42 layer testuali, hidden 2560) vs E2B 5.5B (35 layer, hidden 1536). Il warmup tenta un singolo buffer fp16 di ~14 GiB > MPS single-buffer limit (~9 GiB su Mac Mini 16 GB unified memory).
- **Fix che NON funziona**: caricare con `device_map="cpu"` e poi `language_model.to("mps")` — i pesi sono ancora meta dopo l'init nnsight (P9 esteso). Un trace dummy su CPU non risolve perché il move successivo lascia placeholder su MPS.
- **Fix che funziona**: `device_map="auto"` + `max_memory={"mps": "8GiB", "cpu": "30GiB"}`. Accelerate sharda automaticamente sotto il limit MPS.
  ```python
  model = VisionLanguageModel(
      "google/gemma-4-E4B-it", dtype=torch.bfloat16, device_map="auto",
      max_memory={"mps": "8GiB", "cpu": "30GiB"},
  )
  ```
- **Conseguenza prestazionale**: una frazione dei layer testuali finisce su CPU → l'inferenza diventa più lenta perché ogni layer step può richiedere data transfer MPS↔CPU. Misurare in `exp_003_*`.
- **Verificato**: `exp_003a_decoder_smoke.py` 2026-05-11 — PASS, 42 layer letti, hidden 2560, determinismo bit-exact.
- **Versione**: nnsight 0.7.0, transformers 5.8.0
- **Scoperta**: 2026-05-11

### P13 — `MissedProviderError` su catene lunghe di `layer.output = layer.input` (e sleep MPS)
- **Sintomo**: `nnsight.MissedProviderError: Execution complete but model.model.language_model.layers.N.input.i0 was not provided` durante un trace che assegna `layer.output = layer.input` su molti layer consecutivi. Caso reale: `exp_005` group ablation, skip 7 layer contigui (L28-34) → crash al sesto forward dopo che i primi 5 erano andati a buon fine.
- **Causa A — catene lunghe**: nnsight 0.7 può non riuscire a risolvere il grafo di dipendenze quando N+ `layer.output = layer.input` consecutivi creano una chain dove l'input di un layer dipende dal layer precedente skippato in modo non sequenziale.
- **Causa B — sleep/standby MPS (più probabile)**: su Mac mini headless, se il sistema entra in App Nap / sleep tra trace successive (es. monitor scollegato, energy settings default), il driver Metal può perdere stato dei tensor MPS. Al risveglio nnsight non trova i proxy che si aspettava → MissedProvider su un layer dove prima era OK.
- **Fix A (codice)**: usare **boundary intervention** invece di per-layer skip. Per ogni gruppo contiguo di layer da skippare `[gs, ge)`, 1 solo intervento: `layers[ge-1].output = layers[gs].input`. Stesso effetto matematico (passthrough), 1/N degli interventi, niente catena. Implementato in `decoder/layer_skipper.py:forward`.
- **Fix B (sistema)**: per long-running task su Mac mini headless, prefisso `caffeinate -i -m -s` al python command per disabilitare display sleep, idle sleep e system sleep durante l'esecuzione. Sufficiente: `caffeinate -i /path/to/python -u script.py`.
- **Verificato**: PASS su exp_005 group ablation (8 cat × 3 prompt × 6 gruppi, ~3h) dopo i due fix applicati congiuntamente.
- **Scoperta**: 2026-05-11

### P14 — Skip naïve `output=input` insufficiente su Gemma 4 SOTA
- **Sintomo**: top-1 agreement 0% rispetto al baseline su 7 categorie/8 con k_skip=1 (1 gruppo, 17% layer), e 0/8 con k_skip=2 (33%). KL mean per HIGH path 5–25 nats. Solo `general_qa` mostra parziale resilienza (67% agree, KL=1.73, 7 layer skip).
- **Causa**: Gemma 4 E4B è "densely trained" (ogni layer contribuisce significativamente). La strategia `layer.output = layer.input` (o boundary intervention equivalente) cambia top-1 nella maggioranza dei prompt. Pattern diverso da GPT-2/BLOOM dove la ridondanza inter-layer è maggiore — la letteratura su layer skipping è basata in larga parte su modelli pre-2023.
- **Implicazione**: il go criterion AIS originale (`≥30% skip + ≤5% degrado`) è irraggiungibile con skip naïve su Gemma 4. Vedere `docs/phases.md` per il go criterion ridefinito.
- **Fix futuri possibili**:
  1. **Soft skip / layer interpolation**: `output = α·layer(input) + (1-α)·input`. Tunabile per categoria.
  2. **Single-layer ablation granulare**: 6× più costoso, ma rivela layer specifici skippabili nascosti dal group average.
  3. **MoE-style partial skip**: skippare solo MLP o solo attention, non il blocco intero.
  4. **Distillation/finetune del router come "skip predictor"**: prevedere mask soft a runtime.
- **Verificato**: `exp_006` (k_skip=2, k_skip=1) entrambi FAIL su strict criterion. PASS soft documentato in diary `2026-05-12_fase2_close.md`.
- **Scoperta**: 2026-05-12

### P15 — Layer skip via swap ModuleList NON salva compute su accelerate offload misto
- **Sintomo**: `NativeLayerSkipper` (sostituzione di layer con `HardSkipLayer` no-op) misura latency PEGGIORE del baseline su Mac mini 16GB MPS+CPU. Hard skip 33%: +18% lento. Hard skip 17%: +15% lento. Anche con persistent swap (apply una volta + N forward consecutivi).
- **Causa A — accelerate hooks**: con `device_map="auto"`, accelerate registra pre/post-forward hooks su ogni layer per gestire data transfer MPS↔CPU. I `HardSkipLayer` sostituiti non ereditano questi hook → il flow MPS/CPU si rompe → fallback PyTorch fa `.to()` automatico costoso per ogni hidden state.
- **Causa B — cache MPS fragmentation**: i forward consecutivi mostrano drift +10% lento. Lo swap di ModuleList ricrea sub-grafi che invalidano la cache kernel Metal.
- **Verificato 2026-05-12**: `exp_009b` + `exp_009c` (warmup + 5 run × 3 config). Tutte le configurazioni hard_skip risultano più lente del baseline su E4B (16 GB bf16, NON sta tutto in MPS 8 GB).
- **Fix per il VERO compute saving**:
  1. **Hardware**: deploy su GPU NVIDIA 24GB+ o Mac Studio 128GB, dove il modello sta su un solo device → no accelerate offload → no hook gap.
  2. **Override del forward**: invece di swap di moduli, sub-classe `Gemma4TextModel` e modifica il `forward` per skippare via flag interno (preserva tutti gli accelerate hooks sui layer originali).
  3. **Modelli più piccoli**: su Gemma 4 E2B (~10 GB) il device_map split funziona meglio (single MPS) → saving teoricamente misurabile lì.
- **Implicazione per AIS as product**: la PROMESSA "AIS riduce compute via skip" è valida architettralmente ma richiede hardware adeguato. Su laptop con offload, il saving è teorico. Per Mac mini 16 GB, AIS preserva la qualità (PASS strict) ma NON la latenza.
- **Scoperta**: 2026-05-12

### P5 — Gemma 4 non in cache HF
- **Sintomo**: Prompt Guide AIS §0 dice "Gemma 4 E4B è già scaricato in `~/.cache/huggingface/`" ma `from_pretrained()` parte a scaricare 16GB.
- **Causa**: i modelli `gemma4:*` sono in Ollama (GGUF Q4_K_M), NON in HuggingFace cache. Formati incompatibili.
- **Fix**: scaricare separatamente i pesi HF safetensors. Vedi `modelli.md`.
- **Scoperta**: 2026-05-11

## Metodologia / valutazione

### P6 — GPT-2 Small non predice ' Paris' come top-1
- **Sintomo**: Prompt Guide AIS §3.2 dice "top token deve essere ' Paris' o ' France' — se è altro il modello non si è caricato bene". Su run reale, top-1 è ' the' anche in fp32.
- **Causa**: GPT-2 Small (124M) è troppo piccolo per memorizzare quel fatto con confidenza. ' Paris' è rank 5/50257 in fp32 (p=3.2%), rank 7 in bf16 (p=2.9%). ROME originale (Meng et al. 2022) usa GPT-2 XL (1.5B) e GPT-J (6B), non Small.
- **Fix**: criterio go/no-go di Fase 0 da interpretare con flessibilità. Smoke test usa "distribuzione non-degenerata (max prob > 1%)" invece di top-1 == ' Paris'. Per il causal tracing aspettarsi risultati meno netti della finestra 5-10.
- **Cross-check**: `experiments/exp_000b_precision_xref.py`
- **Scoperta**: 2026-05-11

### P7 — ROUGE come misura di qualità
- **Sintomo**: Prompt Guide AIS §6.3 (Prompt 3.3) usa ROUGE tra output AIS e Ollama come misura di "qualità".
- **Causa**: ROUGE misura overlap lessicale con un riferimento, non qualità rispetto a un gold answer. E Ollama (GGUF Q4_K_M) ≠ HF bf16 — quantizzazione diversa.
- **Fix**: usare benchmark con gold answer (MMLU accuracy, TruthfulQA). Baseline = stesso modello HF bf16 senza AIS, non Ollama.
- **Scoperta**: 2026-05-11 (analisi statica del doc, da confermare in Fase 3)

### P8 — `atol=1e-4` su MPS bf16
- **Sintomo**: Prompt Guide AIS §5.2 (Prompt 2.2) prescrive `atol=1e-4` su `verify_fallback_identity`.
- **Causa**: bf16 ha 7 bit di mantissa, e su MPS potenziali kernel non deterministici. Tuttavia il run smoke test su GPT-2 Small ha mostrato `max_diff == 0.0` esatto su 3 run identici.
- **Fix**: tenere `atol=1e-4` ma aggiungere un secondo test di **rank equivalence** sui top-5 token (più robusto se in futuro MPS introducesse non-determinismo). Sul decoder vero (Gemma 4 E4B) riverificare il determinismo prima di fidarsi della soglia.
- **Scoperta**: 2026-05-11 (analisi + smoke test)

## Template per nuovi pitfall

```
### PN — <titolo breve>
- **Sintomo**: ...
- **Causa**: ...
- **Fix**: ...
- **Versione/contesto**: ...
- **Scoperta**: YYYY-MM-DD
```
