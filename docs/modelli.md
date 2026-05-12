# Modelli — download e uso

## Premessa

Sul sistema dell'utente:
- **Ollama installato** con `gemma4:e2b`, `huihui_ai/gemma-4-abliterated:e4b`, `qwen3.5:*` (formato GGUF Q4_K_M).
- **NO `huggingface-cli`** installato. NO Gemma in cache HF.
- Cache HF attuale: `BAAI/bge-small-en`, `docling-*`, `Llama-2-7b`, `MiniLM`, `gpt2` (scaricato 2026-05-11).

## Ollama vs HuggingFace — quando usare cosa

| Esigenza | Strumento |
|---|---|
| Leggere attivazioni interne, fare hook, fine-tuning | **HuggingFace** (safetensors bf16) — Fasi 0, 1, 2, 3 |
| Solo vedere cosa il modello produce, benchmark output finale | Ollama OK (Fase 3 validazione) |
| Confronto qualità/latenza fra setup quantizzati | Ollama |
| Costruire la mappa topologica | HuggingFace (richiede attivazioni interne) |

**Regola**: se devi vedere DENTRO il modello → HuggingFace. Se devi vedere COSA produce → Ollama va bene.

I modelli GGUF di Ollama **non si possono usare con nnterp** — sono quantizzati e non espongono il grafo PyTorch.

## Download di modelli HuggingFace (no `huggingface-cli`)

Usa `huggingface_hub.snapshot_download` da Python — è già installato come dipendenza di `transformers`:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/gemma-4-E2B-it',
    # cache_dir=None usa default ~/.cache/huggingface/hub
)
```

Oppure direttamente con `from_pretrained()` la prima volta: scarica automaticamente.

## Modelli Gemma — gated

I modelli `google/gemma-*` su HuggingFace sono **gated**: richiedono accettazione T&C + access token. Procedura una-tantum:

1. Crea/accedi all'account su https://huggingface.co
2. Vai alla pagina del modello (es. https://huggingface.co/google/gemma-4-E4B) e clicca "Agree and access repository"
3. Vai in **Settings → Access Tokens** → crea token "Read" (gratuito)
4. Imposta nel terminale: `export HF_TOKEN=hf_xxx` oppure in Python:
   ```python
   from huggingface_hub import login
   login(token='hf_xxx')   # una volta sola, salva in ~/.cache/huggingface/token
   ```

## Modelli del progetto AIS — quando scaricare cosa

| Fase | Modello | Repo HF | Dimensione bf16 | Stato |
|---|---|---|---|---|
| 0 | GPT-2 Small | `gpt2` | 500MB | Scaricato 2026-05-11 |
| 1 | Cervelletto | `google/gemma-4-E2B-it` | 10.3GB | Scaricato 2026-05-11 |
| 2 | Cervellone | `google/gemma-4-E4B-it` | 16.0GB | Da scaricare a inizio Fase 2 |
| 3 | Validazione | `gemma4:e2b`, `:e4b` (Ollama) | già installati | Fase 3 benchmark |

**Correzioni rispetto ai .docx (2026-05-11)**:
- Gemma 4 E2B/E4B sono **multimodali** (text+image+audio+video), pipeline `any-to-any`, registrati come `AutoModelForImageTextToText`. Vedi `pitfalls.md` P11.
- Al 2026-05-11 i repo `google/gemma-4-E*B-it` sono `gated=False` (accesso libero). Il token cached in `~/.cache/huggingface/token` resta utile come fallback se in futuro tornano gated.
- Per caricarli con nnsight serve `VisionLanguageModel` + `device_map` split (vision/audio su CPU), non `StandardizedTransformer`. Pattern documentato in `experiments/exp_002a_gemma_smoke.py`.

**Totale disco bf16**: ~26GB per Gemma 4 E2B + E4B. Sul Mac Mini 16GB con 51GB liberi (stato 2026-05-11) ci stiamo, ma il corpus di attivazioni Fase 1 può occupare altri 20-30GB. **Monitorare `df -h`** prima di lanciare collezione attivazioni.

## Quando scaricare — script template

Da preparare in `scripts/download_models.py` all'inizio di Fase 1:

```python
"""Scarica i modelli HF necessari per Fase 1+. Richiede HF_TOKEN."""
import os
from huggingface_hub import snapshot_download, login

token = os.environ.get('HF_TOKEN')
if not token:
    raise SystemExit("Set HF_TOKEN env var. Vedi docs/modelli.md.")
login(token=token)

for repo_id in ['google/gemma-4-E2B-it', 'google/gemma-4-E4B-it']:
    print(f"Downloading {repo_id}...")
    snapshot_download(repo_id=repo_id)
print("Done.")
```

## Verifica modelli scaricati

```bash
# Cache HF
ls ~/.cache/huggingface/hub/

# Ollama
ollama list
```
