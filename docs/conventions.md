# Code & project conventions

## Codice — non-negoziabili

Quattro regole che, se violate, generano bug o crash garantiti. Una shortlist degli stessi punti è anche in `CLAUDE.md` come reminder permanente. Qui c'è il dettaglio.

### 1. Esegui sempre da file `.py`

Mai con `python -c` o heredoc. nnsight 0.7.0 ha bug in `__init__.py:67`:
```python
__INTERACTIVE__ = (sys.flags.interactive or not sys.argv[0]) and not __IPYTHON__
```
`python -c` e heredoc lasciano `sys.argv[0]=''`, quindi `not ''=True` → attiva un NNsightConsole spurio che crasha su `exec(None,...)`. Soluzione: scrivere in `experiments/exp_NNN_*.py` ed eseguire come file. È anche il pattern del Prompt Guide.

### 2. `dtype=` non `torch_dtype=`

In transformers 5.x `torch_dtype` è deprecato. Esempio corretto:
```python
model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-4-E4B', dtype=torch.bfloat16, device_map='mps',
)
```
Il Prompt Guide AIS usa `torch_dtype=` ovunque — sostituisci sistematicamente.

### 3. API nnterp = `StandardizedTransformer`

NON `NNsightModel` come scritto nel Prompt Guide §3.2/§4.1.
```python
from nnterp import StandardizedTransformer
model = StandardizedTransformer('gpt2', dtype=torch.bfloat16, device_map='mps')
# Accessors:
#   model.layers_output[i]    — output del layer i (residual stream)
#   model.mlps_output[i]      — output MLP del layer i
#   model.attentions_output[i] — output attention del layer i
#   model.logits              — logits finali
```

### 4. Scope nel `with model.trace(...)`

nnsight 0.7.0 fa AST rewriting del body: locals **non sopravvivono fuori**. Pattern corretto:
```python
captured = []                              # init FUORI
with model.trace(prompt):
    for i in range(n_layers):
        captured.append(model.layers_output[i].save())   # append DENTRO
# qui captured[i] è popolato col .value risolto
```

## Altri vincoli di codice

- **bf16, non fp16**: MPS non supporta fp16 in modo affidabile.
- **`torch.no_grad()` attorno ai trace** quando catturi attivazioni in massa (>100 prompt) — altrimenti il grafo autograd esplode su MPS 16GB.
- **`torch.mps.empty_cache()`** dopo batch grossi.
- **Path assoluti** o `Path.home()` — niente hardcoded `/Users/alessandrovadala/...`.
- **Commenti** solo per il WHY non-ovvio. No commenti che ridescrivono il codice. No docstring multi-paragrafo.
- **NaN check** su tutte le attivazioni catturate, non un campione.

## Progetto — convenzioni operative

- **Regola del 20%**: max 1h/giorno finché Signal Noise non ha entrate.
- **Repo separato** da Signal Noise/Noroom_code. Integrazione solo dopo Fase 3.
- **"Claude Code scrive codice, l'utente decide"**: mai delegare a Claude Code "sta funzionando bene?".
- **Go/no-go per fase**: non si procede senza che passi (vedi `phases.md`).
- **Numeri, non aggettivi vaghi** ("buono", "alto" sono banditi nei diary e nei report di test).
- **Diary entry** dopo ogni sessione, max 300 parole, template in `ais-docwriter` agent.

## Esecuzione

Tutti i Python invocations passano attraverso il path assoluto dell'env conda `ais`:
```
/opt/anaconda3/envs/ais/bin/python <script>
```
`conda activate` non persiste tra Bash tool call separati di Claude Code.
