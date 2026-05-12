# CLAUDE.md — Adaptive Inference System

Indice operativo per Claude Code. Letto automaticamente all'inizio di ogni sessione. **I dettagli vivono in `docs/`** — questo file resta volutamente minimale.

## Cos'è AIS in una riga

Infrastruttura di inferenza locale a 3 componenti (Router + Topological Map FAISS + Decoder con layer skipping) che garantisce per costruzione output ≥ baseline. Dettagli in `docs/architecture.md`. Fonte di verità completa nei `.docx` esterni a questo repo (`../*.docx`).

## Setup essenziale

- **Env Python**: `/opt/anaconda3/envs/ais/bin/python` (conda env `ais`, Python 3.12)
- `conda activate` **non persiste** tra Bash tool call → usa sempre path assoluti
- Hardware target: Mac Mini M4 16GB. MPS bf16 (no fp16, no cuda)

## Top 4 pitfall — non-negoziabili (dettagli in `docs/pitfalls.md`)

1. **Esegui da file `.py`**, mai `python -c` o heredoc (nnsight crasha su `sys.argv[0]=''`)
2. **`dtype=` non `torch_dtype=`** (transformers 5.x)
3. **API nnterp = `StandardizedTransformer`**, non `NNsightModel`
4. **`with model.trace(...)`**: init liste FUORI, append DENTRO (AST rewriting nnsight)

## Struttura repo

```
adaptive-inference-system/
├── CLAUDE.md                  ← questo file (indice)
├── docs/                      ← documentazione operativa estesa
├── .claude/agents/            ← ais-reviewer, ais-tester, ais-docwriter
├── skippers/ pipeline/ corpus/ mappa/
├── experiments/               ← exp_NNN_*.py
├── tests/                     ← sanity + performance + fallback
├── diary/                     ← log YYYY-MM-DD per sessione
└── results/                   ← CSV, HTML, JSON di benchmark
```

## Indice `docs/`

| File | Cosa contiene |
|---|---|
| `docs/architecture.md` | Riassunto AIS, 3 componenti, flusso runtime, garanzia FALLBACK |
| `docs/conventions.md` | Convenzioni codice estese (i 4 pitfall + altri), regole di progetto |
| `docs/pitfalls.md` | Tabella completa dei pitfall verificati con sintomo/causa/fix/data |
| `docs/phases.md` | Fasi 0-3, go/no-go, note correttive rispetto ai `.docx` |
| `docs/modelli.md` | Download HF (no `huggingface-cli`), Ollama vs HF, gated models, disco |
| `docs/agents.md` | Quando invocare quale sub-agent, workflow tipico, politica docwriter |

## Sub-agents

| Agent | Quando | Output |
|---|---|---|
| `ais-reviewer` | Dopo aver scritto codice nuovo | Report verdetto + issues con `file:line` |
| `ais-tester` | Dopo nuovo `experiments/exp_NNN_*.py` o milestone | Numeri + PASS/FAIL + interpretazione |
| `ais-docwriter` | A fine sessione | Diary entry + update a `docs/` (non a CLAUDE.md, salvo indice) |

No coder agent dedicato — la conversazione principale fa coding diretto.

## Reference

- Architettura completa: `../Adaptive_Inference_System.docx`
- Roadmap fasi: `../Roadmap_AIS_v2.docx`
- Prompt operativi: `../Prompt_Guide_AIS_v2.docx`

I `.docx` hanno bug noti (vedi `docs/pitfalls.md` P5-P8). In caso di divergenza, **fida di questo CLAUDE.md e di `docs/`**, non dei `.docx`.
