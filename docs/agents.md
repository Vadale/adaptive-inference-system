# Sub-agents disponibili

Tre agent custom in `.claude/agents/`. Vengono caricati nativamente da Claude Code all'avvio della sessione (visibili dopo `/clear` o riavvio di `claude`).

## Quando usare quale

| Agent | Trigger | Cosa produce |
|---|---|---|
| **ais-reviewer** | Dopo aver scritto codice nuovo in `experiments/`, `cervelletto/`, `cervellone/`, `mappa/`, `pipeline/` | Report di review con verdetto (PASS / WITH FIXES / BLOCK), critical/important/minor issues con `file:line`, strengths, sanity check consigliati |
| **ais-tester** | Dopo nuovo `experiments/exp_NNN_*.py`, prima di un milestone, per verificare `verify_fallback_identity` | Numeri concreti + verdetto PASS/FAIL + interpretazione + prossimo passo |
| **ais-docwriter** | A fine sessione o dopo un milestone | Diary `YYYY-MM-DD_*.md` + aggiornamenti mirati a `docs/` (non a `CLAUDE.md` salvo eccezioni) |

## Cosa NON c'è

**No coder agent dedicato**. La conversazione principale fa coding diretto, come da regola "Claude Code scrive codice, l'utente decide". Un coder agent aggiungerebbe un round-trip senza valore.

## Politica del docwriter

Il docwriter aggiorna `docs/` non `CLAUDE.md`. Tocca `CLAUDE.md` **solo** per:
- Nuovo file `docs/` creato → aggiunge 1 riga all'indice
- Nuovo agent in `.claude/agents/` → aggiunge 1 riga alla lista
- Nuova fase iniziata → aggiorna 1 riga nella tabella go/no-go
- Pitfall che entra nella shortlist dei "top non-negoziabili" (max 4-5)

Tutto il resto (dettagli architettura, convenzioni estese, pitfall normali, criteri di fase, modelli) vive in `docs/*.md`.

## Invocazione

In Claude Code:
```
Usa l'agent ais-reviewer per revieware experiments/exp_001_causal_tracing.py
```

Oppure programmaticamente via tool `Agent` con `subagent_type: ais-reviewer`.

Se la sessione corrente non ha gli agent custom caricati (prima volta dopo la loro creazione), fai `/clear` per ricaricare.

## Workflow tipico per Prompt N

```
1. Main thread scrive experiments/exp_NNN_*.py
2. Invoca ais-reviewer → report
3. Main thread applica fix segnalati
4. Invoca ais-tester → esegue + interpreta
5. Se PASS: invoca ais-docwriter → diary entry + eventuale update docs/
6. Se FAIL: diagnostica con main thread, ripeti
```
