# Architecture — sintesi operativa

Riassunto per orientarsi rapidamente. **Fonte di verità completa** nei tre `.docx`:

- `../../Adaptive_Inference_System.docx` — architettura completa
- `../../Roadmap_AIS_v2.docx` — fasi, timeline, milestone
- `../../Prompt_Guide_AIS_v2.docx` — prompt operativi per ogni fase

## I tre componenti

**Cervelletto** — modello small (Gemma 4 E2B, ~5B effettivi). Due ruoli:
- **Router** (runtime): classifica input, interroga la Mappa Topologica, decide se andare in HIGH o FALLBACK.
- **Osservatore** (background): osserva attivazioni del cervellone durante l'inferenza e aggiorna la Mappa.

**Mappa Topologica** — database vettoriale FAISS persistente, separato dai modelli. Schema:
```
chiave:  embedding del tipo di input (256-dim)
valore:  {
  layer_importance: [0.9, 0.2, 0.8, ...],   // per ogni layer del cervellone
  confidence_threshold: 0.75,
  observed_count: 1247,
  domain: 'medical',
  avg_quality_score: 0.91
}
```
Persistente, trasferibile parzialmente tra modelli, condivisibile per dominio, incrementale.

**Cervellone** — modello grande (Gemma 4 E4B ~8B, o altri 14B+ per dominio). Pesi **mai modificati**. Cambia solo quali layer sono attivati per input specifico.

## Flusso runtime

```
input → cervelletto.classify (<50ms)
      → mappa.lookup (<5ms)
      → if confidence >= 0.75:
            cervellone.forward(active_layers=mappa.layers)   [HIGH]
        else:
            cervellone.forward()  # tutti i layer                [FALLBACK]
      → output + confidence_label + confidence_score
      → background: cervelletto osserva, aggiorna mappa
```

## La garanzia fondamentale

**FALLBACK = baseline esatto.** By design, sotto threshold il sistema è bit-equivalente al modello originale. Non può mai essere peggio del baseline. È questa la garanzia commerciale del progetto.

## Differenziazione rispetto alla letteratura

| Esistente | AIS aggiunge |
|---|---|
| MoE (router statico co-trained) | Mappa post-hoc su qualsiasi modello esistente |
| Quantization (pesi compressi) | Pesi intatti, percorso ridotto |
| RAG (retrieve documenti) | Retrieve struttura interna del modello |
| Speculative decoding (predice token) | Predice quali layer servono |
| Pruning (rimuove pesi) | Attivazione selettiva, fallback completo |
| Fine-tuning (riaddestra) | Zero modifiche al cervellone |

Combinazione: agente esterno che osserva il modello, costruisce mappa post-hoc applicabile senza modificarlo, con fallback garantito. Non esiste in letteratura né in prodotti commerciali in questa forma.

## Connessione a Signal Noise

AIS è il **motore tecnico di Signal Noise Tier 2-3** (privacy-first AI on-device). Repo separato da `Noroom_code`. Integrazione come modulo separato solo dopo Fase 3 completa.
