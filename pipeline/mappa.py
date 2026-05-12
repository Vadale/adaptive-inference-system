"""TopologicalMap — indice FAISS per il routing del cervelletto verso layer
del cervellone (vedi `docs/architecture.md`).

Schema della chiave/valore (versione iniziale, layer_importance da popolare in
Fase 2 piena):

  key:   embedding del prompt (cervelletto L9 last-token, hidden=1536, float32)
  value: MapEntry {
    domain: str (categoria, es. 'open_qa', 'summarization', ...)
    layer_importance: np.ndarray[float32, n_cervellone_layers] — quanto ogni
        layer del cervellone è critico per questa categoria (1=critico, 0=salta).
        None finché non popolato.
    confidence_threshold: float (default 0.75)
    observed_count: int
    avg_quality_score: float
  }

Persistenza:
  - `<dir>/index.faiss`         — indice FAISS
  - `<dir>/values.npz`          — array di valori parallelo (lo stesso ordine
                                   degli ID dell'indice)
  - `<dir>/meta.json`           — n_layers cervellone, hidden_dim cerveletto,
                                   dataset/modelli sorgenti, etc.

Index type: `IndexFlatIP` su embedding L2-normalizzati → inner product ≡
cosine similarity. Per 5k vettori 1536-dim: ~30 MB in RAM, query <1ms.
Quando N→100k, migrare a `IndexIVFFlat` (non urgente).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


@dataclass
class MapEntry:
    domain: str
    layer_importance: np.ndarray | None = None   # [n_cervellone_layers] float32
    confidence_threshold: float = 0.75
    observed_count: int = 0
    avg_quality_score: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.layer_importance is not None:
            d["layer_importance"] = self.layer_importance.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MapEntry":
        # Resiliente a schema evolution: ignora chiavi extra (legacy),
        # accetta valori mancanti coi default.
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        li = filtered.get("layer_importance")
        if li is not None:
            filtered["layer_importance"] = np.asarray(li, dtype=np.float32)
        return cls(**filtered)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    # Embedding zero → zero (non x/1e-9 che esploderebbe). Cosine vs zero
    # è indefinita: il vettore zero non match nessuno (sim≈0 con tutti).
    return np.where(norm > 1e-9, x / np.maximum(norm, 1e-9), 0.0)


class TopologicalMap:
    """FAISS index su embedding cervelletto + entries parallele.

    Convenzioni:
      - embedding salvati internamente come float32 L2-normalized.
      - similarity = inner product = cosine in [-1, 1].
    """

    def __init__(self, hidden_dim: int, n_cervellone_layers: int):
        self.hidden_dim = int(hidden_dim)
        self.n_cervellone_layers = int(n_cervellone_layers)
        self.index = faiss.IndexFlatIP(self.hidden_dim)
        self.entries: list[MapEntry] = []

    def __len__(self) -> int:
        n_idx = self.index.ntotal
        assert n_idx == len(self.entries), (
            f"invariante rotta: index.ntotal={n_idx} != len(entries)={len(self.entries)}"
        )
        return n_idx

    def add_batch(
        self, embeddings: np.ndarray, entries: Iterable[MapEntry]
    ) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        assert embeddings.ndim == 2 and embeddings.shape[1] == self.hidden_dim, (
            f"shape={embeddings.shape}, atteso (N, {self.hidden_dim})"
        )
        emb_norm = _l2_normalize(embeddings)
        new_entries = list(entries)
        assert len(new_entries) == len(emb_norm), (
            f"len(entries)={len(new_entries)} != N={len(emb_norm)}"
        )
        self.index.add(emb_norm)
        self.entries.extend(new_entries)

    def lookup(
        self, query: np.ndarray, k: int = 1
    ) -> list[tuple[float, int, MapEntry]]:
        """Ritorna `k` entries più vicini al query embedding.
        Output: lista di (similarity_cosine, idx_interno, MapEntry).
        Similarity in [-1, 1]; 1.0 = match perfetto."""
        q = np.atleast_2d(np.ascontiguousarray(query, dtype=np.float32))
        assert q.shape[1] == self.hidden_dim
        q = _l2_normalize(q)
        sims, idxs = self.index.search(q, k)
        out: list[tuple[float, int, MapEntry]] = []
        for sim, idx in zip(sims[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            out.append((float(sim), int(idx), self.entries[idx]))
        return out

    def save(self, out_dir: Path | str) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "index.faiss"))
        entries_dicts = [e.to_dict() for e in self.entries]
        np.savez_compressed(out_dir / "values.npz",
                            entries=np.array(entries_dicts, dtype=object))
        meta = {
            "hidden_dim": self.hidden_dim,
            "n_cervellone_layers": self.n_cervellone_layers,
            "n_entries": len(self.entries),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, in_dir: Path | str) -> "TopologicalMap":
        in_dir = Path(in_dir)
        meta = json.loads((in_dir / "meta.json").read_text())
        m = cls(hidden_dim=meta["hidden_dim"],
                n_cervellone_layers=meta["n_cervellone_layers"])
        m.index = faiss.read_index(str(in_dir / "index.faiss"))
        entries_dicts = np.load(in_dir / "values.npz", allow_pickle=True)["entries"]
        m.entries = [MapEntry.from_dict(d) for d in entries_dicts]
        return m
