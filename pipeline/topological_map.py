"""TopologicalMap — FAISS index used by the router to look up layer-skip plans
for the decoder model (see `docs/architecture.md`).

Schema (key/value):

  key:   prompt embedding (router pivot-layer last-token, hidden_dim, float32)
  value: MapEntry {
    domain: str (category, e.g. 'open_qa', 'summarization', ...)
    layer_importance: np.ndarray[float32, n_decoder_layers] — how critical each
        decoder layer is for this category (1=critical, 0=safe to skip).
        None until populated by the ablation step.
    confidence_threshold: float (default 0.75)
    observed_count: int
    avg_quality_score: float
  }

Persistence:
  - `<dir>/index.faiss` — FAISS index
  - `<dir>/values.npz`  — parallel array of values (same order as index IDs)
  - `<dir>/meta.json`   — n_decoder_layers, hidden_dim, source dataset/model, ...

Index type: `IndexFlatIP` over L2-normalized embeddings → inner product is
equivalent to cosine similarity. For 5k vectors @ 3072-dim: ~60 MB on disk,
query <1ms. Migrate to `IndexIVFFlat` if N grows past ~100k.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


@dataclass
class MapEntry:
    domain: str
    layer_importance: np.ndarray | None = None   # [n_decoder_layers] float32
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
        # Resilient to schema evolution: ignore unknown keys (legacy),
        # accept missing values via defaults.
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        li = filtered.get("layer_importance")
        if li is not None:
            filtered["layer_importance"] = np.asarray(li, dtype=np.float32)
        return cls(**filtered)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    # Zero embedding → zero (not x/1e-9, which would explode). Cosine vs zero
    # is undefined: the zero vector matches no one (sim ~ 0 with everything).
    return np.where(norm > 1e-9, x / np.maximum(norm, 1e-9), 0.0)


class TopologicalMap:
    """FAISS index over router embeddings + parallel MapEntry list.

    Conventions:
      - Embeddings stored internally as float32, L2-normalized.
      - similarity = inner product = cosine in [-1, 1].
    """

    def __init__(self, hidden_dim: int, n_decoder_layers: int):
        self.hidden_dim = int(hidden_dim)
        self.n_decoder_layers = int(n_decoder_layers)
        self.index = faiss.IndexFlatIP(self.hidden_dim)
        self.entries: list[MapEntry] = []

    def __len__(self) -> int:
        n_idx = self.index.ntotal
        assert n_idx == len(self.entries), (
            f"invariant broken: index.ntotal={n_idx} != len(entries)={len(self.entries)}"
        )
        return n_idx

    def add_batch(
        self, embeddings: np.ndarray, entries: Iterable[MapEntry]
    ) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        assert embeddings.ndim == 2 and embeddings.shape[1] == self.hidden_dim, (
            f"shape={embeddings.shape}, expected (N, {self.hidden_dim})"
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
        """Return the `k` entries closest to the query embedding.
        Output: list of (cosine_similarity, internal_idx, MapEntry).
        Similarity in [-1, 1]; 1.0 = exact match."""
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
            "n_decoder_layers": self.n_decoder_layers,
            "n_entries": len(self.entries),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, in_dir: Path | str) -> "TopologicalMap":
        in_dir = Path(in_dir)
        meta = json.loads((in_dir / "meta.json").read_text())
        # Backward-compat: legacy maps saved with `n_cervellone_layers` key.
        n_layers = meta.get("n_decoder_layers", meta.get("n_cervellone_layers"))
        if n_layers is None:
            raise KeyError(
                "meta.json missing n_decoder_layers (or legacy n_cervellone_layers)"
            )
        m = cls(hidden_dim=meta["hidden_dim"], n_decoder_layers=int(n_layers))
        m.index = faiss.read_index(str(in_dir / "index.faiss"))
        entries_dicts = np.load(in_dir / "values.npz", allow_pickle=True)["entries"]
        m.entries = [MapEntry.from_dict(d) for d in entries_dicts]
        return m
