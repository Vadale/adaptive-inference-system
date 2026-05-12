"""AISInferencePipeline — flow end-to-end del progetto AIS.

Componenti (vedi `docs/architecture.md`):
  1. CERVELLETTO (Gemma 4 E2B): embedda il prompt (L09 last-token con chat
     template, hidden 1536) → query embedding per la Mappa.
  2. MAPPA TOPOLOGICA (FAISS, 5000 entries): lookup del top-k più vicini →
     domain (categoria stimata) + `layer_importance` aggregato.
  3. POLICY: se confidence (cosine similarity) >= threshold → HIGH (skip
     layer non critici del cervellone). Altrimenti → FALLBACK (forward completo
     del cervellone, bit-identico al baseline — garanzia commerciale).
  4. CERVELLONE (Gemma 4 E4B): forward con `active_layers` derivato.

Default conservativo: confidence_threshold=0.999 → praticamente sempre
FALLBACK (output = baseline esatto). Quando i fix futuri (interpolation,
single-layer ablation) miglioreranno il HIGH path, si abbasserà il threshold.

Vincoli: P1-P14. Modelli caricati separatamente — entrambi entrano in
memoria su Mac mini 16GB? E2B su MPS (10GB) + E4B su MPS+CPU (16GB) = ~24GB.
Per ora E2B e E4B caricati on-demand (uno alla volta) — il pipeline tiene
solo il cervellone in memoria, ricreando il cervelletto per ogni query
(slow ma realistic per smoke).
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from nnsight import VisionLanguageModel

from pipeline.mappa import TopologicalMap, MapEntry
from cervellone.layer_skipper import AdaptiveLayerSkipper

CERVELLETTO_MODEL_ID = "google/gemma-4-E2B-it"
CERVELLETTO_PIVOT_LAYER = 9
CERVELLETTO_DEVICE_MAP = {
    "model.vision_tower": "cpu", "model.audio_tower": "cpu",
    "model.embed_vision": "cpu", "model.embed_audio": "cpu",
    "model.language_model": "mps", "lm_head": "mps",
}


@dataclass
class InferenceTrace:
    """Trace di una singola query attraverso la pipeline AIS."""
    prompt: str
    embedding_shape: tuple[int, ...]
    matched_entry: Optional[MapEntry]   # None se la mappa è vuota
    similarity: float                    # cosine [0..1]; -inf se mappa vuota
    confidence_threshold: float
    is_high_path: bool                   # True se HIGH (skip applicato), False FALLBACK
    skipped_layers: list[int]
    logits_last: torch.Tensor            # [vocab] float32 CPU


class _CervellettoEncoder:
    """Wrapper minimale che usa Gemma 4 E2B per produrre l'embedding query."""

    def __init__(self, model_id: str = CERVELLETTO_MODEL_ID,
                 device_map: dict | None = None):
        device_map = device_map or CERVELLETTO_DEVICE_MAP
        self.model = VisionLanguageModel(model_id, dtype=torch.bfloat16,
                                          device_map=device_map)
        self.processor = self.model.processor
        self.pivot = CERVELLETTO_PIVOT_LAYER

    def _chatify(self, text: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        return self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def embed(self, prompt: str, max_chars: int = 300) -> np.ndarray:
        text = self._chatify(prompt[:max_chars])
        holder = [None]
        with torch.no_grad():
            with self.model.trace(text):
                holder[0] = self.model.model.language_model.layers[self.pivot].output.save()
        v = holder[0]
        v = v[0] if isinstance(v, tuple) else v
        return v[0, -1, :].float().cpu().numpy()


def _select_skip_from_importance(li: np.ndarray, threshold: float = 0.10) -> list[int]:
    """Da `layer_importance[n_layers]`, ritorna i layer con importance < threshold.
    Default 0.10 = skip solo layer "very low importance" (più conservativo del
    k_skip=1 di exp_006). In Fase 2 close `general_qa` ha mostrato che 17% skip
    è già marginale: 10% threshold tende a selezionare ~7-14 layer, ma per
    `general_qa` la maggior parte sono in g5 con norm 0.0 e g3 con norm 0.08."""
    return [i for i, v in enumerate(li) if v < threshold]


class AISInferencePipeline:
    """Pipeline AIS minima end-to-end. Carica solo cervellone (E4B) di default;
    il cervelletto (E2B) è caricato on-demand per embedding query — costoso ma
    realistico su 16GB unified memory dove i due non coabitano.
    """

    def __init__(
        self,
        map_dir: Path | str,
        confidence_threshold: float = 0.999,
        skip_importance_threshold: float = 0.10,
        load_cervellone: bool = True,
    ):
        self.map = TopologicalMap.load(map_dir)
        self.confidence_threshold = confidence_threshold
        self.skip_importance_threshold = skip_importance_threshold
        self.cervellone: Optional[AdaptiveLayerSkipper] = (
            AdaptiveLayerSkipper() if load_cervellone else None
        )
        self.encoder: Optional[_CervellettoEncoder] = None  # lazy

    def _ensure_encoder(self) -> _CervellettoEncoder:
        if self.encoder is None:
            self.encoder = _CervellettoEncoder()
        return self.encoder

    def _ensure_cervellone(self) -> AdaptiveLayerSkipper:
        if self.cervellone is None:
            self.cervellone = AdaptiveLayerSkipper()
        return self.cervellone

    def infer_from_embedding(
        self, prompt: str, embedding: np.ndarray
    ) -> InferenceTrace:
        """Variante per smoke/test: prende embedding pre-calcolato (es. dal
        corpus NPZ di Fase 1) invece di chiamare il cervelletto live. Evita
        di tenere E2B + E4B in memoria contemporanea (>16 GB unified)."""
        emb = embedding
        top = self.map.lookup(emb, k=1)
        return self._policy_and_forward(prompt, emb, top)

    def _policy_and_forward(self, prompt, emb, top):
        if not top:
            similarity = float("-inf")
            matched: Optional[MapEntry] = None
        else:
            similarity, _, matched = top[0]
        is_high = (
            matched is not None
            and matched.layer_importance is not None
            and similarity >= self.confidence_threshold
        )
        if is_high:
            assert matched is not None and matched.layer_importance is not None
            skip_layers = _select_skip_from_importance(
                matched.layer_importance, threshold=self.skip_importance_threshold
            )
        else:
            skip_layers = []
        cervellone = self._ensure_cervellone()
        if skip_layers:
            active = set(range(cervellone.n_layers)) - set(skip_layers)
            r = cervellone.forward(prompt, active_layers=active)
        else:
            r = cervellone.forward(prompt, active_layers=None)
        return InferenceTrace(
            prompt=prompt,
            embedding_shape=tuple(emb.shape),
            matched_entry=matched,
            similarity=float(similarity) if similarity != float("-inf") else float("-inf"),
            confidence_threshold=self.confidence_threshold,
            is_high_path=is_high,
            skipped_layers=skip_layers,
            logits_last=r.logits_last,
        )

    def infer(self, prompt: str) -> InferenceTrace:
        """Full pipeline: cervelletto.embed → map.lookup → policy → cervellone.
        NB: tiene cervelletto + cervellone in memoria contemporanea — su
        Mac mini 16 GB rischia OOM (vedi `pipeline.py` docstring). Preferire
        `infer_from_embedding` per smoke/test."""
        encoder = self._ensure_encoder()
        emb = encoder.embed(prompt)
        top = self.map.lookup(emb, k=1)
        return self._policy_and_forward(prompt, emb, top)
