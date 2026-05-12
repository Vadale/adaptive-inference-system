"""AISInferencePipeline — end-to-end flow for the Gemma 4 research path.

Components (see `docs/architecture.md`):
  1. ROUTER (Gemma 4 E2B): embeds the prompt (L09 last-token with chat
     template, hidden=1536) → query embedding for the map.
  2. TOPOLOGICAL MAP (FAISS, 5000 entries): top-k lookup → domain
     (estimated category) + aggregated `layer_importance`.
  3. POLICY: if confidence (cosine similarity) >= threshold → HIGH path
     (skip non-critical decoder layers). Otherwise → FALLBACK (full decoder
     forward, bit-identical to baseline).
  4. DECODER (Gemma 4 E4B): forward with `active_layers` derived from
     the policy.

Conservative default: confidence_threshold=0.999 → almost always FALLBACK
(output = exact baseline). Lower the threshold when the HIGH path quality
improves (interpolation, finer-grained ablation).

Constraints: P1-P14 (see docs/pitfalls.md). Models loaded separately — both
in unified memory on a Mac mini 16GB is tight: E2B on MPS (10GB) + E4B on
MPS+CPU (16GB) ≈ 24GB. Therefore the router (E2B) is loaded on demand for
the embedding query — slow but realistic for a 16GB box.

NOTE: this is the Gemma 4 research pipeline (uses nnsight via
AdaptiveLayerSkipper). For the production-shaped Llama 3.2 path, see
`skippers/llama_skipper.py` and `experiments/exp_013-018`.
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

from pipeline.topological_map import TopologicalMap, MapEntry
from skippers.layer_skipper import AdaptiveLayerSkipper

ROUTER_MODEL_ID = "google/gemma-4-E2B-it"
ROUTER_PIVOT_LAYER = 9
ROUTER_DEVICE_MAP = {
    "model.vision_tower": "cpu", "model.audio_tower": "cpu",
    "model.embed_vision": "cpu", "model.embed_audio": "cpu",
    "model.language_model": "mps", "lm_head": "mps",
}


@dataclass
class InferenceTrace:
    """Trace of a single query through the AIS pipeline."""
    prompt: str
    embedding_shape: tuple[int, ...]
    matched_entry: Optional[MapEntry]   # None if the map is empty
    similarity: float                   # cosine [0..1]; -inf if map empty
    confidence_threshold: float
    is_high_path: bool                  # True if HIGH (skip applied), False FALLBACK
    skipped_layers: list[int]
    logits_last: torch.Tensor           # [vocab] float32 CPU


class _RouterEncoder:
    """Minimal wrapper that uses Gemma 4 E2B to produce the query embedding."""

    def __init__(self, model_id: str = ROUTER_MODEL_ID,
                 device_map: dict | None = None):
        device_map = device_map or ROUTER_DEVICE_MAP
        self.model = VisionLanguageModel(model_id, dtype=torch.bfloat16,
                                          device_map=device_map)
        self.processor = self.model.processor
        self.pivot = ROUTER_PIVOT_LAYER

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
    """Given `layer_importance[n_layers]`, return the indices of layers with
    importance < threshold. Default 0.10 is conservative — it picks the
    universally low-importance layers, leaving anything ambiguous in-place."""
    return [i for i, v in enumerate(li) if v < threshold]


class AISInferencePipeline:
    """End-to-end AIS pipeline (Gemma 4 research path). Loads only the decoder
    (E4B) eagerly; the router (E2B) is created on demand for embedding queries
    — expensive, but realistic on 16GB unified memory where the two don't
    coexist comfortably.
    """

    def __init__(
        self,
        map_dir: Path | str,
        confidence_threshold: float = 0.999,
        skip_importance_threshold: float = 0.10,
        load_decoder: bool = True,
    ):
        self.map = TopologicalMap.load(map_dir)
        self.confidence_threshold = confidence_threshold
        self.skip_importance_threshold = skip_importance_threshold
        self.decoder: Optional[AdaptiveLayerSkipper] = (
            AdaptiveLayerSkipper() if load_decoder else None
        )
        self.encoder: Optional[_RouterEncoder] = None  # lazy

    def _ensure_encoder(self) -> _RouterEncoder:
        if self.encoder is None:
            self.encoder = _RouterEncoder()
        return self.encoder

    def _ensure_decoder(self) -> AdaptiveLayerSkipper:
        if self.decoder is None:
            self.decoder = AdaptiveLayerSkipper()
        return self.decoder

    def infer_from_embedding(
        self, prompt: str, embedding: np.ndarray
    ) -> InferenceTrace:
        """Variant for smoke/test: takes a precomputed embedding (e.g. from
        the Phase 1 corpus NPZ) instead of calling the router live. Avoids
        keeping E2B + E4B in memory at the same time (> 16GB unified)."""
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
        decoder = self._ensure_decoder()
        if skip_layers:
            active = set(range(decoder.n_layers)) - set(skip_layers)
            r = decoder.forward(prompt, active_layers=active)
        else:
            r = decoder.forward(prompt, active_layers=None)
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
        """Full pipeline: router.embed → map.lookup → policy → decoder.
        Keeps router + decoder in memory at the same time — on Mac mini 16GB
        this can OOM. Prefer `infer_from_embedding` for smoke/test."""
        encoder = self._ensure_encoder()
        emb = encoder.embed(prompt)
        top = self.map.lookup(emb, k=1)
        return self._policy_and_forward(prompt, emb, top)
