"""AdaptiveLayerSkipper — research wrapper around Gemma 4 E4B (the decoder).

Core guarantee (see `docs/architecture.md`): in FALLBACK mode the output is
**bit-identical** to the baseline forward pass of the underlying model.

API:
  - `forward(prompt, active_layers=None, alpha=0.0)` → last-token logits.
  - `active_layers=None` or `set(range(n_layers))` → pure FALLBACK: no
    intervention, PyTorch path identical to the baseline.
  - `active_layers != all` → layers NOT in `active_layers` are "skipped"
    with interpolation parameterized by `alpha`. For each contiguous skip
    group `[gs, ge)`:
        layers[ge-1].output = α · orig_output + (1-α) · layers[gs].input
      - α=0.0 (default): HARD SKIP (output = input). Equivalent to the
        ablation pattern used during Phase 2.
      - α=1.0: NO SKIP (output = original). No-op.
      - α∈(0,1): SOFT SKIP. The group's representation is interpolated
        between the transformer output and its input. On modern instruction-
        tuned models like Gemma 4 (P14), α≈0.3-0.7 preserves top-k quality
        much better than α=0.

Compute note: skipping via nnsight intervention does NOT save compute (the
module still runs; its output is just overwritten). For real saving in
deployment, use `NativeLayerSkipper` or `LlamaSkipper` (native PyTorch
ModuleList swap that bypasses execution).

Constraints: P1-P14 (see docs/pitfalls.md). VLM + device_map="auto" +
max_memory (see P12).
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from nnsight import VisionLanguageModel

MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_MAX_MEMORY = {"mps": "8GiB", "cpu": "30GiB"}
# Device map "fisso" che funziona dopo il primo carico (cache hot). vision/audio
# su CPU, language_model+lm_head su MPS. Più predicibile di "auto" (no boundary
# shuffle tra layer testuali → niente cross-device proxy in nnsight).
FIXED_DEVICE_MAP = {
    "model.vision_tower": "cpu", "model.audio_tower": "cpu",
    "model.embed_vision": "cpu", "model.embed_audio": "cpu",
    "model.language_model": "mps", "lm_head": "mps",
}


def _contiguous_groups(layer_ids: set[int]) -> list[tuple[int, int]]:
    """Da un set di layer skippati, ritorna ranges contigui (start, end_excl).
    Es. {28,29,30,31,32,33,34} → [(28, 35)]. {0, 5, 6, 7} → [(0,1), (5,8)]."""
    if not layer_ids:
        return []
    sorted_ids = sorted(layer_ids)
    groups: list[tuple[int, int]] = []
    start = prev = sorted_ids[0]
    for x in sorted_ids[1:]:
        if x == prev + 1:
            prev = x
        else:
            groups.append((start, prev + 1))
            start = prev = x
    groups.append((start, prev + 1))
    return groups


def _unwrap(t):
    return t[0] if isinstance(t, tuple) else t


@dataclass
class ForwardResult:
    """Risultato di un forward — logits del last token (V-dim float32) + meta."""
    logits_last: torch.Tensor   # [vocab] float32, CPU
    n_active: int               # quanti layer attivi
    n_layers: int               # totale layer testuali
    fallback: bool              # True se nessun layer skippato


class AdaptiveLayerSkipper:
    """Wraps the decoder (Gemma 4 E4B). Exposes `forward(prompt, active_layers)`."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        dtype: torch.dtype = torch.bfloat16,
        device_map: dict | str | None = None,
        max_memory: dict | None = None,
    ):
        # Default: device_map="auto" + max_memory (P12). Memoria fredda forza
        # il caching_allocator_warmup MPS — con fixed split eccede 13.9 GiB
        # single-buffer limit. "auto" + max_memory={"mps": "8GiB"} sharda
        # automaticamente sotto il limit. Il boundary cross-device tra layer
        # è gestito dalla boundary intervention (P13).
        if device_map is None:
            device_map = "auto"
        kwargs = {"dtype": dtype, "device_map": device_map}
        if device_map == "auto" and max_memory is None:
            kwargs["max_memory"] = DEFAULT_MAX_MEMORY
        elif max_memory is not None:
            kwargs["max_memory"] = max_memory
        self.model = VisionLanguageModel(model_id, **kwargs)
        self.processor = self.model.processor
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.n_layers = len(self.model._model.model.language_model.layers)
        self._all_layers = set(range(self.n_layers))

    def _chatify(self, text: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        return self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def forward(
        self,
        prompt: str,
        active_layers: Optional[Iterable[int]] = None,
        alpha: float = 0.0,
        chat_template: bool = True,
    ) -> ForwardResult:
        """Forward su prompt con eventuale soft-skip dei layer non attivi.

        Args:
            prompt: testo input.
            active_layers: layer attivi al 100%. None o tutti → FALLBACK.
            alpha: interpolation factor per i layer "skippati":
                output_skip = α·orig_output + (1-α)·layers[gs].input
                α=0 → hard skip (passthrough). α=1 → no skip. α∈(0,1) → soft.
            chat_template: applica Gemma chat template (default True).
        """
        assert 0.0 <= alpha <= 1.0, f"alpha deve essere in [0,1], got {alpha}"

        if active_layers is None:
            active = self._all_layers
        else:
            active = set(int(i) for i in active_layers)
            invalid = [i for i in active if i < 0 or i >= self.n_layers]
            assert not invalid, f"active_layers fuori range: {invalid}"

        skip = self._all_layers - active
        is_fallback = (len(skip) == 0)
        skip_groups = _contiguous_groups(skip)

        text = self._chatify(prompt) if chat_template else prompt
        logits_save = None

        with torch.no_grad():
            with self.model.trace(text):
                if not is_fallback:
                    layers = self.model.model.language_model.layers
                    for (gs, ge) in skip_groups:
                        # boundary_in: ingresso al gruppo. Può essere tupla
                        # (hidden, ...). Per Gemma 4 testuale `layer.input`
                        # è single tensor; nnsight lo gestisce trasparente.
                        boundary_in = layers[gs].input
                        if alpha == 0.0:
                            # Path attuale "hard skip": output = input
                            layers[ge - 1].output = boundary_in
                        else:
                            # Soft skip: α·orig + (1-α)·input
                            orig_out = layers[ge - 1].output
                            new_out = alpha * orig_out + (1.0 - alpha) * boundary_in
                            layers[ge - 1].output = new_out
                logits_save = self.model.lm_head.output.save()

        last = logits_save[0, -1, :].float().cpu()
        return ForwardResult(
            logits_last=last,
            n_active=len(active),
            n_layers=self.n_layers,
            fallback=is_fallback,
        )
