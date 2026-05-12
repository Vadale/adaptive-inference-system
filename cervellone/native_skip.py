"""NativeLayerSkipper — skip layer reale via swap dei moduli PyTorch.

A differenza di `AdaptiveLayerSkipper` (intervention nnsight, che esegue i
layer e ne sovrascrive l'output), questo **bypassa l'esecuzione** dei layer
hard-skippati → vero saving di compute. Necessario per il deploy AIS reale
(misurare latency saving).

API:
  - `NativeLayerSkipper(base_skipper)` riusa il modello E4B già caricato
    da un `AdaptiveLayerSkipper`, niente doppio load.
  - `forward(prompt, hard_skip, soft_skip, ...)`:
      hard_skip: set[int]  → questi layer NON vengono eseguiti (saving).
      soft_skip: dict[int, float] → questi layer vengono eseguiti, e l'output
                                     è interpolato con l'input via α.

Restore garantito tramite try/finally: dopo il forward, `model.language_model.layers`
è sempre riportato allo stato originale (no side effects su future chiamate).

Garanzia FALLBACK: con hard_skip=None e soft_skip=None, il forward è
identico al baseline HF (P11 caveat: il -it senza chat template degenera,
ma è coerente con il baseline).

Vincoli: P1-P14.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_MAX_MEMORY = {"mps": "8GiB", "cpu": "30GiB"}


class HardSkipLayer(nn.Module):
    """No-op: ritorna `hidden_states` identico. Bypassa attention+MLP+norms del
    Gemma4TextDecoderLayer → saving reale di compute. Non ha parametri."""

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class SoftSkipLayer(nn.Module):
    """Esegue orig_layer e interpola l'output con l'input.

    output = α · orig_layer(hidden_states, ...) + (1-α) · hidden_states

    NB: NON salva compute (orig_layer è eseguito normalmente). Serve per
    preservare la qualità senza alterare il flow PyTorch.
    """

    def __init__(self, orig_layer: nn.Module, alpha: float):
        super().__init__()
        self.orig_layer = orig_layer
        assert 0.0 < alpha < 1.0, f"alpha deve essere in (0,1), got {alpha}"
        self.alpha = float(alpha)

    def forward(self, hidden_states, *args, **kwargs):
        out = self.orig_layer(hidden_states, *args, **kwargs)
        return self.alpha * out + (1.0 - self.alpha) * hidden_states


class NativeLayerSkipper:
    """Skip nativo via swap di `language_model.layers`. Carica il modello
    direttamente con `AutoModelForImageTextToText` (puro HF, no nnsight) per
    evitare il problema dei meta-tensor che nnsight lascia su moduli non
    toccati dal trace. Thread-safe non garantito — chiamate seriali."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        dtype: torch.dtype = torch.bfloat16,
        device_map: dict | str = "auto",
        max_memory: dict | None = None,
    ):
        if max_memory is None and device_map == "auto":
            max_memory = DEFAULT_MAX_MEMORY
        kwargs: dict = {"dtype": dtype, "device_map": device_map}
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        self.hf_model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.hf_model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self._lm = self.hf_model.model.language_model
        self._orig_modulelist = self._lm.layers
        self._orig_layers = list(self._orig_modulelist)
        self.n_layers = len(self._orig_layers)

    def _build_layers(
        self,
        hard_skip: set[int],
        soft_skip: dict[int, float],
    ) -> nn.ModuleList:
        new_layers: list[nn.Module] = []
        for i, orig in enumerate(self._orig_layers):
            if i in hard_skip:
                new_layers.append(HardSkipLayer())
            elif i in soft_skip:
                new_layers.append(SoftSkipLayer(orig, soft_skip[i]))
            else:
                new_layers.append(orig)
        return nn.ModuleList(new_layers)

    def _chatify(self, text: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        return self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def apply_skip(
        self,
        hard_skip: Optional[set[int]] = None,
        soft_skip: Optional[dict[int, float]] = None,
    ) -> None:
        """Applica uno skip plan in modo PERSISTENT. I forward successivi
        useranno questo skip senza re-swap interno (più veloce per N forward
        consecutivi sulla stessa categoria). Chiamare `restore()` per ripristinare.
        """
        hard_skip = set(hard_skip) if hard_skip else set()
        soft_skip = dict(soft_skip) if soft_skip else {}
        assert not (hard_skip & set(soft_skip)), "layer in entrambi hard/soft"
        self._lm.layers = self._build_layers(hard_skip, soft_skip)

    def restore(self) -> None:
        """Ripristina i layer originali. Idempotente."""
        self._lm.layers = self._orig_modulelist

    def forward_no_swap(self, prompt: str, chat_template: bool = True) -> torch.Tensor:
        """Forward senza swap interno (assume `apply_skip` già chiamato).
        Usare quando si fanno molti forward con lo stesso skip plan."""
        text = self._chatify(prompt) if chat_template else prompt
        inputs = self.processor(text=text, return_tensors="pt")
        prepared: dict[str, torch.Tensor] = {
            k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        with torch.no_grad():
            outputs = self.hf_model(**prepared)
        return outputs.logits[0, -1, :].float().cpu()

    def forward(
        self,
        prompt: str,
        hard_skip: Optional[set[int]] = None,
        soft_skip: Optional[dict[int, float]] = None,
        chat_template: bool = True,
    ) -> torch.Tensor:
        """Restituisce logits del last token come tensor float32 CPU."""
        hard_skip = set(hard_skip) if hard_skip else set()
        soft_skip = dict(soft_skip) if soft_skip else {}
        # Validazione
        for i in hard_skip:
            assert 0 <= i < self.n_layers, f"hard_skip {i} out of range"
        for i in soft_skip:
            assert 0 <= i < self.n_layers, f"soft_skip {i} out of range"
        assert not (hard_skip & set(soft_skip)), (
            f"layer {hard_skip & set(soft_skip)} sia hard che soft"
        )

        text = self._chatify(prompt) if chat_template else prompt
        # Con device_map="auto", accelerate registra hooks che spostano
        # input/output tra MPS/CPU automaticamente. Lasciare gli input sul
        # device del processor (CPU). Forzare .to(emb_device) può produrre
        # meta-tensor se embed_tokens non è ancora materializzato in alcune
        # versioni di nnsight + accelerate.
        inputs = self.processor(text=text, return_tensors="pt")
        prepared: dict[str, torch.Tensor] = {
            k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }

        # Swap dei layer
        new_layers = self._build_layers(hard_skip, soft_skip)
        self._lm.layers = new_layers
        try:
            with torch.no_grad():
                outputs = self.hf_model(**prepared)
            logits = outputs.logits
            return logits[0, -1, :].float().cpu()
        finally:
            self._lm.layers = self._orig_modulelist
