"""LlamaSkipper — layer skipper for Llama 3.x.

Llama 3.x has a standard transformer architecture (no shared-KV pattern like
Gemma 4), so a `nn.ModuleList` swap works out of the box → real compute
saving when a layer is hard-skipped.

API:
  - `forward(prompt, hard_skip, soft_skip)` — one-shot, auto restore.
  - `apply_skip(hard, soft)` + `forward_no_swap(prompt)` + `restore()` —
    persistent skip mode (apply once, then reuse for many forwards).
  - `embed(prompt, layer)` — runs the model with output_hidden_states=True
    and returns the hidden state of an intermediate layer at last-token
    position. Used as a router embedding (the same model serves as router +
    decoder, which is convenient but expensive — see ARTICLE.md §6).

Fallback guarantee: with hard_skip=None and soft_skip=None, `forward()` is
bit-identical to the HuggingFace baseline. Compute saving on hard skip
scales linearly with the number of layers skipped.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_DEVICE_MAP = "mps"   # Llama 3B fits in 8GB MPS easily (6 GB bf16)


class LlamaHardSkipLayer(nn.Module):
    """Identity for Llama decoder layer: returns hidden_states unchanged.
    Llama decoder layer returns plain tensor (not tuple) → return tensor."""
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class LlamaSoftSkipLayer(nn.Module):
    """Soft skip: run orig layer + interpolate with input.
    output = α · orig(hidden_states, ...) + (1 - α) · hidden_states
    NO compute saving (executes orig). Preserves quality."""
    def __init__(self, orig: nn.Module, alpha: float):
        super().__init__()
        self.orig = orig
        assert 0.0 < alpha < 1.0
        self.alpha = float(alpha)

    def forward(self, hidden_states, *args, **kwargs):
        out = self.orig(hidden_states, *args, **kwargs)
        return self.alpha * out + (1.0 - self.alpha) * hidden_states


class LlamaSkipper:
    """Skipper for Llama 3.x: real compute saving via ModuleList swap."""

    def __init__(self, model_id: str = MODEL_ID,
                 dtype: torch.dtype = torch.bfloat16,
                 device_map: str = DEFAULT_DEVICE_MAP):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device_map
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Pad token (Llama tokenizer may not set it by default; use EOS as pad)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._layers = self.model.model.layers
        self._orig_modulelist = self._layers
        self._orig_layers = list(self._orig_modulelist)
        self.n_layers = len(self._orig_layers)
        self.hidden_size = self.model.config.hidden_size

    def _chatify(self, text: str) -> str:
        msgs = [{"role": "user", "content": text}]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def _build_layers(self, hard: set[int], soft: dict[int, float]) -> nn.ModuleList:
        new = []
        for i, orig in enumerate(self._orig_layers):
            if i in hard:
                new.append(LlamaHardSkipLayer())
            elif i in soft:
                new.append(LlamaSoftSkipLayer(orig, soft[i]))
            else:
                new.append(orig)
        return nn.ModuleList(new)

    def apply_skip(self, hard_skip: Optional[set[int]] = None,
                   soft_skip: Optional[dict[int, float]] = None) -> None:
        hard = set(hard_skip) if hard_skip else set()
        soft = dict(soft_skip) if soft_skip else {}
        for i in hard:
            assert 0 <= i < self.n_layers
        for i in soft:
            assert 0 <= i < self.n_layers
        assert not (hard & set(soft)), "layer in both hard/soft"
        self.model.model.layers = self._build_layers(hard, soft)

    def restore(self) -> None:
        self.model.model.layers = self._orig_modulelist

    def _prepare(self, prompt: str, chat_template: bool):
        text = self._chatify(prompt) if chat_template else prompt
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def forward_no_swap(self, prompt: str, chat_template: bool = True) -> torch.Tensor:
        inputs = self._prepare(prompt, chat_template)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits[0, -1, :].float().cpu()

    def forward(self, prompt: str,
                hard_skip: Optional[set[int]] = None,
                soft_skip: Optional[dict[int, float]] = None,
                chat_template: bool = True) -> torch.Tensor:
        """One-shot: apply skip + forward + auto-restore."""
        hard = set(hard_skip) if hard_skip else set()
        soft = dict(soft_skip) if soft_skip else {}
        if not hard and not soft:
            # FALLBACK: no intervention → bit-identical to HF baseline
            return self.forward_no_swap(prompt, chat_template)
        self.apply_skip(hard, soft)
        try:
            return self.forward_no_swap(prompt, chat_template)
        finally:
            self.restore()

    def embed(self, prompt: str, layer_idx: int = None,
              chat_template: bool = True) -> torch.Tensor:
        """Extract hidden state of an intermediate layer at last-token position.
        Used as router encoder. Default: layer = n_layers // 3 (early-mid).

        NOTE: this uses output_hidden_states=True on the full model — costo
        equivalente a un forward intero. Per encoder dedicato più veloce,
        usare un modello separato più piccolo (es. Llama 3.2 1B)."""
        if layer_idx is None:
            layer_idx = self.n_layers // 3
        inputs = self._prepare(prompt, chat_template)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # hidden_states[0] = embedding output, [i+1] = output of layer i
        h = outputs.hidden_states[layer_idx + 1]   # output of layer `layer_idx`
        return h[0, -1, :].float().cpu()
