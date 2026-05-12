# Adaptive Inference System (AIS)

> **Status: research prototype.** Honest write-up of an attempt to build category-aware
> layer skipping for small LLMs on Apple Silicon. **Not** competitive with state-of-the-art
> efficient inference (vLLM, speculative decoding, LayerSkip). Read this as an engineering
> case study, not as a library to deploy.

📖 **Full write-up: [ARTICLE.md](./ARTICLE.md)** — the story of what worked, what
didn't, and why I'm publishing the failures alongside the wins.

---

## What this is

Three-component inference pipeline for small instruction-tuned LLMs:

1. **Cervelletto** (router) — runs the prompt through the first ~⅓ of the model,
   extracts a hidden-state embedding.
2. **Mappa Topologica** (FAISS index, 5000 entries) — k-NN lookup over the embedding
   returns a category and a per-layer `importance` vector for that category.
3. **Cervellone** (decoder) — runs a full forward, but hard-skips (or soft-interpolates)
   the layers whose importance is below a threshold.

The idea: skip the layers that don't matter for the prompt's category, save compute
without losing quality.

## Headline numbers (Llama 3.2 3B, M4 Mac mini 16GB, bf16, MPS)

### Batch throughput (hard skip 50% of layers, persistent)

| Batch size | Baseline req/s | AIS req/s | Speedup |
|---|---|---|---|
| 1 | 36.6 | 49.8 | **1.36x** |
| 4 | 302 | 576 | **1.90x** |
| 8 | 621 | 1126 | **1.81x** |

### Quality trade-off on MMLU (N=100)

| Mode | Baseline acc | AIS acc | Top-1 agree | Latency saving |
|---|---|---|---|---|
| hard skip 25% | 53% | **25%** ❌ | 73% | +24% |
| soft α=0.7, skip set 25% | 53% | **51%** ✅ | 99% | +2% |

**The honest read:** hard skip ships real compute saving but breaks discriminative
tasks (MMLU collapses to "always answer A"). Soft skip preserves quality but
executes the layer, so the saving evaporates. There is no free lunch.

### Where it's actually useful

| Use case | Strategy | Verdict |
|---|---|---|
| Multi-user API, free-form chat | hard skip 50%, persistent | ✅ ~1.8x more users per machine |
| Single-user, accuracy-critical | — | ❌ Just use the base model |
| MMLU / classification / RAG | — | ❌ Quality loss too large |
| Apple Silicon edge serving | hard skip + batching | ✅ Real throughput win |

## What's in the box

```
adaptive-inference-system/
├── ARTICLE.md                 # long-form write-up (the interesting part)
├── README.md                  # this file
├── cervellone/
│   ├── llama_skipper.py       # LlamaSkipper — real hard-skip via ModuleList swap
│   ├── native_skip.py         # NativeLayerSkipper for Gemma 4 (research only)
│   └── layer_skipper.py       # AdaptiveLayerSkipper via nnsight (research only)
├── pipeline/
│   ├── mappa.py               # TopologicalMap (FAISS IndexFlatIP)
│   └── pipeline.py            # end-to-end AIS pipeline (Gemma 4 path)
├── experiments/               # 18 numbered experiments, reproducible
├── docs/
│   ├── pitfalls.md            # 16 documented gotchas — read this if you do similar work
│   ├── architecture.md        # design notes
│   └── phases.md              # roadmap and go/no-go criteria
├── results/
│   ├── LLAMA_RESULTS.md       # Llama-specific summary
│   ├── TABELLA_COMPARATIVA.md # cross-experiment comparison
│   └── *.npz                  # raw benchmark data
└── scripts/                   # model download helpers
```

## Quick start (reproduce the Llama numbers)

### Prerequisites

- macOS 14+ with Apple Silicon
- Conda / Miniconda
- ~10 GB free disk for Llama 3.2 3B

### Install

```bash
git clone https://github.com/vadale93/adaptive-inference-system.git
cd adaptive-inference-system

conda create -n ais python=3.12 -y
conda activate ais

pip install torch==2.11.0 transformers==5.8.0
pip install faiss-cpu numpy scipy scikit-learn umap-learn matplotlib
pip install datasets huggingface_hub
# Optional, only for Gemma 4 / nnsight research path:
pip install nnsight==0.7.0 nnterp==1.3.0 torchvision

python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### Reproduce the Llama pipeline (~25 min total)

```bash
# 1. Smoke test the skipper
caffeinate -i python -u experiments/exp_012_llama_native_smoke.py

# 2. Build the topological map (~15 min: 5000 forwards for embeddings)
caffeinate -i python -u experiments/exp_013_llama_corpus.py --n 5000
python experiments/exp_014_llama_mappa.py

# 3. Per-category layer importance (~30 s)
caffeinate -i python -u experiments/exp_015_llama_ablation.py

# 4. Validation: hard vs soft skip
caffeinate -i python -u experiments/exp_016_llama_validation.py --mode hard
caffeinate -i python -u experiments/exp_016_llama_validation.py --mode soft --alpha 0.7

# 5. MMLU (~3 min)
caffeinate -i python -u experiments/exp_017_llama_mmlu.py --n 100 --mode soft --alpha 0.7

# 6. Batch throughput (~2 min)
caffeinate -i python -u experiments/exp_018_llama_batch.py
```

### Minimal inference example

```python
from cervellone.llama_skipper import LlamaSkipper

skipper = LlamaSkipper()  # loads Llama 3.2 3B Instruct, bf16, MPS

# Baseline forward (bit-identical to HF native)
logits = skipper.forward("What is the capital of France?")

# AIS hard skip — 14/28 layers bypassed
logits_ais = skipper.forward(
    "What is the capital of France?",
    hard_skip=set(range(7, 14)) | set(range(21, 28)),
)

# Soft interpolation — execute layer + blend (no compute saving)
logits_soft = skipper.forward(
    "What is the capital of France?",
    soft_skip={i: 0.7 for i in range(7, 14)},  # α=0.7
)
```

## Why this isn't a production library

In good faith, here are the techniques you should look at before AIS for any
real workload:

- **[vLLM](https://github.com/vllm-project/vllm)** — PagedAttention + continuous
  batching. ~10–20x throughput improvement, zero quality loss. Industry standard.
- **[LayerSkip](https://arxiv.org/abs/2404.16710)** (Meta, 2024) — Per-token layer
  skipping with LoRA fine-tuning that preserves quality. Same speedup target, much
  better quality preservation than AIS's static category routing.
- **Speculative decoding** — Medusa, EAGLE, look-ahead. 2-3x speedup, lossless.
- **Quantization** — GGUF Q4/Q5, AWQ, GPTQ. 3-4x speedup with marginal quality loss.
- **[Mixture of Depths](https://arxiv.org/abs/2404.02258)** (DeepMind, 2024) —
  Learned per-token router. The right way to do adaptive depth.

AIS's static category-based routing is structurally inferior to learned routing.
This repo is published as a documented engineering exercise, not as a tool to
adopt.

## What's actually worth reading

If you only have 5 minutes:

1. **[ARTICLE.md](./ARTICLE.md)** — the long-form story, including the gotchas
   that would have saved me weeks if I'd known about them.
2. **[docs/pitfalls.md](./docs/pitfalls.md)** — 16 numbered traps (the Gemma 4
   shared-KV breakage at P15-P16 is the one I most wish I'd known about).
3. **[results/LLAMA_RESULTS.md](./results/LLAMA_RESULTS.md)** — concrete numbers
   from the Llama 3.2 3B path.

## License

MIT. The Llama 3.2 3B and Gemma 4 weights used in experiments are subject to
their respective Meta and Google licenses; check those before deploying.

## Citing

```
@misc{vadala2026ais,
  title  = {Adaptive Inference System: A Negative-Result Study of
            Category-Routed Layer Skipping for Small LLMs on Apple Silicon},
  author = {Vadala, Alessandro},
  year   = {2026},
  url    = {https://github.com/vadale93/adaptive-inference-system},
  note   = {Research prototype. Outperformed by LayerSkip and Mixture of Depths.},
}
```
