# I built adaptive layer skipping on an M4 Mac — here's what actually worked, what didn't, and why I'm publishing the failures

**TL;DR.** I spent several weeks building "AIS": a 3-component inference pipeline
that routes prompts through a topological map and skips the layers it thinks
aren't important for that category. On a Mac mini M4 with Llama 3.2 3B, it
delivers a real **1.9x batch throughput speedup at B=4** for free-form
generation. It also **collapses on MMLU** (53% → 25% accuracy when you actually
save compute). Existing techniques — vLLM, LayerSkip, speculative decoding —
do this better. I'm publishing the repo and this write-up because (a) the
gotchas I hit are worth saving someone else from, and (b) honest negative
results are undervalued. If you're considering similar work, read this before
you start.

---

## 1. Motivation

Small LLMs (1B–7B) on edge hardware are everywhere now: Ollama, llama.cpp, MLX,
LM Studio. The promise is "your laptop runs ChatGPT-3.5 quality offline." The
reality is "your laptop runs it slowly, especially for multi-user serving."

I wanted to test a specific hypothesis: **for many "easy" prompts, you don't
need every layer of the model.** If a prompt is asking "what's the capital of
France," do you really need all 28 transformer layers? Or can you skip 7 of
them and get the same answer 25% faster?

Adaptive depth isn't new — see LayerSkip (Meta, 2024), Mixture of Depths
(DeepMind, 2024), early-exit BERT variants. But those approaches mostly require
fine-tuning the base model. **Can you get adaptive depth on a frozen,
off-the-shelf instruction-tuned model?** That was the question.

## 2. The architecture I built ("AIS")

Three components:

```
prompt
  │
  ▼
┌─────────────────────────────┐
│   ROUTER                    │   first 1/3 of the model → embedding
│   Llama 3.2 3B, layers 0-9  │
└─────────────┬───────────────┘
              │ embedding (hidden_size = 3072)
              ▼
┌─────────────────────────────┐
│   TOPOLOGICAL MAP           │   FAISS k-NN over 5000 corpus embeddings
│   IndexFlatIP (cosine)      │   → category + layer_importance vector
└─────────────┬───────────────┘
              │ category, [importance_L0, ..., importance_L27]
              ▼
┌─────────────────────────────┐
│   DECODER                   │   full forward, but skip layers
│   Llama 3.2 3B, layers 0-27 │   with importance < threshold
└─────────────────────────────┘
              │
              ▼
            logits
```

The map was bootstrapped from `databricks/dolly-15k` — 5000 stratified prompts
across 8 categories (`brainstorming`, `classification`, `closed_qa`,
`creative_writing`, `general_qa`, `information_extraction`, `open_qa`,
`summarization`). For each category, I ran a group-ablation experiment: skip
groups of 7 consecutive layers, measure KL divergence vs the full-model
baseline, and call those groups "important" if KL is high.

The intuition: different prompt categories should use different layers. The
router decides which category the prompt belongs to; the importance vector
tells the decoder which layers it can drop.

## 3. What actually happened (the timeline)

### Phase 0: Causal tracing on Gemma 4 E2B

I picked **Gemma 4 E2B** as the starting model because it was the smallest
recent open-weight instruction-tuned model that ran on an M4 16GB Mac. First I
did ROME-style causal tracing to confirm that subject information lives in
identifiable layers (it does, at L09-L13 for capital-city facts — recovery
score 2.04). PASS.

### Phase 1: Building the topological map

Extract a hidden-state embedding for each of 5000 dolly prompts via
`nnsight.VisionLanguageModel(...)`. Pivot layer L09 of Gemma 4 E2B,
last-token pool. Build a `faiss.IndexFlatIP` over L2-normalized embeddings
(= cosine similarity).

**Result:** k-NN (k=10) homogeneity = 0.4485 (4.4x above the random baseline of
1/8 = 0.125 for an 8-class problem). PASS, with caveats: silhouette score is
0.0244 — modest, but text embeddings on overlapping task categories tend to
have low silhouette even when clusters are visible.

### Phase 2: Group ablation for layer importance

For each of the 8 dolly categories, take 3 representative prompts, run
6 group-ablation forwards (skip group g in [0..5]), measure KL(baseline ‖ skip).

**This took 3+ hours on Gemma 4 E4B** because I was using `nnsight`'s
intervention API for the skip, which adds significant overhead. A red flag I
should have caught earlier — see §6.

Result: per-category layer importance vectors, written into the FAISS map.

### Phase 3: Validation — soft skip works, hard skip breaks

I tested two skip modes:

1. **Hard skip:** replace `layers[i]` with an identity module
   (`output = input`). Real compute saving — the layer's matmul is never
   executed.
2. **Soft skip:** still execute the layer, but interpolate with the input:
   `output = α · layer(input) + (1-α) · input`. No compute saving, but quality
   degrades gracefully with α.

On Gemma 4 E2B with hard skip 33%, top-1 accuracy collapsed. With soft skip
α=0.7, 4/8 dolly categories passed the ≥95% top-1 agreement threshold. **Soft
skip preserved quality, hard skip destroyed it.**

This is the classic tension. Soft skip is research-only because it doesn't save
real compute. Hard skip saves compute but breaks the model. The map's job is
supposed to be: identify which layers are safe to hard-skip for which prompts.
It didn't work cleanly on Gemma 4.

### Phase 4: The "62% latency saving" that wasn't

I ran a benchmark suite (MMLU, MMLU Pro, ARC, HellaSwag, MMMLU IT) and saw
*latency reductions of 54-71%* with soft skip α=0.7. **For about a day I
thought I had production-ready results.**

Then a user pushed back: "If it saves 62% latency, does that mean less RAM use?
Because if not, this isn't interesting." I went back to look. The "saving" was
coming from removing `nnsight` interventions from the path. The baseline
forwards I was comparing against were going through `nnsight.trace(...)` too,
adding constant overhead. **The 62% wasn't compute saving. It was overhead
saving — the kind that disappears the moment you ship to a real serving
framework that doesn't use nnsight.**

This was the most important methodological lesson of the project: **measure
compute saving against a clean HuggingFace forward, not against your
instrumentation-laden baseline.** I rewrote `NativeLayerSkipper` to use pure
HF + a `ModuleList` swap, ran the bench again, and the saving was much smaller
— or even negative on Gemma 4 E4B with `accelerate` offload.

### Phase 5: The Gemma 4 trap — shared KV pattern

When I tried hard-skip on Gemma 4 with the clean `NativeLayerSkipper`, it
crashed with `KeyError: 'sliding_attention'`. The reason: **Gemma 4 has a
shared-KV-state pattern across layers.** Some layers reuse KV cache from earlier
layers, identified by an internal "layer type" registry. If you swap a layer
with `Identity`, the downstream layer can't find its expected KV state and
crashes.

There are workarounds (clone the KV state, patch the layer registry), but they
were complex and fragile. So I made a model swap.

### Phase 6: The Llama 3.2 3B rescue

**Llama 3.2 3B has a standard transformer architecture** — no shared KV, every
layer is self-contained. The same `ModuleList`-swap trick that failed on Gemma
4 works out of the box.

I ported the pipeline:
- `LlamaSkipper` class (≈100 LOC): loads HF model, exposes `apply_skip`,
  `restore`, `forward(hard_skip=..., soft_skip=...)`.
- Re-ran the corpus build (5000 prompts, 15 min): k-NN homogeneity **0.5563**
  — 24% better than Gemma 4 E2B (0.4485). Llama embeddings cluster categories
  more cleanly.
- Re-ran the ablation (4 groups of 7, **20 seconds total** — vs 3 hours on
  Gemma): G1 (layers 7-13) emerged as a **universal safe-skip zone** across
  all 8 categories. G0 (layers 0-6) is always critical. G2/G3 vary by category.
- Validation: hard-skip 25% still failed strict top-1 agreement (0/8
  categories). Soft α=0.7 passed 5/8.

## 4. The headline result: batch throughput

Here's the one number that genuinely held up:

| Batch size | Baseline req/s | AIS req/s (hard skip 50%) | Speedup |
|---|---|---|---|
| 1 | 36.6 | 49.8 | **1.36x** |
| 2 | 60.5 | 90.1 | **1.49x** |
| 4 | 302 | **576** | **1.90x** |
| 8 | 621 | **1126** | **1.81x** |

Single forward (one input token, one output token), batch the prompts together,
median over 3 runs after 2 warmups, persistent `apply_skip(hard_skip=...)`.

At B=4, throughput nearly doubles. On an M4 Mac mini, that's the difference
between serving 300 req/s and 576 req/s with the same hardware. **For a chat or
autocomplete API where you don't need MMLU-level discrimination, this is a real
deployment win.**

But:

## 5. The MMLU collapse

Same Llama 3.2 3B. Same setup. N=100 MMLU questions, multi-subject. Skip layers
chosen per-prompt by the AIS router (based on category match in the map).

| Mode | Baseline acc | AIS acc | Δ | Latency saving | Top-1 agree |
|---|---|---|---|---|---|
| hard skip, ~25% layers | 53.0% | **25.0%** | **−28pp** | +23.5% | 73% |
| soft α=0.7, ~25% layers | 53.0% | **51.0%** | −2pp | +2.4% | **99%** |

Hard skip 25% **collapses MMLU accuracy to 25%** — equivalent to random
guessing for a 4-choice question. The model loses its ability to discriminate
between A/B/C/D and defaults to "always answer A" (literally: token argmax goes
to letter ID 362 = " A").

Soft α=0.7 preserves accuracy almost perfectly (−2pp, 99% top-1 agreement
with the baseline). But because soft skip still executes the layer, the
compute saving is essentially gone (+2.4% only — basically noise).

**This is the honest tension at the core of AIS:**

- Hard skip → real saving, but discriminative tasks die.
- Soft skip → quality preserved, but the saving is theatrical.

For free-form generation (chat, brainstorming, autocomplete), the model
"recovers" — the missing layers' contribution is smeared out across the
remaining ones, and the output is still coherent. For tightly constrained
output spaces (multi-choice), there's no recovery, and the missing layers'
exact contribution matters.

## 6. Why this loses to state of the art

I have to be direct: **AIS is structurally inferior to existing techniques.**

| Technique | Speedup | Quality loss | Status |
|---|---|---|---|
| vLLM (PagedAttention + continuous batching) | 10–20x throughput | 0 | production |
| Speculative decoding (Medusa, EAGLE) | 2–3x | 0 (lossless) | production |
| Quantization (GGUF Q4/Q5, AWQ) | 3–4x | −1 to −2 pp | production |
| LayerSkip (Meta, 2024) | ~2x | minimal with LoRA tune | research → prod |
| Mixture of Depths (DeepMind, 2024) | ~2x | preserved by learned router | research |
| **AIS** (this work) | 1.8x batch (free-form only) | −28 pp on MMLU | research |

The closest comparable is LayerSkip: same speedup target, same idea ("skip
layers adaptively"), but with a **learned** router and a small LoRA fine-tune
that re-aligns the model to the skip pattern. The result is dramatically better
quality preservation than AIS's static category-based routing.

The deeper issue: **AIS's "topological map" is a hand-crafted, static
classifier on top of hidden states.** Modern adaptive depth methods use
parametric, end-to-end-trained routers (mixture-of-depths style) that can adapt
per-token, not per-prompt. A 5000-entry FAISS k-NN can't compete with a 2-layer
MLP trained jointly with the model.

The router also has a hidden cost I underestimated: **to route a prompt,
you have to do a partial forward through the model first.** When the router
is the same size as the decoder (as in my Llama setup, where both are
Llama 3.2 3B), that's expensive enough that the skip needs to save more
than the routing cost. This works at B=4 because the routing happens once
per batch and the savings amortize. It does *not* work for single-user
latency.

## 7. Things I learned that might save you time

If you're doing similar work, these are the gotchas I wish I'd known about
before starting. They're all numbered and documented in
[`docs/pitfalls.md`](./docs/pitfalls.md) in the repo.

**P3 / P11. nnsight has serious gotchas.** `with model.trace(...)` rewrites
your block AST. Initializing lists *outside* and appending *inside* works; the
opposite doesn't. Also `.item()` on a meta tensor crashes unless you do a dummy
trace first to materialize weights. For multimodal models like Gemma 4, use
`VisionLanguageModel`, not `NNsightModel`, and you need `lm_head.output`,
not `model.logits`.

**P12. `device_map="auto"` with `accelerate` quietly disables compute saving.**
If a layer ends up on CPU offload, hard-skipping it doesn't save you anything
— the GPU memory was never allocated, and the "saving" is dominated by
host-device transfer. Always check `model.hf_device_map` before claiming a
speedup. On 16GB M4, single-MPS deployment only works up to ~3B parameters in
bf16.

**P13. macOS sleep kills long nnsight traces.** If you have a multi-hour
ablation running, the laptop hitting standby breaks the trace mid-flight and
nnsight gives you a `MissedProviderError`. Always run with `caffeinate -i
python -u ...`.

**P15-P16. Gemma 4 has shared KV across layers** (the `sliding_attention`
type registry). Naive `nn.Identity()` swap breaks the model. If you want to
do hard-skip experiments, **pick a standard-architecture model first** — Llama
3.2, Mistral, Phi — and only then deal with shared-KV variants if you must.

**Methodological: "compute saving" is the wrong metric. Use req/s.** If your
"saving" comes from removing instrumentation overhead, you'll discover this
the moment you compare against a clean baseline. I lost about a day of
optimism to this before catching it. Measure end-to-end throughput against a
plain `model(**inputs)` call, not against your wrapper.

**Transformers 5.x: `torch_dtype=` is deprecated, use `dtype=`.** Easy fix but
silent confusion if you have old docs open.

## 8. What I'd do differently

If I were starting over with the same end goal ("adaptive depth on a frozen
small LLM, Apple Silicon"):

1. **Read LayerSkip and Mixture of Depths first.** I didn't. The week I spent
   on Gemma 4 nnsight interventions was reinventing a technique that already
   has cleaner solutions in the literature.

2. **Start with a standard-arch model** (Llama, Mistral). Don't fight Gemma 4's
   shared KV unless you have a specific reason to use Gemma 4.

3. **Don't use the same model as router and decoder.** The break-even on
   routing cost is brutal when the router is the decoder. Either
   route with a much smaller model (Llama 3.2 1B → Llama 3.2 3B/8B), or skip
   the router entirely and use a fixed skip set with batching.

4. **Train a small router instead of using FAISS k-NN.** Even a 2-layer MLP on
   the same hidden states would dominate static k-NN on classification, with
   no inference overhead.

5. **Or: don't.** vLLM + quantization gets you 80% of what you'd want from
   AIS, with zero quality loss, in a production-grade serving framework. The
   correct adaptive-depth solution probably needs a small fine-tune (à la
   LayerSkip's self-speculative LoRA).

## 9. What's in the repo

Everything to reproduce these numbers, plus the full failure trail:

- **`experiments/exp_012_..._018_*.py`** — the Llama pipeline (smoke, corpus,
  map, ablation, validation, MMLU, batch throughput).
- **`experiments/exp_000_..._011_*.py`** — the Gemma 4 path. Mostly negative
  results, but the methodology (ROME tracing, group ablation, soft-skip
  validation, multi-benchmark suite) is reusable.
- **`skippers/llama_skipper.py`** — the minimal real-compute-saving skipper.
  Read this first.
- **`skippers/native_skip.py`** — the Gemma 4 version. Read this second to
  see how shared-KV breaks the naive approach.
- **`skippers/layer_skipper.py`** — the nnsight-based research version with
  α-interpolation. Read this if you want to do boundary intervention
  experiments without a fine-tune.
- **`pipeline/topological_map.py`** — `TopologicalMap` (FAISS IndexFlatIP
  + per-entry `layer_importance` array). 50 lines, no magic.
- **`docs/pitfalls.md`** — the 16 numbered gotchas.
- **`docs/phases.md`** — the project roadmap, with go/no-go criteria and what
  passed/failed at each gate.
- **`results/LLAMA_RESULTS.md`** — the Llama-only summary table.

## 10. Closing

I'm publishing this because I think the AI/ML community undervalues honest
negative results. The next person trying to do "adaptive layer skipping on a
Mac with a frozen model" can read this, save a few weeks, and either pick a
better technique or push past where I stopped. That's a fine outcome for a
research prototype.

If you find this useful — or you spot something I got wrong — open an issue,
I'd genuinely like to hear it.

— Alessandro Vadala, 2026

---

**Repo:** https://github.com/Vadale/adaptive-inference-system
**Related work worth reading:** [LayerSkip](https://arxiv.org/abs/2404.16710),
[Mixture of Depths](https://arxiv.org/abs/2404.02258),
[vLLM](https://github.com/vllm-project/vllm),
[CALM](https://arxiv.org/abs/2207.07061).
