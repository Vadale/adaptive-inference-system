# AIS — Llama 3.2 3B Results

Validation real compute saving via NativeLayerSkipper (LlamaSkipper, ModuleList swap).

## Setup
- Model: `unsloth/Llama-3.2-3B-Instruct`
- Hardware: M4 16GB, MPS bf16
- Dataset: dolly-15k (5000 stratified), MMLU all/test (N=100)
- No shared-KV pattern → hard skip is real compute saving

## Phase 1 — Topological Map (cervelletto + FAISS)

| Metric | Llama 3.2 3B | Gemma 4 E2B |
|---|---|---|
| pivot layer | L9 | L9 |
| hidden dim | 3072 | 1536 |
| k-NN homog (k=10) | **0.5563** | 0.4485 |
| silhouette (cos) | 0.0143 | 0.0244 |
| centroid recall ≥7/10 | 5/8 | 5/8 |
| verdict (primary OR alt) | **PASS** | PASS |

Llama embedding **+24% better** at category clustering than Gemma E2B.

## Phase 2 — Layer Importance (group ablation, 4 groups of 7)

**Universal pattern across all 8 dolly categories:**
- G0 (L00-06): ALWAYS critical (avg KL 7.3-10.5) — never skip
- G1 (L07-13): ALWAYS least critical (avg KL 1.3-4.6) — universal safe-skip zone
- G2 (L14-20): variable per category
- G3 (L21-27): variable; critical for `creative_writing`, `summarization`

## Phase 3 — Validation (held-out, k=3/cat)

Hard skip G1 (7/28 layer = 25%):
| Mode | Cat PASS | top1 agree mean | KL mean | Verdict |
|---|---|---|---|---|
| hard | 0/8 | ~0.40 | ~2.4 | FAIL |
| soft α=0.7 | 5/8 | ~0.85 | ~0.12 | **PASS** |

Soft α=0.7 preserves output but executes the layer → no compute saving.
Hard skip degrades top-1 on free-form generation.

## Phase 4 — MMLU N=100

| Mode | Baseline acc | AIS acc | Δ | Latency saving | Top-1 agree |
|---|---|---|---|---|---|
| hard skip 25% | 53.0% | **25.0%** | **-28pp** | +23.5% | 73% |
| soft α=0.7 25% | 53.0% | **51.0%** | -2pp | +2.4% | **99%** |

MMLU is much more discriminative-sensitive than free-form: hard skip collapses
to "always A". Soft α=0.7 preserves accuracy but eats the latency saving.

## Phase 5 — Batch Throughput (hard skip 50%, persistent)

| B | base req/s | AIS req/s | saving | speedup |
|---|---|---|---|---|
| 1 | 36.6 | 49.8 | +26.5% | 1.36x |
| 2 | 60.5 | 90.1 | +32.9% | 1.49x |
| 4 | 302 | **576** | +47.5% | **1.90x** |
| 8 | 621 | **1126** | +44.8% | 1.81x |

**This is the headline number**: at B=4, AIS-Llama nearly doubles throughput
(1.90x), serving 576 req/s on a single M4 16GB.

## Tradeoff matrix

| Use case | Skip strategy | Latency saving | Quality |
|---|---|---|---|
| Discriminative (MMLU, classification) | soft α=0.7 | ~0% | preserved (-2pp) |
| Free-form (chat, brainstorm) | hard 25-50% | +24-45% | top-1 varies |
| Batch serving (multi-user API) | hard 50% persistent | +45% | per-prompt routing |

## Comparison to Gemma 4 work

| Aspect | Gemma 4 E2B/E4B | Llama 3.2 3B |
|---|---|---|
| Native hard skip works | ❌ (shared KV, KeyError sliding_attention) | ✅ |
| nnsight overhead | Yes (slow validation) | No (real compute) |
| Best concrete result | Soft α=0.7 PASS 4/8 cat | Batch B=4 → 1.90x speedup |
| Phase 1 cluster quality | 0.4485 homog | 0.5563 homog (+24%) |

## Files
- corpus: `corpus/activations_llama32_3b_n5000_L9_last.npz`
- mappa: `mappa/topology_llama32_3b/` (61.4 MB)
- ablation: `results/layer_importance_llama32_k3_g4.npz`
- validation: `results/llama_validation_k3_kskip1_{hard,soft_a0.7}.npz`
- MMLU: `results/mmlu_llama32_n100_{hard,soft_a0.7}.npz`
- batch: `results/batch_throughput_llama32.txt`
- experiments: `experiments/exp_013_..._018.py`
