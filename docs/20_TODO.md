# 20 — TODO & Task Tracker

> **Prerequisites:** [`19_GLOSSARY.md`](19_GLOSSARY.md).
>
> **Next:** [`21_DEVLOG.md`](21_DEVLOG.md)

---

## Purpose

This document tracks active development tasks, milestone goals, and architecture review action items across the JimmyLabs project. It ensures every folder earns its keep and all implementation milestones are completed systematically.

---

## Architecture Review Action Items (v0.1 Milestone)

- [x] **R1:** Write `SPEC.md` with explicit parameter and memory arithmetic.
- [x] **R2:** Establish weight tying as v0.1 default baseline (`ADR-0003`).
- [x] **R3:** Folder audit — verify every top-level folder holds active artifacts by v0.1.
- [x] **R5:** Write evaluation contract into `SPEC.md`.
- [x] **ADR-0002:** Document Pre-Layer Normalization in `research/design_decisions/ADR-0002-pre-norm.md`.
- [x] **ADR-0003:** Document Weight Tying in `research/design_decisions/ADR-0003-weight-tying-default.md`.

---

## Phase Milestones & Core Documentation

### Phase 0 & 1 — Foundations & Tokenization
- [x] Write `docs/04_MATHEMATICS.md` (vectors, matmul, derivatives, softmax).
- [x] Write `docs/05_NEURAL_NETWORKS.md` (neuron, MLP, backprop, overfit-a-batch).
- [x] Write `docs/06_TOKENIZER.md` (character tokenizer, BPE, round-trip contract).
- [x] Write `docs/07_EMBEDDINGS.md` (token + positional embeddings, weight tying intro).

### Phase 2 — TinyGPT Model Architecture
- [x] Write `docs/08_ATTENTION.md` (scaled dot-product attention, multi-head, causal mask).
- [x] Write `docs/09_TRANSFORMER.md` (pre-norm transformer block).
- [x] Write `docs/10_GPT_ARCHITECTURE.md` (full decoder-only model assembly).

### Phase 3 & 4 — Training & Optimization
- [x] Write `docs/11_TRAINING_PIPELINE.md` (loss, AdamW, LR schedules, grad clip).
- [x] Write `docs/12_INFERENCE.md` (autoregressive loop, sampling knobs, KV cache concept).
- [x] Write `docs/15_EXPERIMENT_GUIDE.md` (reproducible scientific method).
- [x] Write `docs/16_MODEL_CONFIGURATION.md` & `configs/model_v0_1_char_100k.yaml`.

---

## Code Implementation — Phases 0–3 (DONE ✓)

- [x] **Phase 0** — autograd toy + XOR + MNIST (`src/jimmylabs/autograd/`, `scripts/train_mnist.py`).
- [x] **Phase 1** — character tokenizer + data loader (`src/jimmylabs/tokenizer/`, `data/`).
- [x] **Phase 2** — GPT model (`src/jimmylabs/model/{config,embedding,attention,feedforward,block,gpt}.py`); param count == 818,048 (SPEC §5).
- [x] **Phase 3** — training loop, LR schedule, checkpointing, generation (`src/jimmylabs/training/`, `inference/`; `scripts/{prepare_data,train,generate}.py`).
- [x] Test suite: **66 tests** (shape · known-answer · gradient · overfit-a-batch · regression); verified with `./.venv/bin/python -m pytest -q`; CI in `.github/workflows/tests.yml`.
- [x] **v0.1 milestone** — first coherent Shakespeare, val loss 1.54 (`outputs/trained_shakespeare_sample.txt`).

---

## Phase 4 — Optimization (DONE ✓)

- [x] **Baseline benchmark** — `benchmarks/001_baseline.md` (train 65K tok/s, gen 103 tok/s, 258 MB, 10.1 MB ckpt).
- [x] **KV cache** (`OPTIMIZATION_BACKLOG` #2) — `benchmarks/002_kv_cache.md`, gen +86% on the corrected within-`block_size` path. Correctness gated (`test_kv_cache`: cached == naive).
- [x] **Gradient accumulation** (#3) — `benchmarks/003_grad_accum.md`, eff. batch 64 at +67 MB. Gated (`test_grad_accum`: matches a 4× batch).
- [x] **bf16** mixed-precision experiment (#6) — `research/experiments/001_bf16_mixed_precision.md`. **Rejected**: 15–31% slower on M1 (no native bf16 matmul hardware). Correct, honest negative result.
- [x] **Attention visualizer** — `scripts/visualize_attention.py`, reuses the stashed `_attn_weights`.

## Phase 5 — Scale & the Primary Corpus (v0.2 complete)

v0.1 proved the pipeline on Shakespeare (the *sanity* corpus). Per
[`docs/17_DATASET_GUIDE.md`](17_DATASET_GUIDE.md), TinyStories — not Shakespeare — is the
**primary** corpus: its simple English is the actual capacity match for a 1–4M-param model,
and is where genuinely coherent generation is most likely.

- [x] **TinyStories provenance + prep** — `datasets/SOURCES.md`, `scripts/prepare_tinystories.py`.
- [x] **v0.2 config and milestone** (~2.7M: L6 h6 C192 block256) — trained on TinyStories,
      generated, and compared qualitatively to v0.1's Shakespeare sample; see
      [`benchmarks/004_v0_2.md`](../benchmarks/004_v0_2.md).
- [x] Remaining cheap backlog audit: **#4** and **#5** already satisfied by construction;
      **#7** memory-mapped dataset applied; **#8** deferred because the current vocabulary
      is small and profiling does not justify replacing the clear `torch.sort` path.
- [x] Benchmark v0.2 absolute results recorded in `benchmarks/004_v0_2.md`; it is not a
      controlled v0.1 speed delta because model and dataset both changed.

---

> **Next:** [`21_DEVLOG.md`](21_DEVLOG.md)
