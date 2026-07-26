# 21 — Devlog

> **Prerequisites:** [`20_TODO.md`](20_TODO.md).
>
> **Next:** [`22_FUTURE_VERSIONS.md`](22_FUTURE_VERSIONS.md)

---

## Purpose

The Devlog is a chronological record of major project milestones, architectural decisions, and lab notes. It records *how* JimmyLabs evolved from initial design scaffolds to a working 1–4M parameter model.

---

## Chronological Entries

### 2026-07-26 — v0.2 milestone: first TinyStories training run 🎉
- **Milestone:** the first ~2.7M-param model (L=6, h=6, C=192, block=256, vocab=138)
  trained to completion on TinyStories, the project's *primary* corpus. Validation loss
  fell from `ln(138)≈4.93` to **0.8607** (perplexity ≈2.37) at the best checkpoint
  (step 2800/3375). See [`benchmarks/004_v0_2.md`](../benchmarks/004_v0_2.md) for the full
  record and [`outputs/trained_tinystories_v0_2_sample.txt`](../outputs/trained_tinystories_v0_2_sample.txt)
  for the sample — noticeably more fluent than v0.1's Shakespeare sample (real names,
  coherent short sentences, correct dialogue punctuation), matching
  [`docs/17`](17_DATASET_GUIDE.md)'s thesis that TinyStories is the better capacity match
  for a model this size.
- **Laptop-safety correction, applied before running:** the config originally called for
  `batch_size=64`, which visibly strained the machine on a prior attempt (attention
  activation memory scales with `batch_size × block_size²`, and `block_size=256` is 2× v0.1's).
  Lowered to `batch_size=16` with `grad_accum_steps=4` (same effective batch, same
  ~55.3M-token Chinchilla budget) before this run. Even so, the actual run took ~16 hours
  wall-clock — the machine was under real memory pressure from causes independent of this
  process (confirmed: the training process itself used ~60 MB RSS the whole time, while
  system-wide swap reached ~94% used). The wall-clock time is not representative of a
  clean-machine run and is reported as such.
- **A second real bug found and fixed while banking this milestone:** `scripts/generate.py`
  had never been run against a non-v0.1-shaped checkpoint. It built a throwaway model
  hardcoded to v0.1's exact shape just to read the checkpoint's stored config — but
  `load_checkpoint()` always applies the state dict strictly, so this only ever worked by
  coincidence when the checkpoint happened to already be v0.1-shaped. The first real v0.2
  checkpoint hard-crashed with a shape mismatch. Fixed by reading the config via a raw
  `torch.load(...)['config']` first (no model needed for that step), then building the
  correctly-shaped model once. A subprocess-level smoke test
  (`tests/test_generate_script_smoke.py`) now runs the real script against a deliberately
  non-v0.1-shaped checkpoint — closing a gap where no test had ever executed `generate.py`
  end-to-end before this.
- **Recurring lesson, restated:** this is the second script-level bug this project has
  shipped where every unit test passed because nothing actually *ran* the script
  end-to-end (the first was `train.py`'s `lr_max`/`contextlib` bugs). The standing rule —
  every script gets a subprocess-level smoke test, not just unit tests of its importable
  functions — earns its keep again here.

---

### 2026-07-26 — Correction: KV cache correctness past `block_size` + fused-AdamW dead check
- **Finding:** the "+65–81%" KV-cache speedup banked on 2026-07-25 was measured on a run
  (`--gen_tokens 200`, `block_size=128`) where 72 of 200 steps executed a code path that
  silently diverged from the naive (correct) output. Root cause: this model uses learned
  **absolute** positional embeddings; a cached K/V is frozen at the position it had when
  first embedded, so trimming the cache to slide the window desyncs every retained entry's
  position from what the continuing generation assumes. `torch.equal` at `temp=0` never
  caught it — greedy argmax barely moves for a ~1e-2 logit shift, and the existing
  equivalence test's `max_new_tokens` happened not to expose a visible difference.
- **How it surfaced:** found on an independent branch (not part of a dispatched task),
  reproduced and verified here at the **logit level** before trusting it — bit-identical
  through `block_size`, diverging from that step onward on unmodified `main`. The fix
  (stop reusing the cache once context ≥ `block_size`, fall back to the naive path — see
  [`ADR-0004`](../research/design_decisions/ADR-0004-no-cache-reuse-past-block-size.md))
  was verified independently to reduce the divergence to `6e-8` (float noise).
- **Blast radius, checked:** [`outputs/trained_shakespeare_sample.txt`](../outputs/trained_shakespeare_sample.txt)
  is **unaffected** — `scripts/generate.py` never sets `use_cache`. Only
  [`benchmarks/002_kv_cache.md`](../benchmarks/002_kv_cache.md)'s headline number was
  measured against the bug; it now carries a correction (§7): cache is still **+86%** over
  naive on the fixed code, at a **~3%** cost of correctness relative to the buggy version.
- **Also landed:** a second, independent fix — `hasattr(torch.optim.AdamW, 'fused')`
  checked for a class attribute, but `fused` is a constructor keyword, so the check was
  `False` on every torch build and the fused-AdamW path had never actually run. Fixed via
  `inspect.signature(...)`. Benchmarked honestly: **no measurable win** at v0.1's 818K-param
  scale (`benchmarks/005_fused_adamw.md`) — correctly reasoned as expected (optimizer step
  is negligible next to fwd/bwd this small), landed anyway since a dead check is worth
  fixing regardless, and it engages automatically once the model grows.
- **Lesson:** an equivalence test with too weak a comparison (final sampled tokens under
  greedy argmax) can pass while the underlying computation is wrong. Prefer comparing raw
  logits/hidden states directly when verifying "path A == path B" claims, not just the
  argmax of their outputs — this is now `OPTIMIZATION_BACKLOG.md` item #16's cross-reference
  and worth remembering for any future equivalence gate.

---

### 2026-07-25 — Phases 0–3 Code + First Coherent Text (v0.1 milestone) 🎉
- **Milestone:** JimmyLabs trains end-to-end and generates recognizable Shakespeare.
  Validation loss fell **4.174 → 1.54** over 5000 steps on the M1 (MPS).
- **Code shipped:** autograd toy + XOR + MNIST (Phase 0); character tokenizer + data
  loader (Phase 1); the GPT model — `config`, `embedding`, `attention`, `feedforward`,
  `block`, `gpt` (Phase 2); training loop, warmup→cosine LR schedule, checkpointing, and
  autoregressive generation (Phase 3). **39 tests green at that milestone**; CI added
  (`.github/workflows/tests.yml`).
- **Result:** first trained sample in
  [`outputs/trained_shakespeare_sample.txt`](../outputs/trained_shakespeare_sample.txt) —
  real character names and the `NAME:` play format — versus the pre-training garbage in
  `untrained_baseline.txt`. Exactly the expected quality for a 0.82M char-GPT.
- **Postmortem — the flat-loss bug:** the *first* 5000-step run sat at loss = ln(65) =
  4.174 the whole way; the model learned nothing. Root cause: `train.py` called
  `get_batch('train', …)` — passing a **string** where a data tensor belongs — and a
  `try/except` **silently trained on random tokens** instead of failing. Fix: correct the
  call + **delete the silent fallback** (missing data now errors loudly) + a regression
  test (`test_train_integration.py`).
  **Lesson (now a standing rule): never catch-and-substitute data/inputs — fail loudly.**
  A silent fallback turned a one-line typo into a 5000-step waste, and the end-to-end run
  caught what all 37 unit tests structurally could not.

---

### 2026-07-24 — Phase 0–4 Core Documentation Completion
- **Milestone:** Completed foundational, model architecture, training, inference, and experiment documentation across `docs/04` through `docs/16`.
- **Key Deliverables:**
  - `docs/04_MATHEMATICS.md`: Established canonical definitions for matrix multiplication and softmax.
  - `docs/05_NEURAL_NETWORKS.md`: Documented the overfit-a-batch golden rule for ML debugging.
  - `docs/06_TOKENIZER.md` & `docs/07_EMBEDDINGS.md`: Tokenization contracts, positional embeddings, and weight tying concepts.
  - `docs/10_GPT_ARCHITECTURE.md`: Decoder-only model assembly matching `SPEC.md` §5 param math.
  - `docs/11_TRAINING_PIPELINE.md` & `docs/12_INFERENCE.md`: Full training optimization loop and autoregressive sampling mechanics.
  - `docs/15_EXPERIMENT_GUIDE.md` & `docs/16_MODEL_CONFIGURATION.md`: Scientific lab method and annotated `configs/model_v0_1_char_100k.yaml`.
- **Architectural Fixes Actioned:**
  - Added `ADR-0002` (Pre-Layer Normalization by Default).
  - Added `ADR-0003` (Weight Tying as v0.1 Default Baseline).
  - Updated `docs/01_ROADMAP.md` and `research/ARCHITECTURE_REVIEW.md` checkboxes.

---

### 2026-07-24 — Research Platform & Constitution Infrastructure
- **Milestone:** Established repository constitution and specification source of truth.
- **Key Deliverables:**
  - `ENGINEERING_PRINCIPLES.md`: 10 core principles governing performance profiling, reproducibility, and configuration separation.
  - `SPEC.md`: Living technical specification with worked parameter and 8 GB memory arithmetic.
  - `research/ARCHITECTURE_REVIEW.md`: Comprehensive Staff Engineer architecture review.
  - `research/OPTIMIZATION_BACKLOG.md`: Prioritized backlog for Apple Silicon MPS optimization.

---

### 2026-07-24 — Initial Repository Scaffold
- **Milestone:** Created repository structure, documentation dependency graph, and initial flagship documents (`docs/03`, `docs/08`, `docs/09`).
- **Key Rationale:** Documentation-first architecture—designing and understanding components before writing code.

---

> **Next:** [`22_FUTURE_VERSIONS.md`](22_FUTURE_VERSIONS.md)
