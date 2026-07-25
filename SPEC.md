# SPEC.md — JimmyLabs Technical Specification

> **Status:** living source of truth. When code and this document disagree, that's a bug in
> one of them — reconcile explicitly (update the spec *or* fix the code), never silently.
> Numbers here are computed, not guessed; the arithmetic is shown so it can be checked.
>
> Companion docs: [`ARCHITECTURE_REVIEW.md`](research/ARCHITECTURE_REVIEW.md) (why these
> choices), [`ENGINEERING_PRINCIPLES.md`](ENGINEERING_PRINCIPLES.md) (the rules),
> [`docs/`](docs/) (how each piece works).

---

## 1. Project goals

Build and *understand* a decoder-only GPT of **1–4M parameters**, trained and run on a
**MacBook Air M1 (8 GB unified memory)**, entirely from scratch (PyTorch provides only
tensors, autograd, optimizers, MPS). Success = ability to explain, reimplement, train,
optimize, and document every component. Non-goal: competing with frontier models. Full
rationale in [`docs/00_PROJECT_VISION.md`](docs/00_PROJECT_VISION.md).

---

## 2. Architecture

```
   tokens (ids) ─► token emb ─┐
                              (+) ─► [ block × N ] ─► final LN ─► lm head ─► logits
   positions   ─► pos emb ────┘        │                                      │
                                       │ each block (pre-norm):               ▼
                                       │   x = x + MHSA(LN(x))            softmax → next-token
                                       │   x = x + FFN(LN(x))                 distribution
```

- **Type:** decoder-only, causal self-attention, autoregressive next-token prediction.
- **Norm:** pre-norm LayerNorm. **Activation:** GELU. **Positions:** learned embeddings.
- **Weight tying:** token-embedding matrix **shared** with the output projection (default).
- Concept docs: [`08_ATTENTION`](docs/08_ATTENTION.md), [`09_TRANSFORMER`](docs/09_TRANSFORMER.md),
  [`10_GPT_ARCHITECTURE`](docs/10_GPT_ARCHITECTURE.md).

---

## 3. Model versions

| Version | tokenizer | vocab | n_layer | n_head | n_embd | block | ≈ params |
|---------|-----------|-------|---------|--------|--------|-------|----------|
| **v0.1** | char | ~65 | 4 | 4 | 128 | 128 | **~0.82M** |
| v0.2 | char | ~138 | 6 | 6 | 192 | 256 | ~2.7M |
| v0.3 | char | ~100 | 6 | 6 | 256 | 256 | ~4.8M → trim to ≤4M |
| v1.0 | char (BPE candidate) | ~65–512 | 6 | 8 | 256 | 256 | ~4M, tuned |

v0.1 is the **computed baseline** from [`ARCHITECTURE_REVIEW.md`](research/ARCHITECTURE_REVIEW.md).
Grow one axis per experiment; never jump straight to the largest model that fits.

---

## 4. Tensor shapes (the contract every module honors)

Notation: `B`=batch, `T`=sequence length (≤ block_size), `C`=n_embd, `h`=n_head,
`d_k = C/h`, `V`=vocab.

```
   input ids                (B, T)                int64
   token embeddings         (B, T, C)
   + positional embeddings  (B, T, C)
   ── per block ───────────────────────────────────────────
   LayerNorm                (B, T, C)
   Q, K, V                  (B, h, T, d_k)        via (B,T,C)→(B,T,3C)→split
   attention scores QKᵀ     (B, h, T, T)          ⚠ the O(T²) tensor
   softmax(+causal mask)    (B, h, T, T)
   attention out            (B, h, T, d_k) → (B, T, C)
   FFN hidden               (B, T, 4C)            GELU here
   FFN out / block out      (B, T, C)
   ── head ───────────────────────────────────────────────
   final LayerNorm          (B, T, C)
   logits (lm head, tied)   (B, T, V)
   loss (cross-entropy)     scalar                vs targets (B, T)
```

Shape tests assert these first — see [`tests/TESTING_STRATEGY.md`](tests/TESTING_STRATEGY.md).

---

## 5. Parameter count (worked, for v0.1)

`V=65, L=4, h=4, C=128, T=128`, weights tied.

```
   token embedding      V·C  = 65·128        =    8,320
   positional embedding T·C  = 128·128       =   16,384
   per block:
     attention  Wq,Wk,Wv,Wo  4·C²  = 4·16,384 =  65,536   (+biases ≈ 512)
     LayerNorm ×2             2·(2C)          =      512
     FFN  W1 (C·4C)+ W2(4C·C) 8·C²  = 131,072  (+biases ≈ 640)
     ── block total ≈ 12·C² + extras          ≈ 198,272
   blocks: 4 · 198,272                        =  793,088
   final LayerNorm          2C                =      256
   lm head                  tied → 0 extra    =        0
   ────────────────────────────────────────────────────────
   TOTAL v0.1                                 ≈  818,048  (~0.82M)
```

Rule of thumb: **≈ 12·C² per block** dominates; the FFN holds ~2/3 of a block's params.
Untying the head would add `V·C` (here +8,320). See
[`docs/16_MODEL_CONFIGURATION.md`](docs/16_MODEL_CONFIGURATION.md).

### Parameter count (worked, for v0.2)

`V≈138, L=6, h=6, C=192, T=256`, weights tied.

```
   token embedding      V·C  = 138·192       =   26,496
   positional embedding T·C  = 256·192       =   49,152
   per block:
     attention  Wq,Wk,Wv,Wo  4·C²  = 4·36,864 = 147,456   (+biases ≈ 768)
     LayerNorm ×2             2·(2C)          =      768
     FFN  W1 (C·4C)+ W2(4C·C) 8·C²  = 294,912  (+biases ≈ 960)
     ── block total ≈ 12·C² + extras          ≈ 444,864
   blocks: 6 · 444,864                        = 2,669,184
   final LayerNorm          2C                =      384
   lm head                  tied → 0 extra    =        0
   ────────────────────────────────────────────────────────
   TOTAL v0.2                                 ≈ 2,745,216 (~2.7M)
```

---

## 6. Memory usage (fp32, on 8 GB)

**Weights/optimizer are negligible at this scale; activations and the O(T²) attention
matrix are what to watch.**

```
   parameters (fp32)        0.82M · 4 B            ≈ 3.3 MB
   gradients                = params               ≈ 3.3 MB
   Adam state (2 moments)   2 · params             ≈ 6.6 MB
   ── subtotal (fixed)      ─────────────────────  ≈ 13 MB   (trivial)

   activations scale with BATCH. The attention matrix is the ceiling:
   attn tensor = B · h · T · T · 4 B  per layer,  × L layers
   for B=32, h=4, T=128, L=4:  32·4·128·128·4 · 4  ≈ 32 · 1.0 MB  ≈ 32 MB
```

**Takeaway (honest):** v0.1 is *comfortable* on 8 GB — you could train it with room to
spare. The 8 GB constraint only bites as `T` (block_size) and `B` grow, because the
attention term grows as `B·T²`:

```
   attention memory per layer (fp32, h=4), how it explodes with T:
     T=128,  B=32   →   ~1.0 MB/layer      (fine)
     T=256,  B=32   →   ~4.0 MB/layer      (fine)
     T=512,  B=32   →   ~16 MB/layer       (getting real)
     T=1024, B=64   →   ~256 MB/layer  ×L  →  gigabytes  (won't fit)
```

This is *why* JimmyLabs keeps `block_size` modest and treats context length as a memory
decision, not a free knob. Full treatment:
[`docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md`](docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md),
[`architecture/memory_layout.md`](architecture/memory_layout.md).

---

## 7. Training pipeline

```
   corpus → token ids → train/val split (seeded)
        │
        ▼   sample random (B, T) windows
   forward → logits (B,T,V) → cross-entropy vs shifted targets
        │
        ▼
   backward (autograd) → grad clip (max-norm 1.0) → optimizer step (AdamW)
        │
        ▼   LR: warmup → cosine decay
   every K steps: eval on val split · sample fixed prompts · checkpoint if best
```

- **Optimizer:** AdamW, weight decay on matmul weights only (not biases/LayerNorm).
- **LR schedule:** linear warmup then cosine decay (values in `configs/`, not hard-coded).
- **Grad clipping:** global-norm 1.0. **Precision:** fp32 baseline; fp16/bf16 is a
  *benchmarked* experiment, not an assumption (MPS support varies — see doc 13).
- Full detail: [`docs/11_TRAINING_PIPELINE.md`](docs/11_TRAINING_PIPELINE.md),
  [`architecture/training_pipeline.md`](architecture/training_pipeline.md).

---

## 8. Inference pipeline

```
   prompt → ids → forward → logits[:, -1, :]  (last position)
        → temperature scale → top-k / top-p filter → softmax → sample
        → append token → repeat (autoregressive)
```

- **Sampling knobs:** temperature, top-k, top-p (nucleus). Greedy = temperature→0.
- **Cost:** naive generation is O(T²) (recompute all keys/values each step). **KV cache**
  is the planned optimization → O(T) per step. See
  [`docs/12_INFERENCE.md`](docs/12_INFERENCE.md).

---

## 9. Configuration

Every hyperparameter lives in a commented YAML in [`configs/`](configs/), never in code
(principle 8). A model = `config + seed`. Required fields: `vocab_size, n_layer, n_head,
n_embd, block_size, dropout, weight_tying`, plus training block (`batch_size, lr,
warmup_steps, max_steps, weight_decay, grad_clip, eval_interval, seed`). Spec for each
field: [`docs/16_MODEL_CONFIGURATION.md`](docs/16_MODEL_CONFIGURATION.md).

---

## 10. Dependencies

```
   required:  python ≥ 3.10 · torch (with MPS) · pyyaml · numpy
   dev:       pytest · matplotlib (plots) · (optional) tqdm
   forbidden: any prebuilt transformer / attention / tokenizer library
```

Minimal on purpose (principle 9). MLX is a *research comparison* target, not a runtime
dependency — see [`research/tiny_gpt_landscape.md`](research/tiny_gpt_landscape.md).

---

## 11. Failure cases (known, watched)

| Failure | Symptom | Guard |
|---------|---------|-------|
| Missing/incorrect causal mask | loss drops implausibly fast, generation is garbage | mask unit test asserts upper-triangle = −∞ |
| Loss = NaN | exploding activations/grads | grad clip; check LR/warmup; assert finite |
| Loss flat | disconnected graph, LR too low, bug in backward | overfit-a-batch test must reach ~0 |
| OOM on MPS | batch/block too large; attn O(T²) | SPEC §6 budget; reduce B or T first |
| Non-determinism | seed not set on all RNGs incl. MPS | seed-everything util + determinism test |
| Tokenizer not round-tripping | decode(encode(x)) ≠ x | round-trip unit test |
| MPS numerical drift vs CPU | small logit differences | tolerance-based cross-device test |

Details and repro notes: [`docs/11`](docs/11_TRAINING_PIPELINE.md), [`docs/13`](docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md).

---

## 12. Testing strategy (summary)

Pyramid: **shape tests → known-answer tests → gradient checks → overfit-a-batch →
regression tests**. Every `src/` module ships with tests before it's "done" (principle 6).
Full plan: [`tests/TESTING_STRATEGY.md`](tests/TESTING_STRATEGY.md).

---

## 13. Evaluation contract

- **Quantitative:** held-out cross-entropy **and** perplexity on a seeded val split,
  logged every `eval_interval`.
- **Qualitative:** a fixed set of prompts generated at every checkpoint (including a
  *before-training* baseline), stored in [`outputs/`](outputs/) so progress is visible.
- **Performance:** tokens/sec, peak memory, checkpoint size — via
  [`docs/14_BENCHMARKING.md`](docs/14_BENCHMARKING.md).

> No metric is reported without its definition and its seed. A number without conditions is
> noise (principle 2).
