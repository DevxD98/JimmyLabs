# 12 — Inference

> **Prerequisites:** [`08_ATTENTION.md`](08_ATTENTION.md) (causal masking), [`10_GPT_ARCHITECTURE.md`](10_GPT_ARCHITECTURE.md) (GPT forward pass), and [`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md) (trained model checkpoints).
>
> **Next:** [`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md)

---

## Purpose

Training teaches a GPT model the probability distribution of next tokens $P(x_t \mid x_{<t})$. **Inference** is the execution process that uses this learned distribution to generate new text.

This document details the mechanics of text generation in JimmyLabs:
1. **The Autoregressive Loop** — generating text token-by-token.
2. **Sampling Strategies** — Temperature scaling, Top-$K$ filtering, and Top-$P$ (nucleus) sampling.
3. **KV Cache Concept Introduction** — reducing generation complexity from $O(T^2)$ to $O(T)$ per step.

This document is the **canonical home** in JimmyLabs for autoregressive sampling knobs and the conceptual introduction to the KV Cache.

---

## Background

### Training vs Inference execution modes

| Mode | Input Shape | Processing | Output |
|------|-------------|------------|--------|
| **Training** | Entire sequence $(B, T)$ | Parallel across all $T$ positions | Batch loss scalar |
| **Inference** | Growing sequence $(1, t)$ | Sequential step-by-step ($t \to t+1$) | New token IDs |

During training, causal masking ([`08_ATTENTION.md`](08_ATTENTION.md)) allows processing all $T$ tokens simultaneously in one forward pass. During inference, future tokens do not exist yet—the model must generate token $t+1$, append it to the sequence, and feed the expanded prompt back into the model to predict token $t+2$.

---

## Concepts

Core inference concepts:

- **Autoregressive Generation:** Generating sequence elements sequentially, where each output token becomes part of the input context for the next step.
- **Logit Scaling & Temperature ($T_{\text{temp}}$):** Dividing raw logits by temperature before softmax to control sampling randomness.
- **Top-$K$ Sampling:** Truncating probability distribution to only the $K$ highest-scoring tokens.
- **Top-$P$ (Nucleus) Sampling:** Truncating distribution to the smallest set of top tokens whose cumulative probability exceeds threshold $P$.
- **KV Cache:** Caching previously computed Key ($K$) and Value ($V$) tensors across generation steps to avoid redundant $O(T^2)$ matmuls.

---

## Detailed Explanation

### 1. The Autoregressive Generation Loop

Given a prompt sequence of token IDs $[x_0, x_1, \dots, x_k]$:

```
   Prompt: [x0, x1, x2] ──► Model ──► Logits for last position ──► Sample x3
   Next:   [x0, x1, x2, x3] ──► Model ──► Logits for last position ──► Sample x4
```

#### Step-by-Step Loop Protocol
1. Feed current sequence $X$ (shape $1 \times t$) into GPT model.
2. Retrieve raw logit vector at the **last position**: $z = \text{logits}[:, -1, :]$ (shape $1 \times V$).
3. Apply sampling transformation (temperature, top-$k$, top-$p$) to obtain probability distribution $p$.
4. Sample next token ID $x_{t+1} \sim p$.
5. Append $x_{t+1}$ to sequence: $X \leftarrow [X, x_{t+1}]$.
6. Repeat until reaching `max_new_tokens` or end-of-text token.

---

### 2. Sampling Knobs

#### Temperature Scaling ($T_{\text{temp}}$)
Logits $z$ are divided by scalar $T_{\text{temp}} > 0$ before applying softmax ([`04_MATHEMATICS.md`](04_MATHEMATICS.md)):

$$p_i = \frac{e^{z_i / T_{\text{temp}}}}{\sum_{j=1}^{V} e^{z_j / T_{\text{temp}}}}$$

- **Greedy Decoding ($T_{\text{temp}} \to 0$):** Softmax approaches a one-hot argmax. Model always picks the highest-scoring token (deterministic, repetitive).
- **Standard ($T_{\text{temp}} = 1.0$):** Unmodified learned distribution.
- **High Creativity ($T_{\text{temp}} > 1.0$):** Flattens distribution, increasing randomness and diversity (risks nonsensical text at very high values).

```
   Raw Logits z = [4.0, 2.0, 1.0]

   T = 0.5 (Sharp)  ──► Softmax ──► [0.97, 0.03, 0.00]  (High confidence)
   T = 1.0 (Normal) ──► Softmax ──► [0.84, 0.11, 0.04]  (Balanced)
   T = 2.0 (Flat)   ──► Softmax ──► [0.58, 0.26, 0.16]  (High randomness)
```

#### Top-$K$ Filtering
1. Identify the $K$ largest logit values.
2. Set all logits outside the top $K$ to $-\infty$.
3. Apply softmax. Tokens outside top $K$ receive probability $0.0$.

Prevents sampling low-probability tail tokens that cause grammatical collapse.

#### Top-$P$ (Nucleus) Filtering
1. Sort logits in descending order.
2. Compute cumulative softmax probabilities.
3. Remove tokens whose cumulative probability exceeds threshold $P$ (e.g. $P = 0.9$).
4. Set removed logits to $-\infty$ and re-normalize softmax.

Dynamically expands vocabulary pool when uncertainty is high, and contracts pool when model is confident.

---

### 3. Introduction to KV Cache

In naive autoregressive generation, at step $t$, we feed sequence $[x_0, \dots, x_t]$ into the model. Attention computes Query, Key, and Value projections for **all** $t$ tokens.

#### The Waste in Naive Generation
Past keys $K_{0 \dots t-1}$ and values $V_{0 \dots t-1}$ do not change when a new token $x_t$ is added. Recomputing them at every step results in $O(T^2)$ time complexity per generated token.

#### The KV Cache Solution
Instead of recomputing past keys and values, we store (cache) them in memory across steps:

```
   Step t:   Compute Q_t, K_t, V_t for new token ONLY.
             Concat K_t to cached [K_0 ... K_{t-1}].
             Concat V_t to cached [V_0 ... V_{t-1}].
             Compute attention score between Q_t and cached Keys.
```

By caching past keys and values, per-step computation drops from $O(T^2)$ to $O(T)$, providing dramatic speedups during long-context generation. Detailed KV cache implementation specifications live in [`architecture/future_architecture.md`](../architecture/future_architecture.md).

> **Caveat — the cache stops at `block_size`.** The sketch above assumes a cached
> $K_i$ stays valid forever. That holds only while the context is still *growing*. This
> model uses learned **absolute** positions ([`07_EMBEDDINGS.md`](07_EMBEDDINGS.md)), so
> each cached entry is frozen at the position its token had when embedded. Once the
> context outgrows `block_size` the window must slide, and a slid window has to be
> re-embedded from position 0 — which invalidates every cached key at once. JimmyLabs
> therefore disengages the cache at the boundary and falls back to the naive
> sliding-window forward, trading that speedup for correctness. Reusing a *trimmed*
> cache instead is the bug documented in
> [`ADR-0004`](../research/design_decisions/ADR-0004-no-cache-reuse-past-block-size.md):
> it looks right and silently shifts every logit. Relative/rotary positions are what
> actually lift the restriction (backlog #16).

---

## Visual Diagrams

### Autoregressive Sampling Pipeline

```
  Last Position Logits (1, V)
              │
              ▼
    [ Divide by Temperature T ]
              │
              ▼
    [ Top-K / Top-P Masking (-∞) ]
              │
              ▼
    [ Softmax Normalization ]
              │
              ▼
    [ Multinomial Sampling ]
              │
              ▼
   Sampled Token ID ──► Append to Input & Repeat
```

---

## Common Mistakes

- **Re-running Entire Prompt Sequence Naively:** Feeding growing prompt sequences into model without KV cache for long generations, causing generation speed to slow quadratically.
- **Setting Temperature to Zero Directly:** Passing `temperature = 0.0` directly to division `z / T`, causing `ZeroDivisionError` or `NaN`. Greedy decoding must be implemented via `torch.argmax()`.
- **Applying Softmax Before Top-K/Top-P Filtering:** Filtering probabilities *after* softmax requires re-normalizing. Filtering raw logits with $-\infty$ *before* softmax is cleaner and numerically stable.

---

## Future Improvements

- Implement complete KV Cache tensor management in `src/inference/kv_cache.py`.
- Benchmark generation throughput (tokens/sec) with and without KV cache on Apple Silicon MPS hardware.

---

## Glossary

Canonical definitions live in [`19_GLOSSARY.md`](19_GLOSSARY.md). Inference terms:

| Term | Definition |
|------|------------|
| Autoregressive Loop | Sequential text generation where output token is appended as next input |
| Temperature ($T_{\text{temp}}$) | Logit scaling factor controlling distribution sharpness / randomness |
| Top-$K$ | Filtering method restricting sampling to the $K$ highest-probability tokens |
| Top-$P$ (Nucleus) | Filtering method restricting sampling to cumulative probability threshold $P$ |
| KV Cache | Caching past Key and Value tensors to reduce per-step generation complexity to $O(T)$ |

---

## Learning Checklist

You master inference mechanics when you can:

- [ ] Explain why training is parallel while inference is sequential.
- [ ] Trace the autoregressive generation loop step-by-step.
- [ ] Predict how temperature $< 1.0$ vs $> 1.0$ affects output distribution.
- [ ] Explain how Top-$K$ and Top-$P$ filtering prevent sampling tail tokens.
- [ ] Explain why naive generation is $O(T^2)$ and how KV cache reduces it to $O(T)$.

---

## References

- [`SPEC.md` §8](../SPEC.md#8-inference-pipeline) — Technical inference specification.
- Holtzman et al. (2020), *The Curious Case of Neural Text Degeneration* (Nucleus / Top-$P$ sampling).
- [`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md) — Configuration hyperparameters.

## Further Reading

- [`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md) — **Next:** Optimizing inference and training on Apple Silicon 8 GB unified memory.
- [`architecture/inference_pipeline.md`](../architecture/inference_pipeline.md) — Concrete data flow for inference execution in JimmyLabs.

> **Next:** [`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md)
