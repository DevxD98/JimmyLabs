# Optimization Backlog

> The engineering roadmap for making JimmyLabs leaner and faster **on 8 GB Apple Silicon** —
> prioritized, and governed by the iron rule: *nothing here is "done" until a benchmark
> proves the win* (principles 1–2). Each item links to the concept doc that explains it.
>
> **Priority = Impact ÷ Difficulty, filtered by "does it fit the 8 GB / tiny-model
> reality."** Ratings are pre-measurement *hypotheses*; the **Result** column is filled from
> [`benchmarks/`](../benchmarks/) once tested. A hypothesis is not a result.

## Legend

```
   Impact      ★☆☆☆☆ negligible … ★★★★★ transformative (on THIS machine/scale)
   Difficulty  ★☆☆☆☆ trivial … ★★★★★ major undertaking
   Status      applied · planned · experiment-open · deferred · rejected · superseded

   deferred    measured or reasoned to be not worth doing AT THE CURRENT SCALE, with the
               condition that would make it worth revisiting written down. Distinct from
               "rejected" (tried, didn't win) and from "planned" (still intended).
```

## The backlog (priority order)

| # | Optimization | Impact | Difficulty | Priority | Status | Concept |
|---|--------------|--------|------------|----------|--------|---------|
| 1 | **Weight tying** (share token-emb & output proj) | ★★★★☆ | ★☆☆☆☆ | **applied (default)** | applied | [SPEC §5](../SPEC.md), [09](../docs/09_TRANSFORMER.md) |
| 2 | **KV cache** for generation (O(T²)→O(T)/step) | ★★★★★ | ★★★☆☆ | **High** | **applied** (within `block_size` only — [ADR-0004](design_decisions/ADR-0004-no-cache-reuse-past-block-size.md)) | [12](../docs/12_INFERENCE.md), [bench 002](../benchmarks/002_kv_cache.md) |
| 3 | **Gradient accumulation** (big effective batch, low mem) | ★★★★☆ | ★★☆☆☆ | **High** | planned | [13](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| 4 | **Kill hot-loop syncs** (no per-step `.item()`/print) | ★★★☆☆ | ★☆☆☆☆ | **High** | **applied (already satisfied — audited, no code change)** — see note below | [13](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| 5 | **Keep data on-device / avoid CPU↔MPS ping-pong** | ★★★☆☆ | ★☆☆☆☆ | **High** | **applied (already satisfied — audited, no code change)** — see note below | [13](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| 6 | **bf16/fp16 mixed precision** (halve activation mem) | ★★★★☆ | ★★★☆☆ | **Medium** (version-dependent) | experiment-open | [13](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| 7 | **Memory-map dataset** (keep corpus out of the 8 GB) | ★★★☆☆ | ★★☆☆☆ | **Medium** | **applied** — unreclaimable RAM 3,522→283 MB (**12.5x**) on the 414.7M-token corpus; disk 4x smaller; batches proven bit-identical → [006](../benchmarks/006_mmap_dataset.md) | [17](../docs/17_DATASET_GUIDE.md) |
| 8 | **Efficient sampling** (top-k/top-p without full sort) | ★★☆☆☆ | ★★☆☆☆ | **Deferred — not worth optimizing at current vocab scale (≤138)** | deferred | [12](../docs/12_INFERENCE.md) |
| 9 | **Tuned batch/block for the 8 GB ceiling** | ★★★☆☆ | ★★☆☆☆ | **Medium** | experiment-open | [SPEC §6](../SPEC.md) |
| 10 | **Fused/`SDPA` attention** (if MPS path is faster) | ★★★☆☆ | ★★★☆☆ | **Medium** | experiment-open | [08](../docs/08_ATTENTION.md) |
| 11 | **Tiled / FlashAttention-style attn** (avoid T×T mat) | ★★★★☆ | ★★★★★ | **Low (until T is the ceiling)** | planned | [08](../docs/08_ATTENTION.md), [18](../docs/18_RESEARCH_PAPERS.md) |
| 12 | **Grouped-Query Attention** (shrink KV cache) | ★★★☆☆ | ★★★★☆ | **Low** | planned | [future_architecture](../architecture/future_architecture.md) |
| 13 | **Quantized inference** (int8/int4 for deploy) | ★★★☆☆ | ★★★★☆ | **Low (v1.0+)** | planned | [13](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| 14 | **MLX port** as alt runtime | ★★★☆☆ | ★★★★☆ | **Low (research)** | experiment-open | [tiny_gpt_landscape](tiny_gpt_landscape.md) |
| 15 | **`torch.compile`** on MPS (if/when it helps) | ★★☆☆☆ | ★★☆☆☆ | **Low** | experiment-open | [13](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| 16 | **Rotary/relative positions** (unblocks #2 past `block_size`) | ★★★☆☆ | ★★★★☆ | **Low (v0.2+, needs retrain)** | planned | [07](../docs/07_EMBEDDINGS.md), [ADR-0004](design_decisions/ADR-0004-no-cache-reuse-past-block-size.md), [ADR-0006](design_decisions/ADR-0006-rotary-positions-candidate.md) |

## Audit notes (2026-07-25)

Items #4, #5 and #8 were audited by reading the code rather than by assuming the table was
current. Two turned out to be **already satisfied by construction** — the honest close-out
is a citation, not a patch. Changing working code to "apply" an optimization it already has
would be a no-op diff dressed up as a win, and would risk breaking something for nothing.

### #4 — Kill hot-loop syncs → already satisfied

`.item()` forces a sync because the CPU must wait for the queued MPS work to finish before
it can read a scalar. Calling it every step serializes training against the GPU queue. In
`scripts/train.py` this never happens on the hot path:

- `scripts/train.py:124` — `if step % eval_interval == 0 or step == max_steps:` guards the
  entire reporting block.
- `scripts/train.py:130` (`val_loss_tensor.item()`) and `:132` (`loss.item()`) are the only
  `.item()` calls in the training loop, and both sit inside that guard.
- `scripts/train.py:138` and `:144` are the only per-step-eligible `print`s, also inside it.
- `src/jimmylabs/training/schedule.py:36` — `clip_gradients` calls
  `torch.nn.utils.clip_grad_norm_` and **discards the returned norm**. Reading that tensor
  would sync every step; not reading it is what keeps the clip free.
- `src/jimmylabs/inference/generate.py` — the generation loop has no `.item()`/`.cpu()` at
  all; the `float('Inf')` at `:49`/`:69` are Python constants, not device reads, and the
  cache-trim test at `:17` compares `Tensor.size()`, which is a Python int.

The remaining `print`s (`scripts/train.py:37`, `:53`) run once at startup, outside the loop.

### #5 — Keep data on-device / avoid CPU↔MPS ping-pong → already satisfied

`src/jimmylabs/data/loader.py:54` — `get_batch` performs exactly **one** `.to(device)` per
batch, on the assembled `(B, T)` windows, and the tensors are consumed by the model without
ever being moved back. `get_batch_mmap` does the same at `:150`. There is no ping-pong to
remove.

One clarification worth recording, since "keep data on-device" could be misread as "put the
whole corpus on-device": that is **not** the goal here and must not become one. The
TinyStories train split is 414.7M tokens (3.3 GB as int64) against 8 GB of unified memory —
residency is the problem item #7 exists to solve, and #7 moves in the opposite direction, off
the heap entirely. What #5 asks for is that *batches* cross the boundary once, which they do.

### #8 — Efficient sampling → deferred, with the trigger written down

`src/jimmylabs/inference/generate.py:47` already uses `torch.topk` for top-k, which avoids a
full sort. `:56` uses a full `torch.sort` over the vocabulary for top-p.

That sort is **computationally negligible at our scale and will stay that way for v0.1–v0.3**:
the vocabulary is ~65 characters for Shakespeare and ~138 for TinyStories
([`SPEC.md` §3](../SPEC.md)). Sorting ~138 elements once per generated token is far below the
cost of the forward pass that produced the logits, and top-p is off by default. Replacing it
with an incremental/partial-sort scheme would add real complexity and a new correctness
surface to buy a saving that no benchmark can currently detect — a textbook violation of
principle 1 (never optimize before profiling).

**Revisit when the vocabulary grows by an order of magnitude** — specifically if/when the BPE
tokenizer lands ([`docs/06_TOKENIZER.md`](../docs/06_TOKENIZER.md), a v1.0 candidate per
[`ADR-0001`](design_decisions/ADR-0001-character-tokenizer-first.md)), which would take vocab
into the thousands. At that point profile top-p first, then optimize if the profile says so.

## Reading the priorities

- **Items 1–5 are the cheap, high-leverage wins** — mostly loop hygiene and one free
  parameter-saving default. Do these first; several cost almost nothing.
- **Items 2 (KV cache) and 6 (bf16) are the big structural wins** but carry real complexity
  or version-dependence. KV cache is the top *inference* prize; bf16 the top *memory* prize —
  both gated on benchmarks.
- **Items 11–15 are deliberately Low** for now. This is the **overengineering guard** from
  [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md): FlashAttention, GQA, quantization, and
  MLX are powerful but only worth their complexity once a benchmark shows the simpler model
  has hit a wall. Understanding them (papers, notes) is Phase-5 work; *implementing* them is
  earned, not assumed.

## The rule for closing an item

```
   propose (this table)  ─►  benchmark baseline (docs/14)  ─►  apply, one change
        ─►  benchmark after  ─►  win real & repeatable?
              ├─ yes ─► mark "applied", record Δ, link the benchmark
              └─ no  ─► mark "rejected", record WHY (a negative result is a result)
```

Rejected optimizations stay in the table with their reason — so nobody re-proposes them
without new information. Silent removal loses the lesson.

## See also

- [`../docs/14_BENCHMARKING.md`](../docs/14_BENCHMARKING.md) (how to measure) ·
  [`../SPEC.md`](../SPEC.md) (the numeric budget) ·
  [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md) (why several items are intentionally
  Low priority).
