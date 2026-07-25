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
   Status      applied · planned · experiment-open · rejected · superseded
```

## The backlog (priority order)

| # | Optimization | Impact | Difficulty | Priority | Status | Concept |
|---|--------------|--------|------------|----------|--------|---------|
| 1 | **Weight tying** (share token-emb & output proj) | ★★★★☆ | ★☆☆☆☆ | **applied (default)** | applied | [SPEC §5](../SPEC.md), [09](../docs/09_TRANSFORMER.md) |
| 2 | **KV cache** for generation (O(T²)→O(T)/step) | ★★★★★ | ★★★☆☆ | **High** | planned | [12](../docs/12_INFERENCE.md) |
| 3 | **Gradient accumulation** (big effective batch, low mem) | ★★★★☆ | ★★☆☆☆ | **High** | planned | [13](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| 4 | **Kill hot-loop syncs** (no per-step `.item()`/print) | ★★★☆☆ | ★☆☆☆☆ | **High** | planned | [13](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| 5 | **Keep data on-device / avoid CPU↔MPS ping-pong** | ★★★☆☆ | ★☆☆☆☆ | **High** | planned | [13](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| 6 | **bf16/fp16 mixed precision** (halve activation mem) | ★★★★☆ | ★★★☆☆ | **Medium** (version-dependent) | experiment-open | [13](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| 7 | **Memory-map dataset** (keep corpus out of the 8 GB) | ★★★☆☆ | ★★☆☆☆ | **Medium** | **applied** — unreclaimable RAM 3,522→283 MB (**12.5x**) on the 414.7M-token corpus; disk 4x smaller; batches proven bit-identical → [005](../benchmarks/005_mmap_dataset.md) | [17](../docs/17_DATASET_GUIDE.md) |
| 8 | **Efficient sampling** (top-k/top-p without full sort) | ★★☆☆☆ | ★★☆☆☆ | **Medium** | planned | [12](../docs/12_INFERENCE.md) |
| 9 | **Tuned batch/block for the 8 GB ceiling** | ★★★☆☆ | ★★☆☆☆ | **Medium** | experiment-open | [SPEC §6](../SPEC.md) |
| 10 | **Fused/`SDPA` attention** (if MPS path is faster) | ★★★☆☆ | ★★★☆☆ | **Medium** | experiment-open | [08](../docs/08_ATTENTION.md) |
| 11 | **Tiled / FlashAttention-style attn** (avoid T×T mat) | ★★★★☆ | ★★★★★ | **Low (until T is the ceiling)** | planned | [08](../docs/08_ATTENTION.md), [18](../docs/18_RESEARCH_PAPERS.md) |
| 12 | **Grouped-Query Attention** (shrink KV cache) | ★★★☆☆ | ★★★★☆ | **Low** | planned | [future_architecture](../architecture/future_architecture.md) |
| 13 | **Quantized inference** (int8/int4 for deploy) | ★★★☆☆ | ★★★★☆ | **Low (v1.0+)** | planned | [13](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| 14 | **MLX port** as alt runtime | ★★★☆☆ | ★★★★☆ | **Low (research)** | experiment-open | [tiny_gpt_landscape](tiny_gpt_landscape.md) |
| 15 | **`torch.compile`** on MPS (if/when it helps) | ★★☆☆☆ | ★★☆☆☆ | **Low** | experiment-open | [13](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) |

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
