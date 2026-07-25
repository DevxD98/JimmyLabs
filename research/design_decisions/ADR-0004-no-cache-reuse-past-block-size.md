# ADR-0004 — Do Not Reuse the KV Cache Once the Context Outgrows `block_size`

| Field | Value |
|-------|-------|
| Status | accepted |
| Date | 2026-07-25 |
| Deciders | JimmyLabs Core Team |
| Related | research/OPTIMIZATION_BACKLOG.md #2 · benchmarks/002_kv_cache.md · docs/12_INFERENCE.md · docs/07_EMBEDDINGS.md · ADR-0002 |

## Context

The KV cache (`OPTIMIZATION_BACKLOG` #2) stores each layer's keys and values so that an
autoregressive step costs `O(1)` attention work instead of `O(T)`. It shipped in Phase 4
with a measured **+65%** generation throughput win.

It was also silently incorrect past `block_size`.

JimmyLabs uses **learned absolute** positional embeddings
([`docs/07_EMBEDDINGS.md`](../../docs/07_EMBEDDINGS.md)): a token's representation is
fixed to the integer position it was embedded at. `gpt.py` derives that position from the
*length* of the cache:

```python
pos_offset = past_key_values[0][0].size(-2) if past_key_values is not None else 0
```

`generate()` trimmed the cache when it reached `block_size`, keeping the last
`block_size - 1` entries. That keeps `pos_offset` looking correct — it is still the cache
length — while the retained K/V remain frozen at the positions they were *first* embedded
at. The naive path, meanwhile, re-embeds its sliding window from position 0. So the two
paths disagree about where every past token is, drifting one position per slide:

| sequence length | max abs Δlogit (cached vs naive) |
|---|---|
| ≤ `block_size` (16) | ~6e-08 — float noise |
| **17** (first slide) | **5.8e-03** |
| 25 | 2.2e-02 |
| 52 | 3.2e-02 |

Five orders of magnitude at exactly `block_size + 1`, compounding from there. The bug was
invisible because the correctness gate compared *sampled tokens* under greedy decoding,
and an untrained model's argmax rarely moves for a ~1e-2 logit shift — the test passed on
all 20 seeds tried while the logits were wrong.

The constraint is not the trimming arithmetic. It is absolute positions: **once the window
slides, the whole window must be re-embedded**, so no cached key is reusable again.

## Options considered

### Option A — Fix the trimming arithmetic (keep caching past `block_size`)
- Pros:
  - Would preserve the cache speedup for unbounded-length generation.
- Cons:
  - **Not possible with absolute positions.** A cached key encodes its old position; the
    only way to move it is to recompute it, which is exactly what the cache avoids. Any
    arithmetic that makes `pos_offset` "look right" reproduces the current bug.

### Option B — Stop caching at the `block_size` boundary; fall back to the naive sliding-window forward
- Pros:
  - Correct by construction: past the boundary the cached path performs literally the same
    computation as the naive path, so equivalence is structural, not a tuned tolerance.
  - Keeps the full speedup for the common case (generating within `block_size`).
  - Three lines of control flow; no change to the model or the cache format.
- Cons:
  - Generation longer than `block_size` gets no cache benefit — per-step cost returns to
    `O(T)` there.
  - Costs ~3% of measured cached throughput at `gen_tokens=200, block_size=128`.

### Option C — Adopt rotary (RoPE) or relative positional embeddings
- Pros:
  - Removes the restriction entirely: relative positions make a cached key valid wherever
    it sits, so the cache stays reusable for unbounded generation.
  - Matches what modern small GPTs do.
- Cons:
  - Replaces a v0.1 architectural default set in ADR-0002's lineage, invalidating the
    818,048-parameter count in `SPEC.md` §5 and every trained checkpoint.
  - A model-architecture change smuggled in as a bug fix — it needs its own ADR,
    documentation pass, and retrained baseline.

## Decision

We chose **Option B**. The single most important reason: it makes cached and naive
generation *the same computation* past the boundary, so the correctness gate
`OPTIMIZATION_BACKLOG` #2 asks for ("cached output == naive output") holds by construction
rather than by luck. A faster generator that changes the text is a regression, not an
optimization (`benchmarks/002_kv_cache.md` §5).

Option C is the right long-term answer and is now the first item under "future work" in
[`OPTIMIZATION_BACKLOG.md`](../OPTIMIZATION_BACKLOG.md) — but it is an architecture
change, not a fix, and it does not belong in the same PR.

## Consequences

- **Easier now:** cached generation is trustworthy at any length; the equivalence test is
  meaningful instead of decorative; the ~3% cost is bounded and measured.
- **Harder now:** long-context generation (beyond `block_size`) sees no cache win. Anyone
  benchmarking long generations must know the cache disengages at the boundary, or they
  will misread the curve.
- **To watch:** the moment `block_size` grows (v0.2's ~2.7M config) or someone wants
  generation well past the context window, Option C becomes worth the retraining cost.

## Notes

The regression test now re-seeds the RNG before each path and samples with `top_k`, which
makes the whole distribution observable rather than just the argmax. Before the fix, 4/4
seeds fail; after, 0/144 model × sampling-seed combinations disagree. A within-`block_size`
guard rail sits alongside it so the regression cannot pass for the wrong reason.
