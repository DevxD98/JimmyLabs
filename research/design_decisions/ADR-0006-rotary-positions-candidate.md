# ADR-0006 — Rotary/Relative Positions as a Candidate for Longer KV-Cache Reuse

| Field | Value |
|-------|-------|
| Status | proposed |
| Date | 2026-07-26 |
| Deciders | JimmyLabs Core Team |
| Related | [`docs/07_EMBEDDINGS.md`](../../docs/07_EMBEDDINGS.md) · [`research/OPTIMIZATION_BACKLOG.md`](../OPTIMIZATION_BACKLOG.md) #16 · [`ADR-0004`](ADR-0004-no-cache-reuse-past-block-size.md) |

## Context

JimmyLabs currently adds a learned absolute position vector to every token embedding.
That is simple and appropriate for the first educational model, but it makes a cached key
position-specific. Once a sliding generation window passes `block_size`, retaining that key
would reuse a representation built for the wrong absolute position. ADR-0004 therefore
disables cache reuse at the boundary and falls back to the correct naive path.

This ADR records a design candidate only. It does not change the model or any checkpoint.

## What would change

Rotary Position Embeddings (RoPE) apply a position-dependent rotation to query and key
vectors inside attention rather than adding a learned position vector at the input. The
inner product between a query and key then depends on their relative position difference.
Relative-position variants encode the same broad goal through an attention relation rather
than a fixed lookup row.

The important cache consequence is that a key's usable relationship is tied to its relative
offset, not permanently to the input slot where the token first entered the model. A cache
can therefore continue to hold prior keys while generation advances, subject to the chosen
context/window policy, instead of becoming invalid solely because the absolute embedding
table reached `block_size`.

This does not make context unlimited: attention memory still grows with the retained window,
the model still has a configured context policy, and numerical/implementation details need
their own equivalence tests.

## Options considered

### Keep learned absolute positions

- **Pros:** current implementation is small, readable, and already trained; no checkpoint
  migration or retraining cost.
- **Cons:** cache reuse past `block_size` is not correct; longer generation falls back to the
  naive path.

### Adopt RoPE or another relative-position mechanism

- **Pros:** removes the specific absolute-position mismatch that blocks longer cache reuse;
  aligns with a widely used modern attention design.
- **Cons:** changes the model architecture, attention equations, parameter behavior, tests,
  documentation, and checkpoint contract. Existing absolute-position checkpoints are not
  compatible and must not be silently loaded into the new model.

## Retraining and compatibility cost

Existing v0.1 and v0.2 checkpoints would **not** be compatible. Their learned positional
embedding table is part of the trained computation, while a RoPE model would rotate Q/K in
attention and would no longer consume the same positional parameters in the same way. A
RoPE experiment needs a new config/model identity, a fresh seed-controlled training run, a
new checkpoint, and a qualitative/evaluation comparison against the absolute-position
baseline.

The experiment must also add an equivalence gate for any cached path: on the same model,
seed, prompt, and context, cached and uncached logits must agree within an explicitly chosen
floating-point tolerance. Token argmax alone is not sufficient; ADR-0004 documents how that
weaker gate missed a real divergence.

## Recommendation

Keep learned absolute positions for the current educational v0.1/v0.2 line. The corrected
fallback in ADR-0004 is the smallest trustworthy solution, and the current project has not
shown that long-context generation is a measured bottleneck. Revisit this candidate when a
benchmark demonstrates that generation beyond `block_size` is important enough to justify a
new architecture and a full retraining cycle.

Until then, leave backlog item #16 planned and do not implement RoPE as an opportunistic
optimization.
