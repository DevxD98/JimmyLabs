# `research/design_decisions/` — Architecture Decision Records (ADRs)

## Why ADRs

Months from now, someone (probably you) will look at a choice — *why a character tokenizer
first? why pre-norm? why weight tying?* — and want the reasoning, not just the result.
An ADR freezes that reasoning at the moment it was made, including the options rejected.
Rejected options are the most valuable part: they stop you re-litigating settled choices.

## Format

Each record is `ADR-NNNN-short-title.md` and follows:

```
   Status      proposed | accepted | superseded by ADR-XXXX
   Context     what forces a decision here
   Options     the 2–3 real alternatives, with trade-offs
   Decision    what we chose
   Consequences  what this makes easy, what it makes hard, what to watch
```

An ADR is **append-only in spirit**: you don't delete a bad decision, you supersede it
with a new ADR that links back. The trail of superseded decisions *is* the learning.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| `ADR-0001-character-tokenizer-first.md` | Start with a character tokenizer | accepted (example) |
| `ADR-0002-pre-norm.md` | Use Pre-Layer Normalization by Default | accepted |
| `ADR-0003-weight-tying-default.md` | Enable Weight Tying as v0.1 Default Baseline | accepted |
| `ADR-0004-no-cache-reuse-past-block-size.md` | Do Not Reuse the KV Cache Once the Context Outgrows `block_size` | accepted |
| `ADR-0005-tinystories-corpus-scope.md` | Prepare the full TinyStories corpus, train on a Chinchilla-bounded slice | accepted |
| `ADR-0006-rotary-positions-candidate.md` | Consider rotary/relative positions for cache reuse past `block_size` | proposed |
| `ADR-TEMPLATE.md` | Copy me for new decisions | — |

See [`research/design_decisions/`](.) and the concept behind each choice in `docs/`.
