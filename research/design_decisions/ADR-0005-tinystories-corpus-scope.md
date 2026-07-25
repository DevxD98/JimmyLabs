# ADR-0005 — Prepare the full TinyStories corpus, train on a Chinchilla-bounded slice of it

| Field | Value |
|-------|-------|
| Status | accepted |
| Date | 2026-07-25 |
| Deciders | JimmyLabs project |
| Related | [`docs/17_DATASET_GUIDE.md`](../../docs/17_DATASET_GUIDE.md) · [`research/paper_notes/chinchilla.md`](../paper_notes/chinchilla.md) · [`SPEC.md` §3](../../SPEC.md) · [`datasets/SOURCES.md`](../../datasets/SOURCES.md) · backlog [#7](../OPTIMIZATION_BACKLOG.md) / [benchmark 006](../../benchmarks/006_mmap_dataset.md) |

> **Implementation status:** written `proposed`, drafted against an environment where
> `scripts/prepare_tinystories.py`, `configs/model_v0_2.yaml`/`train_tinystories.yaml`, and
> `datasets/SOURCES.md`'s TinyStories rows did not yet exist — following principle 3 ("an
> ADR before or at the moment [decisions] are made"). All of that code and provenance
> already existed on the project's actual `main` branch by the time this ADR was merged
> (see `4a56a09`), and the corpus itself has since been downloaded and prepared (414.7M
> train / 46.6M val tokens, `uint16`). Flipped to `accepted` on merge — the decision this
> ADR documents is not hypothetical, it is what the shipped code does.

## Context

TinyStories is JimmyLabs's **primary corpus** ([`docs/17`](../../docs/17_DATASET_GUIDE.md)):
its deliberately simple English is the best capacity match for a 1–4M-parameter model, and
it is the reason coherent generation is plausible at all at this scale.

The corpus is far larger than the model can use. The numbers, which are what force the
decision:

```
   v0.2 model      L=6, h=6, C=192, block=256, vocab 138, weights tied
                   → 2,745,216 params   (SPEC.md §3; verified by constructing the model)

   Chinchilla ~20:1 token budget         (research/paper_notes/chinchilla.md)
                   2,745,216 × 20      ≈  54,904,320 tokens  ≈ 55M

   TinyStories prepared corpus          ~1.92 GB raw
                   train split          414,700,000 tokens
                   val split             46,600,000 tokens
                   total                461,300,000 tokens

   ratio           414.7M available train tokens ÷ ~55M budget  ≈  7.6x
                   461.3M prepared total         ÷ ~55M budget  ≈  8.4x
```

So a compute-optimal v0.2 run will touch roughly **one token in eight** of what preparation
produces. The question this ADR settles is whether the preparation step should download and
tokenize the whole corpus anyway, or bound itself up front to something near the budget.

The tension is real in both directions. Preparing everything spends disk, bandwidth and
one-time CPU on data no run will read. Preparing only a slice makes the cheap thing (a
one-time download) cheap, but makes the expensive thing (re-downloading and re-tokenizing
because a later experiment wants a bigger budget) recur exactly when experimental momentum
is highest. Note that the training budget is **not** in question here — it stays
Chinchilla-bounded either way; only the *stored corpus* scope is being decided.

## Options considered

### Option A — Prepare the full corpus; bound only the training budget

Download and tokenize all of TinyStories into `train.bin`/`val.bin`, and let
`max_steps × batch_size × block_size` in the training config define the ~55M-token budget
actually consumed.

- **Pros:** one-time cost; the raw corpus stays available, so raising the token budget for a
  later experiment (a longer v0.2 run, a larger v0.3) is a config edit rather than a
  re-download; corpus size does not affect training memory or step time, because `get_batch`
  samples random `(B, T)` windows and never iterates the array; the val split stays large
  enough to be a genuinely independent hold-out; the prepared artifact is a single
  well-defined object, which makes the `datasets/SOURCES.md` provenance row simple and
  honest ("the corpus", not "a subset produced by parameters X, Y, Z").
- **Cons:** ~830 MB of disk for the `uint16` train split (see below) and the bandwidth/CPU of
  a one-time full tokenization pass, in exchange for data ~7.6x of which a single run never
  reads.

### Option B — Bound the download/tokenization to a ~60M-character subset up front

Truncate at the source: fetch and tokenize only enough text to cover the Chinchilla budget
with a small margin, and never materialize the rest.

- **Pros:** minimal disk and bandwidth; the prepared artifact is close in size to what is
  actually consumed, so nothing looks wasteful; fastest possible first-run setup.
- **Cons:** the bound is a **new hyperparameter that silently determines what the model can
  ever see**, and getting it wrong is discovered late — a later experiment that wants a
  larger budget, or simply a different random window distribution, forces a re-download and
  re-tokenization, invalidating the prepared artifact mid-experiment; a truncated corpus
  makes provenance messier (the row must record the truncation rule, and two runs with
  different bounds are not comparable without reading it); sampling random windows from a
  corpus barely larger than the budget means heavy window overlap, so the run sees the same
  text repeatedly in a way a full corpus avoids; and it optimizes a **one-time** cost, which
  is the least valuable kind to optimize.

### Option C — Stream from the upstream source, never store the corpus

Tokenize on the fly from a streaming reader each run.

- **Pros:** near-zero disk footprint.
- **Cons:** every run depends on network availability and upstream stability, which breaks
  reproducibility outright — the same config and seed could see different data on different
  days, and `datasets/SOURCES.md` exists precisely to prevent that; re-tokenizing every run
  wastes far more CPU cumulatively than one full pass; it makes a fixed, seeded train/val
  split hard to guarantee. Rejected on reproducibility grounds (principle 4), independent of
  the disk question.

## Decision

**Option A — prepare the full corpus, bound the training budget in config.**

The single most important reason: **corpus size is decoupled from run cost.** `get_batch`
draws random windows and never touches the corpus proportionally to its length, so a corpus
8x larger than the budget costs nothing per step. Since backlog
[#7](../OPTIMIZATION_BACKLOG.md) landed, it costs almost nothing in RAM either —
[benchmark 006](../../benchmarks/006_mmap_dataset.md) measured the memory-mapped loader
holding the 414.7M-token corpus in **282.6 MB of unreclaimable memory instead of 3,522.5 MB**
(12.5x lower), and stored as `uint16` the train split is **829 MB on disk, not 3.3 GB**. What
Option B economizes on is therefore a one-time disk and bandwidth cost, on a machine that
typically has 19 GB+ free — while what it risks is a mid-experiment re-download, which is
exactly the kind of interruption that derails a run.

Put plainly: this trades a cheap, one-time, recoverable cost against an expensive,
recurring, badly-timed one.

## Consequences

- **Easier now:** raising the token budget for a later experiment is a `max_steps` edit, not
  a data-prep cycle; provenance in `datasets/SOURCES.md` describes one unambiguous artifact;
  the val split is large enough that hold-out loss is not sample-noise-limited; preparation
  has no tuning knob to get wrong.
- **Harder now:** first-run setup is slower and heavier than it strictly needs to be
  (a ~1.92 GB download plus a full tokenization pass, producing ~830 MB of `uint16` train
  split for a run that will read ~55M tokens); anyone with a tight disk budget or a metered
  connection pays for data that run will not use; and the gap between "corpus prepared" and
  "corpus trained on" is a genuine trap for a reader who assumes the model saw all of it —
  which is why it is written down here and must also be stated in `SOURCES.md` and in the
  training config's `max_steps` comment.
- **To watch:** two signals would reopen this. **(1)** Disk pressure — if a full prepared
  corpus stops comfortably fitting alongside checkpoints and the venv, revisit Option B with
  an explicitly recorded bound. **(2)** If v0.3+ settles into consistently reading a *larger*
  fraction of the corpus, the ratio this ADR is built on stops holding and the trade-off
  should be re-derived rather than inherited.

## Notes

This ADR deliberately does **not** decide the token budget itself — that follows from
Chinchilla and the parameter count, and belongs in the training config next to `max_steps`
with its arithmetic shown (principle 5). It decides only how much corpus is *prepared and
kept*.

The `uint16` on-disk format and the memory-mapped read path are not assumptions of this
decision but measured facts underpinning it; both come from backlog
[#7](../OPTIMIZATION_BACKLOG.md) and are recorded in
[benchmark 006](../../benchmarks/006_mmap_dataset.md). Had the corpus still had to be fully
resident as a 3.3 GB int64 tensor, the balance here would be much closer, and Option B would
deserve a second look — the decision leans on #7 having already been applied.
