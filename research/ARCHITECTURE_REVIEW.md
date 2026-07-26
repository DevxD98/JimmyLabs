# Architecture Review — JimmyLabs (v0 design)

> **Reviewer stance:** written as a Staff ML Engineer would review this before any serious
> code is committed. The job of this document is to be *useful*, not kind: to find
> weaknesses, overengineering, missing pieces, and educational gaps while they're still
> cheap to fix — in docs, not in rewritten code.
>
> **Scope reviewed:** [`docs/00`–`02`](../docs/), the flagship docs
> ([`03`](../docs/03_LEARNING_PATH.md), [`08`](../docs/08_ATTENTION.md),
> [`09`](../docs/09_TRANSFORMER.md)), and the folder architecture as of the first scaffold
> commit.

---

## Executive summary

The foundation is **strong and unusually disciplined for a hobby-scale project**: the
documentation-first stance, the ADR/experiment/benchmark scaffolding, and the "why-before-
how" flagship docs are exactly what most educational GPT repos lack. The core architecture
choices (decoder-only, pre-norm, learned positions, char tokenizer first) are correct for
the goal and the hardware.

The main risks are not in *what* is chosen but in three gaps: **(1)** the project has no
explicit numeric budget yet (params, memory, throughput targets) — so "will it fit in 8 GB"
is currently a hope, not a calculation; **(2)** the folder architecture is slightly
over-built relative to the code that exists (a real, minor overengineering risk for an
educational repo); and **(3)** a few high-value educational components are missing from the
plan. All three are addressed below and, where possible, routed to the new
[`SPEC.md`](../SPEC.md).

Overall grade of the *design*: **strong pass, with required fixes before code.**

---

## What's working (keep, don't touch)

- **Decoder-only + causal self-attention + pre-norm.** The right, modern, minimal choice.
  Pre-norm over post-norm is correctly justified in [`docs/09`](../docs/09_TRANSFORMER.md)
  and belongs in an ADR (see gaps).
- **Character tokenizer first** ([`ADR-0001`](design_decisions/ADR-0001-character-tokenizer-first.md)).
  Correct sequencing for an educational project: removes a whole class of bugs from the
  critical early path.
- **Documentation-first + ADRs + experiment/benchmark templates.** This is the project's
  real differentiator. Keep the discipline; it will matter most exactly when momentum
  tempts you to skip it.
- **The `docs/` dependency graph and gated learning path.** Genuinely better than the
  "here's the code, let me narrate it" norm.

---

## Weaknesses & required fixes (before writing code)

### R1 — No numeric budget. *(severity: high)*
The docs say "1–4M params, 8 GB" but never do the arithmetic. Without it, `n_layer`,
`n_head`, `n_embd`, `block_size`, and batch size are guesses, and the O(T²) attention
tensor — often the real memory ceiling — is invisible.
**Fix:** [`SPEC.md`](../SPEC.md) now computes parameter counts, activation memory, and the
attention-matrix cost for each model version. Any config must reference it.

### R2 — Weight tying is mentioned but not planned as default. *(severity: high, cheap)*
At a small `n_embd` with a small vocab, the token-embedding table and the output projection
are each `vocab × n_embd`. Tying them (sharing one matrix) is a **free** parameter and
memory saving with typically neutral-or-better quality, and it's standard in GPT-2. It
should be the *default*, not a Phase-4 "optimization."
**Fix:** make weight tying the v0.1 baseline; the "optimization" framing in
[`docs/01`](../docs/01_ROADMAP.md) undersells it. Log it in the
[optimization backlog](OPTIMIZATION_BACKLOG.md) as already-applied.

### R3 — Folder architecture is slightly overengineered for current code. *(severity: medium)*
Fifteen top-level folders before a single line of `src/` exists is a mild smell — the exact
"unnecessary abstraction" a reviewer flags. It's defensible here *because the folders hold
documentation that's genuinely being written*, but the risk is empty-scaffold theater.
**Fix (accepted trade-off, monitored):** keep the tree, but each folder must earn its keep
with real content this quarter; any folder still holding only a stub README at the v0.1
milestone gets merged or deleted. Tracked as a checklist item in
[`docs/20_TODO.md`](../docs/20_TODO.md).

### R4 — `notes/` vs `research/` vs `docs/` boundary can rot. *(severity: low-medium)*
Three places for prose invites duplication — the very thing the cross-reference rule
forbids.
**Fix:** the rule is already stated (concept defined once, linked elsewhere); add a
lint-in-spirit check to the [testing strategy](../tests/TESTING_STRATEGY.md): a term defined
in two docs is a bug.

### R5 — Evaluation is underspecified. *(severity: medium)*
The roadmap trains on TinyStories/Shakespeare but never says how "good" is measured beyond
"coherent text." Validation loss alone is a weak signal at this scale.
**Fix:** [`SPEC.md`](../SPEC.md) defines the eval contract — held-out cross-entropy *and*
perplexity, plus a small fixed prompt set generated at every checkpoint for qualitative
tracking, stored in [`outputs/`](../outputs/).

---

## Missing components (educational gaps worth adding)

These aren't bugs — they're opportunities the current plan leaves on the table.

| Missing | Why it matters educationally | Where it should land |
|---------|------------------------------|----------------------|
| **A tiny autograd toy** (scalar, ~Karpathy micrograd) | Backprop is *understood* by building it once, then trusting PyTorch's | Phase 0, `docs/05` companion |
| **Explicit gradient-checking** | Proves your hand-derived understanding matches autograd | [`tests/TESTING_STRATEGY.md`](../tests/TESTING_STRATEGY.md) |
| **Attention visualization** | Seeing heads specialize is the single most convincing "it works" moment | new `scripts/visualize_attention.py` (Phase 4), `outputs/` |
| **A generation *before* training baseline** | Random-weight samples make "learning" visceral by contrast | [`outputs/`](../outputs/) |
| **Determinism harness** | Reproducibility (principle 4) needs a seed-everything utility, incl. MPS caveats | `src/` util + spec |
| **Model card** | Professional repos ship one; forces you to state data, limits, intended use | root `MODEL_CARD.md` at v1.0 |

The attention-visualizer, in particular, turns the abstract "heads learn different
relationships" claim from [`docs/08`](../docs/08_ATTENTION.md) into a picture — high
educational ROI for low effort. This aligns with the "build a research *platform*, not just
a model" framing.

---

## Overengineering watch (things NOT to add)

A good review also prevents work. For a 1–4M-param educational model on 8 GB, **do not**:

- Build a custom autograd engine *for the real model* (build the toy, then use PyTorch —
  reinventing autograd for production here is pure yak-shaving).
- Add distributed/multi-GPU anything. There is one GPU. There will always be one GPU.
- Implement FlashAttention from scratch early. Understand it, cite it, and only reach for a
  tiled implementation if benchmarks prove the O(T²) memory term is your actual ceiling.
- Add a config framework (Hydra/etc.). A commented YAML + a small loader is enough and more
  readable (principle 9).
- Support multiple tokenizers before one works. BPE is a *later, benchmarked* upgrade
  ([`ADR-0001`](design_decisions/ADR-0001-character-tokenizer-first.md)).

---

## Scalability note (honest framing)

This architecture does **not** scale to a large model, and that's fine — it's not supposed
to. The relevant "scalability" here is **scaling understanding and experimentation**, not
parameters. The design supports that well: shape-invariant blocks make depth/width trivially
adjustable, and the config-driven approach makes sweeps easy. The one real scaling ceiling
to respect is O(T²) attention memory vs. 8 GB — quantified in [`SPEC.md`](../SPEC.md).

---

## Recommended better architecture (concrete v0.1 baseline)

Rather than "1–4M params" as a range, commit to a **specific, computed** starting point and
grow deliberately:

```
   v0.1  (first real model — target: fits training comfortably in 8 GB)
   ─────────────────────────────────────────────────────────────
   tokenizer     character-level  (vocab ≈ 65–100 for Shakespeare)
   n_layer       4
   n_head        4
   n_embd        128        (d_k = 32 per head)
   block_size    128
   dropout       0.1
   weight tying  ON  (default, per R2)
   norm          pre-norm LayerNorm
   activation    GELU
   ≈ params      ~0.8–1.0M   (see SPEC.md for the arithmetic)
```

Then scale one axis at a time toward v1.0 (2–4M), logging each step as an experiment. Don't
jump to the biggest model that fits; grow into it so every increase is attributable.

---

## Verdict & required actions before code

**Verdict:** the design is sound and the discipline is exceptional. Proceed to code *after*
these fixes land in docs:

- [ ] **R1** — `SPEC.md` with real param/memory arithmetic (done in this batch).
- [x] **R2** — make weight tying the v0.1 default; update roadmap framing.
- [ ] **R3** — commit to the "every folder earns its keep by v0.1" checklist in `20_TODO`.
- [ ] **R5** — write the evaluation contract into `SPEC.md`.
- [x] Add ADRs for **pre-norm** and **weight-tying-by-default** (ADRs 0002 and 0003).
- [ ] Add the **attention visualizer** and **generation-before-training baseline** to the
      Phase plan.

> The best thing this project could do next is exactly what it's doing: design hard before
> coding. This review should save more rewriting than another thousand lines would have
> produced.

## See also

- [`SPEC.md`](../SPEC.md) · [`OPTIMIZATION_BACKLOG.md`](OPTIMIZATION_BACKLOG.md) ·
  [`ENGINEERING_PRINCIPLES.md`](../ENGINEERING_PRINCIPLES.md)
- [`docs/02_ARCHITECTURE.md`](../docs/02_ARCHITECTURE.md) — the design under review
