# AGENTS.md — briefing for AI agents contributing to JimmyLabs

You are probably an AI coding agent (Claude, Codex, Gemini, Cursor, or similar) that someone
has pointed at this repository. Read this file **in full before your first edit**. It is not
a style guide — it is the accumulated result of real bugs this project has shipped, and
every rule below exists because something broke.

Humans should read [`CONTRIBUTING.md`](CONTRIBUTING.md) instead; it covers the doc/ADR/
experiment/benchmark process. This file covers the things agents specifically get wrong.

---

## 1. What this project is

JimmyLabs is an **educational, documentation-first** GPT built from scratch on a MacBook Air
M1 (8 GB, MPS, no CUDA). The point is not to have a fast language model — you can download a
better one. The point is that every line is understood and every claim is measured.

That inverts some normal engineering instincts:

- **Documentation is the deliverable**, code is the demonstration. A clever undocumented
  optimization is worth less here than a plainly-written explanation of why it works.
- **Clarity beats speed.** Attention is hand-rolled rather than fused on purpose. Do not
  "optimize" it into an opaque one-liner.
- **A negative result is a result.** The bf16 experiment was rejected (15–31% *slower* on
  M1) and that rejection is recorded permanently, not deleted.

Read in this order before proposing anything: [`docs/21_DEVLOG.md`](docs/21_DEVLOG.md)
(the chronological history of every decision and bug — start here),
[`ENGINEERING_PRINCIPLES.md`](ENGINEERING_PRINCIPLES.md) (the constitution),
[`SPEC.md`](SPEC.md) (shapes, parameter math, version plan),
[`tests/TESTING_STRATEGY.md`](tests/TESTING_STRATEGY.md),
[`research/OPTIMIZATION_BACKLOG.md`](research/OPTIMIZATION_BACKLOG.md).

**Current state:** v0.2 — a 2,745,216-param model (L=6, h=6, C=192, block=256, vocab=138)
trained on TinyStories, best val loss **0.8607** (perplexity ≈2.37). Suite is **80 passed,
1 skipped**. CI green.

---

## 2. The ten standing rules, and the incident behind each

These are not negotiable, and each one is cheap to follow and expensive to skip.

**1. Never silently catch-and-substitute a bad input. Fail loudly.**
`train.py` once passed the string `'train'` where a data tensor belonged, and a `try/except`
substituted random tokens. It trained on noise for 5000 steps and every one of the 37 unit
tests passed. Later, `generate_baseline.py` substituted a *synthetic vocabulary* when
tokenizer metadata was missing and still wrote its output file — an artifact decoded against
the wrong character mapping, plausible-looking and worthless. If an input is missing or
wrong, raise, and name the fix in the message.

**2. Every script needs a real subprocess-level smoke test.**
Not a unit test of its importable functions — a test that *executes the script end-to-end*.
**Four** bugs have shipped specifically because nothing ever ran a script: `train.py`'s
`lr_max` NameError, `benchmark.py`'s missing `contextlib` import, `generate.py`'s
hardcoded-shape crash, and `generate_baseline.py`'s vocabulary substitution. Note that
`tests/test_mnist_smoke.py` only *imports* from `train_mnist.py` — despite the name it is
not script coverage, and `train_mnist.py` remains the one uncovered script.

**3. No invented numbers, ever.**
Every timing, memory, or quality figure must come from a real run you actually performed,
and must say what hardware and config produced it. Do not estimate, extrapolate, or repeat a
number from a paper as if you measured it. If you did not run it, write *(not measured)*.

**4. Any "faster path" needs an equivalence test proving IDENTICAL output.**
Not "close", not "similar quality" — identical, or bit-identical where the math allows.
See [`tests/test_kv_cache.py`](tests/test_kv_cache.py) and
[`tests/test_loader_mmap.py`](tests/test_loader_mmap.py).

**5. Never hardcode a shape or config that could be read from the artifact itself.**
`generate.py` built a throwaway model hardcoded to v0.1's shape just to read a checkpoint's
config; it worked by coincidence until the first v0.2 checkpoint, then hard-crashed. A
checkpoint is self-describing. So is a tokenizer — `tokenizer.vocab_size` exists, use it.

**6. Check for ADR and benchmark numbering collisions before merging.**
This has happened **three** times on parallel branches. Currently used: `ADR-0001`–`ADR-0006`
(next free: **0007**) and `benchmarks/001`–`006` (next free: **007**). Check the directories,
do not assume.

**7. Verify an "environment mismatch" report before assuming confusion — and verify *which
remote* it is about.** Agents working from forks have repeatedly been *correct* about their
own environment and *wrong about the world*. One reported that a fix "was not on main" — true
of its own fork, which was stale, and false of the real upstream. It then rebased onto an
already-merged branch and duplicated a day of work. If you are on a fork, add the upstream
remote explicitly and verify against **that**.

**8. Laptop safety.** Attention activation memory scales with `batch_size × block_size²`.
Smoke-test tiny before any real config. A `batch_size=64` / `block_size=256` run visibly
strained the target machine; the fix was a smaller `batch_size` plus `grad_accum_steps` to
preserve the same effective batch and token budget. Never launch a long training run without
being asked to.

**9. `torch.load()` defaults to `weights_only=False`.** Always pass `weights_only=True`.

**10. One task = one branch = one PR.** Never push to `main`. Never merge your own PR — a
human reviews and merges. Do not bundle unrelated changes; a reviewer who has to untangle
three concerns will reject all three.

---

## 3. What "verified" means here

The project's most expensive lesson is that **a passing test can be a vacuous test**. The KV
cache had an equivalence test that passed while the two paths computed different things —
it compared final sampled tokens under greedy argmax, which is robust enough to absorb a
~1e-2 logit shift. The bug survived a review and got a benchmark number published against it.

So, when you add a gate, you owe two things:

1. **Compare at the strongest available level.** Prefer raw logits or hidden states over the
   argmax of sampled output. `torch.equal` on the final tokens is the weakest possible check.
2. **Prove the gate is non-vacuous.** Temporarily reintroduce the bug — revert the fix,
   break the invariant — and confirm your test *fails*. A test never observed failing is a
   test you have not verified. State this proof in the PR description.

If you claim "it works", say precisely what you ran and paste the real output.

---

## 4. Environment traps that have actually cost time

**The venv is an editable install.** `pip install -e .` resolves `jimmylabs` to
`<repo>/src/jimmylabs`. If you run pytest from a **git worktree**, it will still import the
**main checkout's** library code — so edits to `src/` inside the worktree are silently
ignored and your verification is meaningless. Override with
`PYTHONPATH=<worktree>/src`, which wins over the editable install. (Changes confined to
`scripts/` and `tests/` are unaffected, since those resolve by path.)

**MPS vs CPU is a real behavioral difference, not just speed.** `torch.set_rng_state`
restores the **CPU** generator only. On MPS, sampling draws from the MPS generator seeded by
`torch.mps.manual_seed`. This made a genuine `--seed` bug *invisible on the Mac* while it was
live on CPU — the regression test fails on forced CPU and passes on MPS. Consequences:
a test can be vacuous locally and meaningful only in CI, and resumed MPS training is not
bit-reproducible (tracked as backlog #17). **CI runs Linux/CPU on Python 3.11; local
development is macOS/MPS on Python 3.14.** A local green run is not proof CI is green.

**`.freebuff/`** sometimes appears in the working tree. It is unrelated to this project and
deliberately not gitignored. Never stage or commit it. Stage files explicitly; avoid
`git add -A`.

---

## 5. Practical workflow

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install pytest
pytest -q                                    # 80 passed, 1 skipped

python scripts/prepare_data.py               # downloads + tokenizes tiny Shakespeare
python scripts/train.py --config configs/train_shakespeare.yaml
python scripts/generate.py --checkpoint checkpoints/best_model.pt
```

Useful flags, both **off by default** so existing behavior is unchanged:
`generate.py --use_cache` (KV cache) and `train.py --use_mmap` (memory-mapped corpus).

Before you open a PR:

- [ ] `pytest -q` passes, and you pasted the real output
- [ ] every script you touched has a subprocess smoke test that executes it
- [ ] every number you wrote came from a run you performed
- [ ] any "faster path" has an equivalence test, **proven non-vacuous**
- [ ] no ADR/benchmark numbering collision (check the directories)
- [ ] one concern, one branch; you did not push to `main` or merge yourself
- [ ] `.freebuff/` is not staged

---

## 6. When to stop and ask

Ask rather than guess when: a task requires a training run longer than a smoke test; you
believe a standing rule above is wrong for your case; you found a bug outside your task's
scope (report it, don't silently expand); the work would require retraining a banked model;
or your environment disagrees with this document — in which case **report the disagreement
with evidence** rather than working around it. Rule 7 exists because those reports have been
right before.

Finally: if you find a real bug while doing something else, **say so loudly and separately**.
Three of this project's most valuable findings came from agents noticing something adjacent
to their assigned task and reporting it instead of quietly routing around it.

---

*"every large model was once a small one that someone refused to stop understanding."*
