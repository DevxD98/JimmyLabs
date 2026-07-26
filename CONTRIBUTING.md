# Contributing to JimmyLabs

Welcome to the lab. JimmyLabs is an educational, documentation-first project. To keep the repository focused on understanding, we operate under a strict constitution. 

Before you write any code, read the **[`ENGINEERING_PRINCIPLES.md`](ENGINEERING_PRINCIPLES.md)**. If a PR violates these principles, the principles win.

> **If you are an AI agent** — or you are directing one — read **[`AGENTS.md`](AGENTS.md)**
> instead of this file. This guide covers the human-facing doc/ADR/experiment/benchmark
> process; `AGENTS.md` covers the standing rules, the verification bar, and the environment
> traps that agents specifically get wrong.

---

## 1. The 12-Section Doc Template & Cross-References
Our documentation is the deliverable. Every concept is explained *before* it is built.
- Every new concept document in `docs/` must follow the strict 12-section template: `Purpose`, `Background`, `Concepts`, `Detailed Explanation`, `Visual Diagrams (ASCII)`, `Common Mistakes`, `Future Improvements`, `Glossary`, `Learning Checklist`, `References`, `Further Reading`, plus a top prerequisite/next nav strip.
- **Cross-reference rule:** Every concept must have one canonical defining document. When mentioning a concept (e.g., AdamW, block_size, softmax), link to its canonical home. Do not redefine it.

## 2. The ADR Flow (Architecture Decisions)
Every architectural decision that is non-trivial or hard-to-reverse must be documented.
- **Where:** [`research/design_decisions/`](research/design_decisions/)
- **What:** Write an Architecture Decision Record (ADR) detailing the context, the rejected options (this is the most valuable part), and the chosen path. Document the trade-offs.

## 3. The Experiment Flow
Experiments are the lifeblood of this project, but an unrecorded experiment is just a script.
- **Template:** Copy the experiment template from `research/experiment_templates/`.
- **Hypothesis First:** Write your hypothesis *before* you run the code.
- **Seed & Config:** Every experiment must use a fixed seed and a configuration file so that it is 100% reproducible.

## 4. The Benchmark Flow
No optimization is real until it is measured.
- **Before/After:** You must show a before-and-after measurement for any optimization PR.
- **Warmup & Median:** Follow benchmarking best practices. Warm up the hardware, run multiple trials, and report the median.
- **Location:** Benchmarks live in `benchmarks/` and follow the guidelines in [`docs/14_BENCHMARKING.md`](docs/14_BENCHMARKING.md).

## 5. The Testing Contract
We test rigorously. No `src/` module is done without tests.
- **Strategy:** Read [`tests/TESTING_STRATEGY.md`](tests/TESTING_STRATEGY.md).
- **Shape Tests First:** The most common bugs in transformers are tensor shape mismatches. Assert your shapes first, then move to known-answer tests, gradient checks, and overfit-a-batch tests.

## 6. PR Conventions
Keep pull requests focused and reviewable.
- **One Change Per Experiment:** Change one variable at a time. Confounded experiments teach nothing.
- **One PR Per Batch/Experiment:** Do not bundle multiple architectural changes or distinct documentation rewrites into a single PR.
- **Branches:** Create a new branch for your specific experiment or documentation batch.

---

*“every large model was once a small one that someone refused to stop understanding.”*
