# Benchmark 006 — Memory-mapped dataset (Optimization backlog #7)

| Field | Value |
|-------|-------|
| ID | 005 |
| Date | 2026-07-25 |
| Type | optimization |
| Backlog item | [#7 — Memory-map dataset](../research/OPTIMIZATION_BACKLOG.md) |
| Related | [`docs/17_DATASET_GUIDE.md`](../docs/17_DATASET_GUIDE.md) · [`SPEC.md` §6](../SPEC.md) |

> ⚠️ **Measured off-target.** Every other benchmark in this folder (001–003) was taken on
> the MacBook Air M1 / MPS. This one was taken on an **x86_64 Linux container (4 cores,
> 15 GB RAM, no MPS, no CUDA)** — the only machine available for this work. Read §2 before
> quoting any number here, and see §6 for exactly which conclusions survive the machine
> change and which do not. **Do not compare these figures to 001–003.**

## 1. What & why

`scripts/train.py` loads the corpus with `torch.load(train.pt)`, which makes the entire
token array resident for the whole run. That is fine for Shakespeare (~1M tokens) and
untenable for TinyStories (~414.7M tokens): as an int64 tensor the train split alone is
**3.3 GB of a machine that has 8 GB total**, competing directly with model activations.

`docs/17_DATASET_GUIDE.md` already states the intended design ("Datasets are streamed or
memory-mapped, never fully loaded into RAM"), and backlog #7 tracks it. This benchmark
measures whether the new memory-mapped path in `src/jimmylabs/data/loader.py`
(`get_batch_mmap`) actually delivers that, and by how much.

**This is a MEMORY benchmark, not a throughput benchmark.** The docs/14 MPS timing
protocol (warmup, `torch.mps.synchronize()`, median-of-N) does not apply — no GPU work is
being timed. What carries over from docs/14 is the honesty discipline: fixed seed,
recorded machine state, measured rather than estimated, written down.

## 2. Machine state

- Chip / platform: **x86_64**, `Linux-6.18.5-x86_64-with-glibc2.39` (container)
- Cores / RAM: **4 cores / 15 GB** — *not* the 8 GB M1 Air the project targets
- Python / PyTorch / NumPy: **3.11.15 / torch 2.13.0 / numpy 2.4.6**
- Device: **cpu** (no MPS, no CUDA available) · dtype: n/a (no model involved)
- Thermal state: n/a — no sustained compute; this measurement is allocation-bound, not
  thermally sensitive, which is the one reason it transfers across machines at all
- Filesystem readahead: `/sys/block/vda/queue/read_ahead_kb` = **8192** (relevant to §6)

## 3. Configuration

- Corpus: **414,700,000 tokens**, vocab 138 — the TinyStories train-split size recorded in
  `datasets/SOURCES.md`. **Synthetic** (seeded random ids), because residency is a function
  of corpus *length and dtype*, not of what the text says. This keeps the benchmark
  reproducible without a 1.9 GB download.
- Seed: **1337**
- Sampling: **200 batches**, B=16, T=256 (the v0.2 shape)
- Two storage formats compared:
  - `train.pt` — int64 tensor via `torch.save`, i.e. exactly what `scripts/prepare_data.py`
    writes today
  - `train.bin` — flat `uint16` binary via `save_memmap`

## 4. Metric definition

Peak RSS via `getrusage(RUSAGE_SELF).ru_maxrss`, normalized to MB (kB on Linux, bytes on
macOS). Each path runs in **its own subprocess** — peak RSS is a whole-process high-water
mark, so measuring both in one process would let the first contaminate the second.

Two measurement bugs were found and fixed while producing this record; both had to be
fixed rather than reported around:

1. **Inherited high-water mark.** A forked child inherits the parent's `ru_maxrss`. With
   corpus prep running in the parent, both paths reported an identical, meaningless
   3,755 MB "peak". Fixed by preparing the corpus in a separate subprocess *and* resetting
   the mark via `/proc/self/clear_refs` (`reset_peak_rss`).
2. **Total RSS hides the thing that matters.** Total RSS lumps together heap memory and
   mapped file pages, which behave completely differently under pressure. The record now
   reports the `/proc/self/status` split:
   - **RssAnon** — heap/tensor pages. The kernel *cannot* reclaim these; under pressure it
     can only swap them.
   - **RssFile** — clean page-cache pages mapped from a file. The kernel drops these
     instantly under pressure and re-faults them from disk on next use.

## 5. Results

Measured with one command (§7). Baseline (interpreter + torch import, ~510 MB) is reported
separately so the corpus-attributable figure is visible.

| Metric | before (in-memory) | after (mmap) | Δ |
|--------|-------------------:|-------------:|--:|
| **RssAnon — unreclaimable** | **3,522.5 MB** | **282.6 MB** | **−3,239.9 MB (12.5x lower)** |
| RssFile — reclaimable | 232.9 MB | 1,018.9 MB | +786.0 MB |
| Peak RSS (total) | 3,755.2 MB | 1,301.3 MB | −2,453.9 MB (2.9x lower) |
| — of which baseline | 510.2 MB | 510.3 MB | +0.1 MB |
| — corpus-attributable | 3,245.0 MB | 791.0 MB | −2,454.0 MB |
| Corpus on disk | 3,317.6 MB | 829.4 MB | −2,488.2 MB (4x smaller) |
| Corpus load time | 10.43 s | 0.00 s | −10.43 s |
| 200 batches | 0.10 s | 0.06 s | −0.04 s |

**Correctness gate:** `tests/test_loader_mmap.py::test_mmap_matches_in_memory_batches` and
`::test_mmap_matches_in_memory_over_many_draws` assert that, from the same seed, the two
paths return **bit-identical** batches (`torch.equal`, 25 consecutive draws). Both paths
draw start indices with the same `torch.randint` call against the same global RNG stream.
This mirrors the KV-cache equivalence gate in `tests/test_kv_cache.py`: an optimization
that changes results is a bug, not an optimization.

## 6. Interpretation

**The win is real, and the headline number is RssAnon, not total RSS.** Holding the corpus
as a resident int64 tensor costs **3.5 GB of unreclaimable memory** — on an 8 GB Air that
is ~44% of the machine gone before a single activation is allocated, and it is memory the
kernel can only swap, not drop. The mmap path moves that to **282.6 MB** of anon plus ~1 GB
of *clean, evictable* page cache. Under memory pressure the kernel reclaims the latter for
free, so the corpus stops competing with the model. This is precisely the property
`docs/17` asks for.

**Total RSS understates the win (2.9x) because RssFile counts pages the OS is holding on
our behalf, not memory we are denying to anything else.** The mmap run's ~1 GB of RssFile
is an artifact of the corpus having just been *written* by the prep step, leaving the whole
file hot in page cache; touching a window then maps an already-cached page in. Two things
confirm the page cache — not I/O readahead — is responsible: opening the memmap costs
**0.0 MB** (nothing is read at map time), and `madvise(MADV_RANDOM)` changes the figure not
at all (789.8 MB with and without). On a machine where the corpus is *not* already hot, the
same code faults in only the pages a batch touches.

**Secondary wins, both incidental but real:** storing ids as `uint16` instead of the int64
`torch.save` writes makes the corpus **4x smaller on disk** (3.3 GB → 829 MB), and startup
drops by **10.4 s** because there is no longer a multi-gigabyte deserialization before step
1. On a 3,375-step run that startup cost is noise; during iterative debugging it is not.

**What transfers to the M1 Air and what does not.** The memory results are allocation
behavior, not compute: a 414.7M-token int64 tensor is 3.3 GB on any machine, and macOS
likewise distinguishes anonymous from file-backed pages, so the *shape* of this result
holds. The exact MB figures do not transfer — macOS reports `ru_maxrss` in bytes, its page
cache and memory-pressure policy differ, and unified memory changes what "competing with
the model" means in practice. **The batch-sampling times must be ignored entirely**: they
are CPU-only, tiny, and measured on the wrong machine.

**Not measured here:** end-to-end training throughput with the mmap loader, which needs the
M1 and a real v0.2 run. `scripts/train.py` still uses the in-memory `get_batch` — this
change adds the capability and proves it equivalent; wiring it into the training script is
a separate, separately-benchmarked change (principle 7, one variable at a time).

**Next bottleneck:** with the corpus off the heap, the 8 GB ceiling is set by activations
and the O(B·T²) attention tensor again — i.e. backlog #9 (tuned batch/block), exactly as
`SPEC.md` §6 predicts.

## 7. Reproduce

```bash
python scripts/benchmark_mmap.py --both
```

Defaults are the values in §3 (414.7M tokens, vocab 138, B=16, T=256, 200 batches, seed
1337). The corpus is generated on first run into `datasets/_bench_mmap/` (git-ignored) and
reused afterwards; it needs ~4.2 GB of free disk. For a fast sanity run:

```bash
python scripts/benchmark_mmap.py --both --tokens 2000000 --batches 50
```

Correctness gate:

```bash
pytest -q tests/test_loader.py tests/test_loader_mmap.py
```
