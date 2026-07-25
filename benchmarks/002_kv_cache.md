# Benchmark 002 — KV Cache (Optimization C1)

| Field | Value |
|-------|-------|
| ID | 002 |
| Date | 2026-07-25 |
| Type | optimization |
| Baseline | `benchmarks/001_baseline.md` |

## 1. What & why
Implementation of KV caching for autoregressive generation (`src/jimmylabs/inference/generate.py` and `attention.py`). By caching the Key and Value tensors for previously generated tokens, we drop the computational complexity of generating the $N$th token from $O(T^2)$ to $O(T)$, avoiding redundant recalculation of past attention scores.

## 2. Machine state
- Mac / chip: MacBook Air M1
- macOS: **26.5.2** (arm64)
- Python / PyTorch: **torch 2.13.0**
- Device: **mps** · dtype: **fp32**
- Protocol: warmup=10 discarded, **median of 50**, `torch.mps.synchronize()` around each timed region

## 3. Configuration
- Model: L=4, h=4, C=128, block=128 → **818,048 params** (v0.1, `configs/train_shakespeare.yaml`)
- Batch size: 16 · sequence length: 128
- Generation: 200 new tokens from a 1-token prompt, using `--use_cache`

## 4. Results (median, warm — same-session naive vs cache)

Measured back-to-back with `scripts/benchmark.py` (3 runs each, all warm). We compare naive
vs cache **in the same session** rather than against 001, because 001's naive figure
(103 tok/s) was taken in a cooler thermal state; a same-session A/B is the honest delta
(docs/14).

| Metric | naive (`--use_cache` off) | KV cache (`--use_cache`) | Delta |
|--------|-------|-------|-------|
| **Generate throughput** | ~83 tok/s (81/84/85) | **~137 tok/s (137/138/137)** | **+65%** |
| **Train throughput** | 64,944 tok/s | 64,944 tok/s | unaffected (cache is inference-only) |

Correctness gate: `tests/test_kv_cache.py::test_kv_cache_equivalence` asserts the cached
output is **bit-identical** to the naive path (`torch.equal`, temp=0) — the optimization
changes speed, not results.

> ⚠️ **The correctness gate above did not hold.** `--gen_tokens 200` at
> `block_size=128` means 72 of those 200 steps ran past the context window, on a code
> path where the cached and naive logits disagree. The `+65%` in §4 was therefore
> measured against an incorrect implementation. See §7 for the corrected A/B, and
> [`ADR-0004`](../research/design_decisions/ADR-0004-no-cache-reuse-past-block-size.md)
> for why absolute positions make cache reuse impossible once the window slides.
> `torch.equal` at `temp=0` was too weak to notice: greedy argmax barely moves for a
> ~1e-2 logit shift.

## 5. Interpretation
- **Speedup is real and stable**: ~+65% generation throughput (naive ~83 → cache ~137
  tok/s), reproducible across runs. KV caching avoids recomputing keys/values for past
  tokens, so per-step work drops from $O(T)$ toward $O(1)$ (overall generation $O(T^2) \to
  O(T)$). At this tiny `block_size=128` the constant Python-loop overhead still caps the
  win; at larger context the gap widens sharply.
- **Correctness first**: the win only counts because the equivalence test proves cached ==
  naive output. A faster generator that changed the text would be a regression, not an
  optimization.
- **Note on the prior draft**: an earlier version of this record reported +25% (103→129);
  those figures predated a fix to `scripts/benchmark.py` (a missing `contextlib` import made
  it crash) and were not script-measured. The numbers above are measured with the fixed
  harness and reproduce.

## 6. Reproduce
```bash
python scripts/benchmark.py --use_cache
```

---

## 7. Correction (2026-07-25) — cost of making the cache correct

The §4 figures were measured while cached generation was silently diverging from naive
generation past `block_size` (see the warning in §4 and `ADR-0004`). This section re-runs
the A/B with the fix in place, so the speedup quoted is one the correctness gate actually
backs.

**Machine state** (differs from §2 — report your own rather than comparing across machines):
- Apple Silicon · Darwin 27.0.0 (arm64) · Python 3.12.12 · **torch 2.10.0**
- Device: **mps** · dtype: **fp32**
- Protocol: `--iters 15 --warmup 5 --gen_tokens 200`, 6 runs per arm, **median**, first
  (cold) run discarded — a cold start reads ~45 tok/s and is not representative.

| Arm | Generation throughput (median of 6) |
|-----|--------------------------------------|
| naive (`--use_cache` off) | **129 tok/s** (127 128 128 129 129 129) |
| KV cache — before fix, buggy trimming | **249 tok/s** (206 244 246 252 252 255) |
| KV cache — after fix | **240 tok/s** (239 239 240 241 242 243) |

- **Cost of correctness: ~3%** (249 → 240 tok/s).
- **Cache still wins big: +86%** over naive (129 → 240 tok/s) on this machine.
- The `+86%` here vs `+65%` in §4 is a machine/thermal difference, not an effect of the
  fix; only the two cache arms above are comparable to each other.

Why the cost is small: the cache disengages only *past* `block_size`. At
`gen_tokens=200, block_size=128` that is 72 of 200 steps, and those steps were already the
cheapest fraction of the run in wall-clock terms. Generating entirely within `block_size`
is unaffected.

**Reproduce the correction:**
```bash
python scripts/benchmark.py --iters 15 --warmup 5 --gen_tokens 200            # naive
python scripts/benchmark.py --iters 15 --warmup 5 --gen_tokens 200 --use_cache # cached
```
