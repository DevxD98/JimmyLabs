# Benchmark 004 — Fused AdamW (detection fix)

| Field | Value |
|-------|-------|
| ID | 004 |
| Date | 2026-07-25 |
| Type | optimization (result: **no measurable win at v0.1 scale**) |
| Baseline | `benchmarks/001_baseline.md` |

## 1. What & why

`GPT.configure_optimizers` intends to use PyTorch's fused AdamW on accelerators:

```python
use_fused = device_type in ('cuda', 'mps') and hasattr(torch.optim.AdamW, 'fused')
```

`fused` is a **constructor keyword**, not an attribute of the class, so
`hasattr(torch.optim.AdamW, 'fused')` is `False` on every torch build. The optimization
had therefore never run — on any machine, in any of the numbers recorded in 001–003.

This benchmark answers the question that fixing the detection raises: *does turning it on
actually help?*

## 2. Machine state

- Apple Silicon · Darwin 27.0.0 (arm64) · Python 3.12.12 · **torch 2.13.0**
- Device: **mps** · dtype: **fp32**
- Protocol: 35 steps, first 10 discarded as warmup, **median** of the rest,
  `torch.mps.synchronize()` around each timed region.

Note: `scripts/benchmark.py` builds a plain `torch.optim.AdamW(model.parameters())` and
never calls `configure_optimizers`, so it cannot measure this. Measured with a direct
harness driving the real `configure_optimizers` path instead.

## 3. Configuration

- Model: L=4, h=4, C=128, block=128 → **818,048 params** (v0.1, `configs/train_shakespeare.yaml`)
- Batch size 16 · sequence length 128 → 2,048 tokens/step
- Seed 1337, dropout 0.1. One variable changes: the `fused` kwarg.

## 4. Results (median, warm)

| Arm | ms/step | Throughput |
|-----|---------|------------|
| fused **off** (behaviour before the fix) | **22.00 ms** | 93,092 tok/s |
| fused **on** (after the fix) | **22.07 ms** | 92,775 tok/s |

**Delta: −0.3% — a wash, inside run-to-run noise.**

Correctness check: overfitting a single batch produces an *identical* loss trajectory on
both arms (`4.188 → 1.360 → 0.234 → 0.076` at steps 0/20/40/59), so enabling fused changes
speed, not results.

## 5. Interpretation

- **No win at this scale, and that is expected.** Fusing AdamW collapses the per-parameter
  elementwise update into one kernel. At 818K parameters that update is a rounding error
  next to the forward/backward pass, so there is nothing to recover. The win arrives when
  the optimizer step is a real share of step time — much larger models, or CUDA.
- **The fix is still worth landing.** The bug is that the code cannot do what it says: a
  reader (or a future benchmark) would reasonably assume fused was active on MPS. Correct
  detection with a measured "no effect here" is honest; a dead branch that looks alive is
  not.
- **Reasonable alternative:** delete the fused branch entirely as speculative complexity.
  This record exists so that choice can be made from data rather than assumption. Left
  enabled because it is behaviour-neutral and engages automatically if the model grows.
- **`hasattr` vs signature.** The test added in `tests/test_train_step.py` asserts against
  `inspect.signature(torch.optim.AdamW).parameters` rather than a hardcoded `True`, so it
  stays correct on torch builds that add or drop fused support.

## 6. Reproduce

```bash
python -m pytest tests/test_train_step.py::test_fused_adamw_detection_matches_torch_support -q
```

The timing harness is not checked in (it duplicates `scripts/benchmark.py`'s protocol
against `configure_optimizers`); making `scripts/benchmark.py` use the real optimizer path
would let this be reproduced with the standard tool and is worth a follow-up.
