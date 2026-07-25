# Dataset Sources & Provenance

> The recipe, not the groceries. Raw corpora are git-ignored; this file records **exactly**
> where each came from, its license, and how it was prepared — so any training run is
> reproducible and legally clean (principle 4). **No corpus is trained on until its row here
> is filled in.** The rationale for these choices is in
> [`../docs/17_DATASET_GUIDE.md`](../docs/17_DATASET_GUIDE.md).

## Template (copy per corpus)

```
### <corpus name>
- Source URL:
- Version / snapshot date:
- Downloaded on:
- License:               (and link)
- Size (raw):
- Intended use:          (sanity / primary / stretch / experiment)
- Cleaning applied:      (encoding, whitespace, casing, control chars — list every choice)
- Vocab produced:        (char set size, for char tokenizer)
- Train/val split:       (ratio + seed)
- Prepared artifact:     (path to the git-ignored *.idx, and the command that made it)
- Notes / gotchas:
```

---

## Corpora

### Tiny Shakespeare  — *planned: sanity / overfit set*
- Source URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Version / snapshot date: 2026-07-24 (latest from master branch)
- Downloaded on: 2026-07-24
- License: **public domain** (Shakespeare's works)
- Size (raw): 1,115,394 bytes (~1.1 MB)
- Intended use: **sanity** — deliberate overfit to prove the training loop learns; first
  target for v0.1.
- Cleaning applied: None (original casing and whitespace preserved)
- Vocab produced: 65 characters
- Train/val split: 90/10 contiguous sequential split (no shuffling)
- Prepared artifact: `datasets/shakespeare/train.pt` and `datasets/shakespeare/val.pt`
- Notes: memorizes quickly — good for the overfit test, not a generalization claim.

### TinyStories — *planned: primary corpus*
- Source URL: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
- Version / snapshot date: 2026-07-25
- Downloaded on: 2026-07-25
- License: **CDLA-Sharing-1.0** (Community Data License Agreement, verified via HF dataset card)
- Size (raw): ~1.92 GB
- Intended use: **primary** — best capacity match for a 1–4M-param model.
- Cleaning applied: None (original casing and whitespace preserved, streamed directly)
- Vocab produced: 138 unique characters
- Train/val split: 90/10, seeded with 42
- Prepared artifact: `datasets/tinystories/train.pt` and `datasets/tinystories/val.pt`
- Notes: the reason we expect coherent output at all (see the TinyStories paper).

### WikiText-2 — *optional: stretch complexity*
- Status: candidate only; add full provenance if/when used.
- License: CC BY-SA (verify current terms).

### (excluded) OpenWebText / SlimPajama / FineWeb
- **Not used for training at this scale** — recorded here so the exclusion is deliberate and
  documented, not an oversight. Rationale: too large/diverse for a tiny model (mush), and
  impractical on an 8 GB fanless Air. See [`../docs/17_DATASET_GUIDE.md`](../docs/17_DATASET_GUIDE.md).

---

> Reminder: fill `_tbd_` fields **before** the corresponding training run, not after. A
> number produced on unrecorded data does not count (principles 2 & 4).
