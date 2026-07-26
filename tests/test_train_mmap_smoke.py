"""Smoke + equivalence tests for scripts/train.py --use_mmap (backlog #7 wiring).

Two things are being proved here, and only one of them is a smoke test:

1. `train.py --use_mmap` actually runs end-to-end and checkpoints (standing rule 2 — this
   project has shipped three bugs in scripts that no test ever executed).
2. The loss trajectory is IDENTICAL with the flag on and off, for the same seed. A memory
   optimization that changes what the model learns is a bug, not an optimization
   (standing rule 5). This is the training-loop counterpart to the batch-level gate in
   tests/test_loader_mmap.py.

Note on the corpus: it is random token ids, so the loss legitimately sits flat near
ln(vocab) — there is nothing to learn in noise. That is expected here and must NOT be read
as the silent-random-data bug from the devlog; these tests assert equality between two
runs, never that the loss falls.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

STEP_RE = re.compile(
    r"Step\s+(\d+)/\d+ \| train loss ([\d.]+) \| val loss ([\d.]+)"
)


@pytest.fixture
def corpus_and_config(tmp_path):
    """A tiny real corpus on disk plus a tiny, fast training config."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.manual_seed(0)
    torch.save(torch.randint(0, 65, (50_000,)), data_dir / "train.pt")
    torch.save(torch.randint(0, 65, (5_000,)), data_dir / "val.pt")

    cfg = dict(vocab_size=65, n_layer=2, n_head=2, n_embd=32, block_size=16, dropout=0.0,
               weight_tying=True, batch_size=8, lr=1e-3, warmup_steps=2, max_steps=12,
               weight_decay=0.1, grad_clip=1.0, eval_interval=2, seed=1337)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return data_dir, cfg_path


def run_train(cfg_path, data_dir, *extra, timeout=600):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train.py"),
         "--config", str(cfg_path), "--data_dir", str(data_dir), *extra],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=timeout,
    )


def parse_losses(stdout):
    """Extract [(step, train_loss, val_loss)] from the training log."""
    return [(int(s), t, v) for s, t, v in STEP_RE.findall(stdout)]


def test_train_script_runs_with_mmap(corpus_and_config):
    """Smoke test: --use_mmap completes and checkpoints."""
    data_dir, cfg_path = corpus_and_config

    result = run_train(cfg_path, data_dir, "--use_mmap")

    assert result.returncode == 0, (
        f"train.py --use_mmap crashed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "Dataset: memory-mapped" in result.stdout, result.stdout
    assert "Step" in result.stdout, "no training step was logged"
    assert "Saved new best model" in result.stdout, "training never checkpointed"
    # The conversion must actually have produced the binaries it claims to use.
    assert (data_dir / "train.bin").exists(), "train.bin was never written"
    assert (data_dir / "val.bin").exists(), "val.bin was never written"


def test_mmap_and_in_memory_produce_identical_loss_trajectories(corpus_and_config):
    """THE equivalence gate: same seed, same losses, flag on vs off."""
    data_dir, cfg_path = corpus_and_config

    r_mem = run_train(cfg_path, data_dir)
    r_mmap = run_train(cfg_path, data_dir, "--use_mmap")

    assert r_mem.returncode == 0, r_mem.stderr
    assert r_mmap.returncode == 0, r_mmap.stderr

    losses_mem = parse_losses(r_mem.stdout)
    losses_mmap = parse_losses(r_mmap.stdout)

    assert losses_mem, f"no losses parsed from in-memory run:\n{r_mem.stdout}"
    assert len(losses_mem) == len(losses_mmap), "different number of eval points"
    assert losses_mem == losses_mmap, (
        "--use_mmap changed the loss trajectory, so it changed what the model learns.\n"
        f"in-memory: {losses_mem}\nmmap     : {losses_mmap}"
    )


def test_memmap_conversion_is_cached_and_invalidated(corpus_and_config):
    """The conversion must happen once, be reused, and go stale when the corpus changes."""
    data_dir, cfg_path = corpus_and_config

    first = run_train(cfg_path, data_dir, "--use_mmap")
    assert first.returncode == 0, first.stderr
    assert "converting" in first.stdout, "first run should build the memmap"

    second = run_train(cfg_path, data_dir, "--use_mmap")
    assert second.returncode == 0, second.stderr
    assert "reusing cached memmap" in second.stdout, (
        f"second run should reuse the cached binary, not reconvert:\n{second.stdout}"
    )
    assert "converting" not in second.stdout, "memmap was rebuilt despite being current"

    # Re-preparing the dataset must invalidate the cache, or training would silently use
    # the previous corpus.
    train_pt = data_dir / "train.pt"
    train_pt.touch()
    import os, time
    os.utime(train_pt, (time.time() + 10, time.time() + 10))

    third = run_train(cfg_path, data_dir, "--use_mmap")
    assert third.returncode == 0, third.stderr
    assert "stale memmap" in third.stdout, (
        f"a newer train.pt should invalidate the cached binary:\n{third.stdout}"
    )


def test_mmap_still_fails_loudly_without_data(tmp_path):
    """Standing rule 1 must survive the new flag: missing data raises, never substitutes."""
    cfg = dict(vocab_size=65, n_layer=2, n_head=2, n_embd=32, block_size=16, dropout=0.0,
               weight_tying=True, batch_size=4, lr=1e-3, warmup_steps=1, max_steps=2,
               weight_decay=0.1, grad_clip=1.0, eval_interval=1, seed=0)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    result = run_train(cfg_path, tmp_path / "does_not_exist", "--use_mmap", timeout=120)

    assert result.returncode != 0, "train.py --use_mmap should fail loudly without data"
    assert "not found" in (result.stdout + result.stderr).lower()
