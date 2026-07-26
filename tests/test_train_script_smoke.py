"""Smoke test that actually RUNS scripts/train.py end-to-end.

Why this exists: twice now, train.py shipped a bug that every unit test missed because
the tests never execute the script's main() —
  1. get_batch called with a string (silent random-data fallback -> flat loss), and
  2. a NameError on `lr_max` after a refactor.
Both would have been caught instantly by running the script for two steps. This test does
exactly that, in a tmp dir with a tiny synthetic dataset, so it's CI-safe (no network, no
real data) and never touches the real datasets/ directory.
"""
import sys
import subprocess
from pathlib import Path

import torch
import yaml


def test_train_script_runs_two_steps(tmp_path):
    # tiny synthetic dataset in a tmp data dir (never the real datasets/shakespeare)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.save(torch.randint(0, 65, (2000,)), data_dir / "train.pt")
    torch.save(torch.randint(0, 65, (500,)), data_dir / "val.pt")

    # tiny config: 2 steps, tiny model
    cfg = dict(vocab_size=65, n_layer=2, n_head=2, n_embd=32, block_size=16, dropout=0.0,
               weight_tying=True, batch_size=4, lr=1e-3, warmup_steps=1, max_steps=2,
               weight_decay=0.1, grad_clip=1.0, eval_interval=1, seed=0)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "train.py"),
         "--config", str(cfg_path), "--data_dir", str(data_dir),
         # MUST stay pointed at tmp_path. train.py used to hardcode 'checkpoints/' relative
         # to the CWD, so this test -- which runs from the repo root -- overwrote the real
         # banked checkpoint with its 2-step toy model on every suite run. That is how the
         # trained v0.2 weights were lost.
         "--out_dir", str(tmp_path / "checkpoints")],
        capture_output=True, text=True, cwd=repo_root, timeout=180,
    )
    # The script must run to completion (exit 0) and print a step line.
    assert result.returncode == 0, f"train.py crashed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "Step" in result.stdout, f"no training step logged:\n{result.stdout}"


def test_train_script_fails_loudly_without_data(tmp_path):
    """Missing data must raise, not silently train on noise (the flat-loss bug)."""
    cfg = dict(vocab_size=65, n_layer=2, n_head=2, n_embd=32, block_size=16, dropout=0.0,
               weight_tying=True, batch_size=4, lr=1e-3, warmup_steps=1, max_steps=2,
               weight_decay=0.1, grad_clip=1.0, eval_interval=1, seed=0)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "train.py"),
         "--config", str(cfg_path), "--data_dir", str(tmp_path / "does_not_exist")],
        capture_output=True, text=True, cwd=repo_root, timeout=60,
    )
    assert result.returncode != 0, "train.py should fail loudly when the dataset is missing"
    assert "not found" in (result.stdout + result.stderr).lower()


def test_training_never_writes_into_the_repo_checkpoints_dir(tmp_path):
    """The test suite must not be able to destroy a banked milestone checkpoint.

    This is a regression test for a real loss, not a hypothetical. train.py hardcoded
    `os.path.join('checkpoints', 'best_model.pt')` relative to the CWD, and the smoke
    tests above run with `cwd=repo_root`. So *every full pytest run* silently overwrote
    `checkpoints/best_model.pt` with a 2-step, 30,656-param toy model. `checkpoints/**` is
    gitignored, so there was no copy in version control and no way back: the trained v0.2
    weights (2,745,216 params, val loss 0.8607) were destroyed exactly this way.

    The fix is `--out_dir`, and this test is the guard on it -- it runs training from the
    repo root, the same as the smoke tests, and asserts the real artifact is untouched.
    """
    repo_root = Path(__file__).resolve().parents[1]
    real_ckpt = repo_root / "checkpoints" / "best_model.pt"
    before = real_ckpt.read_bytes() if real_ckpt.exists() else None

    cfg = dict(vocab_size=65, n_layer=2, n_head=2, n_embd=32, block_size=16, dropout=0.0,
               weight_tying=True, batch_size=4, lr=1e-3, warmup_steps=1, max_steps=2,
               weight_decay=0.1, grad_clip=1.0, eval_interval=1, seed=0)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.save(torch.randint(0, 65, (2000,), dtype=torch.long), data_dir / "train.pt")
    torch.save(torch.randint(0, 65, (500,), dtype=torch.long), data_dir / "val.pt")

    out_dir = tmp_path / "checkpoints"
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "train.py"),
         "--config", str(cfg_path), "--data_dir", str(data_dir),
         "--out_dir", str(out_dir)],
        capture_output=True, text=True, cwd=repo_root, timeout=180,
    )
    assert result.returncode == 0, f"train.py crashed:\nSTDERR:\n{result.stderr}"

    # It wrote where it was told...
    assert (out_dir / "best_model.pt").exists(), "no checkpoint written to --out_dir"

    # ...and nowhere else. Byte-compare rather than mtime: an overwrite with identical
    # content is still an overwrite, and mtime resolution can hide a fast one.
    after = real_ckpt.read_bytes() if real_ckpt.exists() else None
    assert after == before, (
        "training from the repo root modified checkpoints/best_model.pt. That path holds "
        "banked milestone weights, it is gitignored, and there is no way to recover it."
    )

