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
         "--config", str(cfg_path), "--data_dir", str(data_dir)],
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

