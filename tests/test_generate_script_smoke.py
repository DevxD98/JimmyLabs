"""Smoke test that actually RUNS scripts/generate.py end-to-end.

History: generate.py read a checkpoint's config by first constructing a model
HARDCODED to v0.1's shape (n_layer=4, n_embd=128, block_size=128, vocab_size=65) and
calling load_checkpoint() on it just to "peek" at the stored config -- but
load_checkpoint() always calls model.load_state_dict(strict=True) internally, so this
only ever worked by coincidence for v0.1-shaped checkpoints. The first real v0.2
checkpoint (a different shape) hard-crashed with a size-mismatch error. No test caught
it because nothing had ever run the script against a non-v0.1 checkpoint.

This test creates a small checkpoint at a DELIBERATELY DIFFERENT shape than v0.1's
hardcoded one and runs the real script against it via subprocess, so a future regression
of this exact "assumes one hardcoded shape" bug class fails immediately.
"""
import sys
import subprocess
from pathlib import Path

import torch

from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.tokenizer.char import CharTokenizer


def _make_checkpoint(path, vocab_size, n_layer, n_head, n_embd, block_size):
    """A minimal, valid checkpoint at an arbitrary (non-v0.1) shape."""
    config = GPTConfig(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                        n_embd=n_embd, block_size=block_size, dropout=0.0, weight_tying=True)
    model = GPT(config)
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': {},
        'config': {'vocab_size': vocab_size, 'n_layer': n_layer, 'n_head': n_head,
                   'n_embd': n_embd, 'block_size': block_size, 'weight_tying': True},
        'step': 1,
        'val_loss': 1.0,
        'rng_state': torch.get_rng_state(),
    }, path)


def test_generate_script_runs_on_a_non_v0_1_shaped_checkpoint(tmp_path):
    # Deliberately NOT v0.1's shape (4, 4, 128, 128, 65) -- this is what a hardcoded
    # "peek" model would fail size-mismatch against.
    vocab_size, block_size = 20, 32
    ckpt_path = tmp_path / "ckpt.pt"
    _make_checkpoint(ckpt_path, vocab_size=vocab_size, n_layer=2, n_head=2,
                     n_embd=24, block_size=block_size)

    meta_path = tmp_path / "meta.json"
    CharTokenizer(vocab=list("abcdefghijklmnopqrst")[:vocab_size]).save(meta_path)

    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "sample.txt"
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "generate.py"),
         "--checkpoint", str(ckpt_path), "--meta", str(meta_path),
         "--prompt", "a", "--max_new_tokens", "5", "--out", str(out_path)],
        capture_output=True, text=True, cwd=repo_root, timeout=60,
    )
    assert result.returncode == 0, (
        f"generate.py crashed on a non-v0.1-shaped checkpoint:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert out_path.exists() and out_path.read_text(), "no sample was written"
