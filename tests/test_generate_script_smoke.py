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


# ── KV cache wiring (C1) ────────────────────────────────────────────────────────
#
# scripts/generate.py gained --use_cache, so the script-level counterpart to
# tests/test_kv_cache.py belongs here: --use_cache must be a pure speed knob. If it
# changes the generated text it is a bug, not an optimization (standing rule 5).

import pytest


def _run_generate(tmp_path, ckpt, meta, out, *extra, timeout=120):
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "generate.py"),
         "--checkpoint", str(ckpt), "--meta", str(meta), "--out", str(out), *extra],
        capture_output=True, text=True, cwd=repo_root, timeout=timeout,
    )


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """A small non-v0.1-shaped checkpoint plus matching tokenizer metadata.

    block_size=32 is small on purpose: it makes generating past the cache boundary cheap,
    which is where the two paths used to disagree.
    """
    vocab = list("abcdefghijklmnopqrst")
    ckpt_path, meta_path = tmp_path / "ckpt.pt", tmp_path / "meta.json"
    _make_checkpoint(ckpt_path, vocab_size=len(vocab), n_layer=2, n_head=2,
                     n_embd=24, block_size=32)
    CharTokenizer(vocab=vocab).save(meta_path)
    return ckpt_path, meta_path


def test_generate_script_runs_with_use_cache(tiny_checkpoint, tmp_path):
    """The cached path must actually execute and produce output."""
    ckpt, meta = tiny_checkpoint
    out = tmp_path / "cached.txt"

    result = _run_generate(tmp_path, ckpt, meta, out, "--prompt", "a",
                           "--max_new_tokens", "20", "--use_cache")

    assert result.returncode == 0, (
        f"generate.py --use_cache crashed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "kv_cache on" in result.stdout, result.stdout
    assert out.exists() and out.read_text(encoding="utf-8"), "no sample was written"


@pytest.mark.parametrize("max_new_tokens", [
    10,   # stays inside block_size=32
    120,  # well past it, where the cache stops being reused (ADR-0004)
])
def test_use_cache_produces_identical_text(tiny_checkpoint, tmp_path, max_new_tokens):
    """THE equivalence gate, at script level, on both sides of the block_size boundary.

    Past block_size is the interesting case: cached generation used to silently disagree
    with the naive path there, because cached K/V stay frozen at the absolute positions
    they were first embedded at while the naive path re-bases its sliding window. ADR-0004
    fixed that by dropping cache reuse at the boundary; this asserts the script-level
    consequence, so a regression shows up as different text rather than as a subtle
    quality drift nobody attributes to the flag.
    """
    ckpt, meta = tiny_checkpoint
    naive_out, cached_out = tmp_path / "naive.txt", tmp_path / "cached.txt"
    common = ["--prompt", "a", "--max_new_tokens", str(max_new_tokens),
              "--seed", "1234", "--temperature", "0.8", "--top_k", "10"]

    r_naive = _run_generate(tmp_path, ckpt, meta, naive_out, *common)
    r_cached = _run_generate(tmp_path, ckpt, meta, cached_out, *common, "--use_cache")

    assert r_naive.returncode == 0, r_naive.stderr
    assert r_cached.returncode == 0, r_cached.stderr

    naive_text = naive_out.read_text(encoding="utf-8")
    cached_text = cached_out.read_text(encoding="utf-8")
    assert naive_text == cached_text, (
        f"--use_cache changed the generated text at max_new_tokens={max_new_tokens}.\n"
        f"naive : {naive_text!r}\ncached: {cached_text!r}"
    )


def test_seed_flag_actually_controls_sampling(tiny_checkpoint, tmp_path):
    """--seed must mean something.

    load_checkpoint restores the RNG state stored in the checkpoint, so seeding before the
    load silently discarded --seed: every run replayed the trainer's RNG state and two
    different --seed values produced byte-identical text.
    """
    ckpt, meta = tiny_checkpoint
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    common = ["--prompt", "a", "--max_new_tokens", "60",
              "--temperature", "0.8", "--top_k", "10"]

    assert _run_generate(tmp_path, ckpt, meta, a, *common, "--seed", "7").returncode == 0
    assert _run_generate(tmp_path, ckpt, meta, b, *common, "--seed", "999").returncode == 0

    assert a.read_text(encoding="utf-8") != b.read_text(encoding="utf-8"), (
        "different --seed values produced identical text, so --seed is being ignored"
    )


def test_same_seed_is_reproducible(tiny_checkpoint, tmp_path):
    """The other half of the seed contract: same seed, same text."""
    ckpt, meta = tiny_checkpoint
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    common = ["--prompt", "a", "--max_new_tokens", "40", "--seed", "42",
              "--temperature", "0.8", "--top_k", "10"]

    assert _run_generate(tmp_path, ckpt, meta, a, *common).returncode == 0
    assert _run_generate(tmp_path, ckpt, meta, b, *common).returncode == 0

    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
