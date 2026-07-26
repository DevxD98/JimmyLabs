"""Subprocess-level smoke tests for the scripts that still had none (standing rule 2).

Audit context: three bugs have shipped in this project specifically because a script was
never actually executed by a test — a NameError in train.py, a missing import in
benchmark.py, and a hardcoded-shape assumption in generate.py. At the time this file was
written, `scripts/benchmark.py` — one of those three — still had no test that ran it.

Coverage audit of scripts/ at that point:

    train.py                 covered  tests/test_train_script_smoke.py
    generate.py              covered  tests/test_generate_script_smoke.py
    generate_baseline.py     covered  tests/test_generate_baseline_script_smoke.py
    prepare_tinystories.py   covered  tests/test_prepare_tinystories_smoke.py
    benchmark_mmap.py        covered  tests/test_benchmark_mmap_smoke.py
    benchmark.py             GAP   -> covered here
    prepare_data.py          GAP   -> covered here
    visualize_attention.py   GAP   -> covered here (skips without matplotlib)
    train_mnist.py           GAP   -> still uncovered, see note below

`train_mnist.py` is deliberately left out: it takes no CLI arguments, hardcodes 5 epochs,
and unconditionally downloads MNIST, so a subprocess test would be slow and
network-dependent. Making it testable means giving the script argument parsing first,
which is a change to the script rather than to the tests. Note also that
tests/test_mnist_smoke.py, despite the name, only imports MNISTMLP and get_device — it
never executes main(), so it is not script coverage.

Every test here runs the real script via subprocess with a tiny config in a tmp cwd,
touches no network, and stays away from the repository's own datasets/ and outputs/.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

TINY_CFG = dict(vocab_size=65, n_layer=2, n_head=2, n_embd=32, block_size=16, dropout=0.0,
                weight_tying=True, batch_size=4, lr=1e-3, warmup_steps=1, max_steps=2,
                weight_decay=0.1, grad_clip=1.0, eval_interval=1, seed=0)


def run_script(name, *args, cwd=REPO_ROOT, timeout=600):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / name), *args],
        capture_output=True, text=True, cwd=cwd, timeout=timeout,
    )


def test_benchmark_script_runs(tmp_path):
    """scripts/benchmark.py — previously shipped a missing import with no test running it."""
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(TINY_CFG))

    result = run_script("benchmark.py", "--config", str(cfg_path),
                        "--warmup", "2", "--iters", "3", "--gen_tokens", "10")

    assert result.returncode == 0, (
        f"benchmark.py crashed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    # It must report machine state and real numbers — a benchmark that prints nothing
    # measurable is indistinguishable from one that silently did nothing.
    assert "MACHINE STATE" in result.stdout, result.stdout
    assert "train throughput" in result.stdout, result.stdout
    assert "gen throughput" in result.stdout, result.stdout
    assert "model params     : 28,064" in result.stdout, (
        f"benchmark did not report the tiny config's real param count:\n{result.stdout}"
    )


def test_benchmark_script_runs_with_kv_cache(tmp_path):
    """The --use_cache path must execute too, not just the default one."""
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(TINY_CFG))

    result = run_script("benchmark.py", "--config", str(cfg_path),
                        "--warmup", "2", "--iters", "3", "--gen_tokens", "10", "--use_cache")

    assert result.returncode == 0, (
        f"benchmark.py --use_cache crashed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "gen throughput" in result.stdout


def test_prepare_data_script_runs_offline(tmp_path):
    """scripts/prepare_data.py — end-to-end tokenize + split + save.

    A pre-seeded input.txt makes the download branch a no-op, so this never hits the
    network. Run in a tmp cwd because the script hardcodes relative dataset paths.
    """
    data_dir = tmp_path / "datasets" / "shakespeare"
    data_dir.mkdir(parents=True)
    (data_dir / "input.txt").write_text("hello world.\nthe quick brown fox.\n" * 200,
                                        encoding="utf-8")

    result = run_script("prepare_data.py", cwd=tmp_path)

    assert result.returncode == 0, (
        f"prepare_data.py crashed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    for artifact in ("train.pt", "val.pt", "meta.json"):
        assert (data_dir / artifact).exists(), f"prepare_data.py did not write {artifact}"

    import torch
    train = torch.load(data_dir / "train.pt", weights_only=True)
    val = torch.load(data_dir / "val.pt", weights_only=True)
    assert len(train) > 0 and len(val) > 0
    assert abs(len(train) / (len(train) + len(val)) - 0.9) < 0.01, "split is not 90/10"


def test_visualize_attention_script_runs(tmp_path):
    """scripts/visualize_attention.py — needs matplotlib and a real checkpoint.

    Skipped rather than omitted where matplotlib is absent (it is not a hard dependency in
    pyproject.toml), so this still provides coverage on a dev machine that has it.
    """
    pytest.importorskip("matplotlib", reason="visualize_attention.py requires matplotlib")

    import torch
    from jimmylabs.model.config import GPTConfig
    from jimmylabs.model.gpt import GPT
    from jimmylabs.tokenizer.char import CharTokenizer
    from jimmylabs.training.checkpoint import save_checkpoint

    corpus = "\n" + " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.:;!?'-"
    tokenizer = CharTokenizer(corpus=corpus)
    meta_path = tmp_path / "meta.json"
    tokenizer.save(str(meta_path))

    cfg = dict(TINY_CFG, vocab_size=tokenizer.vocab_size)
    model = GPT(GPTConfig(vocab_size=cfg['vocab_size'], n_layer=cfg['n_layer'],
                          n_head=cfg['n_head'], n_embd=cfg['n_embd'],
                          block_size=cfg['block_size'], dropout=0.0, weight_tying=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = tmp_path / "tiny.pt"
    save_checkpoint(str(ckpt), model, optimizer, cfg, step=1, val_loss=1.0)

    result = run_script("visualize_attention.py", "--ckpt", str(ckpt),
                        "--vocab", str(meta_path), "--prompt", "hello",
                        cwd=tmp_path)

    assert result.returncode == 0, (
        f"visualize_attention.py crashed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
