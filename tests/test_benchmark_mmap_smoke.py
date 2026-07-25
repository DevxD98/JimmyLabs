"""Smoke test that actually RUNS scripts/benchmark_mmap.py end-to-end.

Why this exists: this project has twice shipped a script with a NameError or a missing
import that every unit test passed straight through, because nothing executed the
script's main(). benchmark.py was one of them. So the new benchmark script gets the same
treatment as train.py (see tests/test_train_script_smoke.py): run it as a subprocess with
a tiny corpus in tmp_path, assert it exits 0 and prints the numbers it promises.

Kept tiny (a few hundred thousand tokens) so it is CI-safe — the real measurement uses
the full 414.7M-token corpus and is recorded in benchmarks/006_mmap_dataset.md.
"""
import subprocess
import sys
from pathlib import Path


def run_benchmark(tmp_path, *extra):
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "benchmark_mmap.py"),
         "--data_dir", str(tmp_path / "corpus"),
         "--tokens", "200000", "--batches", "5",
         "--block_size", "16", "--batch_size", "4", *extra],
        capture_output=True, text=True, cwd=repo_root, timeout=600,
    )


def test_benchmark_mmap_runs_both_paths(tmp_path):
    result = run_benchmark(tmp_path, "--both")

    assert result.returncode == 0, (
        f"benchmark_mmap.py crashed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    # It must actually report the comparison, not just exit quietly.
    assert "RESULTS — corpus residency" in result.stdout, result.stdout
    assert "peak RSS reduction" in result.stdout, result.stdout
    assert "MACHINE STATE" in result.stdout, "machine state is required for reproducibility"


def test_benchmark_mmap_single_mode_emits_parseable_result(tmp_path):
    """--mode must print the machine-readable RESULT line that --both parses."""
    result = run_benchmark(tmp_path, "--mode", "mmap")

    assert result.returncode == 0, (
        f"benchmark_mmap.py crashed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "RESULT {" in result.stdout, result.stdout


def test_benchmark_mmap_requires_a_mode(tmp_path):
    """No mode is a usage error, not a silent no-op run."""
    result = run_benchmark(tmp_path)

    assert result.returncode != 0
    assert "--both" in (result.stdout + result.stderr)
