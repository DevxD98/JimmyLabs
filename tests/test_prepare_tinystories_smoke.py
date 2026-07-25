"""Smoke test that actually RUNS scripts/prepare_tinystories.py end-to-end.
"""
import sys
import subprocess
from pathlib import Path

def test_prepare_tinystories_smoke(tmp_path):
    """Smoke test for prepare_tinystories.py to ensure it runs end-to-end without network."""
    data_dir = tmp_path / "tinystories"
    
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "prepare_tinystories.py"
    
    result = subprocess.run(
        [sys.executable, str(script_path), "--data_dir", str(data_dir), "--test-mode"],
        capture_output=True, text=True, cwd=repo_root, timeout=60,
    )
    
    assert result.returncode == 0, f"prepare_tinystories.py crashed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    
    # Verify it created the expected artifacts
    assert (data_dir / "train.pt").exists()
    assert (data_dir / "val.pt").exists()
    assert (data_dir / "meta.json").exists()
