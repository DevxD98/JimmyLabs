import os
import torch
from pathlib import Path
import sys
import pytest

# Add src to python path for imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
from jimmylabs.tokenizer.char import CharTokenizer

def test_prepare_tinystories(tmp_path):
    """
    Test the tokenization, splitting, and saving logic by mocking the input.
    """
    data_dir = tmp_path / "tinystories"
    data_dir.mkdir()
    
    input_file = data_dir / "TinyStories-train.txt"
    # Create a small dataset with multiple stories
    stories = [
        "This is story one.\n<|endoftext|>\n",
        "And this is story two which is a bit longer.\n<|endoftext|>\n",
        "Here is the third story, also quite nice.\n<|endoftext|>\n",
        "Story four is very very short.\n<|endoftext|>\n",
        "Story five is the last one in this tiny dataset.\n<|endoftext|>\n",
    ] * 20  # 100 stories total
    
    input_file.write_text("".join(stories), encoding="utf-8")
    
    # We will call the main of prepare_tinystories script directly by modifying sys.argv
    # or by importing and running it.
    import sys
    script_path = repo_root / "scripts" / "prepare_tinystories.py"
    
    # We can run it via subprocess to avoid namespace pollution
    import subprocess
    result = subprocess.run([
        sys.executable, str(script_path), 
        "--data_dir", str(data_dir),
        "--test-mode"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Script failed: {result.stdout} {result.stderr}"
    
    # Verify artifacts
    train_path = data_dir / "train.pt"
    val_path = data_dir / "val.pt"
    meta_path = data_dir / "meta.json"
    
    assert train_path.exists()
    assert val_path.exists()
    assert meta_path.exists()
    
    # Load them
    tokenizer = CharTokenizer.load(str(meta_path))
    train_tensor = torch.load(train_path, weights_only=True)
    val_tensor = torch.load(val_path, weights_only=True)
    
    # Verify split logic
    total_tokens = len(train_tensor) + len(val_tensor)
    train_ratio = len(train_tensor) / total_tokens
    
    # Should be around 90%, but since it's random by story, it might fluctuate.
    # Given 100 stories, 90/10 split should be between 0.8 and 1.0
    assert 0.8 <= train_ratio <= 1.0, f"Split ratio {train_ratio} is wildly off"
    
    # Verify tokens are in range
    assert train_tensor.max().item() < tokenizer.vocab_size
    assert train_tensor.min().item() >= 0
    assert val_tensor.max().item() < tokenizer.vocab_size
    assert val_tensor.min().item() >= 0
    
    # Verify roundtrip
    # We decode a chunk of the validation set to ensure it's readable text
    sample_ids = val_tensor[:20].tolist()
    text = tokenizer.decode(sample_ids)
    assert len(text) == 20
    assert isinstance(text, str)
