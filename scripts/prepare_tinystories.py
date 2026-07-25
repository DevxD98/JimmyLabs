import os
import urllib.request
import torch
import json
import random
import array
from pathlib import Path
import sys

# Add src to python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jimmylabs.tokenizer.char import CharTokenizer

def download_file(url: str, dest_path: str):
    if not os.path.exists(dest_path):
        print(f"Downloading TinyStories from {url}...")
        # Since it's large, we'll download in chunks and show progress
        import urllib.request
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    else:
        print(f"File {dest_path} already exists, skipping download.")

def main():
    import argparse
    import shutil
    
    # Fail loudly if < 3GB free disk space
    free_space_gb = shutil.disk_usage(".").free / (1024**3)
    if free_space_gb < 3.0:
        raise RuntimeError(f"Insufficient disk space! Need at least 3GB, but only {free_space_gb:.2f}GB available.")
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.path.join('datasets', 'tinystories'))
    parser.add_argument('--test-mode', action='store_true', help='Use for smoke tests (skip large downloads)')
    args = parser.parse_args()

    dataset_dir = args.data_dir
    os.makedirs(dataset_dir, exist_ok=True)
    
    input_path = os.path.join(dataset_dir, 'TinyStories-train.txt')
    train_path = os.path.join(dataset_dir, 'train.pt')
    val_path = os.path.join(dataset_dir, 'val.pt')
    meta_path = os.path.join(dataset_dir, 'meta.json')
    
    # 1. Download
    if args.test_mode:
        print("Running in TEST MODE. Skipping massive download.")
        if not os.path.exists(input_path):
            with open(input_path, 'w', encoding='utf-8') as f:
                f.write("This is a tiny test story.\n<|endoftext|>\nThis is another test story.\n<|endoftext|>\n")
    else:
        url = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt'
        download_file(url, input_path)

    # 2. First pass: compute vocabulary efficiently
    print("Pass 1: Computing vocabulary...")
    vocab = set()
    with open(input_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(1024 * 1024 * 10)  # 10 MB chunks
            if not chunk:
                break
            vocab.update(chunk)
    
    tokenizer = CharTokenizer(corpus="".join(sorted(list(vocab))))
    tokenizer.save(meta_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Update SOURCES.md if we're not in test mode
    if not args.test_mode:
        sources_path = os.path.join('datasets', 'SOURCES.md')
        if os.path.exists(sources_path):
            with open(sources_path, 'r', encoding='utf-8') as f:
                sources = f.read()
            if "_tbd_ (will be filled after script runs)" in sources:
                sources = sources.replace("_tbd_ (will be filled after script runs)", str(tokenizer.vocab_size))
                with open(sources_path, 'w', encoding='utf-8') as f:
                    f.write(sources)
                print("Updated SOURCES.md with vocab size.")

    # 3. Second pass: Tokenize and Split
    print("Pass 2: Tokenizing and splitting (90/10 seeded)...")
    random.seed(42)
    
    # Use array for memory efficiency before creating tensors
    # 'B' is unsigned 1-byte char (uint8), perfect since vocab < 256
    train_arr = array.array('B')
    val_arr = array.array('B')
    
    current_story = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            current_story.append(line)
            if "<|endoftext|>" in line:
                story_text = "".join(current_story)
                ids = tokenizer.encode(story_text)
                if random.random() < 0.9:
                    train_arr.extend(ids)
                else:
                    val_arr.extend(ids)
                current_story = []
        
        # Handle the last story if it doesn't end with <|endoftext|>
        if current_story:
            story_text = "".join(current_story)
            ids = tokenizer.encode(story_text)
            if random.random() < 0.9:
                train_arr.extend(ids)
            else:
                val_arr.extend(ids)

    print(f"Train split: {len(train_arr):,} tokens (~90%)")
    print(f"Val split: {len(val_arr):,} tokens (~10%)")
    
    # 4. Save as PyTorch tensors
    print(f"Saving to {train_path} and {val_path}...")
    
    # Convert arrays to tensors using uint8
    train_tensor = torch.tensor(train_arr, dtype=torch.uint8)
    val_tensor = torch.tensor(val_arr, dtype=torch.uint8)
    
    torch.save(train_tensor, train_path)
    torch.save(val_tensor, val_path)
    print("Done!")

if __name__ == '__main__':
    main()
