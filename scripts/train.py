import os
import time
import math
import yaml
import torch
import argparse
import contextlib
from pathlib import Path

from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.training.schedule import get_lr, clip_gradients
from jimmylabs.training.checkpoint import save_checkpoint
from jimmylabs.utils.seed import seed_everything
# Assuming we have the dataset loader from Phase 1
from jimmylabs.data.loader import get_batch, get_batch_mmap, open_memmap, save_memmap


def ensure_memmap(pt_path: Path, bin_path: Path) -> Path:
    """Convert a prepared *.pt token tensor to the flat binary the memmap loader reads.

    Cached: the conversion is skipped when `bin_path` already exists and is newer than
    `pt_path`. The mtime check (rather than mere existence) is what stops a stale binary
    from silently outliving the dataset it was built from — re-running a prepare script
    must invalidate it, or training would quietly continue against the previous corpus.

    Returns the path to the binary.
    """
    if bin_path.exists() and bin_path.stat().st_mtime >= pt_path.stat().st_mtime:
        print(f"  reusing cached memmap {bin_path}")
        return bin_path

    why = "stale" if bin_path.exists() else "missing"
    print(f"  {why} memmap -> converting {pt_path} to {bin_path} (one-time)")
    save_memmap(torch.load(pt_path, weights_only=True), bin_path)
    return bin_path


def main():
    parser = argparse.ArgumentParser(description="Train JimmyLabs GPT")
    parser.add_argument('--config', type=str, default='configs/train_shakespeare.yaml', help='Path to training config')
    parser.add_argument('--data_dir', type=str, default='datasets/shakespeare', help='Directory holding train.pt/val.pt')
    parser.add_argument('--use_mmap', action='store_true',
                        help='Read the corpus via a memory-mapped binary instead of loading '
                             'it fully into RAM (OPTIMIZATION_BACKLOG #7, benchmarks/006). '
                             'Default off; identical batches either way.')
    parser.add_argument('--out_dir', type=str, default='checkpoints',
                        help='Directory to write best_model.pt into. Tests MUST point this at '
                             'a temp dir: the path was previously hardcoded relative to the '
                             'CWD, so every smoke test that ran train.py from the repo root '
                             'silently overwrote the real banked checkpoint with a 2-step toy.')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    seed_everything(config_dict['seed'])
    
    # Select device
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
        
    print(f"Using device: {device}")
    
    # Initialize model
    model_config = GPTConfig(
        vocab_size=config_dict['vocab_size'],
        n_layer=config_dict['n_layer'],
        n_head=config_dict['n_head'],
        n_embd=config_dict['n_embd'],
        block_size=config_dict['block_size'],
        dropout=config_dict.get('dropout', 0.1),
        weight_tying=config_dict.get('weight_tying', True)
    )
    
    model = GPT(model_config)
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config_dict['weight_decay'],
        learning_rate=config_dict['lr'],
        device_type=device
    )
    
    # Training Loop params
    max_steps = config_dict['max_steps']
    warmup_steps = config_dict['warmup_steps']
    lr_max = config_dict['lr']
    grad_clip = config_dict['grad_clip']
    eval_interval = config_dict['eval_interval']
    grad_accum_steps = config_dict.get('grad_accum_steps', 1)
    batch_size = config_dict['batch_size']
    block_size = config_dict['block_size']

    # Load the prepared dataset. Fail LOUDLY if it's missing — never silently train on
    # random data (a flat loss at ln(vocab) is the symptom of exactly that mistake).
    data_dir = Path(args.data_dir)
    train_path, val_path = data_dir / 'train.pt', data_dir / 'val.pt'
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Dataset not found in {data_dir}. Run `python scripts/prepare_data.py` first."
        )
    # Two ways to hold the corpus, one sampling contract. get_batch and get_batch_mmap
    # take the same arguments and, for the same RNG state, return bit-identical batches
    # (tests/test_loader_mmap.py) — so the choice below is purely about memory, never
    # about what the model sees.
    if args.use_mmap:
        print("Dataset: memory-mapped (--use_mmap)")
        train_data = open_memmap(ensure_memmap(train_path, data_dir / 'train.bin'))
        val_data = open_memmap(ensure_memmap(val_path, data_dir / 'val.bin'))
        get_batch_fn = get_batch_mmap
    else:
        print("Dataset: in-memory")
        train_data = torch.load(train_path, weights_only=True)
        val_data = torch.load(val_path, weights_only=True)
        get_batch_fn = get_batch

    # Setup checkpoints dir
    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    # Pre-zero gradients
    optimizer.zero_grad(set_to_none=True)
    
    t0 = time.time()
    time_start = time.time()
    
    # Setup mixed precision context
    dtype_str = config_dict.get('dtype', 'fp32')
    ptdtype = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[dtype_str]
    ctx = torch.autocast(device_type=device, dtype=ptdtype) if dtype_str != 'fp32' else contextlib.nullcontext()
    
    for step in range(1, max_steps + 1):
        
        # 1. Update learning rate
        lr = get_lr(step, max_steps, warmup_steps, lr_max)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # 2. Forward, Backward & Accumulate over grad_accum_steps
        for micro_step in range(grad_accum_steps):
            X, Y = get_batch_fn(train_data, block_size, batch_size, device)
                
            with ctx:
                logits, loss = model(X, Y)
            
            # Scale loss for mathematical equivalence to a large batch
            loss = loss / grad_accum_steps
            
            loss.backward()
            
        # 3. Clip gradients (applied to accumulated gradients)
        clip_gradients(model, grad_clip)
        
        # 4. Optimizer Step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # 5. Evaluation & Checkpointing
        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                X_val, Y_val = get_batch_fn(val_data, block_size, batch_size, device)
                    
                _, val_loss_tensor = model(X_val, Y_val)
                val_loss = val_loss_tensor.item()
                # Unscale the loss of the last micro-step for reporting
                train_loss = loss.item() * grad_accum_steps
                
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                
                print(f"Step {step:4d}/{max_steps} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | lr {lr:.4e} | time {dt:.2f}s")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(args.out_dir, 'best_model.pt')
                    save_checkpoint(checkpoint_path, model, optimizer, config_dict, step, val_loss)
                    print(f"Saved new best model with val loss {val_loss:.4f} to {checkpoint_path}")
                    
            model.train()
            
    total_time = time.time() - time_start
    print(f"Training completed in {total_time/60:.2f} minutes.")

if __name__ == '__main__':
    main()
