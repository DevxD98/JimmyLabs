"""Generate text from a trained JimmyLabs checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "\n" \
        --max_new_tokens 400 --temperature 0.8 --top_k 40

Loads the model config straight from the checkpoint (a checkpoint is self-describing),
restores the trained weights, and generates with the naive autoregressive sampler.
"""
import os
import argparse
import torch

from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.inference.generate import generate
from jimmylabs.tokenizer.char import CharTokenizer
from jimmylabs.training.checkpoint import load_checkpoint
from jimmylabs.utils.seed import seed_everything


def main():
    p = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    p.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    p.add_argument('--meta', default='datasets/shakespeare/meta.json')
    p.add_argument('--prompt', default='\n')
    p.add_argument('--max_new_tokens', type=int, default=400)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=40)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use_cache', action='store_true',
                   help='Generate with the KV cache. Identical tokens, fewer FLOPs, and '
                        'automatically disabled once the context outgrows block_size '
                        '(ADR-0004). Default off, so existing behavior is unchanged.')
    p.add_argument('--out', default='outputs/trained_shakespeare_sample.txt')
    args = p.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # A checkpoint is self-describing: read its config directly (no model needed for this --
    # load_checkpoint() always calls model.load_state_dict(strict=True) internally, so it
    # can only be used AFTER a correctly-shaped model exists, never to "peek" at the shape.
    raw = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = raw['config']
    config = GPTConfig(
        vocab_size=cfg['vocab_size'], n_layer=cfg['n_layer'], n_head=cfg['n_head'],
        n_embd=cfg['n_embd'], block_size=cfg['block_size'],
        dropout=0.0, weight_tying=cfg.get('weight_tying', True),
    )
    model = GPT(config)
    step, val_loss, _ = load_checkpoint(args.checkpoint, model)
    model.to(device).eval()

    # Seed AFTER the checkpoint load, not before. load_checkpoint restores the RNG state
    # saved in the checkpoint (checkpoint.py), so seeding first meant --seed was silently
    # discarded and every run replayed the trainer's RNG state instead of the requested
    # one. Sampling must be controlled by the flag the user actually passed.
    seed_everything(args.seed)

    tokenizer = CharTokenizer.load(args.meta)

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = generate(model, idx, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_k=args.top_k,
                   use_cache=args.use_cache)
    text = tokenizer.decode(out[0].tolist())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"checkpoint: {args.checkpoint}  (step {step}, val_loss {val_loss:.4f})")
    print(f"temperature {args.temperature}, top_k {args.top_k}, seed {args.seed}, "
          f"kv_cache {'on' if args.use_cache else 'off'}\n")
    print("--- trained sample ---")
    print(text)
    print("----------------------")
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    main()
