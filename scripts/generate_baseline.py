import os
import torch
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.inference.generate import generate
from jimmylabs.tokenizer.char import CharTokenizer
from jimmylabs.utils.seed import seed_everything

def main():
    # 1. Tokenizer FIRST — it is the source of truth for vocab_size, and the baseline
    # artifact is only meaningful if it was decoded with the same vocabulary the trained
    # model will use. Missing metadata is a hard error, never a substituted vocabulary:
    # a synthetic fallback vocab silently writes an untrained_baseline.txt decoded against
    # the wrong character mapping, which looks plausible and is worthless as a comparison.
    meta_path = os.path.join('datasets', 'shakespeare', 'meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Tokenizer metadata not found at {meta_path}. "
            "Run `python scripts/prepare_data.py` first — this script will not substitute "
            "a synthetic vocabulary, because the resulting baseline would be decoded with "
            "the wrong character mapping and silently invalid as a comparison."
        )
    tokenizer = CharTokenizer.load(meta_path)

    # 2. Configuration (v0.1 model size) — vocab_size is read from the tokenizer, never
    # hardcoded, so it cannot drift from the corpus this baseline is compared against.
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=128,
        dropout=0.1,
        weight_tying=True
    )

    # 3. Setup
    seed_everything(42)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f"Initializing fresh untrained GPT model (v0.1, vocab_size={tokenizer.vocab_size})...")
    model = GPT(config)
    model.to(device)
    model.eval()

    # Start with a single newline character to kick off generation
    context = "\n"
    idx = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=device)
    
    print("Generating baseline output (200 tokens)...")
    # 4. Generate (naive autoregressive)
    out_idx = generate(model, idx, max_new_tokens=200, temperature=1.0, top_k=10)
    
    # Decode
    out_text = tokenizer.decode(out_idx[0].tolist())
    
    # 5. Save to outputs
    os.makedirs('outputs', exist_ok=True)
    out_path = os.path.join('outputs', 'untrained_baseline.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(out_text)
        
    print(f"\n--- Untrained Baseline Output ---\n{out_text}\n---------------------------------")
    print(f"Saved to {out_path}")
    
if __name__ == '__main__':
    main()
