import torch
from jimmylabs.model.gpt import GPT

@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None, top_p: float = None, use_cache: bool = False) -> torch.Tensor:
    """
    Autoregressive generation loop taking O(N) forward passes without KV cache (Naive phase)
    or O(T) with KV cache.
    """
    block_size = model.config.block_size
    past_key_values = None
    
    for _ in range(max_new_tokens):
        # The cache is only reusable while the context is still *growing*.
        #
        # This model uses learned absolute positions (embedding.py), and a cached
        # K/V is frozen at the position its token had when first embedded. Once the
        # context outgrows block_size the window has to slide, and the naive path
        # re-embeds that window from position 0 — so every cached entry is now one
        # position stale, and one more per slide. Trimming the cache hides this:
        # the offset still *looks* right because it is derived from the cache
        # length, while the retained keys silently disagree about where they are.
        #
        # There is nothing to salvage by re-slicing; with absolute positions a slid
        # window must be re-embedded. So stop caching at the boundary and fall back
        # to the naive sliding-window forward, which is correct by construction.
        # (Relative/rotary positions would lift this restriction — see ADR-0004.)
        cache_usable = use_cache and idx.size(1) < block_size

        if cache_usable and past_key_values is not None:
            idx_cond = idx[:, -1:]
        else:
            past_key_values = None
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # Forward pass to get logits for the sequence
        if cache_usable:
            logits, _, past_key_values = model(idx_cond, use_cache=True, past_key_values=past_key_values)
        else:
            logits, _ = model(idx_cond)
        
        # We only care about the last step's logits
        logits = logits[:, -1, :] # (B, V)
        
        # Temperature scaling
        # Avoid division by zero by handling deterministic greedy decoding explicitly
        if temperature < 1e-5:
            # Deterministic greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
            continue
            
        logits = logits / temperature
        
        # Top-K sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # Everything less than the minimum value in the top-k is masked
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Top-p (Nucleus) sampling
        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter the mask back to the original ordering
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            
            # Apply the mask to logits to prevent sampling
            logits[indices_to_remove] = -float('Inf')
            
            # Recompute probs with the masked logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
        # Sample the next token
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        
        # Append to the sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx
