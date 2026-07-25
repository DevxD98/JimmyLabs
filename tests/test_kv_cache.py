import pytest
import torch
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.inference.generate import generate

BLOCK_SIZE = 16
VOCAB_SIZE = 65


@pytest.fixture
def dummy_model():
    # Fixed seed: CONTRIBUTING.md §3 requires every test to be reproducible.
    # Without it both the model init and the prompt were random, so a failure
    # here could not be reproduced from the test alone.
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=VOCAB_SIZE, n_layer=2, n_head=2, n_embd=32, block_size=BLOCK_SIZE
    )
    model = GPT(config)
    model.eval()
    return model

def test_kv_cache_equivalence(dummy_model):
    """
    Generate with and without KV cache should produce EXACTLY the same tokens.
    """
    B, T = 1, 4
    idx = torch.randint(0, VOCAB_SIZE, (B, T))

    # Use greedy decoding to avoid sampling noise
    max_new_tokens = 20

    out_naive = generate(dummy_model, idx, max_new_tokens, temperature=0.0, use_cache=False)
    out_cached = generate(dummy_model, idx, max_new_tokens, temperature=0.0, use_cache=True)

    assert torch.equal(out_naive, out_cached), "KV cache generation does not match naive generation"


def _both_paths(model, idx, max_new_tokens, seed=0):
    """
    Run the cached and naive paths over the same RNG stream.

    Greedy decoding is too blunt to test this invariant: the sampled token only
    moves if the perturbation crosses the argmax margin, so a cache bug can shift
    every logit and still return identical tokens. Re-seeding before each call
    and sampling with top_k makes the *whole* distribution observable, so any
    logit drift shows up as a different token.
    """
    torch.manual_seed(seed)
    naive = generate(model, idx, max_new_tokens, temperature=1.0, top_k=5, use_cache=False)
    torch.manual_seed(seed)
    cached = generate(model, idx, max_new_tokens, temperature=1.0, top_k=5, use_cache=True)
    return naive, cached


@pytest.mark.parametrize("seed", range(4))
def test_kv_cache_equivalence_within_block_size(dummy_model, seed):
    """
    Guard rail for the regression below: while the sequence still fits inside
    block_size the cache is never rolled, so the two paths must already agree.
    Keeps the next test honest -- if sampling alone made the paths differ, this
    would fail too.
    """
    idx = torch.randint(0, VOCAB_SIZE, (1, 4))
    naive, cached = _both_paths(dummy_model, idx, BLOCK_SIZE - 4, seed=seed)  # final len == block_size

    assert torch.equal(naive, cached)


@pytest.mark.parametrize("seed", range(4))
def test_kv_cache_equivalence_past_block_size(dummy_model, seed):
    """
    Regression: generation must stay correct after the context exceeds block_size.

    Once the sequence outgrows block_size the cache has to be rolled. The naive
    path re-embeds a sliding window starting at position 0, so the cached path
    must arrive at the same token->position assignment. Deriving the position
    offset from the *length* of a trimmed cache does not: the retained K/V keep
    the positions they were first computed at, so the paths drift by one
    position per roll and the error compounds (~3e-2 on the logits by 50 tokens).
    """
    idx = torch.randint(0, VOCAB_SIZE, (1, 4))
    naive, cached = _both_paths(dummy_model, idx, 3 * BLOCK_SIZE, seed=seed)

    mismatches = (naive != cached).nonzero()
    first = mismatches[0, 1].item() if mismatches.numel() else None
    assert torch.equal(naive, cached), (
        f"KV cache diverges from naive generation past block_size={BLOCK_SIZE}: "
        f"{mismatches.size(0)} mismatched tokens, first at index {first}"
    )
