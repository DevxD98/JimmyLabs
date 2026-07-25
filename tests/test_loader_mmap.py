"""Correctness gate for the memory-mapped loader (optimization backlog #7).

A memory optimization that changes results is a bug, not an optimization. So the headline
test here is *equivalence*: from the same seed, `get_batch_mmap` must return batches that
are numerically identical to `get_batch`. This mirrors `tests/test_kv_cache.py`, which
gates the KV cache the same way.

The rest of the tests cover the round-trip and the loud-failure behavior — no silent
fallbacks, per the project's standing rule that bad data must raise, not be papered over.
"""
import numpy as np
import pytest
import torch

from jimmylabs.data.loader import (
    MEMMAP_DTYPE,
    get_batch,
    get_batch_mmap,
    open_memmap,
    save_memmap,
)


@pytest.fixture
def corpus(tmp_path):
    """A synthetic corpus written both ways: resident tensor + on-disk memmap."""
    torch.manual_seed(0)
    data = torch.randint(0, 65, (5000,), dtype=torch.int64)
    path = save_memmap(data, tmp_path / "train.bin")
    return data, open_memmap(path)


def test_mmap_matches_in_memory_batches(corpus):
    """THE correctness gate: same seed -> numerically identical batches from both paths."""
    data, mm = corpus
    block_size, batch_size = 32, 8

    torch.manual_seed(1337)
    x_mem, y_mem = get_batch(data, block_size, batch_size)

    torch.manual_seed(1337)
    x_mm, y_mm = get_batch_mmap(mm, block_size, batch_size)

    assert torch.equal(x_mem, x_mm), "mmap path returned different inputs than the in-memory path"
    assert torch.equal(y_mem, y_mm), "mmap path returned different targets than the in-memory path"


def test_mmap_matches_in_memory_over_many_draws(corpus):
    """Equivalence must hold across a stream of draws, not just the first one."""
    data, mm = corpus
    block_size, batch_size = 16, 4

    torch.manual_seed(7)
    mem_batches = [get_batch(data, block_size, batch_size) for _ in range(25)]

    torch.manual_seed(7)
    mm_batches = [get_batch_mmap(mm, block_size, batch_size) for _ in range(25)]

    for i, ((x_a, y_a), (x_b, y_b)) in enumerate(zip(mem_batches, mm_batches)):
        assert torch.equal(x_a, x_b), f"inputs diverged on draw {i}"
        assert torch.equal(y_a, y_b), f"targets diverged on draw {i}"


def test_mmap_batch_shapes_and_types(corpus):
    """Same shape/dtype/shift contract the in-memory loader is held to."""
    _, mm = corpus
    block_size, batch_size = 10, 4

    x, y = get_batch_mmap(mm, block_size, batch_size)

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    assert x.dtype == torch.int64, "embedding layers index with int64"
    assert y.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1]), "targets are not shifted by one"


def test_memmap_roundtrip_preserves_every_token(tmp_path):
    """save -> open must return the corpus unchanged; a corrupted corpus is a silent disaster."""
    data = torch.randint(0, 65, (1234,), dtype=torch.int64)
    mm = open_memmap(save_memmap(data, tmp_path / "rt.bin"))

    assert len(mm) == len(data)
    assert torch.equal(torch.from_numpy(np.asarray(mm).astype(np.int64)), data)


def test_memmap_stays_off_the_heap(tmp_path):
    """The mapping itself must not materialize the corpus as a resident array."""
    data = torch.randint(0, 65, (100_000,), dtype=torch.int64)
    mm = open_memmap(save_memmap(data, tmp_path / "big.bin"))

    assert isinstance(mm, np.memmap)
    assert mm.dtype == MEMMAP_DTYPE, "ids should be stored narrow on disk, not as int64"


def test_save_memmap_raises_on_ids_too_wide_for_dtype(tmp_path):
    """Ids that don't fit must RAISE, never wrap silently into a garbage corpus."""
    data = torch.tensor([0, 1, 70_000], dtype=torch.int64)
    with pytest.raises(ValueError, match="does not fit"):
        save_memmap(data, tmp_path / "overflow.bin")


def test_save_memmap_rejects_non_1d(tmp_path):
    data = torch.randint(0, 65, (10, 10), dtype=torch.int64)
    with pytest.raises(ValueError, match="1D"):
        save_memmap(data, tmp_path / "twod.bin")


def test_open_memmap_raises_when_corpus_missing(tmp_path):
    """Missing data fails loudly — the flat-loss-at-ln(vocab) bug must not recur."""
    with pytest.raises(FileNotFoundError, match="not found"):
        open_memmap(tmp_path / "nope.bin")
