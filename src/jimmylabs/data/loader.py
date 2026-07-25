"""Batch sampling for JimmyLabs.

Two paths, deliberately kept side by side:

- :func:`get_batch` — the original in-memory path. The whole corpus is a resident
  ``torch.Tensor``. Simple, and fine for Shakespeare (~1M tokens).
- :func:`get_batch_mmap` — a memory-mapped path (optimization backlog #7). The corpus
  stays on disk as a flat binary file; the OS pages in only the windows a batch actually
  touches. This is what makes a ~400M-token corpus trainable without spending gigabytes
  of the 8 GB budget on data. See `docs/17_DATASET_GUIDE.md`.

The two paths are *numerically identical* for the same RNG state: they draw their start
indices with the same `torch.randint` call and read the same windows. That equivalence is
the correctness gate for the optimization (`tests/test_loader_mmap.py`), the same way
`tests/test_kv_cache.py` gates the KV cache. A memory optimization that changes results
is a bug, not an optimization.
"""
from pathlib import Path

import numpy as np
import torch

# Token-id dtype used for the on-disk memmap. uint16 covers vocabularies up to 65,535,
# which is far above anything JimmyLabs plans (char vocab ~65 for Shakespeare, ~100 for
# v0.3; SPEC.md §3). Storing ids as uint16 instead of the int64 that `torch.save` writes
# is a 4x disk saving on its own, before mmap saves the RAM.
MEMMAP_DTYPE = np.uint16


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Yields random (B, T) int64 tensors of inputs and shifted targets.

    Args:
        data: A 1D torch.Tensor of integer token IDs representing the full dataset.
        block_size: The sequence length T.
        batch_size: The number of sequences B in the batch.
        device: The device to move the tensors to (e.g., 'cpu', 'mps', 'cuda').

    Returns:
        x: Input tensor of shape (batch_size, block_size)
        y: Target tensor of shape (batch_size, block_size), shifted by 1 relative to x.
    """
    # Generate random starting indices for the sequences in the batch
    # The max index we can start from is len(data) - block_size - 1
    # (since we need block_size elements for x, plus 1 more for the shifted y)
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Extract inputs and targets using list comprehension and torch.stack
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # Move to the desired device and ensure dtype is long (int64)
    x, y = x.to(device, dtype=torch.long), y.to(device, dtype=torch.long)

    return x, y


def save_memmap(data: torch.Tensor, path: str | Path, dtype: np.dtype = MEMMAP_DTYPE) -> Path:
    """Write a 1D token-id tensor to a flat binary file that :func:`open_memmap` can map.

    The file is a raw dump of token ids with no header — the shape is recoverable from the
    file size and the dtype, which is why `dtype` must be passed back to `open_memmap`.
    Record it alongside the corpus in `datasets/SOURCES.md`.

    Args:
        data: 1D tensor of integer token IDs.
        path: Destination path (conventionally ``*.bin``).
        dtype: NumPy dtype to store ids as. Must be able to represent every id in `data`.

    Returns:
        The path written.

    Raises:
        ValueError: if `data` is not 1D, or holds ids the dtype cannot represent. We raise
            rather than clamp or wrap — a silently truncated corpus trains a model on
            garbage, which is exactly the failure mode this project has been bitten by.
    """
    if data.ndim != 1:
        raise ValueError(f"expected a 1D token tensor, got shape {tuple(data.shape)}")

    info = np.iinfo(dtype)
    lo, hi = int(data.min()), int(data.max())
    if lo < info.min or hi > info.max:
        raise ValueError(
            f"token ids span [{lo}, {hi}], which does not fit in {np.dtype(dtype).name} "
            f"(range [{info.min}, {info.max}]). Choose a wider dtype."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(path, dtype=dtype, mode='w+', shape=(len(data),))
    arr[:] = data.numpy().astype(dtype, copy=False)
    arr.flush()
    del arr
    return path


def open_memmap(path: str | Path, dtype: np.dtype = MEMMAP_DTYPE) -> np.memmap:
    """Open a token-id binary file as a read-only memmap.

    Nothing is read here — this only establishes the mapping. Pages are faulted in by the
    OS when :func:`get_batch_mmap` actually touches a window, which is the entire point.

    Raises:
        FileNotFoundError: if the corpus is missing. Never fall back to synthetic data:
            training on noise looks like a flat loss at ln(vocab) and has silently burned
            a full run on this project before.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Token-id corpus not found at {path}. Prepare it first (see scripts/prepare_data.py)."
        )
    return np.memmap(path, dtype=dtype, mode='r')


def get_batch_mmap(data: np.memmap, block_size: int, batch_size: int, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-mapped twin of :func:`get_batch` — same contract, same numbers, no resident corpus.

    Args:
        data: A 1D ``np.memmap`` (or any 1D NumPy array) of integer token IDs, e.g. from
            :func:`open_memmap`.
        block_size: The sequence length T.
        batch_size: The number of sequences B in the batch.
        device: The device to move the tensors to.

    Returns:
        x, y: int64 tensors of shape (batch_size, block_size), y shifted by 1 vs x —
        bit-identical to what :func:`get_batch` returns from the same RNG state.
    """
    # Identical draw to get_batch: same call, same argument, same global RNG stream. This
    # is what makes the two paths comparable at all — do not "improve" it to torch.randint
    # with a different signature or a NumPy generator without re-checking the equivalence
    # test, which exists precisely to catch that.
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Copy each window out of the mapping and widen to int64 (the dtype the embedding
    # layer indexes with). Only these (B, T) windows are ever faulted into RAM — the rest
    # of the corpus stays on disk.
    x = torch.stack([
        torch.from_numpy(np.asarray(data[i:i + block_size]).astype(np.int64))
        for i in ix.tolist()
    ])
    y = torch.stack([
        torch.from_numpy(np.asarray(data[i + 1:i + block_size + 1]).astype(np.int64))
        for i in ix.tolist()
    ])

    x, y = x.to(device), y.to(device)

    return x, y
