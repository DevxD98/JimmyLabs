"""Measure the RAM cost of the in-memory loader vs the memory-mapped loader (backlog #7).

This answers one question: how much resident memory does the *corpus* cost during
training? It is a MEMORY benchmark, not a throughput benchmark, so the docs/14 MPS
timing protocol (warmup, sync, median-of-N) does not apply — there is nothing on the GPU
here. What does carry over from docs/14 is the honesty part: fix the seed, record machine
state, measure rather than estimate, and write it down.

Each mode runs in its OWN subprocess, because peak RSS is a high-water mark for the whole
process — measuring both paths in one process would let the first contaminate the second.

Usage:
    python scripts/benchmark_mmap.py --both                  # full comparison
    python scripts/benchmark_mmap.py --mode inmemory         # one path, for debugging
    python scripts/benchmark_mmap.py --both --tokens 1000000 # quick run
"""
import argparse
import json
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from jimmylabs.data.loader import (
    MEMMAP_DTYPE,
    get_batch,
    get_batch_mmap,
    open_memmap,
    save_memmap,
)
from jimmylabs.utils.seed import seed_everything

# TinyStories train split size per datasets/SOURCES.md. Used as the default so the
# measurement reflects the corpus this optimization actually exists to serve.
DEFAULT_TOKENS = 414_700_000

# Vocab size of the TinyStories char corpus (~138). Only affects id range, not memory.
DEFAULT_VOCAB = 138


def peak_rss_mb() -> float:
    """Peak resident set size of THIS process, in MB.

    ru_maxrss is kilobytes on Linux but bytes on macOS — a classic way to publish a
    number that is off by 1024x. Normalize explicitly.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if platform.system() == 'Darwin' else raw / 1e3


def reset_peak_rss() -> bool:
    """Zero this process's RSS high-water mark, if the OS lets us.

    Why this is necessary: on Linux a forked child INHERITS the parent's ru_maxrss
    high-water mark, so a child spawned from a parent that had just built a 3.3 GB tensor
    reports that 3.3 GB as its own "peak" before allocating anything. The first run of
    this benchmark did exactly that and reported an identical, meaningless peak for both
    paths. Writing "5" to /proc/self/clear_refs resets the mark (Linux-only).

    We also prepare the corpus in a separate subprocess (see main) so the measuring
    children fork from a lean parent — that is the portable half of the fix, and it is
    what keeps the number honest on macOS, where clear_refs does not exist.

    Returns True if the mark was actually reset.
    """
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
        return True
    except OSError:
        return False


def rss_composition_mb() -> dict:
    """Split current RSS into anonymous vs file-backed pages (Linux only).

    This is the distinction that actually decides whether a corpus "fits" in 8 GB:

    - **RssAnon** — heap/tensor memory. The kernel CANNOT reclaim it; under pressure it
      can only swap it, and on a fanless Air that means the machine crawls. A corpus held
      as a resident torch.Tensor lives entirely here.
    - **RssFile** — clean page-cache pages mapped from a file. Under pressure the kernel
      drops these instantly and re-faults them later from disk. A mmap-ed corpus lives
      here, so its footprint is *elastic*: it looks large when memory is free and shrinks
      to nothing when the model needs the RAM.

    Total RSS alone hides this, which is why the mmap path can show a few hundred MB of
    RSS and still not compete with the model for memory.

    Returns {} where /proc is unavailable (e.g. macOS).
    """
    try:
        fields = {}
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith(('RssAnon:', 'RssFile:', 'RssShmem:', 'VmHWM:')):
                    key, value = line.split(':', 1)
                    fields[key] = int(value.split()[0]) / 1e3  # kB -> MB
        return fields
    except OSError:
        return {}


def prepare_corpus(data_dir: Path, tokens: int, vocab: int) -> tuple[Path, Path]:
    """Create the synthetic corpus in both on-disk formats, if not already present.

    Random ids are fine here: this measures how the loader *holds* a corpus, which is a
    function of length and dtype, not of what the text says. The corpus is synthetic on
    purpose so the benchmark is reproducible without a 1.9 GB download.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    pt_path, bin_path = data_dir / 'train.pt', data_dir / 'train.bin'

    if pt_path.exists() and bin_path.exists():
        print(f"  corpus already present in {data_dir} — reusing")
        return pt_path, bin_path

    print(f"  generating {tokens:,} synthetic token ids (vocab {vocab})...")
    ids = torch.randint(0, vocab, (tokens,), dtype=torch.int64)

    print(f"  writing {pt_path} (int64 tensor — what scripts/prepare_data.py writes today)")
    torch.save(ids, pt_path)

    print(f"  writing {bin_path} ({np.dtype(MEMMAP_DTYPE).name} flat binary — the mmap format)")
    save_memmap(ids, bin_path)

    del ids
    return pt_path, bin_path


def run_mode(mode: str, data_dir: Path, block_size: int, batch_size: int, batches: int, seed: int) -> dict:
    """Load the corpus one way, draw a stream of batches, report peak RSS."""
    reset_ok = reset_peak_rss()
    seed_everything(seed)
    baseline = peak_rss_mb()

    t0 = time.perf_counter()
    if mode == 'inmemory':
        data = torch.load(data_dir / 'train.pt', weights_only=True)
        sampler = get_batch
    elif mode == 'mmap':
        data = open_memmap(data_dir / 'train.bin')
        sampler = get_batch_mmap
    else:
        raise ValueError(f"unknown mode {mode!r}")
    load_s = time.perf_counter() - t0

    after_load = peak_rss_mb()

    t0 = time.perf_counter()
    for _ in range(batches):
        x, y = sampler(data, block_size, batch_size)
    batch_s = time.perf_counter() - t0
    composition = rss_composition_mb()

    result = dict(
        mode=mode,
        tokens=len(data),
        load_seconds=round(load_s, 3),
        batch_seconds=round(batch_s, 3),
        baseline_rss_mb=round(baseline, 1),
        after_load_rss_mb=round(after_load, 1),
        peak_rss_mb=round(peak_rss_mb(), 1),
        hiwater_reset=reset_ok,
        rss_anon_mb=round(composition.get('RssAnon', float('nan')), 1),
        rss_file_mb=round(composition.get('RssFile', float('nan')), 1),
        corpus_bytes_on_disk=(data_dir / ('train.pt' if mode == 'inmemory' else 'train.bin')).stat().st_size,
    )
    # Machine-readable line so --both can parse a child run without guessing at prose.
    print("RESULT " + json.dumps(result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['inmemory', 'mmap'], help='measure a single path')
    p.add_argument('--both', action='store_true', help='measure both paths in separate subprocesses')
    p.add_argument('--data_dir', default='datasets/_bench_mmap', help='where the synthetic corpus lives')
    p.add_argument('--tokens', type=int, default=DEFAULT_TOKENS)
    p.add_argument('--vocab', type=int, default=DEFAULT_VOCAB)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--batches', type=int, default=200)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--skip_prepare', action='store_true', help='assume the corpus already exists')
    p.add_argument('--prepare_only', action='store_true', help='build the corpus and exit (internal)')
    args = p.parse_args()

    if not args.both and args.mode is None and not args.prepare_only:
        p.error("pass --both, --prepare_only, or --mode {inmemory,mmap}")

    data_dir = Path(args.data_dir)

    # Corpus prep peaks at several GB. It runs in its own short-lived process so that the
    # measuring children fork from a lean parent and inherit a lean RSS high-water mark.
    if args.prepare_only:
        prepare_corpus(data_dir, args.tokens, args.vocab)
        return

    # A child process measuring one path must NOT re-prepare the corpus (that would put
    # the full int64 tensor in its RSS and poison the very number we're taking).
    if args.mode and args.skip_prepare:
        run_mode(args.mode, data_dir, args.block_size, args.batch_size, args.batches, args.seed)
        return

    print("=" * 72)
    print("MACHINE STATE")
    print(f"  platform   : {platform.platform()}")
    print(f"  processor  : {platform.processor() or 'n/a'}")
    print(f"  python     : {platform.python_version()}   torch: {torch.__version__}   numpy: {np.__version__}")
    print(f"  corpus     : {args.tokens:,} tokens, vocab {args.vocab} (synthetic, seed {args.seed})")
    print(f"  sampling   : {args.batches} batches of B={args.batch_size} T={args.block_size}")
    print(f"  metric     : peak RSS (ru_maxrss), one subprocess per path")
    print("=" * 72)

    repo_root = Path(__file__).resolve().parents[1]

    print("CORPUS PREP (subprocess — keeps this process lean, see reset_peak_rss)")
    prep = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), '--prepare_only',
         '--data_dir', str(data_dir), '--tokens', str(args.tokens), '--vocab', str(args.vocab)],
        text=True, cwd=repo_root, timeout=3600,
    )
    if prep.returncode != 0:
        raise RuntimeError("corpus preparation failed")

    if not args.both:
        run_mode(args.mode, data_dir, args.block_size, args.batch_size, args.batches, args.seed)
        return
    results = {}
    for mode in ('inmemory', 'mmap'):
        print(f"\nMEASURING mode={mode} (subprocess)")
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--mode', mode,
             '--data_dir', str(data_dir), '--block_size', str(args.block_size),
             '--batch_size', str(args.batch_size), '--batches', str(args.batches),
             '--seed', str(args.seed), '--skip_prepare'],
            capture_output=True, text=True, cwd=repo_root, timeout=3600,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{mode} run failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        match = re.search(r'^RESULT (\{.*\})$', proc.stdout, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"{mode} run printed no RESULT line:\n{proc.stdout}")
        results[mode] = json.loads(match.group(1))
        print(f"  peak RSS {results[mode]['peak_rss_mb']:,.1f} MB   "
              f"load {results[mode]['load_seconds']:.2f}s   "
              f"{args.batches} batches {results[mode]['batch_seconds']:.2f}s")

    before, after = results['inmemory'], results['mmap']
    saved = before['peak_rss_mb'] - after['peak_rss_mb']
    # Peak RSS includes a fixed interpreter+torch-import baseline (hundreds of MB) that
    # has nothing to do with the corpus. Subtracting it isolates what the *data* costs,
    # which is the quantity this optimization actually moves.
    corpus_before = before['peak_rss_mb'] - before['baseline_rss_mb']
    corpus_after = after['peak_rss_mb'] - after['baseline_rss_mb']
    print("\n" + "=" * 72)
    print("RESULTS — corpus residency")
    print(f"  {'metric':<26} {'in-memory':>14} {'mmap':>14} {'delta':>14}")
    print(f"  {'-' * 26} {'-' * 14:>14} {'-' * 14:>14} {'-' * 14:>14}")
    print(f"  {'peak RSS (MB)':<26} {before['peak_rss_mb']:>14,.1f} {after['peak_rss_mb']:>14,.1f} {-saved:>14,.1f}")
    print(f"  {'  of which baseline':<26} {before['baseline_rss_mb']:>14,.1f} {after['baseline_rss_mb']:>14,.1f} "
          f"{after['baseline_rss_mb']-before['baseline_rss_mb']:>14,.1f}")
    print(f"  {'  corpus-attributable':<26} {corpus_before:>14,.1f} {corpus_after:>14,.1f} "
          f"{corpus_after-corpus_before:>14,.1f}")
    print(f"  {'RssAnon  (unreclaimable)':<26} {before['rss_anon_mb']:>14,.1f} {after['rss_anon_mb']:>14,.1f} "
          f"{after['rss_anon_mb']-before['rss_anon_mb']:>14,.1f}")
    print(f"  {'RssFile  (reclaimable)':<26} {before['rss_file_mb']:>14,.1f} {after['rss_file_mb']:>14,.1f} "
          f"{after['rss_file_mb']-before['rss_file_mb']:>14,.1f}")
    print(f"  {'corpus on disk (MB)':<26} {before['corpus_bytes_on_disk']/1e6:>14,.1f} "
          f"{after['corpus_bytes_on_disk']/1e6:>14,.1f} "
          f"{(after['corpus_bytes_on_disk']-before['corpus_bytes_on_disk'])/1e6:>14,.1f}")
    print(f"  {'corpus load (s)':<26} {before['load_seconds']:>14,.2f} {after['load_seconds']:>14,.2f} "
          f"{after['load_seconds']-before['load_seconds']:>14,.2f}")
    print(f"  {f'{args.batches} batches (s)':<26} {before['batch_seconds']:>14,.2f} {after['batch_seconds']:>14,.2f} "
          f"{after['batch_seconds']-before['batch_seconds']:>14,.2f}")
    print(f"\n  peak RSS reduction: {saved:,.1f} MB "
          f"({saved / before['peak_rss_mb'] * 100:.1f}% of the in-memory peak, "
          f"{before['peak_rss_mb'] / after['peak_rss_mb']:.1f}x lower)")
    anon_saved = before['rss_anon_mb'] - after['rss_anon_mb']
    if anon_saved == anon_saved:  # not NaN
        print(f"  unreclaimable (RssAnon) reduction: {anon_saved:,.1f} MB "
              f"({before['rss_anon_mb'] / max(after['rss_anon_mb'], 1e-9):.1f}x lower) "
              f"— the number that decides whether the corpus fits alongside the model")
    print("=" * 72)


if __name__ == '__main__':
    main()
