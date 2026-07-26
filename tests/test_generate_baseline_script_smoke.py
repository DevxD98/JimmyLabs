"""Smoke test that actually RUNS scripts/generate_baseline.py end-to-end.

History: this script used to *silently substitute* a synthetic 65-character vocabulary
when `datasets/shakespeare/meta.json` was missing -- it printed a warning and carried on.
That is standing rule 1's exact failure mode (never catch-and-substitute an input): the
script would still write `outputs/untrained_baseline.txt`, but decoded against a character
mapping that has nothing to do with the corpus the trained model is compared against. The
artifact looks plausible and is worthless as a baseline, and nothing would tell you.

It also hardcoded `vocab_size=65` while holding a tokenizer that knows its own vocab_size
-- the same "hardcode a shape the artifact already carries" bug class that crashed
generate.py on the first v0.2 checkpoint (see test_generate_script_smoke.py).

Both gates below run the real script via subprocess, because three separate bugs have
shipped in this project specifically because only a script's importable functions were
ever tested and nothing executed the script itself.
"""
import sys
import subprocess
from pathlib import Path

from jimmylabs.tokenizer.char import CharTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_baseline.py"


def _run(cwd):
    """Run the real script with `cwd` as its working directory.

    The script resolves both its tokenizer metadata and its output path relative to the
    CWD, so pointing the CWD at a tmp dir fully isolates the test from the repo's real
    datasets/ and outputs/ -- it can neither read the committed corpus nor clobber the
    banked outputs/untrained_baseline.txt.
    """
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, cwd=cwd, timeout=300,
    )


def test_fails_loudly_when_tokenizer_metadata_is_missing(tmp_path):
    """Rule 1: a missing vocabulary must be a hard error, not a substituted one."""
    result = _run(tmp_path)

    assert result.returncode != 0, (
        "generate_baseline.py exited 0 with NO tokenizer metadata present -- it must have "
        f"silently substituted a vocabulary.\nSTDOUT:\n{result.stdout}"
    )
    assert not (tmp_path / "outputs" / "untrained_baseline.txt").exists(), (
        "a baseline artifact was written despite there being no real vocabulary to decode "
        "it with -- this is the silent-corruption case the hard error exists to prevent"
    )
    # The error must tell the operator how to fix it, not just fail.
    assert "prepare_data.py" in result.stderr, (
        f"the failure does not point at the fix (prepare_data.py):\nSTDERR:\n{result.stderr}"
    )


def test_runs_and_derives_vocab_size_from_the_tokenizer(tmp_path):
    """Rule 5: vocab_size comes from meta.json, never from a hardcoded 65."""
    # A DELIBERATELY non-65 vocabulary. If vocab_size were still hardcoded to 65, the
    # model's output ids would range over 65 values while the tokenizer only knows 6 --
    # decoding would raise, or silently emit characters outside this vocab.
    vocab = list("\nabcde")
    assert len(vocab) != 65, "the point of this fixture is that it is NOT v0.1's vocab size"

    meta_path = tmp_path / "datasets" / "shakespeare" / "meta.json"
    meta_path.parent.mkdir(parents=True)
    CharTokenizer(vocab=vocab).save(meta_path)

    result = _run(tmp_path)
    assert result.returncode == 0, (
        f"generate_baseline.py crashed on a non-65 vocabulary:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    out_text = (tmp_path / "outputs" / "untrained_baseline.txt").read_text(encoding="utf-8")
    assert out_text, "no baseline artifact was written"
    # Every character must come from the vocabulary we supplied -- proof the output was
    # decoded with the real tokenizer and not a synthetic stand-in.
    unexpected = set(out_text) - set(vocab)
    assert not unexpected, (
        f"output contains characters outside the supplied vocabulary: {sorted(unexpected)} "
        "-- it was decoded with a different mapping than the one in meta.json"
    )
