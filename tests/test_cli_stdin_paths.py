"""TS-009: the documented ``--stdin`` input path had no test at all.

``docs/cli.md`` lists ``--stdin`` in the global-flag table ("Read input file
paths from standard input (stdin is also read implicitly when no paths are
given and stdin is not a terminal)") and shows a worked example
(``find . -name "*.fits" | torchfits info --stdin -f jsonl``). Seven
subcommands route ``args.stdin`` into ``cli.common.resolve_paths``:

    cmds_info, cmds_header, cmds_verify, cmds_stats, cmds_table,
    cmds_probe, cmds_setkey

and every one of them was untested. The flag table also documents a *second*
behaviour with no test: stdin is consumed implicitly when no paths are given
and stdin is not a terminal.

A mechanical boolean-flag coverage sweep (one-sided / never-passed flags) is
what surfaced this; the flag has no keyword call in any test file, and the
only ``stdin`` references under ``tests/`` are ``subprocess`` plumbing.

Every expectation below was measured against the real CLI before being
asserted (not filled in from the source or the docs).
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits


def _run_cli(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run the CLI with explicit stdin control.

    ``stdin=""`` is a *closed-but-not-a-tty* stdin, which is exactly the
    condition ``resolve_paths`` tests for the implicit read. ``stdin=None``
    uses DEVNULL for the negative cases.
    """
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        input=stdin,
        stdin=None if stdin is not None else subprocess.DEVNULL,
    )


def _image(path, value: float = 1.0) -> str:
    fits.PrimaryHDU(np.full((8, 8), value, dtype=np.float32)).writeto(
        str(path), overwrite=True
    )
    return str(path)


@pytest.fixture
def two_images(tmp_path):
    return _image(tmp_path / "a.fits", 1.0), _image(tmp_path / "b.fits", 2.0)


def test_stdin_flag_reads_every_path(two_images):
    """--stdin is the documented way to feed paths; both must be reported."""
    a, b = two_images
    result = _run_cli("info", "--stdin", stdin=f"{a}\n{b}\n")

    assert result.returncode == 0, result.stderr
    assert f"file='{a}'" in result.stdout
    assert f"file='{b}'" in result.stdout


def test_stdin_is_read_implicitly_when_no_paths_given(two_images):
    """The flag table promises the implicit read; no --stdin is needed."""
    a, _b = two_images
    result = _run_cli("info", stdin=f"{a}\n")

    assert result.returncode == 0, result.stderr
    assert f"file='{a}'" in result.stdout


def test_stdin_blank_lines_and_padding_are_skipped(two_images):
    """find(1) output is newline-terminated and padded; blanks are not paths."""
    a, _b = two_images
    result = _run_cli("info", "--stdin", stdin=f"\n   {a}   \n\n")

    assert result.returncode == 0, result.stderr
    assert f"file='{a}'" in result.stdout
    assert result.stdout.count("file=") == 1


def test_argv_paths_and_stdin_paths_are_merged_in_order(two_images):
    """--stdin appends to argv paths rather than replacing them."""
    a, b = two_images
    result = _run_cli("info", a, "--stdin", stdin=f"{b}\n")

    assert result.returncode == 0, result.stderr
    assert result.stdout.index(f"file='{a}'") < result.stdout.index(f"file='{b}'")


def test_empty_stdin_with_explicit_flag_is_a_usage_error():
    """--stdin with nothing to read is a usage error, not a silent no-op."""
    result = _run_cli("info", "--stdin", stdin="")

    assert result.returncode == 2
    assert "no input paths (argv or stdin)" in result.stderr
    assert result.stdout == ""


def test_unreadable_stdin_path_reports_the_path_from_stdin(two_images, tmp_path):
    """A bad path that arrived over stdin is still attributed by name."""
    _a, _b = two_images
    missing = tmp_path / "nope.fits"
    result = _run_cli("info", "--stdin", stdin=f"{missing}\n")

    assert result.returncode == 3
    assert str(missing) in result.stderr
    assert result.stdout == ""


def test_second_subcommand_honours_stdin(two_images):
    """The flag is routed by every subcommand, not just ``info``.

    ``header`` announces each file with its own banner rather than the
    key=value row ``info`` prints, so the attribution is pinned per format.
    """
    a, b = two_images
    result = _run_cli("header", "--stdin", stdin=f"{a}\n{b}\n")

    assert result.returncode == 0, result.stderr
    assert f"# HDU 0 (PRIMARY) in {a}:" in result.stdout
    assert f"# HDU 0 (PRIMARY) in {b}:" in result.stdout
    assert result.stdout.index(f"in {a}:") < result.stdout.index(f"in {b}:")
