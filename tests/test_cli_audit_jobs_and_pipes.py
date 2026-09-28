"""CLI audit (deep-review unit 9) regressions.

Four defects in ``src/torchfits/cli`` that no existing test covered:

* **CL-001** ``run_file_jobs(..., torch_runtime=True)`` called
  ``torch.set_num_threads(1)`` in the *serial* branch and in every worker, so
  the value ``configure_torch_jobs(-j)`` had just resolved was silently
  discarded. ``transform -j 8 in.fits out.fits`` ran single-threaded.
* **CL-002** ``main()`` mapped ``BrokenPipeError`` to ``EXIT_OK``, but stdout
  is block-buffered when piped, so the write only failed during interpreter
  shutdown: ``torchfits info ... | head -1`` exited **120** and printed
  ``Exception ignored on flushing sys.stdout: BrokenPipeError``.
* **CL-003** ``convert --to png`` band reads surfaced the bare native text
  ``Could not move to HDU`` (or ``Could not open FITS file: ...``), naming
  neither the band file nor its position in a multi-band invocation.
* **CL-004** ``table -n -1`` silently rendered no preview rows instead of
  rejecting a negative count, unlike every other numeric CLI flag.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

_CHILD_THREAD_PROBE = """
import sys, threading
import torch
import torchfits.cli.common as common

_seen = []
_original = common.run_file_jobs


def _spy(items, fn, jobs, *, torch_runtime=False, **kwargs):
    def _wrap(item):
        _seen.append(torch.get_num_threads())
        return fn(item)

    return _original(items, _wrap, jobs, torch_runtime=torch_runtime, **kwargs)


common.run_file_jobs = _spy
import torchfits.cli.cmds_transform as ct
ct.run_file_jobs = _spy
from torchfits.cli.main import main

rc = main(sys.argv[1:])
print("THREADS", ",".join(str(value) for value in _seen))
print("RC", rc)
"""


def _run_cli(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        input=stdin,
        stdin=None if stdin is not None else subprocess.DEVNULL,
    )


def _image(path: Path, value: float = 1.0, extra_hdus: int = 0) -> str:
    hdus = [fits.PrimaryHDU(np.full((16, 16), value, dtype=np.float32))]
    for i in range(extra_hdus):
        hdus.append(fits.ImageHDU(np.full((16, 16), value + i, dtype=np.float32)))
    fits.HDUList(hdus).writeto(str(path), overwrite=True)
    return str(path)


def _thread_probe(*args: str) -> tuple[int, list[int]]:
    """Run the CLI in a child and report the intra-op thread count per file."""
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_THREAD_PROBE, *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path.cwd()),
    )
    assert result.returncode == 0, result.stderr
    threads: list[int] = []
    rc = -1
    for line in result.stdout.splitlines():
        if line.startswith("THREADS "):
            raw = line[len("THREADS ") :].strip()
            threads = [int(value) for value in raw.split(",") if value]
        elif line.startswith("RC "):
            rc = int(line.split()[1])
    return rc, threads


# --------------------------------------------------------------------------
# CL-001: -j must survive run_file_jobs
# --------------------------------------------------------------------------


def test_serial_file_job_honours_jobs_flag(tmp_path):
    """A single file (the default -J 1) must keep the -j thread count."""
    src = _image(tmp_path / "in.fits")
    out = tmp_path / "out.fits"
    rc, threads = _thread_probe(
        "transform", "--name", "ArcsinhStretch", "-j", "3", src, str(out)
    )
    assert rc == 0
    assert threads == [3], f"-j 3 discarded, ran with {threads} intra-op thread(s)"


def test_explicit_serial_file_jobs_honours_jobs_flag(tmp_path):
    src = _image(tmp_path / "in.fits")
    out = tmp_path / "out.fits"
    rc, threads = _thread_probe(
        "transform", "--name", "ArcsinhStretch", "-j", "2", "-J", "1", src, str(out)
    )
    assert rc == 0
    assert threads == [2], f"-J 1 with -j 2 discarded, ran with {threads}"


def test_default_jobs_resolves_to_cpu_count_in_serial_run(tmp_path):
    """-j 0 means "all CPU cores"; the serial path must not pin it to 1."""
    import os

    src = _image(tmp_path / "in.fits")
    out = tmp_path / "out.fits"
    rc, threads = _thread_probe(
        "transform", "--name", "ArcsinhStretch", "-j", "0", src, str(out)
    )
    assert rc == 0
    assert threads == [max(1, os.cpu_count() or 1)], f"-j 0 resolved to {threads}"


def test_fan_out_still_caps_each_worker_to_one_thread(tmp_path):
    """The oversubscription guard is real: -J > 1 keeps workers at 1 thread."""
    srcs = [_image(tmp_path / f"i{i}.fits", float(i + 1)) for i in range(4)]
    out_dir = tmp_path / "out"
    rc, threads = _thread_probe(
        "transform",
        "--name",
        "ArcsinhStretch",
        "-j",
        "4",
        "-J",
        "4",
        *srcs,
        "--out-dir",
        str(out_dir),
    )
    assert rc == 0
    assert sorted(threads) == [1, 1, 1, 1], f"-J 4 workers ran with {threads}"
    assert len(list(out_dir.glob("*.fits"))) == 4


# --------------------------------------------------------------------------
# CL-002: a closed downstream pipe is exit 0, not 120
# --------------------------------------------------------------------------


def test_broken_pipe_exits_zero_without_shutdown_noise(tmp_path):
    """`torchfits info many.fits | head -1` must not exit 120 or print noise.

    Block-buffered stdout means the failing write happens while the
    interpreter flushes at shutdown, which CPython reports as
    "Exception ignored on flushing sys.stdout" and exit code 120.
    """
    srcs = [_image(tmp_path / f"i{i:03d}.fits", float(i)) for i in range(300)]
    shell = (
        f"{sys.executable} -m torchfits.cli info {' '.join(srcs)} "
        "| head -1 >/dev/null; echo $?"
    )
    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", shell],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "Exception ignored" not in result.stderr, result.stderr
    assert "BrokenPipeError" not in result.stderr, result.stderr
    assert result.stdout.strip().endswith("0"), (
        f"expected exit 0 from a truncated pipe, got {result.stdout.strip()!r}; "
        f"stderr={result.stderr!r}"
    )


def test_broken_pipe_does_not_hide_a_real_error(tmp_path):
    """A failing input before the truncation point must still exit 3.

    The broken-pipe contract is "the consumer stopped reading", so it must
    never turn a genuine I/O error into a success.
    """
    missing = str(tmp_path / "does_not_exist.fits")
    srcs = [_image(tmp_path / f"i{i:03d}.fits", float(i)) for i in range(300)]
    shell = (
        f"{sys.executable} -m torchfits.cli info {missing} {' '.join(srcs)} "
        "| head -1 >/dev/null; echo $?"
    )
    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", shell],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip().endswith("3"), (
        f"expected the I/O error exit 3, got {result.stdout.strip()!r}; "
        f"stderr={result.stderr!r}"
    )


# --------------------------------------------------------------------------
# CL-003: convert PNG errors name the band file
# --------------------------------------------------------------------------


def test_png_band_hdu_error_names_the_offending_file(tmp_path):
    """An out-of-range --bands index must say which file and which HDU."""
    bands = [_image(tmp_path / f"b{i}.fits", float(i + 1)) for i in range(3)]
    out = tmp_path / "out.png"
    result = _run_cli("convert", *bands, str(out), "--to", "png", "--bands", "0,1,2")
    assert result.returncode != 0
    stderr = result.stderr
    assert "b1.fits" in stderr, f"error does not name the failing band file: {stderr!r}"
    assert "b0.fits" not in stderr.splitlines()[-1], stderr


def test_png_missing_band_file_names_the_path_and_band(tmp_path):
    bands = [
        _image(tmp_path / "b0.fits", 1.0),
        str(tmp_path / "missing_band.fits"),
        _image(tmp_path / "b2.fits", 3.0),
    ]
    out = tmp_path / "out.png"
    result = _run_cli("convert", *bands, str(out), "--to", "png")
    assert result.returncode == 3, result.stderr
    assert "missing_band.fits" in result.stderr, result.stderr
    assert "Could not open FITS file" not in result.stderr or (
        "missing_band.fits" in result.stderr
    )


# --------------------------------------------------------------------------
# CL-004: table -n rejects a negative preview count
# --------------------------------------------------------------------------


def test_table_negative_rows_is_a_usage_error(tmp_path):
    path = tmp_path / "t.fits"
    fits.BinTableHDU.from_columns(
        [fits.Column(name="A", format="J", array=np.arange(4))]
    ).writeto(path, overwrite=True)
    result = _run_cli("table", str(path), "-n", "-1")
    assert result.returncode == 2, (
        f"expected usage exit 2 for -n -1, got {result.returncode}: {result.stderr!r}"
    )
    assert "--rows" in result.stderr or "rows" in result.stderr, result.stderr


@pytest.mark.parametrize("rows", ["0", "2"])
def test_table_nonnegative_rows_still_works(tmp_path, rows):
    path = tmp_path / "t.fits"
    fits.BinTableHDU.from_columns(
        [fits.Column(name="A", format="J", array=np.arange(4))]
    ).writeto(path, overwrite=True)
    result = _run_cli("table", str(path), "-n", rows)
    assert result.returncode == 0, result.stderr
