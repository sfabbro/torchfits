"""The `tests/cpp` self-checks that pytest *can* build, plus guards on the rest.

`tests/cpp/` holds five standalone C++ programs. Two are already driven by
pytest (`test_security.py` compiles the bracket detector, `test_byteswap.py` the
byte-swap helpers). This file adds the third that needs nothing but a compiler
and `core/parallel.cpp`: `test_parallel_for_nesting`, whose regression is a
**deadlock** in `core::parallel_for` when a worker issues a nested call
(CP-001). Until now that check only ran through `tests/cpp/run_all.sh`, which
no CI job invokes, so the regression it guards had no CI coverage at all.

The other two — `test_open_for_write_conflict` and `test_fitsreader_threads` —
link CFITSIO and are deliberately **not** wired here, because a skip that
always fires is worse than no test: it reports coverage that does not exist
(the TS-015 shape). They need a standalone CFITSIO library, which is not
installed into the pixi environment and, in a `pip install -e .` CI job, is
statically linked into the extension instead of shipped separately. Their
coverage is `run_all.sh`, and the tests below keep that honest.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CPP_SRC = REPO / "src" / "torchfits" / "cpp_src"
CHECKS = REPO / "tests" / "cpp"
RUNNER = CHECKS / "run_all.sh"

_compiler = shlex.split(os.environ.get("CXX", "c++"))


@pytest.fixture(scope="module")
def nesting_binary(tmp_path_factory) -> Path:
    """Build the nesting self-check; it needs no built artifact, only a compiler."""
    out = tmp_path_factory.mktemp("cpp_self_checks") / "test_parallel_for_nesting"
    try:
        result = subprocess.run(
            [
                *_compiler,
                "-std=c++17",
                "-O2",
                "-I",
                str(CPP_SRC),
                str(CHECKS / "test_parallel_for_nesting.cpp"),
                str(CPP_SRC / "core" / "parallel.cpp"),
                "-lpthread",
                "-o",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        # A CXX that does not exist raises here rather than returning non-zero,
        # and an uncaught OSError in a fixture is an *error*, not a skip.
        pytest.skip(f"CXX={_compiler[0]!r} is not runnable: {exc}")
    if result.returncode != 0:
        pytest.skip(
            f"no usable C++ compiler for the self-check ({_compiler[0]!r}): "
            f"{result.stderr.strip()[-300:]}"
        )
    return out


@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_parallel_for_nesting_self_check(nesting_binary: Path, threads: int) -> None:
    """`parallel_for` nesting, error propagation and chunk partitioning.

    Run at several pool sizes because the deadlock this guards only appears at
    size >= 2, and the chunk-partition property has to hold both inline (one
    thread) and across a real pool. The timeout is the point: a regression in
    the worker-inline branch makes this program hang rather than fail, so
    without a bound the suite would hang instead of reporting.
    """
    try:
        result = subprocess.run(
            [str(nesting_binary), str(threads)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        # A regression in the worker-inline branch deadlocks this program, so
        # "still running" is the expected failure shape and has to be reported
        # as a failure rather than left to time out the whole session. The
        # bound is 60 s because a healthy run takes ~1-3 s (measured: all four
        # thread counts, plus the compile, in 16.6 s) and a deadlock never
        # returns at all, so a longer bound only makes the failure slower.
        pytest.fail(
            f"the nesting self-check never returned at TORCHFITS_NUM_THREADS="
            f"{threads}: the worker-inline branch in core/parallel.cpp "
            f"deadlocks, which is the regression this check exists for"
        )
    assert result.returncode == 0, (
        f"TORCHFITS_NUM_THREADS={threads} failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "all checks passed" in result.stdout, result.stdout


def test_every_cpp_self_check_is_in_the_runner() -> None:
    """No self-check may quietly disappear from the harness.

    A check that is neither run nor listed is worse than one that fails: the
    suite says nothing about a program nobody executes. This parses the runner's
    case table and compares it against the directory.
    """
    text = RUNNER.read_text(encoding="utf-8")
    # The case table's rows look like:  "test_bswap_helpers|no|<flags>||"
    listed = set(re.findall(r'^\s*"(test_[a-z_0-9]+)\|', text, flags=re.MULTILINE))
    on_disk = {p.stem for p in CHECKS.glob("*.cpp")}
    assert listed == on_disk, (
        f"run_all.sh covers {sorted(listed)}, tests/cpp holds {sorted(on_disk)}"
    )


def test_library_linked_self_checks_are_documented_as_manual() -> None:
    """The two CFITSIO-linked checks must stay honestly described.

    They cannot be built by a pytest fixture (no standalone CFITSIO library is
    installed, and a CI job links it statically into the extension), so the
    README has to say so rather than let a reader assume CI covers them.
    """
    readme = (CHECKS / "README.md").read_text(encoding="utf-8")
    for name in ("test_open_for_write_conflict", "test_fitsreader_threads"):
        assert name in readme, f"{name} is not mentioned in tests/cpp/README.md"
    assert "no pytest fixture builds" in readme or "stay behind the" in readme, (
        "tests/cpp/README.md no longer says why the library-linked checks are "
        "not run by pytest"
    )
