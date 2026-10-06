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
import shutil
import subprocess
import sys
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
    """The CFITSIO-linked checks must stay honestly described.

    They cannot be built by a pytest fixture (no standalone CFITSIO library is
    installed, and a CI job links it statically into the extension), so the
    README has to say so rather than let a reader assume CI covers them. The
    README must also point at whatever *does* cover each one's Python-visible
    half, so "manual" never reads as "unverified".
    """
    readme = (CHECKS / "README.md").read_text(encoding="utf-8")
    for name in (
        "test_open_for_write_conflict",
        "test_fitsreader_threads",
        "test_raw_fd_retry",
    ):
        assert name in readme, f"{name} is not mentioned in tests/cpp/README.md"
    assert "test_shared_read_cache_does_not_retain_one_descriptor_per_path" in readme, (
        "test_raw_fd_retry's Python half (the retained-descriptor cap) is not "
        "named in tests/cpp/README.md, so a reader would take the cap for "
        "unverified"
    )
    assert "no pytest fixture builds" in readme or "stay behind the" in readme, (
        "tests/cpp/README.md no longer says why the library-linked checks are "
        "not run by pytest"
    )


# --- the CFITSIO link gate ------------------------------------------------
#
# `check_core_link.cmake` is the only thing standing between a build that ships
# a broken wheel and one that does not: it fails the build when `_C` carries an
# undefined `fits_*`/`ff*` symbol that nothing will fill at import time. It had
# **no test at all**, and it makes one assumption about the shape of `nm`
# output -- untested, and false for one of the two formats ld64 emits.
#
# These tests drive the real gate. A stand-in `nm` is the injection point, so
# the parsing under test is the shipped one rather than a copy of it; a copy
# would agree with whatever the copy says.

GATE = CPP_SRC / "check_core_link.cmake"

_FAKE_NM = """#!/bin/sh
# Stand-in `nm`: prints the undefined-symbol list in a chosen shape so the
# gate's decision can be exercised on any host, without a broken library.
# The gate calls it two ways -- `-m -u` on macOS, `-D --undefined-only` on
# ELF -- and both print the same list here; the caller supplies the shape.
# R2_NM_RC != 0 makes it fail, which is how the "nm is broken" paths get tested.
if [ -n "${R2_NM_RC:-}" ]; then
  echo "fake nm: cannot read $2" >&2
  exit "$R2_NM_RC"
fi
printf '%s' "$R2_NM_OUT"
"""

# `_ffclos` bound against the core. Two spellings: current ld64 puts the
# library on the same line, older ld64 ends the line with "from:" and names it
# on the next.
NM_MAC_BOUND_SINGLELINE = (
    "                 (undefined) external _ffclos (from libtorchfits_core)\n"
)
NM_MAC_BOUND_TWOLINE = (
    '                 "_ffclos", referenced from:\n'
    "                   libtorchfits_core.dylib (which is installed)\n"
)
NM_MAC_UNBOUND = "                 (undefined) external _ffclos\n"
# ELF prints the type letter and padding before the symbol.
NM_ELF_BOUND = "                 U _ffclos\n"
NM_ELF_UNBOUND = "                 U _ffclos (dynamic)\n"


def _run_gate(
    tmp_path: Path, nm_output: str, nm_rc: int = 0
) -> subprocess.CompletedProcess[str]:
    """Run the real gate with `nm` replaced, and report what it decided."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    nm = fake_bin / "nm"
    nm.write_text(_FAKE_NM, encoding="utf-8")
    nm.chmod(0o755)
    target = tmp_path / "libtorchfits_fake.so"
    target.write_bytes(b"")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "R2_NM_OUT": nm_output,
    }
    if nm_rc:
        env["R2_NM_RC"] = str(nm_rc)
    return subprocess.run(
        ["cmake", f"-DCMAKE_TARGET={target}", "-P", str(GATE)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake not installed")
def test_core_link_gate_fails_on_an_unbound_cfitsio_symbol(tmp_path: Path) -> None:
    """The gate must still have teeth: an unresolved `ff*` fails the build.

    Without this, a gate that matched nothing at all would pass every build
    and the coverage would be indistinguishable from a working check.
    """
    result = _run_gate(
        tmp_path, NM_MAC_UNBOUND if sys.platform == "darwin" else NM_ELF_UNBOUND
    )
    assert result.returncode != 0, (
        "check_core_link.cmake passed a library with an undefined CFITSIO "
        f"symbol; it must fail the build. Output: {result.stdout}{result.stderr}"
    )
    assert "undefined CFITSIO symbol" in (result.stdout + result.stderr), (
        f"the gate failed for the wrong reason: {result.stdout}{result.stderr}"
    )


@pytest.mark.skipif(
    sys.platform != "darwin", reason="the two-line nm -m form is macOS-only"
)
@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake not installed")
def test_core_link_gate_accepts_both_ways_nm_names_a_bound_symbol(
    tmp_path: Path,
) -> None:
    """A *correctly bound* symbol must pass in either `nm -m` spelling.

    Older ld64 ends the symbol's line with `referenced from:` and names the
    library on the next line, so the gate has to recognise that shape too. When
    it did not, every bound CFITSIO symbol read as unresolved and the build
    failed with "libtorchfits_core must export CFITSIO" on a library that was
    exporting it perfectly well -- and only on the toolchain that prints it.
    """
    for label, output in (
        ("single-line", NM_MAC_BOUND_SINGLELINE),
        ("two-line", NM_MAC_BOUND_TWOLINE),
    ):
        result = _run_gate(tmp_path, output)
        assert result.returncode == 0, (
            f"check_core_link.cmake rejected a correctly bound symbol in the "
            f"{label} `nm -m` form: {result.stdout}{result.stderr}"
        )


@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake not installed")
def test_core_link_gate_never_reports_success_without_looking(tmp_path: Path) -> None:
    """If `nm` fails, the gate must not say the library resolved everything.

    When `nm` cannot read the target, the macOS branch set its result to empty
    and fell through to the success message, so a build in which the gate
    examined nothing at all still logged "resolves every CFITSIO symbol" --
    a positive claim with zero evidence behind it, on the branch that produces
    every shipped macOS wheel. The ELF branch already skipped loudly; this
    says the message may only appear when a symbol list was actually parsed.
    """
    result = _run_gate(tmp_path, "", nm_rc=1)
    output = result.stdout + result.stderr
    assert "resolves every CFITSIO symbol" not in output, (
        "check_core_link.cmake reported success although nm failed and nothing "
        f"was examined: {output}"
    )
    assert "skipping" in output, (
        f"a failed nm should say so rather than pass quietly: {output}"
    )
