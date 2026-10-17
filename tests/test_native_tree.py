"""Contracts that read the torchfits source tree.

These checks compile the C++ probes under tests/cpp or read headers in
src/torchfits/cpp_src. They stay with the library. Parity against astropy
and fitsio lives in torchfits-harness.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_contiguous_copy_helper_rejects_count_larger_than_the_array():
    """Structural gate: the shared copy helper must bound ``nelements`` itself.

    ``ensure_c_contiguous_ndarray`` is the memory-safety boundary for every image
    and table write. It takes ``nelements`` — the count the caller will hand to
    CFITSIO — which is *not* the array's size, and its contiguous fast path
    returned the source pointer without ever consulting it, so an over-large count
    meant CFITSIO read past the end of the buffer. Every current caller derives the
    count from the array, so the guard is defence in depth rather than a live fix;
    it is pinned structurally because the code it protects is a header-inline
    function with no Python-reachable entry point to exercise it.
    """
    from pathlib import Path

    header = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "torchfits"
        / "cpp_src"
        / "nb_ndarray_utils.h"
    ).read_text()
    body = header.split("inline void* ensure_c_contiguous_ndarray", 1)[1]
    body = body.split("ndarray_is_c_contiguous(t))", 1)[0]
    assert "t.size()" in body, (
        "ensure_c_contiguous_ndarray must validate nelements against the array "
        "size before returning the contiguous pointer"
    )


def test_vendored_cfitsio_has_niobuf_mindirect_guards():
    """The vendored CFITSIO headers must carry the #ifndef guards that let
    -DTORCHFITS_NIOBUF / -DTORCHFITS_MINDIRECT override the defaults."""
    fitsio_h = ROOT / "extern/cfitsio/fitsio.h"
    fitsio2_h = ROOT / "extern/cfitsio/fitsio2.h"
    if not fitsio_h.exists():
        pytest.skip("vendored CFITSIO not materialized")
    assert "#ifndef NIOBUF" in fitsio_h.read_text()
    assert "#ifndef MINDIRECT" in fitsio2_h.read_text()


def test_hdu_keyed_caches_position_the_cursor_themselves() -> None:
    """Structural gate: the three HDU-keyed lookups must call ensure_hdu_checked.

    These three cache on ``hdu_num`` but read the CFITSIO current-HDU cursor on a
    miss. Positioning was left to the caller, which is a precondition nothing
    states and nothing enforces — every one of the twelve call sites happened to
    satisfy it, so the suite passed either way. That is precisely the kind of
    invariant a test should not rely on being true by convention, so the calls
    are pinned here.

    Structural rather than behavioural: ``FITSFile`` symbols are hidden in the
    extension, so a C++ probe cannot link against it the way the core-library
    probes link ``libtorchfits_core``. The behavioural test above guards the
    observable contract; this one guards the enforcement.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "torchfits"
        / "cpp_src"
        / "fits_file.cpp"
    ).read_text()

    for method in (
        "FITSFile::get_scale_info",
        "FITSFile::is_compressed_image_cached",
        "FITSFile::get_image_info",
    ):
        start = source.index(method)
        body = source[start : source.index("\n}", start)]
        assert "ensure_hdu_checked(" in body, (
            f"{method} caches on hdu_num but reads the CFITSIO cursor on a miss; "
            "it must position the cursor itself"
        )


def test_every_capsule_reader_binding_holds_the_reader_mutex() -> None:
    """Any binding handed a TableReader capsule must hold its mutex.

    ``TableReader::io_mutex_`` is declared in ``table_reader.h`` and acquired
    nowhere in that header -- the bindings that hand out a *shared* reader take it
    instead, because the only shared reader is the ``open_fits_mmap_reader``
    capsule. That split is easy to misread: the header comment described the
    serialization without saying where it happened, which is misleading enough
    that reading the header alone suggests the mutex is dead code.

    This gate makes the coupling explicit and keeps it true. A new binding that
    accepts a capsule without locking would otherwise be a silent data race on
    the CFITSIO cursor and on ``scratch_buffer_``, with the GIL released.
    """
    cpp_src = Path(__file__).resolve().parents[1] / "src" / "torchfits" / "cpp_src"
    table_bindings = (cpp_src / "table_bindings.cpp").read_text()
    table_reader = (cpp_src / "table_reader.h").read_text()

    assert "mutable std::mutex io_mutex_;" in table_reader, (
        "TableReader must keep a mutex for the shared-capsule path"
    )

    # Producer: open_fits_mmap_reader is the only thing that hands out a capsule.
    producers = table_bindings.count("-> nb::capsule")
    assert producers == 1, (
        f"expected one capsule producer, found {producers}; a second sharing path "
        "needs its own locking decision"
    )

    # Consumers: every lambda taking an nb::capsule parameter must lock. Matched
    # by scanning for the parameter rather than by counting, so a new consumer is
    # covered without editing this test.
    consumers = 0
    # A C++ lambda is [capture](params) -> ret: the capsule is a *parameter*, so
    # it appears inside the parentheses, not the brackets.
    pattern = r"\[\s*[^\]]*\]\s*\(([^)]*nb::capsule[^)]*)\)\s*->"
    for match in re.finditer(pattern, table_bindings):
        consumers += 1
        body_start = table_bindings.index("{", match.end())
        # The lambda body: brace-match from its opening brace.
        depth, idx = 0, body_start
        while idx < len(table_bindings):
            if table_bindings[idx] == "{":
                depth += 1
            elif table_bindings[idx] == "}":
                depth -= 1
                if depth == 0:
                    break
            idx += 1
        body = table_bindings[body_start : idx + 1]
        assert "io_mutex_" in body, (
            "a binding that accepts a TableReader capsule must hold io_mutex_ for "
            "the whole call (GIL released), or the CFITSIO cursor and "
            "scratch_buffer_ are raced"
        )
        assert "gil_scoped_release" in body, (
            "the lock is only useful if the GIL is released around the read"
        )
    assert consumers >= 1, "no capsule consumers found -- has the binding moved?"


def test_native_bswap_helpers_match_a_byte_wise_reference(tmp_path) -> None:
    """The SIMD helpers themselves, not just what a FITS read produces with them.

    The rest of this file checks byteswap *through the reader*, which means the
    tests only ever see whatever the vector loops happened to produce. These
    helpers had no direct test, and the real-observation suites cannot reach
    them: the unsigned fast paths in ``read_tensor_canonical`` are gated on
    ``!compressed``, while the CFHT corpus is Rice-compressed throughout. So
    neither the unit tests nor the real-data tests could tell a correct shuffle
    from a wrong one on an uncompressed unsigned frame -- and a wrong shuffle is
    silent, producing plausible pixels that are simply byte-reversed.

    A standalone C++ probe compares all five helpers against a byte-wise
    reference at element counts straddling every vector width and the scalar
    tail, and prints which SIMD branch was compiled in, because a branch that
    silently did nothing would still pass a comparison built from it.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = os.path.join(repo_root, "tests", "cpp", "test_bswap_helpers.cpp")
    include = os.path.join(repo_root, "src", "torchfits", "cpp_src")
    executable = str(tmp_path / "test_bswap_helpers")
    # CXX may carry flags as well as a program name ("ccache c++", "c++ -pipe"),
    # so split it the way tests/test_security.py does for the same job: passing
    # the raw string as one argv element raises FileNotFoundError on a perfectly
    # ordinary value.
    compiler = shlex.split(os.environ.get("CXX", "c++"))

    subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-O2",
            "-I",
            include,
            source,
            "-o",
            executable,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [executable], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, (
        f"a byteswap helper disagrees with the byte-wise reference:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "match the byte-wise reference" in result.stdout, result.stdout
    # Report which branch was exercised, so a run that silently fell through to
    # the scalar tail is visible in CI output rather than mistaken for coverage.
    print(result.stdout.splitlines()[0])


def test_native_cfitsio_bracket_detector_probe_runs(tmp_path: Path):
    """Compile and run the direct C++ detector regression in CI."""
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "tests" / "cpp" / "test_bracket_detection.cpp"
    executable = tmp_path / "test_bracket_detection"
    subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-I",
            str(repo_root / "src" / "torchfits" / "cpp_src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True
    )
    assert "all checks passed" in result.stdout
