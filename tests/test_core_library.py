"""``libtorchfits_core`` is the real thing, not a wrapper.

The native stack is two libraries now:

* ``libtorchfits_core`` — CFITSIO, the shared per-path metadata cache, the
  FITS inspection rules, and its own thread pool. No libtorch, no nanobind.
* ``torchfits._C`` — the torch-linked extension. It binds the same
  ``FitsFile``/``TableReader`` classes and resolves its ``fits_*`` symbols
  against the core.

Two things make that split worth anything, and both are easy to lose silently:

1. **One implementation.** ``_core.read_header_dict`` and
   ``_C.read_header_dict`` must not be two code paths that agree today. They
   are separate bindings over one C++ function, and these tests compare their
   output on the same inputs rather than trusting that.
2. **One build.** ``_C``/``_core``/``libtorchfits_core`` carry a build id. A
   mismatch is a cross-version call into a C library, whose symptom is a
   segfault; the guard has to be a loud ``ImportError`` at the boundary.

The remaining cross-library invariants (torch-free import, no libtorch in the
core's link line) live in ``tests/test_torch_boundary.py``, which runs in fresh
interpreters.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torchfits  # noqa: E402

core = pytest.importorskip("torchfits._core")

C = pytest.importorskip("torchfits._C")


@pytest.fixture(scope="module")
def sample(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """A scaled image, a binary table, and a file with several HDUs."""
    root = tmp_path_factory.mktemp("core-library")
    image = root / "image.fits"
    torchfits.write(
        str(image),
        torch.arange(16, dtype=torch.float32).reshape(4, 4) * 3.0,
        header={"OBJECT": "CORE", "EXPTIME": 30.5, "INHERIT": 1},
        overwrite=True,
    )
    table = root / "table.fits"
    torchfits.table.write(
        str(table),
        {
            "RA": torch.tensor([1.5, 2.5], dtype=torch.float64),
            "MAG": torch.tensor([19.5, 20.5], dtype=torch.float32),
            "N": torch.tensor([7, 9], dtype=torch.int32),
        },
        overwrite=True,
        extname="MY_TABLE",
    )
    return {"image": str(image), "table": str(table)}


def test_read_header_dict_matches_the_torch_linked_extension(sample) -> None:
    """Both modules must produce byte-identical card lists."""
    for hdu in (0, 1):
        via_core = core.read_header_dict(sample["table"], hdu)
        via_extension = C.read_header_dict(sample["table"], hdu)
        assert via_core == via_extension
        assert via_core, "a real header is never empty"


def test_read_header_string_matches_the_torch_linked_extension(sample) -> None:
    """The extension's header-text binding is handle-based; open one to compare."""
    for path, hdus in ((sample["table"], (0, 1)), (sample["image"], (0,))):
        for hdu in hdus:
            handle = C.open_fits_file(path, "r")
            try:
                via_extension = C.read_header_string(handle, hdu)
            finally:
                handle.close()
            assert core.read_header_string(path, hdu) == via_extension


def test_structural_probes_match_the_torch_linked_extension(sample) -> None:
    assert core.read_num_hdus(sample["table"]) == C.read_num_hdus(sample["table"])
    assert core.read_nrows(sample["table"], 1) == C.read_nrows(sample["table"], 1)
    assert core.read_colnames(sample["table"], 1) == C.read_colnames(sample["table"], 1)
    assert core.read_hdu_type(sample["table"], 1) == C.read_hdu_type(sample["table"], 1)
    assert core.read_table_info(sample["table"], 1) == C.read_table_info(
        sample["table"], 1
    )
    bitpix_core, shape_core = core.read_shape(sample["image"], 0)
    assert (bitpix_core, shape_core) == C.read_shape(sample["image"], 0)


def test_read_keys_types_match_the_torch_linked_extension(sample) -> None:
    """Keyword typing is the subtle part: ``17`` is an int, ``T`` a bool."""
    keys = ["EXPTIME", "INHERIT", "OBJECT", "NAXIS1", "MISSINGISH"]
    via_core = core.read_keys(
        sample["image"], 0, [k for k in keys if k != "MISSINGISH"]
    )
    via_extension = C.read_keys(
        sample["image"], 0, [k for k in keys if k != "MISSINGISH"]
    )
    assert via_core == via_extension
    assert via_core["EXPTIME"] == pytest.approx(30.5)
    assert via_core["INHERIT"] == 1
    assert via_core["OBJECT"] == "CORE"
    with pytest.raises(RuntimeError, match="keyword not found"):
        core.read_keys(sample["image"], 0, ["NOSUCHKEY"])


def test_metadata_handle_agrees_with_the_public_api(sample) -> None:
    """``Metadata`` is a second door to the same room, not a second room."""
    with core.Metadata(sample["table"]) as md:
        assert md.num_hdus() == torchfits.read_num_hdus(sample["table"])
        assert md.hdu_type(1) == torchfits.read_hdu_type(sample["table"], 1)
        assert md.nrows(1) == torchfits.read_nrows(sample["table"], 1)
        assert md.colnames(1) == torchfits.read_colnames(sample["table"], 1)
        assert md.shape(0) == list(torchfits.read_shape(sample["table"], 0)[1])
        assert md.bitpix(0) == torchfits.read_shape(sample["table"], 0)[0]
        assert md.is_compressed_image(0) is False
        assert md.scale_info(0) == (False, True, 1.0, 0.0)  # primary array, unscaled
        assert len(md.header(1)) == len(torchfits.read_header(sample["table"], 1))
        assert md.keywords(1, ["EXTNAME"])["EXTNAME"] == "MY_TABLE"
        # The primary HDU of a table file is a zero-length array, so compare
        # against what read_shape reports rather than an assumed geometry.
        bitpix, shape = torchfits.read_shape(sample["table"], 0)
        assert md.image_info(0)[:2] == (bitpix, len(shape))
    assert md.closed()


def test_shape_and_image_info_state_their_axis_order(tmp_path) -> None:
    """CP-005: the same dimensions in two orders, and only one of them is a shape.

    ``Metadata.shape`` and ``read_shape`` return the row-major (torch) shape --
    NAXISn reversed. ``Metadata.image_info`` reports ``(bitpix, naxis,
    NAXIS1..NAXISn)`` as written in the header. For any non-square HDU the two
    are exact transposes of one another, and the error is silent: the element
    count is identical, so only the axis labels are swapped, and nothing in
    either name hints that an order is in play.

    A non-square image is therefore the only frame that can prove the contract,
    which is what makes this worth its own test -- the module ``sample`` fixture
    is 4x4, where the two orderings are indistinguishable by construction.

    The contract is asserted twice: as measured behaviour, and as the docstring
    a caller actually gets from ``help()``. The second half is the part that was
    missing; the behaviour was already correct, just undiscoverable. The
    committed ``_core.pyi`` inherits the same text via
    ``scripts/gen_native_stub.py``, and ``tests/test_native_stub.py`` gates the
    stub against the extension, so checking the runtime docstring covers both.
    """
    fits = pytest.importorskip("astropy.io.fits")

    path = tmp_path / "nonsquare.fits"
    torchfits.write(
        str(path),
        torch.arange(30, dtype=torch.float32).reshape(5, 6),
        overwrite=True,
    )

    with core.Metadata(str(path)) as md:
        shape = tuple(md.shape(0))
        bitpix, naxis, dims = md.image_info(0)
        fits_order = tuple(int(dims[i]) for i in range(naxis))

    # Row-major, i.e. the shape the tensor paths round-trip.
    assert shape == (5, 6)
    assert core.read_shape(str(path), 0) == (bitpix, shape)

    # FITS order, i.e. the header as written -- the transpose of the above.
    assert fits_order == (6, 5)
    assert fits_order == tuple(reversed(shape))
    assert fits_order != shape, "a square frame would prove nothing"

    # Both readings are the header's own words, so astropy is the arbiter.
    with fits.open(path) as hdul:
        assert tuple(hdul[0].shape) == shape
        assert (hdul[0].header["NAXIS1"], hdul[0].header["NAXIS2"]) == fits_order

    # The part that was actually missing: nothing told the caller which is which.
    shape_doc = (core.Metadata.shape.__doc__ or "").lower()
    image_info_doc = (core.Metadata.image_info.__doc__ or "").lower()
    read_shape_doc = (core.read_shape.__doc__ or "").lower()
    assert "row-major" in shape_doc, core.Metadata.shape.__doc__
    assert "row-major" in read_shape_doc, core.read_shape.__doc__
    assert "not row-major" in image_info_doc, core.Metadata.image_info.__doc__
    # Each names the other, so a caller who landed on the wrong one is told
    # where the right one is.
    assert "image_info()" in shape_doc, core.Metadata.shape.__doc__
    assert "shape()" in image_info_doc, core.Metadata.image_info.__doc__


def test_scaled_image_metadata_round_trips(sample, tmp_path) -> None:
    """The core's BSCALE detection must agree with what a read actually applies.

    ``read_full_raw_with_scale`` reports the factors the tensor reader decided
    on, which is the independent consumer of the same detection. If the two
    disagreed, a ``BZERO`` file would be described one way and read another.

    A ``uint16`` image is the real scaled case: torchfits stores it as BITPIX=16
    with ``BZERO=32768``, which is exactly the offset the unsigned convention
    depends on. The float32 fixture is the unscaled control.
    """
    import torchfits._cpp as cpp

    unsigned = tmp_path / "unsigned.fits"
    values = np.arange(5, dtype=np.uint16) + 40000
    torchfits.write(str(unsigned), values, overwrite=True)

    with core.Metadata(str(unsigned)) as md:
        scaled, trusted, bscale, bzero = md.scale_info(0)
    raw, reader_scaled, reader_bscale, reader_bzero = cpp.read_full_raw_with_scale(
        str(unsigned), 0, False
    )
    assert scaled is True and trusted is True
    assert (bscale, bzero) == pytest.approx((1.0, 32768.0))
    assert reader_scaled is True
    assert (reader_bscale, reader_bzero) == pytest.approx((bscale, bzero))
    # read_full_raw_with_scale hands back the stored (offset) values plus the
    # factors; applying them must reproduce what was written.
    restored = raw.to(torch.int32) + int(reader_bzero)
    assert torch.equal(restored, torch.from_numpy(values.astype(np.int32)))

    with core.Metadata(sample["image"]) as md:
        assert md.scale_info(0) == (False, True, 1.0, 0.0)
    assert float(torchfits._cpp.read_full_raw(sample["image"], 0, False).max()) == (
        pytest.approx(45.0)
    )  # arange(16) * 3.0


def test_shapes_and_typenames_stay_correct_for_a_bintable(sample) -> None:
    bitpix, shape = core.read_shape(sample["image"], 0)
    assert (bitpix, shape) == (-32, (4, 4))
    # A BINTABLE's NAXIS2 is its row count, which read_shape reports as a shape.
    assert core.read_nrows(sample["table"], 1) == 2
    naxis2 = [
        value
        for key, value, _ in core.read_header_dict(sample["table"], 1)
        if key == "NAXIS2"
    ]
    assert naxis2 == ["2"]


def _core_build_artifacts(repo_root: Path) -> tuple[Path, Path] | None:
    """Locate the built core library and its static CFITSIO, or None.

    The FitsReader thread probe links the real library rather than recompiling
    the sources, because the defect it guards is about the interaction between
    FitsReader and the CFITSIO cursor it holds. Skips when there is no build.
    """
    libs = [
        p
        for p in repo_root.glob("build/*/libtorchfits_core.*")
        if p.suffix in {".dylib", ".so"}
    ]
    if not libs:
        return None
    build_dir = libs[0].parent
    archives = [
        p
        for p in build_dir.glob("cfitsio_build/libcfitsio.*")
        if p.suffix in {".a", ".so", ".dylib"}
    ]
    if not archives:
        return None
    return libs[0], archives[0]


def test_fitsreader_is_safe_to_share_across_threads(tmp_path: Path) -> None:
    """One ``Metadata`` handle, many threads: every answer must match its HDU.

    CFITSIO keeps a single mutable current-HDU cursor per ``fitsfile``, and
    ``FitsReader``'s accessors used to release the lock between the move and the
    query that depended on it. Python could not reach that -- the ``Metadata``
    bindings hold the GIL -- so it needed a C++ probe, and on a 37-HDU
    Rice-compressed MegaCam MEF with four threads it produced wrong answers,
    spurious ``Could not read image dimensions``, and a segfault. The fix holds
    the lock across the whole move-then-read.
    """
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = _core_build_artifacts(repo_root)
    if artifacts is None:
        pytest.skip("no build tree with libtorchfits_core; run `pixi run dev` first")
    library, cfitsio = artifacts

    # A synthetic multi-extension file is not enough: the probe needs several
    # HDUs whose type and shape differ, which the real corpus guarantees.
    mef_candidates = sorted(
        (repo_root / "benchmarks_data" / "cfht_megacam").glob("*.fits.fz")
    )
    mef = mef_candidates[0] if mef_candidates else None
    if mef is None:
        pytest.skip(
            "no real MegaCam MEF; run scripts/fetch_cfht_megacam_sample.sh first"
        )

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    source = repo_root / "tests" / "cpp" / "test_fitsreader_threads.cpp"
    executable = tmp_path / "test_fitsreader_threads"
    subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(repo_root / "src" / "torchfits" / "cpp_src"),
            "-I",
            str(repo_root / "extern" / "cfitsio"),
            str(source),
            str(library),
            str(cfitsio),
            "-lpthread",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = f"{library.parent}:{env.get('DYLD_LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{library.parent}:{env.get('LD_LIBRARY_PATH', '')}"
    result = subprocess.run(
        [str(executable), str(mef)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )
    assert result.returncode == 0, (
        f"a shared FitsReader returned another HDU's answer or threw spuriously:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "all checks passed" in result.stdout


def test_cfitsio_readonly_handle_blocks_readwrite_open(tmp_path: Path) -> None:
    """The CFITSIO contract ``open_fits_for_write``'s one-shot retry rests on.

    ``fits_rw.h`` opens a file READWRITE and, on ``FILE_NOT_OPENED``, evicts any
    cached ``TableReader`` and retries once. That works because CFITSIO refuses
    to reopen a file READWRITE while one of its handles is still registered
    READONLY -- six of the nine write call sites in ``table_ops.cpp`` rely on it,
    since only the three in ``fits_bindings.cpp`` evict first.

    The status is load-bearing and nothing else in the suite could notice it
    changing, so it is pinned here rather than left to a comment. This was worth
    pinning because the comment above the retry named ``fits_already_open`` as
    if it were the status; it is a CFITSIO *function*. Looking 104 up in
    ``fitsio.h`` yields ``FILE_NOT_OPENED`` -- "could not open the named file" --
    which reads as unrelated to a cache conflict, and made the retry look like
    dead code to a reviewer on this audit.

    The probe also records that the same status is returned for a missing
    directory, so the retry is unavoidably broad. That is a stated limitation of
    the workaround, not a defect, and the assertion exists so that if CFITSIO
    ever starts distinguishing the two, the narrow fix is obvious.
    """
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = _core_build_artifacts(repo_root)
    if artifacts is None:
        pytest.skip("no build tree with libtorchfits_core; run `pixi run dev` first")
    library, _cfitsio = artifacts

    # A real FITS file with a readable primary HDU: CFITSIO answers 252
    # (UNKNOWN_REC) for an empty one, which would make the first assertion pass
    # for the wrong reason.
    path = tmp_path / "conflict.fits"
    torchfits.write(
        str(path), torch.arange(16, dtype=torch.int16).reshape(4, 4), overwrite=True
    )

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    source = repo_root / "tests" / "cpp" / "test_open_for_write_conflict.cpp"
    executable = tmp_path / "test_open_for_write_conflict"
    # Link the shared core rather than the static CFITSIO archive: the core
    # force-loads the whole archive (CMakeLists.txt, "so the core really is *the*
    # CFITSIO"), so every fits_* symbol resolves and zlib/bzip2/curl come along
    # transitively. Naming those libraries here instead would hard-code a
    # platform-specific link line.
    subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(repo_root / "extern" / "cfitsio"),
            str(source),
            str(library),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = f"{library.parent}:{env.get('DYLD_LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{library.parent}:{env.get('LD_LIBRARY_PATH', '')}"
    result = subprocess.run(
        [str(executable), str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    assert result.returncode == 0, (
        f"CFITSIO no longer refuses a READWRITE open while a READONLY handle is "
        f"held, or stopped returning FILE_NOT_OPENED when nothing is held:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "all checks passed" in result.stdout


def test_read_then_write_survives_a_cached_table_reader(tmp_path: Path) -> None:
    """Read-then-mutate on the public API must work, and must say why if it cannot.

    A cached ``TableReader`` leaves a CFITSIO handle registered READONLY, so
    mutating the same file needs the eviction-and-retry in
    ``open_fits_for_write``. ``append_rows`` evicts the cache itself first (via
    ``_mutation_cache_barrier``), so the warm-cache case is the easy one; it is
    pinned here because it is ordinary usage and the release smoke only ever
    writes *then* reads.

    The second half is the finding. A reader from ``torchfits.open_table_reader``
    is a *live* handle: it owns its own CFITSIO handle, so eviction cannot
    release it and CFITSIO keeps refusing the READWRITE open. That is a
    reasonable thing for a user to do -- open a reader, then write -- and it used
    to fail with CFITSIO's bare "could not open the named file", which names
    neither the cause nor the fix. It now names both.
    """
    path = tmp_path / "read_then_write.fits"
    torchfits.table.write(
        str(path),
        {"A": np.array([1, 2], dtype=np.int32)},
        overwrite=True,
        extname="T",
    )
    # Warm the per-thread reader cache for this path.
    assert torchfits.table.read(str(path), hdu=1).column("A").to_pylist() == [1, 2]
    assert torchfits.table.read(str(path), hdu=1).column("A").to_pylist() == [1, 2]

    torchfits.table.append_rows(str(path), rows={"A": np.array([3], dtype=np.int32)})
    assert torchfits.table.read(str(path), hdu=1).column("A").to_pylist() == [1, 2, 3]

    # A live reader handle cannot be evicted, so the write must still fail --
    # but the error has to tell the user which of the two causes it is.
    reader = torchfits.open_table_reader(str(path), hdu=1)
    assert reader.num_rows() == 3
    with pytest.raises(RuntimeError) as excinfo:
        torchfits.table.append_rows(
            str(path), rows={"A": np.array([4], dtype=np.int32)}
        )
    message = str(excinfo.value)
    assert "open_table_reader" in message, message
    assert "close it" in message, message
    # The old message named neither the cause nor the remedy.
    assert "could not open the named file" not in message, message

    # Once the reader is released the same write succeeds, so the message is
    # accurate advice rather than a dead end.
    del reader
    torchfits.table.append_rows(str(path), rows={"A": np.array([4], dtype=np.int32)})
    assert torchfits.table.read(str(path), hdu=1).column("A").to_pylist() == [
        1,
        2,
        3,
        4,
    ]


def test_thread_pool_survives_worker_side_nesting(tmp_path: Path) -> None:
    """A ``parallel_for`` issued from inside a worker must not deadlock.

    The core pool's only current call site is ``xor_sign_bit_u8``, whose body is
    pure arithmetic, so this is a guard on the *next* call site rather than a
    fix for a live hang. It matters because the original comment claimed
    nesting could never deadlock; measured, it deadlocked at every pool size
    >= 2, because a worker that submits and waits needs workers that are
    themselves inside user code. The worker-inline branch in ``run_chunks``
    makes nesting safe without fanning out a second time.

    The C++ probe is compiled and run at several pool sizes with a wall-clock
    timeout, so a regression fails the test rather than hanging the suite.
    """
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "tests" / "cpp" / "test_parallel_for_nesting.cpp"
    parallel_cpp = repo_root / "src" / "torchfits" / "cpp_src" / "core" / "parallel.cpp"
    executable = tmp_path / "test_parallel_for_nesting"
    subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-I",
            str(repo_root / "src" / "torchfits" / "cpp_src"),
            str(source),
            str(parallel_cpp),
            "-lpthread",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for threads in ("1", "2", "4", "8"):
        result = subprocess.run(
            [str(executable), threads],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, (
            f"parallel_for nesting failed at TORCHFITS_NUM_THREADS={threads} "
            f"(rc={result.returncode})\n{result.stdout}\n{result.stderr}"
        )
        assert "all checks passed" in result.stdout
        assert f"thread_count={threads}" in result.stdout


def test_thread_pool_is_bounded_and_pinned(sample) -> None:
    """``thread_count`` honours TORCHFITS_NUM_THREADS and never exceeds 64."""
    assert 1 <= core.thread_count() <= 64

    probe = subprocess.run(
        [sys.executable, "-c", "import torchfits._core as c; print(c.thread_count())"],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, TORCHFITS_NUM_THREADS="3"),
    )
    assert probe.stdout.strip().splitlines()[-1] == "3"


def test_parallel_signbit_xor_matches_a_single_threaded_read(tmp_path) -> None:
    """The core's own pool must chunk the sign-bit XOR exactly like a serial run.

    The parallel branch is taken above ``TORCHFITS_XOR_PARALLEL_MIN_BYTES`` with
    a 1 MiB grain, so the column has to exceed 2 MiB for more than one chunk to
    exist (below that ``parallel_for`` runs inline by design). A chunked XOR
    that dropped or double-applied a byte would silently corrupt every
    signed-byte column, and nothing else in the suite exercises the core pool.

    The tensor reader is the consumer that applies the XOR; the Arrow path hands
    back the raw FITS bytes, still unsigned, so read via ``read_torch``.
    """
    path = tmp_path / "signed.fits"
    # 3 MiB: three 1 MiB chunks, so two of them are handed to pool workers.
    data = np.arange(-(1 << 21), 1 << 21, dtype=np.int64).astype(np.int8)[::1][
        : 3 << 20
    ]
    torchfits.table.write(str(path), {"B": data}, overwrite=True)

    def read(min_bytes: str) -> list[int]:
        script = (
            "import json\n"
            "import torchfits.table\n"
            f"out = torchfits.table.read_torch({str(path)!r}, 1)\n"
            "print(json.dumps(out['B'].tolist()))\n"
        )
        env = dict(os.environ, TORCHFITS_XOR_PARALLEL_MIN_BYTES=min_bytes)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        return json.loads(proc.stdout.strip().splitlines()[-1])

    assert data.size > 2 << 20, "the fixture must span more than two 1 MiB chunks"
    parallel = read("1")
    assert parallel == read(str(1 << 30)), "chunked sign-bit XOR changed the values"
    assert len(parallel) == data.size


def test_build_ids_agree_across_all_three_artifacts() -> None:
    assert C.__core_build_id__ == core.__build_id__
    assert C.__core_build_id__ == core.core_library_build_id()


def test_mismatched_core_library_build_is_rejected(monkeypatch) -> None:
    """A stale library must raise at the boundary, not segfault later."""
    from torchfits import _core_api

    class _Stale:
        __build_id__ = "torchfits-core/1 build=stale"

        @staticmethod
        def core_library_build_id() -> str:
            return "torchfits-core/1 build=other"

    with pytest.raises(ImportError, match="different builds"):
        _core_api.verify_core_link(_Stale())  # type: ignore[arg-type]


def test_shared_metadata_cache_is_shared_between_the_two_modules(sample) -> None:
    """One cache, or the two modules could disagree after a file is rewritten.

    Two independent registries would both return correct answers and each keep
    its own idea of a file's shape; the failure only shows up as a stale answer
    from whichever module was not told. Assert the entry count, which is a
    direct read of the single registry, and that clearing one clears both.
    """
    core.clear_shared_read_meta_cache()
    assert core.shared_meta_entry_count() == 0

    core.read_num_hdus(sample["table"])
    after_core = core.shared_meta_entry_count()
    assert after_core >= 1

    # Reading through the extension must not create a second entry for the same
    # path: that is exactly what a duplicate registry would do.
    C.read_num_hdus(sample["table"])
    assert core.shared_meta_entry_count() == after_core

    C.read_nrows(sample["table"], 1)
    assert core.shared_meta_entry_count() == after_core

    core.clear_shared_read_meta_cache()
    assert core.shared_meta_entry_count() == 0
