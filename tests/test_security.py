import os
from pathlib import Path
import shlex
import shutil
import subprocess

import numpy as np
import pytest
import torch
from astropy.io import fits

import torchfits


def test_image_with_more_than_nine_axes_is_rejected_safely(tmp_path):
    path = tmp_path / "ten-dimensional.fits"
    fits.PrimaryHDU(np.zeros((1,) * 10, dtype=np.uint8)).writeto(path)

    with pytest.raises(RuntimeError, match="at most 9 axes"):
        torchfits.read_tensor(str(path))


def test_read_path_with_literal_bracket_in_directory(tmp_path):
    """Regression: a literal '[' in a directory component
    (not a trailing CFITSIO extended-filename section) must not be
    misdetected as extension syntax. CFITSIO's URL-aware `fits_open_file`
    genuinely fails to parse such paths ("parse error in input file URL"),
    so torchfits must route them through `fits_open_diskfile` instead.
    """
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    src = tmp_path / "image.fits"
    torchfits.write(str(src), data, overwrite=True)

    bracket_dir = tmp_path / "[data]"
    bracket_dir.mkdir()
    path = str(bracket_dir / "image.fits")
    shutil.copy(src, path)

    for use_mmap in (True, False):
        out = torchfits.read(path, mmap=use_mmap)
        assert torch.equal(out, data), f"mismatch for mmap={use_mmap}"

    hdr = torchfits.read_header(path, 0)
    assert hdr["NAXIS"] == 2


@pytest.mark.parametrize(
    "filename",
    [
        "| echo 'pwned'",
        " | ls",
        "valid.fits |",
        "valid.fits | ",
        "|/bin/sh -c 'touch /tmp/pwned'",
        "!| echo 'pwned'",
        "!! | ls",
        "! !| id",
        "sh://echo 'pwned'",
        " !sh://ls",
        "! ! sh://id",
        "SH://touch /tmp/pwned",
        "! \tSh://id",
    ],
)
def test_security_cve_cfitsio_command_injection(filename):
    """Every CFITSIO open rejects native command-injection syntax.

    Exercise the raw extension as well as the Python façade: pipe and
    ``sh://`` rejection belongs to the native CFITSIO boundary, while the
    Python guard owns network address classification.
    """
    with pytest.raises(RuntimeError, match="Security Error"):
        torchfits.read(filename)

    import torchfits._C as native

    with pytest.raises(RuntimeError, match="Security Error"):
        native.open_fits_file(filename, "r")


# Every native entry point in torchfits._cpp._PATH_FIRST -- i.e. every symbol
# whose FIRST positional argument is a FITS path -- called directly on
# torchfits._C so the C++ check_fits_filename_security guard is what is under
# test (torchfits._cpp additionally wraps these in the Python SSRF guard,
# which is a different layer with a different error).
#
# The CVE test above pins exactly two of them (the Python façade and
# native.open_fits_file). This table pins the whole surface, so dropping the
# guard from any one C++ entry point -- or adding a new path-taking one and
# forgetting it -- fails here instead of silently reopening the hole.
#
# The trailing-argument tuples are the minimum each binding needs to reach its
# open call. They are deliberately NON-trivial: insert_rows, update_rows and
# delete_rows all `return` early on an empty payload / zero row count *before*
# the security check, so a no-op call never reaches the guard. That ordering is
# benign (the early return happens before any file is opened, so nothing can be
# executed), but it means a no-op argument tuple would make this gate pass
# vacuously. insert/update/delete therefore get a one-row payload.
_PATH_FIRST_GUARD_ARGS = {
    "append_fits_table_rows": (1, {"A": torch.zeros(1, dtype=torch.float32)}),
    "delete_fits_table_rows": (1, 1, 1),
    "delete_hdu_header_key": (1, "KEY"),
    "drop_fits_table_columns": (1, []),
    "insert_fits_table_rows": (1, {"A": torch.zeros(1, dtype=torch.float32)}, 1),
    "open_and_read_headers": (0,),
    "open_fits_file": ("r",),
    "read_colnames": (0,),
    "read_fits_table": (),
    "read_fits_table_filtered": (1, [], []),
    "read_fits_table_rows": (),
    "read_fits_table_rows_numpy": (),
    "read_full": (0,),
    "read_full_cached": (0, False),
    "read_full_nocache": (0, False),
    "read_full_numpy": (0,),
    "read_full_numpy_cached": (0,),
    "read_full_raw": (0,),
    "read_full_raw_with_scale": (0,),
    "read_full_scaled_cpu": (0,),
    "read_full_unmapped": (0,),
    "read_full_unmapped_raw": (0,),
    "read_hdus_batch": ([1],),
    "read_hdus_sequence_last": ([1],),
    "read_header_dict": (0,),
    "read_hdu_type": (0,),
    "read_keys": (0, []),
    "read_nrows": (0,),
    "read_num_hdus": (),
    "read_shape": (0,),
    "read_table_info": (0,),
    "rename_fits_table_columns": (1, {}),
    "resolve_hdu_name_cached": ("SCI",),
    "update_fits_table_rows": (1, {"A": torch.zeros(1, dtype=torch.float32)}, 1, 1),
    "update_fits_table_rows_mmap": (
        1,
        {"A": torch.zeros(1, dtype=torch.float32)},
        1,
        1,
    ),
    "verify_hdu_checksums": (),
    "write_fits_file": ([], False),
    "write_fits_file_compressed_images": ([], False),
    "write_fits_table": ({}, {}, False),
    "write_hdu_checksums": (),
    "write_hdu_header_cards": (0, []),
}


def test_path_first_native_entry_points_are_all_classified():
    """No path-taking native entry point may escape the guard table below.

    Keeps the table honest when _cpp.__all__/_PATH_FIRST grows: an unclassified
    name fails here rather than being silently untested.
    """
    from torchfits._cpp import _PATH_FIRST

    assert set(_PATH_FIRST_GUARD_ARGS) == set(_PATH_FIRST), (
        "torchfits._cpp._PATH_FIRST and _PATH_FIRST_GUARD_ARGS disagree; "
        f"unclassified={sorted(set(_PATH_FIRST) - set(_PATH_FIRST_GUARD_ARGS))} "
        f"stale={sorted(set(_PATH_FIRST_GUARD_ARGS) - set(_PATH_FIRST))}"
    )


@pytest.mark.parametrize("name", sorted(_PATH_FIRST_GUARD_ARGS))
@pytest.mark.parametrize(
    "filename", ["| echo pwned", "sh://echo pwned", "valid.fits |", "!| echo pwned"]
)
def test_native_path_entry_points_reject_command_injection(
    name, filename, tmp_path, monkeypatch
):
    """Every path-first native entry point rejects CFITSIO command-injection syntax.

    Runs with the CWD inside tmp_path: if a guard were ever removed from a
    write-capable entry point, the call would try to create the hostile
    filename and this test must not leave that behind in the repo.
    """
    import torchfits._C as native

    monkeypatch.chdir(tmp_path)
    entry = getattr(native, name, None)
    assert entry is not None, f"{name} is no longer exported by torchfits._C"

    with pytest.raises(RuntimeError, match="Security Error"):
        entry(filename, *_PATH_FIRST_GUARD_ARGS[name])

    assert os.listdir(tmp_path) == [], f"{name} created a file for a rejected filename"


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


def test_forced_overwrite_prefix_allowed():
    """Leading '!' is valid CFITSIO overwrite syntax and must not bypass pipe checks."""
    # A bare `except Exception: pass` used to guard this, which made the test
    # unfailable: the guard raises HttpBlockedError (an OSError, not a
    # RuntimeError), so a block landed in the swallowing branch and passed.
    # Assert the *failure shape* instead -- CFITSIO's open error, not a block.
    with pytest.raises(Exception) as excinfo:  # noqa: B017 - type is asserted below
        torchfits.read("!nonexistent_file.fits")
    message = str(excinfo.value)
    assert "security" not in message.lower(), f"path was blocked, not opened: {message}"
    assert "Could not open FITS file" in message, message


@pytest.mark.performance
def test_header_large_dict_construction_fast():
    """Regression: Header(dict) must stay O(N), not O(N^2) (PR #172)."""
    import time

    from torchfits.hdu import Header

    d = {f"KEY{i}": i for i in range(2000)}
    t0 = time.perf_counter()
    h = Header(d)
    elapsed = time.perf_counter() - t0
    assert len(h) == 2000
    assert elapsed < 0.5, f"Header(2000) took {elapsed:.3f}s; expected sub-second"


def test_valid_filenames_allowed():
    """Test that normal filenames are still allowed.

    Regression (deep-review unit 10): this was
    ``try: ... except RuntimeError: assert ... except FileNotFoundError: pass
    except Exception: pass``. HttpBlockedError subclasses OSError, so a guard
    that blocked *every* path fell into the final branch and the test still
    passed -- verified by forcing every guard to raise. It now asserts the
    failure shape: CFITSIO's open error, with no security text anywhere.
    """
    with pytest.raises(Exception) as excinfo:  # noqa: B017 - type is asserted below
        torchfits.read("nonexistent_file.fits")
    message = str(excinfo.value)
    assert "security" not in message.lower(), f"path was blocked, not opened: {message}"
    assert "Could not open FITS file" in message, message


def test_read_blocks_private_cfitsio_http_url():
    """Core read must SSRF-block private http URLs before CFITSIO opens them."""
    from torchfits.http_util import HttpBlockedError

    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.read("http://127.0.0.1:9/x.fits")
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.read("https://10.0.0.1/x.fits[1]")
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.read("!ftp://192.168.1.1/x.fits")
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.read_header("http://127.0.0.1:9/x.fits")
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.read_batch(["http://127.0.0.1:9/x.fits"])
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.read_subset("ftp://192.168.1.1/x.fits", 0, 0, 0, 1, 1)
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.open_subset_reader("!http://127.0.0.1:9/x.fits")
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.HDUList.fromfile("http://127.0.0.1:9/x.fits")
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.TableHDU.from_fits("http://127.0.0.1:9/x.fits")
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.table.scan("http://127.0.0.1:9/x.fits")
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.table.scan_torch("http://127.0.0.1:9/x.fits")
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.HDUList([torchfits.TensorHDU(data=torch.zeros(2, 2))]).write(
            "http://127.0.0.1:9/x.fits", overwrite=True
        )
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.read_batch_info(["http://127.0.0.1:9/x.fits"])
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.table.scan_polars("http://127.0.0.1:9/x.fits")
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.cpp.open_fits_file("http://127.0.0.1:9/x.fits", "r")
    with pytest.raises(HttpBlockedError, match="private"):
        torchfits.cpp.TableReader("http://127.0.0.1:9/x.fits", 1)


def test_guard_allows_public_network_url_for_cfitsio(monkeypatch):
    """Public network URLs pass the guard unchanged (CFITSIO still opens them)."""
    from torchfits import http_util
    from torchfits._io_engine.paths import guard_fits_path

    monkeypatch.setattr(http_util, "is_internal_url", lambda _url: False)
    assert (
        guard_fits_path("http://example.com/public.fits[1:10,1:10]")
        == "http://example.com/public.fits[1:10,1:10]"
    )
