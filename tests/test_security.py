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
    try:
        torchfits.read("!nonexistent_file.fits")
    except RuntimeError as e:
        assert "Security Error" not in str(e)
    except FileNotFoundError:
        pass
    except Exception:
        pass


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
    """Test that normal filenames are still allowed."""
    try:
        torchfits.read("nonexistent_file.fits")
    except RuntimeError as e:
        assert "Security Error" not in str(e)
    except FileNotFoundError:
        pass
    except Exception:
        pass


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


