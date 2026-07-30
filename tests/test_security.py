import shutil

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
    """Regression (deep review C2): a literal '[' in a directory component
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


def test_security_cve_cfitsio_command_injection():
    """
    Test that filenames starting or ending with '|' are rejected to prevent
    CFITSIO command injection vulnerabilities.
    """
    # Filenames that should be rejected
    dangerous_filenames = [
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
        "sh://touch /tmp/pwned",
    ]

    for filename in dangerous_filenames:
        with pytest.raises(RuntimeError, match="Security Error"):
            torchfits.read(filename)


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


def test_guard_allows_public_network_url_for_cfitsio(monkeypatch):
    """Public network URLs pass the guard unchanged (CFITSIO still opens them)."""
    from torchfits import http_util
    from torchfits._io_engine.paths import guard_fits_path

    monkeypatch.setattr(http_util, "is_internal_url", lambda _url: False)
    assert (
        guard_fits_path("http://example.com/public.fits[1:10,1:10]")
        == "http://example.com/public.fits[1:10,1:10]"
    )


def test_security_rejects_uppercase_sh_scheme():
    with pytest.raises(RuntimeError, match="Security Error"):
        torchfits.read("SH://echo pwned")
    with pytest.raises(RuntimeError, match="Security Error"):
        torchfits.read("! Sh://id")
