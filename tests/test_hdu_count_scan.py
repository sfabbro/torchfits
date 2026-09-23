"""HDU-scan completeness: a truncated/hostile HDU-header tail must not
silently under-report HDUs (r7a-08).

CFITSIO's ffthdu walk stops at the first header it cannot parse and discards
the error, so a file cut mid-header used to report fewer HDUs and every caller
exited 0 with the wrong inventory. The scan entry points must raise a typed
error naming the file when trailing bytes begin an HDU header the walk could
not finish. Contract boundary (pinned below): trailing bytes that do NOT begin
an HDU (arbitrary garbage) stay tolerated like in the read paths, and a file
truncated after a complete header set under-reports nothing — that error
surfaces at read time.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits as astropy_fits

import torchfits
import torchfits._cpp as cpp


@pytest.fixture()
def three_hdu_file(tmp_path):
    path = tmp_path / "three.fits"
    hdul = astropy_fits.HDUList(
        [
            astropy_fits.PrimaryHDU(np.zeros((2, 2), np.float32)),
            astropy_fits.ImageHDU(np.ones((3, 3), np.int16)),
            astropy_fits.ImageHDU(np.full((3, 3), 7, np.int16)),
        ]
    )
    hdul.writeto(path)
    return path


def _hdu_header_offset(path, hdu_index: int) -> int:
    with astropy_fits.open(path) as hdul:
        return int(hdul[hdu_index]._header_offset)


def _cut(path, out, cut: int):
    out.write_bytes(path.read_bytes()[:cut])
    return str(out)


@pytest.fixture()
def truncated_tail_header(three_hdu_file, tmp_path):
    """Multi-HDU file cut mid-header of its last HDU (the under-report case)."""
    cut = _hdu_header_offset(three_hdu_file, 2) + 100
    return _cut(three_hdu_file, tmp_path / "trunc_hdr.fits", cut)


def test_read_num_hdus_truncated_tail_raises_naming_path(truncated_tail_header):
    with pytest.raises(RuntimeError) as excinfo:
        torchfits.read_num_hdus(truncated_tail_header)
    msg = str(excinfo.value)
    assert "truncat" in msg
    assert "trunc_hdr.fits" in msg


def test_cpp_read_num_hdus_truncated_tail_raises(truncated_tail_header):
    with pytest.raises(RuntimeError):
        cpp.read_num_hdus(truncated_tail_header)


def test_open_truncated_tail_raises(truncated_tail_header):
    with pytest.raises(RuntimeError):
        torchfits.open(truncated_tail_header)


def test_open_and_read_headers_truncated_tail_raises(truncated_tail_header):
    with pytest.raises(RuntimeError) as excinfo:
        cpp.open_and_read_headers(truncated_tail_header, 0)
    assert "trunc_hdr.fits" in str(excinfo.value)


def test_get_num_hdus_truncated_tail_raises(truncated_tail_header):
    handle = cpp.open_fits_file(truncated_tail_header, "r")
    try:
        with pytest.raises(RuntimeError):
            cpp.get_num_hdus(handle)
        with pytest.raises(RuntimeError):
            handle.get_num_hdus()
    finally:
        handle.close()


def test_clean_three_hdu_scan_exact(three_hdu_file):
    """No false positives: a whole file scans exactly."""
    path = str(three_hdu_file)
    assert torchfits.read_num_hdus(path) == 3
    assert len(cpp.open_and_read_headers(path, 0)[1]) == 3
    with torchfits.open(path) as hdul:
        assert len(hdul) == 3


def test_clean_heap_table_scan_exact(tmp_path):
    """No false positives on table heaps: the VLA heap is part of the last
    HDU's extent and must not read as a truncated tail."""
    path = tmp_path / "heap.fits"
    rng = np.random.default_rng(5)
    vla = np.empty(40, dtype=object)
    for i in range(40):
        vla[i] = rng.normal(size=(i % 17 + 1)).astype("<f8")
    cols = [
        astropy_fits.Column(name="N", format="J", array=np.arange(40, dtype="<i4")),
        astropy_fits.Column(name="V", format="PD()", array=vla),
    ]
    astropy_fits.HDUList(
        [astropy_fits.PrimaryHDU(), astropy_fits.BinTableHDU.from_columns(cols)]
    ).writeto(path)
    assert torchfits.read_num_hdus(str(path)) == 2


def test_garbage_tail_is_tolerated(three_hdu_file, tmp_path):
    """Trailing bytes that do not begin an HDU header are ignored — the same
    contract as the read paths (test_malformed_fits's
    garbage-after-end-is-ignored-or-reported). Only an HDU-header tail means
    the scan under-reported."""
    raw = three_hdu_file.read_bytes()
    dirty = tmp_path / "garbage_tail.fits"
    dirty.write_bytes(raw + b"\xff" * 2880)
    assert torchfits.read_num_hdus(str(dirty)) == 3
    with torchfits.open(str(dirty)) as hdul:
        assert len(hdul) == 3


def test_data_truncated_scan_reports_full_count(three_hdu_file, tmp_path):
    """Cut mid-DATA of the last HDU: its header is whole, so the scan count is
    honest (3). Truncation surfaces at read time, not as an HDU-count lie."""
    with astropy_fits.open(str(three_hdu_file)) as hdul:
        data_off = int(hdul[2]._data_offset)
    cut = _cut(three_hdu_file, tmp_path / "trunc_data.fits", data_off + 50)
    assert torchfits.read_num_hdus(cut) == 3
    with pytest.raises(RuntimeError):
        torchfits.read_tensor(cut, hdu=2, mmap=False)
