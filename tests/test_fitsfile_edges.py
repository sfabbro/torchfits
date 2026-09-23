"""Image/file engine edges: cross-path read agreement, snapshot semantics,
and zero-pixel boundaries for the FITSFile/SubsetReader C++ layer.

Every assertion here pins observable behavior a plausible engine bug would
violate: silent cross-path disagreement on decoded pixels, cross-generation
byte mixing after an out-of-band file replacement, or off-by-one shapes on
empty windows.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torchfits  # noqa: E402
from torchfits import _cpp  # noqa: E402
from astropy.io import fits as afits  # noqa: E402


def _assert_bits_equal(got: np.ndarray, want: np.ndarray, label: str) -> None:
    """Exact 32-bit pattern comparison, with NaN payloads normalized.

    NaN payload bits are not preserved by every codec; everything else
    (including the sign of zero and the sign of infinity) must match bit
    for bit.
    """
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.shape == want.shape, f"{label}: shape {got.shape} != {want.shape}"
    g = np.asarray(got, dtype=np.float32).copy().view(np.uint32)
    w = np.asarray(want, dtype=np.float32).copy().view(np.uint32)
    nan_g = np.isnan(got)
    nan_w = np.isnan(want)
    assert np.array_equal(nan_g, nan_w), (
        f"{label}: NaN placement differs: {got.ravel().tolist()} vs {want.ravel().tolist()}"
    )
    g[nan_g] = np.uint32(0x7FC00000)
    w[nan_w] = np.uint32(0x7FC00000)
    assert np.array_equal(g, w), (
        f"{label}: bit pattern differs: {got.ravel().tolist()} vs {want.ravel().tolist()}"
    )


def _write_compressed_float(path, arr, algorithm):
    _cpp.write_fits_file_compressed_images(
        str(path), [{"data": arr, "header": {}}], True, algorithm
    )


def _image_read_paths(path, hdu, width, height):
    return {
        "read_full": lambda: np.asarray(_cpp.read_full(str(path), hdu, True)),
        "read_full_numpy": lambda: np.asarray(_cpp.read_full_numpy(str(path), hdu, True)),
        "read_full_numpy_cached": lambda: np.asarray(
            _cpp.read_full_numpy_cached(str(path), hdu, True)
        ),
        "read_full_unmapped": lambda: np.asarray(_cpp.read_full_unmapped(str(path), hdu)),
        "SubsetReader": lambda: np.asarray(
            _cpp.SubsetReader(str(path), hdu).read(0, 0, width, height)
        ),
        "read_subset": lambda: np.asarray(
            _cpp.open_fits_file(str(path), "r").read_subset(hdu, 0, 0, width, height)
        ),
    }


@pytest.mark.parametrize("algorithm", ["GZIP_1", "RICE_1"])
def test_compressed_float_specials_match_astropy_everywhere(tmp_path, algorithm):
    """-0.0 / +-Inf / NaN must survive every image read path of a compressed
    float HDU exactly as astropy reads them (r9b-01). NaN as the CFITSIO nulval
    turns Inf and -0.0 into undefined values (fnan); float storage must be read
    without a nulval on every path."""
    path = tmp_path / f"comp_float_{algorithm}.fits"
    arr = np.array(
        [[-0.0, np.inf, -np.inf, np.nan, np.float32(1e-45), 1.0, 2.5, -1.25, 3.0]],
    )
    _write_compressed_float(path, arr, algorithm)
    truth = np.asarray(afits.getdata(str(path), 1), dtype=np.float32)
    for name, fn in _image_read_paths(path, 1, 9, 1).items():
        _assert_bits_equal(fn(), truth, f"{name} [{algorithm}]")


def test_all_image_paths_agree_on_blank_promotion(tmp_path):
    """BLANK present promotes integer storage to float32 with NaN at the
    undefined pixels (blank-nulval) on every path; the float32 result dtype
    pins today's float32 scale accumulation so the 2.0 float64 switch must be
    deliberate (A-06)."""
    path = tmp_path / "blank_i16.fits"
    raw = np.array([[1, 2, -32768, 4], [5, -32768, 7, 8]], dtype=np.int16)
    hdu = afits.ImageHDU(data=raw)
    hdu.header["BLANK"] = -32768
    afits.HDUList([afits.PrimaryHDU(), hdu]).writeto(path)

    want = np.array([[1, 2, np.nan, 4], [5, np.nan, 7, 8]], dtype=np.float32)
    for name, fn in _image_read_paths(path, 1, 4, 2).items():
        got = fn()
        assert got.dtype == np.float32, f"{name}: dtype {got.dtype} != float32"
        _assert_bits_equal(got, want, name)


def test_unsigned_short_convention_stays_uint16_everywhere(tmp_path):
    """BZERO=32768 integer storage reads as uint16 on every image path."""
    path = tmp_path / "u16.fits"
    want = np.array([[0, 1, 32767, 32768], [40000, 65535, 7, 8]], dtype=np.uint16)
    hdu = afits.ImageHDU(data=(want.astype(np.int32) - 32768).astype(np.int16))
    hdu.header["BSCALE"] = 1
    hdu.header["BZERO"] = 32768
    afits.HDUList([afits.PrimaryHDU(), hdu]).writeto(path)

    for name, fn in _image_read_paths(path, 1, 4, 2).items():
        got = fn()
        assert got.dtype == np.uint16, f"{name}: dtype {got.dtype} != uint16"
        assert np.array_equal(got, want), f"{name}: {got.ravel().tolist()}"


def test_handle_read_returns_open_time_snapshot_after_replace(tmp_path):
    """A persistent FITSFile handle must return the pixels of the file
    generation it opened (open-time snapshot semantics, as documented for
    SubsetReader in _io_engine/http_subset.py), never bytes from a file that
    replaced the path after open (r9b-04)."""
    path = tmp_path / "snap.fits"
    afits.PrimaryHDU(data=np.ones((100, 100), dtype=np.int16)).writeto(path)
    fh = _cpp.open_fits_file(str(path), "r")
    try:
        new_path = tmp_path / "snap_new.fits"
        afits.PrimaryHDU(data=np.full((200, 200), 7, dtype=np.int16)).writeto(new_path)
        os.replace(new_path, path)  # new inode at the same path
        got = np.asarray(_cpp.read_full(fh, 0, True))
    finally:
        fh.close()
    assert got.shape == (100, 100)
    assert got.dtype == np.int16
    assert np.all(got == 1), (
        f"read after out-of-band replacement returned foreign pixels: "
        f"unique values {np.unique(got).tolist()}"
    )


def test_subset_reader_refuses_replaced_file_bytes(tmp_path):
    """A SubsetReader whose lazy mmap cannot be established at open time must
    still never serve pixels from a file that replaced the path afterwards
    (r9b-04). The original generation is truncated here, so the honest outcome
    is a typed read error — not the replacement's bytes."""
    path = tmp_path / "sub_snap.fits"
    afits.PrimaryHDU(data=np.ones((100, 100), dtype=np.int16)).writeto(path)
    with open(path, "rb+") as fh:
        fh.truncate(2880)  # header only: mmap init must fall back
    reader = _cpp.SubsetReader(str(path), 0)
    try:
        new_path = tmp_path / "sub_snap_new.fits"
        afits.PrimaryHDU(data=np.full((200, 200), 7, dtype=np.int16)).writeto(new_path)
        os.replace(new_path, path)
        # Model an observed replacement (the shared meta validated the change).
        _cpp.clear_shared_read_meta_cache()
        with pytest.raises(RuntimeError):
            reader.read(0, 0, 4, 4)
    finally:
        reader.close()


def test_zero_pixel_image_and_empty_box_shapes(tmp_path):
    """Zero-width images and degenerate windows keep their non-degenerate
    extent instead of collapsing to (0, 0) or crashing (r9b zero-edge pin)."""
    path = tmp_path / "zero_pix.fits"
    afits.HDUList(
        [afits.PrimaryHDU(), afits.ImageHDU(data=np.zeros((5, 0), np.float32))]
    ).writeto(path)
    assert np.asarray(_cpp.read_full(str(path), 1, True)).shape == (5, 0)
    fh = _cpp.open_fits_file(str(path), "r")
    try:
        assert np.asarray(fh.read_subset(1, 0, 0, 3, 5)).shape == (5, 0)
    finally:
        fh.close()
    reader = _cpp.SubsetReader(str(path), 1)
    try:
        assert np.asarray(reader.read(0, 0, 3, 5)).shape == (5, 0)
    finally:
        reader.close()

    empty_primary = tmp_path / "empty_primary.fits"
    afits.PrimaryHDU().writeto(empty_primary)
    assert np.asarray(_cpp.read_full(str(empty_primary), 0, True)).shape == (0,)


def test_table_hdu_through_image_readers_raises(tmp_path):
    """Image readers on a table HDU must fail loudly, never return the table's
    raw bytes as an image (r9b edge pin)."""
    path = tmp_path / "table_hdu.fits"
    cols = [afits.Column(name="A", format="J", array=np.arange(5))]
    afits.HDUList([afits.PrimaryHDU(), afits.BinTableHDU.from_columns(cols)]).writeto(
        path
    )
    with pytest.raises(RuntimeError):
        _cpp.read_full(str(path), 1, True)
    with pytest.raises(RuntimeError):
        _cpp.read_full_numpy(str(path), 1, True)
    with pytest.raises(RuntimeError):
        _cpp.read_full_unmapped(str(path), 1)
    with pytest.raises(RuntimeError):
        _cpp.SubsetReader(str(path), 1)
    with pytest.raises(RuntimeError):
        torchfits.read_tensor(str(path), hdu=1)
