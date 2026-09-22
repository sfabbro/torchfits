"""Subset vs HTTP Range cutout parity, incl. the blank-nulval BLANK rule.

Same file via the local subset reader and via a mocked HTTP Range fetch must
agree bitwise (r4a-07). Scaled/BLANK/unsigned-convention images are refused by
the Range path so the CFITSIO fallback applies the blank-nulval rules. The
Range header walk happens exactly once per open (r4a-03) and malformed prior
HDUs fall back cleanly (r4a-08).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from astropy.io import fits as afits

import torchfits
from torchfits._io_engine import http_subset
from torchfits._io_engine.http_subset import HttpRangeUnsupported, read_subset_http


def _serve_payload(monkeypatch, payload: bytes) -> None:
    """Mock http_read_range with inclusive-end Range semantics."""

    def fake(url, start, end):
        return payload[start : end + 1]

    monkeypatch.setattr(http_subset, "http_read_range", fake)


def _write_image(tmp_path, data, name="img.fits", **cards) -> str:
    path = str(tmp_path / name)
    hdu = afits.PrimaryHDU(np.asarray(data))
    for key, value in cards.items():
        hdu.header[key] = value
    hdu.writeto(path, overwrite=True)
    return path


URL = "http://example.com/remote.fits"

# (x1, y1, x2, y2): interior, band, zero-width, zero-height, clamped overhang,
# inverted x, y overhang past NAXIS2.
WINDOWS = [
    (0, 0, 6, 4),
    (1, 1, 5, 3),
    (3, 0, 3, 4),
    (0, 2, 6, 2),
    (-2, -2, 8, 6),
    (5, 1, 2, 4),
    (0, 0, 6, 10),
    (-3, 0, -1, 5),
    (8, 0, 11, 3),
]


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.int16, np.int32, np.int64, np.float32, np.float64],
)
def test_range_cutout_matches_local_subset_bitwise(tmp_path, monkeypatch, dtype):
    """Raw unscaled 2D images: Range cutout == local subset, bitwise (r4a-07)."""
    data = np.arange(4 * 6, dtype=dtype).reshape(4, 6)
    if np.issubdtype(dtype, np.floating):
        data = (data * 0.5).astype(dtype)
    path = _write_image(tmp_path, data)
    payload = open(path, "rb").read()
    _serve_payload(monkeypatch, payload)

    for x1, y1, x2, y2 in WINDOWS:
        local = torchfits.read_subset(path, 0, x1, y1, x2, y2)
        remote = read_subset_http(URL, 0, x1, y1, x2, y2)
        assert local.dtype == remote.dtype, (dtype, x1, y1, x2, y2)
        assert local.shape == remote.shape, (dtype, x1, y1, x2, y2)
        assert torch.equal(local, remote), (dtype, x1, y1, x2, y2)


def test_blank_subset_is_nan_float_and_matches_full_read(tmp_path):
    """blank-nulval: BLANK (even with identity BSCALE/BZERO) promotes the
    image to the scaled-float path and nulval=NaN applies on read_subset."""
    data = np.arange(4 * 6, dtype=np.int16).reshape(4, 6)
    data[0, 0] = -32768
    data[2, 3] = -32768
    data[3, 5] = -32768
    path = _write_image(tmp_path, data, BLANK=-32768)

    full = torchfits.read(path, hdu=0)
    sub = torchfits.read_subset(path, 0, 1, 1, 5, 4)
    assert sub.dtype == torch.float32
    np.testing.assert_array_equal(sub.numpy(), full.numpy()[1:4, 1:5])
    assert np.isnan(sub.numpy()).sum() == 1  # only (2,3) falls in the window
    assert np.isnan(full.numpy()[0, 0]) and np.isnan(full.numpy()[3, 5])


def test_range_refuses_scaled_blank_and_unsigned_conventions(tmp_path, monkeypatch):
    """HTTP Range treats BLANK/scale/unsigned conventions as scaled and
    refuses: the CFITSIO fallback owns nulval and dtype conventions (r4a-07)."""
    cases = {
        "blank": (np.arange(8, dtype=np.int16).reshape(2, 4), {"BLANK": -32768}),
        "bscale": (np.arange(8, dtype=np.int16).reshape(2, 4), {"BSCALE": 2.0}),
        "bzero": (np.arange(8, dtype=np.int16).reshape(2, 4), {"BZERO": 0.5}),
        "uint16": (np.arange(8, dtype=np.uint16).reshape(2, 4), {}),
        "int8": (np.arange(8, dtype=np.int8).reshape(2, 4), {}),
    }
    for name, (data, cards) in cases.items():
        path = _write_image(tmp_path, data, name=f"{name}.fits", **cards)
        _serve_payload(monkeypatch, open(path, "rb").read())
        with pytest.raises(HttpRangeUnsupported):
            read_subset_http(URL, 0, 0, 0, 2, 2)


def _counting_locate(monkeypatch) -> list:
    calls: list = []
    real = http_subset.locate_uncompressed_2d

    def counted(url, hdu):
        calls.append((url, hdu))
        return real(url, hdu)

    monkeypatch.setattr(http_subset, "locate_uncompressed_2d", counted)
    return calls


def test_public_read_subset_walks_headers_once(tmp_path, monkeypatch):
    """One public cutout = one HDU header walk (r4a-03; was open-walk +
    per-cutout walk)."""
    data = np.arange(4 * 6, dtype=np.int16).reshape(4, 6)
    path = _write_image(tmp_path, data)
    _serve_payload(monkeypatch, open(path, "rb").read())
    calls = _counting_locate(monkeypatch)

    cut = torchfits.read_subset(URL, 0, 0, 0, 2, 2)
    assert cut.shape == (2, 2)
    assert len(calls) == 1


def test_open_subset_reader_walks_headers_once_for_many_cutouts(tmp_path, monkeypatch):
    """A persistent reader walks once at open and never again per cutout
    (r4a-03; was one walk per cutout on top of the open walk)."""
    data = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    path = _write_image(tmp_path, data)
    _serve_payload(monkeypatch, open(path, "rb").read())
    calls = _counting_locate(monkeypatch)

    with torchfits.open_subset_reader(URL, hdu=0) as reader:
        for window in [(0, 0, 2, 2), (1, 1, 4, 3), (2, 0, 6, 4)]:
            cut = reader.read_subset(*window)
            x1, y1, x2, y2 = window
            assert torch.equal(cut, torchfits.read_subset(path, 0, x1, y1, x2, y2))
    assert len(calls) == 1


def _header_block(cards: list[str]) -> bytes:
    out = b"".join(card.ljust(80)[:80].encode("latin-1") for card in cards)
    out += b"END".ljust(80)
    return out.ljust(2880, b" ")


@pytest.mark.parametrize("malformed", ["missing_bitpix", "garbage_naxis1"])
def test_malformed_prior_hdu_falls_back_not_raw_error(tmp_path, monkeypatch, malformed):
    """Hostile/truncated cards in a prior HDU must raise HttpRangeUnsupported
    (full-file fallback), never leak KeyError/ValueError (r4a-08)."""
    if malformed == "missing_bitpix":
        bad = _header_block(
            [
                "SIMPLE  =                    T",
                "NAXIS   =                    2",
                "NAXIS1  =                    4",
                "NAXIS2  =                    4",
            ]
        )
    else:
        bad = _header_block(
            [
                "SIMPLE  =                    T",
                "BITPIX  =                   16",
                "NAXIS   =                    2",
                "NAXIS1  = garbage-not-an-int  ",
                "NAXIS2  =                    4",
            ]
        )
    target = (
        _header_block(
            [
                "XTENSION= 'IMAGE   '           ",
                "BITPIX  =                   16",
                "NAXIS   =                    2",
                "NAXIS1  =                    2",
                "NAXIS2  =                    2",
            ]
        )
        + (np.zeros((2, 2), dtype=">i2")).tobytes()
    )
    _serve_payload(monkeypatch, bad + target)

    with pytest.raises(HttpRangeUnsupported):
        read_subset_http(URL, 1, 0, 0, 2, 2)
