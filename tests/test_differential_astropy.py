"""Differential parity against astropy across the FITS I/O surface.

This is the safety net for the IO/header/C++ audit: every refactor of those
engines must keep these equalities. It is deliberately seeded and
deterministic so it can run in CI.

Deliberately *not* covered here (pinned by their own tests instead, because
torchfits has a documented behaviour that differs from a naive reading of
astropy): duplicate keyword resolution and COMMENT/HISTORY mapping
visibility. See ``tests/test_header_duplicate_keys.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torchfits  # noqa: E402
import torchfits.table  # noqa: E402
from astropy.io import fits  # noqa: E402

BITPIX_DTYPE = {
    8: np.uint8,
    16: np.int16,
    32: np.int32,
    64: np.int64,
    -32: np.float32,
    -64: np.float64,
}

_RNG = np.random.default_rng(20260912)


# --------------------------------------------------------------------------- images


@pytest.mark.parametrize("bitpix", sorted(BITPIX_DTYPE))
def test_image_values_match_astropy(tmp_path, bitpix):
    """Every BITPIX with identity scaling must match astropy elementwise."""
    dt = BITPIX_DTYPE[bitpix]
    path = str(tmp_path / f"img_{bitpix}.fits")
    data = ((np.arange(48).reshape(6, 8) % 7) + 1).astype(dt)
    fits.PrimaryHDU(data).writeto(path, overwrite=True)

    got = torchfits.read(path)
    exp = np.asarray(fits.getdata(path))

    assert tuple(got.shape) == exp.shape
    assert np.array_equal(np.asarray(got), exp)


@pytest.mark.parametrize("bitpix", [16, 32, -32, -64])
@pytest.mark.parametrize("bscale,bzero", [(2.5, 10.0), (1.0, 0.0), (-0.5, 100.0)])
def test_scaled_image_matches_astropy(tmp_path, bitpix, bscale, bzero):
    """BSCALE/BZERO must be applied identically to astropy."""
    dt = BITPIX_DTYPE[bitpix]
    path = str(tmp_path / f"scaled_{bitpix}_{bscale}.fits")
    data = ((np.arange(48).reshape(6, 8) % 7) + 1).astype(dt)
    hdu = fits.PrimaryHDU(data)
    hdu.header["BSCALE"] = bscale
    hdu.header["BZERO"] = bzero
    hdu.writeto(path, overwrite=True)

    got = np.asarray(torchfits.read(path), dtype="f8")
    exp = np.asarray(fits.getdata(path), dtype="f8")

    assert got.shape == exp.shape
    assert np.allclose(got, exp, rtol=1e-6, atol=1e-6)


def test_multihdu_shapes_match_astropy(tmp_path):
    """Multi-extension files must resolve per-HDU shape and values."""
    path = str(tmp_path / "mef.fits")
    primary = fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32))
    ext1 = fits.ImageHDU(np.ones((3, 5), dtype=np.int16), name="SCI")
    ext2 = fits.ImageHDU(np.full((2, 2), 7, dtype=np.int32), name="ERR")
    fits.HDUList([primary, ext1, ext2]).writeto(path, overwrite=True)

    for idx in (1, 2):
        got = np.asarray(torchfits.read(path, idx))
        exp = np.asarray(fits.getdata(path, idx))
        assert got.shape == exp.shape
        assert np.array_equal(got, exp)


# --------------------------------------------------------------------------- tables


def _typed_table():
    n = 5
    return [
        fits.Column(
            name="L", format="L", array=np.array([True, False, True, True, False])
        ),
        fits.Column(name="B", format="B", array=np.arange(n, dtype=np.uint8)),
        fits.Column(name="I", format="I", array=np.arange(n, dtype=np.int16)),
        fits.Column(name="J", format="J", array=np.arange(n, dtype=np.int32)),
        fits.Column(name="K", format="K", array=np.arange(n, dtype=np.int64)),
        fits.Column(name="E", format="E", array=np.arange(n, dtype=np.float32) + 0.5),
        fits.Column(name="D", format="D", array=np.arange(n) + 0.25),
        fits.Column(
            name="A", format="A8", array=np.array(["ab", "cde", "f", "gh", "ij"])
        ),
    ]


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("name", ["B", "I", "J", "K", "E", "D"])
def test_typed_table_columns_match_astropy(tmp_path, mmap, name):
    """Scalar column types must read back elementwise identical."""
    path = str(tmp_path / "typed.fits")
    fits.BinTableHDU.from_columns(_typed_table()).writeto(path, overwrite=True)

    got = torchfits.table.read_torch(path, mmap=mmap)[name].numpy()
    exp = np.asarray(fits.getdata(path, 1)[name])

    assert got.shape == exp.shape
    assert np.allclose(got.astype("f8"), exp.astype("f8"))


def test_scaled_table_column_matches_astropy(tmp_path):
    """TSCAL/TZERO on an integer column must be applied like astropy."""
    path = str(tmp_path / "scaled_col.fits")
    col = fits.Column(
        name="S", format="I", array=np.array([100, 200, 300], dtype=np.int16)
    )
    hdu = fits.BinTableHDU.from_columns([col])
    hdu.header["TSCAL1"] = 0.1
    hdu.header["TZERO1"] = 5.0
    hdu.writeto(path, overwrite=True)

    got = torchfits.table.read_torch(path, mmap=False)["S"].numpy()
    exp = np.asarray(fits.getdata(path, 1)["S"], dtype="f8")

    assert np.allclose(got.astype("f8"), exp, rtol=1e-6, atol=1e-6)


def test_repeat_column_matches_astropy(tmp_path):
    """Multi-dimensional columns keep row x repeat shape."""
    path = str(tmp_path / "repeat.fits")
    col = fits.Column(
        name="V", format="3E", array=np.arange(12, dtype=np.float32).reshape(4, 3)
    )
    fits.BinTableHDU.from_columns([col]).writeto(path, overwrite=True)

    got = torchfits.table.read_torch(path, mmap=False)["V"].numpy()
    exp = np.asarray(fits.getdata(path, 1)["V"])

    assert got.shape == exp.shape
    assert np.allclose(got, exp)


def test_variable_length_array_matches_astropy(tmp_path):
    """P-format VLA columns must yield the same per-row arrays."""
    path = str(tmp_path / "vla.fits")
    col = fits.Column(
        name="P",
        format="PJ()",
        array=[
            np.arange(3, dtype=np.int32),
            np.arange(1, dtype=np.int32),
            np.arange(5, dtype=np.int32),
        ],
    )
    fits.BinTableHDU.from_columns([col]).writeto(path, overwrite=True)

    got = [np.asarray(x) for x in torchfits.table.read_torch(path, mmap=False)["P"]]
    exp = [np.asarray(x) for x in fits.getdata(path, 1)["P"]]

    assert len(got) == len(exp)
    for a, b in zip(got, exp):
        assert a.shape == b.shape
        assert np.array_equal(a, b)


def test_zero_row_table_matches_astropy(tmp_path):
    """An empty table must not raise and must report zero rows."""
    path = str(tmp_path / "empty.fits")
    col = fits.Column(name="J", format="J", array=np.array([], dtype=np.int32))
    fits.BinTableHDU.from_columns([col], nrows=0).writeto(path, overwrite=True)

    assert torchfits.read_nrows(path) == len(fits.getdata(path, 1)) == 0


# --------------------------------------------------------------------------- headers


def _header_file(tmp_path, specs=(), values=None, name="hdr.fits"):
    """Write a file whose extra cards are valid by construction.

    ``specs`` are full 80-char cards built with ``Card.fromstring`` (a single
    call cannot express a CONTINUE chain, so long values go through
    ``values`` instead, which lets astropy split them).
    ``warnings.simplefilter("error")`` makes astropy reject any non-standard
    card, so a fixture that writes cleanly is proven valid.
    """
    import warnings

    path = str(tmp_path / name)
    hdu = fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for spec in specs:
            hdu.header.append(fits.Card.fromstring(spec))
        for key, value in (values or {}).items():
            hdu.header[key] = value
    hdu.writeto(path, overwrite=True)
    return path


@pytest.mark.parametrize(
    "spec,key",
    [
        ("DEXP    = 1.0D+10", "DEXP"),
        ("DEXP2   = -2.5D-3", "DEXP2"),
        ("SEXP    = 1.5E+3", "SEXP"),
        ("DATEK   = '1999-01-01'", "DATEK"),
        ("NEG     = -42", "NEG"),
        ("ZERO    = 0", "ZERO"),
    ],
)
def test_header_numeric_forms_match_astropy(tmp_path, spec, key):
    """FITS numeric notations (including D exponents) must parse like astropy."""
    path = _header_file(tmp_path, [spec])

    got = torchfits.read_header(path)[key]
    with fits.open(path) as f:
        exp = f[0].header[key]

    assert got == exp


@pytest.mark.parametrize("length", [68, 69, 70, 200])
def test_header_long_string_matches_astropy(tmp_path, length):
    """Values past the 68-char card limit use CONTINUE and must round-trip."""
    value = ("abcdefghij" * (length // 10 + 1))[:length]
    path = _header_file(tmp_path, values={"LONG": value})

    got = torchfits.read_header(path)["LONG"]
    with fits.open(path) as f:
        exp = f[0].header["LONG"]

    assert got == exp == value


def test_header_hierarch_matches_astropy(tmp_path):
    """HIERARCH keywords must keep their full key and value."""
    path = _header_file(tmp_path, ["HIERARCH ESO DET CHIP NAME = 'value1'"])

    got = torchfits.read_header(path)["ESO DET CHIP NAME"]
    with fits.open(path) as f:
        exp = f[0].header["ESO DET CHIP NAME"]

    assert got == exp == "value1"


def test_header_commentary_cards_preserved_in_cards(tmp_path):
    """HISTORY/COMMENT lines must survive losslessly as cards."""
    path = str(tmp_path / "comments.fits")
    hdu = fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32))
    hdu.header.add_comment("first comment")
    hdu.header.add_history("first history")
    hdu.header.add_comment("second comment")
    hdu.writeto(path, overwrite=True)

    header = torchfits.read_header(path)
    got_comments = [c.value for c in header.cards if c.key == "COMMENT"]
    got_history = [c.value for c in header.cards if c.key == "HISTORY"]

    # astropy spells it .keyword; torchfits uses .key.
    with fits.open(path) as f:
        exp_comments = [c.value for c in f[0].header.cards if c.keyword == "COMMENT"]
        exp_history = [c.value for c in f[0].header.cards if c.keyword == "HISTORY"]

    assert got_comments == exp_comments
    assert got_history == exp_history


def test_header_roundtrip_write_read_matches_astropy(tmp_path):
    """A torchfits-written header must read identically via astropy."""
    path = str(tmp_path / "roundtrip.fits")
    header = torchfits.Header()
    header["OBJECT"] = "M31"
    header["EXPTIME"] = 12.5
    header["NSTARS"] = 42
    header["FLAG"] = True
    torchfits.write_tensor(
        path, torch.zeros((4, 4), dtype=torch.float32), header=header, overwrite=True
    )

    with fits.open(path) as f:
        ref = f[0].header

    assert ref["OBJECT"] == "M31"
    assert ref["EXPTIME"] == 12.5
    assert ref["NSTARS"] == 42
    assert bool(ref["FLAG"]) is True
