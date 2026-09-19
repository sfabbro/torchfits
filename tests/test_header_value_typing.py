"""Header value *types* must not depend on which API you call, or on the file.

``read_header`` types values from the fast C++ header string (int/bool/float),
but two routes handed callers the raw CFITSIO string dict instead, where every
value is a ``str``:

* ``read_tensor(..., return_header=True)`` and ``read_hdus(...,
  return_header=True)`` preferred the native string dict over the typed getter
  the caller already passed in.
* ``read_header`` itself silently fell back to that same dict whenever the
  header contained a byte >= 0x80, because nanobind's UTF-8 conversion of the
  header string raised.

Both made ``header["BITPIX"] // 8`` work on one file (or through one API) and
raise ``TypeError`` on another, with no error anywhere. astropy and
``read_keys`` (CFITSIO) are independent oracles; these tests require all four
routes to agree with them on both value and type.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torchfits  # noqa: E402
from astropy.io import fits  # noqa: E402

# Keys written by us plus the structural ones, with the type each must have.
_TYPED = {
    "COUNT": 17,  # int
    "FLAG": True,  # bool
    "SCALE": 2.5,  # float
    "NAME": "abc",  # str
    "BITPIX": -32,  # int (structural)
    "NAXIS": 2,  # int (structural)
}


@pytest.fixture()
def typed_file(tmp_path: Path) -> Path:
    path = tmp_path / "typed.fits"
    torchfits.write(
        path,
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
        header={"COUNT": 17, "FLAG": True, "SCALE": 2.5, "NAME": "abc"},
        overwrite=True,
    )
    return path


def test_read_header_types_match_astropy(typed_file):
    header = torchfits.read_header(typed_file)
    reference = fits.getheader(str(typed_file))
    for key, expected in _TYPED.items():
        assert header[key] == expected, key
        assert type(header[key]) is type(reference[key]), key


def test_typing_matches_read_keys(typed_file):
    """read_keys goes through CFITSIO, so it is an independent type oracle."""
    assert torchfits.read_keys(typed_file, ["COUNT", "FLAG", "SCALE"], hdu=0) == {
        "COUNT": 17,
        "FLAG": True,
        "SCALE": 2.5,
    }


def test_return_header_typing_is_consistent(typed_file):
    """Every return_header=True route must match read_header exactly."""
    reference = torchfits.read_header(typed_file)

    _, from_tensor = torchfits.read_tensor(typed_file, return_header=True)
    _, headers = torchfits.read_hdus(typed_file, [0], return_header=True)
    _, from_read = torchfits.read(typed_file, return_header=True)

    for route, header in (
        ("read_tensor", from_tensor),
        ("read_hdus", headers[0]),
        ("read", from_read),
    ):
        for key, expected in _TYPED.items():
            assert header[key] == expected, f"{route}[{key}]"
            assert type(header[key]) is type(reference[key]), f"{route}[{key}]"


def test_non_ascii_header_keeps_typed_values(tmp_path):
    """A stray high-bit byte must not downgrade every header value to a string.

    It used to: the header string failed UTF-8 decoding inside nanobind, so the
    read fell back to the raw dict and ``COUNT``/``FLAG`` came back as ``'17'``
    and ``'T'``. The byte is still dropped (the documented lenient read), but the
    fast path -- and therefore the typing -- survives.
    """
    path = tmp_path / "legacy.fits"
    torchfits.write(
        path,
        torch.zeros(2, 2),
        header={"COUNT": 17, "FLAG": True, "NAME": "ab"},
        overwrite=True,
    )

    raw = bytearray(path.read_bytes())
    at = raw.find(b"NAME")
    assert at != -1
    value_at = raw.index(b"'", at)
    raw[value_at + 1] = 0xCE  # splice a non-ASCII byte into the string value
    path.write_bytes(bytes(raw))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        header = torchfits.read_header(path)

    assert not [w for w in caught if "fast path failed" in str(w.message)]
    assert header["COUNT"] == 17 and type(header["COUNT"]) is int
    assert header["FLAG"] is True and type(header["FLAG"]) is bool
    assert header["BITPIX"] == -32 and type(header["BITPIX"]) is int
    assert all(ord(ch) < 128 for ch in str(header.get("NAME", "")))


def test_non_ascii_return_header_typing_matches_read_header(tmp_path):
    """The return_header routes must agree even on a non-ASCII header."""
    path = tmp_path / "legacy2.fits"
    torchfits.write(
        path,
        torch.zeros(2, 2),
        header={"COUNT": 17, "NAME": "ab"},
        overwrite=True,
    )
    raw = bytearray(path.read_bytes())
    value_at = raw.index(b"'", raw.find(b"NAME"))
    raw[value_at + 1] = 0xCE
    path.write_bytes(bytes(raw))

    reference = torchfits.read_header(path)
    _, from_tensor = torchfits.read_tensor(path, return_header=True)
    for key in ("COUNT", "BITPIX"):
        assert type(from_tensor[key]) is type(reference[key]), key
        assert from_tensor[key] == reference[key], key
