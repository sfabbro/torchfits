"""Non-ASCII FITS text: loud on write, lenient on read.

FITS header values, comments and string columns are restricted to printable
ASCII (codes 32..126). ``sanitize_fits_string`` used to be applied to both
directions, which meant user input was *silently rewritten*: writing
``header={"UNI": "λ-cold"}`` stored ``'-cold'`` — a different string, no error,
no warning. astropy rejects the same input with a ``ValueError``.

The write paths now reject non-ASCII, while the read paths keep stripping it so
that sloppy files still open. These tests pin both halves, because the tempting
"simplification" is to make one helper do both jobs again.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torchfits  # noqa: E402
import torchfits.table  # noqa: E402
from astropy.io import fits  # noqa: E402

LAMBDA = "\u03bb"  # two UTF-8 bytes: 0xCE 0xBB


def _assert_ascii_rejected(exc_info):
    """The reason must survive the public wrapper's ``from e`` chaining."""
    cause = exc_info.value.__cause__
    message = f"{exc_info.value} {cause}"
    assert "printable ASCII" in message, message


@pytest.mark.parametrize(
    "header",
    [
        {"UNI": f"{LAMBDA}-cold"},  # plain string value
        {"COM": (1, f"{LAMBDA}-comment")},  # value + comment
        {"HISTORY": f"{LAMBDA}-history"},
        {"COMMENT": f"{LAMBDA}-comment"},
        {f"{LAMBDA}KEY": 1},  # keyword itself
    ],
    ids=["value", "comment", "history", "comment-card", "keyword"],
)
def test_non_ascii_image_header_is_rejected(tmp_path, header):
    path = tmp_path / "hdr.fits"
    with pytest.raises(Exception) as exc_info:
        torchfits.write(path, torch.zeros(2, 2), header=header, overwrite=True)
    _assert_ascii_rejected(exc_info)
    assert not path.exists(), "a rejected write must not leave a partial file"


def test_non_ascii_column_name_is_rejected(tmp_path):
    with pytest.raises(Exception) as exc_info:
        torchfits.table.write(
            tmp_path / "cols.fits",
            {LAMBDA: np.arange(3, dtype=np.int32)},
            overwrite=True,
        )
    _assert_ascii_rejected(exc_info)


def test_non_ascii_string_column_value_is_rejected(tmp_path):
    with pytest.raises(Exception) as exc_info:
        torchfits.table.write(
            tmp_path / "vals.fits",
            {"S": np.array([f"{LAMBDA}b"], dtype="U2")},
            overwrite=True,
        )
    _assert_ascii_rejected(exc_info)


def test_ascii_values_are_stored_verbatim(tmp_path):
    """The strictness must not disturb ordinary ASCII round-trips."""
    path = tmp_path / "ok.fits"
    long_value = "x" * 200  # exercises the long-string (CONTINUE) path
    torchfits.write(
        path,
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
        header={
            "LONGSTR": long_value,
            "PLAIN": "plain-cold",
            "INT": 42,
            "FLT": 1.5,
            "LOG": True,
            "WITHCOMM": (7, "a comment"),
        },
        overwrite=True,
    )
    header = fits.getheader(path)
    assert header["LONGSTR"] == long_value
    assert header["PLAIN"] == "plain-cold"
    assert header["WITHCOMM"] == 7
    assert header.comments["WITHCOMM"] == "a comment"
    assert torchfits.read(path).tolist() == [[0.0, 1.0], [2.0, 3.0]]


def test_ascii_table_round_trip(tmp_path):
    path = tmp_path / "t.fits"
    torchfits.table.write(
        path,
        {
            "A": np.arange(3, dtype=np.int32),
            "S": np.array(["ab", "cd", "ef"], dtype="U2"),
        },
        overwrite=True,
    )
    table = torchfits.table.read(path)
    assert table["A"].to_pylist() == [0, 1, 2]
    assert table["S"].to_pylist() == ["ab", "cd", "ef"]


def test_reading_non_ascii_on_disk_stays_lenient(tmp_path):
    """A file written elsewhere may carry stray bytes; reading must not raise.

    The reader keeps using the lenient sanitizer so such files stay openable —
    rejecting them on read would turn a cosmetic defect into an unreadable file.
    """
    path = tmp_path / "legacy.fits"
    hdu = fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32))
    hdu.header["PLAIN"] = "X-cold"
    hdu.writeto(path)

    raw = bytearray(path.read_bytes())
    card = raw.find(b"PLAIN")
    assert card != -1
    value_at = raw.index(b"'", card)
    raw[value_at + 1] = 0xCE  # non-ASCII byte spliced into the string value
    path.write_bytes(bytes(raw))

    header = torchfits.read_header(path)  # must not raise
    assert all(ord(ch) < 128 for ch in str(header.get("PLAIN", "")))
    # The stray byte must not push the read onto the raw-string fallback, which
    # would type every value as str: BITPIX would read '-32' instead of -32.
    assert header["BITPIX"] == -32 and type(header["BITPIX"]) is int
    assert header["NAXIS"] == 2 and type(header["NAXIS"]) is int
