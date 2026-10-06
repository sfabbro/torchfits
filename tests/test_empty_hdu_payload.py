"""The HDU writers must refuse an empty payload (R2-017).

A FITS file must contain at least a primary HDU. The C++ writer accepts an
empty payload and writes a 2880-byte file whose only card is `END` — not a
FITS file, and one no reader can open. Because the rewrite paths stage through
`_atomic_replace_target`, that invalid file was then renamed over a perfectly
good one.

Measured before the fix:

* `delete_hdu(path, 0)` on a single-HDU file **returned normally** and replaced
  5760 valid bytes with 2880 unreadable ones — silent destruction of the file.
* `write(path, HDUList([]))` left the same END-only file behind.

The refusal now sits in `_write_hdus_uncompressed` (below every rewrite path)
with an API-level check in `delete_hdu` so the caller gets an actionable
message before any work happens.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
import torch
from astropy.io import fits

import torchfits
from torchfits._io_engine import _hdu_rewrite as hr
from torchfits._io_engine import write_api


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()


def _single_hdu(tmp_path, name="one.fits", extname=None):
    path = str(tmp_path / name)
    if extname is None:
        torchfits.write(path, torch.arange(16, dtype=torch.float32).reshape(4, 4))
    else:
        primary = fits.PrimaryHDU(np.arange(16, dtype=np.float32).reshape(4, 4))
        primary.header["EXTNAME"] = extname
        fits.HDUList([primary]).writeto(path)
    return path


# --- the headline case: the file must survive ------------------------------


@pytest.mark.parametrize("by", ["index", "extname"])
def test_delete_the_last_hdu_is_refused_and_the_file_survives(tmp_path, by):
    # EXTNAME selection needs an EXTNAME card, which a bare primary lacks --
    # delete_hdu matches on the card, so "PRIMARY" is not a selectable name.
    path = _single_hdu(tmp_path, extname="SCI" if by == "extname" else None)
    before = _sha(path)
    assert torchfits.read(path).shape == (4, 4)

    target = 0 if by == "index" else "SCI"
    with pytest.raises(ValueError, match="last HDU"):
        write_api.delete_hdu(path, target)

    # The bytes must be untouched: this is the whole point.
    assert _sha(path) == before, "the refused delete modified the file"
    assert torchfits.read(path).shape == (4, 4)
    assert torchfits.read_num_hdus(path) == 1


def test_delete_the_primary_from_a_two_hdu_file_still_works(tmp_path):
    """Deleting down to exactly one HDU is legal and must keep the file valid.

    A file whose only HDU is a BINTABLE is not constructible through astropy
    (it insists on a primary), so the interesting boundary is the other side:
    a primary-plus-table file where the *primary* goes and the table remains.
    The writer re-emits an empty primary for it, so the HDU count does not drop
    -- what matters is that the table survives and the file still opens.
    """
    path = str(tmp_path / "primary_table.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="A", format="J", array=np.arange(4))]
            ),
        ]
    ).writeto(path)
    write_api.delete_hdu(path, 0)
    assert torchfits.read_num_hdus(path) == 2  # empty primary + the table
    assert torchfits.read_colnames(path, 1) == ["A"]
    assert torchfits.read_nrows(path, 1) == 4


# --- the writer-level backstop --------------------------------------------


def test_writer_refuses_an_empty_payload(tmp_path):
    path = str(tmp_path / "w.fits")
    with pytest.raises(ValueError, match="At least one writable HDU"):
        hr._write_hdus_uncompressed(path, [], overwrite=False)
    assert not os.path.exists(path), "the refused write created a file"


def test_empty_payload_inside_the_atomic_rewrite_leaves_the_original(tmp_path):
    """The backstop must fire *below* the rename, or the file is still lost."""
    path = _single_hdu(tmp_path)
    before = _sha(path)
    with pytest.raises(ValueError, match="At least one writable HDU"):
        with hr._atomic_replace_target(path) as temp_path:
            hr._write_hdus_uncompressed(temp_path, [], overwrite=False)
    assert _sha(path) == before, "the original was replaced despite the refusal"
    assert torchfits.read(path).shape == (4, 4)
    # No temp file left behind either.
    leftovers = [
        n
        for n in os.listdir(os.path.dirname(path))
        if n.startswith(".") and "tmp" in n.lower()
    ]
    assert not leftovers, f"temp files left behind: {leftovers}"


def test_write_with_an_empty_hdulist_leaves_no_file(tmp_path):
    """The other route that produced an END-only file."""
    path = str(tmp_path / "empty_list.fits")
    with pytest.raises((ValueError, RuntimeError)):
        torchfits.write(path, torchfits.HDUList([]))
    assert not os.path.exists(path), "an unreadable file was left behind"


def test_empty_hdulist_write_is_refused(tmp_path):
    """HDUList.write routes into the same writer, so it inherits the refusal."""
    path = str(tmp_path / "hdulist_write.fits")
    with pytest.raises((ValueError, RuntimeError)):
        torchfits.HDUList([]).write(path)
    assert not os.path.exists(path)


# --- non-vacuity: the refusal must not be over-broad ----------------------


def test_deleting_the_last_extension_of_a_two_hdu_file_still_works(tmp_path):
    """Removing the only *extension* is legal: the primary remains."""
    path = str(tmp_path / "two.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.zeros((2, 2), dtype=np.float32), name="SCI"),
        ]
    ).writeto(path)
    write_api.delete_hdu(path, 1)
    assert torchfits.read_num_hdus(path) == 1
    assert torchfits.read_header(path, 0).get("SIMPLE") is True


def test_deleting_from_a_three_hdu_file_twice_still_works(tmp_path):
    path = str(tmp_path / "three.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.zeros((2, 2), dtype=np.float32), name="A"),
            fits.ImageHDU(np.zeros((3, 3), dtype=np.float32), name="B"),
        ]
    ).writeto(path)
    write_api.delete_hdu(path, 2)
    assert torchfits.read_num_hdus(path) == 2
    write_api.delete_hdu(path, "A")
    assert torchfits.read_num_hdus(path) == 1
    # ...and now the last HDU is off limits, as it should be.
    with pytest.raises(ValueError, match="last HDU"):
        write_api.delete_hdu(path, 0)


def test_insert_and_replace_are_unaffected(tmp_path):
    """The other two rewrite entry points must still work on a 1-HDU file."""
    path = _single_hdu(tmp_path)
    write_api.insert_hdu(path, torch.zeros((2, 2)), index=1)
    assert torchfits.read_num_hdus(path) == 2
    write_api.replace_hdu(path, 0, torch.ones((3, 3)))
    assert torchfits.read(path).shape == (3, 3)
    assert torchfits.read_num_hdus(path) == 2
