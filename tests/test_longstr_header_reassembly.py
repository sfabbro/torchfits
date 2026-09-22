"""LONGSTRN '&'+CONTINUE reassembly parity for open_hdulist (r4c-15)."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

import torchfits


def _write_header(path, pairs) -> str:
    hdul = fits.HDUList([fits.PrimaryHDU()])
    for key, value in pairs:
        hdul[0].header[key] = value
    hdul.writeto(str(path), overwrite=True)
    return str(path)


def test_open_header_reassembles_longstr_chain(tmp_path):
    """torchfits.open must agree with read_header on LONGSTRN values."""
    value = "A" * 80
    path = _write_header(tmp_path / "longstr.fits", [("LONGSTR", value)])

    assert torchfits.read_header(path, hdu=0)["LONGSTR"] == value
    header = torchfits.open(path)[0].header
    assert header["LONGSTR"] == value
    assert all(card.key != "CONTINUE" for card in header.cards)


def test_open_header_reassembles_multi_continue_chain(tmp_path):
    value = "B" * 200
    path = _write_header(tmp_path / "longstr3.fits", [("LONGSTR", value)])
    assert torchfits.read_header(path, hdu=0)["LONGSTR"] == value
    assert torchfits.open(path)[0].header["LONGSTR"] == value


def test_open_header_chain_preserves_inner_ampersands(tmp_path):
    """Only the chain marker '&' is notation; content '&' survives joining."""
    value = "abc&" + "D" * 70
    path = _write_header(tmp_path / "ampmid.fits", [("AMPEND", value)])
    assert torchfits.read_header(path, hdu=0)["AMPEND"] == value
    assert torchfits.open(path)[0].header["AMPEND"] == value


def test_open_header_keeps_literal_trailing_ampersand_without_continue(tmp_path):
    """A '&' with no CONTINUE card is content and must be restored verbatim."""
    value = "xy&"
    path = _write_header(tmp_path / "ampend.fits", [("AMPEND", value)])
    assert torchfits.read_header(path, hdu=0)["AMPEND"] == value
    assert torchfits.open(path)[0].header["AMPEND"] == value


def test_open_header_normal_cards_unchanged(tmp_path):
    """Non-chain cards must pass through the wrapper byte-identically.

    open() headers expose raw card values as strings by pinned design
    (tests/test_hdu_file_ops.py); typed values live on the read_header route.
    """
    data = np.zeros((2, 3), dtype=np.float32)
    path = str(tmp_path / "plain.fits")
    fits.PrimaryHDU(data).writeto(path, overwrite=True)
    direct = torchfits.HDUList.fromfile(path)[0].header
    wrapped = torchfits.open(path)[0].header
    assert dict(wrapped) == dict(direct)
    assert [c.key for c in wrapped.cards] == [c.key for c in direct.cards]
    assert torchfits.read_header(path, hdu=0)["BITPIX"] == -32
