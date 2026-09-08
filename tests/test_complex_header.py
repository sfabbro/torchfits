import os

import numpy as np
from astropy.io import fits

import torchfits


def create_complex_header(filename):
    hdu = fits.PrimaryHDU()
    header = hdu.header

    # Standard keyword
    header["SIMPLE"] = True

    # HIERARCH keyword
    header["HIERARCH LONG KEYWORD"] = "Some value"

    # Comments
    header["KEYWITHC"] = ("Value", "This is a comment")

    # HISTORY and COMMENT
    header.add_history("First history entry")
    header.add_history("Second history entry")
    header.add_comment("First comment entry")
    header.add_comment("Second comment entry")

    hdu.writeto(filename, overwrite=True)


def test_complex_header():
    filename = "test_complex_header.fits"
    create_complex_header(filename)

    try:
        # Open with torchfits
        hdul = torchfits.HDUList.fromfile(filename)
        header = hdul[0].header

        # Check HIERARCH
        # Note: Astropy might normalize HIERARCH keys.
        # 'HIERARCH LONG KEYWORD' might become 'LONG KEYWORD' or similar depending on access.
        # But in the file it is HIERARCH.
        # TorchFits read_header uses fits_read_keyn which returns the key as stored.

        # Check if HIERARCH key exists
        # We might need to iterate to find it if exact name match is tricky
        found_hierarch = False
        for k in header.keys():
            if "LONG KEYWORD" in k:
                found_hierarch = True
                break

        if not found_hierarch:
            assert False, "FAILED: HIERARCH keyword not found"

        # Check HISTORY
        history = header.get_history()
        assert len(history) >= 2, "HISTORY missing or incomplete"

        # Check Comments
        comments = header.get_comment()
        assert len(comments) >= 2

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_fromfile_keeps_duplicate_history_and_comment(tmp_path) -> None:
    path = tmp_path / "hist.fits"
    header = fits.Header()
    header["OBJECT"] = "M13"
    header.add_history("first history entry")
    header.add_history("second history entry")
    header.add_comment("first comment")
    header.add_comment("second comment")
    fits.PrimaryHDU(data=np.arange(4, dtype=np.float32), header=header).writeto(
        str(path), overwrite=True
    )

    with torchfits.HDUList.fromfile(str(path)) as hdul:
        hdr = hdul[0].header
        assert hdr.get_history() == ["first history entry", "second history entry"]
        assert hdr.get_comment() == ["first comment", "second comment"]
        history_cards = [c for c in hdr.cards if c.key == "HISTORY"]
        assert len(history_cards) == 2


if __name__ == "__main__":
    test_complex_header()
