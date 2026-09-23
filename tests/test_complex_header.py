import numpy as np
import pytest
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


def test_complex_header(tmp_path):
    filename = str(tmp_path / "test_complex_header.fits")
    create_complex_header(filename)

    hdul = torchfits.HDUList.fromfile(filename)
    try:
        header = hdul[0].header

        found_hierarch = False
        for k in header.keys():
            if "LONG KEYWORD" in k:
                found_hierarch = True
                break

        if not found_hierarch:
            assert False, "FAILED: HIERARCH keyword not found"

        history = header.get_history()
        assert len(history) >= 2, "HISTORY missing or incomplete"

        comments = header.get_comment()
        assert len(comments) >= 2
    finally:
        hdul.close()


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


def test_fromfile_internal_attribute_error_is_not_silently_rerouted(
    tmp_path, monkeypatch
):
    """An AttributeError raised *inside* the batch open must surface (as the
    documented wrapped RuntimeError), not silently trigger the legacy fallback
    and read the file through a different code path."""
    import torchfits._C as cpp

    path = str(tmp_path / "attr.fits")
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32)).writeto(
        path, overwrite=True
    )

    def boom(*args, **kwargs):
        raise AttributeError("internal binding bug")

    def no_fallback(*args, **kwargs):
        raise AssertionError("legacy fallback must not run")

    monkeypatch.setattr(cpp, "open_and_read_headers", boom)
    monkeypatch.setattr(cpp, "open_fits_file", no_fallback)

    with pytest.raises(RuntimeError) as exc_info:
        torchfits.HDUList.fromfile(path)
    assert isinstance(exc_info.value.__cause__, AttributeError)


def test_hdu_list_name_lookup_tracks_header_renames(tmp_path):
    """EXTNAME lookups must reflect header mutations, not a stale name index."""
    path = str(tmp_path / "rename.fits")
    hdu = fits.ImageHDU(data=np.zeros((2, 2), dtype=np.float32), name="OLD")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)

    hdul = torchfits.HDUList.fromfile(path)
    try:
        assert hdul["OLD"] is hdul[1]  # first lookup may build a name index
        hdul[1].header["EXTNAME"] = "NEW"
        assert hdul["NEW"] is hdul[1]
        with pytest.raises(KeyError):
            hdul["OLD"]
    finally:
        hdul.close()


def test_hdu_list_index_contract(tmp_path):
    """Integer indexing accepts anything __index__ (numpy ints included);
    non-index garbage raises TypeError, not a misleading KeyError."""
    path = str(tmp_path / "idx.fits")
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32)).writeto(
        path, overwrite=True
    )
    hdul = torchfits.HDUList.fromfile(path)
    try:
        assert hdul[np.int64(0)] is hdul[0]
        assert hdul[-1] is hdul[0]
        with pytest.raises(TypeError):
            hdul[1.5]
    finally:
        hdul.close()
