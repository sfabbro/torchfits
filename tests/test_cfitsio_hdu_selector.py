"""CFITSIO HDU-selector filters: `file.fits[N]` must mean the same thing
everywhere (R2-015).

CFITSIO's selector is 0-based and *scopes* the file: opening `mef.fits[1]`
parks the handle on absolute HDU 2 and treats it as the new first HDU, so a
caller's `hdu=0` names that HDU. Measured on a PRIMARY/SCI/ERR/CAT file:
`[0]`->PRIMARY, `[1]`->SCI, `[2]`->ERR, `[3]`->CAT.

The data path applied that offset (FITSFile::ensure_hdu computes
`hdu_num + start_hdu_`); the path-based metadata probes moved to absolute
`hdu + 1` and discarded the position CFITSIO had already given them. So for a
single path `read(..., hdu=0)` returned SCI's pixels while `read_header`,
`read_shape` and `read_hdu_type` described PRIMARY.

These tests pin the CFITSIO convention rather than a torchfits invention:
indexing is relative to the filter, the HDU *count* stays absolute, and an
out-of-scope index raises on both paths alike.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import torchfits

# name, shape -- deliberately all different, so a shape mismatch can never be
# mistaken for agreement. Absolute HDU is the 1-based position in this list.
HDUS = [
    ("PRIMARY", (1, 1)),
    ("SCI", (4, 4)),
    ("ERR", (2, 6)),
    ("CAT", (5, 7)),
]


@pytest.fixture
def mef(tmp_path):
    path = str(tmp_path / "mef.fits")
    hdus = [
        fits.PrimaryHDU(np.zeros(HDUS[0][1], dtype=np.float32)),
        *[
            fits.ImageHDU(np.full(shape, float(i + 1), dtype=np.float32), name=name)
            for i, (name, shape) in enumerate(HDUS[1:])
        ],
    ]
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def _data_shape(path: str, hdu: int):
    return tuple(torchfits.read(path, hdu=hdu).shape)


def _meta_shape(path: str, hdu: int):
    _bitpix, shape = torchfits.read_shape(path, hdu)
    return tuple(shape)


# (filter, relative hdu) -> absolute 0-based HDU it must name.
CASES = [
    ("", 0),
    ("", 1),
    ("", 2),
    ("[0]", 0),
    ("[1]", 0),
    ("[1]", 1),
    ("[2]", 0),
    ("[2]", 1),
    ("[3]", 0),
]


def _absolute_index(suffix: str, hdu: int) -> int:
    """Absolute 0-based HDU that (filter, relative hdu) must name.

    `[N]` is 0-based, so it names absolute 0-based index N; `hdu` then counts
    forward from there. No filter means hdu is already the absolute index.
    """
    if not suffix:
        return hdu
    return int(suffix[1:-1]) + hdu


@pytest.mark.parametrize(("suffix", "hdu"), CASES)
def test_hdu_selector_is_zero_based_and_scopes_the_file(mef, suffix, hdu):
    """`[N]` names absolute 0-based HDU N, so `hdu` counts from there."""
    expected = HDUS[_absolute_index(suffix, hdu)][1]
    assert _data_shape(mef + suffix, hdu) == expected
    assert _meta_shape(mef + suffix, hdu) == expected


@pytest.mark.parametrize(("suffix", "hdu"), CASES)
def test_metadata_probes_agree_with_the_data_path(mef, suffix, hdu):
    """The bug: read() honoured the filter while read_shape/read_header did not."""
    q = mef + suffix
    assert _meta_shape(q, hdu) == _data_shape(q, hdu), (
        "image metadata described a different HDU than the data path read"
    )


@pytest.mark.parametrize(("suffix", "hdu"), CASES)
def test_header_names_the_same_hdu_the_data_path_read(mef, suffix, hdu):
    """read_header must name the same HDU read() returned pixels for."""
    q = mef + suffix
    expected_name = HDUS[_absolute_index(suffix, hdu)][0]
    assert str(torchfits.read_header(q, hdu).get("EXTNAME", "PRIMARY")) == (
        expected_name
    )
    assert torchfits.read_hdu_type(q, hdu) == "IMAGE"


@pytest.mark.parametrize("suffix", ["[4]", "[9]"])
def test_out_of_scope_selector_raises_on_every_path(mef, suffix):
    """A filter past the last HDU fails identically for data and metadata."""
    q = mef + suffix
    with pytest.raises((OSError, RuntimeError)):
        torchfits.read(q, hdu=0)
    with pytest.raises((OSError, RuntimeError)):
        torchfits.read_shape(q, 0)
    with pytest.raises((OSError, RuntimeError)):
        torchfits.read_header(q, 0)


def test_hdu_count_stays_absolute_under_a_filter(mef):
    """CFITSIO reports the whole file even when the path scopes it.

    fits_get_num_hdus is absolute -- it is the *indexing* that is scoped. So a
    filter must not shrink the count, or a caller looping over
    read_num_hdus() would stop early on a filtered path.
    """
    for suffix in ("", "[0]", "[1]", "[3]"):
        assert torchfits.read_num_hdus(mef + suffix) == len(HDUS)


def test_pixel_section_filter_is_unaffected(tmp_path):
    """`[1:2,1:2]` is a different CFITSIO feature and must keep working.

    This is the form the CLI cutout uses (test_cutout_cfitsio_section in
    tests/test_cli.py), so it is the one form with real users; the HDU-selector
    fix must not disturb it. A section filter selects pixels, not an HDU, so
    the HDU index must stay absolute -- which is the property that distinguishes
    it from `[N]` and that the fix could plausibly have broken.
    """
    path = str(tmp_path / "cut.fits")
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    fits.HDUList([fits.PrimaryHDU(data)]).writeto(path, overwrite=True)

    # CFITSIO sections are 1-based inclusive; [1:2,1:2] == the top-left 2x2.
    assert torchfits.read(path, hdu=0).shape == (4, 4)
    assert torchfits.read(path + "[1:2,1:2]", hdu=0).shape == (2, 2)
    # A section selects pixels, not an HDU, so no HDU offset applies and both
    # paths still describe the same HDU -- they report the section's extent.
    assert _meta_shape(path + "[1:2,1:2]", 0) == _data_shape(path + "[1:2,1:2]", 0)


def test_table_hdu_indexing_is_relative_to_the_filter(tmp_path):
    """read_colnames defaults to hdu=1, so the filter shifts what that means.

    On `tab.fits[1]` (absolute HDU 2) the default index 1 is absolute HDU 3,
    which does not exist -- and it must raise rather than silently answering
    about some other HDU. Passing hdu=0 names the table inside the scope.
    """
    path = str(tmp_path / "tab.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="A", format="J", array=np.arange(5))]
            ),
        ]
    ).writeto(path, overwrite=True)

    assert torchfits.read_colnames(path + "[1]", 0) == ["A"]
    assert torchfits.read_nrows(path + "[1]", 0) == 5
    with pytest.raises((OSError, RuntimeError)):
        torchfits.read_colnames(path + "[1]", 1)
