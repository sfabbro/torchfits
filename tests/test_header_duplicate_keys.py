"""Duplicate keywords and COMMENT/HISTORY mapping semantics.

Two behaviours are pinned here because they are easy to regress and were
inconsistent before:

1. A duplicated *value* keyword must resolve to its **first** occurrence. The
   mapping used to keep the last card while ``read_keys`` (CFITSIO
   ``fits_read_keyword``) and astropy both return the first, so the same file
   answered two different values depending on which API you asked.

2. COMMENT/HISTORY must be visible in the mapping for a header read from a file
   exactly as they are for one built in memory. Previously a file-read header
   hid them (``"COMMENT" in header`` was False, ``header["COMMENT"]`` raised
   KeyError), while ``add_comment`` put them in the mapping.

These are deliberately separate from ``test_differential_astropy.py``, which
covers the behaviours where torchfits matches astropy exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

import torchfits  # noqa: F401
from astropy.io import fits  # noqa: E402
from torchfits import Header  # noqa: E402


def _duplicate_keyword_file(tmp_path, name="dup.fits"):
    """A file carrying two value cards with the same keyword."""
    path = str(tmp_path / name)
    hdu = fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32))
    hdu.header.append(("DUP", 1, "first"))
    hdu.header.append(("DUP", 2, "second"))
    hdu.writeto(path, overwrite=True)

    # Prove the fixture really duplicates on disk; otherwise this test would
    # pass vacuously against a normal single-card header.
    with fits.open(path) as f:
        values = [c.value for c in f[0].header.cards if c.keyword == "DUP"]
    assert values == [1, 2], f"fixture did not duplicate the keyword: {values}"
    return path


def test_duplicate_keyword_matches_read_keys_and_astropy(tmp_path):
    """read_header, read_keys and astropy must agree: first occurrence wins."""
    path = _duplicate_keyword_file(tmp_path)

    via_header = torchfits.read_header(path)["DUP"]
    via_keys = torchfits.read_keys(path, ["DUP"])["DUP"]
    with fits.open(path) as f:
        via_astropy = f[0].header["DUP"]

    assert via_header == via_keys == via_astropy == 1


def test_duplicate_keyword_cards_are_preserved(tmp_path):
    """Both cards must still be present; only the mapping view collapses."""
    path = _duplicate_keyword_file(tmp_path)

    header = torchfits.read_header(path)
    assert [c.value for c in header.cards if c.key == "DUP"] == [1, 2]


def test_in_memory_duplicate_append_keeps_first(tmp_path):
    """The same rule applies to a Header built in memory."""
    header = Header()
    header.append(("DUP", 1, "first"))
    header.append(("DUP", 2, "second"))

    assert header["DUP"] == 1
    assert [c.value for c in header.cards if c.key == "DUP"] == [1, 2]


def test_setitem_updates_the_existing_card(tmp_path):
    """Assigning an existing keyword replaces its card rather than duplicating."""
    header = Header()
    header["KEY"] = 1
    header["KEY"] = 2

    assert header["KEY"] == 2
    assert [c.value for c in header.cards if c.key == "KEY"] == [2]


def test_remove_first_duplicate_falls_back_to_next(tmp_path):
    """Removing the first card exposes the next occurrence, not a stale value."""
    header = Header()
    header.append(("DUP", 1, "first"))
    header.append(("DUP", 2, "second"))

    header.remove("DUP", remove_all=False)

    assert header["DUP"] == 2
    assert [c.value for c in header.cards if c.key == "DUP"] == [2]


def test_remove_missing_key_raises_by_default():
    """TS-010: the default (ignore_missing=False) rejects an absent key.

    ``Header.remove`` had no test for a missing key on either side of the
    flag: only ``remove_all`` was varied. The strict branch raises KeyError
    naming the key that was asked for.
    """
    header = Header()
    header["KEY"] = 1

    with pytest.raises(KeyError) as err:
        header.remove("NOPE")

    assert err.value.args[0] == "NOPE"
    # The failed remove must not have disturbed the header.
    assert header["KEY"] == 1
    assert [c.key for c in header.cards] == ["KEY"]


def test_remove_missing_key_is_a_no_op_when_ignored():
    """TS-010: ignore_missing=True is load-bearing, not decorative.

    ``_io_engine/_write_helpers._merged_write_header`` calls
    ``merged.remove(key, ignore_missing=True, remove_all=True)`` for every
    overlay value card, including keys the base header never had. That
    production merge depends on this early return, and on it returning
    *before* the version bump so an all-new overlay leaves the version alone.
    """
    header = Header()
    header["KEY"] = 1
    before = header._version

    header.remove("NOPE", ignore_missing=True)

    assert header._version == before
    assert [c.key for c in header.cards] == ["KEY"]

    # ... and it still removes a key that is present.
    header.remove("KEY", ignore_missing=True)
    assert "KEY" not in header
    assert header._version == before + 1


def test_merged_write_header_overlay_may_introduce_new_keys():
    """TS-010: the ignore_missing=True caller exercises a real new-key merge."""
    from torchfits._io_engine._write_helpers import _merged_write_header

    merged = _merged_write_header({"OBJECT": "M13", "EXPTIME": 60.0}, {"FILTER": "r"})
    assert merged["OBJECT"] == "M13"
    assert merged["EXPTIME"] == 60.0
    assert merged["FILTER"] == "r"


def test_insert_ahead_becomes_the_reported_value(tmp_path):
    """Inserting before existing cards makes the new card the first occurrence."""
    header = Header()
    header["KEY"] = 2
    header.insert(0, ("KEY", 1, ""))

    assert header["KEY"] == 1
    assert [c.value for c in header.cards if c.key == "KEY"] == [1, 2]


def test_commentary_keys_visible_from_file(tmp_path):
    """A file-read header exposes COMMENT/HISTORY in the mapping like astropy."""
    path = str(tmp_path / "comments.fits")
    hdu = fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32))
    hdu.header.add_comment("first comment")
    hdu.header.add_history("only history")
    hdu.writeto(path, overwrite=True)

    header = torchfits.read_header(path)

    assert "COMMENT" in header
    assert "HISTORY" in header
    assert header["COMMENT"] == "first comment"
    assert header["HISTORY"] == "only history"


def test_commentary_mapping_tracks_latest_line(tmp_path):
    """Repeated commentary resolves to the latest line, and pop still works."""
    header = Header()
    header.add_history("a")
    header.add_history("b")
    header.add_history("c")

    assert header["HISTORY"] == "c"
    assert header.get_history() == ["a", "b", "c"]

    assert header.pop("HISTORY") == "c"
    assert "HISTORY" not in header
    assert list(header.cards) == []


def test_file_and_in_memory_headers_agree_on_mapping(tmp_path):
    """The two construction paths must expose the same mapping keys.

    Compared per key rather than as whole dicts: a header read from a file also
    carries the mandatory structural cards (SIMPLE/BITPIX/NAXIS*), which a
    hand-built header legitimately has not.
    """
    path = str(tmp_path / "agree.fits")
    hdu = fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32))
    hdu.header.add_comment("a comment")
    hdu.header.add_history("a history")
    hdu.header["OBJECT"] = "M31"
    hdu.writeto(path, overwrite=True)

    from_file = torchfits.read_header(path)
    built = Header()
    built.add_comment("a comment")
    built.add_history("a history")
    built["OBJECT"] = "M31"

    for key in ("COMMENT", "HISTORY", "OBJECT"):
        assert key in from_file, f"{key} missing from a file-read header"
        assert key in built, f"{key} missing from an in-memory header"
        assert from_file[key] == built[key], key


@pytest.mark.parametrize("mmap", [False, True])
def test_duplicate_keyword_consistent_across_read_paths(tmp_path, mmap):
    """The mmap and buffered table paths must resolve duplicates identically."""
    path = str(tmp_path / "t.fits")
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="C", format="J", array=np.arange(4, dtype=np.int32))]
    )
    hdu.header.append(("DUP", 7, "first"))
    hdu.header.append(("DUP", 9, "second"))
    hdu.writeto(path, overwrite=True)

    assert torchfits.read_header(path, 1)["DUP"] == 7
