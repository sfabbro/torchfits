"""Duplicate ``TTYPE`` cards must not corrupt or crash table reads.

FITS does not forbid repeated column names, and files in the wild carry them.
The C++ reader keyed its working map by column *name*, so a second ``TTYPE``
overwrote the first; the assembly loop then moved the same entry out twice and
the surviving column came back as a null tensor (``None`` in Python), while
``table.read`` raised ``TypeError: 'NoneType' object is not iterable``.

The fixture patches the on-disk card on purpose. astropy re-syncs ``TTYPE``
from ``ColDefs`` during ``writeto``, so building the file with astropy and then
poking ``hdu.header["TTYPE2"]`` yields an ordinary two-name table — such a test
passes with or without the fix. The duplicate must be forced at byte level, and
``_write_duplicate_ttype`` re-reads the header to prove it did.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torchfits  # noqa: E402
import torchfits.table  # noqa: E402
from astropy.io import fits  # noqa: E402


def _duplicate_ttype_file(path, formats=("J", "J"), values=((1, 2), (3, 4))):
    """Write a 2-column table whose *second* column re-uses the first name."""
    col1 = fits.Column(
        name="A", format=formats[0], array=np.array(values[0], dtype=np.int32)
    )
    col2 = fits.Column(
        name="B", format=formats[1], array=np.array(values[1], dtype=np.int32)
    )
    fits.BinTableHDU.from_columns([col1, col2]).writeto(path, overwrite=True)

    raw = bytearray(path.read_bytes())
    card = raw.find(b"TTYPE2")
    assert card != -1, "TTYPE2 card missing from the written header"
    quote = raw.index(b"'", card)
    raw[quote + 1] = ord("A")  # single byte: card width and data offsets stay valid
    path.write_bytes(bytes(raw))

    # Guard the fixture: if this stops producing a genuine duplicate the tests
    # below would silently become vacuous again.
    with fits.open(path) as hdul:
        names = [hdul[1].columns[k].name for k in range(len(hdul[1].columns))]
    assert names == ["A", "A"], f"fixture failed to duplicate TTYPE: {names}"
    return str(path)


@pytest.mark.parametrize("mmap", [False, True])
def test_duplicate_ttype_column_is_not_null(tmp_path, mmap):
    """The reported defect: a duplicated name returned a null tensor."""
    path = _duplicate_ttype_file(tmp_path / "dup.fits")
    res = torchfits.table.read_torch(path, mmap=mmap)

    assert list(res) == ["A"]
    assert res["A"] is not None, "duplicate TTYPE returned a null column"
    assert torch.is_tensor(res["A"])
    assert res["A"].tolist() == [3, 4]


def test_duplicate_ttype_does_not_break_table_read(tmp_path):
    """``table.read`` used to raise TypeError on the same file."""
    path = _duplicate_ttype_file(tmp_path / "dup_read.fits")
    table = torchfits.table.read(path)
    assert table is not None
    assert table["A"].to_pylist() == [3, 4]  # pyarrow ChunkedArray


@pytest.mark.parametrize("mmap", [False, True])
def test_duplicate_ttype_mixed_types(tmp_path, mmap):
    """Divergent types on the duplicated name must not leak between slots."""
    path = _duplicate_ttype_file(tmp_path / "dup_mixed.fits", formats=("J", "E"))
    res = torchfits.table.read_torch(path, mmap=mmap)

    assert res["A"] is not None
    # Second column wins in a Python dict, and must be its OWN tensor — the
    # float32 column, not the int32 one that shares its name.
    assert res["A"].dtype == torch.float32
    assert res["A"].tolist() == [3.0, 4.0]


def test_mmap_and_buffered_paths_agree(tmp_path):
    """Both reader paths must return the same values for a duplicated name."""
    path = _duplicate_ttype_file(tmp_path / "dup_agree.fits")
    assert (
        torchfits.table.read_torch(path, mmap=False)["A"].tolist()
        == torchfits.table.read_torch(path, mmap=True)["A"].tolist()
    )


def test_unique_names_are_untouched(tmp_path):
    """Guard against over-fixing: ordinary tables keep both columns."""
    path = str(tmp_path / "unique.fits")
    col1 = fits.Column(name="A", format="J", array=np.array([1, 2], dtype=np.int32))
    col2 = fits.Column(name="B", format="J", array=np.array([3, 4], dtype=np.int32))
    fits.BinTableHDU.from_columns([col1, col2]).writeto(path, overwrite=True)

    res = torchfits.table.read_torch(path, mmap=False)
    assert sorted(res) == ["A", "B"]
    assert res["A"].tolist() == [1, 2]
    assert res["B"].tolist() == [3, 4]
