"""A nameless table column must look the same to schema and to the readers.

FITS permits a column with no ``TTYPE`` card.  The readers synthesize
``COL<i>``; the Python schema used to drop the column entirely (shifting every
later index) and, after a first fix, named it ``""``.  Either way ``schema``
disagreed with ``read``/``read_torch``, so nothing could address the column.
"""

import pytest

import torchfits
from torchfits._io_engine._write_helpers import _delete_header_key_if_supported


@pytest.fixture()
def nameless_second_column(tmp_path):
    path = str(tmp_path / "nameless.fits")
    torchfits.table.write(path, {"x": [0, 1, 2], "y": [10, 20, 30]})
    # Same route as `torchfits setkey --delete`: CFITSIO card delete, so the
    # file really has no TTYPE2 (writing a mutated Header re-adds the card).
    _delete_header_key_if_supported(path, 1, "TTYPE2")
    return path


def test_schema_and_readers_agree_on_nameless_column(nameless_second_column):
    path = nameless_second_column

    with torchfits.open(path) as hdul:
        names = [c["name"] for c in hdul[1].schema["columns"]]
    assert names == ["x", "COL2"]

    arrow = torchfits.table.read(path, hdu=1)
    assert arrow.column_names == names
    assert arrow.to_pydict() == {"x": [0, 1, 2], "COL2": [10, 20, 30]}

    tensors = torchfits.table.read_torch(path, hdu=1)
    assert list(tensors.keys()) == names
    assert tensors["COL2"].tolist() == [10, 20, 30]


def test_nameless_column_is_not_duplicated_on_sliced_or_filtered_reads(
    nameless_second_column,
):
    """One nameless column must stay one column once slices or filters are involved.

    The filtered and sliced paths build their field list from the header walk
    plus (historically) a separate unnamed-column pass. Once the walk stopped
    dropping nameless columns, the two sources listed it twice and an empty or
    filtered read grew a duplicate ``COL2``.
    """
    path = nameless_second_column

    empty = torchfits.table.read(path, hdu=1, row_slice=(0, 0))
    assert empty.column_names == ["x", "COL2"]
    assert empty.num_rows == 0

    filtered = torchfits.table.read(path, hdu=1, where="x > 0")
    assert filtered.column_names == ["x", "COL2"]
    assert filtered.to_pydict() == {"x": [1, 2], "COL2": [20, 30]}

    filtered_torch = torchfits.table.read_torch(path, hdu=1, where="x > 0")
    assert list(filtered_torch.keys()) == ["x", "COL2"]
    assert filtered_torch["COL2"].tolist() == [20, 30]
