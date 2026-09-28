"""TableHDU.filter grammar, injection rejection, and Arrow WHERE masks."""

import numpy as np
import pyarrow as pa
import pytest
import torch

import torchfits
from torchfits._table.read import _where_mask_for_table
from torchfits.hdu import Header, TableHDU


def test_tablehdu_filter_accepts_grammar_and_rejects_python():
    data = {
        "x": torch.tensor([1, 2, 3]),
        "y": torch.tensor([4, 5, 6]),
    }
    table = TableHDU(data)

    # 1. Valid condition using the new evaluator
    filtered = table.filter("x > 1")
    assert filtered.num_rows == 2
    assert torch.equal(filtered["x"].flatten(), torch.tensor([2, 3]))

    # 2. Test IN and BETWEEN
    filtered_in = table.filter("x IN (1, 3)")
    assert filtered_in.num_rows == 2

    filtered_between = table.filter("x BETWEEN 1 AND 2")
    assert filtered_between.num_rows == 2

    # 3. Code injection attempts
    # The parser should fail on these because they don't match the SQL-like grammar
    with pytest.raises(ValueError):
        table.filter("__import__('os').system('echo malicious')")

    with pytest.raises(ValueError):
        table.filter("print('hello')")

    # 4. Attempting to access globals/locals via expression
    # Even if it bypasses the parser somehow, the evaluator only looks at data_map
    with pytest.raises(ValueError):
        table.filter("unknown_var > 0")


def test_where_mask_for_table_direct():
    data = {"a": np.array([1, 2, 3, 4, 5])}
    table = pa.table(data)

    # Test complex logical expression
    mask = _where_mask_for_table(table, "(a > 1 AND a < 5) OR a == 1")
    np.testing.assert_array_equal(mask.to_numpy(), [True, True, True, True, False])

    # Test NOT
    mask = _where_mask_for_table(table, "NOT a == 3")
    np.testing.assert_array_equal(mask.to_numpy(), [True, True, False, True, True])

    # Test IS NULL (Arrow uses None, not NaN, for null)
    table_nulls = pa.table({"b": pa.array([1.0, None, 3.0], type=pa.float64())})
    mask = _where_mask_for_table(table_nulls, "b IS NULL")
    np.testing.assert_array_equal(mask.to_numpy(), [False, True, False])

    mask = _where_mask_for_table(table_nulls, "b IS NOT NULL")
    np.testing.assert_array_equal(mask.to_numpy(), [True, False, True])


def _write_mixed_table(path):
    """A table with the three column kinds pyarrow.array() cannot ingest raw."""
    torchfits.write(
        str(path),
        {
            "ID": torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
            "NAME": ["alpha", "beta", "gamma", "delta", "epsilon"],
            "CH": ["a", "b", "c", "d", "e"],
        },
        header={
            "NAXIS": 2,
            "NAXIS1": 80,
            "NAXIS2": 5,
            "XTENSION": "BINTABLE",
            "BITPIX": 8,
            "PCOUNT": 0,
            "GCOUNT": 1,
            "TFIELDS": 3,
            "TTYPE1": "ID",
            "TFORM1": "1J",
            "TTYPE2": "NAME",
            "TFORM2": "8A",
            "TTYPE3": "CH",
            "TFORM3": "1A",
        },
        overwrite=True,
    )
    return str(path)


def test_filter_works_on_a_table_with_string_columns(tmp_path):
    """A packed FITS string column is a (rows, width) uint8 tensor.

    pyarrow.array() rejects N-dimensional input, and filter() used to hand it
    every column, so *any* filter on a table holding a string column raised
    ArrowInvalid -- including a numeric predicate that never mentions it.
    """
    import torchfits

    path = _write_mixed_table(tmp_path / "mixed.fits")
    table = torchfits.open(path)[1].materialize()

    numeric = table.filter("ID > 2")
    assert numeric.num_rows == 3
    assert numeric["ID"].tolist() == [3, 4, 5]
    # Non-numeric columns ride along unfiltered-but-sliced, shape intact.
    assert numeric["NAME"].shape == (3, 7)
    assert numeric["CH"].shape == (3, 1)

    by_name = table.filter("NAME == 'alpha'")
    assert by_name.num_rows == 1
    assert by_name.get_string_column("NAME") == ["alpha"]

    both = table.filter("ID > 2 AND NAME == 'gamma'")
    assert both.num_rows == 1
    assert both["ID"].tolist() == [3]

    # A 1-character string column decodes like any other: CHAR == 'c' is a
    # value comparison, not a comparison against raw char codes.
    one_char = table.filter("CH == 'c'")
    assert one_char.num_rows == 1
    assert one_char["ID"].tolist() == [3]


def test_filter_works_on_a_table_with_a_variable_length_column(tmp_path):
    path = str(tmp_path / "vla.fits")
    torchfits.write(
        path,
        {
            "ID": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            "VEC": [
                torch.tensor([1, 2]),
                torch.tensor([3]),
                torch.tensor([4, 5, 6]),
                torch.tensor([7]),
            ],
        },
        header={
            "NAXIS": 2,
            "NAXIS1": 80,
            "NAXIS2": 4,
            "XTENSION": "BINTABLE",
            "BITPIX": 8,
            "PCOUNT": 0,
            "GCOUNT": 1,
            "TFIELDS": 2,
            "TTYPE1": "ID",
            "TFORM1": "1J",
            "TTYPE2": "VEC",
            "TFORM2": "0J",
        },
        overwrite=True,
    )
    table = torchfits.open(path)[1].materialize()

    # The VLA column cannot be a predicate operand, but it must still be
    # sliced to the surviving rows rather than crashing the whole filter.
    out = table.filter("ID > 2")
    assert out.num_rows == 2
    assert [t.tolist() for t in out["VEC"]] == [[4, 5, 6], [7]]


def test_filter_works_on_a_table_with_a_vector_column(tmp_path):
    path = str(tmp_path / "vec.fits")
    torchfits.write(
        path,
        {
            "ID": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            "V": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]]),
        },
        header={
            "NAXIS": 2,
            "NAXIS1": 80,
            "NAXIS2": 4,
            "XTENSION": "BINTABLE",
            "BITPIX": 8,
            "PCOUNT": 0,
            "GCOUNT": 1,
            "TFIELDS": 2,
            "TTYPE1": "ID",
            "TFORM1": "1J",
            "TTYPE2": "V",
            "TFORM2": "3J",
        },
        overwrite=True,
    )
    table = torchfits.open(path)[1].materialize()

    out = table.filter("ID > 2")
    assert out.num_rows == 2
    assert tuple(out["V"].shape) == (2, 3)


def test_filter_on_a_columnless_table_reports_it_instead_of_keeping_every_row():
    """A 0-column BINTABLE still has NAXIS2 rows; a filter on it cannot match
    any column, and silently returning every row hid a typo'd column name."""
    table = TableHDU({}, None, Header({"NAXIS2": 6}))
    with pytest.raises(ValueError, match="no columns"):
        table.filter("NOPE > 1")
