"""TableHDU.filter grammar, injection rejection, and Arrow WHERE masks."""

import numpy as np
import pyarrow as pa
import pytest
import torch

from torchfits._table.read import _where_mask_for_table
from torchfits.hdu import TableHDU


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
