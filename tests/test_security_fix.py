import pytest
import numpy as np
import torch
from torchfits.hdu import TableHDU


def test_tablehdu_filter_security():
    # Create a simple table
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


def test_evaluate_where_direct():
    from torchfits._where import _parse_where_expression, evaluate_where

    data = {"a": np.array([1, 2, 3, 4, 5])}

    # Test complex logical expression
    ast = _parse_where_expression("(a > 1 AND a < 5) OR a == 1")
    mask = evaluate_where(ast, data)
    np.testing.assert_array_equal(mask, [True, True, True, True, False])

    # Test NOT
    ast = _parse_where_expression("NOT a == 3")
    mask = evaluate_where(ast, data)
    np.testing.assert_array_equal(mask, [True, True, False, True, True])

    # Test IS NULL
    data_nulls = {"b": np.array([1.0, np.nan, 3.0])}
    ast = _parse_where_expression("b IS NULL")
    mask = evaluate_where(ast, data_nulls)
    np.testing.assert_array_equal(mask, [False, True, False])

    ast = _parse_where_expression("b IS NOT NULL")
    mask = evaluate_where(ast, data_nulls)
    np.testing.assert_array_equal(mask, [True, False, True])
