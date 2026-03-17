import pytest
import torch
from torchfits.hdu import TableHDU


def test_tablehdu_filter_no_eval_vulnerability():
    # Create a simple table
    data = {
        "x": torch.tensor([1, 2, 3]),
        "y": torch.tensor([4, 5, 6]),
    }
    table = TableHDU(data)

    # 1. Valid condition using numexpr
    filtered = table.filter("x > 1")
    assert filtered.num_rows == 2

    # 2. Condition attempting to use python functions should fail
    # as numexpr does not support this and _safe_eval fallback was removed.
    with pytest.raises(Exception):
        table.filter("__import__('os').system('echo malicious')")

    with pytest.raises(Exception):
        table.filter("print('hello')")
