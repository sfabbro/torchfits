import pytest
import torch
import torchfits
import torchfits.hdu


def test_security_eval_no_arbitrary_calls():
    # Setup data
    data_table = {
        "col1": torch.tensor([1, 2, 3], dtype=torch.int32),
        "col2": torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32),
    }
    header_table = torchfits.Header({"TBLKEY": "TBLVAL"})
    hdu_table = torchfits.TableHDU(data_table, header=header_table)

    # Filtering with valid numpy func
    res = hdu_table.filter("np.abs(col1) > 1")
    assert res.num_rows == 2

    # Filtering with arbitrary attribute of np not in allowed list
    with pytest.raises(ValueError):
        hdu_table.filter("np.polyfit(col1, col2, 1) > 0")

    # Filtering with arbitrary builtin wrapped via something
    with pytest.raises(Exception):
        hdu_table.filter("max([1, 2]) > 1")


def test_tablehdu_filter_no_eval_vulnerability():
    # Create a simple table
    data = {
        "x": torch.tensor([1, 2, 3]),
        "y": torch.tensor([4, 5, 6]),
    }
    table = torchfits.TableHDU(data)

    # 1. Valid condition using numexpr
    filtered = table.filter("x > 1")
    assert filtered.num_rows == 2

    # 2. Condition attempting to use python functions should fail
    # as numexpr does not support this and _safe_eval fallback was removed.
    with pytest.raises(Exception):
        table.filter("__import__('os').system('echo malicious')")

    with pytest.raises(Exception):
        table.filter("print('hello')")
