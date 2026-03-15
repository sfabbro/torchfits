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
    with pytest.raises(AttributeError, match="is not allowed for security reasons"):
        hdu_table.filter("np.polyfit(col1, col2, 1) > 0")

    # Filtering with arbitrary builtin wrapped via something
    with pytest.raises(Exception):
        hdu_table.filter("max([1, 2]) > 1")
