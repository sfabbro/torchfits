
import pytest
import torch
from torchfits.hdu import HDUList, TensorHDU, TableHDU, Header

def test_hdu_list_repr():
    # Mock data
    h1 = TensorHDU(header=Header([("EXTNAME", "PRIMARY", ""), ("NAXIS", 2, ""), ("NAXIS1", 100, ""), ("NAXIS2", 100, "")]))
    h1._data = torch.zeros(100, 100)

    h2 = TableHDU({"col1": torch.tensor([1, 2, 3])}, header=Header([("EXTNAME", "EVENTS", "")]))

    hdul = HDUList([h1, h2])

    repr_str = repr(hdul)
    print(repr_str)

    assert "Filename:" in repr_str
    assert "No." in repr_str
    assert "PRIMARY" in repr_str
    assert "TensorHDU" in repr_str
    assert "(100, 100)" in repr_str
    assert "EVENTS" in repr_str
    assert "TableHDU" in repr_str
    assert "3R x 1C" in repr_str

def test_tensor_hdu_repr():
    h1 = TensorHDU(header=Header([("EXTNAME", "SCI", "")]))
    h1._data = torch.zeros(10, 10, dtype=torch.float32)

    repr_str = repr(h1)
    assert "<TensorHDU SCI" in repr_str
    assert "data=(10, 10)" in repr_str
    assert "dtype=torch.float32" in repr_str

def test_table_hdu_repr():
    h2 = TableHDU({"col1": torch.tensor([1, 2])}, header=Header([("EXTNAME", "CATALOG", "")]))

    repr_str = repr(h2)
    assert "<TableHDU CATALOG" in repr_str
    assert "2 rows x 1 cols" in repr_str
