import pytest
import torch

import torchfits as tf


def test_append_table_to_fits(tmp_path):
    # Create initial table
    table1 = {"A": torch.arange(5), "B": torch.arange(5, 10)}
    out = tmp_path / "append_table.fits"
    tf.write(str(out), table1, header={"EXTNAME": "TBL1"})

    # Append a second table
    table2 = {"A": torch.arange(10, 15), "B": torch.arange(15, 20)}
    tf.write(str(out), table2, header={"EXTNAME": "TBL2"}, append=True)

    # Read back both tables as (dict, header)
    t1_tuple = tf.read(str(out), hdu=1, format="tensor")
    t2_tuple = tf.read(str(out), hdu=2, format="tensor")
    assert isinstance(t1_tuple, tuple) and len(t1_tuple) == 2
    assert isinstance(t2_tuple, tuple) and len(t2_tuple) == 2
    t1, h1 = t1_tuple
    t2, h2 = t2_tuple
    assert set(t1.keys()) == {"A", "B"}
    assert set(t2.keys()) == {"A", "B"}
    assert torch.allclose(t1["A"], table1["A"])
    assert torch.allclose(t2["A"], table2["A"])
    assert torch.allclose(t1["B"], table1["B"])
    assert torch.allclose(t2["B"], table2["B"])
    # Check EXTNAMEs if available
    if isinstance(h1, dict) and h1 is not None:
        assert h1.get("EXTNAME", "").upper() == "TBL1"
    if isinstance(h2, dict) and h2 is not None:
        assert h2.get("EXTNAME", "").upper() == "TBL2"
