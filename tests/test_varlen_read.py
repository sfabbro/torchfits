import numpy as np
import pytest
import torch

import torchfits as tf


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_varlen_read_roundtrip(tmp_path, dtype):
    # Create a VLA table via writer
    arrays = [torch.arange(i + 1, dtype=dtype) * 0.5 for i in range(5)]
    path = tmp_path / "vla_roundtrip.fits"
    tf.write_variable_length_array(
        str(path), arrays, header={"EXTNAME": "VLA"}, overwrite=True
    )

    # Read back via torchfits
    data, header = tf.read(str(path), hdu=1, format="tensor")
    assert isinstance(data, dict)
    assert "ARRAY_DATA" in data
    col = data["ARRAY_DATA"]
    # Column should be a list of tensors matching row counts
    assert isinstance(col, list)
    assert len(col) == len(arrays)
    # Writer uses 'D' (double) for VLA, so dtype should be float64 on read
    for got, exp in zip(col, arrays):
        assert isinstance(got, torch.Tensor)
        assert got.dtype == torch.float64
        np.testing.assert_allclose(got.numpy(), exp.to(torch.float64).numpy())


def test_varlen_read_partial_rows(tmp_path):
    # Build a table with varying lengths, including empty row
    arrays = [
        torch.tensor([], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64),
        torch.tensor([6.0, 7.0, 8.0, 9.0], dtype=torch.float64),
        torch.tensor([10.0], dtype=torch.float64),
    ]
    path = tmp_path / "vla_partial.fits"
    tf.write_variable_length_array(
        str(path), arrays, header={"EXTNAME": "VLA"}, overwrite=True
    )

    # Read subset of rows (rows 2..5 -> 4 rows)
    start_row = 2
    num_rows = 4
    data, header = tf.read(
        str(path), hdu=1, start_row=start_row, num_rows=num_rows, format="tensor"
    )
    col = data["ARRAY_DATA"]
    assert isinstance(col, list)
    assert len(col) == num_rows
    for i, got in enumerate(col):
        exp = arrays[start_row + i]
        np.testing.assert_allclose(got.numpy(), exp.numpy())
