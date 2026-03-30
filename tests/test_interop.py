import os
import tempfile
from unittest import mock

import numpy as np
import pytest
import torch
from astropy.table import Table

import torchfits


def test_to_pandas_decode_bytes():
    pytest.importorskip("pandas")

    table = Table(
        {
            "RA": np.array([10.1, 10.2], dtype=np.float64),
            "NAME": np.array(["STAR_A", "STAR_B"], dtype="U8"),
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        table.write(f.name, format="fits", overwrite=True)
        path = f.name

    try:
        data, _ = torchfits.read(path, hdu=1, return_header=True)
        df = torchfits.to_pandas(data, decode_bytes=True)

        assert df.shape[0] == 2
        assert df["NAME"].tolist() == ["STAR_A", "STAR_B"]
        assert np.allclose(df["RA"].to_numpy(), [10.1, 10.2])
    finally:
        os.unlink(path)


def test_to_arrow_vla_list():
    pytest.importorskip("pyarrow")

    vla = np.array([np.array([1, 2]), np.array([3])], dtype=object)
    table = Table(
        {
            "RA": np.array([10.1, 10.2], dtype=np.float64),
            "VLA": vla,
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        table.write(f.name, format="fits", overwrite=True)
        path = f.name

    try:
        data, _ = torchfits.read(path, hdu=1, return_header=True)
        arrow = torchfits.to_arrow(data, vla_policy="list")

        assert arrow.num_rows == 2
        assert "VLA" in arrow.column_names
    finally:
        os.unlink(path)


def test_to_arrow_vla_invalid_policy():
    pytest.importorskip("pyarrow")

    data = {
        "RA": torch.tensor([10.1, 10.2], dtype=torch.float64),
        "VLA": [torch.tensor([1, 2]), torch.tensor([3])],
    }
    with pytest.raises(ValueError, match="vla_policy must be 'list' or 'drop'"):
        torchfits.to_arrow(data, vla_policy="invalid_policy")


def test_to_pandas_missing_pandas():
    with mock.patch.dict("sys.modules", {"pandas": None}):
        with pytest.raises(
            ImportError, match="Pandas is required for to_pandas conversion."
        ):
            torchfits.to_pandas({"a": torch.tensor([1, 2, 3])})
