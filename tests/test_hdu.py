import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from torchfits.hdu import TableHDU


def create_test_file(filename):
    if not os.path.exists(filename):
        data = {
            "ra": np.array([200.0, 201.0, 202.0], dtype=np.float64),
            "dec": np.array([45.0, 46.0, 47.0], dtype=np.float64),
            "flux": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "id": np.array([1, 2, 3], dtype=np.int32),
            "comments": np.array(
                ["This is star 1", "This is star 2", "This is star 3"], dtype="U20"
            ),
            "flag": np.array([True, False, True], dtype=bool),  # Boolean col
        }
        table = Table(data)
        hdu = fits.BinTableHDU(table, name="MY_TABLE")
        hdu.writeto(filename, overwrite=True)


@pytest.fixture(scope="module")
def fits_file(tmp_path_factory):
    # Write into a session tmp dir: never into the repo tree.
    file_path = str(tmp_path_factory.mktemp("hdu") / "table_example.fits")
    create_test_file(file_path)
    return file_path


def test_tablehdu_from_fits(fits_file):
    # Read the table from the FITS file
    table_hdu = TableHDU.from_fits(fits_file, hdu_index=1)

    # Check that the table has the correct number of rows and columns
    assert table_hdu.num_rows == 3
    # The "comments" column is now included as a byte tensor
    assert len(table_hdu.col_names) == 6
    assert "comments" in table_hdu.col_names

    # Check that the column names are correct
    assert "ra" in table_hdu.col_names
    assert "dec" in table_hdu.col_names
    assert "flux" in table_hdu.col_names
    assert "id" in table_hdu.col_names
    assert "flag" in table_hdu.col_names


def test_tablehdu_from_fits_uses_public_read_pipeline(fits_file, monkeypatch):
    from torchfits import table

    original = table.read_torch
    calls = []

    def traced_read_torch(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(table, "read_torch", traced_read_torch)
    table_hdu = TableHDU.from_fits(fits_file, hdu_index=1)

    assert table_hdu.num_rows == 3
    assert len(calls) == 1
    assert calls[0][1] == {"hdu": 1, "return_header": True}


def test_tablehdu_rejects_mismatched_column_lengths():
    with pytest.raises(ValueError, match="ra=2, dec=1"):
        TableHDU(
            {
                "ra": np.array([1.0, 2.0]),
                "dec": np.array([3.0]),
            }
        )


def test_tablehdu_rejects_non_mapping_input():
    with pytest.raises(TypeError, match="tensor_dict must be a dictionary"):
        TableHDU([1, 2, 3])  # type: ignore[arg-type]


def test_hdu_repr_html():
    import html

    import torch

    import torchfits
    from torchfits.hdu import TableHDURef, TensorHDU

    tensor = TensorHDU(
        data=torch.zeros(2, 3),
        header=torchfits.Header({"EXTNAME": "IMG"}),
    )
    tensor_html = tensor._repr_html_()
    assert "TensorHDU" in tensor_html
    assert html.escape("IMG") in tensor_html
    assert "(2, 3)" in tensor_html

    table = TableHDU({"x": torch.tensor([1.0, 2.0, 3.0])})
    table_html = table._repr_html_()
    assert "TableHDU" in table_html
    assert "table" in table_html.lower()
    assert "3" in table_html

    ref = TableHDURef(
        header=torchfits.Header({"EXTNAME": "CAT", "NAXIS2": 5}),
        source_path="/tmp/x.fits",
        source_hdu=1,
        columns=["a", "b"],
    )
    ref_html = ref._repr_html_()
    assert "TableHDURef" in ref_html
    assert html.escape("CAT") in ref_html
