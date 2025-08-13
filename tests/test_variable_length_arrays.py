import numpy as np
import pytest
import torch

import torchfits as tf


@pytest.mark.skipif(not hasattr(tf, "write"), reason="write API missing")
def test_write_variable_length_array(tmp_path):
    # Using advanced API exposed via fits_reader_cpp: write_variable_length_array
    filename = tmp_path / "var_arr.fits"
    arrays = [torch.randn(i + 3, dtype=torch.float64) for i in range(5)]

    # Direct call into C++ backend (Python wrapper not yet provided)
    from torchfits import fits_reader_cpp  # type: ignore

    fits_reader_cpp.write_variable_length_array(str(filename), arrays, {}, True)

    # Read back via astropy for structure validation
    try:
        from astropy.io import fits
    except ImportError:
        pytest.skip("astropy required for validation")

    with fits.open(str(filename)) as hdul:
        assert len(hdul) == 2  # primary + table
        tbl = hdul[1].data
        # Each row is a variable-length array; verify lengths
        lengths = [len(x) for x in tbl["ARRAY_DATA"]]
        expected = [a.numel() for a in arrays]
        assert lengths == expected
        # Numerical content compare
        for row, arr in zip(tbl["ARRAY_DATA"], arrays):
            np.testing.assert_allclose(row, arr.numpy())
