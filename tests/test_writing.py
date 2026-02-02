import os

import numpy as np
import torch
from astropy.io import fits

import torchfits


def test_writing():
    filename = "test_write.fits"

    # Create TensorHDU
    data_tensor = torch.randn(10, 10)
    header_tensor = torchfits.Header({"TESTKEY": "TESTVAL"})
    hdu_tensor = torchfits.TensorHDU(data=data_tensor, header=header_tensor)

    # Create TableHDU
    data_table = {
        "col1": torch.tensor([1, 2, 3], dtype=torch.int32),
        "col2": torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32),
    }
    header_table = torchfits.Header({"TBLKEY": "TBLVAL"})
    hdu_table = torchfits.TableHDU(data_table, header=header_table)

    # Create HDUList
    hdul = torchfits.HDUList([hdu_tensor, hdu_table])

    def assert_written(path: str) -> None:
        with fits.open(path) as hdul_astro:
            # Verify TensorHDU (Primary)
            # Handle endianness for PyTorch
            data_numpy = hdul_astro[0].data
            if data_numpy.dtype.byteorder == ">":
                data_numpy = data_numpy.astype(data_numpy.dtype.newbyteorder("<"))
            data_read = torch.tensor(data_numpy)
            assert torch.allclose(data_read, data_tensor)
            assert hdul_astro[0].header["TESTKEY"] == "TESTVAL"

            # Verify TableHDU (Extension 1)
            # Note: Astropy might read as BinTableHDU
            data_read_table = hdul_astro[1].data
            assert np.allclose(data_read_table["col1"], data_table["col1"].numpy())
            assert np.allclose(data_read_table["col2"], data_table["col2"].numpy())
            assert hdul_astro[1].header["TBLKEY"] == "TBLVAL"

    filenames = [filename, "test_write_hdulist.fits"]

    try:
        # Write via HDUList
        hdul.write(filenames[0], overwrite=True)
        assert_written(filenames[0])

        # Write via top-level helper
        torchfits.write(filenames[1], hdul, overwrite=True)
        assert_written(filenames[1])
    finally:
        for path in filenames:
            if os.path.exists(path):
                os.remove(path)
