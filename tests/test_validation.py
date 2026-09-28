import os

import numpy as np
import torch
from astropy.io import fits

import torchfits


def create_valid_fits(filename):
    hdu = fits.PrimaryHDU(data=torch.zeros(10, 10).numpy())
    hdu.writeto(filename, overwrite=True)


def test_validation(tmp_path):
    filename = str(tmp_path / "test_valid.fits")
    create_valid_fits(filename)

    hdul = torchfits.HDUList.fromfile(filename)
    is_valid = hdul.validate()
    assert is_valid


def test_validate_is_false_for_a_detached_image_hdu(tmp_path):
    """validate() must reach the data, not just the header.

    The TensorHDU branch was guarded on `if hdu._file_handle:`, which is
    exactly the state mark_closed() clears, so a closed HDU -- the state
    HDUList.close() leaves every HDU in -- skipped the check and validated.
    """
    filename = str(tmp_path / "detached.fits")
    create_valid_fits(filename)

    hdul = torchfits.HDUList.fromfile(filename)
    assert hdul.validate()
    hdul[0].mark_closed()
    assert not hdul.validate()


def test_validate_is_false_for_a_detached_table_hdu(tmp_path):
    """A file-backed table HDU is a TableHDURef, which the TableHDU-only
    isinstance check skipped, so no table was ever validated."""
    filename = str(tmp_path / "table.fits")
    fits.BinTableHDU.from_columns(
        [fits.Column(name="X", format="1J", array=np.array([1, 2, 3]))]
    ).writeto(filename, overwrite=True)

    hdul = torchfits.HDUList.fromfile(filename)
    assert hdul.validate()
    hdul[1]._source_path = str(tmp_path / "gone.fits")
    assert not hdul.validate()


def test_validate_reads_a_table_hdu_and_a_broken_file_is_invalid(tmp_path):
    filename = str(tmp_path / "table2.fits")
    fits.BinTableHDU.from_columns(
        [fits.Column(name="X", format="1J", array=np.array([1, 2, 3]))]
    ).writeto(filename, overwrite=True)

    hdul = torchfits.HDUList.fromfile(filename)
    assert hdul.validate()
    # Delete the file under the open handle: the header is still in memory,
    # so only an actual read notices.
    os.remove(filename)
    assert not hdul.validate()
