import pytest
from torchfits.hdu import TableHDU
import os
from astropy.io import fits
from astropy.table import Table
import numpy as np

def create_test_file(filename):
    if not os.path.exists(filename):
        names = ["ra", "dec", "flux", "id", "comments", "flag"]
        formats = ["D", "D", "E", "J", "20A", "B"]  # Added a boolean column
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
def fits_file():
    # Get the absolute path to the example file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'table_example.fits')
    create_test_file(file_path)
    return file_path

def test_tablehdu_from_fits(fits_file):
    # Read the table from the FITS file
    table_hdu = TableHDU.from_fits(fits_file, hdu_index=1)
    
    # Check that the table has the correct number of rows and columns
    assert table_hdu.num_rows == 3
    # The "comments" column is skipped for now, so there are 5 columns
    assert len(table_hdu.col_names) == 5
    
    # Check that the column names are correct
    assert "ra" in table_hdu.col_names
    assert "dec" in table_hdu.col_names
    assert "flux" in table_hdu.col_names
    assert "id" in table_hdu.col_names
    assert "flag" in table_hdu.col_names
