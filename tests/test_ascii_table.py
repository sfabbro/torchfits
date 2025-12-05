import torch
import torchfits
import numpy as np
from astropy.io import fits
from astropy.table import Table
import os


def create_ascii_table(filename):
    # Create a simple table
    t = Table()
    t["a"] = [1, 2, 3]
    t["b"] = [4.5, 5.5, 6.5]
    t["c"] = ["x", "y", "z"]

    # Write as ASCII table (TableHDU)
    # Note: BinTableHDU is binary. TableHDU is ASCII.
    # We need to convert Table to FITS columns
    # fits.TableHDU(data=t) doesn't work directly with Table object in some versions?
    # Let's try converting to array
    # But ASCII table requires specific handling.

    # Easiest way:
    # hdu = fits.table_to_hdu(t) # This creates BinTableHDU

    # Manually create TableHDU
    # We need to define columns.
    # ASCII table columns are TFORM='I4', 'F8.2', 'A10' etc.

    col1 = fits.Column(name="a", format="I4", array=np.array([1, 2, 3]))
    col2 = fits.Column(name="b", format="F8.2", array=np.array([4.5, 5.5, 6.5]))
    col3 = fits.Column(name="c", format="A10", array=np.array(["x", "y", "z"]))
    cols = fits.ColDefs([col1, col2, col3])
    hdu = fits.TableHDU.from_columns(cols)
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(filename, overwrite=True)


def test_ascii_table():
    filename = "test_ascii.fits"
    create_ascii_table(filename)

    # Test standard reading
    hdul = torchfits.HDUList.fromfile(filename)
    table_hdu = hdul[1]

    data = table_hdu.data

    assert torch.allclose(data["a"], torch.tensor([1, 2, 3], dtype=torch.int32))
    assert torch.allclose(data["b"], torch.tensor([4.5, 5.5, 6.5], dtype=torch.float64))

    # Strings might be bytes or converted?
    # torchfits usually returns byte tensor for strings.
    # 'x' -> 120.

    # Verify string content
    # Row 0 should start with 'x' (120)
    assert data["c"][0, 0] == 120
    # Row 1 should start with 'y' (121)
    assert data["c"][1, 0] == 121
    # Row 2 should start with 'z' (122)
    assert data["c"][2, 0] == 122

    # Test mmap reading (should fail or fallback?)
    # HDUList.fromfile doesn't expose mmap flag for tables directly,
    # but TableHDU might use it if configured.
    # Currently TableHDU uses `read_fits_table` which takes `mmap` arg.
    # But `HDUList` calls `read_fits_table` without mmap arg (defaults to false) or with?
    # Let's check hdu.py.
    pass

    # Clean up
    if os.path.exists(filename):
        os.remove(filename)

    # Clean up
    if os.path.exists(filename):
        os.remove(filename)
