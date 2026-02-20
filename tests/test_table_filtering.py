
import pytest
import numpy as np
from astropy.io import fits

@pytest.fixture
def fits_file(tmp_path):
    path = str(tmp_path / "test_filter.fits")
    n_rows = 1000
    
    # Create data
    # FLOAT column 'MAG' : 0..100
    # INT column 'ID' : 0..1000
    # STRING column 'LABEL': 'A', 'B' alternating
    
    mag = np.linspace(0, 100, n_rows, dtype=np.float32)
    ids = np.arange(n_rows, dtype=np.int32)
    short_col = np.arange(n_rows, dtype=np.int16)
    
    c1 = fits.Column(name='MAG', format='E', array=mag)
    c2 = fits.Column(name='ID', format='J', array=ids)
    c3 = fits.Column(name='SHORT_VAL', format='I', array=short_col)
    
    # Add a string column if possible, but let's start with numeric
    
    hdu = fits.BinTableHDU.from_columns([c1, c2, c3])
    hdu.writeto(path)
    return path

def test_filter_lt(fits_file):
    import torchfits.cpp

    
    # MAG < 50.0. linspace(0,100,1000) means values are 0.0, 0.1, ... 99.9.
    # < 50.0 should be indices 0 to 499 (500 items).
    # 500th item is 50.0 approx.
    
    filters = [("MAG", "<", 50.0)]
    cols = ["ID", "MAG"]
    
    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, cols, filters)
    
    assert "ID" in data
    assert "MAG" in data
    
    ids = data["ID"]
    mags = data["MAG"]
    
    assert len(ids) == 500  # approximately
    # Exact check
    assert (mags < 50.0).all()
    assert len(mags) > 490 and len(mags) < 510

def test_filter_gt(fits_file):
    import torchfits.cpp
    
    filters = [("ID", ">", 800)]
    cols = ["ID"]
    
    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, cols, filters)
    ids = data["ID"]
    
    # 801 to 999 -> 199 items
    assert len(ids) == 199
    assert (ids > 800).all()

def test_table_read_integration(fits_file):
    import torchfits
    
    # Test integration via torchfits.table.read(where=...)
    # MAG < 50.0 should use fast path
    
    t = torchfits.table.read(fits_file, where="MAG < 50.0")
    assert len(t) == 500 # approx
    mags = t["MAG"].to_numpy()
    assert (mags < 50.0).all()
    
    # Test fallback for OR (should work via slow path)
    t_slow = torchfits.table.read(fits_file, where="MAG < 10.0 OR MAG > 90.0")
    # 0..10 (approx 100) + 90..100 (approx 100) = 200
    assert len(t_slow) > 180 and len(t_slow) < 220
    mags_slow = t_slow["MAG"].to_numpy()
    assert ((mags_slow < 10.0) | (mags_slow > 90.0)).all()


def test_table_read_where_torch_backend(fits_file):
    import torchfits

    t = torchfits.table.read(fits_file, where="MAG > 10.0 AND MAG < 20.0", backend="torch")
    mags = t["MAG"].to_numpy()
    assert len(t) > 90 and len(t) < 110
    assert ((mags > 10.0) & (mags < 20.0)).all()


def test_filter_eq(fits_file):
    import torchfits.cpp
    filters = [("ID", "==", 500)]
    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, ["ID"], filters)
    assert len(data["ID"]) == 1
    assert data["ID"][0].item() == 500

def test_filter_compound(fits_file):
    import torchfits.cpp
    # ID > 100 AND ID < 200
    filters = [("ID", ">", 100), ("ID", "<", 200)]
    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, ["ID"], filters)
    ids = data["ID"]
    
    assert len(ids) == 99 # 101 to 199
    assert ids.min() == 101
    assert ids.max() == 199

def test_filter_short(fits_file):
    import torchfits.cpp
    # SHORT_VAL is int16. Filter on it.
    filters = [("SHORT_VAL", "==", 10)]
    data = torchfits.cpp.read_fits_table_filtered(fits_file, 1, ["SHORT_VAL"], filters)
    assert len(data["SHORT_VAL"]) == 1
    assert data["SHORT_VAL"][0].item() == 10
