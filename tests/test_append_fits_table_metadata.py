import pytest
import torch

import torchfits as tf
from torchfits import ColumnInfo, FitsTable


def test_append_fitstable_with_metadata_units_and_desc(tmp_path):
    # Base table written first
    t1 = {"X": torch.arange(3, dtype=torch.int32)}
    out = tmp_path / "append_fitstable_meta.fits"
    tf.write(str(out), t1, header={"EXTNAME": "T1"})

    # Build FitsTable with metadata
    data = {
        "RA": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        "DEC": torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32),
        "ID": torch.tensor([10, 11, 12], dtype=torch.int32),
    }
    meta = {
        "RA": ColumnInfo(
            name="RA", dtype=torch.float32, unit="deg", description="Right Ascension"
        ),
        "DEC": ColumnInfo(
            name="DEC", dtype=torch.float32, unit="deg", description="Declination"
        ),
        "ID": ColumnInfo(name="ID", dtype=torch.int32, description="Identifier"),
    }
    ft = FitsTable(data, metadata=meta)

    # Append FitsTable using unified write()
    tf.write(str(out), ft, header={"EXTNAME": "CAT"}, append=True)

    # Validate via read() header dictionary
    t2_tuple = tf.read(str(out), hdu=2, format="tensor")
    assert isinstance(t2_tuple, tuple) and len(t2_tuple) == 2
    t2, hdr = t2_tuple
    assert set(t2.keys()) == {"RA", "DEC", "ID"}

    # If astropy is available, verify TUNIT/TCOMM placement
    try:
        from astropy.io import fits
    except ImportError:
        pytest.skip("astropy required for header validation")

    with fits.open(str(out)) as hdul:
        h = hdul[2].header
        # Units should be present (positions depend on column ordering)
        tunit_vals = [h.get(f"TUNIT{i}") for i in range(1, 4)]
        assert "deg" in tunit_vals
        # Descriptions are typically stored in TCOMM (or as comments on TTYPE); accept presence in either
        tcomm_vals = [h.get(f"TCOMM{i}") for i in range(1, 4)]
        assert any(v for v in tcomm_vals)
