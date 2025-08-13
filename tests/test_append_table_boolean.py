import pytest
import torch

import torchfits as tf


def test_append_table_with_boolean_column(tmp_path):
    # Base table
    base = {"A": torch.arange(3, dtype=torch.int32)}
    out = tmp_path / "append_bool_table.fits"
    tf.write(str(out), base, header={"EXTNAME": "BASE"})

    # Table with boolean column to append
    t_append = {
        "FLAG": torch.tensor([True, False, True], dtype=torch.bool),
        "VAL": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
    }
    tf.write(str(out), t_append, header={"EXTNAME": "BOOL"}, append=True)

    # Read back appended table (HDU=2)
    table, hdr = tf.read(str(out), hdu=2, format="tensor")
    assert set(table.keys()) == {"FLAG", "VAL"}
    assert table["FLAG"].dtype == torch.bool
    assert torch.equal(table["FLAG"], t_append["FLAG"])  # round-trip
    assert torch.allclose(table["VAL"], t_append["VAL"])  # numeric column intact

    # Optional: verify CFITSIO logical column via astropy when available
    try:
        from astropy.io import fits
    except ImportError:
        return
    with fits.open(str(out)) as hdul:
        hdr2 = hdul[2].header
        # TFORMn should indicate logical (L); position depends on column order
        tforms = [hdr2.get(f"TFORM{i}", "") for i in range(1, 3)]
        assert any("L" in str(t) for t in tforms)
