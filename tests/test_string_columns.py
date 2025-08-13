import pytest
import torch

import torchfits as tf


def test_write_table_all_empty_strings(tmp_path):
    out = tmp_path / "table_all_empty.fits"
    data = {"ID": torch.arange(5, dtype=torch.int32), "NAME": ["" for _ in range(5)]}
    tf.write_table(str(out), data, header={"EXTNAME": "CAT"}, overwrite=True)
    tbl = tf.read(str(out), hdu=1, format="table")
    names = tbl.data["NAME"]
    assert all(isinstance(s, str) and s == "" for s in names)
    try:
        from astropy.io import fits
    except ImportError:
        return
    with fits.open(str(out)) as hdul:
        hdr = hdul[1].header
        # find NAME column and verify minimal width formatting (1A acceptable)
        for i in range(1, hdr.get("TFIELDS", 0) + 1):
            if hdr.get(f"TTYPE{i}") == "NAME":
                assert hdr.get(f"TFORM{i}").lower().endswith("a")
                break


def test_write_table_mixed_length_strings(tmp_path):
    out = tmp_path / "table_mixed_strings.fits"
    long_str = "X" * 200
    data = {
        "ID": torch.arange(4, dtype=torch.int16),
        "NAME": ["A", "", long_str, "MIDLEN"],
    }
    tf.write_table(str(out), data, header={"EXTNAME": "CAT"}, overwrite=True)
    tbl = tf.read(str(out), hdu=1, format="table")
    names = tbl.data["NAME"]
    assert (
        names[0] == "A"
        and names[1] == ""
        and names[2] == long_str
        and names[3] == "MIDLEN"
    )
    try:
        from astropy.io import fits
    except ImportError:
        return
    with fits.open(str(out)) as hdul:
        hdr = hdul[1].header
        for i in range(1, hdr.get("TFIELDS", 0) + 1):
            if hdr.get(f"TTYPE{i}") == "NAME":
                assert hdr.get(f"TFORM{i}") in ("200A", "200a")
                break


def test_write_table_trailing_space_strings(tmp_path):
    out = tmp_path / "table_trailing_spaces.fits"
    data = {
        "VAL": torch.arange(3, dtype=torch.int16),
        "TXT": ["Alpha  ", "Beta \t", "Gamma"],
    }
    tf.write_table(str(out), data, header={"EXTNAME": "CAT"}, overwrite=True)
    tbl = tf.read(str(out), hdu=1, format="table")
    expected = ["Alpha", "Beta \t", "Gamma"]
    assert list(tbl.data["TXT"]) == expected
