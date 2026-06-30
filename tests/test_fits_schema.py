"""Tests for shared FITS table schema parsing."""

from __future__ import annotations

import torch

from torchfits import fits_schema


def test_parse_tform_scalar_and_vla():
    info = fits_schema.parse_tform("20A")
    assert info.vla is False
    assert info.code == "A"
    assert info.repeat == 20
    assert info.is_string is True

    vla = fits_schema.parse_tform("1PJ")
    assert vla.vla is True
    assert vla.code == "J"
    assert vla.vla_descriptor == "P"


def test_build_table_schema_dict():
    header = {
        "TFIELDS": 2,
        "TTYPE1": "NAME",
        "TFORM1": "10A",
        "TTYPE2": "FLUX",
        "TFORM2": "1E",
    }
    schema = fits_schema.build_table_schema_dict(header)
    assert [c["name"] for c in schema["columns"]] == ["NAME", "FLUX"]
    assert schema["string_columns"] == ["NAME"]
    assert schema["vla_columns"] == []


def test_unsigned_columns_from_header():
    header = {
        "TFIELDS": 1,
        "TTYPE1": "PIX",
        "TFORM1": "1J",
        "TSCAL1": 1.0,
        "TZERO1": 2147483648.0,
    }
    dtypes = fits_schema.unsigned_column_dtypes_from_header(header)
    assert dtypes["PIX"] == torch.uint32
