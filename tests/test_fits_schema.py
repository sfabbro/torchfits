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
        "TFIELDS": 3,
        "TTYPE1": "PIX",
        "TFORM1": "1J",
        "TSCAL1": 1.0,
        "TZERO1": 2147483648.0,
        "TTYPE2": "IMPRECISE_J",
        "TFORM2": "1J",
        "TSCAL2": 1.0000000001,
        "TZERO2": 2147483648.0000001,
        "TTYPE3": "IMPRECISE_I",
        "TFORM3": "1I",
        "TSCAL3": 1.0,
        "TZERO3": 32768.0000000001,
    }
    dtypes = fits_schema.unsigned_column_dtypes_from_header(header)
    assert dtypes["PIX"] == torch.uint32
    assert dtypes["IMPRECISE_J"] == torch.uint32
    assert dtypes["IMPRECISE_I"] == torch.uint16


def test_case_variant_header_keys_resolve():
    """FITS keywords are case-insensitive; lowercase cards must not be dropped."""
    header = {
        "tfields": 1,
        "ttype1": "PIX",
        "tform1": "1J",
        "tscal1": 1.0,
        "tzero1": 2147483648.0,
        "tnull1": 7,
    }
    assert fits_schema.unsigned_column_dtypes_from_header(header) == {
        "PIX": torch.uint32
    }
    assert fits_schema.column_tnull_map(header) == {"PIX": 7}
    schema = fits_schema.build_table_schema_dict(header)
    assert [c["name"] for c in schema["columns"]] == ["PIX"]


def test_mixed_case_fast_path_columns_resolve():
    """TFIELDS fast path must resolve case-variant TTYPE/TFORM/TZERO cards."""
    header = {
        "TFIELDS": 2,
        "TTYPE1": "A",
        "TFORM1": "1J",
        "ttype2": "B",
        "tform2": "1I",
        "tscal2": 1.0,
        "tzero2": 32768.0,
    }
    cols = {c.name: c for c in fits_schema.table_columns(header)}
    assert sorted(cols) == ["A", "B"]
    assert cols["B"].tzero == 32768.0
    assert fits_schema.unsigned_column_dtypes_from_header(header)["B"] == torch.uint16


def test_complex_tform_schema_and_bignum_repeat():
    info = fits_schema.parse_tform("3C")
    assert (info.code, info.repeat) == ("C", 3)
    info_m = fits_schema.parse_tform("2M")
    assert (info_m.code, info_m.repeat) == ("M", 2)

    header = {"TFIELDS": 1, "TTYPE1": "CPLX", "TFORM1": "3C"}
    entry = fits_schema.build_table_schema_dict(header)["columns"][0]
    assert entry["code"] == "C"
    assert entry["repeat"] == 3

    # Repeat comes from the header; no int32 truncation of absurd values.
    big = fits_schema.parse_tform("4294967296J")
    assert big.repeat == 4294967296
