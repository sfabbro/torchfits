"""Tests for shared FITS table schema parsing."""

from __future__ import annotations

import pytest
import torch

import torchfits
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


# ---------------------------------------------------------------------------
# Deep-review unit 10, TE-005: the `selected=` column-subset filter
# ---------------------------------------------------------------------------
#
# `iter_table_columns(selected=...)` is what makes `TableHDURef.select()`
# report narrowed *metadata* rather than the whole header. Nothing pinned it:
# the one existing test that reached the branch used a ONE-column table,
# where a selection and "all columns" are the same answer by construction.
# Inverting the filter was caught only because it produced zero columns;
# ignoring it entirely was completely silent.


def test_iter_table_columns_selected_narrows_the_walk():
    """`selected` keeps only the named columns, in header order."""
    header = {
        "TFIELDS": 3,
        "TTYPE1": "A",
        "TFORM1": "1J",
        "TTYPE2": "NAME",
        "TFORM2": "10A",
        "TTYPE3": "V",
        "TFORM3": "1E",
    }
    assert [c.name for c in fits_schema.iter_table_columns(header)] == [
        "A",
        "NAME",
        "V",
    ]
    assert [
        c.name for c in fits_schema.iter_table_columns(header, selected={"NAME", "V"})
    ] == ["NAME", "V"]
    # A selection that names nothing yields nothing -- not everything.
    assert list(fits_schema.iter_table_columns(header, selected=set())) == []


def test_table_columns_and_string_column_names_honour_selected():
    header = {
        "TFIELDS": 3,
        "TTYPE1": "A",
        "TFORM1": "1J",
        "TTYPE2": "NAME",
        "TFORM2": "10A",
        "TTYPE3": "OTHER",
        "TFORM3": "8A",
    }
    assert [c.name for c in fits_schema.table_columns(header, selected={"A"})] == ["A"]
    assert fits_schema.string_column_names(header) == ["NAME", "OTHER"]
    # A selection that drops a string column must drop it here too.
    assert fits_schema.string_column_names(header, selected={"A"}) == []
    assert fits_schema.string_column_names(header, selected={"OTHER"}) == ["OTHER"]


def test_build_table_schema_dict_honours_selected_columns():
    header = {
        "TFIELDS": 3,
        "TTYPE1": "A",
        "TFORM1": "1J",
        "TTYPE2": "NAME",
        "TFORM2": "10A",
        "TTYPE3": "V",
        "TFORM3": "1PJ",
    }
    full = fits_schema.build_table_schema_dict(header)
    assert [c["name"] for c in full["columns"]] == ["A", "NAME", "V"]
    assert full["string_columns"] == ["NAME"]
    assert full["vla_columns"] == ["V"]

    narrowed = fits_schema.build_table_schema_dict(header, selected_columns=["V"])
    assert [c["name"] for c in narrowed["columns"]] == ["V"]
    assert narrowed["string_columns"] == []
    assert narrowed["vla_columns"] == ["V"]


def test_tablehduref_select_narrows_schema_metadata(tmp_path):
    """The public route: `TableHDURef.select()` must narrow `.schema`.

    The data projection after `select()` is pinned elsewhere; this is the
    metadata half, which a one-column fixture cannot distinguish.
    """
    import numpy as np

    import torchfits

    path = tmp_path / "three.fits"
    torchfits.table.write(
        str(path),
        data={
            "A": torch.tensor([1.0, 2.0], dtype=torch.float64),
            "B": torch.tensor([3, 4], dtype=torch.int32),
            "C": np.array([b"xx", b"yy"], dtype="S2"),
        },
        overwrite=True,
    )
    with torchfits.open(str(path)) as hdul:
        ref = hdul[1]
        assert [c["name"] for c in ref.schema["columns"]] == ["A", "B", "C"]
        assert ref.string_columns == ["C"]

        selected = ref.select(["A", "C"])
        assert [c["name"] for c in selected.schema["columns"]] == ["A", "C"]
        assert selected.schema["string_columns"] == ["C"]
        assert selected.string_columns == ["C"]

        # Dropping the string column must drop it from string_columns too,
        # which a fixture where the only column is the selected one cannot see.
        only_b = ref.select(["B"])
        assert [c["name"] for c in only_b.schema["columns"]] == ["B"]
        assert only_b.schema["string_columns"] == []
        assert only_b.string_columns == []


# ---------------------------------------------------------------------------
# Deep-review unit 10, TS-006: the unsigned convention must require TZERO
# ---------------------------------------------------------------------------
#
# `unsigned_column_dtype_names_from_header` recognises the two standard
# unsigned FITS conventions by pairing a code with an exact TZERO. Mutating
# the TZERO test away (`if code == "I"`) left 143 tests green, because every
# fixture with an `I` column happened to be an unsigned one. A *signed* int16
# column -- including an offset column, where TZERO is a small non-zero value
# rather than 32768 -- is the discriminating case, and nothing had one.


def test_signed_columns_are_not_classified_unsigned():
    """A plain signed int16/int32 column must stay signed."""
    header = {
        "TFIELDS": 3,
        "TTYPE1": "SIGNED_I",
        "TFORM1": "1I",
        "TTYPE2": "SIGNED_J",
        "TFORM2": "1J",
        "TTYPE3": "OFFSET_I",
        "TFORM3": "1I",
        "TZERO3": 100.0,
    }
    assert fits_schema.unsigned_column_dtype_names_from_header(header) == {}
    assert fits_schema.unsigned_column_dtypes_from_header(header) == {}


def test_unsigned_convention_still_needs_tzero_not_just_the_code():
    """The TZERO value is what distinguishes the two, so pin both sides."""
    plain = {"TFIELDS": 1, "TTYPE1": "C", "TFORM1": "1I"}
    assert fits_schema.unsigned_column_dtype_names_from_header(plain) == {}

    u16 = dict(plain, TZERO1=32768.0)
    assert fits_schema.unsigned_column_dtype_names_from_header(u16) == {"C": "uint16"}

    u32 = {"TFIELDS": 1, "TTYPE1": "C", "TFORM1": "1J", "TZERO1": 2147483648.0}
    assert fits_schema.unsigned_column_dtype_names_from_header(u32) == {"C": "uint32"}

    # A J column with the int16 TZERO is still not the unsigned-int32
    # convention -- the mutation would have let it through.
    wrong = {"TFIELDS": 1, "TTYPE1": "C", "TFORM1": "1J", "TZERO1": 32768.0}
    assert fits_schema.unsigned_column_dtype_names_from_header(wrong) == {}


def test_signed_int16_table_column_round_trips_as_signed(tmp_path):
    """End to end: a signed int16 column must not be read back as uint16."""
    import pyarrow as pa

    path = tmp_path / "signed.fits"
    torchfits.table.write(
        str(path),
        data={
            "OFFSET": torch.tensor([1, 2, 3], dtype=torch.int16),
            "BIG": torch.tensor([4, 5, 6], dtype=torch.int32),
        },
        overwrite=True,
    )
    out = torchfits.read(str(path), hdu=1, mode="table")
    # This is the route the unsigned dtype map feeds: the table-tensor
    # destination, where a misclassification would silently change a signed
    # column's dtype.
    assert out["OFFSET"].dtype == torch.int16, out["OFFSET"].dtype
    assert out["BIG"].dtype == torch.int32, out["BIG"].dtype
    assert out["OFFSET"].tolist() == [1, 2, 3]

    # And the Arrow route agrees.
    arrow = torchfits.table.read(str(path), hdu=1)
    assert arrow.schema.field("OFFSET").type == pa.int16(), arrow.schema
    assert arrow.schema.field("BIG").type == pa.int32(), arrow.schema


# --------------------------------------------------------------------------
# TS-013: the VLA predicates. `column_is_vla` had no caller anywhere in the
# repository and no test; `table_has_vla` is reached only through
# `selected_includes_vla(columns=None)`. The two per-column forms are *not*
# interchangeable on a duplicated TTYPE, which is precisely the distinction an
# untested helper hides -- so it is pinned here rather than left implicit.
# --------------------------------------------------------------------------


def _vla_header(tforms) -> dict:
    """Build a bin-table header from (name, tform) *pairs*.

    A list, not a dict, because a duplicated TTYPE has to land on two distinct
    card indices (TTYPE1/TTYPE2): ``_iter_tfields_indexed`` keys by index, so a
    repeated dict key would silently collapse into one column.
    """
    pairs = list(tforms.items()) if isinstance(tforms, dict) else list(tforms)
    header = {"TFIELDS": len(pairs)}
    for i, (name, tform) in enumerate(pairs, start=1):
        header[f"TTYPE{i}"] = name
        header[f"TFORM{i}"] = tform
    return header


def test_table_has_vla_detects_p_and_q_codes():
    """`table_has_vla` is the "any column" form; P and Q are the heap codes."""
    plain = _vla_header({"A": "1J", "B": "20A"})
    assert fits_schema.table_has_vla(plain) is False

    assert fits_schema.table_has_vla(_vla_header({"A": "1J", "B": "1PJ"})) is True
    assert fits_schema.table_has_vla(_vla_header({"A": "1QJ", "B": "1J"})) is True


def test_column_is_vla_answers_for_one_named_column():
    """`column_is_vla` is the per-column form and is otherwise unreferenced."""
    header = _vla_header({"SCALAR": "1J", "ROWS": "1PJ", "NAME": "20A"})

    assert fits_schema.column_is_vla(header, "ROWS") is True
    assert fits_schema.column_is_vla(header, "SCALAR") is False
    assert fits_schema.column_is_vla(header, "NAME") is False
    # An absent column is False, not an error.
    assert fits_schema.column_is_vla(header, "NOPE") is False


def test_column_is_vla_and_selected_includes_vla_differ_on_duplicate_names():
    """The two per-column forms scan differently, and the difference is real.

    ``column_is_vla`` answers about the *first* card carrying the name and
    returns; ``selected_includes_vla`` keeps scanning and answers whether *any*
    card with a wanted name is a VLA. On a duplicated TTYPE -- which
    ``test_bug_table_duplicate_names.py`` shows this library really does see --
    the two disagree. Both are intentional; neither was pinned, so a future
    "simplify one into the other" would have looked free.
    """
    header = _vla_header([("DUP", "1J"), ("DUP", "1PJ")])
    assert fits_schema.column_is_vla(header, "DUP") is False
    assert fits_schema.selected_includes_vla(header, ["DUP"]) is True

    # Reversed order: the per-column form now reports True, and so does the
    # projection form, so the disagreement is genuinely about scanning, not
    # about which card is "the" column.
    reversed_header = _vla_header([("DUP", "1PJ"), ("DUP", "1J")])
    assert fits_schema.column_is_vla(reversed_header, "DUP") is True
    assert fits_schema.selected_includes_vla(reversed_header, ["DUP"]) is True


def test_selected_includes_vla_none_delegates_to_table_has_vla():
    """`columns=None` is the whole-table case, so the two must agree."""
    with_vla = _vla_header({"A": "1J", "B": "1PJ"})
    without = _vla_header({"A": "1J", "B": "1J"})

    assert fits_schema.selected_includes_vla(with_vla, None) is True
    assert fits_schema.selected_includes_vla(without, None) is False
    assert fits_schema.selected_includes_vla(with_vla, None) == (
        fits_schema.table_has_vla(with_vla)
    )
    # An empty selection reads no columns, so it can never include a VLA one.
    assert fits_schema.selected_includes_vla(with_vla, []) is False


@pytest.mark.parametrize("decoy", ["1P", "P", "1Q", "-1PJ", "1P "])
def test_selected_includes_vla_keeps_scanning_past_a_non_vla_pq_column(decoy):
    """A TFORM that *contains* P/Q but does not parse as a VLA must not end the scan.

    ``_tform_might_be_vla`` is the cheap gate (any of ``PpQq`` anywhere in the
    TFORM) and ``parse_tform(...).vla`` is the precise answer. They disagree for
    an incomplete or malformed code -- a bare ``P``, a ``P`` with no element
    type, a negative repeat -- and that gap is the only place the loop's
    *continue scanning* behaviour is observable. Folding the precise check into
    the gate's ``if`` (an equivalent-looking "simplification") would stop at the
    decoy and miss the real VLA column after it; this test is what makes that
    mutation fail rather than survive.
    """
    header = _vla_header([("DECOY", decoy), ("REAL", "1PJ")])

    assert fits_schema._tform_might_be_vla(decoy) is True, "decoy must pass the gate"
    assert fits_schema.parse_tform(decoy).vla is False, "decoy must fail the parse"
    assert fits_schema.selected_includes_vla(header, ["DECOY", "REAL"]) is True
    # Selecting only the decoy still reads as "no VLA", not as an error.
    assert fits_schema.selected_includes_vla(header, ["DECOY"]) is False
    # ... and table_has_vla agrees, since it has the same gate/parse pair.
    assert fits_schema.table_has_vla(header) is True
