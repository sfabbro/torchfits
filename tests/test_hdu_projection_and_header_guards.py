"""Guards for the unit-7 findings: HDU projections, row windows, header cards.

R2-041 TableHDURef.num_rows re-derived the row window with weaker rules than
        the shared ``_normalize_row_slice``, so ``len(ref)`` reported a row
        count for windows ``read()``/``iter_rows()`` refuse.
R2-042 TableHDU.iter_rows(range(0, n, -1)) silently yielded no rows, while its
        own package peer TableHDURef.iter_rows refuses the same argument.
R2-043 TableHDURef(columns=[]) / select([]) read *every* column, because the
        projection could not tell "no columns" from "unset".
R2-044 Header.update(other_header) copied the mapping view, dropping every card
        comment and collapsing a HISTORY/COMMENT block to its last line.
R2-045 TableHDU.num_rows was a functools.cached_property fed by the mutable
        header, so a columnless table's row count froze at first read.
"""

import subprocess
import sys

import pytest
import torch

from torchfits.hdu import Header, TableHDU, TableHDURef

NROWS = 10


def _write_table(path):
    afits = pytest.importorskip("astropy.io.fits")
    import numpy as np

    cols = afits.ColDefs(
        [
            afits.Column(name="x", format="1D", array=np.arange(float(NROWS))),
            afits.Column(name="y", format="1E", array=np.zeros(NROWS, dtype="f4")),
        ]
    )
    afits.HDUList(
        [afits.PrimaryHDU(), afits.BinTableHDU.from_columns(cols, nrows=NROWS)]
    ).writeto(str(path), overwrite=True)
    return str(path)


def _build_ref(tmp_path, **kwargs):
    import torchfits

    path = _write_table(tmp_path / "t.fits")
    header = torchfits.read_header(path, 1)
    return TableHDURef(header=header, source_path=path, source_hdu=1, **kwargs)


def _open_ref(tmp_path, **kwargs):
    """A ref on a real file, metadata only (no HDUList handle held open)."""
    return _build_ref(tmp_path, **kwargs)


# --------------------------------------------------------------------------
# R2-041: num_rows and read() must agree about which row windows are legal
# --------------------------------------------------------------------------

_ILLEGAL_WINDOWS = [
    pytest.param(slice(-2, None), id="negative-start"),
    pytest.param(slice(0, -1), id="negative-stop"),
    pytest.param(slice(0, 3, 2), id="step-two"),
    pytest.param((0, 2, 9), id="three-tuple"),
    pytest.param((3,), id="one-tuple"),
]


@pytest.mark.parametrize("window", _ILLEGAL_WINDOWS)
def test_num_rows_refuses_every_window_read_refuses(tmp_path, window):
    ref = _open_ref(tmp_path, row_slice=window)

    with pytest.raises(ValueError) as counted:
        _ = ref.num_rows
    with pytest.raises(ValueError) as read_it:
        ref.read()

    # One definition of the window: same exception, same message.
    assert type(counted.value) is type(read_it.value)
    assert str(counted.value) == str(read_it.value)
    assert "row_slice" in str(counted.value)


@pytest.mark.parametrize(
    "message",
    [
        "row_slice start must be >= 0",
        "row_slice negative stop is not supported",
        "row_slice step must be 1 for FITS row streaming",
        "row_slice tuple must be \\(start, stop\\)",
    ],
)
def test_num_rows_names_the_rule_read_names(tmp_path, message):
    """Each illegal window shape is refused by its own documented rule."""
    window = {
        "row_slice start must be >= 0": slice(-2, None),
        "row_slice negative stop is not supported": slice(0, -1),
        "row_slice step must be 1 for FITS row streaming": slice(0, 3, 2),
        "row_slice tuple must be \\(start, stop\\)": (0, 2, 9),
    }[message]
    ref = _open_ref(tmp_path, row_slice=window)
    with pytest.raises(ValueError, match=message):
        _ = ref.num_rows


_LEGAL_WINDOWS = [
    pytest.param(None, NROWS, 0.0, id="whole-table"),
    pytest.param(slice(2, None), 8, 2.0, id="offset-open-stop"),
    pytest.param(slice(0, 3), 3, 0.0, id="closed-window"),
    pytest.param(slice(2, 99), 8, 2.0, id="stop-past-the-end"),
    pytest.param(slice(4, 2), 0, None, id="empty-window"),
    pytest.param((2, 4), 2, 2.0, id="tuple-window"),
    pytest.param(slice(9, 10), 1, 9.0, id="last-row"),
]


@pytest.mark.parametrize("window,expected,first_value", _LEGAL_WINDOWS)
def test_a_legal_window_still_counts_and_still_reads(
    tmp_path, window, expected, first_value
):
    """The smallest legal cases keep working: count and data agree."""
    ref = _open_ref(tmp_path, row_slice=window)

    assert ref.num_rows == expected
    assert len(ref) == expected

    if expected:
        data = ref.read()
        assert len(data["x"]) == expected
        assert len(data["y"]) == expected
        assert data["x"][0].item() == first_value
        assert data["x"][-1].item() == first_value + expected - 1


def test_a_single_row_window_still_reads_that_row(tmp_path):
    ref = _open_ref(tmp_path, row_slice=slice(3, 4))
    assert ref.num_rows == 1
    data = ref.read()
    assert data["x"].tolist() == [3.0]


def test_the_unwindowed_ref_reports_the_whole_table(tmp_path):
    ref = _open_ref(tmp_path)
    assert ref.num_rows == NROWS
    assert len(ref) == NROWS
    chunks = list(ref.iter_rows(batch_size=4))
    assert len(chunks) == 3


def test_the_metadata_only_path_still_avoids_the_table_io_package(tmp_path):
    """``len(ref)`` on an un-windowed ref must not pull in _table.utils.

    ``num_rows`` shares ``_normalize_row_slice`` with read()/iter_rows(), but
    that helper lives behind an import the metadata path never paid for
    (~15 ms measured). A ref with no row window must stay free of it.
    """
    path = _write_table(tmp_path / "meta.fits")
    child = (
        "import sys, torchfits\n"
        f"hdul = torchfits.open({path!r})\n"
        "assert len(hdul[1]) == 10, len(hdul[1])\n"
        "hdul.close()\n"
        "print('UTILS' if 'torchfits._table.utils' in sys.modules else 'CLEAN')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "CLEAN" in out.stdout, out.stdout + out.stderr[-2000:]


def test_a_windowed_ref_may_import_the_shared_helper(tmp_path):
    """Positive counterpart: the windowed path is the one that needs it."""
    ref = _open_ref(tmp_path, row_slice=slice(0, 4))
    assert ref.num_rows == 4
    assert ref.head(2).num_rows == 2


# --------------------------------------------------------------------------
# R2-042: iter_rows refuses a non-positive batch size, like its lazy peer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [0, -1, -5, -100])
def test_iter_rows_refuses_a_non_positive_batch_size(batch_size):
    table = TableHDU({"x": torch.arange(4.0)})
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        list(table.iter_rows(batch_size=batch_size))


def test_both_iter_rows_implementations_refuse_the_same_batch_sizes(tmp_path):
    ref = _open_ref(tmp_path)
    table = ref.materialize()
    for batch_size in (0, -1):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            list(ref.iter_rows(batch_size=batch_size))
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            list(table.iter_rows(batch_size=batch_size))


@pytest.mark.parametrize(
    "batch_size,expected_chunks,expected_rows",
    [
        pytest.param(1, 3, 3, id="one-row-batches"),
        pytest.param(2, 2, 3, id="two-row-batches"),
        pytest.param(3, 1, 3, id="exactly-the-table"),
        pytest.param(99, 1, 3, id="wider-than-the-table"),
    ],
)
def test_a_positive_batch_size_still_streams_every_row(
    batch_size, expected_chunks, expected_rows
):
    """The smallest legal batch size still yields every row, in order."""
    table = TableHDU({"x": torch.arange(3.0)})
    chunks = list(table.iter_rows(batch_size=batch_size))
    assert len(chunks) == expected_chunks
    assert [row for chunk in chunks for row in chunk["x"].tolist()] == [0.0, 1.0, 2.0]
    assert sum(len(chunk["x"]) for chunk in chunks) == expected_rows


def test_a_single_row_table_still_streams(tmp_path):
    ref = _open_ref(tmp_path, row_slice=slice(0, 1))
    chunks = list(ref.iter_rows(batch_size=4))
    assert len(chunks) == 1
    assert chunks[0]["x"].tolist() == [0.0]


# --------------------------------------------------------------------------
# R2-043: an empty projection is refused, not widened to every column
# --------------------------------------------------------------------------


def test_select_refuses_an_empty_projection(tmp_path):
    ref = _open_ref(tmp_path)
    assert ref.columns == ["x", "y"]
    with pytest.raises(ValueError, match="select\\(\\) requires at least one column"):
        ref.select([])


def test_constructing_a_ref_with_no_columns_is_refused(tmp_path):
    with pytest.raises(ValueError, match="columns must name at least one column"):
        _open_ref(tmp_path, columns=[])


def test_select_with_a_single_column_still_projects(tmp_path):
    """Smallest legal projection: one named column, nothing else."""
    ref = _open_ref(tmp_path)
    one = ref.select(["x"])
    assert one.columns == ["x"]
    assert one.num_rows == NROWS
    data = one.read()
    assert sorted(data) == ["x"]
    assert len(data["x"]) == NROWS


def test_select_with_every_column_still_projects(tmp_path):
    ref = _open_ref(tmp_path)
    both = ref.select(["x", "y"])
    assert both.columns == ["x", "y"]
    assert sorted(both.read()) == ["x", "y"]


def test_select_does_not_mutate_the_source_projection(tmp_path):
    ref = _open_ref(tmp_path)
    ref.select(["x"])
    assert ref.columns == ["x", "y"]
    assert sorted(ref.read()) == ["x", "y"]


# --------------------------------------------------------------------------
# R2-044: Header.update copies cards, not the mapping view
# --------------------------------------------------------------------------


def _rich_header():
    src = Header({"A": 1, "B": "two"})
    src.add_history("first history")
    src.add_history("second history")
    src.add_comment("a comment")
    src.append(("C", 3, "the c comment"))
    src["D"] = (4, "tuple comment")
    return src


def test_update_from_a_header_keeps_every_card():
    src = _rich_header()
    dst = Header()
    dst.update(src)
    assert dst.cards == src.cards


def test_update_from_a_header_keeps_every_history_and_comment_line():
    src = _rich_header()
    dst = Header()
    dst.update(src)
    assert dst.get_history() == ["first history", "second history"]
    assert dst.get_comment() == ["a comment"]
    assert dst.card("C").comment == "the c comment"
    assert dst.card("D").comment == "tuple comment"


def test_update_from_a_header_matches_the_header_constructor():
    """update() must be as lossless as Header(source)."""
    src = _rich_header()
    via_update = Header()
    assert via_update.update(src) is None
    assert via_update.cards == Header(src).cards
    assert dict(via_update) == dict(Header(src))


def test_update_from_a_header_keeps_the_first_occurrence_of_a_value_key():
    src = Header([("K", 1, "first"), ("K", 2, "second")])
    dst = Header()
    dst.update(src)
    assert dst["K"] == dict(src)["K"] == 1
    assert dst.card("K").comment == "first"


def test_update_from_a_plain_mapping_still_sets_values_and_comments():
    dst = Header()
    dst.update({"A": 1, "B": (2, "b comment")})
    dst.update(C=3)
    assert dict(dst) == {"A": 1, "B": 2, "C": 3}
    assert dst.card("B").comment == "b comment"


def test_update_from_a_card_sequence_still_sets_values():
    dst = Header()
    dst.update([("A", 1), ("B", 2)])
    assert dict(dst) == {"A": 1, "B": 2}


def test_update_onto_an_existing_header_still_replaces_values():
    dst = Header({"A": 1})
    dst["A"] = 9
    dst.update({"A": 5})
    assert dst["A"] == 5
    assert len(dst.cards) == 1


def test_update_rejects_a_second_positional_argument():
    with pytest.raises(TypeError, match="at most 1 positional argument"):
        Header().update({"A": 1}, {"B": 2})


def test_update_with_nothing_to_do_leaves_the_version_alone():
    dst = Header({"A": 1})
    before = dst._version
    dst.update({})
    assert dst._version == before
    dst.update(Header())
    assert dst._version == before


# --------------------------------------------------------------------------
# R2-045: num_rows follows the header it is derived from
# --------------------------------------------------------------------------


def test_a_header_edit_moves_the_row_count_of_a_columnless_table():
    header = Header({"TFIELDS": 0, "NAXIS2": 10})
    table = TableHDU({}, None, header)
    assert table.num_rows == 10

    header["NAXIS2"] = 4
    assert table.num_rows == 4
    assert len(table.data) == 4

    header["NAXIS2"] = 7
    assert table.num_rows == 7
    assert "rows=7" in repr(table)


def test_the_header_constructor_copy_also_tracks_the_row_count():
    src = Header({"TFIELDS": 0, "NAXIS2": 5})
    table = TableHDU({}, None, src)
    assert table.num_rows == 5
    src["NAXIS2"] = 2
    assert table.num_rows == 2


def test_a_table_with_columns_ignores_naxis2():
    """The row count of a materialized table comes from its column data."""
    table = TableHDU({"x": torch.arange(10.0)}, None, Header({"NAXIS2": 10}))
    assert table.num_rows == 10
    table.header["NAXIS2"] = 2
    assert table.num_rows == 10


def test_a_table_without_columns_or_header_reports_no_rows():
    assert TableHDU({}).num_rows == 0
    assert TableHDU({}, None, Header({})).num_rows == 0


def test_repeated_row_counts_agree_with_head():
    header = Header({"TFIELDS": 0, "NAXIS2": 10})
    table = TableHDU({}, None, header)
    assert table.num_rows == table.head(10).num_rows
    header["NAXIS2"] = 3
    assert table.num_rows == 3
    assert table.head(3).num_rows == 3
