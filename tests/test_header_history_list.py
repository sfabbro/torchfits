"""HISTORY/COMMENT cards accept multi-line payloads from every entry point.

A list used to reach the C++ writer as an unchecked ``nb::cast<std::string>``
and surface as ``std::bad_cast``.  All the ways to set a header card must agree
that a list/tuple means "one card per line", and a non-str line must be
coerced rather than crash the write.
"""

import pytest

import torchfits
from torchfits import Card, Header


def _write_and_read(tmp_path, header, name):
    path = str(tmp_path / f"{name}.fits")
    torchfits.table.write(path, {"A": [1]}, header=header)
    hdul = torchfits.HDUList.fromfile(path)
    try:
        return hdul[1].header.get_history(), hdul[1].header.get_comment()
    finally:
        hdul.close()


def _setitem():
    header = Header()
    header["HISTORY"] = ["line 1", "line 2"]
    header["COMMENT"] = ["comment 1", "comment 2"]
    return header


def _ctor():
    return Header(
        {"HISTORY": ["line 1", "line 2"], "COMMENT": ["comment 1", "comment 2"]}
    )


def _update():
    header = Header()
    header.update(
        {"HISTORY": ["line 1", "line 2"], "COMMENT": ["comment 1", "comment 2"]}
    )
    return header


def _append():
    header = Header()
    header.append(("HISTORY", ["line 1", "line 2"]))
    header.append(Card("COMMENT", ["comment 1", "comment 2"]))
    return header


def _insert():
    header = Header()
    header.insert(0, ("HISTORY", ["line 1", "line 2"]))
    header.insert(0, ("COMMENT", ["comment 1", "comment 2"]))
    return header


@pytest.mark.parametrize(
    "build", [_setitem, _ctor, _update, _append, _insert], ids=lambda f: f.__name__
)
def test_list_payload_writes_one_card_per_line(tmp_path, build):
    history, comment = _write_and_read(tmp_path, build(), build.__name__)
    assert history == ["line 1", "line 2"]
    assert comment == ["comment 1", "comment 2"]


def test_tuple_payload_unfolds_via_card(tmp_path):
    # A 2-tuple through the mapping means (value, comment), so the tuple case is
    # exercised where it is unambiguous: a Card handed to append().
    header = Header()
    header.append(Card("HISTORY", ("line 1", "line 2")))
    history, _ = _write_and_read(tmp_path, header, "tuple")
    assert history == ["line 1", "line 2"]


def test_two_tuple_assignment_writes_one_card(tmp_path):
    # Mapping assignment reads a 2-tuple as (value, comment); only a list means
    # "several cards".  A HISTORY card has no comment slot, so the value is all
    # that reaches the file - but it must not be split into two lines.
    header = Header()
    header["HISTORY"] = ("line 1", "the comment")
    history, _ = _write_and_read(tmp_path, header, "tuple-comment")
    assert history == ["line 1"]


def test_non_str_lines_are_coerced_not_crashing(tmp_path):
    # A numeric HISTORY value is text, not a header value: the writer's
    # TLOGICAL/TLONGLONG branches must not see it.
    header = Header()
    header["HISTORY"] = [1, 2.5]
    history, _ = _write_and_read(tmp_path, header, "numeric")
    assert history == ["1", "2.5"]


def test_empty_list_writes_no_cards(tmp_path):
    header = Header()
    header["HISTORY"] = []
    assert _write_and_read(tmp_path, header, "empty")[0] == []


def test_low_level_writer_reports_type_instead_of_bad_cast(tmp_path):
    """The raw binding must name the problem; std::bad_cast told the caller nothing."""
    import torchfits._C as cpp

    path = str(tmp_path / "guard.fits")
    torchfits.table.write(path, {"A": [1]})
    with pytest.raises(RuntimeError, match="HISTORY text must be a str, got list"):
        cpp.write_hdu_header_cards(path, 1, [("HISTORY", ["line 1"])])
