from torchfits.hdu import Header
from torchfits.header_parser import fast_parse_header_cards


def test_header_versioning():
    h = Header()
    assert h._version == 0

    h["a"] = 1
    assert h._version == 1

    h.update({"b": 2})
    assert h._version == 2

    h.setdefault("c", 3)
    assert h._version == 3

    # setdefault existing
    h.setdefault("a", 10)
    assert h._version == 4  # We decided to increment anyway

    val = h.pop("a")
    assert val == 1
    assert h._version == 5

    del h["b"]
    assert h._version == 6

    h.clear()
    assert h._version == 7


def test_fast_parse_header_cards_empty_comment_is_str():
    """Cards without '/' must get comment '' (not None → Header 'None')."""
    cards = [
        "SIMPLE  =                    T / file does conform to FITS standard             ",
        "BITPIX  =                   16                                                  ",
        "END                                                                             ",
    ]
    header_string = "".join(c.ljust(80) for c in cards)
    parsed = fast_parse_header_cards(header_string)
    by_key = {k: (v, c) for k, v, c in parsed}
    assert by_key["SIMPLE"][1] != ""
    assert by_key["BITPIX"][1] == ""
    h = Header(parsed)
    bitpix_cards = [c for c in h.cards if c.key == "BITPIX"]
    assert bitpix_cards and bitpix_cards[0].comment == ""


def test_header_remove_all_history_preserves_other_cards():
    h = Header()
    h["SIMPLE"] = True
    h.add_history("h1")
    h.add_history("h2")
    h.add_history("h3")
    h["BITPIX"] = 16
    v = h._version
    h.remove("HISTORY", remove_all=True)
    assert h._version == v + 1
    assert [c.key for c in h.cards] == ["SIMPLE", "BITPIX"]
    assert "HISTORY" not in h
    assert h["SIMPLE"] is True
    assert h["BITPIX"] == 16


def test_header_remove_first_history_only():
    h = Header()
    h.add_history("h1")
    h.add_history("h2")
    h.add_history("h3")
    v = h._version
    h.remove("HISTORY", remove_all=False)
    assert h._version == v + 1
    assert [c.value for c in h.cards if c.key == "HISTORY"] == ["h2", "h3"]


def test_header_remove_all_many_history_is_linear():
    """Smoke: many HISTORY cards must not take quadratic delete time."""
    n = 20_000
    h = Header()
    for i in range(n):
        h.add_history(f"h{i}")
    h["OBJECT"] = "keep"
    h.remove("HISTORY", remove_all=True)
    assert [c.key for c in h.cards] == ["OBJECT"]
    assert h["OBJECT"] == "keep"
