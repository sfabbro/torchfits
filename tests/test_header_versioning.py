from torchfits.hdu import Header


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
