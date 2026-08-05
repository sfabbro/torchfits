from torchfits.hdu import Header, TableHDU


def test_tablehdu_cache_invalidation():
    header = Header()
    header["TFIELDS"] = 1
    header["TTYPE1"] = "OLD_NAME"
    header["TFORM1"] = "A10"

    hdu = TableHDU({}, header=header)
    assert hdu.string_columns == ["OLD_NAME"]

    header["TTYPE1"] = "NEW_NAME"
    assert hdu.string_columns == ["NEW_NAME"]


def test_tablehdu_schema_invalidation_on_del():
    header = Header()
    header["TFIELDS"] = 2
    header["TTYPE1"] = "x"
    header["TFORM1"] = "1D"
    header["TTYPE2"] = "y"
    header["TFORM2"] = "1D"

    hdu = TableHDU({}, header=header)
    assert [c["name"] for c in hdu.schema["columns"]] == ["x", "y"]

    del header["TTYPE2"]
    assert [c["name"] for c in hdu.schema["columns"]] == ["x"]
