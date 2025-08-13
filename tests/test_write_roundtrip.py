import pytest
import torch

import torchfits as tf


def test_write_image_roundtrip(tmp_path):
    data = torch.randint(0, 1000, (32, 32), dtype=torch.int16)
    header = {"OBJECT": "UnitTest", "EXPTIME": "123.4"}
    out = tmp_path / "image_rt.fits"
    tf.write(str(out), data, header)
    rt_data, rt_header = tf.read(str(out), hdu=0)
    assert torch.equal(data.to(rt_data.dtype), rt_data)
    assert rt_header.get("OBJECT") == "UnitTest"
    assert float(rt_header.get("EXPTIME")) == 123.4


def test_write_table_roundtrip(tmp_path):
    table = {"RA": torch.linspace(0, 1, 10), "DEC": torch.linspace(-1, 0, 10)}
    out = tmp_path / "table_rt.fits"
    tf.write(str(out), table, {"EXTNAME": "CAT"})
    rt_table = tf.read(str(out), hdu=1, format="table")
    assert set(rt_table.data.keys()) == {"RA", "DEC"}
    assert torch.allclose(rt_table.data["RA"], table["RA"])
    assert torch.allclose(rt_table.data["DEC"], table["DEC"])


def test_append_hdu(tmp_path):
    base = tmp_path / "mef.fits"
    primary = torch.zeros(8, 8)
    tf.write(str(base), primary, {"EXTNAME": "PRIMARY"})
    extra = torch.ones(4, 4)
    tf.append_hdu(str(base), extra, {"EXTNAME": "SCI"})
    f = tf.FITS(str(base))
    assert len(f) >= 2
    d0, _ = tf.read(str(base), hdu=0)
    d1, _ = tf.read(str(base), hdu=1)
    assert d0.shape == (8, 8)
    assert d1.shape == (4, 4)


def test_write_append_flag_image(tmp_path):
    base = tmp_path / "append_flag.fits"
    primary = torch.zeros(5, 5)
    tf.write(str(base), primary, {"EXTNAME": "PRIMARY"})
    extra = torch.full((3, 3), 2.0)
    # Use unified write with append flag
    tf.write(str(base), extra, {"EXTNAME": "EXTRA"}, append=True)
    f = tf.FITS(str(base))
    assert len(f) == 2
    d0, _ = tf.read(str(base), hdu=0)
    d1, h1 = tf.read(str(base), hdu=1)
    assert torch.allclose(d0, primary)
    assert torch.allclose(d1, extra)
    assert h1.get("EXTNAME") == "EXTRA"


def test_write_mef_roundtrip(tmp_path):
    out = tmp_path / "multi.fits"
    tensors = [
        torch.arange(16, dtype=torch.int16).view(4, 4),
        torch.full((2, 2), 7, dtype=torch.int16),
        torch.zeros(3, 3, dtype=torch.int16),
    ]
    headers = [
        {"OBSTYPE": "PRIMARY"},
        {"OBSTYPE": "SCI"},
        {"OBSTYPE": "ERR"},
    ]
    extnames = ["PRIMARY", "SCI", "ERR"]
    tf.write_mef(str(out), tensors, headers=headers, extnames=extnames)

    f = tf.FITS(str(out))
    assert len(f) == 3
    for i, t in enumerate(tensors):
        d, h = tf.read(str(out), hdu=i)
        assert torch.equal(d.to(t.dtype), t)
        # Header OBSTYPE and EXTNAME should be present
        assert h.get("OBSTYPE") == headers[i]["OBSTYPE"]
        # EXTNAME might only appear for extensions, primary may or may not carry it
        if i == 0:
            # accept either PRIMARY or absence
            if "EXTNAME" in h:
                assert h["EXTNAME"] == "PRIMARY"
        else:
            assert h.get("EXTNAME") == extnames[i]


def test_update_header(tmp_path):
    out = tmp_path / "upd_header.fits"
    base = torch.zeros(6, 6, dtype=torch.float32)
    tf.write(str(out), base, {"OBJECT": "Init"})
    tf.update_header(str(out), {"OBJECT": "Updated", "EXPTIME": "42"}, hdu=1)
    _, hdr = tf.read(str(out), hdu=0)
    assert hdr.get("OBJECT") == "Updated"
    assert float(hdr.get("EXPTIME")) == 42.0


def test_update_header_object_method(tmp_path):
    out = tmp_path / "upd_header_obj.fits"
    base = torch.zeros(4, 4, dtype=torch.float32)
    tf.write(str(out), base, {"OBJECT": "Init"})
    with tf.FITS(str(out)) as f:
        hdu0 = f[0]
        assert hdu0.header.get("OBJECT") == "Init"
        hdu0.update_header({"OBJECT": "Changed", "OBSERVER": "UnitTest"})
        # Force refresh
        hdr2 = hdu0.refresh_header()
        assert hdr2.get("OBJECT") == "Changed"
        assert hdr2.get("OBSERVER") == "UnitTest"


def test_update_data_full_and_partial(tmp_path):
    out = tmp_path / "upd_data.fits"
    base = torch.zeros(10, 10, dtype=torch.float32)
    tf.write(str(out), base, {"OBJECT": "Init"})
    new_full = torch.ones(10, 10, dtype=torch.float32)
    tf.update_data(str(out), new_full, hdu=1)
    full_after, _ = tf.read(str(out), hdu=0)
    # If this unexpectedly fails, inspect raw file via astropy to distinguish read vs write issue.
    if not torch.allclose(full_after, new_full):
        try:
            from astropy.io import fits

            with fits.open(str(out)) as hdul:
                import numpy as np

                assert np.allclose(hdul[0].data, new_full.numpy())
        except ImportError:
            pass
        # Fallback assert (will fail giving diagnostic):
        assert torch.allclose(full_after, new_full)
    patch = torch.full((4, 4), 5.0, dtype=torch.float32)
    tf.update_data(str(out), patch, hdu=1, start=[3, 3], shape=[4, 4])
    full_after2, _ = tf.read(str(out), hdu=0)
    if not torch.allclose(full_after2[3:7, 3:7], patch):
        try:
            from astropy.io import fits

            with fits.open(str(out)) as hdul:
                import numpy as np

                assert np.allclose(hdul[0].data[3:7, 3:7], patch.numpy())
        except ImportError:
            pass
        assert torch.allclose(full_after2[3:7, 3:7], patch)
    # Ensure outside untouched (should remain ones from full update)
    assert full_after2[2, 2] == 1.0


def test_update_data_partial_subset(tmp_path):
    out = tmp_path / "upd_partial.fits"
    base = torch.zeros(10, 10, dtype=torch.float32)
    tf.write(str(out), base, {"OBJECT": "Init"})
    patch = torch.full((4, 4), 5.0, dtype=torch.float32)
    tf.update_data(str(out), patch, hdu=1, start=[3, 3], shape=[4, 4])
    full, _ = tf.read(str(out), hdu=0)
    assert torch.allclose(full[3:7, 3:7], patch)
    assert full[2, 2] == 0.0


def test_write_variable_length_array_wrapper(tmp_path):
    arrays = [torch.arange(i + 3, dtype=torch.float32) for i in range(5)]
    out = tmp_path / "var_arr_wrap.fits"
    tf.write_variable_length_array(str(out), arrays, {"OBJECT": "VAR"}, overwrite=True)
    # Validate via astropy if available
    try:
        import numpy as np
        from astropy.io import fits
    except ImportError:
        pytest.skip("astropy required for validation")
    with fits.open(str(out)) as hdul:
        assert len(hdul) == 2
        tbl = hdul[1].data
        lengths = [len(x) for x in tbl["ARRAY_DATA"]]
        assert lengths == [a.numel() for a in arrays]
        for row, arr in zip(tbl["ARRAY_DATA"], arrays):
            np.testing.assert_allclose(row, arr.numpy().astype(row.dtype))


def test_update_header_by_extname(tmp_path):
    out = tmp_path / "name_update.fits"
    tensors = [torch.zeros(4, 4), torch.ones(2, 2)]
    headers = [{"EXTNAME": "PRIMARY"}, {"EXTNAME": "SCI", "OBJECT": "Init"}]
    tf.write_mef(str(out), tensors, headers=headers, extnames=["PRIMARY", "SCI"])
    tf.update_header(str(out), {"OBJECT": "Updated"}, hdu="SCI")
    d, h = tf.read(str(out), hdu=1)
    assert h.get("OBJECT") == "Updated"


def test_update_data_by_extname(tmp_path):
    out = tmp_path / "name_update_data.fits"
    tensors = [torch.zeros(4, 4), torch.zeros(4, 4)]
    tf.write_mef(
        str(out),
        tensors,
        headers=[{"EXTNAME": "PRIMARY"}, {"EXTNAME": "SCI"}],
        extnames=["PRIMARY", "SCI"],
    )
    patch = torch.full((2, 2), 9.0)
    tf.update_data(str(out), patch, hdu="SCI", start=[1, 1], shape=[2, 2])
    d, h = tf.read(str(out), hdu=1)
    assert torch.allclose(d[1:3, 1:3], patch)
    assert d[0, 0] == 0.0


def test_append_hdu_image_roundtrip(tmp_path):
    # Create initial file with one image
    base = tmp_path / "append_base.fits"
    img1 = torch.ones(5, 5)
    tf.write_image(str(base), img1, header={"EXTNAME": "IMG1"}, overwrite=True)
    # Append second image
    img2 = torch.full((3, 3), 7.0)
    tf.append_hdu(str(base), img2, header={"EXTNAME": "IMG2"})
    # Read back both
    d0, h0 = tf.read(str(base), hdu=0)
    d1, h1 = tf.read(str(base), hdu=1)
    assert torch.allclose(d0, img1)
    assert torch.allclose(d1, img2)
    assert h1.get("EXTNAME") == "IMG2"


def test_write_cube_roundtrip(tmp_path):
    # 3D cube write helper should set CTYPE3 by default and preserve shape
    cube = torch.arange(2 * 4 * 6, dtype=torch.int16).view(2, 4, 6)
    out = tmp_path / "cube_rt.fits"
    tf.write_cube(str(out), cube, header={"OBJECT": "CubeRT"}, overwrite=True)
    data, header = tf.read(str(out), hdu=0)
    assert data.shape == cube.shape
    # dtype may differ depending on storage; compare after cast
    assert torch.equal(cube.to(data.dtype), data)
    # CTYPE3 default should be present if not provided
    assert header.get("CTYPE3") == "WAVE"


def test_write_table_with_strings_and_nulls(tmp_path):
    out = tmp_path / "table_strings.fits"
    data = {
        "ID": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        "NAME": ["Alpha", "Beta", "Gamma", ""],
        "VAL": torch.tensor([10, -9999, 30, -9999], dtype=torch.int32),
    }
    null_sentinels = {"VAL": -9999}
    tf.write_table(
        str(out),
        data,
        header={"EXTNAME": "CAT"},
        null_sentinels=null_sentinels,
        overwrite=True,
    )
    # Read back using low-level header to verify TNULL is present via astropy if available
    try:
        from astropy.io import fits
    except ImportError:
        pytest.skip("astropy required for validation")
    with fits.open(str(out)) as hdul:
        hdr = hdul[1].header
        assert (
            hdr.get("TNULL3") == -9999 or hdr.get("TNULL2") == -9999
        )  # position depends on column ordering excluding string
        # Validate string column round-trip length
        name_col = hdul[1].data["NAME"]
        assert name_col[0].strip() == "Alpha"
        assert name_col[3].strip() == ""


def test_read_table_null_masks(tmp_path):
    out = tmp_path / "table_strings.fits"
    data = {
        "ID": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        "NAME": ["Alpha", "Beta", "Gamma", ""],
        "VAL": torch.tensor([10, -9999, 30, -9999], dtype=torch.int32),
    }
    null_sentinels = {"VAL": -9999}
    tf.write_table(
        str(out),
        data,
        header={"EXTNAME": "CAT"},
        null_sentinels=null_sentinels,
        overwrite=True,
    )
    from torchfits import read_table_with_null_masks

    table_dict, header, masks = read_table_with_null_masks(str(out), hdu=1)
    assert "VAL" in masks, "Null mask for VAL column not returned"
    mask = masks["VAL"]
    assert mask.dtype == torch.bool
    # Mask should be True where sentinel present
    expected = torch.tensor([False, True, False, True])
    assert torch.equal(mask, expected)
    # Underlying data still has sentinel values (preserve policy)
    assert table_dict["VAL"][1].item() == -9999
    assert table_dict["VAL"][3].item() == -9999


def test_read_table_multi_null_masks(tmp_path):
    # Two columns with different sentinels, one integer one int64 to ensure dtype handling
    out = tmp_path / "table_multi_nulls.fits"
    data = {
        "ID": torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
        "A": torch.tensor([11, -1, 13, -1, 15], dtype=torch.int32),
        "B": torch.tensor([100, 200, -999999999, 400, -999999999], dtype=torch.int64),
    }
    null_sentinels = {"A": -1, "B": -999999999}
    tf.write_table(
        str(out),
        data,
        header={"EXTNAME": "CAT"},
        null_sentinels=null_sentinels,
        overwrite=True,
    )
    from torchfits import get_header, read_table_with_null_masks

    table_dict, header, masks = read_table_with_null_masks(str(out), hdu=1)
    # Verify both masks present
    assert set(masks.keys()) == {"A", "B"}
    # Shapes match data
    assert masks["A"].shape[0] == data["A"].shape[0]
    assert masks["B"].shape[0] == data["B"].shape[0]
    # Check correctness
    expected_a = torch.tensor([False, True, False, True, False])
    expected_b = torch.tensor([False, False, True, False, True])
    assert torch.equal(masks["A"], expected_a)
    assert torch.equal(masks["B"], expected_b)
    # Header must contain both TNULL entries (positions depend on ordering)
    hdr = get_header(str(out), 1)
    # Header dictionary may have TNULLn inserted before TTYPE ordering; just check pairs exist
    found_a = any(
        k.startswith("TNULL") and v == -1
        for k, v in ((k, int(v)) for k, v in hdr.items() if k.startswith("TNULL"))
    )
    found_b = any(
        k.startswith("TNULL") and int(v) == -999999999 for k, v in hdr.items()
    )
    assert found_a and found_b, f"Missing TNULL entries in header: {hdr}"


def test_table_cutout_return_null_masks(tmp_path):
    out = tmp_path / "table_cutout_masks.fits"
    data = {
        "ID": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        "VAL": torch.tensor([-1, 2, -1, 4], dtype=torch.int32),
    }
    tf.write_table(
        str(out),
        data,
        header={"EXTNAME": "CAT"},
        null_sentinels={"VAL": -1},
        overwrite=True,
    )
    from torchfits.dataset import read_table_cutout

    (row, masks) = read_table_cutout(
        str(out), hdu=1, row_start=0, row_count=4, return_null_masks=True
    )
    assert set(row.keys()) == {"ID", "VAL"}
    assert set(masks.keys()) == {"VAL"}
    assert masks["VAL"].dtype == torch.bool
    assert torch.equal(masks["VAL"], torch.tensor([True, False, True, False]))


def test_multi_table_cutouts_stack_with_masks(tmp_path):
    out1 = tmp_path / "tab1.fits"
    out2 = tmp_path / "tab2.fits"
    data1 = {"X": torch.tensor([1, -1, 3], dtype=torch.int32)}
    data2 = {"X": torch.tensor([-1, 5, 6], dtype=torch.int32)}
    tf.write_table(
        str(out1),
        data1,
        header={"EXTNAME": "T"},
        null_sentinels={"X": -1},
        overwrite=True,
    )
    tf.write_table(
        str(out2),
        data2,
        header={"EXTNAME": "T"},
        null_sentinels={"X": -1},
        overwrite=True,
    )
    from torchfits.dataset import TableCutoutSpec, read_multi_table_cutouts

    specs = [
        TableCutoutSpec(
            path=str(out1), hdu=1, row_start=0, row_count=3, return_null_masks=True
        ),
        TableCutoutSpec(
            path=str(out2), hdu=1, row_start=0, row_count=3, return_null_masks=True
        ),
    ]
    stacked_data, stacked_masks = read_multi_table_cutouts(
        specs, parallel=False, stack=True
    )
    assert torch.equal(stacked_data["X"], torch.stack([data1["X"], data2["X"]]))
    exp_masks = torch.stack(
        [torch.tensor([False, True, False]), torch.tensor([True, False, False])]
    )
    assert torch.equal(stacked_masks["X"], exp_masks)


def test_apply_null_masks_helpers_dict_and_table():
    import torchfits as tf

    data = {
        "A": torch.tensor([1, -1, 3, -1], dtype=torch.int32),
        "B": torch.tensor([10.0, 20.0, float("nan"), 40.0], dtype=torch.float32),
    }
    masks = {"A": data["A"].eq(-1)}
    # Dict helper: default NaN fill -> A becomes float with NaNs at masked
    out = tf.apply_null_masks_to_dict(data, masks)
    assert out["A"].dtype.is_floating_point
    assert torch.isnan(out["A"][1]) and torch.isnan(out["A"][3])
    # Custom fill value
    out2 = tf.apply_null_masks_to_dict(data, masks, fill_value={"A": 0.0})
    assert out2["A"].dtype.is_floating_point
    assert out2["A"][1].item() == 0.0 and out2["A"][3].item() == 0.0
    # FitsTable helper: use ColumnInfo null_value
    from torchfits import ColumnInfo, FitsTable

    meta = {"A": ColumnInfo("A", dtype=torch.int32, null_value=-1)}
    ft = FitsTable(data, metadata=meta)
    masks2 = ft.null_masks
    assert set(masks2.keys()) == {"A"}
    ft_clean = ft.with_applied_null_masks()
    assert ft_clean.data["A"].dtype.is_floating_point
    assert torch.isnan(ft_clean.data["A"][1]) and torch.isnan(ft_clean.data["A"][3])


def test_table_torch_frame_round_trip(tmp_path):
    # Create simple table
    data = {
        "X": torch.arange(10, dtype=torch.int32),
        "Y": torch.linspace(0, 1, 10, dtype=torch.float32),
    }
    out = tmp_path / "round_trip_df.fits"
    tf.write_table(str(out), data, header={"EXTNAME": "DAT"}, overwrite=True)
    try:
        from torchfits import dataframe_round_trip
    except ImportError:
        pytest.skip("torch_frame not available")
    df, back = dataframe_round_trip(str(out), hdu=1)
    # Value parity
    assert torch.equal(back.data["X"], data["X"])  # int preserved
    assert torch.allclose(back.data["Y"].float(), data["Y"].float(), atol=1e-7)


def test_table_torch_frame_full_file_round_trip(tmp_path):
    # Original table with units
    data = {
        "RA": torch.linspace(0, 10, 5, dtype=torch.float32),
        "DEC": torch.linspace(-5, 5, 5, dtype=torch.float32),
        "ID": torch.arange(5, dtype=torch.int32),
    }
    units = ["deg", "deg", ""]
    out_src = tmp_path / "src_tbl.fits"
    out_dst = tmp_path / "rt_tbl.fits"
    tf.write_table(
        str(out_src),
        data,
        header={"EXTNAME": "SRC"},
        column_units=units,
        overwrite=True,
    )
    try:
        from torchfits import torch_frame_round_trip_file
    except ImportError:
        pytest.skip("torch_frame not available")
    orig, df, new = torch_frame_round_trip_file(str(out_src), str(out_dst), hdu=1)
    # Dtype parity (int column stays int)
    assert orig.data["ID"].dtype == new.data["ID"].dtype
    # Unit parity for RA, DEC
    assert orig.column_info["RA"].unit == new.column_info["RA"].unit == "deg"
    assert orig.column_info["DEC"].unit == new.column_info["DEC"].unit == "deg"
    # Numeric value parity
    assert torch.allclose(orig.data["RA"], new.data["RA"], atol=1e-7)
    assert torch.allclose(orig.data["DEC"], new.data["DEC"], atol=1e-7)


def test_write_variable_length_array_fuzz(tmp_path):
    import random

    arrays = []
    for _ in range(20):
        n = random.randint(0, 15)
        if n == 0:
            arrays.append(torch.empty(0, dtype=torch.float32))
        else:
            arrays.append(torch.randn(n, dtype=torch.float32))
    out = tmp_path / "var_arr_fuzz.fits"
    tf.write_variable_length_array(str(out), arrays, {"OBJECT": "FUZZ"}, overwrite=True)
    try:
        import numpy as np
        from astropy.io import fits
    except ImportError:
        pytest.skip("astropy required for validation")
    with fits.open(str(out)) as hdul:
        tbl = hdul[1].data
        col = tbl["ARRAY_DATA"]
        assert [a.numel() for a in arrays] == [len(r) for r in col]
        for r, a in zip(col, arrays):
            np.testing.assert_allclose(
                r, a.numpy().astype(r.dtype), rtol=1e-6, atol=1e-6
            )
