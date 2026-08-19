import os
import tempfile

import numpy as np
import pytest
import torch

import torchfits


def _make_basic_table_file() -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    torchfits.write(
        handle.name,
        {
            "ID": np.array([1, 2, 3], dtype=np.int32),
            "VAL": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "FLAG": np.array([True, False, True], dtype=np.bool_),
        },
        overwrite=True,
    )
    return handle.name


def test_table_write_negative_stride_column_roundtrip():
    """A reversed (negative-stride) column must be copied correctly on write."""
    base = np.arange(1, 6, dtype=np.int32)  # [1, 2, 3, 4, 5]
    ids = base[::-1]  # negative-stride view: [5, 4, 3, 2, 1]
    vals = base.astype(np.float32)[::-1]
    assert ids.strides[0] < 0 and vals.strides[0] < 0
    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    try:
        torchfits.write(handle.name, {"ID": ids, "VAL": vals}, overwrite=True)
        with torchfits.open(handle.name) as hdul:
            out_ids = hdul[1]["ID"].tolist()
            out_vals = hdul[1]["VAL"].numpy()
        assert out_ids == [5, 4, 3, 2, 1]
        assert np.allclose(out_vals, [5.0, 4.0, 3.0, 2.0, 1.0], atol=1e-6)
    finally:
        os.unlink(handle.name)


def test_update_rows_mmap_forced_failure_not_swallowed(tmp_path):
    path = str(tmp_path / "test_mmap_swallow.fits")
    torchfits.table.write(path, {"A": np.array([1, 2, 3])})

    import torchfits._C as cpp

    original_mmap = cpp.update_fits_table_rows_mmap

    def mock_mmap(*args, **kwargs):
        raise RuntimeError("Mock mmap error")

    cpp.update_fits_table_rows_mmap = mock_mmap

    try:
        with pytest.raises(RuntimeError, match="Mock mmap error"):
            torchfits.table.update_rows(
                path, {"A": np.array([4, 5])}, row_slice=slice(0, 2), mmap="mmap"
            )

        with pytest.raises(RuntimeError, match="Mock mmap error"):
            torchfits.table.update_rows(
                path, {"A": np.array([4, 5])}, row_slice=slice(0, 2), mmap=True
            )

        # auto should swallow the error and fall back
        torchfits.table.update_rows(
            path, {"A": np.array([4, 5])}, row_slice=slice(0, 2), mmap="auto"
        )

        # Verify it actually updated by reading it
        table = torchfits.table.read_torch(path, columns=["A"])
        assert table["A"].tolist() == [4, 5, 3]

    finally:
        cpp.update_fits_table_rows_mmap = original_mmap


def test_table_append_update_rename_drop():
    path = _make_basic_table_file()
    try:
        torchfits.table.append_rows(
            path,
            {
                "ID": np.array([4, 5], dtype=np.int32),
                "VAL": np.array([0.4, 0.5], dtype=np.float32),
                "FLAG": np.array([False, True], dtype=np.bool_),
            },
            hdu=1,
        )
        with torchfits.open(path) as hdul:
            table_hdu = hdul[1]
            ids = table_hdu["ID"]
            vals = table_hdu["VAL"]
            flags = table_hdu["FLAG"]
            assert isinstance(ids, torch.Tensor)
            assert isinstance(vals, torch.Tensor)
            assert isinstance(flags, torch.Tensor)
            assert ids.tolist() == [1, 2, 3, 4, 5]
            assert np.allclose(vals.numpy(), [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-6)
            assert flags.tolist() == [True, False, True, False, True]

        with pytest.raises(ValueError):
            torchfits.table.append_rows(
                path,
                {
                    "ID": [6],
                    "VAL": [0.6],
                    "FLAG": [False],
                    "EXTRA": [1],
                },
                hdu=1,
            )

        torchfits.table.update_rows(
            path,
            {"VAL": np.array([9.9, 8.8], dtype=np.float32)},
            row_slice=slice(1, 3),
            hdu=1,
        )
        with torchfits.open(path) as hdul:
            table_hdu = hdul[1]
            vals = table_hdu["VAL"].numpy()
            assert np.allclose(vals, [0.1, 9.9, 8.8, 0.4, 0.5], atol=1e-6)

        torchfits.table.rename_columns(path, {"VAL": "FLUX"}, hdu=1)
        with torchfits.open(path) as hdul:
            table_hdu = hdul[1]
            assert "VAL" not in table_hdu.columns
            assert "FLUX" in table_hdu.columns

        torchfits.table.drop_columns(path, ["FLAG"], hdu=1)
        with torchfits.open(path) as hdul:
            table_hdu = hdul[1]
            assert "FLAG" not in table_hdu.columns
    finally:
        os.unlink(path)


def test_tablehdu_to_fits_rich_types_roundtrip():
    dst = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    dst.close()
    table_hdu = torchfits.TableHDU(
        {
            "ID": np.array([1, 2, 3], dtype=np.int32),
            "NAME": ["alpha", "beta", "gamma"],
            "Z": np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64),
            "VLA": [
                np.array([1, 2], dtype=np.int32),
                np.array([3], dtype=np.int32),
                np.array([4, 5, 6], dtype=np.int32),
            ],
        }
    )

    try:
        table_hdu.to_fits(dst.name, overwrite=True)
        with torchfits.open(dst.name) as hdul:
            table = hdul[1]
            assert table.get_string_column("NAME") == ["alpha", "beta", "gamma"]
            vals = table["Z"]
            assert np.allclose(
                vals.numpy(), np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
            )
            vla = table.get_vla_column("VLA")
            assert [v.tolist() for v in vla] == [[1, 2], [3], [4, 5, 6]]
    finally:
        if os.path.exists(dst.name):
            os.unlink(dst.name)


def test_table_vla_roundtrip_and_updates():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        table = {
            "ID": np.array([1, 2, 3], dtype=np.int32),
            "VLA": [
                np.array([1, 2], dtype=np.int16),
                np.array([3], dtype=np.int16),
                np.array([], dtype=np.int16),
            ],
        }
        torchfits.write(path.name, table, overwrite=True)

        with torchfits.open(path.name) as hdul:
            vla = hdul[1].get_vla_column("VLA")
            assert [v.tolist() for v in vla] == [[1, 2], [3], []]

        torchfits.table.append_rows(
            path.name,
            {
                "ID": np.array([4], dtype=np.int32),
                "VLA": [np.array([9, 10], dtype=np.int16)],
            },
            hdu=1,
        )
        with torchfits.open(path.name) as hdul:
            vla = hdul[1].get_vla_column("VLA")
            assert vla[-1].tolist() == [9, 10]

        torchfits.table.update_rows(
            path.name,
            {"VLA": [np.array([7], dtype=np.int16)]},
            row_slice=slice(1, 2),
            hdu=1,
        )
        with torchfits.open(path.name) as hdul:
            vla = hdul[1].get_vla_column("VLA")
            assert vla[1].tolist() == [7]
    finally:
        os.unlink(path.name)


def test_table_write_schema_roundtrip():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchfits.table.write(
            path.name,
            data={"ID": np.array([1, 2, 3], dtype=np.int32)},
            schema={"ID": {"format": "J"}},
            overwrite=True,
        )
        with torchfits.open(path.name) as hdul:
            table = hdul[1]
            assert table["ID"].tolist() == [1, 2, 3]
            assert str(table.header.get("TFORM1", "")).upper().startswith("J")
    finally:
        os.unlink(path.name)


def test_table_write_ascii_roundtrip():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchfits.table.write(
            path.name,
            data={"A": np.array([1, 2, 3], dtype=np.int32)},
            table_type="ascii",
            overwrite=True,
        )
        with torchfits.open(path.name) as hdul:
            table = hdul[1]
            assert table["A"].tolist() == [1, 2, 3]
            assert str(table.header.get("XTENSION", "")).upper() == "TABLE"
    finally:
        os.unlink(path.name)


def test_table_reader_cache_invalidated_after_append():
    path = _make_basic_table_file()
    try:
        # Prime Arrow/cpp reader cache for this path.
        initial = torchfits.table.read(path, hdu=1)
        assert initial.num_rows == 3

        torchfits.table.append_rows(
            path,
            {
                "ID": np.array([4], dtype=np.int32),
                "VAL": np.array([0.4], dtype=np.float32),
                "FLAG": np.array([False], dtype=np.bool_),
            },
            hdu=1,
        )

        updated = torchfits.table.read(path, hdu=1)
        assert updated.num_rows == 4
        assert updated.column("ID").to_pylist() == [1, 2, 3, 4]
    finally:
        os.unlink(path)


def test_table_insert_rows_mid_table_preserves_order():
    path = _make_basic_table_file()
    try:
        torchfits.table.insert_rows(
            path,
            {
                "ID": np.array([99, 100], dtype=np.int32),
                "VAL": np.array([9.9, 10.0], dtype=np.float32),
                "FLAG": np.array([False, True], dtype=np.bool_),
            },
            row=1,
            hdu=1,
        )

        with torchfits.open(path) as hdul:
            table_hdu = hdul[1]
            assert table_hdu["ID"].tolist() == [1, 99, 100, 2, 3]
            assert np.allclose(
                table_hdu["VAL"].numpy(),
                [0.1, 9.9, 10.0, 0.2, 0.3],
                atol=1e-6,
            )
            assert table_hdu["FLAG"].tolist() == [
                True,
                False,
                True,
                False,
                True,
            ]
    finally:
        os.unlink(path)


def test_table_delete_rows_slice_and_single():
    path = _make_basic_table_file()
    try:
        torchfits.table.append_rows(
            path,
            {
                "ID": np.array([4, 5], dtype=np.int32),
                "VAL": np.array([0.4, 0.5], dtype=np.float32),
                "FLAG": np.array([False, True], dtype=np.bool_),
            },
            hdu=1,
        )
        torchfits.table.delete_rows(path, 1, hdu=1)
        torchfits.table.delete_rows(path, slice(2, 4), hdu=1)

        with torchfits.open(path) as hdul:
            table_hdu = hdul[1]
            assert table_hdu["ID"].tolist() == [1, 3]
            assert np.allclose(table_hdu["VAL"].numpy(), [0.1, 0.3], atol=1e-6)
            assert table_hdu["FLAG"].tolist() == [True, True]
    finally:
        os.unlink(path)


def test_append_rows_partial_payload_string_vla_defaults():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchfits.write(
            path.name,
            {
                "ID": np.array([1, 2], dtype=np.int32),
                "NAME": ["alpha", "beta"],
                "VLA": [
                    np.array([10, 11], dtype=np.int16),
                    np.array([20], dtype=np.int16),
                ],
            },
            overwrite=True,
        )

        torchfits.table.append_rows(
            path.name,
            {"ID": np.array([3], dtype=np.int32)},
            hdu=1,
        )

        with torchfits.open(path.name) as hdul:
            table_hdu = hdul[1]
            assert table_hdu["ID"].tolist() == [1, 2, 3]
            assert table_hdu.get_string_column("NAME") == ["alpha", "beta", ""]
            vla = table_hdu.get_vla_column("VLA")
            assert [v.tolist() for v in vla] == [[10, 11], [20], []]
    finally:
        os.unlink(path.name)


def test_append_rows_partial_payload_respects_tnull():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchfits.table.write(
            path.name,
            data={
                "ID": np.array([1, 2], dtype=np.int32),
                "QUAL": np.array([7, 8], dtype=np.int16),
            },
            schema={
                "ID": {"format": "J"},
                "QUAL": {"format": "I", "tnull": -999},
            },
            overwrite=True,
        )

        torchfits.table.append_rows(
            path.name,
            {"ID": np.array([3], dtype=np.int32)},
            hdu=1,
        )

        with torchfits.open(path.name) as hdul:
            table_hdu = hdul[1]
            assert table_hdu["ID"].tolist() == [1, 2, 3]
            assert table_hdu["QUAL"].tolist() == [7, 8, -999]
    finally:
        os.unlink(path.name)


def test_table_insert_delete_with_vla_and_string_columns():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchfits.write(
            path.name,
            {
                "ID": np.array([1, 2, 3], dtype=np.int32),
                "NAME": ["a", "b", "c"],
                "VLA": [
                    np.array([1], dtype=np.int16),
                    np.array([2, 3], dtype=np.int16),
                    np.array([4], dtype=np.int16),
                ],
            },
            overwrite=True,
        )

        torchfits.table.insert_rows(
            path.name,
            {"ID": np.array([99], dtype=np.int32)},
            row=1,
            hdu=1,
        )
        torchfits.table.delete_rows(path.name, slice(2, 3), hdu=1)

        with torchfits.open(path.name) as hdul:
            table_hdu = hdul[1]
            assert table_hdu["ID"].tolist() == [1, 99, 3]
            assert table_hdu.get_string_column("NAME") == ["a", "", "c"]
            vla = table_hdu.get_vla_column("VLA")
            assert [v.tolist() for v in vla] == [[1], [], [4]]
    finally:
        os.unlink(path.name)


def test_insert_column_with_explicit_format_metadata():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchfits.table.write(
            path.name,
            data={
                "ID": np.array([1, 2], dtype=np.int32),
                "QUAL": np.array([10, 20], dtype=np.int16),
            },
            schema={
                "ID": {"format": "J"},
                "QUAL": {"format": "I", "unit": "adu", "tnull": -999},
            },
            overwrite=True,
        )

        torchfits.table.insert_column(
            path.name,
            "FLAGS",
            np.array([7, 8], dtype=np.int16),
            hdu=1,
            index=1,
            format="I",
            unit="flag",
            tnull=-1,
        )

        with torchfits.open(path.name) as hdul:
            table = hdul[1]
            assert table.columns == ["ID", "FLAGS", "QUAL"]
            assert table["FLAGS"].tolist() == [7, 8]
            assert table.header.get("TTYPE2") == "FLAGS"
            assert str(table.header.get("TFORM2", "")).upper().startswith("I")
            assert table.header.get("TUNIT2") == "flag"
            assert int(table.header.get("TNULL2")) == -1
            # Existing metadata moves with the original column.
            assert table.header.get("TTYPE3") == "QUAL"
            assert table.header.get("TUNIT3") == "adu"
            assert int(table.header.get("TNULL3")) == -999
    finally:
        os.unlink(path.name)


def test_insert_column_infers_format_from_numpy_and_list():
    """insert_column without format= must infer TFORM (regression: NameError np)."""
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchfits.table.write(
            path.name,
            data={"ID": np.array([1, 2], dtype=np.int32)},
            schema={"ID": {"format": "J"}},
            overwrite=True,
        )

        torchfits.table.insert_column(
            path.name, "FA", np.array([1.5, 2.5], dtype=np.float64), hdu=1
        )
        torchfits.table.insert_column(path.name, "LB", [7, 8], hdu=1)

        with torchfits.open(path.name) as hdul:
            table = hdul[1]
            assert set(table.columns) == {"ID", "FA", "LB"}
            assert table["FA"].tolist() == [1.5, 2.5]
            assert table["LB"].tolist() == [7, 8]
    finally:
        os.unlink(path.name)


def test_replace_column_preserves_metadata_contract():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchfits.table.write(
            path.name,
            data={
                "ID": np.array([1, 2, 3], dtype=np.int32),
                "QUAL": np.array([11, 12, 13], dtype=np.int16),
            },
            schema={
                "ID": {"format": "J"},
                "QUAL": {"format": "I", "unit": "adu", "tnull": -999},
            },
            overwrite=True,
        )

        torchfits.table.replace_column(
            path.name,
            "QUAL",
            np.array([101, 102, 103], dtype=np.int16),
            hdu=1,
        )

        with torchfits.open(path.name) as hdul:
            table = hdul[1]
            assert table["QUAL"].tolist() == [101, 102, 103]
            assert str(table.header.get("TFORM2", "")).upper().startswith("I")
            assert table.header.get("TUNIT2") == "adu"
            assert int(table.header.get("TNULL2")) == -999
    finally:
        os.unlink(path.name)


@pytest.mark.parametrize(
    "dtype,signed_name",
    [
        (np.uint16, "int16"),
        (np.uint32, "int32"),
        (np.uint64, "int64"),
    ],
)
def test_infer_fits_scalar_code_unsigned_dtype_raises_helpful_type_error(
    dtype, signed_name
):
    """uint16/32/64 have no native FITS TFORM; the error should say why and

    point at a fix (cast to signed, or use the BZERO unsigned write path)
    instead of the opaque "Cannot infer FITS TFORM" message.
    """
    from torchfits._table.mutation import _infer_fits_scalar_code

    with pytest.raises(TypeError) as excinfo:
        _infer_fits_scalar_code(np.array([1, 2], dtype=dtype))
    message = str(excinfo.value)
    assert "unsigned" in message.lower()
    assert signed_name in message
    assert "BZERO" in message


def test_infer_fits_scalar_code_uint8_still_maps_to_b():
    from torchfits._table.mutation import _infer_fits_scalar_code

    assert _infer_fits_scalar_code(np.array([1, 2], dtype=np.uint8)) == "B"


def test_table_write_noncontiguous_numeric_roundtrip():
    """A7: non-C-contiguous column views must still write correctly."""
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        col = np.arange(10, dtype=np.float32)[::2]
        assert not col.flags["C_CONTIGUOUS"]
        torchfits.table.write(
            path.name,
            data={"VAL": col},
            overwrite=True,
        )
        with torchfits.open(path.name) as hdul:
            got = hdul[1]["VAL"].detach().cpu().numpy()
        np.testing.assert_array_equal(got, col)
    finally:
        os.unlink(path.name)
