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
            assert ids.squeeze(-1).tolist() == [1, 2, 3, 4, 5]
            assert np.allclose(
                vals.squeeze(-1).numpy(), [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-6
            )
            assert flags.squeeze(-1).tolist() == [True, False, True, False, True]

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
            vals = table_hdu["VAL"].squeeze(-1).numpy()
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


def test_tablehdu_to_fits_rich_types_pending_cfitsio_impl():
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
            vals = table["Z"].squeeze(-1)
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


def test_table_write_schema_pending_cfitsio_impl():
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
            assert table["ID"].squeeze(-1).tolist() == [1, 2, 3]
            assert str(table.header.get("TFORM1", "")).upper().startswith("J")
    finally:
        os.unlink(path.name)


def test_table_write_ascii_pending_cfitsio_impl():
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
            assert table["A"].squeeze(-1).tolist() == [1, 2, 3]
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
            assert table_hdu["ID"].squeeze(-1).tolist() == [1, 99, 100, 2, 3]
            assert np.allclose(
                table_hdu["VAL"].squeeze(-1).numpy(),
                [0.1, 9.9, 10.0, 0.2, 0.3],
                atol=1e-6,
            )
            assert table_hdu["FLAG"].squeeze(-1).tolist() == [
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
            assert table_hdu["ID"].squeeze(-1).tolist() == [1, 3]
            assert np.allclose(
                table_hdu["VAL"].squeeze(-1).numpy(), [0.1, 0.3], atol=1e-6
            )
            assert table_hdu["FLAG"].squeeze(-1).tolist() == [True, True]
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
            assert table_hdu["ID"].squeeze(-1).tolist() == [1, 2, 3]
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
            assert table_hdu["ID"].squeeze(-1).tolist() == [1, 2, 3]
            assert table_hdu["QUAL"].squeeze(-1).tolist() == [7, 8, -999]
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
            assert table_hdu["ID"].squeeze(-1).tolist() == [1, 99, 3]
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
            assert table["FLAGS"].squeeze(-1).tolist() == [7, 8]
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
            assert table["QUAL"].squeeze(-1).tolist() == [101, 102, 103]
            assert str(table.header.get("TFORM2", "")).upper().startswith("I")
            assert table.header.get("TUNIT2") == "adu"
            assert int(table.header.get("TNULL2")) == -999
    finally:
        os.unlink(path.name)
