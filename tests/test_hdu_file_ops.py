import pytest
import torch
import time

import torchfits


def test_hdu_file_ops_insert_replace_delete(tmp_path):
    path = tmp_path / "ops.fits"
    base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torchfits.write(str(path), base, overwrite=True)

    payload1 = torch.zeros((4, 4), dtype=torch.float32)
    torchfits.insert_hdu(str(path), payload1, index=1, header={"EXTNAME": "SCI"})

    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 2
        assert hdul["SCI"].header.get("EXTNAME") == "SCI"
    assert torch.equal(torchfits.read(str(path), hdu=1), payload1)

    payload2 = torch.ones((4, 4), dtype=torch.float32)
    torchfits.replace_hdu(str(path), "SCI", payload2)
    assert torch.equal(torchfits.read(str(path), hdu=1), payload2)

    torchfits.delete_hdu(str(path), "SCI")
    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 1
    with pytest.raises(Exception):
        _ = torchfits.read(str(path), hdu=1)


def test_external_overwrite_invalidates_cached_handle(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    path = tmp_path / "external_overwrite.fits"

    original = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    updated = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 100.0

    torchfits.write(str(path), original, overwrite=True)
    first = torchfits.read(str(path), hdu=0)
    assert torch.equal(first, original)

    fits.PrimaryHDU(updated.numpy()).writeto(path, overwrite=True)
    second = torchfits.read(str(path), hdu=0)
    deadline = time.monotonic() + 3.0
    while not torch.equal(second, updated) and time.monotonic() < deadline:
        time.sleep(0.01)
        second = torchfits.read(str(path), hdu=0)
    assert torch.equal(second, updated)


def test_tablehduref_file_mutators_roundtrip(tmp_path):
    path = tmp_path / "table_mutators.fits"
    torchfits.write(
        str(path),
        {
            "ID": torch.tensor([1, 2, 3], dtype=torch.int32),
            "VAL": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            "FLAG": torch.tensor([True, False, True], dtype=torch.bool),
        },
        overwrite=True,
    )

    with torchfits.open(str(path)) as hdul:
        table = hdul[1]
        table = table.append_rows_file({"ID": [4], "VAL": [0.4], "FLAG": [False]})
        table = table.insert_rows_file(
            {"ID": [99], "VAL": [9.9], "FLAG": [True]}, row=1
        )
        table = table.update_rows_file({"VAL": [8.8]}, row_slice=slice(0, 1))
        table = table.rename_columns_file({"VAL": "FLUX"})
        table = table.drop_columns_file(["FLAG"])

    with torchfits.open(str(path)) as hdul:
        table = hdul[1]
        assert table["ID"].squeeze(-1).tolist() == [1, 99, 2, 3, 4]
        assert table["FLUX"].squeeze(-1).shape[0] == 5
        assert abs(float(table["FLUX"].squeeze(-1)[0]) - 8.8) < 1e-6
        assert "FLAG" not in table.columns


def test_tablehduref_mutation_refreshes_schema_and_rowcount(tmp_path):
    path = tmp_path / "table_refresh.fits"
    torchfits.write(
        str(path),
        {
            "ID": torch.tensor([1, 2], dtype=torch.int32),
            "VAL": torch.tensor([10.0, 20.0], dtype=torch.float32),
        },
        overwrite=True,
    )

    with torchfits.open(str(path)) as hdul:
        table = hdul[1]
        updated = table.append_rows_file({"ID": [3], "VAL": [30.0]})
        assert updated.num_rows == 3
        renamed = updated.rename_columns_file({"ID": "OBJID"})
        assert "OBJID" in renamed.columns
        assert "ID" not in renamed.columns


def test_tablehduref_column_mutators_roundtrip(tmp_path):
    path = tmp_path / "table_column_mutators.fits"
    torchfits.table.write(
        str(path),
        data={
            "ID": torch.tensor([1, 2], dtype=torch.int32),
            "QUAL": torch.tensor([10, 20], dtype=torch.int16),
        },
        schema={
            "ID": {"format": "J"},
            "QUAL": {"format": "I", "unit": "adu", "tnull": -999},
        },
        overwrite=True,
    )

    with torchfits.open(str(path)) as hdul:
        table = hdul[1]
        table = table.insert_column_file(
            "FLAGS",
            [7, 8],
            index=1,
            format="I",
            unit="flag",
            tnull=-1,
        )
        table = table.replace_column_file("QUAL", [101, 102])
        assert table.columns == ["ID", "FLAGS", "QUAL"]

    with torchfits.open(str(path)) as hdul:
        table = hdul[1]
        assert table["FLAGS"].squeeze(-1).tolist() == [7, 8]
        assert table["QUAL"].squeeze(-1).tolist() == [101, 102]
        assert table.header.get("TUNIT2") == "flag"
        assert int(table.header.get("TNULL2")) == -1
        assert table.header.get("TUNIT3") == "adu"
        assert int(table.header.get("TNULL3")) == -999


def test_hdulist_write_preserves_table_extension_metadata(tmp_path):
    src = tmp_path / "table_src.fits"
    dst = tmp_path / "table_dst.fits"

    torchfits.table.write(
        str(src),
        data={
            "ID": torch.tensor([1, 2], dtype=torch.int32),
            "QUAL": torch.tensor([10, 20], dtype=torch.int16),
        },
        schema={
            "ID": {"format": "J"},
            "QUAL": {"format": "I", "unit": "adu", "tnull": -999},
        },
        overwrite=True,
    )

    with torchfits.open(str(src)) as hdul:
        hdul.write(str(dst), overwrite=True)

    with torchfits.open(str(dst)) as hdul:
        assert len(hdul) == 2
        table = hdul[1]
        assert table.columns == ["ID", "QUAL"]
        assert table["QUAL"].squeeze(-1).tolist() == [10, 20]
        assert str(table.header.get("TFORM2", "")).upper().startswith("I")
        assert table.header.get("TUNIT2") == "adu"
        assert int(table.header.get("TNULL2")) == -999


def test_insert_hdu_compressed_image_only(tmp_path):
    path = tmp_path / "insert_compressed.fits"
    base = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    torchfits.write(str(path), base, overwrite=True, compress=True)

    inserted = torch.full((4, 4), 7, dtype=torch.int16)
    torchfits.insert_hdu(
        str(path),
        inserted,
        index=1,
        header={"EXTNAME": "SCI"},
        compress=True,
    )

    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 3
        assert hdul[1].header.get("EXTNAME") == "SCI"
    assert torch.equal(torchfits.read(str(path), hdu=1), inserted)
    assert torch.equal(torchfits.read(str(path), hdu=2), base)


def test_replace_hdu_compressed_image_only(tmp_path):
    path = tmp_path / "replace_compressed.fits"
    base = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    torchfits.write(str(path), base, overwrite=True, compress=True)

    replacement = torch.full((4, 4), 3, dtype=torch.int16)
    torchfits.replace_hdu(
        str(path),
        1,
        replacement,
        header={"EXTNAME": "SCI_REPLACED"},
        compress="RICE_1",
    )

    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 2
        assert hdul[1].header.get("EXTNAME") == "SCI_REPLACED"
    assert torch.equal(torchfits.read(str(path), hdu=1), replacement)


def test_insert_hdu_compressed_mixed_hdus_roundtrip(tmp_path):
    path = tmp_path / "insert_compressed_mixed.fits"
    image = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    table = {"ID": torch.tensor([1, 2, 3], dtype=torch.int32)}
    torchfits.write(
        str(path),
        torchfits.HDUList([torchfits.TensorHDU(image), torchfits.TableHDU(table)]),
        overwrite=True,
    )

    inserted = torch.full((4, 4), 11, dtype=torch.int16)
    torchfits.insert_hdu(
        str(path),
        inserted,
        index=1,
        header={"EXTNAME": "SCI_INSERT"},
        compress=True,
    )

    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 4
        assert hdul[2].header.get("EXTNAME") == "SCI_INSERT"
    assert torch.equal(torchfits.read(str(path), hdu=2), inserted)
    table_out = torchfits.read(str(path), hdu=3)
    assert table_out["ID"].squeeze(-1).tolist() == [1, 2, 3]


def test_delete_hdu_compressed_image_only(tmp_path):
    path = tmp_path / "delete_compressed.fits"
    img0 = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    img1 = torch.full((4, 4), 9, dtype=torch.int16)
    torchfits.write(
        str(path),
        torchfits.HDUList([torchfits.TensorHDU(img0), torchfits.TensorHDU(img1)]),
        overwrite=True,
        compress=True,
    )

    torchfits.delete_hdu(str(path), 1, compress=True)
    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 2
    assert torch.equal(torchfits.read(str(path), hdu=1), img1)


def test_delete_hdu_compressed_mixed_hdus_roundtrip(tmp_path):
    path = tmp_path / "delete_compressed_mixed.fits"
    image = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    table = {"ID": torch.tensor([1, 2], dtype=torch.int32)}
    torchfits.write(
        str(path),
        torchfits.HDUList([torchfits.TensorHDU(image), torchfits.TableHDU(table)]),
        overwrite=True,
    )

    torchfits.delete_hdu(str(path), 0, compress=True)
    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 2
    table_out = torchfits.read(str(path), hdu=1)
    assert table_out["ID"].squeeze(-1).tolist() == [1, 2]


def test_replace_hdu_compressed_mixed_hdus_roundtrip(tmp_path):
    path = tmp_path / "replace_compressed_mixed.fits"
    image = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    table = {"ID": torch.tensor([1, 2], dtype=torch.int32)}
    torchfits.write(
        str(path),
        torchfits.HDUList([torchfits.TensorHDU(image), torchfits.TableHDU(table)]),
        overwrite=True,
    )

    replacement = {"ID": torch.tensor([10, 20, 30], dtype=torch.int32)}
    torchfits.replace_hdu(
        str(path),
        1,
        replacement,
        header={"EXTNAME": "CAT_REPLACED"},
        compress=True,
    )

    with torchfits.open(str(path)) as hdul:
        assert len(hdul) == 3
        assert hdul[2].header.get("EXTNAME") == "CAT_REPLACED"
    table_out = torchfits.read(str(path), hdu=2)
    assert table_out["ID"].squeeze(-1).tolist() == [10, 20, 30]
