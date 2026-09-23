import os
import tempfile

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.table import Table

import torchfits


def _make_table_file():
    table = Table(
        {
            "RA": np.array([10.1, 10.2, 10.3], dtype=np.float64),
            "ID": np.array([1, 2, 3], dtype=np.int64),
            "NAME": np.array(["STAR_A", "STAR_B", "STAR_C"], dtype="U8"),
        }
    )
    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    table.write(handle.name, format="fits", overwrite=True)
    return handle.name


def _make_tnull_table_file(vector: bool = False):
    if vector:
        values = np.array([[1, -999], [3, 4], [-999, 6]], dtype=np.int16)
    else:
        values = np.array([1, -999, 3], dtype=np.int16)

    table = Table({"A": values})
    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    table.write(handle.name, format="fits", overwrite=True)

    with fits.open(handle.name, mode="update") as hdul:
        hdul[1].header["TNULL1"] = -999

    return handle.name


def _make_scaled_table_file(vector: bool = False):
    if vector:
        values = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16)
    else:
        values = np.array([1, 2, 3, 4], dtype=np.int16)

    table = Table({"A": values})
    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    table.write(handle.name, format="fits", overwrite=True)

    with fits.open(handle.name, mode="update") as hdul:
        hdul[1].header["TSCAL1"] = 0.5
        hdul[1].header["TZERO1"] = 1.25

    return handle.name


def _make_bit_vla_table_file():
    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()

    bit = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    vla = np.empty(3, dtype=object)
    vla[0] = np.array([1, 2], dtype=np.int32)
    vla[1] = np.array([3], dtype=np.int32)
    vla[2] = np.array([4, 5, 6], dtype=np.int32)

    cols = [
        fits.Column(name="BITS", format="8X", array=bit),
        fits.Column(name="VLA", format="PJ()", array=vla),
    ]
    fits.BinTableHDU.from_columns(cols).writeto(handle.name, overwrite=True)
    return handle.name


def test_arrow_scan_and_read():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        batches = list(
            torchfits.table.scan(
                path, hdu=1, batch_size=2, decode_bytes=True, include_fits_metadata=True
            )
        )
        assert len(batches) == 2
        assert batches[0].num_rows == 2
        assert batches[1].num_rows == 1

        arrow_table = torchfits.table.read(
            path, hdu=1, batch_size=2, decode_bytes=True, include_fits_metadata=True
        )
        assert arrow_table.num_rows == 3
        assert set(arrow_table.column_names) == {"RA", "ID", "NAME"}
        md = arrow_table.schema.field("RA").metadata or {}
        assert b"fits_tform" in md
    finally:
        os.unlink(path)


def test_arrow_cpp_backend_matches_default():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        import torchfits.cpp as cpp

        if not hasattr(cpp, "read_fits_table_rows"):
            pytest.skip("cpp table backend not available")

        t_default = torchfits.table.read(
            path, hdu=1, decode_bytes=True, backend="torch"
        )
        t_cpp = torchfits.table.read(path, hdu=1, decode_bytes=True, backend="cpp")

        assert t_default.num_rows == t_cpp.num_rows
        assert sorted(t_default.schema.names) == sorted(t_cpp.schema.names)
        assert t_default.column("ID").to_pylist() == t_cpp.column("ID").to_pylist()
    finally:
        os.unlink(path)


def test_arrow_row_slice():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        arrow_table = torchfits.table.read(
            path, hdu=1, row_slice=slice(1, 3), decode_bytes=True
        )
        assert arrow_table.num_rows == 2
        assert arrow_table.column("ID").to_pylist() == [2, 3]
    finally:
        os.unlink(path)


def test_arrow_rows_selection_preserves_order():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        arrow_table = torchfits.table.read(path, hdu=1, rows=[2, 0], decode_bytes=True)
        assert arrow_table.num_rows == 2
        assert arrow_table.column("ID").to_pylist() == [3, 1]
    finally:
        os.unlink(path)


def test_scan_torch_cpu_batches():
    path = _make_table_file()
    try:
        chunks = list(
            torchfits.table.scan_torch(path, hdu=1, batch_size=2, device="cpu")
        )
        assert len(chunks) == 2
        assert isinstance(chunks[0]["RA"], torch.Tensor)
        assert chunks[0]["RA"].shape[0] == 2
    finally:
        os.unlink(path)


def test_scan_torch_accelerator_if_available():
    path = _make_table_file()
    try:
        device = None
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        if device is None:
            pytest.skip("No accelerator available")

        chunk = next(
            torchfits.table.scan_torch(path, hdu=1, batch_size=2, device=device)
        )
        assert isinstance(chunk["RA"], torch.Tensor)
        assert chunk["RA"].device.type == device
    finally:
        os.unlink(path)


def test_scan_torch_fallback_for_bit_vla_mmap():
    path = _make_bit_vla_table_file()
    try:
        chunk = next(
            torchfits.table.scan_torch(path, hdu=1, batch_size=2, device="cpu")
        )
        assert isinstance(chunk["BITS"], torch.Tensor)
        assert chunk["BITS"].shape[0] == 2
    finally:
        os.unlink(path)


def test_scan_torch_columns_vla_mmap_true_falls_back():
    path = _make_bit_vla_table_file()
    try:
        chunk = next(
            torchfits.table.scan_torch(
                path, hdu=1, columns=["VLA"], batch_size=2, mmap=True, device="cpu"
            )
        )
        v = chunk["VLA"]
        # Depending on backend, VLA may be materialized as:
        # - list[Tensor] (torch path), or
        # - (flat_values, offsets) tuple (cpp numpy path).
        if isinstance(v, list):
            assert len(v) == 2
            assert [np.asarray(x).tolist() for x in v] == [[1, 2], [3]]
        else:
            assert isinstance(v, tuple) and len(v) == 2
            flat, offsets = v
            flat = np.asarray(flat)
            offsets = np.asarray(offsets, dtype=np.int64)
            out = []
            for i in range(2):
                a = int(offsets[i])
                b = int(offsets[i + 1])
                out.append(flat[a:b].tolist())
            assert out == [[1, 2], [3]]
    finally:
        os.unlink(path)


def test_arrow_read_columns_vla_mmap_true_falls_back():
    pytest.importorskip("pyarrow")
    path = _make_bit_vla_table_file()
    try:
        table = torchfits.table.read(path, hdu=1, columns=["VLA"], mmap=True)
        assert table.column_names == ["VLA"]
        assert table.column("VLA").to_pylist() == [[1, 2], [3], [4, 5, 6]]
    finally:
        os.unlink(path)


def test_arrow_to_pandas_stream():
    pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")
    path = _make_table_file()
    try:
        dfs = list(
            torchfits.table.to_pandas(
                torchfits.table.scan(path, hdu=1, batch_size=2, decode_bytes=True),
                stream=True,
            )
        )
        assert len(dfs) == 2
        merged = pd.concat(dfs, ignore_index=True)
        assert merged.shape[0] == 3
        assert merged["ID"].tolist() == [1, 2, 3]
    finally:
        os.unlink(path)


def test_arrow_schema_minimal():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        sch = torchfits.table.schema(
            path, hdu=1, columns=["ID", "RA"], include_fits_metadata=True
        )
        assert sch.names == ["ID", "RA"]
        assert sch.metadata is not None
        assert b"fits_hdu" in sch.metadata
    finally:
        os.unlink(path)


def test_arrow_reader_and_dataset():
    pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.dataset")
    path = _make_table_file()
    try:
        reader = torchfits.table.reader(path, hdu=1, decode_bytes=True, batch_size=2)
        batches = list(reader)
        assert len(batches) == 2

        dset = torchfits.table.dataset(path, hdu=1, decode_bytes=True)
        tbl = dset.to_table()
        assert tbl.num_rows == 3
        assert set(tbl.column_names) == {"RA", "ID", "NAME"}
    finally:
        os.unlink(path)


def test_arrow_scanner_filter_projection():
    ds = pytest.importorskip("pyarrow.dataset")
    path = _make_table_file()
    try:
        scanner = torchfits.table.scanner(
            path,
            hdu=1,
            decode_bytes=True,
            columns=["ID"],
            filter=ds.field("ID") >= 2,
            batch_size=2,
        )
        tbl = scanner.to_table()
        assert tbl.column_names == ["ID"]
        assert tbl.column("ID").to_pylist() == [2, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_projection_pushdown():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ID"],
            where="ID >= 2",
            backend="cpp",
        )
        assert table.column_names == ["ID"]
        assert table.column("ID").to_pylist() == [2, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_with_row_slice():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ID"],
            row_slice=slice(0, 2),
            where="ID >= 2",
            backend="cpp",
        )
        assert table.column("ID").to_pylist() == [2]
    finally:
        os.unlink(path)


def test_arrow_scan_where_batches():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        batches = list(
            torchfits.table.scan(
                path,
                hdu=1,
                columns=["ID"],
                where="ID >= 2",
                batch_size=1,
                backend="cpp",
            )
        )
        assert len(batches) == 2
        assert batches[0].column("ID").to_pylist() == [2]
        assert batches[1].column("ID").to_pylist() == [3]
    finally:
        os.unlink(path)


def test_arrow_scanner_where_projection():
    pytest.importorskip("pyarrow.dataset")
    path = _make_table_file()
    try:
        scan = torchfits.table.scanner(
            path,
            hdu=1,
            columns=["ID"],
            where="ID >= 2",
        )
        table = scan.to_table()
        assert table.column("ID").to_pylist() == [2, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_invalid_expression():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        with pytest.raises(ValueError):
            _ = torchfits.table.read(path, hdu=1, where="ID ~~ 2", backend="cpp")
    finally:
        os.unlink(path)


def test_arrow_read_where_and_or():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ID"],
            where="ID == 1 OR ID == 3",
            backend="cpp",
        )
        assert table.column("ID").to_pylist() == [1, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_parentheses_precedence():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ID"],
            where="(ID == 1 OR ID == 2) AND RA > 10.15",
            backend="cpp",
        )
        assert table.column("ID").to_pylist() == [2]
    finally:
        os.unlink(path)


def test_arrow_read_where_unbalanced_parentheses():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        with pytest.raises(ValueError):
            _ = torchfits.table.read(
                path, hdu=1, where="(ID == 1 OR ID == 2", backend="cpp"
            )
    finally:
        os.unlink(path)


def test_arrow_read_where_not_clause():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ID"],
            where="NOT (ID == 2)",
            backend="cpp",
        )
        assert table.column("ID").to_pylist() == [1, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_not_precedence():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ID"],
            where="NOT ID == 1 AND ID <= 3",
            backend="cpp",
        )
        assert table.column("ID").to_pylist() == [2, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_trailing_not_invalid():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        with pytest.raises(ValueError):
            _ = torchfits.table.read(
                path, hdu=1, where="ID == 1 AND NOT", backend="cpp"
            )
    finally:
        os.unlink(path)


def test_arrow_read_where_in_list():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ID"],
            where="ID IN (1, 3)",
            backend="cpp",
        )
        assert table.column("ID").to_pylist() == [1, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_not_in_list():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ID"],
            where="ID NOT IN (2)",
            backend="cpp",
        )
        assert table.column("ID").to_pylist() == [1, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_in_with_strings_and_or():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ID"],
            where="NAME IN ('STAR_A', 'STAR_C') OR ID == 2",
            backend="cpp",
        )
        assert table.column("ID").to_pylist() == [1, 2, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_in_missing_parenthesis_invalid():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        with pytest.raises(ValueError):
            _ = torchfits.table.read(path, hdu=1, where="ID IN (1, 2", backend="cpp")
    finally:
        os.unlink(path)


def test_arrow_read_where_between():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ID"],
            where="ID BETWEEN 2 AND 3",
            backend="cpp",
        )
        assert table.column("ID").to_pylist() == [2, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_not_between():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["ID"],
            where="ID NOT BETWEEN 2 AND 2",
            backend="cpp",
        )
        assert table.column("ID").to_pylist() == [1, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_is_null():
    pytest.importorskip("pyarrow")
    path = _make_tnull_table_file(vector=False)
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["A"],
            where="A IS NULL",
            backend="cpp",
            apply_fits_nulls=True,
        )
        assert table.column("A").to_pylist() == [None]
    finally:
        os.unlink(path)


def test_arrow_read_where_is_not_null():
    pytest.importorskip("pyarrow")
    path = _make_tnull_table_file(vector=False)
    try:
        table = torchfits.table.read(
            path,
            hdu=1,
            columns=["A"],
            where="A IS NOT NULL",
            backend="cpp",
            apply_fits_nulls=True,
        )
        assert table.column("A").to_pylist() == [1, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_between_missing_and_invalid():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        with pytest.raises(ValueError):
            _ = torchfits.table.read(path, hdu=1, where="ID BETWEEN 1 3", backend="cpp")
    finally:
        os.unlink(path)


def test_scan_where_matches_python_filter():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        where = "ID >= 2 AND RA < 10.3"
        full = torchfits.table.read(path, hdu=1, decode_bytes=True, backend="cpp")
        ids = full.column("ID").to_pylist()
        ras = full.column("RA").to_pylist()
        expected = [i for i, r in zip(ids, ras) if i >= 2 and r < 10.3]

        batches = list(
            torchfits.table.scan(
                path,
                hdu=1,
                where=where,
                decode_bytes=True,
                backend="cpp",
                batch_size=1,
            )
        )
        got = []
        for batch in batches:
            got.extend(batch.column("ID").to_pylist())
        assert got == expected
    finally:
        os.unlink(path)


def test_scan_where_with_projection():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        batches = list(
            torchfits.table.scan(
                path,
                hdu=1,
                where="ID IN (1, 3)",
                columns=["ID"],
                decode_bytes=True,
                backend="cpp",
                batch_size=2,
            )
        )
        assert batches
        ids = []
        for batch in batches:
            assert batch.schema.names == ["ID"]
            ids.extend(batch.column("ID").to_pylist())
        assert ids == [1, 3]
    finally:
        os.unlink(path)


def test_dataset_path_uses_reader_not_read(monkeypatch):
    pytest.importorskip("pyarrow.dataset")
    path = _make_table_file()
    try:
        monkeypatch.setattr(
            torchfits.table,
            "read",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("read called")),
        )
        dset = torchfits.table.dataset(path, hdu=1, decode_bytes=True, batch_size=2)
        tbl = dset.to_table()
        assert tbl.num_rows == 3
    finally:
        os.unlink(path)


def test_arrow_write_parquet_stream():
    pq = pytest.importorskip("pyarrow.parquet")
    path = _make_table_file()
    out = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    out.close()
    try:
        r = torchfits.table.reader(path, hdu=1, decode_bytes=True, batch_size=2)
        torchfits.table.write_parquet(out.name, r, stream=True)
        tbl = pq.read_table(out.name)
        assert tbl.num_rows == 3
        assert tbl.column("ID").to_pylist() == [1, 2, 3]
    finally:
        os.unlink(path)
        os.unlink(out.name)


def test_arrow_bytes_without_decode_are_fixed_binary():
    pa = pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(path, hdu=1, decode_bytes=False, backend="cpp")
        assert table.num_rows == 3
        assert pa.types.is_fixed_size_binary(table.schema.field("NAME").type)
    finally:
        os.unlink(path)


def test_to_pandas_path_accepts_io_kwargs():
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    path = _make_table_file()
    try:
        df = torchfits.table.to_pandas(
            path,
            row_slice=slice(1, 3),
            decode_bytes=True,
            backend="cpp",
        )
        assert df.shape[0] == 2
        assert df["ID"].tolist() == [2, 3]
    finally:
        os.unlink(path)


def test_tnull_scalar_to_arrow_nulls():
    pytest.importorskip("pyarrow")
    path = _make_tnull_table_file(vector=False)
    try:
        t_with_nulls = torchfits.table.read(
            path, hdu=1, backend="cpp", apply_fits_nulls=True
        )
        t_without_nulls = torchfits.table.read(
            path, hdu=1, backend="cpp", apply_fits_nulls=False
        )
        assert t_with_nulls.column("A").to_pylist() == [1, None, 3]
        assert t_without_nulls.column("A").to_pylist() == [1, -999, 3]
    finally:
        os.unlink(path)


def test_tnull_vector_to_arrow_nulls():
    pytest.importorskip("pyarrow")
    path = _make_tnull_table_file(vector=True)
    try:
        table = torchfits.table.read(path, hdu=1, backend="cpp", apply_fits_nulls=True)
        assert table.column("A").to_pylist() == [[1, None], [3, 4], [None, 6]]
    finally:
        os.unlink(path)


def test_scaled_scalar_column_preserves_physical_values():
    pytest.importorskip("pyarrow")
    path = _make_scaled_table_file(vector=False)
    try:
        with fits.open(path) as hdul:
            expected = hdul[1].data["A"].astype(np.float64)

        table = torchfits.table.read(path, hdu=1, backend="cpp")
        got = np.asarray(table.column("A"))
        assert got.dtype == np.float64
        assert np.allclose(got, expected)
    finally:
        os.unlink(path)


def test_scaled_vector_column_preserves_physical_values():
    pytest.importorskip("pyarrow")
    path = _make_scaled_table_file(vector=True)
    try:
        with fits.open(path) as hdul:
            expected = hdul[1].data["A"].astype(np.float64)

        table = torchfits.table.read(path, hdu=1, backend="cpp")
        got = np.asarray(table.column("A").to_pylist(), dtype=np.float64)
        assert got.shape == expected.shape
        assert np.allclose(got, expected)
    finally:
        os.unlink(path)


def test_legacy_table_read_scaled_column_is_float_physical():
    path = _make_scaled_table_file(vector=False)
    try:
        with fits.open(path) as hdul:
            expected = hdul[1].data["A"].astype(np.float64)

        result = torchfits.read(path, hdu=1)
        got = result["A"].detach().cpu().numpy()
        assert got.dtype == np.float64
        assert np.allclose(got, expected)
    finally:
        os.unlink(path)


def test_bit_and_vla_columns_readable():
    pytest.importorskip("pyarrow")
    path = _make_bit_vla_table_file()
    try:
        legacy = torchfits.read(path, hdu=1)
        assert "BITS" in legacy and "VLA" in legacy
        assert legacy["BITS"].shape == (3, 8)
        assert legacy["BITS"].dtype == torch.bool
        assert legacy["BITS"].tolist()[0] == [
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ]
        assert len(legacy["VLA"]) == 3

        table = torchfits.table.read(path, hdu=1, backend="cpp")
        assert set(table.column_names) == {"BITS", "VLA"}
        assert table.column("VLA").to_pylist() == [[1, 2], [3], [4, 5, 6]]
        bits_py = table.column("BITS").to_pylist()
        assert bits_py[0] == [
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ]
    finally:
        os.unlink(path)


def test_cpp_vla_returns_flat_tuple():
    path = _make_bit_vla_table_file()
    try:
        import torchfits.cpp as cpp

        chunk = cpp.read_fits_table_rows_numpy(path, 1, ["VLA"], 1, -1, False)
        assert "VLA" in chunk
        v = chunk["VLA"]
        assert isinstance(v, tuple)
        assert len(v) == 2
        flat, offsets = v
        assert isinstance(flat, np.ndarray)
        assert isinstance(offsets, np.ndarray)
        assert offsets.ndim == 1
        assert offsets[0] == 0
        assert offsets[-1] == len(flat)
    finally:
        os.unlink(path)


def test_read_polars_basic():
    """read_polars should return a FITSPolarsFrame with data and metadata."""
    pytest.importorskip("pyarrow")
    pl = pytest.importorskip("polars")
    path = _make_table_file()
    try:
        result = torchfits.table.read_polars(path, hdu=1, decode_bytes=True)
        assert isinstance(result, torchfits.table.FITSPolarsFrame)
        assert isinstance(result.frame, pl.DataFrame)
        assert result.height == 3
        assert result["ID"].to_list() == [1, 2, 3]
        # FITS metadata should be preserved
        assert "fits_hdu" in result.table_meta
        assert result.table_meta["fits_hdu"] == "1"
        # Column-level metadata (TFORM) should be present
        assert "RA" in result.field_meta
        assert "fits_tform" in result.field_meta["RA"]
    finally:
        os.unlink(path)


def test_read_polars_with_where():
    """read_polars should support where= filtering."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    path = _make_table_file()
    try:
        result = torchfits.table.read_polars(
            path, hdu=1, columns=["ID"], where="ID >= 2", decode_bytes=True
        )
        assert result.height == 2
        assert result["ID"].to_list() == [2, 3]
    finally:
        os.unlink(path)


def test_read_polars_with_columns():
    """read_polars should support column projection."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    path = _make_table_file()
    try:
        result = torchfits.table.read_polars(
            path, hdu=1, columns=["ID", "RA"], decode_bytes=True
        )
        assert set(result.columns) == {"ID", "RA"}
        # Metadata should only include selected columns
        assert "ID" in result.field_meta
        assert "NAME" not in result.field_meta
    finally:
        os.unlink(path)


def test_read_polars_attribute_delegation():
    """FITSPolarsFrame should delegate attribute access to the wrapped DataFrame."""
    pytest.importorskip("pyarrow")
    pl = pytest.importorskip("polars")
    path = _make_table_file()
    try:
        result = torchfits.table.read_polars(path, hdu=1, decode_bytes=True)
        # Delegated attributes
        assert result.height == 3
        assert result.width == 3
        assert result.columns == ["RA", "ID", "NAME"]
        # Delegated methods
        filtered = result.filter(pl.col("ID") >= 2)
        assert filtered.height == 2
        # __len__ delegation
        assert len(result) == 3
    finally:
        os.unlink(path)


def test_read_polars_rechunk_true():
    """read_polars with rechunk=True should still work correctly."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    path = _make_table_file()
    try:
        result = torchfits.table.read_polars(
            path, hdu=1, decode_bytes=True, rechunk=True
        )
        assert result.height == 3
        assert result["ID"].to_list() == [1, 2, 3]
        # Metadata should still be preserved with rechunk=True
        assert "fits_hdu" in result.table_meta
    finally:
        os.unlink(path)


def test_read_polars_import_error(monkeypatch):
    """read_polars should raise ImportError when polars is not installed."""
    pytest.importorskip("pyarrow")
    import sys

    path = _make_table_file()
    try:
        monkeypatch.setitem(sys.modules, "polars", None)
        with pytest.raises(ImportError, match="polars is required"):
            torchfits.table.read_polars(path, hdu=1)
    finally:
        os.unlink(path)


def test_fitspolarsframe_repr():
    """FITSPolarsFrame repr should include metadata when present."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    path = _make_table_file()
    try:
        result = torchfits.table.read_polars(path, hdu=1, decode_bytes=True)
        repr_str = repr(result)
        assert "FITSPolarsFrame" in repr_str
        assert "field_meta" in repr_str
        assert "table_meta" in repr_str
    finally:
        os.unlink(path)


def test_to_polars_rechunk_false_default():
    """to_polars should pass rechunk=False to pl.from_arrow by default."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    path = _make_table_file()
    try:
        df = torchfits.table.to_polars(path, hdu=1, decode_bytes=True)
        assert df.height == 3
        assert df["ID"].to_list() == [1, 2, 3]
        # With rechunk=False the chunked array keeps original chunk structure.
        # Verify data correctness regardless of chunking.
        assert df["RA"].to_list() == [10.1, 10.2, 10.3]
    finally:
        os.unlink(path)


def test_to_polars_stream_rechunk_false():
    """to_polars with stream=True should also use rechunk=False."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    path = _make_table_file()
    try:
        frames = list(
            torchfits.table.to_polars(
                path, hdu=1, batch_size=2, decode_bytes=True, stream=True
            )
        )
        assert len(frames) == 2
        all_ids = []
        for frame in frames:
            all_ids.extend(frame["ID"].to_list())
        assert all_ids == [1, 2, 3]
    finally:
        os.unlink(path)


def test_to_polars_rechunk_true_explicit():
    """to_polars with rechunk=True should still work correctly."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    path = _make_table_file()
    try:
        df = torchfits.table.to_polars(path, hdu=1, decode_bytes=True, rechunk=True)
        assert df.height == 3
        assert df["ID"].to_list() == [1, 2, 3]
    finally:
        os.unlink(path)


def test_scan_polars_basic():
    """scan_polars should yield pl.DataFrame batches without full materialization."""
    pytest.importorskip("pyarrow")
    pl = pytest.importorskip("polars")
    path = _make_table_file()
    try:
        frames = list(
            torchfits.table.scan_polars(path, hdu=1, batch_size=2, decode_bytes=True)
        )
        assert len(frames) == 2
        assert isinstance(frames[0], pl.DataFrame)
        assert frames[0].height == 2
        assert frames[1].height == 1

        all_ids: list[int] = []
        for frame in frames:
            all_ids.extend(frame["ID"].to_list())
        assert all_ids == [1, 2, 3]
    finally:
        os.unlink(path)


def test_scan_polars_with_where():
    """scan_polars should support where= filtering."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    path = _make_table_file()
    try:
        frames = list(
            torchfits.table.scan_polars(
                path, hdu=1, batch_size=10, where="ID >= 2", decode_bytes=True
            )
        )
        all_ids: list[int] = []
        for frame in frames:
            all_ids.extend(frame["ID"].to_list())
        assert all_ids == [2, 3]
    finally:
        os.unlink(path)


def test_scan_polars_with_columns():
    """scan_polars should support column projection."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    path = _make_table_file()
    try:
        frames = list(
            torchfits.table.scan_polars(
                path, hdu=1, batch_size=10, columns=["ID"], decode_bytes=True
            )
        )
        assert frames
        assert set(frames[0].columns) == {"ID"}
    finally:
        os.unlink(path)


def test_scan_polars_import_error(monkeypatch):
    """scan_polars should raise ImportError when polars is not installed."""
    pytest.importorskip("pyarrow")
    import sys

    path = _make_table_file()
    try:
        monkeypatch.setitem(sys.modules, "polars", None)
        with pytest.raises(ImportError, match="polars is required"):
            next(torchfits.table.scan_polars(path, hdu=1))
    finally:
        os.unlink(path)


def test_duckdb_query_on_fits_table():
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    path = _make_table_file()
    try:
        result = torchfits.table.duckdb_query(
            path,
            "SELECT COUNT(*) AS n FROM fits_table WHERE ID >= 2",
            hdu=1,
            decode_bytes=True,
        )
        if hasattr(result, "read_all"):
            table = result.read_all()
        elif hasattr(result, "to_table"):
            table = result.to_table()
        else:
            table = result
        assert table.column("n").to_pylist() == [2]
    finally:
        os.unlink(path)


def test_to_duckdb():
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")

    data = pa.table({"ID": [1, 2, 3], "RA": [10.1, 10.2, 10.3]})

    rel = torchfits.table.to_duckdb(data)
    res = rel.filter("ID >= 2").aggregate("COUNT(*) AS n")

    if hasattr(res, "arrow"):
        table = res.arrow()
    elif hasattr(res, "to_arrow_table"):
        table = res.to_arrow_table()
    else:
        table = res.execute().fetch_arrow_table()

    if hasattr(table, "read_all"):
        table = table.read_all()

    assert table.column("n").to_pylist() == [2]


def test_to_duckdb_custom_table_name():
    pa = pytest.importorskip("pyarrow")
    duckdb = pytest.importorskip("duckdb")

    data = pa.table({"ID": [1, 2, 3], "RA": [10.1, 10.2, 10.3]})

    con = duckdb.connect()
    torchfits.table.to_duckdb(data, "my_custom_table", con)
    res = con.sql("SELECT COUNT(*) AS n FROM my_custom_table WHERE ID >= 2")

    if hasattr(res, "arrow"):
        table = res.arrow()
    elif hasattr(res, "to_arrow_table"):
        table = res.to_arrow_table()
    else:
        table = res.execute().fetch_arrow_table()

    if hasattr(table, "read_all"):
        table = table.read_all()

    assert table.column("n").to_pylist() == [2]


def test_to_duckdb_missing_dependency(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "duckdb", None)

    with pytest.raises(
        ImportError, match="duckdb is required for to_duckdb conversion"
    ):
        torchfits.table.to_duckdb({"ID": [1, 2, 3]})


def test_row_slice_negative_stop_raises():
    """Negative stop in row_slice must raise (total row count unknown at parse time)."""
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        with pytest.raises(ValueError, match="negative stop"):
            torchfits.table.read(path, hdu=1, row_slice=slice(0, -1))
        with pytest.raises(ValueError, match="negative stop"):
            torchfits.table.read(path, hdu=1, row_slice=slice(1, -1))
    finally:
        os.unlink(path)


def test_empty_where_preserves_schema():
    """Empty WHERE result must preserve column names (no KeyError on column access)."""
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(path, hdu=1, where="ID > 9999", backend="cpp")
        assert table.num_rows == 0
        assert set(table.column_names) == {"RA", "ID", "NAME"}
        # Accessing a column must not raise KeyError.
        assert table.column("ID").to_pylist() == []
    finally:
        os.unlink(path)


def test_empty_where_torch_path_preserves_schema():
    """Empty WHERE via the torch-tensor filter path must also preserve schema."""
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(path, hdu=1, where="ID > 9999", backend="torch")
        assert table.num_rows == 0
        assert set(table.column_names) == {"RA", "ID", "NAME"}
    finally:
        os.unlink(path)


def test_empty_where_with_projection_preserves_schema():
    """Empty WHERE with column projection must preserve selected column names."""
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(
            path, hdu=1, columns=["ID", "RA"], where="ID > 9999", backend="cpp"
        )
        assert table.num_rows == 0
        assert table.column_names == ["ID", "RA"]
    finally:
        os.unlink(path)


def test_empty_row_slice_preserves_schema():
    """row_slice selecting 0 rows must preserve column names (e.g. slice(0,0))."""
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(path, hdu=1, row_slice=slice(0, 0), backend="cpp")
        assert table.num_rows == 0
        assert set(table.column_names) == {"RA", "ID", "NAME"}
    finally:
        os.unlink(path)


def test_empty_rows_preserves_schema():
    """Empty rows=[] must preserve column names."""
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        table = torchfits.table.read(path, hdu=1, rows=[], backend="cpp")
        assert table.num_rows == 0
        assert set(table.column_names) == {"RA", "ID", "NAME"}
    finally:
        os.unlink(path)


def test_chunk_repeat1_vector_surfaces_scalar():
    """Repeat-1 TFORM chunks surface Arrow scalars, matching read() (r5c-03)."""
    pa = pytest.importorskip("pyarrow")

    from torchfits._table.arrow_convert import _chunk_to_record_batch

    values = np.arange(6, dtype=np.int32).reshape(3, 2)[:, :1]  # (3, 1)
    for tform in ("1J", "J"):
        batch = _chunk_to_record_batch(
            {"A": values},
            False,
            "ascii",
            True,
            field_meta={"A": {"fits_tform": tform}},
        )
        assert batch.schema.field("A").type == pa.int32(), tform
        assert batch.column("A").to_pylist() == [0, 2, 4]

    # width-2 vectors keep their fixed-size-list type
    batch = _chunk_to_record_batch(
        {"A": np.arange(6, dtype=np.int32).reshape(3, 2)},
        False,
        "ascii",
        True,
        field_meta={"A": {"fits_tform": "2J"}},
    )
    assert pa.types.is_fixed_size_list(batch.schema.field("A").type)


def test_chunk_repeat1_tensor_surfaces_scalar():
    """The torch tensor route agrees with the numpy route (r5c-03)."""
    pa = pytest.importorskip("pyarrow")

    from torchfits._table.arrow_convert import _chunk_to_record_batch

    batch = _chunk_to_record_batch(
        {"A": torch.arange(3, dtype=torch.int32).reshape(3, 1)},
        False,
        "ascii",
        True,
        field_meta={"A": {"fits_tform": "1J"}},
    )
    assert batch.schema.field("A").type == pa.int32()
    assert batch.column("A").to_pylist() == [0, 1, 2]


def test_chunk_repeat1_bit_surfaces_scalar_bool():
    """A 1-bit column is a scalar boolean, not FixedSizeList<bool>[1] (r5c-03)."""
    pa = pytest.importorskip("pyarrow")

    from torchfits._table.arrow_convert import _chunk_to_record_batch

    batch = _chunk_to_record_batch(
        {"B": np.zeros((3, 1), np.uint8)},
        False,
        "ascii",
        True,
        field_meta={"B": {"fits_tform": "1X"}},
    )
    assert batch.schema.field("B").type == pa.bool_()
    # multi-bit columns keep the fixed-size-list-of-bool mapping
    batch = _chunk_to_record_batch(
        {"B": np.zeros((3, 8), np.uint8)},
        False,
        "ascii",
        True,
        field_meta={"B": {"fits_tform": "8X"}},
    )
    assert pa.types.is_fixed_size_list(batch.schema.field("B").type)


def test_width1_fixed_list_to_astropy_is_scalar():
    """Arrow width-1 lists surface as scalars on the way back out (r5c-03)."""
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("astropy")

    from torchfits.table import to_astropy

    arr = pa.array([[1], [2], [3]], type=pa.list_(pa.int32(), 1))
    col = to_astropy(pa.table({"w": arr}))["w"]
    assert np.shape(col) == (3,), np.shape(col)
    assert col.dtype.kind == "i"
    assert np.asarray(col).tolist() == [1, 2, 3]


def test_repeat1_round_trip_agrees_with_astropy(tmp_path):
    """(N, 1) + TFORM 1J round-trips to (N,) exactly like astropy (r5c-03)."""
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("astropy")
    from astropy.io import fits as afits
    from astropy.table import Table as AstropyTable

    from torchfits._table.arrow_convert import _chunk_to_record_batch
    from torchfits.table import to_astropy

    path = tmp_path / "r1j.fits"
    afits.BinTableHDU.from_columns(
        [
            afits.Column(
                name="S", format="1J", array=np.arange(3, dtype="<i4").reshape(3, 1)
            )
        ]
    ).writeto(str(path), overwrite=True)
    gt = AstropyTable.read(str(path))["S"]
    assert np.shape(gt) == (3,)

    batch = _chunk_to_record_batch(
        {"S": np.arange(3, dtype=np.int32).reshape(3, 1)},
        False,
        "ascii",
        True,
        field_meta={"S": {"fits_tform": "1J"}},
    )
    col = to_astropy(batch)["S"]
    assert np.shape(col) == np.shape(gt) == (3,)
    assert np.asarray(col).tolist() == np.asarray(gt).tolist()


def test_byte_vector_column_is_uint8_fsl(tmp_path):
    """TFORM 'B' byte vectors map to FixedSizeList<uint8>[w], never strings.

    r5c-14 (BLOCKER): the data path must agree with table.schema() and
    read_torch on every decode/mmap combination — decoding byte vectors as
    text silently corrupted every value.
    """
    pa = pytest.importorskip("pyarrow")
    from astropy.io import fits as afits

    import torchfits

    path = tmp_path / "bytevec.fits"
    values = np.arange(24, dtype=np.uint8).reshape(3, 8)
    afits.BinTableHDU.from_columns(
        [afits.Column(name="BV", format="8B", array=values)]
    ).writeto(str(path), overwrite=True)

    for mmap in (True, False):
        for decode_bytes in (True, False):
            tbl = torchfits.table.read(str(path), decode_bytes=decode_bytes, mmap=mmap)
            col = tbl["BV"]
            assert pa.types.is_fixed_size_list(col.type), (mmap, decode_bytes, col.type)
            assert col.type.list_size == 8
            assert col.type.value_type == pa.uint8()
            assert col.to_pylist() == values.tolist(), (mmap, decode_bytes)

    rt = torchfits.table.read_torch(str(path))
    assert tuple(rt["BV"].shape) == (3, 8)
    assert rt["BV"].dtype == torch.uint8
    assert rt["BV"].numpy().tolist() == values.tolist()


def test_chunk_byte_vector_vs_char_dispatch():
    """Only TFORM 'A' matrices take the string/bytes path; 'B' vectors stay
    byte vectors (r5c-14)."""
    pa = pytest.importorskip("pyarrow")

    from torchfits._table.arrow_convert import _chunk_to_record_batch

    values = np.arange(24, dtype=np.uint8).reshape(3, 8)
    for decode in (True, False):
        batch = _chunk_to_record_batch(
            {"BV": values},
            decode,
            "ascii",
            True,
            column_tforms={"BV": "8B"},
        )
        field = batch.schema.field("BV")
        assert pa.types.is_fixed_size_list(field.type), (decode, field.type)
        assert field.type.value_type == pa.uint8()
        assert batch.column("BV").to_pylist() == values.tolist(), decode

    # char matrices keep their documented decode contract
    chars = np.array([[65, 66, 67, 68], [69, 70, 71, 72]], dtype=np.uint8)
    decoded = _chunk_to_record_batch(
        {"S": chars}, True, "ascii", True, column_tforms={"S": "4A"}
    )
    assert pa.types.is_string(decoded.schema.field("S").type)
    assert decoded.column("S").to_pylist() == ["ABCD", "EFGH"]
    raw = _chunk_to_record_batch(
        {"S": chars}, False, "ascii", True, column_tforms={"S": "4A"}
    )
    assert pa.types.is_fixed_size_binary(raw.schema.field("S").type)
