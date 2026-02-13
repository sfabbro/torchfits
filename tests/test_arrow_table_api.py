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


def test_arrow_cpp_numpy_backend_matches_default():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        import torchfits.cpp as cpp

        if not hasattr(cpp, "read_fits_table_rows_numpy_from_handle"):
            pytest.skip("cpp numpy backend not available")

        t_default = torchfits.table.read(
            path, hdu=1, decode_bytes=True, backend="torch"
        )
        t_cpp = torchfits.table.read(
            path, hdu=1, decode_bytes=True, backend="cpp_numpy"
        )

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
            path, hdu=1, columns=["RA", "ID"], include_fits_metadata=True
        )
        assert sch.names == ["RA", "ID"]
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
            backend="cpp_numpy",
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
            backend="cpp_numpy",
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
                backend="cpp_numpy",
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
            _ = torchfits.table.read(path, hdu=1, where="ID ~~ 2", backend="cpp_numpy")
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
            backend="cpp_numpy",
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
            backend="cpp_numpy",
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
                path, hdu=1, where="(ID == 1 OR ID == 2", backend="cpp_numpy"
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
            backend="cpp_numpy",
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
            backend="cpp_numpy",
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
                path, hdu=1, where="ID == 1 AND NOT", backend="cpp_numpy"
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
            backend="cpp_numpy",
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
            backend="cpp_numpy",
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
            backend="cpp_numpy",
        )
        assert table.column("ID").to_pylist() == [1, 2, 3]
    finally:
        os.unlink(path)


def test_arrow_read_where_in_missing_parenthesis_invalid():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        with pytest.raises(ValueError):
            _ = torchfits.table.read(
                path, hdu=1, where="ID IN (1, 2", backend="cpp_numpy"
            )
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
            backend="cpp_numpy",
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
            backend="cpp_numpy",
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
            backend="cpp_numpy",
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
            backend="cpp_numpy",
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
            _ = torchfits.table.read(
                path, hdu=1, where="ID BETWEEN 1 3", backend="cpp_numpy"
            )
    finally:
        os.unlink(path)


def test_scan_where_matches_python_filter():
    pytest.importorskip("pyarrow")
    path = _make_table_file()
    try:
        where = "ID >= 2 AND RA < 10.3"
        full = torchfits.table.read(path, hdu=1, decode_bytes=True, backend="cpp_numpy")
        ids = full.column("ID").to_pylist()
        ras = full.column("RA").to_pylist()
        expected = [i for i, r in zip(ids, ras) if i >= 2 and r < 10.3]

        batches = list(
            torchfits.table.scan(
                path,
                hdu=1,
                where=where,
                decode_bytes=True,
                backend="cpp_numpy",
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
                backend="cpp_numpy",
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
        table = torchfits.table.read(
            path, hdu=1, decode_bytes=False, backend="cpp_numpy"
        )
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
            backend="cpp_numpy",
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
            path, hdu=1, backend="cpp_numpy", apply_fits_nulls=True
        )
        t_without_nulls = torchfits.table.read(
            path, hdu=1, backend="cpp_numpy", apply_fits_nulls=False
        )
        assert t_with_nulls.column("A").to_pylist() == [1, None, 3]
        assert t_without_nulls.column("A").to_pylist() == [1, -999, 3]
    finally:
        os.unlink(path)


def test_tnull_vector_to_arrow_nulls():
    pytest.importorskip("pyarrow")
    path = _make_tnull_table_file(vector=True)
    try:
        table = torchfits.table.read(
            path, hdu=1, backend="cpp_numpy", apply_fits_nulls=True
        )
        assert table.column("A").to_pylist() == [[1, None], [3, 4], [None, 6]]
    finally:
        os.unlink(path)


def test_scaled_scalar_column_preserves_physical_values():
    pytest.importorskip("pyarrow")
    path = _make_scaled_table_file(vector=False)
    try:
        with fits.open(path) as hdul:
            expected = hdul[1].data["A"].astype(np.float64)

        table = torchfits.table.read(path, hdu=1, backend="cpp_numpy")
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

        table = torchfits.table.read(path, hdu=1, backend="cpp_numpy")
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
        assert legacy["BITS"].dtype == torch.uint8
        assert len(legacy["VLA"]) == 3

        table = torchfits.table.read(path, hdu=1, backend="cpp_numpy")
        assert set(table.column_names) == {"BITS", "VLA"}
        assert table.column("VLA").to_pylist() == [[1, 2], [3], [4, 5, 6]]
        bits_py = table.column("BITS").to_pylist()
        assert bits_py[0] == b"\x01\x00\x01\x00\x01\x00\x01\x00"
    finally:
        os.unlink(path)


def test_cpp_numpy_vla_returns_flat_tuple():
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


def test_to_polars_lazy_expression():
    pytest.importorskip("pyarrow")
    pl = pytest.importorskip("polars")
    path = _make_table_file()
    try:
        lf = torchfits.table.to_polars_lazy(path, hdu=1, decode_bytes=True)
        out = lf.filter(pl.col("ID") >= 2).select(pl.col("ID")).collect()
        assert out["ID"].to_list() == [2, 3]
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
