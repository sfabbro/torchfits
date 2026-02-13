import os
import tempfile

import numpy as np
import pytest
from astropy.table import Table

import torchfits


def _make_table_file() -> str:
    table = Table(
        {
            "RA": np.array([10.1, 10.2, 10.3], dtype=np.float64),
            "DEC": np.array([-1.0, 0.5, 1.5], dtype=np.float64),
            "ID": np.array([1, 2, 3], dtype=np.int64),
            "NAME": np.array(["STAR_A", "STAR_B", "STAR_C"], dtype="U8"),
        }
    )
    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    table.write(handle.name, format="fits", overwrite=True)
    return handle.name


def test_table_docs_workflow_smoke():
    pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.dataset")

    path = _make_table_file()
    try:
        # read + where + projection
        subset = torchfits.table.read(
            path,
            hdu=1,
            columns=["ID", "RA"],
            where="DEC > 0",
            decode_bytes=True,
            backend="cpp_numpy",
        )
        assert subset.column_names == ["ID", "RA"]
        assert subset.column("ID").to_pylist() == [2, 3]

        # scan batches with predicate
        ids = []
        for batch in torchfits.table.scan(
            path,
            hdu=1,
            columns=["ID"],
            where="ID >= 2",
            batch_size=1,
            backend="cpp_numpy",
        ):
            ids.extend(batch.column("ID").to_pylist())
        assert ids == [2, 3]

        # reader + scanner flow
        reader = torchfits.table.reader(
            path,
            hdu=1,
            columns=["ID"],
            where="ID IN (1, 3)",
            backend="cpp_numpy",
        )
        from_reader = []
        for batch in reader:
            from_reader.extend(batch.column("ID").to_pylist())
        assert from_reader == [1, 3]

        scanner = torchfits.table.scanner(
            path,
            hdu=1,
            columns=["ID"],
            where="ID NOT IN (2)",
        )
        scanned = scanner.to_table()
        assert scanned.column("ID").to_pylist() == [1, 3]
    finally:
        os.unlink(path)
