from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

fitsio = pytest.importorskip("fitsio")

import torchfits  # noqa: E402


def _table_data() -> dict[str, np.ndarray]:
    return {
        "INDEX": np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        "X": np.array([0.5, 4.0, 10.0, 2.0, 9.0, 7.0], dtype=np.float64),
        "Y": np.array([10.0, 7.0, 3.0, 12.0, 4.0, 1.0], dtype=np.float64),
        "FLAG": np.array([True, False, True, False, True, False], dtype=np.bool_),
    }


def test_fitsio_readme_subset_and_where_workflows_match_torchfits() -> None:
    table = _table_data()
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)

    try:
        torchfits.write(path.as_posix(), table, overwrite=True)

        rows = [1, 4]
        cols = ["INDEX", "X", "Y"]
        tf_subset = torchfits.table.read(
            path.as_posix(), hdu=1, rows=rows, columns=cols
        )
        fits_subset = fitsio.read(path.as_posix(), ext=1, rows=rows, columns=cols)
        for name in cols:
            np.testing.assert_allclose(
                np.asarray(tf_subset.column(name).to_pylist()),
                fits_subset[name],
                atol=0.0,
                rtol=0.0,
            )

        expr = "X > 3 && Y < 8"
        tf_filtered = torchfits.table.read(
            path.as_posix(), hdu=1, columns=cols, where=expr
        )
        with fitsio.FITS(path.as_posix()) as fits:
            where_rows = fits[1].where(expr)
            fits_filtered = fits[1].read(rows=where_rows, columns=cols)
        for name in cols:
            np.testing.assert_allclose(
                np.asarray(tf_filtered.column(name).to_pylist()),
                fits_filtered[name],
                atol=0.0,
                rtol=0.0,
            )
    finally:
        path.unlink(missing_ok=True)


def test_fitsio_readme_table_mutation_header_and_checksum_workflows() -> None:
    table = _table_data()
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)

    try:
        torchfits.write(
            path.as_posix(),
            table,
            header={"OBSERVER": "TORCHFITS", "EXTNAME": "CATALOG"},
            overwrite=True,
        )

        torchfits.table.append_rows(
            path.as_posix(),
            {
                "INDEX": np.array([6], dtype=np.int32),
                "X": np.array([11.0], dtype=np.float64),
                "Y": np.array([0.5], dtype=np.float64),
                "FLAG": np.array([True], dtype=np.bool_),
            },
            hdu=1,
        )
        torchfits.table.insert_rows(
            path.as_posix(),
            {
                "INDEX": np.array([99], dtype=np.int32),
                "X": np.array([1.25], dtype=np.float64),
                "Y": np.array([1.5], dtype=np.float64),
                "FLAG": np.array([False], dtype=np.bool_),
            },
            row=2,
            hdu=1,
        )
        torchfits.table.update_rows(
            path.as_posix(),
            {"Y": np.array([42.0, 43.0], dtype=np.float64)},
            row_slice=slice(0, 2),
            hdu=1,
        )
        torchfits.table.rename_columns(path.as_posix(), {"X": "FLUX"}, hdu=1)
        torchfits.table.drop_columns(path.as_posix(), ["FLAG"], hdu=1)
        torchfits.write_checksums(path.as_posix(), hdu=1)

        status = torchfits.verify_checksums(path.as_posix(), hdu=1)
        assert status["ok"]

        data = fitsio.read(path.as_posix(), ext=1)
        header = fitsio.read_header(path.as_posix(), ext=1)
        assert header["OBSERVER"] == "TORCHFITS"
        assert header["EXTNAME"] == "CATALOG"
        assert data.dtype.names == ("INDEX", "FLUX", "Y")
        np.testing.assert_array_equal(
            data["INDEX"], np.array([0, 1, 99, 2, 3, 4, 5, 6])
        )
        np.testing.assert_allclose(
            data["FLUX"], np.array([0.5, 4.0, 1.25, 10.0, 2.0, 9.0, 7.0, 11.0])
        )
        np.testing.assert_allclose(
            data["Y"], np.array([42.0, 43.0, 1.5, 3.0, 12.0, 4.0, 1.0, 0.5])
        )
        assert "CHECKSUM" in header
        assert "DATASUM" in header
    finally:
        path.unlink(missing_ok=True)


def test_fitsio_image_compression_and_slice_workflows_match_torchfits() -> None:
    image = np.arange(64, dtype=np.int16).reshape(8, 8)
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)

    try:
        torchfits.write(
            path.as_posix(), torch.from_numpy(image), overwrite=True, compress=True
        )

        full_tf = torchfits.read(path.as_posix(), hdu=1).cpu().numpy()
        full_fitsio = fitsio.read(path.as_posix(), ext=1)
        np.testing.assert_array_equal(full_tf, image)
        np.testing.assert_array_equal(full_fitsio, image)

        with fitsio.FITS(path.as_posix()) as fits:
            subset = fits[1][2:5, 3:7]
        np.testing.assert_array_equal(subset, image[2:5, 3:7])
    finally:
        path.unlink(missing_ok=True)
