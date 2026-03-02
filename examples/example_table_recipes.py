"""
Example: Recommended FITS table workflow using Arrow scanner, Polars, and DuckDB.
"""

import os
import tempfile

import numpy as np
from astropy.table import Table

import torchfits


def _write_catalog(path: str, offset: int = 0) -> None:
    table = Table(
        {
            "OBJID": np.array([1, 2, 3, 4], dtype=np.int64),
            "RA": np.array([10.1, 11.2, 12.3, 13.4], dtype=np.float64),
            "DEC": np.array([-1.0, 0.5, 1.2, 2.1], dtype=np.float64),
            "MAG_G": np.array([20.1, 19.4, 18.9, 21.0], dtype=np.float32) + offset,
            "BAND": np.array(["g", "g", "r", "r"], dtype="U1"),
        }
    )
    table.write(path, format="fits", overwrite=True)


def _write_labels(path: str) -> None:
    table = Table(
        {
            "OBJID": np.array([1, 2, 3, 4], dtype=np.int64),
            "CLASS": np.array(["star", "galaxy", "qso", "star"], dtype="U8"),
        }
    )
    table.write(path, format="fits", overwrite=True)


def main() -> None:
    catalog_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    labels_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    catalog_file.close()
    labels_file.close()
    _write_catalog(catalog_file.name)
    _write_labels(labels_file.name)

    try:
        # 1) Projection/filter pushdown with Arrow scanner
        try:
            import pyarrow.dataset as ds
        except ImportError:
            print("pyarrow.dataset not installed; skipping scanner recipe")
        else:
            scanner = torchfits.table.scanner(
                catalog_file.name,
                hdu=1,
                columns=["OBJID", "RA", "DEC"],
                filter=ds.field("DEC") > 0,
            )
            north = scanner.to_table()
            print("Scanner rows with DEC > 0:", north.num_rows)

        # 2) Complex expressions with Polars LazyFrame
        try:
            import polars as pl
        except ImportError:
            print("polars not installed; skipping LazyFrame recipe")
        else:
            lf = torchfits.table.to_polars_lazy(
                catalog_file.name, hdu=1, decode_bytes=True
            )
            summary = (
                lf.filter(pl.col("MAG_G").is_not_null())
                .group_by("BAND")
                .agg(pl.col("MAG_G").mean().alias("mag_g_mean"), pl.len().alias("n"))
                .sort("n", descending=True)
                .collect()
            )
            print("Polars summary:")
            print(summary)

        # 3) SQL joins with DuckDB
        try:
            import duckdb
        except ImportError:
            print("duckdb not installed; skipping SQL recipe")
        else:
            con = duckdb.connect()
            torchfits.table.to_duckdb(
                catalog_file.name, hdu=1, relation_name="catalog", connection=con
            )
            torchfits.table.to_duckdb(
                labels_file.name, hdu=1, relation_name="labels", connection=con
            )
            joined = con.sql(
                """
                SELECT c.OBJID, c.RA, l.CLASS
                FROM catalog c
                JOIN labels l USING (OBJID)
                WHERE c.DEC > 0
                """
            ).arrow()
            print("DuckDB join rows:", joined.num_rows)
    finally:
        os.unlink(catalog_file.name)
        os.unlink(labels_file.name)


if __name__ == "__main__":
    main()
