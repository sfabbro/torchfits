"""
Example: FITS table recipes with Arrow scanner, Polars LazyFrame, and DuckDB SQL.
"""

import os
import tempfile

import numpy as np
from astropy.table import Table

import torchfits


def _write_catalog(path: str, offset: float = 0.0) -> None:
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
        # 1) Predicate + projection pushdown via table.read
        north = torchfits.table.read(
            catalog_file.name,
            hdu=1,
            columns=["OBJID", "RA", "DEC"],
            where="DEC > 0",
        )
        print(f"table.read (DEC > 0): {north.num_rows} rows")

        # 2) PyArrow Dataset scanner for composable filters
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
            scanned = scanner.to_table()
            print(f"table.scanner (DEC > 0): {scanned.num_rows} rows")

        # 3) Polars LazyFrame for aggregations
        try:
            import polars as pl
        except ImportError:
            print("Polars not installed; skipping LazyFrame recipe")
        else:
            summary = (
                torchfits.table.to_polars_lazy(
                    catalog_file.name, hdu=1, decode_bytes=True
                )
                .filter(pl.col("MAG_G").is_not_null())
                .group_by("BAND")
                .agg(
                    pl.col("MAG_G").mean().alias("mag_g_mean"),
                    pl.len().alias("n"),
                )
                .sort("n", descending=True)
                .collect()
            )
            print("Polars summary:")
            print(summary)

        # 4) DuckDB SQL joins across FITS files
        try:
            import duckdb
        except ImportError:
            print("DuckDB not installed; skipping SQL recipe")
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
            # DuckDB ≥1.5 + PyArrow ≥24 may return a RecordBatchReader instead of Table.
            if hasattr(joined, "read_all"):
                joined = joined.read_all()
            print(f"DuckDB join (DEC > 0): {joined.num_rows} rows")
    finally:
        os.unlink(catalog_file.name)
        os.unlink(labels_file.name)


if __name__ == "__main__":
    main()
