"""
Example: Polars integration via torchfits.table.to_polars and to_polars_lazy.
"""

import os
import tempfile
import time

import numpy as np
from astropy.table import Table

import torchfits


def _create_catalog(path: str, n_rows: int = 50_000) -> None:
    table = Table(
        {
            "id": np.arange(n_rows, dtype=np.int64),
            "ra": np.random.uniform(0, 360, n_rows).astype(np.float64),
            "dec": np.random.uniform(-90, 90, n_rows).astype(np.float64),
            "flux_g": np.random.exponential(100.0, n_rows).astype(np.float32),
            "flux_r": np.random.exponential(150.0, n_rows).astype(np.float32),
            "class": np.random.randint(0, 3, n_rows).astype(np.int16),
        }
    )
    table.write(path, format="fits", overwrite=True)


def main() -> None:
    try:
        import polars as pl
    except ImportError:
        print("Polars not installed; install with: pip install polars")
        return

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        _create_catalog(path, n_rows=50_000)
        print(f"Catalog: {path} ({50_000:,} rows)")

        # Direct FITS -> Polars (Arrow-backed, no manual tensor conversion)
        t0 = time.perf_counter()
        df = torchfits.table.to_polars(
            path,
            hdu=1,
            columns=["id", "ra", "dec", "flux_g", "class"],
            where="flux_g > 500",
        )
        t1 = time.perf_counter()
        print(f"\ntable.to_polars (with where=): {df.height} rows in {t1 - t0:.3f}s")
        print(df.head(3))

        # LazyFrame for complex aggregations
        summary = (
            torchfits.table.to_polars_lazy(path, hdu=1)
            .filter(pl.col("flux_g") > 500)
            .group_by("class")
            .agg(
                pl.col("ra").mean().alias("avg_ra"),
                pl.col("flux_g").max().alias("max_g"),
                pl.len().alias("count"),
            )
            .sort("class")
            .collect()
        )
        print("\nLazyFrame summary:")
        print(summary)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
