"""
Example: Polars integration via torchfits.table.to_polars, read_polars, and scan_polars.
"""

import os
import tempfile
import time

import numpy as np

import torchfits


def _create_catalog(path: str, n_rows: int = 50_000) -> None:
    torchfits.table.write(
        path,
        {
            "id": np.arange(n_rows, dtype=np.int64),
            "ra": np.random.uniform(0, 360, n_rows).astype(np.float64),
            "dec": np.random.uniform(-90, 90, n_rows).astype(np.float64),
            "flux_g": np.random.exponential(100.0, n_rows).astype(np.float32),
            "flux_r": np.random.exponential(150.0, n_rows).astype(np.float32),
            "class": np.random.randint(0, 3, n_rows).astype(np.int16),
        },
        overwrite=True,
    )


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

        # read_polars: one-call FITS-to-Polars with FITS metadata preserved
        t0 = time.perf_counter()
        result = torchfits.table.read_polars(
            path, hdu=1, columns=["id", "ra", "flux_g", "class"]
        )
        t1 = time.perf_counter()
        print(f"\ntable.read_polars: {result.height} rows in {t1 - t0:.3f}s")
        print(f"  FITS metadata keys: {list(result.table_meta.keys())}")
        print(f"  Column metadata (flux_g): {result.field_meta.get('flux_g', {})}")
        print(result.head(3))

        # LazyFrame for complex aggregations via scan_polars
        summary = (
            pl.concat(torchfits.table.scan_polars(path, hdu=1))
            .lazy()
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

        # Streaming Polars: process large FITS tables in batches
        # without materializing the entire table at once.
        total = 0
        for frame in torchfits.table.scan_polars(path, hdu=1, batch_size=10_000):
            total += frame.filter(pl.col("flux_g") > 500).height
        print(f"\nscan_polars streaming: {total} rows with flux_g > 500")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
