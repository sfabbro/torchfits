"""
Example: Polars Integration for Efficient Data Analysis

This example demonstrates how to use torchfits with Polars for
high-performance tabular data analysis, leveraging Apache Arrow
for zero-copy (or near-zero-copy) data transfer.
"""

import tempfile
import time
import numpy as np
import torch
import torchfits
import polars as pl


def create_sample_catalog(n_rows=100_000):
    """Create a sample astronomical catalog FITS file."""
    print(f"Creating sample catalog with {n_rows:,} rows...")

    # Create dictionary of numpy arrays
    data = {
        "id": np.arange(n_rows, dtype=np.int64),
        "ra": np.random.uniform(0, 360, n_rows).astype(np.float64),
        "dec": np.random.uniform(-90, 90, n_rows).astype(np.float64),
        "flux_g": np.random.exponential(100.0, n_rows).astype(np.float32),
        "flux_r": np.random.exponential(150.0, n_rows).astype(np.float32),
        "class": np.random.randint(0, 3, n_rows).astype(np.int16),
    }

    # Write to FITS
    filename = tempfile.mktemp(suffix=".fits")

    # Using astropy to write initial file (as torchfits write support is mainly for tensors)
    from astropy.table import Table

    t = Table(data)
    t.write(filename, format="fits", overwrite=True)

    print(f"  File: {filename}")
    return filename


def example_polars_integration():
    """Demonstrate reading FITS into Polars via torchfits."""
    filename = create_sample_catalog()

    print("\n" + "=" * 50)
    print("TorchFITS -> Polars Integration")
    print("=" * 50)

    # Measure read time
    t0 = time.time()

    # 1. Read using torchfits (returns dictionary of Tensors/Arrays)
    # limit_rows=None means read all
    data = torchfits.read(filename, hdu=1)

    t1 = time.time()
    print(f"Read time (torchfits): {t1 - t0:.4f}s")

    # 2. Convert to Polars DataFrame
    # torchfits returns torch tensors by default.
    # While Polars can accept tensors (via dlpack or numpy),
    # converting to numpy first is often robust.

    # Efficient conversion:
    # If data is on CPU, tensor.numpy() is zero-copy for contiguous arrays.
    polars_data = {}
    for col, tensor in data.items():
        if isinstance(tensor, torch.Tensor):
            polars_data[col] = tensor.numpy()
        else:
            polars_data[col] = tensor

    df = pl.DataFrame(polars_data)

    t2 = time.time()
    print(f"Conversion to Polars:  {t2 - t1:.4f}s")
    print(f"Total time:            {t2 - t0:.4f}s")

    # 3. Demonstrate Polars operations
    print("\nPolars DataFrame Schema:")
    print(df.schema)

    print("\nSample operations:")
    # Filter by flux and calculate mean coords
    result = (
        df.filter(pl.col("flux_g") > 500)
        .group_by("class")
        .agg(
            [
                pl.col("ra").mean().alias("avg_ra"),
                pl.col("flux_g").max().alias("max_g"),
                pl.len().alias("count"),
            ]
        )
        .sort("class")
    )

    print(result)


if __name__ == "__main__":
    try:
        example_polars_integration()
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback

        traceback.print_exc()
