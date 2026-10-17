"""Example: torchfits.data table and cutout datasets."""

from __future__ import annotations

import os
import tempfile

import numpy as np

import torchfits
from torchfits.data import (
    FitsCutoutDataset,
    FitsTableDataset,
    FitsTableIterableDataset,
    make_loader,
)


def _write_image(path: str) -> None:
    data = np.arange(1024, dtype=np.float32).reshape(32, 32)
    torchfits.write(path, data, overwrite=True)


def _write_table(path: str) -> None:
    torchfits.table.write(
        path,
        {
            "flux": np.arange(8, dtype=np.float32) + 1.0,
            "mag": np.arange(8, dtype=np.float32) + 10.0,
        },
        overwrite=True,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="torchfits_data_catalogs_") as tmp:
        image_path = os.path.join(tmp, "mosaic.fits")
        table_path = os.path.join(tmp, "catalog.fits")
        _write_image(image_path)
        _write_table(table_path)

        # Map-style table: eager load, random access by row index
        table_ds = FitsTableDataset(table_path, columns=["flux", "mag"])
        print(f"FitsTableDataset rows: {len(table_ds)}")
        row0, label0 = table_ds[0]
        print(f"  row0 flux={row0['flux'].item():.1f}, mag={row0['mag'].item():.1f}")

        # Iterable table: constant memory via table.scan
        stream_ds = FitsTableIterableDataset(
            table_path, columns=["flux"], where="flux > 4", batch_size=3
        )
        streamed = list(stream_ds)
        print(f"FitsTableIterableDataset filtered rows: {len(streamed)}")
        print(f"  first flux={streamed[0]['flux'].item():.1f}")

        loader = make_loader(stream_ds, batch_size=4, optimize_cache=False)
        batch = next(iter(loader))
        print(f"  loader batch flux shape: {batch['flux'].shape}")

        # Cutout dataset: many windows from one mosaic
        cutout_ds = FitsCutoutDataset(
            [
                (image_path, 0, 0, 0, 8, 8),
                (image_path, 0, 8, 8, 16, 16),
            ]
        )
        print(f"FitsCutoutDataset cutouts: {len(cutout_ds)}")
        print(f"  cutout0 sum={cutout_ds[0].sum().item():.0f}")
        print(f"  cutout1 sum={cutout_ds[1].sum().item():.0f}")


if __name__ == "__main__":
    main()
