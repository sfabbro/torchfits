# Examples

Runnable scripts covering the main torchfits workflows. Each script is self-contained and creates temporary FITS files as needed.

## Images

| Script | What it demonstrates |
|---|---|
| [`example_image.py`](../examples/example_image.py) | Read a FITS image into a tensor, inspect the header |
| [`example_image_cutouts.py`](../examples/example_image_cutouts.py) | Extract sub-regions with `read_subset` (0-based pixel bounds) |
| [`example_image_cube.py`](../examples/example_image_cube.py) | Read and slice 3D data cubes (e.g. RA, Dec, velocity) |
| [`example_image_mef.py`](../examples/example_image_mef.py) | Work with multi-extension FITS files (primary + image + table HDUs) |

## Tables

| Script | What it demonstrates |
|---|---|
| [`example_table.py`](../examples/example_table.py) | Read binary tables into tensors, access columns, write back |
| [`example_table_interop.py`](../examples/example_table_interop.py) | Variable-length array columns, interop with [Astropy](https://www.astropy.org/) |
| [`example_polars.py`](../examples/example_polars.py) | Table workflows with [Polars](https://pola.rs/) and [Apache Arrow](https://arrow.apache.org/) |
| [`example_table_recipes.py`](../examples/example_table_recipes.py) | Arrow scanner, [Polars](https://pola.rs/) lazy frames, and [DuckDB](https://duckdb.org/) SQL queries on FITS tables |

## WCS

| Script | What it demonstrates |
|---|---|
| [`example_wcs_transform.py`](../examples/example_wcs_transform.py) | Pixel-to-world and world-to-pixel transforms for 2D images and 1D spectra |

## HEALPix & Spherical Harmonics

| Script | What it demonstrates |
|---|---|
| [`example_healpix.py`](../examples/example_healpix.py) | `ang2pix`, `pix2ang`, disc queries, and map sampling with `torchfits.sphere` |
| [`example_spectral_sht.py`](../examples/example_spectral_sht.py) | Spherical harmonic transforms: power spectra, `synfast`, `map2alm`, smoothing |

## ML Integration

| Script | What it demonstrates |
|---|---|
| [`example_image_dataset.py`](../examples/example_image_dataset.py) | PyTorch `Dataset` for FITS images with caching and `DataLoader` |
| [`example_ml_pipeline.py`](../examples/example_ml_pipeline.py) | End-to-end pipeline: data loading, transforms, caching, distributed training |

## Running

```bash
pixi run python examples/example_image.py
```

Or directly:

```bash
python examples/example_image.py
```

Some examples require optional dependencies (e.g. Polars, DuckDB, Astropy). Install them with:

```bash
pip install torchfits[examples]
```
