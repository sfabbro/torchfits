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

## PyTorch dataset pattern

| Script | What it demonstrates |
|---|---|
| [`example_image_dataset.py`](../examples/example_image_dataset.py) | Minimal PyTorch `Dataset` using `torchfits.read` and `torchfits.get_header` |

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
