# Examples

Runnable scripts covering the main torchfits workflows. Each script is self-contained, creates temporary FITS files as needed, and cleans up after itself.

## Arrays and Tensors

| Script | What it demonstrates |
|---|---|
| [`example_image.py`](../examples/example_image.py) | `read_tensor`, `read`, `get_header`, and `write_tensor` round-trip |
| [`example_image_cutouts.py`](../examples/example_image_cutouts.py) | `read_subset`, tensor slicing, and `open_subset_reader` |
| [`example_image_cube.py`](../examples/example_image_cube.py) | 3D cubes with `read_tensor` and tensor slicing |
| [`example_image_mef.py`](../examples/example_image_mef.py) | Multi-extension files with `open`, `read_hdus`, and table `filter` |

## Tables

| Script | What it demonstrates |
|---|---|
| [`example_table.py`](../examples/example_table.py) | `read_table`, `table.read` with `where=`, `stream_table`, and `table.write` |
| [`example_table_interop.py`](../examples/example_table_interop.py) | VLA columns and `to_pandas` / `to_arrow` / `to_polars` conversion |
| [`example_polars.py`](../examples/example_polars.py) | Direct FITS → Polars via `table.to_polars` and `table.to_polars_lazy` |
| [`example_table_recipes.py`](../examples/example_table_recipes.py) | Arrow scanner, Polars lazy frames, and DuckDB SQL on FITS tables |

## PyTorch dataset pattern

| Script | What it demonstrates |
|---|---|
| [`example_image_dataset.py`](../examples/example_image_dataset.py) | PyTorch `Dataset` with `read_tensor`, `get_header`, and `read_batch` |

## Running

Run every example (including optional-deps skips):

```bash
pixi run python examples/test_examples.py
```

Or run one script directly:

```bash
pixi run python examples/example_image.py
```

Some examples require optional dependencies (Polars, DuckDB). They exit cleanly with a message when those packages are missing. Install them with:

```bash
pip install polars duckdb pyarrow
```
