# Quick Start

Get up and running with torchfits in minutes. Current cut is **1.0.0** (not yet released; **1.0.0rc5** is the PyPI line,
built for the PyTorch 2.13 lane); see [Changelog](changelog.md) for release notes.

## Install

One bare command — torchfits wheels are ABI-matched to PyTorch 2.13.x (Linux
x86_64 and aarch64, macOS arm64, CPython 3.10–3.14), and the wheel metadata
pins that range, so pip installs or upgrades torch for you. The 1.0 line is
still a release candidate on PyPI, so add `--pre` until the final tag:

```bash
pip install --pre torchfits
```

Pre-built wheels, no system CFITSIO needed (it's vendored). Prefer a CUDA /
CPU / MPS flavor, or must keep a specific torch minor? See
[Installation](install.md).

## Shell tools

```bash
torchfits info image.fits
torchfits header image.fits --keyword BITPIX --json
torchfits verify image.fits
```

Full command reference: [CLI guide](cli.md). Job-first shell recipes:
[CLI recipes](cli-recipes.md).

## Real data + a first figure

Worked examples with printed output and plots: [Examples](examples.md).
Transform gallery: [Transform gallery](examples-transforms.md).
Multi-file datasets: [ML with FITS](examples-ml.md).

```bash
pixi run python examples/gallery_images.py   # writes examples/output/
```

The generated PNGs are copied into the documentation gallery by the docs
build; they are not written directly by this command.

## Your First Read

```python
import torchfits

# Read a FITS image as a PyTorch tensor
tensor = torchfits.read_tensor("image.fits", hdu=0, device="cpu")
print(tensor.shape, tensor.dtype)
# Shape and dtype depend on the FITS image.
```

`hdu=0` selects the **HDU** (Header Data Unit) — FITS files are structured as
a stack of numbered sections, each with a header and data block. HDU 0 is
typically the primary image; higher-numbered HDUs hold tables or additional
images.

!!! tip "Memory-mapped reads"
    `mmap=True` selects the eligible memory-mapped input path, but the result is
    still a newly allocated tensor. It is not a lazy tensor view, and float or
    compressed images may use a buffered path. Use `read_subset` or
    `open_subset_reader` when you need bounded-memory cutouts.

!!! tip "GPU transfer"
    Pass `device="cuda"` or `device="mps"` to request accelerator placement.
    torchfits reads on the host and transfers the resulting tensor:
    ```python
    tensor = torchfits.read_tensor("image.fits", hdu=0, device="cuda")
    ```

## Read an Image with Header

```python
data, header = torchfits.read("image.fits", hdu=0, return_header=True)
print(header["OBJECT"])  # e.g. "M31"
```

## Filter a Table

```python
df = torchfits.table.read(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20.0 AND CLASS_STAR > 0.9",
)
print(df.num_rows)  # number of rows in this pyarrow.Table

# Optional integration: install Polars separately; it is not a core dependency.
pl_df = torchfits.table.read_polars("catalog.fits", hdu=1)

cols = torchfits.table.read_torch("catalog.fits", hdu=1, columns=["RA", "DEC"])
```

## Stream a Large Table

```python
for batch in torchfits.table.scan("survey.fits", hdu=1, batch_size=50_000):
    print(batch.num_rows)  # pyarrow.RecordBatch
```

## Many files with Datasets

| Layer | Use when |
|-------|----------|
| `read_tensor` / `table.read` | One file, inspect, write |
| `torchfits.transforms` | Reusable stretch / normalize for display or model input |
| `Fits*Dataset` + `make_loader` | Many files or rows with shuffle / workers |

```python
from torchfits.data import FitsImageDataset, make_loader

ds = FitsImageDataset("observations/*.fits", label_key="CLASS")
loader = make_loader(ds, batch_size=32, num_workers=4)

for images, labels in loader:
    pass
```

Details: [Data module](api-data.md), [Transforms](api-transforms.md).

## Write Back

```python
import torch

tensor = torch.zeros((8, 8), dtype=torch.float32)
table_dict = {
    "ID": torch.tensor([1, 2], dtype=torch.int32),
    "FLUX": torch.tensor([1.5, 2.5], dtype=torch.float32),
}

torchfits.write("output.fits", tensor, header={"OBJECT": "M31"}, overwrite=True)

# Table write
torchfits.table.write("catalog_out.fits", table_dict, overwrite=True)
```

## Multi-HDU Files

```python
with torchfits.open("multi_ext.fits") as hdul:
    img = hdul[0].to_tensor()
    tbl = hdul[1].to_tensor_dict()
    filtered = hdul[1].filter("FLUX > 100")
```

HDUs can also be addressed by **EXTNAME** labels (e.g., `'SCI'`, `'EVENTS'`)
instead of integer indices. `read_hdus(path, hdus=["SCI"])` reads named image
HDUs and returns a list; use `table.read(path, hdu="EVENTS")` for a named table.

## What's Next?

- [Python workflows](python-workflows.md) — images, tables, cutouts, datasets
- [Core I/O](api-core-io.md) — `read_tensor`, `read_subset`, writes, headers
- [Tables](api-tables.md) — `table.read`, filters, Polars/DuckDB
- [Examples](examples.md) — runnable scripts
- [ML with FITS](examples-ml.md) — Datasets and loaders
- [Data module](api-data.md) — Dataset / loader API
- [Transforms](api-transforms.md) — stretches, normalizers, clip
- [CLI](cli.md) — shell inspect / cutout / convert
- [Architecture](architecture.md) — CFITSIO, mmap, caching
