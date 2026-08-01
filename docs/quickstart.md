# Quick Start

Get up and running with torchfits in minutes. Current PyPI line is **1.0.0rc4**
(prerelease); see [Changelog](changelog.md) for release notes.

## Install

```bash
pip install torchfits "torch>=2.10,<2.11"
```

Pre-built wheels for Linux x86_64 and macOS arm64. No system CFITSIO needed — it's
vendored. Requires Python 3.10+ and **PyTorch 2.10** (ABI-matched wheels). For
other torch minors or CUDA/MPS install details, see [Installation](install.md).

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
pixi run python examples/gallery_images.py   # → examples/output/ (+ docs assets)
```

## Your First Read

```python
import torchfits

# Read a FITS image as a PyTorch tensor
tensor = torchfits.read_tensor("image.fits", hdu=0, device="cpu")
print(tensor.shape, tensor.dtype)
# torch.Size([4096, 4096]) torch.float32
```

`hdu=0` selects the **HDU** (Header Data Unit) — FITS files are structured as
a stack of numbered sections, each with a header and data block. HDU 0 is
typically the primary image; higher-numbered HDUs hold tables or additional
images.

!!! tip "Memory-mapped reads"
    By default, `read_tensor` uses **mmap** (memory mapping) — the file stays
    on disk and pages are loaded on demand, so a 10 GB image uses minimal RAM.
    Pass `mmap=False` to read the entire file into memory instead.

!!! tip "GPU transfer"
    Pass `device="cuda"` or `device="mps"` to read directly to GPU:
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
print(df.num_rows)  # pyarrow.Table

pl_df = torchfits.table.read_polars("catalog.fits", hdu=1)

cols = torchfits.table.read_torch("catalog.fits", hdu=1, columns=["RA", "DEC"])
```

## Stream a Large Table

```python
for batch in torchfits.table.scan("survey.fits", hdu=1, batch_size=50_000):
    process(batch)  # pyarrow.RecordBatch
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
torchfits.write("output.fits", tensor, header={"OBJECT": "M31"}, overwrite=True)

# Table write
torchfits.table.write("catalog_out.fits", table_dict, overwrite=True)
```

## Multi-HDU Files

```python
with torchfits.open("multi_ext.fits") as hdul:
    img = hdul[0].data       # image tensor
    tbl = hdul[1].data       # table accessor
    filtered = hdul[1].filter("FLUX > 100")
```

HDUs can also be addressed by **EXTNAME** labels (e.g., `'SCI'`, `'EVENTS'`)
instead of integer indices. Use `torchfits.read_hdus(path, hdus=['SCI'])` to
read by name.

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
