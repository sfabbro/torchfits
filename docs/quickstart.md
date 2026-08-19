# Quick Start

Get up and running with `torchfits` in minutes.

---

## 1. Installation

Install the prebuilt binary wheel:

```bash
pip install torchfits
```

Prebuilt binary wheels include vendored CFITSIO (no C++ compiler or external dependencies required). For CPU-only installs, custom CUDA builds, or existing PyTorch 2.11/2.12 environments, see the [Installation guide](install.md).

---

## 2. Reading Images to PyTorch Tensors

```python
import torchfits

# Read primary image onto CPU
image = torchfits.read_tensor("science.fits", hdu=0)
print(f"Shape: {image.shape}, Dtype: {image.dtype}")

# Direct decode to GPU (CUDA on Linux, MPS on Apple Silicon)
gpu_image = torchfits.read_tensor("science.fits", hdu=0, device="cuda")

# Read pixel data alongside header metadata
data, header = torchfits.read("science.fits", hdu=0, return_header=True)
print("Target Object:", header.get("OBJECT"))
print("Exposure Time:", header.get("EXPTIME"))
```

---

## 3. Extracting Fast Cutouts

Extract sub-regions without reading or decompressing the entire image:

```python
import torchfits

# Coordinates use 0-based, half-open indexing [x1, y1, x2, y2)
stamp = torchfits.read_subset(
    "giant_mosaic.fits",
    hdu=0,
    x1=100,
    y1=100,
    x2=228,
    y2=228,
)
print(stamp.shape)  # torch.Size([128, 128])
```

---

## 4. Reading and Filtering Catalogs

Read FITS binary and ASCII tables directly into PyArrow tables with SQL-style pushdown filtering:

```python
import torchfits

# Load filtered catalog into PyArrow
table = torchfits.table.read(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20.0 AND CLASS_STAR > 0.8",
)
print(f"Filtered rows: {table.num_rows}")

# Convert to Pandas or Polars DataFrame
df_pandas = table.to_pandas()

# Load columns directly as a dictionary of PyTorch tensors
tensors = torchfits.table.read_torch(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC"],
)
```

---

## 5. Streaming Massive Tables

Iterate over catalogs larger than system RAM in configurable batch sizes:

```python
import torchfits

for batch in torchfits.table.scan("huge_catalog.fits", hdu=1, batch_size=50_000):
    # batch is a pyarrow.RecordBatch
    process_batch(batch)
```

---

## 6. Multi-Extension FITS (MEF)

Inspect and load multiple extensions using a Pythonic context manager:

```python
import torchfits

with torchfits.open("observation.fits") as hdul:
    print(f"Total HDUs: {len(hdul)}")

    # Access primary header
    primary_header = hdul[0].header

    # Access extensions by name (EXTNAME)
    science_image = hdul["SCI"].to_tensor()
    catalog_table = hdul["CATALOG"].read()
```

---

## 7. Writing Tensors and Tables

Save tensors and tabular data back to standard FITS files with optional tile compression:

```python
import torch
import torchfits

# Write an image tensor with custom header metadata
image_data = torch.randn(512, 512, dtype=torch.float32)
torchfits.write(
    "output_image.fits",
    image_data,
    header={"OBJECT": "M31", "FILTER": "r"},
    overwrite=True,
)

# Write tile-compressed FITS using Rice algorithm
torchfits.write("compressed.fits", image_data, compress="RICE_1", overwrite=True)

# Write a dictionary of column tensors as a binary table
catalog_data = {
    "ID": torch.tensor([1, 2, 3], dtype=torch.int64),
    "FLUX": torch.tensor([10.5, 23.1, 45.0], dtype=torch.float32),
}
torchfits.table.write("output_table.fits", catalog_data, overwrite=True)
```

---

## 8. Machine Learning Pipelines

Use `torchfits.data` to build multi-worker PyTorch `DataLoader` pipelines:

```python
from torchfits.data import FitsImageDataset, make_loader
from torchfits.transforms import ArcsinhStretch, Compose, ZScaleNormalize

# Preprocessing transform
transforms = Compose(
    [
        ArcsinhStretch(a=0.05),
        ZScaleNormalize(),
    ]
)

# Create Dataset across observation files
dataset = FitsImageDataset(
    "data/survey/*.fits",
    hdu=0,
    label_key="CLASS_ID",
    transform=transforms,
)

# Build high-performance DataLoader
loader = make_loader(dataset, batch_size=32, num_workers=4, shuffle=True)

for images, labels in loader:
    # Train step
    pass
```

---

## 9. Shell CLI in 30 Seconds

The `torchfits` command line tool provides immediate shell access to FITS inspection and transformation:

```bash
# Print HDU overview
torchfits info science.fits

# Dump and filter header keywords
torchfits header science.fits -k OBJECT -k 'NAXIS*'

# Compute pixel statistics
torchfits stats science.fits -e 0

# Extract a cutout to a new FITS file
torchfits cutout 'science.fits[100:256,100:256]' cutout.fits

# Convert table to Parquet with a row filter
torchfits convert catalog.fits bright.parquet -e 1 -w "MAG_G < 18.0"

# Verify CHECKSUM / DATASUM
torchfits verify science.fits
```

---

## Where to Go Next

- [Python Workflows](python-workflows.md): Comprehensive workflow patterns and best practices.
- [CLI Guide](cli.md) & [CLI Recipes](cli-recipes.md): Command-line reference and shell recipes.
- [Transforms Gallery](examples-transforms.md): Visual before/after figures for astronomical stretches.
- [API Reference](api.md): Detailed signatures and parameters for all modules.
