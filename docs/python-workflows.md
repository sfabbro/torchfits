# Python Workflows

Practical patterns and best practices for working with FITS files in Python with PyTorch, PyArrow, and modern data science tools.

If you are migrating from existing libraries, see also the [Astropy migration guide](migration_astropy.md) or [fitsio migration guide](migration_fitsio.md).

---

## Workflow Selector

| Goal | Primary API | Key Benefit | Full Reference |
|---|---|---|---|
| **Read image to Tensor** | `read_tensor(path, device="cuda")` | Direct decode to target device (CPU/CUDA/MPS) | [Images & HDUs](#images-and-hdus) |
| **Read image + Header** | `read(path, return_header=True)` | Unified access to pixel tensor and card dictionary | [Headers](#headers) |
| **Write Tensor to FITS** | `write(path, tensor, compress="RICE_1")` | Lossless / lossy tile compression & header preservation | [Writing Images](#writing-images) |
| **Filter & load catalogs** | `table.read(path, where="MAG < 20")` | Zero-copy PyArrow table with SQL pushdown filtering | [Tables & Catalogs](#tables-as-dataframes) |
| **Catalog columns as Tensors** | `table.read_torch(path, columns=[...])` | Dictionary of PyTorch tensors ready for model inputs | [Tables as Tensors](#tables-as-tensors) |
| **Stream huge catalogs** | `table.scan(path, batch_size=50_000)` | Out-of-core chunked reader for datasets larger than RAM | [Streaming Large Catalogs](#streaming-large-catalogs) |
| **Extract image cutouts** | `read_subset(path, x1, y1, x2, y2)` | Fast bounding box extraction without loading full frame | [Cutouts](#cutouts-and-mefs) |
| **High-throughput cutouts** | `open_subset_reader(path)` | Reuses open file handle across thousands of crops | [Subset Reader](#reusable-subset-reader) |
| **Multi-Extension FITS (MEF)** | `with torchfits.open(path) as hdul:` | Pythonic HDUList navigation by index or EXTNAME | [HDULists](#mef) |
| **PyTorch DataLoader pipelines** | `FitsImageDataset` + `make_loader` | Multi-worker parallel data loading for deep learning | [Datasets & Loaders](#datasets-and-loaders) |
| **Image stretches & transforms** | `torchfits.transforms` | Differentiable transforms (Arcsinh, ZScale, SigmaClip) | [Transforms](#transforms) |

---

## 1. Reading and Writing Images {#images-and-hdus}

### Reading Images as PyTorch Tensors

`torchfits.read_tensor` reads an IMAGE HDU directly into a `torch.Tensor`. Use the `device` argument to place tensors on GPU or Apple Silicon unified memory directly:

```python
import torchfits

# Read primary image HDU (HDU 0) onto CPU
image = torchfits.read_tensor("science.fits", hdu=0)
print(f"Shape: {image.shape}, Dtype: {image.dtype}, Device: {image.device}")

# Direct transfer to GPU (CUDA on Linux, MPS on macOS)
gpu_image = torchfits.read_tensor("science.fits", hdu=0, device="cuda")

# Select HDU by extension name (EXTNAME)
sci_image = torchfits.read_tensor("mef_survey.fits", hdu="SCI")
```

### Writing Images {#writing-images}

`torchfits.write` saves PyTorch tensors or NumPy arrays to FITS format, with optional tile compression and custom headers:

```python
import torch
import torchfits

tensor = torch.randn(1024, 1024, dtype=torch.float32)

# Write uncompressed FITS
torchfits.write("output.fits", tensor, overwrite=True)

# Write tile-compressed FITS using RICE_1 (creates smaller, fast-reading files)
torchfits.write("compressed.fits", tensor, compress="RICE_1", overwrite=True)
```

---

## 2. Working with FITS Headers {#headers}

Use `torchfits.read(..., return_header=True)` or `torchfits.read_header(...)` to access metadata:

```python
import torchfits

# Read image data and header together
image, header = torchfits.read("science.fits", hdu=0, return_header=True)

# Inspect header keywords
print("Object:", header.get("OBJECT"))
print("Exposure time:", header.get("EXPTIME", 0.0))

# Iterate over keywords
for key, value in header.items():
    print(f"{key} = {value}")

# Read header only without loading pixel data
header_only = torchfits.read_header("science.fits", hdu=0)
```

To update or set keywords when saving:

```python
import torch
import torchfits

image_data = torch.randn(256, 256, dtype=torch.float32)
header = {
    "OBJECT": "M31",
    "OBSERVER": "Astronomer",
    "HISTORY": "Calibrated with flat field",
}
torchfits.write("calibrated.fits", image_data, header=header, overwrite=True)
```

---

## 3. Catalogs and Tables {#tables-as-dataframes}

### Tables as DataFrames & Astropy Tables

`torchfits.table.read` decodes binary and ASCII tables into a zero-copy `pyarrow.Table`. You can apply column projections and SQL-like predicate filters directly at the C++ reader level:

```python
import polars as pl
import torchfits

# Read table with column selection and SQL pushdown filtering
table = torchfits.table.read(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G", "CLASS_STAR"],
    where="MAG_G < 20.0 AND CLASS_STAR > 0.8",
)
print(f"Loaded {table.num_rows} rows.")

# Zero-copy export to Pandas or Polars
df_pandas = table.to_pandas()
df_polars = pl.from_arrow(table)

# Or read directly into an Astropy Table
astropy_table = torchfits.table.read_astropy(
    "catalog.fits", hdu=1, where="MAG_G < 20.0"
)

# Write Astropy Tables or DataFrames directly back to FITS
torchfits.table.write("filtered.fits", astropy_table, overwrite=True)
```

### Tables as Tensors {#tables-as-tensors}

If you are training models on tabular catalog features, `torchfits.table.read_torch` returns a dictionary of PyTorch tensors:

```python
import torchfits

# Load table columns directly as a dictionary of PyTorch tensors
tensors = torchfits.table.read_torch(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "FLUX"],
    where="FLUX > 10.0",
)

ra_tensor = tensors["RA"]  # torch.Tensor
flux_tensor = tensors["FLUX"]  # torch.Tensor
```

### Streaming Large Catalogs {#streaming-large-catalogs}

For catalogs that exceed available RAM, `torchfits.table.scan` iterates over batches of rows without loading the entire catalog:

```python
import torchfits

# Stream through a multi-million row catalog in chunks of 50,000 rows
for batch in torchfits.table.scan(
    "huge_catalog.fits", hdu=1, batch_size=50_000, columns=["RA", "DEC"]
):
    # batch is a pyarrow.RecordBatch
    process_batch(batch)
```

---

## 4. Fast Image Cutouts {#cutouts-and-mefs}

### Single Cutout

`torchfits.read_subset` extracts a rectangular sub-region without reading unneeded pixel blocks:

```python
import torchfits

# Coordinates use 0-based, half-open indexing [x1, y1, x2, y2)
cutout = torchfits.read_subset(
    "giant_mosaic.fits",
    hdu=0,
    x1=100,
    y1=100,
    x2=228,
    y2=228,
)
print(cutout.shape)  # torch.Size([128, 128])
```

### Reusable Subset Reader {#reusable-subset-reader}

When extracting hundreds or thousands of stamps from the same mosaic (e.g. galaxy postage stamps from a survey image), use `open_subset_reader` to reuse the open file descriptor:

```python
import torchfits

with torchfits.open_subset_reader("survey_mosaic.fits", hdu=0) as reader:
    # Extremely fast repeated extractions
    stamp1 = reader.read_subset(100, 100, 228, 228)
    stamp2 = reader.read_subset(200, 200, 328, 328)
```

---

## 5. Multi-Extension FITS (HDUList) {#mef}

`torchfits.open` provides a context manager for navigating Multi-Extension FITS files:

```python
import torchfits

with torchfits.open("observation.fits") as hdul:
    print(f"Total HDUs: {len(hdul)}")

    # Access HDUs by integer index
    primary_header = hdul[0].header

    # Access HDUs by extension name (EXTNAME)
    science_data = hdul["SCI"].to_tensor()
    catalog_dict = hdul["CATALOG"].read()

    # Iterate through all extensions
    for i, hdu in enumerate(hdul):
        extname = hdu.header.get("EXTNAME", f"HDU_{i}")
        print(f"HDU {i}: {extname} ({type(hdu).__name__})")
```

---

## 6. Machine Learning Datasets and Loaders {#datasets-and-loaders}

The `torchfits.data` module provides native PyTorch `Dataset` implementations optimized for multi-worker DataLoaders.

```python
from torchfits.data import FitsImageDataset, make_loader
from torchfits.transforms import ArcsinhStretch, Compose, ZScaleNormalize

# Define preprocessing pipeline
pipeline = Compose(
    [
        ArcsinhStretch(a=0.05),
        ZScaleNormalize(),
    ]
)

# Create Dataset across survey image files
dataset = FitsImageDataset(
    "data/survey/*.fits",
    hdu=0,
    label_key="CLASS_ID",
    transform=pipeline,
)

# Build high-throughput DataLoader with multi-processing workers
loader = make_loader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
)

# Training loop
for batch_images, batch_labels in loader:
    pass
```

---

## 7. Astronomical Transforms and Preprocessing {#transforms}

The `torchfits.transforms` module provides astronomical stretch, scaling, and normalization functions designed for tensors:

```python
import torchfits
from torchfits.transforms import (
    ArcsinhStretch,
    Compose,
    LogStretch,
    PercentileClipNormalize,
    SigmaClip,
    ZScaleNormalize,
    lupton_rgb,
    rgb,
)

image = torchfits.read_tensor("science.fits", hdu=0)

# Apply Arcsinh contrast stretch
stretched = ArcsinhStretch(a=0.1)(image)

# Apply IRAF-style ZScale normalization
transform_pipeline = Compose(
    [
        SigmaClip(n_sigma=3.0),
        LogStretch(),
        PercentileClipNormalize(lower_pct=1.0, upper_pct=99.0),
    ]
)
processed = transform_pipeline(image)

# Combine filter images into color RGB (shortest wavelength first)
g = r = i = image
rgb_img = rgb(g, r, i)
# Astropy-parity Lupton (reddest first)
lupton = lupton_rgb(i, r, g, Q=8.0, stretch=0.5)
```

---

## Next Steps

- [Quick Start](quickstart.md): Step-by-step introduction to reading, writing, and transforming data.
- [API Reference](api.md): Complete module documentation and parameter details.
- [Transform Gallery](examples-transforms.md): Visual before/after gallery of astronomical transforms.
- [CLI Reference](cli.md): Fast command-line inspection and batch processing.
