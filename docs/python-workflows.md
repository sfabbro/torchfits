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
| **In-memory byte buffers** | `read_bytes(buffer)`, `write_bytes(tensor)` | Cloud/S3/HTTP microservice and Lambda I/O | [In-Memory Buffers](#in-memory-buffers) |

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

# Inspect header cards
print("Object:", header.get("OBJECT"))
print("Exposure time:", header.get("EXPTIME", default=0.0))

# Iterate over all cards
for card in header.cards:
    print(f"{card.key} = {card.value} / {card.comment}")

# Read header only without loading pixel data
header_only = torchfits.read_header("science.fits", hdu=0)
```

To update or set keywords when saving:

```python
header["OBSERVER"] = "Astronomer"
header["HISTORY"] = "Calibrated with flat field"

torchfits.write("calibrated.fits", image, header=header, overwrite=True)
```

---

## 3. Catalogs and Tables {#tables-as-dataframes}

### Tables as DataFrames (PyArrow, Pandas, Polars)

`torchfits.table.read` decodes binary and ASCII tables into a zero-copy `pyarrow.Table`. You can apply column projections and SQL-like predicate filters directly at the C++ reader level:

```python
import torchfits

# Read table with column selection and SQL pushdown filtering
table = torchfits.table.read(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G", "MAG_R"],
    where="MAG_G < 20.0 AND MAG_R > 15.0",
)
print(f"Loaded {table.num_rows} rows.")

# Zero-copy export to Pandas or Polars
df_pandas = table.to_pandas()

import polars as pl

df_polars = pl.from_arrow(table)
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
    where="FLUX > 100",
)

ra_tensor = tensors["RA"]  # torch.Tensor
flux_tensor = tensors["FLUX"]  # torch.Tensor
```

### Streaming Large Catalogs {#streaming-large-catalogs}

For catalogs that exceed available RAM, `torchfits.table.scan` iterates over batches of rows without loading the entire catalog:

```python
import torchfits

# Stream through a 100-million row catalog in chunks of 50,000 rows
for batch in torchfits.table.scan(
    "huge_survey.fits", hdu=1, batch_size=50_000, columns=["RA", "DEC"]
):
    # batch is a pyarrow.RecordBatch
    process_chunk(batch)
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
    x1=500,
    y1=500,
    x2=628,
    y2=628,
    device="cuda",
)
print(cutout.shape)  # torch.Size([128, 128])
```

### Reusable Subset Reader {#reusable-subset-reader}

When extracting hundreds or thousands of stamps from the same mosaic (e.g. galaxy postage stamps from a survey image), use `open_subset_reader` to reuse the open file descriptor:

```python
import torchfits

with torchfits.open_subset_reader("mosaic.fits", hdu=0) as reader:
    # Extremely fast repeated extractions
    stamp1 = reader.read_subset(100, 100, 228, 228)
    stamp2 = reader.read_subset(500, 500, 628, 628)
    stamp3 = reader.read_subset(1200, 800, 1328, 928)
```

---

## 5. Multi-Extension FITS (HDUList) {#mef}

`torchfits.open` provides a context manager for navigating Multi-Extension FITS files:

```python
import torchfits

with torchfits.open("mef_observation.fits") as hdul:
    print(f"Total HDUs: {len(hdul)}")

    # Access HDUs by integer index
    primary_header = hdul[0].header

    # Access HDUs by extension name (EXTNAME)
    science_data = hdul["SCI"].data  # Tensor or Arrow table
    variance_data = hdul["VAR"].data

    # Iterate through all extensions
    for i, hdu in enumerate(hdul):
        print(f"HDU {i}: {hdu.name} ({type(hdu).__name__})")
```

---

## 6. Machine Learning Datasets and Loaders {#datasets-and-loaders}

The `torchfits.data` module provides native PyTorch `Dataset` implementations optimized for multi-worker DataLoaders.

```python
from torchfits.data import FitsImageDataset, make_loader
from torchfits.transforms import Compose, ArcsinhStretch, ZScaleNormalize

# Define preprocessing pipeline
pipeline = Compose(
    [
        ArcsinhStretch(factor=0.05),
        ZScaleNormalize(),
    ]
)

# Create Dataset across thousands of image files
dataset = FitsImageDataset(
    "data/survey/*.fits",
    hdu=0,
    label_key="CLASS_ID",
    transform=pipeline,
)

# Build high-throughput DataLoader with multi-processing workers
loader = make_loader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# Training loop
for batch_images, batch_labels in loader:
    # batch_images: [64, 1, H, W]
    train_step(batch_images, batch_labels)
```

Available Datasets in `torchfits.data`:

- `FitsImageDataset`: Load 2D/3D images from file lists or glob patterns.
- `FitsCutoutDataset`: Extract on-the-fly random or coordinate-based cutouts from large survey mosaics.
- `FitsTableDataset`: In-memory tabular dataset for catalogue-driven models.
- `FitsTableIterableDataset`: Out-of-core streaming dataset for massive catalogs.

---

## 7. Astronomical Transforms and Preprocessing {#transforms}

The `torchfits.transforms` module provides astronomical stretch, scaling, and normalization functions designed for tensors:

```python
import torchfits
from torchfits.transforms import (
    ArcsinhStretch,
    LogStretch,
    SqrtStretch,
    ZScaleNormalize,
    PercentileClipNormalize,
    SigmaClip,
    lupton_rgb,
    Compose,
)

image = torchfits.read_tensor("galaxy.fits", hdu=0)

# Apply Arcsinh contrast stretch
stretched = ArcsinhStretch(factor=0.1)(image)

# Apply IRAF-style ZScale normalization
normalized = ZScaleNormalize()(image)

# Chain transforms together into a single pipeline
transform_pipeline = Compose(
    [
        SigmaClip(sigma=3.0),
        LogStretch(),
        PercentileClipNormalize(lower_pct=1.0, upper_pct=99.0),
    ]
)
processed = transform_pipeline(image)

# Combine 3 band images into a color Lupton RGB tensor
r = torchfits.read_tensor("r.fits")
g = torchfits.read_tensor("g.fits")
b = torchfits.read_tensor("b.fits")
rgb = lupton_rgb(r, g, b, q=8.0, stretch=0.5)
```

---

## 8. In-Memory Buffers and Cloud I/O {#in-memory-buffers}

For web APIs, cloud microservices, AWS Lambda, or object storage streams (S3/GCS/HTTP):

```python
import torch
import torchfits

# Serialize a tensor directly into an in-memory FITS byte buffer
data = torch.ones((256, 256), dtype=torch.float32)
fits_bytes = torchfits.write_bytes(data)

# Read directly from an in-memory byte buffer without writing to disk
tensor_from_bytes = torchfits.read_bytes(fits_bytes)
print(tensor_from_bytes.shape)  # torch.Size([256, 256])
```

---

## Next Steps

- [Quick Start](quickstart.md): Step-by-step introduction to reading, writing, and transforming data.
- [API Reference](api.md): Complete module documentation and parameter details.
- [Transform Gallery](examples-transforms.md): Visual before/after gallery of astronomical transforms.
- [CLI Reference](cli.md): Fast command-line inspection and batch processing.
