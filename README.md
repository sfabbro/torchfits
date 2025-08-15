# torchfits

**IN-PROGRESS -STILL EXPERIMENTAL.**

[![PyPI version](https://badge.fury.io/py/torchfits.svg)](https://badge.fury.io/py/torchfits) [![License](https://img.shields.io/badge/License-GPL_2.0-blue.svg)](https://opensource.org/licenses/GPL-2.0)

`torchfits` is a high-performance Python package for reading FITS files directly into PyTorch tensors. It leverages `cfitsio` and `wcslib` for speed and accuracy, making it ideal for astronomy and machine learning applications that require efficient access to large FITS datasets. It is designed to be easy to use, with a simple and consistent API.

<!-- perf:begin -->

Performance highlights (auto-updated):

- Image full read (torchfits): 256: 0.10 ms, 512: 0.10 ms, 1024: 0.22 ms

See artifacts/benchmarks/plots for charts.

<!-- perf:end -->









## Features

* **Fast FITS I/O:** Uses a highly optimized C++ backend (powered by `cfitsio`) for rapid data access.
* **Direct PyTorch Tensors:** Reads FITS data directly into PyTorch tensors, avoiding unnecessary data copies.
* **Flexible Cutout Reading:** Supports CFITSIO-style cutout strings (e.g., `'myimage.fits[1][10:20,30:40]'`).
* **Simplified Cutout Definition:** Provides an easy way to read rectangular regions using `start` and `shape` parameters.
* **Automatic Data Type Handling:** Automatically determines the correct PyTorch data type.
* **Image, Cube, and Table Support:** Reads data from image HDUs (1D, 2D, and 3D) and binary/ASCII table HDUs.
* **WCS Support:** Includes functions for world-to-pixel and pixel-to-world coordinate transformations using `wcslib`. WCS information is automatically updated when reading cutouts. (Note: these functions are internal, WCS use is integrated directly into `read`).
* **Header Access:** Provides functions to access the full FITS header as a Python dictionary, or to retrieve individual header keyword values.
* **HDU Information:** Functions to get the number of HDUs, HDU types, and image dimensions.
* **Designed for PyTorch Datasets:** The API is designed to integrate seamlessly with PyTorch's `Dataset` and `DataLoader` classes, including distributed data loading.
* **HDU selection**: Select HDU either by name or number.

## Installation

### Prerequisites

Before installing `torchfits`, you need to install the required system libraries:

**Linux (Debian/Ubuntu):**

```bash
sudo apt-get install libcfitsio-dev libwcs-dev
```

**Linux (Fedora/CentOS/RHEL):**

```bash
sudo yum install cfitsio-devel wcslib-devel
```

**macOS (Homebrew):**

```bash
brew install cfitsio wcslib
```

**macOS (MacPorts):**

```bash
sudo port install cfitsio wcslib
```

### Install torchfits

Once prerequisites are installed:

```bash
pip install torchfits
```

### Development Installation

For development or the latest features:

```bash
git clone https://github.com/sfabbro/torchfits.git
cd torchfits
pip install -e .
```

## Quickstart

Get started with `torchfits` in just a few lines of code! Here are the most common use cases:

### Basic Usage

```python
import torchfits

# Read an entire FITS image
data, header = torchfits.read("my_image.fits")
print(f"Image shape: {data.shape}, Data type: {data.dtype}")
print(f"NAXIS1: {header['NAXIS1']}, NAXIS2: {header['NAXIS2']}")

# Read a specific extension
sci_data, sci_header = torchfits.read("my_image.fits", hdu="SCI")

# Read a cutout (faster for large images)
cutout, cutout_header = torchfits.read("my_image.fits", start=[100, 200], shape=[50, 50])
```

### Working with Tables

```python
# Read all columns from a table
table_data = torchfits.read("catalog.fits", hdu="CATALOG")
print(f"Available columns: {list(table_data.keys())}")

# Read specific columns only
ra_dec = torchfits.read("catalog.fits", hdu="CATALOG", columns=["RA", "DEC"])
print(f"RA shape: {ra_dec['RA'].shape}")

# Read a subset of rows
first_1000 = torchfits.read("catalog.fits", hdu="CATALOG", num_rows=1000)
```

### Variable-Length Array (VLA) columns

Binary tables with variable-length array columns are supported. When reading a VLA column, `torchfits.read(..., format='tensor')` returns a Python list of 1D tensors: one tensor per row.

You can easily pad/densify these ragged sequences with the `pad_ragged` helper:

```python
import torch, torchfits as tf

# Write a VLA test file (one ragged array per row)
arrays = [torch.arange(i, dtype=torch.float32) for i in range(5)]
tf.write_variable_length_array("vla.fits", arrays, header={"EXTNAME": "VLA"})

# Read back
tbl = tf.read("vla.fits", hdu="VLA")
vla_col = tbl["ARRAY"]  # list[Tensor], length == nrows

# Convert to dense [rows, max_len] with lengths
from torchfits import pad_ragged
padded, lengths = pad_ragged(vla_col, pad_value=0.0)
print(padded.shape, lengths)
```

Notes:

* The base numeric dtype of the VLA column is preserved (writer uses double by default).
* Table operations like filtering and sorting in `FitsTable` work with list-valued columns.
* Keep VLA columns as ragged lists when possible; use `pad_ragged` for batching models that need dense tensors.

### Batched reads and stacking

Use the high-level dataset/batch helpers to read across many files efficiently. When all outputs are tensors of the same shape and dtype, you can request a stacked tensor.

```python
from torchfits.dataset import BatchReadSpec, read_batch

specs = [
    BatchReadSpec(path="im1.fits", hdu=0, start=(0,0), shape=(64,64)),
    BatchReadSpec(path="im2.fits", hdu=0, start=(0,0), shape=(64,64)),
]

# Returns a tensor with shape [N, H, W]
stacked = read_batch(specs, stack=True)

# If you need to preserve headers when stacking image outputs,
# request a tuple (stacked_tensor, headers_list):
stacked, headers = read_batch(specs, stack=True, preserve_headers_on_stack=True)
```

For table cutouts, `read_multi_table_cutouts(..., stack=True)` stacks per-column tensors when shapes and dtypes match across the batch, returning a dict of stacked tensors keyed by column name. If you also request null masks (see below), stacking returns a tuple `(stacked_data_dict, stacked_masks_dict)`.

### Column slicing aliases for tables

Instead of listing column names, you can select a contiguous range of columns by index using `col_start`/`col_count`.

```python
from torchfits.dataset import TableCutoutSpec, read_multi_table_cutouts, BatchReadSpec, read_batch

# Read first two columns (indices 0 and 1) for 5 rows
spec = TableCutoutSpec(path="tab.fits", hdu=1, row_start=0, row_count=5, col_start=0, col_count=2)
sub = read_multi_table_cutouts([spec], parallel=False)[0]

# Same idea via batch read helper
bspec = BatchReadSpec(path="tab.fits", hdu=1, row_start=10, row_count=3, col_start=1, col_count=1)
row = read_batch([bspec], parallel=False)[0]  # dict with a single column

### Null handling for tables (TNULL masks)

When tables are written with integer null sentinels (TNULLn), torchfits preserves the sentinel values in the data tensors and exposes boolean null masks separately.

- Low-level helper: `from torchfits import read_table_with_null_masks` returns `(data_dict, header, masks_dict)`.
- Convenience cutouts: `read_table_cutout(..., return_null_masks=True)` returns `(data_dict, masks_dict)`.
- Batched cutouts: `read_multi_table_cutouts([...], stack=True)` will stack both data and masks when `return_null_masks=True` is set in each `TableCutoutSpec`, returning `(stacked_data_dict, stacked_masks_dict)`.

Example:

```python
from torchfits.dataset import read_table_cutout, TableCutoutSpec, read_multi_table_cutouts

# Single cutout with masks
data, masks = read_table_cutout("catalog.fits", hdu=1, row_start=0, row_count=100, return_null_masks=True)
val_mask = masks.get("VAL")  # True where null

# Batched stacking with masks
specs = [
    TableCutoutSpec(path="catalog1.fits", hdu=1, row_start=0, row_count=32, return_null_masks=True),
    TableCutoutSpec(path="catalog2.fits", hdu=1, row_start=0, row_count=32, return_null_masks=True),
]
stacked_data, stacked_masks = read_multi_table_cutouts(specs, parallel=True, stack=True)

# Apply masks to replace nulls with NaN in a dict-of-tensors
from torchfits import apply_null_masks_to_dict
clean = apply_null_masks_to_dict(data, masks)

# Or for FitsTable objects
from torchfits import FitsTable
ft = FitsTable(data)
ft_clean = ft.with_applied_null_masks(masks)
```
```

## Writing & Updating FITS

Use the unified write() API to create images, tables, and multi-extension files. You can also append new HDUs and update headers or pixel data in-place.

Basic image write and append:

```python
import torch, torchfits as tf

# Write an image (primary)
img = torch.randint(0, 1000, (32, 32), dtype=torch.int16)
tf.write("image.fits", img, {"OBJECT": "UnitTest"})

# Append another image as a new HDU
extra = torch.ones(16, 16)
tf.write("image.fits", extra, {"EXTNAME": "SCI"}, append=True)
```

Write a table (dict) with null sentinels and append a second table:

```python
import torch, torchfits as tf

table = {
    "ID": torch.tensor([1,2,3,4], dtype=torch.int32),
    "NAME": ["Alpha", "Beta", "Gamma", ""],  # strings supported
    "VAL": torch.tensor([10, -9999, 30, -9999], dtype=torch.int32),
}
tf.write("catalog.fits", table, {"EXTNAME": "CAT"}, null_sentinels={"VAL": -9999})

# Append another table HDU
more = {"A": torch.arange(5), "B": torch.arange(5,10)}
tf.write("catalog.fits", more, {"EXTNAME": "CAT2"}, append=True)
```

Append a FitsTable using per-column metadata (units/descriptions/null_value) via ColumnInfo:

```python
from torchfits import FitsTable, ColumnInfo

data = {
    "RA": torch.tensor([1.0, 2.0, 3.0]),
    "DEC": torch.tensor([-1.0, -2.0, -3.0]),
}
meta = {
    "RA": ColumnInfo("RA", dtype=torch.float32, unit="deg", description="Right Ascension"),
    "DEC": ColumnInfo("DEC", dtype=torch.float32, unit="deg", description="Declination"),
}
ft = FitsTable(data, metadata=meta)
tf.write("catalog.fits", ft, {"EXTNAME": "META"}, append=True)
```

Update headers and data in-place:

```python
# Update/add header keywords on a given HDU (by index or EXTNAME)
tf.update_header("image.fits", {"OBJECT": "Updated", "EXPTIME": "42"}, hdu=1)

# Update entire image data
tf.update_data("image.fits", torch.zeros(32,32), hdu=1)

# Update a sub-region (start is 0-based; shape in pixels)
tf.update_data("image.fits", torch.full((8,8), 5.0), hdu=1, start=[4,4], shape=[8,8])
```

Notes:

* write() auto-detects HDU type (image vs table) for tensors, dicts, and FitsTable.
* For tables, you can pass null_sentinels to emit TNULLn and preserve masks; FitsTable ColumnInfo.null_value is also honored when present.
* Compression writing is supported (GZIP, RICE, HCOMPRESS). When reading, if the primary HDU is empty and the next HDU is a tile‑compressed image, `read(hdu=0)` will transparently return that image.

Write a 3D cube:

```python
import torch, torchfits as tf

cube = torch.randn(4, 64, 64)
tf.write_cube("cube.fits", cube, header={"OBJECT": "Demo"}, overwrite=True)
# Later
data, hdr = tf.read("cube.fits")
print(data.shape)
```


### File Information

```python
# Get basic file information
num_hdus = torchfits.get_num_hdus("my_image.fits")
dims = torchfits.get_dims("my_image.fits", hdu=1)
hdu_type = torchfits.get_hdu_type("my_image.fits", hdu=1)

print(f"File has {num_hdus} HDUs")
print(f"Primary HDU dimensions: {dims}")
print(f"Primary HDU type: {hdu_type}")
```

### Object-Oriented Approach (Recommended for Multiple Operations)

When you need to perform multiple operations on the same file, use the object-oriented interface for better performance:

```python
import torchfits

with torchfits.FITS("my_multiextension_file.fits") as f:
    print(f"File has {len(f)} HDUs")
    
    # Access HDUs by index or name
    primary = f[0]
    sci_hdu = f["SCI"]
    
    # Read data and access headers
    primary_data = primary.read()
    sci_data = sci_hdu.read(start=[0, 0], shape=[100, 100])  # Cutout
    
    print(f"Primary HDU shape: {primary_data.shape}")
    print(f"Science HDU cutout shape: {sci_data.shape}")
    print(f"EXPTIME: {sci_hdu.header['EXPTIME']}")
```

## Examples

The `examples/` directory contains several example scripts demonstrating various use cases:

* **`example_basic_reading.py`:** Basic reading of full HDUs and headers.  Demonstrates using `torchfits.read` with simple filenames and accessing header information.
* **`example_cutouts.py`:**  Shows how to read cutouts using both CFITSIO-style strings (passed directly to `torchfits.read`) and the `start`/`shape` parameters of the `torchfits.read` function.
* **`example_mef.py`:**  Illustrates working with Multi-Extension FITS (MEF) files, including iterating through HDUs and accessing HDUs by number and name.
* **`example_tables.py`:** Focuses on reading data from FITS binary tables, showing how to access individual columns.
* **`example_datacube.py`:**  Demonstrates reading slices and 1D spectra from a 3D data cube, using both CFITSIO strings and `start`/`shape` parameters.
* **`example_dataset.py`:**  A basic example of integrating `torchfits` with PyTorch's `Dataset` and `DataLoader` for efficient data loading.
* **`example_mnist.py`:** A complete, self-contained example that downloads the MNIST dataset, converts it to FITS, and trains a simple CNN classifier using `torchfits` for FITS I/O.
* **`example_sdss_classification.py`:**  A complete example that downloads a small subset of SDSS spectroscopic data, loads the spectra using `torchfits`, and trains a simple CNN classifier to distinguish between stars, galaxies, and quasars.
* **`example_variable_length_write.py`:** Write a variable-length array table (ragged arrays per row) and read it back.
* **`example_remote_caching.py`:** Read a remote FITS over HTTPS with SmartCache, showing cache hits and statistics.

To run the examples, navigate to the `examples/` directory and run, for instance:

```bash
python example_basic_reading.py
```

The examples that use external datasets will automatically download and cache the data.

## API Reference

### Core Reading Function

#### `torchfits.read(filename_or_url, hdu=1, start=None, shape=None, columns=None, start_row=0, num_rows=None, cache_capacity=0, device='cpu')`

The main function for reading FITS data into PyTorch tensors. Handles images, data cubes, and tables seamlessly.

**Parameters:**

* `filename_or_url` (str or dict): Path to FITS file, CFITSIO-compatible URL, or fsspec parameters dict for remote files
* `hdu` (int or str, optional): HDU number (1-based) or name. Defaults to 1 (primary HDU)
* `start` (list[int], optional): Starting pixel coordinates (0-based) for cutouts. For 2D: `[row, col]`
* `shape` (list[int], optional): Cutout shape. Required if `start` given. Use `-1` to read to end of dimension
* `columns` (list[str], optional): Column names for table HDUs. Reads all columns if `None`
* `start_row` (int, optional): Starting row (0-based) for table reads. Default: 0
* `num_rows` (int, optional): Number of rows to read from table. Reads all if `None`
* `cache_capacity` (int, optional): Cache size in MB. Default: auto-sized. Set 0 to disable
* `device` (str, optional): Target device ('cpu' or 'cuda'). Default: 'cpu'

**Returns:**

* For images/cubes: tuple `(data, header)` where `data` is PyTorch tensor, `header` is dict
* For tables: dict with column names as keys, PyTorch tensors as values

### File Information Functions

* **`torchfits.get_header(filename, hdu=1)`**: Get complete FITS header as dictionary
* **`torchfits.get_num_hdus(filename)`**: Get total number of HDUs in file  
* **`torchfits.get_dims(filename, hdu=1)`**: Get dimensions of image/cube HDU
* **`torchfits.get_header_value(filename, hdu, key)`**: Get value of specific header keyword
* **`torchfits.get_hdu_type(filename, hdu=1)`**: Get HDU type ("IMAGE", "BINTABLE", "TABLE", "UNKNOWN")

### World Coordinate System (WCS) Functions

* **`torchfits.world_to_pixel(world_coords, header)`**: Convert world coordinates to pixel coordinates using WCS information from header
* **`torchfits.pixel_to_world(pixel_coords, header)`**: Convert pixel coordinates to world coordinates using WCS information from header

### Object-Oriented Interface

For more complex file operations, use the object-oriented interface:

#### `torchfits.FITS(filename)`

Context manager for accessing FITS files. Provides access to individual HDUs.

```python
with torchfits.FITS("file.fits") as f:
    num_hdus = len(f)              # Number of HDUs
    primary = f[0]                 # Access by index
    sci_hdu = f['SCI']             # Access by name
    data = primary.read()          # Read HDU data
    header = primary.header        # Get HDU header
```

#### `torchfits.HDU(filename, hdu_spec)`

Represents a single HDU. Has `read()` method and `header` property.

### WCS Functions

* **`torchfits.world_to_pixel(world_coords, header)`**: Converts world coordinates to pixel coordinates.
* **`torchfits.pixel_to_world(pixel_coords, header)`**: Converts pixel coordinates to world coordinates.

### Classes

* **`torchfits.FITS(filename)`**: A context manager for accessing FITS files.
* **`torchfits.HDU(filename, hdu_spec)`**: Represents a single HDU in a FITS file.

## Reading Remote FITS Files

`torchfits` supports reading FITS files from remote locations using `fsspec` parameters. This allows you to read FITS files directly from cloud storage or other remote sources without downloading them first.

### Example

```python
import torchfits

# Define fsspec parameters for reading a remote FITS file
fsspec_params = {
    'protocol': 'https',
    'host': 'example.com',
    'path': 'path/to/remote_file.fits'
}

# Read the remote FITS file
data, header = torchfits.read(fsspec_params)

print(f"Data shape: {data.shape}, Header: {header}")
```

## Remote & Caching

TorchFits includes a production-ready SmartCache that can transparently fetch and cache remote files and also cache decoded tensors/tables when beneficial.

* Environment variable: set TORCHFITS_CACHE to control the cache base directory.
* Public helpers: configure_cache(), get_cache(), and get_cache_manager() for stats and maintenance.
* Protocols: file, http(s), s3, gs (gcs), and other fsspec-backed URIs when fsspec and the relevant plugins are installed.

Quick start:

```python
import os, torchfits as tf
os.environ["TORCHFITS_CACHE"] = os.path.expanduser("~/.cache/torchfits")
tf.configure_cache(max_size_gb=2.0)

url = "https://example.org/data/sample.fits"
data, header = tf.read(url)
print(tf.get_cache_manager().get_cache_statistics())
```

See examples/example_remote_caching.py for a runnable demo.

## Performance knobs

TorchFits includes opt-in switches to experiment with advanced CFITSIO paths:

* TORCHFITS_ENABLE_MMAP=1 or read(..., enable_mmap=True): opt-in memory-mapped full-image read for uncompressed images. Only applies to full-image reads; compressed images or subsets fall back.
* TORCHFITS_ENABLE_BUFFERED=1 — enable buffered/tiled image read fast-paths (no cutouts). This prefers tiled reads for compressed images and scaled reads for uncompressed images. Falls back automatically on any error.

Additional knobs and notes:

- TORCHFITS_SUBSET_FULL_FRAC: when reading a small window (start, shape) from an image, if the requested area is ≤ this fraction of the full image (default 0.05 = 5%) and the full image has ≤ TORCHFITS_SUBSET_FULL_MAX_PIXELS pixels (default 2048x2048), torchfits will read the full image once and slice the window from it. The full image is cached for reuse across repeated cutouts on the same HDU+device.
- TORCHFITS_SUBSET_FULL_MAX_PIXELS: safety cap for the above heuristic (default 4194304 pixels).
- Memory-mapped fast path currently supports only uncompressed, unscaled images (BSCALE=1 and BZERO=0) and only for full-image reads. For other cases torchfits automatically falls back to the buffered/standard path.
* TORCHFITS_PAR_READ=1 — attempt parallel scalar column reads for wide tables (4+ scalar numeric columns).
* TORCHFITS_PIN_MEMORY=1 — allocate pinned host memory for CPU tensors to improve host→GPU transfer via .to(device).

These are safe to toggle off at any time; the default behavior is conservative.

## Parity & Benchmarks Artifacts

You can generate quick project artifacts to track API parity and basic I/O performance:

* API Parity Matrix (symbols implemented vs tests covered):
    * Run: pixi run parity-matrix
    * Output: artifacts/validation/parity_matrix.md (+ JSON sidecar)

* Basic I/O micro-benchmarks (reads example files if present):
    * Run: pixi run benchmark-json
    * Output: artifacts/validation/bench_basic.jsonl

* Buffered vs default image read (micro):
    * Run: pixi run bench-buffered
    * Output: artifacts/validation/bench_buffered.jsonl

* Memory-mapped vs default image read (micro):
    * Run: pixi run bench-mmap
    * Output: artifacts/validation/bench_mmap.jsonl

These assist validation toward the v1.0 roadmap goals.

## Contributing

Contributions are welcome! Traditional GitHub contributions style welcome.

### Compression Support & Tests

TorchFits supports writing compressed FITS images using CFITSIO tile compression (GZIP, RICE, HCOMPRESS) and reading them transparently. By convention, tile‑compressed images are stored in an extension while the primary HDU may be empty. TorchFits detects this and, when you call `read(hdu=0)`, will auto‑advance to the first compressed image extension and return its data.

Why they are skipped by default:

* Some platforms (notably macOS with both Homebrew and certain PyPI wheels installed) may load two OpenMP runtimes, leading to initialization errors or sporadic crashes inside dependency libraries unrelated to `torchfits` core.
* Core (non‑compression) functionality should remain testable and reliable while the environment issue is investigated.

How to run the compression tests locally (opt‑in):

```bash
export TORCHFITS_ENABLE_COMPRESSION_TESTS=1
pixi run test-safe -k compression
```

or with plain pytest:

```bash
TORCHFITS_ENABLE_COMPRESSION_TESTS=1 pytest -k compression
```

If your environment is healthy (single OpenMP runtime), the tests will execute; otherwise you may see a skip or an early diagnostic from `torchfits.openmp_guard`.

Troubleshooting duplicate OpenMP runtime (macOS example):

1. Prefer a single distribution source for scientific packages (either all conda / pixi, or all PyPI wheels). Mixing can introduce another `libomp`.
2. Ensure Homebrew's `libomp` path is not redundantly injected if a wheel already bundles one.
3. Use `otool -L $(python -c 'import torch; import inspect, os; import torchfits; print(torchfits.__file__)')` to inspect linked libraries (advanced).
4. Temporarily unset `KMP_DUPLICATE_LIB_OK` if previously exported; rely on detection instead of forcing acceptance.

Once the broader environment duplication is resolved the default skip may be removed and failures will then guard compression regressions. Compression round‑trip tests assert lossless equality for integer images (GZIP/RICE) and tolerance‑based comparisons for floating point with HCOMPRESS.

## License

This project is licensed under the GPL-2 License - see the LICENSE file for details.

