# torchfits API Reference

Complete API documentation for torchfits.

## Table of Contents

- [Core I/O Functions](#core-io-functions)
- [HDU Classes](#hdu-classes)
- [WCS Functionality](#wcs-functionality)
- [Datasets](#datasets)
- [DataLoaders](#dataloaders)
- [Transforms](#transforms)
- [Utility Functions](#utility-functions)
- [Streaming](#streaming)
- [Interop](#interop)

---
## Streaming

### `stream_table()`

Iterate over a FITS table in row chunks.

```python
for chunk in torchfits.stream_table("catalog.fits", hdu=1, chunk_rows=10000):
    # chunk is a dict of column tensors (or lists for VLA)
    ...
```

**Parameters:**
- `file_path` (str): Path to FITS file
- `hdu` (int): HDU index (default: 1)
- `columns` (list[str] | None): Column names to read (default: all)
- `start_row` (int): 1-based starting row
- `num_rows` (int): Rows to read, -1 = all
- `chunk_rows` (int): Rows per yielded chunk (default: 10000)
- `mmap` (bool): Use memory mapping when available
- `max_chunks` (int | None): Stop after N chunks (default: None)

### `read_large_table()`

Read large FITS table with memory-aware chunking.

```python
data = torchfits.read_large_table("catalog.fits", hdu=1, streaming=True)

# Or return an iterator
chunks = torchfits.read_large_table("catalog.fits", hdu=1, streaming=True, return_iterator=True)
for chunk in chunks:
    ...
```

**Parameters:**
- `file_path` (str): Path to FITS file
- `hdu` (int): HDU index (default: 1)
- `max_memory_mb` (int): Memory budget used to size chunks
- `streaming` (bool): Enable chunked reads
- `return_iterator` (bool): Return an iterator over chunks instead of a dict
  - If `streaming=False`, returns a single-chunk iterator over the full table.

---
## Interop

### `to_pandas()`

Convert a dict of tensors to a Pandas DataFrame.

```python
df = torchfits.to_pandas(table, decode_bytes=True, vla_policy="object")
```

Parameters:
- `decode_bytes` (bool): Decode uint8 string columns into Python strings
- `encoding` (str): String encoding (default: ascii)
- `strip` (bool): Strip trailing spaces/nulls (default: True)
- `vla_policy` (str): "object" or "drop"

### `to_arrow()`

Convert a dict of tensors to a PyArrow Table.

```python
tbl = torchfits.to_arrow(table, decode_bytes=True, vla_policy="list")
```

Parameters:
- `decode_bytes` (bool): Decode uint8 string columns into Python strings
- `encoding` (str): String encoding (default: ascii)
- `strip` (bool): Strip trailing spaces/nulls (default: True)
- `vla_policy` (str): "list" or "drop"

### `torchfits.table.to_polars_lazy()`

Create a Polars `LazyFrame` from FITS table data (via Arrow).

```python
lf = torchfits.table.to_polars_lazy("catalog.fits", hdu=1, decode_bytes=True)
result = lf.filter(pl.col("MAG_G") < 20).group_by("BAND").len().collect()
```

### `torchfits.table.to_duckdb()`

Register FITS table data in DuckDB and return a relation.

```python
rel = torchfits.table.to_duckdb("catalog.fits", hdu=1, relation_name="cat")
```

### `torchfits.table.duckdb_query()`

Execute SQL against FITS table data using DuckDB.

```python
out = torchfits.table.duckdb_query(
    "catalog.fits",
    "SELECT COUNT(*) AS n FROM fits_table WHERE DEC > 0",
    hdu=1,
)
```

### `torchfits.table.write()`

Write a FITS binary table through the CFITSIO-native `torchfits.write` path.

```python
torchfits.table.write(
    "catalog.fits",
    data={"ID": [1, 2], "RA": [10.1, 10.2]},
    header={"EXTNAME": "CATALOG"},
    overwrite=True,
)
```

Notes:
- Schema-first table writing and ASCII-table output are supported via `torchfits.table.write(...)`.

### `torchfits.table.append_rows()`

Append rows in-place to an existing table HDU.

### `torchfits.table.update_rows()`

Update a row slice in-place for selected columns.

### `torchfits.table.rename_columns()`

Rename one or more columns in-place.

### `torchfits.table.drop_columns()`

Drop one or more columns in-place.

### `torchfits.table.insert_rows()`

Insert rows at a 0-based row index (missing columns get deterministic defaults).

### `torchfits.table.delete_rows()`

Delete rows by 0-based index/slice.

### `torchfits.table.insert_column()`

Insert a new column with optional explicit FITS metadata (`format`, `unit`, `dim`, `tnull`, `tscal`, `tzero`).

### `torchfits.table.replace_column()`

Replace an existing column while preserving metadata unless explicitly overridden.

### Table Recipes

Recommended split of responsibilities:
- Use `where="..."` in `read/scan/reader/scanner` for C++ row predicate pushdown
  (comparisons, `IN`/`NOT IN`, `BETWEEN`/`NOT BETWEEN`, `IS NULL`/`IS NOT NULL`,
  and `NOT`/`AND`/`OR` with parentheses).
- Use `torchfits.table.scanner()` for Arrow expression filtering/projection on top of pushed rows.
- Use `torchfits.table.to_polars_lazy()` for expression-heavy analytics.
- Use `torchfits.table.to_duckdb()` / `duckdb_query()` for SQL joins/windows.

```python
import duckdb
import polars as pl
import pyarrow.dataset as ds
import torchfits

# 1) Push down rows + columns in torchfits, then optional Arrow filter
scanner = torchfits.table.scanner(
    "catalog.fits",
    hdu=1,
    columns=["OBJID", "RA", "DEC"],
    where="DEC > 0",
    filter=ds.field("DEC") > 0,
)
north = scanner.to_table()

# 2) Polars LazyFrame for complex expressions
lf = torchfits.table.to_polars_lazy("catalog.fits", hdu=1, decode_bytes=True)
summary = (
    lf.filter(pl.col("MAG_G").is_not_null())
    .group_by("BAND")
    .agg(pl.col("MAG_G").mean().alias("mag_g_mean"), pl.len().alias("n"))
    .collect()
)

# 3) DuckDB SQL for joins/window functions
con = duckdb.connect()
torchfits.table.to_duckdb("catalog_a.fits", hdu=1, relation_name="a", connection=con)
torchfits.table.to_duckdb("catalog_b.fits", hdu=1, relation_name="b", connection=con)
joined = con.sql(
    "SELECT a.OBJID, a.RA, b.CLASS FROM a JOIN b USING (OBJID)"
).arrow()
```


## Core I/O Functions

### `read()`

Read FITS data as PyTorch tensors.

```python
torchfits.read(path, hdu=0, device='cpu', mmap='auto', fp16=False, bf16=False,
               columns=None, start_row=1, num_rows=-1, cache_capacity=10,
               fast_header=True, return_header=False)
```

**Parameters:**
- `path` (str): Path to FITS file or URL
- `hdu` (int | str): HDU index or name (default: 0)
- `device` (str): Target device - 'cpu', 'cuda', 'mps', or 'cuda:N' (default: 'cpu')
- `mmap` (bool | "auto"): Memory mapping mode (default: `"auto"`).
  - `"auto"`: compressed image HDUs default to non-mmap; uncompressed images use latency heuristics
  - `True` / `False`: explicit override
- `fp16` (bool): Convert to half precision (default: False). Not recommended for photometry or astrometry.
- `bf16` (bool): Convert to bfloat16 (default: False). Better for ML than fp16 but still lossy.
- `columns` (list[str] | None): Column names for table reading (default: None = all)
- `start_row` (int): Starting row for tables, 1-based FITS indexing (default: 1)
- `num_rows` (int): Number of rows to read, -1 = all (default: -1)
- `cache_capacity` (int): Max number of cached entries (default: 10)
- `fast_header` (bool): Use fast bulk header parsing (default: True)
- `return_header` (bool): Whether to return header (default: False)

**Returns:**
- If `return_header=True`:
  - For images: `(torch.Tensor, Header)` - Image data and header
  - For tables: `(dict[str, torch.Tensor], Header)` - Dictionary of column tensors and header
- If `return_header=False`:
  - For images: `torch.Tensor`
  - For tables: `dict[str, torch.Tensor | list]`

**Example:**

```python
import torchfits

# Read image - preserves original dtype by default
data, header = torchfits.read("image.fits", device='cuda', return_header=True)
print(data.shape, data.device)  # torch.Size([2048, 2048]) cuda:0

# Read table with column selection for catalogs
table, header = torchfits.read("catalog.fits", hdu=1,
                               columns=['RA', 'DEC', 'MAG'],
                               return_header=True)
print(table['RA'].shape)  # torch.Size([100000])

# Read table row range for processing subsets
table, header = torchfits.read("catalog.fits", hdu=1,
                               start_row=1000, num_rows=5000,
                               return_header=True)
```

**Notes:**
- Data types are preserved from FITS by default for numerical accuracy
- Default `mmap='auto'` is recommended for typical workloads, especially mixed compressed/uncompressed archives
- Avoid `fp16`/`bf16` for astrometry or photometry requiring full precision
- URLs supported via cfitsio (http://, https://, ftp://)

---

### `write()`

Write PyTorch tensors to FITS files.

```python
torchfits.write(path, data, header=None, overwrite=False, compress=False)
```

**Parameters:**
- `path` (str): Output file path
- `data` (torch.Tensor | dict | HDUList): Data to write
  - `torch.Tensor`: Single image HDU
  - `dict[str, torch.Tensor]`: Table HDU
  - `HDUList`: Multiple HDUs
- `header` (Header | dict | None): Optional FITS header (default: None)
- `overwrite` (bool): Overwrite existing file (default: False)
- `compress` (bool): Use tile compression (Rice algorithm) (default: False)

**Example:**

```python
import torch
import torchfits

# Write image
data = torch.randn(512, 512)
torchfits.write("output.fits", data, overwrite=True)

# Write with header
header = {'OBJECT': 'M31', 'EXPTIME': 300.0, 'FILTER': 'g'}
torchfits.write("output.fits", data, header=header, overwrite=True)

# Write table
table = {
    'RA': torch.randn(1000),
    'DEC': torch.randn(1000),
    'MAG_G': torch.randn(1000),
    'MAG_R': torch.randn(1000)
}
torchfits.write("catalog.fits", table, overwrite=True)
```

---

### `insert_hdu()`

Insert an HDU into an existing file at a given index.

### `replace_hdu()`

Replace an HDU by index or `EXTNAME`.

### `delete_hdu()`

Delete an HDU by index or `EXTNAME`.

---

### `read_subset()`

Read a rectangular cutout from an image HDU without loading the full image. Memory-efficient for extracting postage stamps or regions of interest from large survey mosaics.

```python
torchfits.read_subset(path, hdu, x1, y1, x2, y2)
```

**Parameters:**
- `path` (str): Path to FITS file
- `hdu` (int): HDU index
- `x1, y1` (int): Start coordinates (0-based Python indexing, inclusive)
- `x2, y2` (int): End coordinates (0-based Python indexing, exclusive)

**Returns:**
- `torch.Tensor`: Cutout data `[y2-y1, x2-x1]`

**Example:**

```python
# Read 100x100 cutout from large mosaic
cutout = torchfits.read_subset("large.fits", hdu=0,
                               x1=1000, y1=1000, x2=1100, y2=1100)
print(cutout.shape)  # torch.Size([100, 100])
```

**Notes:**
- Uses 0-based Python indexing, not 1-based FITS indexing
- Only loads the requested region into memory
- Coordinates are (x, y) but returned tensor is [height, width]

---

### `open()`

Open FITS file for multi-HDU access with context management.

```python
torchfits.open(path, mode='r')
```

**Parameters:**
- `path` (str): File path
- `mode` (str): File mode - 'r' for read, 'w' for write (default: 'r')

**Returns:**
- `HDUList`: HDU list object

**Example:**

```python
with torchfits.open("multi.fits") as hdul:
    # Access HDUs by index
    primary = hdul[0]
    extension = hdul[1]
    
    # Access HDUs by name
    science = hdul['SCI']
    
    # Get data
    data = primary.data  # torch.Tensor
    
    # Get header
    header = primary.header  # Header object

    # Tables are lazy by default (safe): hdul[ext] returns a TableHDURef handle,
    # which supports streaming / column-projection without materializing the full table.
```

---

### `get_header()`

Read only the header from a FITS file.

```python
torchfits.get_header(path, hdu=0)
```

**Parameters:**
- `path` (str): Path to FITS file
- `hdu` (int | str): HDU index or name (default: 0)

**Returns:**
- `Header`: Header object

**Example:**

```python
header = torchfits.get_header("image.fits")
print(header['NAXIS1'], header['NAXIS2'])
print(header.get('OBJECT', 'Unknown'))
```

---

## HDU Classes

### `HDUList`

Container for multiple HDUs with context management.

**Methods:**

```python
# Access HDUs
hdu = hdul[0]           # By index
hdu = hdul['SCI']       # By name

# Iteration
for hdu in hdul:
    print(hdu.header['EXTNAME'])

# Length
num_hdus = len(hdul)

# Close
hdul.close()

# Write
hdul.write("output.fits", overwrite=True)
```

**Example:**

```python
with torchfits.open("multi.fits") as hdul:
    print(f"File has {len(hdul)} HDUs")
    
    for i, hdu in enumerate(hdul):
        print(f"HDU {i}: {hdu.header.get('EXTNAME', 'PRIMARY')}")
        if hasattr(hdu, 'data'):
            print(f"  Shape: {hdu.data.shape}")
```

---

### `TensorHDU`

Image HDU with lazy loading and WCS support.

**Attributes:**
- `data` (torch.Tensor): Image data (lazy loaded)
- `header` (Header): FITS header
- `wcs` (WCS | None): WCS object if WCS keywords present

**Methods:**

```python
# Convert to tensor with device placement
tensor = hdu.to_tensor(device='cuda')

# Get WCS if available
if hdu.wcs:
    world = hdu.wcs.pixel_to_world(pixels)
```

**Example:**

```python
with torchfits.open("image.fits") as hdul:
    hdu = hdul[0]
    
    # Lazy access - data not loaded yet
    print(hdu.header['NAXIS1'])
    
    # Load data
    data = hdu.data  # torch.Tensor on CPU
    
    # Or load directly to GPU
    data_gpu = hdu.to_tensor(device='cuda')
```

---

### `TableHDU`

Materialized table (in-memory) for tabular data.

**Attributes:**
- `data` (dict[str, torch.Tensor]): Table data as dictionary
- `header` (Header): FITS header

**Example:**

```python
with torchfits.open("catalog.fits") as hdul:
    table_ref = hdul[1]  # TableHDURef (lazy handle)

    # Column access reads only that column
    ra = table_ref["RA"]

    # Materialize explicitly (loads full table into memory)
    table_hdu = table_ref.materialize()
    dec = table_hdu["DEC"]
```

---

### `TableHDURef`

Lazy, file-backed table handle returned by `torchfits.open(...)` for table HDUs.

Useful for out-of-core workflows:

```python
with torchfits.open("catalog.fits") as hdul:
    t = hdul[1]  # TableHDURef

    # Stream in bounded memory
    for chunk in t.iter_rows(batch_size=100_000):
        ...

    # Arrow-native streaming
    for batch in t.scan_arrow(batch_size=100_000):
        ...
```

To load the full table: `t.materialize()`.

### `Header`

FITS header dictionary with FITS-specific methods.

**Methods:**

```python
# Dictionary-like access
value = header['KEYWORD']
value = header.get('KEYWORD', default)

# Set values
header['KEYWORD'] = value

# Iteration
for key, value in header.items():
    print(f"{key} = {value}")
```

**Example:**

```python
_, header = torchfits.read("image.fits", return_header=True)

# Access values
naxis1 = header['NAXIS1']
object_name = header.get('OBJECT', 'Unknown')

# Modify
header['COMMENT'] = 'Processed with torchfits'

# Write back
torchfits.write("output.fits", data, header=header, overwrite=True)
```

---

## WCS Functionality

### `WCS`

World Coordinate System transformations with batch processing. Supports standard projections (TAN, SIN, etc.) via wcslib.

**Methods:**

```python
wcs.pixel_to_world(pixels, batch_size=None)
wcs.world_to_pixel(coords, batch_size=None)
```

**Example:**

```python
import torch

with torchfits.open("image.fits") as hdul:
    wcs = hdul[0].wcs
    
    if wcs:
        # Transform pixel coordinates to world (RA, DEC in degrees)
        pixels = torch.tensor([[100.0, 200.0],
                               [300.0, 400.0]])
        world = wcs.pixel_to_world(pixels)
        print(world)  # [[ra1, dec1], [ra2, dec2]] in degrees
        
        # Transform back to pixels
        pixels_back = wcs.world_to_pixel(world)
        
        # Batch processing for catalogs or large coordinate lists
        large_pixels = torch.randn(1000000, 2)
        world_coords = wcs.pixel_to_world(large_pixels, batch_size=10000)
```

**Notes:**
- WCS automatically detected from header keywords (CTYPE, CRVAL, CRPIX, CD/CDELT)
- Batch processing uses OpenMP parallelization
- Coordinates use FITS 1-based convention internally but accept 0-based Python coordinates
- Returns sky coordinates in degrees

---

## PyTorch-Frame Integration

Convert FITS tables to [pytorch-frame](https://pytorch-frame.readthedocs.io/) TensorFrame objects for tabular deep learning.

### `read_tensor_frame()`

Read a FITS table directly as a TensorFrame.

```python
torchfits.read_tensor_frame(path, hdu=1, columns=None)
```

**Parameters:**
- `path` (str): Path to FITS file
- `hdu` (int): HDU index (default: 1 for first table extension)
- `columns` (list[str] | None): Optional list of columns to read

**Returns:**
- `TensorFrame`: pytorch-frame TensorFrame object with automatic semantic type inference

**Example:**

```python
import torchfits
from torch_frame import stype

# Read catalog as TensorFrame  
tf = torchfits.read_tensor_frame("catalog.fits", hdu=1)

# Access features by semantic type
print(tf.feat_dict[stype.numerical])    # Numerical features
print(tf.feat_dict[stype.categorical])  # Categorical features
print(tf.col_names_dict)  # Column names grouped by type
```

**Semantic Type Inference:**
- `float32`, `float64`, `int32`, `int16`, `uint8` → numerical
- `int64`, `bool` → categorical
- String columns currently skipped

### `to_tensor_frame()`

Convert a dictionary of tensors (from `read()`) to TensorFrame.

```python
torchfits.to_tensor_frame(data)
```

**Parameters:**
- `data` (dict[str, torch.Tensor]): Dictionary of column tensors

**Returns:**
- `TensorFrame`: pytorch-frame TensorFrame object

**Example:**

```python
# Read table first
data, header = torchfits.read("catalog.fits", hdu=1, return_header=True)

# Convert to TensorFrame
tf = torchfits.to_tensor_frame(data)

# Use with pytorch-frame models
from torch_frame.nn import LinearEncoder
encoder = LinearEncoder(tf.num_features)
```

### `write_tensor_frame()`

Write a TensorFrame back to a FITS file.

```python
torchfits.write_tensor_frame(path, tf, overwrite=False)
```

**Parameters:**
- `path` (str): Output file path
- `tf` (TensorFrame): TensorFrame object to write
- `overwrite` (bool): Whether to overwrite existing files

**Example:**

```python
# Read, process, and write back
tf = torchfits.read_tensor_frame("input.fits", hdu=1)
# ... process tf ...
torchfits.write_tensor_frame("output.fits", tf, overwrite=True)
```

**Notes:**
- Automatically converts numerical and categorical features back to appropriate FITS column types
- Preserves column names and data types
- Useful for saving preprocessed features or model predictions

---

## Datasets

### `FITSDataset`

Map-style dataset for random access to FITS files.

```python
torchfits.FITSDataset(file_paths, hdu=0, transform=None, device='cpu')
```

**Parameters:**
- `file_paths` (list[str]): List of FITS file paths
- `hdu` (int): HDU index to read (default: 0)
- `transform` (callable | None): Optional transform function (default: None)
- `device` (str): Target device (default: 'cpu')
- `include_header` (bool): Return `(data, header)` tuples (default: False)

**Example:**

```python
from torchfits import FITSDataset
from torchfits.transforms import ZScale

# Create dataset
files = ["img1.fits", "img2.fits", "img3.fits"]
dataset = FITSDataset(files, transform=ZScale(), device='cuda')

# Access samples
sample = dataset[0]  # torch.Tensor on CUDA

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

---

### `IterableFITSDataset`

Streaming dataset for large-scale data.

```python
torchfits.IterableFITSDataset(index_url, hdu=0, transform=None, 
                              device='cpu', shard_size=1000)
```

**Parameters:**
- `index_url` (str): URL to sharded index file
- `hdu` (int): HDU index (default: 0)
- `transform` (callable | None): Optional transform (default: None)
- `device` (str): Target device (default: 'cpu')
- `shard_size` (int): Shard size for distributed processing (default: 1000)

**Example:**

```python
from torchfits import IterableFITSDataset

dataset = IterableFITSDataset("https://example.com/index.json",
                              transform=transform,
                              device='cuda')

for sample in dataset:
    # Process streaming data
    process(sample)
```

---

### `TableChunkDataset`

Iterable dataset yielding table chunks.

```python
torchfits.TableChunkDataset(file_paths, hdu=1, chunk_rows=10000)
```

**Parameters:**
- `file_paths` (list[str]): FITS files to stream
- `hdu` (int): Table HDU index (default: 1)
- `columns` (list[str] | None): Column names (default: all)
- `chunk_rows` (int): Rows per chunk (default: 10000)
- `max_chunks` (int | None): Stop after N chunks (default: None)
- `mmap` (bool): Use memory mapping if available
- `device` (str): Target device (default: 'cpu')
- `transform` (callable | None): Optional transform
- `include_header` (bool): Return `(chunk, header)` tuples

---

## DataLoaders

### `create_dataloader()`

Create optimized DataLoader for FITS data.

```python
torchfits.create_dataloader(dataset, batch_size=32, num_workers=4, 
                            shuffle=True, **kwargs)
```

**Parameters:**
- `dataset` (Dataset): FITS dataset
- `batch_size` (int): Batch size (default: 32)
- `num_workers` (int): Number of worker processes (default: 4)
- `shuffle` (bool): Shuffle data (default: True)
- `**kwargs`: Additional arguments passed to DataLoader

**Returns:**
- `DataLoader`: Configured PyTorch DataLoader

**Example:**

```python
from torchfits import FITSDataset, create_dataloader

dataset = FITSDataset(file_paths, device='cuda')
loader = create_dataloader(dataset, batch_size=32, num_workers=4)

for batch in loader:
    # batch is already on GPU
    output = model(batch)
```

---

### `create_table_dataloader()`

Create a DataLoader that yields table chunks.

```python
torchfits.create_table_dataloader(file_paths, hdu=1, chunk_rows=10000)
```

## Transforms

All transforms work on GPU and can be composed.

### `ZScale`

Automatic image normalization using the IRAF ZScale algorithm. Determines display range based on image statistics while rejecting outliers.

```python
torchfits.transforms.ZScale(contrast=0.25, max_reject=0.5)
```

**Parameters:**
- `contrast` (float): Contrast parameter (default: 0.25)
- `max_reject` (float): Maximum fraction of pixels to reject (default: 0.5)

**Example:**

```python
from torchfits.transforms import ZScale

transform = ZScale()
data, _ = torchfits.read("image.fits", device='cuda', return_header=True)
normalized = transform(data)  # Normalized to [0, 1]
```

**Notes:**
- Implements the IRAF/DS9 ZScale algorithm
- Automatically determines appropriate display range rejecting cosmic rays and bad pixels
- Output range is [0, 1]

---

### `AsinhStretch`

Asinh (inverse hyperbolic sine) stretch for high dynamic range images. Commonly used for displaying images with bright cores (e.g., galaxies, star clusters) while preserving faint features.

```python
torchfits.transforms.AsinhStretch(a=0.1, Q=8.0)
```

**Parameters:**
- `a` (float): Softening parameter (default: 0.1)
- `Q` (float): Stretch parameter controlling contrast (default: 8.0)

**Example:**

```python
from torchfits.transforms import AsinhStretch

stretch = AsinhStretch(Q=10.0)
stretched = stretch(data)
```

**Notes:**
- Linear for faint pixels, logarithmic for bright pixels
- Higher Q = more contrast in faint regions
- Commonly used in survey imaging (SDSS, HST, etc.)

---

### `Normalize`

Standard normalization (mean=0, std=1).

```python
torchfits.transforms.Normalize(mean=None, std=None)
```

**Example:**

```python
from torchfits.transforms import Normalize

# Auto-compute mean and std
norm = Normalize()
normalized = norm(data)

# Use specific values
norm = Normalize(mean=0.5, std=0.2)
normalized = norm(data)
```

---

### `RandomCrop`

Random crop for data augmentation.

```python
torchfits.transforms.RandomCrop(size)
```

**Parameters:**
- `size` (int | tuple[int, int]): Output size (height, width) or single int for square

**Example:**

```python
from torchfits.transforms import RandomCrop

crop = RandomCrop(224)
cropped = crop(data)  # Random 224x224 crop
```

---

### `CenterCrop`

Center crop transformation.

```python
torchfits.transforms.CenterCrop(size)
```

**Example:**

```python
from torchfits.transforms import CenterCrop

crop = CenterCrop((256, 256))
cropped = crop(data)  # Center 256x256 crop
```

---

### `RandomFlip`

Random flip for augmentation.

```python
torchfits.transforms.RandomFlip(horizontal=True, vertical=True, p=0.5)
```

**Example:**

```python
from torchfits.transforms import RandomFlip

flip = RandomFlip(p=0.5)
flipped = flip(data)
```

---

### `GaussianNoise`

Add Gaussian noise.

```python
torchfits.transforms.GaussianNoise(std=0.01, snr_based=False)
```

**Example:**

```python
from torchfits.transforms import GaussianNoise

noise = GaussianNoise(std=0.02)
noisy = noise(data)
```

---

### `ToDevice`

Move tensor to specified device.

```python
torchfits.transforms.ToDevice(device)
```

**Example:**

```python
from torchfits.transforms import ToDevice

to_gpu = ToDevice('cuda')
data_gpu = to_gpu(data)
```

---

### `Compose`

Compose multiple transformations.

```python
torchfits.transforms.Compose(transforms)
```

**Example:**

```python
from torchfits.transforms import Compose, ZScale, AsinhStretch, RandomCrop

transform = Compose([
    RandomCrop(512),
    ZScale(),
    AsinhStretch(),
])

data, _ = torchfits.read("image.fits", device='cuda', return_header=True)
transformed = transform(data)
```

---

### Convenience Functions

```python
# Pre-configured transform pipelines
torchfits.transforms.create_training_transform(crop_size=224, normalize=True, augment=True)
torchfits.transforms.create_validation_transform(crop_size=224, normalize=True)
torchfits.transforms.create_inference_transform(normalize=True)
```

**Example:**

```python
from torchfits.transforms import create_training_transform

train_transform = create_training_transform(crop_size=256, augment=True)
val_transform = create_validation_transform(crop_size=256)

# Use in dataset
train_dataset = FITSDataset(train_files, transform=train_transform)
val_dataset = FITSDataset(val_files, transform=val_transform)
```

---

## Utility Functions

### Cache Management

```python
# Configure cache for environment
torchfits.configure_for_environment()

# Get cache statistics
stats = torchfits.get_cache_stats()
print(stats)  # {'hits': 100, 'misses': 20, ...}

# Clear cache
torchfits.clear_cache()
torchfits.clear_file_cache()
```

### Buffer Management

```python
# Configure buffers
torchfits.configure_buffers()

# Get buffer statistics
stats = torchfits.get_buffer_stats()

# Clear buffers
torchfits.clear_buffers()
```

### Batch Operations

```python
# Read multiple files
tensors = torchfits.read_batch(file_paths, device='cuda')

# Get batch info
info = torchfits.get_batch_info(file_paths)
print(info)  # {'num_files': 100, 'valid_files': 98}

# Get cache performance
perf = torchfits.get_cache_performance()
print(perf)  # {'hit_rate': 0.85, 'miss_rate': 0.15, ...}
```

---

## Complete Example

```python
import torch
import torchfits
from torchfits import FITSDataset, create_dataloader
from torchfits.transforms import Compose, ZScale, RandomCrop, RandomFlip

# Define transforms
transform = Compose([
    RandomCrop(256),
    RandomFlip(p=0.5),
    ZScale(),
])

# Create dataset
files = [f"image_{i:04d}.fits" for i in range(1000)]
dataset = FITSDataset(files, transform=transform, device='cuda')

# Create dataloader
loader = create_dataloader(dataset, batch_size=32, num_workers=4)

# Training loop
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for batch in loader:
        # batch is already on GPU and transformed
        output = model(batch)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## See Also

- [README](README.md) - Quick start and overview
- [CHANGELOG](CHANGELOG.md) - Version history and changes
- [Examples](examples/) - Working code examples
