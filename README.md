# torchfits

[![PyPI version](https://badge.fury.io/py/torchfits.svg)](https://badge.fury.io/py/torchfits)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

High-performance FITS I/O library for PyTorch. Provides zero-copy tensor operations and native GPU support for astronomical data processing and machine learning workflows.

## Installation

```bash
pip install torchfits
```

Optional TensorFrame support:

```bash
pip install "torchfits[frame]"
```

## Features

- **Fast I/O**: Zero-copy tensor creation from FITS data with SIMD-optimized type conversions
- **Multi-device**: Direct tensor creation on CPU, CUDA, or MPS (Apple Silicon) devices
- **FITS tables**: Read binary tables as dictionaries of tensors with column and row selection
- **WCS support**: Batch coordinate transformations using wcslib with OpenMP parallelization (Optimized for 2D TAN/SIP/TPV)
- **Data transforms**: GPU-accelerated astronomical normalization and augmentation (ZScale, asinh stretch, etc.)
- **Smart caching**: Multi-level caching (L1 memory + L2 disk) for remote files and repeated access
- **FITS compliant**: Built on cfitsio for standards compliance and robust file handling

## Quick Start

### Reading Images

```python
import torchfits

# Read FITS image as PyTorch tensor
data, header = torchfits.read("image.fits", device='cuda', return_header=True)
print(data.shape, data.device)  # torch.Size([2048, 2048]) cuda:0

# Read specific HDU
data, header = torchfits.read("multi.fits", hdu=1, return_header=True)

# Read cutout from large file
cutout = torchfits.read_subset("large.fits", hdu=0, 
                               x1=100, y1=100, x2=200, y2=200)
```

### Reading Tables

```python
# Read FITS table as dictionary of tensors
table, header = torchfits.read("catalog.fits", hdu=1, return_header=True)

# Access columns as tensors
ra = table['RA']      # torch.Tensor
dec = table['DEC']    # torch.Tensor
mag = table['MAG_G']  # torch.Tensor

# Select specific columns and row ranges
table, _ = torchfits.read("catalog.fits", hdu=1,
                          return_header=True,
                          columns=['RA', 'DEC'],
                          start_row=1000, num_rows=5000)
```

Decoding string columns from `TableHDU`:

```python
with torchfits.open("catalog.fits") as hdul:
    table = hdul[1]  # TableHDURef (lazy handle)
    if "NAME" in table.string_columns:
        names = table.get_string_column("NAME")
```

Streaming tables in chunks:

```python
for chunk in torchfits.stream_table("catalog.fits", hdu=1, chunk_rows=10000):
    # chunk is a dict of column tensors (or lists for VLA)
    ...
```

Memory-budgeted streaming iterator:

```python
chunks = torchfits.read_large_table("catalog.fits", hdu=1, streaming=True, return_iterator=True)
for chunk in chunks:
    ...
```

VLA lengths:

```python
with torchfits.open("catalog.fits") as hdul:
    table = hdul[1]
    lengths = table.vla_lengths  # dict column -> list of lengths
```

Streaming table dataset:

```python
from torchfits import TableChunkDataset, create_table_dataloader

dataset = TableChunkDataset(["catalog.fits"], hdu=1, chunk_rows=5000)
dataloader = create_table_dataloader(["catalog.fits"], hdu=1, chunk_rows=5000)
```

Interoperability helpers:

```python
df = torchfits.to_pandas(table, decode_bytes=True)
arrow = torchfits.to_arrow(table, decode_bytes=True)
```

Arrow-native table API (streaming/out-of-core friendly):

```python
# Read as Arrow table
arrow_table = torchfits.table.read("catalog.fits", hdu=1, decode_bytes=True)
# Push down row predicates before materializing result rows.
# Supports comparisons, IN/NOT IN, BETWEEN/NOT BETWEEN, IS NULL/IS NOT NULL,
# plus NOT/AND/OR with parentheses.
arrow_north = torchfits.table.read("catalog.fits", hdu=1, where="DEC > 0")
# FITS null sentinels (TNULLn) are converted to Arrow nulls by default.
# Disable for maximum throughput when you don't need null semantics:
arrow_fast = torchfits.table.read(
    "catalog.fits", hdu=1, decode_bytes=False, apply_fits_nulls=False
)

# Stream Arrow record batches
for batch in torchfits.table.scan(
    "catalog.fits", hdu=1, where="ID >= 2", columns=["ID"], batch_size=100_000
):
    ...

# Build a RecordBatchReader / Arrow dataset (ecosystem-native)
reader = torchfits.table.reader("catalog.fits", hdu=1, batch_size=100_000)
dset = torchfits.table.dataset("catalog.fits", hdu=1)
scanner = torchfits.table.scanner(
    "catalog.fits", hdu=1, columns=["RA", "DEC"], where="DEC > 0", filter=None
)

# Stream directly to accelerator-friendly torch chunks
for chunk in torchfits.table.scan_torch("catalog.fits", hdu=1, batch_size=100_000, device="cuda"):
    ...

# Convert to pandas / polars
df = torchfits.table.to_pandas(arrow_table)
pl_df = torchfits.table.to_polars(arrow_table)
lf = torchfits.table.to_polars_lazy("catalog.fits", hdu=1, decode_bytes=True)

# Use DuckDB for SQL-style joins/group-bys/windows
agg = torchfits.table.duckdb_query(
    "catalog.fits",
    "SELECT AVG(RA) AS ra_mean, COUNT(*) AS n FROM fits_table WHERE DEC > 0",
    hdu=1,
)

# Export stream to parquet without materializing full table
torchfits.table.write_parquet("catalog.parquet", reader, stream=True)

# Schema-first table writing and ASCII tables are now supported via `torchfits.table.write(...)`.
```

Tip: keep `decode_bytes=False` for maximum read throughput and decode only when needed.
Scaled table columns (`TSCALn`/`TZEROn`) are returned as physical values.

Design note: torchfits focuses on FITS-native I/O, schema/null/scaling semantics, and fast Arrow/torch conversion.
For complex dataframe operations (multi-table joins, window functions, advanced group-bys), prefer Polars/DuckDB via these adapters.

Table recipes (recommended pattern):

```python
# 1) Projection/filter pushdown with Arrow scanner
import pyarrow.dataset as ds
scanner = torchfits.table.scanner(
    "catalog.fits",
    hdu=1,
    columns=["OBJID", "RA", "DEC"],
    filter=ds.field("DEC") > 0,
)
north = scanner.to_table()

# 2) Complex expressions with Polars LazyFrame
import polars as pl
lf = torchfits.table.to_polars_lazy("catalog.fits", hdu=1, decode_bytes=True)
summary = (
    lf.filter(pl.col("MAG_G").is_not_null())
    .group_by("BAND")
    .agg(pl.col("MAG_G").mean().alias("mag_g_mean"), pl.len().alias("n"))
    .sort("n", descending=True)
    .collect()
)

# 3) SQL joins/windows with DuckDB
import duckdb
con = duckdb.connect()
torchfits.table.to_duckdb("catalog_a.fits", hdu=1, relation_name="a", connection=con)
torchfits.table.to_duckdb("catalog_b.fits", hdu=1, relation_name="b", connection=con)
joined = con.sql(
    \"\"\"
    SELECT a.OBJID, a.RA, b.CLASS
    FROM a
    JOIN b USING (OBJID)
    WHERE a.DEC > 0
    \"\"\"
).arrow()
```

For a runnable end-to-end version, see `examples/example_table_recipes.py`.

### Writing FITS Files

```python
import torch

# Write tensor as FITS image
data = torch.randn(512, 512)
torchfits.write("output.fits", data, overwrite=True)

# Write with header
header = {'OBJECT': 'M31', 'EXPTIME': 300.0}
torchfits.write("output.fits", data, header=header, overwrite=True)

# Write table
table = {
    'RA': torch.randn(1000),
    'DEC': torch.randn(1000),
    'MAG': torch.randn(1000)
}
torchfits.write("catalog.fits", table, overwrite=True)

# File-level HDU insert/replace/delete APIs are planned for CFITSIO-native implementation.
```

### Data Processing

```python
from torchfits.transforms import ZScale, AsinhStretch, Compose

# Create transformation pipeline
transform = Compose([
    ZScale(),           # Normalize using IRAF ZScale algorithm
    AsinhStretch(),     # Asinh stretch for high dynamic range
])

# Apply transformations on GPU
data, _ = torchfits.read("image.fits", device='cuda', return_header=True)
stretched = transform(data)
```

### Machine Learning Integration

```python
from torchfits import FITSDataset, create_dataloader

# Create dataset
dataset = FITSDataset(
    file_paths=["img1.fits", "img2.fits", ...],
    transform=transform,
    device='cuda'
)

# Create DataLoader compatible with PyTorch training loops
dataloader = create_dataloader(dataset, batch_size=32, num_workers=4)

for batch in dataloader:
    # batch is already on GPU
    output = model(batch)
```

## Performance

Performance measurements on M2 MacBook Air with 2048×2048 float32 images and 1M row tables:

| Operation | torchfits | astropy | fitsio | Best |
|-----------|-----------|---------|--------|------|
| Read 2k×2k image | 3 ms | 45 ms | 5 ms | torchfits (1.7× vs fitsio) |
| Read 1M row table | 12 ms | 850 ms | 15 ms | torchfits (1.3× vs fitsio) |
| WCS transform (1M pts) | 8 ms | 420 ms | N/A | torchfits |

Performance characteristics:
- Zero-copy operations provide consistent speedup across data sizes
- SIMD-optimized type conversions reduce overhead (Optimized for float32)
- Direct GPU placement eliminates host-device transfer for ML workflows
- Competitive with fitsio while providing PyTorch tensor output
- Largest gains over astropy for tables and large arrays

See `benchmarks/` for detailed methodology and scaling behavior.

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.0
- cfitsio (bundled)
- wcslib (system dependency)
- pytorch-frame (optional; required for TensorFrame integration)

## Device Support

- **CPU**: Standard CPU tensors with SIMD acceleration
- **CUDA**: NVIDIA GPU acceleration  
- **MPS**: Apple Silicon GPU (M1/M2/M3) - note some overhead for small workloads

```python
# Specify device when reading
data, _ = torchfits.read("image.fits", device='mps', return_header=True)  # Apple Silicon
data, _ = torchfits.read("image.fits", device='cuda', return_header=True) # NVIDIA GPU
data, _ = torchfits.read("image.fits", device='cpu', return_header=True)  # CPU
```

For more examples, see the `examples/` directory.

## PyTorch-Frame Integration

Seamlessly convert FITS tables to [pytorch-frame](https://pytorch-frame.readthedocs.io/) `TensorFrame` objects for tabular deep learning:

```python
import torchfits

# Read FITS table directly as TensorFrame
tf = torchfits.read_tensor_frame("catalog.fits", hdu=1)

# Or convert from dict
data, header = torchfits.read("catalog.fits", hdu=1, return_header=True)
tf = torchfits.to_tensor_frame(data)

# Use with pytorch-frame models
print(tf.feat_dict)  # {stype.numerical: tensor, stype.categorical: tensor}
print(tf.col_names_dict)  # Column names grouped by semantic type

# Write TensorFrame back to FITS
torchfits.write_tensor_frame("output.fits", tf, overwrite=True)
```

Automatic semantic type inference:
- `float32/float64` → numerical features
- `int32/int16/uint8` → numerical features  
- `int64/bool` → categorical features

See `examples/example_frame.py` for a complete workflow.

## Documentation

- [API Reference](API.md) - Complete API documentation
- [Examples](examples/) - Working examples for common workflows
- [CHANGELOG](CHANGELOG.md) - Version history

## License

GPL-2.0 License. See [LICENSE](LICENSE) for details.

## Citation

If you use torchfits in your research, please cite:

```bibtex
@software{torchfits2025,
  author = {Fabbro, Seb},
  title = {torchfits: High-performance FITS I/O for PyTorch},
  year = {2025},
  url = {https://github.com/sfabbro/torchfits}
}
```

## Acknowledgments

- [cfitsio](https://heasarc.gsfc.nasa.gov/fitsio/) - FITS file I/O library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [nanobind](https://github.com/wjakob/nanobind) - C++/Python bindings
