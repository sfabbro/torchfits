# API Reference

## Quick Paths

| Goal | Entry point |
|---|---|
| Read image or table | `torchfits.read(path, hdu=..., return_header=True)` |
| Read image only | `torchfits.read_image(path, hdu=0, mmap=True)` |
| Read table only | `torchfits.read_table(path, hdu=1, columns=[...])` |
| Row slice | `torchfits.read_table_rows(path, hdu=1, start_row=1, num_rows=N)` |
| Cutout | `torchfits.read_subset(path, hdu, x1, y1, x2, y2)` |
| Multi-HDU images | `torchfits.read_hdus(path, hdus=[0,1,2])` |
| Repeated cutouts | `torchfits.open_subset_reader(path, hdu)` |
| Stream table | `torchfits.stream_table(path, chunk_rows=10000)` |
| Write | `torchfits.write(path, data, header=None, overwrite=False)` |
| WCS | `torchfits.get_wcs(path, hdu="auto")` |
| Header only | `torchfits.get_header(path, hdu=0)` |
| Multi-HDU handle | `with torchfits.open(path) as hdul: ...` |
| Table with pushdown | `where=` parameter in `read` / `stream_table` |
| Arrow / Polars / DuckDB | `torchfits.table.to_polars_lazy(...)`, `torchfits.table.to_duckdb(...)` |
| Sphere / HEALPix | See [sphere.md](sphere.md) |

---

## Core I/O

### `read(path, hdu=0, *, mode='auto', device='cpu', mmap='auto', fp16=False, bf16=False, columns=None, start_row=1, num_rows=-1, where=None, cache_capacity=10, handle_cache_capacity=16, fast_header=True, return_header=False)`

Unified reader. Auto-detects image or table based on HDU type.

- `hdu`: integer index, EXTNAME string, or `"auto"` / `None` (first data HDU).
- `mode`: `"auto"` | `"image"` | `"table"`.
- `mmap`: `True` | `False` | `"auto"` (heuristic based on file size).
- `fp16` / `bf16`: lossy downcasting. Avoid for astrometry/photometry.
- `where`: SQL-style predicate pushdown for tables (see [Predicate Pushdown](#predicate-pushdown)).
- `return_header=True`: returns `(data, Header)` instead of just `data`.

```python
data, hdr = torchfits.read("image.fits", device="cuda", return_header=True)
table = torchfits.read("cat.fits", hdu=1, columns=["RA", "DEC"], where="MAG_G < 20")
```

### `read_image(path, hdu=0, *, device='cpu', mmap=True, handle_cache=True, fp16=False, bf16=False, raw_scale=False, return_header=False)`

Low-level image reader. Bypasses auto-detection.

- `raw_scale=True`: return raw integer values without BSCALE/BZERO.

### `read_table(path, hdu=1, *, columns=None, start_row=1, num_rows=-1, device='cpu', mmap=True, return_header=False)`

Table-focused wrapper around `read()`.

### `read_table_rows(path, hdu=1, *, start_row=1, num_rows=1000, columns=None, device='cpu', mmap=True)`

Read a row slice. `start_row` is 1-based (FITS convention).

### `read_hdus(path, hdus, *, device='cpu', mmap=True, return_header=False)`

Read multiple image HDUs in one open.

```python
sci, wht, msk = torchfits.read_hdus("mef.fits", hdus=["SCI", "WHT", "MASK"])
```

### `read_subset(path, hdu, x1, y1, x2, y2, handle_cache_capacity=16)`

Read a rectangular cutout without loading the full image. Coordinates are 0-based.

### `open_subset_reader(path, hdu=0, device='cpu') -> SubsetReader`

Persistent cutout reader. Keeps the FITS handle open for repeated reads.

```python
with torchfits.open_subset_reader("mosaic.fits", hdu=0) as r:
    stamp1 = r(0, 0, 256, 256)
    stamp2 = r(256, 256, 512, 512)
```

### `stream_table(file_path, hdu=1, *, columns=None, start_row=1, num_rows=-1, chunk_rows=10000, mmap=False, max_chunks=None)`

Iterate over a table in row chunks. Yields `dict[str, Tensor | list]`.

### `read_large_table(file_path, hdu=1, *, max_memory_mb=100, streaming=False, return_iterator=False)`

Memory-aware chunked table read.

---

### `write(path, data, header=None, overwrite=False, compress=False)`

Write to FITS.

- `data`: `Tensor` (image), `dict[str, Tensor]` (table), or `HDUList` (multi-HDU).
- `compress`: `False` | `True` (Rice) | `"RICE"` | `"GZIP"` | `"HCOMPRESS"`.

```python
torchfits.write("out.fits", torch.randn(512, 512), header={"OBJECT": "M31"}, overwrite=True)
torchfits.write("cat.fits", {"RA": ra, "DEC": dec}, overwrite=True)
```

### HDU manipulation

```python
torchfits.insert_hdu(path, data, index=1, header=None, compress=False)
torchfits.replace_hdu(path, hdu, data, header=None, compress=False)
torchfits.delete_hdu(path, hdu)
```

### Checksums

```python
torchfits.write_checksums(path, hdu=0)
result = torchfits.verify_checksums(path, hdu=0)
# result: {"datastatus": 1, "hdustatus": 1, "ok": True}
# status values: 1 = ok, 0 = absent, -1 = bad
```

---

### `open(path, mode='r') -> HDUList`

Multi-HDU handle with context management.

```python
with torchfits.open("mef.fits") as hdul:
    hdu = hdul[0]          # by index
    hdu = hdul["SCI"]      # by EXTNAME
    data = hdu.data        # Tensor
    hdr  = hdu.header      # Header
    tbl  = hdul[1]         # TableHDURef (lazy)
```

### `get_header(path, hdu=0) -> Header`

Read header only. `hdu` accepts index, EXTNAME, `"auto"`, or `None`.

### `get_wcs(path, hdu="auto", device=None) -> WCS`

Construct a WCS from a FITS file. See [WCS](#wcs).

---

## HDU Types

### `TensorHDU`

Image HDU. Attributes: `.data` (Tensor, lazy), `.header`, `.wcs`.

### `TableHDU`

In-memory table. Attribute: `.data` (dict[str, Tensor | list]).

Methods: `.filter(condition)` applies a SQL-style predicate and returns a filtered `TableHDU`.

### `TableHDURef`

Lazy file-backed table handle. Returned by `hdul[n]` for table HDUs.

```python
with torchfits.open("cat.fits") as hdul:
    t = hdul[1]               # TableHDURef
    ra = t["RA"]              # read single column
    for chunk in t.iter_rows(batch_size=100_000): ...
    for batch in t.scan_arrow(batch_size=100_000): ...
    full = t.materialize()    # -> TableHDU
```

**In-place file mutations:**
`append_rows_file`, `update_rows_file`, `insert_rows_file`, `delete_rows_file`,
`insert_column_file`, `replace_column_file`, `rename_columns_file`, `drop_columns_file`.

**Query/projection:**
`select(columns)`, `head(n)`, `read(columns=, row_slice=)`, `to_arrow()`, `scan_arrow()`.

### `Header`

Dict-like FITS header. Supports `header["KEY"]`, `header.get("KEY", default)`, `header["KEY"] = val`, `.items()`.

---

## WCS

```python
from torchfits.wcs import WCS

wcs = WCS(header_dict)
wcs = torchfits.get_wcs("image.fits", hdu="auto")

ra, dec = wcs.pixel_to_world(x, y)
x, y = wcs.world_to_pixel(ra, dec)
```

Pure-PyTorch implementation. Runs on CPU, CUDA, or MPS. Follows the same `pixel_to_world` / `world_to_pixel` interface as [astropy.wcs](https://docs.astropy.org/en/stable/wcs/).

**Supported projections (13):** TAN, SIN, ARC, ZPN, ZEA, STG, CEA, CAR, MER, AIT, MOL, HPX, SFL. Distortions: SIP, TPV, TNX, ZPX.

**Not yet implemented:** CYP (forward and inverse raise `NotImplementedError`). Projections not listed above (e.g. BON, PCO, TSC, QSC) are not supported.

Methods: `.pixel_to_world(*args, origin=0)`, `.world_to_pixel(*args, iterative=True, origin=0)`, `.to(device)`, `.compile(**kwargs)`.

Coordinates use 0-based pixel convention. Results are in degrees.

---

## Table Module (`torchfits.table`)

### Read / write

```python
torchfits.table.read(path, hdu=1, columns=None, where=None)
torchfits.table.scan(path, hdu=1, chunk_rows=10000)
torchfits.table.reader(path, hdu=1)
torchfits.table.write(path, data, header=None, overwrite=False)
```

### In-place mutations

```python
torchfits.table.append_rows(path, rows, hdu=1)
torchfits.table.update_rows(path, rows, row_slice, hdu=1)
torchfits.table.insert_rows(path, rows, *, row, hdu=1)
torchfits.table.delete_rows(path, row_slice, hdu=1)
torchfits.table.insert_column(path, name, values, hdu=1, index=None, **meta)
torchfits.table.replace_column(path, name, values, hdu=1)
torchfits.table.rename_columns(path, mapping, hdu=1)
torchfits.table.drop_columns(path, columns, hdu=1)
```

### Interop

```python
torchfits.table.to_polars_lazy(path, hdu=1, decode_bytes=True)     # -> polars.LazyFrame
torchfits.table.to_duckdb(path, hdu=1, relation_name="tbl", connection=con)
torchfits.table.duckdb_query(path, sql, hdu=1)
torchfits.table.scanner(path, hdu=1, columns=None, where=None)     # -> pyarrow.Scanner
torchfits.to_arrow(table_dict, decode_bytes=True, vla_policy="list")
torchfits.to_pandas(table_dict, decode_bytes=True, vla_policy="object")
```

Interop with [Pandas](https://pandas.pydata.org/), [Polars](https://pola.rs/), [DuckDB](https://duckdb.org/), and [PyArrow](https://arrow.apache.org/docs/python/).

### Predicate pushdown

The `where=` parameter filters rows in C++ before data reaches Python.

**Operators:** `=`, `!=`, `<`, `>`, `<=`, `>=`, `AND`, `OR`, `NOT`, `IN (...)`, `NOT IN (...)`, `BETWEEN ... AND ...`, `IS NULL`, `IS NOT NULL`.

```python
torchfits.read("cat.fits", hdu=1, where="MAG_G < 20 AND DEC > 0")
torchfits.read("cat.fits", hdu=1, where="FILTER IN ('g', 'r') AND REDSHIFT BETWEEN 0.1 AND 0.5")
```

### Limitations

- VLA (variable-length array) columns are read via buffered I/O; mmap is not supported for VLA.
- Bit columns (`X` format) and complex columns are not supported for mmap reads or in-place updates.
- Scaled columns (TSCAL/TZERO) are not supported for mmap updates; use the buffered path.
- ASCII tables are read as strings; the main focus is binary table extensions.

---

## Spectral Types (`torchfits.spectral`)

Experimental helpers for 1D spectra and IFU cubes. Not part of the core FITS I/O contract.

### `Spectrum1D`

```python
from torchfits.spectral import Spectrum1D, SpectralAxis, SpectralReader

sp = SpectralReader.read_spectrum_1d("spec.fits", hdu=0,
         flux_col="FLUX", wave_col="WAVELENGTH", ivar_col="IVAR", mask_col="MASK")
sp.wavelength     # Angstroms
sp.frequency      # Hz
sp.error          # 1/sqrt(ivar)
sp.apply_mask()   # masked pixels -> NaN
sp.resample(new_wave_tensor)
```

### `DataCube`

```python
cube = SpectralReader.read_data_cube("cube.fits", hdu=0)
cube.extract_spectrum(y=100, x=100)
cube.collapse_spectral(method="mean")
cube.extract_slice(wavelength=6563.0, width=5)
cube.to(device)
```

### `SpectralAxis`

Fields: `values`, `unit`, `type`, `rest_frequency`, `redshift`. Methods: `.to_wavelength()`, `.to_frequency()`.

---

## Datasets & DataLoaders

### `FITSDataset`

Map-style [PyTorch Dataset](https://pytorch.org/docs/stable/data.html#map-style-datasets) for random-access to FITS images.

```python
from torchfits import FITSDataset
from torchfits.transforms import Compose, ZScale, RandomCrop

ds = FITSDataset(file_paths, hdu="auto",
    transform=Compose([RandomCrop(256), ZScale()]),
    device="cuda", mmap="auto")
sample = ds[0]
```

### `IterableFITSDataset`

Streaming [IterableDataset](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) for large file collections with multi-worker support.

### `TableChunkDataset`

Iterable dataset yielding table row chunks.

### DataLoader helpers

```python
torchfits.create_dataloader(dataset, batch_size=32, num_workers=4, shuffle=True)
torchfits.create_fits_dataloader(file_paths, *, hdu="auto", transform=None, device="cpu",
                                  batch_size=32, num_workers=4)
torchfits.create_table_dataloader(file_paths, *, hdu=1, chunk_rows=10000, columns=None, device="cpu")
```

---

## Transforms (`torchfits.transforms`)

All transforms accept and return a `Tensor`. GPU-safe.

| Transform | Description |
|---|---|
| `ZScale(contrast, max_reject)` | Quantile-based normalization to [0, 1] |
| `AsinhStretch(a, Q)` | Asinh stretch for high dynamic range |
| `LogStretch(a)` | Logarithmic stretch |
| `PowerStretch(gamma)` | Power-law stretch |
| `Normalize(mean, std)` | Z-score normalization (auto-computes if None) |
| `MinMaxScale()` | Scale to [0, 1] |
| `RobustScale(quantile)` | IQR-based normalization |
| `RandomCrop(size)` | Random crop |
| `CenterCrop(size)` | Center crop |
| `RandomFlip(horizontal, vertical, p)` | Random flip |
| `RandomRotation(degrees)` | Random rotation |
| `GaussianNoise(std)` | Additive Gaussian noise |
| `PoissonNoise(scale)` | Poisson noise augmentation |
| `PerturbByError(scale)` | Gaussian noise scaled by error map |
| `RedshiftShift(z_range)` | Spectral redshift augmentation |
| `ToDevice(device)` | Move to device |
| `Compose([...])` | Sequential pipeline |

**Convenience pipelines:** `create_training_transform`, `create_validation_transform`, `create_inference_transform`.

---

## Batch Utilities

```python
torchfits.read_batch(file_paths, hdu=0, device="cpu")  # -> list[Tensor]
torchfits.get_batch_info(file_paths)                    # -> {"num_files": N, "valid_files": M}
```

---

## Cache & Runtime

```python
torchfits.configure_for_environment()     # auto-tune caches
torchfits.get_cache_stats()               # hit/miss stats
torchfits.clear_file_cache(data=True, handles=True, meta=True, cpp=True)
torchfits.clear_cache()
torchfits.clear_buffers()
torchfits.get_buffer_stats()
torchfits.configure_buffers()
```

---

## See Also

- [Sphere & HEALPix reference](sphere.md)
- [Examples](examples.md)
- [Benchmarks](benchmarks.md)
- [Changelog](changelog.md)
