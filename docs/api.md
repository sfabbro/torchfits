# torchfits API Reference

## Quick Paths

| Goal | Entry point |
|---|---|
| Read image/table | `torchfits.read(path, hdu=..., return_header=True)` |
| Read image only | `torchfits.read_image(path, hdu=0, mmap=True)` |
| Read table only | `torchfits.read_table(path, hdu=1, columns=[...])` |
| Row slice | `torchfits.read_table_rows(path, hdu=1, start_row=1, num_rows=N)` |
| Cutout from image | `torchfits.read_subset(path, hdu, x1, y1, x2, y2)` |
| Multi-HDU images | `torchfits.read_hdus(path, hdus=[0,1,2])` |
| Repeated cutouts | `torchfits.open_subset_reader(path, hdu)` |
| Stream table | `torchfits.stream_table(path, chunk_rows=10000)` |
| Write | `torchfits.write(path, data, header=None, overwrite=False)` |
| WCS | `torchfits.get_wcs(path, hdu="auto")` |
| Header only | `torchfits.get_header(path, hdu=0)` |
| Multi-HDU handle | `with torchfits.open(path) as hdul: ...` |
| Table pushdown SQL | `where=` parameter in `read/stream_table` |
| Arrow / Polars | `torchfits.table.to_polars_lazy(...)`, `torchfits.table.to_duckdb(...)` |

---

## Core I/O

### `read(path, hdu=0, *, mode='auto', device='cpu', mmap='auto', fp16=False, bf16=False, columns=None, start_row=1, num_rows=-1, where=None, cache_capacity=10, handle_cache_capacity=16, fast_header=True, return_header=False)`

Unified FITS reader — auto-detects image or table.

- `hdu`: index, EXTNAME string, `"auto"`/`None` (first payload HDU).
- `mode`: `"auto"` | `"image"` | `"table"`.
- `mmap`: `True` | `False` | `"auto"` (heuristic, recommended).
- `fp16` / `bf16`: lossy downcasting — avoid for astrometry/photometry.
- `where`: C++ predicate pushdown, table only (see [Predicate Pushdown](#predicate-pushdown)).
- `return_header=True`: returns `(data, Header)`.

```python
data, hdr = torchfits.read("image.fits", device="cuda", return_header=True)
table = torchfits.read("cat.fits", hdu=1, columns=["RA", "DEC"], where="MAG_G < 20")
```

---

### `read_image(path, hdu=0, *, device='cpu', mmap=True, handle_cache=True, fp16=False, bf16=False, raw_scale=False, return_header=False)`

Explicit low-level image reader — bypasses policy dispatch.

---

### `read_table(path, hdu=1, *, columns=None, start_row=1, num_rows=-1, device='cpu', mmap=True, return_header=False)`

Table-focused convenience wrapper around `read()`.

---

### `read_table_rows(path, hdu=1, *, start_row=1, num_rows=1000, columns=None, device='cpu', mmap=True)`

Read a row slice. `start_row` is 1-based (FITS convention).

---

### `read_hdus(path, hdus, *, device='cpu', mmap=True, return_header=False)`

Read multiple image HDUs from a single file in one open. `hdus` accepts a list of indices or EXTNAME strings.

```python
sci, wht, msk = torchfits.read_hdus("mef.fits", hdus=["SCI", "WHT", "MASK"])
```

---

### `read_subset(path, hdu, x1, y1, x2, y2, handle_cache_capacity=16)`

Read a rectangular image cutout without loading the full HDU. Coordinates are 0-based, `[x1:x2, y1:y2]`.

```python
stamp = torchfits.read_subset("large.fits", hdu=0, x1=1000, y1=1000, x2=1100, y2=1100)
```

### `open_subset_reader(path, hdu=0, device='cpu') → SubsetReader`

Open a persistent cutout reader — keeps the FITS handle open for repeated reads on the same HDU.

```python
with torchfits.open_subset_reader("mosaic.fits", hdu=0) as r:
    stamp1 = r(0, 0, 256, 256)
    stamp2 = r(256, 256, 512, 512)
```

`SubsetReader` also exposes `.hdu`, `.shape`, `.read_subset(x1,y1,x2,y2)`, `.close()`.

---

### `stream_table(file_path, hdu=1, *, columns=None, start_row=1, num_rows=-1, chunk_rows=10000, mmap=False, max_chunks=None)`

Iterate over a table in row chunks. Yields `dict[str, Tensor | list]`.

---

### `read_large_table(file_path, hdu=1, *, max_memory_mb=100, streaming=False, return_iterator=False)`

Memory-aware chunked table read. If `return_iterator=True`, returns a chunk iterator.

---

### `write(path, data, header=None, overwrite=False, compress=False)`

Write to FITS.

- `data`: `Tensor` (image), `dict[str, Tensor]` (table), or `HDUList` (multi-HDU).
- `compress`: `False` | `True` (Rice) | algorithm string (`"RICE"`, `"GZIP"`, `"HCOMPRESS"`).

```python
torchfits.write("out.fits", torch.randn(512, 512), header={"OBJECT": "M31"}, overwrite=True)
torchfits.write("cat.fits", {"RA": ra_tensor, "DEC": dec_tensor}, overwrite=True)
```

---

### HDU Manipulation

```python
torchfits.insert_hdu(path, data, index=1, header=None, compress=False)
torchfits.replace_hdu(path, hdu, data, header=None, compress=False)
torchfits.delete_hdu(path, hdu)
```

---

### Checksums

```python
torchfits.write_checksums(path, hdu=0)   # compute & write DATASUM/CHECKSUM
result = torchfits.verify_checksums(path, hdu=0)
# result: {"datastatus": 1, "hdustatus": 1, "ok": True}
# status: 1=ok, 0=absent, -1=bad
```

---

### `open(path, mode='r') → HDUList`

Multi-HDU handle with context management.

```python
with torchfits.open("mef.fits") as hdul:
    hdu = hdul[0]          # by index
    hdu = hdul["SCI"]      # by EXTNAME
    data = hdu.data        # Tensor
    hdr  = hdu.header      # Header
    tbl  = hdul[1]         # TableHDURef (lazy)
```

---

### `get_header(path, hdu=0) → Header`

Read only the header. `hdu` may be index, EXTNAME, `"auto"`, or `None`.

### `get_wcs(path, hdu="auto", device=None) → WCS`

Construct a `WCS` from a FITS file (see [WCS](#wcs)).

---

## HDU Types

### `TensorHDU`
Image HDU. Key attributes: `.data` (Tensor, lazy), `.header`, `.wcs`.

### `TableHDU`
Materialized in-memory table. Key attribute: `.data` (dict[str, Tensor]).

### `TableHDURef`
Lazy, file-backed table handle. Access from `hdul[n]` for table HDUs.

```python
with torchfits.open("cat.fits") as hdul:
    t = hdul[1]               # TableHDURef
    ra = t["RA"]              # read single column
    for chunk in t.iter_rows(batch_size=100_000): ...
    for batch in t.scan_arrow(batch_size=100_000): ...
    full = t.materialize()    # → TableHDU (loads all into memory)
```

**In-place file-mutation methods** (modify underlying FITS file):
- `append_rows_file(rows)`, `update_rows_file(rows, row_slice)`, `insert_rows_file(rows, row)`, `delete_rows_file(row_slice)`
- `insert_column_file(name, values, index=None, **meta)`, `replace_column_file(name, values)`, `rename_columns_file(mapping)`, `drop_columns_file(columns)`

**Query/projection:**
- `select(columns)` → new `TableHDURef`, `head(n)` → new `TableHDURef`
- `read(columns=None, row_slice=None)`, `to_arrow()`, `scan_arrow()`

### `Header`
Dict-like FITS header. Supports `header["KEY"]`, `header.get("KEY", default)`, `header["KEY"] = val`, `.items()`.

---

## WCS

```python
from torchfits.wcs import WCS

wcs = WCS(header_dict)           # from a dict / FITS header
wcs = torchfits.get_wcs("image.fits", hdu="auto")

ra, dec    = wcs.pixel_to_world(x, y)     # → degrees
x_out, y_out = wcs.world_to_pixel(ra, dec)
```

**Supported projections:** `TAN`, `SIN`, `ARC`, `ZPN`, `STG`, `ZEA`, `CEA`, `MER`, `CYP`, `AIT`, `MOL`, `HPX`, plus distortions `SIP`, `TPV`, `TNX`, `ZPX`.

Key methods: `.pixel_to_world(*args, origin=0)`, `.world_to_pixel(*args, iterative=True, origin=0)`, `.to(device)`, `.compile(**kwargs)`.

Coordinates use 0-based Python pixel convention. Results are in degrees.

---

## Table Module (`torchfits.table`)

### Standalone read/write

```python
torchfits.table.read(path, hdu=1, columns=None, where=None)
torchfits.table.scan(path, hdu=1, chunk_rows=10000)
torchfits.table.reader(path, hdu=1)
torchfits.table.write(path, data, header=None, overwrite=False)
```

### In-place mutations (functional wrappers)

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
torchfits.table.to_polars_lazy(path, hdu=1, decode_bytes=True)  # → polars.LazyFrame
torchfits.table.to_duckdb(path, hdu=1, relation_name="tbl", connection=con)
torchfits.table.duckdb_query(path, sql, hdu=1)
torchfits.table.scanner(path, hdu=1, columns=None, where=None, filter=None)  # → PyArrow scanner
torchfits.to_arrow(table_dict, decode_bytes=True, vla_policy="list")   # → pyarrow.Table
torchfits.to_pandas(table_dict, decode_bytes=True, vla_policy="object") # → pd.DataFrame
```

### Predicate Pushdown

The `where=` parameter routes filtering to the C++ layer before data reaches Python.

**Operators:** `=`, `!=`, `<`, `>`, `<=`, `>=`, `AND`, `OR`, `NOT`, `IN (...)`, `NOT IN (...)`, `BETWEEN ... AND ...`, `IS NULL`, `IS NOT NULL`.

```python
torchfits.read("cat.fits", hdu=1, where="MAG_G < 20 AND DEC > 0")
torchfits.read("cat.fits", hdu=1, where="FILTER IN ('g', 'r') AND REDSHIFT BETWEEN 0.1 AND 0.5")
```

---

## Spectral Types (`torchfits.spectral`)

> Experimental helper layer for 1D spectra and IFU cubes. Not part of the core FITS I/O surface.

### `Spectrum1D`

```python
from torchfits.spectral import Spectrum1D, SpectralAxis, SpectralReader

sp = SpectralReader.read_spectrum_1d("spec.fits", hdu=0,
         flux_col="FLUX", wave_col="WAVELENGTH", ivar_col="IVAR", mask_col="MASK")
# sp.flux, sp.spectral_axis, sp.ivar, sp.mask
sp.wavelength     # Angstroms
sp.frequency      # Hz
sp.error          # 1/sqrt(ivar)
sp.apply_mask()   # → Spectrum1D with masked pixels NaN
sp.resample(new_wave_tensor)
```

### `DataCube`

```python
cube = SpectralReader.read_data_cube("cube.fits", hdu=0)
# cube.data [n_spectral, ny, nx], cube.spectral_axis, cube.spatial_wcs
cube.extract_spectrum(y=100, x=100)           # → Spectrum1D
cube.collapse_spectral(method="mean")          # → 2D Tensor
cube.extract_slice(wavelength=6563.0, width=5) # → 2D Tensor
cube.to(device)
```

### `SpectralAxis`

Fields: `values`, `unit`, `type`, `rest_frequency`, `redshift`. Methods: `.to_wavelength()`, `.to_frequency()`.

---

## Datasets & DataLoaders

### `FITSDataset`

Map-style dataset for random-access to FITS image files.

```python
from torchfits import FITSDataset
from torchfits.transforms import Compose, ZScale, RandomCrop

ds = FITSDataset(
    file_paths,
    hdu="auto",
    transform=Compose([RandomCrop(256), ZScale()]),
    device="cuda",
    mmap="auto",
    include_header=False,
    cache_capacity=0,
    handle_cache_capacity=64,
    raw_scale=False,
)
sample = ds[0]   # Tensor (or (Tensor, Header) if include_header=True)
```

### `TableChunkDataset`

Iterable dataset yielding table row chunks.

```python
torchfits.TableChunkDataset(file_paths, hdu=1, columns=None, chunk_rows=10000,
                             device="cpu", transform=None, include_header=False)
```

### DataLoader helpers

```python
torchfits.create_dataloader(dataset, batch_size=32, num_workers=4, shuffle=True, **kwargs)
torchfits.create_fits_dataloader(file_paths, *, hdu="auto", transform=None, device="cpu",
                                  batch_size=32, num_workers=4, **kwargs)
torchfits.create_table_dataloader(file_paths, *, hdu=1, chunk_rows=10000,
                                   columns=None, device="cpu", **kwargs)
```

---

## Transforms (`torchfits.transforms`)

All transforms are callable objects that accept a `Tensor` and return a `Tensor`. GPU-safe.

| Transform | Signature | Notes |
|---|---|---|
| `ZScale` | `ZScale(contrast=0.25, max_reject=0.5)` | Quantile-based normalization → [0, 1] |
| `AsinhStretch` | `AsinhStretch(a=0.1, Q=8.0)` | Hi-dynamic-range stretch |
| `LogStretch` | `LogStretch(a=1000.0)` | Log stretch |
| `PowerStretch` | `PowerStretch(gamma=2.0)` | Power-law stretch |
| `Normalize` | `Normalize(mean=None, std=None)` | Z-score; auto-computes if None |
| `MinMaxScale` | `MinMaxScale()` | Scales to [0, 1] |
| `RobustScale` | `RobustScale(quantile=0.25)` | IQR-based normalization |
| `RandomCrop` | `RandomCrop(size)` | Random HxW crop |
| `CenterCrop` | `CenterCrop(size)` | Center crop |
| `RandomFlip` | `RandomFlip(horizontal=True, vertical=True, p=0.5)` | Random flip |
| `RandomRotation` | `RandomRotation(degrees=90)` | Random rotation |
| `GaussianNoise` | `GaussianNoise(std=0.01)` | Additive Gaussian noise |
| `PoissonNoise` | `PoissonNoise(scale=1.0)` | Poisson noise augmentation |
| `PerturbByError` | `PerturbByError(scale=1.0)` | Gaussian noise scaled by error map |
| `RedshiftShift` | `RedshiftShift(z_range=(0.0, 0.1))` | Spectral redshift augmentation |
| `ToDevice` | `ToDevice(device)` | Move to device |
| `Compose` | `Compose([t1, t2, ...])` | Sequential pipeline |

**Convenience pipelines:**

```python
from torchfits.transforms import (create_training_transform,
                                   create_validation_transform,
                                   create_inference_transform)

train_t = create_training_transform(crop_size=256, normalize=True, augment=True)
val_t   = create_validation_transform(crop_size=256, normalize=True)
inf_t   = create_inference_transform(normalize=True)
```

---

## Batch Utilities

```python
torchfits.read_batch(file_paths, hdu=0, device="cpu")  # → list[Tensor]
torchfits.get_batch_info(file_paths)                   # → {"num_files": N, "valid_files": M}
```

---

## Cache & Runtime

```python
torchfits.configure_for_environment()          # auto-tune buffers/handles

torchfits.get_cache_stats()                    # Python-side hit/miss stats
torchfits.get_cache_performance()             # extended cache statistics
torchfits.clear_file_cache(data=True, handles=True, meta=True, cpp=True)
torchfits.clear_cache()
torchfits.clear_buffers()
torchfits.get_buffer_stats()
torchfits.configure_buffers()
```

---

## Spherical Geometry & HEALPix

See [sphere.md](sphere.md) for the complete `torchfits.sphere` reference.

---

## See Also

- [README](../README.md) — Quick start and overview
- [Changelog](changelog.md) — Version history
- [Examples](examples.md) — Working code examples
- [Benchmarks](benchmarks.md) — Performance numbers
