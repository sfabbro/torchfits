# API Reference

`torchfits` owns FITS file I/O: images, HDUs, headers, binary/ASCII tables,
checksums, compression, caching, and table interop.

It does not own WCS, HEALPix, sphere, spectral-domain modelling, datasets, or
training transforms. Those domains are out of scope for torchfits.

## Quick Paths

| Goal | Entry point |
|---|---|
| Read image or table | `torchfits.read(path, hdu=..., return_header=True)` |
| Read N-D array/tensor | `torchfits.read_tensor(path, hdu=0, mmap=True)` |
| Read table only | `torchfits.read_table(path, hdu=1, columns=[...])` |
| Row slice | `torchfits.read_table_rows(path, hdu=1, start_row=1, num_rows=N)` |
| Cutout | `torchfits.read_subset(path, hdu, x1, y1, x2, y2)` |
| Multi-HDU arrays | `torchfits.read_hdus(path, hdus=[0, 1, 2])` |
| Repeated cutouts | `torchfits.open_subset_reader(path, hdu)` |
| Stream table | `torchfits.stream_table(path, chunk_rows=10000)` |
| Write generic | `torchfits.write(path, data, header=None, overwrite=False)` |
| Write tensor | `torchfits.write_tensor(path, tensor, header=None, overwrite=False)` |
| Header only | `torchfits.get_header(path, hdu=0)` |
| Multi-HDU handle | `with torchfits.open(path) as hdul: ...` |
| Table with pushdown | `where=` parameter in `read` / `read_table` / `stream_table` |
| Arrow / Polars / DuckDB | `torchfits.table.to_polars_lazy(...)`, `torchfits.table.to_duckdb(...)` |

## Core I/O

### `read(...)`

```python
data, hdr = torchfits.read("image.fits", hdu="auto", return_header=True)
table = torchfits.read("cat.fits", hdu=1, columns=["RA", "DEC"], where="MAG_G < 20")
```

Unified reader. Auto-detects image or table HDUs when `mode="auto"`.

- `hdu`: integer index, EXTNAME string, `"auto"`, or `None`.
- `mode`: `"auto"`, `"image"`, or `"table"`.
- `mmap`: `True`, `False`, or `"auto"`.
- `return_header=True`: returns `(data, Header)`.
- `where`: SQL-style table predicate pushdown.

### Array and Tensor reads

```python
# Read any N-dimensional array directly as a PyTorch Tensor
data = torchfits.read_tensor("image.fits", hdu=0, device="cpu", mmap=True)
# Apple Silicon: device="mps"; Linux NVIDIA: device="cuda"
sci, wht, msk = torchfits.read_hdus("mef.fits", hdus=["SCI", "WHT", "MASK"])
stamp = torchfits.read_subset("mosaic.fits", 0, 0, 0, 256, 256)

with torchfits.open_subset_reader("mosaic.fits", hdu=0) as reader:
    stamp = reader(0, 0, 256, 256)
```

`read_tensor` options: `device`, `mmap`, `handle_cache`, `fp16`, `bf16`,
`raw_scale`, `return_header`. Requires an explicit integer `hdu` (not `"auto"`).

**GPU integer dtypes:** `read(..., device="cuda", scale_on_device=True)` (default) applies
BSCALE/BZERO on the device. FITS signed-byte and unsigned-integer conventions keep narrow
storage dtypes (int8, uint16, uint32) on H2D; generic scaled pixels still become
`float32` for ML. For fitsio-matching native storage dtypes, use
`read_tensor(..., raw_scale=True)`.

**Training loops:** call `torchfits.cache.optimize_for_dataset(file_paths, avg_file_size_mb=…)`
before `DataLoader` epochs to warm handle caches (see `examples/example_image_dataset.py`).

### Table reads

```python
rows = torchfits.read_table("cat.fits", hdu=1, columns=["RA", "DEC"])
subset = torchfits.read_table_rows(
    "cat.fits",
    hdu=1,
    start_row=1,
    num_rows=1000,
    columns=["RA", "DEC"],
)

for chunk in torchfits.stream_table("cat.fits", hdu=1, chunk_rows=100_000):
    ...
```

### Writes and HDU mutation

```python
torchfits.write("out.fits", image, header={"OBJECT": "M31"}, overwrite=True)
torchfits.write_tensor("out.fits", tensor, header={"OBJECT": "M31"}, overwrite=True)
torchfits.write("cat.fits", {"RA": ra, "DEC": dec}, overwrite=True)

torchfits.insert_hdu(path, data, index=1, header=None, compress=False)
torchfits.replace_hdu(path, hdu, data, header=None, compress=False)
torchfits.delete_hdu(path, hdu)
```

`write_tensor` accepts a single `torch.Tensor` image payload and optional
`compress=True` (or a compression type string). `write` also accepts tensors,
dict tables, or an `HDUList`.

### Checksums

```python
torchfits.write_checksums(path, hdu=0)
result = torchfits.verify_checksums(path, hdu=0)
```

`result` contains `datastatus`, `hdustatus`, and `ok`.

## Handles, HDUs, And Headers

```python
with torchfits.open("mef.fits") as hdul:
    primary = hdul[0]
    sci = hdul["SCI"]
    data = sci.data
    header = sci.header
```

- `TensorHDU`: image HDU with lazy `.data` and `.header`.
- `TableHDU`: in-memory table HDU.
- `TableHDURef`: lazy file-backed table handle.
- `Header`: dict-like FITS header preserving FITS card semantics.

## Table Module

```python
torchfits.table.read(path, hdu=1, columns=None, where=None)
torchfits.table.scan(path, hdu=1, chunk_rows=10000)
torchfits.table.reader(path, hdu=1)
torchfits.table.write(path, data, header=None, overwrite=False)
```

In-place table mutation:

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

Interop:

```python
torchfits.table.to_polars_lazy(path, hdu=1, decode_bytes=True)
torchfits.table.to_duckdb(path, hdu=1, relation_name="tbl", connection=con)
torchfits.table.duckdb_query(path, sql, hdu=1)
torchfits.table.scanner(path, hdu=1, columns=None, where=None)
torchfits.to_arrow(table_dict, decode_bytes=True, vla_policy="list")
torchfits.to_pandas(table_dict, decode_bytes=True, vla_policy="object")
```

Advanced table utilities (optional dependencies may apply):

```python
torchfits.table.schema(path, hdu=1)
torchfits.table.scan_torch(path, hdu=1, batch_size=10000, device="cpu")
torchfits.table.dataset(path, hdu=1, decode_bytes=True)
torchfits.table.write_parquet("out.parquet", "catalog.fits", hdu=1)
```

## Predicate Pushdown

The `where=` parameter filters table rows before data reaches Python.

Supported operators include `=`, `!=`, `<`, `>`, `<=`, `>=`, `AND`, `OR`,
`NOT`, `IN (...)`, `NOT IN (...)`, `BETWEEN ... AND ...`, `IS NULL`, and
`IS NOT NULL`.

```python
torchfits.read("cat.fits", hdu=1, where="MAG_G < 20 AND DEC > 0")
torchfits.table.read("cat.fits", hdu=1, where="MAG_G < 20", backend="auto")
```

### Table read backends

`torchfits.table.read` and `scan` accept `backend=`:

| Backend | Behavior |
|---|---|
| `"auto"` (default) | Prefer fast C++ numpy path; for `where=`, choose Arrow filter vs C++ pushdown from table size and column layout |
| `"cpp_numpy"` | C++ row/table reads materialized through NumPy → Arrow |
| `"torch"` | `torchfits.stream_table` chunked path |

Public constant: `torchfits.table.TABLE_BACKENDS`.

Environment tuning:

| Variable | Default | Effect |
|---|---|---|
| `TORCHFITS_TABLE_SCANNER_THRESHOLD` | `100000` (or `1000` for VLA tables) | Row count below which `where=` uses read-then-filter instead of C++ pushdown |
| `TORCHFITS_TABLE_HANDLE_CACHE` | `1` | Set `0` to disable LRU cache of open FITS file handles |
| `TORCHFITS_TABLE_READER_CACHE` | `1` | Set `0` to disable cached `TableReader` instances per `(path, hdu)` |

## Batch And Cache Utilities

```python
torchfits.read_batch(file_paths, hdu=0, device="cpu")
torchfits.get_batch_info(file_paths)

# Root I/O cache helpers (file handles, metadata, C++ CFITSIO cache)
torchfits.get_cache_performance()
torchfits.clear_file_cache(data=True, handles=True, meta=True, cpp=True)

# Higher-level cache manager (auto-tuning, aggregate stats)
torchfits.cache.configure_for_environment()
torchfits.cache.get_cache_stats()
torchfits.cache.clear_cache()
```

The first I/O call also runs `torchfits.cache.configure_for_environment()` once
at import time. Call it explicitly at startup if you want tuning before any
reads.

## Deprecated aliases (0.5.0b2)

Still supported; prefer the canonical paths above:

- `read_large_table` → `read_table` / `stream_table` / `table.scan`
- `read_fast` → `read` / `read_tensor`
- `read_image` → `read_tensor`

## Limitations

- VLA columns are read via buffered I/O; mmap reads and in-place updates are not
  supported for VLA.
- Scaled table columns are not supported for mmap updates; use the buffered path.
- Non-CPU tensors are copied to host before FITS writes.
- Compressed writes support tensor image payloads; dict HDU payloads for
  compressed writes must contain tensor image data.
- `torchfits` intentionally does not expose WCS/sphere/domain modelling APIs.
