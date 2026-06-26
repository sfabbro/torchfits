# API Reference

`torchfits` owns FITS file I/O: images, HDUs, headers, binary/ASCII tables,
checksums, compression, caching, and table interop.

It does not own WCS, HEALPix, sphere, spectral-domain modelling, datasets, or
training transforms. Use `torchsky` for sky-domain tensor models.

## Quick Paths

| Goal | Entry point |
|---|---|
| Read image or table | `torchfits.read(path, hdu=..., return_header=True)` |
| Read image only | `torchfits.read_image(path, hdu=0, mmap=True)` |
| Read table only | `torchfits.read_table(path, hdu=1, columns=[...])` |
| Row slice | `torchfits.read_table_rows(path, hdu=1, start_row=1, num_rows=N)` |
| Cutout | `torchfits.read_subset(path, hdu, x1, y1, x2, y2)` |
| Multi-HDU images | `torchfits.read_hdus(path, hdus=[0, 1, 2])` |
| Repeated cutouts | `torchfits.open_subset_reader(path, hdu)` |
| Stream table | `torchfits.stream_table(path, chunk_rows=10000)` |
| Write | `torchfits.write(path, data, header=None, overwrite=False)` |
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

### Image reads

```python
image = torchfits.read_image("image.fits", hdu=0, device="cpu", mmap=True)
sci, wht, msk = torchfits.read_hdus("mef.fits", hdus=["SCI", "WHT", "MASK"])
stamp = torchfits.read_subset("mosaic.fits", 0, 0, 0, 256, 256)

with torchfits.open_subset_reader("mosaic.fits", hdu=0) as reader:
    stamp = reader(0, 0, 256, 256)
```

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
torchfits.write("cat.fits", {"RA": ra, "DEC": dec}, overwrite=True)

torchfits.insert_hdu(path, data, index=1, header=None, compress=False)
torchfits.replace_hdu(path, hdu, data, header=None, compress=False)
torchfits.delete_hdu(path, hdu)
```

`data` may be a tensor image, a `dict[str, Tensor | list]` table, or an
`HDUList`.

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

## Predicate Pushdown

The `where=` parameter filters table rows before data reaches Python.

Supported operators include `=`, `!=`, `<`, `>`, `<=`, `>=`, `AND`, `OR`,
`NOT`, `IN (...)`, `NOT IN (...)`, `BETWEEN ... AND ...`, `IS NULL`, and
`IS NOT NULL`.

```python
torchfits.read("cat.fits", hdu=1, where="MAG_G < 20 AND DEC > 0")
```

## Batch And Cache Utilities

```python
torchfits.read_batch(file_paths, hdu=0, device="cpu")
torchfits.get_batch_info(file_paths)

torchfits.configure_for_environment()
torchfits.get_cache_stats()
torchfits.clear_file_cache(data=True, handles=True, meta=True, cpp=True)
torchfits.clear_cache()
```

## Limitations

- VLA columns are read via buffered I/O; mmap is not supported for VLA.
- Bit columns and complex columns are not supported for mmap reads or in-place
  updates.
- Scaled columns are not supported for mmap updates; use the buffered path.
- Non-CPU tensors are copied to host before FITS writes.
- Compressed writes support tensor image payloads; dict HDU payloads for
  compressed writes must contain tensor image data.
- `torchfits` intentionally does not expose WCS/sphere/domain modelling APIs.
