# Table Reference

FITS tables are columnar catalogs. Read them as Arrow (`table.read`), as a
column → tensor map (`table.read_torch`), or as Polars (`table.read_polars`).

| Destination | Call | Returns |
|---|---|---|
| Arrow table | `table.read` / `table.read_arrow` | `pyarrow.Table` |
| Column → tensor map | `table.read_torch` | `dict[str, torch.Tensor]` |
| Polars | `table.read_polars` | Polars DataFrame-like |

Supports `where=` filters, column projection, streaming, mutations, and
handoff to Polars, DuckDB, Pandas, and PyArrow.

---

## `table.read()`

Read a FITS table as a `pyarrow.Table`, with optional `where=` and column
projection. `table.read_arrow` is the same function under another name.

```python
torchfits.table.read(
    path, hdu=1, columns=None, row_slice=None, rows=None, where=None,
    batch_size=65536, mmap=True, decode_bytes=True, encoding="ascii",
    strip=True, include_fits_metadata=False, apply_fits_nulls=True,
    backend="auto",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdu` | `int` or `str` | `1` | Table HDU index or EXTNAME |
| `columns` | `list[str]` or `None` | `None` | Columns to read (`None` = all) |
| `row_slice` | `slice` or `tuple[int,int]` or `None` | `None` | Row range |
| `rows` | `list[int]` or `None` | `None` | Specific row indices |
| `where` | `str` or `None` | `None` | SQL-like row filter |
| `batch_size` | `int` | `65536` | Read batch size |
| `mmap` | `bool` | `True` | Memory-mapped reads |
| `decode_bytes` | `bool` | `True` | Decode byte-string columns |
| `backend` | `str` | `"auto"` | `"auto"`, `"cpp"`, or `"torch"` |
| `include_fits_metadata` | `bool` | `False` | Preserve FITS column metadata |

**Returns:** `pyarrow.Table`

```python
df = torchfits.table.read(
    "catalog.fits", hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20 AND DEC > 0",
)
print(df.num_rows, df.column_names)
```

---

## `table.read_torch()`

Read selected columns as `torch.Tensor` values.

```python
torchfits.table.read_torch(
    path, hdu=1, columns=None, start_row=1, num_rows=-1,
    device="cpu", mmap="auto", cache_capacity=10,
    handle_cache_capacity=16, fast_header=True, return_header=False,
    where=None,
)
```

**Returns:** `dict[str, torch.Tensor]`

`cache_capacity`, `handle_cache_capacity`, and `fast_header` are accepted for
compatibility but **ignored**.

Optional `where` uses the same predicate dialect as `table.read`. With
`read_torch`, torchfits reads the needed columns (project) and applies a
torch **mask** so only matching rows remain.

```python
cols = torchfits.table.read_torch("catalog.fits", hdu=1, columns=["RA", "DEC"])
bright = torchfits.table.read_torch(
    "catalog.fits", hdu=1, columns=["MAG"], where="MAG < 20"
)
```

For many column reads on one file, use
`torchfits.open_table_reader(path, hdu=1)`.

---

## `table.scan()`

Streaming dataframe scanner yielding `pyarrow.RecordBatch` objects without
materializing the entire table.

```python
torchfits.table.scan(
    path, hdu=1, columns=None, row_slice=None, where=None,
    batch_size=65536, mmap=True, decode_bytes=True, encoding="ascii",
    strip=True, include_fits_metadata=False, apply_fits_nulls=True,
    backend="auto",
)
```

**Yields:** `pyarrow.RecordBatch`

```python
for batch in torchfits.table.scan("survey.fits", hdu=1, batch_size=50_000):
    process(batch)  # pyarrow.RecordBatch
```

!!! info "When to use"
    Use `scan()` when the table is too large to fit in memory, or when you
    want to process rows in streaming fashion. For Polars-specific streaming,
    use `scan_polars()`.

---

## `table.scan_torch()`

Stream dataframe rows as tensor-column chunks.

```python
torchfits.table.scan_torch(
    path, hdu=1, columns=None, row_slice=None, batch_size=65536,
    mmap=True, device="cpu", non_blocking=True, pin_memory=False,
)
```

**Yields:** `dict[str, torch.Tensor]`

```python
for batch in torchfits.table.scan_torch("survey.fits", hdu=1, batch_size=10000):
    # batch: dict[str, torch.Tensor]
    process(batch)
```

---

## `table.reader()`

Open a FITS table as a `pyarrow.RecordBatchReader` for streaming.

```python
torchfits.table.reader(
    path, hdu=1, columns=None, row_slice=None, where=None,
    batch_size=65536, mmap=True, decode_bytes=True, encoding="ascii",
    strip=True, include_fits_metadata=True, apply_fits_nulls=True,
    backend="auto",
)
```

**Returns:** `pyarrow.RecordBatchReader`

---

## `table.write()`

Write a columnar dictionary as a FITS binary or ASCII table.

```python
torchfits.table.write(path, data, *, schema=None, header=None,
                      overwrite=False, extname=None, table_type="binary",
                      quantize=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | Output path |
| `data` | `dict[str, array-like]` | *(required)* | Column name to values |
| `header` | `dict` or `None` | `None` | FITS header key-value pairs |
| `overwrite` | `bool` | `False` | Overwrite existing file |
| `table_type` | `str` | `"binary"` | `"binary"` or `"ascii"` |
| `quantize` | `None` or `str` or `dict` | `None` | Opt-in robust `TFORM=I` + `TSCAL`/`TZERO` for float columns (`"robust"` for all floats, `{"FLUX": "robust"}` per column, or options `{"lo_q","hi_q","keep_zero"}`). Integer columns are left alone. Default keeps native float `TFORM`. |

```python
torchfits.table.write("out.fits", {"RA": ra, "DEC": dec}, overwrite=True)
torchfits.table.write(
    "packed.fits",
    {"ID": ids, "FLUX": flux},
    quantize={"FLUX": "robust"},
    overwrite=True,
)
```

---

## Predicate Pushdown

Pass `where=` to keep only matching rows. How filtering runs depends on the
call:

- **`table.read_torch`:** reads the projected columns, then applies a torch
  **mask**.
- **`table.read` / `table.scan`:** filters in C++ when the table layout
  allows; otherwise filters after the Arrow read.

Use `torchfits.table.read()` or `torchfits.table.scan()` for filtered
catalogs. Root `torchfits.read()` does not take `where=`.

**Supported operators:**

| Operator | Example |
|---|---|
| `=` / `!=` | `where="CLASS = 'star'"` |
| `<` / `>` / `<=` / `>=` | `where="MAG_G < 20"` |
| `AND` / `OR` | `where="MAG_G < 20 AND DEC > 0"` |
| `NOT` | `where="NOT CLASS = 'star'"` |
| `IN (...)` | `where="id IN (1, 2, 3)"` |
| `NOT IN (...)` | `where="id NOT IN (4, 5)"` |
| `BETWEEN ... AND ...` | `where="MAG_G BETWEEN 15 AND 20"` |
| `IS NULL` / `IS NOT NULL` | `where="DEC IS NOT NULL"` |

### Backend Selection

| Backend | Behavior |
|---|---|
| `"auto"` (default) | C++ pushdown when (a) no VLA columns in projection and (b) `backend="cpp"` or (`backend="auto"` and `mmap=True`); Arrow filter otherwise |
| `"cpp"` | C++ row reads as torch tensors, converted to Arrow |
| `"torch"` | `table.scan_torch` chunked path |

### Environment Tuning

Table reads open a private CFITSIO handle per call. Shared metadata and disk
cache roots use the `TORCHFITS_*` variables in [Architecture](architecture.md)
and [Core I/O](api-core-io.md).

---

## Predicate Helpers

The `torchfits.where` module provides predicate parsing and evaluation
outside of table reads.

```python
from torchfits.where import evaluate_where, parse_where_expression

ast = parse_where_expression("MAG_G < 20 AND DEC IS NOT NULL")
mask = evaluate_where(ast, {"MAG_G": magnitudes, "DEC": declinations})
# mask: np.ndarray[bool]
```

Additional public names: `parse_where_literal`, `tokenize_where_expression`,
`normalize_where_syntax`, `where_columns_from_ast`.

---

## In-Place Table Mutation

```python
# Row operations
torchfits.table.append_rows(path, rows, hdu=1)
torchfits.table.insert_rows(path, rows, row=0, hdu=1)
torchfits.table.update_rows(path, rows, row_slice, hdu=1)
torchfits.table.delete_rows(path, row_slice, hdu=1)

# Column operations
torchfits.table.insert_column(path, name, values, hdu=1, index=None)
torchfits.table.replace_column(path, name, values, hdu=1)
torchfits.table.rename_columns(path, {"old_name": "new_name"}, hdu=1)
torchfits.table.drop_columns(path, ["col_a", "col_b"], hdu=1)
```

`insert_column` and `replace_column` accept optional `format`, `unit`,
`dim`, `tnull`, `tscal`, `tzero` metadata kwargs.

---

## Interop

### Polars

Native DataFrame path — FITS table → Polars dataframe in one call.

```python
# One-call FITS to Polars (preserves FITS metadata)
df = torchfits.table.read_polars("catalog.fits", hdu=1)
# df: FITSPolarsFrame — wraps pl.DataFrame with .field_meta, .table_meta

# Streaming FITS to Polars (no full materialization)
for batch in torchfits.table.scan_polars("catalog.fits", hdu=1):
    process(batch)  # pl.DataFrame

# Eager Polars then LazyFrame (materializes once)
lazy = torchfits.table.to_polars("catalog.fits", hdu=1).lazy()

# From table dict
polars_df = torchfits.to_polars(table_dict, decode_bytes=True)
```

!!! tip "True streaming"
    `scan_polars()` yields one DataFrame per batch without building the full
    table. For a LazyFrame over an already-materialized table, use
    `to_polars(...).lazy()` or `pl.concat(scan_polars(...)).lazy()`.

!!! info "rechunk=False default"
    All Polars conversion functions default to `rechunk=False`. Pass
    `rechunk=True` to restore the old chunk-concatenation behavior.

### DuckDB

```python
# Register and query
con = torchfits.table.to_duckdb("catalog.fits", hdu=1, relation_name="tbl")
result = torchfits.table.duckdb_query(
    "catalog.fits",
    "SELECT * FROM tbl WHERE MAG < 20",
    hdu=1,
)
# result: pyarrow.Table (by default)
```

### Arrow and Pandas

```python
arrow_table = torchfits.to_arrow(table_dict, decode_bytes=True)
pandas_df = torchfits.to_pandas(table_dict, decode_bytes=True)
```

---

## Schema

Infer the Arrow schema from FITS TFORM header cards without reading any data.

```python
schema = torchfits.table.schema("catalog.fits", hdu=1)
# schema: pyarrow.Schema
```

---

## Additional Utilities

```python
# PyArrow dataset and scanner
ds = torchfits.table.dataset("catalog.fits", hdu=1)
sc = torchfits.table.scanner("catalog.fits", columns=["RA", "DEC"])

# Parquet export
torchfits.table.write_parquet("out.parquet", "catalog.fits", hdu=1)

# Cache cleanup
torchfits.table.clear_cache()
```

The public constant `torchfits.table.TABLE_BACKENDS` lists recognized table
backend names for callers that select an explicit backend.
