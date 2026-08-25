# Table Reference

FITS tables are columnar catalogs. Read them as Arrow (`table.read`), as a
column → tensor map (`table.read_torch`), or as Polars (`table.read_polars`).

| Destination | Call | Returns |
|---|---|---|
| Arrow table | `table.read` / `table.read_arrow` | `pyarrow.Table` |
| Column → tensor map | `table.read_torch` | `dict[str, torch.Tensor]` (VLA columns use list/tuple values) |
| Astropy Table | `table.read_astropy` | `astropy.table.Table` |
| Polars | `table.read_polars` | `FITSPolarsFrame` wrapper around `pl.DataFrame` |

Supports `where=` filters, column projection, streaming, mutations, and
handoff to Astropy, Polars, DuckDB, Pandas, and PyArrow.

---

## `table.read()`

Read a FITS table as a `pyarrow.Table`, with optional `where=` and column
projection. `table.read_arrow` is the same function under another name.

```python
torchfits.table.read(
    path,
    hdu=1,
    columns=None,
    row_slice=None,
    rows=None,
    where=None,
    batch_size=65536,
    mmap=True,
    decode_bytes=True,
    encoding="ascii",
    strip=True,
    include_fits_metadata=False,
    apply_fits_nulls=True,
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
| `where` | `str` or `None` | `None` | SQL-like row filter (full dialect) |
| `batch_size` | `int` | `65536` | Read batch size |
| `mmap` | `bool` | `True` | Memory-mapped reads |
| `decode_bytes` | `bool` | `True` | Decode byte-string columns |
| `encoding` | `str` | `"ascii"` | Byte decoding encoding |
| `strip` | `bool` | `True` | Strip trailing spaces on strings |
| `include_fits_metadata` | `bool` | `False` | Attach FITS column metadata on the Arrow schema |
| `apply_fits_nulls` | `bool` | `True` | Honor `TNULL` as nulls |
| `backend` | `str` | `"auto"` | `"auto"`, `"cpp"`, or `"torch"` |

**Returns:** `pyarrow.Table`

```python
df = torchfits.table.read(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20 AND DEC > 0",
)
print(df.num_rows, df.column_names)
```

---

## `table.read_torch()`

Read selected columns as `torch.Tensor` values
(`dict[str, torch.Tensor]`).

```python
torchfits.table.read_torch(
    path,
    hdu=1,
    columns=None,
    start_row=1,
    num_rows=-1,
    device="cpu",
    mmap="auto",
    cache_capacity=10,
    handle_cache_capacity=16,
    fast_header=True,
    return_header=False,
    where=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdu` | `int` or `str` | `1` | Table HDU index or EXTNAME |
| `columns` | `list[str]` or `None` | `None` | Columns to return (`None` = all) |
| `start_row` | `int` | `1` | 1-based first row |
| `num_rows` | `int` | `-1` | Row count (`-1` = through end) |
| `device` | `str` | `"cpu"` | `"cpu"`, `"cuda"`, `"mps"` |
| `mmap` | `bool` or `str` | `"auto"` | `True` / `False` / `"auto"` (`"auto"` → mmap on) |
| `cache_capacity` | `int` | `10` | Accepted, **ignored** |
| `handle_cache_capacity` | `int` | `16` | Accepted, **ignored** |
| `fast_header` | `bool` | `True` | Accepted, **ignored** |
| `return_header` | `bool` | `False` | Also return the HDU `Header` |
| `where` | `str` or `None` | `None` | Simple numeric filter (see below) |

**Returns:** `dict[str, torch.Tensor]` (scalar columns; VLA columns use
list/tuple values), or `(dict, Header)` when `return_header=True`.

#### Scalar-column shapes

FITS scalar columns (repeat == 1) are returned as rank-1 tensors of shape
`(N,)` on every access path: `table.read_torch`, `torchfits.read`,
`hdul[n].data[col]`, `hdul[n][col]`, `TableHDURef`, `iter_rows`, and
streaming chunks. Vector columns (repeat > 1) keep `(N, repeat)`; packed
string columns stay `(N, width)` uint8 matrices.

`where=` on `read_torch` accepts only **simple** predicates: comparisons
(`==`, `!=`, `<`, `<=`, `>`, `>=`), `BETWEEN`, and `AND` of those. Expressions
with `OR`, `IN`, `IS NULL`, or `NOT` raise `ValueError` — use
`table.read(..., where=...)` for the full dialect. Matching rows are kept by
reading the needed columns and applying a torch mask. When combined with a
row window (`start_row` / `num_rows`), the window selects file rows first and
the predicate filters inside it — identical to `table.read(row_slice=..., where=...)`.

```python
cols = torchfits.table.read_torch("catalog.fits", hdu=1, columns=["RA", "DEC"])
bright = torchfits.table.read_torch(
    "catalog.fits", hdu=1, columns=["MAG"], where="MAG < 20"
)
```

For many unfiltered column reads on one file, use
`torchfits.open_table_reader(path, hdu=1)` (no `where=` on the handle).

---

## `table.scan()`

Streaming dataframe scanner yielding `pyarrow.RecordBatch` objects without
materializing the entire table.

```python
torchfits.table.scan(
    path,
    hdu=1,
    columns=None,
    row_slice=None,
    where=None,
    batch_size=65536,
    mmap=True,
    decode_bytes=True,
    encoding="ascii",
    strip=True,
    include_fits_metadata=False,
    apply_fits_nulls=True,
    backend="auto",
)
```

**Yields:** `pyarrow.RecordBatch`

```python
for batch in torchfits.table.scan("survey.fits", hdu=1, batch_size=50_000):
    print(batch.num_rows, batch.column_names)  # pyarrow.RecordBatch
```

!!! info "When to use"
    Use `scan()` when the table is too large to fit in memory, or when you
    want to process rows in streaming fashion. With `where=`, the predicate is
    evaluated **per batch as rows stream past**, so peak memory stays bounded
    by `batch_size`, not table size. For Polars-specific streaming,
    use `scan_polars()`.

---

## `table.scan_torch()`

Stream row chunks as `dict[str, torch.Tensor]`. No `where=` — filter with
`table.scan(..., where=...)` or mask batches yourself.

```python
torchfits.table.scan_torch(
    path,
    hdu=1,
    columns=None,
    row_slice=None,
    batch_size=65536,
    mmap=True,
    device="cpu",
    non_blocking=True,
    pin_memory=False,
)
```

**Yields:** `dict[str, torch.Tensor]`

```python
for batch in torchfits.table.scan_torch("survey.fits", hdu=1, batch_size=10000):
    print(batch.keys())  # dict[str, torch.Tensor]
```

---

## `table.reader()`

Open a FITS table as a `pyarrow.RecordBatchReader` for streaming.

```python
torchfits.table.reader(
    path,
    hdu=1,
    columns=None,
    row_slice=None,
    where=None,
    batch_size=65536,
    mmap=True,
    decode_bytes=True,
    encoding="ascii",
    strip=True,
    include_fits_metadata=True,
    apply_fits_nulls=True,
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
| `data` | `dict`, `Table`, or DataFrame | *(required)* | Column dictionary, Astropy Table, PyArrow Table, or DataFrame |
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

## Row filters (`where=`)

Pass `where=` to keep matching rows.

| Call | Dialect | How it filters |
|---|---|---|
| `table.read` / `table.scan` | Full (see operators below) | C++ mmap scan when safe (`backend="auto"` + `mmap=True`, no VLA in the projection); otherwise read then Arrow filter |
| `table.read_torch` | Simple only: compare / `BETWEEN` / `AND` | Read projected columns, then torch mask. `OR` / `IN` / `IS NULL` / `NOT` raise `ValueError` |

Root `torchfits.read()` has no `where=` parameter.

**Operators on `table.read` / `table.scan`:**

| Operator | Example |
|---|---|
| `==` / `!=` | `where="CLASS == 'star'"` |
| `<` / `>` / `<=` / `>=` | `where="MAG_G < 20"` |
| `AND` / `OR` | `where="MAG_G < 20 AND DEC > 0"` |
| `NOT` | `where="NOT CLASS == 'star'"` |
| `IN (...)` | `where="id IN (1, 2, 3)"` |
| `NOT IN (...)` | `where="id NOT IN (4, 5)"` |
| `BETWEEN ... AND ...` | `where="MAG_G BETWEEN 15 AND 20"` |
| `IS NULL` / `IS NOT NULL` | `where="DEC IS NOT NULL"` |

!!! note "Use `==` for equality"
    The WHERE dialect is Python-expression based — single `=` is not an
    operator; write `==` (also accepts `&&` / `||` / `~` C-style forms).

### `backend=` on `table.read` / `table.scan`

| Value | Behavior |
|---|---|
| `"auto"` (default) | With `where=` and `mmap=True`, C++ pushdown when the header is readable and the projection has no VLA columns; otherwise Arrow filter. With `mmap=False`, buffered read then filter. |
| `"cpp"` | Prefer C++ column / pushdown paths when safe |
| `"torch"` | Chunked path via `table.scan_torch` |

Table reads open a private CFITSIO handle per call. Disk cache roots and
shared metadata use the `TORCHFITS_*` variables in
[Architecture](architecture.md) and [Core I/O](api-core-io.md).

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

## Table mutation

Rewrite helpers on an existing table HDU:

```python
torchfits.table.append_rows(path, rows, hdu=1)
torchfits.table.insert_rows(path, rows, row=0, hdu=1)
torchfits.table.update_rows(path, rows, row_slice, hdu=1, mmap="auto")
torchfits.table.delete_rows(path, row_slice, hdu=1)

torchfits.table.insert_column(
    path,
    name,
    values,
    hdu=1,
    index=None,
    format=None,
    unit=None,
    dim=None,
    tnull=None,
    tscal=None,
    tzero=None,
)
torchfits.table.replace_column(
    path,
    name,
    values,
    hdu=1,
    format=None,
    unit=None,
    dim=None,
    tnull=None,
    tscal=None,
    tzero=None,
)
torchfits.table.rename_columns(path, {"old_name": "new_name"}, hdu=1)
torchfits.table.drop_columns(path, ["col_a", "col_b"], hdu=1)
```

`row_slice` is a `slice` or `(start, stop)` tuple (0-based Python style for
update/delete). `insert_rows` requires keyword `row=` (0-based insert index).

---

## Interop

### Astropy

Direct reading, conversion, and writing for `astropy.table.Table`:

```python
# Read directly into an Astropy Table (supports SQL WHERE filters)
tbl = torchfits.table.read_astropy(
    "catalog.fits", hdu=1, where="MAG_G < 20.0 AND DEC > 0"
)

# Convert an in-memory tensor dict to an Astropy Table
tbl_from_tensors = torchfits.to_astropy(tensor_dict)

# Write an Astropy Table directly to FITS
torchfits.table.write("out.fits", tbl, overwrite=True)
```

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
polars_df = torchfits.to_polars(table_dict, decode_bytes=False)
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
# Register and query (share the relation name between both calls)
con = torchfits.table.to_duckdb("catalog.fits", hdu=1, relation_name="tbl")
result = torchfits.table.duckdb_query(
    "catalog.fits",
    "SELECT * FROM tbl WHERE MAG < 20",
    hdu=1,
    relation_name="tbl",
)
# result: pyarrow object (Table or RecordBatchReader by version);
# call result.read_all() to force a pyarrow.Table
```

### Arrow and Pandas

```python
arrow_table = torchfits.to_arrow(table_dict, decode_bytes=False)
pandas_df = torchfits.to_pandas(table_dict, decode_bytes=False)
```

---

## Schema

Infer an Arrow schema from FITS `TFORM` / `TTYPE` cards (no data read when
`where=None`).

```python
torchfits.table.schema(
    path,
    hdu=1,
    columns=None,
    where=None,
    decode_bytes=True,
    encoding="ascii",
    strip=True,
    include_fits_metadata=False,
    apply_fits_nulls=False,
    backend="auto",
)
```

**Returns:** `pyarrow.Schema`

```python
schema = torchfits.table.schema("catalog.fits", hdu=1)
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
