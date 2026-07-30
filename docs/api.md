# API Reference

`torchfits` covers FITS file I/O (IMAGE HDUs → tensors, tables → dataframes,
headers, compression) and ML helpers (`torchfits.data`, `torchfits.transforms`).
Sky-domain modelling is out of scope. Current line: **1.0.0rc4** (prerelease).

For job-oriented patterns (which API for which task), start with
[Python workflows](python-workflows.md). This page lists signatures, defaults,
and edge cases.

**FITS images → tensors. FITS tables → dataframes** (Arrow by default;
Polars/Pandas one call away; tensor columns when you train). In code the
namespace is `torchfits.table` because that is the FITS name; the object
model is a columnar dataframe.

---

## Which reader?

| Goal | Call | Returns | mmap default |
|---|---|---|---|
| Image / cube / spectrum | `read_tensor(path, hdu=0)` | `torch.Tensor` | `True` |
| Catalog as dataframe (default) | `table.read(path, hdu=1, where=…)` | `pyarrow.Table` | `True` |
| Dataframe columns as tensors | `table.read_torch(path, hdu=1)` | `dict[str, torch.Tensor]` | `True` |
| Native Polars dataframe | `table.read_polars(path, hdu=1)` | Polars DataFrame-like | (via `table.read`) |

`table.read_arrow` is an exact synonym of `table.read` (destination-qualified
spelling alongside `read_torch` / `read_polars`). Root `read_table` /
`stream_table` / `read_table_rows` / `get_header` / `get_batch_info` were
**removed** in 1.0 — use `table.*`, `read_header`, and `read_batch_info`.

**Caches.** Root I/O metadata caches are cleared via
`clear_file_cache()` / `get_cache_performance()`. Training-oriented sizing and
warm-up live under `torchfits.cache` (`configure_for_environment`,
`optimize_for_dataset`, `clear_cache`). C++ shared-handle pooling is gone
(Option A); `configure_cache` is a no-op. See [Core I/O → Cache Utilities](api-core-io.md#cache-utilities).

---

## Quick Paths

### Images and Files

| Goal | Entry point | Reference |
|---|---|---|
| Read N-D array as tensor | `read_tensor(path, hdu=0, device="cpu", mmap=True)` | [Core I/O](api-core-io.md#read_tensor) |
| Read image or table (auto-detect) | `read(path, hdu=None, return_header=False)` | [Core I/O](api-core-io.md#read) |
| Rectangular cutout | `read_subset(path, hdu, x1, y1, x2, y2)` | [Core I/O](api-core-io.md#read_subset) |
| Repeated cutouts from one file | `open_subset_reader(path, hdu=0)` | [Core I/O](api-core-io.md#open_subset_reader) |
| Repeated table column reads | `open_table_reader(path, hdu=1)` | [Core I/O](api-core-io.md#open_table_reader) |
| Multiple HDUs at once | `read_hdus(path, hdus=[0, 1, 2])` | [Core I/O](api-core-io.md#read_hdus) |
| Write a tensor | `write_tensor(path, tensor, ..., quantize=None)` | [Core I/O](api-core-io.md#write_tensor) |
| Read header only | `read_header(path, hdu=0)` | [Core I/O](api-core-io.md#read_header) |
| Table row count (skinny) | `read_nrows(path, hdu=1)` | [Core I/O](api-core-io.md#read_nrows) |
| Selected header keys (skinny) | `read_keys(path, keys, hdu=0)` | [Core I/O](api-core-io.md#read_keys) |
| Image BITPIX+shape (skinny) | `read_shape(path, hdu=0)` | [Core I/O](api-core-io.md#read_shape) |
| HDU type / count (skinny) | `read_hdu_type` / `read_num_hdus` | [Core I/O](api-core-io.md#skinny-hdu-type-count-extname) |
| Table colnames / info (skinny) | `read_colnames` / `read_table_info` | [Core I/O](api-core-io.md#skinny-colnames-table-info) |
| Multi-HDU context manager | `open(path, mode="r")` | [Core I/O](api-core-io.md#open) |
| Batch-read many files | `read_batch(file_paths, hdu=0)` | [Core I/O](api-core-io.md#read_batch) |

### Tables as dataframes

| Goal | Entry point | Reference |
|---|---|---|
| Read dataframe (Arrow) | `table.read(path, hdu=1, columns=None, where=None)` | [Tables](api-tables.md#tableread) |
| Stream dataframe batches | `table.scan(path, hdu=1, batch_size=65536)` | [Tables](api-tables.md#tablescan) |
| Dataframe columns as tensors | `table.read_torch(path, hdu=1, columns=None)` | [Tables](api-tables.md#tableread_torch) |
| Stream tensor-column chunks | `table.scan_torch(path, hdu=1, batch_size=65536)` | [Tables](api-tables.md#tablescan_torch) |
| Native Polars dataframe | `table.read_polars(path, hdu=1)` | [Tables](api-tables.md#polars) |
| Streaming Polars batches | `table.scan_polars(path, hdu=1)` | [Tables](api-tables.md#polars) |
| DuckDB SQL | `table.duckdb_query(path, sql, hdu=1)` | [Tables](api-tables.md#duckdb) |

Destination-qualified spelling of `table.read` (same object): `table.read_arrow`.


### ML Training

Prefer raw I/O until you need epochs/shuffle/workers — see
[Data module](api-data.md) and [Transforms](api-transforms.md#when-to-use).

| Goal | Entry point | Reference |
|---|---|---|
| General N-D image, map-style | `FitsTensorDataset(paths, hdu=0, label_key=None)` | [Data](api-data.md#fitstensordataset) |
| General N-D image, iterable (multi-worker) | `FitsTensorIterableDataset(paths, shuffle=False)` | [Data](api-data.md#fitstensoriterabledataset) |
| 2D image peer | `FitsImageDataset(paths, hdu=0)` | [Data](api-data.md#fitsimagedataset-fitscubedataset) |
| 3D+ cube peer | `FitsCubeDataset(paths, hdu=0, slice_index=None)` | [Data](api-data.md#fitsimagedataset-fitscubedataset) |
| 2D image iterable peer | `FitsImageIterableDataset(paths, shuffle=False)` | [Data](api-data.md#fitstensoriterabledataset) |
| 1D / multi-arm spectrum | `FitsSpectrumDataset(paths, hdu=0, layout="dict")` | [Data](api-data.md#fitsspectrumdataset) |
| Table map-style (fits in RAM) | `FitsTableDataset(path, hdu=1)` | [Data](api-data.md#fitstabledataset) |
| Table streaming (large) | `FitsTableIterableDataset(path, hdu=1, batch_size=65536)` | [Data](api-data.md#fitstableiterabledataset) |
| Cutout patches | `FitsCutoutDataset(cutouts)` | [Data](api-data.md#fitscutoutdataset) |
| DataLoader + cache defaults | `make_loader(dataset, batch_size=32)` | [Data](api-data.md#make_loader) |
| Image stretches, normalizers, clip | `from torchfits.transforms import …` | [Transforms](api-transforms.md) |
| Shell inspect / convert | `torchfits info|header|verify|…` | [CLI](cli.md) |

---

## Reference Pages

| Page | What it covers |
|---|---|
| [CLI](cli.md) | `torchfits` command-line tools, exit codes, MEF defaults |
| [Core I/O](api-core-io.md) | `read`, `read_tensor`, `read_subset`, `read_hdus`, `write_tensor`, `write`, `open`, headers, HDU mutation, checksums, batch reads, cache |
| [Tables](api-tables.md) | FITS tables as dataframes: `table.read` / `read_torch` / `read_polars`, mutations, interop |
| [Data](api-data.md) | `FitsTensorDataset`, `FitsTensorIterableDataset`, `FitsCubeDataset`, `FitsSpectrumDataset`, table/cutout datasets, `make_loader`, remote prefetch |
| [Transforms](api-transforms.md) | Transform classes (callable protocol, not `nn.Module`) with verified math, parameters, invertibility, and when-to-use guidance |
| [Architecture](architecture.md) | C++/Python layering, I/O paths, caching, threading, CFITSIO mapping, environment variables |

---

## Package Namespaces

| Namespace | Purpose |
|---|---|
| `torchfits` (root) | I/O functions and HDU classes |
| `torchfits.hdu` | HDU/header types (`Header`, `Card`, `HDUList`, …) |
| `torchfits.table` | FITS table / dataframe I/O, mutation, interop |
| `torchfits.data` | Dataset classes and loader factory |
| `torchfits.transforms` | Transform classes |
| `torchfits.cache` | Cache configuration and management |
| `torchfits.where` | Predicate parser and evaluator |
| `torchfits.cpp` | Low-level native compatibility surface |

`torchfits.cpp` is the low-level native compatibility surface used by
performance-sensitive downstream packages. Its `__all__` is the
function-level compatibility contract; new compiled-extension symbols are
private until promoted there.

---

## HDU Types

| Class | Description |
|---|---|
| `TensorHDU` | Image HDU with lazy `.data` (returns `DataView`) and `.header` |
| `TableHDU` | In-memory table HDU with tensor columns |
| `TableHDURef` | Lazy file-backed table handle |
| `Header` | Dict-like FITS header preserving card order and semantics |
| `Card` | Single FITS header card: `Card(key, value, comment)` |

---

## Removed names

| Old | Use instead |
|---|---|
| `read_fast(...)` | `read(...)` or `read_tensor(...)` |
| `read_image(...)` | `read_tensor(...)` |
| `read_table(...)` (root) | `table.read_torch(...)` or `table.read(...)` |
| `stream_table(...)` (root) | `table.scan_torch(...)` |
| `read_table_rows(...)` (root) | `table.read_torch(..., start_row=, num_rows=)` |
| `get_header(...)` | `read_header(...)` |
| `get_batch_info(...)` | `read_batch_info(...)` |
| Root transform classes (e.g. `torchfits.ArcsinhStretch`) | `torchfits.transforms.*` |

Transforms are not re-exported at the package root. Import them from
`torchfits.transforms`. HDU helpers are available as root names and via
`torchfits.hdu`.

---

## Limitations

- VLA columns use buffered I/O; mmap reads and in-place updates are not supported.
- Scaled table columns do not support mmap updates; use the buffered path.
- Non-CPU tensors are copied to host before FITS writes.
- Compressed writes accept tensor IMAGE-HDU payloads; dict payloads must contain tensor data.
