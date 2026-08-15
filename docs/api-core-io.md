# Core I/O Reference

Fundamental read and write operations for FITS images, tables, headers,
multi-extension files, checksums, and caching.

---

## `read()`

Unified FITS reader. Auto-detects image or table HDUs.

```python
torchfits.read(
    path,
    hdu=0,
    device="cpu",
    mmap="auto",
    mode="auto",
    options=None,
    return_header=False,
    **kwargs,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` or `PathLike` | *(required)* | FITS file path |
| `hdu` | `int` or `str` or `None` | `0` | HDU index, EXTNAME, or `None`/`"auto"` for autodetection |
| `device` | `str` | `"cpu"` | `"cpu"`, `"cuda"`, `"mps"` |
| `mmap` | `bool` or `str` | `"auto"` | `True`, `False`, or `"auto"` |
| `mode` | `str` | `"auto"` | `"auto"`, `"image"`, or `"table"` |
| `return_header` | `bool` | `False` | Return `(data, Header)` tuple |

**Returns:** `torch.Tensor` (images), a column mapping for tables (scalar
columns are tensors; VLA columns may use list/tuple values), or a tuple if
`return_header=True`.

!!! note "Advanced options"
    Prefer the explicit parameters and keyword arguments shown here. The
    internal `options=` helper is not part of the public facade — do not
    import it from a private module in application code.

!!! tip "When to mmap"
    Mmap can help eligible large local integer IMAGE HDUs and repeated
    cutouts. It still returns a normal tensor, and float / compressed images
    use a buffered path. Prefer `mmap=False` when many workers open the same
    files, on cold network filesystems, and for VLA / scaled tables. Dataset
    docs: [Data module](api-data.md).

!!! info "When to use"
    Use `read()` for quick exploration (`hdu=0` by default; `hdu=None`
    autodetection). Prefer `read_tensor()` for images and `table.read()` for
    catalogs with `where=` / Arrow. Root `read()` on a table HDU returns a
    column → tensor dict and has no `where=` parameter.

```python
# Image with header
data, hdr = torchfits.read("image.fits", hdu=0, return_header=True)

# Auto-detect table → column → tensor dict
columns = torchfits.read("catalog.fits", hdu=1)
```

---

## `read_tensor()`

Read any N-dimensional FITS array directly as a PyTorch Tensor.

```python
torchfits.read_tensor(
    path,
    hdu=0,
    device="cpu",
    mmap=True,
    fp16=False,
    bf16=False,
    raw_scale=False,
    return_header=False,
    fallback_get_header=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdu` | `int` or `str` | `0` | HDU index or EXTNAME |
| `device` | `str` | `"cpu"` | `"cpu"`, `"cuda"`, `"mps"` |
| `mmap` | `bool` | `True` | Request the eligible memory-mapped input path |
| `fp16` | `bool` | `False` | Read as float16 |
| `bf16` | `bool` | `False` | Read as bfloat16 |
| `raw_scale` | `bool` | `False` | Skip BSCALE/BZERO, return native storage dtype |
| `return_header` | `bool` | `False` | Return `(tensor, Header)` |

**Returns:** `torch.Tensor` (or tuple if `return_header=True`).

!!! info "When to use"
    Read an IMAGE HDU as a single tensor (spectrum, image, or cube). For
    multi-extension files use `read_hdus()`; for cutouts use `read_subset()`.

!!! tip "GPU reads"
    Pass `device="cuda"` or `device="mps"` to place the result on device.
    Generic BSCALE/BZERO scaling still yields `float32` unless you opt into
    storage dtypes with `raw_scale=True`. For example, signed-byte and
    pseudo-unsigned conventions expose their on-disk `uint8` / `int16` storage
    dtypes rather than the logical `int8` / `uint16` dtypes.

```python
# Read to GPU
tensor = torchfits.read_tensor("image.fits", hdu=0, device="cuda")

# Native integer dtype (matches fitsio)
tensor = torchfits.read_tensor("image.fits", hdu=0, raw_scale=True)

# 3D cube
cube = torchfits.read_tensor("cube.fits", hdu=0)
# cube.shape = (nz, ny, nx)
```

---

## `read_subset()`

Read a rectangular pixel subset from an image HDU. On **3D+ cubes**, the
window applies to the trailing `(y, x)` axes; the leading depth axis is kept
in full.

```python
torchfits.read_subset(path, hdu, x1, y1, x2, y2)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path or HTTP(S) URL |
| `hdu` | `int` or `str` | *(required)* | HDU index or EXTNAME |
| `x1, y1, x2, y2` | `int` | *(required)* | Half-open pixel window `[x1,x2)×[y1,y2)` |

**Returns:** `torch.Tensor`

HTTP(S) **uncompressed 2D** images Range-fetch a row-band (no full download).
Compressed / scaled remotes and `vos:` / `vault:` paths materialize into the
remote cache first, then cut out locally.

```python
stamp = torchfits.read_subset("mosaic.fits", hdu=0, x1=0, y1=0, x2=256, y2=256)
```

CFITSIO image sections on the path (1-based inclusive) work the same way
users expect from `imcopy` / CFITSIO, e.g.
`read_tensor("mosaic.fits[1:256,1:256]")` or
`torchfits cutout 'mosaic.fits[1:256,1:256]' out.fits`. For Python /
NumPy-style windows use `read_subset` or CLI `--box` (0-based half-open).
Do not stack a path section with `read_subset` / `--box` on the same call.
Binspec / complex CFITSIO filters are not a certified torchfits surface —
use `table.read(..., where=)` for catalog predicates.

---

## `open_subset_reader()`

Reusable reader for repeated cutout access on a single image HDU. Opens the
file once; each call reads a region without re-opening.

```python
torchfits.open_subset_reader(path, hdu=0, device="cpu")
```

`hdu` accepts an integer index or EXTNAME string (same as `read_subset`).

**Returns:** Context manager yielding a `SubsetReader`. Call `reader(x1, y1, x2, y2)` for each cutout.

```python
with torchfits.open_subset_reader("mosaic.fits", hdu=0) as reader:
    stamp1 = reader(0, 0, 256, 256)
    stamp2 = reader(128, 128, 256, 256)
```

!!! info "When to use"
    Many cutouts from the same large mosaic without reopening the file each
    time.

---

## `read_hdus()`

Read multiple image HDUs from a single FITS file as a list of tensors.

```python
torchfits.read_hdus(path, hdus, *, device="cpu", mmap=True, return_header=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdus` | `list[int | str]` or `tuple[int | str, ...]` | *(required)* | Image HDU indices or EXTNAME strings |
| `device` | `str` | `"cpu"` | Target device |
| `mmap` | `bool` | `True` | Request the eligible memory-mapped input path |
| `return_header` | `bool` | `False` | Also return a parallel list of `Header` objects |

**Returns:** `list[torch.Tensor]`, or `(list[torch.Tensor], list[Header])`
when `return_header=True`.

```python
sci, wht, msk = torchfits.read_hdus("mef.fits", hdus=["SCI", "WHT", "MASK"])
```

---

## Table Tensors (see [Tables](api-tables.md))

For reading and streaming catalog columns directly as PyTorch tensors:

| Need | API |
|---|---|
| Full / sliced column tensors | [`table.read_torch`](api-tables.md#tableread_torch) |
| Streaming tensor chunks | [`table.scan_torch`](api-tables.md#tablescan_torch) |
| Arrow / Polars / `where=` | [`table.read`](api-tables.md) / `table.read_polars` |

```python
cols = torchfits.table.read_torch("catalog.fits", hdu=1, columns=["RA", "DEC"])
for chunk in torchfits.table.scan_torch("survey.fits", hdu=1, batch_size=100_000):
    print(chunk.keys())
```

---

## `read_batch()`

Read the same HDU from multiple FITS files.

```python
torchfits.read_batch(file_paths, hdu=0, device="cpu", *, strict=False)
```

**Returns:** `list[torch.Tensor]` — one tensor per successfully read file (not a
stacked batch). With the default ``strict=False``, files that fail to read are
skipped with a ``RuntimeWarning``; pass ``strict=True`` to raise on the first
failure.

```python
tensors = torchfits.read_batch(["img1.fits", "img2.fits"], hdu=0)
```

---

## `read_batch_info()`

Inspect shape and dtype consistency across files before batch reading.

```python
torchfits.read_batch_info(file_paths)
```

**Returns:** `dict` with `num_files` (paths supplied) and `existing_files`
(paths present on disk).

---

## `open()`

Multi-HDU context manager for low-level HDU/header access.

```python
torchfits.open(path, mode="r")
```

**Returns:** `HDUList` context manager.

```python
with torchfits.open("mef.fits") as hdul:
    primary = hdul[0]  # TensorHDU
    sci = hdul["SCI"]  # TensorHDU by EXTNAME
    data = sci.data  # DataView (lazy)
    header = sci.header  # Header (dict-like)
```

Paths may include a CFITSIO **image section** (`file.fits[10:100,20:200]`);
existence checks use the base path before `[`. Prefer `hdu=` / EXTNAME
indexing over path HDU selectors (`file.fits[1]`) — those are not a certified
torchfits `open` surface yet.

!!! warning "The open model is read-oriented"
    `TensorHDU.data` is a read-oriented `DataView`; `HDUList` has no
    `flush()` method or in-place write protocol. To create a modified file,
    construct the desired HDUs and call `hdul.write(output_path, overwrite=...)`.

!!! info "EXTNAME lookup returns first match"
    When indexing by EXTNAME (`hdul["SCI"]`), only the **first** HDU with
    that name is returned. FITS files with duplicate EXTNAMEs (e.g.,
    multi-chip detectors with repeated ``SCI`` extensions) must use
    numeric indices for the second and subsequent occurrences.

---

## `read_header()`

Read only the FITS header from an HDU.

```python
torchfits.read_header(path, hdu=0)
```

**Returns:** `Header` (dict-like). Default `hdu=0`; pass `hdu=None` / `"auto"`
to autodetect.

```python
header = torchfits.read_header("image.fits", hdu=0)
print(header["EXPTIME"])  # e.g. 300.0
```

For bloated headers when you only need a few cards or the row count, prefer
the skinny helpers below — they skip the full header dump.

## `read_nrows()`

Table row count via CFITSIO `fits_get_num_rows` (no full header materialize).

```python
torchfits.read_nrows(path, hdu=1)
```

**Returns:** `int`. Default `hdu=1`. Raises if the HDU is not a table.

```python
n = torchfits.read_nrows("catalog.fits", hdu=1)
```

## `read_keys()`

Selected header keywords via CFITSIO `fits_read_keyword` (no full header dump).

```python
torchfits.read_keys(path, keys, hdu=0)
```

**Returns:** `dict[str, Any]`. Missing keys raise. Default `hdu=0`.

```python
meta = torchfits.read_keys("image.fits", ["BITPIX", "NAXIS1", "NAXIS2"], hdu=0)
```

## `read_shape()`

Image BITPIX + shape via CFITSIO image params (no full header).

```python
torchfits.read_shape(path, hdu=0)
```

**Returns:** `(bitpix, shape)` with torch / row-major `shape`.

## Skinny HDU type / count / EXTNAME

`read_hdu_type()` / `read_num_hdus()` / `read_extname()`

```python
torchfits.read_hdu_type(path, hdu=0)  # "IMAGE" / "BINARY_TABLE" / ...
torchfits.read_num_hdus(path)
torchfits.read_extname(path, hdu=1)  # EXTNAME or None
```

## Skinny colnames / table info

`read_colnames()` / `read_table_info()`

```python
torchfits.read_colnames(path, hdu=1)
torchfits.read_table_info(path, hdu=1)  # {nrows, colnames, tforms}
```

One open each; no full header dump. Prefer these over `read_header` for
counts, dims, and a handful of cards.

## `open_table_reader()`

Open one table HDU once; call `read_torch` repeatedly without reopening.

```python
torchfits.open_table_reader(path, hdu=1)
```

**Returns:** context manager yielding a handle with:

- `num_rows()` → `int`
- `read_torch(columns=None, start_row=1, num_rows=-1, device="cpu")` →
  `dict[str, torch.Tensor]`

No `where=` on the handle — use `table.read_torch(..., where=...)` for
filtered reads (each call opens the file).

```python
with torchfits.open_table_reader("catalog.fits", hdu=1) as reader:
    n = reader.num_rows()
    cols = reader.read_torch(columns=["RA", "DEC"])
```

---

## Writes

### `write()`

Write a tensor, numpy array, dict table, or HDUList to FITS.

```python
torchfits.write(path, data, header=None, overwrite=False, compress=False, quantize=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` or `PathLike` | *(required)* | Output path |
| `data` | `Tensor`, numpy array, `dict`, or `HDUList` | *(required)* | Data to write. Numpy arrays and tensors write an image HDU; `dict` writes a table. |
| `header` | `dict` or `Header` or `None` | `None` | FITS header key-value pairs |
| `overwrite` | `bool` | `False` | Overwrite existing file |
| `compress` | `bool` or `str` | `False` | `True`, `"gzip"`, `"rice"`, etc. GZIP_1 and integer RICE_1 writes are lossless; float RICE_1 / HCOMPRESS_1 use CFITSIO default quantization (lossy, same as astropy/fitsio defaults). |
| `quantize` | `None` or `str` or `dict` | `None` | Opt-in robust `BITPIX=16` pack for float images (`"robust"` or `{"lo_q", "hi_q", "keep_zero"}`). Default keeps native float. |

!!! tip "Skewed float → int16"
    Linear min→max packing onto int16 wastes codes on rare extremes. Prefer
    native float (`quantize=None`). When size forces int16, use
    `quantize="robust"` (default percentiles `lo_q=0.1`, `hi_q=99.9` + clip)
    or a dict `{"lo_q", "hi_q", "keep_zero"}` — not global min/max. Same helper
    packs table columns via `table.write(..., quantize=)`. See
    [`example_quantize_int16.py`](published-examples/example_quantize_int16.py).

!!! note "uint64 payloads are rejected"
    FITS has no native uint64 storage (`BITPIX=-64` is not standard, and a
    `BZERO=2**64` pseudo-unsigned convention is not interoperable).
    `write()` raises `ValueError` for uint64 image tensors and uint64 table
    columns with guidance: convert to `int64` (values `< 2**63`) or `float64`
    before writing. Unsigned `uint16`/`uint32` are supported natively via the
    standard signed storage + `BSCALE`/`BZERO` convention.

### `write_tensor()`

Write a single PyTorch Tensor to a FITS image extension.

```python
torchfits.write_tensor(
    path, tensor, header=None, overwrite=False, compress=False, quantize=None
)
```

```python
torchfits.write_tensor("out.fits", tensor, header={"OBJECT": "M31"}, overwrite=True)
torchfits.write_tensor("packed.fits", tensor, quantize="robust", overwrite=True)
```

---

## HDU Mutation

```python
torchfits.insert_hdu(path, data, index=1, header=None, compress=False)
torchfits.replace_hdu(path, hdu, data, header=None, compress=False)
torchfits.delete_hdu(path, hdu, compress=False)
```

---

## Checksums

```python
torchfits.write_checksums(path, hdu=0)
result = torchfits.verify_checksums(path, hdu=0)
# result: dict with "datastatus", "hdustatus", "ok", and "status" ("ok" / "no_checksums" / "fail")
```

---

## Cache Utilities

| Layer | Entry points | Role |
|---|---|---|
| Disk / policy | `torchfits.cache.configure_for_environment()`, `get_cache_stats()`, `clear_cache()`, `clear_cache(disk=True)` / `clear_all_caches()`, `optimize_for_dataset(paths, avg_file_size_mb=10.0)` | Environment policy and on-disk roots |
| I/O metadata | `get_cache_performance()`, `clear_file_cache(...)` | In-process header / meta / data caches |
| Shared metadata (C++) | `clear_file_cache(..., cpp=True)` | Per-path metadata across private CFITSIO handles |

```python
torchfits.clear_file_cache(
    data=True, handles=True, meta=True, hdu_types=True, stats=True, cpp=True
)
torchfits.get_cache_performance()

torchfits.cache.configure_for_environment()
torchfits.cache.get_cache_stats()
torchfits.cache.clear_cache()  # in-process only (default)
torchfits.clear_all_caches()  # in-process + disk cache_root()
torchfits.cache.optimize_for_dataset(file_paths, avg_file_size_mb=10.0)
```

`clear_cache()` clears in-process policy, data, and metadata caches. Pass
`disk=True` (or call root `clear_all_caches()`) to also remove downloaded
files under `cache_root()`.

`clear_file_cache` keyword-only flags (all default `True`): `data`, `handles`,
`meta`, `hdu_types`, `stats`, `cpp`. Optional `cpp_module=` overrides the
extension module used for the C++ clear.

Advanced helpers on `torchfits.io`: `cache_subsystem_policy(name)` /
`clear_cache_subsystem(name)` (`"all"` clears every subsystem).

!!! tip "Many-file datasets"
    Call `torchfits.cache.optimize_for_dataset(paths, avg_file_size_mb=...)`
    before multi-worker loops. `make_loader(..., optimize_cache=True)` does
    this automatically. Each read uses a private CFITSIO handle.

### Disk cache directories

Remote downloads and example samples use
`TORCHFITS_CACHE_DIR` / `TORCHFITS_REMOTE_CACHE` / `TORCHFITS_SAMPLE_CACHE`
([Environment variables](architecture.md#environment-variables)). Those roots
are separate from `clear_file_cache` / `get_cache_performance`. Remote-capable
image, tensor, cube, and spectrum datasets accept `cache_dir=` to override the
remote root; see [Data module](api-data.md#cache-how-and-when).
