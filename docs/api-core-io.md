# Core I/O Reference

Fundamental read and write operations for FITS images, tables, headers,
multi-extension files, checksums, and caching.

---

## `read()`

Unified FITS reader. Auto-detects image or table HDUs.

```python
torchfits.read(path, hdu=0, device="cpu", mmap="auto", mode="auto",
               options=None, return_header=False, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` or `PathLike` | *(required)* | FITS file path |
| `hdu` | `int` or `str` or `None` | `0` | HDU index, EXTNAME, or `None`/`"auto"` for autodetection |
| `device` | `str` | `"cpu"` | `"cpu"`, `"cuda"`, `"mps"` |
| `mmap` | `bool` or `str` | `"auto"` | `True`, `False`, or `"auto"` |
| `mode` | `str` | `"auto"` | `"auto"`, `"image"`, or `"table"` |
| `return_header` | `bool` | `False` | Return `(data, Header)` tuple |

**Returns:** `torch.Tensor` (images), `dict[str, torch.Tensor]` (tables), or
tuple if `return_header=True`.

!!! warning "options=" vs kwargs"
    ``options=`` (a `ReadOptions` instance) and individual keyword arguments
    (``fp16=``, ``mmap=``, etc.) are **mutually exclusive** — passing both
    raises a ``TypeError``. Pick one style: either ``options=ReadOptions(...)``
    or ``read(..., mmap=True, fp16=False)``.

!!! tip "When to mmap"
    Mmap helps large **local** IMAGE HDUs and repeated cutouts. Prefer
    `mmap=False` with multi-worker `DataLoader` on the same files if handle
    contention appears; also prefer non-mmap on cold network filesystems and
    for VLA / scaled tables. Dataset docs: [Data module](api-data.md).

!!! info "When to use"
    Use `read()` for quick exploration. Default `hdu=0` (primary). Pass
    `hdu=None` for first image/table autodetection. For explicit tensor reads,
    prefer `read_tensor()`. For dataframe workflows (predicate pushdown
    `where=`, Arrow/Polars), use `table.read()`. Root `read()` returns tensor
    columns for table HDUs and has no `where=` parameter.

```python
# Image with header
data, hdr = torchfits.read("image.fits", hdu=0, return_header=True)

# Auto-detect table → tensor-column dict (not a pyarrow.Table)
columns = torchfits.read("catalog.fits", hdu=1)
```

---

## `read_tensor()`

Read any N-dimensional FITS array directly as a PyTorch Tensor.

```python
torchfits.read_tensor(path, hdu=0, device="cpu", mmap=True,
                      fp16=False, bf16=False, raw_scale=False,
                      return_header=False, fallback_get_header=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdu` | `int` or `str` | `0` | HDU index or EXTNAME (required) |
| `device` | `str` | `"cpu"` | `"cpu"`, `"cuda"`, `"mps"` |
| `mmap` | `bool` | `True` | Memory-mapped reads (faster for repeated access) |
| `fp16` | `bool` | `False` | Read as float16 |
| `bf16` | `bool` | `False` | Read as bfloat16 |
| `raw_scale` | `bool` | `False` | Skip BSCALE/BZERO, return native storage dtype |
| `return_header` | `bool` | `False` | Return `(tensor, Header)` |

**Returns:** `torch.Tensor` (or tuple if `return_header=True`).

!!! info "When to use"
    This is the primary function for reading FITS images as tensors. Use it
    when you want a single tensor (1D spectra, 2D images, 3D cubes). For
    multi-extension files, use `read_hdus()`. For cutouts, use
    `read_subset()`.

!!! tip "GPU reads"
    Pass `device="cuda"` or `device="mps"` to place the result on device.
    Generic BSCALE/BZERO scaling still yields `float32` unless you opt into
    storage dtypes with `raw_scale=True` (e.g. int8 / uint16 parity with fitsio).

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
    Use `open_subset_reader` when you need many cutouts from the same large
    file (e.g., training on patches from a mosaic). It avoids repeatedly
    opening and closing the FITS handle.

---

## `read_hdus()`

Read multiple HDUs from a single FITS file as a list of tensors.

```python
torchfits.read_hdus(path, hdus, *, device="cpu", mmap=True, return_header=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdus` | `list[int` or `str]` | *(required)* | HDU indices or EXTNAME strings |
| `device` | `str` | `"cpu"` | Target device |
| `mmap` | `bool` | `True` | Memory-mapped reads |
| `return_header` | `bool` | `False` | Return list of `(tensor, Header)` tuples |

**Returns:** `list[torch.Tensor]`

```python
sci, wht, msk = torchfits.read_hdus("mef.fits", hdus=["SCI", "WHT", "MASK"])
```

---

## Table tensors (see [Tables](api-tables.md))

Root aliases `read_table` / `read_table_rows` / `stream_table` were removed in
1.0. Use:

| Need | API |
|---|---|
| Full / sliced column tensors | [`table.read_torch`](api-tables.md#tableread_torch) |
| Streaming tensor chunks | [`table.scan_torch`](api-tables.md#tablescan_torch) |
| Arrow / Polars / `where=` | [`table.read`](api-tables.md) / `table.read_polars` |

```python
cols = torchfits.table.read_torch("catalog.fits", hdu=1, columns=["RA", "DEC"])
for chunk in torchfits.table.scan_torch(
    "survey.fits", hdu=1, batch_size=100_000
):
    process(chunk)
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

**Returns:** `dict` with shape, dtype, and file info.

---

## `open()`

Multi-HDU context manager for low-level HDU/header access.

```python
torchfits.open(path, mode="r")
```

**Returns:** `HDUList` context manager.

```python
with torchfits.open("mef.fits") as hdul:
    primary = hdul[0]          # TensorHDU
    sci = hdul["SCI"]          # TensorHDU by EXTNAME
    data = sci.data            # DataView (lazy)
    header = sci.header        # Header (dict-like)
```

Paths may include a CFITSIO **image section** (`file.fits[10:100,20:200]`);
existence checks use the base path before `[`. Prefer `hdu=` / EXTNAME
indexing over path HDU selectors (`file.fits[1]`) — those are not a certified
torchfits `open` surface yet.

!!! warning "mmap writes require flush"
    When writing via `open()` with mmap-backed HDUs, changes are held in
    memory-mapped pages and **not persisted to disk** until you call
    `hdul.flush()`. Always flush before closing the context manager if
    you modified data in-place.

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
torchfits.read_hdu_type(path, hdu=0)   # "IMAGE" / "BINARY_TABLE" / ...
torchfits.read_num_hdus(path)
torchfits.read_extname(path, hdu=1)    # EXTNAME or None
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

Reusable table handle (open once, many column/row reads). `hdu` accepts an
integer index or EXTNAME string (default `1`, the usual table location).

```python
torchfits.open_table_reader(path, hdu=1)
```

```python
with torchfits.open_table_reader("catalog.fits", hdu=1) as reader:
    n = reader.num_rows()
    cols = reader.read_torch(columns=["RA", "DEC"])
```

Mirrors `open_subset_reader` for images. There is no handle-based
filtered/`where=` read; for cold filtered reads use
`table.read_torch(..., where=...)`, which reopens the file per call.

---

## Writes

### `write()`

Write a tensor, numpy array, dict table, or HDUList to FITS.

```python
torchfits.write(path, data, header=None, overwrite=False, compress=False,
                quantize=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` or `PathLike` | *(required)* | Output path |
| `data` | `Tensor` or `ndarray` or `dict` or `HDUList` | *(required)* | Data to write |
| `header` | `dict` or `Header` or `None` | `None` | FITS header key-value pairs |
| `overwrite` | `bool` | `False` | Overwrite existing file |
| `compress` | `bool` or `str` | `False` | `True`, `"gzip"`, `"rice"`, etc. |
| `quantize` | `None` or `str` or `dict` | `None` | Opt-in robust `BITPIX=16` pack for float images (`"robust"` or `{"lo_q", "hi_q", "keep_zero"}`). Default keeps native float. |

!!! tip "Skewed float → int16"
    Linear min→max packing onto int16 wastes codes on rare extremes. Prefer
    native float (`quantize=None`). When size forces int16, use
    `quantize="robust"` (default percentiles `lo_q=0.1`, `hi_q=99.9` + clip)
    or a dict `{"lo_q", "hi_q", "keep_zero"}` — not global min/max. Same helper
    packs table columns via `table.write(..., quantize=)`. See
    [`example_quantize_int16.py`](published-examples/example_quantize_int16.py).

### `write_tensor()`

Write a single PyTorch Tensor to a FITS image extension.

```python
torchfits.write_tensor(path, tensor, header=None, overwrite=False, compress=False,
                       quantize=None)
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
# result: dict with "datastatus", "hdustatus", "ok"
```

---

## Cache Utilities

Two layers (do not merge in call sites — pick one intentionally):

| Layer | Entry points | Role |
|---|---|---|
| Root I/O / metadata caches | `get_cache_performance()`, `clear_file_cache(...)` | Python LRUs + SharedReadMeta clear used by `read` / `read_tensor` |
| `torchfits.cache` manager | `cache.configure_for_environment()`, `cache.get_cache_stats()`, `cache.clear_cache()`, `cache.optimize_for_dataset(...)` | Higher-level training / Dataset policy (C++ handle-pool `configure_cache` is a no-op) |

```python
# Root I/O cache
torchfits.get_cache_performance()
torchfits.clear_file_cache(data=True, handles=True, meta=True, cpp=True)

# Higher-level cache manager
torchfits.cache.configure_for_environment()
torchfits.cache.get_cache_stats()
torchfits.cache.clear_cache()
torchfits.cache.optimize_for_dataset(file_paths, avg_file_size_mb=10.0)
```

Advanced (import from `torchfits.io`, not re-exported at package root):
`cache_subsystem_policy(name)` / `clear_cache_subsystem(name)` inspect or clear
a named engine subsystem (`"all"` clears every subsystem). Prefer the root /
`torchfits.cache` helpers above unless you are debugging cache splits.

!!! tip "Training loops"
    Call `torchfits.cache.optimize_for_dataset(paths, avg_file_size_mb=...)`
    before `DataLoader` epochs to set cache policy for the file set. When using
    `make_loader(..., optimize_cache=True)` (the default), this happens
    automatically. Per-call CFITSIO handles are not pooled across threads.

### Disk cache directories

Remote prefetch and example samples live under a base directory resolved
once per process: `TORCHFITS_CACHE_DIR` / `TORCHFITS_REMOTE_CACHE` /
`TORCHFITS_SAMPLE_CACHE` — see the User-facing table in
[Environment variables](architecture.md#environment-variables) for defaults.

These roots are independent of `get_cache_performance` / `clear_file_cache`
above, which govern the in-process handle and metadata caches, not files on
disk. Dataset classes accept `cache_dir=` to override the remote root
per-instance; see [Data module](api-data.md#cache-how-and-when) for the
Dataset/`make_loader` cache-warming path and when it no-ops (table datasets,
single-file reads).
