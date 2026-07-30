# Data Module

`torchfits.data` provides PyTorch `Dataset` / `IterableDataset` classes for
FITS images and tables, plus `make_loader` — a thin factory around
`torch.utils.data.DataLoader` with torchfits cache warm-up defaults.

**Not training?** Stay on [Core I/O](api-core-io.md) (`read_tensor`) or
[Tables](api-tables.md) (`table.read`). Datasets exist for multi-sample epochs,
shuffle, and worker parallelism — not for one-off reads.

**Layering:** `read_tensor` / `table.read` → optional
[`transforms`](api-transforms.md) → `Fits*Dataset` → `make_loader` → training
loop. Pass a transform into the dataset when every sample needs the same
preprocess; call transforms yourself when exploring a single file.

---

## Choosing a Dataset

| Your data | Catalog size | Use | Why |
|---|---|---|---|
| General N-D IMAGE (any rank) | Any | `FitsTensorDataset` (map) | Umbrella / escape hatch; multi-HDU flux channels |
| General N-D IMAGE, many workers | Any | `FitsTensorIterableDataset` | Deterministic sharding |
| 2D images / multi-band | Any | `FitsImageDataset` | Peer; multi-band HDUs → `[C,H,W]` |
| 3D+ cubes | Any | `FitsCubeDataset` | Peer; optional `slice_index` |
| 1D / multi-arm spectra | Any | `FitsSpectrumDataset` | Peer; `layout=` dict/stack/concat; DESI `row=` |
| One table HDU | Fits in RAM | `FitsTableDataset` | Row-indexed `(dict[str, Tensor], label)` with predicate pushdown |
| One table HDU | Too large for RAM | `FitsTableIterableDataset` | Constant-memory streaming via `table.scan` |
| Fixed `(path, hdu, x, y, size)` cutouts | Any | `FitsCutoutDataset` | Patch training from a mosaic |
| One-off inspect / write / Arrow analysis | — | Core I/O / Tables | No Dataset needed |

**Channel semantics:** flux bands / arms / CCDs stack as tensor channels.
IVAR and mask are always **companion** tensors (or dict fields) — never mixed
into flux channels. Pass `ivar_hdu=` / `mask_hdu=` (or table `ivar_column=`).

**Dataset `transform=` signature:** each dataset calls `transform(payload)`
with one positional argument — the full payload (a `Tensor`, or a
`{"flux", "ivar"?, "mask"?}` dict when `ivar_hdu=` / `mask_hdu=` are set).
There is no separate `mask=` kwarg at the Dataset boundary, even though
`FITSTransform.forward(x, mask=None)` accepts one when you call a transform
directly on a tensor. Write custom transforms to branch on dict vs tensor
input if they need mask-aware Dataset wiring — see
[custom transforms](api-transforms.md#writing-a-custom-transform).

!!! tip "When to mmap"
    Good for large local IMAGE HDUs and repeated cutouts. Prefer
    `mmap=False` with multi-worker DataLoader on the same files if handle
    contention appears; also prefer non-mmap on cold network FS and VLA /
    scaled tables.

!!! tip "Disk cache (remotes + samples only)"
    Dataset HTTP(S)/vos prefetch and example samples write under
    `TORCHFITS_CACHE_DIR` (default `$XDG_CACHE_HOME/torchfits` or
    `~/.cache/torchfits`). Subdirs: `remote/`, `samples/`. Override with
    `TORCHFITS_REMOTE_CACHE` / `TORCHFITS_SAMPLE_CACHE`, or pass
    `cache_dir=` into Datasets / rely on `make_loader` prefetch. Point the
    root at scratch on shared HPC homes — no silent writes elsewhere under
    `$HOME`. HTTP auth: `TORCHFITS_HTTP_AUTHORIZATION` or
    `TORCHFITS_HTTP_TOKEN`. Uncompressed 2D HTTP cutouts (`read_subset` /
    `FitsCutoutDataset`) use Range GETs when possible; compressed and vos
    paths cache the full file first.

---

## Cache: how and when

Two independent caches exist: the **disk cache** (remote downloads, example
samples) and the **in-process handle/metadata cache** (`torchfits.cache`).
Dataset training loops touch both; a single-file `read_tensor` call touches
neither.

**Disk cache roots:** `TORCHFITS_CACHE_DIR` / `TORCHFITS_REMOTE_CACHE` /
`TORCHFITS_SAMPLE_CACHE` — see the User-facing table in
[Environment variables](architecture.md#environment-variables) for defaults.

Datasets accept `cache_dir=` to override the remote materialization
directory per-instance, taking priority over the environment variables for
that Dataset only.

**Handle/metadata cache:** `make_loader(ds, optimize_cache=True)` (default)
calls `cache.optimize_for_dataset(ds.files, avg_file_size_mb=...)`, which
updates Python cache policy for the file count (`configure_cache` is a
no-op for C++ handles). This only fires when the
dataset exposes a `files` list — every `Fits*Dataset` built on
`FitsTensorDataset` (`FitsImageDataset`, `FitsCubeDataset`,
`FitsSpectrumDataset`, `FitsTensorIterableDataset`, `FitsCutoutDataset`) has
one. `FitsTableDataset` / `FitsTableIterableDataset` read one file through
CFITSIO's own buffered path and do not expose `files`; `optimize_cache=True`
is then a documented no-op, not an error.

```python
from torchfits import cache

cache.get_cache_stats()   # hit rate, config (cpp handle pool size is always 0)
cache.clear_cache()       # drop Python I/O LRUs + SharedReadMeta
```

**When *not* to warm:** a single local file read (`read_tensor`, one-off
`table.read`) opens and closes one handle — there is nothing to warm and
`optimize_for_dataset` / `configure_for_environment` add overhead without
benefit. Warm-up pays off once a loader iterates many files (remote
prefetch, multi-worker `DataLoader`, repeated cutouts from a mosaic via
`open_subset_reader`).

---

## `FitsTensorDataset`

Map-style **general N-D** IMAGE reader (any rank). Multi-HDU `hdu=[…]` stacks
**flux** on dim 0; optional `ivar_hdu` / `mask_hdu` return companion tensors.

```python
from torchfits.data import FitsTensorDataset, make_loader

ds = FitsTensorDataset(
    "observations/*.fits",
    hdu=0,
    label_key="CLASS",        # header keyword → int label
    transform=None,            # optional callable
    device="cpu",
    mmap=True,
    add_channel_dim=False,     # Tensor default: leave rank alone
    cache_dir=None,            # optional remote materialization dir
)
loader = make_loader(ds, batch_size=32, num_workers=4)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `paths` | `str` or `list[str]` | *(required)* | File paths, glob, or HTTP(S) URLs |
| `hdu` | `int` or `str` or `sequence` | `0` | Flux HDU(s); multi → channel stack |
| `ivar_hdu` / `mask_hdu` | same arity as `hdu` | `None` | Companion HDUs (not flux channels) |
| `label_key` | `str` or `None` | `None` | Header keyword for classification labels |
| `labels` | `list[int]` or `None` | `None` | Explicit per-file labels (overrides `label_key`) |
| `transform` | `callable` or `None` | `None` | Applied to each payload |
| `device` | `str` | `"cpu"` | Torch device |
| `mmap` | `bool` or `str` | `True` | Memory-mapped reads |
| `add_channel_dim` | `bool` | `False` | Prepend channel dim for 2D |
| `cache_dir` | `str` or `Path` or `None` | `None` | Override remote prefetch directory |

**Returns per item:** `(payload, label)` where payload is a `Tensor` or
`{"flux", "ivar"?, "mask"?}`.

!!! info "When to use"
    Use `FitsTensorDataset` when rank/layout is unknown or non-2D. Prefer
    `FitsImageDataset` / `FitsCubeDataset` / `FitsSpectrumDataset` for those
    shapes. For 100k+ files, prefer `FitsTensorIterableDataset`.

---

## `FitsImageDataset` / `FitsCubeDataset`

Subclasses of `FitsTensorDataset` — same constructor signature and returns,
with two defaults changed:

- **Image** — 2D; `add_channel_dim=True` by default; multi-band HDUs → `[C,H,W]`.
- **Cube** — 3D+; `add_channel_dim=False` by default; adds `slice_index`.

```python
from torchfits.data import FitsImageDataset, FitsCubeDataset

images = FitsImageDataset("obs/*.fits", hdu=0, label_key="CLASS")
cubes = FitsCubeDataset("cubes/*.fits", hdu=0, slice_index=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `paths` | `str` or `list[str]` | *(required)* | File paths, glob, or HTTP(S) URLs |
| `hdu` | `int` or `str` or `sequence` | `0` | Flux HDU(s); multi → channel stack |
| `add_channel_dim` | `bool` | `True` (Image) / `False` (Cube) | Prepend channel dim for rank-2 payloads |
| `slice_index` | `int` or `None` | `None` (`FitsCubeDataset` only) | Index the leading axis after read |
| `ivar_hdu` / `mask_hdu` / `label_key` / `labels` / `transform` / `device` / `mmap` / `cache_dir` | — | see `FitsTensorDataset` | Passed through unchanged |

**Returns per item:** same as `FitsTensorDataset` — `(payload, label)`.

`FitsImageIterableDataset` stays first-class for loaders (same knobs as the
tensor iterable, channel dim default on).

---

## `FitsSpectrumDataset`

1D spectra (IMAGE `NAXIS=1` or table `column=` + optional `ivar_column=`), plus
DESI-style 2D `[nspec, nwave]` via `row=`. Multi-arm (MOS B/R/Z) uses `layout=`:

| `layout` | Behavior |
|---|---|
| `"dict"` (default) | Per-arm `{name: {flux, ivar?, mask?}}` (flat keys if one arm) |
| `"stack"` | Flux `[C, nwave]` only if all arms share `nwave` |
| `"concat"` | One 1D flux + parallel ivar/mask along wavelength |

```python
from torchfits.data import FitsSpectrumDataset

ds = FitsSpectrumDataset(
    "spectra/*.fits", hdu=["B", "R", "Z"], layout="dict",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `paths` | `str` or `list[str]` | *(required)* | File paths, glob, or HTTP(S) URLs |
| `hdu` | `int` or `str` or `sequence` | `0` | IMAGE flux HDU(s), one per arm |
| `ivar_hdu` / `mask_hdu` | same arity as `hdu` | `None` | Companion IMAGE HDUs per arm |
| `column` | `str` or `None` | `None` | Table flux column (mutually exclusive with `hdu` image path) |
| `ivar_column` | `str` or `None` | `None` | Table ivar column |
| `row` | `int` or `None` | `None` | Row index into a DESI-style `[nspec, nwave]` HDU/column |
| `layout` | `str` | `"dict"` | `"dict"`, `"stack"`, or `"concat"` |
| `transform` | `callable` or `None` | `None` | Applied to the laid-out payload |
| `device` | `str` | `"cpu"` | Torch device |
| `mmap` | `bool` or `str` | `True` | Memory-mapped reads |
| `cache_dir` | `str` or `Path` or `None` | `None` | Override remote prefetch directory |

**Returns per item:** payload per `layout` (see table above) — no label.

Synthetic DESI-shaped demo: `examples/desi_shaped_spectrum.py`.

---

## `FitsTensorIterableDataset`

Iterable dataset for multi-worker sharded tensor loading. Each worker processes
a deterministic subset — every file is seen exactly once per epoch.

```python
from torchfits.data import FitsTensorIterableDataset

ds = FitsTensorIterableDataset(
    "observations/*.fits",
    hdu=0,
    shuffle=True,
    seed=42,
    add_channel_dim=False,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `paths` | `str` or `list[str]` | *(required)* | File paths, glob, or HTTP(S) URLs |
| `hdu` | `int` or `str` or `sequence` | `0` | Flux HDU(s) |
| `transform` | `callable` or `None` | `None` | Applied to each payload |
| `device` | `str` | `"cpu"` | Torch device |
| `mmap` | `bool` or `str` | `True` | Memory-mapped reads |
| `shuffle` | `bool` | `False` | Shuffle file order per epoch |
| `seed` | `int` | `0` | Base seed for shuffling |
| `add_channel_dim` | `bool` | `False` | Prepend channel dimension |
| `cache_dir` | `str` or `Path` or `None` | `None` | Override remote prefetch directory |

**Returns per item:** payload (`Tensor` or flux/ivar/mask dict) — no label.

!!! info "When to use"
    Use when you have many files and want deterministic multi-worker loading
    without file duplication. Unlike map-style + `num_workers`, each worker
    gets a disjoint shard of files.

---

## `FitsTableDataset`

Map-style dataset for row-indexable FITS catalogs. Loads the full filtered
table at `__init__` — use only when the catalog fits in RAM.

```python
from torchfits.data import FitsTableDataset

ds = FitsTableDataset(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20",        # predicate pushdown at load time
    labels=[0, 1, 0, 1, ...],  # optional per-row labels (default 0)
    transform=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdu` | `int` | `1` | Table HDU index |
| `columns` | `list[str]` or `None` | `None` | Column names (None = all) |
| `where` | `str` or `None` | `None` | SQL-like predicate for row filtering |
| `labels` | `list[int]` or `None` | `None` | Per-row integer labels (default 0 when omitted) |
| `transform` | `callable` or `None` | `None` | Applied to each row dict |
| `device` | `str` | `"cpu"` | Torch device |
| `mmap` | `bool` or `str` | `"auto"` | Memory-mapped reads |

**Returns per item:** `(dict[str, Tensor], torch.Tensor)` — row dict and `torch.long` label.

!!! info "When to use"
    Use for catalogs up to a few million rows that fit in memory. The
    `where=` predicate is pushed down so only matching rows are loaded. For
    larger catalogs, use `FitsTableIterableDataset`.

---

## `FitsTableIterableDataset`

Streams table rows in constant memory. Each `__getitem__` yields one
`dict[str, Tensor]` row.

```python
from torchfits.data import FitsTableIterableDataset

ds = FitsTableIterableDataset(
    "survey.fits",
    hdu=1,
    columns=["RA", "DEC"],
    where="DEC > 0",
    batch_size=65536,
    mmap="auto",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | FITS file path |
| `hdu` | `int` | `1` | Table HDU index |
| `columns` | `list[str]` or `None` | `None` | Column projection |
| `where` | `str` or `None` | `None` | SQL-like predicate |
| `batch_size` | `int` | `65536` | Rows per internal scan batch |
| `transform` | `callable` or `None` | `None` | Applied to each row dict |
| `device` | `str` | `"cpu"` | Torch device |
| `mmap` | `bool` or `str` | `"auto"` | Memory-mapped reads |

**Returns per item:** `dict[str, Tensor | Any]` — one row.

!!! warning "where= performance"
    The ``where=`` path uses ``table.scan()`` which yields Arrow
    ``RecordBatch`` objects and converts each row to Python individually.
    This is slower than the non-``where=`` path which uses
    ``table.scan_torch`` for direct tensor-column chunks. For
    performance-sensitive filtered streaming on large catalogs, consider
    reading the full table with ``table.scan_torch`` and filtering in
    PyTorch.

!!! info "When to use"
    Use for large catalogs that don't fit in memory. Workers shard by scan
    batch index (`batch_idx % num_workers == worker_id`), so each row is
    seen exactly once. Row order within a batch is preserved.

---

## `FitsCutoutDataset`

Map-style dataset for fixed cutout windows from one or more FITS images.

```python
from torchfits.data import FitsCutoutDataset

cutouts = [
    ("mosaic.fits", 0, 100, 200, 164),   # (path, hdu, x, y, size)
    ("mosaic.fits", 0, 300, 400, 164),
]
ds = FitsCutoutDataset(cutouts, transform=None, device="cpu")
```

Accepts `(path, hdu, x, y, size)` or `(path, hdu, x1, y1, x2, y2)` tuples.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cutouts` | `Sequence` | *(required)* | List of cutout specs |
| `transform` | `callable` or `None` | `None` | Applied to each cutout tensor |
| `device` | `str` | `"cpu"` | Torch device |
| `add_channel_dim` | `bool` | `True` | Prepend channel dimension |

**Returns per item:** `Tensor` — raw cutout tensor without a label. This dataset
is unsupervised by design (it has no `labels` parameter). For supervised
patch classification, subclass `FitsCutoutDataset` and override
`__getitem__` to return `(tensor, label)`, or create a custom `Dataset`
that pairs `read_subset` calls with labels.

!!! info "When to use"
    Use for patch training from a mosaic. Each cutout uses **pixel coordinates**
    supplied in the spec — torchfits has no WCS layer. If most cutouts share one
    file, `open_subset_reader` avoids repeated handle opens.

---

## `make_loader()`

Thin factory: `torch.utils.data.DataLoader(dataset, ...)` plus an optional
call to `cache.optimize_for_dataset` before the loader is returned. Any
`Dataset` — torchfits or plain PyTorch — works with either path; the two
produce identical batches when `optimize_cache=False`.

```python
from torchfits.data import make_loader

loader = make_loader(
    ds,
    batch_size=32,
    num_workers=4,
    pin_memory=True,       # faster CPU → GPU transfer
    optimize_cache=True,   # set cache policy for the file set (no C++ handle pool)
    shuffle=True,          # default for map-style; False for iterable
)
```

**Build `DataLoader` yourself** when any of these apply:

- You already have a `collate_fn`, `sampler`, or `batch_sampler` that
  `make_loader`'s keyword surface doesn't expose directly (pass through
  `**loader_kwargs`, but a hand-built call is clearer once you're combining
  several).
- Your dataset has no `files` attribute — `optimize_cache=True` is then a
  silent no-op (see [Cache: how and when](#cache-how-and-when)); skip the
  torchfits wrapper and set `optimize_cache=False` or call `DataLoader`
  directly.
- You manage handle-cache warm-up elsewhere (e.g. once per training run
  rather than per loader construction).

Side-by-side runnable comparison:
`examples/example_make_loader_vs_dataloader.py`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | `Dataset` or `IterableDataset` | *(required)* | A torchfits dataset |
| `batch_size` | `int` | `32` | Batch size |
| `shuffle` | `bool` or `None` | `None` | Auto: `True` for map-style, `False` for iterable |
| `num_workers` | `int` | `0` | Worker processes |
| `pin_memory` | `bool` | `False` | Pin for GPU transfers |
| `prefetch_factor` | `int` | `2` | Prefetch per worker |
| `drop_last` | `bool` | `False` | Drop incomplete batches |
| `optimize_cache` | `bool` | `True` | Call `cache.optimize_for_dataset` |
| `avg_file_size_mb` | `float` | `10.0` | For cache sizing |
| `collate_fn` | `callable` | `fits_collate_fn` | Batch collation |
| `**loader_kwargs` | | | Passed to `DataLoader` |

**Returns:** `torch.utils.data.DataLoader`

---

## `fits_collate_fn`

Default collation function for torchfits datasets:

- `list[Tensor]` → stacked `Tensor` (all must have the same shape)
- `list[(Tensor, Tensor)]` → `(stacked_images, stacked_labels)`
- `list[dict[str, Tensor]]` → `dict[str, stacked_Tensor]`
- `list[(dict[str, Tensor], Tensor)]` → `(stacked_dict, stacked_labels)`

All tensors in a batch must have identical shapes — `torch.stack` is used
under the hood. Raises `ValueError` on non-tensor columns (strings, VLA
lists). For variable-size images, either pad externally before collation or
supply a custom `collate_fn`.

---

## Worker Sharding

### Images (`FitsImageIterableDataset`)

File indices are partitioned by `worker_id`:

```
per_worker = total // num_workers
remainder  = total %  num_workers
start      = worker_id * per_worker + min(worker_id, remainder)
size       = per_worker + (1 if worker_id < remainder else 0)
```

Every file is seen exactly once per epoch. Shuffling permutes within each
worker's shard (epoch-seeded).

### Tables (`FitsTableIterableDataset`)

Scan batches are assigned to workers via `batch_idx % num_workers == worker_id`.
Row order within a batch is preserved. Workers do not interleave individual
rows.

!!! tip "Multi-worker tip"
    Use `num_workers=2` or more for parallel reads. Each worker independently
    opens files and reads data — no GIL contention on the I/O path. Add
    `persistent_workers=True` (passed via `**loader_kwargs`) to keep workers
    alive between epochs — this avoids re-opening file handles each epoch and
    significantly speeds up epoch startup for mmap-backed datasets.
