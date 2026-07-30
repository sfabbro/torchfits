# Python workflows

Which torchfits API to use for common FITS jobs — between
[Quick start](quickstart.md) (shortest path) and the
[API Reference](api.md) (signatures and kwargs).

Coming from **Astropy** or **fitsio**? Skim the
[Astropy](migration_astropy.md) / [fitsio](migration_fitsio.md) migration
tables first, then use this page for torchfits-native patterns.

---

## Choose by job

| Job | Start with | Then see |
|-----|------------|----------|
| One image → tensor | `read_tensor` | [Images](#images-and-hdus) |
| Image + header | `read(..., return_header=True)` | [Core I/O](api-core-io.md) |
| Catalog filter / project | `table.read(..., where=)` | [Tables](#tables-as-dataframes) |
| Columns as tensors | `table.read_torch` | [Tables](api-tables.md) |
| Huge catalog, stream | `table.scan` / `scan_polars` | [Tables](api-tables.md) |
| Many cutouts from one mosaic | `open_subset_reader` | [Cutouts](#cutouts-and-mefs) |
| Multi-extension file | `open` / `read_hdus` | [Images](#images-and-hdus) |
| Train on many files / rows | `Fits*Dataset` + `make_loader` | [Datasets](#datasets-and-loaders) |
| Shell inspect / one-off | `torchfits` CLI | [CLI](cli.md) |

---

## Images and HDUs

```python
import torchfits

# IMAGE HDU → tensor (mmap=True by default)
tensor = torchfits.read_tensor("image.fits", hdu=0, device="cpu")

# Same data + header cards
data, header = torchfits.read("image.fits", hdu=0, return_header=True)
print(header.get("OBJECT"))
```

`hdu` is an integer index or an **EXTNAME** string (`"SCI"`, `"EVENTS"`, …).
When several HDUs share a name, only the **first** match is returned — use a
numeric index for the others.

!!! tip "mmap and devices"
    Keep `mmap=True` for large local images. Prefer `mmap=False` when many
    workers open the same files. `device="cuda"` / `"mps"` copies to the
    accelerator after the host read.

`torchfits.read` without `return_header` returns a tensor for IMAGE HDUs, or
a `dict[str, Tensor]` for table HDUs. For catalogs as dataframes, prefer
`table.read` below instead of root `read`.

More: [Core I/O](api-core-io.md) · [Examples](examples.md)

---

## Tables as dataframes

```python
df = torchfits.table.read(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20",
)
print(df.num_rows)  # pyarrow.Table

cols = torchfits.table.read_torch(
    "catalog.fits", hdu=1, columns=["RA", "DEC"]
)

for batch in torchfits.table.scan("survey.fits", hdu=1, batch_size=50_000):
    process(batch)  # pyarrow.RecordBatch
```

!!! warning "Two table shapes"
    - `torchfits.table.read` → **Arrow** (`pyarrow.Table`)
    - `torchfits.table.read_torch` / root `read` on a table HDU → **`dict[str, Tensor]`**

    Pick one shape per code path. Do not assume `table.read` returns tensors.

For tables that do not fit in RAM, use `scan` / `scan_polars`. Avoid
materializing the full table only to call `.lazy()` afterward.

More: [Tables](api-tables.md) · [Astropy migration](migration_astropy.md)

---

## Cutouts and MEFs

Python / `--box` windows are **0-based, half-open**
`[x1, x2) × [y1, y2)` (NumPy slicing). Shell users can keep the familiar
**CFITSIO path section** (`file.fits[1:256,1:256]`, 1-based inclusive).
Do not mix the two coordinate systems on one call.

```python
# One stamp
stamp = torchfits.read_subset(
    "mosaic.fits", hdu=0, x1=100, y1=100, x2=200, y2=200
)

# Many stamps from one open (preferred for large mosaics)
with torchfits.open_subset_reader("mosaic.fits", hdu=0) as reader:
    a = reader.read_subset(0, 0, 64, 64)
    b = reader.read_subset(128, 128, 192, 192)

# Multi-extension: by index or first matching EXTNAME
with torchfits.open("mef.fits") as hdul:
    primary = hdul[0].data
    sci = hdul["SCI"]  # first EXTNAME == "SCI"
```

Shell:

```bash
torchfits cutout 'mosaic.fits[101:200,101:200]' stamp.fits
torchfits cutout mosaic.fits stamp.fits --box 100,100,200,200
```

See [CLI](cli.md) and [CLI recipes](cli-recipes.md).

More: [`read_subset`](api-core-io.md#read_subset) ·
[`open_subset_reader`](api-core-io.md#open_subset_reader)

---

## Datasets and loaders

`torchfits.data` builds PyTorch Datasets over many FITS files or rows.

```python
from torchfits.data import FitsImageDataset, make_loader

ds = FitsImageDataset("observations/*.fits", label_key="CLASS")
loader = make_loader(ds, batch_size=32, num_workers=4)

for images, labels in loader:
    ...
```

| Dataset | Typical use | Item shape |
|---------|-------------|------------|
| `FitsImageDataset` | Many 2D images | `(tensor, label)` |
| `FitsTableDataset` | Catalog in RAM | `(row_dict, label)` |
| `FitsCutoutDataset` | Patches from a mosaic | tensor |
| `FitsTableIterableDataset` | Huge catalog stream | `row_dict` |

Walkthroughs: [ML with FITS](examples-ml.md).
Reference: [Data module](api-data.md) · [Transforms](api-transforms.md)

---

## CLI vs Python

| Prefer | When |
|--------|------|
| In-process Python API | Pipelines and many reads in one process |
| `torchfits` CLI | Inspect, checksums, one-off cutouts, shell scripts |

Each CLI invocation pays a PyTorch/extension import (~1 s on many laptops).
For tight loops, stay in Python.

Tour: [CLI](cli.md) · Recipes: [CLI recipes](cli-recipes.md)

---

## What's next?

1. [Quick start](quickstart.md) — install → first tensor in minutes  
2. [Examples](examples.md) — runnable scripts and galleries  
3. [API Reference](api.md) — full signatures  
4. [Parity](parity.md) — supported vs out-of-scope vs Astropy / fitsio / CFITSIO  

!!! tip "Optional: coding agents"
    You can point an agent at this docs site or the repo for help on *your*
    files. Skimming this page (or Quick start) first keeps the conversation
    grounded in the real public API.
