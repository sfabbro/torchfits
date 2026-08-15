# Migrating from fitsio to torchfits

Side-by-side comparison and migration recipes for common astronomical FITS I/O tasks between `fitsio` and `torchfits`.

For a complete breakdown of supported FITS standard features, see the [Feature Parity Matrix](parity.md). For end-to-end Python examples, see [Python Workflows](python-workflows.md) and [Examples](examples.md).

---

## Reading Images & Datacubes

| Task | `fitsio` | `torchfits` |
|---|---|---|
| **Read image array** | `fitsio.read(path)` | `torchfits.read(path)` |
| **Read as PyTorch Tensor** | `torch.from_numpy(fitsio.read(path))` | `torchfits.read_tensor(path, hdu=0)` |
| **Read with memory-mapping** | *(Not supported in fitsio)* | `torchfits.read_tensor(path, hdu=0, mmap=True)` |
| **Direct GPU placement** | `torch.from_numpy(fitsio.read(path)).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda")` |
| **Read header only** | `fitsio.read_header(path)` | `torchfits.read_header(path, hdu=0)` |
| **Windowed cutout read** | `fitsio.read(path, ext=0, rows=[y1, y2], cols=[x1, x2])` | `torchfits.read_subset(path, hdu=0, x1=x1, y1=y1, x2=x2, y2=y2)` |

---

## Reading Tables & Catalogs

| Task | `fitsio` | `torchfits` |
|---|---|---|
| **Read entire table** | `fitsio.read(path, ext=1)` | `torchfits.table.read(path, hdu=1)` *(Returns Arrow table)* |
| **Filtered read (`WHERE`)** | `fitsio.FITS(path)[1].where("RA > 0")` | `torchfits.table.read(path, hdu=1, where="RA > 0")` |
| **Column projection** | `fitsio.read(path, ext=1, columns=['RA', 'DEC'])` | `torchfits.table.read(path, hdu=1, columns=["RA", "DEC"])` |
| **Columns as PyTorch tensors** | `[torch.from_numpy(col) for col in cols]` | `torchfits.table.read_torch(path, hdu=1)` |
| **Stream tensor chunks** | *(Manual chunking)* | `for chunk in torchfits.table.scan_torch(path, hdu=1, batch_size=10000): …` |
| **Polars DataFrame** | *(Manual conversion)* | `torchfits.table.read_polars(path, hdu=1)` |

---

## Writing & Table Mutations

| Task | `fitsio` | `torchfits` |
|---|---|---|
| **Write image tensor** | `fitsio.write(path, tensor.numpy())` | `torchfits.write_tensor(path, tensor)` |
| **Write table catalog** | `fitsio.write(path, table_dict)` | `torchfits.table.write(path, table_dict)` |
| **Append rows to table** | `f.append(table_dict)` | `torchfits.table.append_rows(path, rows, hdu=1)` |
| **Update rows in place** | `f[1].update(rows, row_slice)` | `torchfits.table.update_rows(path, rows, row_slice, hdu=1)` |
| **Insert table column** | `f[1].insert_column(name, values)` | `torchfits.table.insert_column(path, name, values, hdu=1)` |
| **Rename table columns** | `f[1].rename_column(old, new)` | `torchfits.table.rename_columns(path, {old: new}, hdu=1)` |
| **Drop table columns** | `f[1].delete_column(name)` | `torchfits.table.drop_columns(path, [name], hdu=1)` |

---

## Key Behavioral Differences

### 1. Multi-Processing & DataLoader Fork Safety
- **fitsio:** Long-lived `fitsio.FITS` file handles shared across `torch.utils.data.DataLoader` worker forks can encounter CFITSIO state conflicts or descriptor issues.
- **torchfits:** Native PyTorch `Dataset` implementations in `torchfits.data` (e.g. `FitsImageDataset`, `FitsCutoutDataset`) open independent private CFITSIO handles per call and release the GIL during decoding, ensuring safe execution across multi-worker data loaders.

### 2. Table Mutations & Cache Invalidation
- **fitsio:** In-place modifications to FITS tables require manual handle management and do not synchronize with read caches.
- **torchfits:** In-place table operations in `torchfits.table` automatically invalidate internal metadata caches, ensuring immediate consistency for subsequent reads.

### 3. Variable-Length Arrays (VLAs)
- **fitsio:** Reads VLA columns as NumPy arrays with Python object pointers (`dtype=object`), incurring Python object overhead.
- **torchfits:** Decodes VLAs directly into standard Apache Arrow `ListArray` structures, maintaining memory-contiguous CPU representations.

