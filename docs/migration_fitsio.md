# Migration from fitsio to torchfits

Side-by-side replacements for common FITS I/O. See
[Benchmarks](benchmarks.md#performance-deficits) for cases where fitsio still
wins, and [Python workflows](python-workflows.md) for torchfits-native
patterns.

## Reading an image

| Operation | fitsio | torchfits |
|-----------|--------|-----------|
| Read image | `fitsio.read(path)` | `torchfits.read(path)` |
| Read image as tensor | `torch.from_numpy(fitsio.read(path))` | `torchfits.read_tensor(path, hdu=0)` |
| Read image with mmap | *(fitsio has no mmap mode; use slice reads)* | `torchfits.read_tensor(path, hdu=0, mmap=True)` |
| Read image to GPU | `torch.from_numpy(fitsio.read(path)).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda")` |
| Read header | `fitsio.read_header(path)` | `torchfits.read_header(path, hdu=0)` |

## Reading a table

| Operation | fitsio | torchfits |
|-----------|--------|-----------|
| Read all rows | `fitsio.read(path, ext=1)` | `torchfits.table.read(path, hdu=1)` |
| Read with WHERE | `fitsio.FITS(path)[1].where("RA > 0")` | `torchfits.table.read(path, hdu=1, where="RA > 0")` |
| Subset of columns | `fitsio.read(path, ext=1, columns=['RA','DEC'])` | `torchfits.table.read(path, hdu=1, columns=["RA","DEC"])` |
| Columns as tensors | per-column `fitsio.read` + `torch.from_numpy` | `torchfits.table.read_torch(path, hdu=1)` |
| Stream tensor chunks | iterate fitsio rows | `torchfits.table.scan_torch(path, hdu=1, batch_size=10000)` |
| Polars | *(manual)* | `torchfits.table.read_polars(path, hdu=1)` |

`table.read_torch(..., where=)` accepts only simple compare / `BETWEEN` /
`AND`. For `OR` / `IN` / `IS NULL`, use `table.read(..., where=...)`.

## Writing

| Operation | fitsio | torchfits |
|-----------|--------|-----------|
| Write tensor | `fitsio.write(path, tensor.numpy())` | `torchfits.write_tensor(path, tensor)` |
| Write table | `fitsio.write(path, table_dict)` | `torchfits.table.write(path, table_dict)` |
| Append rows | `f.append(table_dict)` | `torchfits.table.append_rows(path, rows, hdu=1)` |
| Update rows | `f[1].update(rows, row_slice)` | `torchfits.table.update_rows(path, rows, row_slice, hdu=1)` |
| Insert column | `f[1].insert_column(name, values)` | `torchfits.table.insert_column(path, name, values, hdu=1)` |
| Rename column | `f[1].rename_column(old, new)` | `torchfits.table.rename_columns(path, {old: new}, hdu=1)` |
| Drop column | `f[1].delete_column(name)` | `torchfits.table.drop_columns(path, [name], hdu=1)` |

## Predicate pushdown

| Operation | fitsio | torchfits |
|-----------|--------|-----------|
| Column filter | `fitsio.FITS(path)[1].where("MAG_G < 20")` | `torchfits.table.read(path, hdu=1, where="MAG_G < 20")` |
| Compound predicate | `fitsio.FITS(path)[1].where("MAG_G < 20 AND DEC > 0")` | `torchfits.table.read(path, hdu=1, where="MAG_G < 20 AND DEC > 0")` |
| IN list | `fitsio.FITS(path)[1].where("id IN (1,2,3)")` | `torchfits.table.read(path, hdu=1, where="id IN (1, 2, 3)")` |

## GPU transfer

| Operation | fitsio | torchfits |
|-----------|--------|-----------|
| Image to GPU | `torch.from_numpy(fitsio.read(path)).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda")` |
| Narrow dtype GPU (uint16) | `torch.from_numpy(fitsio.read(path)).to(torch.uint16).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda", raw_scale=True)` |

## Performance notes

| Metric | fitsio | torchfits |
|--------|--------|-----------|
| Large float32 image (16 MB, CPU) | 5.25 ms | 3.85 ms (**~1.4× faster**) |
| Same read @ CUDA | 12.22 ms | 8.37 ms (**~1.5× faster**) |
| Compressed Rice image (CPU) | 18.03 ms | 18.16 ms (**~parity**) |
| 50× repeated 100×100 cutouts (CPU) | 18.16 ms | 13.60 ms (**~1.3× faster**) |
| Table read (100k rows, 8 cols, mixed) | 31.45 ms | 5.65 ms (**~5.6× faster**) |

*Medians from Round-3 scorecard (`exhaustive_mps_20260719_143706` / CUDA
`exhaustive_cuda_20260719_144457`); see [Benchmarks](benchmarks.md) host
scorecard. Absolute highlight times refresh with each soak — prefer
`docs/benchmarks.md#performance-highlights` over this migration snapshot when
numbers disagree.*

## Key Behavioral Differences

### 1. Multi-Processing Fork Safety
* **fitsio**: High-performance CFITSIO reads into NumPy; there is no true OS
  `mmap` toggle (`memmap=` is ignored). Long-lived `FITS` handles across
  `DataLoader` forks can still share CFITSIO state poorly — prefer reopen-per-
  worker or torchfits datasets.
* **torchfits**: Use `torchfits.data` datasets with `make_loader` for multi-worker loading. Map-style datasets are worker-safe (each worker reads independently); iterable datasets shard by `worker_id`. Since rc2, image/table reads use private CFITSIO handles per call — no shared-handle LRU across threads. Call `torchfits.cache.optimize_for_dataset(paths)` when the dataset exposes `files` (also invoked by `make_loader` by default) to size metadata caches before training.

### 2. Table Mutations
* **fitsio**: In-place updates can corrupt FITS tables if not handled carefully, and do not invalidate read buffers automatically.
* **torchfits**: Functions under `torchfits.table` (like `append_rows`, `update_rows`, `insert_column`, `rename_columns`, `drop_columns`) perform parallel columns reconstruction and automatically invalidate all Python-side and C++ handle/meta caches, preventing stale reads.

### 3. Variable Length Arrays (VLAs)
* **fitsio**: Reads VLA columns as NumPy arrays of object pointers (`object` dtype).
* **torchfits**: Translates VLAs to standard PyArrow `ListArray` types, allowing high-performance, memory-contiguous vectorization on the CPU.

