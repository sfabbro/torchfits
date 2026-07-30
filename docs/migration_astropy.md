# Migration from astropy to torchfits

Side-by-side replacements for common **FITS I/O** tasks. torchfits covers
tensor/dataframe FITS I/O — see [Parity](parity.md) for the full matrix. For
torchfits-native job patterns, see [Python workflows](python-workflows.md).
For runnable workflows, start with [Examples](examples.md).

## Reading an image

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Read image | `astropy.io.fits.getdata(path)` | `torchfits.read(path)` |
| Read image as tensor | `torch.from_numpy(astropy.io.fits.getdata(path))` | `torchfits.read_tensor(path, hdu=0)` |
| Read image with mmap | `astropy.io.fits.getdata(path, use_mmap=True)` | `torchfits.read_tensor(path, hdu=0, mmap=True)` |
| Read image to GPU | `torch.from_numpy(astropy.io.fits.getdata(path)).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda")` |
| Read image + header | `hdul = astropy.io.fits.open(path); data = hdul[0].data; hdr = hdul[0].header` | `data, header = torchfits.read(path, hdu=0, return_header=True)` |

## Reading a table (dataframe path)

FITS tables are dataframes on disk. Prefer `torchfits.table.read` (Arrow);
use `table.read_torch` for tensor columns.

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Read all rows (dataframe via Arrow) | `astropy.io.fits.getdata(path, ext=1)` | `torchfits.table.read(path, hdu=1)` |
| Read with WHERE | `t = astropy.io.fits.getdata(path, ext=1); mask = t['RA'] > 0; t[mask]` | `torchfits.table.read(path, hdu=1, where="RA > 0")` |
| Read subset of columns | `astropy.io.fits.getdata(path, ext=1)[['RA','DEC']]` | `torchfits.table.read(path, hdu=1, columns=["RA","DEC"])` |
| Dataframe columns as tensors | `t = astropy.io.fits.getdata(path, ext=1); {n: torch.from_numpy(t[n]) for n in t.names}` | `torchfits.table.read_torch(path, hdu=1)` |
| Native Polars dataframe | *(manual)* | `torchfits.table.read_polars(path, hdu=1)` |

## Writing

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Write tensor | `astropy.io.fits.PrimaryHDU(tensor.numpy()).writeto(path)` | `torchfits.write_tensor(path, tensor)` |
| Write table | `astropy.io.fits.BinTableHDU(table).writeto(path)` | `torchfits.table.write(path, table_dict)` |
| Write with header | `hdu = astropy.io.fits.PrimaryHDU(data); hdu.header['KEY'] = val; hdu.writeto(path)` | `torchfits.write(path, data, header={'KEY': val})` |

## Multi-HDU access

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Open MEF | `hdul = astropy.io.fits.open(path)` | `hdul = torchfits.open(path)` |
| Read by EXTNAME | `hdul['SCI'].data` | `torchfits.read_hdus(path, hdus=['SCI'])` |
| Read multiple HDUs | `[hdul[i].data for i in range(3)]` | `torchfits.read_hdus(path, hdus=[0, 1, 2])` |

## GPU transfer

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Image to GPU | `torch.from_numpy(astropy.io.fits.getdata(path)).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda")` |
| Unsigned integer GPU (correct) | `torch.from_numpy(astropy.io.fits.getdata(path)).to(torch.int32).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda")` narrow H2D |

## Compression & checksums

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Read compressed | `astropy.io.fits.open(path)[1].data` | `torchfits.read(path, hdu=1)` (auto-detected) |
| Verify checksums | manual | `torchfits.verify_checksums(path)` |
| Write checksums | `hdul.writeto(path, checksum=True)` | `torchfits.write_checksums(path)` |

## Performance notes

| Metric | astropy | torchfits |
|--------|---------|-----------|
| Large float32 image (16 MB, CPU) | 11.35 ms | 3.85 ms (**~3× faster**) |
| Same read @ CUDA | 15.94 ms | 8.37 ms (**~1.9× faster**) |
| Compressed Rice image (CPU) | 75.33 ms | 18.16 ms (**~4× faster**) |
| 50× repeated 100×100 cutouts (CPU) | 269.20 ms | 13.60 ms (**~20× faster**) |
| Table read (100k rows, 8 cols, mixed) | 98.48 ms | 5.65 ms (**~17× faster**) |

*Medians from Round-3 scorecard (`exhaustive_mps_20260719_143706`); see
[Benchmarks](benchmarks.md) host scorecard. Absolute highlight times refresh
with each soak — prefer `docs/benchmarks.md#performance-highlights` when
numbers disagree. Narrow-table `predicate_filter` can still lag fitsio on some
hosts; MegaCam `torchfits_cached` vs `fitsio_cached` was only ~6.5% behind on
Round-3 medians (and led on a 2026-07-30 re-soak).*

## Key Behavioral Differences

### 1. Data Scaling & Type Promotion
* **Astropy**: Scaling (applying `BSCALE` and `BZERO` keywords) is applied on the CPU when the HDU data is initialized. Integer types (like `uint16` or `int32`) are promoted to double-precision `float64` in memory if the scaling yields floating-point numbers.
* **torchfits**: Optional on-device scaling via `torchfits.read(..., scale_on_device=True)` (via `**kwargs` into the read pipeline). Raw integers transfer to GPU/MPS; `BSCALE`/`BZERO` apply in device registers, yielding `float32` instead of Astropy's default `float64`. `read_tensor` has no `scale_on_device` parameter — use `read()` or pass `raw_scale=True` on `read_tensor` for storage dtypes.

### 2. Table Representation
* **Astropy**: Tables are represented as `astropy.table.Table` or `numpy.recarray`.
* **torchfits**: FITS tables are dataframes on disk. Default path is
  `torchfits.table.read` → `pyarrow.Table` (portable dataframe). Tensor columns
  use `table.read_torch`. Native Polars uses
  `table.read_polars`. VLAs become Arrow list columns.

### 3. Thread-Safety & Multi-Processing
* **Astropy**: HDU handles (`HDUList`) are not thread-safe. Opening the same file in multiple background threads can lead to file descriptor and read-pointer conflicts.
* **torchfits**: Since 1.0.0rc2, concurrent reads open a **private** CFITSIO `fitsfile*` per call (CFITSIO R2). Shared metadata (`SharedReadMeta`) and the raw `fd` use `pread` and stay mutex-guarded; they do not share CHDU state across threads. For PyTorch `DataLoader` workers, use `torchfits.data` datasets with `make_loader`: map-style datasets read independently per worker; iterable datasets shard by `worker_id`. Call `torchfits.cache.optimize_for_dataset(paths)` when the dataset exposes a `files` list to set Python cache policy (C++ handle pooling is disabled).

