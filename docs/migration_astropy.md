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

## Reading a table

| Operation | astropy | torchfits |
|-----------|---------|-----------|
| Read all rows | `astropy.io.fits.getdata(path, ext=1)` | `torchfits.table.read(path, hdu=1)` |
| Read with WHERE | `t = …; mask = t['RA'] > 0; t[mask]` | `torchfits.table.read(path, hdu=1, where="RA > 0")` |
| Read subset of columns | `…[['RA','DEC']]` | `torchfits.table.read(path, hdu=1, columns=["RA","DEC"])` |
| Columns as tensors | `torch.from_numpy(t[n])` per column | `torchfits.table.read_torch(path, hdu=1)` |
| Polars | *(manual)* | `torchfits.table.read_polars(path, hdu=1)` |

`table.read_torch(..., where=)` accepts only simple compare / `BETWEEN` /
`AND`. For `OR` / `IN` / `IS NULL`, use `table.read(..., where=...)`.

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

See [Benchmarks → Performance highlights](benchmarks.md#performance-highlights)
for the live table. Snapshot (rc5 CPU exhaustive + Round-3 CUDA):

| Metric | astropy | torchfits |
|--------|---------|-----------|
| Large float32 image (16 MB, CPU) | 5.02 ms | 3.85 ms (**~1.3× faster**) |
| Same read @ CUDA | 5.22 ms | 3.28 ms (**~1.6× faster**) |
| Compressed Rice image (CPU) | 32.85 ms | 12.40 ms (**~2.6× faster**) |
| 50× repeated 100×100 cutouts (CPU) | 50.85 ms | 0.47 ms (**~108× faster**) |
| Table read (100k rows, 8 cols, mixed) | 34.81 ms | 2.56 ms (**~13.6× faster**) |

## Key Behavioral Differences

### 1. Data Scaling & Type Promotion
* **Astropy**: Applies `BSCALE` / `BZERO` on the CPU when HDU data is loaded.
  Integers may promote to `float64` when scaling yields floats.
* **torchfits**: Scaling yields `float32` (on-device by default; pass
  `scale_on_device=False` to read through the host pipeline). On
  `read_tensor`, pass `raw_scale=True` for storage dtypes (e.g. the
  signed-byte / pseudo-unsigned conventions expose `uint8` / `int16`).

### 2. Table Representation
* **Astropy**: `astropy.table.Table` or `numpy.recarray`.
* **torchfits**: `table.read` → `pyarrow.Table`; `table.read_torch` → column
  tensors; `table.read_polars` → Polars. VLAs become Arrow list columns.

### 3. Thread-Safety & Multi-Processing
* **Astropy**: `HDUList` handles are not thread-safe across concurrent opens
  of the same file.
* **torchfits**: Concurrent reads open a private CFITSIO handle per call.
  Shared metadata and the raw `fd` use `pread` under mutexes. For multi-worker
  Datasets, use `torchfits.data` with `make_loader`, and
  `torchfits.cache.optimize_for_dataset(paths)` when the dataset exposes
  `files`.

