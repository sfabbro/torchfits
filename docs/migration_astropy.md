# Migrating from Astropy to torchfits

Side-by-side comparison and migration recipes for common astronomical FITS I/O tasks between `astropy.io.fits` and `torchfits`.

For a complete breakdown of supported FITS standard features, see the [Feature Parity Matrix](parity.md). For end-to-end Python examples, see [Python Workflows](python-workflows.md) and [Examples](examples.md).

---

## Reading Images & Datacubes

| Task | `astropy.io.fits` | `torchfits` |
|---|---|---|
| **Read image array** | `astropy.io.fits.getdata(path)` | `torchfits.read(path)` |
| **Read as PyTorch Tensor** | `torch.from_numpy(astropy.io.fits.getdata(path))` | `torchfits.read_tensor(path, hdu=0)` |
| **Read with memory-mapping** | `astropy.io.fits.getdata(path, use_mmap=True)` | `torchfits.read_tensor(path, hdu=0, mmap=True)` |
| **Direct GPU placement** | `torch.from_numpy(astropy.io.fits.getdata(path)).cuda()` | `torchfits.read_tensor(path, hdu=0, device="cuda")` |
| **Read image & header** | `hdul = astropy.io.fits.open(path); data = hdul[0].data; hdr = hdul[0].header` | `data, header = torchfits.read(path, hdu=0, return_header=True)` |
| **Windowed cutout read** | `astropy.io.fits.open(path, memmap=True)[0].section[y1:y2, x1:x2]` | `torchfits.read_subset(path, hdu=0, x1=x1, y1=y1, x2=x2, y2=y2)` |

---

## Reading Tables & Catalogs

| Task | `astropy.io.fits` | `torchfits` |
|---|---|---|
| **Read entire table** | `astropy.io.fits.getdata(path, ext=1)` | `torchfits.table.read(path, hdu=1)` *(Returns Arrow table)* |
| **Filtered read (`WHERE`)** | `t = …; mask = t['RA'] > 0; t[mask]` | `torchfits.table.read(path, hdu=1, where="RA > 0")` |
| **Column projection** | `t[['RA', 'DEC']]` | `torchfits.table.read(path, hdu=1, columns=["RA", "DEC"])` |
| **Columns as PyTorch tensors** | `[torch.from_numpy(t[col]) for col in cols]` | `torchfits.table.read_torch(path, hdu=1)` |
| **Stream row batches** | *(Manual chunking)* | `for batch in torchfits.table.scan(path, hdu=1, batch_size=50000): …` |
| **Polars DataFrame** | *(Manual conversion)* | `torchfits.table.read_polars(path, hdu=1)` |

---

## Writing FITS Files

| Task | `astropy.io.fits` | `torchfits` |
|---|---|---|
| **Write image tensor** | `astropy.io.fits.PrimaryHDU(tensor.numpy()).writeto(path)` | `torchfits.write_tensor(path, tensor)` |
| **Write table catalog** | `astropy.io.fits.BinTableHDU(table).writeto(path)` | `torchfits.table.write(path, table_dict)` |
| **Write with header metadata** | `hdu = astropy.io.fits.PrimaryHDU(data); hdu.header['OBJECT'] = 'M31'; hdu.writeto(path)` | `torchfits.write(path, data, header={'OBJECT': 'M31'})` |

---

## Multi-Extension Files (MEF)

| Task | `astropy.io.fits` | `torchfits` |
|---|---|---|
| **Open MEF container** | `hdul = astropy.io.fits.open(path)` | `with torchfits.open(path) as hdul: …` |
| **Read HDU by `EXTNAME`** | `hdul['SCI'].data` | `torchfits.read_hdus(path, hdus=['SCI'])` |
| **Read multiple HDUs at once** | `[hdul[i].data for i in range(3)]` | `torchfits.read_hdus(path, hdus=[0, 1, 2])` |

---

## Compression & Checksums

| Task | `astropy.io.fits` | `torchfits` |
|---|---|---|
| **Read Rice-compressed (`.fz`)** | `astropy.io.fits.open(path)[1].data` | `torchfits.read(path, hdu=1)` *(Auto-decompressed)* |
| **Write tile-compressed image** | `astropy.io.fits.CompImageHDU(data, compression_type='RICE_1').writeto(path)` | `torchfits.write(path, data, compress="rice")` |
| **Verify FITS checksums** | *(Manual calculation)* | `torchfits.verify_checksums(path)` |
| **Write standard checksums** | `hdul.writeto(path, checksum=True)` | `torchfits.write_checksums(path)` |

---

## Key Behavioral Differences

### 1. Data Scaling & Precision
- **Astropy:** Applies `BSCALE`/`BZERO` scaling on the CPU when data is loaded, often promoting integer arrays to 64-bit float (`float64`).
- **torchfits:** Fuses scaling directly into vectorized SIMD loops, outputting standard 32-bit float (`float32`) tensors ideal for PyTorch models and GPU memory efficiency. Use `raw_scale=True` to preserve raw storage integers without conversion.

### 2. Tabular Data Representation
- **Astropy:** Returns custom `astropy.table.Table` or NumPy record arrays (`numpy.recarray`).
- **torchfits:** Returns zero-copy Apache Arrow tables (`pyarrow.Table`), PyTorch tensor dictionaries (`dict[str, torch.Tensor]`), or Polars DataFrames (`FITSPolarsFrame`), providing immediate compatibility with modern data science ecosystems.

### 3. Thread Safety & Multi-Worker Loaders
- **Astropy:** `HDUList` instances are not thread-safe and can cause file descriptor corruption when shared across threads or PyTorch `DataLoader` worker processes.
- **torchfits:** Every read opens an independent private C++ handle and releases the Python GIL during decoding, ensuring safe execution across multi-threaded pipelines and multi-worker `DataLoader` instances.

