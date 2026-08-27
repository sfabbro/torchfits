# Feature Parity with Astropy & fitsio

A comprehensive comparison of astronomical FITS standard features supported by `torchfits` compared to `astropy.io.fits` and `fitsio`.

For environment, Python, and PyTorch ABI compatibility, see [Environment Compatibility](compatibility.md).

---

## Capabilities Overview

Status values:

- **Supported**: Fully implemented and verified against standard FITS test suites.
- **Partial**: Supported for common workflows with documented boundary behavior.
- **Out of Scope**: High-level astronomy models (coordinates, cosmology, units) that belong in dedicated domain packages.

---

## 1. Images, Datacubes & Multi-Extension Files (MEF)

| Feature | Status | Comparator | Implementation Details & Behavior |
|---|:---:|---|---|
| **2D Image Reading & Writing** | **Supported** | `astropy.io.fits`, `fitsio` | Direct C++ decoding to PyTorch tensors (`float32`, `float64`, `int16`, `int32`, `int64`, `uint8`). |
| **3D/4D Datacubes** | **Supported** | `astropy.io.fits`, `fitsio` | Reads multi-dimensional cubes (e.g. IFU datacubes, radio velocity channels) into $[D, H, W]$ tensors. |
| **Multi-Extension (MEF)** | **Supported** | `astropy.io.fits.HDUList` | HDU iteration by integer index or extension name (`hdul["SCI"]`), lazy metadata scanning. |
| **Windowed Cutout Reads** | **Supported** | `astropy.io.fits`, `fitsio` | Fast pixel sub-region extraction via `read_subset` and zero-overhead `open_subset_reader`. |
| **Unsigned Integers (`uint16`/`uint32`)** | **Supported** | `BZERO` convention | Vectorized SIMD decoding of unsigned integers into native PyTorch integer tensors. |
| **Physical Scaling (`BSCALE`/`BZERO`)** | **Supported** | FITS Standard | Automatically scales raw detector counts to physical float32/float64 values. |

---

## 2. Tile-Compressed Images (`.fits.fz`)

| Compression Algorithm | Status | Read Support | Write Support |
|---|:---:|---|---|
| **Rice (`RICE_1`)** | **Supported** | Fast tile decompression via CFITSIO backend | Compressed tile writing (`compress="rice"`) |
| **Gzip (`GZIP_1`, `GZIP_2`)** | **Supported** | Byte-level and tile-compressed gzip streams | Compressed tile writing (`compress="gzip"`) |
| **H-Compress (`HCOMPRESS_1`)** | **Supported** | Multi-resolution astronomical wavelet decoding | Compressed tile writing (`compress="hcompress"`) |
| **PLIO (`PLIO_1`)** | **Supported** | IRAF-style pixel list mask decoding | Supported |

---

## 3. Astronomical Catalogs & Tables (`torchfits.table`)

| Table Feature | Status | Comparator | Implementation Details & Behavior |
|---|:---:|---|---|
| **Binary Tables (`BINTABLE`)** | **Supported** | `astropy.io.fits`, `fitsio` | High-throughput Arrow decoding into tables and PyTorch tensors. |
| **ASCII Tables (`TABLE`)** | **Supported** | `astropy.io.fits.TableHDU` | Full column parsing and type inference for fixed-width ASCII tables. |
| **Column Projection** | **Supported** | `columns=["ra", "dec"]` | Reads only requested columns from disk, skipping unneeded byte offsets. |
| **Row Slicing (`start_row`, `num_rows`)** | **Supported** | `fitsio` row limits | Reads contiguous row ranges without scanning earlier or later records. |
| **Predicate Filtering (`where=`)** | **Supported** | `fitsio` WHERE clauses | SQL expression pushdown (`"mag < 21.0 AND flag == 0"`) executed during scanning. |
| **In-Place Table Mutation** | **Supported** | `table.update_rows(...)` | Fast in-place column and cell updating on disk via memory-mapping. |

---

## 4. Advanced Table Data Types

| Column Data Type | FITS Format Code | Support Status | Notes |
|---|:---:|:---:|---|
| **Numeric Columns** | `B`, `I`, `J`, `K`, `E`, `D` | **Supported** | Full integer and floating-point support. |
| **Fixed-Width Strings** | `nA` | **Supported** | Decoded as Arrow string arrays with automatic space padding on update. |
| **Boolean Bitmasks** | `X` | **Supported** | Bit-level MSB-first decoding into boolean tensors. |
| **Complex Numbers** | `C`, `M` | **Partial** | Tensor path (`table.read_torch`) decodes `torch.complex64` / `complex128`. Arrow `table.read` raises `NotImplementedError`. |
| **Variable-Length Arrays (VLA)** | `P`, `Q` | **Partial** | Supported via buffered CFITSIO reading; memory-mapped updates not supported. |
| **Scaled Columns (`TSCALn`/`TZEROn`)** | `TSCAL`, `TZERO` | **Supported** | Linear physical count scaling to floating-point tensors. |

---

## 5. Headers & Metadata

| Capability | Status | Notes |
|---|:---:|---|
| **Header Card Access** | **Supported** | Dict-like and attribute access to header keywords, comments, and values. |
| **Header Modification & Export** | **Supported** | Mutate cards and preserve metadata when writing new FITS files. |
| **Checksum & Datasum Verification** | **Supported** | Computes and validates standard FITS `CHECKSUM` and `DATASUM` cards. |

---

## 6. PyTorch ML & Accelerator Integration

| Feature | Status | Notes |
|---|:---:|---|
| **Direct GPU Tensor Placement** | **Supported** | Load FITS images directly onto `device="cuda"` or `device="mps"` without NumPy intermediate steps. |
| **PyTorch `Dataset` Integration** | **Supported** | Built-in `FitsImageDataset`, `FitsCutoutDataset`, `FitsCubeDataset`, and `FitsTableDataset`. |
| **Multi-Worker `DataLoader`** | **Supported** | Factory `make_loader` with automatic batch collation and worker cache warmup. |
| **Header-Aware Transforms** | **Supported** | Astronomical image stretches (Arcsinh, ZScale, Lupton RGB) implemented as PyTorch transforms. |

---

## Scope & Design Philosophy

`torchfits` is designed specifically for high-throughput, low-latency FITS tensor and table I/O.

High-level astronomical analysis tools (such as world coordinate transformations with `astropy.wcs` or unit modeling with `astropy.units`) remain the domain of Astropy and companion libraries. `torchfits` integrates with these libraries by outputting standard PyTorch tensors and Arrow tables.
