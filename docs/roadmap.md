# Project Roadmap

The vision, planned milestones, and architectural evolution of `torchfits`.

---

## 1.0.0 — Foundation (Current)

The 1.0.0 release establishes the high-performance core for FITS tensor and table I/O:

- **Zero-Copy Tensor I/O:** Memory-mapped reads with SIMD-vectorized byte swapping for 1D–4D FITS image extensions.
- **Apache Arrow Table Engine:** Columnar binary and ASCII table reads with SQL predicate pushdown (`where=`) and fast column projection.
- **PyTorch ML Data Loaders:** Native `Dataset` classes (`FitsImageDataset`, `FitsCutoutDataset`, `FitsCubeDataset`, `FitsTableDataset`) and multi-worker `make_loader`.
- **Command-Line Suite:** Unix-style CLI tools (`info`, `header`, `cutout`, `convert`, `checksum`).
- **Feature Parity:** Comprehensive format support verified against standard FITS test suites.

---

## 1.1.0 — Streaming & Codec Enhancements (Planned)

The 1.1 minor release focuses on remote streaming efficiency and expanded compression features:

- **Enhanced Remote HTTP/S3 Range Reads:** Improved row-band caching and range-fetching for remote mosaic cutouts and cloud object stores.
- **Direct Write-Compression Tuning:** Optimized parallel tile compression for multi-CCD mosaic creation.
- **Expanded Table Interoperability:** Zero-copy streaming bridges for DuckDB, Polars, and Pandas.
- **Extended Instrument Profiles:** Pre-tuned cutout readers for large astronomical survey archives.

---

## 2.0.0 — Native C++ / GPU-Direct Architecture (Future)

The 2.0 major release aims to drop external legacy C library dependencies in favor of a modern, native C++/CUDA I/O engine:

- **Direct Storage-to-GPU Transport (GPUDirect Storage):** Direct DMA transfers from NVMe/object storage directly into NVIDIA GPU device memory (cuFile/GDS) without intermediate host-memory bouncing.
- **Native Astronomical Tile Codecs:** Pure C++20 and CUDA implementations of Rice, H-Compress, and Gzip decompression.
- **Asynchronous Batch Execution:** Fully non-blocking multi-file decoders scheduled via CUDA streams and CPU worker pools.
- **Stable Python API:** Maintaining full backwards compatibility with the 1.x `read_tensor`, `table.read`, and `torchfits.data` APIs.

---

## Permanent Scope & Design Boundaries

To ensure focus, long-term maintainability, and peak performance, `torchfits` maintains strict scope boundaries:

- **No Celestial Coordinate Systems or WCS Math:** Coordinate transformations belong in `astropy.wcs`. `torchfits` outputs raw pixel tensors with standard header metadata for Astropy consumption.
- **No Physical Units Engine:** Quantity conversions belong in `astropy.units`.
- **No High-Level Astronomy Modeling:** Source extraction, PSF fitting, and continuum fitting belong in domain analysis packages (e.g. Photutils, SEP).
- **Format Integrity:** Strict compliance with the official IAU FITS standard.
