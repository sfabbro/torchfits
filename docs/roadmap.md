# Project Roadmap

The vision, planned milestones, and architectural evolution of `torchfits`.

---

## Released

### 1.0 — Foundation (2026-08-09)

The 1.0 release established the high-performance core for FITS tensor and table I/O:

- **Zero-Copy Tensor I/O:** Memory-mapped reads with SIMD-vectorized byte swapping for 1D–4D FITS image extensions.
- **Columnar Table Engine:** Binary and ASCII table reads with SQL predicate pushdown (`where=`) and fast column projection.
- **PyTorch ML Data Loaders:** Native `Dataset` classes (`FitsImageDataset`, `FitsCutoutDataset`, `FitsCubeDataset`, `FitsTableDataset`) and multi-worker `make_loader`.
- **Command-Line Suite:** Unix-style CLI tools (`info`, `header`, `cutout`, `convert`, `verify`).
- **Feature Parity:** Comprehensive format support verified against standard FITS test suites.

### 1.1 — Streaming, Correctness & Remote Hardening (beta soak)

On the same PyTorch ABI lane; the [changelog](changelog.md) carries the full list:

- **Checksum-stamped writes:** `write(..., checksum=True)` plus `verify_checksums`.
- **GIL-free hot reads:** DataLoader workers no longer serialize behind one Python thread during disk or network access.
- **Auto-adaptive RGB compositing:** `transforms.rgb(*bands)` and `convert --recipe auto`.
- **Memory-bounded streaming filters:** `scan(..., where=...)` evaluates predicates per batch, so peak RAM tracks `batch_size`.
- **Multiprocess-safe remote downloads:** OS-file-lock dedupe, resumable partials, explicit completeness warnings.
- **Silent-corruption fixes:** BIT column writes, multi-chunk buffered reads, unsigned-column `where=` pushdown.

---

## Current focus

- **Single-pass arena decode for buffered table reads.** Removes the one
  remaining significant benchmark deficit vs `fitsio` (narrow-table full
  reads with `mmap=False`, ~6–17%): decode straight into caller-visible,
  strided tensors instead of staging whole rows in scratch chunks. An
  API-visible change targeted at the next minor.
- **Selective-projection fast path** in the same reader, so filtered scans
  stop paying for whole-row pread when only a few columns are needed.
- **Table semantics polish:** complex-column dtypes in `schema()`,
  consistent error types across the mutation API.
- **Object-store recipes:** row-band caching and range-fetch patterns for
  S3-style archives on top of the hardened HTTP downloader.
- **CLI wave 3:** thin `fitsverify` helper and fpack-style tile controls
  (no CFITSIO HTTPS drivers — torchfits keeps its own HTTP stack).

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
