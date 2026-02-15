# Changelog

All notable changes to torchfits will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-02-14

### Performance Improvements
- **Int32 Optimization**: Routed `int32` image reads through the efficient CFITSIO path, achieving a **1.45x** speedup (vs `fitsio`) and fixing a previous regression.
- **Intelligent Chunking**: Implemented 128MB chunking for large file reads. This prevents memory spikes during type conversion, maintaining high throughput for multi-GB files without OOM risks.
- **Table Reads**: Verified massive speedups (**~1500x**) for table operations using memory mapping.
- **No Regressions**: Confirmed 100% win rate against `fitsio` across 88 benchmark cases in the exhaustive suite.

### Documentation
- Updated `docs/benchmarks.md` with the latest exhaustive benchmark results (100% win rate).
- Updated `docs/performance_attempts.md` with successful optimization logs (Int32 routing, Chunking).

### Maintenance
- Cleaned up internal benchmark scripts and temporary logs.
- Updated CI workflows and pre-commit hooks.

## [0.2.0] - 2026-02-14

### Added
- Arrow-native table API in `torchfits.table`: `read()`, `scan()`, `reader()`, `dataset()`, `scanner()`.
- In-place FITS table mutation APIs: `append_rows()`, `insert_rows()`, `delete_rows()`, `update_rows()`, `insert_column()`, `replace_column()`, `rename_columns()`, `drop_columns()`.
- `TableHDURef` file-backed mutation helpers (`*_file`) and expanded lazy-handle workflow.
- HTML representation for `HDUList` in notebook environments.
- Additional compressed image parity coverage (including hcompress-focused tests).
- Expanded benchmark tooling and workflows (`bench-fast`, `bench-fast-stable`, focused/core runners, Arrow-table benchmark scripts) to track regressions across image, cache, transform, and table paths.
- `hdu="auto"`/`hdu=None` payload detection in `read()` and `get_header()`, plus `get_wcs()` convenience API for file-based WCS initialization.
- `benchmark_ml_loader.py` for end-to-end DataLoader throughput comparisons.

### Changed
- Minimum supported Python version is now `>=3.11`.
- Build and packaging flow updated around vendored native dependencies (cfitsio/wcslib), with wheel/CI configuration refresh.
- WCS path improvements, including TPV/SIP handling refinements and correction flow updates.
- Performance work across image read/cache paths, mmap/read-path selection, and SIMD tuning in core C++ paths.
- AVX2/SSSE3-focused optimization work landed for key numeric conversion and table/image hot paths.
- Documentation reorganized: README condensed to overview and API coverage moved/expanded in `docs/api.md`.

### Fixed
- FITS filename handling hardening and header/column-name sanitization on write paths.
- WCS/TAN GPU approximation correctness fixes.
- CI/build-system reliability fixes (build isolation, formatter compliance, dependency setup).
- Multiple interoperability and non-table regression fixes with test coverage updates.

### Performance Notes
- Benchmarks were re-baselined and workflow noise was reduced (repeat control, focused suites, stable fast runs) to make before/after performance comparisons more reproducible.
- Full 0.2.0 release benchmark (with tables) shows torchfits ahead in 87/88 `read_full` rows vs `fitsio` and 87/88 vs `fitsio_torch`; only `medium_float64_2d` remains a slight loss in this snapshot.
- ML-loader benchmark configuration was aligned for fairer comparisons (persistent workers + payload HDU detection); current CPU results are near parity with run-to-run variance (uncompressed median 0.985x vs `fitsio`, compressed median 1.008x), so performance claims should use repeated runs rather than single-shot output.

## [0.1.1] - 2025-12-05

## [0.1.0] - 2025-12-05

### Added

#### Core I/O
- High-performance FITS file reading with zero-copy tensor creation
- Direct tensor creation on CPU, CUDA, and MPS devices
- FITS table reading as dictionaries of tensors
- Column selection and row range reading for tables
- Cutout/subset reading without loading full images
- FITS file writing for images and tables
- Multi-HDU file support with context management

#### WCS Support
- Batch coordinate transformations using wcslib
- `pixel_to_world()` and `world_to_pixel()` transformations
- OpenMP parallelization for large coordinate arrays

#### Datasets & DataLoaders
- `FITSDataset` for map-style random access
- `IterableFITSDataset` for streaming large-scale data
- Optimized DataLoader factory functions
- PyTorch DataLoader compatibility

#### Transforms
- GPU-accelerated astronomical transforms:
  - `ZScale` - Automatic normalization
  - `AsinhStretch` - Asinh stretch for HDR images
  - `LogStretch` - Logarithmic stretch
  - `PowerStretch` - Power law stretch
  - `Normalize` - Standard normalization
  - `MinMaxScale` - Min-max scaling
  - `RobustScale` - Robust scaling with IQR
- Data augmentation transforms:
  - `RandomCrop` - Random cropping
  - `CenterCrop` - Center cropping
  - `RandomFlip` - Random flipping
  - `RandomRotation` - Random rotation
  - `GaussianNoise` - Gaussian noise addition
  - `PoissonNoise` - Poisson noise
- Spectroscopy transforms:
  - `RedshiftShift` - Simulate redshift effects
  - `PerturbByError` - Error-based perturbation
- Utility transforms:
  - `ToDevice` - Device placement
  - `Compose` - Transform composition
- Pre-configured transform pipelines for training, validation, and inference

#### Performance Features
- C++ engine with nanobind for Python bindings
- Zero-copy tensor operations
- SIMD-optimized data type conversions
- Multi-level caching (L1 memory + L2 disk)
- Memory pools for efficient allocation
- Tile-aware reading for compressed images
- Aggressive buffering with adaptive buffer sizes

#### Integration
- pytorch-frame TensorFrame support for tabular data
- PyArrow and Pandas conversion utilities
- Remote file support (HTTP/HTTPS/FTP via cfitsio)

### Performance
- 10-100x faster than astropy for large arrays
- Zero-copy tensor creation from FITS data
- Parallel I/O with OpenMP acceleration
- Optimized memory usage with shared buffers

### Documentation
- Comprehensive README with quick start guide
- Complete API reference with examples
- Working examples for common use cases
- Benchmark suite demonstrating performance

### Known Limitations
- VLA (Variable Length Array) columns in tables have limited support
- Some advanced WCS projections may not be fully supported
- Compressed image writing uses Rice compression only
- MPS (Apple Silicon) shows overhead for small workloads

### Dependencies
Historical note for `0.1.0` at release time:
- Python ≥ 3.11
- PyTorch ≥ 2.0
- pytorch-frame ≥ 0.2.0
- NumPy ≥ 1.20
- cfitsio (bundled as submodule)
- wcslib (system dependency)

---

## [Unreleased]

### Planned
- Additional compression algorithms (GZIP, HCOMPRESS)
- Improved VLA column support
- Async I/O for remote files
- More WCS projection types
- Performance optimizations for MPS device

[0.1.0]: https://github.com/sfabbro/torchfits/releases/tag/v0.1.0
[0.1.1]: https://github.com/sfabbro/torchfits/releases/tag/v0.1.1
[0.2.0]: https://github.com/sfabbro/torchfits/releases/tag/v0.2.0
[0.2.1]: https://github.com/sfabbro/torchfits/releases/tag/v0.2.1
