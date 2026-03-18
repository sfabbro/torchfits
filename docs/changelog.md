# Changelog

All notable changes to torchfits will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-03-18

### Changed
- Version synchronization across `pyproject.toml`, `pixi.toml`, and `__init__.py` (was inconsistent in 0.3.1).
- Benchmark snapshot updated to run `20260318_083620` with improved results across all domains.
- CI workflow cleanup: `ci.yml` now references `pyarrow` instead of `pytorch-frame`.
- Default pixi environment now includes comparison libraries (`healpy`, `hpgeom`, `healsparse`, `astropy-healpix`, `spherical-geometry`) so parity tests run by default; skipped tests reduced from ~142 to ~16.

### Fixed
- WCS projection count corrected to 13 in README and API docs (was incorrectly listing CYP as supported).
- Changelog v0.3.0 projection list corrected (was listing CYP as implemented; now noted as stubbed).
- `test_healsparse_upstream.py` updated to use actual `SparseHealpixMap` API (`from_pixels`, `get_values`, `ud_grade`, `to_dense`); tests for unimplemented APIs (`make_empty`, `sum_union`/`sum_intersection`) marked skip; known API mismatches marked xfail.
- `test_real_data_validation.py` BSCALE/BZERO float32 tolerance relaxed to 1e-4 (appropriate for float32 arithmetic).

## [0.3.1] - 2026-03-18

### Added
- Comprehensive `test_synalm_parity.py` to validate parity with [healpy](https://healpy.readthedocs.io/) generation paths (`synalm`, `synfast`, etc.).
- Real-data validation lane (`tests/test_real_data_validation.py`, `benchmarks/replays/replay_real_data_validation.py`) using installed FITS fixtures and public reference data from [Astropy sample datasets](https://docs.astropy.org/en/stable/utils/data.html).
- Security hardening: replaced `numexpr.evaluate` in table filtering with a secure expression parser. `&`/`|`/`~` syntax shimmed to SQL-style `AND`/`OR`/`NOT`.
- Documentation rewrite across all docs: `README.md`, `api.md`, `benchmarks.md`, `changelog.md`, `contributing.md`, `examples.md`, `install.md`, `release.md`, `sphere.md`.

### Fixed
- Resolved minor precision bugs in `torchfits.sphere.spectral` harmonic primitives.
- Mixed-HDU write paths: schema-aware normalization for tables with scaled columns, string columns, and `torch.Tensor` inputs.
- Cache invalidation now clears both Python and C++ file/handle/metadata caches.
- `_extract_table_schema_from_header` parses `TNULL`/`TSCAL`/`TZERO` as numeric values.
- Table column ordering preserved during mixed-HDU round-trips.
- Header `BSCALE`/`BZERO` handling in real-data validation (copy before data access to avoid Astropy in-place removal).

### Performance
- Integrated performance fixes from open PRs: `alm_size` O(1) formula, SIP power-cache extraction, TNX regex caching, header parser string-op optimization, loop-level import hoisting in `get_batch_info` and `IterableFITSDataset._process_shard`.
- Hardware detection on macOS uses `sysctlbyname` instead of `popen`.
- Re-baselined the exhaustive benchmark suite across all four domains (run `20260318_083620`).

## [0.3.0] - 2026-03-06

### Added
- **WCS (`torchfits.wcs`)**: Pure-PyTorch implementation of astronomical coordinate transformations.
    - Projections: TAN, SIN, ARC, ZPN, ZEA, STG, CEA, CAR, MER, AIT, MOL, HPX, SFL. CYP is stubbed but not yet implemented.
    - Distortions: SIP, TPV, TNX, ZPX.
    - Batch `pixel_to_world` and `world_to_pixel` on CPU, CUDA, and MPS.
    - `torchfits.get_wcs()` for initialization from FITS headers.
- **Sphere (`torchfits.sphere`)**: PyTorch-native HEALPix and spherical geometry.
    - Core primitives: `ang2pix`, `pix2ang`, `nest2ring`, `ring2nest`, `neighbors`, interpolation.
    - Spherical polygons: non-convex region queries, area, containment.
    - Spherical harmonic transforms: `map2alm`, `alm2map` (scalar and spin).
    - Multi-Order Coverage (MOC) maps.
    - HealSparse-compatible sparse map API.
    - `torchfits.sphere.compat` healpy-compatible API surface.
- **Spectral (`torchfits.spectral`)**: Experimental 1D spectrum and IFU data cube containers.
- **ML loaders**: Improved `FITSDataset` and `create_fits_dataloader` with `hdu="auto"` payload detection and multi-worker robustness.

### Changed
- `torchfits.get_wcs` now uses the native PyTorch WCS implementation exclusively — no runtime dependency on wcslib.
- Removed all WCSLIB detection and linking logic from the build system. The C++ engine now only vendors [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/).
- Benchmark output directory standardized to `benchmarks_results/`.

### Removed
- WCSLIB C++ dependency. Wheels are now fully self-contained.
- Legacy monolithic `wcs.py` and redundant `torchfits.fits` namespace.

## [0.2.1] - 2026-02-14

### Fixed
- Routed `int32` image reads through the CFITSIO fast path, fixing a regression.
- Implemented 128 MB chunking for large file reads to prevent memory spikes during type conversion.

### Performance
- Table reads via memory mapping show substantial speedups for large catalogs.

## [0.2.0] - 2026-02-14

### Added
- Arrow-native table API in `torchfits.table`: `read()`, `scan()`, `reader()`, `dataset()`, `scanner()`.
- In-place FITS table mutation APIs: `append_rows()`, `insert_rows()`, `delete_rows()`, `update_rows()`, `insert_column()`, `replace_column()`, `rename_columns()`, `drop_columns()`.
- `TableHDURef` file-backed mutation helpers and lazy-handle workflow.
- HTML representation for `HDUList` in notebook environments.
- `hdu="auto"` / `hdu=None` payload detection in `read()` and `get_header()`.
- `get_wcs()` convenience API for file-based WCS initialization.
- ML loader benchmark script (`bench_ml_loader.py`).

### Changed
- Minimum supported Python version raised to 3.11 (later relaxed to 3.10 in 0.3.0).
- Build system updated for vendored CFITSIO. AVX2/SSSE3 optimizations in C++ hot paths.

### Fixed
- FITS filename handling and header/column-name sanitization on write paths.
- WCS/TAN GPU approximation correctness.
- CI/build-system reliability (build isolation, formatter compliance).

## [0.1.1] - 2025-12-05

Patch release. No user-visible changes.

## [0.1.0] - 2025-12-05

### Added
- FITS image and table reading with zero-copy tensor creation on CPU, CUDA, and MPS.
- Column selection, row slicing, and cutout/subset reading.
- FITS file writing for images and tables.
- Multi-HDU support with context management.
- Batch WCS coordinate transformations via wcslib (later replaced by native PyTorch in 0.3.0).
- `FITSDataset` and `IterableFITSDataset` for [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html) integration.
- GPU-accelerated astronomical transforms: ZScale, AsinhStretch, LogStretch, PowerStretch, Normalize, MinMaxScale, RobustScale.
- Data augmentation transforms: RandomCrop, CenterCrop, RandomFlip, RandomRotation, GaussianNoise, PoissonNoise.
- Spectroscopy transforms: RedshiftShift, PerturbByError.
- C++ engine with [nanobind](https://nanobind.readthedocs.io/) bindings, SIMD-optimized type conversion, multi-level caching.
- [PyArrow](https://arrow.apache.org/docs/python/) and [Pandas](https://pandas.pydata.org/) conversion utilities.
- Remote file support (HTTP/HTTPS/FTP via CFITSIO).

### Known Limitations
- Variable-length array (VLA) table columns have limited support.
- Compressed image writing supports Rice compression only.
- MPS (Apple Silicon) shows overhead for small workloads due to transfer latency.

[0.1.0]: https://github.com/sfabbro/torchfits/releases/tag/v0.1.0
[0.1.1]: https://github.com/sfabbro/torchfits/releases/tag/v0.1.1
[0.2.0]: https://github.com/sfabbro/torchfits/releases/tag/v0.2.0
[0.2.1]: https://github.com/sfabbro/torchfits/releases/tag/v0.2.1
[0.3.0]: https://github.com/sfabbro/torchfits/releases/tag/v0.3.0
[0.3.1]: https://github.com/sfabbro/torchfits/releases/tag/v0.3.1
[0.3.2]: https://github.com/sfabbro/torchfits/releases/tag/v0.3.2
