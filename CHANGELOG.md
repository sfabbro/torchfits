# Changelog

## [0.2.1] - 2026-02-15

### Added
- Intelligent chunking for large uncompressed images to optimize memory and speed.
- Support for optional `psutil` dependency with graceful fallbacks.
- Python 3.14 classifier to `pyproject.toml`.

### Changed
- **Performance**: Optimized `int32` image reads by routing through CFITSIO, achieving ~1.45x speedup.
- **CI/CD**: Hardened wheel building process for Linux and MacOS.
  - Linux: Forced CPU-only PyTorch index to prevent CUDA dependency issues during build.
  - MacOS: Improved `delocate` process by robustly excluding `libtorch` libraries.
- **Dependencies**: Dropped official support for Python 3.11 (now requiring >=3.12).
- **Tests**: Marked flaky cache invalidation tests to be skipped in CI.

### Fixed
- Fixed MacOS wheel build repairs where `delocate` would fail on missing external dependencies.
- Resolved build isolation failures on systems without CUDA drivers.

## [0.2.0] - 2026-02-14
- Initial release with PyTorch frame support and native C++ backend.
