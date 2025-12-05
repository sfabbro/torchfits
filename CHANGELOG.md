# Changelog

All notable changes to torchfits will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-04

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
