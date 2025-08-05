# TorchFits v1.0 Implementation Plan

## 🎯 Core Mission

To provide the fast, intuitive, and PyTorch-native library for accessing FITS data, empowering machine learning applications in astronomy.

---

## 🌟 Guiding Principles

1. **Performance First**: Outperform existing libraries like `astropy.io.fits` and `fitsio` in all common use cases, from small images to massive data cubes and tables.
2. **PyTorch Native**: Data should be read directly into `torch.Tensor` objects, ready for GPU acceleration and ML pipelines, without intermediate copies.
3. **Familiar API**: Provide a user-friendly and familiar API for astronomers accustomed to `astropy.io.fits` and `fitsio`.
4. **Seamless `torch-frame` Integration**: Treat FITS tables as first-class citizens in the PyTorch ecosystem, with deep and meaningful integration with `torch-frame`.
5. **Robust Remote Access**: Make working with remote datasets (HTTP, S3, etc.) as simple and efficient as local files, with an intelligent caching system optimized for ML training.
6. **Full Functionality**: Strive for feature parity with the core functionalities of `astropy.io.fits` and `fitsio`.

---

## 现状 (Current Status - v0.3)

`torchfits` has successfully established a strong foundation, delivering on key initial promises.

* ✅ **High-Performance Core**: C++ backend leveraging CFITSIO for direct-to-tensor reading, already outperforming competitors in many benchmarks.
* ✅ **Enhanced Table Operations**: The `FitsTable` class provides a powerful, pure-PyTorch, pandas-like interface for table manipulation.
* ✅ **Basic Remote & Cache Support**: Foundational infrastructure for reading from URLs and caching files is in place.
* ✅ **WCS Utilities**: Core WCS transformations are supported.
* ✅ **Initial `torch-frame` Integration**: FITS tables can be converted to `torch_frame.DataFrame` objects.
* ✅ **Familiar API**: `read()` function and `FITS`/`HDU` objects offer an interface that is intuitive for users of existing FITS libraries.

---

## 🚀 Roadmap to v1.0

The path to v1.0 is focused on three pillars: **Performance**, **Complete Feature Set**, and **Intelligent Data Handling**.

### Pillar 1: Performance 

**Goal**: Make `torchfits` the undisputed performance leader for FITS I/O in Python.

**Key Initiatives**:

1. **Advanced CFITSIO Integration**:
    * **Memory Mapping**: Implement `ffgmem` for memory-mapped access to large local files, minimizing memory overhead and I/O.
    * **Buffered I/O**: Optimize buffer sizes and use CFITSIO's tile-based access for significant speedups on compressed files and cutouts.
    * **Iterator Functions**: Use `fits_iterate_data` for highly efficient table column processing, especially for row-wise filtering and transformations.

2. **Parallelization**:
    * **Multi-threaded Column Reading**: Read multiple table columns in parallel using thread-safe CFITSIO calls.
    * **Parallel HDU Processing**: Develop strategies for parallel reading of multiple HDUs from a single MEF file.

3. **GPU-Direct Pipeline**:
    * **Pinned Memory**: Read data into pinned (page-locked) memory to enable faster, asynchronous CPU-to-GPU transfers.
    * **CUDA Streams**: Overlap data reading and GPU transfers using CUDA streams to hide latency in ML data loading pipelines.

### Pillar 2: Complete Feature Set & API Parity

**Goal**: Achieve functional parity with the most critical features of `astropy.io.fits` and `fitsio`, ensuring users can fully migrate to `torchfits`.

**Key Initiatives**:

1. **Writing and Updating FITS Files**:
    * Implement `torchfits.write()` to save tensors and `FitsTable` objects to new FITS files.
    * Support for updating existing FITS files (in-place modification of data, appending HDUs).
    * Header manipulation: Add, remove, and update header keywords.

2. **Expanded FITS Standard Support**:
    * **Compressed Images**: Natively handle FITS images compressed with Rice, GZIP, and HCOMPRESS.
    * **Variable-Length Arrays**: Support for reading table columns with variable-length arrays, a common feature in astronomical catalogs.
    * **Random Groups**: Support for this legacy but still-present FITS format.

3. **Enhanced `FitsTable` Functionality**:
    * **String Column Support**: More robust and efficient handling of string columns.
    * **Advanced Joining**: Implement more complex join operations (`outer`, `right`) between `FitsTable` objects.
    * **Missing Data**: More explicit handling of null values (`TNULL`) during read and in `FitsTable` operations.

### Pillar 3: Intelligent Data Handling (Remote & ML)

**Goal**: Create a best-in-class experience for large-scale, remote datasets, specifically tailored for ML training workflows.

**Key Initiatives**:

1. **ML-Optimized Smart Cache**:
    * **Training-Aware Prefetching**: Implement a `TorchFitsDataset` that intelligently prefetches the next files needed for training based on the dataloader's access pattern.
    * **Epoch-Aware Cache Management**: The cache should understand the concept of training epochs, keeping data for recent epochs and evicting older data.
    * **Cache Resiliency**: Implement checksum verification, automatic cleanup of corrupted files, and robust error handling for network failures.

2. **Deep `torch-frame` Integration**:
    * **Automatic `stype` Inference**: Automatically map FITS column metadata (units, keywords) to `torch_frame` semantic types (`stype.numerical`, `stype.categorical`, etc.). Develop new astronomy-specific `stypes` if needed (e.g., `stype.celestial_coord`).
    * **Bi-directional Conversion**: Enable seamless conversion from a `torch_frame.DataFrame` back to a `FitsTable` or FITS file.
    * **WCS in DataFrames**: Explore methods to associate WCS information with tables in a `torch-frame` context.

3. **Production-Ready Usability**:
    * **Comprehensive Documentation**: Create a documentation portal with a user guide, API reference, and a gallery of examples for common astronomy ML tasks.
    * **Actionable Error Messages**: Provide clear, helpful error messages that guide the user to a solution.
    * **Cross-Platform CI**: Rigorous testing on Linux, macOS (Intel & Apple Silicon), and Windows to ensure reliability.

---

## ✅ v1.0 Success Checklist

### Performance
- [ ] **Tables**: Outperforms `fitsio` and `astropy` by >1.1x on any type of table, image, datacube
- [ ] **Remote (Cached)**: Achieves <10% overhead compared to local file access.
- [ ] **GPU Workflows**: `TorchFitsDataset` shows >2x faster end-to-end training throughput than a naive `astropy`-based dataset.

### Features
- [ ] **Writing**: Can write any readable `torchfits` object to a FITS file.
- [ ] **Feature Parity**: All code examples from the `astropy.io.fits` and `fitsio` documentation can be translated to `torchfits`.
- [ ] **`torch-frame`**: `read(..., format='dataframe')` returns a `DataFrame` with correct `stypes` inferred from FITS headers.

### Robustness
- [ ] **Remote Training**: A multi-epoch training job on a remote dataset completes successfully even with simulated network interruptions.
- [ ] **Documentation**: A new user can successfully build a galaxy classification model using a remote FITS dataset by following the tutorials.
- [ ] **Cross-Platform**: All tests pass on all targeted OS and Python/PyTorch versions.

This plan outlines a clear path to establishing `torchfits` as the essential, high-performance tool for astronomical data analysis in the modern machine learning era.
