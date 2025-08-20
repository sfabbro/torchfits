# TorchFITS: Top 10 Strategic Work Packages

This document outlines the 10 most impactful engineering tasks required to elevate torchfits to a state-of-the-art FITS I/O library. The priorities are derived from a comprehensive analysis of the existing codebase, project objectives, performance benchmarks, and a review of how leading astronomical libraries leverage cfitsio.

---

## 1. Implement a True Zero-Copy C++ Core Reader

**Objective:** Eliminate the primary performance bottleneck by rewriting the C++ data reading functions to achieve a true zero-copy data path.

**Justification:** Benchmarks show torchfits is slower than fitsio. High-performance libraries like EleFits and CCfits avoid intermediate memory buffers. The current C++ backend likely performs a cfitsio → temporary buffer → torch::Tensor copy. This is the single most critical performance issue to fix.

**Implementation:**

- Modify `fits.cpp` and `table.cpp`.
- For any read operation, first allocate a `torch::Tensor` of the correct size and data type on the target device (CPU or CUDA).
- Get the raw data pointer from this tensor using `.data_ptr()`.
- Pass this pointer directly to cfitsio functions (`fits_read_subset`, `fits_read_col`, etc.) to have cfitsio write the data directly into the tensor's memory.
- This work package is the foundation for all other performance goals.

---

## 2. Add Optimized Tiled Decompression for Cutouts

**Objective:** Achieve massive speedups when reading small sections (cutouts) from large, compressed FITS files.

**Justification:** Modern surveys (LSST, Euclid) produce petabytes of data, almost always stored in fpack-compressed FITS files with tiled compression. Reading a small cutout without this optimization forces cfitsio to decompress the entire image, which is catastrophically slow. This is a mandatory feature for modern survey science.

**Implementation:**

- In the C++ ImageHDU reader, when a section is requested, check if the FITS file is compressed using `fits_get_img_comp_type`.
- If it is, use `fits_read_compressed_img` instead of the standard read functions. This cfitsio routine is specifically designed to only decompress the necessary tiles to reconstruct the requested cutout.

---

## 3. Build a Robust and Performant Table Reader

**Objective:** Fix the currently non-functional table reader to support massive catalogs (200M+ rows) with high performance.

**Justification:** Astronomical machine learning is increasingly reliant on large catalogs. A high-performance FITS library must excel at reading tables.

**Implementation:**

- Rewrite the `table.cpp` module from the ground up.
- Implement a columnar reading strategy. The `read_table(columns=[...])` function should iterate through the requested columns.
- For each column, perform a single, contiguous read using `fits_read_col` directly into a pre-allocated `torch::Tensor` (as per Work Package #1).
- Add robust type handling to correctly map FITS data types (e.g., TSTRING, TDOUBLE, TLOGICAL) to torch dtypes.

---

## 4. Implement a C++ fitsfile Pointer Cache

**Objective:** Drastically reduce file I/O overhead in iterative machine learning workflows (e.g., inside a DataLoader).

**Justification:** Opening and closing a file is an expensive OS operation. In a typical training loop, the same files are accessed repeatedly. Caching the opened `fitsfile*` pointers in memory avoids this overhead. The LSST Science Pipelines use a similar strategy for performance.

**Implementation:**

- Create a C++ singleton class, `FITSCache`, which holds an `std::map<std::string, fitsfile*>`.
- When `torchfits.open(path, cache=True)` is called, the C++ backend first checks if the path exists in the `FITSCache`.
- If it exists, return the cached `fitsfile*`. If not, open the file with `fits_open_file`, store the pointer in the map, and then return it.
- Provide a Python function `torchfits.clear_cache()` to close all cached files.

---

## 5. Ensure Python GIL Release for True Parallelism

**Objective:** Enable true multi-worker performance in `torch.utils.data.DataLoader`.

**Justification:** Without releasing Python's Global Interpreter Lock (GIL), all C++ operations will be serialized, making `num_workers > 1` in a DataLoader completely ineffective. This is a simple but critical step for high-throughput data loading.

**Implementation:**

- In `bindings.cpp`, wrap all C++ function calls that perform significant I/O or computation (e.g., `read_image`, `read_table`) with `py::call_guard<py::gil_scoped_release>()`.

---

## 6. Add a Full-Featured torchfits.writeto() Implementation

**Objective:** Provide essential write functionality to make torchfits a complete I/O solution.

**Justification:** A read-only library is incomplete. Users need to save model outputs, processed data, and generated catalogs. This is a major missing feature required for end-to-end workflows.

**Implementation:**

- Create a new C++ function `write_to_file`.
- For array data (`torch.Tensor`), use `fits_create_img` and `fits_write_img`, passing the tensor's `.data_ptr()`.
- For table data (`dict[str, torch.Tensor]`), use `fits_create_tbl`, followed by a loop that calls `fits_write_col` for each column.
- Expose this functionality via a simple `torchfits.writeto(path, data, header)` function in Python.

---

## 7. Optimize WCS Transformations for GPU Tensors

**Objective:** Enable high-performance WCS calculations directly on the GPU, avoiding costly CPU round-trips.

**Justification:** For many ML applications (e.g., de-projection, data augmentation), WCS transformations are part of the data pipeline. Performing these on the GPU alongside other augmentations is far more efficient.

**Implementation:**

- Modify `wcs.cpp` to accept tensors that may reside on a CUDA device.
- If the input tensor is on the GPU, the C++ code must copy the coordinate data from the GPU to a CPU-side buffer.
- Pass the CPU buffer to the wcslib functions (which are CPU-only).
- Copy the results from wcslib back into a new tensor allocated on the original GPU device. While not a "native" GPU implementation, this C++-managed round-trip is significantly faster than a Python-level `.to('cpu')` call.

---

## 8. Implement mmap Support for Read Operations

**Objective:** Provide an alternative, high-performance reading mode for specific access patterns, particularly repeated random access on large files.

**Justification:** Memory mapping (mmap) offloads file buffering and caching to the operating system, which can be more efficient than application-level buffering for certain workloads. `fitsio` offers this, and for feature parity and performance flexibility, torchfits should as well.

**Implementation:**

- Add an `mmap=False` flag to `torchfits.open()`.
- If True, the C++ backend should use cfitsio's URL syntax to open the file with memory mapping: `mem://#filename`. cfitsio handles the underlying mmap call.

---

## 9. Create an Intelligent torchfits.Dataset Class

**Objective:** Simplify the user experience for common ML tasks by providing a high-level, opinionated Dataset class.

**Justification:** While the core API should be flexible, many users have similar needs (e.g., loading an image from one extension and a label from another). A pre-built Dataset class reduces boilerplate and makes the library more accessible.

**Implementation:**

- Create a `torchfits.SpectroscopicDataset` in `datasets.py` that takes a list of files and extension names for flux, ivar, and metadata as input.
- Create a `torchfits.ImagingDataset` that can take extension names for science, mask, and weight images.
- These classes will use the core `torchfits.open` API internally but present a much simpler interface to the end-user.

---

## 10. Develop a Comprehensive Benchmarking Suite

**Objective:** Create a robust, reproducible benchmark suite to prove performance superiority and prevent regressions.

**Justification:** The primary goal is to be faster than fitsio. This claim must be backed by rigorous, public benchmarks covering all major use cases (large images, cutouts, compressed files, large tables, spectra). This is essential for user trust and for guiding future optimization.

**Implementation:**

- Expand the existing `benchmarks/` directory.
- Create benchmark scripts that compare torchfits, fitsio, and astropy on a standardized set of large, publicly available FITS files.
- Measure both speed and memory usage.
- Integrate these benchmarks into the CI/CD pipeline to automatically catch performance regressions before they are merged.