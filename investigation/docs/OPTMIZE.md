# torchfits: Strategic Work Packages

This document outlines the 15 most impactful engineering tasks required to elevate torchfits to a state-of-the-art FITS I/O library. The priorities are derived from a comprehensive analysis of the existing codebase, project objectives, performance benchmarks, and a review of how leading astronomical and high-performance computing libraries solve common data loading challenges.

---

## 1. Bindings: Switch C++ Bindings from pybind11 to nanobind

**Objective:** Achieve a significant, library-wide performance boost and reduce complexity with a simple dependency change.

**Justification:** nanobind is the modern successor to pybind11, designed by the same author to be faster and lighter. Benchmarks show up to 4x faster compile times, 5x smaller binaries, and 10x lower runtime overhead for Python-to-C++ function calls. For a performance-critical library like torchfits, this is a massive "easy win."

**Implementation:**

- Update `CMakeLists.txt` to find and link against the nanobind library instead of pybind11.
- Perform a project-wide search-and-replace in the C++ source files from the `py::` namespace to `nb::`.
- Update include directives from `<pybind11/...>` to `<nanobind/...>` and add explicit includes for STL types (e.g., `<nanobind/stl/string.h>`). The API is nearly identical, making this a low-effort, high-reward migration.

---

## 2. Core I/O: Implement a True Zero-Copy C++ Core Reader

**Objective:** Eliminate the primary performance bottleneck by rewriting the C++ data reading functions to achieve a true zero-copy data path.

**Justification:** This is the most critical optimization. The current benchmarks lag fitsio because of intermediate data copies. The C++ backend must be modified to first allocate a `torch::Tensor` and then pass its raw data pointer directly to cfitsio functions.

**Implementation:**

- Modify `fits.cpp` and `table.cpp`.
- For any read operation, first allocate a `torch::Tensor` of the correct size and data type on the target device (CPU or CUDA).
- Get the raw data pointer from this tensor using `.data_ptr()`.
- Pass this pointer directly to cfitsio functions (`fits_read_subset`, `fits_read_col`, etc.) to have cfitsio write the data directly into the tensor's memory.

---

## 3. Compression: Add Optimized Tiled Decompression for Cutouts

**Objective:** Achieve massive speedups when reading small sections (cutouts) from large, compressed FITS files.

**Justification:** Modern surveys (LSST, Euclid) rely heavily on tiled compression. A standard read decompresses the whole image, which is incredibly slow for small cutouts. This is a mandatory feature for modern survey science.

**Implementation:**

- In the C++ ImageHDU reader, when a section is requested, check if the FITS file is compressed using `fits_get_img_comp_type`.
- If it is, use `fits_read_compressed_img` instead of the standard read functions. This cfitsio routine is specifically designed to only decompress the necessary tiles to reconstruct the requested cutout.

---

## 4. Tables: Build a Robust and Performant Table Reader

**Objective:** Fix the currently non-functional table reader to support massive catalogs (200M+ rows) with high performance.

**Justification:** A high-performance FITS library must excel at reading tables.

**Implementation:**

- Rewrite the `table.cpp` module from the ground up.
- Implement a columnar reading strategy. The `read_table(columns=[...])` function should iterate through the requested columns.
- For each column, perform a single, contiguous read using `fits_read_col` directly into a pre-allocated `torch::Tensor` (as per Work Package #2).

---

## 5. Headers: Implement fitsio's Fast Header Parsing Strategy

**Objective:** Dramatically accelerate header reading by minimizing Python/C++ boundary crossings.

**Justification:** Reading a header keyword-by-keyword involves many slow round-trips between Python and C++. fitsio's speed comes from avoiding this.

**Implementation:**

- Create a single C++ function, `read_header_to_string()`.
- Inside this function, call the CFITSIO routine `fits_hdr2str` to dump the entire header block into a single C string in one operation.
- Return this single string to Python.
- Create a fast Python-side parser to split the string into a dictionary of keywords. This "do bulk work in C++, make one call" pattern is vastly more efficient.

---

## 6. Caching: Implement a C++ fitsfile Pointer Cache

**Objective:** Drastically reduce file I/O overhead in iterative machine learning workflows (e.g., inside a DataLoader).

**Justification:** Opening and closing a file is an expensive OS operation. Caching the opened `fitsfile*` pointers in memory avoids this overhead.

**Implementation:**

- Create a C++ singleton class, `FITSCache`, which holds an `std::map<std::string, fitsfile*>`.
- When `torchfits.open(path, cache=True)` is called, the C++ backend first checks if the path exists in the `FITSCache`.
- If it exists, return the cached `fitsfile*`. If not, open the file with `fits_open_file`, store the pointer in the map, and then return it.

---

## 7. Parallelism: Ensure Python GIL is Released During All I/O

**Objective:** Enable true multi-worker performance in `torch.utils.data.DataLoader`.

**Justification:** Without releasing Python's Global Interpreter Lock (GIL), all C++ operations will be serialized, making `num_workers > 0` in a DataLoader completely ineffective.

**Implementation:**

- In your binding code, wrap all C++ function calls that perform significant I/O or computation (e.g., `read_image`, `read_table`) with `nb::call_guard<nb::gil_scoped_release>()`.

---

## 8. Writing: Add a Full-Featured torchfits.writeto() Implementation

**Objective:** Provide essential write functionality to make torchfits a complete I/O solution.

**Justification:** A read-only library is incomplete. Users need to save model outputs, processed data, and generated catalogs.

**Implementation:**

- Create a new C++ function `write_to_file`.
- For array data (`torch.Tensor`), use `fits_create_img` and `fits_write_img`, passing the tensor's `.data_ptr()`.
- For table data (`dict[str, torch.Tensor]`), use `fits_create_tbl`, followed by a loop that calls `fits_write_col` for each column.

---

## 9. WCS: Optimize WCS Transformations for GPU Tensors

**Objective:** Enable high-performance WCS calculations directly on the GPU, avoiding costly CPU round-trips.

**Justification:** For many ML applications, WCS transformations are part of the data pipeline. Performing these on the GPU is far more efficient.

**Implementation:**

- Modify `wcs.cpp` to accept tensors that may reside on a CUDA device.
- If the input tensor is on the GPU, the C++ code must copy the coordinate data from the GPU to a CPU-side buffer, pass it to wcslib, and copy the results back to a new GPU tensor. This C++-managed round-trip is significantly faster than a Python-level `.to('cpu')` call.

---

## 10. Memory: Implement a Simple Tensor Buffer Pool

**Objective:** Eliminate memory allocation overhead in the data loading loop.

**Justification:** Repeatedly allocating and deallocating GPU/CPU memory for tensors is slow and causes fragmentation. A buffer pool solves this.

**Implementation:**

- Create a Python class `torchfits.BufferPool` that, upon initialization, pre-allocates a list of tensors of a specific shape and device.
- In the `Dataset.__getitem__`, instead of creating a new tensor, request one from the pool (`pool.get()`).
- The C++ backend writes data directly into this pre-allocated tensor's memory. This is a standard pattern in high-performance vision libraries.

---

## 11. Prefetching: Implement a Python-Level Prefetching IterableDataset

**Objective:** Hide I/O latency by overlapping data loading (CPU) with model computation (GPU).

**Justification:** Even with a fast loader, the GPU can sit idle waiting for the next batch. Prefetching solves this by loading batch N+1 while the GPU works on batch N.

**Implementation:**

- Create a `torchfits.PrefetchingIterableDataset` class that wraps a standard torchfits dataset.
- The `__iter__` method will spawn a background `threading.Thread`.
- This thread's job is to continuously load items from the underlying dataset and put them into a `queue.Queue`.
- The main thread simply yields items from this queue. This is a simple and powerful way to build an asynchronous data pipeline in Python.

---

## 12. Accuracy: Use wcslib for All WCS Logic, Period.

**Objective:** Guarantee accuracy and standards compliance for all WCS transformations.

**Justification:** `astropy.wcs` is the gold standard because it is a wrapper around the canonical wcslib C library. Reinventing this logic is unnecessary and guarantees errors.

**Implementation:**

- Mandate that the `wcs.cpp` backend is a pure, thin wrapper. It should not contain any custom projection math.
- Its only role is to efficiently pass coordinate data from PyTorch tensors to the standard wcslib C functions (`wcsbth`, `wcspih`, etc.) and return the results. This ensures torchfits has the same WCS accuracy as astropy.

---

## 13. mmap: Add mmap Support for Read Operations

**Objective:** Provide an alternative, high-performance reading mode for specific access patterns, particularly repeated random access on large files.

**Justification:** Memory mapping (mmap) offloads file buffering to the OS, which can be more efficient than application-level buffering.

**Implementation:**

- Add an `mmap=False` flag to `torchfits.open()`.
- If True, the C++ backend should use cfitsio's URL syntax to open the file with memory mapping: `mem://#filename`.

---

## 14. Usability: Create an Intelligent torchfits.Dataset Class

**Objective:** Simplify the user experience for common ML tasks by providing a high-level, opinionated Dataset class.

**Justification:** Many users have similar needs (e.g., loading an image from one extension and a label from another). A pre-built Dataset class reduces boilerplate.

**Implementation:**

- Create a `torchfits.SpectroscopicDataset` in `datasets.py` that takes extension names for flux, ivar, and metadata as input.
- Create a `torchfits.ImagingDataset` that can take extension names for science, mask, and weight images.

---

## 15. Benchmarking: Develop a Comprehensive Benchmarking Suite

**Objective:** Create a robust, reproducible benchmark suite to prove performance superiority and prevent regressions.

**Justification:** The primary goal is to be faster than fitsio. This claim must be backed by rigorous, public benchmarks.

**Implementation:**

- Expand the existing `benchmarks/` directory.
- Create benchmark scripts that compare torchfits, fitsio, and astropy on a standardized set of large, publicly available FITS files.
- Measure both speed and memory usage.
- Integrate these benchmarks into the CI/CD pipeline to automatically catch performance regressions.