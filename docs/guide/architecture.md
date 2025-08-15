# TorchFits architecture and data flow

This document gives a high-level view of the runtime architecture, active modules, data flows, and the places where performance is won. It also lists redundant/legacy files that can be removed or migrated.

## Layers and responsibilities

- Python API (public surface)
  - `src/torchfits/fits_reader.py`: High-level read API, object-oriented `FITS`/`HDU`, glue around C++ extension, small conveniences (stacking, null-mask helpers).
  - `src/torchfits/fits_writer.py`: High-level write/update APIs for images, tables, MEFs.
  - `src/torchfits/dataset.py`: Batch and dataset helpers, batched sky cutouts via wcslib + C++ cutout path.
  - `src/torchfits/wcs_utils.py`: wcslib-backed WCS transforms and utilities.
  - `src/torchfits/table.py`: Table helpers and types (`FitsTable`, `ColumnInfo`, etc.).
  - `src/torchfits/remote.py` + `src/torchfits/smart_cache.py`: Remote fetching and Python-level cache manager for remote/local files.

- C++ core (performance-critical)
  - `src/torchfits/bindings.cpp`: pybind11 module, exports the core C++ functions to Python.
  - `src/torchfits/fits_reader.cpp` (+ `fits_reader.h`): The runtime entry point for reads. Implements image/table read, batched cutouts (single- and multi-HDU), small-window full-read-then-slice, tile-aware compressed cutouts, VLA handling, and cache integration.
  - `src/torchfits/fits_writer.cpp` (+ `fits_writer.h`): Unified writer for images/tables/MEFs and in-place updates.
  - `src/torchfits/fits_utils.cpp` (+ `fits_utils.h`): Thin CFITSIO helpers, error handling, header decode, etc.
  - `src/torchfits/wcs_utils.cpp` (+ `wcs_utils.h`): wcslib-backed WCS conversions.
  - `src/torchfits/cfitsio_enhanced.cpp` (+ `cfitsio_enhanced.h`): Opt-in CFITSIO enhancements (mmap full-image fast path; buffered reads scaffolding).
  - Cache: `src/torchfits/real_cache.cpp` (+ `real_cache.h`): Global LRU-like tensor cache used in hot read paths (full-image reuse; tile cache).
  - `src/torchfits/remote.cpp` (+ `remote.h`): Localize remote URLs (fsspec-style) for the C++ layer.

## Data flows

- Image read (full or cutout)
  1. Python `torchfits.read()` -> C++ `fits_reader_cpp.read` -> `read_impl`.
  2. `read_impl` resolves file/HDU, then:
     - If subset and small-window heuristic triggers: read full image once, cache it, slice view; reuses cached full across calls.
     - Else: `read_image_data(...)` via CFITSIO; for compressed images and batches, use tile-aware stitching with a global tile cache.
     - Optional mmap fast path applies to full, uncompressed, unscaled images when enabled.
  3. Returns tensor (+ header) directly to Python.

- Batched cutouts
  - Same-HDU: `read_many_cutouts` coalesces nearby windows or uses tile-aware reads for compressed inputs; results are returned as a list of tensors.
  - Multi-HDU (MEF): `read_many_cutouts_multi_hdu` groups by HDU and applies the same strategy per HDU, maintaining original order.

- Tables
  - Scalar columns read in bulk; optional parallel scalar-col read path for wide tables.
  - VLA columns: per-row reads (optionally parallel across rows using multiple handles); returned as Python lists of 1D tensors.
  - Null masks: one-pass integer TNULL masks via `read_table_with_null_masks`.

- WCS
  - wcslib-backed transforms; batched sky cutouts converge to the C++ cutout path after world->pixel conversion.

## Performance levers (where speed comes from)

- Minimizing CFITSIO calls: coalescing neighboring cutouts; batched read APIs; stitching tiles for compressed images.
- Reuse via caches: full-image reuse for small-window slices; global tile cache to avoid re-decompression.
- Zero/one-copy paths: opt-in zero-copy mmap for full, uncompressed, unscaled images; device-aware tensor creation.
- Parallelism: scalar-column parallel reads; VLA row-parallel with capped threads.
- WCS batching: convert world->pixel once and batch the cutout operations through the C++ path.

## Active vs legacy files (to consolidate)

Active (keep):

- Core: `bindings.cpp`, `fits_reader.cpp/h`, `fits_writer.cpp/h`, `fits_utils.cpp/h`, `wcs_utils.cpp/h`, `cfitsio_enhanced.cpp/h`, `real_cache.cpp/h`, `remote.cpp/h`.
- Python: `fits_reader.py`, `fits_writer.py`, `dataset.py`, `wcs_utils.py`, `table.py`, `remote.py`, `smart_cache.py`.

Legacy/redundant (removed or superseded):

- `performance_v2.cpp/.h`, `performance.cpp/.h` (scaffolding; removed).
- `smart_cache.cpp/.h` (C++ cache scaffolding; removed in favor of Python `smart_cache.py`).
- `memory_optimizer.cpp/.h` (empty; removed).
- `cache.cpp/.h` (legacy cache shim; removed after consolidating on `real_cache`).

Recommendation: Remove the above legacy C++ files from the tree (or move to an `attic/`), keep Python `smart_cache.py`.

## Proposed target layout (clean)

- Reads: `fits_reader.cpp/h` only.
- Writes: `fits_writer.cpp/h` only.
- Utils: `fits_utils`, `wcs_utils`.
- Perf opts: `cfitsio_enhanced` behind flags.
- Cache: `real_cache` only.
- Python API: `fits_reader.py`, `fits_writer.py`, `dataset.py`, `table.py`, `wcs_utils.py`, `smart_cache.py`, `remote.py`.

This keeps the hot path small and clear, while retaining optional hooks (mmap/tile cache) behind env flags.
