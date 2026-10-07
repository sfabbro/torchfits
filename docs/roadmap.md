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

### 1.1 — Streaming, Correctness & Remote Hardening (beta)

On the same PyTorch ABI lane; the [changelog](changelog.md) carries the full list:

- **Checksum-stamped writes:** `write(..., checksum=True)` plus `verify_checksums`.
- **GIL-free hot reads:** DataLoader workers no longer serialize behind one Python thread during disk or network access.
- **Auto-adaptive RGB compositing:** `transforms.rgb(*bands)` and `convert --recipe auto`.
- **Memory-bounded streaming filters:** `scan(..., where=...)` evaluates predicates per batch, so peak RAM tracks `batch_size`.
- **Multiprocess-safe remote downloads:** OS-file-lock dedupe, resumable partials, explicit completeness warnings.
- **Silent-corruption fixes:** BIT column writes, multi-chunk buffered reads, unsigned-column `where=` pushdown.

---

### 1.2 — The torch boundary (in progress)

Metadata has never needed PyTorch, but every metadata call paid about a second
for it, because the single native extension links `TORCH_LIBRARIES` and imports
`torch` in its module body. The rule this work establishes — **PyTorch loads
only for a call whose documented return type is a `torch.Tensor` or that takes
`device=`** — is enforced by `tests/test_torch_boundary.py` and measured by
`benchmarks/bench_import_boundary.py`.

Landed so far:

- `import torchfits.hdu` 883 ms → 2.6 ms and `import torchfits.io` 882 ms →
  34.6 ms, both with torch absent; `Header`/`Card` work in a torch-free process.
- The native module and metadata calls no longer initialize the Python
  `torch` package. `read_header`, `read_keys`, shape/HDU inventory and checksum
  verification pass with an import blocker installed.
- Arrow table reads now cross a real native raw-buffer boundary. Fixed columns
  and flat VLA values/offsets are returned as typed, Python-owned memoryviews
  without `THPVariable_Wrap`; strings, bits, vectors and VLA rows are materialized
  into Arrow in Python. `table.read`, `table.schema` and the `table` CLI command
  pass the same fresh-process blocker test.
- All seven metadata CLI commands (`info`, `header`, `probe`, `verify`, `table`,
  `copy`, `setkey`) are parser-safe and runtime-safe without torch. Pixel commands
  declare their tensor runtime explicitly and retain their thread controls.
- An interpreter-exit defect fixed: the cache hook imported the extension
  unconditionally, so a process that never loaded it printed a traceback (or a
  warning) on every exit.

- **The dynamic dependency boundary is split.** `libtorchfits_core` is a separate
  shared library holding CFITSIO, the shared read-metadata cache, the FITS
  inspection rules, and its own `parallel_for`; it links no Torch target. It is
  bound as `torchfits._core`, and the path-based metadata probes
  (`read_header`, `read_keys`, `read_colnames`, `read_nrows`, `read_num_hdus`,
  `read_hdu_type`, `read_shape`, `read_table_info`) now run entirely through it
  — so they never dlopen libtorch. Cold start for those calls went 332–337 ms →
  53 ms. `_C` resolves its `fits_*` symbols against the core (one CFITSIO in
  the process; `check_core_link.cmake` fails the build if any is left unbound),
  a build-id constant stamped into all three artifacts rejects a mismatched
  pair at import, and `tests/test_core_library.py` compares the two modules
  answer-for-answer.
- The split is now checked against **real observations**, not only synthetic
  fixtures. `tests/test_core_library_real_data.py` compares the two modules on
  all 409 (frame, HDU) records of the fetched CFHT sample data (three 1.6 GB
  MegaPipe mosaics, ten Rice-compressed MegaCam MEFs) and walks the whole
  corpus with `torch` and `numpy` blocked.
  `tests/test_reads_real_data.py` goes past the header and holds the *reads*
  against `astropy.io.fits` — a separate implementation with its own Rice
  decompressor: 4644x2112 decompressed frames match exactly, the raw VLA tile
  stream is byte-identical, and 435 megapixels of real mosaic match in every
  sampled window. A synthetic fixture cannot reach a 358-card header, a
  `NAXIS=0` primary whose `NAXIS1` is genuinely absent, or a compressed HDU
  that is an `IMAGE` to one library and a table to another.

Remaining, in order:

1. **Zero-copy Arrow assembly.** The raw transport is correctness-first: native
   tensors are copied once into Python-owned buffers, then the established Arrow
   conversion materializes typed NumPy views. Build primitive and list Arrow
   buffers directly from those memoryviews and benchmark before removing the
   staging copy.

   This is also what stands between the Arrow path and numpy. `torchfits.table`
   is torch-free today but not numpy-free, and auditing `src/torchfits/_table/`
   showed the numpy imports are overwhelmingly in the staging step
   (`arrow_convert.py` has nine function-scope `import numpy as np`, plus
   `_read_scan.py` and `_read_where.py`), not in the native transport. So
   "zero-copy" and "the table path stops needing numpy" are the same piece of
   work, not two. Note that pyarrow itself imports numpy when `pyarrow.compute`
   loads, so a fully numpy-free `table.read` is not reachable while the Arrow
   conversion goes through pyarrow at all — the achievable goal is that
   *torchfits* stops adding its own numpy dependency to the process.
2. **Image payloads in the core**, which also makes `compress`/`decompress`
   torch-free — they re-encode bytes and never do pixel math. Tensor
   destinations can then share one owned byte arena via `torch.from_blob`.
3. **A torch-free `torchfits.open()`.** `HDUList` owns a native handle that is
   the same object the tensor readers take, so it still loads `_C` (204 ms cold
   against 53 ms for a path-based probe). Enumerating the inventory through
   `libtorchfits_core` and opening the handle lazily on first tensor access
   closes the gap.

   Auditing `examples/` with an import blocker found a *second*, independent
   mechanism, which the dlopen half of this item does not fix:
   `hdu_list.py` dispatches on `isinstance(hdu, _table_hdu_types())` to decide
   what an HDU is, and resolving those classes imports
   `torchfits._hdu.table_hdu`, which imports `torch` at module scope. So
   `torchfits.open(...)` followed by nothing but `hdul[1].header` pulls in the
   Python `torch` package purely to answer a dispatch question — an
   `examples/example_mef_header.py` run with `torch` blocked fails on exactly
   that. `TableHDURef.materialize()` reaches the same module. Both need a
   torch-free marker on the classes (an `is_table` attribute checked before the
   `isinstance`) rather than the class import; that is why the item is a
   dispatch refactor and not just a lazy handle.
4. **Packaging the split.** Decide whether to publish `libtorchfits_core` as its
   own distribution: it helps metadata and table users, but pixel-math commands
   still need torch, so a separate package would only pay off for callers that
   never import torch at all.

## Current focus

- **Single-pass arena decode for buffered table reads.** Removes the
  largest remaining significant Linux benchmark deficits vs `fitsio` (narrow-table
  full reads with `mmap=False`, 21–22% on CPU and 13% on CUDA, plus
  `predicate_filter` on the `ascii_10000` catalog, 15% on CPU): decode
  straight into caller-visible, strided tensors instead of staging whole rows
  in scratch chunks. An API-visible change targeted at the next minor. The
  authoritative numbers are in [Benchmarks](benchmarks.md#performance-deficits);
  the per-run CSVs are published there.
- **Selective-projection fast path** in the same reader, so filtered scans
  stop paying for whole-row pread when only a few columns are needed.
- **Table semantics polish:** complex-column dtypes in `schema()`,
  consistent error types across the mutation API.
- **Native binding signatures:** `read_fits_table_filtered` declares a default
  on `column_names` ahead of a required `filters`, which is legal in C++ but
  has no Python signature (and the default is dead — nothing can omit it).
  `nb::sig()` on the bindings would make the generated stub exact instead of
  hand-corrected, and would also give `help()`/`inspect.signature` real names
  where stubgen currently emits `arg0`/`arg1`.
- **Object-store recipes:** row-band caching and range-fetch patterns for
  S3-style archives on top of the hardened HTTP downloader.
- **CLI wave 3:** thin `fitsverify` helper and fpack-style tile controls
  (no CFITSIO HTTPS drivers — torchfits keeps its own HTTP stack).

---

## Tooling decisions

Re-evaluated 2026-09; recorded so they are not silently revisited.

- **`ty` (Astral) is deferred to 1.0.** Latest is 0.0.80 — pre-1.0 beta, and
  not a drop-in for mypy (different defaults, different diagnostics). mypy
  `--strict` is the gate. Trigger to revisit: the measured cost it would
  remove — mypy runs cold in ~20 s over 95 source files, which is already
  tolerable in CI, so the case is ergonomic rather than blocking. Any trial
  should start as a non-blocking CI job alongside mypy, not a replacement.
- **nanobind split mode is not applicable.** It collapses the wheel matrix to
  one wheel per platform by targeting the Python 3.10 stable ABI, but the real
  constraint here is `libtorch_python`, which is CPython-version-specific, so
  torchfits would still ship one wheel per Python version. Adopted nanobind 3
  for the API/perf improvements only.
- **Dependency floors are aspirational, not tested.** Floors name the oldest
  release with wheels for the minimum supported Python (3.10); nothing
  installs them. Follow-up that would make them real: a lowest-direct
  resolution job (`pip install --resolution lowest-direct`, or a pixi minimum
  env) so the metadata cannot drift from reality again.

---

## 2.0.0 — Native C++ / GPU-Direct Architecture (Future)

The 2.0 major release aims to drop external legacy C library dependencies in favor of a modern, native C++/CUDA I/O engine:

- **Direct Storage-to-GPU Transport (GPUDirect Storage):** Direct DMA transfers from NVMe/object storage directly into NVIDIA GPU device memory (cuFile/GDS) without intermediate host-memory bouncing.
- **Native Astronomical Tile Codecs:** Pure C++20 and CUDA implementations of Rice, H-Compress, and Gzip decompression.
- **Asynchronous Batch Execution:** Fully non-blocking multi-file decoders scheduled via CUDA streams and CPU worker pools.
- **Stable Python API:** Maintaining full backwards compatibility with the 1.x `read_tensor`, `table.read`, and `torchfits.data` APIs.

### Scheduled 2.0 removals

Deprecated 1.x shims that will be dropped in 2.0:

- `configure_cpp_cache` (deprecated in favor of the cache-manager
  configuration path).
- `handle_cache_capacity` (per-path handle caching was removed; the
  argument is accepted and ignored).
- Legacy `write()` dict-vs-tensor argument conventions are documented
  but will be tightened to explicit keyword-only forms.

---

## Permanent Scope & Design Boundaries

To ensure focus, long-term maintainability, and peak performance, `torchfits` maintains strict scope boundaries:

- **No Celestial Coordinate Systems or WCS Math:** Coordinate transformations belong in `astropy.wcs`. `torchfits` outputs raw pixel tensors with standard header metadata for Astropy consumption.
- **No Physical Units Engine:** Quantity conversions belong in `astropy.units`.
- **No High-Level Astronomy Modeling:** Source extraction, PSF fitting, and continuum fitting belong in domain analysis packages (e.g. Photutils, SEP).
- **Format Integrity:** Strict compliance with the official IAU FITS standard.
