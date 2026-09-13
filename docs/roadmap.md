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
- An interpreter-exit defect fixed: the cache hook imported the extension
  unconditionally, so a process that never loaded it printed a traceback (or a
  warning) on every exit.

Remaining, in order:

1. **Torch-free core library.** Extract the inspection half of `FITSFile`, the
   `SharedReadMeta` caches and a `parallel_for` of our own into a single
   `libtorchfits_core` shared library; bind it as `torchfits._core` with a
   build-id guard against `_C`. Target: `read_header` 1136 ms → ~5 ms, and the
   metadata CLI commands (`info`, `header`, `probe`, `verify`, `copy`, `setkey`,
   `table`) off torch entirely.
2. **Buffer transport for tables.** Return owned byte arenas instead of
   `torch::Tensor` buffers, build Arrow from them zero-copy, and make
   `table.read_torch` one `torch.from_blob` destination. Target: `table.read`
   1555 ms → ~140 ms with no torch installed, and one copy fewer on the tensor
   path.
3. **Image payloads in the core**, which also makes `compress`/`decompress`
   torch-free — they re-encode bytes and never do pixel math.
4. **Mechanical proof and packaging.** A test asserting the core's dynamic
   dependencies contain no libtorch, docs, and a decision on publishing the core
   as its own distribution (it helps the metadata and table audience, not the
   pixel-math commands, which will always need torch).

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
