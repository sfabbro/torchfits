# Roadmap

`torchfits` is a focused FITS I/O package for PyTorch. The roadmap is organized
around an explicit compatibility contract rather than broad claims of full
Astropy, fitsio, or CFITSIO replacement.

## Parity tiers

| Tier | Target | Meaning |
|---|---|---|
| 0 | Public contract | README, API docs, examples, and release notes describe only the implemented FITS I/O surface. |
| 1 | fitsio core workflows | Common image/table/header/checksum/compression workflows interoperate with `fitsio`. |
| 2 | Astropy common workflows | Common `astropy.io.fits` HDU, header, image, compressed-image, and table workflows interoperate with torchfits. |
| 3 | Selected CFITSIO behavior | Native backend behavior is documented where torchfits intentionally exposes CFITSIO-backed semantics. |
| 4 | Explicit non-goals | Full CFITSIO API parity, WCS solving/modeling, sphere geometry, HEALPix, and sky-domain simulation are outside torchfits. |

## Near-term work

- Treat parity as a tested compatibility surface, not a blanket claim that
  torchfits reimplements Astropy, fitsio, or the CFITSIO C API.
- Keep the package boundary clean: torchfits owns FITS I/O only; sky-domain
  tensor models and simulation workflows stay outside this repository.
- Expand parity smoke tests for `fitsio` and `astropy.io.fits` whenever a public
  claim is added to `docs/parity.md`.
- Keep unsupported mmap behavior explicit for VLA and scaled table columns.
- Keep benchmark evidence scoped to FITS images and FITS tables, with separate
  rows for mmap fairness, compression, scaling, and table pushdown.
- Maintain release gates that scan docs for stale WCS/sphere/HEALPix ownership
  claims.

### Permanent design decisions (not gaps)

These Partial items are inherent format limitations or deliberate architectural
choices, not work items to be closed:

- **VLA mmap reads/updates** — Variable-length arrays use a separate heap with
  pointer indirection; flat `mmap` cannot stride across variable-length records.
  The buffered CFITSIO path is the correct solution.
- **Scaled column mmap updates** — Reverse-scaling arithmetic risks precision
  loss and overflow when writing floats back through integer storage. Unsafe by
  design.
- **GPU writes** — The CFITSIO C API requires host `void*` pointers. Bypassing
  it requires CUDA kernels or GPUDirect Storage, massive engineering for
  marginal gain. Host-copy is intentional.
- **Arrow/Pandas/Polars/DuckDB interop as optional** — Keeping these as optional
  dependencies preserves PyTorch's lightweight package boundaries.

## Longer-term candidates

- Broaden Astropy table parity where it is useful for FITS users: additional
  header/card round-trips, richer ASCII table schemas, and more variable-length
  array cases.
- Improve compressed-image write coverage beyond the current supported tensor
  image payloads, while keeping unsupported compressed table/dict payload cases
  explicit.
- Add benchmark replay snapshots for representative public FITS files so
  performance claims are tied to reproducible inputs.
- Consider additional CFITSIO-backed capabilities only when they can be exposed
  through a small PyTorch-native API and covered by tests.

## 0.6.0 — core module decomposition

Engineering milestone after the 0.5.0 beta quick wins (shared `fits_schema`, where-read
policy, table handle cache split). Goal: keep behavior stable while making the Python I/O
layer maintainable — no file in the hot path should sprawl past ~1k lines without a
compelling reason.

### Priority 1 — split `torchfits.table` into `_table/`

`table.py` is still ~3.7k lines. Decompose into focused modules and keep
`torchfits.table` as a thin re-export surface:

| Module | Owns |
|---|---|
| `_table/cache.py` | C++ FITS file / TableReader LRU caches *(done in 0.5.0)* |
| `_table/schema.py` | Thin wrappers over `fits_schema` for path-based metadata |
| `_table/arrow_convert.py` | numpy/tensor/VLA/complex → Arrow conversion |
| `_table/read.py` | `read`, `scan`, `_read_cpp_numpy_table`, batch assembly |
| `_table/where.py` | where masks, C++ pushdown, `_filter_table_with_where` |
| `_table/mutate.py` | insert/delete/update columns and rows, HDU rewrite |
| `_table/interop.py` | pandas, polars, duckdb, parquet helpers |

**Exit criteria:** each module ≤ ~600 lines; public `torchfits.table` API unchanged;
existing table tests pass without modification.

### Priority 2 — tame `read_dispatch.read_unified`

`read_dispatch.py` crossed 1k lines via fast-path accretion. Refactor to:

- `ReadDeps` dataclass (replace 18 injected callbacks)
- Explicit strategy list: batch paths → CPU fast → generic fast → fallback
- Each strategy returns `NotApplicable` or a result; no nested `recursive_read` closure

**Exit criteria:** `read_unified` ≤ ~150 lines; no behavior change in image/table/batch reads.

### Priority 3 — unify table read surfaces

Today there are three paths to table bytes: `torchfits.table.read` (Arrow),
`torchfits.read(..., mode="table")` (tensors), and `TableHDU` / `TableHDURef` (HDU workflow).
Pick two canonical pipelines (Arrow + tensor) that share the same C++ reader layer.

**Exit criteria:** documented ownership boundary in `docs/api.md`; no duplicated schema
walks outside `fits_schema`.

### Priority 4 — C++ table read consolidation *(optional, high effort)*

Replace the seven-deep `_read_cpp_numpy_table` API fallback chain with one C++
`read_table_chunk(...)` entry that picks mmap vs buffered vs row-ranges internally.

**Exit criteria:** one Python call site for table chunk reads; fallback chain deleted.

### Out of scope for 0.6.0

- Full CFITSIO API parity
- GPU FITS writes
- Rewriting `TableHDU.__init__` type coercion (separate hardening pass)

## Release gate

A release may claim parity only for rows that have one of:

- a passing test listed in `benchmarks/replays/upstream_sources.json`;
- a benchmark row in the FITS or FITS-table benchmark suites;
- an explicit unsupported or out-of-scope entry in `docs/parity.md`.
