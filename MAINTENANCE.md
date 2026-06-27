# torchfits maintenance policy

**Active development is on torchfits** — a focused FITS I/O library for PyTorch.

## Scope

- FITS image and table read/write
- HDUs, headers, checksums, compression, caching
- Table interop (Arrow, Pandas, Polars, DuckDB)

WCS, HEALPix, sphere geometry, sky-domain simulation, and training pipelines are
out of scope for this repository.

## Release lines

- **v0.3.2** and the **0.3.x** branch remain available for users who need the
  older broader package surface.
- New FITS I/O work lands on the current mainline.

## Where to contribute

- FITS I/O features, parity tests, benchmarks, and docs: **torchfits**
- See [docs/roadmap.md](docs/roadmap.md) and [docs/parity.md](docs/parity.md) for
  the supported contract.

## C++ code conventions

- **No inline RAII structs in `.cpp` files.** All resource-management guards
  live in shared headers:
  - `FitsHandleGuard` (`cache.h`) — `fitsfile*` handles (cached/non-cached).
  - `MMapHandle` (`hardware.h`) — mmap regions (filename-based or adopt existing).
- If new C++ resources need RAII wrappering, add the guard to the appropriate
  header rather than defining an inline struct at the usage site.
- See `scripts/check_duplicate_cpp.py` for CI-enforced duplicate-function checks.
