# Compatibility and Parity Matrix

This matrix is the public compatibility contract for torchfits. It is based on
the current source, tests, and local comparator packages, not on older overview
docs. Status values are:

- **Supported**: implemented and covered by tests or benchmark gates.
- **Partial**: usable for documented cases with listed limitations.
- **Unsupported**: known limitation with explicit tests or source evidence.
- **Out of scope**: intentionally belongs outside torchfits.

## Summary

| Area | Status | Comparator / source | Evidence |
|---|---:|---|---|
| FITS image read/write | Supported | `astropy.io.fits`, `fitsio`, CFITSIO backend | `tests/test_api.py`, `tests/test_writing.py`, `tests/test_astropy_upstream_smoke.py`, `tests/test_fitsio_upstream_smoke.py` |
| Multi-extension FITS files | Supported | `astropy.io.fits` HDUList workflows | `tests/test_hdu.py`, `tests/test_astropy_upstream_smoke.py` |
| FITS headers and cards | Supported | Astropy/fitsio header reads and torchfits `Header` | `tests/test_header_parser.py`, `tests/test_complex_header.py`, `tests/test_astropy_upstream_smoke.py`, `tests/test_fitsio_upstream_smoke.py` |
| Checksums | Supported | fitsio/CFITSIO checksum workflows | `tests/test_checksum.py`, `tests/test_fitsio_upstream_smoke.py` |
| Compressed image reads | Supported | Astropy `CompImageHDU`, fitsio image reads | `tests/test_compression.py`, `tests/test_astropy_upstream_smoke.py`, `tests/test_fitsio_upstream_smoke.py` |
| Compressed image writes | Supported | CFITSIO compressed-image writer | `tests/test_writing.py`; tensor and numpy array/list payloads are supported for compressed image HDUs |
| Unsigned image convention | Supported | Astropy/fitsio `BZERO` convention | `uint16`/`uint32` image reads and writes preserve unsigned integer semantics, including HDUList writes |
| Binary table reads/writes | Supported | `astropy.io.fits`, `fitsio` | `tests/test_table.py`, `tests/test_table_file_ops.py`, `tests/test_astropy_upstream_smoke.py`, `tests/test_fitsio_upstream_smoke.py` |
| ASCII table reads/writes | Supported | Astropy `TableHDU` | `tests/test_ascii_table.py`, `tests/test_table_file_ops.py`, `tests/test_astropy_upstream_smoke.py` |
| Table projection, row slicing, filtering | Supported | fitsio rows/columns/where workflows | `tests/test_table_filtering.py`, `tests/test_fitsio_upstream_smoke.py` |
| Table mutation | Supported | fitsio-readable results | `tests/test_table_file_ops.py`, `tests/test_fitsio_upstream_smoke.py` |
| VLA table columns | Partial | Astropy/fitsio variable-length arrays | buffered reads/writes are covered; mmap reads/updates are unsupported |
| Complex table columns | Partial | Astropy complex FITS columns | buffered and mmap reads are covered; mmap updates are unsupported |
| Bit table columns | Supported | Astropy/fitsio `X` columns | read/write returns boolean bit arrays; mmap updates remain unsupported |
| Unsigned table integer convention | Supported | Astropy/fitsio `TZERO` convention | `uint16`/`uint32` table reads and writes preserve unsigned integer semantics through root, `table.write`, and HDUList paths |
| Scaled image data | Supported | FITS BSCALE/BZERO semantics | `tests/test_astropy_upstream_smoke.py`, `tests/test_integration.py`, `benchmarks/bench_scaled.py` |
| Scaled table columns | Partial | CFITSIO-backed table path | buffered reads are covered; mmap updates are unsupported |
| GPU reads | Supported | PyTorch device transfer after FITS decode | `tests/test_api.py`, examples |
| GPU writes | Partial | torch tensor inputs | non-CPU tensors are copied to host before FITS write |
| Arrow/Pandas/Polars/DuckDB interop | Partial | optional ecosystem libraries | `tests/test_interop.py`, `tests/test_arrow_table_api.py`; optional dependencies control availability |
| Full Astropy API parity | Out of scope | Astropy package surface | torchfits targets common FITS I/O workflows only |
| Full fitsio API parity | Out of scope | fitsio package surface | torchfits targets common FITS I/O workflows only |
| Full CFITSIO API parity | Out of scope | CFITSIO C API | torchfits exposes selected PyTorch-native behavior only |
| WCS solving/modeling | Out of scope | Astropy WCS / wcslib-style domains | use specialized WCS packages |
| Sky coordinates, sphere geometry, HEALPix | Out of scope | sky-domain packages | outside torchfits |
| Sky-domain simulation and training pipelines | Out of scope | application/domain code | belongs outside torchfits |

## Why full parity is not claimed

Astropy, fitsio, and CFITSIO each expose more than the FITS I/O workflows that
torchfits intentionally owns. Full Astropy parity would include the wider
Astropy package and many object-model conveniences. Full fitsio parity would
include every method and keyword of its Python wrapper. Full CFITSIO parity
would mean exposing a large low-level C API surface, including file drivers and
update modes that are not PyTorch-native.

The target is narrower and testable: common FITS image, HDU, header, checksum,
compression, table, and slicing/filtering workflows, plus selected
CFITSIO-backed behavior that torchfits exposes directly.

## Known mmap limitations

The high-level table readers keep `mmap=True` ergonomic: cases that require
buffered CFITSIO handling fall back to the safe buffered path. Forced mmap table
updates reject unsupported layouts instead of rewriting them through an unsafe
path.

Affected layouts:

- VLA columns;
- scaled columns;
- mmap updates for VLA, bit, scaled, string, and complex columns.

Use the buffered table path for those cases.

## Benchmark scope

Benchmarks compare FITS image I/O and FITS table I/O against
`astropy.io.fits` and `fitsio`. WCS, sphere, HEALPix, and sky-domain benchmark
suites are not torchfits benchmarks.
