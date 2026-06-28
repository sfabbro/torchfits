# Changelog

All notable changes to torchfits are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Refocused torchfits as a FITS I/O package: images, HDUs, headers, checksums,
  compression, FITS tables, caching, and table interop.
- Removed stale public claims that torchfits owns WCS, sphere geometry, HEALPix,
  sky-domain simulation, or training pipelines. Those domains belong outside
  torchfits.
- Added a roadmap and compatibility matrix that distinguish supported, partial,
  unsupported, and out-of-scope behavior.
- Replaced broad parity claims with test-backed parity tiers for common fitsio,
  Astropy, and selected CFITSIO-backed workflows.

### Added

- `docs/roadmap.md` for the FITS I/O roadmap and parity tiers.
- `docs/parity.md` for the public compatibility matrix.
- Astropy upstream smoke coverage for common image, HDU, compressed-image,
  table, ASCII table, VLA, complex column, and scaled-image workflows.
- Documentation integrity checks for stale WCS/sphere/HEALPix ownership claims.
- Supported-status promotion for in-place mmap table updates on COMPLEX
  (`1C`/`1M`), BIT (`8X`), and fixed-width STRING (`12A`-style) columns.
  `torchfits.table.update_rows(..., mmap=True)` now writes these column
  types correctly on disk. Verified via raw byte inspection
  (`scripts/diag_string_bytes_v2.py`) and an astropy upstream-reader
  roundtrip. VLA columns remain explicitly unsupported in the mmap
  fast path by design.
- Astropy and fitsio upstream smoke coverage that exercises the
  COMPLEX / BIT / fixed-width STRING mmap-update parity shift,
  including right-padding to the declared column width and verification
  vs the upstream readers. The 8A-string assertion falls back to
  astropy because the local fitsio upstream misdecodes updated `8A`
  rows (the on-disk bytes are bit-exact to the expected layout; this
  is an upstream-reader limitation, not a torchfits writer bug).
- `tests/test_astropy_upstream_smoke.py::test_astropy_compimage_compression_variants_match_torchfits`
  exercising additional `astropy.io.fits.CompImageHDU` compression
  variants (RICE / HCOMPRESS / PLIO) round-tripped against torchfits.

### Removed

- Dataset/training helper namespace from the torchfits package contract.

## Earlier releases

Earlier 0.1.x through 0.3.x releases included broader experimental astronomy
domains. The current package contract is FITS I/O only; consult the current
README, API reference, roadmap, and parity matrix for supported behavior.

[0.1.0]: https://github.com/sfabbro/torchfits/releases/tag/v0.1.0
[0.1.1]: https://github.com/sfabbro/torchfits/releases/tag/v0.1.1
[0.2.0]: https://github.com/sfabbro/torchfits/releases/tag/v0.2.0
[0.2.1]: https://github.com/sfabbro/torchfits/releases/tag/v0.2.1
[0.3.0]: https://github.com/sfabbro/torchfits/releases/tag/v0.3.0
[0.3.1]: https://github.com/sfabbro/torchfits/releases/tag/v0.3.1
[0.3.2]: https://github.com/sfabbro/torchfits/releases/tag/v0.3.2
