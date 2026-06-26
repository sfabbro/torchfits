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

- Keep the split clean: torchfits owns FITS I/O; torchsky owns sky-domain tensor
  models and depends on torchfits at file boundaries.
- Expand parity smoke tests for `fitsio` and `astropy.io.fits` whenever a public
  claim is added to `docs/parity.md`.
- Keep unsupported mmap behavior explicit for VLA, bit, scaled, string, and
  complex table cases.
- Keep benchmark evidence scoped to FITS images and FITS tables, with separate
  rows for mmap fairness, compression, scaling, and table pushdown.
- Maintain release gates that scan docs for stale WCS/sphere/HEALPix ownership
  claims.

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

## Release gate

A release may claim parity only for rows that have one of:

- a passing test listed in `benchmarks/replays/upstream_sources.json`;
- a benchmark row in the FITS or FITS-table benchmark suites;
- an explicit unsupported or out-of-scope entry in `docs/parity.md`.
