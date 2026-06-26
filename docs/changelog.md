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
