# torchfits

[![PyPI](https://img.shields.io/pypi/v/torchfits)](https://pypi.org/project/torchfits/)

[![CI](https://github.com/sfabbro/torchfits/actions/workflows/ci.yml/badge.svg)](https://github.com/sfabbro/torchfits/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-green)](LICENSE)

**torchfits** is a focused FITS I/O library for PyTorch. It reads and writes
FITS images, headers, HDUs, compressed images, and FITS tables through a
multi-threaded C++ engine with vendored CFITSIO, returning tensor-native data
without requiring users to build NumPy-to-torch glue code.

It is not a full replacement for Astropy, fitsio, or CFITSIO. The supported
surface is documented explicitly in [docs/parity.md](docs/parity.md), with
source-backed tests for each claimed parity area. WCS, sky-coordinate models,
HEALPix, sphere geometry, and sky-domain simulation workflows belong in
[`torchsky`](https://github.com/sfabbro/torchsky).

## At a Glance

| Task | Traditional stack | torchfits equivalent |
|---|---|---|
| Read image to GPU | astropy/fitsio &rarr; numpy &rarr; torch &rarr; `.to(device)` | `torchfits.read("img.fits", device="cuda")` |
| Write tensor to FITS | tensor &rarr; numpy &rarr; astropy HDU &rarr; writeto | `torchfits.write("out.fits", tensor)` |
| Filter large table | load all rows &rarr; mask in Python | `where="MAG < 20"` pushdown in C++ |
| Read multi-extension files | manual HDU dispatch | `with torchfits.open("mef.fits") as hdul: ...` |
| Verify FITS checksums | comparator-specific helpers | `torchfits.verify_checksums(path)` |

## Features

**FITS I/O** &mdash; Multi-threaded C++ core with SIMD-optimized type conversion,
memory-mapped image reads, intelligent chunking, and adaptive buffering. Reads
and writes images, binary/ASCII tables, compressed images, and multi-extension
FITS files with header round-trip coverage.

**Table Engine** &mdash; Arrow-native table API with predicate pushdown (`where=`), column projection, row slicing, streaming `scan()`, and in-place mutations (append, insert, update, delete rows and columns). Interop with Pandas, Polars, DuckDB, and PyArrow.

**Compatibility Contract** &mdash; Parity is tracked by tier: truthful public docs,
fitsio core workflow parity, Astropy common workflow parity, selected CFITSIO
backend behavior, and explicit non-goals. See [docs/parity.md](docs/parity.md).

## Install

```bash
pip install torchfits
```

Pre-built wheels are available for Linux and macOS (x86_64, arm64). No system CFITSIO needed&mdash;it's vendored and compiled automatically.

From source:

```bash
git clone https://github.com/sfabbro/torchfits.git
cd torchfits
pip install -e .
```

Requires Python 3.10+, a C++17 compiler, CMake 3.21+, and PyTorch 2.0+.

## Quick Start

### Read an image to GPU

```python
import torchfits

data, header = torchfits.read("science.fits", device="cuda", return_header=True)
# data: torch.Tensor on CUDA, shape e.g. (4096, 4096), dtype torch.float32
```

### Filter and stream a catalog

```python
# Predicate pushdown — only matching rows leave C++
table = torchfits.table.read(
    "catalog.fits",
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20.0 AND CLASS_STAR > 0.9",
)

# Stream 100M rows in constant memory
for batch in torchfits.table.scan("survey.fits", batch_size=50_000):
    process(batch)
```

### Multi-HDU access

```python
with torchfits.open("multi_ext.fits") as hdul:
    print(hdul)            # pretty-printed summary
    img = hdul[0].data     # image tensor
    tbl = hdul[1].data     # table dict
    tbl_filtered = hdul[1].filter("FLUX > 100 AND FLAG = 0")
```

### Write back

```python
torchfits.write("output.fits", data, header=header, overwrite=True)
torchfits.table.write("catalog_out.fits", table_dict, header=header)
```

## Benchmarks

torchfits benchmark evidence is limited to FITS image I/O and FITS table I/O.
Comparators are `astropy.io.fits` and `fitsio`; selected CFITSIO behavior is
validated through the torchfits native backend and smoke tests. Sky-domain
benchmark suites live with torchsky.

Methodology, reproducible commands, results, and known deficits: [`docs/benchmarks.md`](docs/benchmarks.md)

## Documentation

| | |
|---|---|
| [API Reference](docs/api.md) | Full public API with signatures and examples |
| [Roadmap](docs/roadmap.md) | FITS I/O roadmap and parity tiers |
| [Parity Matrix](docs/parity.md) | Supported, partial, unsupported, and out-of-scope features |
| [Examples](docs/examples.md) | Runnable scripts for every major workflow |
| [Installation](docs/install.md) | Build from source, GPU setup, troubleshooting |
| [Benchmarks](docs/benchmarks.md) | Methodology, commands, and latest numbers |
| [Changelog](docs/changelog.md) | Version history and migration notes |
| [Release Checklist](docs/release.md) | Maintainer guide for cutting releases |

## Contributing

```bash
git clone https://github.com/sfabbro/torchfits.git
cd torchfits
pixi install
pixi run test
```

The project uses [pixi](https://pixi.sh) for environment management, [ruff](https://github.com/astral-sh/ruff) for linting, and [pytest](https://docs.pytest.org) for testing.

## License

[GPL-2.0](LICENSE)
