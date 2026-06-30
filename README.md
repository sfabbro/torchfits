# torchfits

[![PyPI](https://img.shields.io/pypi/v/torchfits)](https://pypi.org/project/torchfits/)

[![CI](https://github.com/astroai/torchfits/actions/workflows/ci.yml/badge.svg)](https://github.com/astroai/torchfits/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-green)](LICENSE)

**torchfits** is a focused FITS I/O library for PyTorch. It reads and writes
FITS images, headers, HDUs, compressed images, and FITS tables through a
multi-threaded C++ engine with vendored CFITSIO, returning tensor-native data
without requiring users to build NumPy-to-torch glue code.

It is not a full replacement for Astropy, fitsio, or CFITSIO. The supported
surface is documented explicitly in [docs/parity.md](docs/parity.md), with
source-backed tests for each claimed parity area. WCS, sky-coordinate models,
HEALPix, sphere geometry, and sky-domain simulation workflows are out of scope
for torchfits.

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

## What's New in 0.5.0

0.5.0 is a focused FITS I/O release: tensor-native reads/writes, Arrow tables with
C++ predicate pushdown, mmap in-place table updates (BIT, complex, fixed-width strings),
unsigned integer conventions, compressed-image parity, and test-backed compatibility
with common `astropy.io.fits` and `fitsio` workflows.

Recent beta improvements:

- Shared FITS table schema parsing (`fits_schema`) — one code path for TFORM/VLA/string/bit/unsigned columns across table, HDU, and read dispatch.
- Smarter `where=` reads — automatic choice between Arrow filter and C++ pushdown based on table size and column layout.
- Cleaner cache boundaries — table handle caches live in `_table.cache` without circular imports through `io`.

Full history: [docs/changelog.md](docs/changelog.md). Engineering decomposition plan for 0.6.0: [docs/roadmap.md](docs/roadmap.md).

## Performance

Median wall-clock from the lab exhaustive benchmark suite (`0.5.0b3` snapshot, H100 CUDA where noted). See [docs/benchmarks.md](docs/benchmarks.md) for methodology and reproducible commands.

| Case | torchfits | astropy | Speedup |
|---|---:|---:|---:|
| Large float32 image read (16 MB, CPU) | 4.26 ms | 15.19 ms | **3.6×** |
| Same read @ CUDA | 3.61 ms | 16.16 ms | **4.6×** |
| Compressed Rice image (CPU) | 9.20 ms | 29.33 ms | **3.2×** |
| 50× repeated 100×100 cutouts (CPU) | 5.53 ms | 80.93 ms | **15.5×** |
| Table read (100k rows, 8 cols) | 93 μs | 6.33 ms | **68×** |

Tables and GPU transports are CPU-resident in all backends today; GPU rows measure host decode + H2D copy, not disk→GPU bypass.

## Install

```bash
pip install torchfits
```

Pre-built wheels are available for Linux and macOS (x86_64, arm64). No system CFITSIO needed&mdash;it's vendored and compiled automatically.

From source:

```bash
git clone https://github.com/astroai/torchfits.git
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
# table: pyarrow.Table

# Stream 100M rows in constant memory
for batch in torchfits.table.scan("survey.fits", batch_size=50_000):
    process(batch)  # batch: pyarrow.RecordBatch
```

### Multi-HDU access

```python
with torchfits.open("multi_ext.fits") as hdul:
    print(hdul)            # pretty-printed summary
    img = hdul[0].data     # image tensor
    tbl = hdul[1].data     # dict-like table accessor
    tbl_filtered = hdul[1].filter("FLUX > 100 AND FLAG = 0")
```

### Write back

```python
torchfits.write("output.fits", data, header=header, overwrite=True)
# table_dict is a dict of column names to 1D arrays/tensors
torchfits.table.write("catalog_out.fits", table_dict, header=header, overwrite=True)
```

## Benchmarks

torchfits benchmark evidence is limited to FITS image I/O and FITS table I/O.
Comparators are `astropy.io.fits` and `fitsio`; selected CFITSIO behavior is
validated through the torchfits native backend and smoke tests.

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
git clone https://github.com/astroai/torchfits.git
cd torchfits
pixi install
pixi run test
```

The project uses [pixi](https://pixi.sh) for environment management, [ruff](https://github.com/astral-sh/ruff) for linting, and [pytest](https://docs.pytest.org) for testing.

## License

[GPL-2.0](LICENSE)
