# torchfits

[![PyPI](https://img.shields.io/pypi/v/torchfits)](https://pypi.org/project/torchfits/)
[![Wheels](https://github.com/sfabbro/torchfits/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/sfabbro/torchfits/actions/workflows/build_wheels.yml)
[![CI](https://github.com/sfabbro/torchfits/actions/workflows/ci.yml/badge.svg)](https://github.com/sfabbro/torchfits/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-green)](LICENSE)

**torchfits** reads and writes FITS files directly as PyTorch tensors on CPU, CUDA, or Apple Silicon MPS. It is built on a multi-threaded C++ engine with vendored CFITSIO and covers the same ground as `astropy.io.fits`, `fitsio`, `healpy`, `healsparse`, and `astropy.wcs`&mdash;but with native tensor output and no intermediate NumPy copies.

## At a Glance

| Task | Traditional stack | torchfits equivalent |
|---|---|---|
| Read image to GPU | astropy/fitsio &rarr; numpy &rarr; torch &rarr; `.to(device)` | `torchfits.read("img.fits", device="cuda")` |
| Write tensor to FITS | tensor &rarr; numpy &rarr; astropy HDU &rarr; writeto | `torchfits.write("out.fits", tensor)` |
| Filter large table | load all rows &rarr; mask in Python | `where="MAG < 20"` pushdown in C++ |
| WCS coordinate transform | astropy.wcs / PyAST / Kapteyn | `torchfits.get_wcs()` &mdash; pure PyTorch, batch, any device |
| HEALPix pixelization | healpy / hpgeom / astropy-healpix | `torchfits.sphere.ang2pix()` &mdash; CPU + CUDA + MPS |
| Spherical harmonics | healpy (CPU, NumPy) | `torchfits.sphere.map2alm()` / `alm2map()` |
| Sparse HEALPix maps | healsparse (NumPy) | `torchfits.sphere.sparse` (tensor-native) |
| Spherical polygons | spherical-geometry (NumPy) | `torchfits.sphere.geom` (non-convex, GPU) |

## Features

**FITS I/O** &mdash; Multi-threaded C++ core with SIMD-optimized type conversion, memory-mapped reads, intelligent chunking, and adaptive buffering. Reads and writes images, binary tables, compressed tiles (Rice, HCOMPRESS, GZIP, PLIO), and multi-extension FITS files with full header round-trip fidelity.

**Table Engine** &mdash; Arrow-native table API with predicate pushdown (`where=`), column projection, row slicing, streaming `scan()`, and in-place mutations (append, insert, update, delete rows and columns). Interop with Pandas, Polars, DuckDB, and PyArrow.

**WCS** &mdash; Pure-PyTorch implementation of 13 projections (TAN, SIN, ARC, ZPN, ZEA, STG, CEA, CAR, MER, AIT, MOL, HPX, SFL) with SIP, TPV, TNX, and ZPX polynomial distortions. Batch `pixel_to_world` / `world_to_pixel` on any device. Validated against astropy.wcs, PyAST, and Kapteyn.

**Sphere** &mdash; HEALPix primitives (`ang2pix`, `pix2ang`, `nest2ring`, `ring2nest`, `neighbors`, interpolation), spherical polygons (non-convex region queries, area, containment), Multi-Order Coverage (MOC) maps, HealSparse interop, and spherical harmonic transforms (`map2alm`, `alm2map`, scalar and spin). Benchmarked against healpy, hpgeom, astropy-healpix, mhealpy, healsparse, and spherical-geometry.

**Compatibility** &mdash; `torchfits.sphere.compat` provides a healpy-compatible API surface (`ang2pix`, `pix2ang`, `query_circle`, `map2alm`, `alm2map`, `synalm`, `synfast`, `anafast`, `smoothing`, and more) so existing healpy code can switch with minimal changes. WCS objects follow the same `pixel_to_world` / `world_to_pixel` interface as astropy.wcs.

**ML Integration** &mdash; `FITSDataset` and `IterableFITSDataset` work with `torch.utils.data.DataLoader` for multi-worker streaming. Built-in astronomical transforms (ZScale, Asinh, Log, Power stretches; crop, flip, rotation augmentations; redshift shift, error perturbation) with composable pipelines.

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

### WCS coordinate transforms

```python
wcs = torchfits.get_wcs("image.fits")

# Batch pixel → sky on GPU
ra, dec = wcs.pixel_to_world(x_pixels, y_pixels)  # torch.Tensor in, Tensor out
```

### HEALPix and spherical harmonics

```python
from torchfits.sphere import ang2pix, map2alm, alm2map

ipix = ang2pix(nside=2048, theta=theta, phi=phi, nest=True)  # GPU-accelerated
alm = map2alm(healpix_map, lmax=512)
smoothed = alm2map(alm, nside=2048, lmax=512)
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

torchfits is benchmarked against astropy, fitsio, healpy, hpgeom, astropy-healpix, mhealpy, healsparse, spherical-geometry, PyAST, and Kapteyn across four domains (FITS I/O, tables, WCS, sphere). Correctness is validated against each upstream library using their own test fixtures and public reference data.

Methodology, reproducible commands, results, and known deficits: [`docs/benchmarks.md`](docs/benchmarks.md)

## Documentation

| | |
|---|---|
| [API Reference](docs/api.md) | Full public API with signatures and examples |
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
