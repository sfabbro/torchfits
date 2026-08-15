# torchfits

[![PyPI](https://img.shields.io/pypi/v/torchfits)](https://pypi.org/project/torchfits/)
[![GitHub release](https://img.shields.io/github/v/release/astroai/torchfits?include_prereleases&label=latest%20release)](https://github.com/astroai/torchfits/releases)
[![CI](https://github.com/astroai/torchfits/actions/workflows/ci.yml/badge.svg)](https://github.com/astroai/torchfits/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

FITS I/O for PyTorch: read and write FITS as tensors and tables — cutouts,
predicate filters, and a shell CLI (C++ engine, vendored CFITSIO).

Prebuilt binary wheels for **Python 3.10–3.14** across **PyTorch 2.11, 2.12, and 2.13** on Linux (x86_64, aarch64) and macOS (Apple Silicon arm64) with vendored CFITSIO.

Docs: [stable](https://astroai.github.io/torchfits/) · [edge](https://astroai.github.io/torchfits/edge/) (`main` tip)

## Install

```bash
pip install torchfits
```

Prebuilt wheels automatically enable GPU acceleration (CUDA on Linux, MPS on macOS) when available, or fall back to CPU. For CPU-only installs, custom CUDA builds, or existing PyTorch 2.11/2.12 environments, see the [Installation guide](docs/install.md).

## At a glance

| Task | API |
|---|---|
| Image → tensor | `torchfits.read_tensor("img.fits", device="cuda")` |
| Write a tensor | `torchfits.write("out.fits", tensor)` |
| Filter a catalog | `torchfits.table.read(..., where="MAG < 20")` |
| Columns as tensors | `torchfits.table.read_torch(..., where=...)` |
| Open a MEF | `with torchfits.open("mef.fits") as hdul: …` |
| Train | `FitsImageDataset` + `make_loader(..., num_workers=4)` |
| Shell | `torchfits info` / `header` / `convert` / `cutout` / … |

## Quick start

```python
import torchfits

tensor = torchfits.read_tensor("image.fits", hdu=0, device="cpu")

table = torchfits.table.read(
    "catalog.fits",
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20.0",
)
```

```bash
torchfits info science.fits
torchfits convert catalog.fits out.parquet --hdu 1
torchfits cutout 'science.fits[100:256,100:256]' cutout.fits
```

## Learn more

| | |
|---|---|
| [Documentation](https://astroai.github.io/torchfits/) | Quick start, Python workflows, API, CLI |
| [Python workflows](https://astroai.github.io/torchfits/python-workflows/) | Which API for images, tables, cutouts, training |
| [Examples](https://astroai.github.io/torchfits/examples/) | Runnable scripts + transform plots |
| [Benchmarks](https://astroai.github.io/torchfits/benchmarks/) | Methodology and scorecards |
| [Changelog](https://astroai.github.io/torchfits/changelog/) | Release notes |

## Develop

Pixi-first (do not use bare `python` for project work):

```bash
git clone https://github.com/astroai/torchfits.git
cd torchfits
pixi install
pixi run preflight-push   # fast gate while editing
pixi run test             # full unit suite
pixi run ci-local         # pre-push parity
```

Agent conventions: [`AGENTS.md`](AGENTS.md). Release process: [`docs/release.md`](docs/release.md).

## License

[MIT](LICENSE)
