# torchfits

[![PyPI](https://img.shields.io/pypi/v/torchfits)](https://pypi.org/project/torchfits/)
[![GitHub release](https://img.shields.io/github/v/release/astroai/torchfits?include_prereleases&label=latest%20release)](https://github.com/astroai/torchfits/releases)
[![CI](https://github.com/astroai/torchfits/actions/workflows/ci.yml/badge.svg)](https://github.com/astroai/torchfits/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**torchfits** is high-performance FITS I/O: read and write FITS as tensors and
tables — cutouts, filters, and a shell CLI (C++ engine, vendored CFITSIO).
Optional datasets / transforms sit on top; you do not need an ML workflow to
benefit.

**Current:** [1.0.0](https://pypi.org/project/torchfits/) — built for the
**PyTorch 2.13 lane** (wheels are ABI-matched to the torch minor they ship
for). Docs: [stable](https://astroai.github.io/torchfits/) (latest `v*` tag) ·
[edge](https://astroai.github.io/torchfits/edge/) (`main` tip).

## Install

torchfits wheels are ABI-matched to the **PyTorch 2.13.x** minor, and the
wheel's metadata pins that range (`torch>=2.13,<2.14`) — so the install
command needs no torch restriction: pip installs or upgrades torch for you.

```bash
pip install torchfits
```

Works with any Python 3.10+ and any installed PyTorch ≥ 2.10: if you already
have torch 2.13.x (any flavor — CPU or CUDA), it is left untouched; an older
minor is upgraded to 2.13.x automatically (the torch C++ ABI is per-minor, so
torchfits must load against its lane). Pin explicitly only when you must keep
a specific torch minor:

```bash
pip install torchfits "torch>=2.13,<2.14"
```

Requires **Python 3.10+** and **PyTorch ≥ 2.10** (wheels are built for the
2.13.x lane; pip aligns your torch automatically). Pre-built wheels for Linux
x86_64 and macOS arm64 (CFITSIO is vendored).

Choose a PyTorch flavor in one line:

| You want | Command |
|---|---|
| Default (CUDA + CPU) | `pip install torchfits "torch>=2.13,<2.14"` |
| CPU-only (thin) | `pip install torchfits "torch>=2.13,<2.14" --extra-index-url https://download.pytorch.org/whl/cpu` |
| CUDA build (e.g. cu129) | `pip install torchfits "torch>=2.13,<2.14" --extra-index-url https://download.pytorch.org/whl/cu129` |

The `[cpu]` / `[cuda]` extras pin the matching torch build exactly
(`torch==2.13.0+cpu` / `torch==2.13.0+cu129`) after a one-time index setting
(`PIP_EXTRA_INDEX_URL` / `pip.conf`). Both extras are **Linux-only** — on
macOS they are no-ops (MPS ships inside the default wheel). In zsh, quote the
extra (`pip install 'torchfits[cpu]'`). Details:
[Install](https://astroai.github.io/torchfits/install/).

## At a glance

| Task | API |
|---|---|
| Image → tensor | `torchfits.read_tensor("img.fits", device="cuda")` |
| Write a tensor | `torchfits.write("out.fits", tensor)` |
| Filter a catalog in C++ | `torchfits.table.read(..., where="MAG < 20")` |
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

Coding agents can use the docs site or this repo; humans should still skim
[Quick start](https://astroai.github.io/torchfits/quickstart/) or
[Python workflows](https://astroai.github.io/torchfits/python-workflows/).

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
