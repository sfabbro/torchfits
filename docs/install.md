# Installation

## From PyPI

```bash
pip install torchfits
```

Pre-built wheels are available for Linux (x86_64, aarch64) and macOS (x86_64, arm64). The wheels bundle [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/) — no system library needed.

Requires Python 3.10+ and [PyTorch](https://pytorch.org/) 2.0+.

## From source

### Prerequisites

- Python 3.10+
- C++17 compiler (GCC 10+, Clang 14+, or MSVC 2019+)
- [CMake](https://cmake.org/) 3.21+
- [Ninja](https://ninja-build.org/) (recommended)
- [PyTorch](https://pytorch.org/) 2.0+
- [NumPy](https://numpy.org/) 1.20+

On macOS, install Xcode Command Line Tools if not present:

```bash
xcode-select --install
```

### Build

```bash
git clone https://github.com/sfabbro/torchfits.git
cd torchfits
./extern/vendor.sh      # download vendored CFITSIO sources
pip install -e .        # editable install
```

For a release build:

```bash
pip install .
```

The build uses [scikit-build-core](https://scikit-build-core.readthedocs.io/) and [nanobind](https://nanobind.readthedocs.io/) for the C++ extension.

### Vendored CFITSIO

torchfits vendors CFITSIO to avoid system-library version mismatches. The `extern/vendor.sh` script downloads the source and places it in `extern/cfitsio/`. By default it resolves the latest published tag; pin a version with:

```bash
./extern/vendor.sh --cfitsio-version cfitsio-4.6.2
```

To link against a system CFITSIO instead, set:

```bash
pip install -e . --config-settings=cmake.args="-DTORCHFITS_USE_VENDORED_CFITSIO=OFF"
```

## GPU support

torchfits reads FITS data on the CPU and places the resulting tensor on the requested device. No GPU-specific build steps are required — just install PyTorch with CUDA or MPS support:

```bash
# CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# MPS (Apple Silicon) — included in the default macOS PyTorch wheel
pip install torch
```

Then pass `device="cuda"` or `device="mps"` to `torchfits.read()`.

## Development setup (pixi)

The project uses [pixi](https://pixi.sh/) for reproducible environments:

```bash
pixi install
pixi run test           # run tests
pixi run lint           # ruff lint
pixi run bench-all      # exhaustive benchmarks
```

## Optional dependencies

| Extra | Installs | Use |
|---|---|---|
| `pip install torchfits[cache]` | psutil | Adaptive cache sizing |
| `pip install torchfits[dev]` | pytest, ruff, mypy, ipykernel | Development |
| `pip install torchfits[bench]` | astropy, fitsio, pandas, matplotlib, seaborn | Benchmarking |
| `pip install torchfits[test]` | pytest, pytest-cov, pytest-benchmark | Testing |
| `pip install torchfits[examples]` | torchvision, requests, matplotlib | Running examples |

## Troubleshooting

**`ModuleNotFoundError: No module named 'torchfits.cpp'`** — The native extension did not build. Check that CMake, a C++17 compiler, and Ninja are installed. Re-run `pip install -e . -v` for verbose build output.

**`./extern/vendor.sh` fails** — Ensure `curl` or `wget` is available. If behind a proxy, set `HTTPS_PROXY`.

**`ImportError: ... symbol not found`** — Version mismatch between the compiled extension and the installed PyTorch. Rebuild: `pip install -e . --no-build-isolation --force-reinstall`.

**Slow first read** — The first call compiles internal caches. Subsequent reads are faster. Call `torchfits.configure_for_environment()` at startup for auto-tuning.
