# Installation

`torchfits` is distributed as prebuilt binary wheels for **Linux (x86_64, aarch64)** and **macOS (Apple Silicon arm64)** supporting **Python 3.10–3.14**. CFITSIO is vendored directly into the wheels, so no C++ compiler, system libraries, or manual build steps are required.

## Quick install

### 1. Standard install (CUDA + CPU auto-detection) — Recommended

For standard environments (Linux with NVIDIA GPU, CPU-only Linux, or macOS Apple Silicon):

```bash
pip install torchfits
```

- **On Linux with an NVIDIA GPU:** `device="cuda"` copies the host-decoded tensor onto CUDA.
- **On Linux without a GPU:** falls back to CPU automatically (`device="cpu"`).
- **On macOS (Apple Silicon):** `device="mps"` copies the host-decoded tensor onto Metal.

### 2. CPU-only install (Minimal footprint)

On headless servers, CI runners, or containers where CUDA runtime libraries are not needed, use the lightweight CPU index with `[cpu]` extra:

```bash
pip install "torchfits[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu
# Or explicitly pinning the CPU PyTorch build:
pip install torchfits "torch==2.13.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
```

### 3. Specific CUDA toolkit versions

To match a specific CUDA version installed on your system (e.g. `cu126`, `cu129`, `cu130`):

```bash
pip install "torchfits[cuda]" --extra-index-url https://download.pytorch.org/whl/cu129
# Or explicitly pinning the CUDA PyTorch build:
pip install torchfits "torch==2.13.0+cu129" --extra-index-url https://download.pytorch.org/whl/cu129
```

---

## PyTorch Version Compatibility

The PyPI release is built against one PyTorch ABI lane at a time; the current lane is **PyTorch 2.13.x** (`torch>=2.13,<2.14`). Wheels are ABI-matched to that minor, so keep your PyTorch on 2.13.x.

If you must stay on an older PyTorch (**≥ 2.10**), install from source against your environment (see below) — prebuilt wheels are not published for other lanes.

---

## Verify your installation

```python
import torchfits
import torch

print(f"torchfits {torchfits.__version__} with torch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

```bash
torchfits info --help
```

---

## Building from source

Builds link libbz2 when available (conda prefix or system via CMake), which enables `BZIP2_1` tile compression and transparent reading of whole-file `.bz2` FITS (`torchfits._C.HAS_BZIP2` reports the capability). Without it, `.bz2` inputs raise an actionable error.

Building from source is only needed if you are developing torchfits or targeting a custom PyTorch build (≥ 2.10).

### Prerequisites
- Python 3.10+
- C++17 compiler (GCC 10+, Clang 14+, MSVC 2019+)
- [CMake](https://cmake.org/) 3.21+ and [Ninja](https://ninja-build.org/)
- [PyTorch](https://pytorch.org/) ≥ 2.10 and [NumPy](https://numpy.org/)

=== "Linux"

    ```bash
    sudo apt install build-essential cmake ninja-build
    ```

=== "macOS"

    ```bash
    xcode-select --install
    brew install cmake ninja
    ```

=== "Windows"

    Windows is **unsupported**. There are no wheels, and the native extension
    is not tested with MSVC. Use WSL2, Linux, or macOS.

### Build steps

```bash
git clone https://github.com/astroai/torchfits.git
cd torchfits
./extern/vendor.sh      # downloads vendored CFITSIO

# Install build dependencies and compile against your environment
pip install numpy scikit-build-core nanobind
pip install --no-deps --no-build-isolation -e .
```

To link against a system-installed CFITSIO instead of the vendored source:

```bash
pip install -e . --no-deps --no-build-isolation --config-settings=cmake.args="-DTORCHFITS_USE_VENDORED_CFITSIO=OFF"
```

---

## Development setup (Pixi)

For local development and running the test suite, we recommend [pixi](https://pixi.sh/):

```bash
pixi install
pixi run test           # run test suite
pixi run lint           # ruff linter and formatter
pixi run bench-all      # benchmarks
```

---

## Optional dependencies

| Extra | Included packages | Purpose |
|---|---|---|
| `torchfits[dev]` | pytest, ruff, mypy, astropy, fitsio, pandas, matplotlib | Full development suite |
| `torchfits[bench]` | astropy, fitsio, pandas, matplotlib | Benchmarking suite |
| `torchfits[test]` | pytest, pytest-cov, astropy, fitsio, psutil, pyarrow | Unit testing extras (`pip install torchfits[test]`) |
| `torchfits[examples]` | matplotlib | Running tutorial scripts |

PyArrow is installed automatically for tabular operations (`torchfits.table`). Interoperability with [Pandas](https://pandas.pydata.org/), [Polars](https://pola.rs/), and [DuckDB](https://duckdb.org/) is supported seamlessly:

```bash
pip install pandas polars duckdb
```

---

## Troubleshooting

**`OMP: Error #15: Initializing libomp.dylib`**

Two OpenMP libraries are loaded in the same process — typical on macOS when PyTorch's `libomp` meets Homebrew or conda `llvm-openmp`. Importing `torchfits` sets `KMP_DUPLICATE_LIB_OK=TRUE` before PyTorch loads. Import `torchfits` first, or export that variable in the shell before `import torch`:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

**`No matching distribution found for torchfits`**

Prebuilt binary wheels are available for Linux (x86_64, aarch64) and macOS (Apple Silicon arm64) on CPython 3.10–3.14. Windows and x86_64 macOS are unsupported (no wheels, no tested MSVC recipe).

**`ImportError: ... ABI mismatch`**

The binary extension was built against a different PyTorch minor version than the one currently active in Python. Either install the matching prebuilt wheel for your PyTorch version or rebuild from source.

**`./extern/vendor.sh fails`**

Ensure `curl` and `tar` are installed and reachable. If behind a proxy, set `HTTPS_PROXY`.
