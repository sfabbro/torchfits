# Installation

torchfits **1.0.0rc5** is built against **PyTorch 2.13.x** — the wheel ABI lane
the release ships for. The wheel's metadata pins that range
(`torch>=2.13,<2.14`), so the install command needs no torch restriction: pip
installs or upgrades torch for you.

torchfits wheels need **Python 3.10+** and **PyTorch 2.13.x**. If you already
have that minor (any flavor — CPU or CUDA), it is left untouched; an older
minor is upgraded automatically because the torch C++ ABI is per-minor. Source
builds can target **PyTorch ≥2.10** when the build frontend is installed and
the project is installed with `--no-deps`.

- **PyPI wheels** are ABI-matched to **PyTorch 2.13.x** (Linux x86_64, macOS
  arm64). No system CFITSIO — CFITSIO is vendored into the wheel.
- **Other torch minors (≥ 2.10):** build from source against the torch already
  installed (see [From source](#from-source) and [Compatibility](compatibility.md)).

## Quick install (wheels)

torchfits wheels are ABI-matched to **PyTorch 2.13.x**, and the wheel metadata
pins that range — so the default install is one bare command:

```bash
pip install torchfits
```

The recipes below show the pin explicitly; that form only matters when you
must keep a specific torch minor (pip would otherwise upgrade an older one).

### One line: full version (CUDA + CPU) — the default

```bash
pip install torchfits "torch>=2.13,<2.14"
```

This installs the default PyTorch build from PyPI, which **bundles the CUDA
runtime** on Linux x86_64 (CUDA 12.x; on other platforms the default is CPU /
MPS). It works whether or not the machine has an NVIDIA GPU:

- **With a GPU** (NVIDIA driver installed): `device="cuda"` reads work out of
  the box — `torch.cuda.is_available()` is `True`.
- **Without a GPU**: torch falls back to CPU. `torch.cuda.is_available()` is
  `False`, the CUDA libraries are never invoked, and every torchfits API
  (`read_tensor(device="cpu")`, tables, cutouts, the CLI) works unchanged.

torchfits itself ships no CUDA artifact — its C++ extension has no `.cu`
kernels. Installing the CUDA-enabled torch is what unlocks `device="cuda"`.

This also installs the `torchfits` CLI (`torchfits --help`). See the
[CLI guide](cli.md).

**Next steps:** [Quick start](quickstart.md), [Python workflows](python-workflows.md),
[Examples](examples.md), [API reference](api.md)

### One line: CPU-only (thin, no CUDA libraries)

Prefer this when you never touch NVIDIA GPUs. The **CPU** PyTorch wheel is much
thinner — it does **not** pull the CUDA runtime / cuDNN stack into the env.

Self-contained one-liner (no config needed):

```bash
pip install torchfits "torch>=2.13,<2.14" --extra-index-url https://download.pytorch.org/whl/cpu
```

Or, with the CPU index configured once (`export
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu`, or `pip config set
global.extra-index-url https://download.pytorch.org/whl/cpu` — the latter is
user-global; the env var is per-shell and a project-level `pip.conf` also
works), the deterministic extras form works:

```bash
pip install 'torchfits[cpu]'   # Linux: torch 2.13.0+cpu; macOS: no-op (MPS is the default)
```

!!! note
    zsh (the default macOS shell) treats `[cpu]` as a glob — always quote the
    extra (`'torchfits[cpu]'`) to be safe.

torchfits itself has no CUDA build artifact either way. You still get tensors
(on `cpu`), Arrow tables, cutouts, and the full CLI. Skip `device="cuda"` /
CUDA checks.

### One line: a specific CUDA build

Point pip at the CUDA toolkit you need (cu118 / cu121 / cu124 / cu126 / cu128
/ **cu129** / … — see <https://pytorch.org/get-started/locally/> for the
current matrix; this lane's extras pin cu129):

```bash
pip install torchfits "torch>=2.13,<2.14" --extra-index-url https://download.pytorch.org/whl/cu129
```

With the index configured once (`PIP_EXTRA_INDEX_URL` or `pip config set
global.extra-index-url https://download.pytorch.org/whl/cu129`):

```bash
pip install 'torchfits[cuda]'   # Linux: torch 2.13.0+cu129; macOS: no-op (MPS is the default)
```

These CUDA builds also run on machines **without** a GPU — see above.

### Why the `--extra-index-url` recipes “just work”

PyTorch’s own wheels use PEP 440 **local versions** (`2.13.0+cpu`,
`2.13.0+cu129`), which sort *above* the plain PyPI build (`2.13.0+cpu >
2.13.0`). When the PyTorch index is visible, pip therefore automatically
prefers its build for the pinned `torch>=2.13,<2.14` range — no need to choose
one index or the other: torchfits comes from PyPI, torch comes from the extra
index, in the same command. With `PIP_EXTRA_INDEX_URL` (or `pip.conf`
`extra-index-url`) set once, even a plain `pip install torchfits
"torch>=2.13,<2.14"` picks your flavor up automatically.

The `torchfits[cpu]` / `torchfits[cuda]` extras **cannot embed index URLs**, so
they resolve only when the matching PyTorch index is reachable (the one-time
setting above). Keep exactly one PyTorch index configured at a time. uv works
the same way: `uv pip install torchfits "torch>=2.13,<2.14"
--extra-index-url https://download.pytorch.org/whl/cpu` (or `UV_EXTRA_INDEX_URL`).

Apple Silicon: the default macOS torch wheel includes **MPS**, so the default
recipe is all you need; there is no macOS `+cu129` **or `+cpu`** build — both
`[cpu]` / `[cuda]` extras are Linux-only no-ops there (the exact `+local` pins
have no macOS wheel to resolve).

On first runtime use, torchfits configures cache defaults for CPU, CUDA, or MPS.

**Disk cache** (HTTP remotes + example samples only): default
`$XDG_CACHE_HOME/torchfits` or `~/.cache/torchfits`. Override with
`TORCHFITS_CACHE_DIR` — see [Environment variables](architecture.md#environment-variables)
for the full list. This is separate from the in-memory handle/
`configure_for_environment` path.

!!! tip "Verify what you got"
    ```python
    import torch

    print(
        torch.__version__, torch.version.cuda
    )  # "2.13.0" "12.9"; None means no CUDA (CPU / MPS build)
    print(torch.cuda.is_available())  # False on GPU-less machines is expected
    ```

---

## From source

Use this when you need torchfits against a **non-2.13** torch (≥ 2.10), or when
contributing.

### Prerequisites

- Python 3.10+
- C++17 compiler (GCC 10+, Clang 14+, or MSVC 2019+)
- [CMake](https://cmake.org/) 3.21+
- [Ninja](https://ninja-build.org/) (recommended)
- [PyTorch](https://pytorch.org/) **≥ 2.10** already installed (the build
  embeds that torch's major.minor as the ABI tag)
- [NumPy](https://numpy.org/) 1.20+
- `scikit-build-core` and `nanobind` (the Python build frontend)

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

    Install Visual Studio 2019+ with C++ workload, then:

    ```bash
    pip install cmake ninja
    ```

### Build

```bash
git clone https://github.com/astroai/torchfits.git
cd torchfits
./extern/vendor.sh      # download vendored CFITSIO sources

# Install the torch minor you want and the build frontend first, then build
# against that environment. Use --no-deps so pip does not replace the torch.
pip install "torch>=2.10" numpy scikit-build-core nanobind
pip install --no-deps --no-build-isolation -e .
```

For a release build:

```bash
pip install --no-deps --no-build-isolation .
```

The extension records the torch **major.minor** it was built with. Import fails
if the running torch minor differs — rebuild after upgrading torch.

The build uses [scikit-build-core](https://scikit-build-core.readthedocs.io/)
and [nanobind](https://nanobind.readthedocs.io/) for the C++ extension.

### Vendored CFITSIO

torchfits vendors CFITSIO to avoid system-library version mismatches. The
`extern/vendor.sh` script downloads the source into `extern/cfitsio/`.

Pin a specific version:

```bash
./extern/vendor.sh --cfitsio-version cfitsio-4.6.4
```

Link against system CFITSIO instead:

```bash
pip install -e . --no-deps --no-build-isolation --config-settings=cmake.args="-DTORCHFITS_USE_VENDORED_CFITSIO=OFF"
```

---

## Verify install

```python
import torchfits

print(torchfits.__version__)
_ = torchfits.read_tensor  # extension loaded
```

```bash
torchfits --help
torchfits info --help
```

---

## Development setup (pixi)

The project uses [pixi](https://pixi.sh/) for reproducible environments
(dev pixi stays on the 2.13 ABI lane for wheel parity):

```bash
pixi install
pixi run test           # run tests
pixi run lint           # ruff lint
pixi run bench-all      # exhaustive benchmarks
```

---

## Optional dependencies

| Extra | Installs | Use |
|---|---|---|
| `torchfits[dev]` | pytest, ruff, mypy, astropy, fitsio, pandas, matplotlib | Development (test + bench + examples deps) |
| `torchfits[bench]` | astropy, fitsio, pandas, matplotlib | Benchmarking |
| `torchfits[test]` | pytest, pytest-cov | Testing |
| `torchfits[examples]` | matplotlib | Running examples |
| `torchfits[cpu]` | `torch==2.13.0+cpu` (Linux; needs the PyTorch **CPU** index configured) | Thin CPU-only torch |
| `torchfits[cuda]` | `torch==2.13.0+cu129` (Linux; needs the **cu129** index configured) | CUDA-enabled torch |

The `[cpu]` / `[cuda]` extras are the flavors from
[Quick install](#quick-install-wheels): they pin an exact torch build from a
PyTorch index, so that index must be reachable via `PIP_EXTRA_INDEX_URL` or
`pip.conf` first.

Notebooks: `_repr_html_` works with any Jupyter kernel — **ipykernel is not**
a torchfits dependency.

PyArrow is a core dependency (`torchfits.table` is Arrow-native). Pandas,
Polars, and DuckDB remain optional integrations:

```bash
pip install pandas polars duckdb  # optional table interop
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torchfits._C'`** (or `'torchfits.cpp'`)

The compiled extension (`torchfits._C`) did not build. `torchfits.cpp` is a
pure-Python compatibility shim that re-exports it, so either import failure
points at the missing extension. Check that CMake, a C++17 compiler, and
Ninja are installed. Re-run with verbose output:

```bash
pip install -e . --no-build-isolation -v
```

**`./extern/vendor.sh` fails**

Ensure `curl` and `tar` are available. Behind a proxy? Set `HTTPS_PROXY`.

**`ImportError: ... ABI mismatch` / symbol not found**

The extension was built for a different torch major.minor than the one
imported. This usually happens after a bare `pip install torchfits` pulled a
newer torch than the wheel lane. First try pinning the wheel lane and
reimporting:

```bash
pip install "torch>=2.13,<2.14"
```

This release publishes only the 2.13 wheel lane. If you need a different torch
minor, rebuild torchfits from source in an environment containing that minor:

```bash
pip install --no-deps -e . --no-build-isolation --force-reinstall
```

**Slow first read**

Runtime initialization configures the environment on first API use. Rebuild
or reinstall if the imported torch minor changes.
