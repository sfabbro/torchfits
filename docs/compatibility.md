# Compatibility matrix

Supported combinations for **torchfits** wheels and source builds.

| Component | Wheels | Source builds |
|-----------|--------|----------------|
| Python | **3.10 – 3.14** | **3.10+** |
| PyTorch | **2.11.x, 2.12.x, 2.13.x** (default PyPI release tracks 2.13.x; multi-lane wheels for 2.11/2.12 available on GitHub Releases) | **≥ 2.10** when building from source with `--no-deps` |
| CUDA Flavors | **cu126, cu128, cu129, cu130** and **CPU-only (+cpu)** | all supported PyTorch CUDA builds |
| NumPy | **≥ 1.20** | same |
| PyArrow | **≥ 5.0** | same |
| Platforms | **Linux x86_64 + aarch64**, **macOS arm64** | git checkout + `extern/vendor.sh` |

Optional: Polars / Pandas / DuckDB via env; astropy / fitsio for test / bench only
(not imported by runtime I/O).

## Release lanes & multi-lane wheels

Each torchfits wheel is compiled against **one PyTorch minor** — the wheel "lane" —
because PyTorch has no stable C++ ABI across minors; the C++ extension embeds the
torch ABI tag it was built with and refuses to import under a different minor.

| PyTorch lane | Wheel availability | Notes |
|---|---|---|
| **2.13.x** | **Default PyPI release** (`torchfits 1.0.0`) | Default `pip install torchfits` line |
| **2.12.x** | **GitHub Releases / Wheel Index** (`1.0.0+torch212`) | Prebuilt wheel for PyTorch 2.12 environments |
| **2.11.x** | **GitHub Releases / Wheel Index** (`1.0.0+torch211`) | Prebuilt wheel for PyTorch 2.11 environments |

The release lane map lives in `scripts/torch_lanes.json` (verified via
`pixi run check-lane`).

## Wheels vs source

- **PyPI wheels** are compiled against PyTorch 2.13 (CPython 3.10–3.14). Plain
  `pip install torchfits` automatically installs or upgrades torch to 2.13.
  There is no sdist on PyPI, so pip never compiles.
- **Prebuilt wheels for PyTorch 2.11 / 2.12:** install the corresponding lane
  wheel directly from GitHub Releases or via `--find-links` to run on PyTorch 2.11
  or 2.12 without upgrading torch or compiling.
- **Other PyTorch minors (≥ 2.10):** from a git checkout, pre-install that
  torch and the build frontend (`scikit-build-core`, `nanobind`, CMake, Ninja,
  C++17 compiler, and NumPy), then build with `pip install --no-deps
  --no-build-isolation .` so pip does not replace the selected torch minor.
- **CUDA / CPU-only installs:** a single CPU-linked torchfits wheel works across
  all CUDA flavors (`cu126`, `cu128`, `cu129`, `cu130`) as well as CPU-only
  (`+cpu`) builds of that PyTorch lane (see [Install](install.md)); CUDA builds
  also run seamlessly on GPU-less machines via CPU fallback.

## Downstream guidance

| Consumer | Guidance |
|----------|----------|
| ML training (`Dataset` / `make_loader`) | Prefer wheel + the matching torch lane, or rebuild torchfits from source for your torch minor |
| Arrow catalogs | Prefer `torchfits.table.read` / `scan` |
| Tensor columns | Prefer `torchfits.table.read_torch` / `scan_torch` |
| Root `read_table` / `stream_table` / `read_table_rows` / `get_header` / `get_batch_info` | **Removed** in 1.0 — use the explicit mappings in [API Reference](api.md#removed-names), plus `read_header` / `read_batch_info` |

## Wheel install smoke

```bash
bash scripts/clean_install_smoke.sh
```

CI publishes the cibuildwheel matrix on tagged releases
(`.github/workflows/build_wheels.yml`); local builds go through
`scripts/build_wheels_local.sh` (per-lane, per-Python matrix) and are checked
with `scripts/verify_wheel_matrix.sh`.
