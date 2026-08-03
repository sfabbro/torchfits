# Compatibility matrix

Supported combinations for **torchfits** wheels and source builds.

| Component | Wheels | Source builds |
|-----------|--------|----------------|
| Python | **3.10 – 3.14** | **3.10+** |
| PyTorch | **the ABI lane the wheel ships for** (2.13.x on the current release) | **≥ 2.10** (build against the torch already installed) |
| NumPy | **≥ 1.20** | same |
| PyArrow | **≥ 5.0** | same |
| Platforms | **Linux x86_64**, **macOS arm64** | other arches via source |

Optional: Polars / Pandas / DuckDB via env; astropy / fitsio for test / bench only
(not imported by runtime I/O).

## Release lanes

Each torchfits release targets **one PyTorch minor** — the wheel "lane" —
because PyTorch has no stable C++ ABI across minors; the extension embeds the
torch major.minor it was built with and refuses to import under a different
minor.

| PyTorch lane | torchfits release |
|---|---|
| **2.13.x** | **1.0.0** (current) |

Install the pair together, e.g. `pip install torchfits "torch>=2.13,<2.14"` —
see [Install](install.md). The lane map lives in `scripts/torch_lanes.json`
(single source of truth; `pixi run check-lane` verifies all files agree).
Additional lanes are added only when a real backport release is cut, with that
release's actual version.

## Wheels vs source

- **PyPI wheels** are compiled against the lane's PyTorch minor. Install the
  matching torch first (`torch>=2.13,<2.14`), then `pip install torchfits`.
- **Other PyTorch minors (≥ 2.10):** pre-install that torch (and build tools —
  CMake, Ninja, C++17 compiler, NumPy), then build from source with
  `pip install --no-build-isolation .`.
- **CUDA / CPU-only installs:** install recipes pin the wheel ABI lane
  (`torch>=2.13,<2.14`). `torchfits[cpu]` / `torchfits[cuda]` extras and the
  `--extra-index-url` one-liners select the PyTorch build (see
  [Install](install.md)); CUDA builds also run on GPU-less machines via CPU
  fallback.

## Downstream guidance

| Consumer | Guidance |
|----------|----------|
| ML training (`Dataset` / `make_loader`) | Prefer wheel + the matching torch lane, or rebuild torchfits from source for your torch minor |
| Arrow catalogs | Prefer `torchfits.table.read` / `scan` |
| Tensor columns | Prefer `torchfits.table.read_torch` / `scan_torch` |
| Root `read_table` / `stream_table` / `read_table_rows` / `get_header` / `get_batch_info` | **Removed** in 1.0 — use the replacements above / `read_header` / `read_batch_info` |

## Wheel install smoke

```bash
bash scripts/clean_install_smoke.sh
```

CI publishes the cibuildwheel matrix on tagged releases
(`.github/workflows/build_wheels.yml`); local builds go through
`scripts/build_wheels_local.sh` (per-lane, per-Python matrix) and are checked
with `scripts/verify_wheel_matrix.sh`.
