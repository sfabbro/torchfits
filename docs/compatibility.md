# Compatibility matrix

Supported combinations for **torchfits** wheels and source builds.

| Component | Wheels | Source builds |
|-----------|--------|----------------|
| Python | **3.10 – 3.13** | **3.10+** |
| PyTorch | **2.10.x** (ABI-matched) | **≥ 2.10** (build against the torch already installed) |
| NumPy | **≥ 1.20** | same |
| PyArrow | **≥ 5.0** | same |
| Platforms | **Linux x86_64**, **macOS arm64** | other arches via source |

Optional: Polars / Pandas / DuckDB via env; astropy / fitsio for test / bench only
(not imported by runtime I/O).

## Wheels vs source

- **PyPI wheels** are compiled against **PyTorch 2.10**. Install matching torch
  first (`torch>=2.10,<2.11`), then `pip install torchfits`.
- **Other PyTorch minors (≥ 2.10):** pre-install that torch (and build tools —
  CMake, Ninja, C++17 compiler, NumPy), then build from source with
  `pip install --no-build-isolation .`. The extension embeds the torch
  major.minor it was built with and refuses to import under a different minor.
- **CUDA / CPU-only installs:** install recipes pin `torch>=2.10,<2.11` (the
  wheel ABI lane). `torchfits[cpu]` / `torchfits[cuda]` extras and the
  `--extra-index-url` one-liners select the PyTorch build (see
  [Install](install.md)); CUDA builds also run on GPU-less machines via CPU
  fallback.

## Downstream guidance

| Consumer | Guidance |
|----------|----------|
| ML training (`Dataset` / `make_loader`) | Prefer wheel + PyTorch 2.10, or rebuild torchfits from source for your torch minor |
| Arrow catalogs | Prefer `torchfits.table.read` / `scan` |
| Tensor columns | Prefer `torchfits.table.read_torch` / `scan_torch` |
| Root `read_table` / `stream_table` / `read_table_rows` / `get_header` / `get_batch_info` | **Removed** in 1.0 — use the replacements above / `read_header` / `read_batch_info` |

## Wheel install smoke

```bash
bash scripts/clean_install_smoke.sh
```

CI publishes the cibuildwheel matrix on tagged releases
(`.github/workflows/build_wheels.yml`).
