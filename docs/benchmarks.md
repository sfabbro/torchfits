# Benchmarks

`torchfits` benchmarks cover FITS image I/O and FITS table I/O. WCS, HEALPix,
sphere, and sky-domain benchmarks are out of scope for this repository.

## Comparison Targets

| Domain | torchfits module | Compared against |
|---|---|---|
| FITS image I/O | `torchfits.read` / `torchfits.write` | `astropy.io.fits`, `fitsio` |
| FITS table I/O | `torchfits.table` | `astropy.io.fits`, `fitsio` |

## Methodology

Each case measures median wall-clock time over multiple repetitions. Cases are
grouped into two families:

- **smart** — the idiomatic high-level API, such as `torchfits.read()` vs
  `astropy.io.fits.getdata()` plus `torch.from_numpy()`.
- **specialized** — lower-level paths with explicit mmap, compression, or table
  streaming controls.

Fairness controls:

- Rows with mismatched mmap behavior are marked `SKIPPED` and excluded from
  rankings.
- FITS comparators must be official released distributions.
- Warm-cache and cold-cache profiles are kept separate.

## Correctness Gates

| Gate | Command | Validates |
|---|---|---|
| fitsio parity | `pixi run pytest tests/test_fitsio_upstream_smoke.py -q` | Common fitsio image, header, table, compression, and checksum workflows |
| Astropy parity | `pixi run pytest tests/test_astropy_upstream_smoke.py -q` | Common Astropy HDU, header, image, compressed-image, table, and scaled-data workflows |
| Package isolation | `pixi run pytest tests/test_package_isolation.py tests/test_docs_integrity.py -q` | Clean FITS-only package boundary and docs contract |

## Reproducing

```bash
pixi run bench-fits
pixi run bench-fitstable
pixi run bench-all
# Full transport matrix (mmap on + off, doubles CPU rows; GPU rows for both when CUDA/MPS):
pixi run -e bench-gpu python benchmarks/bench_all.py --profile lab --scope all --mmap-matrix
```

For focused FITS partitions:

```bash
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(tiny_)'
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(small_)'
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(medium_|large_)'
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(scaled_|compressed_|mef_)'
```

## Benchmark Scripts

| Script | Domain | Description |
|---|---|---|
| `bench_all.py` | fits / fitstable | FITS benchmark orchestrator |
| `bench_fits_io.py` | fits | Image I/O across dtypes, sizes, compression, scaling, MEF, and cutouts |
| `bench_fitstable_io.py` | fitstable | Table I/O across row counts, schemas, projection, row slicing, predicates, and streaming |
| `bench_fast.py` | fits | Low-level image/header fast-path checks |
| `bench_table.py` | fitstable | Table API timing |
| `bench_arrow_tables.py` | fitstable | Arrow-oriented table workflows |
| `bench_gpu_transports.py` | fits (GPU) | CUDA/MPS image reads, cutouts, repeated cutouts (`disk→CPU→GPU` / `disk→RAM→GPU` rows) |
| `bench_ml_loader.py` | fits (diagnostic) | PyTorch `DataLoader` throughput (not merged into `bench-all` CSV) |
| `bench_gpu_memory.py` | fits (diagnostic) | GPU memory/leak checks (non-gating) |

## Coverage matrix

What the exhaustive `bench-all` suite measures today, and what is intentionally out of
scope or not yet wired into the published tables.

| Dimension | Covered? | Where | Gap / caveat |
|---|---|---|---|
| Backends (torchfits / astropy / fitsio) | Yes | `bench_fits_io.py`, `bench_fitstable_io.py` | `fitsio` often excluded from mmap-fairness summaries; **uint** image comparators may be torchfits-only when astropy requires buffered fallback |
| CPU vs GPU device | Partial | CPU: full matrix; GPU: image reads only | GPU requires CUDA/MPS hardware (`pixi run -e bench-gpu bench-gpu` or local CUDA); **CI weekly bench is CPU-only** |
| I/O transport `disk→RAM→CPU` | Yes | `bench-all` mmap-on pass | Median mixes many ops/sizes — coarse aggregate |
| I/O transport `disk→CPU` (non-mmap) | Yes | `bench-all --mmap-matrix` mmap-off pass | Buffered host decode; use `--mmap-matrix` (or `--no-mmap`) to populate |
| I/O transport `disk→RAM→GPU` | Partial | `bench_gpu_transports.py` (mmap on) | Image `read_full`, cutouts, repeated cutouts only; **no tables** |
| I/O transport `disk→CPU→GPU` | Partial | `bench_gpu_transports.py` (mmap off) | Same GPU ops with buffered host decode + H2D copy |
| I/O transport `disk→GPU` | No | — | No Python FITS backend supports true disk→GPU (GPUDirect / cuFile); row stays empty by design |
| BITPIX / dtypes | Partial | int8–int64, float32/64 × 1D/2D/3D | Native **uint16/uint32** 2D fixtures (`small/medium/large_uint*_2d`); unsigned via BZERO also in `scaled_*` |
| Image dimensions / sizes | Yes | tiny → large categories | Large 3D cubes skipped (size cap) |
| Compression | Yes | gzip, rice, hcompress, plio | Write-side compression not benchmarked |
| Scaling (BSCALE/BZERO) | Yes | `scaled_small/medium/large` | Table-column scaling not isolated |
| Random / repeated access | Yes | cutouts, `random_ext_full_reads_200`, `open_subset_reader` repeated cutouts | MEF random ext reads only on selected fixtures |
| Multi-extension (MEF) | Yes | `mef_*`, `multi_mef_10ext` | — |
| Table full read / projection / slice | Yes | `bench_fitstable_io.py` | — |
| Table predicate / scan | Yes | `predicate_filter`, `scan_count` | Arrow `table.scan` streaming not identical to `scan_count` row |
| Table schemas | Partial | mixed / narrow / wide / varlen | **typed** (BIT/complex/string) and **ascii** table fixtures at selected row counts |
| Table GPU | No | — | All comparators are CPU-resident; not a meaningful apples-to-apples GPU row today |
| Writes | No | — | Read-heavy suite; write parity validated in tests, not bench CSV |
| FITS physical units (BUNIT/TUNIT) | No | — | Metadata semantics, not I/O transport — covered by parity tests only |
| ML DataLoader pattern | Diagnostic | `bench_ml_loader.py` | Not merged into `docs/benchmarks.md` tables |

### Why the I/O transport table looks sparse on GPU

1. **`disk→GPU` is always empty** — every backend decodes on the host first (CFITSIO /
   astropy / fitsio into host RAM), then copies with `.to(device)`. `device="cuda"` does
   **not** mean a native disk→GPU bypass (that would require GPUDirect Storage / cuFile,
   which none of these Python FITS stacks use).
2. **`disk→CPU→GPU` vs `disk→RAM→GPU`** — the former is the mmap-off GPU path (buffered
   host decode + H2D); the latter is mmap-on decode + H2D. Both still touch host memory.
3. **`disk→RAM→GPU` is populated only when GPU rows exist in the CSV** — produced by
   `bench_gpu_transports.py` inside `bench-all` when `torch.cuda.is_available()` or MPS
   is available. GitHub Actions `bench-report` installs **CPU PyTorch**, so weekly CI
   runs will **not** refresh GPU cells; the published CUDA numbers come from a manual
   lab run (`exhaustive_mmap_0.5.0b3_20260630_063118`, via `pixi run -e bench-gpu bench-exhaustive`).
4. **FITS tables have no GPU transport rows** — astropy/fitsio/torchfits table paths are
   CPU-buffered; GPU table benchmarks would mostly measure PyTorch copy overhead, not FITS
   decode, and are deliberately omitted.

### Refreshing GPU numbers

```bash
# Linux + NVIDIA
pixi run -e bench-gpu bench-gpu

# Apple Silicon (MPS transport rows; separate from CUDA lab numbers)
pixi run bench-mps

# Re-render docs from the merged CSV
pixi run -e bench-gpu bench-exhaustive
# or, from an existing run directory:
pixi run bench-table-render -- --csv benchmarks_results/<run-id>/results.csv
python scripts/patch_bench_docs.py --csv ... --deficits ... --run-id <run-id>
```

## I/O Transport × Backend

> **GPU summary:** Image **`disk→CPU→GPU`** and **`disk→RAM→GPU`** rows appear only when the benchmark CSV was
> produced on CUDA or MPS hardware. **`disk→GPU`** is intentionally empty (unsupported by
> all backends). **Table GPU transports are not benchmarked.** CI weekly `bench-report`
> uses CPU PyTorch and will not update GPU cells.


<!-- BENCH_IOPATH_BEGIN -->
Source: `benchmarks_results/exhaustive_mmap_0.5.0b3_20260630_063118/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.15 ms` (n=269) | `0.77 ms` (n=269) | — | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.15 ms` (n=269) | `0.67 ms` (n=219) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.21 ms` (n=180) | `0.83 ms` (n=90) | `0.28 ms` (n=90) | — |
| `disk→RAM→GPU` | `0.18 ms` (n=180) | `1.03 ms` (n=90) | `0.24 ms` (n=90) | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.10 ms` (n=180) | `3.28 ms` (n=162) | — | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.10 ms` (n=180) | `3.06 ms` (n=162) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |
<!-- BENCH_IOPATH_END -->

### Notes on the layout

- Rows are **I/O transports** (`disk→CPU`, `disk→RAM→CPU`, `disk→GPU`,
  `disk→CPU→GPU`, `disk→RAM→GPU`).
- Columns are **backends** (`torchfits` / `astropy` / `fitsio` / `cfitsio-direct`).
- `cfitsio` is the C engine used by `torchfits`; no standalone `cfitsio`-only
  benchmark row is generated by `bench-all`, so the cell is documented as
  "engine exposed under `torchfits`".
- Cell `n=` counts comparable OK rows in the bucket; `—` indicates the
  bucket is empty (no rows match, or rows were excluded under
  `strict_mmap_fairness` in the original `bench-all` summary).
- Median is computed over heterogeneous operations (`read_full`,
  `cutout_100x100`, `header_read`, `predicate_filter`, `projection`,
  `row_slice`, etc.) and payload sizes; treat the per-cell ms as a
  coarse representative number, not a precise benchmark.

## Performance Highlights

<!-- BENCH_HIGHLIGHTS_BEGIN -->
The following table showcases median wall-clock execution times of key representative FITS benchmarks.
In almost all core I/O paths, `torchfits` is significantly faster than standard astronomical tools, with extra performance wins from persistent handle caches and direct-to-device transfers.

| Benchmark Case | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Win vs Astropy | Win vs fitsio |
|---|---|---:|---:|---:|---:|---:|---:|
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **4.26 ms** | 4.19 ms | 15.19 ms | — | **3.63x** | **—** |
| Large Image Read (Float32 2D @ CUDA) | CUDA | **3.61 ms** | 3.53 ms | 16.16 ms | 5.57 ms | **4.58x** | **1.58x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **9.20 ms** | 9.08 ms | 29.33 ms | — | **3.23x** | **—** |
| Compressed Image Read (Rice @ CUDA) | CUDA | **8.97 ms** | 9.00 ms | 28.97 ms | 9.24 ms | **3.23x** | **1.03x** |
| Repeated Cutouts (50x 100x100) | CPU | **5.53 ms** | 5.21 ms | 80.93 ms | — | **15.53x** | **—** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **93.4 μs** | 99.4 μs | 6.33 ms | — | **67.79x** | **—** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **94.3 μs** | 101.1 μs | 3.39 ms | — | **35.94x** | **—** |
<!-- BENCH_HIGHLIGHTS_END -->

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured benchmark configurations.

| Domain | Benchmark Case | Operation | Size | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | **—** | 165.2 μs | 2.14 ms | — | **12.94x** | **—** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | **16.16 ms** | 16.11 ms | 41.19 ms | — | **2.56x** | **—** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | **—** | 165.2 μs | 2.15 ms | — | **13.00x** | **—** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | **15.69 ms** | 15.65 ms | 67.51 ms | — | **4.32x** | **—** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | **—** | 177.7 μs | 2.26 ms | — | **12.70x** | **—** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | **30.63 ms** | 30.60 ms | 36.86 ms | — | **1.20x** | **—** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | **943.2 μs** | 937.8 μs | 10.64 ms | — | **11.34x** | **—** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | **—** | 174.5 μs | 2.25 ms | — | **12.89x** | **—** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | **9.20 ms** | 9.08 ms | 29.33 ms | — | **3.23x** | **—** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | **—** | 88.0 μs | 613.0 μs | — | **6.97x** | **—** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | **1.06 ms** | 1.00 ms | 2.21 ms | — | **2.20x** | **—** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | **—** | 92.1 μs | 649.7 μs | — | **7.05x** | **—** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | **4.26 ms** | 4.19 ms | 15.19 ms | — | **3.63x** | **—** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | **—** | 91.1 μs | 616.8 μs | — | **6.77x** | **—** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | **1.97 ms** | 1.97 ms | 3.83 ms | — | **1.95x** | **—** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | **—** | 92.6 μs | 651.7 μs | — | **7.03x** | **—** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | **10.91 ms** | 10.87 ms | 24.24 ms | — | **2.23x** | **—** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | **—** | 86.5 μs | 612.3 μs | — | **7.08x** | **—** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | **583.0 μs** | 588.1 μs | 1.43 ms | — | **2.44x** | **—** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | **—** | 89.8 μs | 653.6 μs | — | **7.28x** | **—** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | **2.14 ms** | 2.13 ms | 4.62 ms | — | **2.17x** | **—** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | **—** | 87.9 μs | 609.8 μs | — | **6.94x** | **—** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | **997.5 μs** | 1.00 ms | 2.20 ms | — | **2.21x** | **—** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | **—** | 91.8 μs | 646.6 μs | — | **7.05x** | **—** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | **4.05 ms** | 4.04 ms | 14.95 ms | — | **3.70x** | **—** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | **—** | 88.0 μs | 607.6 μs | — | **6.91x** | **—** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | **1.97 ms** | 1.97 ms | 3.83 ms | — | **1.94x** | **—** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | **—** | 89.7 μs | 643.7 μs | — | **7.18x** | **—** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | **10.74 ms** | 10.64 ms | 23.85 ms | — | **2.24x** | **—** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | **—** | 92.9 μs | 673.1 μs | — | **7.25x** | **—** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | **357.4 μs** | 356.3 μs | 1.24 ms | — | **3.47x** | **—** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | **—** | 97.9 μs | 714.9 μs | — | **7.30x** | **—** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | **1.03 ms** | 1.04 ms | 2.56 ms | — | **2.47x** | **—** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | **—** | 96.1 μs | 706.8 μs | — | **7.35x** | **—** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | **8.56 ms** | 5.01 ms | 6.46 ms | — | **1.29x** | **—** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | **—** | 96.7 μs | 713.2 μs | — | **7.38x** | **—** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | **14.53 ms** | 7.38 ms | 10.30 ms | — | **1.39x** | **—** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | **—** | 86.4 μs | 622.9 μs | — | **7.21x** | **—** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | **249.0 μs** | 211.7 μs | 789.8 μs | — | **3.73x** | **—** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | **—** | 92.1 μs | 653.2 μs | — | **7.09x** | **—** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | **1.11 ms** | 1.06 ms | 2.33 ms | — | **2.18x** | **—** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | **—** | 95.0 μs | 680.3 μs | — | **7.16x** | **—** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | **1.68 ms** | 1.62 ms | 3.38 ms | — | **2.09x** | **—** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | **—** | 88.6 μs | 625.1 μs | — | **7.05x** | **—** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | **302.8 μs** | 307.2 μs | 967.2 μs | — | **3.19x** | **—** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | **—** | 92.3 μs | 649.1 μs | — | **7.03x** | **—** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | **2.13 ms** | 2.09 ms | 4.56 ms | — | **2.19x** | **—** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | **—** | 94.9 μs | 671.5 μs | — | **7.07x** | **—** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | **3.19 ms** | 3.17 ms | 5.95 ms | — | **1.87x** | **—** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | **—** | 90.4 μs | 616.6 μs | — | **6.82x** | **—** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | **170.3 μs** | 176.3 μs | 709.7 μs | — | **4.17x** | **—** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | **—** | 91.6 μs | 647.0 μs | — | **7.06x** | **—** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | **663.4 μs** | 663.4 μs | 1.53 ms | — | **2.31x** | **—** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | **—** | 93.5 μs | 674.7 μs | — | **7.22x** | **—** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | **948.1 μs** | 954.6 μs | 2.03 ms | — | **2.14x** | **—** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | **—** | 87.0 μs | 616.1 μs | — | **7.08x** | **—** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | **209.3 μs** | 214.2 μs | 799.4 μs | — | **3.82x** | **—** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | **—** | 91.6 μs | 646.5 μs | — | **7.06x** | **—** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | **1.10 ms** | 1.11 ms | 2.41 ms | — | **2.19x** | **—** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | **—** | 91.8 μs | 670.5 μs | — | **7.30x** | **—** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | **1.70 ms** | 1.71 ms | 3.46 ms | — | **2.03x** | **—** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | **—** | 86.4 μs | 604.7 μs | — | **7.00x** | **—** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | **322.4 μs** | 317.5 μs | 991.4 μs | — | **3.12x** | **—** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | **—** | 90.9 μs | 646.9 μs | — | **7.11x** | **—** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | **2.13 ms** | 2.13 ms | 4.62 ms | — | **2.17x** | **—** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | **—** | 93.5 μs | 670.7 μs | — | **7.17x** | **—** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | **3.29 ms** | 3.29 ms | 6.11 ms | — | **1.86x** | **—** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | **—** | 94.6 μs | 680.6 μs | — | **7.19x** | **—** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | **145.2 μs** | 158.9 μs | 816.9 μs | — | **5.63x** | **—** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | **—** | 96.6 μs | 711.0 μs | — | **7.36x** | **—** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | **388.6 μs** | 401.2 μs | 1.31 ms | — | **3.38x** | **—** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | **—** | 98.3 μs | 739.9 μs | — | **7.53x** | **—** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | **525.2 μs** | 539.6 μs | 1.59 ms | — | **3.04x** | **—** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | **—** | 94.9 μs | 707.3 μs | — | **7.45x** | **—** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | **1.87 ms** | 972.0 μs | 2.11 ms | — | **2.17x** | **—** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | **—** | 95.5 μs | 706.5 μs | — | **7.40x** | **—** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | **2.65 ms** | 1.49 ms | 2.92 ms | — | **1.96x** | **—** |
| fits | mef_medium | header_read | 7.02 MB | CPU | **—** | 104.8 μs | 992.7 μs | — | **9.47x** | **—** |
| fits | mef_medium | read_full | 7.02 MB | CPU | **353.5 μs** | 365.7 μs | 1.54 ms | — | **4.36x** | **—** |
| fits | mef_small | header_read | 0.45 MB | CPU | **—** | 105.2 μs | 997.5 μs | — | **9.48x** | **—** |
| fits | mef_small | read_full | 0.45 MB | CPU | **139.9 μs** | 152.6 μs | 1.09 ms | — | **7.80x** | **—** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | **105.4 μs** | 103.3 μs | 3.52 ms | — | **34.03x** | **—** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | **—** | 106.7 μs | 994.6 μs | — | **9.32x** | **—** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | **6.82 ms** | 6.82 ms | 10.92 ms | — | **1.60x** | **—** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | **138.8 μs** | 153.8 μs | 1.09 ms | — | **7.85x** | **—** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | **5.53 ms** | 5.21 ms | 80.93 ms | — | **15.53x** | **—** |
| fits | scaled_large | header_read | 8.00 MB | CPU | **—** | 97.2 μs | 718.0 μs | — | **7.39x** | **—** |
| fits | scaled_large | read_full | 8.00 MB | CPU | **3.70 ms** | 3.64 ms | 7.54 ms | — | **2.07x** | **—** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | **—** | 97.7 μs | 711.7 μs | — | **7.28x** | **—** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | **1.06 ms** | 1.02 ms | 2.46 ms | — | **2.40x** | **—** |
| fits | scaled_small | header_read | 0.13 MB | CPU | **—** | 97.5 μs | 720.2 μs | — | **7.38x** | **—** |
| fits | scaled_small | read_full | 0.13 MB | CPU | **222.8 μs** | 195.4 μs | 918.9 μs | — | **4.70x** | **—** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | **—** | 86.9 μs | 610.6 μs | — | **7.02x** | **—** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | **171.2 μs** | 133.9 μs | 639.7 μs | — | **4.78x** | **—** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | **—** | 89.9 μs | 652.3 μs | — | **7.26x** | **—** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | **222.5 μs** | 188.0 μs | 752.6 μs | — | **4.00x** | **—** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | **—** | 96.5 μs | 677.6 μs | — | **7.02x** | **—** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | **320.5 μs** | 281.1 μs | 961.4 μs | — | **3.42x** | **—** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | **—** | 87.2 μs | 613.6 μs | — | **7.04x** | **—** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | **139.1 μs** | 142.7 μs | 659.2 μs | — | **4.74x** | **—** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | **—** | 88.3 μs | 648.8 μs | — | **7.34x** | **—** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | **245.8 μs** | 247.9 μs | 873.6 μs | — | **3.55x** | **—** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | **—** | 91.6 μs | 677.9 μs | — | **7.40x** | **—** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | **445.9 μs** | 449.7 μs | 1.24 ms | — | **2.77x** | **—** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | **—** | 87.1 μs | 614.0 μs | — | **7.05x** | **—** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | **130.0 μs** | 134.0 μs | 625.3 μs | — | **4.81x** | **—** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | **—** | 89.1 μs | 649.3 μs | — | **7.29x** | **—** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | **152.9 μs** | 159.3 μs | 695.4 μs | — | **4.55x** | **—** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | **—** | 91.8 μs | 673.2 μs | — | **7.33x** | **—** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | **204.5 μs** | 206.2 μs | 806.1 μs | — | **3.94x** | **—** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | **—** | 86.2 μs | 612.4 μs | — | **7.10x** | **—** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | **132.7 μs** | 135.5 μs | 634.0 μs | — | **4.78x** | **—** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | **—** | 89.8 μs | 639.4 μs | — | **7.12x** | **—** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | **182.8 μs** | 188.5 μs | 747.5 μs | — | **4.09x** | **—** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | **—** | 93.9 μs | 681.0 μs | — | **7.26x** | **—** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | **275.1 μs** | 278.4 μs | 958.4 μs | — | **3.48x** | **—** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | **—** | 86.8 μs | 607.7 μs | — | **7.00x** | **—** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | **136.6 μs** | 142.9 μs | 649.5 μs | — | **4.76x** | **—** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | **—** | 89.5 μs | 659.4 μs | — | **7.37x** | **—** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | **242.4 μs** | 250.2 μs | 879.3 μs | — | **3.63x** | **—** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | **—** | 93.3 μs | 672.4 μs | — | **7.21x** | **—** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | **441.7 μs** | 440.8 μs | 1.25 ms | — | **2.84x** | **—** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | **—** | 94.8 μs | 680.6 μs | — | **7.18x** | **—** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | **107.7 μs** | 115.6 μs | 782.8 μs | — | **7.27x** | **—** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | **—** | 95.2 μs | 702.3 μs | — | **7.38x** | **—** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | **137.2 μs** | 148.4 μs | 819.4 μs | — | **5.97x** | **—** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | **—** | 97.7 μs | 739.9 μs | — | **7.57x** | **—** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | **163.1 μs** | 177.0 μs | 866.1 μs | — | **5.31x** | **—** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | **—** | 95.4 μs | 702.3 μs | — | **7.36x** | **—** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | **414.9 μs** | 323.0 μs | 795.4 μs | — | **2.46x** | **—** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | **—** | 96.7 μs | 709.0 μs | — | **7.34x** | **—** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | **462.3 μs** | 358.9 μs | 837.8 μs | — | **2.33x** | **—** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | **—** | 91.1 μs | 642.6 μs | — | **7.05x** | **—** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | **216.9 μs** | 182.6 μs | 727.0 μs | — | **3.98x** | **—** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | **—** | 89.5 μs | 645.4 μs | — | **7.21x** | **—** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | **220.3 μs** | 184.1 μs | 735.9 μs | — | **4.00x** | **—** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | **—** | 92.0 μs | 640.4 μs | — | **6.96x** | **—** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | **217.0 μs** | 185.2 μs | 740.7 μs | — | **4.00x** | **—** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | **—** | 91.0 μs | 647.7 μs | — | **7.11x** | **—** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | **217.4 μs** | 181.2 μs | 736.9 μs | — | **4.07x** | **—** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | **—** | 88.2 μs | 643.7 μs | — | **7.30x** | **—** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | **215.2 μs** | 176.6 μs | 739.3 μs | — | **4.19x** | **—** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | **—** | 88.8 μs | 615.8 μs | — | **6.94x** | **—** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | **148.0 μs** | 112.8 μs | 611.4 μs | — | **5.42x** | **—** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | **—** | 90.8 μs | 643.5 μs | — | **7.09x** | **—** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | **169.1 μs** | 133.6 μs | 642.3 μs | — | **4.81x** | **—** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | **—** | 93.9 μs | 681.4 μs | — | **7.25x** | **—** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | **172.8 μs** | 133.3 μs | 653.9 μs | — | **4.90x** | **—** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | **—** | 92.5 μs | 617.0 μs | — | **6.67x** | **—** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | **112.4 μs** | 113.4 μs | 618.3 μs | — | **5.50x** | **—** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | **—** | 91.4 μs | 643.5 μs | — | **7.04x** | **—** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | **132.2 μs** | 139.9 μs | 649.9 μs | — | **4.92x** | **—** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | **—** | 92.7 μs | 673.5 μs | — | **7.27x** | **—** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | **132.2 μs** | 141.9 μs | 691.3 μs | — | **5.23x** | **—** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | **—** | 86.0 μs | 610.7 μs | — | **7.10x** | **—** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | **108.0 μs** | 110.4 μs | 616.0 μs | — | **5.70x** | **—** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | **—** | 88.2 μs | 645.6 μs | — | **7.32x** | **—** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | **116.4 μs** | 120.2 μs | 640.1 μs | — | **5.50x** | **—** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | **—** | 91.3 μs | 671.4 μs | — | **7.35x** | **—** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | **113.0 μs** | 119.2 μs | 662.3 μs | — | **5.86x** | **—** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | **—** | 87.4 μs | 607.2 μs | — | **6.95x** | **—** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | **107.9 μs** | 112.4 μs | 610.7 μs | — | **5.66x** | **—** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | **—** | 90.1 μs | 650.2 μs | — | **7.21x** | **—** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | **124.9 μs** | 130.4 μs | 642.9 μs | — | **5.15x** | **—** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | **—** | 92.6 μs | 675.7 μs | — | **7.30x** | **—** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | **126.8 μs** | 134.3 μs | 665.2 μs | — | **5.25x** | **—** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | **—** | 87.2 μs | 613.8 μs | — | **7.04x** | **—** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | **106.8 μs** | 112.0 μs | 618.3 μs | — | **5.79x** | **—** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | **—** | 89.9 μs | 641.7 μs | — | **7.14x** | **—** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | **130.9 μs** | 137.3 μs | 657.2 μs | — | **5.02x** | **—** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | **—** | 90.9 μs | 677.9 μs | — | **7.46x** | **—** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | **133.2 μs** | 140.0 μs | 673.2 μs | — | **5.05x** | **—** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | **—** | 94.7 μs | 679.4 μs | — | **7.17x** | **—** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | **103.4 μs** | 113.7 μs | 774.4 μs | — | **7.49x** | **—** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | **—** | 98.0 μs | 707.8 μs | — | **7.22x** | **—** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | **108.3 μs** | 119.7 μs | 792.5 μs | — | **7.32x** | **—** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | **—** | 96.8 μs | 738.1 μs | — | **7.62x** | **—** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | **108.2 μs** | 121.3 μs | 818.9 μs | — | **7.57x** | **—** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CUDA | **15.98 ms** | 16.00 ms | 40.64 ms | 17.66 ms | **2.54x** | **1.11x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CUDA | **15.50 ms** | 15.54 ms | 67.05 ms | 17.38 ms | **4.33x** | **1.12x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CUDA | **30.52 ms** | 30.54 ms | 36.24 ms | 29.28 ms | **1.19x** | **0.96x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CUDA | **1.07 ms** | 1.06 ms | 12.84 ms | 1.25 ms | **12.08x** | **1.18x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CUDA | **8.97 ms** | 9.00 ms | 28.97 ms | 9.24 ms | **3.23x** | **1.03x** |
| fits | large_float32_1d | read_full | 3.82 MB | CUDA | **833.8 μs** | 834.6 μs | 1.63 ms | 1.33 ms | **1.95x** | **1.59x** |
| fits | large_float32_2d | read_full | 16.00 MB | CUDA | **3.61 ms** | 3.53 ms | 16.16 ms | 5.57 ms | **4.58x** | **1.58x** |
| fits | large_float64_1d | read_full | 7.63 MB | CUDA | **1.55 ms** | 1.56 ms | 3.06 ms | 2.04 ms | **1.97x** | **1.32x** |
| fits | large_float64_2d | read_full | 32.00 MB | CUDA | **12.20 ms** | 11.80 ms | 26.20 ms | 13.30 ms | **2.22x** | **1.13x** |
| fits | large_int16_1d | read_full | 1.91 MB | CUDA | **507.5 μs** | 503.0 μs | 1.07 ms | 610.3 μs | **2.13x** | **1.21x** |
| fits | large_int16_2d | read_full | 8.00 MB | CUDA | **1.74 ms** | 1.74 ms | 4.10 ms | 2.09 ms | **2.35x** | **1.20x** |
| fits | large_int32_1d | read_full | 3.82 MB | CUDA | **835.7 μs** | 831.7 μs | 1.62 ms | 1.33 ms | **1.94x** | **1.60x** |
| fits | large_int32_2d | read_full | 16.00 MB | CUDA | **3.66 ms** | 3.55 ms | 16.21 ms | 5.54 ms | **4.57x** | **1.56x** |
| fits | large_int64_1d | read_full | 7.63 MB | CUDA | **625.9 μs** | 1.55 ms | 3.09 ms | 2.06 ms | **4.94x** | **3.29x** |
| fits | large_int64_2d | read_full | 32.00 MB | CUDA | **2.39 ms** | 11.63 ms | 26.64 ms | 12.78 ms | **11.14x** | **5.34x** |
| fits | large_int8_1d | read_full | 0.96 MB | CUDA | **1.07 ms** | 311.7 μs | 953.8 μs | 375.2 μs | **3.06x** | **1.20x** |
| fits | large_int8_2d | read_full | 4.00 MB | CUDA | **2.24 ms** | 937.6 μs | 1.99 ms | 1.13 ms | **2.12x** | **1.20x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CUDA | **4.95 ms** | 4.98 ms | 5.47 ms | 2.36 ms | **1.10x** | **0.48x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CUDA | **8.13 ms** | 8.06 ms | 10.28 ms | 6.00 ms | **1.28x** | **0.74x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CUDA | **133.5 μs** | 136.6 μs | 560.3 μs | 213.6 μs | **4.20x** | **1.60x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CUDA | **874.9 μs** | 875.1 μs | 1.71 ms | 1.41 ms | **1.95x** | **1.61x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CUDA | **1.30 ms** | 1.31 ms | 2.45 ms | 2.08 ms | **1.88x** | **1.60x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CUDA | **225.8 μs** | 229.8 μs | 698.3 μs | 302.4 μs | **3.09x** | **1.34x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CUDA | **1.64 ms** | 1.64 ms | 3.41 ms | 2.16 ms | **2.08x** | **1.32x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CUDA | **2.57 ms** | 2.59 ms | 6.67 ms | 3.39 ms | **2.59x** | **1.32x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CUDA | **90.0 μs** | 86.3 μs | 484.7 μs | 130.2 μs | **5.62x** | **1.51x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CUDA | **528.7 μs** | 530.5 μs | 1.18 ms | 648.7 μs | **2.24x** | **1.23x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CUDA | **752.4 μs** | 756.7 μs | 1.47 ms | 920.3 μs | **1.95x** | **1.22x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CUDA | **132.1 μs** | 136.5 μs | 548.6 μs | 213.8 μs | **4.15x** | **1.62x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CUDA | **881.6 μs** | 881.4 μs | 1.73 ms | 1.42 ms | **1.96x** | **1.61x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CUDA | **1.31 ms** | 1.31 ms | 2.45 ms | 2.08 ms | **1.87x** | **1.59x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CUDA | **153.1 μs** | 227.4 μs | 709.7 μs | 306.9 μs | **4.63x** | **2.00x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CUDA | **652.6 μs** | 1.67 ms | 5.49 ms | 2.20 ms | **8.41x** | **3.37x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CUDA | **962.2 μs** | 3.02 ms | 6.87 ms | 3.48 ms | **7.14x** | **3.62x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CUDA | **217.0 μs** | 69.4 μs | 595.0 μs | 101.7 μs | **8.58x** | **1.47x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CUDA | **971.2 μs** | 323.7 μs | 1.27 ms | 386.9 μs | **3.92x** | **1.20x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CUDA | **1.01 ms** | 436.2 μs | 1.19 ms | 529.8 μs | **2.73x** | **1.21x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CUDA | **782.5 μs** | 784.3 μs | 1.76 ms | 704.2 μs | **2.25x** | **0.90x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CUDA | **1.27 ms** | 1.26 ms | 2.35 ms | 1.53 ms | **1.87x** | **1.22x** |
| fits | mef_medium | read_full | 7.02 MB | CUDA | **702.4 μs** | 325.3 μs | 1.51 ms | 417.5 μs | **4.64x** | **1.28x** |
| fits | mef_small | read_full | 0.45 MB | CUDA | **140.6 μs** | 68.0 μs | 817.9 μs | 136.3 μs | **12.03x** | **2.01x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CUDA | **56.5 μs** | 54.9 μs | 4.11 ms | 300.3 μs | **74.83x** | **5.47x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CUDA | **131.0 μs** | 70.0 μs | 828.8 μs | 189.0 μs | **11.84x** | **2.70x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | CUDA | **8.54 ms** | 7.54 ms | 103.01 ms | 8.38 ms | **13.67x** | **1.11x** |
| fits | scaled_large | read_full | 8.00 MB | CUDA | **3.75 ms** | 4.67 ms | 11.56 ms | 5.08 ms | **3.09x** | **1.35x** |
| fits | scaled_medium | read_full | 2.01 MB | CUDA | **936.4 μs** | 1.11 ms | 1.95 ms | 1.30 ms | **2.09x** | **1.39x** |
| fits | scaled_small | read_full | 0.13 MB | CUDA | **166.6 μs** | 131.2 μs | 680.7 μs | 163.4 μs | **5.19x** | **1.25x** |
| fits | small_float32_1d | read_full | 42.2 KB | CUDA | **52.7 μs** | 57.2 μs | 419.4 μs | 82.4 μs | **7.97x** | **1.56x** |
| fits | small_float32_2d | read_full | 0.26 MB | CUDA | **97.6 μs** | 103.8 μs | 529.4 μs | 174.4 μs | **5.42x** | **1.79x** |
| fits | small_float32_3d | read_full | 0.63 MB | CUDA | **200.0 μs** | 202.3 μs | 677.2 μs | 296.0 μs | **3.39x** | **1.48x** |
| fits | small_float64_1d | read_full | 0.08 MB | CUDA | **59.1 μs** | 62.0 μs | 436.9 μs | 94.1 μs | **7.40x** | **1.59x** |
| fits | small_float64_2d | read_full | 0.51 MB | CUDA | **165.2 μs** | 171.0 μs | 609.6 μs | 230.1 μs | **3.69x** | **1.39x** |
| fits | small_float64_3d | read_full | 1.26 MB | CUDA | **354.0 μs** | 359.0 μs | 900.6 μs | 464.7 μs | **2.54x** | **1.31x** |
| fits | small_int16_1d | read_full | 22.5 KB | CUDA | **49.0 μs** | 50.8 μs | 399.9 μs | 70.9 μs | **8.17x** | **1.45x** |
| fits | small_int16_2d | read_full | 0.13 MB | CUDA | **70.8 μs** | 74.3 μs | 475.8 μs | 102.1 μs | **6.72x** | **1.44x** |
| fits | small_int16_3d | read_full | 0.32 MB | CUDA | **119.3 μs** | 123.4 μs | 557.2 μs | 171.1 μs | **4.67x** | **1.43x** |
| fits | small_int32_1d | read_full | 42.2 KB | CUDA | **55.1 μs** | 200.4 μs | 1.62 ms | 81.8 μs | **29.46x** | **1.49x** |
| fits | small_int32_2d | read_full | 0.26 MB | CUDA | **329.0 μs** | 173.0 μs | 902.9 μs | 389.3 μs | **5.22x** | **2.25x** |
| fits | small_int32_3d | read_full | 0.63 MB | CUDA | **294.6 μs** | 230.9 μs | 812.8 μs | 350.4 μs | **3.52x** | **1.52x** |
| fits | small_int64_1d | read_full | 0.08 MB | CUDA | **77.6 μs** | 72.0 μs | 558.4 μs | 112.7 μs | **7.75x** | **1.56x** |
| fits | small_int64_2d | read_full | 0.51 MB | CUDA | **131.9 μs** | 199.0 μs | 749.2 μs | 266.4 μs | **5.68x** | **2.02x** |
| fits | small_int64_3d | read_full | 1.26 MB | CUDA | **232.2 μs** | 414.5 μs | 1.07 ms | 540.4 μs | **4.60x** | **2.33x** |
| fits | small_int8_1d | read_full | 14.1 KB | CUDA | **94.0 μs** | 56.5 μs | 679.0 μs | 92.2 μs | **12.02x** | **1.63x** |
| fits | small_int8_2d | read_full | 0.07 MB | CUDA | **179.6 μs** | 77.8 μs | 750.2 μs | 113.5 μs | **9.64x** | **1.46x** |
| fits | small_int8_3d | read_full | 0.16 MB | CUDA | **292.1 μs** | 109.9 μs | 829.3 μs | 149.1 μs | **7.55x** | **1.36x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CUDA | **219.3 μs** | 224.9 μs | 749.7 μs | 140.8 μs | **3.42x** | **0.64x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CUDA | **301.4 μs** | 310.1 μs | 832.3 μs | 231.8 μs | **2.76x** | **0.77x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CUDA | **144.6 μs** | 142.9 μs | 735.3 μs | 234.8 μs | **5.15x** | **1.64x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CUDA | **142.4 μs** | 155.8 μs | 749.8 μs | 238.4 μs | **5.26x** | **1.67x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CUDA | **152.1 μs** | 165.2 μs | 814.8 μs | 260.8 μs | **5.36x** | **1.71x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CUDA | **151.6 μs** | 153.2 μs | 855.0 μs | 263.0 μs | **5.64x** | **1.73x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CUDA | **156.6 μs** | 156.9 μs | 814.3 μs | 270.5 μs | **5.20x** | **1.73x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CUDA | **53.6 μs** | 55.6 μs | 584.0 μs | 89.0 μs | **10.89x** | **1.66x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CUDA | **60.9 μs** | 64.4 μs | 636.9 μs | 103.5 μs | **10.46x** | **1.70x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CUDA | **63.1 μs** | 70.3 μs | 666.9 μs | 116.6 μs | **10.58x** | **1.85x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CUDA | **55.6 μs** | 57.7 μs | 615.1 μs | 101.9 μs | **11.06x** | **1.83x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CUDA | **67.5 μs** | 82.9 μs | 656.1 μs | 111.4 μs | **9.72x** | **1.65x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CUDA | **70.9 μs** | 74.5 μs | 705.9 μs | 120.8 μs | **9.96x** | **1.70x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CUDA | **52.7 μs** | 54.8 μs | 595.9 μs | 94.1 μs | **11.31x** | **1.79x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CUDA | **56.7 μs** | 56.6 μs | 622.6 μs | 102.1 μs | **11.01x** | **1.80x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CUDA | **55.0 μs** | 60.6 μs | 667.6 μs | 99.3 μs | **12.14x** | **1.81x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CUDA | **53.1 μs** | 56.3 μs | 597.3 μs | 94.4 μs | **11.24x** | **1.78x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CUDA | **62.2 μs** | 64.2 μs | 632.4 μs | 101.9 μs | **10.16x** | **1.64x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CUDA | **65.1 μs** | 68.7 μs | 664.6 μs | 108.6 μs | **10.21x** | **1.67x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CUDA | **67.6 μs** | 55.6 μs | 587.2 μs | 95.5 μs | **10.57x** | **1.72x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CUDA | **77.1 μs** | 69.2 μs | 645.4 μs | 111.4 μs | **9.32x** | **1.61x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CUDA | **93.0 μs** | 74.4 μs | 708.4 μs | 119.7 μs | **9.52x** | **1.61x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CUDA | **88.0 μs** | 57.1 μs | 793.4 μs | 92.0 μs | **13.90x** | **1.61x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CUDA | **97.5 μs** | 58.7 μs | 809.4 μs | 99.4 μs | **13.78x** | **1.69x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CUDA | **100.4 μs** | 62.7 μs | 824.7 μs | 95.7 μs | **13.16x** | **1.53x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | **1.09 ms** | 352.0 μs | 4.77 ms | — | **13.56x** | **—** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | **92.4 μs** | 96.8 μs | 11.31 ms | — | **122.38x** | **—** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | **91.4 μs** | 96.8 μs | 2.16 ms | — | **23.69x** | **—** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | **91.7 μs** | 98.8 μs | 2.60 ms | — | **28.36x** | **—** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | **135.9 μs** | 106.7 μs | 3.83 ms | — | **35.89x** | **—** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | **746.2 μs** | 290.4 μs | 2.97 ms | — | **10.22x** | **—** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | **98.4 μs** | 102.8 μs | 3.21 ms | — | **32.62x** | **—** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | **94.8 μs** | 102.2 μs | 1.99 ms | — | **21.01x** | **—** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | **96.3 μs** | 102.0 μs | 2.50 ms | — | **25.99x** | **—** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | **135.9 μs** | 110.4 μs | 2.29 ms | — | **20.73x** | **—** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | **17.18 ms** | 1.50 ms | 107.65 ms | — | **71.54x** | **—** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | **89.5 μs** | 96.6 μs | 19.56 ms | — | **218.48x** | **—** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | **90.5 μs** | 99.3 μs | 57.98 ms | — | **640.59x** | **—** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | **93.1 μs** | 96.8 μs | 20.91 ms | — | **224.56x** | **—** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | **149.9 μs** | 106.7 μs | 19.65 ms | — | **184.23x** | **—** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | **2.90 ms** | 498.2 μs | 11.65 ms | — | **23.38x** | **—** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | **92.4 μs** | 101.1 μs | 4.13 ms | — | **44.73x** | **—** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | **93.4 μs** | 99.4 μs | 6.33 ms | — | **67.79x** | **—** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | **93.5 μs** | 100.6 μs | 5.38 ms | — | **57.58x** | **—** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | **148.1 μs** | 111.9 μs | 4.11 ms | — | **36.76x** | **—** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | **1.01 ms** | 338.4 μs | 4.58 ms | — | **13.53x** | **—** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | **91.2 μs** | 96.6 μs | 2.92 ms | — | **32.09x** | **—** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | **90.1 μs** | 95.0 μs | 3.10 ms | — | **34.40x** | **—** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | **91.3 μs** | 99.2 μs | 3.98 ms | — | **43.57x** | **—** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | **147.4 μs** | 105.5 μs | 2.88 ms | — | **27.29x** | **—** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | **769.6 μs** | 261.5 μs | 3.84 ms | — | **14.70x** | **—** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | **92.9 μs** | 100.6 μs | 2.73 ms | — | **29.40x** | **—** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | **94.2 μs** | 96.6 μs | 2.76 ms | — | **29.25x** | **—** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | **94.3 μs** | 100.1 μs | 3.78 ms | — | **40.16x** | **—** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | **147.3 μs** | 106.7 μs | 2.70 ms | — | **25.33x** | **—** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | **13.05 ms** | 1.53 ms | 36.75 ms | — | **23.94x** | **—** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | **93.7 μs** | 96.2 μs | 6.06 ms | — | **64.72x** | **—** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | **91.4 μs** | 92.9 μs | 10.10 ms | — | **110.48x** | **—** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | **90.6 μs** | 95.2 μs | 6.77 ms | — | **74.73x** | **—** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | **128.3 μs** | 108.3 μs | 6.05 ms | — | **55.92x** | **—** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | **2.04 ms** | 512.3 μs | 6.18 ms | — | **12.06x** | **—** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | **91.0 μs** | 97.5 μs | 2.54 ms | — | **27.94x** | **—** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | **90.2 μs** | 96.5 μs | 2.93 ms | — | **32.44x** | **—** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | **91.6 μs** | 98.2 μs | 3.18 ms | — | **34.68x** | **—** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | **131.1 μs** | 108.6 μs | 2.49 ms | — | **22.98x** | **—** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | **822.9 μs** | 360.8 μs | 2.96 ms | — | **8.22x** | **—** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | **91.3 μs** | 94.6 μs | 2.08 ms | — | **22.82x** | **—** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | **91.6 μs** | 95.5 μs | 2.10 ms | — | **22.88x** | **—** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | **93.0 μs** | 97.8 μs | 2.66 ms | — | **28.58x** | **—** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | **130.3 μs** | 104.8 μs | 2.04 ms | — | **19.50x** | **—** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | **723.3 μs** | 282.3 μs | 2.64 ms | — | **9.35x** | **—** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | **93.8 μs** | 97.9 μs | 2.02 ms | — | **21.55x** | **—** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | **95.6 μs** | 98.5 μs | 2.05 ms | — | **21.42x** | **—** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | **92.8 μs** | 99.0 μs | 2.57 ms | — | **27.69x** | **—** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | **131.6 μs** | 106.4 μs | 1.97 ms | — | **18.54x** | **—** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | **1.98 ms** | 488.2 μs | 5.93 ms | — | **12.14x** | **—** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | **100.3 μs** | 102.9 μs | 39.48 ms | — | **393.50x** | **—** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | **97.6 μs** | 102.7 μs | 3.55 ms | — | **36.34x** | **—** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | **101.1 μs** | 107.7 μs | 3.34 ms | — | **33.07x** | **—** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | **135.0 μs** | 112.0 μs | 2.64 ms | — | **23.59x** | **—** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | **874.7 μs** | 355.9 μs | 3.15 ms | — | **8.86x** | **—** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | **98.7 μs** | 104.1 μs | 5.81 ms | — | **58.85x** | **—** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | **100.1 μs** | 103.8 μs | 2.30 ms | — | **22.99x** | **—** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | **100.7 μs** | 108.4 μs | 2.85 ms | — | **28.28x** | **—** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | **135.9 μs** | 115.0 μs | 2.22 ms | — | **19.29x** | **—** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | **1.25 ms** | 472.8 μs | 5.99 ms | — | **12.67x** | **—** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | **98.3 μs** | 101.0 μs | 780.28 ms | — | **7938.48x** | **—** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | **94.3 μs** | 101.1 μs | 3.39 ms | — | **35.94x** | **—** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | **99.2 μs** | 104.8 μs | 3.13 ms | — | **31.52x** | **—** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | **132.4 μs** | 111.5 μs | 2.51 ms | — | **22.54x** | **—** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | **563.4 μs** | 345.3 μs | 2.91 ms | — | **8.41x** | **—** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | **93.3 μs** | 97.3 μs | 79.49 ms | — | **851.90x** | **—** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | **90.6 μs** | 94.9 μs | 2.10 ms | — | **23.22x** | **—** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | **95.1 μs** | 102.6 μs | 2.54 ms | — | **26.70x** | **—** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | **129.6 μs** | 109.9 μs | 2.01 ms | — | **18.25x** | **—** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | **699.8 μs** | 284.3 μs | 2.51 ms | — | **8.83x** | **—** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | **89.9 μs** | 96.9 μs | 9.82 ms | — | **109.28x** | **—** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | **92.0 μs** | 95.8 μs | 1.97 ms | — | **21.37x** | **—** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | **93.3 μs** | 100.0 μs | 2.41 ms | — | **25.84x** | **—** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | **127.7 μs** | 104.0 μs | 1.90 ms | — | **18.32x** | **—** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | **6.58 ms** | 518.5 μs | 57.48 ms | — | **110.88x** | **—** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | **93.8 μs** | 97.5 μs | 12.44 ms | — | **132.63x** | **—** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | **93.3 μs** | 96.8 μs | 34.16 ms | — | **365.93x** | **—** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | **94.1 μs** | 97.8 μs | 18.56 ms | — | **197.21x** | **—** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | **296.1 μs** | 107.0 μs | 12.44 ms | — | **116.32x** | **—** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | **1.93 ms** | 361.6 μs | 15.68 ms | — | **43.37x** | **—** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | **89.6 μs** | 97.1 μs | 8.79 ms | — | **98.11x** | **—** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | **92.2 μs** | 95.6 μs | 10.08 ms | — | **109.37x** | **—** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | **92.2 μs** | 97.9 μs | 13.56 ms | — | **147.11x** | **—** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | **294.3 μs** | 107.6 μs | 8.74 ms | — | **81.19x** | **—** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | **1.37 ms** | 284.7 μs | 13.33 ms | — | **46.83x** | **—** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | **90.7 μs** | 95.5 μs | 8.35 ms | — | **92.05x** | **—** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | **90.7 μs** | 95.1 μs | 8.51 ms | — | **93.82x** | **—** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | **89.4 μs** | 95.0 μs | 12.99 ms | — | **145.32x** | **—** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | **284.3 μs** | 104.6 μs | 8.30 ms | — | **79.34x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Domain | Case | torchfits | Winner | Lag ratio |
|---|---|---|---:|---:|
| fits | large_int8_1d [read_full @ cuda] | 0.0010218159877695143 | fitsio/fitsio_torch_device | 2.723339425476888 |
| fits | medium_int8_2d [read_full @ cuda] | 0.0009670520084910095 | fitsio/fitsio_torch_device | 2.499462461034819 |
| fits | medium_int8_1d [read_full @ cuda] | 0.00021702301455661654 | fitsio/fitsio_torch_device | 2.1691240081379375 |
| fits | large_uint16_2d [read_full @ cuda] | 0.00494849297683686 | fitsio/fitsio_torch_device | 2.094280180766235 |
| fits | large_int8_2d [read_full @ cuda] | 0.0022349689970724285 | fitsio/fitsio_torch_device | 1.9805390562824416 |
| fits | medium_int8_3d [read_full @ cuda] | 0.0010004760115407407 | fitsio/fitsio_torch_device | 1.8884888197328742 |
| fits | small_int8_3d [read_full @ cuda] | 0.0002291480195708573 | fitsio/fitsio_torch_device | 1.8696659021904167 |
| fits | mef_medium [read_full @ cuda] | 0.0006966640357859433 | fitsio/fitsio_torch_device | 1.6687841935202312 |
| fits | large_uint32_2d [read_full @ cuda] | 0.008125699998345226 | fitsio/fitsio_torch_device | 1.3547073605312203 |
| fits | small_int8_2d [read_full @ cuda] | 0.0001369729870930314 | fitsio/fitsio_torch_device | 1.3536085059777456 |
| fits | small_uint16_2d [read_full @ cuda] | 0.00015342101687565446 | fitsio/fitsio_torch_device | 1.345869234202171 |
| fits | large_uint16_2d [read_full] | 0.007956657034810632 | astropy/astropy_torch | 1.2254218934229926 |
| fits | small_int8_1d [read_full @ cuda] | 8.49139760248363e-05 | fitsio/fitsio_torch_device | 1.121717072300533 |
| fits | medium_uint16_2d [read_full @ cuda] | 0.0007824960048310459 | fitsio/fitsio_torch_device | 1.1112142617074365 |
| fits | small_uint32_2d [read_full @ cuda] | 0.00019244995201006532 | fitsio/fitsio_torch_device | 1.0964747481145694 |
| fits | tiny_int8_3d [read_full @ cuda] | 9.894301183521748e-05 | fitsio/fitsio_torch_device | 1.054761956471154 |
| fits | scaled_small [read_full @ cuda] | 0.00016565900295972824 | fitsio/fitsio_torch_device | 1.043587182431947 |
| fits | compressed_hcompress_1 [read_full @ cuda] | 0.03048878302797675 | fitsio/fitsio_torch_device | 1.0412706641641978 |
| fits | mef_small [read_full @ cuda] | 0.00014061201363801956 | fitsio/fitsio_torch_device | 1.0312958266112076 |
| fits | tiny_int8_1d [read_full @ cuda] | 8.79999715834856e-05 | fitsio/fitsio_torch_device | 1.0228032598121128 |
| fits | repeated_cutouts_50x_100x100 @ cuda | 0.00854046200402081 | fitsio/fitsio_torch_device | 1.0193726421009917 |
| fits | tiny_int8_2d [read_full @ cuda] | 9.523401968181133e-05 | fitsio/fitsio_torch_device | 1.014509127189096 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `exhaustive_mmap_0.5.0b3_20260630_063118` | fits + fitstable (lab) | 3474 | 22 | lab bench-all `--mmap-matrix` + CUDA (H100) |
<!-- BENCH_SNAPSHOT_END -->

Latest local quick benchmark evidence:

| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| `20260625_213448` | FITS image I/O | `pixi run python benchmarks/bench_all.py --profile user --fits-only --quick` | 27 | 0 |
| `20260625_213459` | FITS table I/O | `pixi run python benchmarks/bench_all.py --profile user --fitstable-only --quick` | 90 | 0 |

Keep this page current with the latest FITS and FITS-table benchmark
run before making performance claims. Historical WCS/sphere benchmark results
are no longer maintained here.
