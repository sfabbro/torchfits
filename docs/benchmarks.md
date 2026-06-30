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
| ML DataLoader pattern | Diagnostic | `bench_ml_loader.py` | Not merged into `docs/benchmarks.md` tables; README cites local CPU diagnostic (Rice **1.12×** vs fitsio, 30×512² files) |

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
   lab run (`exhaustive_mmap_0.5.0b4_20260630_162835`, via `pixi run -e bench-gpu bench-exhaustive`).
4. **FITS tables have no GPU transport rows** — astropy/fitsio/torchfits table paths are
   CPU-buffered; GPU table benchmarks would mostly measure PyTorch copy overhead, not FITS
   decode, and are deliberately omitted.

### GPU integer dtype comparisons (0.5.0+)

The **deficit table** below compares default
`torchfits.read(..., scale_on_device=True)` against `torch.from_numpy(fitsio.read(...)).to(cuda)`.
That pairing is **not dtype-equivalent** for generic scaled integer FITS (see table).
After 0.5.0 narrow-integer H2D fixes, the lab snapshot dropped from **22 → 13** deficits;
remaining gaps are mostly **≤20% on tiny CUDA int8** or **cold CPU uint32** vs astropy.

| FITS convention | fitsio @ CUDA | default `read` @ CUDA (before 0.5.0 fixes) | 0.5.0 behavior |
|---|---|---|---|
| Signed byte (BITPIX=8, BZERO=-128) | native `int8` H2D | promoted to `float32` on GPU | narrow `int8` H2D + offset on device |
| Unsigned uint16/uint32 (BZERO offset) | native uint H2D | int64 widen on CPU, then cast | narrow storage H2D, offset on device |
| Generic BSCALE/BZERO scaling | often native storage dtype | `float32` on device (intentional for ML) | unchanged `float32` on device |

For apples-to-apples integer GPU timing, the exhaustive suite also records
**`torchfits_dtype_fair_device`** (`read_tensor(..., raw_scale=True)`).

**Training loops:** cold single-shot reads can lose to astropy on native uint32 CPU;
call `torchfits.cache.optimize_for_dataset(paths, avg_file_size_mb=…)` before
`DataLoader` epochs so handle caches stay warm (see `examples/example_image_dataset.py`).

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
Source: `benchmarks_results/exhaustive_mmap_0.5.0b4_20260630_162835/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.16 ms` (n=269) | `0.81 ms` (n=269) | — | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.15 ms` (n=269) | `0.71 ms` (n=219) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.14 ms` (n=256) | `0.64 ms` (n=90) | `0.21 ms` (n=90) | — |
| `disk→RAM→GPU` | `0.22 ms` (n=256) | `1.30 ms` (n=90) | `0.34 ms` (n=90) | — |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.12 ms` (n=180) | `3.40 ms` (n=162) | — | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.12 ms` (n=180) | `3.41 ms` (n=162) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large Image Read (Float32 2D, 16.0 MB) | CPU | **4.89 ms** | 4.82 ms | 15.66 ms | — | **3.25x** | **—** |
| Large Image Read (Float32 2D @ CUDA) | CUDA | **3.26 ms** | 3.33 ms | 15.46 ms | 5.10 ms | **4.75x** | **1.57x** |
| Compressed Image Read (Rice, 1.1 MB) | CPU | **9.22 ms** | 9.13 ms | 28.70 ms | — | **3.14x** | **—** |
| Compressed Image Read (Rice @ CUDA) | CUDA | **8.86 ms** | 8.91 ms | 28.08 ms | 9.15 ms | **3.17x** | **1.03x** |
| Repeated Cutouts (50x 100x100) | CPU | **6.34 ms** | 6.05 ms | 80.25 ms | — | **13.27x** | **—** |
| Table Read (100k rows, 8 cols, mixed) | CPU | **102.0 μs** | 110.7 μs | 6.37 ms | — | **62.45x** | **—** |
| Varlen Table Read (100k rows, 3 cols) | CPU | **109.8 μs** | 116.9 μs | 3.53 ms | — | **32.12x** | **—** |
<!-- BENCH_HIGHLIGHTS_END -->

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured benchmark configurations.

| Domain | Benchmark Case | Operation | Size | Device | torchfits | torchfits (persistent) | astropy (via torch) | fitsio (via torch) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fits | compressed_gzip_1 | header_read | 1.29 MB | CPU | **—** | 182.5 μs | 2.16 ms | — | **11.86x** | **—** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CPU | **16.29 ms** | 20.67 ms | 51.25 ms | — | **3.15x** | **—** |
| fits | compressed_gzip_2 | header_read | 0.89 MB | CPU | **—** | 181.0 μs | 2.19 ms | — | **12.08x** | **—** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CPU | **20.10 ms** | 20.21 ms | 82.53 ms | — | **4.11x** | **—** |
| fits | compressed_hcompress_1 | header_read | 0.82 MB | CPU | **—** | 192.3 μs | 2.28 ms | — | **11.84x** | **—** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CPU | **30.74 ms** | 30.75 ms | 36.91 ms | — | **1.20x** | **—** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | **940.9 μs** | 931.7 μs | 10.54 ms | — | **11.31x** | **—** |
| fits | compressed_rice_1 | header_read | 0.90 MB | CPU | **—** | 192.7 μs | 2.26 ms | — | **11.73x** | **—** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CPU | **9.22 ms** | 9.13 ms | 28.70 ms | — | **3.14x** | **—** |
| fits | large_float32_1d | header_read | 3.82 MB | CPU | **—** | 104.4 μs | 642.7 μs | — | **6.16x** | **—** |
| fits | large_float32_1d | read_full | 3.82 MB | CPU | **1.25 ms** | 1.20 ms | 2.40 ms | — | **2.00x** | **—** |
| fits | large_float32_2d | header_read | 16.00 MB | CPU | **—** | 110.2 μs | 674.0 μs | — | **6.12x** | **—** |
| fits | large_float32_2d | read_full | 16.00 MB | CPU | **4.89 ms** | 4.82 ms | 15.66 ms | — | **3.25x** | **—** |
| fits | large_float64_1d | header_read | 7.63 MB | CPU | **—** | 103.5 μs | 644.7 μs | — | **6.23x** | **—** |
| fits | large_float64_1d | read_full | 7.63 MB | CPU | **2.33 ms** | 2.34 ms | 4.17 ms | — | **1.78x** | **—** |
| fits | large_float64_2d | header_read | 32.00 MB | CPU | **—** | 108.4 μs | 685.7 μs | — | **6.32x** | **—** |
| fits | large_float64_2d | read_full | 32.00 MB | CPU | **12.44 ms** | 12.42 ms | 25.75 ms | — | **2.07x** | **—** |
| fits | large_int16_1d | header_read | 1.91 MB | CPU | **—** | 104.4 μs | 636.1 μs | — | **6.09x** | **—** |
| fits | large_int16_1d | read_full | 1.91 MB | CPU | **740.7 μs** | 741.7 μs | 1.62 ms | — | **2.18x** | **—** |
| fits | large_int16_2d | header_read | 8.00 MB | CPU | **—** | 105.2 μs | 681.0 μs | — | **6.48x** | **—** |
| fits | large_int16_2d | read_full | 8.00 MB | CPU | **2.69 ms** | 2.70 ms | 6.52 ms | — | **2.42x** | **—** |
| fits | large_int32_1d | header_read | 3.82 MB | CPU | **—** | 104.6 μs | 630.0 μs | — | **6.02x** | **—** |
| fits | large_int32_1d | read_full | 3.82 MB | CPU | **1.27 ms** | 1.28 ms | 2.49 ms | — | **1.96x** | **—** |
| fits | large_int32_2d | header_read | 16.00 MB | CPU | **—** | 107.3 μs | 667.1 μs | — | **6.21x** | **—** |
| fits | large_int32_2d | read_full | 16.00 MB | CPU | **5.14 ms** | 5.14 ms | 15.97 ms | — | **3.11x** | **—** |
| fits | large_int64_1d | header_read | 7.63 MB | CPU | **—** | 101.6 μs | 640.0 μs | — | **6.30x** | **—** |
| fits | large_int64_1d | read_full | 7.63 MB | CPU | **2.48 ms** | 2.49 ms | 4.34 ms | — | **1.75x** | **—** |
| fits | large_int64_2d | header_read | 32.00 MB | CPU | **—** | 108.5 μs | 665.9 μs | — | **6.14x** | **—** |
| fits | large_int64_2d | read_full | 32.00 MB | CPU | **12.34 ms** | 12.32 ms | 25.57 ms | — | **2.08x** | **—** |
| fits | large_int8_1d | header_read | 0.96 MB | CPU | **—** | 109.5 μs | 706.1 μs | — | **6.45x** | **—** |
| fits | large_int8_1d | read_full | 0.96 MB | CPU | **419.6 μs** | 430.5 μs | 1.33 ms | — | **3.17x** | **—** |
| fits | large_int8_2d | header_read | 4.00 MB | CPU | **—** | 110.3 μs | 739.1 μs | — | **6.70x** | **—** |
| fits | large_int8_2d | read_full | 4.00 MB | CPU | **1.35 ms** | 1.36 ms | 2.85 ms | — | **2.11x** | **—** |
| fits | large_uint16_2d | header_read | 8.00 MB | CPU | **—** | 112.2 μs | 727.9 μs | — | **6.49x** | **—** |
| fits | large_uint16_2d | read_full | 8.00 MB | CPU | **8.36 ms** | 6.75 ms | 10.38 ms | — | **1.54x** | **—** |
| fits | large_uint32_2d | header_read | 16.00 MB | CPU | **—** | 113.7 μs | 737.8 μs | — | **6.49x** | **—** |
| fits | large_uint32_2d | read_full | 16.00 MB | CPU | **26.70 ms** | 9.47 ms | 11.20 ms | — | **1.18x** | **—** |
| fits | medium_float32_1d | header_read | 0.38 MB | CPU | **—** | 103.3 μs | 637.3 μs | — | **6.17x** | **—** |
| fits | medium_float32_1d | read_full | 0.38 MB | CPU | **289.7 μs** | 248.3 μs | 863.7 μs | — | **3.48x** | **—** |
| fits | medium_float32_2d | header_read | 4.00 MB | CPU | **—** | 107.8 μs | 680.1 μs | — | **6.31x** | **—** |
| fits | medium_float32_2d | read_full | 4.00 MB | CPU | **1.47 ms** | 1.39 ms | 2.90 ms | — | **2.09x** | **—** |
| fits | medium_float32_3d | header_read | 6.25 MB | CPU | **—** | 107.0 μs | 694.7 μs | — | **6.49x** | **—** |
| fits | medium_float32_3d | read_full | 6.25 MB | CPU | **2.15 ms** | 2.11 ms | 3.94 ms | — | **1.87x** | **—** |
| fits | medium_float64_1d | header_read | 0.77 MB | CPU | **—** | 105.1 μs | 643.6 μs | — | **6.12x** | **—** |
| fits | medium_float64_1d | read_full | 0.77 MB | CPU | **368.1 μs** | 373.2 μs | 1.06 ms | — | **2.88x** | **—** |
| fits | medium_float64_2d | header_read | 8.00 MB | CPU | **—** | 108.0 μs | 668.2 μs | — | **6.19x** | **—** |
| fits | medium_float64_2d | read_full | 8.00 MB | CPU | **2.72 ms** | 2.74 ms | 6.46 ms | — | **2.38x** | **—** |
| fits | medium_float64_3d | header_read | 12.51 MB | CPU | **—** | 111.8 μs | 703.7 μs | — | **6.30x** | **—** |
| fits | medium_float64_3d | read_full | 12.51 MB | CPU | **4.14 ms** | 4.15 ms | 7.11 ms | — | **1.72x** | **—** |
| fits | medium_int16_1d | header_read | 0.20 MB | CPU | **—** | 102.8 μs | 633.4 μs | — | **6.16x** | **—** |
| fits | medium_int16_1d | read_full | 0.20 MB | CPU | **185.6 μs** | 196.5 μs | 762.3 μs | — | **4.11x** | **—** |
| fits | medium_int16_2d | header_read | 2.01 MB | CPU | **—** | 106.9 μs | 673.8 μs | — | **6.30x** | **—** |
| fits | medium_int16_2d | read_full | 2.01 MB | CPU | **805.9 μs** | 806.7 μs | 1.75 ms | — | **2.17x** | **—** |
| fits | medium_int16_3d | header_read | 3.13 MB | CPU | **—** | 103.8 μs | 703.2 μs | — | **6.77x** | **—** |
| fits | medium_int16_3d | read_full | 3.13 MB | CPU | **1.16 ms** | 1.31 ms | 2.65 ms | — | **2.29x** | **—** |
| fits | medium_int32_1d | header_read | 0.38 MB | CPU | **—** | 101.9 μs | 638.3 μs | — | **6.26x** | **—** |
| fits | medium_int32_1d | read_full | 0.38 MB | CPU | **279.2 μs** | 250.5 μs | 841.8 μs | — | **3.36x** | **—** |
| fits | medium_int32_2d | header_read | 4.00 MB | CPU | **—** | 106.5 μs | 673.6 μs | — | **6.33x** | **—** |
| fits | medium_int32_2d | read_full | 4.00 MB | CPU | **1.36 ms** | 1.36 ms | 2.79 ms | — | **2.06x** | **—** |
| fits | medium_int32_3d | header_read | 6.25 MB | CPU | **—** | 104.5 μs | 712.8 μs | — | **6.82x** | **—** |
| fits | medium_int32_3d | read_full | 6.25 MB | CPU | **2.04 ms** | 2.04 ms | 3.74 ms | — | **1.84x** | **—** |
| fits | medium_int64_1d | header_read | 0.77 MB | CPU | **—** | 101.0 μs | 638.9 μs | — | **6.33x** | **—** |
| fits | medium_int64_1d | read_full | 0.77 MB | CPU | **360.5 μs** | 361.8 μs | 1.04 ms | — | **2.90x** | **—** |
| fits | medium_int64_2d | header_read | 8.00 MB | CPU | **—** | 104.1 μs | 683.4 μs | — | **6.57x** | **—** |
| fits | medium_int64_2d | read_full | 8.00 MB | CPU | **2.63 ms** | 2.65 ms | 6.30 ms | — | **2.39x** | **—** |
| fits | medium_int64_3d | header_read | 12.51 MB | CPU | **—** | 105.6 μs | 705.8 μs | — | **6.68x** | **—** |
| fits | medium_int64_3d | read_full | 12.51 MB | CPU | **4.07 ms** | 4.07 ms | 6.91 ms | — | **1.70x** | **—** |
| fits | medium_int8_1d | header_read | 0.10 MB | CPU | **—** | 108.8 μs | 717.7 μs | — | **6.59x** | **—** |
| fits | medium_int8_1d | read_full | 0.10 MB | CPU | **160.0 μs** | 168.8 μs | 880.1 μs | — | **5.50x** | **—** |
| fits | medium_int8_2d | header_read | 1.01 MB | CPU | **—** | 110.5 μs | 745.2 μs | — | **6.75x** | **—** |
| fits | medium_int8_2d | read_full | 1.01 MB | CPU | **445.2 μs** | 453.3 μs | 1.37 ms | — | **3.08x** | **—** |
| fits | medium_int8_3d | header_read | 1.57 MB | CPU | **—** | 113.1 μs | 767.4 μs | — | **6.78x** | **—** |
| fits | medium_int8_3d | read_full | 1.57 MB | CPU | **617.8 μs** | 625.7 μs | 1.67 ms | — | **2.70x** | **—** |
| fits | medium_uint16_2d | header_read | 2.01 MB | CPU | **—** | 111.2 μs | 734.9 μs | — | **6.61x** | **—** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CPU | **2.07 ms** | 1.02 ms | 2.26 ms | — | **2.21x** | **—** |
| fits | medium_uint32_2d | header_read | 4.00 MB | CPU | **—** | 108.9 μs | 741.6 μs | — | **6.81x** | **—** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CPU | **3.23 ms** | 1.78 ms | 3.18 ms | — | **1.79x** | **—** |
| fits | mef_medium | header_read | 7.02 MB | CPU | **—** | 119.1 μs | 1.03 ms | — | **8.65x** | **—** |
| fits | mef_medium | read_full | 7.02 MB | CPU | **419.0 μs** | 435.9 μs | 1.60 ms | — | **3.83x** | **—** |
| fits | mef_small | header_read | 0.45 MB | CPU | **—** | 119.7 μs | 1.05 ms | — | **8.73x** | **—** |
| fits | mef_small | read_full | 0.45 MB | CPU | **148.8 μs** | 168.3 μs | 1.12 ms | — | **7.53x** | **—** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | **106.4 μs** | 107.4 μs | 3.48 ms | — | **32.71x** | **—** |
| fits | multi_mef_10ext | header_read | 2.68 MB | CPU | **—** | 119.9 μs | 1.03 ms | — | **8.61x** | **—** |
| fits | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | **6.57 ms** | 6.57 ms | 9.96 ms | — | **1.52x** | **—** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CPU | **148.7 μs** | 168.4 μs | 1.13 ms | — | **7.60x** | **—** |
| fits | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | **6.34 ms** | 6.05 ms | 80.25 ms | — | **13.27x** | **—** |
| fits | scaled_large | header_read | 8.00 MB | CPU | **—** | 113.6 μs | 760.1 μs | — | **6.69x** | **—** |
| fits | scaled_large | read_full | 8.00 MB | CPU | **4.34 ms** | 4.28 ms | 7.62 ms | — | **1.78x** | **—** |
| fits | scaled_medium | header_read | 2.01 MB | CPU | **—** | 113.4 μs | 751.7 μs | — | **6.63x** | **—** |
| fits | scaled_medium | read_full | 2.01 MB | CPU | **1.23 ms** | 1.18 ms | 2.50 ms | — | **2.12x** | **—** |
| fits | scaled_small | header_read | 0.13 MB | CPU | **—** | 112.8 μs | 747.3 μs | — | **6.62x** | **—** |
| fits | scaled_small | read_full | 0.13 MB | CPU | **251.9 μs** | 211.7 μs | 964.6 μs | — | **4.56x** | **—** |
| fits | small_float32_1d | header_read | 42.2 KB | CPU | **—** | 99.9 μs | 653.6 μs | — | **6.55x** | **—** |
| fits | small_float32_1d | read_full | 42.2 KB | CPU | **181.3 μs** | 151.1 μs | 685.8 μs | — | **4.54x** | **—** |
| fits | small_float32_2d | header_read | 0.26 MB | CPU | **—** | 106.0 μs | 680.7 μs | — | **6.42x** | **—** |
| fits | small_float32_2d | read_full | 0.26 MB | CPU | **251.3 μs** | 215.2 μs | 808.8 μs | — | **3.76x** | **—** |
| fits | small_float32_3d | header_read | 0.63 MB | CPU | **—** | 109.9 μs | 707.1 μs | — | **6.43x** | **—** |
| fits | small_float32_3d | read_full | 0.63 MB | CPU | **367.2 μs** | 323.9 μs | 1.01 ms | — | **3.11x** | **—** |
| fits | small_float64_1d | header_read | 0.08 MB | CPU | **—** | 104.5 μs | 651.6 μs | — | **6.24x** | **—** |
| fits | small_float64_1d | read_full | 0.08 MB | CPU | **153.1 μs** | 164.3 μs | 697.9 μs | — | **4.56x** | **—** |
| fits | small_float64_2d | header_read | 0.51 MB | CPU | **—** | 104.6 μs | 677.7 μs | — | **6.48x** | **—** |
| fits | small_float64_2d | read_full | 0.51 MB | CPU | **276.2 μs** | 282.4 μs | 928.9 μs | — | **3.36x** | **—** |
| fits | small_float64_3d | header_read | 1.26 MB | CPU | **—** | 107.7 μs | 707.0 μs | — | **6.56x** | **—** |
| fits | small_float64_3d | read_full | 1.26 MB | CPU | **508.5 μs** | 511.0 μs | 1.30 ms | — | **2.56x** | **—** |
| fits | small_int16_1d | header_read | 22.5 KB | CPU | **—** | 102.1 μs | 640.1 μs | — | **6.27x** | **—** |
| fits | small_int16_1d | read_full | 22.5 KB | CPU | **138.7 μs** | 145.3 μs | 671.2 μs | — | **4.84x** | **—** |
| fits | small_int16_2d | header_read | 0.13 MB | CPU | **—** | 108.3 μs | 674.5 μs | — | **6.23x** | **—** |
| fits | small_int16_2d | read_full | 0.13 MB | CPU | **171.1 μs** | 177.4 μs | 746.9 μs | — | **4.37x** | **—** |
| fits | small_int16_3d | header_read | 0.32 MB | CPU | **—** | 105.3 μs | 703.9 μs | — | **6.68x** | **—** |
| fits | small_int16_3d | read_full | 0.32 MB | CPU | **224.8 μs** | 232.5 μs | 848.7 μs | — | **3.78x** | **—** |
| fits | small_int32_1d | header_read | 42.2 KB | CPU | **—** | 100.4 μs | 646.1 μs | — | **6.44x** | **—** |
| fits | small_int32_1d | read_full | 42.2 KB | CPU | **144.5 μs** | 147.3 μs | 668.0 μs | — | **4.62x** | **—** |
| fits | small_int32_2d | header_read | 0.26 MB | CPU | **—** | 105.5 μs | 675.2 μs | — | **6.40x** | **—** |
| fits | small_int32_2d | read_full | 0.26 MB | CPU | **206.0 μs** | 213.2 μs | 792.8 μs | — | **3.85x** | **—** |
| fits | small_int32_3d | header_read | 0.63 MB | CPU | **—** | 107.6 μs | 705.2 μs | — | **6.55x** | **—** |
| fits | small_int32_3d | read_full | 0.63 MB | CPU | **317.5 μs** | 319.4 μs | 987.3 μs | — | **3.11x** | **—** |
| fits | small_int64_1d | header_read | 0.08 MB | CPU | **—** | 102.8 μs | 640.9 μs | — | **6.23x** | **—** |
| fits | small_int64_1d | read_full | 0.08 MB | CPU | **153.1 μs** | 157.6 μs | 695.6 μs | — | **4.54x** | **—** |
| fits | small_int64_2d | header_read | 0.51 MB | CPU | **—** | 103.5 μs | 671.9 μs | — | **6.49x** | **—** |
| fits | small_int64_2d | read_full | 0.51 MB | CPU | **279.8 μs** | 280.9 μs | 921.9 μs | — | **3.30x** | **—** |
| fits | small_int64_3d | header_read | 1.26 MB | CPU | **—** | 106.2 μs | 698.8 μs | — | **6.58x** | **—** |
| fits | small_int64_3d | read_full | 1.26 MB | CPU | **519.8 μs** | 523.5 μs | 1.32 ms | — | **2.53x** | **—** |
| fits | small_int8_1d | header_read | 14.1 KB | CPU | **—** | 108.4 μs | 699.3 μs | — | **6.45x** | **—** |
| fits | small_int8_1d | read_full | 14.1 KB | CPU | **117.7 μs** | 130.0 μs | 811.4 μs | — | **6.89x** | **—** |
| fits | small_int8_2d | header_read | 0.07 MB | CPU | **—** | 110.9 μs | 736.0 μs | — | **6.64x** | **—** |
| fits | small_int8_2d | read_full | 0.07 MB | CPU | **150.0 μs** | 161.0 μs | 869.2 μs | — | **5.79x** | **—** |
| fits | small_int8_3d | header_read | 0.16 MB | CPU | **—** | 112.8 μs | 760.3 μs | — | **6.74x** | **—** |
| fits | small_int8_3d | read_full | 0.16 MB | CPU | **185.6 μs** | 192.8 μs | 923.3 μs | — | **4.97x** | **—** |
| fits | small_uint16_2d | header_read | 0.13 MB | CPU | **—** | 109.4 μs | 736.9 μs | — | **6.73x** | **—** |
| fits | small_uint16_2d | read_full | 0.13 MB | CPU | **413.6 μs** | 305.6 μs | 851.0 μs | — | **2.78x** | **—** |
| fits | small_uint32_2d | header_read | 0.26 MB | CPU | **—** | 109.0 μs | 737.0 μs | — | **6.76x** | **—** |
| fits | small_uint32_2d | read_full | 0.26 MB | CPU | **473.8 μs** | 361.1 μs | 897.8 μs | — | **2.49x** | **—** |
| fits | timeseries_frame_000 | header_read | 0.26 MB | CPU | **—** | 103.7 μs | 681.0 μs | — | **6.56x** | **—** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CPU | **245.7 μs** | 205.5 μs | 787.5 μs | — | **3.83x** | **—** |
| fits | timeseries_frame_001 | header_read | 0.26 MB | CPU | **—** | 107.0 μs | 671.7 μs | — | **6.28x** | **—** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CPU | **246.0 μs** | 205.7 μs | 777.7 μs | — | **3.78x** | **—** |
| fits | timeseries_frame_002 | header_read | 0.26 MB | CPU | **—** | 106.4 μs | 676.5 μs | — | **6.36x** | **—** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CPU | **249.1 μs** | 201.5 μs | 797.7 μs | — | **3.96x** | **—** |
| fits | timeseries_frame_003 | header_read | 0.26 MB | CPU | **—** | 104.3 μs | 674.2 μs | — | **6.46x** | **—** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CPU | **241.6 μs** | 201.8 μs | 788.8 μs | — | **3.91x** | **—** |
| fits | timeseries_frame_004 | header_read | 0.26 MB | CPU | **—** | 105.9 μs | 686.3 μs | — | **6.48x** | **—** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CPU | **245.4 μs** | 204.9 μs | 785.0 μs | — | **3.83x** | **—** |
| fits | tiny_float32_1d | header_read | 8.4 KB | CPU | **—** | 101.2 μs | 648.1 μs | — | **6.41x** | **—** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CPU | **162.2 μs** | 122.0 μs | 660.5 μs | — | **5.42x** | **—** |
| fits | tiny_float32_2d | header_read | 19.7 KB | CPU | **—** | 104.2 μs | 677.6 μs | — | **6.50x** | **—** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CPU | **182.6 μs** | 146.8 μs | 689.5 μs | — | **4.70x** | **—** |
| fits | tiny_float32_3d | header_read | 25.3 KB | CPU | **—** | 106.6 μs | 708.8 μs | — | **6.65x** | **—** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CPU | **189.0 μs** | 144.3 μs | 702.1 μs | — | **4.86x** | **—** |
| fits | tiny_float64_1d | header_read | 11.2 KB | CPU | **—** | 101.5 μs | 646.7 μs | — | **6.37x** | **—** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CPU | **119.4 μs** | 125.7 μs | 650.0 μs | — | **5.45x** | **—** |
| fits | tiny_float64_2d | header_read | 36.6 KB | CPU | **—** | 106.3 μs | 672.2 μs | — | **6.32x** | **—** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CPU | **137.9 μs** | 148.0 μs | 695.4 μs | — | **5.04x** | **—** |
| fits | tiny_float64_3d | header_read | 45.0 KB | CPU | **—** | 108.3 μs | 703.5 μs | — | **6.50x** | **—** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CPU | **139.0 μs** | 152.6 μs | 719.4 μs | — | **5.18x** | **—** |
| fits | tiny_int16_1d | header_read | 5.6 KB | CPU | **—** | 104.9 μs | 646.1 μs | — | **6.16x** | **—** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CPU | **115.0 μs** | 121.8 μs | 655.1 μs | — | **5.70x** | **—** |
| fits | tiny_int16_2d | header_read | 11.2 KB | CPU | **—** | 106.2 μs | 675.2 μs | — | **6.36x** | **—** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CPU | **118.5 μs** | 126.4 μs | 673.9 μs | — | **5.69x** | **—** |
| fits | tiny_int16_3d | header_read | 14.1 KB | CPU | **—** | 106.6 μs | 696.7 μs | — | **6.54x** | **—** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CPU | **122.4 μs** | 129.9 μs | 699.1 μs | — | **5.71x** | **—** |
| fits | tiny_int32_1d | header_read | 8.4 KB | CPU | **—** | 99.6 μs | 638.1 μs | — | **6.41x** | **—** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CPU | **117.0 μs** | 124.8 μs | 646.2 μs | — | **5.52x** | **—** |
| fits | tiny_int32_2d | header_read | 19.7 KB | CPU | **—** | 106.4 μs | 667.7 μs | — | **6.27x** | **—** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CPU | **137.0 μs** | 142.8 μs | 675.9 μs | — | **4.93x** | **—** |
| fits | tiny_int32_3d | header_read | 25.3 KB | CPU | **—** | 108.1 μs | 704.4 μs | — | **6.52x** | **—** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CPU | **138.7 μs** | 146.5 μs | 705.5 μs | — | **5.09x** | **—** |
| fits | tiny_int64_1d | header_read | 11.2 KB | CPU | **—** | 100.3 μs | 644.8 μs | — | **6.43x** | **—** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CPU | **120.3 μs** | 122.4 μs | 658.5 μs | — | **5.47x** | **—** |
| fits | tiny_int64_2d | header_read | 36.6 KB | CPU | **—** | 104.3 μs | 663.5 μs | — | **6.36x** | **—** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CPU | **142.3 μs** | 152.5 μs | 682.0 μs | — | **4.79x** | **—** |
| fits | tiny_int64_3d | header_read | 45.0 KB | CPU | **—** | 107.1 μs | 705.5 μs | — | **6.59x** | **—** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CPU | **146.6 μs** | 154.1 μs | 709.8 μs | — | **4.84x** | **—** |
| fits | tiny_int8_1d | header_read | 5.6 KB | CPU | **—** | 108.2 μs | 696.9 μs | — | **6.44x** | **—** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CPU | **115.9 μs** | 126.9 μs | 805.8 μs | — | **6.95x** | **—** |
| fits | tiny_int8_2d | header_read | 8.4 KB | CPU | **—** | 112.3 μs | 731.9 μs | — | **6.52x** | **—** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CPU | **118.6 μs** | 129.0 μs | 827.8 μs | — | **6.98x** | **—** |
| fits | tiny_int8_3d | header_read | 8.4 KB | CPU | **—** | 114.6 μs | 756.3 μs | — | **6.60x** | **—** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CPU | **119.7 μs** | 132.2 μs | 852.4 μs | — | **7.12x** | **—** |
| fits | compressed_gzip_1 | read_full | 1.29 MB | CUDA | **15.86 ms** | 15.94 ms | 40.19 ms | 17.54 ms | **2.53x** | **1.11x** |
| fits | compressed_gzip_2 | read_full | 0.89 MB | CUDA | **15.41 ms** | 15.48 ms | 64.88 ms | 17.30 ms | **4.21x** | **1.12x** |
| fits | compressed_hcompress_1 | read_full | 0.82 MB | CUDA | **30.36 ms** | 30.45 ms | 36.43 ms | 29.14 ms | **1.20x** | **0.96x** |
| fits | compressed_rice_1 | cutout_100x100 | 0.90 MB | CUDA | **1.11 ms** | 833.5 μs | 9.70 ms | 962.3 μs | **11.64x** | **1.15x** |
| fits | compressed_rice_1 | read_full | 0.90 MB | CUDA | **8.86 ms** | 8.91 ms | 28.08 ms | 9.15 ms | **3.17x** | **1.03x** |
| fits | large_float32_1d | read_full | 3.82 MB | CUDA | **709.5 μs** | 733.7 μs | 1.42 ms | 1.22 ms | **2.00x** | **1.71x** |
| fits | large_float32_2d | read_full | 16.00 MB | CUDA | **3.26 ms** | 3.33 ms | 15.46 ms | 5.10 ms | **4.75x** | **1.57x** |
| fits | large_float64_1d | read_full | 7.63 MB | CUDA | **1.29 ms** | 1.33 ms | 2.88 ms | 1.84 ms | **2.23x** | **1.42x** |
| fits | large_float64_2d | read_full | 32.00 MB | CUDA | **11.72 ms** | 11.86 ms | 25.68 ms | 12.59 ms | **2.19x** | **1.07x** |
| fits | large_int16_1d | read_full | 1.91 MB | CUDA | **418.8 μs** | 438.8 μs | 957.9 μs | 542.9 μs | **2.29x** | **1.30x** |
| fits | large_int16_2d | read_full | 8.00 MB | CUDA | **1.46 ms** | 1.48 ms | 5.32 ms | 1.85 ms | **3.64x** | **1.27x** |
| fits | large_int32_1d | read_full | 3.82 MB | CUDA | **709.3 μs** | 730.2 μs | 1.42 ms | 1.21 ms | **2.01x** | **1.71x** |
| fits | large_int32_2d | read_full | 16.00 MB | CUDA | **3.25 ms** | 3.40 ms | 15.63 ms | 5.24 ms | **4.81x** | **1.61x** |
| fits | large_int64_1d | read_full | 7.63 MB | CUDA | **526.6 μs** | 1.38 ms | 2.77 ms | 1.86 ms | **5.25x** | **3.53x** |
| fits | large_int64_2d | read_full | 32.00 MB | CUDA | **2.27 ms** | 11.92 ms | 26.18 ms | 13.30 ms | **11.54x** | **5.86x** |
| fits | large_int8_1d | read_full | 0.96 MB | CUDA | **212.6 μs** | 289.1 μs | 888.5 μs | 349.1 μs | **4.18x** | **1.64x** |
| fits | large_int8_2d | read_full | 4.00 MB | CUDA | **610.3 μs** | 859.9 μs | 1.78 ms | 1.04 ms | **2.91x** | **1.71x** |
| fits | large_uint16_2d | read_full | 8.00 MB | CUDA | **1.46 ms** | 4.64 ms | 5.33 ms | 2.03 ms | **3.65x** | **1.39x** |
| fits | large_uint32_2d | read_full | 16.00 MB | CUDA | **3.20 ms** | 9.63 ms | 9.56 ms | 5.65 ms | **2.99x** | **1.76x** |
| fits | medium_float32_1d | read_full | 0.38 MB | CUDA | **104.9 μs** | 131.7 μs | 512.4 μs | 206.0 μs | **4.88x** | **1.96x** |
| fits | medium_float32_2d | read_full | 4.00 MB | CUDA | **746.6 μs** | 766.9 μs | 1.53 ms | 1.31 ms | **2.05x** | **1.75x** |
| fits | medium_float32_3d | read_full | 6.25 MB | CUDA | **1.10 ms** | 1.14 ms | 2.20 ms | 1.92 ms | **2.00x** | **1.74x** |
| fits | medium_float64_1d | read_full | 0.77 MB | CUDA | **192.9 μs** | 216.0 μs | 639.2 μs | 284.2 μs | **3.31x** | **1.47x** |
| fits | medium_float64_2d | read_full | 8.00 MB | CUDA | **1.42 ms** | 1.46 ms | 3.23 ms | 1.95 ms | **2.28x** | **1.38x** |
| fits | medium_float64_3d | read_full | 12.51 MB | CUDA | **2.26 ms** | 2.33 ms | 6.19 ms | 3.09 ms | **2.74x** | **1.37x** |
| fits | medium_int16_1d | read_full | 0.20 MB | CUDA | **64.9 μs** | 87.4 μs | 459.0 μs | 123.6 μs | **7.08x** | **1.91x** |
| fits | medium_int16_2d | read_full | 2.01 MB | CUDA | **447.4 μs** | 467.8 μs | 1.70 ms | 562.7 μs | **3.81x** | **1.26x** |
| fits | medium_int16_3d | read_full | 3.13 MB | CUDA | **634.8 μs** | 657.3 μs | 1.28 ms | 785.8 μs | **2.02x** | **1.24x** |
| fits | medium_int32_1d | read_full | 0.38 MB | CUDA | **110.7 μs** | 131.9 μs | 511.9 μs | 209.6 μs | **4.62x** | **1.89x** |
| fits | medium_int32_2d | read_full | 4.00 MB | CUDA | **748.1 μs** | 771.6 μs | 2.84 ms | 1.29 ms | **3.80x** | **1.72x** |
| fits | medium_int32_3d | read_full | 6.25 MB | CUDA | **1.11 ms** | 1.13 ms | 2.15 ms | 1.91 ms | **1.95x** | **1.72x** |
| fits | medium_int64_1d | read_full | 0.77 MB | CUDA | **139.8 μs** | 218.2 μs | 648.8 μs | 288.9 μs | **4.64x** | **2.07x** |
| fits | medium_int64_2d | read_full | 8.00 MB | CUDA | **549.5 μs** | 1.42 ms | 5.02 ms | 1.99 ms | **9.14x** | **3.63x** |
| fits | medium_int64_3d | read_full | 12.51 MB | CUDA | **787.7 μs** | 2.38 ms | 10.54 ms | 3.26 ms | **13.38x** | **4.14x** |
| fits | medium_int8_1d | read_full | 0.10 MB | CUDA | **43.9 μs** | 71.5 μs | 564.4 μs | 101.3 μs | **12.85x** | **2.31x** |
| fits | medium_int8_2d | read_full | 1.01 MB | CUDA | **222.3 μs** | 304.1 μs | 1.20 ms | 360.1 μs | **5.41x** | **1.62x** |
| fits | medium_int8_3d | read_full | 1.57 MB | CUDA | **300.0 μs** | 407.2 μs | 1.07 ms | 480.2 μs | **3.56x** | **1.60x** |
| fits | medium_uint16_2d | read_full | 2.01 MB | CUDA | **435.0 μs** | 1.14 ms | 1.66 ms | 618.8 μs | **3.81x** | **1.42x** |
| fits | medium_uint32_2d | read_full | 4.00 MB | CUDA | **729.2 μs** | 1.40 ms | 2.12 ms | 1.41 ms | **2.90x** | **1.93x** |
| fits | mef_medium | read_full | 7.02 MB | CUDA | **215.9 μs** | 297.1 μs | 1.43 ms | 380.9 μs | **6.63x** | **1.76x** |
| fits | mef_small | read_full | 0.45 MB | CUDA | **48.1 μs** | 74.2 μs | 807.1 μs | 128.8 μs | **16.78x** | **2.68x** |
| fits | multi_mef_10ext | cutout_100x100 | 2.68 MB | CUDA | **41.5 μs** | 44.4 μs | 3.13 ms | 208.6 μs | **75.36x** | **5.02x** |
| fits | multi_mef_10ext | read_full | 2.68 MB | CUDA | **46.6 μs** | 72.6 μs | 788.8 μs | 191.1 μs | **16.92x** | **4.10x** |
| fits | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | CUDA | **5.57 ms** | 5.07 ms | 78.96 ms | 5.55 ms | **15.56x** | **1.09x** |
| fits | scaled_large | read_full | 8.00 MB | CUDA | **1.44 ms** | 4.13 ms | 12.41 ms | 4.53 ms | **8.61x** | **3.15x** |
| fits | scaled_medium | read_full | 2.01 MB | CUDA | **437.7 μs** | 1.05 ms | 1.79 ms | 1.21 ms | **4.08x** | **2.77x** |
| fits | scaled_small | read_full | 0.13 MB | CUDA | **56.6 μs** | 139.8 μs | 666.8 μs | 165.9 μs | **11.78x** | **2.93x** |
| fits | small_float32_1d | read_full | 42.2 KB | CUDA | **39.8 μs** | 56.1 μs | 413.6 μs | 79.0 μs | **10.40x** | **1.99x** |
| fits | small_float32_2d | read_full | 0.26 MB | CUDA | **73.3 μs** | 100.1 μs | 489.1 μs | 163.5 μs | **6.67x** | **2.23x** |
| fits | small_float32_3d | read_full | 0.63 MB | CUDA | **165.0 μs** | 193.1 μs | 643.9 μs | 291.5 μs | **3.90x** | **1.77x** |
| fits | small_float64_1d | read_full | 0.08 MB | CUDA | **43.4 μs** | 61.5 μs | 436.2 μs | 87.4 μs | **10.06x** | **2.02x** |
| fits | small_float64_2d | read_full | 0.51 MB | CUDA | **135.9 μs** | 162.7 μs | 582.6 μs | 218.0 μs | **4.29x** | **1.60x** |
| fits | small_float64_3d | read_full | 1.26 MB | CUDA | **292.1 μs** | 321.3 μs | 809.9 μs | 426.5 μs | **2.77x** | **1.46x** |
| fits | small_int16_1d | read_full | 22.5 KB | CUDA | **36.2 μs** | 50.2 μs | 393.1 μs | 77.9 μs | **10.87x** | **2.15x** |
| fits | small_int16_2d | read_full | 0.13 MB | CUDA | **52.6 μs** | 72.2 μs | 446.4 μs | 100.7 μs | **8.49x** | **1.92x** |
| fits | small_int16_3d | read_full | 0.32 MB | CUDA | **99.5 μs** | 120.2 μs | 522.6 μs | 162.6 μs | **5.25x** | **1.63x** |
| fits | small_int32_1d | read_full | 42.2 KB | CUDA | **37.8 μs** | 53.6 μs | 403.4 μs | 84.2 μs | **10.68x** | **2.23x** |
| fits | small_int32_2d | read_full | 0.26 MB | CUDA | **75.6 μs** | 99.5 μs | 492.6 μs | 165.5 μs | **6.52x** | **2.19x** |
| fits | small_int32_3d | read_full | 0.63 MB | CUDA | **164.3 μs** | 191.9 μs | 627.3 μs | 279.7 μs | **3.82x** | **1.70x** |
| fits | small_int64_1d | read_full | 0.08 MB | CUDA | **64.2 μs** | 60.2 μs | 417.6 μs | 87.8 μs | **6.94x** | **1.46x** |
| fits | small_int64_2d | read_full | 0.51 MB | CUDA | **104.4 μs** | 159.8 μs | 585.6 μs | 217.4 μs | **5.61x** | **2.08x** |
| fits | small_int64_3d | read_full | 1.26 MB | CUDA | **185.1 μs** | 313.9 μs | 810.1 μs | 413.1 μs | **4.38x** | **2.23x** |
| fits | small_int8_1d | read_full | 14.1 KB | CUDA | **31.4 μs** | 48.3 μs | 533.4 μs | 71.6 μs | **16.99x** | **2.28x** |
| fits | small_int8_2d | read_full | 0.07 MB | CUDA | **49.2 μs** | 64.4 μs | 569.5 μs | 92.6 μs | **11.57x** | **1.88x** |
| fits | small_int8_3d | read_full | 0.16 MB | CUDA | **52.6 μs** | 86.9 μs | 621.2 μs | 122.3 μs | **11.82x** | **2.33x** |
| fits | small_uint16_2d | read_full | 0.13 MB | CUDA | **54.7 μs** | 136.6 μs | 552.7 μs | 113.9 μs | **10.11x** | **2.08x** |
| fits | small_uint32_2d | read_full | 0.26 MB | CUDA | **77.7 μs** | 185.1 μs | 588.5 μs | 166.2 μs | **7.57x** | **2.14x** |
| fits | timeseries_frame_000 | read_full | 0.26 MB | CUDA | **74.8 μs** | 100.2 μs | 495.0 μs | 164.2 μs | **6.62x** | **2.20x** |
| fits | timeseries_frame_001 | read_full | 0.26 MB | CUDA | **74.4 μs** | 100.2 μs | 493.3 μs | 163.1 μs | **6.63x** | **2.19x** |
| fits | timeseries_frame_002 | read_full | 0.26 MB | CUDA | **74.6 μs** | 99.2 μs | 492.1 μs | 162.3 μs | **6.60x** | **2.18x** |
| fits | timeseries_frame_003 | read_full | 0.26 MB | CUDA | **73.0 μs** | 98.3 μs | 488.9 μs | 159.2 μs | **6.70x** | **2.18x** |
| fits | timeseries_frame_004 | read_full | 0.26 MB | CUDA | **70.9 μs** | 97.0 μs | 487.8 μs | 159.1 μs | **6.88x** | **2.24x** |
| fits | tiny_float32_1d | read_full | 8.4 KB | CUDA | **28.0 μs** | 44.3 μs | 382.9 μs | 69.1 μs | **13.66x** | **2.47x** |
| fits | tiny_float32_2d | read_full | 19.7 KB | CUDA | **33.4 μs** | 50.8 μs | 413.3 μs | 74.0 μs | **12.39x** | **2.22x** |
| fits | tiny_float32_3d | read_full | 25.3 KB | CUDA | **34.4 μs** | 53.7 μs | 439.3 μs | 75.0 μs | **12.76x** | **2.18x** |
| fits | tiny_float64_1d | read_full | 11.2 KB | CUDA | **29.0 μs** | 44.9 μs | 390.3 μs | 71.4 μs | **13.47x** | **2.46x** |
| fits | tiny_float64_2d | read_full | 36.6 KB | CUDA | **36.4 μs** | 56.6 μs | 427.0 μs | 78.6 μs | **11.72x** | **2.16x** |
| fits | tiny_float64_3d | read_full | 45.0 KB | CUDA | **41.1 μs** | 56.9 μs | 449.4 μs | 78.5 μs | **10.93x** | **1.91x** |
| fits | tiny_int16_1d | read_full | 5.6 KB | CUDA | **28.9 μs** | 41.7 μs | 395.0 μs | 66.1 μs | **13.67x** | **2.29x** |
| fits | tiny_int16_2d | read_full | 11.2 KB | CUDA | **30.1 μs** | 46.2 μs | 407.5 μs | 73.6 μs | **13.55x** | **2.45x** |
| fits | tiny_int16_3d | read_full | 14.1 KB | CUDA | **30.2 μs** | 47.1 μs | 420.5 μs | 73.4 μs | **13.92x** | **2.43x** |
| fits | tiny_int32_1d | read_full | 8.4 KB | CUDA | **29.3 μs** | 42.6 μs | 390.0 μs | 67.5 μs | **13.32x** | **2.31x** |
| fits | tiny_int32_2d | read_full | 19.7 KB | CUDA | **35.0 μs** | 50.0 μs | 412.8 μs | 76.9 μs | **11.80x** | **2.20x** |
| fits | tiny_int32_3d | read_full | 25.3 KB | CUDA | **36.1 μs** | 52.2 μs | 431.2 μs | 76.0 μs | **11.95x** | **2.10x** |
| fits | tiny_int64_1d | read_full | 11.2 KB | CUDA | **54.7 μs** | 43.2 μs | 380.8 μs | 68.7 μs | **8.82x** | **1.59x** |
| fits | tiny_int64_2d | read_full | 36.6 KB | CUDA | **59.5 μs** | 54.7 μs | 414.1 μs | 76.6 μs | **7.57x** | **1.40x** |
| fits | tiny_int64_3d | read_full | 45.0 KB | CUDA | **59.9 μs** | 56.4 μs | 439.9 μs | 82.5 μs | **7.80x** | **1.46x** |
| fits | tiny_int8_1d | read_full | 5.6 KB | CUDA | **28.6 μs** | 45.9 μs | 522.9 μs | 66.9 μs | **18.27x** | **2.34x** |
| fits | tiny_int8_2d | read_full | 8.4 KB | CUDA | **29.2 μs** | 47.7 μs | 538.6 μs | 71.2 μs | **18.42x** | **2.43x** |
| fits | tiny_int8_3d | read_full | 8.4 KB | CUDA | **30.0 μs** | 49.3 μs | 548.7 μs | 71.7 μs | **18.28x** | **2.39x** |
| fitstable | ascii_10000 | predicate_filter | 0.44 MB | CPU | **1.13 ms** | 314.2 μs | 6.52 ms | — | **20.75x** | **—** |
| fitstable | ascii_10000 | projection | 0.44 MB | CPU | **107.2 μs** | 115.4 μs | 11.45 ms | — | **106.76x** | **—** |
| fitstable | ascii_10000 | read_full | 0.44 MB | CPU | **106.3 μs** | 111.9 μs | 2.24 ms | — | **21.03x** | **—** |
| fitstable | ascii_10000 | row_slice | 0.44 MB | CPU | **114.8 μs** | 117.8 μs | 2.66 ms | — | **23.20x** | **—** |
| fitstable | ascii_10000 | scan_count | 0.44 MB | CPU | **182.1 μs** | 154.5 μs | 3.96 ms | — | **25.64x** | **—** |
| fitstable | ascii_1000 | predicate_filter | 50.6 KB | CPU | **800.0 μs** | 268.3 μs | 3.02 ms | — | **11.26x** | **—** |
| fitstable | ascii_1000 | projection | 50.6 KB | CPU | **115.1 μs** | 119.4 μs | 3.30 ms | — | **28.68x** | **—** |
| fitstable | ascii_1000 | read_full | 50.6 KB | CPU | **114.7 μs** | 130.6 μs | 2.04 ms | — | **17.82x** | **—** |
| fitstable | ascii_1000 | row_slice | 50.6 KB | CPU | **111.9 μs** | 117.8 μs | 2.55 ms | — | **22.76x** | **—** |
| fitstable | ascii_1000 | scan_count | 50.6 KB | CPU | **154.3 μs** | 127.6 μs | 2.34 ms | — | **18.37x** | **—** |
| fitstable | mixed_1000000 | predicate_filter | 50.55 MB | CPU | **18.60 ms** | 1.52 ms | 110.58 ms | — | **72.71x** | **—** |
| fitstable | mixed_1000000 | projection | 50.55 MB | CPU | **103.5 μs** | 113.7 μs | 20.75 ms | — | **200.51x** | **—** |
| fitstable | mixed_1000000 | read_full | 50.55 MB | CPU | **102.9 μs** | 113.2 μs | 60.29 ms | — | **586.11x** | **—** |
| fitstable | mixed_1000000 | row_slice | 50.55 MB | CPU | **111.9 μs** | 114.6 μs | 22.22 ms | — | **198.53x** | **—** |
| fitstable | mixed_1000000 | scan_count | 50.55 MB | CPU | **163.3 μs** | 128.8 μs | 20.99 ms | — | **162.96x** | **—** |
| fitstable | mixed_100000 | predicate_filter | 5.06 MB | CPU | **3.01 ms** | 470.3 μs | 11.60 ms | — | **24.68x** | **—** |
| fitstable | mixed_100000 | projection | 5.06 MB | CPU | **119.6 μs** | 114.1 μs | 4.27 ms | — | **37.39x** | **—** |
| fitstable | mixed_100000 | read_full | 5.06 MB | CPU | **102.0 μs** | 110.7 μs | 6.37 ms | — | **62.45x** | **—** |
| fitstable | mixed_100000 | row_slice | 5.06 MB | CPU | **105.6 μs** | 113.2 μs | 5.45 ms | — | **51.60x** | **—** |
| fitstable | mixed_100000 | scan_count | 5.06 MB | CPU | **164.8 μs** | 121.5 μs | 4.24 ms | — | **34.88x** | **—** |
| fitstable | mixed_10000 | predicate_filter | 0.51 MB | CPU | **1.07 ms** | 313.2 μs | 4.65 ms | — | **14.85x** | **—** |
| fitstable | mixed_10000 | projection | 0.51 MB | CPU | **105.1 μs** | 111.6 μs | 3.01 ms | — | **28.65x** | **—** |
| fitstable | mixed_10000 | read_full | 0.51 MB | CPU | **106.7 μs** | 116.1 μs | 3.18 ms | — | **29.77x** | **—** |
| fitstable | mixed_10000 | row_slice | 0.51 MB | CPU | **105.5 μs** | 111.9 μs | 4.04 ms | — | **38.30x** | **—** |
| fitstable | mixed_10000 | scan_count | 0.51 MB | CPU | **167.6 μs** | 121.6 μs | 2.96 ms | — | **24.32x** | **—** |
| fitstable | mixed_1000 | predicate_filter | 0.06 MB | CPU | **807.1 μs** | 233.9 μs | 3.94 ms | — | **16.82x** | **—** |
| fitstable | mixed_1000 | projection | 0.06 MB | CPU | **104.0 μs** | 112.5 μs | 2.80 ms | — | **26.87x** | **—** |
| fitstable | mixed_1000 | read_full | 0.06 MB | CPU | **105.0 μs** | 112.6 μs | 2.81 ms | — | **26.76x** | **—** |
| fitstable | mixed_1000 | row_slice | 0.06 MB | CPU | **108.2 μs** | 111.0 μs | 3.81 ms | — | **35.21x** | **—** |
| fitstable | mixed_1000 | scan_count | 0.06 MB | CPU | **163.2 μs** | 119.4 μs | 2.78 ms | — | **23.29x** | **—** |
| fitstable | narrow_1000000 | predicate_filter | 12.40 MB | CPU | **13.50 ms** | 1.50 ms | 37.57 ms | — | **24.98x** | **—** |
| fitstable | narrow_1000000 | projection | 12.40 MB | CPU | **109.2 μs** | 117.5 μs | 6.43 ms | — | **58.90x** | **—** |
| fitstable | narrow_1000000 | read_full | 12.40 MB | CPU | **123.5 μs** | 125.4 μs | 10.70 ms | — | **86.65x** | **—** |
| fitstable | narrow_1000000 | row_slice | 12.40 MB | CPU | **112.0 μs** | 116.0 μs | 7.09 ms | — | **63.34x** | **—** |
| fitstable | narrow_1000000 | scan_count | 12.40 MB | CPU | **148.6 μs** | 119.3 μs | 6.32 ms | — | **52.98x** | **—** |
| fitstable | narrow_100000 | predicate_filter | 1.25 MB | CPU | **2.09 ms** | 492.0 μs | 6.26 ms | — | **12.71x** | **—** |
| fitstable | narrow_100000 | projection | 1.25 MB | CPU | **104.9 μs** | 111.2 μs | 2.64 ms | — | **25.19x** | **—** |
| fitstable | narrow_100000 | read_full | 1.25 MB | CPU | **104.3 μs** | 113.0 μs | 3.02 ms | — | **28.94x** | **—** |
| fitstable | narrow_100000 | row_slice | 1.25 MB | CPU | **106.1 μs** | 115.1 μs | 3.24 ms | — | **30.59x** | **—** |
| fitstable | narrow_100000 | scan_count | 1.25 MB | CPU | **147.9 μs** | 120.6 μs | 2.60 ms | — | **21.60x** | **—** |
| fitstable | narrow_10000 | predicate_filter | 0.13 MB | CPU | **846.4 μs** | 326.2 μs | 3.03 ms | — | **9.28x** | **—** |
| fitstable | narrow_10000 | projection | 0.13 MB | CPU | **105.2 μs** | 111.2 μs | 2.14 ms | — | **20.37x** | **—** |
| fitstable | narrow_10000 | read_full | 0.13 MB | CPU | **104.0 μs** | 110.7 μs | 2.14 ms | — | **20.54x** | **—** |
| fitstable | narrow_10000 | row_slice | 0.13 MB | CPU | **106.8 μs** | 116.1 μs | 2.70 ms | — | **25.31x** | **—** |
| fitstable | narrow_10000 | scan_count | 0.13 MB | CPU | **147.9 μs** | 120.7 μs | 2.09 ms | — | **17.31x** | **—** |
| fitstable | narrow_1000 | predicate_filter | 19.7 KB | CPU | **765.3 μs** | 247.5 μs | 2.68 ms | — | **10.83x** | **—** |
| fitstable | narrow_1000 | projection | 19.7 KB | CPU | **105.6 μs** | 112.3 μs | 2.08 ms | — | **19.74x** | **—** |
| fitstable | narrow_1000 | read_full | 19.7 KB | CPU | **109.2 μs** | 112.6 μs | 2.08 ms | — | **19.05x** | **—** |
| fitstable | narrow_1000 | row_slice | 19.7 KB | CPU | **106.0 μs** | 115.4 μs | 2.63 ms | — | **24.82x** | **—** |
| fitstable | narrow_1000 | scan_count | 19.7 KB | CPU | **148.3 μs** | 121.5 μs | 2.03 ms | — | **16.71x** | **—** |
| fitstable | typed_100000 | predicate_filter | 2.39 MB | CPU | **2.13 ms** | 479.6 μs | 6.07 ms | — | **12.66x** | **—** |
| fitstable | typed_100000 | projection | 2.39 MB | CPU | **112.4 μs** | 122.6 μs | 40.01 ms | — | **355.97x** | **—** |
| fitstable | typed_100000 | read_full | 2.39 MB | CPU | **117.0 μs** | 131.0 μs | 3.74 ms | — | **31.97x** | **—** |
| fitstable | typed_100000 | row_slice | 2.39 MB | CPU | **116.6 μs** | 124.5 μs | 3.48 ms | — | **29.80x** | **—** |
| fitstable | typed_100000 | scan_count | 2.39 MB | CPU | **155.8 μs** | 128.6 μs | 2.78 ms | — | **21.65x** | **—** |
| fitstable | typed_10000 | predicate_filter | 0.24 MB | CPU | **936.3 μs** | 326.0 μs | 3.30 ms | — | **10.12x** | **—** |
| fitstable | typed_10000 | projection | 0.24 MB | CPU | **113.9 μs** | 119.8 μs | 5.99 ms | — | **52.55x** | **—** |
| fitstable | typed_10000 | read_full | 0.24 MB | CPU | **127.2 μs** | 129.5 μs | 2.39 ms | — | **18.78x** | **—** |
| fitstable | typed_10000 | row_slice | 0.24 MB | CPU | **118.4 μs** | 123.6 μs | 2.92 ms | — | **24.68x** | **—** |
| fitstable | typed_10000 | scan_count | 0.24 MB | CPU | **154.6 μs** | 131.9 μs | 2.29 ms | — | **17.36x** | **—** |
| fitstable | varlen_100000 | predicate_filter | 3.06 MB | CPU | **1.29 ms** | 450.3 μs | 9.66 ms | — | **21.44x** | **—** |
| fitstable | varlen_100000 | projection | 3.06 MB | CPU | **114.8 μs** | 116.5 μs | 758.44 ms | — | **6606.30x** | **—** |
| fitstable | varlen_100000 | read_full | 3.06 MB | CPU | **109.8 μs** | 116.9 μs | 3.53 ms | — | **32.12x** | **—** |
| fitstable | varlen_100000 | row_slice | 3.06 MB | CPU | **113.5 μs** | 116.6 μs | 3.21 ms | — | **28.30x** | **—** |
| fitstable | varlen_100000 | scan_count | 3.06 MB | CPU | **172.7 μs** | 142.8 μs | 3.09 ms | — | **21.65x** | **—** |
| fitstable | varlen_10000 | predicate_filter | 0.31 MB | CPU | **584.0 μs** | 318.9 μs | 2.95 ms | — | **9.26x** | **—** |
| fitstable | varlen_10000 | projection | 0.31 MB | CPU | **109.7 μs** | 116.2 μs | 78.75 ms | — | **717.99x** | **—** |
| fitstable | varlen_10000 | read_full | 0.31 MB | CPU | **106.4 μs** | 117.9 μs | 2.18 ms | — | **20.49x** | **—** |
| fitstable | varlen_10000 | row_slice | 0.31 MB | CPU | **165.0 μs** | 160.6 μs | 2.60 ms | — | **16.17x** | **—** |
| fitstable | varlen_10000 | scan_count | 0.31 MB | CPU | **146.7 μs** | 123.7 μs | 2.07 ms | — | **16.73x** | **—** |
| fitstable | varlen_1000 | predicate_filter | 39.4 KB | CPU | **757.0 μs** | 255.0 μs | 2.58 ms | — | **10.13x** | **—** |
| fitstable | varlen_1000 | projection | 39.4 KB | CPU | **106.3 μs** | 115.3 μs | 9.72 ms | — | **91.48x** | **—** |
| fitstable | varlen_1000 | read_full | 39.4 KB | CPU | **106.9 μs** | 115.8 μs | 2.04 ms | — | **19.12x** | **—** |
| fitstable | varlen_1000 | row_slice | 39.4 KB | CPU | **109.2 μs** | 115.5 μs | 2.49 ms | — | **22.76x** | **—** |
| fitstable | varlen_1000 | scan_count | 39.4 KB | CPU | **144.4 μs** | 117.9 μs | 1.98 ms | — | **16.82x** | **—** |
| fitstable | wide_100000 | predicate_filter | 20.71 MB | CPU | **7.17 ms** | 478.2 μs | 73.01 ms | — | **152.67x** | **—** |
| fitstable | wide_100000 | projection | 20.71 MB | CPU | **106.0 μs** | 110.5 μs | 12.87 ms | — | **121.42x** | **—** |
| fitstable | wide_100000 | read_full | 20.71 MB | CPU | **105.5 μs** | 109.9 μs | 33.11 ms | — | **313.70x** | **—** |
| fitstable | wide_100000 | row_slice | 20.71 MB | CPU | **107.0 μs** | 112.2 μs | 18.90 ms | — | **176.72x** | **—** |
| fitstable | wide_100000 | scan_count | 20.71 MB | CPU | **375.3 μs** | 135.7 μs | 15.73 ms | — | **115.93x** | **—** |
| fitstable | wide_10000 | predicate_filter | 2.08 MB | CPU | **2.07 ms** | 330.7 μs | 15.80 ms | — | **47.77x** | **—** |
| fitstable | wide_10000 | projection | 2.08 MB | CPU | **105.6 μs** | 113.4 μs | 8.97 ms | — | **84.93x** | **—** |
| fitstable | wide_10000 | read_full | 2.08 MB | CPU | **103.1 μs** | 110.1 μs | 10.22 ms | — | **99.12x** | **—** |
| fitstable | wide_10000 | row_slice | 2.08 MB | CPU | **104.6 μs** | 115.9 μs | 13.76 ms | — | **131.52x** | **—** |
| fitstable | wide_10000 | scan_count | 2.08 MB | CPU | **308.8 μs** | 121.6 μs | 8.91 ms | — | **73.32x** | **—** |
| fitstable | wide_1000 | predicate_filter | 0.22 MB | CPU | **1.45 ms** | 249.0 μs | 13.41 ms | — | **53.86x** | **—** |
| fitstable | wide_1000 | projection | 0.22 MB | CPU | **102.3 μs** | 110.1 μs | 8.43 ms | — | **82.44x** | **—** |
| fitstable | wide_1000 | read_full | 0.22 MB | CPU | **104.3 μs** | 111.6 μs | 8.59 ms | — | **82.31x** | **—** |
| fitstable | wide_1000 | row_slice | 0.22 MB | CPU | **105.2 μs** | 112.8 μs | 13.16 ms | — | **125.08x** | **—** |
| fitstable | wide_1000 | scan_count | 0.22 MB | CPU | **303.7 μs** | 118.3 μs | 8.40 ms | — | **71.02x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency).

Latest lab run: **13 deficits** (down from 22 pre–integer H2D fixes). Largest remaining gap:
`large_uint32_2d` cold CPU read vs astropy (~1.5×); CUDA int8 cases are ≤1.2× vs fitsio.

| Domain | Case | torchfits | Winner | Lag ratio |
|---|---|---|---:|---:|
| fits | large_uint32_2d [read_full] | 0.017325995955616236 | astropy/astropy_torch | 1.5248161130488542 |
| fits | tiny_int8_1d [read_full @ cuda] | 8.04349547252059e-05 | fitsio/fitsio_torch_device | 1.202999965177421 |
| fits | small_int8_1d [read_full @ cuda] | 8.484802674502134e-05 | fitsio/fitsio_torch_device | 1.1844328864050209 |
| fits | tiny_int8_3d [read_full @ cuda] | 8.179800352081656e-05 | fitsio/fitsio_torch_device | 1.1415055947038157 |
| fits | tiny_int8_2d [read_full @ cuda] | 8.105701999738812e-05 | fitsio/fitsio_torch_device | 1.13900155814838 |
| fits | compressed_rice_1 [cutout_100x100 @ cuda] | 0.0010645200381986797 | fitsio/fitsio_torch_device | 1.1062741687071713 |
| fits | small_int8_2d [read_full @ cuda] | 0.00010107900016009808 | fitsio/fitsio_torch_device | 1.091483466752484 |
| fits | small_uint16_2d [read_full @ cuda] | 0.00012334302300587296 | fitsio/fitsio_torch_device | 1.0828299500491843 |
| fits | compressed_hcompress_1 [read_full @ cuda] | 0.030440815025940537 | fitsio/fitsio_torch_device | 1.0455709030073166 |
| fits | medium_int8_1d [read_full @ cuda] | 0.00010562798706814647 | fitsio/fitsio_torch_device | 1.0424361874584243 |
| fits | scaled_small [read_full @ cuda] | 0.0001682970323599875 | fitsio/fitsio_torch_device | 1.0141982514651802 |
| fits | repeated_cutouts_50x_100x100 @ cuda | 0.005570091016124934 | fitsio/fitsio_torch_device | 1.0039347500037794 |
| fits | small_int8_3d [read_full @ cuda] | 0.00012267299462109804 | fitsio/fitsio_torch_device | 1.003304349067989 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `exhaustive_mmap_0.5.0b4_20260630_162835` | fits + fitstable (lab) | 3626 | 13 | lab bench-all + `--mmap-matrix` + CUDA/MPS |
<!-- BENCH_SNAPSHOT_END -->

Latest local quick benchmark evidence:

| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| `20260625_213448` | FITS image I/O | `pixi run python benchmarks/bench_all.py --profile user --fits-only --quick` | 27 | 0 |
| `20260625_213459` | FITS table I/O | `pixi run python benchmarks/bench_all.py --profile user --fitstable-only --quick` | 90 | 0 |

Keep this page current with the latest FITS and FITS-table benchmark
run before making performance claims. Historical WCS/sphere benchmark results
are no longer maintained here.
