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
| `bench_compressed.py` | fits | Compressed image throughput |
| `bench_decompression.py` | fits | Compression-kernel timing |
| `bench_mmap.py` | fits | mmap path timing |
| `bench_scaled.py` | fits | BSCALE/BZERO scaled-image timing |
| `bench_scaled_path.py` | fits | Scaled-path microbenchmarks |
| `bench_table.py` | fitstable | Table API timing |
| `bench_arrow_tables.py` | fitstable | Arrow-oriented table workflows |
| `bench_arrow_tables_diverse.py` | fitstable | Diverse Arrow/table schemas |
| `bench_decompressed_complex_bit_string.py` | fitstable | `update_rows` throughput on COMPLEX/BIT/fixed-width STRING columns (mmap=True vs buffered) |

## I/O Transport × Backend

> **GPU columns (`disk→GPU`, `disk→RAM→GPU`) will be populated by**
> `pixi run -e bench-gpu bench-gpu`. **Cells presently marked**
> `_pending bench-gpu_` **are deliberate reservations for that run, not
> missing data.** Once `benchmarks_results/gpu_<id>/results.csv` lands,
> re-run `scripts/render_bench_iopath_table.py` with that path to fill
> the cells. This section is regenerated from CSV by the script — do
> not hand-edit.

Source: `benchmarks_results/run_20260627_235744/results.csv` (CPU-only run).
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules; per-operation / per-size-bucket splits are a future
refinement once the GPU run unlocks finer partitioning.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | _no measured row (this run is mmap-on)_ | _no measured row (this run is mmap-on)_ | _no measured row (this run is mmap-on)_ | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.11 ms` (n=168) | `0.68 ms` (n=211) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ |
| `disk→RAM→GPU` | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | _no measured row (this run is mmap-on)_ | _no measured row (this run is mmap-on)_ | _no measured row (this run is mmap-on)_ | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.10 ms` (n=140) | `4.00 ms` (n=126) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ |
| `disk→RAM→GPU` | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ |

### Notes on the layout

- Rows are **I/O transports** (`disk→CPU`, `disk→RAM→CPU`, `disk→GPU`, `disk→RAM→GPU`).
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

## Release Snapshot

Latest local quick benchmark evidence:

| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| `20260625_213448` | FITS image I/O | `pixi run python benchmarks/bench_all.py --profile user --fits-only --quick` | 27 | 0 |
| `20260625_213459` | FITS table I/O | `pixi run python benchmarks/bench_all.py --profile user --fitstable-only --quick` | 90 | 0 |

Latest pre-extraction full-suite audit:

| Date | Scope | Comparable wins | Deficits | Report |
|---|---|---:|---:|---|
| 2026-06-26 | 81 FITS files / 84 image workflows and full table matrix | 308 / 308 | 0 | [Full report](benchmark_report_2026-06-26.md) |

Keep this page current with the latest committed FITS and FITS-table benchmark
run before making performance claims. Historical WCS/sphere benchmark results
are no longer maintained here.
