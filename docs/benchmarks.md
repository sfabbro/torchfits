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
| `bench_table.py` | fitstable | Table API timing |
| `bench_arrow_tables.py` | fitstable | Arrow-oriented table workflows |

## I/O Transport × Backend

> **GPU columns (`disk→GPU`, `disk→RAM→GPU`)** are populated from
> `bench_gpu_transports.py` rows (metadata `io_transport`). On Apple Silicon
> use `pixi run bench-mps` (`device=mps`); on Linux CUDA use
> `pixi run -e bench-gpu bench-gpu`. CUDA columns for **0.5.0 final** are
> validated on Linux; **0.5.0b1** ships MPS evidence from macOS runs.

<!-- BENCH_IOPATH_BEGIN -->
Source: `benchmarks_results/20260629_114056_gpu/results.csv` (MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### FITS image I/O (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | _no measured row (this run is mmap-on)_ | _no measured row (this run is mmap-on)_ | _no measured row (this run is mmap-on)_ | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.13 ms` (n=249) | `0.80 ms` (n=211) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ |
| `disk→RAM→GPU` | `0.42 ms` (n=20) | _pending bench-gpu_ | `0.31 ms` (n=20) | _pending bench-gpu_ |

### FITS table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | _no measured row (this run is mmap-on)_ | _no measured row (this run is mmap-on)_ | _no measured row (this run is mmap-on)_ | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.12 ms` (n=70) | `5.08 ms` (n=126) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ |
| `disk→RAM→GPU` | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ | _pending bench-gpu_ |
<!-- BENCH_IOPATH_END -->

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

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (documented for transparency; not fixed in this release).

| Domain | Case | torchfits | Winner | Lag ratio |
|---|---|---|---:|---:|
| fits | tiny_int16_1d [read_full @ mps] | 0.0004003749927505851 | fitsio/fitsio_torch_device | 2.3416608849056204 |
| fits | timeseries_frame_001 [read_full @ mps] | 0.00040087499655783176 | fitsio/fitsio_torch_device | 1.7521411445761064 |
| fits | scaled_small [read_full @ mps] | 0.0004840835463255644 | fitsio/fitsio_torch_device | 1.7192784501940068 |
| fits | timeseries_frame_002 [read_full @ mps] | 0.0003870624932460487 | fitsio/fitsio_torch_device | 1.6831874965353795 |
| fits | tiny_int32_1d [read_full @ mps] | 0.0002795205218717456 | fitsio/fitsio_torch_device | 1.6594957504615158 |
| fits | tiny_int8_1d [read_full @ mps] | 0.00031602103263139725 | fitsio/fitsio_torch_device | 1.6109797772959622 |
| fits | timeseries_frame_004 [read_full @ mps] | 0.00034795800456777215 | fitsio/fitsio_torch_device | 1.596745380211369 |
| fits | wcs_image [read_full @ mps] | 0.0004939584759995341 | fitsio/fitsio_torch_device | 1.5734267756934661 |
| fits | timeseries_frame_000 [read_full @ mps] | 0.00036395795177668333 | fitsio/fitsio_torch_device | 1.5356841644566634 |
| fits | medium_int16_2d [read_full @ mps] | 0.0009762919507920742 | fitsio/fitsio_torch_device | 1.5335926052105637 |
| fits | scaled_medium [read_full @ mps] | 0.0017416665214113891 | fitsio/fitsio_torch_device | 1.4193308289557955 |
| fits | small_int32_3d [read_full @ mps] | 0.0004101460799574852 | fitsio/fitsio_torch_device | 1.3214510828622499 |
| fits | tiny_float64_2d [read_full @ mps] | 0.00027295801555737853 | fitsio/fitsio_torch_device | 1.254382356088166 |
| fits | timeseries_frame_003 [read_full @ mps] | 0.00037372851511463523 | fitsio/fitsio_torch_device | 1.2053326610140283 |
| fits | multi_mef_10ext [read_full @ mps] | 0.0004381045000627637 | fitsio/fitsio_torch_device | 1.15779507278043 |
| fits | medium_float32_3d [read_full @ mps] | 0.002103958046063781 | fitsio/fitsio_torch_device | 1.1510404550851165 |
| fits | medium_int32_3d [read_full @ mps] | 0.0020045004785060883 | fitsio/fitsio_torch_device | 1.0983062026277086 |
| fits | large_int32_1d [read_full @ mps] | 0.0011520834523253143 | fitsio/fitsio_torch_device | 1.0440659371208085 |
| fits | compressed_hcompress_1 [read_full @ mps] | 0.03341781202470884 | fitsio/fitsio_torch_device | 1.029113785206993 |
| fits | compressed_rice_1 [read_full @ mps] | 0.010672708100173622 | fitsio/fitsio_torch_device | 1.009830395871548 |
| fits | large_int16_2d [read_full] | 0.00326020794454962 | astropy/astropy | 1.064108808298818 |
<!-- BENCH_DEFICITS_END -->

## Release Snapshot

Latest full lab benchmark:

| Run ID | Scope | Rows | Deficits | Notes |
|---|---|---:|---:|---|
<!-- BENCH_SNAPSHOT_BEGIN -->
| `20260629_114056_gpu` | fits + fitstable (lab) | (see CSV) | (see deficits CSV) | CI weekly bench-all |
<!-- BENCH_SNAPSHOT_END -->

Latest local quick benchmark evidence:

| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| `20260625_213448` | FITS image I/O | `pixi run python benchmarks/bench_all.py --profile user --fits-only --quick` | 27 | 0 |
| `20260625_213459` | FITS table I/O | `pixi run python benchmarks/bench_all.py --profile user --fitstable-only --quick` | 90 | 0 |

Keep this page current with the latest FITS and FITS-table benchmark
run before making performance claims. Historical WCS/sphere benchmark results
are no longer maintained here.
