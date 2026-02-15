# TorchFits Benchmarks

This page tracks benchmark methodology and the 0.2.0 release snapshot.

## Main Scripts

| Script | Description |
|--------|-------------|
| `benchmark_all.py` | Exhaustive suite across image/table/compression/WCS/cutout/random-extension paths. |
| `benchmark_sentinel.py` | Fast 10-case gate with early-stop significance checks for optimization loops. |
| `benchmark_ml_loader.py` | End-to-end DataLoader throughput benchmark (uncompressed + compressed). |
| `benchmark_fast.py` | Fast regression loop for image-path changes. |
| `benchmark_compressed_stability.py` | Randomized-order compressed benchmark for RICE/HCOMPRESS tuning with lower order/cache bias (`library_default` vs forced config modes). |
| `benchmark_table.py` | FITS table-path read/scan/column access checks. |
| `benchmark_arrow_tables.py` | Arrow conversion benchmark for table workloads. |
| `benchmark_wcs.py` | WCS transform throughput checks. |

## Standard Commands

```bash
# Exhaustive release-style run (can exceed 1 hour)
pixi run python benchmarks/benchmark_all.py --profile user --include-tables --output-dir benchmark_results/<run_id>

# Fast optimization gate (10 diverse cases, early-stop enabled)
pixi run python benchmarks/benchmark_sentinel.py --initial-repeats 3 --full-repeats 9 --seed 123

# Compressed-path stability run (randomized config order)
pixi run python benchmarks/benchmark_compressed_stability.py --rounds 4 --repeats 12 --warmup 3

# ML loader throughput run
pixi run python benchmarks/benchmark_ml_loader.py --device cpu --shape 2048,2048 --n-files 50 --batch-size 4 --num-workers 4 --epochs 3 --repeats 5 --warm-cache
```

## Optimization History

For a concise log of attempted optimizations and outcomes (including dead ends), see:
- `docs/performance_attempts.md`

## 0.2.0 Release Snapshot

Run metadata:
- Date: `2026-02-13`
- Exhaustive output: `benchmark_results/release_0_2_0_20260213_154654`
- Profile: `user` (`cache=10`, `handle_cache=16`, `hot_cache=10`)
- Tables included: `True`

### Exhaustive Read Results (`operation=read_full`)

Rows: `88`

| Baseline | Wins | Median speedup | Min speedup | P10 | P90 |
|----------|-----:|---------------:|------------:|----:|----:|
| `fitsio` | 87/88 | 2.465x | 0.949x | 1.129x | 5.246x |
| `fitsio_torch` | 87/88 | 2.618x | 0.932x | 1.136x | 5.634x |
| `astropy` | 88/88 | 37.577x | 1.617x | 2.335x | 126.868x |
| `astropy_torch` | 88/88 | 28.727x | 1.351x | 2.057x | 124.898x |

Speedup definition: `baseline_median_time / torchfits_median_time`.

### Size Breakdown (vs `fitsio`)

| Size | Rows | Wins | Median | Min |
|------|-----:|-----:|-------:|----:|
| `tiny` | 18 | 18 | 4.389x | 3.868x |
| `small` | 18 | 18 | 2.487x | 1.458x |
| `medium` | 18 | 17 | 1.378x | 0.949x |
| `large` | 12 | 12 | 1.183x | 1.071x |

### Data-Type Breakdown (vs `fitsio`)

| Data type | Rows | Wins | Median | Min |
|-----------|-----:|-----:|-------:|----:|
| `int8` | 11 | 11 | 4.514x | 3.342x |
| `int16` | 11 | 11 | 2.069x | 1.317x |
| `int32` | 11 | 11 | 2.065x | 1.125x |
| `int64` | 11 | 11 | 1.984x | 1.071x |
| `float32` | 11 | 11 | 2.053x | 1.120x |
| `float64` | 11 | 10 | 1.954x | 0.949x |

### Compression Snapshot (`read_full`, vs `fitsio`)

| Group | Rows | Wins | Median | Min |
|-------|-----:|-----:|-------:|----:|
| Uncompressed | 84 | 83 | 2.548x | 0.949x |
| Compressed | 4 | 4 | 1.386x | 1.021x |

Worst compressed case in this run: `compressed_rice_1` at `1.021x` (near parity).

### Cutout / Extension Access Rows (`operation != read_full`)

| Case | TorchFits median (s) | Fitsio median (s) | Astropy median (s) | Speedup vs Fitsio |
|------|---------------------:|------------------:|-------------------:|------------------:|
| `multi_mef_10ext cutout_100x100` | 0.000061 | 0.000387 | 0.006587 | 6.390x |
| `compressed_rice_1 cutout_100x100` | 0.000771 | 0.001041 | 0.011462 | 1.351x |
| `multi_mef_10ext random_ext_full_reads_200` | 0.007146 | 0.008494 | 0.010003 | 1.189x |

### Known Weak Spots (Release Snapshot)

- `medium_float64_2d` is the only `read_full` loss vs `fitsio` (`0.949x`) and `fitsio_torch` (`0.932x`).
- `large` uncompressed cases are wins but narrower (`~1.07x` to `~1.25x`) than tiny/small cases.

## ML Loader Snapshot (0.2.0)

Command used:

```bash
pixi run python benchmarks/benchmark_ml_loader.py --device cpu --shape 2048,2048 --n-files 50 --batch-size 4 --num-workers 4 --epochs 3 --repeats 5 --warm-cache
```

Uncompressed benchmark (`torchfits` / `fitsio + numpy`):
- Run speedups: `1.074x`, `0.969x`, `0.982x`, `1.131x`, `0.959x`
- Median speedup: `0.985x`

Compressed benchmark (`torchfits (comp)` / `fitsio (comp)`):
- Run speedups: `0.990x`, `1.009x`, `1.079x`, `1.033x`, `1.008x`
- Median speedup: `1.008x`

Interpretation:
- Uncompressed loader path is currently near parity but slightly behind on median.
- Compressed loader path is effectively parity/slight win.
- Always report repeat medians/ranges, not single runs.
