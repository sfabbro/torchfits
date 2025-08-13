# TorchFits Benchmark Suite

Unified documentation for all benchmarking assets (replaces former `BENCHMARKS.md`, `benchmarks/docs/README.md`, and `benchmarks/docs/OVERVIEW.md`).

## 1. Overview

The TorchFits benchmark suite validates performance, memory efficiency, and correctness against reference libraries (`fitsio`, `astropy`). It includes:

* Official comparative benchmark suite (pytest-based)
* Focused micro-benchmarks (e.g. table bulk column read, batched cutouts)
* Tunable performance knobs (env vars) for advanced runs

### Performance Targets (Current Working Goals)

| Domain | Target (relative to fastest competitor) |
|--------|------------------------------------------|
| 2D / 3D image full reads | Competitive (≥0.8× and improving) |
| Table column subset reads | 1.2–5× (scalar numeric multi-column optimized) |
| Cutouts | < 100 ms typical; small < 50 ms |
| Memory footprint | < 2× worst competitor; aim for parity |
| Remote (future) | <15% overhead with caching |

## 2. Repository Layout

```text
benchmarks/
  README.md                # This file
  table_bulk_read_micro.py # Micro-benchmark for table column selection path
  cutout_batch_micro.py    # Micro-benchmark for random batched cutouts (+ async GPU)
  image_cutout_compare.py  # Compare cutout pipelines across torchfits/astropy/fitsio
  compare_readers.py       # All-in-one comparison (image full, cutouts+WCS, table columns)
  run_all.py               # Convenience runner that executes all benchmarks
  prefetch_iterable_bench.py # Micro-benchmark for iterable dataset prefetch benefit
tests/
  test_official_benchmark_suite.py  # Main comparative benchmark tests (still in tests/ for pytest discovery)
```

> NOTE: The main suite remains under `tests/` so it can be selectively invoked in CI with standard pytest patterns. Micro-benchmarks and docs live here.

## 3. Running the Official Suite

### Quick (small/medium representative cases)

```bash
pytest tests/test_official_benchmark_suite.py::test_image_performance -v
pytest tests/test_official_benchmark_suite.py::test_table_performance -v
```

### Full benchmark grouping (same orchestration used previously by the runner script)

```bash
pytest tests/test_official_benchmark_suite.py::test_image_performance -v
pytest tests/test_official_benchmark_suite.py::test_cutout_performance -v
pytest tests/test_official_benchmark_suite.py::test_column_selection_performance -v
pytest tests/test_official_benchmark_suite.py::test_memory_efficiency -v
pytest tests/test_official_benchmark_suite.py::test_error_handling -v
pytest tests/test_official_benchmark_suite.py::test_performance_summary -v
```

### All at once (pytest will build the temporary data files once per session)

```bash
pytest tests/test_official_benchmark_suite.py -v
```

### JSON Output

After running enough tests to populate results, a JSON artifact is written in the temporary benchmark output directory (see summary printed by the suite). You can parse this for automated regression tracking.

## 4. Micro-Benchmarks

### Table Bulk Column Read Micro-Benchmark

Measures the scalar numeric multi-column selection path (bulk `fits_read_col` + optional parallelism / pinned memory).

Run:

```bash
python benchmarks/table_bulk_read_micro.py
```

Example output (compact table):

```text
== Table bulk column read (200000 rows, 4 cols) ==
Impl      | API                    | mean ms | stdev | notes            
----------+------------------------+---------+-------+------------------
torchfits | read(columns)->tensor  | 34.12   | 1.02  | native tensor dict
astropy   | numpy->torch (cols)    | 66.48   | 2.31  | np->torch        
fitsio    | numpy->torch (cols)    | 52.07   | 1.78  | np->torch        
```

Interpretation: Improvement ≈ 1.23×. Track deltas over time; flag regressions >10%.

### Batched Random Cutout Micro-Benchmark

Measures throughput of parallel random cutouts across multiple files and optional async GPU transfer.

Run (CPU baseline):

```bash
python benchmarks/cutout_batch_micro.py --batch 64 --rep 5
```

Run with GPU & async pipeline (if CUDA available):

```bash
TORCHFITS_ASYNC_GPU=1 python benchmarks/cutout_batch_micro.py --cuda --batch 64 --rep 5
```

Outputs: JSON-like dict lines with ms_mean/ms_stdev; async speedup printed if faster.

New table format example:

```text
== Batched cutouts ==
Impl      | Mode      | mean ms | stdev | notes             
----------+-----------+---------+-------+-------------------
torchfits | CPU       | 12.35   | 0.44  | batch=64, sz=64   
torchfits | GPU sync  | 8.73    | 0.39  | batch=64, sz=64   
torchfits | GPU async | 6.11    | 0.27  | batch=64, sz=64   
(async speedup 1.43x vs sync)
```

### Image Cutout Comparison (single-threaded)

Compare per-cutout pipeline cost across libraries, including numpy->torch conversion for competitors and optional WCS creation on astropy.

Run:

```bash
python benchmarks/image_cutout_compare.py --size 1024 --cutouts 10 --cutout-size 64 --reps 5
```

### All-in-one Reader Comparison

Run:

```bash
python benchmarks/compare_readers.py --size 1024 --cutouts 10 --cutout-size 64 --reps 5 --table-rows 200000
```

This prints three tables: image full read, random cutouts (+WCS on astropy), and table column subset with mixed dtypes (ints, floats, strings, bools).

### Run Everything

```bash
python benchmarks/run_all.py
```

## 5. Performance Tuning Knobs

| Env Var | Values | Effect |
|---------|--------|--------|
| `TORCHFITS_PAR_READ` | `1` / unset | Enable parallel per-column table reads (>=4 scalar numeric columns) using independent CFITSIO handles. |
| `TORCHFITS_PIN_MEMORY` | `1` / unset | Allocate pinned host memory for table columns (beneficial if transferring to GPU). |
| `TORCHFITS_BATCH_THREADS` | int | Max threads for batch/multi-cutout parallelism (default min(N,32)). |
| `TORCHFITS_ASYNC_GPU` | `1` / unset | Force async GPU transfers in batch read pipeline. |

Usage example:

```bash
TORCHFITS_PAR_READ=1 TORCHFITS_PIN_MEMORY=1 pytest tests/test_official_benchmark_suite.py::test_column_selection_performance -k table_medium -v
```

## 6. Interpreting Results

Key metrics (printed per case):

* `read_time_ms` – wall-clock read time
* `throughput_mbs` – file_size / time (rough efficiency proxy)
* `speedup_vs_baseline` – ratio vs fastest available competitor (fitsio preferred, else astropy)
* `memory_usage_mb` – peak RSS delta (process-level)

Warnings in the suite indicate performance below soft thresholds; only hard assertions will fail CI.

## 7. Adding New Benchmarks

1. Extend data generator in `test_official_benchmark_suite.py` if new data form needed.
2. Add a parametrized test referencing new file key / operation.
3. Keep per-test runtime modest (<2–3 s) to remain CI friendly; move very large scale runs to a separate, opt-in script.

## 8. Roadmap / Future Enhancements

* Remote I/O + cache latency benchmarks
* Compressed FITS read performance (when compression path enabled)
* GPU transfer + end-to-end pipeline timings
* Write performance symmetry suite
* Trend tracking (store JSON history & plot)

## 9. Troubleshooting

| Symptom | Cause | Mitigation |
|---------|-------|------------|
| Slow table subset reads | Parallel path disabled | Set `TORCHFITS_PAR_READ=1` (ensure ≥4 scalar numeric cols) |
| No speedup reported | Baseline lib missing | Install `fitsio` / `astropy` into environment |
| High memory delta | Large temporary intermediate tensors | Verify selection uses column subset not full table |
| Cutout assertion fails | Cutout outside bounds | Ensure slice ranges inside shape or add bounds clipping |

## 10. Attribution & Original Content

This consolidated README incorporates and supersedes the original comprehensive benchmark narrative that previously lived in `BENCHMARKS.md` with detailed examples, goals, data generation logic, error handling notes, and extension roadmap.

---
Maintainers: update this file when modifying performance targets or adding environment flags so users have a single source of truth.
