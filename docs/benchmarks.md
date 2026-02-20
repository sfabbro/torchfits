# TorchFits Benchmarks

This page documents the benchmark methodology. The snapshot tables below reflect the **0.2.1** release; update with each release run.

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
| `benchmark_healpix.py` | HEALPix CPU/CUDA parity + throughput + quality gates (`ang2pix`, `pix2ang`, `ring/nest`). |
| `benchmark_healpix_advanced.py` | Advanced HEALPix parity + throughput checks (`get_all_neighbours`, `get_interp_weights`, `get_interp_val`). |
| `benchmark_sphere_geometry.py` | Cross-library spherical geometry benchmark (TorchFits vs HEALPix ecosystem releases). |
| `benchmark_sphere_core.py` | Sphere-core primitives benchmark (multi-band sampling, pairwise distances, ellipse query). |
| `benchmark_sphere_polygons.py` | Non-convex spherical polygon benchmark (contains/query/area with optional spherical-geometry parity). |
| `benchmark_sphere_spectral.py` | CPU-first spherical harmonic primitive benchmark (`map2alm`, `map2alm_lsq`, `alm2map`, `almxfl`, `alm2cl`, `anafast`, `map2alm_spin`, `alm2map_spin`) plus compatibility-generation paths (`synalm`, `synfast`, `bl2beam`, `beam2bl`) as needed. |
| `benchmark_pipeline_table_sphere.py` | End-to-end benchmark: table predicate pushdown (`where`) + HEALPix spherical reduction pipeline. |
| `replay_upstream_healpy_interp_edges.py` | Replays interpolation edge cases (lon wrap, poles, pixel-center inputs) against official healpy with parity/perf gates. |
| `replay_upstream_healpy_spin.py` | Replays spin transform parity/throughput against official healpy releases (`map2alm_spin`, `alm2map_spin`). |
| `replay_upstream_spherical_geometry_polygons.py` | Replays spherical-geometry upstream polygon fixtures/data (`test_intersects_*`, `difficult_intersections.txt`) against TorchFits with multi-NSIDE difficult-overlap area gates. |

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

# WCS parity+throughput run with mixed interior/boundary sampling and p99 gates
pixi run python benchmarks/benchmark_wcs.py --cases TAN,SIN,ARC,AIT,MOL,HPX,CEA,MER,TAN_SIP,TPV --sample-profile mixed --n-points 200000 --runs 5 --p99-angular-error-arcsec 1e-2 --p99-inverse-pixel-error 1e-2

# HEALPix parity+throughput run on CPU (strict parity gates)
pixi run python benchmarks/benchmark_healpix.py --device cpu --sample-profile mixed --nside 1024 --n-points 200000 --runs 5 --max-index-mismatches 0 --max-pix2ang-dra-deg 1e-10 --max-pix2ang-ddec-deg 1e-10

# HEALPix CUDA run with CPU comparison gate
pixi run python benchmarks/benchmark_healpix.py --device cuda --compare-cpu --sample-profile mixed --nside 1024 --n-points 200000 --runs 5 --min-cuda-speedup-vs-cpu 2.0

# HEALPix Apple GPU (MPS) run with CPU comparison (float32 thresholds)
pixi run python benchmarks/benchmark_healpix.py --device mps --compare-cpu --sample-profile uniform --nside 1024 --n-points 200000 --runs 5 --max-index-mismatches 10 --max-pix2ang-dra-deg 5e-4 --max-pix2ang-ddec-deg 5e-4

# Advanced HEALPix benchmark (neighbors + interpolation)
pixi run -e sphere-bench sphere-bench-healpix-advanced

# Advanced HEALPix benchmark gate (ratio thresholds vs healpy)
pixi run -e sphere-bench sphere-bench-healpix-advanced-gate

# Advanced HEALPix small-N gate (overhead-sensitive)
pixi run -e sphere-bench sphere-bench-healpix-advanced-gate-small

# Sphere-core benchmark (multi-band sampling + geometry primitives)
pixi run -e sphere-bench sphere-bench-core

# Non-convex polygon benchmark (contains/query/area)
pixi run -e sphere-bench sphere-bench-polygons

# Spectral primitive benchmark (CPU-first scalar + spin transforms)
pixi run -e sphere-bench sphere-bench-spectral

# End-to-end table pushdown + sphere pipeline benchmark
pixi run -e sphere-bench sphere-bench-pipeline

# End-to-end table pushdown + sphere pipeline gate (ratio thresholds)
pixi run -e sphere-bench sphere-bench-pipeline-gate

# Install optional sphere benchmark comparators from official releases
pixi run -e sphere-bench sphere-bench-bootstrap

# Install only core HEALPix comparator packages (lean setup)
pixi run -e sphere-bench sphere-bench-bootstrap-core

# Sync upstream release source artifacts and extract tests/data fixtures
pixi run -e sphere-bench sphere-upstream-sync

# Replay astropy-healpix upstream test-style HEALPix cases on TorchFits vs healpy
pixi run -e sphere-bench sphere-upstream-gate

# Replay upstream test functions directly (including shape semantics + hypothesis inner tests)
pixi run -e sphere-bench sphere-upstream-test-gate

# Replay/gate upstream healpy spin transform cases
pixi run -e sphere-bench sphere-upstream-spin-gate

# Optional extended spin replay matrix (higher-complexity spin-2 cases)
pixi run -e sphere-bench python benchmarks/replay_upstream_healpy_spin.py --case-set extended --runs 5

# Replay/gate upstream interpolation edge cases (pole/wrap/pixel-boundary semantics)
pixi run -e sphere-bench sphere-upstream-interp-edge-gate

# Replay/gate upstream spherical-geometry polygon fixtures/data
pixi run -e sphere-bench sphere-upstream-polygon-gate

# Cross-library sphere geometry benchmark (fails if comparator comes from local/editable/VCS install)
pixi run -e sphere-bench sphere-bench-geometry-fast

# Ecosystem benchmark snapshot against released packages (healpy/hpgeom/astropy-healpix/healpix/mhealpy)
pixi run -e sphere-bench sphere-bench-geometry-ecosystem

# Cross-library sphere geometry matrix with correctness + median ratio gates
pixi run -e sphere-bench sphere-bench-geometry-gate

# Small-N matrix/profile gate to catch overhead-sensitive regressions
pixi run -e sphere-bench sphere-bench-geometry-gate-small
```

## Sphere Benchmark Policy

- Comparator libraries must come from official released distributions (pip/uv/conda), not local repo clones.
- `benchmark_healpix.py` and `benchmark_sphere_geometry.py` now print comparator package provenance and fail by default on local/editable/VCS installs.
- `benchmark_sphere_matrix.py` runs `uniform/boundary/mixed` profiles and can enforce median TorchFits/healpy speed-ratio floors.
- `benchmark_sphere_polygons.py` benchmarks non-convex polygon contains/query/area and optionally checks parity against released `spherical-geometry`.
- `replay_upstream_spherical_geometry_polygons.py` replays polygon fixtures/data from upstream spherical-geometry test suite with correctness gates (contains + pair intersections + difficult-case nonempty parity + multi-NSIDE overlap-area/convergence checks, default ladder `128..16384`).
- `sync_upstream_sphere_sources.py` downloads release artifacts for selected packages and extracts upstream `tests/` + `data/` trees with a pinned manifest.
- `replay_upstream_astropy_healpy.py` replays astropy-healpix test-style HEALPix cases (same NSIDE range and boundary example-style vectors) against TorchFits/healpy with correctness + ratio gates.
- `replay_upstream_test_functions.py` executes selected upstream astropy-healpix test functions directly against TorchFits adapters for closer semantic parity checks.
- `replay_upstream_healpy_interp_edges.py` replays interpolation edge cases (lon wrap, poles, edge pixels) against healpy with mismatch/error and ratio gates.
- `replay_upstream_healpy_spin.py` replays spin transform cases (`map2alm_spin`, `alm2map_spin`) against healpy with per-case error gates and median throughput ratio thresholds.
- Spin replay defaults to the current validated profile (`TORCHFITS_RING_FOURIER_CPP=1`, recurrence off, `TORCHFITS_SPIN_RING_AUTO_MIN_BYTES=32 MiB`) unless explicitly overridden in the environment.
- Override only for local development experiments with:
  - `benchmark_healpix.py --allow-nonrelease-healpy`
  - `benchmark_sphere_geometry.py --allow-nonrelease-distributions`

## Optimization History

For a concise log of attempted optimizations and outcomes (including dead ends), see:
- `docs/performance_attempts.md`

## 0.2.1 Release Snapshot

Run metadata:
- Date: `2026-02-14`
- Exhaustive output: `benchmark_results/exhaustive_20260214_211602`
- Profile: `user` (`cache=10`, `handle_cache=16`, `hot_cache=10`)
- Tables included: `True`

### Exhaustive Read Results (`operation=read_full`)

Rows: `88`

| Baseline | Wins | Median speedup | Min speedup | P10 | P90 |
|----------|-----:|---------------:|------------:|----:|----:|
| `fitsio` | 88/88 | 2.478x | 1.01x | 1.14x | 5.34x |
| `fitsio_torch` | 88/88 | 2.651x | 1.01x | 1.14x | 5.86x |
| `astropy` | 88/88 | 37.577x | 1.617x | 2.335x | 126.868x |
| `astropy_torch` | 88/88 | 28.727x | 1.351x | 2.057x | 124.898x |

Speedup definition: `baseline_median_time / torchfits_median_time`.

### Size Breakdown (vs `fitsio`)

| Size | Rows | Wins | Median | Min |
|------|-----:|-----:|-------:|----:|
| `tiny` | 18 | 18 | 4.412x | 3.19x |
| `small` | 18 | 18 | 2.512x | 1.33x |
| `medium` | 18 | 18 | 1.392x | 1.08x |
| `large` | 12 | 12 | 1.151x | 1.01x |

### Data-Type Breakdown (vs `fitsio`)

| Data type | Rows | Wins | Median | Min |
|-----------|-----:|-----:|-------:|----:|
| `int8` | 11 | 11 | 4.62x | 3.47x |
| `int16` | 11 | 11 | 2.20x | 1.16x |
| `int32` | 11 | 11 | 2.26x | 1.06x |
| `int64` | 11 | 11 | 1.99x | 1.08x |
| `float32` | 11 | 11 | 2.16x | 1.15x |
| `float64` | 11 | 11 | 1.68x | 1.08x |

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

- **No regressions found** (100% win rate against baseline).
- `medium_float64_2d` gap is closed (now `1.14x` faster).

## ML Loader Snapshot (0.2.1)

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
