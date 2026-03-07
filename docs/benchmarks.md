# torchfits Benchmarks

This page documents the benchmark methodology. The snapshot tables below reflect the **0.3.1** release; update with each release run.

## Main Scripts

| Script | Description |
|--------|-------------|
| `bench_all.py` | 4-domain orchestrator: FITS I/O, FITS Table I/O, WCS, Sphere. |
| `bench_fits_io.py` | Authoritative FITS image-domain runner (1D/2D/3D, dtype/size/compression/scaled, MEFs, cutouts, headers). |
| `bench_fitstable_io.py` | Authoritative FITS table-domain runner (row scales, schema mixes, varlen, projection/slice/filter/scan). |
| `bench_wcs_suite.py` | Authoritative WCS projection sweep runner with forward+inverse and `N={1k..10M}` tiers. |
| `bench_sphere_suite.py` | Authoritative sphere-domain runner aggregating geometry/advanced/sparse/spectral/polygon/core benches. |
| `bench_legacy_all.py` | Legacy monolithic exhaustive harness (opt-in only, no longer default contract). |
| `bench_sentinel.py` | Fast 10-case gate with early-stop significance checks for optimization loops. |
| `bench_ml_loader.py` | End-to-end DataLoader throughput benchmark (uncompressed + compressed). |
| `bench_fast.py` | Fast regression loop for image-path changes. |
| `bench_compressed_stability.py` | Randomized-order compressed benchmark for RICE/HCOMPRESS tuning with lower order/cache bias (`library_default` vs forced config modes). |
| `bench_table.py` | FITS table-path read/scan/column access checks. |
| `bench_arrow_tables.py` | Arrow conversion benchmark for table workloads. |
| `bench_wcs.py` | WCS transform throughput checks. |
| `bench_healpix.py` | HEALPix CPU/CUDA parity + throughput + quality gates (`ang2pix`, `pix2ang`, `ring/nest`). |
| `bench_healpix_advanced.py` | Advanced HEALPix parity + throughput checks (`get_all_neighbours`, `get_interp_weights`, `get_interp_val`). |
| `bench_sphere_geometry.py` | Cross-library spherical geometry benchmark (torchfits vs HEALPix ecosystem releases). |
| `bench_sphere_core.py` | Sphere-core primitives benchmark (multi-band sampling, pairwise distances, ellipse query). |
| `bench_sphere_polygons.py` | Non-convex spherical polygon benchmark (contains/query/area with optional spherical-geometry parity). |
| `bench_sphere_spectral.py` | CPU-first spherical harmonic primitive benchmark (`map2alm`, `map2alm_lsq`, `alm2map`, `almxfl`, `alm2cl`, `anafast`, `map2alm_spin`, `alm2map_spin`) plus compatibility-generation paths (`synalm`, `synfast`, `bl2beam`, `beam2bl`) as needed. |
| `bench_pipeline_table_sphere.py` | End-to-end benchmark: table predicate pushdown (`where`) + HEALPix spherical reduction pipeline. |
| `replay_upstream_healpy_interp_edges.py` | Replays interpolation edge cases (lon wrap, poles, pixel-center inputs) against official healpy with parity/perf gates. |
| `replay_upstream_healpy_spin.py` | Replays spin transform parity/throughput against official healpy releases (`map2alm_spin`, `alm2map_spin`). |
| `replay_upstream_spherical_geometry_polygons.py` | Replays spherical-geometry upstream polygon fixtures/data (`test_intersects_*`, `difficult_intersections.txt`) against torchfits with multi-NSIDE difficult-overlap area gates. |

## Standard Commands

```bash
# Contract run (all four domains)
pixi run bench-all

# Exact scope aliases
pixi run bench-fits       # == pixi run bench-all -- --fits-only
pixi run bench-fitstable  # == pixi run bench-all -- --fitstable-only
pixi run bench-wcs        # == pixi run bench-all -- --wcs-only
pixi run bench-sphere     # == pixi run bench-all -- --sphere-only

# Direct CLI (equivalent)
pixi run python benchmarks/bench_all.py --scope all
pixi run python benchmarks/bench_all.py --fits-only
pixi run python benchmarks/bench_all.py --fitstable-only
pixi run python benchmarks/bench_all.py --wcs-only
pixi run python benchmarks/bench_all.py --sphere-only

# Output contract
# benchmarks_results/<run_id>/
#   - results.csv
#   - torchfits_deficits.csv
#   - summary.md

# Fast optimization gate (10 diverse cases, early-stop enabled)
pixi run python benchmarks/bench_sentinel.py --initial-repeats 3 --full-repeats 9 --seed 123

# Compressed-path stability run (randomized config order)
pixi run python benchmarks/bench_compressed_stability.py --rounds 4 --repeats 12 --warmup 3

# ML loader throughput run
pixi run python benchmarks/bench_ml_loader.py --device cpu --shape 2048,2048 --n-files 50 --batch-size 4 --num-workers 4 --epochs 3 --repeats 5 --warm-cache

# WCS parity+throughput run with mixed interior/boundary sampling and p99 gates
pixi run python benchmarks/bench_wcs.py --cases TAN,SIN,ARC,AIT,MOL,HPX,CEA,MER,TAN_SIP,TPV --sample-profile mixed --n-points 200000 --runs 5 --p99-angular-error-arcsec 1e-2 --p99-inverse-pixel-error 1e-2

# HEALPix parity+throughput run on CPU (strict parity gates)
pixi run python benchmarks/bench_healpix.py --device cpu --sample-profile mixed --nside 1024 --n-points 200000 --runs 5 --max-index-mismatches 0 --max-pix2ang-dra-deg 1e-10 --max-pix2ang-ddec-deg 1e-10

# HEALPix CUDA run with CPU comparison gate
pixi run python benchmarks/bench_healpix.py --device cuda --compare-cpu --sample-profile mixed --nside 1024 --n-points 200000 --runs 5 --min-cuda-speedup-vs-cpu 2.0

# HEALPix Apple GPU (MPS) run with CPU comparison (float32 thresholds)
pixi run python benchmarks/bench_healpix.py --device mps --compare-cpu --sample-profile uniform --nside 1024 --n-points 200000 --runs 5 --max-index-mismatches 10 --max-pix2ang-dra-deg 5e-4 --max-pix2ang-ddec-deg 5e-4

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

# Replay astropy-healpix upstream test-style HEALPix cases on torchfits vs healpy
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
- `bench_healpix.py` and `bench_sphere_geometry.py` now print comparator package provenance and fail by default on local/editable/VCS installs.
- `bench_sphere_matrix.py` runs `uniform/boundary/mixed` profiles and can enforce median torchfits/healpy speed-ratio floors.
- `bench_sphere_polygons.py` benchmarks non-convex polygon contains/query/area and optionally checks parity against released `spherical-geometry`.
- `replay_upstream_spherical_geometry_polygons.py` replays polygon fixtures/data from upstream spherical-geometry test suite with correctness gates (contains + pair intersections + difficult-case nonempty parity + multi-NSIDE overlap-area/convergence checks, default ladder `128..16384`).
- `sync_upstream_sphere_sources.py` downloads release artifacts for selected packages and extracts upstream `tests/` + `data/` trees with a pinned manifest.
- `replay_upstream_astropy_healpy.py` replays astropy-healpix test-style HEALPix cases (same NSIDE range and boundary example-style vectors) against torchfits/healpy with correctness + ratio gates.
- `replay_upstream_test_functions.py` executes selected upstream astropy-healpix test functions directly against torchfits adapters for closer semantic parity checks.
- `replay_upstream_healpy_interp_edges.py` replays interpolation edge cases (lon wrap, poles, edge pixels) against healpy with mismatch/error and ratio gates.
- `replay_upstream_healpy_spin.py` replays spin transform cases (`map2alm_spin`, `alm2map_spin`) against healpy with per-case error gates and median throughput ratio thresholds.
- Spin replay defaults to the current validated profile (`TORCHFITS_RING_FOURIER_CPP=1`, recurrence off, `TORCHFITS_SPIN_RING_AUTO_MIN_BYTES=32 MiB`) unless explicitly overridden in the environment.
- Override only for local development experiments with:
  - `bench_healpix.py --allow-nonrelease-healpy`
  - `bench_sphere_geometry.py --allow-nonrelease-distributions`

## Optimization History

For a concise log of attempted optimizations and outcomes (including dead ends), see:
- `docs/performance_attempts.md`

## 0.3.1 Release Snapshot

Run metadata:
- Date: `2026-03-06`
- Exhaustive output: `benchmarks_results/20260306_191113`
- Profile: `user` (`cache=10`, `handle_cache=16`, `hot_cache=10`)
- Tables included: `True`

### Astronomer Scorecard (Win Rates)

| Domain | Family | torchfits First | Win Rate | Legacy In Ranking |
|---|---|---:|---:|---:|
| fits | smart | 83/84 | 98.8% | 0 |
| fits | specialized | 165/165 | 100.0% | 0 |
| fitstable | smart | 70/70 | 100.0% | 0 |
| fitstable | specialized | 70/70 | 100.0% | 0 |
| sphere | specialized | 16/18 | 88.9% | 0 |
| wcs | smart | 121/150 | 80.7% | 75 |

### FITS I/O Representative Medians (vs `fitsio`)

| Case | torchfits median (s) | Fitsio median (s) | Speedup vs Fitsio |
|------|---------------------:|------------------:|------------------:|
| `tiny_int8_1d [read_full]` | 0.000044 | 0.000185 | 4.20x |
| `small_float32_2d [read_full]` | 0.000075 | 0.000212 | 2.82x |
| `medium_int16_3d [read_full]` | 0.000524 | 0.000782 | 1.49x |
| `large_float64_2d [read_full]` | 0.005312 | 0.005470 | 1.03x |

### Known Deficits (Target for 0.4.0)

- **Sphere (Spin Transforms)**: `map2alm_spin` and `alm2map_spin` show ~6x-7x lag vs `healpy`. These will be optimized with a custom RING-Fouier kernel.
- **WCS (Small-N)**: Forward transformations for small coordinate arrays (N < 1000) show lag vs `pyast` (legacy). torchfits is optimized for batch throughput (N > 10k), where it leads by 1.1x - 1.3x.
- **ZPN/AIT Projections**: Slight visible deficits in intermediate size tiers.

## ML Loader Snapshot (0.3.0)

Command used:

```bash
pixi run python benchmarks/bench_ml_loader.py --device cpu --shape 2048,2048 --n-files 50 --batch-size 4 --num-workers 4 --epochs 3 --repeats 5 --warm-cache
```

Results (Median of 5 runs):
- Uncompressed: `0.985x` (vs `fitsio + numpy`)
- Compressed (RICE): `1.008x` (vs `fitsio`)
