# Benchmarks

torchfits is benchmarked for both **correctness** and **throughput** against upstream libraries across four domains. This page documents the methodology, how to reproduce the results, and the latest release snapshot.

## Comparison Targets

| Domain | torchfits module | Compared against |
|---|---|---|
| FITS image I/O | `torchfits.read` / `torchfits.write` | astropy.io.fits, fitsio |
| FITS table I/O | `torchfits.table` | astropy.io.fits, fitsio |
| WCS | `torchfits.wcs` | astropy.wcs, PyAST, Kapteyn |
| HEALPix | `torchfits.sphere` | healpy, hpgeom, astropy-healpix, mhealpy |
| Sparse maps | `torchfits.sphere.sparse` | healsparse |
| Spherical polygons | `torchfits.sphere.geom` | spherical-geometry |
| Spherical harmonics | `torchfits.sphere.spectral` | healpy |

## Methodology

### Throughput

Each case measures **median wall-clock time** over multiple repetitions (typically 3-9 runs). Cases are grouped into two families:

- **smart** &mdash; the idiomatic, high-level API a user would call (e.g. `torchfits.read()` vs `astropy.io.fits.getdata()` + `torch.from_numpy()`). Primary adoption view.
- **specialized** &mdash; lower-level or library-specific call paths with explicit mmap control. Used for controlled I/O comparisons.

Fairness controls:

- **mmap parity** &mdash; rows where libraries have mismatched mmap behavior are marked `SKIPPED` and excluded from rankings. In the current suite fitsio is skipped in FITS I/O and table I/O because its mmap mode is not independently controllable.
- **Comparator provenance** &mdash; sphere-domain comparators must come from official released packages, not local or editable installs.
- **Warm cache** &mdash; FITS benchmarks use configurable file/handle caching (default: `cache=10`, `handle_cache=16`). Cold-start variants available via `--profile cold`.

### Correctness gates

Correctness is validated separately using upstream test fixtures and public reference data. Each gate runs as a pass/fail check:

| Gate | Command | Validates |
|---|---|---|
| WCS parity | `pixi run wcs-upstream-gate` | `pixel_to_world` / `world_to_pixel` vs astropy.wcs across all projections |
| FITS I/O parity | `pixi run fits-upstream-gate` | Image, header, table, compression round-trips vs fitsio test suite |
| HealSparse parity | `pixi run healsparse-upstream-gate` | Sparse map I/O, degrade, geometry vs healsparse test suite |
| HEALPix parity | `pixi run sphere-upstream-gate` | Primitives vs healpy / astropy-healpix test-style cases |
| Spin transforms | `pixi run sphere-upstream-spin-gate` | `map2alm_spin` / `alm2map_spin` vs healpy |
| Interpolation edges | `pixi run sphere-upstream-interp-edge-gate` | Pole, wrap, pixel-boundary edge cases vs healpy |
| Polygon parity | `pixi run sphere-upstream-polygon-gate` | Non-convex polygon fixtures vs spherical-geometry |
| Real-data | `pixi run real-data-gate` | End-to-end on installed FITS fixtures + public Astropy sample image |

### Output format

Every benchmark run produces three artifacts in `benchmarks_results/<run_id>/`:

| File | Contents |
|---|---|
| `results.csv` | One row per (case, library, family) with median time, throughput, rank, lag ratio |
| `torchfits_deficits.csv` | Only cases where torchfits is not first in its family |
| `summary.md` | Scorecard with win rates, adoption checks, and deficit table |

---

## Reproducing

### Full suite

```bash
pixi run bench-all                # all four domains, sequential
```

### Parallel by domain

```bash
pixi run bench-fits &             # FITS image I/O
pixi run bench-fitstable &        # FITS table I/O
pixi run bench-wcs &              # WCS projections
pixi run bench-sphere &           # HEALPix + harmonics + polygons
wait
```

### Partitioned FITS (for large case counts)

```bash
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(tiny_)'
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(small_)'
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(medium_|large_)'
pixi run -e bench-all python benchmarks/bench_all.py --scope fits --filter '^(scaled_|compressed_|mef_)'
```

### Specialized runs

```bash
# Fast regression gate (10 cases, early-stop)
pixi run python benchmarks/bench_sentinel.py --initial-repeats 3 --full-repeats 9

# ML DataLoader throughput
pixi run python benchmarks/bench_ml_loader.py --device cpu --shape 2048,2048 --n-files 50 --batch-size 4 --num-workers 4 --epochs 3 --repeats 5

# Compressed image stability
pixi run python benchmarks/bench_compressed_stability.py --rounds 4 --repeats 12

# Cross-library sphere geometry ecosystem snapshot
pixi run -e sphere-bench sphere-bench-geometry-ecosystem
```

---

## Release Snapshot (0.3.2)

Run ID: `20260318_083620`. Run date: 2026-03-18. Profile: `user`. Platform: macOS arm64 (Apple Silicon).

### Scorecard

| Domain | Family | Cases | torchfits First | Win Rate |
|---|---|---:|---:|---:|
| fits | smart | 84 | 84 | 100.0% |
| fits | specialized | 165 | 165 | 100.0% |
| fitstable | smart | 70 | 70 | 100.0% |
| fitstable | specialized | 70 | 70 | 100.0% |
| wcs | smart | 150 | 128 | 85.3% |
| sphere | specialized | 18 | 16 | 88.9% |

**Large-N leadership** (N &ge; 100K): 100% across all domains. Zero large-N deficits.

Note: fitsio is present in the suite but excluded from FITS and table rankings due to mmap fairness enforcement (its mmap mode is not independently controllable). The comparable pairs are torchfits vs astropy.

---

### FITS Image I/O

Compared against **astropy** (smart family). Coverage: 7 dtypes (int8 &ndash; float64), 1D/2D/3D, 4 size tiers, Rice/HCOMPRESS/GZIP compressed, BSCALE/BZERO scaled, MEF, cutouts, header-only reads.

**Representative `read_full` results (smart family, torchfits vs astropy):**

| Case | torchfits | astropy | Ratio (astropy / tf) |
|---|---:|---:|---:|
| tiny_int16_1d | 73 &mu;s | 6.5 ms | 89x |
| tiny_int32_1d | 31 &mu;s | 2.6 ms | 87x |
| small_float32_1d | 55 &mu;s | 1.1 ms | 19x |
| small_float64_2d | 145 &mu;s | 3.6 ms | 25x |
| medium_float32_1d | 104 &mu;s | 3.7 ms | 35x |
| medium_int16_1d | 125 &mu;s | 2.0 ms | 16x |
| medium_float64_2d | 1.5 ms | 5.8 ms | 3.8x |
| large_float32_1d | 621 &mu;s | 1.6 ms | 2.6x |
| large_int16_2d | 1.3 ms | 4.8 ms | 3.6x |
| large_float64_2d | 6.0 ms | 8.5 ms | 1.4x |
| compressed_rice_1 | 6.6 ms | 19.8 ms | 3.0x |
| compressed_gzip_2 | 7.7 ms | 45.9 ms | 6.0x |
| compressed_hcompress_1 | 21.9 ms | 28.4 ms | 1.3x |

Win rate: **100%** across all 84 smart-family and 165 specialized-family comparable cases. No deficits.

---

### FITS Table I/O

Compared against **astropy** (smart family). Coverage: narrow/wide/mixed/varlen schemas, 1K &ndash; 1M rows, column projection, row slicing, predicate filter, streaming scan.

**Representative results (smart family, torchfits vs astropy):**

| Case | torchfits | astropy | Ratio |
|---|---:|---:|---:|
| narrow_1000 [read_full] | 58 &mu;s | 1.6 ms | 27x |
| narrow_1000000 [read_full] | 131 &mu;s | 22.1 ms | 168x |
| wide_100000 [read_full] | 67 &mu;s | 25.8 ms | 383x |
| mixed_1000 [row_slice] | 57 &mu;s | 3.1 ms | 54x |
| mixed_100000 [row_slice] | 61 &mu;s | 3.9 ms | 63x |
| mixed_1000000 [read_full] | 78 &mu;s | 38.9 ms | 502x |
| varlen_100000 [row_slice] | 67 &mu;s | 8.2 ms | 123x |
| mixed_100000 [scan_count] | 142 &mu;s | 3.0 ms | 21x |
| narrow_1000 [predicate_filter] | 1.3 ms | 2.1 ms | 1.6x |

Win rate: **100%** across all 70 smart-family and 70 specialized-family comparable cases. No deficits.

---

### WCS

Compared against **astropy.wcs** (smart family). PyAST and Kapteyn are available as legacy comparators when bridge data is present but were not included in this run. Coverage: 15 projections &times; forward/inverse &times; N = {1K, 10K, 100K, 1M, 10M}.

**Results at N = 10M (smart family, torchfits vs astropy.wcs):**

| Projection | Forward | Inverse | Forward ratio | Inverse ratio |
|---|---:|---:|---:|---:|
| TAN | 455 ms | 232 ms | 2.3x | 5.2x |
| TAN_SIP | 317 ms | 839 ms | 3.3x | 2.9x |
| TPV | 413 ms | 5.01 s | 2.9x | &mdash; |
| SIN | 301 ms | 320 ms | 2.8x | 3.4x |
| ARC | 396 ms | 460 ms | 2.3x | 2.8x |
| ZPN | 267 ms | 284 ms | 3.1x | 3.8x |
| ZEA | 419 ms | 425 ms | 2.4x | 2.8x |
| STG | 404 ms | 432 ms | 2.3x | 2.9x |
| CEA | 56 ms | 61 ms | 5.6x | 2.8x |
| CAR | 31 ms | 48 ms | 6.4x | 3.4x |
| MER | 77 ms | 98 ms | 5.8x | 4.1x |
| AIT | 521 ms | 224 ms | 1.4x | 2.0x |
| MOL | 263 ms | 521 ms | 1.8x | 16.2x |
| SFL | 80 ms | 60 ms | 3.9x | 2.9x |
| HPX | 200 ms | 123 ms | 1.9x | 2.8x |

**Results at N = 100K:**

| Projection | Forward | Inverse | Forward ratio | Inverse ratio |
|---|---:|---:|---:|---:|
| TAN | 4.6 ms | 2.3 ms | 2.4x | 5.7x |
| TAN_SIP | 4.0 ms | 9.1 ms | 2.6x | 2.5x |
| SIN | 4.8 ms | 4.7 ms | 1.9x | 2.4x |
| ARC | 3.7 ms | 3.6 ms | 2.2x | 2.8x |
| ZPN | 3.0 ms | 3.2 ms | 2.6x | 3.2x |
| ZEA | 8.3 ms | 8.2 ms | 1.3x | 1.5x |
| CEA | 442 &mu;s | 483 &mu;s | 7.0x | 3.1x |
| CAR | 228 &mu;s | 340 &mu;s | 8.4x | 3.8x |
| HPX | 1.9 ms | 1.0 ms | 1.8x | 2.9x |

Win rate: **85.3%** overall. At N &ge; 100K: **100%** (90/90 cases).

**Deficits:** 22 deficits total: 21 at N = 1K and 1 at N = 10K (ZEA forward, 2.9x). PyTorch dispatch overhead dominates at small N. No deficits at N &ge; 100K.

---

### Sphere

Compared against **healpy**, **hpgeom**, **astropy-healpix**, **healsparse**, and **spherical-geometry** (specialized family). Coverage: `ang2pix` (ring/nested), `pix2ang`, `nest2ring`, `ring2nest`, neighbors, interpolation weights/values, sparse map operations, non-convex polygon contains/area/query, and spin-weighted harmonic transforms.

**HEALPix geometry (CPU):**

| Operation | torchfits | healpy | Ratio (healpy / tf) |
|---|---:|---:|---:|
| ang2pix ring | 2.6 ms | 5.6 ms | 2.1x |
| ang2pix nested | 3.9 ms | 5.8 ms | 1.5x |
| pix2ang ring | 2.2 ms | 3.2 ms | 1.5x |
| pix2ang nested | 3.5 ms | 4.1 ms | 1.2x |
| ring2nest | 1.4 ms | 1.9 ms | 1.4x |
| nest2ring | 1.8 ms | 1.8 ms | 1.0x |

**Advanced HEALPix:**

| Operation | torchfits | healpy | Ratio |
|---|---:|---:|---:|
| neighbors nested | 1.6 ms | 4.1 ms | 2.6x |
| neighbors ring | 6.7 ms | 9.4 ms | 1.4x |
| interp_weights nested | 20.5 ms | 29.2 ms | 1.4x |
| interp_weights ring | 14.3 ms | 17.2 ms | 1.2x |
| interp_val nested | 33.4 ms | 35.9 ms | 1.1x |
| interp_val ring | 18.4 ms | 26.1 ms | 1.4x |

**Spherical polygons:**

| Operation | torchfits | spherical-geometry | Ratio |
|---|---:|---:|---:|
| contains (non-convex, N points) | 9.2 ms | 28.2 s | 3072x |

**Sparse maps:**

| Operation | torchfits | dense CPU baseline | Ratio |
|---|---:|---:|---:|
| ud_grade | 24.8 ms | 60.5 ms | 2.4x |

**Spherical harmonics:**

| Operation | torchfits | healpy | Ratio (healpy / tf) |
|---|---:|---:|---:|
| map2alm_spin | 1.1 ms | 170 &mu;s | 0.2x |
| alm2map_spin | 926 &mu;s | 111 &mu;s | 0.1x |

Win rate: **88.9%** (16/18). Large-N: **100%** (13/13). Two deficits listed below.

---

### Known Deficits

| Domain | Case | Lag | Notes |
|---|---|---:|---|
| wcs | N &le; 10K forward/inverse (22 cases) | 1.0x &ndash; 2.9x | Fixed PyTorch dispatch overhead at small N (21 at N = 1K, 1 at N = 10K). All under 3 ms absolute. |
| sphere | `map2alm_spin` | 6.3x | Spin transform not yet optimized; healpy uses compiled C. Target for 0.4.x. |
| sphere | `alm2map_spin` | 8.3x | Same as above. |

---

## Benchmark Scripts

| Script | Domain | Description |
|---|---|---|
| `bench_all.py` | all | Four-domain orchestrator |
| `bench_fits_io.py` | fits | Image I/O across dtypes, sizes, compression |
| `bench_fitstable_io.py` | fitstable | Table I/O across row counts, schemas, operations |
| `bench_wcs_suite.py` | wcs | Projection sweep, forward + inverse, N = 1K &ndash; 10M |
| `bench_sphere_suite.py` | sphere | Aggregates geometry, advanced, sparse, spectral, polygon benchmarks |
| `bench_sentinel.py` | fits | Fast 10-case regression gate |
| `bench_ml_loader.py` | fits | End-to-end DataLoader throughput |
| `bench_compressed_stability.py` | fits | Randomized compressed-image stability |
| `bench_wcs.py` | wcs | WCS throughput with parity gates |
| `bench_healpix.py` | sphere | HEALPix CPU/CUDA/MPS parity + throughput |
| `bench_healpix_advanced.py` | sphere | Neighbors + interpolation benchmarks |
| `bench_sphere_geometry.py` | sphere | Cross-library HEALPix ecosystem comparison |
| `bench_sphere_core.py` | sphere | Multi-band sampling, pairwise distances, ellipse queries |
| `bench_sphere_polygons.py` | sphere | Non-convex polygon operations vs spherical-geometry |
| `bench_sphere_spectral.py` | sphere | Spherical harmonic primitives (scalar + spin) |
| `bench_pipeline_table_sphere.py` | mixed | Table pushdown + HEALPix reduction pipeline |

### Upstream replay scripts

| Script | Upstream | What it replays |
|---|---|---|
| `replay_upstream_astropy_wcs.py` | astropy.wcs | Fixture-heavy WCS projection tests |
| `replay_upstream_fits_workflows.py` | astropy, fitsio | README/docs example workflows |
| `replay_upstream_astropy_healpy.py` | astropy-healpix | HEALPix test-style primitives |
| `replay_upstream_test_functions.py` | astropy-healpix | Selected upstream test functions via adapters |
| `replay_upstream_healpy_interp_edges.py` | healpy | Interpolation edge cases (poles, wrap, boundaries) |
| `replay_upstream_healpy_spin.py` | healpy | Spin transform parity + throughput |
| `replay_upstream_spherical_geometry_polygons.py` | spherical-geometry | Polygon fixtures, difficult intersections, multi-NSIDE area convergence |
| `replay_upstream_healsparse.py` | healsparse | I/O, degrade, geometry lifecycle |
| `replay_real_data_validation.py` | astropy, fitsio, healpy | Installed FITS fixtures + public reference image |

## Sphere Benchmark Policy

- Comparator libraries must come from official released distributions (pip/conda), not local or editable installs.
- `bench_healpix.py` and `bench_sphere_geometry.py` print comparator provenance and reject local/VCS installs by default. Override with `--allow-nonrelease-healpy` or `--allow-nonrelease-distributions` for development.
- Spin replay uses the validated profile (`TORCHFITS_RING_FOURIER_CPP=1`, recurrence off) unless overridden.

## Optimization History

Past optimization attempts and their outcomes are tracked in commit history and PR descriptions.
