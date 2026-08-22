# Benchmarks

`torchfits` benchmarks cover FITS **tensor** I/O (IMAGE HDUs, typically 1D–4D)
and FITS **table** I/O vs Astropy and fitsio across CPU and GPU hardware.

**Honesty:** Headline ratios below are lab medians from reproducible benchmark runs across our test suites — not guarantees on your specific filesystem, file mix, or PyTorch version. Check [Performance comparisons & limitations](#performance-deficits) for a transparent breakdown of cases where peer libraries are competitive or faster.

## How to read this page

| If you want… | Jump to |
|---|---|
| Headline wins | [Performance highlights](#performance-highlights) |
| Cases where torchfits is not #1 (CPU and GPU) | [Performance comparisons & limitations](#performance-deficits) |
| GPU transport rows | [I/O transport and backend](#io-transport-and-backend) |
| Python × PyTorch version variance | [Version matrix variance](#python-pytorch-matrix-variance) |
| Reproduce numbers | [Reproducing](#reproducing) |
| Every measured configuration | [Exhaustive benchmark results](#exhaustive-benchmark-results) |
| Raw CSV | [Published CSVs](#published-csvs) |

Published GPU/CPU numbers come from the multi-host release scorecard
(`exhaustive_mps_*`, `exhaustive_cpu_*`, `exhaustive_cuda_*`). Manual
`workflow_dispatch` on `.github/workflows/bench-report.yml` is CPU-only and
does not refresh GPU cells.

## Comparison targets

| Domain | torchfits API | Compared against |
|---|---|---|
| Tensor (IMAGE HDU) | `read` / `read_tensor` / `write` | `astropy.io.fits`, `fitsio` |
| Table (dataframe) | `torchfits.table` | `astropy.io.fits`, `fitsio` |

## Methodology

Each case measures median wall-clock time over multiple repetitions, plus
**peak process RSS** (and peak CUDA alloc when on CUDA). Performance ranking is
**time-based**; RSS is reported alongside times.

Cases are grouped into two families:

- **default** — high-level API (`torchfits.read` / `table.read`, etc.).
- **specialized** — `torchfits_specialized` methods (open-once handle /
  `open_subset_reader` paths). Empty specialized cells mean that path was not
  measured for the case.

Fairness controls:

- Rows with mismatched mmap behavior are marked `SKIPPED` and excluded from
  rankings.
- **Why fitsio has no mmap rows:** fitsio does not expose a comparable mmap
  toggle. Under `mmap_target=on` / `strict_mmap_fairness`, fitsio rows are
  non-comparable and show as skipped in transport tables (see
  `scripts/render_bench_iopath_table.py`).
- Warm-cache and cold-cache profiles are kept separate.

### Disk to GPU

True **disk→GPU** (GPUDirect Storage / cuFile, or a CFITSIO path that never
touches host RAM) is **not** implemented. Every Python FITS stack here
decodes on the host, then copies with `.to(device)`. Exploring a direct path
is a **1.1** candidate (see [Roadmap](roadmap.md)) — not a 1.0 claim.

### Tables on GPU transports

Table GPU transport rows compare `table.read_torch(..., device=cpu)` against
`device=cuda` / `device=mps` on a medium mixed catalog case
(`mixed_100000`). Decode still happens on the host; the GPU column measures
host decode plus H2D copy into tensor columns.

### CUDA Host-to-Device Transfer & Small Payloads

For small tensor payloads (e.g. 1D arrays and small $64 \times 64$ sub-regions), fixed kernel launch latency and Host-to-Device (H2D) memory transfer dominate over raw decode throughput.

In `torchfits`, the C++ engine optimizes memory transfers by coordinating host buffers and asynchronous CUDA streams:
- For larger images, direct memory transfers match peak PCIe bus bandwidth.
- For small payloads, latency remains competitive with in-memory transfers, operating at parity with baseline libraries on NVIDIA CUDA and Apple Silicon MPS.

### Vectorized SIMD Integer Decoding (Unsigned Integers & BZERO)

Standard astronomical FITS stores unsigned 16-bit and 32-bit integers using signed formats paired with standard `BZERO` offsets ($y = \text{raw} + 32768$).

`torchfits` fuses big-endian byte-swapping and `BZERO` offset calculations directly into vectorized SIMD loops within the C++ engine:
- Eliminates secondary scalar normalization passes over memory.
- Delivers up to $3\times$ speedups on large `uint16` and `uint32` image arrays compared to two-stage Python conversions.

## Python & PyTorch Matrix Variance

The headline benchmarks on this page are reported using the Pareto-optimal (champion) environment combination measured across our CANFAR exhaustive matrix: **PyTorch 2.12 + Python 3.11**.

Below is the measured performance variance across the full matrix grid (**Python 3.10–3.14** × **PyTorch 2.10–2.13** × **CPU / CUDA**), tracking the average latency delta and overhead relative to the champion configuration.

### Summary: Average Performance Overhead vs Champion

- **CPU Host Workloads:**
  - **Optimal Baseline:** PyTorch 2.12 + Python 3.11 (Geometric-mean latency: **0.106 ms**)
  - **Average Python Variance:** Across all Python versions (3.10–3.14), average latency penalty is **+9.5%** (+7.6% on 3.10, +11.2% on 3.11, +10.5% on 3.12, +8.4% on 3.13, +12.8% on 3.14).
  - **Average PyTorch Variance:** Across PyTorch minor versions (2.10–2.13), average latency penalty is **+9.8%** (+10.7% on 2.10, +12.3% on 2.11, +7.4% on 2.12, +9.9% on 2.13).

- **CUDA Workloads (NVIDIA GPU):**
  - **Optimal Baseline:** PyTorch 2.12 + Python 3.11 (Geometric-mean latency: **0.187 ms**)
  - **Average Python Variance:** Across all Python versions (3.10–3.14), average latency penalty is **+5.8%** (+8.6% on 3.10, +4.0% on 3.11, +5.0% on 3.12, +4.9% on 3.13, +6.3% on 3.14).
  - **Average PyTorch Variance:** Across PyTorch minor versions (2.10–2.13), average latency penalty is **+5.7%** (+3.7% on 2.10, +4.3% on 2.11, +3.9% on 2.12, +11.0% on 2.13).

### Full Matrix Benchmark Comparison

#### CPU Transport Matrix

| PyTorch | Python | Device | Geom Mean (ms) | Delta vs Best | Relative Perf |
|---|---|---|---:|---:|---:|
| 2.12 | 3.11 | CPU | 0.106 | **Baseline (Best)** | **1.00×** |
| 2.10 | 3.10 | CPU | 0.108 | +2.0% slower | 1.02× |
| 2.13 | 3.12 | CPU | 0.111 | +5.0% slower | 1.05× |
| 2.11 | 3.10 | CPU | 0.113 | +6.4% slower | 1.06× |
| 2.11 | 3.13 | CPU | 0.113 | +6.4% slower | 1.06× |
| 2.12 | 3.13 | CPU | 0.113 | +6.6% slower | 1.07× |
| 2.12 | 3.10 | CPU | 0.114 | +7.3% slower | 1.07× |
| 2.13 | 3.13 | CPU | 0.114 | +7.3% slower | 1.07× |
| 2.12 | 3.14 | CPU | 0.114 | +7.4% slower | 1.07× |
| 2.13 | 3.14 | CPU | 0.115 | +8.6% slower | 1.09× |
| 2.10 | 3.12 | CPU | 0.116 | +8.9% slower | 1.09× |
| 2.10 | 3.14 | CPU | 0.117 | +10.2% slower | 1.10× |
| 2.11 | 3.11 | CPU | 0.119 | +11.7% slower | 1.12× |
| 2.11 | 3.12 | CPU | 0.119 | +11.9% slower | 1.12× |
| 2.10 | 3.13 | CPU | 0.120 | +13.2% slower | 1.13× |
| 2.13 | 3.11 | CPU | 0.121 | +13.9% slower | 1.14× |
| 2.13 | 3.10 | CPU | 0.122 | +14.9% slower | 1.15× |
| 2.12 | 3.12 | CPU | 0.123 | +16.0% slower | 1.16× |
| 2.10 | 3.11 | CPU | 0.126 | +19.2% slower | 1.19× |
| 2.11 | 3.14 | CPU | 0.132 | +24.9% slower | 1.25× |

#### CUDA Transport Matrix

| PyTorch | Python | Device | Geom Mean (ms) | Delta vs Best | Relative Perf |
|---|---|---|---:|---:|---:|
| 2.12 | 3.11 | CUDA | 0.187 | **Baseline (Best)** | **1.00×** |
| 2.10 | 3.14 | CUDA | 0.189 | +1.1% slower | 1.01× |
| 2.10 | 3.13 | CUDA | 0.190 | +1.8% slower | 1.02× |
| 2.10 | 3.12 | CUDA | 0.190 | +1.8% slower | 1.02× |
| 2.12 | 3.13 | CUDA | 0.191 | +1.9% slower | 1.02× |
| 2.11 | 3.12 | CUDA | 0.193 | +3.1% slower | 1.03× |
| 2.11 | 3.14 | CUDA | 0.193 | +3.3% slower | 1.03× |
| 2.12 | 3.12 | CUDA | 0.195 | +4.1% slower | 1.04× |
| 2.11 | 3.11 | CUDA | 0.196 | +5.0% slower | 1.05× |
| 2.11 | 3.13 | CUDA | 0.197 | +5.1% slower | 1.05× |
| 2.10 | 3.11 | CUDA | 0.197 | +5.2% slower | 1.05× |
| 2.11 | 3.10 | CUDA | 0.197 | +5.2% slower | 1.05× |
| 2.12 | 3.10 | CUDA | 0.197 | +5.4% slower | 1.05× |
| 2.13 | 3.11 | CUDA | 0.198 | +5.8% slower | 1.06× |
| 2.12 | 3.14 | CUDA | 0.202 | +8.1% slower | 1.08× |
| 2.10 | 3.10 | CUDA | 0.203 | +8.8% slower | 1.09× |
| 2.13 | 3.12 | CUDA | 0.207 | +10.8% slower | 1.11× |
| 2.13 | 3.13 | CUDA | 0.207 | +10.9% slower | 1.11× |
| 2.13 | 3.14 | CUDA | 0.211 | +12.6% slower | 1.13× |
| 2.13 | 3.10 | CUDA | 0.215 | +15.1% slower | 1.15× |

## Published Benchmark Data {#published-csvs}

Exhaustive benchmark datasets and analysis CSVs (`results.csv`, `torchfits_deficits.csv`) are published with each release and mirrored under `docs/assets/bench/<run-id>/`:

- [`exhaustive_cpu_20260807_013736/results.csv`](assets/bench/exhaustive_cpu_20260807_013736/results.csv)
- [`exhaustive_cuda_20260807_013736/results.csv`](assets/bench/exhaustive_cuda_20260807_013736/results.csv)
- [`exhaustive_mps_20260719_143706/results.csv`](assets/bench/exhaustive_mps_20260719_143706/results.csv)
- [`exhaustive_cpu_20260719_144337/results.csv`](assets/bench/exhaustive_cpu_20260719_144337/results.csv)
- [`exhaustive_cuda_20260719_144457/results.csv`](assets/bench/exhaustive_cuda_20260719_144457/results.csv)

### Modular Suites & Release Exhaustives

Named suites live in `benchmarks/suites.py` and resolve to `bench_all.py` flags
(`--scope` / `--filter` / `--operation` / GPU / mmap / profile):

```bash
pixi run bench-suite hcompress
pixi run bench-suite compressed_rice -- --no-mmap
pixi run bench-suite fitstable_predicate
pixi run bench-deficit-focus          # registry-driven deficit clusters
```

Release composition is the `release` suite (full fits + fitstable, mmap matrix,
GPU when present). Host recipes:

| Task | Host | Run ID prefix |
|---|---|---|
| `pixi run bench-exhaustive-local` | Mac CPU + MPS | `exhaustive_mps_*` |
| `pixi run bench-exhaustive-canfar-cpu` | CANFAR multicore CPU | `exhaustive_cpu_*` |
| `pixi run bench-exhaustive-canfar-cuda` | CANFAR CUDA | `exhaustive_cuda_*` |
| `pixi run bench-release-scorecard -- <run_dir>...` | meta | patches multi-host docs |
| `pixi run bench-cfitsio-direct` | local C | full-suite pure vendored CFITSIO (`--profile full`) |
| `pixi run bench-megacam` | local | CFHT MegaCam MEF cutouts (requires fetched sample data) |
| `pixi run bench-ml` | local | PyTorch DataLoader throughput vs fitsio |

### CFHT MegaCam Cutout Suite

Public CFHT MegaCam MEF samples (CADC Direct Data Service) exercise **Rice
`.fz`** repeated cutouts with peer ranking:

| Method | Role |
|---|---|
| `torchfits_cached` / `fitsio_cached` | Open once + N× subset (comparable family) |
| `torchfits_materialize` | Decompress plane once, then host slices (isolates Rice vs cutout API) |
| `torchfits_naive` | Re-open per cutout (pathological baseline; not ranked) |

Uses `ZNAXIS*` for tile-compressed sizes; throughput is cutout **payload** MB/s.

```bash
bash scripts/fetch_cfht_megacam_sample.sh   # once; idempotent
pixi run bench-megacam
```

Outputs land in `benchmarks_results/<run-id>/megacam_results.csv`.

On multi-extension CFHT MegaCam exposures (40 cutouts $\times 256 \times 256$ per HDU), `torchfits_cached` outperforms `fitsio_cached` by 7.5%–15.2% across sampled HDUs due to optimized tile decompression handles.

For **uncompressed** survey mosaics (e.g. CFHTLS MegaPipe float32 stacks),
`open_subset_reader` maps the data segment once and slices cutouts with
endian swap into torch tensors — see
[ML with FITS](examples-ml.md#survey-mosaic-cutouts-cfht-megapipe). Rice `.fz`
MegaCam cutouts remain a separate comparison (tile decompress inside CFITSIO).

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
pixi run bench-ml
bash scripts/fetch_cfht_megacam_sample.sh && pixi run bench-megacam
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

Named **focused-benchmark** recipes (mmap on+off, no unrelated GPU matrix when scoped to tables):

```bash
pixi run bench-deficit-focus              # hcompress + tiny_int8 + narrow predicates
pixi run bench-deficit-focus hcompress
pixi run bench-deficit-focus tiny_int8
pixi run bench-deficit-focus predicate
```

Rankings and comparisons group by `(domain, case_id, family, mmap_target)` so
mmap-on and mmap-off peers are never cross-compared.
## Benchmark Scripts

| Script | Domain | Description |
|---|---|---|
| `bench_all.py` | fits / fitstable | FITS benchmark orchestrator |
| `bench_fits_io.py` | fits | Image I/O across dtypes, sizes, compression, scaling, MEF, and cutouts |
| `bench_fitstable_io.py` | fitstable | Table I/O across row counts, schemas, projection, row slicing, predicates, and streaming |
| `bench_all.py` / `bench-fits` | fits | Scorecard path |
| `bench_table.py` | fitstable | Table API timing |
| `bench_arrow_tables.py` | fitstable | Arrow-oriented table workflows |
| `bench_gpu_transports.py` | fits (GPU) | CUDA/MPS image reads, cutouts, repeated cutouts (`disk→CPU→GPU` / `disk→RAM→GPU` rows) |
| `bench_ml_loader.py` | fits (diagnostic) | PyTorch `DataLoader` throughput (not merged into `bench-all` CSV) |
| `bench_gpu_memory.py` | fits (diagnostic) | GPU memory/leak checks (non-gating) |
| `bench_denoise.py` | ml (scientific) | Noise2Noise CR-cleaning on real CFHT MegaCam frames (dark→blank framing, `torchfits` loaders vs Astropy; see [denoise-pipeline.md](denoise-pipeline.md)) |

## Coverage matrix

What the exhaustive `bench-all` suite measures today, and what is intentionally out of
scope or not yet wired into the published tables.

| Dimension | Covered? | Where | Gap / caveat |
|---|---|---|---|
| Backends (torchfits / astropy / fitsio) | Yes | `bench_fits_io.py`, `bench_fitstable_io.py` | `fitsio` often excluded from mmap-fairness summaries; **uint** image comparators may be torchfits-only when astropy requires buffered fallback |
| CPU vs GPU device | Partial | CPU: full matrix; GPU: tensor reads | GPU requires CUDA/MPS (`pixi run -e bench-gpu`); manual CI bench is CPU-only |
| I/O transport `disk→RAM→CPU` | Yes | `bench-all` mmap-on pass | Median mixes many ops/sizes — coarse aggregate |
| I/O transport `disk→CPU` (non-mmap) | Yes | `bench-all --mmap-matrix` mmap-off pass | Buffered host decode |
| I/O transport `disk→RAM→GPU` | Partial | `bench_gpu_transports.py` (mmap on) | Tensor `read_full`, cutouts, repeated cutouts; tables until suite lands |
| I/O transport `disk→CPU→GPU` | Partial | `bench_gpu_transports.py` (mmap off) | Same with buffered host decode + H2D |
| I/O transport `disk→GPU` | No | — | No host-bypass path yet (see Methodology); 1.1 candidate |
| BITPIX / dtypes | Partial | int8–int64, float32/64 × 1D/2D/3D | Native **uint16/uint32** 2D sample datasets; unsigned via BZERO in `scaled_*` |
| Tensor dimensions / sizes | Yes | tiny → large; 1D–3D (4D where sample datasets exist) | Large 3D cubes may hit size caps |
| Compression (read) | Yes | gzip, rice, hcompress, plio | Write→compress cases are being added to the suite |
| Scaling (BSCALE/BZERO) | Yes | `scaled_small/medium/large` | Table-column scaling not isolated |
| Random / repeated access | Yes | cutouts, `random_ext_full_reads_200`, `open_subset_reader` | MEF random ext reads on selected sample datasets |
| Multi-extension (MEF) | Yes | `mef_*`, `multi_mef_10ext`, MegaCam suite | — |
| Table full read / projection / slice | Yes | `bench_fitstable_io.py` | — |
| Table predicate / scan | Yes | `predicate_filter` (dense ~50% keep), `predicate_filter_selective` (~5–7%), `scan_count` | Both keep-rate regimes; fused gather ≠ project+mask |
| Table schemas | Partial | mixed / narrow / wide / varlen | typed / ascii at selected row counts |
| Table GPU vs CPU | Partial | GPU transports / fitstable | Expanding into published tables |
| Writes / write→compress | Partial | suite expansion | Read-heavy historically; write parity also in tests |
| ML DataLoader | Yes | `bench_ml_loader.py` | Reported in highlights / dedicated section |

### Why the I/O transport table looks sparse on GPU

1. **`disk→GPU` is always empty** — backends decode on the host first, then
   `.to(device)`. See [Disk to GPU](#disk-to-gpu).
2. **`disk→CPU→GPU` vs `disk→RAM→GPU`** — mmap-off vs mmap-on host decode + H2D.
3. **GPU rows need CUDA/MPS hardware** — published CUDA numbers come from
   CANFAR staging (`exhaustive_cuda_20260807_013736`, refreshed 2026-08-07).
4. **Tables** — see [Tables on GPU transports](#tables-on-gpu-transports).

### GPU integer dtype comparisons

The **deficit table** compares default
`torchfits.read(..., scale_on_device=True)` against
`torch.from_numpy(fitsio.read(...)).to(cuda)`. That pairing is not
dtype-equivalent for every scaled integer FITS file.

| FITS convention | fitsio @ CUDA | default `read` @ CUDA |
|---|---|---|
| Signed byte (BITPIX=8, BZERO=-128) | native `int8` H2D | narrow `int8` H2D + offset on device |
| Unsigned uint16/uint32 (BZERO) | native uint H2D | narrow storage H2D, offset on device |
| Generic BSCALE/BZERO | often native storage | `float32` on device (ML-friendly) |

For apples-to-apples integer GPU timing, the suite also records
`torchfits_dtype_fair_device` (`read_tensor(..., raw_scale=True)`).

**Training loops:** call
`torchfits.cache.optimize_for_dataset(paths, avg_file_size_mb=…)` before
`DataLoader` epochs so handle caches stay warm.

### Refreshing GPU numbers (CANFAR staging)

CUDA lab numbers come from a headless GPU session on `@staging`. From a
machine with `canfar` x509 auth:

```bash
bash scripts/selfcheck_canfar_launcher.sh
TORCHFITS_CANFAR_IMAGE=astroai/notebook:latest TORCHFITS_BENCH_MODE=exhaustive \
  pixi run bench-canfar-gpu
bash scripts/fetch_canfar_bench_vos.sh exhaustive_cuda_<stamp>
bash scripts/patch_canfar_exhaustive_docs.sh exhaustive_cuda_<stamp>
```

```bash
# Local CI + docs before push
bash scripts/ci_local.sh
# Apple Silicon (MPS transport rows)
pixi run bench-mps
```

## I/O transport and backend

> **GPU summary:** Tensor **`disk→CPU→GPU`** / **`disk→RAM→GPU`** rows appear
> only when the CSV was produced on CUDA or MPS. **`disk→GPU`** stays empty
> (unsupported). Table GPU cells stay empty until the table-GPU suite lands.


<!-- BENCH_IOPATH_BEGIN -->
Source: `benchmarks_results/exhaustive_cpu_20260822_152204/results.csv` (mmap on+off matrix.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### Tensor I/O (IMAGE HDU) (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.12 ms` (n=174) | `0.72 ms` (n=253) | `0.20 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.13 ms` (n=174) | `0.64 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### Table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.19 ms` (n=216) | `2.58 ms` (n=184) | `0.59 ms` (n=216) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.17 ms` (n=216) | `2.55 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |
<!-- BENCH_IOPATH_END -->

### Notes on the layout

- Rows are **I/O transports** (`disk→CPU`, `disk→RAM→CPU`, `disk→GPU`,
  `disk→CPU→GPU`, `disk→RAM→GPU`).
- Columns are **backends** (`torchfits` / `astropy` / `fitsio` / `cfitsio-direct`).
- Pure-C CFITSIO (vendored): `pixi run bench-cfitsio-direct` runs the **full**
  image+table scorecard fixture set with op→API mapping in
  `benchmarks/cfitsio_direct/bench_cfitsio_direct.c`
  (`fits_read_img` / `fits_read_subset` / `fits_read_record` /
  `fits_read_tblbytes` / `fits_read_col`). CSV:
  `benchmarks_results/<run-id>/cfitsio_direct.csv`.
- Cell `n=` counts comparable OK rows in the bucket; `—` indicates the
  bucket is empty (no rows match, or rows were excluded under
  `strict_mmap_fairness` in the original `bench-all` summary).
- Median is computed over heterogeneous operations (`read_full`,
  `cutout_100x100`, `header_read`, `predicate_filter`, `projection`,
  `row_slice`, etc.) and payload sizes; treat the per-cell ms as a
  coarse representative number, not a precise benchmark.

## Performance highlights

<!-- BENCH_HIGHLIGHTS_BEGIN -->
The following table showcases median wall-clock times for key FITS tensor and table cases. The **specialized** column is `torchfits_specialized` (open-once / subset-reader paths); it is empty when that path was not measured.

| Benchmark Case | Device | torchfits | torchfits (specialized) | astropy (via torch) | fitsio (via torch) | Win vs Astropy | Win vs fitsio |
|---|---|---:|---:|---:|---:|---:|---:|
| Table read (100k rows, 8 cols, mixed) | CPU | **2.16 ms** | 2.06 ms | 31.01 ms | 10.14 ms | **15.03x** | **4.91x** |
| Varlen table read (100k rows, 3 cols) | CPU | **70.64 ms** | 71.02 ms | 552.80 ms | 113.05 ms | **7.83x** | **1.60x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Benchmark category summary

CPU category rows aggregate the CPU exhaustive
(`exhaustive_cpu_20260807_082931_reader_cache`, source of the generated
[highlights](#performance-highlights) and [full table](#exhaustive-benchmark-results)
above); the GPU (CUDA) rows come from `exhaustive_cuda_20260807_013736`
(see host scorecard for deficit honesty — all lags listed, floors label
noise vs significant). Category ranges are the last regenerated aggregation
shape; for absolute times prefer the generated tables above.

### FITS image I/O

| Category | Cases | torchfits median | astropy median | fitsio median | Typical speedup vs astropy | Typical speedup vs fitsio |
|---|---:|---:|---:|---:|---:|---:|
| **1D** (float32/64, int8–int64, tiny–large) | 24 | 29 μs – 1.28 ms | 302 μs – 2.57 ms | 61 μs – 1.69 ms | **2.0–13.5×** | **1.30–2.4×** |
| **2D** (float32/64, int8–int64, uint16/32, tiny–large) | 30 | 37 μs – 7.07 ms | 361 μs – 13.10 ms | 75 μs – 8.92 ms | **1.8–12.9×** | **1.26–2.2×** |
| **3D** (float32/64, int8–int64, tiny–medium) | 18 | 45 μs – 1.96 ms | 423 μs – 4.35 ms | 83 μs – 2.98 ms | **2.2–15.4×** | **1.41–2.1×** |
| **Compressed** (gzip, hcompress, rice) | 5 | 1.33–45.56 ms | 11.43–72.27 ms | 1.38–44.34 ms | **1.1–8.6×** | **0.58–1.1×** |
| **Scaled** (BSCALE/BZERO, small–large) | 3 | 76 μs – 2.93 ms | 496 μs – 6.09 ms | 128 μs – 3.87 ms | **2.1–6.6×** | **1.32–1.7×** |
| **MEF** (multi-extension, small/medium) | 2 | 69–220 μs | 614–954 μs | 155–342 μs | **4.3–8.9×** | **1.55–2.2×** |
| **Multi-MEF** (10 extensions, cutouts + random reads) | 3 | 64 μs – 8.15 ms | 634 μs – 12.01 ms | 194 μs – 11.04 ms | **1.5–40.4×** | **1.36–3.0×** |
| **Repeated cutouts** (50× 100×100) | 1 | 698 μs | 88.49 ms | 5.33 ms | **126.7×** | **7.63×** |
| **Time series frames** (5 frames) | 5 | 64–91 μs | 492–663 μs | 143–195 μs | **6.5–7.8×** | **1.95–2.3×** |
| **Header read** (all fixture types) | 87 | 14–51 μs | 257 μs – 2.46 ms | 25–267 μs | **15.1–48.2×** | **1.58–5.2×** |

**GPU (CUDA) device lanes** — 85 comparable `read_full` / cutout cases:

| Category | torchfits median | astropy median | fitsio median | Typical speedup vs astropy | Typical speedup vs fitsio |
|---|---:|---:|---:|---:|---:|
| **1D** (tiny–large) | 107 μs – 1.57 ms | 438 μs – 2.90 ms | 106 μs – 2.05 ms | **1.8–6.8×** | **0.88–1.5×** |
| **2D** (tiny–large) | 114 μs – 11.92 ms | 450 μs – 22.88 ms | 112 μs – 13.09 ms | **1.9–6.9×** | **0.91–1.5×** |
| **3D** (tiny–medium) | 111 μs – 2.76 ms | 479 μs – 5.75 ms | 110 μs – 3.55 ms | **1.8–7.4×** | **0.95–1.6×** |
| **Compressed** (gzip, hcompress, rice) | 989 μs – 30.66 ms | 9.63–67.38 ms | 1.03–29.60 ms | **1.2–9.7×** | **0.97–1.1×** |
| **Scaled** | 198 μs – 4.34 ms | 920 μs – 10.93 ms | 203 μs – 4.91 ms | **1.9–4.6×** | **1.03–1.1×** |
| **MEF + Multi-MEF** | 143–391 μs | 1.14–2.80 ms | 173–451 μs | **4.0–17.0×** | **1.16–1.7×** |
| **Repeated cutouts (GPU)** | 1.47 ms | 61.22 ms | 6.19 ms | **41.7×** | **4.21×** |

### FITS table I/O

| Category | Cases | torchfits median | astropy median | fitsio median | Typical speedup vs astropy | Typical speedup vs fitsio |
|---|---:|---:|---:|---:|---:|---:|
| **read_full** (all schemas incl. 1M rows, varlen) | 18 | 175 μs – 74.74 ms | 2.40–681.18 ms | 0.26–485.33 ms | **1.3–32×** | **0.73–22×** |
| **projection** (column subset) | 18 | 171 μs – 72.73 ms | 2.29–526.19 ms | 0.30–107.85 ms | **1.4–38×** | **1.48–13×** |
| **row_slice** (row range) | 18 | 113 μs – 6.96 ms | 1.99–59.12 ms | 0.25–31.15 ms | **7.8–55×** | **1.55–17×** |
| **predicate_filter** (WHERE clause, dense + selective) | 36 | 71 μs – 11.13 ms | 1.51–2.64 ms | 0.12–20.44 ms | **6.9–11×** | **0.98–3×** |
| **scan_count** (streaming) | 18 | 26–65 μs | 0.40–0.98 ms | 0.06–0.43 ms | **14.4–17×** | **2.12–7×** |

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured configurations. Empty cells mean that method was not run for the case (for example `torchfits_specialized` is only used for open-once / subset-reader paths). Domain `tensor` = IMAGE HDU payloads (1D–4D); `table` = binary/ASCII tables.

| Domain | Benchmark Case | Operation | Size | Device | mmap | torchfits | torchfits (specialized) | astropy (via torch) | fitsio (via torch) | cfitsio (direct) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| tensor | compressed_gzip_1:header_read | header_read | 1.29 MB | CPU | n/a | **—** | 30.9 μs | 1.39 ms | 138.7 μs | — | **44.90x** | **4.49x** |
| tensor | compressed_gzip_2:header_read | header_read | 0.89 MB | CPU | n/a | **—** | 30.5 μs | 1.39 ms | 139.8 μs | — | **45.65x** | **4.58x** |
| tensor | compressed_hcompress_1:header_read | header_read | 0.82 MB | CPU | n/a | **—** | 31.5 μs | 1.48 ms | 157.9 μs | — | **46.82x** | **5.01x** |
| tensor | compressed_rice_1:cutout_100x100 | cutout_100x100 | 0.90 MB | CPU | n/a | **731.1 μs** | 745.3 μs | 6.75 ms | 801.4 μs | — | **9.24x** | **1.10x** |
| tensor | compressed_rice_1:header_read | header_read | 0.90 MB | CPU | n/a | **—** | 32.3 μs | 1.45 ms | 157.9 μs | — | **44.89x** | **4.89x** |
| tensor | large_float32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 15.3 μs | 263.5 μs | 26.3 μs | — | **17.26x** | **1.73x** |
| tensor | large_float32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 17.9 μs | 297.0 μs | 28.9 μs | — | **16.57x** | **1.61x** |
| tensor | large_float64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 15.6 μs | 264.4 μs | 28.9 μs | — | **16.94x** | **1.85x** |
| tensor | large_float64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 15.5 μs | 286.0 μs | 27.5 μs | — | **18.49x** | **1.78x** |
| tensor | large_int16_1d:header_read | header_read | 1.91 MB | CPU | n/a | **—** | 14.3 μs | 263.3 μs | 27.0 μs | — | **18.45x** | **1.89x** |
| tensor | large_int16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 16.9 μs | 291.3 μs | 29.0 μs | — | **17.25x** | **1.72x** |
| tensor | large_int32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 15.7 μs | 267.7 μs | 25.5 μs | — | **17.09x** | **1.63x** |
| tensor | large_int32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 16.3 μs | 297.8 μs | 28.4 μs | — | **18.32x** | **1.74x** |
| tensor | large_int64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 16.3 μs | 259.7 μs | 27.0 μs | — | **15.95x** | **1.66x** |
| tensor | large_int64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 17.8 μs | 289.9 μs | 27.9 μs | — | **16.29x** | **1.57x** |
| tensor | large_int8_1d:header_read | header_read | 0.96 MB | CPU | n/a | **—** | 15.4 μs | 311.8 μs | 32.6 μs | — | **20.25x** | **2.11x** |
| tensor | large_int8_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 16.0 μs | 327.5 μs | 34.2 μs | — | **20.53x** | **2.15x** |
| tensor | large_uint16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 17.3 μs | 334.3 μs | 33.3 μs | — | **19.35x** | **1.93x** |
| tensor | large_uint32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 18.2 μs | 336.6 μs | 33.4 μs | — | **18.49x** | **1.83x** |
| tensor | medium_float32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 15.2 μs | 279.2 μs | 26.8 μs | — | **18.36x** | **1.76x** |
| tensor | medium_float32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 16.6 μs | 291.3 μs | 29.5 μs | — | **17.57x** | **1.78x** |
| tensor | medium_float32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 17.2 μs | 318.8 μs | 31.2 μs | — | **18.58x** | **1.82x** |
| tensor | medium_float64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 14.6 μs | 272.0 μs | 26.4 μs | — | **18.59x** | **1.80x** |
| tensor | medium_float64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 49.0 μs | 857.6 μs | 85.9 μs | — | **17.50x** | **1.75x** |
| tensor | medium_float64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 22.0 μs | 448.5 μs | 43.9 μs | — | **20.39x** | **1.99x** |
| tensor | medium_int16_1d:header_read | header_read | 0.20 MB | CPU | n/a | **—** | 16.1 μs | 267.4 μs | 28.5 μs | — | **16.57x** | **1.77x** |
| tensor | medium_int16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 14.9 μs | 294.3 μs | 28.0 μs | — | **19.70x** | **1.88x** |
| tensor | medium_int16_3d:header_read | header_read | 3.13 MB | CPU | n/a | **—** | 16.3 μs | 322.6 μs | 29.4 μs | — | **19.81x** | **1.81x** |
| tensor | medium_int32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 15.1 μs | 268.8 μs | 25.7 μs | — | **17.79x** | **1.70x** |
| tensor | medium_int32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 15.3 μs | 293.5 μs | 29.4 μs | — | **19.16x** | **1.92x** |
| tensor | medium_int32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 16.0 μs | 310.4 μs | 32.2 μs | — | **19.42x** | **2.01x** |
| tensor | medium_int64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 16.6 μs | 274.6 μs | 26.9 μs | — | **16.55x** | **1.62x** |
| tensor | medium_int64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 16.6 μs | 302.8 μs | 28.6 μs | — | **18.20x** | **1.72x** |
| tensor | medium_int64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 16.4 μs | 313.9 μs | 31.3 μs | — | **19.12x** | **1.91x** |
| tensor | medium_int8_1d:header_read | header_read | 0.10 MB | CPU | n/a | **—** | 15.5 μs | 311.1 μs | 31.5 μs | — | **20.14x** | **2.04x** |
| tensor | medium_int8_2d:header_read | header_read | 1.01 MB | CPU | n/a | **—** | 16.4 μs | 335.5 μs | 36.0 μs | — | **20.50x** | **2.20x** |
| tensor | medium_int8_3d:header_read | header_read | 1.57 MB | CPU | n/a | **—** | 17.2 μs | 348.9 μs | 35.1 μs | — | **20.33x** | **2.04x** |
| tensor | medium_uint16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 17.0 μs | 339.2 μs | 34.3 μs | — | **19.95x** | **2.02x** |
| tensor | medium_uint32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 16.1 μs | 333.3 μs | 33.4 μs | — | **20.69x** | **2.07x** |
| tensor | mef_medium:header_read | header_read | 7.02 MB | CPU | n/a | **—** | 18.9 μs | 536.0 μs | 43.8 μs | — | **28.36x** | **2.32x** |
| tensor | mef_small:header_read | header_read | 0.45 MB | CPU | n/a | **—** | 18.7 μs | 532.9 μs | 45.0 μs | — | **28.47x** | **2.40x** |
| tensor | multi_mef_10ext:cutout_100x100 | cutout_100x100 | 2.68 MB | CPU | n/a | **89.8 μs** | 88.3 μs | 2.22 ms | 170.9 μs | — | **25.14x** | **1.93x** |
| tensor | multi_mef_10ext:header_read | header_read | 2.68 MB | CPU | n/a | **—** | 19.7 μs | 534.5 μs | 43.8 μs | — | **27.08x** | **2.22x** |
| tensor | multi_mef_10ext:random_ext_full_reads_200 | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **5.19 ms** | 5.23 ms | 6.50 ms | 7.12 ms | — | **1.25x** | **1.37x** |
| tensor | repeated_cutouts_50x_100x100:repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **466.1 μs** | 475.0 μs | 51.26 ms | 3.10 ms | — | **109.97x** | **6.66x** |
| tensor | scaled_large:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 16.4 μs | 335.3 μs | 34.1 μs | — | **20.44x** | **2.08x** |
| tensor | scaled_medium:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 17.6 μs | 331.7 μs | 33.6 μs | — | **18.84x** | **1.91x** |
| tensor | scaled_small:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 16.6 μs | 334.1 μs | 33.4 μs | — | **20.09x** | **2.01x** |
| tensor | small_float32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 15.2 μs | 272.5 μs | 26.9 μs | — | **17.91x** | **1.77x** |
| tensor | small_float32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 15.8 μs | 288.8 μs | 30.5 μs | — | **18.24x** | **1.93x** |
| tensor | small_float32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 15.6 μs | 310.3 μs | 29.2 μs | — | **19.88x** | **1.87x** |
| tensor | small_float64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 16.7 μs | 271.1 μs | 26.5 μs | — | **16.22x** | **1.59x** |
| tensor | small_float64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 15.5 μs | 284.7 μs | 27.4 μs | — | **18.40x** | **1.77x** |
| tensor | small_float64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 15.3 μs | 309.4 μs | 29.3 μs | — | **20.20x** | **1.91x** |
| tensor | small_int16_1d:header_read | header_read | 22.5 KB | CPU | n/a | **—** | 14.4 μs | 258.1 μs | 25.0 μs | — | **17.95x** | **1.74x** |
| tensor | small_int16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 14.8 μs | 284.0 μs | 26.5 μs | — | **19.23x** | **1.79x** |
| tensor | small_int16_3d:header_read | header_read | 0.32 MB | CPU | n/a | **—** | 16.7 μs | 311.1 μs | 30.1 μs | — | **18.66x** | **1.81x** |
| tensor | small_int32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 14.9 μs | 270.4 μs | 28.1 μs | — | **18.09x** | **1.88x** |
| tensor | small_int32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.2 μs | 292.1 μs | 28.2 μs | — | **18.04x** | **1.74x** |
| tensor | small_int32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 17.1 μs | 313.2 μs | 31.2 μs | — | **18.36x** | **1.83x** |
| tensor | small_int64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 15.4 μs | 262.2 μs | 27.0 μs | — | **17.03x** | **1.75x** |
| tensor | small_int64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 16.7 μs | 285.6 μs | 29.2 μs | — | **17.08x** | **1.74x** |
| tensor | small_int64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 16.0 μs | 304.8 μs | 29.8 μs | — | **19.02x** | **1.86x** |
| tensor | small_int8_1d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 16.1 μs | 306.4 μs | 30.3 μs | — | **18.98x** | **1.88x** |
| tensor | small_int8_2d:header_read | header_read | 0.07 MB | CPU | n/a | **—** | 17.0 μs | 330.3 μs | 33.6 μs | — | **19.49x** | **1.98x** |
| tensor | small_int8_3d:header_read | header_read | 0.16 MB | CPU | n/a | **—** | 16.1 μs | 351.7 μs | 36.2 μs | — | **21.91x** | **2.26x** |
| tensor | small_uint16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 16.4 μs | 333.0 μs | 33.2 μs | — | **20.33x** | **2.02x** |
| tensor | small_uint32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.8 μs | 335.7 μs | 33.3 μs | — | **20.00x** | **1.98x** |
| tensor | timeseries_frame_000:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.6 μs | 301.1 μs | 28.4 μs | — | **18.14x** | **1.71x** |
| tensor | timeseries_frame_001:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.2 μs | 296.1 μs | 28.0 μs | — | **18.29x** | **1.73x** |
| tensor | timeseries_frame_002:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.6 μs | 290.2 μs | 29.6 μs | — | **17.53x** | **1.79x** |
| tensor | timeseries_frame_003:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.9 μs | 300.2 μs | 29.5 μs | — | **17.78x** | **1.75x** |
| tensor | timeseries_frame_004:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.2 μs | 297.4 μs | 29.6 μs | — | **18.38x** | **1.83x** |
| tensor | tiny_float32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 15.3 μs | 270.4 μs | 27.7 μs | — | **17.65x** | **1.81x** |
| tensor | tiny_float32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 15.4 μs | 286.4 μs | 28.2 μs | — | **18.61x** | **1.83x** |
| tensor | tiny_float32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 16.0 μs | 309.9 μs | 30.8 μs | — | **19.33x** | **1.92x** |
| tensor | tiny_float64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 15.8 μs | 259.1 μs | 27.4 μs | — | **16.43x** | **1.73x** |
| tensor | tiny_float64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 15.5 μs | 292.2 μs | 30.1 μs | — | **18.80x** | **1.94x** |
| tensor | tiny_float64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 16.6 μs | 311.5 μs | 30.4 μs | — | **18.74x** | **1.83x** |
| tensor | tiny_int16_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 14.9 μs | 268.4 μs | 24.7 μs | — | **17.99x** | **1.66x** |
| tensor | tiny_int16_2d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 16.4 μs | 283.8 μs | 29.4 μs | — | **17.31x** | **1.79x** |
| tensor | tiny_int16_3d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 15.7 μs | 304.9 μs | 30.5 μs | — | **19.44x** | **1.95x** |
| tensor | tiny_int32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 15.9 μs | 268.7 μs | 25.7 μs | — | **16.85x** | **1.61x** |
| tensor | tiny_int32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 16.2 μs | 287.9 μs | 30.8 μs | — | **17.79x** | **1.90x** |
| tensor | tiny_int32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 16.0 μs | 313.3 μs | 30.6 μs | — | **19.57x** | **1.91x** |
| tensor | tiny_int64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 15.0 μs | 270.7 μs | 27.7 μs | — | **18.06x** | **1.85x** |
| tensor | tiny_int64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 16.0 μs | 293.1 μs | 26.6 μs | — | **18.30x** | **1.66x** |
| tensor | tiny_int64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 16.8 μs | 316.2 μs | 30.3 μs | — | **18.86x** | **1.81x** |
| tensor | tiny_int8_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 15.9 μs | 309.7 μs | 31.6 μs | — | **19.49x** | **1.99x** |
| tensor | tiny_int8_2d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 16.3 μs | 332.9 μs | 34.5 μs | — | **20.45x** | **2.12x** |
| tensor | tiny_int8_3d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 17.1 μs | 357.9 μs | 36.2 μs | — | **20.97x** | **2.12x** |
| tensor | write_compress_hcompress_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **48.56 ms** | — | 58.89 ms | — | — | **1.21x** | **—** |
| tensor | write_compress_rice_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **37.80 ms** | — | 68.75 ms | — | — | **1.82x** | **—** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | off | **23.53 ms** | 23.53 ms | 45.85 ms | 26.35 ms | — | **1.95x** | **1.12x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | off | **20.18 ms** | 20.18 ms | 72.39 ms | 22.98 ms | — | **3.59x** | **1.14x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | off | **45.78 ms** | 45.72 ms | 51.76 ms | 44.44 ms | — | **1.13x** | **0.97x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | off | **7.30 ms** | 7.30 ms | 19.12 ms | 7.28 ms | — | **2.62x** | **1.00x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | off | **491.6 μs** | 681.6 μs | 1.14 ms | 746.6 μs | — | **2.31x** | **1.52x** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | off | **2.78 ms** | 2.65 ms | 9.53 ms | 3.21 ms | — | **3.59x** | **1.21x** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | off | **871.4 μs** | 888.3 μs | 1.89 ms | 1.21 ms | — | **2.17x** | **1.39x** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | off | **5.55 ms** | 5.02 ms | 10.39 ms | 5.35 ms | — | **2.07x** | **1.06x** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | off | **278.7 μs** | 292.7 μs | 718.5 μs | 342.0 μs | — | **2.58x** | **1.23x** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | off | **986.1 μs** | 973.5 μs | 3.69 ms | 1.27 ms | — | **3.79x** | **1.30x** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | off | **478.0 μs** | 477.6 μs | 1.11 ms | 734.6 μs | — | **2.33x** | **1.54x** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | off | **2.53 ms** | 2.54 ms | 9.40 ms | 3.08 ms | — | **3.71x** | **1.22x** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | off | **1.27 ms** | 870.0 μs | 1.89 ms | 1.21 ms | — | **2.17x** | **1.40x** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | off | **5.05 ms** | 4.89 ms | 10.27 ms | 5.23 ms | — | **2.10x** | **1.07x** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | off | **232.5 μs** | 199.3 μs | 637.4 μs | 186.5 μs | — | **3.20x** | **0.94x** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | off | **661.7 μs** | 632.4 μs | 1.56 ms | 655.4 μs | — | **2.47x** | **1.04x** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | off | **1.25 ms** | 1.25 ms | 4.00 ms | 1.54 ms | — | **3.19x** | **1.23x** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | off | **3.10 ms** | 3.12 ms | 6.67 ms | 3.63 ms | — | **2.15x** | **1.17x** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | off | **48.7 μs** | 95.0 μs | 384.2 μs | 116.2 μs | — | **7.90x** | **2.39x** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | off | **484.7 μs** | 477.9 μs | 1.19 ms | 771.3 μs | — | **2.48x** | **1.61x** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | off | **735.9 μs** | 735.0 μs | 1.66 ms | 1.16 ms | — | **2.26x** | **1.58x** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | off | **133.7 μs** | 131.4 μs | 457.2 μs | 154.2 μs | — | **3.48x** | **1.17x** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | off | **930.3 μs** | 911.8 μs | 2.54 ms | 1.28 ms | — | **2.78x** | **1.40x** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | off | **1.86 ms** | 1.72 ms | 6.24 ms | 2.95 ms | — | **3.62x** | **1.71x** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | off | **103.3 μs** | 99.4 μs | 506.7 μs | 108.0 μs | — | **5.10x** | **1.09x** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | off | **425.4 μs** | 419.9 μs | 1.18 ms | 535.4 μs | — | **2.81x** | **1.28x** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | off | **616.7 μs** | 658.2 μs | 1.54 ms | 798.6 μs | — | **2.50x** | **1.29x** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | off | **86.9 μs** | 114.5 μs | 598.4 μs | 172.1 μs | — | **6.88x** | **1.98x** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | off | **737.0 μs** | 702.7 μs | 1.80 ms | 1.22 ms | — | **2.56x** | **1.74x** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | off | **1.20 ms** | 1.09 ms | 2.50 ms | 1.85 ms | — | **2.30x** | **1.70x** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | off | **154.5 μs** | 172.6 μs | 708.3 μs | 246.5 μs | — | **4.59x** | **1.60x** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | off | **1.34 ms** | 1.36 ms | 3.98 ms | 1.92 ms | — | **2.98x** | **1.43x** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | off | **2.61 ms** | 2.41 ms | 6.31 ms | 3.04 ms | — | **2.62x** | **1.26x** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | off | **81.4 μs** | 76.1 μs | 698.0 μs | 101.3 μs | — | **9.17x** | **1.33x** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | off | **316.6 μs** | 320.5 μs | 1.06 ms | 304.2 μs | — | **3.34x** | **0.96x** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | off | **452.7 μs** | 454.6 μs | 1.31 ms | 436.0 μs | — | **2.89x** | **0.96x** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | off | **522.5 μs** | 489.0 μs | 2.10 ms | 632.7 μs | — | **4.30x** | **1.29x** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | off | **943.9 μs** | 949.8 μs | 2.76 ms | 1.46 ms | — | **2.92x** | **1.54x** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | off | **316.0 μs** | 320.8 μs | 1.37 ms | 343.7 μs | — | **4.33x** | **1.09x** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | off | **70.6 μs** | 91.3 μs | 959.7 μs | 150.2 μs | — | **13.60x** | **2.13x** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | off | **57.6 μs** | 97.9 μs | 976.5 μs | 238.8 μs | — | **16.95x** | **4.15x** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | off | **5.46 ms** | 5.65 ms | 8.44 ms | 5.48 ms | — | **1.55x** | **1.00x** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | off | **1.10 ms** | 1.10 ms | 2.12 ms | 1.30 ms | — | **1.93x** | **1.18x** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | off | **101.4 μs** | 144.2 μs | 751.0 μs | 161.4 μs | — | **7.41x** | **1.59x** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | off | **67.0 μs** | 46.6 μs | 454.6 μs | 71.3 μs | — | **9.75x** | **1.53x** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | off | **104.0 μs** | 100.5 μs | 580.6 μs | 143.6 μs | — | **5.78x** | **1.43x** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | off | **162.7 μs** | 122.9 μs | 711.8 μs | 259.2 μs | — | **5.79x** | **2.11x** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | off | **69.5 μs** | 48.1 μs | 462.9 μs | 86.4 μs | — | **9.62x** | **1.79x** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | off | **149.8 μs** | 129.3 μs | 658.7 μs | 186.9 μs | — | **5.09x** | **1.45x** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | off | **270.2 μs** | 272.4 μs | 948.3 μs | 364.3 μs | — | **3.51x** | **1.35x** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | off | **69.3 μs** | 52.5 μs | 456.7 μs | 71.9 μs | — | **8.70x** | **1.37x** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | off | **76.2 μs** | 76.7 μs | 516.2 μs | 103.1 μs | — | **6.78x** | **1.35x** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | off | **121.4 μs** | 116.2 μs | 613.1 μs | 137.5 μs | — | **5.28x** | **1.18x** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | off | **54.2 μs** | 55.8 μs | 457.7 μs | 75.2 μs | — | **8.44x** | **1.39x** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | off | **95.3 μs** | 110.2 μs | 571.0 μs | 135.9 μs | — | **5.99x** | **1.43x** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | off | **174.3 μs** | 130.0 μs | 716.6 μs | 257.6 μs | — | **5.51x** | **1.98x** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | off | **56.9 μs** | 74.3 μs | 469.5 μs | 84.0 μs | — | **8.25x** | **1.48x** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | off | **91.6 μs** | 158.0 μs | 656.3 μs | 190.2 μs | — | **7.17x** | **2.08x** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | off | **281.7 μs** | 283.1 μs | 962.8 μs | 366.1 μs | — | **3.42x** | **1.30x** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | off | **72.7 μs** | 46.2 μs | 632.5 μs | 79.7 μs | — | **13.69x** | **1.72x** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | off | **89.1 μs** | 60.3 μs | 672.7 μs | 90.0 μs | — | **11.16x** | **1.49x** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | off | **106.0 μs** | 107.5 μs | 731.3 μs | 112.1 μs | — | **6.90x** | **1.06x** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | off | **69.3 μs** | 88.4 μs | 654.4 μs | 104.4 μs | — | **9.44x** | **1.51x** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | off | **112.0 μs** | 110.5 μs | 693.8 μs | 147.0 μs | — | **6.28x** | **1.33x** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | off | **82.3 μs** | 99.5 μs | 404.5 μs | 93.5 μs | — | **4.92x** | **1.14x** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | off | **57.6 μs** | 70.8 μs | 362.2 μs | 90.4 μs | — | **6.29x** | **1.57x** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | off | **71.1 μs** | 54.3 μs | 360.6 μs | 90.3 μs | — | **6.64x** | **1.66x** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | off | **60.1 μs** | 66.9 μs | 352.6 μs | 90.2 μs | — | **5.87x** | **1.50x** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | off | **68.4 μs** | 39.9 μs | 365.1 μs | 90.8 μs | — | **9.14x** | **2.27x** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | off | **31.9 μs** | 36.2 μs | 259.5 μs | 39.6 μs | — | **8.14x** | **1.24x** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | off | **30.2 μs** | 39.9 μs | 271.8 μs | 43.7 μs | — | **8.99x** | **1.44x** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | off | **41.3 μs** | 31.6 μs | 293.1 μs | 44.6 μs | — | **9.27x** | **1.41x** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | off | **30.3 μs** | 35.7 μs | 265.1 μs | 42.7 μs | — | **8.75x** | **1.41x** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | off | **33.2 μs** | 42.1 μs | 280.7 μs | 46.9 μs | — | **8.45x** | **1.41x** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | off | **45.3 μs** | 31.4 μs | 299.3 μs | 49.3 μs | — | **9.52x** | **1.57x** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | off | **28.9 μs** | 45.9 μs | 266.8 μs | 41.9 μs | — | **9.24x** | **1.45x** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | off | **41.0 μs** | 31.1 μs | 274.4 μs | 42.7 μs | — | **8.82x** | **1.37x** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | off | **37.2 μs** | 42.7 μs | 287.2 μs | 41.8 μs | — | **7.72x** | **1.12x** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | off | **41.8 μs** | 25.0 μs | 262.3 μs | 41.9 μs | — | **10.51x** | **1.68x** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | off | **31.6 μs** | 28.1 μs | 278.9 μs | 42.7 μs | — | **9.93x** | **1.52x** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | off | **31.3 μs** | 42.7 μs | 291.0 μs | 45.2 μs | — | **9.30x** | **1.44x** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | off | **30.9 μs** | 33.6 μs | 255.2 μs | 42.4 μs | — | **8.26x** | **1.37x** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | off | **37.4 μs** | 39.4 μs | 288.0 μs | 46.3 μs | — | **7.71x** | **1.24x** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | off | **44.5 μs** | 38.7 μs | 297.7 μs | 46.6 μs | — | **7.70x** | **1.21x** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | off | **34.1 μs** | 41.0 μs | 360.8 μs | 48.4 μs | — | **10.59x** | **1.42x** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | off | **44.2 μs** | 26.7 μs | 379.5 μs | 48.0 μs | — | **14.19x** | **1.80x** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | off | **46.0 μs** | 31.2 μs | 390.5 μs | 49.7 μs | — | **12.51x** | **1.59x** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | on | **23.73 ms** | 23.71 ms | 46.06 ms | 26.36 ms | — | **1.94x** | **1.11x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | on | **20.17 ms** | 20.24 ms | 72.62 ms | 23.03 ms | — | **3.60x** | **1.14x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | on | **45.71 ms** | 45.66 ms | 51.90 ms | 44.50 ms | — | **1.14x** | **0.97x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | on | **12.46 ms** | 12.56 ms | 33.13 ms | 12.63 ms | — | **2.66x** | **1.01x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | on | **679.8 μs** | 762.8 μs | 1.53 ms | — | — | **2.25x** | **—** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | on | **3.88 ms** | 3.88 ms | 12.50 ms | — | — | **3.22x** | **—** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | on | **1.45 ms** | 1.28 ms | 2.46 ms | — | — | **1.93x** | **—** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | on | **7.34 ms** | 7.29 ms | 11.72 ms | — | — | **1.61x** | **—** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | on | **381.4 μs** | 411.2 μs | 1.05 ms | — | — | **2.76x** | **—** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | on | **1.33 ms** | 1.31 ms | 2.57 ms | — | — | **1.97x** | **—** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | on | **767.6 μs** | 730.7 μs | 2.28 ms | — | — | **3.12x** | **—** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | on | **6.19 ms** | 5.26 ms | 15.13 ms | — | — | **2.88x** | **—** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | on | **1.95 ms** | 2.00 ms | 3.85 ms | — | — | **1.97x** | **—** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | on | **8.35 ms** | 8.62 ms | 14.95 ms | — | — | **1.79x** | **—** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | on | **469.6 μs** | 467.2 μs | — | — | — | **—** | **—** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | on | **1.51 ms** | 1.48 ms | — | — | — | **—** | **—** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | on | **2.09 ms** | 2.12 ms | — | — | — | **—** | **—** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | on | **3.42 ms** | 2.84 ms | — | — | — | **—** | **—** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | on | **80.2 μs** | 80.4 μs | 378.9 μs | — | — | **4.72x** | **—** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | on | **482.1 μs** | 499.4 μs | 1.07 ms | — | — | **2.21x** | **—** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | on | **811.0 μs** | 730.5 μs | 2.92 ms | — | — | **4.00x** | **—** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | on | **170.3 μs** | 163.4 μs | 708.1 μs | — | — | **4.33x** | **—** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | on | **1.36 ms** | 1.31 ms | 2.56 ms | — | — | **1.95x** | **—** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | on | **2.00 ms** | 2.01 ms | 3.73 ms | — | — | **1.87x** | **—** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | on | **119.5 μs** | 109.3 μs | 542.7 μs | — | — | **4.97x** | **—** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | on | **398.7 μs** | 406.3 μs | 1.08 ms | — | — | **2.71x** | **—** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | on | **590.9 μs** | 576.3 μs | 1.37 ms | — | — | **2.38x** | **—** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | on | **122.8 μs** | 140.8 μs | 594.3 μs | — | — | **4.84x** | **—** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | on | **711.1 μs** | 716.2 μs | 1.64 ms | — | — | **2.31x** | **—** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | on | **1.05 ms** | 1.03 ms | 2.16 ms | — | — | **2.10x** | **—** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | on | **195.5 μs** | 202.5 μs | 712.6 μs | — | — | **3.65x** | **—** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | on | **1.28 ms** | 1.30 ms | 2.59 ms | — | — | **2.02x** | **—** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | on | **1.96 ms** | 1.98 ms | 3.75 ms | — | — | **1.91x** | **—** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | on | **98.7 μs** | 89.7 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | on | **270.0 μs** | 322.7 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | on | **415.7 μs** | 454.5 μs | — | — | — | **—** | **—** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | on | **409.3 μs** | 385.4 μs | — | — | — | **—** | **—** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | on | **1.18 ms** | 1.20 ms | — | — | — | **—** | **—** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | on | **327.4 μs** | 328.0 μs | — | — | — | **—** | **—** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | on | **70.0 μs** | 100.8 μs | — | — | — | **—** | **—** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | on | **98.8 μs** | 86.4 μs | — | — | — | **—** | **—** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | on | **5.48 ms** | 5.50 ms | — | — | — | **—** | **—** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | on | **1.09 ms** | 1.07 ms | — | — | — | **—** | **—** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | on | **126.7 μs** | 145.3 μs | — | — | — | **—** | **—** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | on | **59.3 μs** | 50.7 μs | 473.7 μs | — | — | **9.34x** | **—** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | on | **71.7 μs** | 67.2 μs | 586.3 μs | — | — | **8.72x** | **—** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | on | **156.1 μs** | 161.3 μs | 713.6 μs | — | — | **4.57x** | **—** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | on | **80.0 μs** | 51.1 μs | 494.7 μs | — | — | **9.69x** | **—** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | on | **126.7 μs** | 123.4 μs | 673.0 μs | — | — | **5.45x** | **—** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | on | **270.3 μs** | 274.1 μs | 920.1 μs | — | — | **3.40x** | **—** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | on | **72.5 μs** | 60.5 μs | 462.2 μs | — | — | **7.64x** | **—** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | on | **70.1 μs** | 116.9 μs | 544.1 μs | — | — | **7.76x** | **—** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | on | **113.3 μs** | 120.7 μs | 629.5 μs | — | — | **5.55x** | **—** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | on | **55.5 μs** | 61.2 μs | 480.1 μs | — | — | **8.66x** | **—** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | on | **113.9 μs** | 109.2 μs | 578.9 μs | — | — | **5.30x** | **—** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | on | **143.2 μs** | 187.1 μs | 720.4 μs | — | — | **5.03x** | **—** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | on | **86.1 μs** | 71.7 μs | 477.0 μs | — | — | **6.65x** | **—** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | on | **154.7 μs** | 152.9 μs | 646.1 μs | — | — | **4.22x** | **—** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | on | **260.2 μs** | 283.9 μs | 906.4 μs | — | — | **3.48x** | **—** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | on | **46.5 μs** | 78.5 μs | — | — | — | **—** | **—** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | on | **67.6 μs** | 75.7 μs | — | — | — | **—** | **—** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | on | **113.1 μs** | 112.4 μs | — | — | — | **—** | **—** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | on | **91.7 μs** | 78.5 μs | — | — | — | **—** | **—** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | on | **111.3 μs** | 163.9 μs | — | — | — | **—** | **—** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | on | **104.0 μs** | 73.6 μs | 582.4 μs | — | — | **7.91x** | **—** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | on | **72.4 μs** | 100.5 μs | 589.9 μs | — | — | **8.15x** | **—** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | on | **104.6 μs** | 70.0 μs | 586.8 μs | — | — | **8.39x** | **—** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | on | **69.8 μs** | 69.6 μs | 593.7 μs | — | — | **8.54x** | **—** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | on | **103.2 μs** | 88.9 μs | 587.8 μs | — | — | **6.61x** | **—** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | on | **46.1 μs** | 62.1 μs | 467.8 μs | — | — | **10.15x** | **—** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | on | **47.6 μs** | 65.7 μs | 479.9 μs | — | — | **10.07x** | **—** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | on | **64.9 μs** | 43.1 μs | 511.6 μs | — | — | **11.88x** | **—** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | on | **57.9 μs** | 52.9 μs | 457.8 μs | — | — | **8.65x** | **—** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | on | **49.6 μs** | 54.4 μs | 486.5 μs | — | — | **9.82x** | **—** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | on | **66.5 μs** | 60.7 μs | 527.7 μs | — | — | **8.69x** | **—** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | on | **60.0 μs** | 67.6 μs | 447.6 μs | — | — | **7.46x** | **—** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | on | **64.5 μs** | 70.7 μs | 474.0 μs | — | — | **7.35x** | **—** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | on | **51.0 μs** | 76.6 μs | 504.8 μs | — | — | **9.90x** | **—** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | on | **43.8 μs** | 71.9 μs | 453.5 μs | — | — | **10.35x** | **—** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | on | **69.8 μs** | 71.6 μs | 485.5 μs | — | — | **6.96x** | **—** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | on | **66.5 μs** | 55.8 μs | 301.8 μs | — | — | **5.41x** | **—** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | on | **40.8 μs** | 31.8 μs | 262.0 μs | — | — | **8.24x** | **—** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | on | **47.4 μs** | 31.7 μs | 295.4 μs | — | — | **9.31x** | **—** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | on | **41.6 μs** | 38.5 μs | 298.1 μs | — | — | **7.74x** | **—** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | on | **45.4 μs** | 48.3 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | on | **48.9 μs** | 30.5 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | on | **47.2 μs** | 50.2 μs | — | — | — | **—** | **—** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **343.3 μs** | 341.7 μs | 2.66 ms | 377.8 μs | — | **7.80x** | **1.11x** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | off | **323.3 μs** | 345.4 μs | 2.72 ms | 379.7 μs | — | **8.40x** | **1.17x** |
| table | ascii_10000 | projection | 0.44 MB | CPU | off | **1.04 ms** | 959.8 μs | 8.10 ms | 1.98 ms | — | **8.44x** | **2.06x** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | off | **1.04 ms** | 953.2 μs | 8.08 ms | 1.97 ms | — | **8.48x** | **2.07x** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | off | **191.1 μs** | 129.5 μs | 2.64 ms | 524.1 μs | — | **20.41x** | **4.05x** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | off | **30.3 μs** | 30.6 μs | 430.8 μs | 65.2 μs | — | **14.21x** | **2.15x** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **102.4 μs** | 107.1 μs | 1.53 ms | 170.6 μs | — | **14.92x** | **1.67x** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | off | **101.7 μs** | 106.7 μs | 1.51 ms | 172.6 μs | — | **14.86x** | **1.70x** |
| table | ascii_1000 | projection | 50.6 KB | CPU | off | **191.3 μs** | 124.3 μs | 2.20 ms | 347.3 μs | — | **17.65x** | **2.79x** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | off | **190.2 μs** | 126.6 μs | 2.20 ms | 332.5 μs | — | **17.37x** | **2.63x** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | off | **125.0 μs** | 58.1 μs | 1.96 ms | 198.5 μs | — | **33.79x** | **3.42x** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | off | **30.5 μs** | 28.9 μs | 427.2 μs | 63.6 μs | — | **14.79x** | **2.20x** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **10.97 ms** | 10.75 ms | 15.91 ms | 20.52 ms | — | **1.48x** | **1.91x** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | off | **9.90 ms** | 9.90 ms | 12.57 ms | 16.88 ms | — | **1.27x** | **1.70x** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | off | **10.21 ms** | 9.85 ms | 17.26 ms | 31.21 ms | — | **1.75x** | **3.17x** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | off | **27.10 ms** | 25.54 ms | 328.28 ms | 112.99 ms | — | **12.85x** | **4.42x** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **298.0 μs** | 218.7 μs | 13.41 ms | 1.53 ms | — | **61.33x** | **7.01x** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **32.6 μs** | 43.0 μs | 539.1 μs | 88.6 μs | — | **16.56x** | **2.72x** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **1.50 ms** | 1.26 ms | 3.20 ms | 2.22 ms | — | **2.53x** | **1.75x** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | off | **1.13 ms** | 1.14 ms | 2.87 ms | 1.82 ms | — | **2.54x** | **1.62x** |
| table | mixed_100000 | projection | 5.06 MB | CPU | off | **1.12 ms** | 1.03 ms | 3.26 ms | 3.24 ms | — | **3.18x** | **3.16x** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | off | **2.16 ms** | 2.06 ms | 31.01 ms | 10.14 ms | — | **15.03x** | **4.91x** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | off | **289.1 μs** | 212.1 μs | 6.02 ms | 1.53 ms | — | **28.37x** | **7.23x** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | off | **31.1 μs** | 27.1 μs | 468.2 μs | 80.7 μs | — | **17.27x** | **2.98x** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **224.7 μs** | 256.5 μs | 1.99 ms | 390.4 μs | — | **8.88x** | **1.74x** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | off | **158.3 μs** | 189.1 μs | 1.96 ms | 348.9 μs | — | **12.39x** | **2.20x** |
| table | mixed_10000 | projection | 0.51 MB | CPU | off | **187.0 μs** | 113.9 μs | 1.96 ms | 482.3 μs | — | **17.24x** | **4.23x** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | off | **283.6 μs** | 209.2 μs | 4.65 ms | 1.10 ms | — | **22.25x** | **5.27x** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | off | **130.2 μs** | 64.9 μs | 3.03 ms | 330.5 μs | — | **46.61x** | **5.09x** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | off | **30.2 μs** | 29.7 μs | 462.1 μs | 79.8 μs | — | **15.54x** | **2.69x** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **42.9 μs** | 88.2 μs | 1.85 ms | 195.7 μs | — | **43.03x** | **4.56x** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | off | **40.7 μs** | 84.0 μs | 1.85 ms | 191.1 μs | — | **45.48x** | **4.69x** |
| table | mixed_1000 | projection | 0.06 MB | CPU | off | **105.0 μs** | 47.9 μs | 1.86 ms | 199.6 μs | — | **38.86x** | **4.17x** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | off | **129.7 μs** | 62.6 μs | 2.21 ms | 277.3 μs | — | **35.37x** | **4.43x** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | off | **116.9 μs** | 49.4 μs | 2.69 ms | 209.5 μs | — | **54.46x** | **4.24x** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | off | **29.8 μs** | 27.2 μs | 451.1 μs | 74.0 μs | — | **16.57x** | **2.72x** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **6.30 ms** | 5.66 ms | 8.83 ms | 14.80 ms | — | **1.56x** | **2.61x** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | off | **4.89 ms** | 4.89 ms | 5.34 ms | 11.35 ms | — | **1.09x** | **2.32x** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | off | **4.03 ms** | 4.02 ms | 5.73 ms | 23.66 ms | — | **1.42x** | **5.88x** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | off | **5.67 ms** | 5.58 ms | 7.08 ms | 5.80 ms | — | **1.27x** | **1.04x** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **170.2 μs** | 101.5 μs | 4.00 ms | 591.3 μs | — | **39.37x** | **5.82x** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **31.7 μs** | 29.9 μs | 458.5 μs | 74.3 μs | — | **15.33x** | **2.48x** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **1.06 ms** | 786.4 μs | 2.18 ms | 1.64 ms | — | **2.77x** | **2.09x** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | off | **607.0 μs** | 657.9 μs | 1.85 ms | 1.28 ms | — | **3.04x** | **2.11x** |
| table | narrow_100000 | projection | 1.25 MB | CPU | off | **535.0 μs** | 448.4 μs | 1.87 ms | 2.55 ms | — | **4.18x** | **5.69x** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | off | **681.1 μs** | 598.6 μs | 2.01 ms | 707.0 μs | — | **3.35x** | **1.18x** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | off | **164.8 μs** | 101.6 μs | 2.16 ms | 590.7 μs | — | **21.25x** | **5.81x** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | off | **29.4 μs** | 30.1 μs | 449.9 μs | 67.2 μs | — | **15.28x** | **2.28x** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **150.4 μs** | 184.4 μs | 1.41 ms | 310.2 μs | — | **9.40x** | **2.06x** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | off | **91.2 μs** | 133.9 μs | 1.38 ms | 273.5 μs | — | **15.15x** | **3.00x** |
| table | narrow_10000 | projection | 0.13 MB | CPU | off | **140.0 μs** | 75.4 μs | 1.36 ms | 392.1 μs | — | **18.10x** | **5.20x** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | off | **156.5 μs** | 91.9 μs | 1.40 ms | 191.2 μs | — | **15.20x** | **2.08x** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | off | **107.6 μs** | 48.3 μs | 1.81 ms | 203.2 μs | — | **37.51x** | **4.21x** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | off | **28.2 μs** | 26.4 μs | 433.6 μs | 64.7 μs | — | **16.45x** | **2.45x** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **45.9 μs** | 90.5 μs | 1.33 ms | 175.0 μs | — | **28.99x** | **3.81x** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | off | **41.2 μs** | 89.1 μs | 1.31 ms | 171.3 μs | — | **31.89x** | **4.16x** |
| table | narrow_1000 | projection | 19.7 KB | CPU | off | **102.8 μs** | 43.4 μs | 1.33 ms | 181.0 μs | — | **30.62x** | **4.17x** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | off | **106.2 μs** | 44.9 μs | 1.37 ms | 150.5 μs | — | **30.54x** | **3.35x** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | off | **105.7 μs** | 41.2 μs | 1.78 ms | 162.1 μs | — | **43.29x** | **3.94x** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | off | **27.1 μs** | 25.9 μs | 435.3 μs | 61.8 μs | — | **16.81x** | **2.39x** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **813.3 μs** | 841.3 μs | 1.94 ms | 1.42 ms | — | **2.38x** | **1.75x** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | off | **825.9 μs** | 828.0 μs | 1.94 ms | 1.41 ms | — | **2.35x** | **1.71x** |
| table | typed_100000 | projection | 2.39 MB | CPU | off | **3.54 ms** | 3.44 ms | 29.28 ms | 13.16 ms | — | **8.52x** | **3.83x** |
| table | typed_100000 | read_full | 2.39 MB | CPU | off | **5.22 ms** | 5.11 ms | 29.41 ms | 14.31 ms | — | **5.75x** | **2.80x** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | off | **648.5 μs** | 571.1 μs | 4.78 ms | 1.94 ms | — | **8.37x** | **3.39x** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | off | **29.6 μs** | 32.2 μs | 464.1 μs | 70.0 μs | — | **15.69x** | **2.37x** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **148.5 μs** | 191.2 μs | 1.48 ms | 291.1 μs | — | **9.95x** | **1.96x** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | off | **143.7 μs** | 183.3 μs | 1.48 ms | 289.1 μs | — | **10.32x** | **2.01x** |
| table | typed_10000 | projection | 0.24 MB | CPU | off | **480.4 μs** | 407.2 μs | 4.11 ms | 1.47 ms | — | **10.10x** | **3.60x** |
| table | typed_10000 | read_full | 0.24 MB | CPU | off | **645.9 μs** | 573.5 μs | 4.16 ms | 1.59 ms | — | **7.25x** | **2.78x** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | off | **175.3 μs** | 100.1 μs | 2.23 ms | 402.7 μs | — | **22.29x** | **4.02x** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | off | **33.2 μs** | 30.8 μs | 445.3 μs | 68.4 μs | — | **14.47x** | **2.22x** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **841.1 μs** | 715.7 μs | 1.81 ms | 1.27 ms | — | **2.52x** | **1.78x** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | off | **692.9 μs** | 720.2 μs | 1.80 ms | 1.27 ms | — | **2.60x** | **1.84x** |
| table | varlen_100000 | projection | 3.06 MB | CPU | off | **70.95 ms** | 70.82 ms | 553.37 ms | 112.81 ms | — | **7.81x** | **1.59x** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | off | **70.64 ms** | 71.02 ms | 552.80 ms | 113.05 ms | — | **7.83x** | **1.60x** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | off | **7.00 ms** | 6.94 ms | 56.83 ms | 12.21 ms | — | **8.18x** | **1.76x** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | off | **32.8 μs** | 33.4 μs | 461.1 μs | 72.8 μs | — | **14.05x** | **2.22x** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **120.4 μs** | 155.2 μs | 1.37 ms | 276.0 μs | — | **11.37x** | **2.29x** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | off | **109.9 μs** | 154.2 μs | 1.36 ms | 277.8 μs | — | **12.39x** | **2.53x** |
| table | varlen_10000 | projection | 0.31 MB | CPU | off | **7.02 ms** | 6.93 ms | 56.22 ms | 11.23 ms | — | **8.11x** | **1.62x** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | off | **7.08 ms** | 6.94 ms | 56.23 ms | 11.24 ms | — | **8.10x** | **1.62x** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | off | **781.4 μs** | 713.9 μs | 7.33 ms | 1.42 ms | — | **10.27x** | **2.00x** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | off | **29.8 μs** | 28.9 μs | 443.9 μs | 64.7 μs | — | **15.35x** | **2.24x** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **43.9 μs** | 94.4 μs | 1.28 ms | 172.6 μs | — | **29.14x** | **3.94x** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | off | **43.7 μs** | 90.8 μs | 1.29 ms | 169.9 μs | — | **29.54x** | **3.89x** |
| table | varlen_1000 | projection | 39.4 KB | CPU | off | **794.5 μs** | 707.6 μs | 6.89 ms | 1.34 ms | — | **9.73x** | **1.89x** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | off | **773.2 μs** | 710.2 μs | 6.90 ms | 1.29 ms | — | **9.71x** | **1.81x** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | off | **212.3 μs** | 140.9 μs | 2.27 ms | 306.3 μs | — | **16.15x** | **2.17x** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | off | **27.3 μs** | 26.2 μs | 419.1 μs | 59.8 μs | — | **15.98x** | **2.28x** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **3.37 ms** | 3.21 ms | 8.41 ms | 4.34 ms | — | **2.62x** | **1.35x** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | off | **3.05 ms** | 3.06 ms | 8.08 ms | 4.02 ms | — | **2.65x** | **1.32x** |
| table | wide_100000 | projection | 20.71 MB | CPU | off | **2.58 ms** | 2.49 ms | 8.53 ms | 5.28 ms | — | **3.42x** | **2.12x** |
| table | wide_100000 | read_full | 20.71 MB | CPU | off | **16.05 ms** | 16.38 ms | 129.03 ms | 42.46 ms | — | **8.04x** | **2.65x** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | off | **1.15 ms** | 1.08 ms | 21.89 ms | 5.00 ms | — | **20.34x** | **4.65x** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | off | **38.3 μs** | 37.9 μs | 563.4 μs | 254.1 μs | — | **14.86x** | **6.70x** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **434.1 μs** | 459.5 μs | 6.05 ms | 779.5 μs | — | **13.93x** | **1.80x** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | off | **376.0 μs** | 402.6 μs | 5.97 ms | 740.7 μs | — | **15.87x** | **1.97x** |
| table | wide_10000 | projection | 2.08 MB | CPU | off | **337.7 μs** | 264.0 μs | 6.00 ms | 861.6 μs | — | **22.72x** | **3.26x** |
| table | wide_10000 | read_full | 2.08 MB | CPU | off | **1.14 ms** | 1.05 ms | 16.77 ms | 4.52 ms | — | **15.95x** | **4.30x** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | off | **227.9 μs** | 158.1 μs | 10.71 ms | 899.6 μs | — | **67.72x** | **5.69x** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | off | **39.4 μs** | 38.2 μs | 557.5 μs | 251.0 μs | — | **14.58x** | **6.56x** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **56.1 μs** | 103.2 μs | 5.71 ms | 395.2 μs | — | **101.77x** | **7.05x** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | off | **54.9 μs** | 102.3 μs | 5.71 ms | 397.4 μs | — | **104.09x** | **7.24x** |
| table | wide_1000 | projection | 0.22 MB | CPU | off | **125.7 μs** | 53.3 μs | 5.70 ms | 408.2 μs | — | **106.85x** | **7.65x** |
| table | wide_1000 | read_full | 0.22 MB | CPU | off | **223.2 μs** | 158.4 μs | 7.21 ms | 825.6 μs | — | **45.51x** | **5.21x** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | off | **162.0 μs** | 95.3 μs | 9.46 ms | 506.3 μs | — | **99.19x** | **5.31x** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | off | **37.3 μs** | 37.7 μs | 558.9 μs | 252.5 μs | — | **14.99x** | **6.77x** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **408.1 μs** | 393.6 μs | 2.68 ms | — | — | **6.81x** | **—** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | on | **395.2 μs** | 395.3 μs | 2.67 ms | — | — | **6.76x** | **—** |
| table | ascii_10000 | projection | 0.44 MB | CPU | on | **1.09 ms** | 1.02 ms | 8.02 ms | — | — | **7.86x** | **—** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | on | **1.09 ms** | 1.03 ms | 8.03 ms | — | — | **7.83x** | **—** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | on | **254.4 μs** | 191.5 μs | 2.59 ms | — | — | **13.54x** | **—** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | on | **30.8 μs** | 29.5 μs | 438.3 μs | — | — | **14.83x** | **—** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **157.9 μs** | 160.5 μs | 1.55 ms | — | — | **9.79x** | **—** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | on | **160.5 μs** | 174.9 μs | 1.54 ms | — | — | **9.57x** | **—** |
| table | ascii_1000 | projection | 50.6 KB | CPU | on | **252.1 μs** | 195.9 μs | 2.20 ms | — | — | **11.23x** | **—** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | on | **245.7 μs** | 185.9 μs | 2.20 ms | — | — | **11.83x** | **—** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | on | **166.7 μs** | 119.1 μs | 1.98 ms | — | — | **16.63x** | **—** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | on | **29.7 μs** | 31.4 μs | 425.6 μs | — | — | **14.33x** | **—** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **5.74 ms** | 5.03 ms | 12.12 ms | — | — | **2.41x** | **—** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | on | **4.24 ms** | 4.36 ms | 8.74 ms | — | — | **2.06x** | **—** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | on | **7.32 ms** | 7.24 ms | 14.67 ms | — | — | **2.03x** | **—** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | on | **27.83 ms** | 24.09 ms | 318.39 ms | — | — | **13.22x** | **—** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **254.3 μs** | 171.9 μs | 9.14 ms | — | — | **53.20x** | **—** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **39.7 μs** | 32.6 μs | 487.6 μs | — | — | **14.94x** | **—** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **973.6 μs** | 710.0 μs | 2.97 ms | — | — | **4.18x** | **—** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | on | **590.7 μs** | 589.6 μs | 2.62 ms | — | — | **4.45x** | **—** |
| table | mixed_100000 | projection | 5.06 MB | CPU | on | **872.8 μs** | 792.0 μs | 3.02 ms | — | — | **3.82x** | **—** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | on | **1.86 ms** | 1.76 ms | 30.41 ms | — | — | **17.30x** | **—** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | on | **248.0 μs** | 174.2 μs | 5.73 ms | — | — | **32.87x** | **—** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | on | **34.6 μs** | 34.2 μs | 468.4 μs | — | — | **13.69x** | **—** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **178.2 μs** | 191.9 μs | 1.97 ms | — | — | **11.03x** | **—** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | on | **111.0 μs** | 142.3 μs | 1.92 ms | — | — | **17.25x** | **—** |
| table | mixed_10000 | projection | 0.51 MB | CPU | on | **162.1 μs** | 89.9 μs | 1.94 ms | — | — | **21.55x** | **—** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | on | **241.0 μs** | 149.9 μs | 4.60 ms | — | — | **30.67x** | **—** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | on | **141.5 μs** | 72.6 μs | 2.97 ms | — | — | **40.88x** | **—** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | on | **32.7 μs** | 31.9 μs | 456.6 μs | — | — | **14.31x** | **—** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **54.4 μs** | 97.8 μs | 1.86 ms | — | — | **34.14x** | **—** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | on | **50.5 μs** | 91.9 μs | 1.83 ms | — | — | **36.24x** | **—** |
| table | mixed_1000 | projection | 0.06 MB | CPU | on | **121.5 μs** | 54.3 μs | 1.85 ms | — | — | **34.02x** | **—** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | on | **137.8 μs** | 71.7 μs | 2.21 ms | — | — | **30.84x** | **—** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | on | **127.4 μs** | 64.6 μs | 2.69 ms | — | — | **41.59x** | **—** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | on | **32.4 μs** | 29.6 μs | 465.2 μs | — | — | **15.73x** | **—** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **3.24 ms** | 2.40 ms | 8.15 ms | — | — | **3.40x** | **—** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | on | **1.67 ms** | 1.67 ms | 4.74 ms | — | — | **2.84x** | **—** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | on | **2.16 ms** | 2.08 ms | 5.23 ms | — | — | **2.51x** | **—** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | on | **3.26 ms** | 3.14 ms | 6.54 ms | — | — | **2.08x** | **—** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **157.3 μs** | 86.6 μs | 3.44 ms | — | — | **39.70x** | **—** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **42.1 μs** | 29.9 μs | 485.5 μs | — | — | **16.22x** | **—** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **753.7 μs** | 457.8 μs | 2.10 ms | — | — | **4.59x** | **—** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | on | **316.5 μs** | 308.0 μs | 1.75 ms | — | — | **5.68x** | **—** |
| table | narrow_100000 | projection | 1.25 MB | CPU | on | **315.0 μs** | 239.4 μs | 1.78 ms | — | — | **7.42x** | **—** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | on | **426.4 μs** | 331.3 μs | 1.92 ms | — | — | **5.78x** | **—** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | on | **149.8 μs** | 83.5 μs | 2.06 ms | — | — | **24.64x** | **—** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | on | **32.1 μs** | 31.7 μs | 458.6 μs | — | — | **14.46x** | **—** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **151.2 μs** | 180.2 μs | 1.45 ms | — | — | **9.61x** | **—** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | on | **90.5 μs** | 129.3 μs | 1.40 ms | — | — | **15.44x** | **—** |
| table | narrow_10000 | projection | 0.13 MB | CPU | on | **128.5 μs** | 64.5 μs | 1.40 ms | — | — | **21.69x** | **—** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | on | **144.0 μs** | 76.8 μs | 1.44 ms | — | — | **18.77x** | **—** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | on | **121.3 μs** | 57.9 μs | 1.82 ms | — | — | **31.42x** | **—** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | on | **30.2 μs** | 29.5 μs | 474.2 μs | — | — | **16.08x** | **—** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **53.7 μs** | 101.4 μs | 1.35 ms | — | — | **25.16x** | **—** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | on | **52.1 μs** | 98.1 μs | 1.34 ms | — | — | **25.65x** | **—** |
| table | narrow_1000 | projection | 19.7 KB | CPU | on | **115.1 μs** | 50.6 μs | 1.35 ms | — | — | **26.66x** | **—** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | on | **115.2 μs** | 51.9 μs | 1.38 ms | — | — | **26.52x** | **—** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | on | **118.1 μs** | 51.9 μs | 1.78 ms | — | — | **34.24x** | **—** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | on | **30.6 μs** | 31.2 μs | 451.1 μs | — | — | **14.72x** | **—** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **582.2 μs** | 464.8 μs | 1.81 ms | — | — | **3.90x** | **—** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | on | **447.8 μs** | 462.1 μs | 1.80 ms | — | — | **4.02x** | **—** |
| table | typed_100000 | projection | 2.39 MB | CPU | on | **1.24 ms** | 1.13 ms | 28.82 ms | — | — | **25.44x** | **—** |
| table | typed_100000 | read_full | 2.39 MB | CPU | on | **1.35 ms** | 1.24 ms | 28.82 ms | — | — | **23.22x** | **—** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | on | **260.2 μs** | 185.2 μs | 4.53 ms | — | — | **24.48x** | **—** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | on | **33.5 μs** | 34.9 μs | 461.4 μs | — | — | **13.76x** | **—** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **136.0 μs** | 163.7 μs | 1.45 ms | — | — | **10.67x** | **—** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | on | **131.3 μs** | 160.4 μs | 1.46 ms | — | — | **11.14x** | **—** |
| table | typed_10000 | projection | 0.24 MB | CPU | on | **252.9 μs** | 170.6 μs | 4.08 ms | — | — | **23.93x** | **—** |
| table | typed_10000 | read_full | 0.24 MB | CPU | on | **255.6 μs** | 175.1 μs | 4.09 ms | — | — | **23.36x** | **—** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | on | **142.8 μs** | 71.8 μs | 2.19 ms | — | — | **30.46x** | **—** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | on | **32.0 μs** | 30.1 μs | 445.7 μs | — | — | **14.83x** | **—** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **545.1 μs** | 423.3 μs | 1.59 ms | — | — | **3.77x** | **—** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | on | **407.3 μs** | 403.0 μs | 1.59 ms | — | — | **3.95x** | **—** |
| table | varlen_100000 | projection | 3.06 MB | CPU | on | **72.99 ms** | 71.73 ms | 526.88 ms | — | — | **7.34x** | **—** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | on | **72.78 ms** | 72.04 ms | 530.47 ms | — | — | **7.36x** | **—** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | on | **7.12 ms** | 7.03 ms | 54.40 ms | — | — | **7.74x** | **—** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | on | **33.4 μs** | 32.8 μs | 459.6 μs | — | — | **14.03x** | **—** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **131.7 μs** | 169.4 μs | 1.34 ms | — | — | **10.16x** | **—** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | on | **128.3 μs** | 162.3 μs | 1.34 ms | — | — | **10.43x** | **—** |
| table | varlen_10000 | projection | 0.31 MB | CPU | on | **7.09 ms** | 7.04 ms | 53.34 ms | — | — | **7.58x** | **—** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | on | **7.09 ms** | 7.04 ms | 53.66 ms | — | — | **7.62x** | **—** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | on | **863.5 μs** | 817.8 μs | 7.06 ms | — | — | **8.63x** | **—** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | on | **32.7 μs** | 29.5 μs | 447.1 μs | — | — | **15.15x** | **—** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **56.8 μs** | 103.1 μs | 1.30 ms | — | — | **22.81x** | **—** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | on | **58.8 μs** | 101.3 μs | 1.28 ms | — | — | **21.84x** | **—** |
| table | varlen_1000 | projection | 39.4 KB | CPU | on | **866.9 μs** | 809.3 μs | 6.61 ms | — | — | **8.17x** | **—** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | on | **870.6 μs** | 782.8 μs | 6.61 ms | — | — | **8.44x** | **—** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | on | **265.3 μs** | 211.5 μs | 2.26 ms | — | — | **10.70x** | **—** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | on | **29.5 μs** | 27.4 μs | 434.4 μs | — | — | **15.85x** | **—** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **1.53 ms** | 1.31 ms | 7.45 ms | — | — | **5.69x** | **—** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | on | **1.18 ms** | 1.16 ms | 7.14 ms | — | — | **6.17x** | **—** |
| table | wide_100000 | projection | 20.71 MB | CPU | on | **1.63 ms** | 1.54 ms | 7.53 ms | — | — | **4.88x** | **—** |
| table | wide_100000 | read_full | 20.71 MB | CPU | on | **13.88 ms** | 13.56 ms | 126.63 ms | — | — | **9.34x** | **—** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | on | **770.4 μs** | 674.9 μs | 20.63 ms | — | — | **30.58x** | **—** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | on | **42.9 μs** | 40.0 μs | 584.1 μs | — | — | **14.60x** | **—** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **269.3 μs** | 262.4 μs | 5.83 ms | — | — | **22.23x** | **—** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | on | **202.0 μs** | 210.7 μs | 5.78 ms | — | — | **28.61x** | **—** |
| table | wide_10000 | projection | 2.08 MB | CPU | on | **236.6 μs** | 158.6 μs | 5.79 ms | — | — | **36.48x** | **—** |
| table | wide_10000 | read_full | 2.08 MB | CPU | on | **769.9 μs** | 677.2 μs | 16.48 ms | — | — | **24.34x** | **—** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | on | **218.5 μs** | 148.3 μs | 10.43 ms | — | — | **70.34x** | **—** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | on | **47.2 μs** | 42.0 μs | 576.6 μs | — | — | **13.71x** | **—** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **73.5 μs** | 116.0 μs | 5.61 ms | — | — | **76.37x** | **—** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | on | **73.8 μs** | 113.5 μs | 5.63 ms | — | — | **76.20x** | **—** |
| table | wide_1000 | projection | 0.22 MB | CPU | on | **130.5 μs** | 64.5 μs | 5.66 ms | — | — | **87.80x** | **—** |
| table | wide_1000 | read_full | 0.22 MB | CPU | on | **211.9 μs** | 142.3 μs | 7.10 ms | — | — | **49.86x** | **—** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | on | **172.8 μs** | 97.1 μs | 9.35 ms | — | — | **96.28x** | **—** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | on | **40.7 μs** | 40.1 μs | 590.2 μs | — | — | **14.73x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance comparisons & edge cases {#performance-deficits}

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (CPU and GPU). GPU lags may reflect software or hardware limits — they are listed, not hidden.

| Platform | Domain | Case | mmap | torchfits | Peak RSS (MB) | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| Linux x86_64 / CPU | tensor | large_int8_1d [read_full] | off | 232.5 μs | 296.6 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 45.78 ms | 309.4 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 45.71 ms | 294.0 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 45.72 ms | 309.4 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 45.66 ms | 294.0 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | table | narrow_100000 [read_full] | off | 681.1 μs | 380.4 | fitsio/fitsio_torch | 1.13× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 5.67 ms | 443.2 | fitsio/fitsio_torch | 1.06× |
| Linux x86_64 / CUDA | tensor | tiny_float64_1d [read_full @ cuda] | off | 116.1 μs | 760.0 | fitsio/fitsio_torch_device | 1.13× |
| Linux x86_64 / CUDA | tensor | tiny_int32_1d [read_full @ cuda] | off | 107.4 μs | 760.0 | fitsio/fitsio_torch_device | 1.13× |
| Linux x86_64 / CUDA | tensor | small_int16_1d [read_full @ cuda] | off | 115.1 μs | 760.0 | fitsio/fitsio_torch_device | 1.12× |
| Linux x86_64 / CUDA | tensor | small_float64_1d [read_full @ cuda] | off | 133.4 μs | 760.0 | fitsio/fitsio_torch_device | 1.12× |
| Linux x86_64 / CUDA | tensor | tiny_int16_2d [read_full @ cuda] | off | 115.6 μs | 760.0 | fitsio/fitsio_torch_device | 1.10× |
| Linux x86_64 / CUDA | tensor | tiny_int64_2d [read_full @ cuda] | off | 115.5 μs | 760.0 | fitsio/fitsio_torch_device | 1.08× |
| Linux x86_64 / CUDA | tensor | small_int16_2d [read_full @ cuda] | off | 142.6 μs | 760.0 | fitsio/fitsio_torch_device | 1.07× |
| Linux x86_64 / CUDA | tensor | tiny_float32_2d [read_full @ cuda] | off | 111.3 μs | 760.0 | fitsio/fitsio_torch_device | 1.06× |
| Linux x86_64 / CUDA | tensor | small_int32_1d [read_full @ cuda] | off | 124.1 μs | 760.0 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | tiny_float32_1d [read_full @ cuda] | off | 105.5 μs | 760.0 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | tiny_int64_3d [read_full @ cuda] | off | 115.5 μs | 760.0 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | off | 30.71 ms | 728.1 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | tiny_int16_3d [read_full @ cuda] | off | 108.4 μs | 760.0 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | small_int8_2d [read_full @ cuda] | off | 126.8 μs | 760.0 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | small_int8_3d [read_full @ cuda] | off | 158.9 μs | 760.0 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | on | 30.45 ms | 605.8 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | off | 30.37 ms | 727.0 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | tensor | scaled_small [read_full @ cuda] | off | 195.9 μs | 760.0 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | on | 30.45 ms | 698.0 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | tiny_int8_1d [read_full @ cuda] | off | 101.1 μs | 760.0 | fitsio/fitsio_torch_device | 1.02× |
| Linux x86_64 / CUDA | tensor | tiny_int8_3d [read_full @ cuda] | off | 105.9 μs | 760.0 | fitsio/fitsio_torch_device | 1.01× |
| Linux x86_64 / CUDA | tensor | tiny_int32_3d [read_full @ cuda] | off | 110.1 μs | 760.0 | fitsio/fitsio_torch_device | 1.01× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | on | 30.55 ms | 698.0 | fitsio/fitsio_torch_device_specialized | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | off | 30.45 ms | 728.1 | fitsio/fitsio_torch_device_specialized | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | on | 30.33 ms | 605.8 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | off | 30.21 ms | 727.0 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CUDA | tensor | medium_int8_3d [read_full] | off | 463.8 μs | 758.9 | fitsio/fitsio_torch | 1.01× |
| Linux x86_64 / CUDA | table | narrow_1000000 [read_full] | off | 10.39 ms | 757.4 | fitsio/fitsio_torch | 1.17× |
| Linux x86_64 / CUDA | table | narrow_1000000 [read_full] | off | 10.39 ms | 757.4 | fitsio/fitsio | 1.14× |
<!-- BENCH_DEFICITS_END -->

### Host scorecard

| Platform | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| Linux x86_64 / CPU | `exhaustive_cpu_20260822_152204` | 3057 | 7 | 294.0 | lab + mmap-matrix |
| Linux x86_64 / CUDA | `exhaustive_cuda_20260822_152235` | 4315 | 29 | 739.3 | lab + mmap-matrix + GPU |
<!-- BENCH_HOSTS_END -->

Round-3 soak (post thin-I/O): MPS `exhaustive_mps_20260719_143706` (local);
CANFAR staging CPU `exhaustive_cpu_20260719_144337` and CUDA
`exhaustive_cuda_20260719_144457` (clone `bench/thin-io-scorecard` @ 9b9e7cf).
ML loader: `ml_20260719_145743`. MegaCam: `20260719_075555`.



Latest local quick benchmark evidence:

<!-- BENCH_QUICK_BEGIN -->
| Run ID | Scope | Command | Rows | Deficits |
|---|---|---|---:|---:|
| — | FITS image I/O | _(no run yet)_ | — | — |
| — | FITS table I/O | _(no run yet)_ | — | — |
<!-- BENCH_QUICK_END -->

### ML DataLoader throughput

<!-- BENCH_ML_BEGIN -->
_Run `pixi run bench-ml` to populate ML loader throughput._
<!-- BENCH_ML_END -->

### CFHT MegaCam MEF cutouts (local)

<!-- BENCH_MEGACAM_BEGIN -->
Source: `docs/assets/bench/20260719_075555/megacam_results.csv` (160 OK rows).
Median throughput over OK rows (earlier table values were copy-paste μs from unrelated suites).

| Method | Median throughput |
|---|---:|
| `fitsio_cached` | 52.7 MB/s |
| `torchfits_cached` | 49.3 MB/s |
| `torchfits_materialize` | 119.4 MB/s |
| `torchfits_naive` | 50.6 MB/s |
<!-- BENCH_MEGACAM_END -->


Keep this page current with the latest tensor and table benchmark run before
making performance claims.
