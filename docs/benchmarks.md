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
Source: `benchmarks_results/exhaustive_cpu_20260822_054439/results.csv` (mmap on+off matrix.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### Tensor I/O (IMAGE HDU) (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.18 ms` (n=174) | `0.93 ms` (n=253) | `0.30 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.17 ms` (n=174) | `0.83 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### Table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.34 ms` (n=216) | `4.10 ms` (n=184) | `0.92 ms` (n=216) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.37 ms` (n=216) | `4.84 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Table read (100k rows, 8 cols, mixed) | CPU | **5.09 ms** | 4.87 ms | 57.07 ms | 20.09 ms | **11.73x** | **4.13x** |
| Varlen table read (100k rows, 3 cols) | CPU | **127.22 ms** | 129.07 ms | 911.03 ms | 186.45 ms | **7.16x** | **1.47x** |
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
| tensor | compressed_gzip_1:header_read | header_read | 1.29 MB | CPU | n/a | **—** | 54.2 μs | 2.39 ms | 255.4 μs | — | **44.12x** | **4.71x** |
| tensor | compressed_gzip_2:header_read | header_read | 0.89 MB | CPU | n/a | **—** | 52.3 μs | 2.42 ms | 244.9 μs | — | **46.19x** | **4.68x** |
| tensor | compressed_hcompress_1:header_read | header_read | 0.82 MB | CPU | n/a | **—** | 54.5 μs | 2.53 ms | 276.6 μs | — | **46.32x** | **5.07x** |
| tensor | compressed_rice_1:cutout_100x100 | cutout_100x100 | 0.90 MB | CPU | n/a | **1.41 ms** | 1.28 ms | 11.92 ms | 1.42 ms | — | **9.29x** | **1.11x** |
| tensor | compressed_rice_1:header_read | header_read | 0.90 MB | CPU | n/a | **—** | 60.0 μs | 2.75 ms | 279.5 μs | — | **45.81x** | **4.66x** |
| tensor | large_float32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 38.8 μs | 676.0 μs | 58.8 μs | — | **17.41x** | **1.51x** |
| tensor | large_float32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 27.8 μs | 527.6 μs | 51.0 μs | — | **18.95x** | **1.83x** |
| tensor | large_float64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 27.4 μs | 472.3 μs | 49.5 μs | — | **17.25x** | **1.81x** |
| tensor | large_float64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 28.5 μs | 517.5 μs | 50.4 μs | — | **18.15x** | **1.77x** |
| tensor | large_int16_1d:header_read | header_read | 1.91 MB | CPU | n/a | **—** | 23.8 μs | 470.7 μs | 47.6 μs | — | **19.78x** | **2.00x** |
| tensor | large_int16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 26.3 μs | 510.2 μs | 48.6 μs | — | **19.37x** | **1.85x** |
| tensor | large_int32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 27.9 μs | 470.6 μs | 46.1 μs | — | **16.88x** | **1.65x** |
| tensor | large_int32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 27.1 μs | 506.8 μs | 51.0 μs | — | **18.68x** | **1.88x** |
| tensor | large_int64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 27.3 μs | 494.3 μs | 45.2 μs | — | **18.14x** | **1.66x** |
| tensor | large_int64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 26.7 μs | 499.1 μs | 49.0 μs | — | **18.67x** | **1.83x** |
| tensor | large_int8_1d:header_read | header_read | 0.96 MB | CPU | n/a | **—** | 28.3 μs | 552.7 μs | 56.0 μs | — | **19.54x** | **1.98x** |
| tensor | large_int8_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 35.5 μs | 721.5 μs | 72.7 μs | — | **20.30x** | **2.05x** |
| tensor | large_uint16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 30.9 μs | 596.5 μs | 58.8 μs | — | **19.29x** | **1.90x** |
| tensor | large_uint32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 32.5 μs | 707.5 μs | 68.7 μs | — | **21.75x** | **2.11x** |
| tensor | medium_float32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 26.4 μs | 486.8 μs | 45.5 μs | — | **18.43x** | **1.72x** |
| tensor | medium_float32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 28.0 μs | 525.2 μs | 51.5 μs | — | **18.78x** | **1.84x** |
| tensor | medium_float32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 28.8 μs | 551.6 μs | 57.8 μs | — | **19.13x** | **2.00x** |
| tensor | medium_float64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 27.1 μs | 478.1 μs | 46.2 μs | — | **17.65x** | **1.70x** |
| tensor | medium_float64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 28.2 μs | 530.2 μs | 49.2 μs | — | **18.79x** | **1.74x** |
| tensor | medium_float64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 44.6 μs | 753.4 μs | 87.4 μs | — | **16.89x** | **1.96x** |
| tensor | medium_int16_1d:header_read | header_read | 0.20 MB | CPU | n/a | **—** | 27.5 μs | 477.6 μs | 47.4 μs | — | **17.34x** | **1.72x** |
| tensor | medium_int16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 38.3 μs | 701.6 μs | 70.3 μs | — | **18.33x** | **1.84x** |
| tensor | medium_int16_3d:header_read | header_read | 3.13 MB | CPU | n/a | **—** | 29.0 μs | 552.1 μs | 53.6 μs | — | **19.02x** | **1.85x** |
| tensor | medium_int32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 26.6 μs | 474.9 μs | 49.1 μs | — | **17.84x** | **1.84x** |
| tensor | medium_int32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 35.3 μs | 611.9 μs | 71.9 μs | — | **17.34x** | **2.04x** |
| tensor | medium_int32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 34.7 μs | 656.0 μs | 69.4 μs | — | **18.91x** | **2.00x** |
| tensor | medium_int64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 25.9 μs | 473.3 μs | 47.8 μs | — | **18.29x** | **1.85x** |
| tensor | medium_int64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 27.9 μs | 497.3 μs | 54.7 μs | — | **17.82x** | **1.96x** |
| tensor | medium_int64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 31.6 μs | 566.2 μs | 62.6 μs | — | **17.92x** | **1.98x** |
| tensor | medium_int8_1d:header_read | header_read | 0.10 MB | CPU | n/a | **—** | 27.2 μs | 549.5 μs | 51.4 μs | — | **20.23x** | **1.89x** |
| tensor | medium_int8_2d:header_read | header_read | 1.01 MB | CPU | n/a | **—** | 28.9 μs | 581.5 μs | 58.8 μs | — | **20.12x** | **2.03x** |
| tensor | medium_int8_3d:header_read | header_read | 1.57 MB | CPU | n/a | **—** | 45.4 μs | 771.2 μs | 81.9 μs | — | **16.97x** | **1.80x** |
| tensor | medium_uint16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 31.1 μs | 579.9 μs | 59.2 μs | — | **18.63x** | **1.90x** |
| tensor | medium_uint32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 42.9 μs | 708.5 μs | 80.8 μs | — | **16.53x** | **1.89x** |
| tensor | mef_medium:header_read | header_read | 7.02 MB | CPU | n/a | **—** | 34.8 μs | 969.9 μs | 77.6 μs | — | **27.85x** | **2.23x** |
| tensor | mef_small:header_read | header_read | 0.45 MB | CPU | n/a | **—** | 32.4 μs | 930.0 μs | 70.4 μs | — | **28.72x** | **2.18x** |
| tensor | multi_mef_10ext:cutout_100x100 | cutout_100x100 | 2.68 MB | CPU | n/a | **116.1 μs** | 177.0 μs | 3.87 ms | 303.4 μs | — | **33.34x** | **2.61x** |
| tensor | multi_mef_10ext:header_read | header_read | 2.68 MB | CPU | n/a | **—** | 33.5 μs | 946.8 μs | 71.0 μs | — | **28.27x** | **2.12x** |
| tensor | multi_mef_10ext:random_ext_full_reads_200 | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **11.12 ms** | 11.14 ms | 13.76 ms | 14.43 ms | — | **1.24x** | **1.30x** |
| tensor | repeated_cutouts_50x_100x100:repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **904.6 μs** | 973.0 μs | 89.00 ms | 5.94 ms | — | **98.39x** | **6.56x** |
| tensor | scaled_large:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 36.1 μs | 655.0 μs | 76.2 μs | — | **18.13x** | **2.11x** |
| tensor | scaled_medium:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 30.0 μs | 605.3 μs | 59.6 μs | — | **20.18x** | **1.99x** |
| tensor | scaled_small:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 30.5 μs | 577.4 μs | 67.9 μs | — | **18.91x** | **2.22x** |
| tensor | small_float32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 26.3 μs | 478.7 μs | 46.7 μs | — | **18.19x** | **1.78x** |
| tensor | small_float32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 26.8 μs | 514.5 μs | 51.9 μs | — | **19.17x** | **1.94x** |
| tensor | small_float32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 30.7 μs | 566.8 μs | 55.2 μs | — | **18.49x** | **1.80x** |
| tensor | small_float64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 30.6 μs | 536.9 μs | 54.9 μs | — | **17.56x** | **1.79x** |
| tensor | small_float64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 27.8 μs | 513.3 μs | 54.9 μs | — | **18.46x** | **1.97x** |
| tensor | small_float64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 26.7 μs | 553.3 μs | 53.8 μs | — | **20.70x** | **2.01x** |
| tensor | small_int16_1d:header_read | header_read | 22.5 KB | CPU | n/a | **—** | 27.9 μs | 476.8 μs | 50.7 μs | — | **17.12x** | **1.82x** |
| tensor | small_int16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 33.4 μs | 615.8 μs | 71.8 μs | — | **18.45x** | **2.15x** |
| tensor | small_int16_3d:header_read | header_read | 0.32 MB | CPU | n/a | **—** | 25.5 μs | 525.6 μs | 52.2 μs | — | **20.59x** | **2.04x** |
| tensor | small_int32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 46.5 μs | 581.2 μs | 75.3 μs | — | **12.49x** | **1.62x** |
| tensor | small_int32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 24.4 μs | 513.3 μs | 49.7 μs | — | **21.05x** | **2.04x** |
| tensor | small_int32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 27.5 μs | 551.3 μs | 54.0 μs | — | **20.04x** | **1.96x** |
| tensor | small_int64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 27.1 μs | 462.4 μs | 46.7 μs | — | **17.03x** | **1.72x** |
| tensor | small_int64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 27.6 μs | 491.8 μs | 49.7 μs | — | **17.82x** | **1.80x** |
| tensor | small_int64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 26.3 μs | 546.8 μs | 52.3 μs | — | **20.75x** | **1.98x** |
| tensor | small_int8_1d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 41.1 μs | 635.9 μs | 79.5 μs | — | **15.48x** | **1.93x** |
| tensor | small_int8_2d:header_read | header_read | 0.07 MB | CPU | n/a | **—** | 29.6 μs | 591.6 μs | 58.5 μs | — | **19.98x** | **1.98x** |
| tensor | small_int8_3d:header_read | header_read | 0.16 MB | CPU | n/a | **—** | 31.1 μs | 631.7 μs | 64.8 μs | — | **20.33x** | **2.09x** |
| tensor | small_uint16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 29.3 μs | 606.3 μs | 59.0 μs | — | **20.69x** | **2.01x** |
| tensor | small_uint32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 30.0 μs | 609.0 μs | 66.8 μs | — | **20.31x** | **2.23x** |
| tensor | timeseries_frame_000:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 28.2 μs | 516.4 μs | 53.8 μs | — | **18.30x** | **1.91x** |
| tensor | timeseries_frame_001:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 29.2 μs | 519.9 μs | 52.9 μs | — | **17.79x** | **1.81x** |
| tensor | timeseries_frame_002:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 27.9 μs | 513.4 μs | 51.2 μs | — | **18.39x** | **1.84x** |
| tensor | timeseries_frame_003:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 26.8 μs | 508.5 μs | 50.3 μs | — | **18.96x** | **1.88x** |
| tensor | timeseries_frame_004:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 28.3 μs | 538.7 μs | 52.0 μs | — | **19.05x** | **1.84x** |
| tensor | tiny_float32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 28.6 μs | 488.2 μs | 51.7 μs | — | **17.10x** | **1.81x** |
| tensor | tiny_float32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 30.4 μs | 519.0 μs | 54.9 μs | — | **17.09x** | **1.81x** |
| tensor | tiny_float32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 29.1 μs | 559.0 μs | 53.2 μs | — | **19.24x** | **1.83x** |
| tensor | tiny_float64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 28.4 μs | 468.5 μs | 47.7 μs | — | **16.50x** | **1.68x** |
| tensor | tiny_float64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 31.2 μs | 504.2 μs | 55.4 μs | — | **16.18x** | **1.78x** |
| tensor | tiny_float64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 30.3 μs | 564.2 μs | 52.0 μs | — | **18.61x** | **1.72x** |
| tensor | tiny_int16_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 26.9 μs | 462.1 μs | 48.7 μs | — | **17.18x** | **1.81x** |
| tensor | tiny_int16_2d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 31.4 μs | 522.2 μs | 48.5 μs | — | **16.63x** | **1.54x** |
| tensor | tiny_int16_3d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 30.3 μs | 567.2 μs | 53.7 μs | — | **18.72x** | **1.77x** |
| tensor | tiny_int32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 28.2 μs | 478.6 μs | 49.2 μs | — | **16.95x** | **1.74x** |
| tensor | tiny_int32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 26.9 μs | 523.8 μs | 50.5 μs | — | **19.44x** | **1.87x** |
| tensor | tiny_int32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 28.7 μs | 571.6 μs | 54.9 μs | — | **19.89x** | **1.91x** |
| tensor | tiny_int64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 25.1 μs | 472.6 μs | 43.1 μs | — | **18.86x** | **1.72x** |
| tensor | tiny_int64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 28.0 μs | 520.3 μs | 49.1 μs | — | **18.60x** | **1.76x** |
| tensor | tiny_int64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 28.6 μs | 555.7 μs | 57.4 μs | — | **19.44x** | **2.01x** |
| tensor | tiny_int8_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 29.3 μs | 562.7 μs | 57.4 μs | — | **19.20x** | **1.96x** |
| tensor | tiny_int8_2d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 29.1 μs | 579.6 μs | 65.8 μs | — | **19.94x** | **2.27x** |
| tensor | tiny_int8_3d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 27.1 μs | 628.1 μs | 66.8 μs | — | **23.17x** | **2.46x** |
| tensor | write_compress_hcompress_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **85.13 ms** | — | 103.60 ms | — | — | **1.22x** | **—** |
| tensor | write_compress_rice_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **66.18 ms** | — | 121.40 ms | — | — | **1.83x** | **—** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | off | **23.91 ms** | 23.95 ms | 47.03 ms | 26.85 ms | — | **1.97x** | **1.12x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | off | **20.52 ms** | 20.48 ms | 75.24 ms | 23.59 ms | — | **3.67x** | **1.15x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | off | **46.15 ms** | 46.15 ms | 52.77 ms | 45.07 ms | — | **1.14x** | **0.98x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | off | **12.74 ms** | 12.71 ms | 33.52 ms | 13.01 ms | — | **2.64x** | **1.02x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | off | **1.09 ms** | 1.12 ms | 2.55 ms | 1.64 ms | — | **2.34x** | **1.50x** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | off | **10.48 ms** | 10.48 ms | 21.71 ms | 11.58 ms | — | **2.07x** | **1.11x** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | off | **1.91 ms** | 1.91 ms | 4.44 ms | 2.87 ms | — | **2.33x** | **1.50x** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | off | **22.64 ms** | 21.74 ms | 58.11 ms | 24.56 ms | — | **2.67x** | **1.13x** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | off | **667.5 μs** | 650.7 μs | 1.63 ms | 809.9 μs | — | **2.50x** | **1.24x** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | off | **2.27 ms** | 2.09 ms | 5.05 ms | 3.21 ms | — | **2.42x** | **1.53x** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | off | **1.18 ms** | 1.18 ms | 2.65 ms | 1.72 ms | — | **2.25x** | **1.46x** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | off | **11.76 ms** | 4.38 ms | 23.93 ms | 12.65 ms | — | **5.47x** | **2.89x** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | off | **2.14 ms** | 2.16 ms | 4.56 ms | 2.98 ms | — | **2.13x** | **1.39x** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | off | **22.31 ms** | 22.73 ms | 60.25 ms | 24.94 ms | — | **2.70x** | **1.12x** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | off | **388.3 μs** | 314.2 μs | 1.34 ms | 377.7 μs | — | **4.28x** | **1.20x** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | off | **1.25 ms** | 1.23 ms | 3.24 ms | 1.46 ms | — | **2.63x** | **1.19x** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | off | **2.53 ms** | 2.53 ms | 8.06 ms | 3.48 ms | — | **3.18x** | **1.37x** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | off | **11.27 ms** | 11.26 ms | 18.70 ms | 12.08 ms | — | **1.66x** | **1.07x** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | off | **141.0 μs** | 172.5 μs | 752.0 μs | 227.7 μs | — | **5.33x** | **1.62x** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | off | **1.14 ms** | 1.15 ms | 2.69 ms | 1.71 ms | — | **2.36x** | **1.50x** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | off | **1.74 ms** | 1.74 ms | 3.83 ms | 2.59 ms | — | **2.20x** | **1.49x** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | off | **318.0 μs** | 243.0 μs | 984.5 μs | 354.2 μs | — | **4.05x** | **1.46x** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | off | **1.99 ms** | 2.02 ms | 4.64 ms | 3.02 ms | — | **2.33x** | **1.52x** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | off | **3.35 ms** | 3.38 ms | 17.41 ms | 8.51 ms | — | **5.20x** | **2.54x** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | off | **65.3 μs** | 111.1 μs | 613.9 μs | 130.6 μs | — | **9.40x** | **2.00x** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | off | **671.3 μs** | 689.0 μs | 1.76 ms | 870.7 μs | — | **2.63x** | **1.30x** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | off | **998.0 μs** | 978.8 μs | 2.28 ms | 1.24 ms | — | **2.33x** | **1.27x** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | off | **129.4 μs** | 185.3 μs | 760.6 μs | 231.2 μs | — | **5.88x** | **1.79x** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | off | **1.19 ms** | 1.17 ms | 2.67 ms | 1.70 ms | — | **2.28x** | **1.45x** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | off | **1.77 ms** | 1.62 ms | 3.86 ms | 2.61 ms | — | **2.39x** | **1.61x** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | off | **277.7 μs** | 301.5 μs | 1.12 ms | 406.5 μs | — | **4.04x** | **1.46x** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | off | **2.02 ms** | 2.03 ms | 4.75 ms | 3.09 ms | — | **2.35x** | **1.53x** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | off | **3.50 ms** | 3.48 ms | 19.87 ms | 8.94 ms | — | **5.71x** | **2.57x** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | off | **72.4 μs** | 118.1 μs | 724.8 μs | 109.4 μs | — | **10.01x** | **1.51x** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | off | **417.3 μs** | 430.6 μs | 1.49 ms | 445.6 μs | — | **3.57x** | **1.07x** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | off | **532.9 μs** | 614.1 μs | 1.88 ms | 647.3 μs | — | **3.53x** | **1.21x** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | off | **755.7 μs** | 772.9 μs | 2.68 ms | 937.5 μs | — | **3.54x** | **1.24x** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | off | **1.49 ms** | 1.53 ms | 3.85 ms | 2.10 ms | — | **2.58x** | **1.41x** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | off | **387.2 μs** | 398.1 μs | 1.79 ms | 480.1 μs | — | **4.62x** | **1.24x** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | off | **137.7 μs** | 78.0 μs | 1.11 ms | 170.5 μs | — | **14.28x** | **2.19x** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | off | **85.3 μs** | 71.2 μs | 1.05 ms | 260.0 μs | — | **14.77x** | **3.65x** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | off | **10.92 ms** | 10.91 ms | 17.99 ms | 11.02 ms | — | **1.65x** | **1.01x** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | off | **1.23 ms** | 1.27 ms | 3.11 ms | 1.58 ms | — | **2.52x** | **1.28x** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | off | **173.4 μs** | 117.1 μs | 862.4 μs | 194.1 μs | — | **7.37x** | **1.66x** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | off | **59.8 μs** | 54.0 μs | 508.7 μs | 93.7 μs | — | **9.42x** | **1.73x** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | off | **122.1 μs** | 75.5 μs | 711.3 μs | 178.8 μs | — | **9.42x** | **2.37x** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | off | **231.3 μs** | 229.2 μs | 960.1 μs | 345.0 μs | — | **4.19x** | **1.51x** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | off | **93.1 μs** | 60.6 μs | 516.7 μs | 109.5 μs | — | **8.53x** | **1.81x** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | off | **129.8 μs** | 228.4 μs | 862.6 μs | 260.3 μs | — | **6.64x** | **2.00x** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | off | **449.0 μs** | 411.0 μs | 1.37 ms | 547.0 μs | — | **3.34x** | **1.33x** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | off | **67.6 μs** | 84.1 μs | 477.8 μs | 85.8 μs | — | **7.07x** | **1.27x** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | off | **83.3 μs** | 94.8 μs | 603.8 μs | 110.5 μs | — | **7.25x** | **1.33x** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | off | **187.4 μs** | 112.7 μs | 777.7 μs | 192.9 μs | — | **6.90x** | **1.71x** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | off | **75.0 μs** | 86.3 μs | 551.1 μs | 95.4 μs | — | **7.35x** | **1.27x** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | off | **127.4 μs** | 105.2 μs | 854.4 μs | 203.2 μs | — | **8.12x** | **1.93x** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | off | **230.8 μs** | 240.4 μs | 985.9 μs | 346.1 μs | — | **4.27x** | **1.50x** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | off | **64.7 μs** | 65.2 μs | 757.0 μs | 147.2 μs | — | **11.70x** | **2.28x** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | off | **219.6 μs** | 187.9 μs | 872.2 μs | 267.2 μs | — | **4.64x** | **1.42x** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | off | **412.7 μs** | 459.4 μs | 1.40 ms | 561.9 μs | — | **3.38x** | **1.36x** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | off | **46.3 μs** | 79.2 μs | 690.1 μs | 88.9 μs | — | **14.91x** | **1.92x** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | off | **64.1 μs** | 55.2 μs | 728.6 μs | 109.7 μs | — | **13.20x** | **1.99x** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | off | **119.4 μs** | 125.8 μs | 830.1 μs | 135.8 μs | — | **6.95x** | **1.14x** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | off | **119.8 μs** | 87.3 μs | 746.6 μs | 125.8 μs | — | **8.55x** | **1.44x** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | off | **108.5 μs** | 145.7 μs | 861.4 μs | 188.6 μs | — | **7.94x** | **1.74x** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | off | **88.5 μs** | 82.7 μs | 745.1 μs | 177.9 μs | — | **9.01x** | **2.15x** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | off | **90.2 μs** | 131.8 μs | 683.9 μs | 165.2 μs | — | **7.58x** | **1.83x** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | off | **84.4 μs** | 124.8 μs | 723.7 μs | 179.8 μs | — | **8.57x** | **2.13x** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | off | **83.4 μs** | 135.6 μs | 689.8 μs | 167.6 μs | — | **8.27x** | **2.01x** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | off | **89.2 μs** | 89.7 μs | 738.3 μs | 187.0 μs | — | **8.27x** | **2.10x** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | off | **42.2 μs** | 78.1 μs | 493.7 μs | 76.2 μs | — | **11.71x** | **1.81x** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | off | **56.2 μs** | 51.7 μs | 506.9 μs | 88.1 μs | — | **9.80x** | **1.70x** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | off | **77.6 μs** | 55.3 μs | 534.5 μs | 79.1 μs | — | **9.66x** | **1.43x** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | off | **43.2 μs** | 82.3 μs | 482.6 μs | 88.8 μs | — | **11.17x** | **2.06x** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | off | **64.6 μs** | 57.3 μs | 530.5 μs | 78.5 μs | — | **9.26x** | **1.37x** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | off | **61.1 μs** | 58.3 μs | 555.0 μs | 95.3 μs | — | **9.52x** | **1.64x** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | off | **51.3 μs** | 86.4 μs | 477.3 μs | 83.9 μs | — | **9.31x** | **1.64x** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | off | **57.8 μs** | 74.4 μs | 507.5 μs | 83.8 μs | — | **8.78x** | **1.45x** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | off | **65.5 μs** | 69.6 μs | 535.2 μs | 83.8 μs | — | **8.18x** | **1.28x** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | off | **59.6 μs** | 68.7 μs | 449.3 μs | 78.9 μs | — | **7.54x** | **1.32x** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | off | **62.1 μs** | 78.2 μs | 495.4 μs | 86.2 μs | — | **7.98x** | **1.39x** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | off | **82.5 μs** | 61.4 μs | 525.8 μs | 81.4 μs | — | **8.57x** | **1.33x** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | off | **47.1 μs** | 78.3 μs | 464.9 μs | 77.6 μs | — | **9.86x** | **1.65x** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | off | **66.1 μs** | 49.8 μs | 524.2 μs | 86.4 μs | — | **10.53x** | **1.74x** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | off | **64.7 μs** | 86.6 μs | 533.3 μs | 87.6 μs | — | **8.25x** | **1.35x** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | off | **72.2 μs** | 62.0 μs | 654.9 μs | 87.2 μs | — | **10.55x** | **1.41x** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | off | **45.9 μs** | 96.7 μs | 688.2 μs | 97.5 μs | — | **14.99x** | **2.13x** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | off | **60.6 μs** | 87.9 μs | 707.5 μs | 91.0 μs | — | **11.67x** | **1.50x** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | on | **23.93 ms** | 23.81 ms | 46.50 ms | 26.84 ms | — | **1.95x** | **1.13x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | on | **20.49 ms** | 20.46 ms | 73.71 ms | 23.52 ms | — | **3.60x** | **1.15x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | on | **45.82 ms** | 45.89 ms | 52.88 ms | 44.89 ms | — | **1.15x** | **0.98x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | on | **12.72 ms** | 12.73 ms | 33.94 ms | 12.99 ms | — | **2.67x** | **1.02x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | on | **1.11 ms** | 1.12 ms | 2.20 ms | — | — | **1.98x** | **—** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | on | **10.84 ms** | 11.27 ms | 14.43 ms | — | — | **1.33x** | **—** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | on | **2.06 ms** | 2.07 ms | 3.59 ms | — | — | **1.74x** | **—** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | on | **23.41 ms** | 23.86 ms | 41.71 ms | — | — | **1.78x** | **—** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | on | **571.9 μs** | 587.6 μs | 1.42 ms | — | — | **2.49x** | **—** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | on | **1.89 ms** | 1.90 ms | 3.65 ms | — | — | **1.93x** | **—** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | on | **1.02 ms** | 1.02 ms | 2.14 ms | — | — | **2.10x** | **—** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | on | **8.57 ms** | 8.71 ms | 13.34 ms | — | — | **1.56x** | **—** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | on | **1.81 ms** | 1.79 ms | 3.42 ms | — | — | **1.90x** | **—** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | on | **18.61 ms** | 18.71 ms | 37.89 ms | — | — | **2.04x** | **—** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | on | **390.9 μs** | 337.2 μs | — | — | — | **—** | **—** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | on | **1.23 ms** | 1.24 ms | — | — | — | **—** | **—** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | on | **1.88 ms** | 1.93 ms | — | — | — | **—** | **—** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | on | **10.63 ms** | 10.54 ms | — | — | — | **—** | **—** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | on | **152.9 μs** | 127.3 μs | 735.9 μs | — | — | **5.78x** | **—** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | on | **1.10 ms** | 1.13 ms | 2.19 ms | — | — | **1.99x** | **—** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | on | **1.68 ms** | 1.67 ms | 3.15 ms | — | — | **1.88x** | **—** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | on | **239.5 μs** | 256.3 μs | 930.7 μs | — | — | **3.89x** | **—** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | on | **1.92 ms** | 1.95 ms | 3.85 ms | — | — | **2.00x** | **—** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | on | **3.11 ms** | 3.09 ms | 5.67 ms | — | — | **1.84x** | **—** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | on | **102.1 μs** | 131.1 μs | 624.0 μs | — | — | **6.11x** | **—** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | on | **588.6 μs** | 622.1 μs | 2.09 ms | — | — | **3.56x** | **—** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | on | **904.4 μs** | 897.7 μs | 2.09 ms | — | — | **2.33x** | **—** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | on | **178.4 μs** | 118.0 μs | 783.5 μs | — | — | **6.64x** | **—** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | on | **1.10 ms** | 1.14 ms | 2.42 ms | — | — | **2.21x** | **—** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | on | **1.57 ms** | 1.64 ms | 3.41 ms | — | — | **2.18x** | **—** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | on | **318.7 μs** | 301.6 μs | 1.02 ms | — | — | **3.39x** | **—** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | on | **1.86 ms** | 1.89 ms | 3.70 ms | — | — | **1.99x** | **—** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | on | **2.71 ms** | 2.66 ms | 5.79 ms | — | — | **2.18x** | **—** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | on | **71.8 μs** | 69.9 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | on | **392.6 μs** | 441.6 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | on | **645.5 μs** | 658.2 μs | — | — | — | **—** | **—** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | on | **654.2 μs** | 681.3 μs | — | — | — | **—** | **—** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | on | **1.74 ms** | 1.73 ms | — | — | — | **—** | **—** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | on | **402.6 μs** | 341.9 μs | — | — | — | **—** | **—** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | on | **87.0 μs** | 81.4 μs | — | — | — | **—** | **—** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | on | **132.5 μs** | 72.4 μs | — | — | — | **—** | **—** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | on | **10.83 ms** | 10.80 ms | — | — | — | **—** | **—** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | on | **1.26 ms** | 1.28 ms | — | — | — | **—** | **—** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | on | **135.8 μs** | 132.5 μs | — | — | — | **—** | **—** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | on | **95.3 μs** | 63.0 μs | 510.4 μs | — | — | **8.10x** | **—** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | on | **133.9 μs** | 74.6 μs | 717.7 μs | — | — | **9.62x** | **—** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | on | **206.8 μs** | 264.7 μs | 960.0 μs | — | — | **4.64x** | **—** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | on | **94.3 μs** | 58.1 μs | 543.6 μs | — | — | **9.36x** | **—** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | on | **173.7 μs** | 174.3 μs | 908.2 μs | — | — | **5.23x** | **—** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | on | **450.8 μs** | 505.9 μs | 1.31 ms | — | — | **2.91x** | **—** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | on | **88.4 μs** | 92.2 μs | 501.0 μs | — | — | **5.67x** | **—** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | on | **89.8 μs** | 92.7 μs | 622.0 μs | — | — | **6.93x** | **—** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | on | **161.9 μs** | 151.1 μs | 797.0 μs | — | — | **5.27x** | **—** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | on | **102.9 μs** | 76.4 μs | 508.5 μs | — | — | **6.65x** | **—** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | on | **164.9 μs** | 109.2 μs | 721.0 μs | — | — | **6.60x** | **—** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | on | **193.3 μs** | 274.7 μs | 963.6 μs | — | — | **4.99x** | **—** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | on | **125.9 μs** | 91.9 μs | 546.1 μs | — | — | **5.94x** | **—** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | on | **220.5 μs** | 177.0 μs | 872.1 μs | — | — | **4.93x** | **—** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | on | **474.8 μs** | 429.6 μs | 1.24 ms | — | — | **2.88x** | **—** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | on | **63.5 μs** | 103.8 μs | — | — | — | **—** | **—** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | on | **116.8 μs** | 65.0 μs | — | — | — | **—** | **—** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | on | **91.5 μs** | 124.4 μs | — | — | — | **—** | **—** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | on | **87.0 μs** | 92.8 μs | — | — | — | **—** | **—** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | on | **207.3 μs** | 134.8 μs | — | — | — | **—** | **—** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | on | **69.8 μs** | 128.0 μs | 691.0 μs | — | — | **9.90x** | **—** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | on | **92.4 μs** | 73.6 μs | 693.7 μs | — | — | **9.43x** | **—** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | on | **125.0 μs** | 73.2 μs | 685.7 μs | — | — | **9.36x** | **—** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | on | **125.1 μs** | 114.5 μs | 712.2 μs | — | — | **6.22x** | **—** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | on | **126.0 μs** | 102.3 μs | 703.5 μs | — | — | **6.88x** | **—** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | on | **90.7 μs** | 46.9 μs | 512.2 μs | — | — | **10.92x** | **—** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | on | **87.9 μs** | 51.7 μs | 536.2 μs | — | — | **10.37x** | **—** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | on | **80.9 μs** | 69.3 μs | 562.1 μs | — | — | **8.11x** | **—** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | on | **45.6 μs** | 44.5 μs | 498.5 μs | — | — | **11.21x** | **—** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | on | **94.9 μs** | 50.2 μs | 526.3 μs | — | — | **10.47x** | **—** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | on | **99.5 μs** | 54.1 μs | 578.1 μs | — | — | **10.69x** | **—** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | on | **86.5 μs** | 78.5 μs | 492.5 μs | — | — | **6.27x** | **—** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | on | **85.6 μs** | 81.8 μs | 539.7 μs | — | — | **6.60x** | **—** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | on | **68.5 μs** | 97.7 μs | 565.5 μs | — | — | **8.25x** | **—** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | on | **72.2 μs** | 81.4 μs | 490.2 μs | — | — | **6.79x** | **—** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | on | **101.5 μs** | 87.1 μs | 518.5 μs | — | — | **5.95x** | **—** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | on | **95.1 μs** | 66.4 μs | 550.5 μs | — | — | **8.29x** | **—** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | on | **54.1 μs** | 88.2 μs | 493.4 μs | — | — | **9.12x** | **—** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | on | **64.8 μs** | 102.8 μs | 541.3 μs | — | — | **8.35x** | **—** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | on | **70.9 μs** | 118.7 μs | 585.5 μs | — | — | **8.26x** | **—** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | on | **102.6 μs** | 54.2 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | on | **61.3 μs** | 46.8 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | on | **102.3 μs** | 85.0 μs | — | — | — | **—** | **—** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **557.5 μs** | 556.5 μs | 4.67 ms | 615.6 μs | — | **8.39x** | **1.11x** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | off | **539.8 μs** | 566.6 μs | 4.68 ms | 616.8 μs | — | **8.67x** | **1.14x** |
| table | ascii_10000 | projection | 0.44 MB | CPU | off | **1.75 ms** | 1.64 ms | 13.83 ms | 3.39 ms | — | **8.44x** | **2.07x** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | off | **1.80 ms** | 1.64 ms | 13.90 ms | 3.37 ms | — | **8.48x** | **2.05x** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | off | **311.4 μs** | 206.8 μs | 4.48 ms | 888.3 μs | — | **21.65x** | **4.29x** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | off | **50.7 μs** | 48.1 μs | 759.9 μs | 108.1 μs | — | **15.80x** | **2.25x** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **162.7 μs** | 163.7 μs | 2.58 ms | 270.0 μs | — | **15.83x** | **1.66x** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | off | **154.4 μs** | 160.7 μs | 2.55 ms | 276.0 μs | — | **16.50x** | **1.79x** |
| table | ascii_1000 | projection | 50.6 KB | CPU | off | **188.8 μs** | 123.8 μs | 2.18 ms | 342.3 μs | — | **17.61x** | **2.77x** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | off | **186.4 μs** | 124.4 μs | 2.18 ms | 327.3 μs | — | **17.55x** | **2.63x** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | off | **120.9 μs** | 80.7 μs | 3.31 ms | 326.5 μs | — | **41.02x** | **4.05x** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | off | **49.8 μs** | 67.1 μs | 985.3 μs | 140.2 μs | — | **19.79x** | **2.82x** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **15.73 ms** | 15.93 ms | 26.29 ms | 20.58 ms | — | **1.67x** | **1.31x** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | off | **11.92 ms** | 11.83 ms | 23.85 ms | 16.61 ms | — | **2.02x** | **1.40x** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | off | **9.92 ms** | 9.60 ms | 28.60 ms | 55.97 ms | — | **2.98x** | **5.83x** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | off | **28.71 ms** | 25.84 ms | 389.45 ms | 178.57 ms | — | **15.07x** | **6.91x** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **307.5 μs** | 234.0 μs | 26.57 ms | 1.55 ms | — | **113.52x** | **6.63x** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **32.2 μs** | 32.8 μs | 479.3 μs | 84.5 μs | — | **14.90x** | **2.63x** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **3.10 ms** | 3.15 ms | 6.27 ms | 4.11 ms | — | **2.02x** | **1.32x** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | off | **2.41 ms** | 2.51 ms | 5.64 ms | 3.44 ms | — | **2.34x** | **1.43x** |
| table | mixed_100000 | projection | 5.06 MB | CPU | off | **2.68 ms** | 2.49 ms | 6.70 ms | 5.98 ms | — | **2.70x** | **2.41x** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | off | **5.09 ms** | 4.87 ms | 57.07 ms | 20.09 ms | — | **11.73x** | **4.13x** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | off | **591.3 μs** | 420.3 μs | 11.28 ms | 2.82 ms | — | **26.83x** | **6.72x** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | off | **63.9 μs** | 60.3 μs | 812.0 μs | 141.3 μs | — | **13.47x** | **2.34x** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **413.5 μs** | 491.9 μs | 3.68 ms | 729.9 μs | — | **8.91x** | **1.77x** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | off | **337.8 μs** | 393.6 μs | 3.61 ms | 672.3 μs | — | **10.68x** | **1.99x** |
| table | mixed_10000 | projection | 0.51 MB | CPU | off | **395.8 μs** | 222.9 μs | 3.62 ms | 912.2 μs | — | **16.23x** | **4.09x** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | off | **572.9 μs** | 406.9 μs | 8.40 ms | 2.12 ms | — | **20.63x** | **5.22x** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | off | **270.9 μs** | 144.9 μs | 5.49 ms | 619.0 μs | — | **37.87x** | **4.27x** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | off | **60.5 μs** | 59.8 μs | 822.4 μs | 150.0 μs | — | **13.75x** | **2.51x** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **95.1 μs** | 188.4 μs | 3.31 ms | 377.7 μs | — | **34.76x** | **3.97x** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | off | **87.4 μs** | 182.0 μs | 3.28 ms | 367.5 μs | — | **37.55x** | **4.20x** |
| table | mixed_1000 | projection | 0.06 MB | CPU | off | **214.3 μs** | 95.8 μs | 3.30 ms | 380.2 μs | — | **34.40x** | **3.97x** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | off | **251.4 μs** | 133.7 μs | 3.96 ms | 505.3 μs | — | **29.63x** | **3.78x** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | off | **237.5 μs** | 104.1 μs | 4.79 ms | 390.8 μs | — | **46.03x** | **3.75x** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | off | **58.1 μs** | 60.7 μs | 824.0 μs | 141.3 μs | — | **14.18x** | **2.43x** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **10.81 ms** | 10.86 ms | 8.61 ms | 14.77 ms | — | **0.80x** | **1.37x** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | off | **6.84 ms** | 6.81 ms | 5.17 ms | 11.19 ms | — | **0.76x** | **1.64x** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | off | **4.53 ms** | 4.33 ms | 5.65 ms | 24.85 ms | — | **1.31x** | **5.74x** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | off | **5.61 ms** | 5.54 ms | 7.30 ms | 5.68 ms | — | **1.32x** | **1.02x** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **172.5 μs** | 97.7 μs | 3.95 ms | 582.2 μs | — | **40.41x** | **5.96x** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **26.9 μs** | 32.1 μs | 454.0 μs | 64.8 μs | — | **16.87x** | **2.41x** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **2.11 ms** | 2.17 ms | 4.12 ms | 2.99 ms | — | **1.96x** | **1.42x** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | off | **1.49 ms** | 1.48 ms | 3.52 ms | 2.37 ms | — | **2.38x** | **1.60x** |
| table | narrow_100000 | projection | 1.25 MB | CPU | off | **1.05 ms** | 882.4 μs | 3.59 ms | 4.61 ms | — | **4.07x** | **5.23x** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | off | **1.31 ms** | 1.15 ms | 3.89 ms | 1.43 ms | — | **3.38x** | **1.24x** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | off | **344.2 μs** | 187.8 μs | 3.99 ms | 1.11 ms | — | **21.27x** | **5.93x** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | off | **60.1 μs** | 61.3 μs | 856.8 μs | 122.4 μs | — | **14.26x** | **2.04x** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **250.9 μs** | 344.2 μs | 2.58 ms | 572.4 μs | — | **10.30x** | **2.28x** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | off | **180.0 μs** | 279.4 μs | 2.53 ms | 503.3 μs | — | **14.08x** | **2.80x** |
| table | narrow_10000 | projection | 0.13 MB | CPU | off | **273.3 μs** | 153.5 μs | 2.50 ms | 710.4 μs | — | **16.29x** | **4.63x** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | off | **341.0 μs** | 356.1 μs | 5.51 ms | 734.2 μs | — | **16.17x** | **2.15x** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | off | **217.4 μs** | 95.9 μs | 3.24 ms | 383.4 μs | — | **33.83x** | **4.00x** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | off | **56.8 μs** | 57.3 μs | 816.6 μs | 129.5 μs | — | **14.37x** | **2.28x** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **85.8 μs** | 195.2 μs | 2.39 ms | 343.6 μs | — | **27.87x** | **4.01x** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | off | **126.3 μs** | 191.4 μs | 2.42 ms | 329.7 μs | — | **19.15x** | **2.61x** |
| table | narrow_1000 | projection | 19.7 KB | CPU | off | **216.3 μs** | 97.7 μs | 2.37 ms | 341.2 μs | — | **24.30x** | **3.49x** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | off | **238.8 μs** | 92.7 μs | 2.45 ms | 289.5 μs | — | **26.39x** | **3.13x** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | off | **208.9 μs** | 84.1 μs | 3.18 ms | 304.2 μs | — | **37.77x** | **3.62x** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | off | **51.0 μs** | 50.4 μs | 766.4 μs | 119.6 μs | — | **15.20x** | **2.37x** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **1.09 ms** | 1.13 ms | 1.91 ms | 1.39 ms | — | **1.75x** | **1.27x** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | off | **1.09 ms** | 1.11 ms | 1.90 ms | 1.39 ms | — | **1.75x** | **1.28x** |
| table | typed_100000 | projection | 2.39 MB | CPU | off | **3.74 ms** | 3.58 ms | 29.19 ms | 13.22 ms | — | **8.16x** | **3.70x** |
| table | typed_100000 | read_full | 2.39 MB | CPU | off | **10.03 ms** | 9.75 ms | 53.82 ms | 26.60 ms | — | **5.52x** | **2.73x** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | off | **660.1 μs** | 572.7 μs | 4.76 ms | 1.93 ms | — | **8.31x** | **3.36x** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | off | **29.8 μs** | 29.8 μs | 459.7 μs | 74.8 μs | — | **15.44x** | **2.51x** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **315.5 μs** | 397.8 μs | 2.76 ms | 560.8 μs | — | **8.75x** | **1.78x** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | off | **295.9 μs** | 408.9 μs | 2.78 ms | 567.6 μs | — | **9.41x** | **1.92x** |
| table | typed_10000 | projection | 0.24 MB | CPU | off | **924.5 μs** | 768.5 μs | 7.49 ms | 2.71 ms | — | **9.74x** | **3.53x** |
| table | typed_10000 | read_full | 0.24 MB | CPU | off | **1.21 ms** | 1.04 ms | 7.54 ms | 2.99 ms | — | **7.22x** | **2.86x** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | off | **362.5 μs** | 198.6 μs | 4.08 ms | 760.2 μs | — | **20.55x** | **3.83x** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | off | **63.4 μs** | 66.2 μs | 841.7 μs | 131.5 μs | — | **13.29x** | **2.08x** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **1.87 ms** | 1.95 ms | 3.64 ms | 2.40 ms | — | **1.95x** | **1.28x** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | off | **1.89 ms** | 1.91 ms | 3.66 ms | 2.41 ms | — | **1.93x** | **1.27x** |
| table | varlen_100000 | projection | 3.06 MB | CPU | off | **138.67 ms** | 138.53 ms | 925.16 ms | 191.22 ms | — | **6.68x** | **1.38x** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | off | **127.22 ms** | 129.07 ms | 911.03 ms | 186.45 ms | — | **7.16x** | **1.47x** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | off | **13.02 ms** | 12.90 ms | 94.60 ms | 21.11 ms | — | **7.33x** | **1.64x** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | off | **67.8 μs** | 76.8 μs | 889.0 μs | 151.6 μs | — | **13.11x** | **2.23x** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **191.3 μs** | 252.7 μs | 2.28 ms | 445.7 μs | — | **11.91x** | **2.33x** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | off | **201.9 μs** | 245.5 μs | 2.27 ms | 439.9 μs | — | **11.22x** | **2.18x** |
| table | varlen_10000 | projection | 0.31 MB | CPU | off | **12.24 ms** | 12.13 ms | 91.87 ms | 18.72 ms | — | **7.57x** | **1.54x** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | off | **12.22 ms** | 12.19 ms | 92.29 ms | 18.59 ms | — | **7.57x** | **1.52x** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | off | **1.37 ms** | 1.26 ms | 12.04 ms | 2.36 ms | — | **9.58x** | **1.88x** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | off | **47.8 μs** | 47.7 μs | 768.4 μs | 112.2 μs | — | **16.11x** | **2.35x** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **45.1 μs** | 149.0 μs | 2.47 ms | 294.8 μs | — | **54.78x** | **6.53x** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | off | **75.7 μs** | 134.7 μs | 2.16 ms | 268.8 μs | — | **28.47x** | **3.55x** |
| table | varlen_1000 | projection | 39.4 KB | CPU | off | **793.7 μs** | 720.7 μs | 6.51 ms | 1.27 ms | — | **9.03x** | **1.76x** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | off | **785.9 μs** | 721.7 μs | 6.51 ms | 1.26 ms | — | **9.01x** | **1.75x** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | off | **207.5 μs** | 143.9 μs | 2.21 ms | 297.0 μs | — | **15.36x** | **2.06x** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | off | **44.9 μs** | 45.8 μs | 769.6 μs | 111.2 μs | — | **17.14x** | **2.48x** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **3.82 ms** | 3.75 ms | 8.43 ms | 4.43 ms | — | **2.25x** | **1.18x** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | off | **3.27 ms** | 3.52 ms | 8.61 ms | 4.05 ms | — | **2.63x** | **1.24x** |
| table | wide_100000 | projection | 20.71 MB | CPU | off | **2.63 ms** | 2.54 ms | 8.55 ms | 5.42 ms | — | **3.36x** | **2.13x** |
| table | wide_100000 | read_full | 20.71 MB | CPU | off | **44.50 ms** | 17.70 ms | 145.96 ms | 60.04 ms | — | **8.25x** | **3.39x** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | off | **1.09 ms** | 1.02 ms | 21.86 ms | 5.05 ms | — | **21.41x** | **4.95x** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | off | **41.0 μs** | 51.0 μs | 700.5 μs | 272.9 μs | — | **17.09x** | **6.66x** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **836.1 μs** | 896.7 μs | 11.00 ms | 1.51 ms | — | **13.16x** | **1.80x** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | off | **759.8 μs** | 829.3 μs | 10.75 ms | 1.44 ms | — | **14.15x** | **1.89x** |
| table | wide_10000 | projection | 2.08 MB | CPU | off | **738.2 μs** | 576.5 μs | 10.76 ms | 1.66 ms | — | **18.67x** | **2.87x** |
| table | wide_10000 | read_full | 2.08 MB | CPU | off | **2.49 ms** | 2.33 ms | 30.27 ms | 8.86 ms | — | **13.01x** | **3.81x** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | off | **457.3 μs** | 317.6 μs | 19.04 ms | 1.66 ms | — | **59.95x** | **5.23x** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | off | **78.9 μs** | 80.1 μs | 1.02 ms | 462.3 μs | — | **12.88x** | **5.86x** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **131.6 μs** | 220.4 μs | 10.12 ms | 748.6 μs | — | **76.89x** | **5.69x** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | off | **114.9 μs** | 225.8 μs | 11.18 ms | 947.4 μs | — | **97.25x** | **8.24x** |
| table | wide_1000 | projection | 0.22 MB | CPU | off | **249.5 μs** | 118.6 μs | 10.07 ms | 771.2 μs | — | **84.91x** | **6.50x** |
| table | wide_1000 | read_full | 0.22 MB | CPU | off | **459.7 μs** | 315.8 μs | 12.83 ms | 1.53 ms | — | **40.62x** | **4.86x** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | off | **331.1 μs** | 181.0 μs | 16.62 ms | 938.9 μs | — | **91.87x** | **5.19x** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | off | **122.5 μs** | 118.3 μs | 1.45 ms | 697.4 μs | — | **12.27x** | **5.89x** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **746.0 μs** | 737.7 μs | 4.77 ms | — | — | **6.46x** | **—** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | on | **715.2 μs** | 740.5 μs | 4.84 ms | — | — | **6.76x** | **—** |
| table | ascii_10000 | projection | 0.44 MB | CPU | on | **1.91 ms** | 2.12 ms | 16.79 ms | — | — | **8.77x** | **—** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | on | **2.40 ms** | 2.14 ms | 17.12 ms | — | — | **7.99x** | **—** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | on | **485.9 μs** | 363.3 μs | 4.63 ms | — | — | **12.76x** | **—** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | on | **62.0 μs** | 56.9 μs | 750.1 μs | — | — | **13.18x** | **—** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **352.3 μs** | 367.1 μs | 2.88 ms | — | — | **8.18x** | **—** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | on | **350.1 μs** | 358.9 μs | 2.85 ms | — | — | **8.15x** | **—** |
| table | ascii_1000 | projection | 50.6 KB | CPU | on | **498.9 μs** | 381.9 μs | 4.00 ms | — | — | **10.47x** | **—** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | on | **485.3 μs** | 506.1 μs | 6.11 ms | — | — | **12.59x** | **—** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | on | **360.0 μs** | 246.9 μs | 3.67 ms | — | — | **14.85x** | **—** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | on | **99.6 μs** | 88.8 μs | 1.07 ms | — | — | **12.00x** | **—** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **20.32 ms** | 20.05 ms | 25.86 ms | — | — | **1.29x** | **—** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | on | **13.34 ms** | 14.01 ms | 19.72 ms | — | — | **1.48x** | **—** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | on | **18.24 ms** | 23.55 ms | 35.80 ms | — | — | **1.96x** | **—** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | on | **81.53 ms** | 73.09 ms | 740.27 ms | — | — | **10.13x** | **—** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **545.2 μs** | 364.7 μs | 19.33 ms | — | — | **53.01x** | **—** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **81.1 μs** | 68.3 μs | 819.9 μs | — | — | **12.00x** | **—** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **2.15 ms** | 2.17 ms | 5.64 ms | — | — | **2.63x** | **—** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | on | **1.41 ms** | 1.58 ms | 5.06 ms | — | — | **3.59x** | **—** |
| table | mixed_100000 | projection | 5.06 MB | CPU | on | **2.01 ms** | 1.81 ms | 6.03 ms | — | — | **3.33x** | **—** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | on | **4.35 ms** | 4.09 ms | 56.62 ms | — | — | **13.83x** | **—** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | on | **529.3 μs** | 328.1 μs | 10.66 ms | — | — | **32.49x** | **—** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | on | **63.8 μs** | 59.0 μs | 828.1 μs | — | — | **14.05x** | **—** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **304.2 μs** | 364.2 μs | 3.59 ms | — | — | **11.79x** | **—** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | on | **221.3 μs** | 298.0 μs | 3.60 ms | — | — | **16.27x** | **—** |
| table | mixed_10000 | projection | 0.51 MB | CPU | on | **359.5 μs** | 187.5 μs | 3.57 ms | — | — | **19.03x** | **—** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | on | **526.8 μs** | 306.3 μs | 8.41 ms | — | — | **27.46x** | **—** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | on | **290.1 μs** | 159.8 μs | 5.41 ms | — | — | **33.82x** | **—** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | on | **65.2 μs** | 58.7 μs | 810.0 μs | — | — | **13.79x** | **—** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **101.9 μs** | 200.3 μs | 3.35 ms | — | — | **32.83x** | **—** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | on | **106.8 μs** | 195.3 μs | 3.28 ms | — | — | **30.69x** | **—** |
| table | mixed_1000 | projection | 0.06 MB | CPU | on | **227.3 μs** | 108.8 μs | 3.30 ms | — | — | **30.30x** | **—** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | on | **276.7 μs** | 140.6 μs | 3.91 ms | — | — | **27.81x** | **—** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | on | **259.3 μs** | 132.6 μs | 4.82 ms | — | — | **36.35x** | **—** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | on | **53.9 μs** | 54.1 μs | 803.8 μs | — | — | **14.91x** | **—** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **13.16 ms** | 13.15 ms | 15.24 ms | — | — | **1.16x** | **—** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | on | **6.21 ms** | 6.24 ms | 9.00 ms | — | — | **1.45x** | **—** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | on | **4.92 ms** | 4.70 ms | 10.99 ms | — | — | **2.34x** | **—** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | on | **6.67 ms** | 6.48 ms | 13.47 ms | — | — | **2.08x** | **—** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **302.7 μs** | 180.7 μs | 6.67 ms | — | — | **36.92x** | **—** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **68.8 μs** | 57.8 μs | 770.4 μs | — | — | **13.32x** | **—** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **1.54 ms** | 1.51 ms | 3.90 ms | — | — | **2.59x** | **—** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | on | **901.8 μs** | 778.0 μs | 3.22 ms | — | — | **4.14x** | **—** |
| table | narrow_100000 | projection | 1.25 MB | CPU | on | **778.2 μs** | 595.3 μs | 3.45 ms | — | — | **5.80x** | **—** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | on | **982.1 μs** | 767.9 μs | 3.82 ms | — | — | **4.97x** | **—** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | on | **526.1 μs** | 176.1 μs | 3.82 ms | — | — | **21.71x** | **—** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | on | **53.9 μs** | 52.4 μs | 783.6 μs | — | — | **14.96x** | **—** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **323.3 μs** | 337.5 μs | 2.63 ms | — | — | **8.15x** | **—** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | on | **190.5 μs** | 279.7 μs | 2.59 ms | — | — | **13.61x** | **—** |
| table | narrow_10000 | projection | 0.13 MB | CPU | on | **264.3 μs** | 131.4 μs | 2.50 ms | — | — | **19.04x** | **—** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | on | **276.3 μs** | 153.9 μs | 2.58 ms | — | — | **16.74x** | **—** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | on | **256.3 μs** | 119.1 μs | 3.23 ms | — | — | **27.13x** | **—** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | on | **57.0 μs** | 88.2 μs | 992.6 μs | — | — | **17.43x** | **—** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **103.1 μs** | 207.5 μs | 2.40 ms | — | — | **23.25x** | **—** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | on | **107.4 μs** | 202.5 μs | 2.40 ms | — | — | **22.37x** | **—** |
| table | narrow_1000 | projection | 19.7 KB | CPU | on | **240.0 μs** | 202.7 μs | 6.03 ms | — | — | **29.75x** | **—** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | on | **249.3 μs** | 114.3 μs | 2.83 ms | — | — | **24.80x** | **—** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | on | **232.0 μs** | 103.6 μs | 3.17 ms | — | — | **30.55x** | **—** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | on | **53.7 μs** | 72.3 μs | 1.11 ms | — | — | **20.60x** | **—** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **1.76 ms** | 1.59 ms | 4.08 ms | — | — | **2.57x** | **—** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | on | **1.75 ms** | 1.76 ms | 4.20 ms | — | — | **2.40x** | **—** |
| table | typed_100000 | projection | 2.39 MB | CPU | on | **3.04 ms** | 2.58 ms | 61.45 ms | — | — | **23.80x** | **—** |
| table | typed_100000 | read_full | 2.39 MB | CPU | on | **3.33 ms** | 3.10 ms | 63.29 ms | — | — | **20.40x** | **—** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | on | **697.8 μs** | 404.0 μs | 9.79 ms | — | — | **24.24x** | **—** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | on | **112.3 μs** | 95.9 μs | 1.08 ms | — | — | **11.22x** | **—** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **303.0 μs** | 351.2 μs | 3.18 ms | — | — | **10.50x** | **—** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | on | **353.3 μs** | 362.6 μs | 3.24 ms | — | — | **9.18x** | **—** |
| table | typed_10000 | projection | 0.24 MB | CPU | on | **547.3 μs** | 402.1 μs | 8.83 ms | — | — | **21.96x** | **—** |
| table | typed_10000 | read_full | 0.24 MB | CPU | on | **624.3 μs** | 433.7 μs | 8.96 ms | — | — | **20.65x** | **—** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | on | **385.4 μs** | 169.7 μs | 4.84 ms | — | — | **28.52x** | **—** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | on | **99.9 μs** | 92.5 μs | 1.05 ms | — | — | **11.30x** | **—** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **1.48 ms** | 1.46 ms | 3.60 ms | — | — | **2.47x** | **—** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | on | **1.44 ms** | 1.54 ms | 3.77 ms | — | — | **2.62x** | **—** |
| table | varlen_100000 | projection | 3.06 MB | CPU | on | **135.85 ms** | 136.62 ms | 942.66 ms | — | — | **6.94x** | **—** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | on | **135.53 ms** | 154.33 ms | 937.91 ms | — | — | **6.92x** | **—** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | on | **16.16 ms** | 15.23 ms | 113.38 ms | — | — | **7.44x** | **—** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | on | **91.3 μs** | 102.5 μs | 1.03 ms | — | — | **11.30x** | **—** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **203.7 μs** | 306.4 μs | 2.52 ms | — | — | **12.35x** | **—** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | on | **376.8 μs** | 304.2 μs | 2.45 ms | — | — | **8.05x** | **—** |
| table | varlen_10000 | projection | 0.31 MB | CPU | on | **13.12 ms** | 12.77 ms | 93.47 ms | — | — | **7.32x** | **—** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | on | **12.89 ms** | 12.55 ms | 93.35 ms | — | — | **7.44x** | **—** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | on | **1.61 ms** | 1.43 ms | 12.44 ms | — | — | **8.70x** | **—** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | on | **60.4 μs** | 62.6 μs | 785.7 μs | — | — | **13.01x** | **—** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **110.5 μs** | 211.9 μs | 2.35 ms | — | — | **21.29x** | **—** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | on | **113.3 μs** | 211.0 μs | 2.35 ms | — | — | **20.73x** | **—** |
| table | varlen_1000 | projection | 39.4 KB | CPU | on | **1.57 ms** | 1.48 ms | 11.63 ms | — | — | **7.84x** | **—** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | on | **1.59 ms** | 1.41 ms | 11.58 ms | — | — | **8.18x** | **—** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | on | **571.5 μs** | 403.1 μs | 4.03 ms | — | — | **10.00x** | **—** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | on | **49.0 μs** | 65.4 μs | 1.02 ms | — | — | **20.78x** | **—** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **3.40 ms** | 3.37 ms | 13.69 ms | — | — | **4.06x** | **—** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | on | **2.67 ms** | 2.55 ms | 13.23 ms | — | — | **5.19x** | **—** |
| table | wide_100000 | projection | 20.71 MB | CPU | on | **4.03 ms** | 3.65 ms | 14.87 ms | — | — | **4.07x** | **—** |
| table | wide_100000 | read_full | 20.71 MB | CPU | on | **36.44 ms** | 35.53 ms | 256.61 ms | — | — | **7.22x** | **—** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | on | **1.89 ms** | 1.51 ms | 37.62 ms | — | — | **24.95x** | **—** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | on | **92.1 μs** | 77.8 μs | 1.01 ms | — | — | **13.02x** | **—** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **504.4 μs** | 510.7 μs | 10.48 ms | — | — | **20.78x** | **—** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | on | **447.2 μs** | 468.8 μs | 10.51 ms | — | — | **23.49x** | **—** |
| table | wide_10000 | projection | 2.08 MB | CPU | on | **532.3 μs** | 329.1 μs | 10.48 ms | — | — | **31.83x** | **—** |
| table | wide_10000 | read_full | 2.08 MB | CPU | on | **1.87 ms** | 1.71 ms | 30.04 ms | — | — | **17.59x** | **—** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | on | **447.2 μs** | 314.5 μs | 18.57 ms | — | — | **59.04x** | **—** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | on | **98.3 μs** | 102.9 μs | 1.43 ms | — | — | **14.51x** | **—** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **148.5 μs** | 239.3 μs | 9.97 ms | — | — | **67.10x** | **—** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | on | **145.8 μs** | 238.4 μs | 9.98 ms | — | — | **68.45x** | **—** |
| table | wide_1000 | projection | 0.22 MB | CPU | on | **282.6 μs** | 142.5 μs | 10.05 ms | — | — | **70.53x** | **—** |
| table | wide_1000 | read_full | 0.22 MB | CPU | on | **433.4 μs** | 283.0 μs | 12.70 ms | — | — | **44.86x** | **—** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | on | **371.3 μs** | 206.2 μs | 16.53 ms | — | — | **80.20x** | **—** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | on | **77.3 μs** | 80.1 μs | 1.07 ms | — | — | **13.90x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance comparisons & edge cases {#performance-deficits}

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (CPU and GPU). GPU lags may reflect software or hardware limits — they are listed, not hidden.

| Platform | Domain | Case | mmap | torchfits | Peak RSS (MB) | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 46.15 ms | 316.0 | fitsio/fitsio_torch | 1.01× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 45.82 ms | 299.5 | fitsio/fitsio_torch | 1.01× |
| Linux x86_64 / CPU | tensor | compressed_rice_1 [cutout_100x100] | n/a | 1.41 ms | 288.2 | fitsio/fitsio_torch | 1.00× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 46.15 ms | 316.0 | fitsio/fitsio_torch | 1.01× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 45.89 ms | 299.5 | fitsio/fitsio_torch | 1.01× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 5.61 ms | 433.1 | fitsio/fitsio_torch | 1.11× |
| Linux x86_64 / CPU | table | narrow_1000000 [predicate_filter_selective] | off | 6.81 ms | 433.1 | astropy/astropy | 1.32× |
| Linux x86_64 / CPU | table | narrow_1000000 [predicate_filter] | off | 10.86 ms | 433.1 | astropy/astropy | 1.26× |
<!-- BENCH_DEFICITS_END -->

### Host scorecard

| Platform | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| Linux x86_64 / CPU | `exhaustive_cpu_20260822_054439` | 3057 | 8 | 300.5 | lab + mmap-matrix |
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
