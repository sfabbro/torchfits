# Benchmarks

> **Headline (v1.1.0, CANFAR headless, lab profile, mmap on+off matrix):**
> torchfits wins **100% of significant image comparisons** on both CPU and
> CUDA hosts — compressed and uncompressed, every dtype, every transport —
> and 98.6–100% of table comparisons. The single remaining case family
> where a peer is ahead is **narrow-table full reads with `mmap=False`**
> (fitsio, by 6–17%): fitsio reads one column at a time while our buffered
> path stages whole rows; the structural fix (single-pass decode into
> caller-visible memory) lands in 1.2. All other lag rows are sub-1.13x
> noise on shared-CFITSIO decompression paths or sub-0.15 ms GPU launch
> overheads. Full per-cell data below; raw CSVs under `benchmarks_results/`.


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
Source: `benchmarks_results/exhaustive_cpu_20260822_213823/results.csv` (mmap on+off matrix.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### Tensor I/O (IMAGE HDU) (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.10 ms` (n=174) | `0.65 ms` (n=253) | `0.18 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.11 ms` (n=174) | `0.49 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### Table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.19 ms` (n=216) | `3.21 ms` (n=184) | `0.68 ms` (n=216) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.18 ms` (n=216) | `2.65 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Table read (100k rows, 8 cols, mixed) | CPU | **3.18 ms** | 2.93 ms | 52.22 ms | 16.87 ms | **17.84x** | **5.76x** |
| Varlen table read (100k rows, 3 cols) | CPU | **71.45 ms** | 71.93 ms | 523.01 ms | 110.15 ms | **7.32x** | **1.54x** |
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
| tensor | compressed_gzip_1:header_read | header_read | 1.29 MB | CPU | n/a | **—** | 49.8 μs | 2.38 ms | 234.5 μs | — | **47.69x** | **4.70x** |
| tensor | compressed_gzip_2:header_read | header_read | 0.89 MB | CPU | n/a | **—** | 49.2 μs | 2.33 ms | 234.4 μs | — | **47.40x** | **4.77x** |
| tensor | compressed_hcompress_1:header_read | header_read | 0.82 MB | CPU | n/a | **—** | 52.8 μs | 2.49 ms | 266.4 μs | — | **47.28x** | **5.05x** |
| tensor | compressed_rice_1:cutout_100x100 | cutout_100x100 | 0.90 MB | CPU | n/a | **1.34 ms** | 1.29 ms | 11.52 ms | 1.37 ms | — | **8.94x** | **1.06x** |
| tensor | compressed_rice_1:header_read | header_read | 0.90 MB | CPU | n/a | **—** | 51.4 μs | 2.46 ms | 263.5 μs | — | **47.75x** | **5.12x** |
| tensor | large_float32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 27.1 μs | 455.3 μs | 46.9 μs | — | **16.82x** | **1.73x** |
| tensor | large_float32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 32.5 μs | 625.8 μs | 61.1 μs | — | **19.26x** | **1.88x** |
| tensor | large_float64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 18.5 μs | 318.8 μs | 34.6 μs | — | **17.21x** | **1.87x** |
| tensor | large_float64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 15.1 μs | 291.4 μs | 29.2 μs | — | **19.28x** | **1.93x** |
| tensor | large_int16_1d:header_read | header_read | 1.91 MB | CPU | n/a | **—** | 14.4 μs | 263.1 μs | 26.7 μs | — | **18.24x** | **1.85x** |
| tensor | large_int16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 16.6 μs | 286.3 μs | 29.2 μs | — | **17.24x** | **1.76x** |
| tensor | large_int32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 15.1 μs | 261.2 μs | 26.1 μs | — | **17.32x** | **1.73x** |
| tensor | large_int32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 14.9 μs | 293.1 μs | 28.5 μs | — | **19.73x** | **1.92x** |
| tensor | large_int64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 16.0 μs | 264.1 μs | 27.0 μs | — | **16.55x** | **1.69x** |
| tensor | large_int64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 15.7 μs | 292.7 μs | 29.4 μs | — | **18.68x** | **1.88x** |
| tensor | large_int8_1d:header_read | header_read | 0.96 MB | CPU | n/a | **—** | 16.2 μs | 305.0 μs | 31.6 μs | — | **18.81x** | **1.95x** |
| tensor | large_int8_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 17.7 μs | 326.8 μs | 33.2 μs | — | **18.46x** | **1.88x** |
| tensor | large_uint16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 15.7 μs | 331.6 μs | 31.5 μs | — | **21.13x** | **2.01x** |
| tensor | large_uint32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 16.5 μs | 328.0 μs | 35.5 μs | — | **19.87x** | **2.15x** |
| tensor | medium_float32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 14.9 μs | 275.2 μs | 27.7 μs | — | **18.51x** | **1.86x** |
| tensor | medium_float32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 16.0 μs | 292.2 μs | 31.0 μs | — | **18.30x** | **1.94x** |
| tensor | medium_float32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 15.6 μs | 315.2 μs | 32.2 μs | — | **20.22x** | **2.07x** |
| tensor | medium_float64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 15.2 μs | 266.0 μs | 25.7 μs | — | **17.53x** | **1.69x** |
| tensor | medium_float64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 16.7 μs | 291.5 μs | 28.2 μs | — | **17.50x** | **1.69x** |
| tensor | medium_float64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 15.8 μs | 309.7 μs | 30.4 μs | — | **19.57x** | **1.92x** |
| tensor | medium_int16_1d:header_read | header_read | 0.20 MB | CPU | n/a | **—** | 17.1 μs | 275.0 μs | 28.5 μs | — | **16.12x** | **1.67x** |
| tensor | medium_int16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 15.6 μs | 291.0 μs | 28.1 μs | — | **18.68x** | **1.80x** |
| tensor | medium_int16_3d:header_read | header_read | 3.13 MB | CPU | n/a | **—** | 15.7 μs | 316.6 μs | 31.3 μs | — | **20.18x** | **2.00x** |
| tensor | medium_int32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 15.8 μs | 268.0 μs | 27.6 μs | — | **16.95x** | **1.75x** |
| tensor | medium_int32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 17.1 μs | 294.2 μs | 27.3 μs | — | **17.22x** | **1.60x** |
| tensor | medium_int32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 16.4 μs | 310.9 μs | 32.6 μs | — | **18.93x** | **1.99x** |
| tensor | medium_int64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 14.6 μs | 269.1 μs | 26.1 μs | — | **18.39x** | **1.78x** |
| tensor | medium_int64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 16.1 μs | 292.0 μs | 29.5 μs | — | **18.10x** | **1.83x** |
| tensor | medium_int64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 15.7 μs | 310.5 μs | 31.4 μs | — | **19.72x** | **1.99x** |
| tensor | medium_int8_1d:header_read | header_read | 0.10 MB | CPU | n/a | **—** | 16.7 μs | 317.1 μs | 33.4 μs | — | **18.95x** | **2.00x** |
| tensor | medium_int8_2d:header_read | header_read | 1.01 MB | CPU | n/a | **—** | 16.4 μs | 327.7 μs | 33.5 μs | — | **19.98x** | **2.05x** |
| tensor | medium_int8_3d:header_read | header_read | 1.57 MB | CPU | n/a | **—** | 17.8 μs | 350.4 μs | 35.3 μs | — | **19.67x** | **1.98x** |
| tensor | medium_uint16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 16.4 μs | 335.0 μs | 37.0 μs | — | **20.42x** | **2.25x** |
| tensor | medium_uint32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 18.3 μs | 347.8 μs | 38.1 μs | — | **19.00x** | **2.08x** |
| tensor | mef_medium:header_read | header_read | 7.02 MB | CPU | n/a | **—** | 19.5 μs | 558.1 μs | 45.0 μs | — | **28.69x** | **2.31x** |
| tensor | mef_small:header_read | header_read | 0.45 MB | CPU | n/a | **—** | 21.2 μs | 538.8 μs | 45.0 μs | — | **25.42x** | **2.12x** |
| tensor | multi_mef_10ext:cutout_100x100 | cutout_100x100 | 2.68 MB | CPU | n/a | **100.2 μs** | 129.0 μs | 3.76 ms | 273.2 μs | — | **37.55x** | **2.73x** |
| tensor | multi_mef_10ext:header_read | header_read | 2.68 MB | CPU | n/a | **—** | 18.7 μs | 522.0 μs | 40.3 μs | — | **27.88x** | **2.15x** |
| tensor | multi_mef_10ext:random_ext_full_reads_200 | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **8.46 ms** | 8.52 ms | 11.10 ms | 11.94 ms | — | **1.31x** | **1.41x** |
| tensor | repeated_cutouts_50x_100x100:repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **708.2 μs** | 677.8 μs | 87.16 ms | 5.21 ms | — | **128.58x** | **7.69x** |
| tensor | scaled_large:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 15.9 μs | 335.0 μs | 35.3 μs | — | **21.05x** | **2.22x** |
| tensor | scaled_medium:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 17.0 μs | 331.7 μs | 34.9 μs | — | **19.48x** | **2.05x** |
| tensor | scaled_small:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 17.0 μs | 349.9 μs | 36.5 μs | — | **20.53x** | **2.14x** |
| tensor | small_float32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 15.8 μs | 274.4 μs | 26.5 μs | — | **17.37x** | **1.68x** |
| tensor | small_float32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 14.7 μs | 285.1 μs | 28.9 μs | — | **19.38x** | **1.96x** |
| tensor | small_float32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 14.8 μs | 309.1 μs | 29.9 μs | — | **20.88x** | **2.02x** |
| tensor | small_float64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 16.0 μs | 261.1 μs | 28.3 μs | — | **16.33x** | **1.77x** |
| tensor | small_float64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 16.7 μs | 283.7 μs | 29.8 μs | — | **16.96x** | **1.78x** |
| tensor | small_float64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 14.9 μs | 298.3 μs | 29.1 μs | — | **20.06x** | **1.96x** |
| tensor | small_int16_1d:header_read | header_read | 22.5 KB | CPU | n/a | **—** | 14.7 μs | 272.1 μs | 29.5 μs | — | **18.56x** | **2.01x** |
| tensor | small_int16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 14.4 μs | 275.7 μs | 30.9 μs | — | **19.10x** | **2.14x** |
| tensor | small_int16_3d:header_read | header_read | 0.32 MB | CPU | n/a | **—** | 15.6 μs | 304.6 μs | 29.8 μs | — | **19.49x** | **1.90x** |
| tensor | small_int32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 14.7 μs | 261.4 μs | 26.7 μs | — | **17.83x** | **1.82x** |
| tensor | small_int32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 15.6 μs | 287.4 μs | 28.4 μs | — | **18.47x** | **1.82x** |
| tensor | small_int32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 15.4 μs | 311.7 μs | 28.8 μs | — | **20.28x** | **1.87x** |
| tensor | small_int64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 15.8 μs | 264.9 μs | 26.3 μs | — | **16.82x** | **1.67x** |
| tensor | small_int64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 15.8 μs | 289.0 μs | 29.3 μs | — | **18.28x** | **1.85x** |
| tensor | small_int64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 16.1 μs | 307.3 μs | 30.4 μs | — | **19.12x** | **1.89x** |
| tensor | small_int8_1d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 16.3 μs | 303.5 μs | 31.0 μs | — | **18.61x** | **1.90x** |
| tensor | small_int8_2d:header_read | header_read | 0.07 MB | CPU | n/a | **—** | 18.7 μs | 333.1 μs | 35.3 μs | — | **17.86x** | **1.89x** |
| tensor | small_int8_3d:header_read | header_read | 0.16 MB | CPU | n/a | **—** | 15.7 μs | 356.3 μs | 34.4 μs | — | **22.73x** | **2.19x** |
| tensor | small_uint16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 17.0 μs | 324.7 μs | 34.8 μs | — | **19.13x** | **2.05x** |
| tensor | small_uint32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.8 μs | 327.9 μs | 32.4 μs | — | **19.56x** | **1.93x** |
| tensor | timeseries_frame_000:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.0 μs | 292.2 μs | 28.5 μs | — | **18.26x** | **1.78x** |
| tensor | timeseries_frame_001:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 15.1 μs | 293.3 μs | 28.0 μs | — | **19.47x** | **1.86x** |
| tensor | timeseries_frame_002:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 15.7 μs | 291.6 μs | 29.2 μs | — | **18.53x** | **1.86x** |
| tensor | timeseries_frame_003:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.3 μs | 297.6 μs | 30.9 μs | — | **18.24x** | **1.89x** |
| tensor | timeseries_frame_004:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.1 μs | 292.7 μs | 27.5 μs | — | **18.16x** | **1.70x** |
| tensor | tiny_float32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 16.1 μs | 273.2 μs | 27.3 μs | — | **16.96x** | **1.70x** |
| tensor | tiny_float32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 14.0 μs | 291.6 μs | 27.3 μs | — | **20.81x** | **1.95x** |
| tensor | tiny_float32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 14.9 μs | 317.7 μs | 33.4 μs | — | **21.28x** | **2.24x** |
| tensor | tiny_float64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 14.7 μs | 261.4 μs | 29.5 μs | — | **17.73x** | **2.00x** |
| tensor | tiny_float64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 15.5 μs | 284.4 μs | 27.3 μs | — | **18.37x** | **1.76x** |
| tensor | tiny_float64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 15.9 μs | 312.4 μs | 31.6 μs | — | **19.63x** | **1.99x** |
| tensor | tiny_int16_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 15.2 μs | 262.3 μs | 27.4 μs | — | **17.22x** | **1.80x** |
| tensor | tiny_int16_2d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 15.7 μs | 275.9 μs | 28.4 μs | — | **17.53x** | **1.81x** |
| tensor | tiny_int16_3d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 15.8 μs | 303.4 μs | 31.6 μs | — | **19.21x** | **2.00x** |
| tensor | tiny_int32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 14.4 μs | 263.5 μs | 26.7 μs | — | **18.33x** | **1.86x** |
| tensor | tiny_int32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 15.8 μs | 282.3 μs | 28.8 μs | — | **17.89x** | **1.82x** |
| tensor | tiny_int32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 15.0 μs | 302.9 μs | 28.7 μs | — | **20.13x** | **1.91x** |
| tensor | tiny_int64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 16.1 μs | 266.1 μs | 26.5 μs | — | **16.56x** | **1.65x** |
| tensor | tiny_int64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 14.9 μs | 279.8 μs | 27.2 μs | — | **18.82x** | **1.83x** |
| tensor | tiny_int64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 16.2 μs | 299.7 μs | 30.8 μs | — | **18.53x** | **1.90x** |
| tensor | tiny_int8_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 15.2 μs | 311.7 μs | 31.7 μs | — | **20.45x** | **2.08x** |
| tensor | tiny_int8_2d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 16.0 μs | 331.7 μs | 34.0 μs | — | **20.68x** | **2.12x** |
| tensor | tiny_int8_3d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 17.1 μs | 347.2 μs | 36.5 μs | — | **20.30x** | **2.13x** |
| tensor | write_compress_hcompress_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **48.10 ms** | — | 58.23 ms | — | — | **1.21x** | **—** |
| tensor | write_compress_rice_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **37.85 ms** | — | 68.65 ms | — | — | **1.81x** | **—** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | off | **13.71 ms** | 13.74 ms | 26.40 ms | 15.24 ms | — | **1.92x** | **1.11x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | off | **11.90 ms** | 11.77 ms | 41.89 ms | 13.36 ms | — | **3.56x** | **1.14x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | off | **26.32 ms** | 26.35 ms | 29.67 ms | 25.67 ms | — | **1.13x** | **0.98x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | off | **7.31 ms** | 7.24 ms | 19.05 ms | 7.31 ms | — | **2.63x** | **1.01x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | off | **506.5 μs** | 508.0 μs | 1.20 ms | 815.8 μs | — | **2.37x** | **1.61x** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | off | **1.87 ms** | 3.10 ms | 10.23 ms | 3.11 ms | — | **5.47x** | **1.66x** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | off | **1.35 ms** | 921.3 μs | 1.93 ms | 1.25 ms | — | **2.10x** | **1.36x** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | off | **5.64 ms** | 5.10 ms | 10.48 ms | 5.35 ms | — | **2.05x** | **1.05x** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | off | **358.5 μs** | 288.8 μs | 747.3 μs | 364.1 μs | — | **2.59x** | **1.26x** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | off | **1.19 ms** | 990.7 μs | 3.11 ms | 1.25 ms | — | **3.14x** | **1.27x** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | off | **509.4 μs** | 499.5 μs | 1.15 ms | 759.0 μs | — | **2.30x** | **1.52x** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | off | **2.55 ms** | 2.75 ms | 9.45 ms | 3.10 ms | — | **3.71x** | **1.22x** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | off | **915.9 μs** | 905.5 μs | 1.90 ms | 1.22 ms | — | **2.10x** | **1.35x** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | off | **5.62 ms** | 5.01 ms | 10.47 ms | 5.31 ms | — | **2.09x** | **1.06x** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | off | **217.9 μs** | 207.6 μs | 643.1 μs | 192.3 μs | — | **3.10x** | **0.93x** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | off | **675.3 μs** | 656.3 μs | 1.74 ms | 657.6 μs | — | **2.65x** | **1.00x** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | off | **1.24 ms** | 1.27 ms | 3.87 ms | 1.53 ms | — | **3.11x** | **1.23x** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | off | **3.23 ms** | 3.13 ms | 6.61 ms | 3.66 ms | — | **2.11x** | **1.17x** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | off | **80.3 μs** | 97.4 μs | 368.4 μs | 110.5 μs | — | **4.59x** | **1.38x** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | off | **520.0 μs** | 517.6 μs | 1.22 ms | 791.1 μs | — | **2.35x** | **1.53x** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | off | **763.8 μs** | 909.4 μs | 1.66 ms | 1.16 ms | — | **2.17x** | **1.52x** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | off | **92.1 μs** | 125.8 μs | 468.4 μs | 158.4 μs | — | **5.09x** | **1.72x** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | off | **1.00 ms** | 969.1 μs | 2.54 ms | 1.32 ms | — | **2.62x** | **1.37x** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | off | **1.87 ms** | 1.80 ms | 4.84 ms | 2.38 ms | — | **2.68x** | **1.32x** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | off | **45.9 μs** | 60.9 μs | 305.4 μs | 66.7 μs | — | **6.65x** | **1.45x** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | off | **281.5 μs** | 318.9 μs | 776.3 μs | 360.1 μs | — | **2.76x** | **1.28x** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | off | **436.5 μs** | 414.0 μs | 1.01 ms | 541.4 μs | — | **2.45x** | **1.31x** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | off | **75.4 μs** | 84.5 μs | 366.2 μs | 105.6 μs | — | **4.86x** | **1.40x** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | off | **489.0 μs** | 485.7 μs | 1.20 ms | 773.9 μs | — | **2.47x** | **1.59x** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | off | **789.5 μs** | 798.6 μs | 1.68 ms | 1.19 ms | — | **2.12x** | **1.50x** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | off | **137.0 μs** | 128.5 μs | 454.0 μs | 158.2 μs | — | **3.53x** | **1.23x** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | off | **979.4 μs** | 925.6 μs | 2.73 ms | 1.37 ms | — | **2.95x** | **1.48x** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | off | **1.87 ms** | 1.83 ms | 4.62 ms | 2.30 ms | — | **2.53x** | **1.26x** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | off | **34.0 μs** | 64.4 μs | 387.2 μs | 58.7 μs | — | **11.40x** | **1.73x** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | off | **182.3 μs** | 209.2 μs | 750.2 μs | 194.4 μs | — | **4.11x** | **1.07x** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | off | **308.2 μs** | 324.7 μs | 877.3 μs | 297.4 μs | — | **2.85x** | **0.97x** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | off | **380.8 μs** | 334.8 μs | 1.30 ms | 428.8 μs | — | **3.88x** | **1.28x** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | off | **677.7 μs** | 650.9 μs | 1.72 ms | 926.7 μs | — | **2.64x** | **1.42x** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | off | **225.4 μs** | 221.2 μs | 963.2 μs | 234.4 μs | — | **4.35x** | **1.06x** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | off | **41.5 μs** | 58.6 μs | 596.5 μs | 91.5 μs | — | **14.37x** | **2.21x** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | off | **47.5 μs** | 48.7 μs | 614.0 μs | 148.3 μs | — | **12.93x** | **3.12x** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | off | **3.80 ms** | 3.52 ms | 5.48 ms | 3.40 ms | — | **1.56x** | **0.96x** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | off | **679.3 μs** | 699.8 μs | 1.42 ms | 809.9 μs | — | **2.09x** | **1.19x** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | off | **86.5 μs** | 72.5 μs | 463.6 μs | 100.4 μs | — | **6.39x** | **1.39x** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | off | **41.0 μs** | 44.4 μs | 260.0 μs | 47.3 μs | — | **6.35x** | **1.16x** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | off | **63.5 μs** | 70.5 μs | 348.0 μs | 85.6 μs | — | **5.48x** | **1.35x** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | off | **81.8 μs** | 123.3 μs | 458.2 μs | 154.0 μs | — | **5.60x** | **1.88x** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | off | **39.7 μs** | 29.3 μs | 278.9 μs | 47.7 μs | — | **9.53x** | **1.63x** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | off | **102.9 μs** | 62.1 μs | 411.0 μs | 118.1 μs | — | **6.62x** | **1.90x** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | off | **193.2 μs** | 169.9 μs | 618.9 μs | 237.1 μs | — | **3.64x** | **1.40x** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | off | **36.9 μs** | 33.5 μs | 267.4 μs | 44.1 μs | — | **7.98x** | **1.32x** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | off | **51.0 μs** | 50.2 μs | 750.9 μs | 138.1 μs | — | **14.97x** | **2.75x** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | off | **109.9 μs** | 119.3 μs | 601.1 μs | 133.3 μs | — | **5.47x** | **1.21x** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | off | **67.3 μs** | 48.3 μs | 451.7 μs | 76.9 μs | — | **9.36x** | **1.59x** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | off | **83.8 μs** | 91.8 μs | 563.6 μs | 134.1 μs | — | **6.72x** | **1.60x** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | off | **126.9 μs** | 160.9 μs | 712.7 μs | 243.0 μs | — | **5.62x** | **1.92x** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | off | **59.0 μs** | 55.8 μs | 459.6 μs | 81.4 μs | — | **8.23x** | **1.46x** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | off | **94.0 μs** | 154.1 μs | 646.0 μs | 180.1 μs | — | **6.88x** | **1.92x** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | off | **263.0 μs** | 237.2 μs | 944.8 μs | 362.6 μs | — | **3.98x** | **1.53x** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | off | **73.7 μs** | 74.4 μs | 618.2 μs | 75.3 μs | — | **8.39x** | **1.02x** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | off | **83.6 μs** | 64.6 μs | 669.4 μs | 91.8 μs | — | **10.37x** | **1.42x** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | off | **96.3 μs** | 89.9 μs | 715.2 μs | 106.7 μs | — | **7.96x** | **1.19x** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | off | **75.5 μs** | 95.6 μs | 628.1 μs | 95.9 μs | — | **8.32x** | **1.27x** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | off | **111.1 μs** | 81.3 μs | 689.7 μs | 149.1 μs | — | **8.48x** | **1.83x** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | off | **64.5 μs** | 75.3 μs | 578.1 μs | 140.2 μs | — | **8.97x** | **2.18x** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | off | **84.0 μs** | 86.8 μs | 558.3 μs | 138.0 μs | — | **6.65x** | **1.64x** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | off | **92.8 μs** | 77.5 μs | 557.8 μs | 134.7 μs | — | **7.19x** | **1.74x** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | off | **102.0 μs** | 85.8 μs | 554.2 μs | 136.8 μs | — | **6.46x** | **1.59x** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | off | **93.0 μs** | 96.5 μs | 561.6 μs | 134.1 μs | — | **6.04x** | **1.44x** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | off | **65.0 μs** | 41.2 μs | 446.8 μs | 67.4 μs | — | **10.85x** | **1.64x** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | off | **65.8 μs** | 49.3 μs | 469.5 μs | 78.2 μs | — | **9.52x** | **1.58x** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | off | **67.2 μs** | 49.5 μs | 497.3 μs | 82.7 μs | — | **10.05x** | **1.67x** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | off | **44.3 μs** | 67.4 μs | 448.2 μs | 70.5 μs | — | **10.11x** | **1.59x** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | off | **64.2 μs** | 47.5 μs | 477.5 μs | 84.7 μs | — | **10.05x** | **1.78x** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | off | **48.5 μs** | 70.7 μs | 502.1 μs | 80.3 μs | — | **10.34x** | **1.65x** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | off | **46.0 μs** | 66.1 μs | 439.6 μs | 66.0 μs | — | **9.56x** | **1.43x** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | off | **44.2 μs** | 59.3 μs | 453.2 μs | 70.4 μs | — | **10.26x** | **1.59x** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | off | **66.1 μs** | 47.7 μs | 486.8 μs | 78.8 μs | — | **10.20x** | **1.65x** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | off | **56.4 μs** | 61.7 μs | 433.9 μs | 67.8 μs | — | **7.70x** | **1.20x** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | off | **62.4 μs** | 50.3 μs | 471.0 μs | 76.7 μs | — | **9.37x** | **1.52x** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | off | **62.8 μs** | 64.8 μs | 498.1 μs | 73.3 μs | — | **7.94x** | **1.17x** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | off | **50.3 μs** | 56.6 μs | 441.3 μs | 72.4 μs | — | **8.77x** | **1.44x** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | off | **60.5 μs** | 50.1 μs | 470.1 μs | 73.6 μs | — | **9.38x** | **1.47x** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | off | **49.9 μs** | 66.5 μs | 491.1 μs | 79.3 μs | — | **9.85x** | **1.59x** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | off | **42.8 μs** | 58.8 μs | 621.6 μs | 76.8 μs | — | **14.51x** | **1.79x** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | off | **62.2 μs** | 59.5 μs | 643.8 μs | 82.6 μs | — | **10.81x** | **1.39x** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | off | **58.5 μs** | 43.8 μs | 680.2 μs | 73.9 μs | — | **15.53x** | **1.69x** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | on | **23.68 ms** | 23.74 ms | 45.60 ms | 26.33 ms | — | **1.93x** | **1.11x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | on | **20.27 ms** | 20.16 ms | 73.19 ms | 23.02 ms | — | **3.63x** | **1.14x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | on | **45.78 ms** | 45.76 ms | 51.89 ms | 44.53 ms | — | **1.13x** | **0.97x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | on | **12.53 ms** | 12.50 ms | 32.87 ms | 12.57 ms | — | **2.63x** | **1.01x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | on | **677.7 μs** | 715.7 μs | 1.51 ms | — | — | **2.23x** | **—** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | on | **3.87 ms** | 3.90 ms | 12.45 ms | — | — | **3.21x** | **—** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | on | **1.32 ms** | 1.36 ms | 2.45 ms | — | — | **1.86x** | **—** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | on | **7.25 ms** | 7.65 ms | 11.68 ms | — | — | **1.61x** | **—** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | on | **397.2 μs** | 411.2 μs | 1.02 ms | — | — | **2.58x** | **—** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | on | **1.29 ms** | 1.33 ms | 2.58 ms | — | — | **1.99x** | **—** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | on | **712.3 μs** | 676.3 μs | 1.50 ms | — | — | **2.21x** | **—** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | on | **3.69 ms** | 4.03 ms | 12.47 ms | — | — | **3.37x** | **—** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | on | **1.28 ms** | 1.26 ms | 2.45 ms | — | — | **1.94x** | **—** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | on | **6.90 ms** | 6.69 ms | 11.69 ms | — | — | **1.75x** | **—** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | on | **291.3 μs** | 316.7 μs | — | — | — | **—** | **—** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | on | **1.02 ms** | 1.04 ms | — | — | — | **—** | **—** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | on | **1.36 ms** | 1.34 ms | — | — | — | **—** | **—** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | on | **5.48 ms** | 5.94 ms | — | — | — | **—** | **—** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | on | **108.7 μs** | 121.9 μs | 595.2 μs | — | — | **5.48x** | **—** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | on | **729.6 μs** | 726.8 μs | 1.93 ms | — | — | **2.66x** | **—** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | on | **1.08 ms** | 1.12 ms | 2.16 ms | — | — | **2.00x** | **—** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | on | **171.6 μs** | 175.8 μs | 717.9 μs | — | — | **4.18x** | **—** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | on | **1.36 ms** | 1.36 ms | 2.59 ms | — | — | **1.90x** | **—** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | on | **2.05 ms** | 2.07 ms | 3.72 ms | — | — | **1.82x** | **—** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | on | **100.4 μs** | 109.0 μs | 518.8 μs | — | — | **5.17x** | **—** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | on | **383.5 μs** | 416.5 μs | 1.08 ms | — | — | **2.81x** | **—** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | on | **585.6 μs** | 582.7 μs | 1.38 ms | — | — | **2.36x** | **—** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | on | **95.5 μs** | 131.0 μs | 579.4 μs | — | — | **6.06x** | **—** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | on | **743.8 μs** | 697.3 μs | 1.90 ms | — | — | **2.73x** | **—** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | on | **1.06 ms** | 1.05 ms | 2.16 ms | — | — | **2.05x** | **—** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | on | **196.0 μs** | 213.1 μs | 697.7 μs | — | — | **3.56x** | **—** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | on | **1.32 ms** | 1.31 ms | 2.56 ms | — | — | **1.95x** | **—** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | on | **1.97 ms** | 2.00 ms | 3.72 ms | — | — | **1.89x** | **—** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | on | **96.4 μs** | 99.7 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | on | **326.0 μs** | 273.0 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | on | **428.0 μs** | 466.3 μs | — | — | — | **—** | **—** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | on | **420.1 μs** | 435.5 μs | — | — | — | **—** | **—** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | on | **1.17 ms** | 1.21 ms | — | — | — | **—** | **—** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | on | **286.0 μs** | 340.1 μs | — | — | — | **—** | **—** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | on | **101.4 μs** | 69.7 μs | — | — | — | **—** | **—** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | on | **58.9 μs** | 109.1 μs | — | — | — | **—** | **—** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | on | **4.14 ms** | 5.55 ms | — | — | — | **—** | **—** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | on | **1.10 ms** | 1.12 ms | — | — | — | **—** | **—** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | on | **146.5 μs** | 119.0 μs | — | — | — | **—** | **—** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | on | **45.4 μs** | 67.7 μs | 355.2 μs | — | — | **7.83x** | **—** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | on | **46.2 μs** | 71.8 μs | 343.8 μs | — | — | **7.44x** | **—** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | on | **117.6 μs** | 106.3 μs | 462.4 μs | — | — | **4.35x** | **—** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | on | **44.6 μs** | 32.7 μs | 282.3 μs | — | — | **8.62x** | **—** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | on | **107.6 μs** | 94.5 μs | 424.0 μs | — | — | **4.49x** | **—** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | on | **197.3 μs** | 194.8 μs | 589.9 μs | — | — | **3.03x** | **—** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | on | **52.1 μs** | 37.5 μs | 270.5 μs | — | — | **7.22x** | **—** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | on | **73.7 μs** | 73.4 μs | 312.3 μs | — | — | **4.25x** | **—** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | on | **96.0 μs** | 83.0 μs | 383.0 μs | — | — | **4.61x** | **—** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | on | **48.3 μs** | 52.3 μs | 272.0 μs | — | — | **5.63x** | **—** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | on | **82.3 μs** | 81.0 μs | 351.0 μs | — | — | **4.33x** | **—** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | on | **91.2 μs** | 139.6 μs | 445.0 μs | — | — | **4.88x** | **—** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | on | **57.4 μs** | 43.4 μs | 283.8 μs | — | — | **6.53x** | **—** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | on | **105.0 μs** | 114.9 μs | 425.5 μs | — | — | **4.05x** | **—** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | on | **221.2 μs** | 201.7 μs | 584.3 μs | — | — | **2.90x** | **—** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | on | **51.7 μs** | 52.6 μs | — | — | — | **—** | **—** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | on | **57.1 μs** | 59.9 μs | — | — | — | **—** | **—** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | on | **76.8 μs** | 85.6 μs | — | — | — | **—** | **—** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | on | **102.3 μs** | 85.9 μs | — | — | — | **—** | **—** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | on | **107.9 μs** | 92.0 μs | — | — | — | **—** | **—** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | on | **38.7 μs** | 67.9 μs | 352.0 μs | — | — | **9.09x** | **—** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | on | **71.0 μs** | 67.5 μs | 345.0 μs | — | — | **5.11x** | **—** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | on | **70.2 μs** | 66.6 μs | 347.9 μs | — | — | **5.23x** | **—** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | on | **77.8 μs** | 42.9 μs | 344.4 μs | — | — | **8.03x** | **—** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | on | **45.2 μs** | 64.7 μs | 349.2 μs | — | — | **7.72x** | **—** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | on | **31.5 μs** | 41.9 μs | 266.3 μs | — | — | **8.46x** | **—** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | on | **40.7 μs** | 38.9 μs | 285.5 μs | — | — | **7.34x** | **—** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | on | **42.0 μs** | 30.7 μs | 296.4 μs | — | — | **9.66x** | **—** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | on | **31.1 μs** | 36.9 μs | 260.0 μs | — | — | **8.37x** | **—** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | on | **32.0 μs** | 30.8 μs | 289.9 μs | — | — | **9.41x** | **—** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | on | **28.3 μs** | 38.2 μs | 307.1 μs | — | — | **10.85x** | **—** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | on | **35.0 μs** | 45.3 μs | 265.4 μs | — | — | **7.57x** | **—** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | on | **50.0 μs** | 34.6 μs | 286.6 μs | — | — | **8.29x** | **—** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | on | **40.2 μs** | 35.3 μs | 513.8 μs | — | — | **14.57x** | **—** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | on | **30.1 μs** | 51.5 μs | 326.7 μs | — | — | **10.86x** | **—** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | on | **52.5 μs** | 43.6 μs | 286.0 μs | — | — | **6.55x** | **—** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | on | **46.9 μs** | 49.1 μs | 296.8 μs | — | — | **6.32x** | **—** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | on | **39.1 μs** | 36.4 μs | 264.2 μs | — | — | **7.26x** | **—** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | on | **53.9 μs** | 30.2 μs | 290.5 μs | — | — | **9.62x** | **—** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | on | **48.1 μs** | 34.2 μs | 312.8 μs | — | — | **9.14x** | **—** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | on | **47.8 μs** | 27.6 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | on | **51.6 μs** | 34.2 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | on | **30.9 μs** | 31.7 μs | — | — | — | **—** | **—** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **533.1 μs** | 553.9 μs | 4.51 ms | 608.2 μs | — | **8.45x** | **1.14x** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | off | **630.4 μs** | 540.0 μs | 4.47 ms | 612.6 μs | — | **8.28x** | **1.13x** |
| table | ascii_10000 | projection | 0.44 MB | CPU | off | **1.70 ms** | 1.58 ms | 13.91 ms | 3.42 ms | — | **8.81x** | **2.17x** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | off | **1.00 ms** | 925.3 μs | 8.03 ms | 1.97 ms | — | **8.68x** | **2.13x** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | off | **304.7 μs** | 199.0 μs | 4.42 ms | 889.4 μs | — | **22.19x** | **4.47x** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | off | **46.3 μs** | 69.1 μs | 985.6 μs | 148.5 μs | — | **21.28x** | **3.21x** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **99.2 μs** | 102.7 μs | 1.50 ms | 171.3 μs | — | **15.09x** | **1.73x** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | off | **96.1 μs** | 102.3 μs | 1.50 ms | 170.0 μs | — | **15.60x** | **1.77x** |
| table | ascii_1000 | projection | 50.6 KB | CPU | off | **188.6 μs** | 139.0 μs | 2.60 ms | 411.8 μs | — | **18.73x** | **2.96x** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | off | **185.1 μs** | 118.8 μs | 2.15 ms | 329.6 μs | — | **18.06x** | **2.77x** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | off | **118.3 μs** | 52.8 μs | 1.94 ms | 199.2 μs | — | **36.70x** | **3.78x** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | off | **27.6 μs** | 29.0 μs | 422.5 μs | 67.7 μs | — | **15.33x** | **2.46x** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **11.07 ms** | 10.77 ms | 15.84 ms | 20.61 ms | — | **1.47x** | **1.91x** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | off | **10.06 ms** | 10.07 ms | 12.41 ms | 16.81 ms | — | **1.23x** | **1.67x** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | off | **10.83 ms** | 10.11 ms | 16.91 ms | 32.01 ms | — | **1.67x** | **3.17x** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | off | **35.53 ms** | 26.48 ms | 337.08 ms | 119.14 ms | — | **12.73x** | **4.50x** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **297.9 μs** | 223.2 μs | 13.42 ms | 1.52 ms | — | **60.12x** | **6.79x** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **30.4 μs** | 28.5 μs | 467.6 μs | 88.0 μs | — | **16.42x** | **3.09x** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **2.04 ms** | 1.92 ms | 5.10 ms | 3.70 ms | — | **2.66x** | **1.92x** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | off | **1.75 ms** | 1.78 ms | 4.56 ms | 3.04 ms | — | **2.60x** | **1.73x** |
| table | mixed_100000 | projection | 5.06 MB | CPU | off | **1.56 ms** | 1.44 ms | 5.07 ms | 5.33 ms | — | **3.53x** | **3.71x** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | off | **3.18 ms** | 2.93 ms | 52.22 ms | 16.87 ms | — | **17.84x** | **5.76x** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | off | **445.1 μs** | 329.8 μs | 10.00 ms | 2.53 ms | — | **30.33x** | **7.67x** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | off | **48.6 μs** | 47.0 μs | 778.8 μs | 133.3 μs | — | **16.57x** | **2.84x** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **225.4 μs** | 261.4 μs | 1.99 ms | 395.1 μs | — | **8.85x** | **1.75x** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | off | **164.7 μs** | 207.1 μs | 1.94 ms | 350.7 μs | — | **11.76x** | **2.13x** |
| table | mixed_10000 | projection | 0.51 MB | CPU | off | **184.9 μs** | 114.0 μs | 1.95 ms | 487.7 μs | — | **17.07x** | **4.28x** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | off | **288.3 μs** | 299.2 μs | 7.18 ms | 1.65 ms | — | **24.89x** | **5.74x** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | off | **126.1 μs** | 69.0 μs | 2.98 ms | 332.6 μs | — | **43.20x** | **4.82x** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | off | **28.6 μs** | 28.5 μs | 444.8 μs | 77.3 μs | — | **15.61x** | **2.71x** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **44.0 μs** | 86.2 μs | 1.84 ms | 195.2 μs | — | **41.85x** | **4.44x** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | off | **39.8 μs** | 82.9 μs | 1.82 ms | 188.9 μs | — | **45.61x** | **4.74x** |
| table | mixed_1000 | projection | 0.06 MB | CPU | off | **108.9 μs** | 46.2 μs | 1.84 ms | 205.0 μs | — | **39.70x** | **4.43x** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | off | **130.1 μs** | 63.4 μs | 2.18 ms | 274.4 μs | — | **34.46x** | **4.33x** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | off | **117.1 μs** | 50.6 μs | 2.66 ms | 212.3 μs | — | **52.65x** | **4.20x** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | off | **29.5 μs** | 28.9 μs | 459.2 μs | 75.3 μs | — | **15.87x** | **2.60x** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **8.70 ms** | 8.61 ms | 14.32 ms | 25.34 ms | — | **1.66x** | **2.94x** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | off | **7.80 ms** | 7.80 ms | 8.46 ms | 19.16 ms | — | **1.09x** | **2.46x** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | off | **7.28 ms** | 6.77 ms | 8.71 ms | 40.51 ms | — | **1.29x** | **5.99x** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | off | **9.51 ms** | 9.37 ms | 10.69 ms | 9.17 ms | — | **1.14x** | **0.98x** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **264.9 μs** | 154.5 μs | 6.27 ms | 985.1 μs | — | **40.60x** | **6.38x** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **52.0 μs** | 47.4 μs | 766.7 μs | 118.3 μs | — | **16.17x** | **2.49x** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **1.31 ms** | 1.14 ms | 3.54 ms | 2.78 ms | — | **3.10x** | **2.44x** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | off | **955.2 μs** | 978.5 μs | 2.94 ms | 2.14 ms | — | **3.07x** | **2.24x** |
| table | narrow_100000 | projection | 1.25 MB | CPU | off | **853.2 μs** | 935.0 μs | 3.67 ms | 5.50 ms | — | **4.30x** | **6.45x** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | off | **1.11 ms** | 996.0 μs | 3.14 ms | 1.11 ms | — | **3.16x** | **1.11x** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | off | **249.4 μs** | 153.7 μs | 3.47 ms | 961.6 μs | — | **22.55x** | **6.26x** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | off | **45.4 μs** | 47.0 μs | 768.6 μs | 108.9 μs | — | **16.93x** | **2.40x** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **146.8 μs** | 184.6 μs | 1.40 ms | 309.9 μs | — | **9.56x** | **2.11x** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | off | **94.3 μs** | 134.3 μs | 1.42 ms | 277.7 μs | — | **15.12x** | **2.95x** |
| table | narrow_10000 | projection | 0.13 MB | CPU | off | **137.1 μs** | 78.3 μs | 1.36 ms | 391.1 μs | — | **17.37x** | **5.00x** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | off | **158.1 μs** | 92.3 μs | 1.40 ms | 195.8 μs | — | **15.21x** | **2.12x** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | off | **111.9 μs** | 47.7 μs | 1.77 ms | 202.4 μs | — | **37.14x** | **4.25x** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | off | **25.3 μs** | 26.4 μs | 426.6 μs | 64.0 μs | — | **16.87x** | **2.53x** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **48.2 μs** | 94.2 μs | 1.33 ms | 175.7 μs | — | **27.66x** | **3.65x** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | off | **42.7 μs** | 87.8 μs | 1.40 ms | 172.2 μs | — | **32.70x** | **4.04x** |
| table | narrow_1000 | projection | 19.7 KB | CPU | off | **107.5 μs** | 44.7 μs | 1.32 ms | 180.3 μs | — | **29.48x** | **4.03x** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | off | **102.5 μs** | 44.3 μs | 1.36 ms | 153.6 μs | — | **30.80x** | **3.47x** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | off | **103.6 μs** | 39.7 μs | 1.74 ms | 162.6 μs | — | **43.85x** | **4.09x** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | off | **27.3 μs** | 28.3 μs | 427.7 μs | 65.9 μs | — | **15.69x** | **2.42x** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **805.4 μs** | 827.1 μs | 1.89 ms | 1.37 ms | — | **2.34x** | **1.71x** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | off | **805.9 μs** | 818.7 μs | 1.89 ms | 1.39 ms | — | **2.34x** | **1.72x** |
| table | typed_100000 | projection | 2.39 MB | CPU | off | **3.82 ms** | 3.43 ms | 29.10 ms | 13.10 ms | — | **8.49x** | **3.82x** |
| table | typed_100000 | read_full | 2.39 MB | CPU | off | **5.22 ms** | 5.08 ms | 29.11 ms | 14.14 ms | — | **5.73x** | **2.78x** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | off | **634.0 μs** | 561.9 μs | 4.74 ms | 1.95 ms | — | **8.43x** | **3.46x** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | off | **30.6 μs** | 29.9 μs | 436.7 μs | 68.4 μs | — | **14.60x** | **2.29x** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **160.9 μs** | 192.8 μs | 1.51 ms | 300.6 μs | — | **9.40x** | **1.87x** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | off | **145.6 μs** | 188.7 μs | 1.45 ms | 289.8 μs | — | **9.96x** | **1.99x** |
| table | typed_10000 | projection | 0.24 MB | CPU | off | **466.5 μs** | 402.1 μs | 4.08 ms | 1.47 ms | — | **10.14x** | **3.66x** |
| table | typed_10000 | read_full | 0.24 MB | CPU | off | **630.0 μs** | 572.7 μs | 4.33 ms | 1.68 ms | — | **7.56x** | **2.93x** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | off | **161.6 μs** | 93.1 μs | 2.19 ms | 398.8 μs | — | **23.57x** | **4.29x** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | off | **28.6 μs** | 30.0 μs | 452.6 μs | 66.3 μs | — | **15.81x** | **2.32x** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **869.1 μs** | 721.6 μs | 1.76 ms | 1.28 ms | — | **2.44x** | **1.78x** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | off | **692.2 μs** | 717.7 μs | 1.76 ms | 1.28 ms | — | **2.54x** | **1.85x** |
| table | varlen_100000 | projection | 3.06 MB | CPU | off | **72.92 ms** | 71.84 ms | 523.28 ms | 110.15 ms | — | **7.28x** | **1.53x** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | off | **71.45 ms** | 71.93 ms | 523.01 ms | 110.15 ms | — | **7.32x** | **1.54x** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | off | **6.97 ms** | 6.85 ms | 53.83 ms | 11.89 ms | — | **7.86x** | **1.74x** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | off | **29.5 μs** | 28.4 μs | 459.6 μs | 67.2 μs | — | **16.16x** | **2.36x** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **123.4 μs** | 167.4 μs | 1.40 ms | 288.3 μs | — | **11.38x** | **2.34x** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | off | **111.2 μs** | 149.7 μs | 1.34 ms | 273.1 μs | — | **12.03x** | **2.46x** |
| table | varlen_10000 | projection | 0.31 MB | CPU | off | **6.94 ms** | 6.87 ms | 52.84 ms | 10.98 ms | — | **7.69x** | **1.60x** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | off | **6.93 ms** | 6.87 ms | 52.78 ms | 10.91 ms | — | **7.69x** | **1.59x** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | off | **781.8 μs** | 710.4 μs | 7.02 ms | 1.38 ms | — | **9.87x** | **1.94x** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | off | **28.3 μs** | 27.6 μs | 444.1 μs | 63.9 μs | — | **16.10x** | **2.31x** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **46.5 μs** | 88.5 μs | 1.27 ms | 171.7 μs | — | **27.37x** | **3.69x** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | off | **41.7 μs** | 83.3 μs | 1.25 ms | 171.1 μs | — | **30.13x** | **4.11x** |
| table | varlen_1000 | projection | 39.4 KB | CPU | off | **795.5 μs** | 737.4 μs | 6.52 ms | 1.29 ms | — | **8.84x** | **1.75x** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | off | **778.9 μs** | 705.2 μs | 6.50 ms | 1.27 ms | — | **9.21x** | **1.81x** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | off | **195.7 μs** | 133.4 μs | 2.23 ms | 298.9 μs | — | **16.72x** | **2.24x** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | off | **26.4 μs** | 28.2 μs | 429.2 μs | 63.4 μs | — | **16.25x** | **2.40x** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **5.30 ms** | 5.20 ms | 14.11 ms | 7.18 ms | — | **2.71x** | **1.38x** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | off | **5.04 ms** | 5.04 ms | 13.55 ms | 6.60 ms | — | **2.69x** | **1.31x** |
| table | wide_100000 | projection | 20.71 MB | CPU | off | **4.17 ms** | 4.05 ms | 14.43 ms | 8.87 ms | — | **3.57x** | **2.19x** |
| table | wide_100000 | read_full | 20.71 MB | CPU | off | **27.79 ms** | 26.15 ms | 223.81 ms | 72.18 ms | — | **8.56x** | **2.76x** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | off | **1.59 ms** | 1.48 ms | 37.29 ms | 8.30 ms | — | **25.27x** | **5.62x** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | off | **62.9 μs** | 60.4 μs | 954.1 μs | 439.2 μs | — | **15.81x** | **7.28x** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **554.7 μs** | 596.7 μs | 8.84 ms | 1.10 ms | — | **15.94x** | **1.99x** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | off | **594.6 μs** | 632.4 μs | 10.02 ms | 1.21 ms | — | **16.86x** | **2.03x** |
| table | wide_10000 | projection | 2.08 MB | CPU | off | **356.7 μs** | 279.6 μs | 5.93 ms | 884.8 μs | — | **21.23x** | **3.16x** |
| table | wide_10000 | read_full | 2.08 MB | CPU | off | **1.16 ms** | 1.07 ms | 16.80 ms | 4.55 ms | — | **15.72x** | **4.26x** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | off | **229.5 μs** | 160.4 μs | 10.60 ms | 906.8 μs | — | **66.11x** | **5.65x** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | off | **64.8 μs** | 62.6 μs | 941.3 μs | 432.2 μs | — | **15.04x** | **6.91x** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **52.3 μs** | 102.1 μs | 5.63 ms | 399.3 μs | — | **107.53x** | **7.63x** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | off | **60.1 μs** | 100.0 μs | 5.64 ms | 401.9 μs | — | **93.94x** | **6.69x** |
| table | wide_1000 | projection | 0.22 MB | CPU | off | **119.9 μs** | 54.4 μs | 5.67 ms | 412.8 μs | — | **104.17x** | **7.59x** |
| table | wide_1000 | read_full | 0.22 MB | CPU | off | **230.9 μs** | 159.8 μs | 7.12 ms | 836.8 μs | — | **44.53x** | **5.24x** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | off | **158.9 μs** | 88.4 μs | 9.34 ms | 511.9 μs | — | **105.65x** | **5.79x** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | off | **39.7 μs** | 37.6 μs | 558.2 μs | 255.3 μs | — | **14.84x** | **6.79x** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **404.1 μs** | 396.9 μs | 2.67 ms | — | — | **6.72x** | **—** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | on | **387.2 μs** | 406.1 μs | 2.66 ms | — | — | **6.86x** | **—** |
| table | ascii_10000 | projection | 0.44 MB | CPU | on | **1.06 ms** | 988.0 μs | 8.08 ms | — | — | **8.18x** | **—** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | on | **1.06 ms** | 985.5 μs | 8.09 ms | — | — | **8.21x** | **—** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | on | **243.7 μs** | 180.2 μs | 2.61 ms | — | — | **14.49x** | **—** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | on | **29.4 μs** | 28.0 μs | 428.7 μs | — | — | **15.32x** | **—** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **167.0 μs** | 171.9 μs | 1.55 ms | — | — | **9.28x** | **—** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | on | **161.3 μs** | 171.5 μs | 1.55 ms | — | — | **9.59x** | **—** |
| table | ascii_1000 | projection | 50.6 KB | CPU | on | **248.8 μs** | 188.2 μs | 2.20 ms | — | — | **11.70x** | **—** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | on | **248.1 μs** | 264.9 μs | 3.33 ms | — | — | **13.41x** | **—** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | on | **175.0 μs** | 109.5 μs | 1.99 ms | — | — | **18.16x** | **—** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | on | **29.2 μs** | 30.9 μs | 437.0 μs | — | — | **14.99x** | **—** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **6.09 ms** | 7.10 ms | 19.65 ms | — | — | **3.22x** | **—** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | on | **6.31 ms** | 6.28 ms | 13.99 ms | — | — | **2.23x** | **—** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | on | **7.54 ms** | 7.43 ms | 14.93 ms | — | — | **2.01x** | **—** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | on | **31.63 ms** | 25.86 ms | 341.91 ms | — | — | **13.22x** | **—** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **263.3 μs** | 178.4 μs | 9.40 ms | — | — | **52.67x** | **—** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **52.5 μs** | 50.4 μs | 800.7 μs | — | — | **15.88x** | **—** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **1.01 ms** | 735.6 μs | 2.98 ms | — | — | **4.05x** | **—** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | on | **601.6 μs** | 595.4 μs | 2.65 ms | — | — | **4.45x** | **—** |
| table | mixed_100000 | projection | 5.06 MB | CPU | on | **896.8 μs** | 794.6 μs | 3.06 ms | — | — | **3.85x** | **—** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | on | **1.90 ms** | 1.78 ms | 30.69 ms | — | — | **17.23x** | **—** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | on | **253.2 μs** | 164.3 μs | 5.80 ms | — | — | **35.32x** | **—** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | on | **33.6 μs** | 31.7 μs | 447.8 μs | — | — | **14.12x** | **—** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **181.6 μs** | 193.9 μs | 1.98 ms | — | — | **10.91x** | **—** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | on | **127.9 μs** | 142.1 μs | 1.94 ms | — | — | **15.14x** | **—** |
| table | mixed_10000 | projection | 0.51 MB | CPU | on | **167.0 μs** | 86.5 μs | 1.95 ms | — | — | **22.57x** | **—** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | on | **248.4 μs** | 172.4 μs | 4.61 ms | — | — | **26.74x** | **—** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | on | **142.1 μs** | 72.5 μs | 3.00 ms | — | — | **41.43x** | **—** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | on | **27.8 μs** | 29.9 μs | 459.1 μs | — | — | **16.49x** | **—** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **55.8 μs** | 98.2 μs | 1.85 ms | — | — | **33.21x** | **—** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | on | **44.9 μs** | 94.4 μs | 1.88 ms | — | — | **41.88x** | **—** |
| table | mixed_1000 | projection | 0.06 MB | CPU | on | **118.9 μs** | 54.9 μs | 1.85 ms | — | — | **33.72x** | **—** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | on | **143.8 μs** | 72.3 μs | 2.34 ms | — | — | **32.34x** | **—** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | on | **126.4 μs** | 63.8 μs | 2.70 ms | — | — | **42.38x** | **—** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | on | **30.9 μs** | 30.7 μs | 458.8 μs | — | — | **14.95x** | **—** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **3.31 ms** | 2.47 ms | 8.18 ms | — | — | **3.31x** | **—** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | on | **1.71 ms** | 1.72 ms | 4.75 ms | — | — | **2.78x** | **—** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | on | **2.29 ms** | 2.17 ms | 5.34 ms | — | — | **2.46x** | **—** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | on | **3.39 ms** | 3.30 ms | 6.63 ms | — | — | **2.01x** | **—** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **161.7 μs** | 89.2 μs | 3.49 ms | — | — | **39.12x** | **—** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **29.5 μs** | 29.0 μs | 466.7 μs | — | — | **16.08x** | **—** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **740.4 μs** | 475.6 μs | 2.12 ms | — | — | **4.45x** | **—** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | on | **333.9 μs** | 339.0 μs | 1.82 ms | — | — | **5.45x** | **—** |
| table | narrow_100000 | projection | 1.25 MB | CPU | on | **333.5 μs** | 231.3 μs | 1.78 ms | — | — | **7.70x** | **—** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | on | **461.1 μs** | 363.6 μs | 1.90 ms | — | — | **5.24x** | **—** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | on | **150.7 μs** | 95.0 μs | 2.14 ms | — | — | **22.54x** | **—** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | on | **31.3 μs** | 28.3 μs | 456.8 μs | — | — | **16.13x** | **—** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **150.7 μs** | 182.1 μs | 1.44 ms | — | — | **9.54x** | **—** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | on | **89.7 μs** | 128.7 μs | 1.41 ms | — | — | **15.74x** | **—** |
| table | narrow_10000 | projection | 0.13 MB | CPU | on | **138.0 μs** | 64.2 μs | 1.40 ms | — | — | **21.85x** | **—** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | on | **147.6 μs** | 78.2 μs | 1.44 ms | — | — | **18.35x** | **—** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | on | **119.6 μs** | 57.2 μs | 1.82 ms | — | — | **31.91x** | **—** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | on | **27.3 μs** | 27.1 μs | 441.8 μs | — | — | **16.31x** | **—** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **54.9 μs** | 101.2 μs | 1.37 ms | — | — | **25.05x** | **—** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | on | **49.2 μs** | 97.9 μs | 1.34 ms | — | — | **27.18x** | **—** |
| table | narrow_1000 | projection | 19.7 KB | CPU | on | **115.8 μs** | 50.1 μs | 1.41 ms | — | — | **28.21x** | **—** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | on | **117.8 μs** | 51.7 μs | 1.40 ms | — | — | **27.10x** | **—** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | on | **114.0 μs** | 50.3 μs | 1.79 ms | — | — | **35.66x** | **—** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | on | **31.0 μs** | 27.5 μs | 467.8 μs | — | — | **16.98x** | **—** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **618.8 μs** | 452.0 μs | 1.82 ms | — | — | **4.03x** | **—** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | on | **452.9 μs** | 444.3 μs | 1.81 ms | — | — | **4.07x** | **—** |
| table | typed_100000 | projection | 2.39 MB | CPU | on | **1.25 ms** | 1.16 ms | 29.38 ms | — | — | **25.41x** | **—** |
| table | typed_100000 | read_full | 2.39 MB | CPU | on | **1.38 ms** | 1.27 ms | 29.46 ms | — | — | **23.12x** | **—** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | on | **263.5 μs** | 178.7 μs | 4.58 ms | — | — | **25.60x** | **—** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | on | **34.0 μs** | 31.1 μs | 465.5 μs | — | — | **14.97x** | **—** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **133.8 μs** | 161.2 μs | 1.47 ms | — | — | **11.02x** | **—** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | on | **127.4 μs** | 156.2 μs | 1.46 ms | — | — | **11.48x** | **—** |
| table | typed_10000 | projection | 0.24 MB | CPU | on | **254.5 μs** | 171.1 μs | 4.09 ms | — | — | **23.87x** | **—** |
| table | typed_10000 | read_full | 0.24 MB | CPU | on | **260.6 μs** | 174.2 μs | 4.13 ms | — | — | **23.73x** | **—** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | on | **148.7 μs** | 70.5 μs | 2.20 ms | — | — | **31.18x** | **—** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | on | **30.3 μs** | 28.9 μs | 445.9 μs | — | — | **15.41x** | **—** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **553.5 μs** | 406.0 μs | 1.62 ms | — | — | **3.99x** | **—** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | on | **400.1 μs** | 382.0 μs | 1.60 ms | — | — | **4.19x** | **—** |
| table | varlen_100000 | projection | 3.06 MB | CPU | on | **74.22 ms** | 74.46 ms | 524.06 ms | — | — | **7.06x** | **—** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | on | **74.69 ms** | 75.18 ms | 530.43 ms | — | — | **7.10x** | **—** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | on | **7.13 ms** | 7.06 ms | 53.89 ms | — | — | **7.64x** | **—** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | on | **32.1 μs** | 32.1 μs | 452.6 μs | — | — | **14.12x** | **—** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **175.2 μs** | 227.0 μs | 2.24 ms | — | — | **12.79x** | **—** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | on | **183.2 μs** | 226.4 μs | 2.24 ms | — | — | **12.25x** | **—** |
| table | varlen_10000 | projection | 0.31 MB | CPU | on | **12.10 ms** | 12.01 ms | 92.64 ms | — | — | **7.71x** | **—** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | on | **12.10 ms** | 12.00 ms | 92.67 ms | — | — | **7.72x** | **—** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | on | **1.47 ms** | 1.34 ms | 12.08 ms | — | — | **9.03x** | **—** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | on | **47.3 μs** | 29.9 μs | 479.7 μs | — | — | **16.07x** | **—** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **90.3 μs** | 149.8 μs | 2.18 ms | — | — | **24.12x** | **—** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | on | **96.4 μs** | 150.4 μs | 2.18 ms | — | — | **22.57x** | **—** |
| table | varlen_1000 | projection | 39.4 KB | CPU | on | **1.42 ms** | 1.40 ms | 11.27 ms | — | — | **8.07x** | **—** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | on | **1.46 ms** | 1.39 ms | 11.28 ms | — | — | **8.12x** | **—** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | on | **398.9 μs** | 305.4 μs | 3.82 ms | — | — | **12.51x** | **—** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | on | **44.6 μs** | 44.3 μs | 749.8 μs | — | — | **16.91x** | **—** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **1.55 ms** | 1.36 ms | 7.48 ms | — | — | **5.50x** | **—** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | on | **1.22 ms** | 1.21 ms | 7.18 ms | — | — | **5.93x** | **—** |
| table | wide_100000 | projection | 20.71 MB | CPU | on | **1.66 ms** | 1.58 ms | 7.63 ms | — | — | **4.83x** | **—** |
| table | wide_100000 | read_full | 20.71 MB | CPU | on | **14.61 ms** | 16.51 ms | 133.28 ms | — | — | **9.12x** | **—** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | on | **787.4 μs** | 693.9 μs | 20.92 ms | — | — | **30.15x** | **—** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | on | **38.9 μs** | 40.4 μs | 571.4 μs | — | — | **14.69x** | **—** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **258.9 μs** | 276.9 μs | 5.89 ms | — | — | **22.74x** | **—** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | on | **199.6 μs** | 216.8 μs | 5.83 ms | — | — | **29.18x** | **—** |
| table | wide_10000 | projection | 2.08 MB | CPU | on | **241.2 μs** | 167.6 μs | 5.82 ms | — | — | **34.74x** | **—** |
| table | wide_10000 | read_full | 2.08 MB | CPU | on | **800.8 μs** | 705.7 μs | 16.62 ms | — | — | **23.55x** | **—** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | on | **224.0 μs** | 148.7 μs | 10.55 ms | — | — | **70.94x** | **—** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | on | **39.1 μs** | 38.6 μs | 578.9 μs | — | — | **15.01x** | **—** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **64.9 μs** | 115.5 μs | 5.67 ms | — | — | **87.31x** | **—** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | on | **66.7 μs** | 114.3 μs | 5.67 ms | — | — | **85.07x** | **—** |
| table | wide_1000 | projection | 0.22 MB | CPU | on | **130.9 μs** | 63.2 μs | 5.69 ms | — | — | **90.04x** | **—** |
| table | wide_1000 | read_full | 0.22 MB | CPU | on | **219.5 μs** | 141.3 μs | 7.15 ms | — | — | **50.62x** | **—** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | on | **171.2 μs** | 102.3 μs | 9.41 ms | — | — | **91.99x** | **—** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | on | **38.0 μs** | 36.7 μs | 579.7 μs | — | — | **15.81x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance comparisons & edge cases {#performance-deficits}

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (CPU and GPU). GPU lags may reflect software or hardware limits — they are listed, not hidden.

| Platform | Domain | Case | mmap | torchfits | Peak RSS (MB) | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 45.78 ms | 293.9 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 26.32 ms | 309.2 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 45.76 ms | 293.9 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 26.35 ms | 309.2 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | table | narrow_100000 [read_full] | off | 1.11 ms | 380.4 | fitsio/fitsio_torch | 1.25× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 9.51 ms | 443.3 | fitsio/fitsio_torch | 1.23× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 9.37 ms | 443.3 | fitsio/fitsio | 1.02× |
| Linux x86_64 / CUDA | tensor | small_int8_1d [read_full @ cuda] | off | 114.6 μs | 761.5 | fitsio/fitsio_torch_device | 1.13× |
| Linux x86_64 / CUDA | tensor | tiny_int8_2d [read_full @ cuda] | off | 111.5 μs | 761.5 | fitsio/fitsio_torch_device | 1.10× |
| Linux x86_64 / CUDA | tensor | tiny_float32_1d [read_full @ cuda] | off | 110.0 μs | 761.5 | fitsio/fitsio_torch_device | 1.10× |
| Linux x86_64 / CUDA | tensor | tiny_int8_1d [read_full @ cuda] | off | 107.8 μs | 761.5 | fitsio/fitsio_torch_device | 1.10× |
| Linux x86_64 / CUDA | tensor | small_int16_1d [read_full @ cuda] | off | 119.5 μs | 761.5 | fitsio/fitsio_torch_device | 1.10× |
| Linux x86_64 / CUDA | tensor | tiny_float64_1d [read_full @ cuda] | off | 111.5 μs | 761.5 | fitsio/fitsio_torch_device | 1.09× |
| Linux x86_64 / CUDA | tensor | tiny_int16_3d [read_full @ cuda] | off | 108.1 μs | 761.5 | fitsio/fitsio_torch_device | 1.07× |
| Linux x86_64 / CUDA | tensor | tiny_int64_3d [read_full @ cuda] | off | 117.7 μs | 761.5 | fitsio/fitsio_torch_device | 1.07× |
| Linux x86_64 / CUDA | tensor | tiny_float32_3d [read_full @ cuda] | off | 108.9 μs | 761.5 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | small_int8_3d [read_full @ cuda] | off | 159.9 μs | 761.5 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | small_int8_2d [read_full @ cuda] | off | 127.2 μs | 761.5 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | tiny_int64_2d [read_full @ cuda] | off | 115.3 μs | 761.5 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | on | 30.72 ms | 697.9 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | off | 30.62 ms | 729.3 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | off | 30.42 ms | 726.7 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | tensor | tiny_float32_2d [read_full @ cuda] | off | 110.1 μs | 761.5 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | tiny_int64_1d [read_full @ cuda] | off | 104.6 μs | 761.5 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | small_uint16_2d [read_full @ cuda] | off | 147.5 μs | 761.5 | fitsio/fitsio_torch_device | 1.02× |
| Linux x86_64 / CUDA | tensor | small_int64_1d [read_full @ cuda] | off | 128.3 μs | 761.5 | fitsio/fitsio_torch_device | 1.02× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | on | 30.23 ms | 605.8 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CUDA | tensor | medium_int8_1d [read_full @ cuda] | off | 133.5 μs | 761.5 | fitsio/fitsio_torch_device | 1.02× |
| Linux x86_64 / CUDA | tensor | tiny_int32_2d [read_full @ cuda] | off | 109.5 μs | 761.5 | fitsio/fitsio_torch_device | 1.01× |
| Linux x86_64 / CUDA | tensor | small_int32_1d [read_full @ cuda] | off | 116.2 μs | 761.5 | fitsio/fitsio_torch_device | 1.01× |
| Linux x86_64 / CUDA | tensor | tiny_int16_1d [read_full @ cuda] | off | 101.6 μs | 761.5 | fitsio/fitsio_torch_device | 1.00× |
| Linux x86_64 / CUDA | tensor | small_float32_1d [read_full @ cuda] | off | 113.5 μs | 761.5 | fitsio/fitsio_torch_device | 1.00× |
| Linux x86_64 / CUDA | tensor | small_int16_3d [read_full] | off | 162.9 μs | 758.6 | fitsio/fitsio_torch | 1.04× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | off | 30.54 ms | 729.6 | fitsio/fitsio_torch_device_specialized | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | off | 30.44 ms | 726.7 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | on | 30.56 ms | 697.9 | fitsio/fitsio_torch_device_specialized | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | on | 30.38 ms | 605.8 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CUDA | table | narrow_1000000 [read_full] | off | 8.33 ms | 757.3 | fitsio/fitsio_torch | 1.11× |
| Linux x86_64 / CUDA | table | narrow_1000000 [read_full] | off | 8.37 ms | 757.3 | fitsio/fitsio | 1.03× |
<!-- BENCH_DEFICITS_END -->

### Host scorecard

| Platform | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| Linux x86_64 / CPU | `exhaustive_cpu_20260822_213823` | 3057 | 7 | 293.9 | lab + mmap-matrix |
| Linux x86_64 / CUDA | `exhaustive_cuda_20260822_213846` | 4315 | 32 | 739.2 | lab + mmap-matrix + GPU |
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
