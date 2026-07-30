# Benchmarks

`torchfits` benchmarks cover FITS **tensor** I/O (IMAGE HDUs, typically 1D–4D)
and FITS **table** I/O vs Astropy and fitsio. CPU↔GPU comparisons are published
when hardware was available; GPU deficits are listed, not hidden.

**Honesty:** torchfits is a **1.0.0rc4** prerelease. Headline ratios below are
lab medians from named scorecard runs — not guarantees on your filesystem,
file mix, or PyTorch version. Check [Performance deficits](#performance-deficits)
before assuming torchfits wins every case.

## How to read this page

| If you want… | Jump to |
|---|---|
| Headline wins | [Performance highlights](#performance-highlights) |
| Cases where torchfits is not #1 (CPU and GPU) | [Performance deficits](#performance-deficits) |
| GPU transport rows | [I/O transport and backend](#io-transport-and-backend) |
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
**peak process RSS** (and peak CUDA alloc when on CUDA). Deficit ranking is
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

## Published CSVs

Exhaustive `results.csv` / `torchfits_deficits.csv` for the scorecard runs are
linked from GitHub Release assets when published, and mirrored under
`docs/assets/bench/<run-id>/` when size allows. Example local paths used to
build this page:

- `docs/assets/bench/exhaustive_mps_20260719_143706/results.csv`
- `docs/assets/bench/exhaustive_cpu_20260719_144337/results.csv`
- `docs/assets/bench/exhaustive_cuda_20260719_144457/results.csv`

(also under `benchmarks_results/<run-id>/` locally)


Deficit floors (same-mmap peers):

- **Images / cubes / spectra / cutouts** (`domain=fits`): any lag above float-timer
  ε counts — including rice/hcompress (no percent floor).
- **Arrow table interchange** (`domain=fitstable`): allow up to **1.05×**.

### Modular suites and release exhaustives

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

### CFHT MegaCam cutout suite

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

Outputs land in `benchmarks_results/<run-id>/megacam_results.csv`. Sample
FITS files are gitignored under `benchmarks_data/cfht_megacam/`.

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

Named **deficit-cluster** recipes (mmap on+off, no unrelated GPU matrix when scoped to tables):

```bash
pixi run bench-deficit-focus              # hcompress + tiny_int8 + narrow predicates
pixi run bench-deficit-focus hcompress
pixi run bench-deficit-focus tiny_int8
pixi run bench-deficit-focus predicate
```

Rankings and deficit scoring group by `(domain, case_id, family, mmap_target)` so
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
| BITPIX / dtypes | Partial | int8–int64, float32/64 × 1D/2D/3D | Native **uint16/uint32** 2D fixtures; unsigned via BZERO in `scaled_*` |
| Tensor dimensions / sizes | Yes | tiny → large; 1D–3D (4D where fixtures exist) | Large 3D cubes may hit size caps |
| Compression (read) | Yes | gzip, rice, hcompress, plio | Write→compress cases are being added to the suite |
| Scaling (BSCALE/BZERO) | Yes | `scaled_small/medium/large` | Table-column scaling not isolated |
| Random / repeated access | Yes | cutouts, `random_ext_full_reads_200`, `open_subset_reader` | MEF random ext reads on selected fixtures |
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
   CANFAR staging (`exhaustive_cuda_20260719_144457`).
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
Source: `benchmarks_results/exhaustive_mps_20260719_143706/results.csv` (mmap on+off matrix; MPS/CUDA GPU transport rows included.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### Tensor I/O (IMAGE HDU) (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.13 ms` (n=174) | `0.35 ms` (n=253) | `0.15 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.13 ms` (n=174) | `0.33 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | `0.36 ms` (n=152) | `0.72 ms` (n=152) | `0.36 ms` (n=152) | — |
| `disk→RAM→GPU` | `0.36 ms` (n=152) | `0.78 ms` (n=152) | `14.38 ms` (n=8) | — |

### Table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.51 ms` (n=146) | `2.20 ms` (n=164) | `0.75 ms` (n=180) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.33 ms` (n=144) | `2.22 ms` (n=164) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Large tensor read (Float32 2D, 16.0 MB) | CPU | **2.52 ms** | 2.61 ms | 4.77 ms | 2.44 ms | **1.89x** | **0.97x** |
| Large tensor read (Float32 2D @ CUDA) | CUDA | **3.40 ms** | 5.27 ms | 7.02 ms | 4.95 ms | **2.07x** | **1.46x** |
| Compressed tensor read (Rice, 1.1 MB) | CPU | **6.50 ms** | 6.48 ms | 17.60 ms | 6.51 ms | **2.72x** | **1.00x** |
| Compressed tensor read (Rice @ CUDA) | CUDA | **7.11 ms** | 7.12 ms | 18.06 ms | 6.96 ms | **2.54x** | **0.98x** |
| Repeated cutouts (50x 100x100) | CPU | **786.0 μs** | 743.6 μs | 86.01 ms | 5.37 ms | **115.66x** | **7.23x** |
| Table read (100k rows, 8 cols, mixed) | CPU | **2.20 ms** | 2.12 ms | 31.85 ms | 10.44 ms | **15.00x** | **4.92x** |
| Varlen table read (100k rows, 3 cols) | CPU | **77.19 ms** | 76.72 ms | 491.92 ms | 111.95 ms | **6.41x** | **1.46x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Benchmark category summary

Aggregated wins across every domain and operation in the CANFAR CUDA exhaustive
(`exhaustive_cuda_20260719_144457`, 4,087 rows; see host scorecard for
deficit honesty — all lags listed, floors label noise vs significant).
Category ranges below are the last regenerated aggregation shape; for this
run’s absolute times prefer [Performance highlights](#performance-highlights)
and the full table.

### FITS image I/O

| Category | Cases | torchfits median | astropy median | fitsio median | Typical speedup vs astropy | Typical speedup vs fitsio |
|---|---:|---:|---:|---:|---:|---:|
| **1D** (float32/64, int8–int64, tiny–large) | 48 | 85 μs – 1.86 ms | 640 μs – 3.83 ms | 130 μs – 2.39 ms | **2.1–7.8×** | **1.3–2.3×** |
| **2D** (float32/64, int8–int64, uint16/32, tiny–large) | 52 | 101 μs – 10.59 ms | 660 μs – 24.0 ms | 135 μs – 11.24 ms | **2.3–7.7×** | **1.1–2.4×** |
| **3D** (float32/64, int8–int64, small–medium) | 36 | 103 μs – 3.20 ms | 689 μs – 6.12 ms | 140 μs – 4.16 ms | **1.9–7.5×** | **1.2–2.1×** |
| **Compressed** (gzip, hcompress, rice) | 6 | 9.06–30.64 ms | 27.77–40.35 ms | 9.43–29.45 ms | **1.2–4.3×** | **1.0–1.1×** |
| **Scaled** (BSCALE/BZERO, small–large) | 6 | 154 μs – 3.53 ms | 935 μs – 11.44 ms | 277 μs – 4.76 ms | **2.5–6.1×** | **1.3–1.8×** |
| **MEF** (multi-extension, small/medium) | 2 | 112–324 μs | 1.11–1.56 ms | 267–507 μs | **4.8–9.9×** | **1.6–2.4×** |
| **Multi-MEF** (10 extensions, cutouts + random reads) | 3 | 95 μs – 6.62 ms | 3.39–11.25 ms | 361 μs – 10.19 ms | **1.7–35.9×** | **1.1–3.8×** |
| **Repeated cutouts** (50× 100×100) | 1 | 4.68 ms | 75.36 ms | 4.94 ms | **16.7×** | **1.1×** |
| **Time series frames** (5 frames) | 5 | 148–160 μs | 750–763 μs | 272–278 μs | **4.7–5.1×** | **1.7–1.9×** |
| **Header read** (all fixture types) | ~55 | 88–109 μs | 615–1010 μs | 128–159 μs | **6.5–9.5×** | **1.4–1.5×** |

**GPU (CUDA) image reads** — 76 `read_full` cases:

| Category | torchfits median | astropy median | fitsio median | Typical speedup vs astropy | Typical speedup vs fitsio |
|---|---:|---:|---:|---:|---:|
| **1D** (tiny–large) | 27–818 μs | 382–1620 μs | 74–1330 μs | **2.0–14.9×** | **1.3–3.0×** |
| **2D** (tiny–large) | 28–3.51 ms | 382–17.8 ms | 74–5.50 ms | **2.1–19.9×** | **1.1–2.9×** |
| **3D** (tiny–medium) | 33–2.62 ms | 437–10.43 ms | 87–3.51 ms | **1.9–15.1×** | **1.3–2.6×** |
| **Scaled** | 54 μs – 1.68 ms | 674 μs – 11.61 ms | 169–4880 μs | **3.9–12.5×** | **1.6–3.1×** |
| **MEF + Multi-MEF** | 42–238 μs | 795–3220 μs | 134–419 μs | **6.1–78.4×** | **1.8–5.4×** |
| **Repeated cutouts (GPU)** | 6.30 ms | 82.38 ms | 6.35 ms | **14.1×** | **1.1×** |

### FITS table I/O

| Category | Cases | torchfits median | astropy median | fitsio median | Typical speedup vs astropy | Typical speedup vs fitsio |
|---|---:|---:|---:|---:|---:|---:|
| **read_full** (mixed/narrow/wide/varlen, 1k–100k rows) | 20 | 93–184 μs | 2.25–6.74 ms | 3.25–59.84 ms | **24–115×** | **34–628×** |
| **projection** (column subset) | 20 | 93–101 μs | 2.60–13.53 ms | 219 μs – 9.94 ms | **26–147×** | **2.3–91×** |
| **row_slice** (row range) | 20 | 94–103 μs | 2.69–14.18 ms | 308 μs – 15.70 ms | **28–147×** | **3.2–162×** |
| **predicate_filter** (WHERE clause) | 20 | 643 μs – 1.06 ms | 3.10–13.53 ms | 561 μs – 7.60 ms | **3.2–57×** | **0.44–25×** |
| **scan_count** (streaming) | 20 | 134–277 μs | 3.98–11.15 ms | 490 μs – 12.44 ms | **30–85×** | **4.6–55×** |

## Exhaustive Benchmark Results

<!-- BENCH_FULL_TABLE_BEGIN -->
The complete, un-cherrypicked list of all measured configurations. Empty cells mean that method was not run for the case (for example `torchfits_specialized` is only used for open-once / subset-reader paths). Domain `tensor` = IMAGE HDU payloads (1D–4D); `table` = binary/ASCII tables.

| Domain | Benchmark Case | Operation | Size | Device | mmap | torchfits | torchfits (specialized) | astropy (via torch) | fitsio (via torch) | cfitsio (direct) | Speedup vs Astropy | Speedup vs fitsio |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| tensor | compressed_gzip_1 | header_read | 1.29 MB | CPU | n/a | **—** | 128.0 μs | 1.05 ms | 146.7 μs | — | **8.21x** | **1.15x** |
| tensor | compressed_gzip_2 | header_read | 0.89 MB | CPU | n/a | **—** | 113.8 μs | 1.02 ms | 145.5 μs | — | **8.98x** | **1.28x** |
| tensor | compressed_hcompress_1 | header_read | 0.82 MB | CPU | n/a | **—** | 121.8 μs | 1.08 ms | 167.0 μs | — | **8.86x** | **1.37x** |
| tensor | compressed_rice_1 | cutout_100x100 | 0.90 MB | CPU | n/a | **869.9 μs** | 951.9 μs | 6.04 ms | 769.5 μs | — | **6.95x** | **0.88x** |
| tensor | compressed_rice_1 | header_read | 0.90 MB | CPU | n/a | **—** | 116.9 μs | 1.03 ms | 157.7 μs | — | **8.83x** | **1.35x** |
| tensor | large_float32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 55.5 μs | 214.5 μs | 52.3 μs | — | **3.86x** | **0.94x** |
| tensor | large_float32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 64.6 μs | 227.1 μs | 117.6 μs | — | **3.51x** | **1.82x** |
| tensor | large_float64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 64.3 μs | 221.3 μs | 68.1 μs | — | **3.44x** | **1.06x** |
| tensor | large_float64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 55.2 μs | 240.7 μs | 80.0 μs | — | **4.36x** | **1.45x** |
| tensor | large_int16_1d | header_read | 1.91 MB | CPU | n/a | **—** | 57.8 μs | 234.1 μs | 67.9 μs | — | **4.05x** | **1.17x** |
| tensor | large_int16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 85.1 μs | 383.3 μs | 75.1 μs | — | **4.50x** | **0.88x** |
| tensor | large_int32_1d | header_read | 3.82 MB | CPU | n/a | **—** | 57.5 μs | 226.5 μs | 60.1 μs | — | **3.94x** | **1.04x** |
| tensor | large_int32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 55.0 μs | 248.7 μs | 57.2 μs | — | **4.52x** | **1.04x** |
| tensor | large_int64_1d | header_read | 7.63 MB | CPU | n/a | **—** | 48.3 μs | 216.5 μs | 43.6 μs | — | **4.48x** | **0.90x** |
| tensor | large_int64_2d | header_read | 32.00 MB | CPU | n/a | **—** | 60.5 μs | 243.5 μs | 159.2 μs | — | **4.02x** | **2.63x** |
| tensor | large_int8_1d | header_read | 0.96 MB | CPU | n/a | **—** | 55.3 μs | 239.3 μs | 51.9 μs | — | **4.33x** | **0.94x** |
| tensor | large_int8_2d | header_read | 4.00 MB | CPU | n/a | **—** | 58.3 μs | 269.4 μs | 56.2 μs | — | **4.62x** | **0.96x** |
| tensor | large_uint16_2d | header_read | 8.00 MB | CPU | n/a | **—** | 60.2 μs | 261.0 μs | 55.3 μs | — | **4.33x** | **0.92x** |
| tensor | large_uint32_2d | header_read | 16.00 MB | CPU | n/a | **—** | 59.1 μs | 269.0 μs | 72.5 μs | — | **4.55x** | **1.23x** |
| tensor | medium_float32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 47.0 μs | 206.5 μs | 46.6 μs | — | **4.39x** | **0.99x** |
| tensor | medium_float32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 58.4 μs | 234.7 μs | 75.7 μs | — | **4.02x** | **1.30x** |
| tensor | medium_float32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 57.6 μs | 253.8 μs | 63.7 μs | — | **4.41x** | **1.11x** |
| tensor | medium_float64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 56.5 μs | 220.8 μs | 82.8 μs | — | **3.91x** | **1.47x** |
| tensor | medium_float64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 54.3 μs | 237.9 μs | 47.3 μs | — | **4.38x** | **0.87x** |
| tensor | medium_float64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 60.0 μs | 274.7 μs | 56.6 μs | — | **4.57x** | **0.94x** |
| tensor | medium_int16_1d | header_read | 0.20 MB | CPU | n/a | **—** | 53.2 μs | 212.3 μs | 47.0 μs | — | **3.99x** | **0.88x** |
| tensor | medium_int16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 55.4 μs | 224.9 μs | 52.5 μs | — | **4.06x** | **0.95x** |
| tensor | medium_int16_3d | header_read | 3.13 MB | CPU | n/a | **—** | 60.5 μs | 243.8 μs | 51.0 μs | — | **4.03x** | **0.84x** |
| tensor | medium_int32_1d | header_read | 0.38 MB | CPU | n/a | **—** | 52.8 μs | 220.9 μs | 53.2 μs | — | **4.18x** | **1.01x** |
| tensor | medium_int32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 62.8 μs | 253.1 μs | 72.0 μs | — | **4.03x** | **1.15x** |
| tensor | medium_int32_3d | header_read | 6.25 MB | CPU | n/a | **—** | 72.7 μs | 311.9 μs | 66.4 μs | — | **4.29x** | **0.91x** |
| tensor | medium_int64_1d | header_read | 0.77 MB | CPU | n/a | **—** | 47.0 μs | 212.5 μs | 42.8 μs | — | **4.53x** | **0.91x** |
| tensor | medium_int64_2d | header_read | 8.00 MB | CPU | n/a | **—** | 55.7 μs | 228.9 μs | 46.2 μs | — | **4.11x** | **0.83x** |
| tensor | medium_int64_3d | header_read | 12.51 MB | CPU | n/a | **—** | 64.1 μs | 252.6 μs | 72.8 μs | — | **3.94x** | **1.14x** |
| tensor | medium_int8_1d | header_read | 0.10 MB | CPU | n/a | **—** | 60.5 μs | 262.8 μs | 96.7 μs | — | **4.34x** | **1.60x** |
| tensor | medium_int8_2d | header_read | 1.01 MB | CPU | n/a | **—** | 247.0 μs | 1.44 ms | 95.7 μs | — | **5.84x** | **0.39x** |
| tensor | medium_int8_3d | header_read | 1.57 MB | CPU | n/a | **—** | 62.9 μs | 317.0 μs | 62.6 μs | — | **5.04x** | **1.00x** |
| tensor | medium_uint16_2d | header_read | 2.01 MB | CPU | n/a | **—** | 61.3 μs | 269.0 μs | 140.3 μs | — | **4.39x** | **2.29x** |
| tensor | medium_uint32_2d | header_read | 4.00 MB | CPU | n/a | **—** | 62.4 μs | 265.0 μs | 72.0 μs | — | **4.25x** | **1.15x** |
| tensor | mef_medium | header_read | 7.02 MB | CPU | n/a | **—** | 59.2 μs | 406.8 μs | 63.6 μs | — | **6.87x** | **1.07x** |
| tensor | mef_small | header_read | 0.45 MB | CPU | n/a | **—** | 67.2 μs | 396.9 μs | 63.4 μs | — | **5.90x** | **0.94x** |
| tensor | multi_mef_10ext | cutout_100x100 | 2.68 MB | CPU | n/a | **150.3 μs** | 92.5 μs | 2.50 ms | 273.4 μs | — | **27.08x** | **2.96x** |
| tensor | multi_mef_10ext | header_read | 2.68 MB | CPU | n/a | **—** | 62.3 μs | 389.8 μs | 61.9 μs | — | **6.26x** | **0.99x** |
| tensor | multi_mef_10ext | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **—** | — | 7.27 ms | 5.72 ms | — | **—** | **—** |
| tensor | repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **786.0 μs** | 743.6 μs | 86.01 ms | 5.37 ms | — | **115.66x** | **7.23x** |
| tensor | scaled_large | header_read | 8.00 MB | CPU | n/a | **—** | 60.4 μs | 274.1 μs | 123.5 μs | — | **4.54x** | **2.04x** |
| tensor | scaled_medium | header_read | 2.01 MB | CPU | n/a | **—** | 52.4 μs | 263.7 μs | 51.0 μs | — | **5.03x** | **0.97x** |
| tensor | scaled_small | header_read | 0.13 MB | CPU | n/a | **—** | 75.3 μs | 316.8 μs | 68.0 μs | — | **4.20x** | **0.90x** |
| tensor | small_float32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 60.6 μs | 223.9 μs | 159.7 μs | — | **3.69x** | **2.63x** |
| tensor | small_float32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 55.9 μs | 239.1 μs | 55.5 μs | — | **4.28x** | **0.99x** |
| tensor | small_float32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 53.8 μs | 240.3 μs | 51.7 μs | — | **4.46x** | **0.96x** |
| tensor | small_float64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 61.2 μs | 225.2 μs | 47.3 μs | — | **3.68x** | **0.77x** |
| tensor | small_float64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 55.8 μs | 229.0 μs | 60.5 μs | — | **4.10x** | **1.08x** |
| tensor | small_float64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 52.3 μs | 247.6 μs | 53.3 μs | — | **4.73x** | **1.02x** |
| tensor | small_int16_1d | header_read | 22.5 KB | CPU | n/a | **—** | 48.5 μs | 213.1 μs | 45.7 μs | — | **4.39x** | **0.94x** |
| tensor | small_int16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 52.5 μs | 239.8 μs | 48.8 μs | — | **4.56x** | **0.93x** |
| tensor | small_int16_3d | header_read | 0.32 MB | CPU | n/a | **—** | 52.7 μs | 245.6 μs | 49.8 μs | — | **4.66x** | **0.95x** |
| tensor | small_int32_1d | header_read | 42.2 KB | CPU | n/a | **—** | 51.9 μs | 209.1 μs | 115.0 μs | — | **4.03x** | **2.22x** |
| tensor | small_int32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 48.9 μs | 220.4 μs | 47.8 μs | — | **4.51x** | **0.98x** |
| tensor | small_int32_3d | header_read | 0.63 MB | CPU | n/a | **—** | 50.8 μs | 240.6 μs | 51.5 μs | — | **4.74x** | **1.01x** |
| tensor | small_int64_1d | header_read | 0.08 MB | CPU | n/a | **—** | 54.8 μs | 220.0 μs | 57.1 μs | — | **4.02x** | **1.04x** |
| tensor | small_int64_2d | header_read | 0.51 MB | CPU | n/a | **—** | 54.0 μs | 252.0 μs | 57.0 μs | — | **4.66x** | **1.05x** |
| tensor | small_int64_3d | header_read | 1.26 MB | CPU | n/a | **—** | 51.3 μs | 239.4 μs | 49.1 μs | — | **4.67x** | **0.96x** |
| tensor | small_int8_1d | header_read | 14.1 KB | CPU | n/a | **—** | 60.2 μs | 263.8 μs | 70.6 μs | — | **4.39x** | **1.17x** |
| tensor | small_int8_2d | header_read | 0.07 MB | CPU | n/a | **—** | 59.5 μs | 281.3 μs | 57.8 μs | — | **4.72x** | **0.97x** |
| tensor | small_int8_3d | header_read | 0.16 MB | CPU | n/a | **—** | 59.9 μs | 290.2 μs | 61.3 μs | — | **4.84x** | **1.02x** |
| tensor | small_uint16_2d | header_read | 0.13 MB | CPU | n/a | **—** | 55.3 μs | 255.6 μs | 53.1 μs | — | **4.63x** | **0.96x** |
| tensor | small_uint32_2d | header_read | 0.26 MB | CPU | n/a | **—** | 52.2 μs | 253.4 μs | 52.0 μs | — | **4.85x** | **0.99x** |
| tensor | timeseries_frame_000 | header_read | 0.26 MB | CPU | n/a | **—** | 51.2 μs | 224.2 μs | 46.2 μs | — | **4.38x** | **0.90x** |
| tensor | timeseries_frame_001 | header_read | 0.26 MB | CPU | n/a | **—** | 57.1 μs | 244.2 μs | 55.1 μs | — | **4.27x** | **0.96x** |
| tensor | timeseries_frame_002 | header_read | 0.26 MB | CPU | n/a | **—** | 53.9 μs | 235.0 μs | 144.5 μs | — | **4.36x** | **2.68x** |
| tensor | timeseries_frame_003 | header_read | 0.26 MB | CPU | n/a | **—** | 63.0 μs | 262.1 μs | 107.1 μs | — | **4.16x** | **1.70x** |
| tensor | timeseries_frame_004 | header_read | 0.26 MB | CPU | n/a | **—** | 62.4 μs | 244.0 μs | 60.6 μs | — | **3.91x** | **0.97x** |
| tensor | tiny_float32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 57.3 μs | 222.0 μs | 55.9 μs | — | **3.87x** | **0.98x** |
| tensor | tiny_float32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 68.2 μs | 253.2 μs | 78.5 μs | — | **3.71x** | **1.15x** |
| tensor | tiny_float32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 61.5 μs | 239.8 μs | 57.2 μs | — | **3.90x** | **0.93x** |
| tensor | tiny_float64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 54.8 μs | 245.2 μs | 121.7 μs | — | **4.47x** | **2.22x** |
| tensor | tiny_float64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 62.5 μs | 235.5 μs | 53.4 μs | — | **3.77x** | **0.86x** |
| tensor | tiny_float64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 57.2 μs | 246.0 μs | 75.7 μs | — | **4.30x** | **1.32x** |
| tensor | tiny_int16_1d | header_read | 5.6 KB | CPU | n/a | **—** | 52.9 μs | 208.8 μs | 121.5 μs | — | **3.95x** | **2.30x** |
| tensor | tiny_int16_2d | header_read | 11.2 KB | CPU | n/a | **—** | 52.5 μs | 221.0 μs | 50.4 μs | — | **4.21x** | **0.96x** |
| tensor | tiny_int16_3d | header_read | 14.1 KB | CPU | n/a | **—** | 52.8 μs | 255.3 μs | 51.3 μs | — | **4.83x** | **0.97x** |
| tensor | tiny_int32_1d | header_read | 8.4 KB | CPU | n/a | **—** | 55.0 μs | 217.6 μs | 48.0 μs | — | **3.96x** | **0.87x** |
| tensor | tiny_int32_2d | header_read | 19.7 KB | CPU | n/a | **—** | 58.2 μs | 229.0 μs | 68.6 μs | — | **3.94x** | **1.18x** |
| tensor | tiny_int32_3d | header_read | 25.3 KB | CPU | n/a | **—** | 53.2 μs | 244.9 μs | 49.6 μs | — | **4.60x** | **0.93x** |
| tensor | tiny_int64_1d | header_read | 11.2 KB | CPU | n/a | **—** | 54.1 μs | 226.1 μs | 133.7 μs | — | **4.18x** | **2.47x** |
| tensor | tiny_int64_2d | header_read | 36.6 KB | CPU | n/a | **—** | 60.5 μs | 245.4 μs | 57.2 μs | — | **4.06x** | **0.95x** |
| tensor | tiny_int64_3d | header_read | 45.0 KB | CPU | n/a | **—** | 58.5 μs | 253.8 μs | 59.7 μs | — | **4.34x** | **1.02x** |
| tensor | tiny_int8_1d | header_read | 5.6 KB | CPU | n/a | **—** | 68.7 μs | 353.0 μs | 84.0 μs | — | **5.14x** | **1.22x** |
| tensor | tiny_int8_2d | header_read | 8.4 KB | CPU | n/a | **—** | 66.5 μs | 270.5 μs | 64.5 μs | — | **4.07x** | **0.97x** |
| tensor | tiny_int8_3d | header_read | 8.4 KB | CPU | n/a | **—** | 57.0 μs | 281.8 μs | 54.0 μs | — | **4.94x** | **0.95x** |
| tensor | write_compress_hcompress_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **—** | — | 70.41 ms | — | — | **—** | **—** |
| tensor | write_compress_rice_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **35.36 ms** | — | 62.24 ms | — | — | **1.76x** | **—** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | CPU | off | **10.81 ms** | 10.80 ms | 25.01 ms | 14.48 ms | — | **2.32x** | **1.34x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | CPU | off | **11.50 ms** | 11.49 ms | 40.69 ms | 14.22 ms | — | **3.54x** | **1.24x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | CPU | off | **21.62 ms** | 21.55 ms | 25.98 ms | 21.10 ms | — | **1.21x** | **0.98x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | CPU | off | **6.50 ms** | 6.48 ms | 17.60 ms | 6.51 ms | — | **2.72x** | **1.00x** |
| tensor | large_float32_1d | read_full | 3.82 MB | CPU | off | **513.9 μs** | 464.5 μs | 1.31 ms | 478.2 μs | — | **2.82x** | **1.03x** |
| tensor | large_float32_2d | read_full | 16.00 MB | CPU | off | **2.52 ms** | 2.61 ms | 4.77 ms | 2.44 ms | — | **1.89x** | **0.97x** |
| tensor | large_float64_1d | read_full | 7.63 MB | CPU | off | **1.23 ms** | 1.27 ms | 2.23 ms | 1.38 ms | — | **1.82x** | **1.13x** |
| tensor | large_float64_2d | read_full | 32.00 MB | CPU | off | **5.70 ms** | 5.67 ms | 9.43 ms | 5.82 ms | — | **1.66x** | **1.03x** |
| tensor | large_int16_1d | read_full | 1.91 MB | CPU | off | **279.8 μs** | 209.4 μs | 665.2 μs | 236.2 μs | — | **3.18x** | **1.13x** |
| tensor | large_int16_2d | read_full | 8.00 MB | CPU | off | **1.10 ms** | 1.04 ms | 2.28 ms | 1.14 ms | — | **2.20x** | **1.10x** |
| tensor | large_int32_1d | read_full | 3.82 MB | CPU | off | **491.4 μs** | 539.0 μs | 1.16 ms | 487.1 μs | — | **2.36x** | **0.99x** |
| tensor | large_int32_2d | read_full | 16.00 MB | CPU | off | **2.25 ms** | 2.18 ms | 4.65 ms | 2.30 ms | — | **2.13x** | **1.05x** |
| tensor | large_int64_1d | read_full | 7.63 MB | CPU | off | **1.30 ms** | 1.22 ms | 2.20 ms | 1.30 ms | — | **1.80x** | **1.06x** |
| tensor | large_int64_2d | read_full | 32.00 MB | CPU | off | **5.57 ms** | 5.80 ms | 9.50 ms | 5.86 ms | — | **1.71x** | **1.05x** |
| tensor | large_int8_1d | read_full | 0.96 MB | CPU | off | **133.2 μs** | 140.0 μs | 347.5 μs | 520.1 μs | — | **2.61x** | **3.90x** |
| tensor | large_int8_2d | read_full | 4.00 MB | CPU | off | **605.8 μs** | 654.2 μs | 1.18 ms | 2.19 ms | — | **1.95x** | **3.61x** |
| tensor | large_uint16_2d | read_full | 8.00 MB | CPU | off | **2.71 ms** | 2.67 ms | 4.35 ms | 2.82 ms | — | **1.63x** | **1.05x** |
| tensor | large_uint32_2d | read_full | 16.00 MB | CPU | off | **3.89 ms** | 3.87 ms | 7.00 ms | 4.01 ms | — | **1.81x** | **1.04x** |
| tensor | medium_float32_1d | read_full | 0.38 MB | CPU | off | **65.1 μs** | 83.5 μs | 263.5 μs | 74.7 μs | — | **4.05x** | **1.15x** |
| tensor | medium_float32_2d | read_full | 4.00 MB | CPU | off | **492.3 μs** | 562.6 μs | 1.23 ms | 544.8 μs | — | **2.50x** | **1.11x** |
| tensor | medium_float32_3d | read_full | 6.25 MB | CPU | off | **780.2 μs** | 835.0 μs | 1.91 ms | 893.3 μs | — | **2.45x** | **1.15x** |
| tensor | medium_float64_1d | read_full | 0.77 MB | CPU | off | **116.3 μs** | 131.8 μs | 329.2 μs | 117.7 μs | — | **2.83x** | **1.01x** |
| tensor | medium_float64_2d | read_full | 8.00 MB | CPU | off | **1.38 ms** | 1.41 ms | 2.21 ms | 1.33 ms | — | **1.60x** | **0.97x** |
| tensor | medium_float64_3d | read_full | 12.51 MB | CPU | off | **2.05 ms** | 1.97 ms | 3.11 ms | 2.14 ms | — | **1.58x** | **1.09x** |
| tensor | medium_int16_1d | read_full | 0.20 MB | CPU | off | **60.2 μs** | 69.8 μs | 258.1 μs | 69.0 μs | — | **4.29x** | **1.15x** |
| tensor | medium_int16_2d | read_full | 2.01 MB | CPU | off | **218.6 μs** | 220.5 μs | 768.5 μs | 246.8 μs | — | **3.52x** | **1.13x** |
| tensor | medium_int16_3d | read_full | 3.13 MB | CPU | off | **437.0 μs** | 415.4 μs | 943.2 μs | 448.6 μs | — | **2.27x** | **1.08x** |
| tensor | medium_int32_1d | read_full | 0.38 MB | CPU | off | **76.8 μs** | 68.2 μs | 259.3 μs | 80.6 μs | — | **3.80x** | **1.18x** |
| tensor | medium_int32_2d | read_full | 4.00 MB | CPU | off | **479.6 μs** | 561.5 μs | 1.12 ms | 510.2 μs | — | **2.34x** | **1.06x** |
| tensor | medium_int32_3d | read_full | 6.25 MB | CPU | off | **810.2 μs** | 873.8 μs | 1.81 ms | 845.0 μs | — | **2.24x** | **1.04x** |
| tensor | medium_int64_1d | read_full | 0.77 MB | CPU | off | **114.2 μs** | 120.8 μs | 313.1 μs | 130.2 μs | — | **2.74x** | **1.14x** |
| tensor | medium_int64_2d | read_full | 8.00 MB | CPU | off | **1.37 ms** | 1.28 ms | 2.23 ms | 1.41 ms | — | **1.74x** | **1.10x** |
| tensor | medium_int64_3d | read_full | 12.51 MB | CPU | off | **2.03 ms** | 2.19 ms | 3.08 ms | 2.13 ms | — | **1.52x** | **1.05x** |
| tensor | medium_int8_1d | read_full | 0.10 MB | CPU | off | **79.3 μs** | 136.3 μs | 277.0 μs | 107.0 μs | — | **3.49x** | **1.35x** |
| tensor | medium_int8_2d | read_full | 1.01 MB | CPU | off | **127.3 μs** | 137.0 μs | 390.0 μs | 537.9 μs | — | **3.06x** | **4.23x** |
| tensor | medium_int8_3d | read_full | 1.57 MB | CPU | off | **283.6 μs** | 291.1 μs | 700.5 μs | 853.9 μs | — | **2.47x** | **3.01x** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | CPU | off | **643.0 μs** | 644.4 μs | 1.01 ms | 619.9 μs | — | **1.57x** | **0.96x** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | CPU | off | **789.0 μs** | 876.9 μs | 1.65 ms | 859.7 μs | — | **2.09x** | **1.09x** |
| tensor | mef_medium | read_full | 7.02 MB | CPU | off | **164.0 μs** | 152.2 μs | 585.4 μs | 562.3 μs | — | **3.85x** | **3.70x** |
| tensor | mef_small | read_full | 0.45 MB | CPU | off | **175.8 μs** | 79.4 μs | 408.4 μs | 170.0 μs | — | **5.14x** | **2.14x** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | CPU | off | **74.4 μs** | 152.3 μs | 410.5 μs | 250.7 μs | — | **5.52x** | **3.37x** |
| tensor | scaled_large | read_full | 8.00 MB | CPU | off | **2.28 ms** | 2.28 ms | 5.45 ms | 2.51 ms | — | **2.39x** | **1.10x** |
| tensor | scaled_medium | read_full | 2.01 MB | CPU | off | **535.7 μs** | 509.3 μs | 1.56 ms | 585.4 μs | — | **3.06x** | **1.15x** |
| tensor | scaled_small | read_full | 0.13 MB | CPU | off | **141.3 μs** | 81.2 μs | 332.2 μs | 69.3 μs | — | **4.09x** | **0.85x** |
| tensor | small_float32_1d | read_full | 42.2 KB | CPU | off | **131.5 μs** | 131.6 μs | 210.0 μs | 146.6 μs | — | **1.60x** | **1.12x** |
| tensor | small_float32_2d | read_full | 0.26 MB | CPU | off | **71.9 μs** | 80.1 μs | 267.1 μs | 68.5 μs | — | **3.71x** | **0.95x** |
| tensor | small_float32_3d | read_full | 0.63 MB | CPU | off | **75.0 μs** | 95.5 μs | 346.1 μs | 91.3 μs | — | **4.61x** | **1.22x** |
| tensor | small_float64_1d | read_full | 0.08 MB | CPU | off | **138.1 μs** | 55.6 μs | 215.2 μs | 66.6 μs | — | **3.87x** | **1.20x** |
| tensor | small_float64_2d | read_full | 0.51 MB | CPU | off | **82.9 μs** | 87.7 μs | 267.8 μs | 98.2 μs | — | **3.23x** | **1.18x** |
| tensor | small_float64_3d | read_full | 1.26 MB | CPU | off | **162.7 μs** | 152.5 μs | 507.7 μs | 165.4 μs | — | **3.33x** | **1.08x** |
| tensor | small_int16_1d | read_full | 22.5 KB | CPU | off | **43.2 μs** | 68.5 μs | 215.4 μs | 50.8 μs | — | **4.99x** | **1.18x** |
| tensor | small_int16_2d | read_full | 0.13 MB | CPU | off | **57.2 μs** | 83.7 μs | 244.9 μs | 55.1 μs | — | **4.28x** | **0.96x** |
| tensor | small_int16_3d | read_full | 0.32 MB | CPU | off | **73.1 μs** | 72.3 μs | 283.1 μs | 68.7 μs | — | **3.92x** | **0.95x** |
| tensor | small_int32_1d | read_full | 42.2 KB | CPU | off | **43.8 μs** | 45.0 μs | 208.9 μs | 57.8 μs | — | **4.78x** | **1.32x** |
| tensor | small_int32_2d | read_full | 0.26 MB | CPU | off | **62.7 μs** | 140.0 μs | 247.0 μs | 72.4 μs | — | **3.94x** | **1.15x** |
| tensor | small_int32_3d | read_full | 0.63 MB | CPU | off | **76.6 μs** | 89.5 μs | 317.4 μs | 90.5 μs | — | **4.14x** | **1.18x** |
| tensor | small_int64_1d | read_full | 0.08 MB | CPU | off | **57.5 μs** | 133.1 μs | 297.6 μs | 62.7 μs | — | **5.18x** | **1.09x** |
| tensor | small_int64_2d | read_full | 0.51 MB | CPU | off | **95.6 μs** | 89.5 μs | 300.4 μs | 98.7 μs | — | **3.36x** | **1.10x** |
| tensor | small_int64_3d | read_full | 1.26 MB | CPU | off | **167.9 μs** | 172.0 μs | 432.4 μs | 175.7 μs | — | **2.58x** | **1.05x** |
| tensor | small_int8_1d | read_full | 14.1 KB | CPU | off | **44.0 μs** | 41.6 μs | 267.7 μs | 56.1 μs | — | **6.43x** | **1.35x** |
| tensor | small_int8_2d | read_full | 0.07 MB | CPU | off | **152.1 μs** | 60.9 μs | 370.2 μs | 100.3 μs | — | **6.08x** | **1.65x** |
| tensor | small_int8_3d | read_full | 0.16 MB | CPU | off | **67.0 μs** | 58.1 μs | 370.8 μs | 122.1 μs | — | **6.38x** | **2.10x** |
| tensor | small_uint16_2d | read_full | 0.13 MB | CPU | off | **81.3 μs** | 82.2 μs | 391.2 μs | 86.3 μs | — | **4.81x** | **1.06x** |
| tensor | small_uint32_2d | read_full | 0.26 MB | CPU | off | **119.0 μs** | 96.9 μs | 291.0 μs | 89.4 μs | — | **3.00x** | **0.92x** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | CPU | off | **73.0 μs** | 136.6 μs | 295.7 μs | 65.0 μs | — | **4.05x** | **0.89x** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | CPU | off | **66.9 μs** | 66.3 μs | 347.9 μs | 67.0 μs | — | **5.25x** | **1.01x** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | CPU | off | **63.7 μs** | 73.3 μs | 280.0 μs | 69.7 μs | — | **4.40x** | **1.09x** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | CPU | off | **61.7 μs** | 65.6 μs | 246.4 μs | 71.0 μs | — | **3.99x** | **1.15x** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | CPU | off | **202.8 μs** | 63.7 μs | 274.4 μs | 64.5 μs | — | **4.31x** | **1.01x** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | CPU | off | **45.5 μs** | 45.0 μs | 272.2 μs | 58.7 μs | — | **6.04x** | **1.30x** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | CPU | off | **39.0 μs** | 47.8 μs | 228.5 μs | 51.7 μs | — | **5.86x** | **1.33x** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | CPU | off | **43.1 μs** | 39.5 μs | 227.9 μs | 52.9 μs | — | **5.77x** | **1.34x** |
| tensor | tiny_float64_1d | read_full | 11.2 KB | CPU | off | **48.2 μs** | 40.3 μs | 231.3 μs | 61.4 μs | — | **5.75x** | **1.52x** |
| tensor | tiny_float64_2d | read_full | 36.6 KB | CPU | off | **51.8 μs** | 53.9 μs | 217.7 μs | 56.0 μs | — | **4.20x** | **1.08x** |
| tensor | tiny_float64_3d | read_full | 45.0 KB | CPU | off | **62.5 μs** | 134.3 μs | 304.4 μs | 73.7 μs | — | **4.87x** | **1.18x** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | CPU | off | **53.5 μs** | 42.1 μs | 208.2 μs | 46.2 μs | — | **4.95x** | **1.10x** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | CPU | off | **39.0 μs** | 54.9 μs | 218.2 μs | 53.9 μs | — | **5.60x** | **1.38x** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | CPU | off | **45.2 μs** | 40.7 μs | 220.8 μs | 50.3 μs | — | **5.42x** | **1.24x** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | CPU | off | **40.9 μs** | 53.6 μs | 216.6 μs | 54.7 μs | — | **5.30x** | **1.34x** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | CPU | off | **43.8 μs** | 43.5 μs | 221.4 μs | 55.7 μs | — | **5.09x** | **1.28x** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | CPU | off | **50.2 μs** | 44.1 μs | 2.86 ms | 152.0 μs | — | **64.90x** | **3.45x** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | CPU | off | **42.8 μs** | 39.2 μs | 251.2 μs | 64.6 μs | — | **6.41x** | **1.65x** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | CPU | off | **44.5 μs** | 56.1 μs | 228.3 μs | 62.2 μs | — | **5.13x** | **1.40x** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | CPU | off | **136.2 μs** | 95.0 μs | 233.2 μs | 145.8 μs | — | **2.45x** | **1.53x** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | CPU | off | **40.2 μs** | 41.2 μs | 265.4 μs | 55.7 μs | — | **6.59x** | **1.38x** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | CPU | off | **39.0 μs** | 44.3 μs | 286.1 μs | 52.0 μs | — | **7.34x** | **1.33x** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | CPU | off | **42.2 μs** | 48.5 μs | 281.5 μs | 52.3 μs | — | **6.66x** | **1.24x** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | CPU | on | **29.41 ms** | 28.83 ms | 67.23 ms | 33.44 ms | — | **2.33x** | **1.16x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | CPU | on | **32.53 ms** | 32.11 ms | 119.60 ms | 40.08 ms | — | **3.72x** | **1.25x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | CPU | on | **55.06 ms** | 54.13 ms | 77.89 ms | 58.54 ms | — | **1.44x** | **1.08x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | CPU | on | **18.40 ms** | 18.29 ms | 49.14 ms | 18.28 ms | — | **2.69x** | **1.00x** |
| tensor | large_float32_1d | read_full | 3.82 MB | CPU | on | **1.09 ms** | 1.19 ms | 3.06 ms | — | — | **2.82x** | **—** |
| tensor | large_float32_2d | read_full | 16.00 MB | CPU | on | **5.43 ms** | 5.31 ms | 9.32 ms | — | — | **1.75x** | **—** |
| tensor | large_float64_1d | read_full | 7.63 MB | CPU | on | **2.97 ms** | 2.95 ms | 4.21 ms | — | — | **1.43x** | **—** |
| tensor | large_float64_2d | read_full | 32.00 MB | CPU | on | **11.99 ms** | 12.15 ms | 21.85 ms | — | — | **1.82x** | **—** |
| tensor | large_int16_1d | read_full | 1.91 MB | CPU | on | **721.5 μs** | 727.0 μs | 1.90 ms | — | — | **2.63x** | **—** |
| tensor | large_int16_2d | read_full | 8.00 MB | CPU | on | **2.61 ms** | 2.40 ms | 4.58 ms | — | — | **1.91x** | **—** |
| tensor | large_int32_1d | read_full | 3.82 MB | CPU | on | **1.28 ms** | 1.28 ms | 2.41 ms | — | — | **1.89x** | **—** |
| tensor | large_int32_2d | read_full | 16.00 MB | CPU | on | **6.07 ms** | 6.20 ms | 8.29 ms | — | — | **1.37x** | **—** |
| tensor | large_int64_1d | read_full | 7.63 MB | CPU | on | **2.32 ms** | 2.38 ms | 4.35 ms | — | — | **1.87x** | **—** |
| tensor | large_int64_2d | read_full | 32.00 MB | CPU | on | **10.27 ms** | 10.47 ms | 23.71 ms | — | — | **2.31x** | **—** |
| tensor | large_int8_1d | read_full | 0.96 MB | CPU | on | **353.3 μs** | 382.5 μs | — | — | — | **—** | **—** |
| tensor | large_int8_2d | read_full | 4.00 MB | CPU | on | **1.68 ms** | 1.81 ms | — | — | — | **—** | **—** |
| tensor | large_uint16_2d | read_full | 8.00 MB | CPU | on | **6.93 ms** | 6.70 ms | — | — | — | **—** | **—** |
| tensor | large_uint32_2d | read_full | 16.00 MB | CPU | on | **10.00 ms** | 9.72 ms | — | — | — | **—** | **—** |
| tensor | medium_float32_1d | read_full | 0.38 MB | CPU | on | **263.0 μs** | 245.7 μs | 1.60 ms | — | — | **6.53x** | **—** |
| tensor | medium_float32_2d | read_full | 4.00 MB | CPU | on | **1.14 ms** | 1.17 ms | 2.58 ms | — | — | **2.27x** | **—** |
| tensor | medium_float32_3d | read_full | 6.25 MB | CPU | on | **2.48 ms** | 2.61 ms | 3.70 ms | — | — | **1.49x** | **—** |
| tensor | medium_float64_1d | read_full | 0.77 MB | CPU | on | **342.1 μs** | 302.0 μs | 894.8 μs | — | — | **2.96x** | **—** |
| tensor | medium_float64_2d | read_full | 8.00 MB | CPU | on | **2.99 ms** | 2.96 ms | 4.47 ms | — | — | **1.51x** | **—** |
| tensor | medium_float64_3d | read_full | 12.51 MB | CPU | on | **4.64 ms** | 4.91 ms | 6.35 ms | — | — | **1.37x** | **—** |
| tensor | medium_int16_1d | read_full | 0.20 MB | CPU | on | **170.9 μs** | 166.5 μs | 697.1 μs | — | — | **4.19x** | **—** |
| tensor | medium_int16_2d | read_full | 2.01 MB | CPU | on | **722.0 μs** | 747.9 μs | 2.14 ms | — | — | **2.96x** | **—** |
| tensor | medium_int16_3d | read_full | 3.13 MB | CPU | on | **681.0 μs** | 660.9 μs | 1.71 ms | — | — | **2.59x** | **—** |
| tensor | medium_int32_1d | read_full | 0.38 MB | CPU | on | **172.3 μs** | 166.3 μs | 495.6 μs | — | — | **2.98x** | **—** |
| tensor | medium_int32_2d | read_full | 4.00 MB | CPU | on | **1.10 ms** | 946.9 μs | 2.93 ms | — | — | **3.10x** | **—** |
| tensor | medium_int32_3d | read_full | 6.25 MB | CPU | on | **848.2 μs** | 840.4 μs | 1.52 ms | — | — | **1.80x** | **—** |
| tensor | medium_int64_1d | read_full | 0.77 MB | CPU | on | **112.3 μs** | 94.0 μs | 4.35 ms | — | — | **46.27x** | **—** |
| tensor | medium_int64_2d | read_full | 8.00 MB | CPU | on | **1.06 ms** | 1.00 ms | 1.82 ms | — | — | **1.82x** | **—** |
| tensor | medium_int64_3d | read_full | 12.51 MB | CPU | on | **1.67 ms** | 1.65 ms | 2.09 ms | — | — | **1.26x** | **—** |
| tensor | medium_int8_1d | read_full | 0.10 MB | CPU | on | **53.8 μs** | 70.4 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_2d | read_full | 1.01 MB | CPU | on | **170.6 μs** | 121.6 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_3d | read_full | 1.57 MB | CPU | on | **230.7 μs** | 243.9 μs | — | — | — | **—** | **—** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | CPU | on | **649.8 μs** | 637.3 μs | — | — | — | **—** | **—** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | CPU | on | **958.0 μs** | 894.9 μs | — | — | — | **—** | **—** |
| tensor | mef_medium | read_full | 7.02 MB | CPU | on | **143.8 μs** | 143.4 μs | — | — | — | **—** | **—** |
| tensor | mef_small | read_full | 0.45 MB | CPU | on | **69.2 μs** | 144.1 μs | — | — | — | **—** | **—** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | CPU | on | **134.5 μs** | 66.2 μs | — | — | — | **—** | **—** |
| tensor | scaled_large | read_full | 8.00 MB | CPU | on | **2.17 ms** | 2.29 ms | — | — | — | **—** | **—** |
| tensor | scaled_medium | read_full | 2.01 MB | CPU | on | **550.9 μs** | 534.4 μs | — | — | — | **—** | **—** |
| tensor | scaled_small | read_full | 0.13 MB | CPU | on | **77.3 μs** | 62.4 μs | — | — | — | **—** | **—** |
| tensor | small_float32_1d | read_full | 42.2 KB | CPU | on | **109.0 μs** | 103.8 μs | 206.0 μs | — | — | **1.99x** | **—** |
| tensor | small_float32_2d | read_full | 0.26 MB | CPU | on | **76.0 μs** | 74.5 μs | 280.4 μs | — | — | **3.77x** | **—** |
| tensor | small_float32_3d | read_full | 0.63 MB | CPU | on | **83.8 μs** | 97.8 μs | 292.3 μs | — | — | **3.49x** | **—** |
| tensor | small_float64_1d | read_full | 0.08 MB | CPU | on | **50.3 μs** | 48.7 μs | 206.2 μs | — | — | **4.23x** | **—** |
| tensor | small_float64_2d | read_full | 0.51 MB | CPU | on | **79.3 μs** | 93.0 μs | 272.5 μs | — | — | **3.44x** | **—** |
| tensor | small_float64_3d | read_full | 1.26 MB | CPU | on | **249.2 μs** | 164.9 μs | 387.4 μs | — | — | **2.35x** | **—** |
| tensor | small_int16_1d | read_full | 22.5 KB | CPU | on | **52.0 μs** | 46.8 μs | 189.1 μs | — | — | **4.04x** | **—** |
| tensor | small_int16_2d | read_full | 0.13 MB | CPU | on | **128.7 μs** | 126.7 μs | 214.5 μs | — | — | **1.69x** | **—** |
| tensor | small_int16_3d | read_full | 0.32 MB | CPU | on | **81.5 μs** | 88.1 μs | 267.0 μs | — | — | **3.27x** | **—** |
| tensor | small_int32_1d | read_full | 42.2 KB | CPU | on | **129.2 μs** | 138.9 μs | 222.1 μs | — | — | **1.72x** | **—** |
| tensor | small_int32_2d | read_full | 0.26 MB | CPU | on | **131.9 μs** | 68.0 μs | 327.6 μs | — | — | **4.82x** | **—** |
| tensor | small_int32_3d | read_full | 0.63 MB | CPU | on | **122.6 μs** | 102.5 μs | 290.2 μs | — | — | **2.83x** | **—** |
| tensor | small_int64_1d | read_full | 0.08 MB | CPU | on | **113.3 μs** | 115.9 μs | 226.5 μs | — | — | **2.00x** | **—** |
| tensor | small_int64_2d | read_full | 0.51 MB | CPU | on | **99.0 μs** | 78.2 μs | 261.3 μs | — | — | **3.34x** | **—** |
| tensor | small_int64_3d | read_full | 1.26 MB | CPU | on | **142.4 μs** | 167.8 μs | 401.2 μs | — | — | **2.82x** | **—** |
| tensor | small_int8_1d | read_full | 14.1 KB | CPU | on | **46.3 μs** | 47.3 μs | — | — | — | **—** | **—** |
| tensor | small_int8_2d | read_full | 0.07 MB | CPU | on | **63.0 μs** | 114.1 μs | — | — | — | **—** | **—** |
| tensor | small_int8_3d | read_full | 0.16 MB | CPU | on | **64.7 μs** | 53.5 μs | — | — | — | **—** | **—** |
| tensor | small_uint16_2d | read_full | 0.13 MB | CPU | on | **195.3 μs** | 170.4 μs | — | — | — | **—** | **—** |
| tensor | small_uint32_2d | read_full | 0.26 MB | CPU | on | **101.6 μs** | 110.8 μs | — | — | — | **—** | **—** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | CPU | on | **75.5 μs** | 87.7 μs | 266.3 μs | — | — | **3.53x** | **—** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | CPU | on | **78.0 μs** | 59.8 μs | 281.9 μs | — | — | **4.71x** | **—** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | CPU | on | **124.5 μs** | 66.1 μs | 300.9 μs | — | — | **4.55x** | **—** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | CPU | on | **89.8 μs** | 74.5 μs | 279.6 μs | — | — | **3.75x** | **—** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | CPU | on | **68.2 μs** | 81.5 μs | 301.6 μs | — | — | **4.42x** | **—** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | CPU | on | **40.2 μs** | 43.5 μs | 226.9 μs | — | — | **5.64x** | **—** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | CPU | on | **44.1 μs** | 50.6 μs | 223.8 μs | — | — | **5.08x** | **—** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | CPU | on | **42.7 μs** | 42.0 μs | 239.5 μs | — | — | **5.70x** | **—** |
| tensor | tiny_float64_1d | read_full | 11.2 KB | CPU | on | **43.0 μs** | 45.3 μs | 213.1 μs | — | — | **4.96x** | **—** |
| tensor | tiny_float64_2d | read_full | 36.6 KB | CPU | on | **45.9 μs** | 46.5 μs | 222.5 μs | — | — | **4.85x** | **—** |
| tensor | tiny_float64_3d | read_full | 45.0 KB | CPU | on | **90.3 μs** | 101.3 μs | 246.8 μs | — | — | **2.73x** | **—** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | CPU | on | **46.5 μs** | 59.4 μs | 203.5 μs | — | — | **4.38x** | **—** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | CPU | on | **69.7 μs** | 53.5 μs | 231.9 μs | — | — | **4.33x** | **—** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | CPU | on | **68.5 μs** | 51.7 μs | 247.4 μs | — | — | **4.78x** | **—** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | CPU | on | **49.4 μs** | 46.9 μs | 220.2 μs | — | — | **4.69x** | **—** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | CPU | on | **48.1 μs** | 49.2 μs | 220.4 μs | — | — | **4.58x** | **—** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | CPU | on | **41.6 μs** | 41.8 μs | 214.7 μs | — | — | **5.16x** | **—** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | CPU | on | **47.7 μs** | 51.6 μs | 194.2 μs | — | — | **4.07x** | **—** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | CPU | on | **51.7 μs** | 50.2 μs | 206.9 μs | — | — | **4.12x** | **—** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | CPU | on | **109.6 μs** | 89.5 μs | 213.6 μs | — | — | **2.39x** | **—** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | CPU | on | **38.8 μs** | 36.7 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | CPU | on | **39.0 μs** | 47.1 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | CPU | on | **56.5 μs** | 44.0 μs | — | — | — | **—** | **—** |
| tensor | compressed_rice_1 | cutout_100x100 | 0.90 MB | MPS | n/a | **964.8 μs** | 977.0 μs | 6.02 ms | 1.20 ms | — | **6.24x** | **1.25x** |
| tensor | multi_mef_10ext | cutout_100x100 | 2.68 MB | MPS | n/a | **249.6 μs** | 263.0 μs | 3.04 ms | 401.7 μs | — | **12.20x** | **1.61x** |
| tensor | repeated_cutouts_50x_100x100_gpu | repeated_cutouts_50x_100x100 | 4.00 MB | MPS | n/a | **7.61 ms** | 7.58 ms | 96.92 ms | 12.08 ms | — | **12.78x** | **1.59x** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | MPS | off | **12.40 ms** | 11.60 ms | 23.18 ms | 15.12 ms | — | **2.00x** | **1.30x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | MPS | off | **12.23 ms** | 12.17 ms | 41.33 ms | 14.86 ms | — | **3.40x** | **1.22x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | MPS | off | **22.08 ms** | 22.12 ms | 26.75 ms | 21.85 ms | — | **1.21x** | **0.99x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | MPS | off | **7.67 ms** | 7.12 ms | 18.51 ms | 7.13 ms | — | **2.60x** | **1.00x** |
| tensor | large_float32_1d | read_full | 3.82 MB | MPS | off | **886.7 μs** | 849.2 μs | 2.05 ms | 961.2 μs | — | **2.41x** | **1.13x** |
| tensor | large_float32_2d | read_full | 16.00 MB | MPS | off | **3.85 ms** | 5.27 ms | 8.16 ms | 3.67 ms | — | **2.12x** | **0.95x** |
| tensor | large_int16_1d | read_full | 1.91 MB | MPS | off | **553.8 μs** | 523.3 μs | 1.58 ms | 571.4 μs | — | **3.01x** | **1.09x** |
| tensor | large_int16_2d | read_full | 8.00 MB | MPS | off | **1.68 ms** | 1.75 ms | 3.19 ms | 1.75 ms | — | **1.90x** | **1.04x** |
| tensor | large_int32_1d | read_full | 3.82 MB | MPS | off | **952.5 μs** | 979.8 μs | 1.73 ms | 959.4 μs | — | **1.81x** | **1.01x** |
| tensor | large_int32_2d | read_full | 16.00 MB | MPS | off | **3.65 ms** | 3.33 ms | 5.85 ms | 3.48 ms | — | **1.76x** | **1.04x** |
| tensor | large_int64_1d | read_full | 7.63 MB | MPS | off | **2.03 ms** | 1.85 ms | 3.22 ms | 1.85 ms | — | **1.74x** | **1.00x** |
| tensor | large_int64_2d | read_full | 32.00 MB | MPS | off | **7.95 ms** | 8.31 ms | 12.02 ms | 8.08 ms | — | **1.51x** | **1.02x** |
| tensor | large_int8_1d | read_full | 0.96 MB | MPS | off | **446.2 μs** | 370.0 μs | 875.4 μs | 787.6 μs | — | **2.37x** | **2.13x** |
| tensor | large_int8_2d | read_full | 4.00 MB | MPS | off | **1.12 ms** | 1.27 ms | 2.15 ms | 2.79 ms | — | **1.91x** | **2.48x** |
| tensor | large_uint16_2d | read_full | 8.00 MB | MPS | off | **3.76 ms** | 3.48 ms | 5.02 ms | 3.64 ms | — | **1.44x** | **1.05x** |
| tensor | large_uint32_2d | read_full | 16.00 MB | MPS | off | **5.10 ms** | 5.07 ms | 7.98 ms | 5.22 ms | — | **1.57x** | **1.03x** |
| tensor | medium_float32_1d | read_full | 0.38 MB | MPS | off | **281.9 μs** | 276.1 μs | 646.0 μs | 287.7 μs | — | **2.34x** | **1.04x** |
| tensor | medium_float32_2d | read_full | 4.00 MB | MPS | off | **929.1 μs** | 969.2 μs | 2.08 ms | 934.1 μs | — | **2.24x** | **1.01x** |
| tensor | medium_float32_3d | read_full | 6.25 MB | MPS | off | **1.32 ms** | 1.28 ms | 2.54 ms | 1.32 ms | — | **1.98x** | **1.03x** |
| tensor | medium_int16_1d | read_full | 0.20 MB | MPS | off | **293.8 μs** | 360.2 μs | 565.8 μs | 299.8 μs | — | **1.93x** | **1.02x** |
| tensor | medium_int16_2d | read_full | 2.01 MB | MPS | off | **587.9 μs** | 616.2 μs | 1.39 ms | 643.5 μs | — | **2.37x** | **1.09x** |
| tensor | medium_int16_3d | read_full | 3.13 MB | MPS | off | **846.6 μs** | 776.1 μs | 1.58 ms | 901.0 μs | — | **2.03x** | **1.16x** |
| tensor | medium_int32_1d | read_full | 0.38 MB | MPS | off | **282.7 μs** | 292.0 μs | 719.2 μs | 328.9 μs | — | **2.54x** | **1.16x** |
| tensor | medium_int32_2d | read_full | 4.00 MB | MPS | off | **970.2 μs** | 926.2 μs | 1.88 ms | 956.8 μs | — | **2.03x** | **1.03x** |
| tensor | medium_int32_3d | read_full | 6.25 MB | MPS | off | **1.35 ms** | 1.35 ms | 2.60 ms | 1.42 ms | — | **1.93x** | **1.06x** |
| tensor | medium_int64_1d | read_full | 0.77 MB | MPS | off | **367.5 μs** | 364.4 μs | 777.5 μs | 429.7 μs | — | **2.13x** | **1.18x** |
| tensor | medium_int64_2d | read_full | 8.00 MB | MPS | off | **1.91 ms** | 1.87 ms | 3.09 ms | 1.89 ms | — | **1.65x** | **1.01x** |
| tensor | medium_int64_3d | read_full | 12.51 MB | MPS | off | **3.02 ms** | 2.93 ms | 3.94 ms | 2.94 ms | — | **1.34x** | **1.00x** |
| tensor | medium_int8_1d | read_full | 0.10 MB | MPS | off | **334.9 μs** | 375.8 μs | 511.1 μs | 285.0 μs | — | **1.53x** | **0.85x** |
| tensor | medium_int8_2d | read_full | 1.01 MB | MPS | off | **386.7 μs** | 441.2 μs | 833.4 μs | 837.7 μs | — | **2.15x** | **2.17x** |
| tensor | medium_int8_3d | read_full | 1.57 MB | MPS | off | **574.3 μs** | 586.8 μs | 999.2 μs | 1.19 ms | — | **1.74x** | **2.07x** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | MPS | off | **995.2 μs** | 1.03 ms | 1.76 ms | 966.2 μs | — | **1.77x** | **0.97x** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | MPS | off | **1.32 ms** | 1.35 ms | 2.35 ms | 1.33 ms | — | **1.79x** | **1.01x** |
| tensor | mef_medium | read_full | 7.02 MB | MPS | off | **408.5 μs** | 403.7 μs | 826.3 μs | 868.1 μs | — | **2.05x** | **2.15x** |
| tensor | mef_small | read_full | 0.45 MB | MPS | off | **267.9 μs** | 268.0 μs | 619.7 μs | 308.9 μs | — | **2.31x** | **1.15x** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | MPS | off | **249.4 μs** | 270.5 μs | 617.0 μs | 363.5 μs | — | **2.47x** | **1.46x** |
| tensor | scaled_large | read_full | 8.00 MB | MPS | off | **3.32 ms** | 3.32 ms | 7.17 ms | 3.08 ms | — | **2.16x** | **0.93x** |
| tensor | scaled_medium | read_full | 2.01 MB | MPS | off | **984.1 μs** | 987.2 μs | 2.34 ms | 953.2 μs | — | **2.38x** | **0.97x** |
| tensor | scaled_small | read_full | 0.13 MB | MPS | off | **320.8 μs** | 263.7 μs | 611.6 μs | 357.2 μs | — | **2.32x** | **1.35x** |
| tensor | small_float32_1d | read_full | 42.2 KB | MPS | off | **229.2 μs** | 451.0 μs | 408.4 μs | 236.9 μs | — | **1.78x** | **1.03x** |
| tensor | small_float32_2d | read_full | 0.26 MB | MPS | off | **264.6 μs** | 246.2 μs | 537.4 μs | 292.6 μs | — | **2.18x** | **1.19x** |
| tensor | small_float32_3d | read_full | 0.63 MB | MPS | off | **325.3 μs** | 259.6 μs | 751.1 μs | 321.6 μs | — | **2.89x** | **1.24x** |
| tensor | small_int16_1d | read_full | 22.5 KB | MPS | off | **221.4 μs** | 231.1 μs | 471.2 μs | 224.5 μs | — | **2.13x** | **1.01x** |
| tensor | small_int16_2d | read_full | 0.13 MB | MPS | off | **363.2 μs** | 300.1 μs | 932.0 μs | 284.1 μs | — | **3.11x** | **0.95x** |
| tensor | small_int16_3d | read_full | 0.32 MB | MPS | off | **278.3 μs** | 271.8 μs | 775.2 μs | 279.6 μs | — | **2.85x** | **1.03x** |
| tensor | small_int32_1d | read_full | 42.2 KB | MPS | off | **267.0 μs** | 277.6 μs | 523.5 μs | 267.5 μs | — | **1.96x** | **1.00x** |
| tensor | small_int32_2d | read_full | 0.26 MB | MPS | off | **340.3 μs** | 268.0 μs | 572.2 μs | 377.0 μs | — | **2.14x** | **1.41x** |
| tensor | small_int32_3d | read_full | 0.63 MB | MPS | off | **346.5 μs** | 389.4 μs | 638.6 μs | 348.2 μs | — | **1.84x** | **1.00x** |
| tensor | small_int64_1d | read_full | 0.08 MB | MPS | off | **259.5 μs** | 361.2 μs | 591.4 μs | 264.8 μs | — | **2.28x** | **1.02x** |
| tensor | small_int64_2d | read_full | 0.51 MB | MPS | off | **323.4 μs** | 308.8 μs | 655.5 μs | 338.9 μs | — | **2.12x** | **1.10x** |
| tensor | small_int64_3d | read_full | 1.26 MB | MPS | off | **437.0 μs** | 468.0 μs | 767.0 μs | 468.1 μs | — | **1.75x** | **1.07x** |
| tensor | small_int8_1d | read_full | 14.1 KB | MPS | off | **219.0 μs** | 250.3 μs | 520.3 μs | 262.3 μs | — | **2.38x** | **1.20x** |
| tensor | small_int8_2d | read_full | 0.07 MB | MPS | off | **255.9 μs** | 250.1 μs | 496.6 μs | 275.9 μs | — | **1.99x** | **1.10x** |
| tensor | small_int8_3d | read_full | 0.16 MB | MPS | off | **271.5 μs** | 257.8 μs | 547.1 μs | 355.9 μs | — | **2.12x** | **1.38x** |
| tensor | small_uint16_2d | read_full | 0.13 MB | MPS | off | **457.7 μs** | 414.2 μs | 658.5 μs | 314.0 μs | — | **1.59x** | **0.76x** |
| tensor | small_uint32_2d | read_full | 0.26 MB | MPS | off | **328.9 μs** | 320.5 μs | 773.3 μs | 400.4 μs | — | **2.41x** | **1.25x** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | MPS | off | **308.8 μs** | 317.5 μs | 556.3 μs | 312.7 μs | — | **1.80x** | **1.01x** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | MPS | off | **675.8 μs** | 290.6 μs | 565.5 μs | 358.1 μs | — | **1.95x** | **1.23x** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | MPS | off | **376.0 μs** | 309.4 μs | 601.4 μs | 295.4 μs | — | **1.94x** | **0.95x** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | MPS | off | **303.8 μs** | 282.9 μs | 656.9 μs | 288.2 μs | — | **2.32x** | **1.02x** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | MPS | off | **285.9 μs** | 299.7 μs | 526.7 μs | 351.3 μs | — | **1.84x** | **1.23x** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | MPS | off | **285.2 μs** | 234.5 μs | 506.2 μs | 276.7 μs | — | **2.16x** | **1.18x** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | MPS | off | **296.3 μs** | 392.8 μs | 437.6 μs | 265.4 μs | — | **1.48x** | **0.90x** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | MPS | off | **267.1 μs** | 361.3 μs | 465.0 μs | 249.7 μs | — | **1.74x** | **0.94x** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | MPS | off | **226.9 μs** | 248.7 μs | 444.1 μs | 237.9 μs | — | **1.96x** | **1.05x** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | MPS | off | **309.0 μs** | 243.9 μs | 440.1 μs | 249.4 μs | — | **1.80x** | **1.02x** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | MPS | off | **305.7 μs** | 299.5 μs | 467.6 μs | 247.5 μs | — | **1.56x** | **0.83x** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | MPS | off | **284.9 μs** | 229.3 μs | 548.6 μs | 236.9 μs | — | **2.39x** | **1.03x** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | MPS | off | **226.3 μs** | 233.5 μs | 443.5 μs | 254.5 μs | — | **1.96x** | **1.12x** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | MPS | off | **519.1 μs** | 246.4 μs | 468.9 μs | 418.2 μs | — | **1.90x** | **1.70x** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | MPS | off | **625.6 μs** | 254.5 μs | 492.4 μs | 525.0 μs | — | **1.93x** | **2.06x** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | MPS | off | **337.8 μs** | 347.3 μs | 431.0 μs | 241.4 μs | — | **1.28x** | **0.71x** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | MPS | off | **272.4 μs** | 243.6 μs | 533.8 μs | 253.2 μs | — | **2.19x** | **1.04x** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | MPS | off | **230.7 μs** | 229.9 μs | 531.2 μs | 278.5 μs | — | **2.31x** | **1.21x** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | MPS | off | **256.3 μs** | 239.1 μs | 520.5 μs | 351.7 μs | — | **2.18x** | **1.47x** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | MPS | off | **257.8 μs** | 234.7 μs | 511.9 μs | 301.4 μs | — | **2.18x** | **1.28x** |
| tensor | compressed_gzip_1 | read_full | 1.29 MB | MPS | on | **11.46 ms** | 11.53 ms | 23.25 ms | 13.98 ms | — | **2.03x** | **1.22x** |
| tensor | compressed_gzip_2 | read_full | 0.89 MB | MPS | on | **12.12 ms** | 12.18 ms | 41.21 ms | 14.66 ms | — | **3.40x** | **1.21x** |
| tensor | compressed_hcompress_1 | read_full | 0.82 MB | MPS | on | **22.07 ms** | 22.01 ms | 26.63 ms | 21.78 ms | — | **1.21x** | **0.99x** |
| tensor | compressed_rice_1 | read_full | 0.90 MB | MPS | on | **7.24 ms** | 7.10 ms | 18.49 ms | 6.95 ms | — | **2.61x** | **0.98x** |
| tensor | large_float32_1d | read_full | 3.82 MB | MPS | on | **909.7 μs** | 980.8 μs | 1.69 ms | — | — | **1.85x** | **—** |
| tensor | large_float32_2d | read_full | 16.00 MB | MPS | on | **3.29 ms** | 3.27 ms | 4.69 ms | — | — | **1.44x** | **—** |
| tensor | large_int16_1d | read_full | 1.91 MB | MPS | on | **612.0 μs** | 788.3 μs | 2.28 ms | — | — | **3.73x** | **—** |
| tensor | large_int16_2d | read_full | 8.00 MB | MPS | on | **1.97 ms** | 1.79 ms | 2.91 ms | — | — | **1.63x** | **—** |
| tensor | large_int32_1d | read_full | 3.82 MB | MPS | on | **920.9 μs** | 938.2 μs | 1.69 ms | — | — | **1.84x** | **—** |
| tensor | large_int32_2d | read_full | 16.00 MB | MPS | on | **3.42 ms** | 3.72 ms | 4.63 ms | — | — | **1.36x** | **—** |
| tensor | large_int64_1d | read_full | 7.63 MB | MPS | on | **1.57 ms** | 1.62 ms | 2.59 ms | — | — | **1.65x** | **—** |
| tensor | large_int64_2d | read_full | 32.00 MB | MPS | on | **6.60 ms** | 6.62 ms | 8.88 ms | — | — | **1.35x** | **—** |
| tensor | large_int8_1d | read_full | 0.96 MB | MPS | on | **435.6 μs** | 450.8 μs | 1.02 ms | — | — | **2.33x** | **—** |
| tensor | large_int8_2d | read_full | 4.00 MB | MPS | on | **1.19 ms** | 1.12 ms | 2.36 ms | — | — | **2.11x** | **—** |
| tensor | large_uint16_2d | read_full | 8.00 MB | MPS | on | **3.35 ms** | 3.33 ms | 5.09 ms | — | — | **1.53x** | **—** |
| tensor | large_uint32_2d | read_full | 16.00 MB | MPS | on | **4.89 ms** | 4.91 ms | 8.63 ms | — | — | **1.77x** | **—** |
| tensor | medium_float32_1d | read_full | 0.38 MB | MPS | on | **277.7 μs** | 391.6 μs | 515.3 μs | — | — | **1.86x** | **—** |
| tensor | medium_float32_2d | read_full | 4.00 MB | MPS | on | **935.6 μs** | 1.03 ms | 1.60 ms | — | — | **1.71x** | **—** |
| tensor | medium_float32_3d | read_full | 6.25 MB | MPS | on | **1.37 ms** | 1.27 ms | 2.25 ms | — | — | **1.77x** | **—** |
| tensor | medium_int16_1d | read_full | 0.20 MB | MPS | on | **297.6 μs** | 429.1 μs | 553.5 μs | — | — | **1.86x** | **—** |
| tensor | medium_int16_2d | read_full | 2.01 MB | MPS | on | **562.8 μs** | 710.3 μs | 1.03 ms | — | — | **1.83x** | **—** |
| tensor | medium_int16_3d | read_full | 3.13 MB | MPS | on | **809.3 μs** | 791.4 μs | 1.41 ms | — | — | **1.78x** | **—** |
| tensor | medium_int32_1d | read_full | 0.38 MB | MPS | on | **333.0 μs** | 309.7 μs | 600.7 μs | — | — | **1.94x** | **—** |
| tensor | medium_int32_2d | read_full | 4.00 MB | MPS | on | **852.3 μs** | 948.3 μs | 1.66 ms | — | — | **1.95x** | **—** |
| tensor | medium_int32_3d | read_full | 6.25 MB | MPS | on | **1.36 ms** | 1.39 ms | 2.22 ms | — | — | **1.63x** | **—** |
| tensor | medium_int64_1d | read_full | 0.77 MB | MPS | on | **344.7 μs** | 494.6 μs | 562.5 μs | — | — | **1.63x** | **—** |
| tensor | medium_int64_2d | read_full | 8.00 MB | MPS | on | **1.64 ms** | 1.62 ms | 2.63 ms | — | — | **1.62x** | **—** |
| tensor | medium_int64_3d | read_full | 12.51 MB | MPS | on | **2.59 ms** | 2.49 ms | 2.92 ms | — | — | **1.17x** | **—** |
| tensor | medium_int8_1d | read_full | 0.10 MB | MPS | on | **356.3 μs** | 373.6 μs | 717.3 μs | — | — | **2.01x** | **—** |
| tensor | medium_int8_2d | read_full | 1.01 MB | MPS | on | **465.9 μs** | 375.9 μs | 1.40 ms | — | — | **3.72x** | **—** |
| tensor | medium_int8_3d | read_full | 1.57 MB | MPS | on | **534.7 μs** | 608.9 μs | 1.37 ms | — | — | **2.56x** | **—** |
| tensor | medium_uint16_2d | read_full | 2.01 MB | MPS | on | **907.0 μs** | 930.3 μs | 2.01 ms | — | — | **2.22x** | **—** |
| tensor | medium_uint32_2d | read_full | 4.00 MB | MPS | on | **1.33 ms** | 1.42 ms | 2.83 ms | — | — | **2.12x** | **—** |
| tensor | mef_medium | read_full | 7.02 MB | MPS | on | **450.0 μs** | 360.9 μs | 1.36 ms | — | — | **3.78x** | **—** |
| tensor | mef_small | read_full | 0.45 MB | MPS | on | **257.0 μs** | 240.7 μs | 1.04 ms | — | — | **4.33x** | **—** |
| tensor | multi_mef_10ext | read_full | 2.68 MB | MPS | on | **245.3 μs** | 229.3 μs | 1.02 ms | — | — | **4.46x** | **—** |
| tensor | scaled_large | read_full | 8.00 MB | MPS | on | **3.34 ms** | 3.35 ms | 7.37 ms | — | — | **2.21x** | **—** |
| tensor | scaled_medium | read_full | 2.01 MB | MPS | on | **974.2 μs** | 998.7 μs | 2.52 ms | — | — | **2.59x** | **—** |
| tensor | scaled_small | read_full | 0.13 MB | MPS | on | **301.4 μs** | 291.3 μs | 805.4 μs | — | — | **2.76x** | **—** |
| tensor | small_float32_1d | read_full | 42.2 KB | MPS | on | **280.8 μs** | 257.6 μs | 506.6 μs | — | — | **1.97x** | **—** |
| tensor | small_float32_2d | read_full | 0.26 MB | MPS | on | **370.8 μs** | 283.9 μs | 518.8 μs | — | — | **1.83x** | **—** |
| tensor | small_float32_3d | read_full | 0.63 MB | MPS | on | **316.1 μs** | 307.0 μs | 633.1 μs | — | — | **2.06x** | **—** |
| tensor | small_int16_1d | read_full | 22.5 KB | MPS | on | **244.7 μs** | 253.6 μs | 414.4 μs | — | — | **1.69x** | **—** |
| tensor | small_int16_2d | read_full | 0.13 MB | MPS | on | **355.3 μs** | 333.4 μs | 489.8 μs | — | — | **1.47x** | **—** |
| tensor | small_int16_3d | read_full | 0.32 MB | MPS | on | **306.7 μs** | 320.8 μs | 572.5 μs | — | — | **1.87x** | **—** |
| tensor | small_int32_1d | read_full | 42.2 KB | MPS | on | **251.2 μs** | 420.9 μs | 565.1 μs | — | — | **2.25x** | **—** |
| tensor | small_int32_2d | read_full | 0.26 MB | MPS | on | **614.1 μs** | 568.3 μs | 550.9 μs | — | — | **0.97x** | **—** |
| tensor | small_int32_3d | read_full | 0.63 MB | MPS | on | **330.9 μs** | 365.1 μs | 756.6 μs | — | — | **2.29x** | **—** |
| tensor | small_int64_1d | read_full | 0.08 MB | MPS | on | **280.0 μs** | 320.4 μs | 484.0 μs | — | — | **1.73x** | **—** |
| tensor | small_int64_2d | read_full | 0.51 MB | MPS | on | **308.2 μs** | 363.2 μs | 620.3 μs | — | — | **2.01x** | **—** |
| tensor | small_int64_3d | read_full | 1.26 MB | MPS | on | **435.9 μs** | 448.9 μs | 819.8 μs | — | — | **1.88x** | **—** |
| tensor | small_int8_1d | read_full | 14.1 KB | MPS | on | **229.0 μs** | 316.2 μs | 713.3 μs | — | — | **3.11x** | **—** |
| tensor | small_int8_2d | read_full | 0.07 MB | MPS | on | **291.0 μs** | 243.7 μs | 727.6 μs | — | — | **2.99x** | **—** |
| tensor | small_int8_3d | read_full | 0.16 MB | MPS | on | **297.4 μs** | 266.7 μs | 771.9 μs | — | — | **2.89x** | **—** |
| tensor | small_uint16_2d | read_full | 0.13 MB | MPS | on | **387.6 μs** | 403.7 μs | 880.0 μs | — | — | **2.27x** | **—** |
| tensor | small_uint32_2d | read_full | 0.26 MB | MPS | on | **314.5 μs** | 304.6 μs | 849.0 μs | — | — | **2.79x** | **—** |
| tensor | timeseries_frame_000 | read_full | 0.26 MB | MPS | on | **316.1 μs** | 290.5 μs | 635.5 μs | — | — | **2.19x** | **—** |
| tensor | timeseries_frame_001 | read_full | 0.26 MB | MPS | on | **276.5 μs** | 339.9 μs | 590.8 μs | — | — | **2.14x** | **—** |
| tensor | timeseries_frame_002 | read_full | 0.26 MB | MPS | on | **317.0 μs** | 286.0 μs | 567.7 μs | — | — | **1.98x** | **—** |
| tensor | timeseries_frame_003 | read_full | 0.26 MB | MPS | on | **274.6 μs** | 279.4 μs | 544.2 μs | — | — | **1.98x** | **—** |
| tensor | timeseries_frame_004 | read_full | 0.26 MB | MPS | on | **253.7 μs** | 372.6 μs | 481.4 μs | — | — | **1.90x** | **—** |
| tensor | tiny_float32_1d | read_full | 8.4 KB | MPS | on | **334.0 μs** | 247.0 μs | 425.2 μs | — | — | **1.72x** | **—** |
| tensor | tiny_float32_2d | read_full | 19.7 KB | MPS | on | **318.5 μs** | 244.9 μs | 482.6 μs | — | — | **1.97x** | **—** |
| tensor | tiny_float32_3d | read_full | 25.3 KB | MPS | on | **337.9 μs** | 332.5 μs | 436.1 μs | — | — | **1.31x** | **—** |
| tensor | tiny_int16_1d | read_full | 5.6 KB | MPS | on | **237.9 μs** | 231.3 μs | 418.8 μs | — | — | **1.81x** | **—** |
| tensor | tiny_int16_2d | read_full | 11.2 KB | MPS | on | **251.0 μs** | 260.4 μs | 424.7 μs | — | — | **1.69x** | **—** |
| tensor | tiny_int16_3d | read_full | 14.1 KB | MPS | on | **245.1 μs** | 242.5 μs | 434.9 μs | — | — | **1.79x** | **—** |
| tensor | tiny_int32_1d | read_full | 8.4 KB | MPS | on | **231.8 μs** | 243.3 μs | 426.3 μs | — | — | **1.84x** | **—** |
| tensor | tiny_int32_2d | read_full | 19.7 KB | MPS | on | **252.9 μs** | 253.9 μs | 420.7 μs | — | — | **1.66x** | **—** |
| tensor | tiny_int32_3d | read_full | 25.3 KB | MPS | on | **268.9 μs** | 235.3 μs | 500.7 μs | — | — | **2.13x** | **—** |
| tensor | tiny_int64_1d | read_full | 11.2 KB | MPS | on | **238.9 μs** | 219.7 μs | 508.2 μs | — | — | **2.31x** | **—** |
| tensor | tiny_int64_2d | read_full | 36.6 KB | MPS | on | **241.0 μs** | 262.2 μs | 450.7 μs | — | — | **1.87x** | **—** |
| tensor | tiny_int64_3d | read_full | 45.0 KB | MPS | on | **284.1 μs** | 253.4 μs | 501.7 μs | — | — | **1.98x** | **—** |
| tensor | tiny_int8_1d | read_full | 5.6 KB | MPS | on | **269.2 μs** | 315.0 μs | 658.6 μs | — | — | **2.45x** | **—** |
| tensor | tiny_int8_2d | read_full | 8.4 KB | MPS | on | **336.3 μs** | 306.9 μs | 778.7 μs | — | — | **2.54x** | **—** |
| tensor | tiny_int8_3d | read_full | 8.4 KB | MPS | on | **367.6 μs** | 279.0 μs | 760.0 μs | — | — | **2.72x** | **—** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **264.1 μs** | 271.4 μs | 2.57 ms | 357.7 μs | — | **9.72x** | **1.35x** |
| table | ascii_10000 | projection | 0.44 MB | CPU | off | **995.8 μs** | 1.04 ms | 8.46 ms | 2.31 ms | — | **8.50x** | **2.32x** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | off | **1.09 ms** | 984.2 μs | 8.28 ms | 2.32 ms | — | **8.41x** | **2.36x** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | off | **184.5 μs** | 163.6 μs | 2.34 ms | 767.4 μs | — | **14.33x** | **4.69x** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | off | **—** | — | 322.0 μs | 80.7 μs | — | **—** | **—** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **95.3 μs** | 93.2 μs | 1.19 ms | 138.2 μs | — | **12.81x** | **1.48x** |
| table | ascii_1000 | projection | 50.6 KB | CPU | off | **150.9 μs** | 143.1 μs | 1.78 ms | 331.5 μs | — | **12.45x** | **2.32x** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | off | **166.1 μs** | 145.6 μs | 1.81 ms | 325.7 μs | — | **12.46x** | **2.24x** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | off | **94.2 μs** | 88.5 μs | 1.52 ms | 167.2 μs | — | **17.22x** | **1.89x** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | off | **—** | — | 328.1 μs | 78.0 μs | — | **—** | **—** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **12.96 ms** | 13.76 ms | 16.04 ms | 33.62 ms | — | **1.24x** | **2.59x** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | off | **10.38 ms** | 9.67 ms | 17.29 ms | 63.04 ms | — | **1.79x** | **6.52x** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | off | **18.32 ms** | 17.50 ms | 305.45 ms | 109.68 ms | — | **17.46x** | **6.27x** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **408.0 μs** | 798.1 μs | 14.71 ms | 1.70 ms | — | **36.05x** | **4.16x** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **—** | — | 379.5 μs | 104.7 μs | — | **—** | **—** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **1.07 ms** | 1.10 ms | 2.57 ms | 3.37 ms | — | **2.40x** | **3.15x** |
| table | mixed_100000 | projection | 5.06 MB | CPU | off | **933.2 μs** | 1.34 ms | 2.47 ms | 6.44 ms | — | **2.64x** | **6.90x** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | off | **2.20 ms** | 2.12 ms | 31.85 ms | 10.44 ms | — | **15.00x** | **4.92x** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | off | **353.5 μs** | 382.2 μs | 5.73 ms | 1.72 ms | — | **16.22x** | **4.86x** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | off | **—** | — | 356.6 μs | 93.1 μs | — | **—** | **—** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **244.1 μs** | 376.1 μs | 1.76 ms | 621.6 μs | — | **7.20x** | **2.55x** |
| table | mixed_10000 | projection | 0.51 MB | CPU | off | **234.1 μs** | 276.3 μs | 1.44 ms | 725.8 μs | — | **6.14x** | **3.10x** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | off | **288.9 μs** | 731.1 μs | 5.03 ms | 1.51 ms | — | **17.40x** | **5.21x** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | off | **136.6 μs** | 109.3 μs | 2.32 ms | 387.5 μs | — | **21.21x** | **3.55x** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | off | **—** | — | 333.0 μs | 83.5 μs | — | **—** | **—** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **87.0 μs** | 110.8 μs | 1.41 ms | 166.4 μs | — | **16.16x** | **1.91x** |
| table | mixed_1000 | projection | 0.06 MB | CPU | off | **99.8 μs** | 123.6 μs | 1.42 ms | 212.9 μs | — | **14.22x** | **2.13x** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | off | **133.5 μs** | 114.1 μs | 1.66 ms | 301.6 μs | — | **14.58x** | **2.64x** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | off | **137.0 μs** | 106.2 μs | 2.05 ms | 167.7 μs | — | **19.32x** | **1.58x** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | off | **—** | — | 334.2 μs | 89.7 μs | — | **—** | **—** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **8.78 ms** | 8.45 ms | 8.56 ms | 17.44 ms | — | **1.01x** | **2.06x** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | off | **4.34 ms** | 4.66 ms | 7.59 ms | 31.74 ms | — | **1.75x** | **7.31x** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | off | **5.87 ms** | 5.40 ms | 7.85 ms | 6.78 ms | — | **1.45x** | **1.26x** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **293.2 μs** | 321.7 μs | 4.71 ms | 759.3 μs | — | **16.06x** | **2.59x** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **—** | — | 394.6 μs | 111.7 μs | — | **—** | **—** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **949.6 μs** | 963.3 μs | 1.62 ms | 1.83 ms | — | **1.70x** | **1.93x** |
| table | narrow_100000 | projection | 1.25 MB | CPU | off | **452.2 μs** | 410.8 μs | 1.41 ms | 3.19 ms | — | **3.44x** | **7.76x** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | off | **666.7 μs** | 715.7 μs | 1.87 ms | 760.1 μs | — | **2.81x** | **1.14x** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | off | **176.3 μs** | 249.2 μs | 1.66 ms | 724.0 μs | — | **9.44x** | **4.11x** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | off | **—** | — | 363.4 μs | 82.0 μs | — | **—** | **—** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **147.7 μs** | 174.4 μs | 1.01 ms | 286.7 μs | — | **6.86x** | **1.94x** |
| table | narrow_10000 | projection | 0.13 MB | CPU | off | **263.0 μs** | 218.8 μs | 994.7 μs | 467.3 μs | — | **4.55x** | **2.14x** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | off | **194.2 μs** | 123.3 μs | 1.07 ms | 294.7 μs | — | **8.70x** | **2.39x** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | off | **111.9 μs** | 90.9 μs | 1.31 ms | 170.6 μs | — | **14.36x** | **1.88x** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | off | **—** | — | 319.8 μs | 75.6 μs | — | **—** | **—** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **96.3 μs** | 85.7 μs | 865.6 μs | 151.9 μs | — | **10.10x** | **1.77x** |
| table | narrow_1000 | projection | 19.7 KB | CPU | off | **116.1 μs** | 136.7 μs | 929.0 μs | 151.5 μs | — | **8.00x** | **1.31x** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | off | **115.3 μs** | 98.4 μs | 1.05 ms | 143.3 μs | — | **10.63x** | **1.46x** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | off | **109.6 μs** | 98.3 μs | 1.23 ms | 175.0 μs | — | **12.53x** | **1.78x** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | off | **—** | — | 321.5 μs | 75.5 μs | — | **—** | **—** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **526.2 μs** | 614.7 μs | 1.54 ms | 1.55 ms | — | **2.92x** | **2.95x** |
| table | typed_100000 | projection | 2.39 MB | CPU | off | **4.34 ms** | 4.07 ms | 29.74 ms | 14.20 ms | — | **7.30x** | **3.49x** |
| table | typed_100000 | read_full | 2.39 MB | CPU | off | **5.31 ms** | 5.51 ms | 30.03 ms | 16.22 ms | — | **5.65x** | **3.05x** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | off | **625.8 μs** | 812.0 μs | 4.39 ms | 2.67 ms | — | **7.02x** | **4.26x** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | off | **—** | — | 357.3 μs | 74.2 μs | — | **—** | **—** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **248.4 μs** | 213.9 μs | 1.11 ms | 284.7 μs | — | **5.17x** | **1.33x** |
| table | typed_10000 | projection | 0.24 MB | CPU | off | **638.2 μs** | 535.0 μs | 3.88 ms | 1.57 ms | — | **7.25x** | **2.94x** |
| table | typed_10000 | read_full | 0.24 MB | CPU | off | **760.5 μs** | 755.0 μs | 3.89 ms | 1.70 ms | — | **5.16x** | **2.26x** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | off | **137.8 μs** | 115.3 μs | 1.68 ms | 363.5 μs | — | **14.57x** | **3.15x** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | off | **—** | — | 329.9 μs | 74.9 μs | — | **—** | **—** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **599.6 μs** | 512.6 μs | 1.32 ms | 1.41 ms | — | **2.57x** | **2.75x** |
| table | varlen_100000 | projection | 3.06 MB | CPU | off | **76.89 ms** | 77.41 ms | 494.05 ms | 111.85 ms | — | **6.43x** | **1.45x** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | off | **77.19 ms** | 76.72 ms | 491.92 ms | 111.95 ms | — | **6.41x** | **1.46x** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | off | **6.83 ms** | 6.68 ms | 50.51 ms | 12.67 ms | — | **7.56x** | **1.90x** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | off | **—** | — | 334.6 μs | 85.8 μs | — | **—** | **—** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **204.3 μs** | 261.0 μs | 914.5 μs | 378.5 μs | — | **4.48x** | **1.85x** |
| table | varlen_10000 | projection | 0.31 MB | CPU | off | **7.02 ms** | 7.13 ms | 49.59 ms | 11.45 ms | — | **7.06x** | **1.63x** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | off | **7.07 ms** | 6.74 ms | 49.20 ms | 11.26 ms | — | **7.30x** | **1.67x** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | off | **659.6 μs** | 656.0 μs | 6.11 ms | 1.65 ms | — | **9.31x** | **2.52x** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | off | **—** | — | 317.8 μs | 69.9 μs | — | **—** | **—** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **106.7 μs** | 89.1 μs | 876.3 μs | 125.6 μs | — | **9.84x** | **1.41x** |
| table | varlen_1000 | projection | 39.4 KB | CPU | off | **653.7 μs** | 646.3 μs | 5.72 ms | 1.28 ms | — | **8.85x** | **1.98x** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | off | **662.8 μs** | 660.5 μs | 6.01 ms | 1.27 ms | — | **9.09x** | **1.92x** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | off | **161.0 μs** | 140.1 μs | 1.71 ms | 267.3 μs | — | **12.18x** | **1.91x** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | off | **—** | — | 309.5 μs | 65.7 μs | — | **—** | **—** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **2.51 ms** | 2.83 ms | 8.62 ms | 6.18 ms | — | **3.44x** | **2.46x** |
| table | wide_100000 | projection | 20.71 MB | CPU | off | **3.56 ms** | 3.27 ms | 8.84 ms | 7.24 ms | — | **2.70x** | **2.22x** |
| table | wide_100000 | read_full | 20.71 MB | CPU | off | **14.48 ms** | 14.92 ms | 128.59 ms | 54.66 ms | — | **8.88x** | **3.78x** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | off | **1.70 ms** | 1.63 ms | 22.07 ms | 6.28 ms | — | **13.58x** | **3.86x** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | off | **—** | — | 475.2 μs | 261.5 μs | — | **—** | **—** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **257.3 μs** | 303.6 μs | 4.60 ms | 899.8 μs | — | **17.89x** | **3.50x** |
| table | wide_10000 | projection | 2.08 MB | CPU | off | **539.6 μs** | 497.8 μs | 4.82 ms | 1.14 ms | — | **9.67x** | **2.28x** |
| table | wide_10000 | read_full | 2.08 MB | CPU | off | **1.58 ms** | 1.64 ms | 17.14 ms | 5.52 ms | — | **10.87x** | **3.50x** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | off | **516.2 μs** | 498.6 μs | 8.57 ms | 1.04 ms | — | **17.19x** | **2.09x** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | off | **—** | — | 443.8 μs | 251.2 μs | — | **—** | **—** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **140.0 μs** | 129.0 μs | 4.26 ms | 370.9 μs | — | **32.99x** | **2.88x** |
| table | wide_1000 | projection | 0.22 MB | CPU | off | **160.6 μs** | 186.7 μs | 4.31 ms | 373.3 μs | — | **26.81x** | **2.32x** |
| table | wide_1000 | read_full | 0.22 MB | CPU | off | **587.1 μs** | 533.6 μs | 5.76 ms | 935.3 μs | — | **10.79x** | **1.75x** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | off | **399.0 μs** | 395.8 μs | 7.34 ms | 728.3 μs | — | **18.54x** | **1.84x** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | off | **—** | — | 443.3 μs | 251.0 μs | — | **—** | **—** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **127.5 μs** | 136.5 μs | 2.44 ms | — | — | **19.16x** | **—** |
| table | ascii_10000 | projection | 0.44 MB | CPU | on | **217.1 μs** | 152.1 μs | 8.22 ms | — | — | **54.07x** | **—** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | on | **201.6 μs** | 148.3 μs | 8.16 ms | — | — | **55.04x** | **—** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | on | **398.4 μs** | 98.9 μs | 1.97 ms | — | — | **19.88x** | **—** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | on | **—** | — | 311.5 μs | — | — | **—** | **—** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **86.2 μs** | 88.6 μs | 1.08 ms | — | — | **12.48x** | **—** |
| table | ascii_1000 | projection | 50.6 KB | CPU | on | **146.8 μs** | 82.5 μs | 1.66 ms | — | — | **20.17x** | **—** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | on | **178.3 μs** | 94.2 μs | 1.67 ms | — | — | **17.76x** | **—** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | on | **318.2 μs** | 86.4 μs | 1.29 ms | — | — | **14.90x** | **—** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | on | **—** | — | 305.9 μs | — | — | **—** | **—** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **15.61 ms** | 14.36 ms | 16.09 ms | — | — | **1.12x** | **—** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | on | **11.46 ms** | 11.34 ms | 17.11 ms | — | — | **1.51x** | **—** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | on | **21.26 ms** | 20.94 ms | 325.00 ms | — | — | **15.52x** | **—** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **355.6 μs** | 469.7 μs | 10.93 ms | — | — | **30.74x** | **—** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **—** | — | 406.8 μs | — | — | **—** | **—** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **1.18 ms** | 1.31 ms | 2.61 ms | — | — | **2.21x** | **—** |
| table | mixed_100000 | projection | 5.06 MB | CPU | on | **905.7 μs** | 855.2 μs | 3.26 ms | — | — | **3.81x** | **—** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | on | **2.22 ms** | 1.95 ms | 34.07 ms | — | — | **17.43x** | **—** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | on | **370.0 μs** | 264.9 μs | 5.68 ms | — | — | **21.43x** | **—** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | on | **—** | — | 382.8 μs | — | — | **—** | **—** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **175.0 μs** | 186.1 μs | 1.55 ms | — | — | **8.83x** | **—** |
| table | mixed_10000 | projection | 0.51 MB | CPU | on | **201.2 μs** | 142.1 μs | 1.47 ms | — | — | **10.34x** | **—** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | on | **380.9 μs** | 570.5 μs | 5.02 ms | — | — | **13.18x** | **—** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | on | **191.9 μs** | 124.3 μs | 2.35 ms | — | — | **18.93x** | **—** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | on | **—** | — | 354.5 μs | — | — | **—** | **—** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **83.1 μs** | 112.2 μs | 1.38 ms | — | — | **16.56x** | **—** |
| table | mixed_1000 | projection | 0.06 MB | CPU | on | **159.4 μs** | 104.4 μs | 1.36 ms | — | — | **13.00x** | **—** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | on | **174.5 μs** | 132.0 μs | 1.73 ms | — | — | **13.13x** | **—** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | on | **177.1 μs** | 123.8 μs | 2.06 ms | — | — | **16.67x** | **—** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | on | **—** | — | 330.7 μs | — | — | **—** | **—** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **8.47 ms** | 8.11 ms | 7.98 ms | — | — | **0.98x** | **—** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | on | **2.85 ms** | 2.62 ms | 6.74 ms | — | — | **2.57x** | **—** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | on | **3.54 ms** | 4.01 ms | 7.32 ms | — | — | **2.07x** | **—** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **203.3 μs** | 165.2 μs | 3.58 ms | — | — | **21.65x** | **—** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **—** | — | 354.6 μs | — | — | **—** | **—** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **1.04 ms** | 973.6 μs | 2.31 ms | — | — | **2.37x** | **—** |
| table | narrow_100000 | projection | 1.25 MB | CPU | on | **503.1 μs** | 726.5 μs | 2.98 ms | — | — | **5.93x** | **—** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | on | **459.0 μs** | 531.2 μs | 1.84 ms | — | — | **4.01x** | **—** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | on | **207.5 μs** | 277.3 μs | 1.55 ms | — | — | **7.49x** | **—** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | on | **—** | — | 352.3 μs | — | — | **—** | **—** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **142.2 μs** | 154.5 μs | 949.4 μs | — | — | **6.67x** | **—** |
| table | narrow_10000 | projection | 0.13 MB | CPU | on | **270.0 μs** | 115.6 μs | 947.2 μs | — | — | **8.19x** | **—** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | on | **251.9 μs** | 112.8 μs | 924.0 μs | — | — | **8.19x** | **—** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | on | **132.2 μs** | 80.9 μs | 1.19 ms | — | — | **14.70x** | **—** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | on | **—** | — | 347.8 μs | — | — | **—** | **—** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **86.9 μs** | 93.3 μs | 939.2 μs | — | — | **10.81x** | **—** |
| table | narrow_1000 | projection | 19.7 KB | CPU | on | **139.2 μs** | 86.3 μs | 942.2 μs | — | — | **10.91x** | **—** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | on | **278.4 μs** | 105.7 μs | 1.09 ms | — | — | **10.32x** | **—** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | on | **154.1 μs** | 99.3 μs | 1.32 ms | — | — | **13.32x** | **—** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | on | **—** | — | 375.1 μs | — | — | **—** | **—** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **534.3 μs** | 545.0 μs | 1.18 ms | — | — | **2.21x** | **—** |
| table | typed_100000 | projection | 2.39 MB | CPU | on | **1.48 ms** | 1.24 ms | 31.58 ms | — | — | **25.40x** | **—** |
| table | typed_100000 | read_full | 2.39 MB | CPU | on | **1.56 ms** | 1.35 ms | 30.28 ms | — | — | **22.36x** | **—** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | on | **247.6 μs** | 196.9 μs | 4.17 ms | — | — | **21.18x** | **—** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | on | **—** | — | 368.3 μs | — | — | **—** | **—** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **116.5 μs** | 274.0 μs | 931.3 μs | — | — | **7.99x** | **—** |
| table | typed_10000 | projection | 0.24 MB | CPU | on | **278.6 μs** | 222.4 μs | 4.05 ms | — | — | **18.20x** | **—** |
| table | typed_10000 | read_full | 0.24 MB | CPU | on | **269.7 μs** | 210.4 μs | 3.83 ms | — | — | **18.20x** | **—** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | on | **267.8 μs** | 111.8 μs | 1.55 ms | — | — | **13.89x** | **—** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | on | **—** | — | 339.0 μs | — | — | **—** | **—** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **522.0 μs** | 484.1 μs | 1.10 ms | — | — | **2.28x** | **—** |
| table | varlen_100000 | projection | 3.06 MB | CPU | on | **81.73 ms** | 81.94 ms | 496.05 ms | — | — | **6.07x** | **—** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | on | **86.73 ms** | 81.90 ms | 489.86 ms | — | — | **5.98x** | **—** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | on | **7.74 ms** | 8.04 ms | 52.25 ms | — | — | **6.75x** | **—** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | on | **—** | — | 328.7 μs | — | — | **—** | **—** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **135.1 μs** | 124.7 μs | 911.8 μs | — | — | **7.31x** | **—** |
| table | varlen_10000 | projection | 0.31 MB | CPU | on | **8.24 ms** | 8.10 ms | 53.79 ms | — | — | **6.64x** | **—** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | on | **8.04 ms** | 8.21 ms | 52.13 ms | — | — | **6.49x** | **—** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | on | **999.7 μs** | 1.14 ms | 8.62 ms | — | — | **8.62x** | **—** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | on | **—** | — | 367.7 μs | — | — | **—** | **—** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **76.6 μs** | 84.4 μs | 867.3 μs | — | — | **11.32x** | **—** |
| table | varlen_1000 | projection | 39.4 KB | CPU | on | **850.5 μs** | 951.6 μs | 6.16 ms | — | — | **7.24x** | **—** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | on | **880.6 μs** | 934.0 μs | 5.95 ms | — | — | **6.76x** | **—** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | on | **240.2 μs** | 335.8 μs | 1.74 ms | — | — | **7.23x** | **—** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | on | **—** | — | 334.0 μs | — | — | **—** | **—** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **3.17 ms** | 3.02 ms | 7.49 ms | — | — | **2.48x** | **—** |
| table | wide_100000 | projection | 20.71 MB | CPU | on | **2.88 ms** | 1.96 ms | 7.45 ms | — | — | **3.81x** | **—** |
| table | wide_100000 | read_full | 20.71 MB | CPU | on | **18.50 ms** | 15.74 ms | 129.52 ms | — | — | **8.23x** | **—** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | on | **1.25 ms** | 1.49 ms | 25.44 ms | — | — | **20.40x** | **—** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | on | **—** | — | 464.0 μs | — | — | **—** | **—** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **274.3 μs** | 325.1 μs | 4.79 ms | — | — | **17.47x** | **—** |
| table | wide_10000 | projection | 2.08 MB | CPU | on | **357.7 μs** | 296.2 μs | 4.87 ms | — | — | **16.44x** | **—** |
| table | wide_10000 | read_full | 2.08 MB | CPU | on | **1.43 ms** | 1.43 ms | 18.81 ms | — | — | **13.13x** | **—** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | on | **580.7 μs** | 501.0 μs | 8.93 ms | — | — | **17.82x** | **—** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | on | **—** | — | 499.2 μs | — | — | **—** | **—** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **122.5 μs** | 148.7 μs | 8.30 ms | — | — | **67.78x** | **—** |
| table | wide_1000 | projection | 0.22 MB | CPU | on | **278.6 μs** | 141.7 μs | 4.26 ms | — | — | **30.07x** | **—** |
| table | wide_1000 | read_full | 0.22 MB | CPU | on | **536.7 μs** | 440.9 μs | 5.76 ms | — | — | **13.06x** | **—** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | on | **521.9 μs** | 384.5 μs | 7.18 ms | — | — | **18.68x** | **—** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | on | **—** | — | 439.1 μs | — | — | **—** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (CPU and GPU). GPU lags may reflect software or hardware limits — they are listed, not hidden.

| Platform | Domain | Case | mmap | torchfits | Peak RSS (MB) | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| macOS arm64 / CPU | tensor | small_float64_1d [read_full] | off | 138.1 μs | 854.9 | fitsio/fitsio_torch | 2.17× |
| macOS arm64 / MPS | tensor | timeseries_frame_001 [read_full @ mps] | off | 675.8 μs | 856.5 | fitsio/fitsio_torch_device | 1.89× |
| macOS arm64 / CPU | tensor | timeseries_frame_004 [read_full] | off | 202.8 μs | 854.9 | fitsio/fitsio_torch | 1.66× |
| macOS arm64 / MPS | tensor | small_uint16_2d [read_full @ mps] | off | 457.7 μs | 856.5 | fitsio/fitsio_torch_device | 1.46× |
| macOS arm64 / MPS | tensor | tiny_int64_2d [read_full @ mps] | off | 337.8 μs | 856.5 | fitsio/fitsio_torch_device | 1.40× |
| macOS arm64 / CPU | tensor | small_int8_2d [read_full] | off | 152.1 μs | 854.9 | fitsio/fitsio_torch | 1.39× |
| macOS arm64 / CPU | tensor | small_float32_1d [read_full] | off | 131.5 μs | 854.9 | fitsio/fitsio_torch | 1.35× |
| macOS arm64 / CPU | tensor | mef_small [read_full] | off | 175.8 μs | 854.9 | fitsio/fitsio_torch | 1.29× |
| macOS arm64 / MPS | tensor | small_int16_2d [read_full @ mps] | off | 363.2 μs | 856.5 | fitsio/fitsio_torch_device | 1.28× |
| macOS arm64 / MPS | tensor | timeseries_frame_002 [read_full @ mps] | off | 376.0 μs | 856.5 | fitsio/fitsio_torch_device | 1.27× |
| macOS arm64 / MPS | tensor | tiny_int64_1d [read_full @ mps] | off | 625.6 μs | 856.5 | astropy/astropy_torch_device | 1.27× |
| macOS arm64 / MPS | tensor | tiny_int32_3d [read_full @ mps] | off | 519.1 μs | 856.5 | fitsio/fitsio_torch_device | 1.24× |
| macOS arm64 / MPS | tensor | tiny_int16_2d [read_full @ mps] | off | 309.0 μs | 856.5 | fitsio/fitsio_torch_device | 1.24× |
| macOS arm64 / MPS | tensor | tiny_int16_3d [read_full @ mps] | off | 305.7 μs | 856.5 | fitsio/fitsio_torch_device | 1.24× |
| macOS arm64 / CPU | tensor | tiny_int64_3d [read_full] | off | 136.2 μs | 854.9 | fitsio/fitsio_torch | 1.22× |
| macOS arm64 / MPS | tensor | tiny_int32_1d [read_full @ mps] | off | 284.9 μs | 856.5 | fitsio/fitsio_torch_device | 1.20× |
| macOS arm64 / MPS | tensor | medium_int8_1d [read_full @ mps] | off | 334.9 μs | 856.5 | fitsio/fitsio_torch_device | 1.18× |
| macOS arm64 / CPU | tensor | compressed_rice_1 [cutout_100x100] | n/a | 869.9 μs | 854.9 | fitsio/fitsio_torch | 1.14× |
| macOS arm64 / MPS | tensor | tiny_float32_2d [read_full @ mps] | off | 296.3 μs | 856.5 | fitsio/fitsio_torch_device | 1.12× |
| macOS arm64 / MPS | tensor | small_int32_2d [read_full @ mps] | on | 614.1 μs | 574.9 | astropy/astropy_torch_device | 1.11× |
| macOS arm64 / MPS | tensor | large_int64_1d [read_full @ mps] | off | 2.03 ms | 856.5 | fitsio/fitsio_torch_device | 1.10× |
| macOS arm64 / MPS | tensor | scaled_large [read_full @ mps] | off | 3.32 ms | 856.5 | fitsio/fitsio_torch_device | 1.08× |
| macOS arm64 / MPS | tensor | tiny_int64_3d [read_full @ mps] | off | 272.4 μs | 856.5 | fitsio/fitsio_torch_device | 1.08× |
| macOS arm64 / MPS | tensor | compressed_rice_1 [read_full @ mps] | off | 7.67 ms | 856.5 | fitsio/fitsio_torch_device | 1.08× |
| macOS arm64 / MPS | tensor | tiny_float32_3d [read_full @ mps] | off | 267.1 μs | 856.5 | fitsio/fitsio_torch_device | 1.07× |
| macOS arm64 / MPS | tensor | timeseries_frame_003 [read_full @ mps] | off | 303.8 μs | 856.5 | fitsio/fitsio_torch_device | 1.05× |
| macOS arm64 / MPS | tensor | large_float32_2d [read_full @ mps] | off | 3.85 ms | 856.5 | fitsio/fitsio_torch_device | 1.05× |
| macOS arm64 / MPS | tensor | large_int32_2d [read_full @ mps] | off | 3.65 ms | 856.5 | fitsio/fitsio_torch_device | 1.05× |
| macOS arm64 / MPS | tensor | compressed_rice_1 [read_full @ mps] | on | 7.24 ms | 574.3 | fitsio/fitsio_torch_device | 1.04× |
| macOS arm64 / MPS | tensor | scaled_medium [read_full @ mps] | off | 984.1 μs | 856.5 | fitsio/fitsio_torch_device | 1.03× |
| macOS arm64 / MPS | tensor | large_uint16_2d [read_full @ mps] | off | 3.76 ms | 856.5 | fitsio/fitsio_torch_device | 1.03× |
| macOS arm64 / MPS | tensor | tiny_float32_1d [read_full @ mps] | off | 285.2 μs | 856.5 | fitsio/fitsio_torch_device | 1.03× |
| macOS arm64 / MPS | tensor | medium_uint16_2d [read_full @ mps] | off | 995.2 μs | 856.5 | fitsio/fitsio_torch_device | 1.03× |
| macOS arm64 / MPS | tensor | medium_int64_3d [read_full @ mps] | off | 3.02 ms | 856.5 | fitsio/fitsio_torch_device | 1.03× |
| macOS arm64 / MPS | tensor | medium_int32_2d [read_full @ mps] | off | 970.2 μs | 856.5 | fitsio/fitsio_torch_device | 1.01× |
| macOS arm64 / MPS | tensor | compressed_hcompress_1 [read_full @ mps] | on | 22.07 ms | 574.4 | fitsio/fitsio_torch_device | 1.01× |
| macOS arm64 / MPS | tensor | small_float32_3d [read_full @ mps] | off | 325.3 μs | 856.5 | fitsio/fitsio_torch_device | 1.01× |
| macOS arm64 / MPS | tensor | compressed_hcompress_1 [read_full @ mps] | off | 22.08 ms | 856.5 | fitsio/fitsio_torch_device | 1.01× |
| macOS arm64 / MPS | tensor | medium_int64_2d [read_full @ mps] | off | 1.91 ms | 856.5 | fitsio/fitsio_torch_device | 1.01× |
| macOS arm64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 21.62 ms | 854.9 | fitsio/fitsio_torch | 1.00× |

_…and 281 more rows in `torchfits_deficits.csv`._
<!-- BENCH_DEFICITS_END -->

### Host scorecard

| Platform | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| macOS arm64 / MPS | `exhaustive_mps_20260719_143706` | 3931 | 127 | 854.9 | lab + mmap-matrix + GPU |
| Linux x86_64 / CPU | `exhaustive_cpu_20260719_144337` | 2829 | 92 | 289.4 | lab + mmap-matrix |
| Linux x86_64 / CUDA | `exhaustive_cuda_20260719_144457` | 4087 | 102 | 729.8 | lab + mmap-matrix + GPU |
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
Source: `benchmarks_results/exhaustive_mps_20260719_143706/ml_results.csv` (device=cpu).

| Case | Method | Median throughput |
|---|---|---:|
| ml_compressed_rice | `fitsio (comp)` | 443,859,959 pixels/s |
| ml_compressed_rice | `torchfits (comp)` | 432,586,096 pixels/s |
| ml_uncompressed | `fitsio + numpy` | 1,445,528,814 pixels/s |
| ml_uncompressed | `torchfits` | 1,495,169,580 pixels/s |
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
