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
Source: `benchmarks_results/exhaustive_cpu_20260801_202620/results.csv` (mmap on+off matrix.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### Tensor I/O (IMAGE HDU) (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.08 ms` (n=174) | `0.46 ms` (n=253) | `0.15 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.15 ms` (n=174) | `0.73 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### Table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.19 ms` (n=216) | `2.55 ms` (n=184) | `0.59 ms` (n=216) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.23 ms` (n=208) | `2.80 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Table read (100k rows, 8 cols, mixed) | CPU | **2.33 ms** | 2.31 ms | 30.73 ms | 10.17 ms | **13.31x** | **4.41x** |
| Varlen table read (100k rows, 3 cols) | CPU | **75.18 ms** | 11.08 ms | 525.26 ms | 107.73 ms | **47.39x** | **9.72x** |
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
| tensor | compressed_gzip_1:header_read | header_read | 1.29 MB | CPU | n/a | **—** | 29.0 μs | 1.36 ms | 136.3 μs | — | **47.08x** | **4.71x** |
| tensor | compressed_gzip_2:header_read | header_read | 0.89 MB | CPU | n/a | **—** | 28.9 μs | 1.36 ms | 137.2 μs | — | **46.96x** | **4.74x** |
| tensor | compressed_hcompress_1:header_read | header_read | 0.82 MB | CPU | n/a | **—** | 30.1 μs | 1.42 ms | 156.5 μs | — | **47.06x** | **5.20x** |
| tensor | compressed_rice_1:cutout_100x100 | cutout_100x100 | 0.90 MB | CPU | n/a | **779.5 μs** | 739.2 μs | 6.80 ms | 792.7 μs | — | **9.20x** | **1.07x** |
| tensor | compressed_rice_1:header_read | header_read | 0.90 MB | CPU | n/a | **—** | 29.1 μs | 1.42 ms | 152.8 μs | — | **48.82x** | **5.25x** |
| tensor | large_float32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 13.9 μs | 266.5 μs | 25.5 μs | — | **19.14x** | **1.83x** |
| tensor | large_float32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 15.0 μs | 297.5 μs | 26.5 μs | — | **19.83x** | **1.77x** |
| tensor | large_float64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 15.9 μs | 266.6 μs | 26.7 μs | — | **16.80x** | **1.68x** |
| tensor | large_float64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 35.4 μs | 683.4 μs | 65.6 μs | — | **19.31x** | **1.85x** |
| tensor | large_int16_1d:header_read | header_read | 1.91 MB | CPU | n/a | **—** | 17.7 μs | 352.4 μs | 33.1 μs | — | **19.88x** | **1.86x** |
| tensor | large_int16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 13.6 μs | 289.9 μs | 29.7 μs | — | **21.30x** | **2.18x** |
| tensor | large_int32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 15.2 μs | 262.3 μs | 25.1 μs | — | **17.30x** | **1.66x** |
| tensor | large_int32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 14.7 μs | 286.9 μs | 28.7 μs | — | **19.52x** | **1.95x** |
| tensor | large_int64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 14.0 μs | 263.6 μs | 28.0 μs | — | **18.77x** | **2.00x** |
| tensor | large_int64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 15.2 μs | 297.1 μs | 27.4 μs | — | **19.50x** | **1.80x** |
| tensor | large_int8_1d:header_read | header_read | 0.96 MB | CPU | n/a | **—** | 15.8 μs | 315.4 μs | 33.9 μs | — | **19.97x** | **2.15x** |
| tensor | large_int8_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 16.9 μs | 333.2 μs | 35.1 μs | — | **19.67x** | **2.07x** |
| tensor | large_uint16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 15.0 μs | 324.4 μs | 33.0 μs | — | **21.57x** | **2.19x** |
| tensor | large_uint32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 16.1 μs | 336.7 μs | 32.9 μs | — | **20.95x** | **2.05x** |
| tensor | medium_float32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 14.0 μs | 259.9 μs | 25.5 μs | — | **18.53x** | **1.82x** |
| tensor | medium_float32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 15.0 μs | 291.4 μs | 29.8 μs | — | **19.46x** | **1.99x** |
| tensor | medium_float32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 15.0 μs | 310.1 μs | 32.3 μs | — | **20.63x** | **2.15x** |
| tensor | medium_float64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 14.4 μs | 265.9 μs | 24.7 μs | — | **18.50x** | **1.72x** |
| tensor | medium_float64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 15.8 μs | 284.9 μs | 29.3 μs | — | **18.07x** | **1.86x** |
| tensor | medium_float64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 15.4 μs | 313.0 μs | 31.5 μs | — | **20.28x** | **2.04x** |
| tensor | medium_int16_1d:header_read | header_read | 0.20 MB | CPU | n/a | **—** | 15.4 μs | 263.6 μs | 25.5 μs | — | **17.14x** | **1.66x** |
| tensor | medium_int16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 14.3 μs | 292.0 μs | 28.2 μs | — | **20.37x** | **1.97x** |
| tensor | medium_int16_3d:header_read | header_read | 3.13 MB | CPU | n/a | **—** | 14.5 μs | 312.6 μs | 29.6 μs | — | **21.61x** | **2.05x** |
| tensor | medium_int32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 14.8 μs | 258.8 μs | 24.9 μs | — | **17.47x** | **1.68x** |
| tensor | medium_int32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 15.5 μs | 294.1 μs | 30.0 μs | — | **18.98x** | **1.94x** |
| tensor | medium_int32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 14.7 μs | 307.3 μs | 30.0 μs | — | **20.90x** | **2.04x** |
| tensor | medium_int64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 14.5 μs | 264.5 μs | 26.0 μs | — | **18.21x** | **1.79x** |
| tensor | medium_int64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 14.9 μs | 280.0 μs | 26.2 μs | — | **18.78x** | **1.75x** |
| tensor | medium_int64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 14.1 μs | 303.3 μs | 30.0 μs | — | **21.57x** | **2.13x** |
| tensor | medium_int8_1d:header_read | header_read | 0.10 MB | CPU | n/a | **—** | 15.4 μs | 315.3 μs | 33.8 μs | — | **20.45x** | **2.19x** |
| tensor | medium_int8_2d:header_read | header_read | 1.01 MB | CPU | n/a | **—** | 15.9 μs | 338.4 μs | 34.2 μs | — | **21.24x** | **2.14x** |
| tensor | medium_int8_3d:header_read | header_read | 1.57 MB | CPU | n/a | **—** | 15.6 μs | 361.6 μs | 35.2 μs | — | **23.25x** | **2.26x** |
| tensor | medium_uint16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 17.2 μs | 342.4 μs | 35.0 μs | — | **19.94x** | **2.04x** |
| tensor | medium_uint32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 17.6 μs | 333.0 μs | 36.7 μs | — | **18.97x** | **2.09x** |
| tensor | mef_medium:header_read | header_read | 7.02 MB | CPU | n/a | **—** | 17.0 μs | 535.0 μs | 43.8 μs | — | **31.53x** | **2.58x** |
| tensor | mef_small:header_read | header_read | 0.45 MB | CPU | n/a | **—** | 18.9 μs | 535.2 μs | 40.8 μs | — | **28.31x** | **2.16x** |
| tensor | multi_mef_10ext:cutout_100x100 | cutout_100x100 | 2.68 MB | CPU | n/a | **51.3 μs** | 87.6 μs | 2.20 ms | 157.7 μs | — | **42.92x** | **3.07x** |
| tensor | multi_mef_10ext:header_read | header_read | 2.68 MB | CPU | n/a | **—** | 17.2 μs | 528.6 μs | 45.7 μs | — | **30.75x** | **2.66x** |
| tensor | multi_mef_10ext:random_ext_full_reads_200 | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **5.55 ms** | 5.55 ms | 6.62 ms | 6.96 ms | — | **1.19x** | **1.25x** |
| tensor | repeated_cutouts_50x_100x100:repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **420.7 μs** | 450.6 μs | 51.43 ms | 2.91 ms | — | **122.24x** | **6.91x** |
| tensor | scaled_large:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 17.2 μs | 336.3 μs | 36.0 μs | — | **19.56x** | **2.09x** |
| tensor | scaled_medium:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 15.7 μs | 336.7 μs | 33.7 μs | — | **21.42x** | **2.15x** |
| tensor | scaled_small:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 16.3 μs | 330.4 μs | 34.5 μs | — | **20.26x** | **2.11x** |
| tensor | small_float32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 13.9 μs | 270.7 μs | 25.9 μs | — | **19.53x** | **1.87x** |
| tensor | small_float32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 15.1 μs | 286.9 μs | 29.0 μs | — | **18.98x** | **1.92x** |
| tensor | small_float32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 14.2 μs | 319.1 μs | 29.9 μs | — | **22.44x** | **2.10x** |
| tensor | small_float64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 16.9 μs | 327.9 μs | 30.2 μs | — | **19.37x** | **1.78x** |
| tensor | small_float64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 13.9 μs | 285.5 μs | 26.2 μs | — | **20.47x** | **1.88x** |
| tensor | small_float64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 15.9 μs | 303.8 μs | 30.9 μs | — | **19.13x** | **1.95x** |
| tensor | small_int16_1d:header_read | header_read | 22.5 KB | CPU | n/a | **—** | 13.4 μs | 254.3 μs | 25.5 μs | — | **18.92x** | **1.90x** |
| tensor | small_int16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 14.7 μs | 284.8 μs | 28.7 μs | — | **19.32x** | **1.95x** |
| tensor | small_int16_3d:header_read | header_read | 0.32 MB | CPU | n/a | **—** | 15.6 μs | 302.0 μs | 28.2 μs | — | **19.33x** | **1.80x** |
| tensor | small_int32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 14.2 μs | 269.4 μs | 25.7 μs | — | **19.00x** | **1.81x** |
| tensor | small_int32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 14.5 μs | 283.1 μs | 28.1 μs | — | **19.51x** | **1.93x** |
| tensor | small_int32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 15.7 μs | 315.9 μs | 28.7 μs | — | **20.16x** | **1.83x** |
| tensor | small_int64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 13.8 μs | 268.7 μs | 25.7 μs | — | **19.42x** | **1.86x** |
| tensor | small_int64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 15.2 μs | 291.0 μs | 25.4 μs | — | **19.19x** | **1.67x** |
| tensor | small_int64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 14.3 μs | 316.7 μs | 28.4 μs | — | **22.08x** | **1.98x** |
| tensor | small_int8_1d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 16.0 μs | 314.6 μs | 31.0 μs | — | **19.64x** | **1.94x** |
| tensor | small_int8_2d:header_read | header_read | 0.07 MB | CPU | n/a | **—** | 14.2 μs | 328.5 μs | 33.6 μs | — | **23.16x** | **2.37x** |
| tensor | small_int8_3d:header_read | header_read | 0.16 MB | CPU | n/a | **—** | 15.4 μs | 354.5 μs | 35.9 μs | — | **23.01x** | **2.33x** |
| tensor | small_uint16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 15.2 μs | 334.9 μs | 32.5 μs | — | **22.03x** | **2.14x** |
| tensor | small_uint32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.4 μs | 330.8 μs | 32.8 μs | — | **20.12x** | **1.99x** |
| tensor | timeseries_frame_000:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 15.6 μs | 288.6 μs | 27.6 μs | — | **18.44x** | **1.76x** |
| tensor | timeseries_frame_001:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 13.6 μs | 282.2 μs | 27.8 μs | — | **20.69x** | **2.04x** |
| tensor | timeseries_frame_002:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 14.6 μs | 291.1 μs | 27.4 μs | — | **19.90x** | **1.88x** |
| tensor | timeseries_frame_003:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 14.6 μs | 283.7 μs | 26.7 μs | — | **19.39x** | **1.83x** |
| tensor | timeseries_frame_004:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 15.4 μs | 283.4 μs | 27.5 μs | — | **18.41x** | **1.79x** |
| tensor | tiny_float32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 13.8 μs | 260.0 μs | 24.6 μs | — | **18.86x** | **1.78x** |
| tensor | tiny_float32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 13.9 μs | 282.0 μs | 24.6 μs | — | **20.22x** | **1.76x** |
| tensor | tiny_float32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 15.7 μs | 301.7 μs | 29.8 μs | — | **19.24x** | **1.90x** |
| tensor | tiny_float64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 14.9 μs | 260.8 μs | 24.8 μs | — | **17.47x** | **1.66x** |
| tensor | tiny_float64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 13.5 μs | 280.2 μs | 26.1 μs | — | **20.75x** | **1.94x** |
| tensor | tiny_float64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 13.9 μs | 304.4 μs | 27.9 μs | — | **21.84x** | **2.00x** |
| tensor | tiny_int16_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 14.3 μs | 263.6 μs | 25.5 μs | — | **18.39x** | **1.78x** |
| tensor | tiny_int16_2d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 14.4 μs | 291.1 μs | 28.1 μs | — | **20.21x** | **1.95x** |
| tensor | tiny_int16_3d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 14.9 μs | 303.0 μs | 30.3 μs | — | **20.29x** | **2.03x** |
| tensor | tiny_int32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 14.2 μs | 262.6 μs | 26.7 μs | — | **18.50x** | **1.88x** |
| tensor | tiny_int32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 14.6 μs | 275.9 μs | 27.5 μs | — | **18.93x** | **1.89x** |
| tensor | tiny_int32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 14.4 μs | 297.8 μs | 29.5 μs | — | **20.64x** | **2.04x** |
| tensor | tiny_int64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 13.7 μs | 264.7 μs | 25.4 μs | — | **19.38x** | **1.86x** |
| tensor | tiny_int64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 15.5 μs | 279.0 μs | 27.7 μs | — | **17.95x** | **1.78x** |
| tensor | tiny_int64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 14.7 μs | 306.9 μs | 30.2 μs | — | **20.94x** | **2.06x** |
| tensor | tiny_int8_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 17.1 μs | 304.0 μs | 31.6 μs | — | **17.82x** | **1.85x** |
| tensor | tiny_int8_2d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 15.4 μs | 327.7 μs | 34.9 μs | — | **21.29x** | **2.27x** |
| tensor | tiny_int8_3d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 15.3 μs | 345.8 μs | 35.7 μs | — | **22.58x** | **2.33x** |
| tensor | write_compress_hcompress_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **48.16 ms** | — | 58.81 ms | — | — | **1.22x** | **—** |
| tensor | write_compress_rice_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **37.70 ms** | — | 68.84 ms | — | — | **1.83x** | **—** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | off | **23.59 ms** | 23.55 ms | 45.45 ms | 26.42 ms | — | **1.93x** | **1.12x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | off | **20.27 ms** | 20.33 ms | 71.94 ms | 23.10 ms | — | **3.55x** | **1.14x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | off | **45.53 ms** | 45.73 ms | 51.80 ms | 44.06 ms | — | **1.14x** | **0.97x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | off | **12.45 ms** | 12.57 ms | 32.92 ms | 12.52 ms | — | **2.64x** | **1.01x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | off | **685.5 μs** | 678.4 μs | 1.68 ms | 1.14 ms | — | **2.48x** | **1.68x** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | off | **3.89 ms** | 3.88 ms | 15.55 ms | 5.01 ms | — | **4.00x** | **1.29x** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | off | **973.9 μs** | 1.39 ms | 1.91 ms | 1.24 ms | — | **1.96x** | **1.27x** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | off | **4.61 ms** | 4.66 ms | 9.82 ms | 4.97 ms | — | **2.13x** | **1.08x** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | off | **281.9 μs** | 282.4 μs | 737.7 μs | 345.9 μs | — | **2.62x** | **1.23x** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | off | **989.3 μs** | 1.02 ms | 3.68 ms | 1.29 ms | — | **3.72x** | **1.30x** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | off | **479.9 μs** | 487.1 μs | 1.13 ms | 739.3 μs | — | **2.36x** | **1.54x** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | off | **3.67 ms** | 1.92 ms | 9.47 ms | 3.09 ms | — | **4.92x** | **1.61x** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | off | **907.4 μs** | 885.6 μs | 1.91 ms | 1.23 ms | — | **2.15x** | **1.39x** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | off | **4.62 ms** | 4.81 ms | 9.77 ms | 4.94 ms | — | **2.11x** | **1.07x** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | off | **167.4 μs** | 165.7 μs | 651.5 μs | 170.2 μs | — | **3.93x** | **1.03x** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | off | **546.8 μs** | 724.5 μs | 1.52 ms | 648.4 μs | — | **2.79x** | **1.19x** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | off | **1.29 ms** | 1.31 ms | 4.06 ms | 1.60 ms | — | **3.14x** | **1.24x** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | off | **2.33 ms** | 4.15 ms | 6.81 ms | 3.68 ms | — | **2.92x** | **1.58x** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | off | **96.9 μs** | 76.6 μs | 373.0 μs | 111.5 μs | — | **4.87x** | **1.46x** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | off | **497.7 μs** | 515.8 μs | 1.20 ms | 772.3 μs | — | **2.41x** | **1.55x** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | off | **739.3 μs** | 752.4 μs | 1.67 ms | 1.17 ms | — | **2.26x** | **1.58x** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | off | **97.2 μs** | 130.9 μs | 453.5 μs | 148.1 μs | — | **4.66x** | **1.52x** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | off | **936.5 μs** | 919.8 μs | 2.54 ms | 1.30 ms | — | **2.76x** | **1.41x** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | off | **1.94 ms** | 1.43 ms | 4.23 ms | 2.03 ms | — | **2.96x** | **1.42x** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | off | **54.4 μs** | 57.9 μs | 306.7 μs | 64.5 μs | — | **5.64x** | **1.19x** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | off | **303.7 μs** | 295.9 μs | 776.1 μs | 364.6 μs | — | **2.62x** | **1.23x** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | off | **414.4 μs** | 545.7 μs | 1.02 ms | 545.2 μs | — | **2.46x** | **1.32x** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | off | **82.5 μs** | 46.5 μs | 377.9 μs | 105.9 μs | — | **8.12x** | **2.28x** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | off | **502.4 μs** | 508.8 μs | 1.20 ms | 783.0 μs | — | **2.38x** | **1.56x** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | off | **769.6 μs** | 767.1 μs | 1.69 ms | 1.19 ms | — | **2.21x** | **1.55x** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | off | **130.1 μs** | 134.1 μs | 463.4 μs | 150.6 μs | — | **3.56x** | **1.16x** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | off | **957.5 μs** | 956.2 μs | 2.56 ms | 1.31 ms | — | **2.68x** | **1.37x** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | off | **2.04 ms** | 1.92 ms | 4.30 ms | 2.09 ms | — | **2.25x** | **1.09x** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | off | **34.6 μs** | 50.9 μs | 386.6 μs | 54.0 μs | — | **11.17x** | **1.56x** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | off | **207.5 μs** | 178.0 μs | 676.8 μs | 179.0 μs | — | **3.80x** | **1.01x** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | off | **227.0 μs** | 255.7 μs | 858.9 μs | 278.6 μs | — | **3.78x** | **1.23x** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | off | **396.6 μs** | 370.0 μs | 1.34 ms | 440.9 μs | — | **3.63x** | **1.19x** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | off | **669.2 μs** | 687.6 μs | 1.80 ms | 962.2 μs | — | **2.69x** | **1.44x** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | off | **185.5 μs** | 188.6 μs | 866.8 μs | 222.3 μs | — | **4.67x** | **1.20x** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | off | **30.5 μs** | 58.4 μs | 587.1 μs | 80.1 μs | — | **19.23x** | **2.62x** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | off | **58.1 μs** | 35.6 μs | 585.6 μs | 126.4 μs | — | **16.46x** | **3.55x** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | off | **3.40 ms** | 3.50 ms | 5.53 ms | 3.38 ms | — | **1.63x** | **0.99x** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | off | **639.5 μs** | 655.6 μs | 1.42 ms | 789.0 μs | — | **2.21x** | **1.23x** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | off | **82.9 μs** | 73.0 μs | 460.6 μs | 93.4 μs | — | **6.31x** | **1.28x** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | off | **37.3 μs** | 30.2 μs | 268.3 μs | 45.3 μs | — | **8.89x** | **1.50x** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | off | **65.3 μs** | 36.2 μs | 353.8 μs | 80.7 μs | — | **9.77x** | **2.23x** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | off | **120.7 μs** | 117.9 μs | 457.3 μs | 148.7 μs | — | **3.88x** | **1.26x** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | off | **28.1 μs** | 34.9 μs | 280.8 μs | 48.2 μs | — | **9.98x** | **1.71x** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | off | **87.0 μs** | 88.3 μs | 421.6 μs | 111.9 μs | — | **4.85x** | **1.29x** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | off | **186.5 μs** | 199.3 μs | 624.8 μs | 228.7 μs | — | **3.35x** | **1.23x** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | off | **27.2 μs** | 41.6 μs | 265.4 μs | 40.4 μs | — | **9.75x** | **1.48x** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | off | **49.9 μs** | 53.0 μs | 311.3 μs | 56.1 μs | — | **6.24x** | **1.13x** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | off | **46.2 μs** | 92.9 μs | 388.2 μs | 87.4 μs | — | **8.41x** | **1.89x** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | off | **28.3 μs** | 42.2 μs | 269.5 μs | 45.1 μs | — | **9.52x** | **1.59x** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | off | **65.8 μs** | 69.3 μs | 342.3 μs | 81.5 μs | — | **5.20x** | **1.24x** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | off | **119.7 μs** | 121.5 μs | 453.0 μs | 151.8 μs | — | **3.79x** | **1.27x** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | off | **45.2 μs** | 30.1 μs | 269.9 μs | 51.0 μs | — | **8.97x** | **1.70x** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | off | **99.0 μs** | 80.9 μs | 404.8 μs | 111.4 μs | — | **5.00x** | **1.38x** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | off | **206.7 μs** | 188.4 μs | 622.6 μs | 230.3 μs | — | **3.30x** | **1.22x** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | off | **36.7 μs** | 34.7 μs | 366.0 μs | 43.3 μs | — | **10.55x** | **1.25x** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | off | **37.4 μs** | 49.0 μs | 399.6 μs | 54.2 μs | — | **10.68x** | **1.45x** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | off | **35.7 μs** | 66.0 μs | 442.4 μs | 63.7 μs | — | **12.39x** | **1.78x** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | off | **45.5 μs** | 58.2 μs | 396.5 μs | 59.0 μs | — | **8.72x** | **1.30x** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | off | **75.9 μs** | 78.0 μs | 424.7 μs | 83.8 μs | — | **5.59x** | **1.10x** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | off | **58.9 μs** | 58.1 μs | 360.7 μs | 82.9 μs | — | **6.21x** | **1.43x** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | off | **62.9 μs** | 64.1 μs | 353.7 μs | 83.8 μs | — | **5.62x** | **1.33x** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | off | **61.7 μs** | 40.2 μs | 353.1 μs | 84.1 μs | — | **8.79x** | **2.09x** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | off | **35.7 μs** | 65.4 μs | 357.3 μs | 81.0 μs | — | **10.01x** | **2.27x** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | off | **66.4 μs** | 63.1 μs | 360.0 μs | 82.2 μs | — | **5.71x** | **1.30x** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | off | **33.2 μs** | 33.5 μs | 258.8 μs | 41.0 μs | — | **7.79x** | **1.23x** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | off | **23.8 μs** | 35.1 μs | 277.5 μs | 43.4 μs | — | **11.67x** | **1.82x** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | off | **36.8 μs** | 24.4 μs | 305.2 μs | 44.8 μs | — | **12.49x** | **1.83x** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | off | **27.3 μs** | 36.4 μs | 261.5 μs | 38.8 μs | — | **9.56x** | **1.42x** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | off | **34.2 μs** | 40.3 μs | 281.9 μs | 41.4 μs | — | **8.25x** | **1.21x** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | off | **28.4 μs** | 38.4 μs | 302.0 μs | 49.4 μs | — | **10.62x** | **1.74x** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | off | **25.3 μs** | 25.3 μs | 265.5 μs | 38.1 μs | — | **10.51x** | **1.51x** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | off | **23.5 μs** | 37.8 μs | 270.3 μs | 37.4 μs | — | **11.50x** | **1.59x** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | off | **35.9 μs** | 37.0 μs | 290.2 μs | 43.0 μs | — | **8.08x** | **1.20x** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | off | **24.3 μs** | 35.5 μs | 261.0 μs | 39.5 μs | — | **10.73x** | **1.62x** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | off | **29.0 μs** | 38.1 μs | 277.4 μs | 42.0 μs | — | **9.57x** | **1.45x** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | off | **27.0 μs** | 37.4 μs | 299.5 μs | 43.9 μs | — | **11.09x** | **1.63x** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | off | **29.9 μs** | 24.0 μs | 260.9 μs | 40.2 μs | — | **10.89x** | **1.68x** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | off | **39.7 μs** | 39.2 μs | 279.3 μs | 45.3 μs | — | **7.12x** | **1.15x** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | off | **25.9 μs** | 28.6 μs | 292.1 μs | 43.1 μs | — | **11.29x** | **1.67x** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | off | **36.5 μs** | 28.5 μs | 377.3 μs | 44.0 μs | — | **13.22x** | **1.54x** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | off | **31.2 μs** | 38.8 μs | 383.3 μs | 43.4 μs | — | **12.30x** | **1.39x** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | off | **39.5 μs** | 26.6 μs | 389.2 μs | 46.0 μs | — | **14.65x** | **1.73x** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | on | **23.68 ms** | 23.51 ms | 45.61 ms | 26.41 ms | — | **1.94x** | **1.12x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | on | **20.16 ms** | 20.29 ms | 71.91 ms | 23.14 ms | — | **3.57x** | **1.15x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | on | **45.40 ms** | 45.54 ms | 51.52 ms | 43.60 ms | — | **1.13x** | **0.96x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | on | **12.38 ms** | 12.60 ms | 32.98 ms | 12.46 ms | — | **2.66x** | **1.01x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | on | **666.8 μs** | 662.9 μs | 1.49 ms | — | — | **2.24x** | **—** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | on | **3.77 ms** | 2.48 ms | 12.37 ms | — | — | **4.99x** | **—** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | on | **1.22 ms** | 1.22 ms | 2.42 ms | — | — | **1.99x** | **—** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | on | **6.35 ms** | 6.25 ms | 10.93 ms | — | — | **1.75x** | **—** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | on | **397.4 μs** | 454.3 μs | 1.02 ms | — | — | **2.58x** | **—** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | on | **1.28 ms** | 1.32 ms | 2.54 ms | — | — | **1.98x** | **—** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | on | **675.6 μs** | 648.4 μs | 1.48 ms | — | — | **2.29x** | **—** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | on | **3.55 ms** | 2.47 ms | 12.39 ms | — | — | **5.02x** | **—** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | on | **1.21 ms** | 1.29 ms | 2.42 ms | — | — | **2.00x** | **—** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | on | **6.03 ms** | 6.03 ms | 10.86 ms | — | — | **1.80x** | **—** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | on | **247.7 μs** | 248.2 μs | — | — | — | **—** | **—** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | on | **782.3 μs** | 752.4 μs | — | — | — | **—** | **—** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | on | **3.16 ms** | 3.16 ms | — | — | — | **—** | **—** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | on | **5.35 ms** | 5.36 ms | — | — | — | **—** | **—** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | on | **118.2 μs** | 135.3 μs | 599.7 μs | — | — | **5.07x** | **—** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | on | **689.4 μs** | 776.9 μs | 1.55 ms | — | — | **2.25x** | **—** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | on | **1.02 ms** | 1.02 ms | 2.13 ms | — | — | **2.08x** | **—** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | on | **172.2 μs** | 169.5 μs | 708.1 μs | — | — | **4.18x** | **—** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | on | **1.33 ms** | 1.28 ms | 2.54 ms | — | — | **1.99x** | **—** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | on | **1.99 ms** | 1.96 ms | 3.66 ms | — | — | **1.87x** | **—** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | on | **109.0 μs** | 108.4 μs | 536.2 μs | — | — | **4.94x** | **—** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | on | **405.4 μs** | 404.3 μs | 1.08 ms | — | — | **2.67x** | **—** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | on | **575.8 μs** | 543.9 μs | 1.37 ms | — | — | **2.53x** | **—** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | on | **146.7 μs** | 96.7 μs | 590.7 μs | — | — | **6.11x** | **—** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | on | **819.4 μs** | 697.8 μs | 1.58 ms | — | — | **2.26x** | **—** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | on | **1.04 ms** | 1.01 ms | 2.12 ms | — | — | **2.11x** | **—** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | on | **195.7 μs** | 211.3 μs | 728.7 μs | — | — | **3.72x** | **—** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | on | **1.34 ms** | 1.27 ms | 2.55 ms | — | — | **2.01x** | **—** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | on | **1.92 ms** | 2.44 ms | 3.66 ms | — | — | **1.91x** | **—** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | on | **57.6 μs** | 83.4 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | on | **260.7 μs** | 212.7 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | on | **371.2 μs** | 365.7 μs | — | — | — | **—** | **—** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | on | **892.3 μs** | 842.2 μs | — | — | — | **—** | **—** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | on | **1.19 ms** | 1.19 ms | — | — | — | **—** | **—** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | on | **280.9 μs** | 269.5 μs | — | — | — | **—** | **—** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | on | **102.1 μs** | 51.4 μs | — | — | — | **—** | **—** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | on | **90.7 μs** | 82.1 μs | — | — | — | **—** | **—** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | on | **5.48 ms** | 5.46 ms | — | — | — | **—** | **—** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | on | **1.09 ms** | 1.08 ms | — | — | — | **—** | **—** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | on | **101.4 μs** | 113.8 μs | — | — | — | **—** | **—** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | on | **55.4 μs** | 48.6 μs | 459.3 μs | — | — | **9.45x** | **—** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | on | **94.5 μs** | 66.6 μs | 584.0 μs | — | — | **8.76x** | **—** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | on | **154.4 μs** | 151.5 μs | 711.3 μs | — | — | **4.70x** | **—** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | on | **66.8 μs** | 90.2 μs | 483.9 μs | — | — | **7.25x** | **—** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | on | **135.8 μs** | 118.4 μs | 653.7 μs | — | — | **5.52x** | **—** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | on | **252.6 μs** | 258.9 μs | 904.1 μs | — | — | **3.58x** | **—** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | on | **76.7 μs** | 52.5 μs | 454.4 μs | — | — | **8.65x** | **—** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | on | **93.7 μs** | 105.5 μs | 530.2 μs | — | — | **5.66x** | **—** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | on | **125.7 μs** | 127.0 μs | 613.9 μs | — | — | **4.88x** | **—** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | on | **49.7 μs** | 73.8 μs | 469.1 μs | — | — | **9.44x** | **—** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | on | **106.3 μs** | 101.6 μs | 589.4 μs | — | — | **5.80x** | **—** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | on | **137.1 μs** | 172.4 μs | 707.4 μs | — | — | **5.16x** | **—** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | on | **66.5 μs** | 79.5 μs | 483.6 μs | — | — | **7.27x** | **—** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | on | **153.6 μs** | 127.6 μs | 666.5 μs | — | — | **5.22x** | **—** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | on | **270.0 μs** | 302.9 μs | 914.5 μs | — | — | **3.39x** | **—** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | on | **47.2 μs** | 75.1 μs | — | — | — | **—** | **—** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | on | **60.1 μs** | 97.0 μs | — | — | — | **—** | **—** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | on | **62.2 μs** | 134.0 μs | — | — | — | **—** | **—** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | on | **154.0 μs** | 205.1 μs | — | — | — | **—** | **—** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | on | **209.7 μs** | 144.8 μs | — | — | — | **—** | **—** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | on | **133.4 μs** | 113.6 μs | 736.4 μs | — | — | **6.48x** | **—** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | on | **149.7 μs** | 97.8 μs | 744.3 μs | — | — | **7.61x** | **—** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | on | **131.9 μs** | 85.0 μs | 725.3 μs | — | — | **8.54x** | **—** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | on | **80.4 μs** | 135.4 μs | 726.7 μs | — | — | **9.04x** | **—** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | on | **116.3 μs** | 147.0 μs | 724.4 μs | — | — | **6.23x** | **—** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | on | **42.9 μs** | 62.1 μs | 464.2 μs | — | — | **10.82x** | **—** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | on | **70.2 μs** | 49.3 μs | 529.3 μs | — | — | **10.73x** | **—** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | on | **78.6 μs** | 69.9 μs | 541.2 μs | — | — | **7.74x** | **—** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | on | **48.7 μs** | 42.8 μs | 472.7 μs | — | — | **11.05x** | **—** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | on | **78.1 μs** | 55.8 μs | 538.8 μs | — | — | **9.65x** | **—** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | on | **44.1 μs** | 45.7 μs | 575.3 μs | — | — | **13.03x** | **—** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | on | **33.0 μs** | 46.0 μs | 266.6 μs | — | — | **8.08x** | **—** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | on | **43.3 μs** | 33.5 μs | 285.0 μs | — | — | **8.51x** | **—** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | on | **27.9 μs** | 50.1 μs | 293.1 μs | — | — | **10.50x** | **—** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | on | **30.3 μs** | 36.5 μs | 269.7 μs | — | — | **8.91x** | **—** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | on | **26.3 μs** | 42.0 μs | 287.7 μs | — | — | **10.92x** | **—** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | on | **37.8 μs** | 27.0 μs | 298.5 μs | — | — | **11.06x** | **—** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | on | **33.2 μs** | 42.6 μs | 263.4 μs | — | — | **7.94x** | **—** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | on | **36.9 μs** | 48.9 μs | 292.5 μs | — | — | **7.92x** | **—** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | on | **36.6 μs** | 49.4 μs | 314.1 μs | — | — | **8.59x** | **—** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | on | **42.5 μs** | 30.3 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | on | **33.0 μs** | 48.6 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | on | **32.5 μs** | 30.8 μs | — | — | — | **—** | **—** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **311.1 μs** | 319.9 μs | 2.68 ms | 366.1 μs | — | **8.60x** | **1.18x** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | off | **325.3 μs** | 334.3 μs | 2.66 ms | 366.5 μs | — | **8.18x** | **1.13x** |
| table | ascii_10000 | projection | 0.44 MB | CPU | off | **1.05 ms** | 1.03 ms | 8.05 ms | 1.98 ms | — | **7.83x** | **1.92x** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | off | **1.05 ms** | 1.01 ms | 8.05 ms | 1.98 ms | — | **7.94x** | **1.96x** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | off | **201.6 μs** | 171.5 μs | 2.61 ms | 527.3 μs | — | **15.23x** | **3.08x** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | off | **28.0 μs** | 29.4 μs | 412.8 μs | 60.1 μs | — | **14.76x** | **2.15x** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **114.9 μs** | 123.8 μs | 1.52 ms | 165.7 μs | — | **13.21x** | **1.44x** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | off | **128.3 μs** | 130.0 μs | 1.51 ms | 167.6 μs | — | **11.78x** | **1.31x** |
| table | ascii_1000 | projection | 50.6 KB | CPU | off | **190.6 μs** | 172.6 μs | 2.19 ms | 343.3 μs | — | **12.71x** | **1.99x** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | off | **189.0 μs** | 178.6 μs | 2.19 ms | 325.9 μs | — | **12.24x** | **1.82x** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | off | **111.0 μs** | 103.1 μs | 1.96 ms | 194.1 μs | — | **19.00x** | **1.88x** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | off | **28.4 μs** | 28.1 μs | 419.9 μs | 65.3 μs | — | **14.97x** | **2.33x** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **9.66 ms** | 9.57 ms | 15.74 ms | 19.88 ms | — | **1.64x** | **2.08x** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | off | **9.67 ms** | 9.68 ms | 12.38 ms | 16.29 ms | — | **1.28x** | **1.69x** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | off | **10.30 ms** | 11.97 ms | 17.75 ms | 31.19 ms | — | **1.72x** | **3.03x** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | off | **28.99 ms** | 30.17 ms | 329.65 ms | 111.90 ms | — | **11.37x** | **3.86x** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **290.9 μs** | 279.0 μs | 12.79 ms | 1.53 ms | — | **45.83x** | **5.47x** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **31.7 μs** | 31.6 μs | 489.0 μs | 76.6 μs | — | **15.49x** | **2.43x** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **1.09 ms** | 1.09 ms | 3.18 ms | 2.19 ms | — | **2.92x** | **2.01x** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | off | **1.08 ms** | 1.09 ms | 2.86 ms | 1.81 ms | — | **2.64x** | **1.68x** |
| table | mixed_100000 | projection | 5.06 MB | CPU | off | **1.26 ms** | 1.23 ms | 3.25 ms | 3.21 ms | — | **2.64x** | **2.60x** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | off | **2.33 ms** | 2.31 ms | 30.73 ms | 10.17 ms | — | **13.31x** | **4.41x** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | off | **298.8 μs** | 286.3 μs | 5.98 ms | 1.52 ms | — | **20.88x** | **5.32x** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | off | **32.3 μs** | 30.2 μs | 472.8 μs | 78.5 μs | — | **15.65x** | **2.60x** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **157.1 μs** | 194.7 μs | 1.97 ms | 379.1 μs | — | **12.52x** | **2.41x** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | off | **152.5 μs** | 197.1 μs | 1.93 ms | 337.1 μs | — | **12.66x** | **2.21x** |
| table | mixed_10000 | projection | 0.51 MB | CPU | off | **181.5 μs** | 155.5 μs | 1.97 ms | 470.4 μs | — | **12.64x** | **3.02x** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | off | **290.8 μs** | 270.2 μs | 4.62 ms | 1.10 ms | — | **17.09x** | **4.07x** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | off | **143.3 μs** | 120.0 μs | 3.00 ms | 326.9 μs | — | **24.99x** | **2.72x** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | off | **28.7 μs** | 27.8 μs | 440.4 μs | 73.9 μs | — | **15.85x** | **2.66x** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **63.3 μs** | 112.0 μs | 1.82 ms | 187.8 μs | — | **28.80x** | **2.96x** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | off | **59.0 μs** | 111.7 μs | 1.83 ms | 182.4 μs | — | **31.01x** | **3.09x** |
| table | mixed_1000 | projection | 0.06 MB | CPU | off | **115.8 μs** | 90.6 μs | 1.84 ms | 196.9 μs | — | **20.28x** | **2.17x** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | off | **133.0 μs** | 118.6 μs | 2.19 ms | 270.6 μs | — | **18.48x** | **2.28x** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | off | **125.7 μs** | 112.9 μs | 2.67 ms | 203.1 μs | — | **23.64x** | **1.80x** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | off | **30.1 μs** | 27.2 μs | 445.9 μs | 72.3 μs | — | **16.41x** | **2.66x** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **4.96 ms** | 4.88 ms | 8.62 ms | 14.65 ms | — | **1.77x** | **3.00x** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | off | **4.92 ms** | 4.92 ms | 5.23 ms | 11.10 ms | — | **1.06x** | **2.26x** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | off | **4.17 ms** | 4.34 ms | 5.73 ms | 23.65 ms | — | **1.37x** | **5.67x** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | off | **5.90 ms** | 5.85 ms | 7.04 ms | 5.81 ms | — | **1.20x** | **0.99x** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **157.3 μs** | 152.1 μs | 3.95 ms | 583.5 μs | — | **25.93x** | **3.84x** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **28.0 μs** | 29.8 μs | 476.0 μs | 62.9 μs | — | **17.01x** | **2.25x** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **615.1 μs** | 624.3 μs | 2.17 ms | 1.64 ms | — | **3.52x** | **2.66x** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | off | **595.5 μs** | 620.2 μs | 1.84 ms | 1.27 ms | — | **3.08x** | **2.13x** |
| table | narrow_100000 | projection | 1.25 MB | CPU | off | **551.4 μs** | 512.1 μs | 1.86 ms | 2.56 ms | — | **3.64x** | **5.00x** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | off | **680.7 μs** | 681.3 μs | 1.99 ms | 678.0 μs | — | **2.92x** | **1.00x** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | off | **164.6 μs** | 152.6 μs | 2.13 ms | 587.8 μs | — | **13.94x** | **3.85x** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | off | **26.5 μs** | 29.3 μs | 429.5 μs | 63.8 μs | — | **16.21x** | **2.41x** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **104.2 μs** | 150.2 μs | 1.40 ms | 304.6 μs | — | **13.45x** | **2.92x** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | off | **102.9 μs** | 154.6 μs | 1.37 ms | 262.0 μs | — | **13.28x** | **2.55x** |
| table | narrow_10000 | projection | 0.13 MB | CPU | off | **139.1 μs** | 121.7 μs | 1.38 ms | 388.1 μs | — | **11.38x** | **3.19x** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | off | **154.6 μs** | 140.2 μs | 1.42 ms | 189.6 μs | — | **10.10x** | **1.35x** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | off | **107.2 μs** | 94.5 μs | 1.79 ms | 202.7 μs | — | **18.97x** | **2.15x** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | off | **26.3 μs** | 26.6 μs | 433.9 μs | 64.3 μs | — | **16.51x** | **2.45x** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **59.1 μs** | 107.0 μs | 1.31 ms | 169.0 μs | — | **22.09x** | **2.86x** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | off | **55.9 μs** | 107.7 μs | 1.32 ms | 163.4 μs | — | **23.70x** | **2.92x** |
| table | narrow_1000 | projection | 19.7 KB | CPU | off | **98.9 μs** | 82.2 μs | 1.31 ms | 176.9 μs | — | **15.95x** | **2.15x** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | off | **107.2 μs** | 93.5 μs | 1.36 ms | 147.7 μs | — | **14.49x** | **1.58x** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | off | **104.6 μs** | 89.3 μs | 1.78 ms | 153.7 μs | — | **19.92x** | **1.72x** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | off | **26.5 μs** | 24.9 μs | 425.2 μs | 66.5 μs | — | **17.07x** | **2.67x** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **743.5 μs** | 765.0 μs | 1.92 ms | 1.39 ms | — | **2.58x** | **1.86x** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | off | **750.3 μs** | 786.0 μs | 1.94 ms | 1.37 ms | — | **2.59x** | **1.83x** |
| table | typed_100000 | projection | 2.39 MB | CPU | off | **3.59 ms** | 3.56 ms | 28.95 ms | 13.11 ms | — | **8.14x** | **3.69x** |
| table | typed_100000 | read_full | 2.39 MB | CPU | off | **5.32 ms** | 5.29 ms | 29.19 ms | 14.15 ms | — | **5.52x** | **2.67x** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | off | **644.8 μs** | 627.2 μs | 4.78 ms | 1.92 ms | — | **7.62x** | **3.06x** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | off | **31.5 μs** | 33.0 μs | 449.3 μs | 66.6 μs | — | **14.25x** | **2.11x** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **123.3 μs** | 166.8 μs | 1.47 ms | 282.3 μs | — | **11.89x** | **2.29x** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | off | **123.6 μs** | 164.9 μs | 1.48 ms | 286.5 μs | — | **12.01x** | **2.32x** |
| table | typed_10000 | projection | 0.24 MB | CPU | off | **471.3 μs** | 443.0 μs | 4.09 ms | 1.46 ms | — | **9.24x** | **3.29x** |
| table | typed_10000 | read_full | 0.24 MB | CPU | off | **637.1 μs** | 613.1 μs | 4.15 ms | 1.60 ms | — | **6.76x** | **2.60x** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | off | **161.2 μs** | 149.7 μs | 2.21 ms | 397.3 μs | — | **14.76x** | **2.65x** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | off | **30.5 μs** | 28.3 μs | 453.4 μs | 63.0 μs | — | **16.05x** | **2.23x** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **641.2 μs** | 650.2 μs | 1.79 ms | 1.25 ms | — | **2.79x** | **1.94x** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | off | **626.2 μs** | 649.5 μs | 1.78 ms | 1.25 ms | — | **2.85x** | **2.00x** |
| table | varlen_100000 | projection | 3.06 MB | CPU | off | **75.10 ms** | 11.08 ms | 525.29 ms | 107.81 ms | — | **47.42x** | **9.73x** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | off | **75.18 ms** | 11.08 ms | 525.26 ms | 107.73 ms | — | **47.39x** | **9.72x** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | off | **7.57 ms** | 1.11 ms | 54.08 ms | 11.88 ms | — | **48.76x** | **10.71x** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | off | **27.2 μs** | 29.7 μs | 472.6 μs | 61.3 μs | — | **17.37x** | **2.26x** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **106.2 μs** | 164.9 μs | 1.36 ms | 272.9 μs | — | **12.81x** | **2.57x** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | off | **108.0 μs** | 166.2 μs | 1.37 ms | 267.3 μs | — | **12.66x** | **2.47x** |
| table | varlen_10000 | projection | 0.31 MB | CPU | off | **7.38 ms** | 1.12 ms | 53.45 ms | 10.85 ms | — | **47.73x** | **9.69x** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | off | **7.32 ms** | 1.12 ms | 53.33 ms | 10.72 ms | — | **47.74x** | **9.60x** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | off | **829.6 μs** | 197.1 μs | 7.03 ms | 1.38 ms | — | **35.67x** | **7.01x** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | off | **29.6 μs** | 27.7 μs | 456.6 μs | 63.3 μs | — | **16.51x** | **2.29x** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **56.8 μs** | 110.5 μs | 1.27 ms | 165.6 μs | — | **22.42x** | **2.91x** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | off | **57.4 μs** | 106.8 μs | 1.28 ms | 165.3 μs | — | **22.30x** | **2.88x** |
| table | varlen_1000 | projection | 39.4 KB | CPU | off | **812.5 μs** | 195.0 μs | 6.58 ms | 1.28 ms | — | **33.76x** | **6.57x** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | off | **810.9 μs** | 204.5 μs | 6.61 ms | 1.25 ms | — | **32.32x** | **6.09x** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | off | **210.5 μs** | 116.4 μs | 2.23 ms | 294.1 μs | — | **19.15x** | **2.53x** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | off | **25.9 μs** | 22.9 μs | 436.2 μs | 58.7 μs | — | **19.01x** | **2.56x** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **2.97 ms** | 3.01 ms | 8.25 ms | 4.24 ms | — | **2.78x** | **1.43x** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | off | **2.98 ms** | 2.99 ms | 7.92 ms | 3.87 ms | — | **2.66x** | **1.30x** |
| table | wide_100000 | projection | 20.71 MB | CPU | off | **2.98 ms** | 2.95 ms | 8.34 ms | 5.18 ms | — | **2.82x** | **1.75x** |
| table | wide_100000 | read_full | 20.71 MB | CPU | off | **17.19 ms** | 19.85 ms | 129.86 ms | 41.34 ms | — | **7.56x** | **2.41x** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | off | **1.43 ms** | 1.46 ms | 21.63 ms | 5.17 ms | — | **15.10x** | **3.61x** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | off | **39.9 μs** | 39.5 μs | 569.8 μs | 254.7 μs | — | **14.42x** | **6.45x** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **404.3 μs** | 436.9 μs | 5.94 ms | 761.4 μs | — | **14.70x** | **1.88x** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | off | **395.3 μs** | 441.0 μs | 5.92 ms | 730.6 μs | — | **14.97x** | **1.85x** |
| table | wide_10000 | projection | 2.08 MB | CPU | off | **416.2 μs** | 395.3 μs | 5.94 ms | 860.0 μs | — | **15.03x** | **2.18x** |
| table | wide_10000 | read_full | 2.08 MB | CPU | off | **1.44 ms** | 1.45 ms | 16.78 ms | 4.47 ms | — | **11.62x** | **3.10x** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | off | **503.9 μs** | 516.3 μs | 10.63 ms | 892.8 μs | — | **21.09x** | **1.77x** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | off | **38.7 μs** | 37.1 μs | 571.4 μs | 251.8 μs | — | **15.41x** | **6.79x** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **111.3 μs** | 168.2 μs | 5.74 ms | 389.8 μs | — | **51.57x** | **3.50x** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | off | **108.3 μs** | 166.1 μs | 5.68 ms | 389.8 μs | — | **52.43x** | **3.60x** |
| table | wide_1000 | projection | 0.22 MB | CPU | off | **161.9 μs** | 144.4 μs | 5.67 ms | 403.1 μs | — | **39.28x** | **2.79x** |
| table | wide_1000 | read_full | 0.22 MB | CPU | off | **496.7 μs** | 509.4 μs | 7.15 ms | 826.5 μs | — | **14.39x** | **1.66x** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | off | **425.9 μs** | 451.4 μs | 9.44 ms | 504.0 μs | — | **22.16x** | **1.18x** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | off | **37.2 μs** | 39.5 μs | 554.8 μs | 251.9 μs | — | **14.92x** | **6.77x** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **—** | — | 2.64 ms | — | — | **—** | **—** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | on | **—** | — | 2.64 ms | — | — | **—** | **—** |
| table | ascii_10000 | projection | 0.44 MB | CPU | on | **1.08 ms** | 1.13 ms | 8.10 ms | — | — | **7.52x** | **—** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | on | **1.07 ms** | 1.11 ms | 8.08 ms | — | — | **7.52x** | **—** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | on | **230.3 μs** | 274.0 μs | 2.60 ms | — | — | **11.27x** | **—** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | on | **27.8 μs** | 28.3 μs | 418.2 μs | — | — | **15.02x** | **—** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **—** | — | 1.49 ms | — | — | **—** | **—** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | on | **—** | — | 1.50 ms | — | — | **—** | **—** |
| table | ascii_1000 | projection | 50.6 KB | CPU | on | **234.2 μs** | 277.2 μs | 2.20 ms | — | — | **9.38x** | **—** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | on | **235.6 μs** | 280.8 μs | 2.19 ms | — | — | **9.28x** | **—** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | on | **159.2 μs** | 201.0 μs | 1.98 ms | — | — | **12.44x** | **—** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | on | **25.4 μs** | 26.1 μs | 423.2 μs | — | — | **16.67x** | **—** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **5.57 ms** | 4.94 ms | 12.07 ms | — | — | **2.44x** | **—** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | on | **4.08 ms** | 4.24 ms | 8.83 ms | — | — | **2.16x** | **—** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | on | **7.20 ms** | 8.04 ms | 13.07 ms | — | — | **1.81x** | **—** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | on | **26.69 ms** | 23.65 ms | 319.17 ms | — | — | **13.49x** | **—** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **273.1 μs** | 218.2 μs | 9.04 ms | — | — | **41.43x** | **—** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **32.2 μs** | 29.5 μs | 462.3 μs | — | — | **15.68x** | **—** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **1.00 ms** | 728.5 μs | 2.93 ms | — | — | **4.03x** | **—** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | on | **600.0 μs** | 614.3 μs | 2.62 ms | — | — | **4.36x** | **—** |
| table | mixed_100000 | projection | 5.06 MB | CPU | on | **865.5 μs** | 807.3 μs | 3.06 ms | — | — | **3.78x** | **—** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | on | **1.87 ms** | 1.80 ms | 30.57 ms | — | — | **17.03x** | **—** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | on | **267.9 μs** | 205.2 μs | 5.76 ms | — | — | **28.08x** | **—** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | on | **30.5 μs** | 28.9 μs | 443.1 μs | — | — | **15.33x** | **—** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **206.4 μs** | 233.0 μs | 1.97 ms | — | — | **9.54x** | **—** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | on | **140.2 μs** | 175.3 μs | 1.93 ms | — | — | **13.73x** | **—** |
| table | mixed_10000 | projection | 0.51 MB | CPU | on | **171.6 μs** | 126.1 μs | 1.95 ms | — | — | **15.45x** | **—** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | on | **259.7 μs** | 200.4 μs | 4.60 ms | — | — | **22.97x** | **—** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | on | **166.9 μs** | 113.1 μs | 3.01 ms | — | — | **26.64x** | **—** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | on | **28.8 μs** | 29.2 μs | 460.1 μs | — | — | **15.96x** | **—** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **139.0 μs** | 202.9 μs | 3.19 ms | — | — | **22.93x** | **—** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | on | **119.5 μs** | 199.3 μs | 3.16 ms | — | — | **26.45x** | **—** |
| table | mixed_1000 | projection | 0.06 MB | CPU | on | **209.9 μs** | 134.5 μs | 3.18 ms | — | — | **23.66x** | **—** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | on | **245.0 μs** | 179.1 μs | 3.80 ms | — | — | **21.22x** | **—** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | on | **236.0 μs** | 164.5 μs | 4.70 ms | — | — | **28.60x** | **—** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | on | **50.8 μs** | 47.8 μs | 791.6 μs | — | — | **16.56x** | **—** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **3.36 ms** | 2.42 ms | 8.14 ms | — | — | **3.37x** | **—** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | on | **1.63 ms** | 1.61 ms | 4.68 ms | — | — | **2.91x** | **—** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | on | **2.18 ms** | 2.12 ms | 5.22 ms | — | — | **2.47x** | **—** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | on | **3.24 ms** | 3.16 ms | 6.54 ms | — | — | **2.07x** | **—** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **173.2 μs** | 122.4 μs | 3.42 ms | — | — | **27.97x** | **—** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **32.3 μs** | 34.8 μs | 455.2 μs | — | — | **14.11x** | **—** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **788.7 μs** | 509.0 μs | 2.08 ms | — | — | **4.09x** | **—** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | on | **347.9 μs** | 369.1 μs | 1.76 ms | — | — | **5.06x** | **—** |
| table | narrow_100000 | projection | 1.25 MB | CPU | on | **353.1 μs** | 296.6 μs | 1.77 ms | — | — | **5.95x** | **—** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | on | **448.6 μs** | 389.1 μs | 1.90 ms | — | — | **4.88x** | **—** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | on | **162.6 μs** | 111.7 μs | 2.08 ms | — | — | **18.58x** | **—** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | on | **27.5 μs** | 30.5 μs | 445.5 μs | — | — | **16.23x** | **—** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **180.5 μs** | 211.1 μs | 1.45 ms | — | — | **8.03x** | **—** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | on | **124.0 μs** | 159.9 μs | 1.43 ms | — | — | **11.49x** | **—** |
| table | narrow_10000 | projection | 0.13 MB | CPU | on | **147.8 μs** | 96.2 μs | 1.42 ms | — | — | **14.71x** | **—** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | on | **153.2 μs** | 115.3 μs | 1.45 ms | — | — | **12.61x** | **—** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | on | **138.0 μs** | 86.7 μs | 1.86 ms | — | — | **21.46x** | **—** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | on | **27.2 μs** | 26.3 μs | 448.6 μs | — | — | **17.04x** | **—** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **133.9 μs** | 198.0 μs | 2.31 ms | — | — | **17.24x** | **—** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | on | **121.9 μs** | 189.2 μs | 2.32 ms | — | — | **19.04x** | **—** |
| table | narrow_1000 | projection | 19.7 KB | CPU | on | **196.3 μs** | 121.1 μs | 2.30 ms | — | — | **18.97x** | **—** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | on | **202.5 μs** | 125.9 μs | 2.37 ms | — | — | **18.87x** | **—** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | on | **201.7 μs** | 123.5 μs | 3.07 ms | — | — | **24.87x** | **—** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | on | **45.7 μs** | 48.5 μs | 778.6 μs | — | — | **17.03x** | **—** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **666.8 μs** | 508.7 μs | 1.80 ms | — | — | **3.53x** | **—** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | on | **524.6 μs** | 539.9 μs | 1.79 ms | — | — | **3.41x** | **—** |
| table | typed_100000 | projection | 2.39 MB | CPU | on | **1.24 ms** | 1.17 ms | 29.00 ms | — | — | **24.82x** | **—** |
| table | typed_100000 | read_full | 2.39 MB | CPU | on | **1.36 ms** | 1.27 ms | 29.20 ms | — | — | **22.93x** | **—** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | on | **265.1 μs** | 213.4 μs | 4.55 ms | — | — | **21.30x** | **—** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | on | **31.5 μs** | 28.2 μs | 443.1 μs | — | — | **15.70x** | **—** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **165.7 μs** | 203.9 μs | 1.45 ms | — | — | **8.76x** | **—** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | on | **161.4 μs** | 199.8 μs | 1.46 ms | — | — | **9.02x** | **—** |
| table | typed_10000 | projection | 0.24 MB | CPU | on | **249.0 μs** | 198.9 μs | 4.05 ms | — | — | **20.38x** | **—** |
| table | typed_10000 | read_full | 0.24 MB | CPU | on | **260.3 μs** | 203.0 μs | 4.10 ms | — | — | **20.20x** | **—** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | on | **149.6 μs** | 103.0 μs | 2.18 ms | — | — | **21.16x** | **—** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | on | **28.8 μs** | 28.5 μs | 434.0 μs | — | — | **15.21x** | **—** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **635.0 μs** | 455.2 μs | 1.59 ms | — | — | **3.49x** | **—** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | on | **430.7 μs** | 455.4 μs | 1.59 ms | — | — | **3.68x** | **—** |
| table | varlen_100000 | projection | 3.06 MB | CPU | on | **73.93 ms** | 74.13 ms | 524.24 ms | — | — | **7.09x** | **—** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | on | **74.49 ms** | 74.30 ms | 526.65 ms | — | — | **7.09x** | **—** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | on | **7.45 ms** | 7.34 ms | 54.12 ms | — | — | **7.38x** | **—** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | on | **28.6 μs** | 26.3 μs | 445.3 μs | — | — | **16.93x** | **—** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **165.0 μs** | 192.6 μs | 1.36 ms | — | — | **8.22x** | **—** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | on | **156.7 μs** | 191.4 μs | 1.36 ms | — | — | **8.68x** | **—** |
| table | varlen_10000 | projection | 0.31 MB | CPU | on | **7.23 ms** | 7.26 ms | 53.39 ms | — | — | **7.38x** | **—** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | on | **7.25 ms** | 7.24 ms | 53.36 ms | — | — | **7.37x** | **—** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | on | **880.2 μs** | 914.8 μs | 6.98 ms | — | — | **7.93x** | **—** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | on | **28.7 μs** | 27.5 μs | 459.2 μs | — | — | **16.72x** | **—** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **75.7 μs** | 135.3 μs | 1.30 ms | — | — | **17.16x** | **—** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | on | **79.8 μs** | 135.3 μs | 1.30 ms | — | — | **16.24x** | **—** |
| table | varlen_1000 | projection | 39.4 KB | CPU | on | **867.3 μs** | 917.8 μs | 6.55 ms | — | — | **7.55x** | **—** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | on | **867.6 μs** | 901.5 μs | 6.56 ms | — | — | **7.56x** | **—** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | on | **244.1 μs** | 294.6 μs | 2.26 ms | — | — | **9.27x** | **—** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | on | **29.8 μs** | 26.1 μs | 453.7 μs | — | — | **17.39x** | **—** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **1.55 ms** | 1.32 ms | 7.40 ms | — | — | **5.59x** | **—** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | on | **1.17 ms** | 1.18 ms | 7.10 ms | — | — | **6.06x** | **—** |
| table | wide_100000 | projection | 20.71 MB | CPU | on | **1.62 ms** | 1.55 ms | 7.54 ms | — | — | **4.85x** | **—** |
| table | wide_100000 | read_full | 20.71 MB | CPU | on | **13.95 ms** | 13.88 ms | 129.43 ms | — | — | **9.32x** | **—** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | on | **1.05 ms** | 1.00 ms | 20.85 ms | — | — | **20.79x** | **—** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | on | **43.5 μs** | 39.2 μs | 573.0 μs | — | — | **14.62x** | **—** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **313.7 μs** | 326.8 μs | 5.87 ms | — | — | **18.70x** | **—** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | on | **260.9 μs** | 279.0 μs | 5.83 ms | — | — | **22.35x** | **—** |
| table | wide_10000 | projection | 2.08 MB | CPU | on | **288.9 μs** | 240.1 μs | 5.86 ms | — | — | **24.43x** | **—** |
| table | wide_10000 | read_full | 2.08 MB | CPU | on | **1.07 ms** | 1.02 ms | 16.70 ms | — | — | **16.40x** | **—** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | on | **499.3 μs** | 455.0 μs | 10.49 ms | — | — | **23.06x** | **—** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | on | **39.8 μs** | 39.4 μs | 573.7 μs | — | — | **14.56x** | **—** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **124.7 μs** | 178.0 μs | 5.68 ms | — | — | **45.57x** | **—** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | on | **125.7 μs** | 174.0 μs | 5.67 ms | — | — | **45.15x** | **—** |
| table | wide_1000 | projection | 0.22 MB | CPU | on | **174.8 μs** | 127.2 μs | 5.68 ms | — | — | **44.68x** | **—** |
| table | wide_1000 | read_full | 0.22 MB | CPU | on | **502.8 μs** | 452.6 μs | 7.17 ms | — | — | **15.84x** | **—** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | on | **453.6 μs** | 411.5 μs | 9.44 ms | — | — | **22.94x** | **—** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | on | **39.2 μs** | 39.8 μs | 582.2 μs | — | — | **14.85x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (CPU and GPU). GPU lags may reflect software or hardware limits — they are listed, not hidden.

| Platform | Domain | Case | mmap | torchfits | Peak RSS (MB) | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 45.40 ms | 289.0 | fitsio/fitsio_torch | 1.04× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 45.53 ms | 304.4 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 45.54 ms | 289.0 | fitsio/fitsio_torch | 1.04× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 45.73 ms | 304.4 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CPU | tensor | compressed_rice_1 [read_full] | on | 12.60 ms | 289.0 | fitsio/fitsio_torch | 1.00× |
| Linux x86_64 / CPU | table | narrow_100000 [read_full] | off | 680.7 μs | 368.5 | fitsio/fitsio_torch | 1.15× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 5.90 ms | 387.5 | fitsio/fitsio_torch | 1.12× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 5.85 ms | 391.6 | fitsio/fitsio | 1.01× |
| Linux x86_64 / CPU | table | narrow_100000 [read_full] | off | 681.3 μs | 368.5 | fitsio/fitsio | 1.00× |
| Linux x86_64 / CUDA | tensor | tiny_int8_3d [read_full @ cuda] | off | 115.5 μs | 771.4 | fitsio/fitsio_torch_device | 1.15× |
| Linux x86_64 / CUDA | tensor | tiny_int64_3d [read_full @ cuda] | off | 126.3 μs | 771.4 | fitsio/fitsio_torch_device | 1.13× |
| Linux x86_64 / CUDA | tensor | compressed_gzip_1 [read_full @ cuda] | on | 20.01 ms | 709.3 | fitsio/fitsio_torch_device | 1.13× |
| Linux x86_64 / CUDA | tensor | small_int8_1d [read_full @ cuda] | off | 115.2 μs | 771.4 | fitsio/fitsio_torch_device | 1.12× |
| Linux x86_64 / CUDA | tensor | tiny_int8_1d [read_full @ cuda] | off | 112.5 μs | 771.4 | fitsio/fitsio_torch_device | 1.12× |
| Linux x86_64 / CUDA | tensor | tiny_int16_3d [read_full @ cuda] | off | 112.7 μs | 771.4 | fitsio/fitsio_torch_device | 1.11× |
| Linux x86_64 / CUDA | tensor | tiny_int16_1d [read_full @ cuda] | off | 105.0 μs | 771.4 | fitsio/fitsio_torch_device | 1.10× |
| Linux x86_64 / CUDA | tensor | tiny_float64_3d [read_full @ cuda] | off | 120.4 μs | 771.4 | fitsio/fitsio_torch_device | 1.10× |
| Linux x86_64 / CUDA | tensor | tiny_float64_2d [read_full @ cuda] | off | 116.0 μs | 771.4 | fitsio/fitsio_torch_device | 1.06× |
| Linux x86_64 / CUDA | tensor | large_int8_2d [read_full] | off | 944.8 μs | 778.2 | fitsio/fitsio_torch | 1.06× |
| Linux x86_64 / CUDA | tensor | tiny_int32_2d [read_full @ cuda] | off | 111.4 μs | 771.4 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | tiny_float32_3d [read_full @ cuda] | off | 115.5 μs | 771.4 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | small_uint16_2d [read_full @ cuda] | off | 149.3 μs | 771.4 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | tiny_int32_3d [read_full @ cuda] | off | 112.2 μs | 771.4 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | on | 30.56 ms | 711.4 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | off | 30.66 ms | 739.5 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | tiny_int64_2d [read_full @ cuda] | off | 114.5 μs | 771.4 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | small_int16_1d [read_full @ cuda] | off | 110.6 μs | 771.4 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | on | 30.34 ms | 619.9 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | off | 30.17 ms | 778.2 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | tensor | tiny_int16_2d [read_full @ cuda] | off | 105.9 μs | 771.4 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | medium_int8_1d [read_full @ cuda] | off | 134.2 μs | 771.4 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | tiny_float32_2d [read_full @ cuda] | off | 113.6 μs | 771.4 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | tiny_float64_1d [read_full @ cuda] | off | 102.3 μs | 771.4 | fitsio/fitsio_torch_device | 1.02× |
| Linux x86_64 / CUDA | tensor | tiny_float32_1d [read_full @ cuda] | off | 107.1 μs | 771.4 | fitsio/fitsio_torch_device | 1.01× |
| Linux x86_64 / CUDA | tensor | scaled_small [read_full @ cuda] | off | 198.2 μs | 771.4 | fitsio/fitsio_torch_device | 1.01× |
| Linux x86_64 / CUDA | tensor | small_int8_2d [read_full @ cuda] | off | 123.4 μs | 771.4 | fitsio/fitsio_torch_device | 1.01× |
| Linux x86_64 / CUDA | tensor | small_int8_3d [read_full @ cuda] | off | 154.5 μs | 771.4 | fitsio/fitsio_torch_device | 1.00× |
| Linux x86_64 / CUDA | tensor | small_int16_2d [read_full @ cuda] | off | 142.9 μs | 771.4 | fitsio/fitsio_torch_device | 1.00× |
| Linux x86_64 / CUDA | tensor | medium_int16_3d [read_full] | off | 797.3 μs | 778.2 | fitsio/fitsio_torch | 1.12× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | off | 30.59 ms | 739.5 | fitsio/fitsio_torch_device_specialized | 1.04× |

_…and 38 more rows in `torchfits_deficits.csv`._
<!-- BENCH_DEFICITS_END -->

### Host scorecard

| Platform | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| Linux x86_64 / CPU | `exhaustive_cpu_20260801_202620` | 3057 | 9 | 288.9 | lab + mmap-matrix |
| Linux x86_64 / CUDA | `exhaustive_cuda_20260801_202613` | 4315 | 38 | 728.5 | lab + mmap-matrix + GPU |
| Linux x86_64 / CUDA | `exhaustive_0.9.0_20260801_202843` | 4315 | 31 | 731.3 | lab + mmap-matrix + GPU |
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
