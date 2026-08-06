# Benchmarks

`torchfits` benchmarks cover FITS **tensor** I/O (IMAGE HDUs, typically 1D–4D)
and FITS **table** I/O vs Astropy and fitsio. CPU↔GPU comparisons are published
when hardware was available; GPU deficits are listed, not hidden.

**Honesty:** torchfits is a **1.0.0rc5** prerelease. Headline ratios below are
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

### Small-payload CUDA overhead measurement (2026-08-04)

The current development host has no CUDA device, so this follow-up uses the
matched CUDA scorecard artifacts rather than a CPU proxy. Across 21 CUDA lanes
and 3,654 paired `read_full` / `read_full_gpu` rows (mmap-on, same case and
host), the medians were:

| Case group | torchfits host | torchfits GPU | torchfits added | fitsio host | fitsio GPU | fitsio added |
|---|---:|---:|---:|---:|---:|---:|
| tiny images | 0.0656 ms | 0.1158 ms | 0.0509 ms | 0.0911 ms | 0.1094 ms | 0.0179 ms |
| small images | 0.1127 ms | 0.1801 ms | 0.0732 ms | 0.1588 ms | 0.1857 ms | 0.0262 ms |
| `large_uint16_2d` | 3.0271 ms | 3.3141 ms | 0.3606 ms | 2.4247 ms | 2.2133 ms | -0.1251 ms |

The small-payload overhead hypothesis is **supported**: torchfits is faster on
host decode for the tiny/small groups, but its host-to-device path adds more
fixed time and reaches parity with fitsio or slightly loses. The rows prove a
combined launch/H2D/conversion cost, not which component dominates; that needs
Nsight or CUDA event instrumentation on a GPU host. No speculative CUDA-graph
or stream-manager change is being landed. The remaining large uint16 gap also
contains a host decode/dtype-path difference, so it is not explained by the
small-transfer hypothesis alone.

**uint16 host-decode gap fixed (2026-08-05, `4652476`).** The mmap fast path
applied the BZERO=32768 unsigned offset with a scalar second pass over the
buffer after the vectorized byte-swap copy; folding the offset into the SIMD
loops removed that pass (wraps mod 2^16 exactly like the scalar cast). Same
host, mmap-on, median of 3 interleaved runs: `large_uint16_2d`
6.180 → 1.399 ms (3.4× vs standalone fitsio; 1.32 → 0.57× vs the in-family
tensor peer `fitsio` + `torch.from_numpy`), `medium_uint16_2d`
1.830 → 0.408 ms, `small_uint16_2d` 0.178 → 0.124 ms. Bitwise-verified against
the CFITSIO path on full-range uint16 data; parity suite and `ci-local` green.

## Published CSVs

Exhaustive `results.csv` / `torchfits_deficits.csv` for the scorecard runs are
linked from GitHub Release assets when published, and mirrored under
`docs/assets/bench/<run-id>/` when size allows. The newest CPU/CUDA
exhaustives (`exhaustive_cpu_20260806_*`, `exhaustive_cuda_20260806_*`) feed
the generated tables above; their CSVs are archived under
`benchmarks_results/<run-id>/` locally and on the benchmark hosts until the
next release mirrors them as assets. Example local paths from the Round-3
soak that are already mirrored:

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

The 2026-08-05 same-host repro (`megacam-wave4/20260805_004215`, two files,
four HDUs, 40 256x256 cutouts per HDU) did **not** reproduce a torchfits lag:
`torchfits_cached` led `fitsio_cached` on every sampled HDU by approximately
7.5–15.2%. `torchfits_materialize` was faster still, but it is a separate
full-plane algorithm and used higher peak RSS on the larger sample. No Rice
decompression or materialization change is justified without a new host/file
combination that reverses this result.

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
Source: `benchmarks_results/exhaustive_cpu_20260806_012620/results.csv` (mmap on+off matrix.)
Cell values are median wall-clock over all comparable OK rows in the
`(domain × I/O transport × backend)` bucket; throughput is intentionally
omitted because the cell aggregates heterogeneous payloads and would
produce physically-impossible rates when small and large sizes are
median-mixed. See `scripts/render_bench_iopath_table.py` for the
aggregation rules.

### Tensor I/O (IMAGE HDU) (fits)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.09 ms` (n=174) | `0.49 ms` (n=253) | `0.17 ms` (n=261) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.10 ms` (n=174) | `0.42 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
| `disk→GPU` | — | — | — | — |
| `disk→CPU→GPU` | — | — | — | — |
| `disk→RAM→GPU` | — | — | — | — |

### Table I/O (fitstable)

| I/O transport | `torchfits` (libcfitsio) | `astropy` | `fitsio` | `cfitsio` (direct) |
|---|---:|---:|---:|---:|
| `disk→CPU` | `0.19 ms` (n=216) | `2.61 ms` (n=184) | `0.59 ms` (n=216) | — (engine exposed under `torchfits`) |
| `disk→RAM→CPU` | `0.21 ms` (n=216) | `2.55 ms` (n=184) | — (rows skipped under `strict_mmap_fairness`) | — (engine exposed under `torchfits`) |
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
| Table read (100k rows, 8 cols, mixed) | CPU | **2.56 ms** | 2.49 ms | 34.81 ms | 10.36 ms | **13.96x** | **4.15x** |
| Varlen table read (100k rows, 3 cols) | CPU | **72.21 ms** | 10.57 ms | 531.17 ms | 112.24 ms | **50.28x** | **10.62x** |
<!-- BENCH_HIGHLIGHTS_END -->

## Benchmark category summary

CPU category rows aggregate the rc5 CPU exhaustive
(`exhaustive_cpu_20260806_012620`, source of the generated
[highlights](#performance-highlights) and [full table](#exhaustive-benchmark-results)
above); the GPU (CUDA) rows come from `exhaustive_cuda_20260719_144457`
(see host scorecard for deficit honesty — all lags listed, floors label
noise vs significant). Category ranges are the last regenerated aggregation
shape; for absolute times prefer the generated tables above.

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
| **Repeated cutouts** (50× 100×100) | 1 | 467 μs | 50.85 ms | 3.21 ms | **108.8×** | **6.9×** |
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
| tensor | compressed_gzip_1:header_read | header_read | 1.29 MB | CPU | n/a | **—** | 30.0 μs | 1.36 ms | 139.0 μs | — | **45.35x** | **4.64x** |
| tensor | compressed_gzip_2:header_read | header_read | 0.89 MB | CPU | n/a | **—** | 29.9 μs | 1.37 ms | 139.0 μs | — | **45.91x** | **4.64x** |
| tensor | compressed_hcompress_1:header_read | header_read | 0.82 MB | CPU | n/a | **—** | 31.1 μs | 1.44 ms | 155.0 μs | — | **46.34x** | **4.98x** |
| tensor | compressed_rice_1:cutout_100x100 | cutout_100x100 | 0.90 MB | CPU | n/a | **778.2 μs** | 744.7 μs | 6.68 ms | 799.5 μs | — | **8.98x** | **1.07x** |
| tensor | compressed_rice_1:header_read | header_read | 0.90 MB | CPU | n/a | **—** | 30.8 μs | 1.43 ms | 157.1 μs | — | **46.36x** | **5.10x** |
| tensor | large_float32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 15.2 μs | 265.2 μs | 26.5 μs | — | **17.44x** | **1.74x** |
| tensor | large_float32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 16.2 μs | 295.3 μs | 29.1 μs | — | **18.20x** | **1.79x** |
| tensor | large_float64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 15.1 μs | 264.6 μs | 26.8 μs | — | **17.57x** | **1.78x** |
| tensor | large_float64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 15.6 μs | 294.6 μs | 27.9 μs | — | **18.93x** | **1.79x** |
| tensor | large_int16_1d:header_read | header_read | 1.91 MB | CPU | n/a | **—** | 15.2 μs | 266.8 μs | 26.8 μs | — | **17.53x** | **1.76x** |
| tensor | large_int16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 15.6 μs | 292.8 μs | 29.6 μs | — | **18.82x** | **1.90x** |
| tensor | large_int32_1d:header_read | header_read | 3.82 MB | CPU | n/a | **—** | 15.4 μs | 260.9 μs | 26.2 μs | — | **16.96x** | **1.70x** |
| tensor | large_int32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 16.2 μs | 290.9 μs | 28.9 μs | — | **17.95x** | **1.78x** |
| tensor | large_int64_1d:header_read | header_read | 7.63 MB | CPU | n/a | **—** | 14.2 μs | 262.7 μs | 27.1 μs | — | **18.46x** | **1.91x** |
| tensor | large_int64_2d:header_read | header_read | 32.00 MB | CPU | n/a | **—** | 15.3 μs | 287.8 μs | 27.9 μs | — | **18.81x** | **1.82x** |
| tensor | large_int8_1d:header_read | header_read | 0.96 MB | CPU | n/a | **—** | 16.3 μs | 317.2 μs | 31.3 μs | — | **19.49x** | **1.92x** |
| tensor | large_int8_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 17.6 μs | 333.2 μs | 32.8 μs | — | **18.88x** | **1.86x** |
| tensor | large_uint16_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 16.0 μs | 328.7 μs | 34.3 μs | — | **20.48x** | **2.14x** |
| tensor | large_uint32_2d:header_read | header_read | 16.00 MB | CPU | n/a | **—** | 16.8 μs | 322.2 μs | 33.8 μs | — | **19.23x** | **2.02x** |
| tensor | medium_float32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 15.5 μs | 259.2 μs | 27.0 μs | — | **16.73x** | **1.74x** |
| tensor | medium_float32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 16.3 μs | 285.6 μs | 30.0 μs | — | **17.47x** | **1.84x** |
| tensor | medium_float32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 16.1 μs | 314.9 μs | 32.1 μs | — | **19.59x** | **2.00x** |
| tensor | medium_float64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 14.3 μs | 264.4 μs | 26.1 μs | — | **18.47x** | **1.83x** |
| tensor | medium_float64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 14.6 μs | 290.6 μs | 26.6 μs | — | **19.91x** | **1.82x** |
| tensor | medium_float64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 14.9 μs | 317.9 μs | 30.0 μs | — | **21.31x** | **2.01x** |
| tensor | medium_int16_1d:header_read | header_read | 0.20 MB | CPU | n/a | **—** | 15.2 μs | 263.8 μs | 30.8 μs | — | **17.35x** | **2.02x** |
| tensor | medium_int16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 15.5 μs | 289.8 μs | 29.7 μs | — | **18.67x** | **1.91x** |
| tensor | medium_int16_3d:header_read | header_read | 3.13 MB | CPU | n/a | **—** | 14.7 μs | 305.5 μs | 30.5 μs | — | **20.83x** | **2.08x** |
| tensor | medium_int32_1d:header_read | header_read | 0.38 MB | CPU | n/a | **—** | 15.7 μs | 260.4 μs | 27.4 μs | — | **16.57x** | **1.74x** |
| tensor | medium_int32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 15.5 μs | 285.7 μs | 29.2 μs | — | **18.41x** | **1.88x** |
| tensor | medium_int32_3d:header_read | header_read | 6.25 MB | CPU | n/a | **—** | 15.1 μs | 308.7 μs | 28.9 μs | — | **20.40x** | **1.91x** |
| tensor | medium_int64_1d:header_read | header_read | 0.77 MB | CPU | n/a | **—** | 14.7 μs | 264.1 μs | 26.0 μs | — | **17.99x** | **1.77x** |
| tensor | medium_int64_2d:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 14.6 μs | 286.7 μs | 28.0 μs | — | **19.62x** | **1.92x** |
| tensor | medium_int64_3d:header_read | header_read | 12.51 MB | CPU | n/a | **—** | 15.4 μs | 304.0 μs | 29.9 μs | — | **19.79x** | **1.95x** |
| tensor | medium_int8_1d:header_read | header_read | 0.10 MB | CPU | n/a | **—** | 15.6 μs | 304.1 μs | 30.9 μs | — | **19.52x** | **1.98x** |
| tensor | medium_int8_2d:header_read | header_read | 1.01 MB | CPU | n/a | **—** | 17.3 μs | 331.0 μs | 31.5 μs | — | **19.18x** | **1.83x** |
| tensor | medium_int8_3d:header_read | header_read | 1.57 MB | CPU | n/a | **—** | 17.0 μs | 360.6 μs | 35.6 μs | — | **21.17x** | **2.09x** |
| tensor | medium_uint16_2d:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 17.0 μs | 341.2 μs | 33.5 μs | — | **20.06x** | **1.97x** |
| tensor | medium_uint32_2d:header_read | header_read | 4.00 MB | CPU | n/a | **—** | 17.2 μs | 335.3 μs | 33.1 μs | — | **19.50x** | **1.93x** |
| tensor | mef_medium:header_read | header_read | 7.02 MB | CPU | n/a | **—** | 19.8 μs | 534.8 μs | 45.3 μs | — | **26.98x** | **2.29x** |
| tensor | mef_small:header_read | header_read | 0.45 MB | CPU | n/a | **—** | 19.9 μs | 539.4 μs | 46.4 μs | — | **27.05x** | **2.33x** |
| tensor | multi_mef_10ext:cutout_100x100 | cutout_100x100 | 2.68 MB | CPU | n/a | **47.1 μs** | 85.8 μs | 2.17 ms | 163.4 μs | — | **46.18x** | **3.47x** |
| tensor | multi_mef_10ext:header_read | header_read | 2.68 MB | CPU | n/a | **—** | 18.3 μs | 541.5 μs | 43.5 μs | — | **29.58x** | **2.38x** |
| tensor | multi_mef_10ext:random_ext_full_reads_200 | random_ext_full_reads_200 | 2.68 MB | CPU | n/a | **5.41 ms** | 5.45 ms | 6.93 ms | 7.00 ms | — | **1.28x** | **1.29x** |
| tensor | repeated_cutouts_50x_100x100:repeated_cutouts_50x_100x100 | repeated_cutouts_50x_100x100 | 4.00 MB | CPU | n/a | **467.2 μs** | 496.4 μs | 50.85 ms | 3.21 ms | — | **108.84x** | **6.87x** |
| tensor | scaled_large:header_read | header_read | 8.00 MB | CPU | n/a | **—** | 16.0 μs | 339.3 μs | 35.7 μs | — | **21.17x** | **2.23x** |
| tensor | scaled_medium:header_read | header_read | 2.01 MB | CPU | n/a | **—** | 17.1 μs | 355.2 μs | 35.9 μs | — | **20.77x** | **2.10x** |
| tensor | scaled_small:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 17.2 μs | 341.9 μs | 34.1 μs | — | **19.88x** | **1.98x** |
| tensor | small_float32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 15.8 μs | 272.9 μs | 28.6 μs | — | **17.31x** | **1.82x** |
| tensor | small_float32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.3 μs | 290.2 μs | 29.9 μs | — | **17.79x** | **1.83x** |
| tensor | small_float32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 16.4 μs | 319.6 μs | 31.7 μs | — | **19.52x** | **1.93x** |
| tensor | small_float64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 15.1 μs | 267.1 μs | 25.4 μs | — | **17.63x** | **1.68x** |
| tensor | small_float64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 15.2 μs | 298.5 μs | 29.0 μs | — | **19.66x** | **1.91x** |
| tensor | small_float64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 16.9 μs | 314.5 μs | 32.7 μs | — | **18.63x** | **1.94x** |
| tensor | small_int16_1d:header_read | header_read | 22.5 KB | CPU | n/a | **—** | 15.7 μs | 277.8 μs | 27.0 μs | — | **17.75x** | **1.72x** |
| tensor | small_int16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 15.8 μs | 295.0 μs | 29.4 μs | — | **18.69x** | **1.86x** |
| tensor | small_int16_3d:header_read | header_read | 0.32 MB | CPU | n/a | **—** | 16.0 μs | 313.7 μs | 30.9 μs | — | **19.66x** | **1.94x** |
| tensor | small_int32_1d:header_read | header_read | 42.2 KB | CPU | n/a | **—** | 14.5 μs | 267.5 μs | 28.2 μs | — | **18.46x** | **1.95x** |
| tensor | small_int32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 14.7 μs | 290.5 μs | 28.9 μs | — | **19.83x** | **1.97x** |
| tensor | small_int32_3d:header_read | header_read | 0.63 MB | CPU | n/a | **—** | 15.9 μs | 308.6 μs | 30.0 μs | — | **19.39x** | **1.88x** |
| tensor | small_int64_1d:header_read | header_read | 0.08 MB | CPU | n/a | **—** | 15.9 μs | 266.5 μs | 27.3 μs | — | **16.74x** | **1.71x** |
| tensor | small_int64_2d:header_read | header_read | 0.51 MB | CPU | n/a | **—** | 15.0 μs | 278.3 μs | 30.4 μs | — | **18.53x** | **2.02x** |
| tensor | small_int64_3d:header_read | header_read | 1.26 MB | CPU | n/a | **—** | 15.7 μs | 308.8 μs | 30.5 μs | — | **19.64x** | **1.94x** |
| tensor | small_int8_1d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 16.6 μs | 310.0 μs | 31.0 μs | — | **18.69x** | **1.87x** |
| tensor | small_int8_2d:header_read | header_read | 0.07 MB | CPU | n/a | **—** | 17.3 μs | 330.9 μs | 35.3 μs | — | **19.15x** | **2.04x** |
| tensor | small_int8_3d:header_read | header_read | 0.16 MB | CPU | n/a | **—** | 17.3 μs | 355.3 μs | 35.8 μs | — | **20.50x** | **2.07x** |
| tensor | small_uint16_2d:header_read | header_read | 0.13 MB | CPU | n/a | **—** | 15.9 μs | 338.5 μs | 32.8 μs | — | **21.23x** | **2.06x** |
| tensor | small_uint32_2d:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 15.5 μs | 335.3 μs | 32.7 μs | — | **21.67x** | **2.12x** |
| tensor | timeseries_frame_000:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 15.1 μs | 293.9 μs | 29.9 μs | — | **19.52x** | **1.99x** |
| tensor | timeseries_frame_001:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.7 μs | 293.8 μs | 30.0 μs | — | **17.55x** | **1.79x** |
| tensor | timeseries_frame_002:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 14.7 μs | 290.1 μs | 28.9 μs | — | **19.72x** | **1.96x** |
| tensor | timeseries_frame_003:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.2 μs | 295.7 μs | 29.2 μs | — | **18.31x** | **1.81x** |
| tensor | timeseries_frame_004:header_read | header_read | 0.26 MB | CPU | n/a | **—** | 16.9 μs | 297.4 μs | 30.1 μs | — | **17.57x** | **1.78x** |
| tensor | tiny_float32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 15.9 μs | 268.1 μs | 26.9 μs | — | **16.91x** | **1.70x** |
| tensor | tiny_float32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 16.4 μs | 294.6 μs | 30.2 μs | — | **17.98x** | **1.85x** |
| tensor | tiny_float32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 16.4 μs | 320.0 μs | 32.7 μs | — | **19.51x** | **2.00x** |
| tensor | tiny_float64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 16.0 μs | 266.9 μs | 26.2 μs | — | **16.71x** | **1.64x** |
| tensor | tiny_float64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 15.5 μs | 289.1 μs | 30.1 μs | — | **18.66x** | **1.94x** |
| tensor | tiny_float64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 16.8 μs | 315.8 μs | 31.1 μs | — | **18.84x** | **1.86x** |
| tensor | tiny_int16_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 16.0 μs | 263.1 μs | 27.2 μs | — | **16.40x** | **1.70x** |
| tensor | tiny_int16_2d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 16.5 μs | 297.3 μs | 28.1 μs | — | **17.99x** | **1.70x** |
| tensor | tiny_int16_3d:header_read | header_read | 14.1 KB | CPU | n/a | **—** | 15.3 μs | 311.5 μs | 31.3 μs | — | **20.33x** | **2.04x** |
| tensor | tiny_int32_1d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 15.7 μs | 259.3 μs | 26.7 μs | — | **16.55x** | **1.70x** |
| tensor | tiny_int32_2d:header_read | header_read | 19.7 KB | CPU | n/a | **—** | 15.0 μs | 289.1 μs | 29.4 μs | — | **19.22x** | **1.96x** |
| tensor | tiny_int32_3d:header_read | header_read | 25.3 KB | CPU | n/a | **—** | 16.7 μs | 310.7 μs | 32.3 μs | — | **18.62x** | **1.94x** |
| tensor | tiny_int64_1d:header_read | header_read | 11.2 KB | CPU | n/a | **—** | 15.7 μs | 251.7 μs | 28.0 μs | — | **16.00x** | **1.78x** |
| tensor | tiny_int64_2d:header_read | header_read | 36.6 KB | CPU | n/a | **—** | 16.5 μs | 288.0 μs | 30.5 μs | — | **17.43x** | **1.85x** |
| tensor | tiny_int64_3d:header_read | header_read | 45.0 KB | CPU | n/a | **—** | 15.8 μs | 306.4 μs | 28.9 μs | — | **19.34x** | **1.82x** |
| tensor | tiny_int8_1d:header_read | header_read | 5.6 KB | CPU | n/a | **—** | 16.9 μs | 308.7 μs | 31.6 μs | — | **18.31x** | **1.88x** |
| tensor | tiny_int8_2d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 16.9 μs | 337.9 μs | 34.4 μs | — | **20.05x** | **2.04x** |
| tensor | tiny_int8_3d:header_read | header_read | 8.4 KB | CPU | n/a | **—** | 16.8 μs | 353.1 μs | 34.4 μs | — | **21.04x** | **2.05x** |
| tensor | write_compress_hcompress_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **48.40 ms** | — | 58.69 ms | — | — | **1.21x** | **—** |
| tensor | write_compress_rice_medium_float32_2d | write_compress | 4.00 MB | CPU | n/a | **37.61 ms** | — | 68.22 ms | — | — | **1.81x** | **—** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | off | **23.56 ms** | 23.68 ms | 45.54 ms | 26.29 ms | — | **1.93x** | **1.12x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | off | **20.08 ms** | 20.38 ms | 71.18 ms | 23.05 ms | — | **3.55x** | **1.15x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | off | **45.43 ms** | 45.44 ms | 51.54 ms | 44.34 ms | — | **1.13x** | **0.98x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | off | **12.40 ms** | 12.42 ms | 32.85 ms | 12.57 ms | — | **2.65x** | **1.01x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | off | **675.3 μs** | 666.6 μs | 1.69 ms | 1.15 ms | — | **2.53x** | **1.72x** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | off | **3.85 ms** | 3.85 ms | 15.53 ms | 5.02 ms | — | **4.03x** | **1.30x** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | off | **1.28 ms** | 1.25 ms | 2.80 ms | 1.79 ms | — | **2.25x** | **1.44x** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | off | **6.55 ms** | 6.50 ms | 14.16 ms | 7.16 ms | — | **2.18x** | **1.10x** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | off | **482.5 μs** | 403.5 μs | 1.11 ms | 501.7 μs | — | **2.74x** | **1.24x** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | off | **1.44 ms** | 1.43 ms | 6.02 ms | 1.85 ms | — | **4.20x** | **1.29x** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | off | **683.5 μs** | 696.0 μs | 1.66 ms | 1.14 ms | — | **2.43x** | **1.67x** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | off | **3.75 ms** | 4.46 ms | 15.38 ms | 4.86 ms | — | **4.10x** | **1.30x** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | off | **1.28 ms** | 1.28 ms | 2.81 ms | 1.79 ms | — | **2.20x** | **1.40x** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | off | **6.48 ms** | 6.49 ms | 14.09 ms | 7.10 ms | — | **2.17x** | **1.09x** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | off | **255.0 μs** | 260.6 μs | 985.5 μs | 281.6 μs | — | **3.86x** | **1.10x** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | off | **992.4 μs** | 978.8 μs | 2.48 ms | 1.00 ms | — | **2.53x** | **1.02x** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | off | **1.87 ms** | 1.87 ms | 6.32 ms | 2.29 ms | — | **3.39x** | **1.23x** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | off | **3.35 ms** | 4.59 ms | 10.30 ms | 5.71 ms | — | **3.08x** | **1.71x** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | off | **89.3 μs** | 126.9 μs | 577.6 μs | 166.4 μs | — | **6.47x** | **1.86x** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | off | **715.2 μs** | 696.5 μs | 1.79 ms | 1.19 ms | — | **2.57x** | **1.71x** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | off | **1.26 ms** | 1.05 ms | 2.47 ms | 1.81 ms | — | **2.35x** | **1.72x** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | off | **195.5 μs** | 192.4 μs | 720.2 μs | 236.7 μs | — | **3.74x** | **1.23x** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | off | **1.34 ms** | 1.34 ms | 3.90 ms | 1.88 ms | — | **2.91x** | **1.40x** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | off | **2.64 ms** | 2.67 ms | 6.91 ms | 2.88 ms | — | **2.62x** | **1.09x** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | off | **102.8 μs** | 93.7 μs | 416.7 μs | 90.5 μs | — | **4.45x** | **0.97x** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | off | **312.9 μs** | 310.6 μs | 777.1 μs | 373.5 μs | — | **2.50x** | **1.20x** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | off | **446.8 μs** | 425.1 μs | 1.04 ms | 568.0 μs | — | **2.45x** | **1.34x** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | off | **53.9 μs** | 82.3 μs | 364.6 μs | 112.7 μs | — | **6.76x** | **2.09x** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | off | **525.7 μs** | 522.4 μs | 1.22 ms | 813.4 μs | — | **2.34x** | **1.56x** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | off | **777.0 μs** | 1.05 ms | 1.73 ms | 1.22 ms | — | **2.22x** | **1.57x** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | off | **135.8 μs** | 144.8 μs | 460.8 μs | 163.5 μs | — | **3.39x** | **1.20x** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | off | **974.0 μs** | 961.7 μs | 2.56 ms | 1.36 ms | — | **2.66x** | **1.41x** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | off | **2.12 ms** | 1.48 ms | 4.75 ms | 2.06 ms | — | **3.21x** | **1.39x** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | off | **35.0 μs** | 52.8 μs | 374.2 μs | 58.2 μs | — | **10.70x** | **1.66x** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | off | **219.5 μs** | 143.8 μs | 682.3 μs | 194.8 μs | — | **4.74x** | **1.35x** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | off | **240.0 μs** | 279.8 μs | 860.1 μs | 285.0 μs | — | **3.58x** | **1.19x** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | off | **387.4 μs** | 362.1 μs | 1.31 ms | 441.8 μs | — | **3.62x** | **1.22x** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | off | **683.7 μs** | 680.3 μs | 1.78 ms | 968.7 μs | — | **2.61x** | **1.42x** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | off | **149.5 μs** | 197.9 μs | 856.1 μs | 216.6 μs | — | **5.73x** | **1.45x** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | off | **55.7 μs** | 49.4 μs | 563.9 μs | 81.7 μs | — | **11.41x** | **1.65x** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | off | **65.9 μs** | 31.8 μs | 562.0 μs | 142.1 μs | — | **17.67x** | **4.47x** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | off | **3.43 ms** | 3.47 ms | 5.58 ms | 3.39 ms | — | **1.63x** | **0.99x** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | off | **650.3 μs** | 669.2 μs | 1.41 ms | 799.1 μs | — | **2.17x** | **1.23x** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | off | **83.7 μs** | 85.1 μs | 458.9 μs | 95.3 μs | — | **5.48x** | **1.14x** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | off | **26.7 μs** | 27.9 μs | 263.5 μs | 48.3 μs | — | **9.85x** | **1.81x** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | off | **42.7 μs** | 68.5 μs | 353.2 μs | 87.7 μs | — | **8.28x** | **2.05x** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | off | **120.5 μs** | 83.8 μs | 460.9 μs | 162.2 μs | — | **5.50x** | **1.94x** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | off | **31.3 μs** | 34.9 μs | 275.3 μs | 51.0 μs | — | **8.79x** | **1.63x** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | off | **107.5 μs** | 76.9 μs | 416.0 μs | 122.4 μs | — | **5.41x** | **1.59x** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | off | **179.0 μs** | 243.3 μs | 633.2 μs | 240.6 μs | — | **3.54x** | **1.34x** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | off | **29.8 μs** | 36.0 μs | 257.0 μs | 47.4 μs | — | **8.62x** | **1.59x** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | off | **59.2 μs** | 58.3 μs | 309.6 μs | 58.9 μs | — | **5.31x** | **1.01x** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | off | **89.9 μs** | 72.0 μs | 375.4 μs | 89.5 μs | — | **5.22x** | **1.24x** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | off | **39.3 μs** | 33.6 μs | 269.1 μs | 49.2 μs | — | **8.00x** | **1.46x** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | off | **69.8 μs** | 41.6 μs | 344.3 μs | 83.9 μs | — | **8.27x** | **2.02x** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | off | **123.0 μs** | 133.4 μs | 468.4 μs | 159.9 μs | — | **3.81x** | **1.30x** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | off | **41.4 μs** | 31.9 μs | 275.9 μs | 52.1 μs | — | **8.65x** | **1.63x** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | off | **106.1 μs** | 108.1 μs | 413.5 μs | 123.0 μs | — | **3.90x** | **1.16x** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | off | **209.6 μs** | 179.2 μs | 639.6 μs | 252.8 μs | — | **3.57x** | **1.41x** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | off | **41.4 μs** | 32.0 μs | 357.1 μs | 46.3 μs | — | **11.15x** | **1.44x** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | off | **59.5 μs** | 34.0 μs | 388.7 μs | 55.0 μs | — | **11.45x** | **1.62x** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | off | **55.8 μs** | 60.9 μs | 444.7 μs | 66.9 μs | — | **7.97x** | **1.20x** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | off | **61.0 μs** | 38.5 μs | 380.1 μs | 63.4 μs | — | **9.86x** | **1.65x** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | off | **49.9 μs** | 71.7 μs | 415.5 μs | 96.0 μs | — | **8.33x** | **1.93x** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | off | **62.5 μs** | 74.7 μs | 343.6 μs | 85.7 μs | — | **5.50x** | **1.37x** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | off | **65.8 μs** | 79.8 μs | 354.5 μs | 83.4 μs | — | **5.39x** | **1.27x** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | off | **61.5 μs** | 65.4 μs | 350.5 μs | 85.7 μs | — | **5.70x** | **1.39x** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | off | **62.2 μs** | 80.0 μs | 348.9 μs | 84.7 μs | — | **5.61x** | **1.36x** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | off | **44.1 μs** | 71.5 μs | 350.0 μs | 88.0 μs | — | **7.94x** | **2.00x** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | off | **41.0 μs** | 32.4 μs | 257.9 μs | 40.9 μs | — | **7.96x** | **1.26x** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | off | **38.9 μs** | 31.4 μs | 274.9 μs | 41.1 μs | — | **8.75x** | **1.31x** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | off | **39.3 μs** | 37.2 μs | 292.3 μs | 47.8 μs | — | **7.86x** | **1.29x** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | off | **36.7 μs** | 32.8 μs | 261.8 μs | 41.6 μs | — | **7.98x** | **1.27x** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | off | **33.0 μs** | 37.4 μs | 286.5 μs | 47.0 μs | — | **8.68x** | **1.42x** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | off | **31.7 μs** | 38.2 μs | 389.9 μs | 64.7 μs | — | **12.31x** | **2.04x** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | off | **35.2 μs** | 31.8 μs | 255.5 μs | 39.4 μs | — | **8.03x** | **1.24x** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | off | **30.0 μs** | 28.2 μs | 269.3 μs | 41.0 μs | — | **9.54x** | **1.45x** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | off | **36.8 μs** | 35.5 μs | 286.8 μs | 44.2 μs | — | **8.09x** | **1.25x** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | off | **36.4 μs** | 38.3 μs | 261.0 μs | 45.9 μs | — | **7.16x** | **1.26x** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | off | **35.0 μs** | 24.0 μs | 270.7 μs | 43.0 μs | — | **11.29x** | **1.79x** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | off | **25.8 μs** | 42.2 μs | 288.9 μs | 45.5 μs | — | **11.20x** | **1.76x** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | off | **36.7 μs** | 28.8 μs | 252.9 μs | 41.2 μs | — | **8.78x** | **1.43x** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | off | **44.0 μs** | 32.6 μs | 271.4 μs | 46.0 μs | — | **8.32x** | **1.41x** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | off | **31.8 μs** | 47.8 μs | 297.2 μs | 48.7 μs | — | **9.36x** | **1.53x** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | off | **40.8 μs** | 28.2 μs | 365.1 μs | 47.3 μs | — | **12.93x** | **1.68x** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | off | **41.0 μs** | 26.8 μs | 373.0 μs | 46.9 μs | — | **13.91x** | **1.75x** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | off | **34.0 μs** | 32.4 μs | 379.2 μs | 49.8 μs | — | **11.69x** | **1.53x** |
| tensor | compressed_gzip_1:read_full | read_full | 1.29 MB | CPU | on | **23.50 ms** | 23.67 ms | 45.67 ms | 26.42 ms | — | **1.94x** | **1.12x** |
| tensor | compressed_gzip_2:read_full | read_full | 0.89 MB | CPU | on | **20.17 ms** | 20.49 ms | 71.86 ms | 23.17 ms | — | **3.56x** | **1.15x** |
| tensor | compressed_hcompress_1:read_full | read_full | 0.82 MB | CPU | on | **45.69 ms** | 46.02 ms | 51.84 ms | 44.65 ms | — | **1.13x** | **0.98x** |
| tensor | compressed_rice_1:read_full | read_full | 0.90 MB | CPU | on | **12.45 ms** | 12.69 ms | 32.90 ms | 12.56 ms | — | **2.64x** | **1.01x** |
| tensor | large_float32_1d:read_full | read_full | 3.82 MB | CPU | on | **436.4 μs** | 446.6 μs | 957.7 μs | — | — | **2.19x** | **—** |
| tensor | large_float32_2d:read_full | read_full | 16.00 MB | CPU | on | **2.36 ms** | 3.18 ms | 7.46 ms | — | — | **3.16x** | **—** |
| tensor | large_float64_1d:read_full | read_full | 7.63 MB | CPU | on | **1.22 ms** | 811.1 μs | 1.55 ms | — | — | **1.91x** | **—** |
| tensor | large_float64_2d:read_full | read_full | 32.00 MB | CPU | on | **5.22 ms** | 4.08 ms | 7.06 ms | — | — | **1.73x** | **—** |
| tensor | large_int16_1d:read_full | read_full | 1.91 MB | CPU | on | **255.6 μs** | 260.2 μs | 658.4 μs | — | — | **2.58x** | **—** |
| tensor | large_int16_2d:read_full | read_full | 8.00 MB | CPU | on | **832.6 μs** | 891.4 μs | 1.63 ms | — | — | **1.96x** | **—** |
| tensor | large_int32_1d:read_full | read_full | 3.82 MB | CPU | on | **422.6 μs** | 446.2 μs | 950.9 μs | — | — | **2.25x** | **—** |
| tensor | large_int32_2d:read_full | read_full | 16.00 MB | CPU | on | **2.31 ms** | 1.66 ms | 7.44 ms | — | — | **4.48x** | **—** |
| tensor | large_int64_1d:read_full | read_full | 7.63 MB | CPU | on | **826.9 μs** | 787.4 μs | 1.55 ms | — | — | **1.97x** | **—** |
| tensor | large_int64_2d:read_full | read_full | 32.00 MB | CPU | on | **4.55 ms** | 4.68 ms | 7.08 ms | — | — | **1.56x** | **—** |
| tensor | large_int8_1d:read_full | read_full | 0.96 MB | CPU | on | **169.8 μs** | 169.1 μs | — | — | — | **—** | **—** |
| tensor | large_int8_2d:read_full | read_full | 4.00 MB | CPU | on | **518.5 μs** | 519.4 μs | — | — | — | **—** | **—** |
| tensor | large_uint16_2d:read_full | read_full | 8.00 MB | CPU | on | **854.6 μs** | 854.5 μs | — | — | — | **—** | **—** |
| tensor | large_uint32_2d:read_full | read_full | 16.00 MB | CPU | on | **3.32 ms** | 3.33 ms | — | — | — | **—** | **—** |
| tensor | medium_float32_1d:read_full | read_full | 0.38 MB | CPU | on | **73.7 μs** | 92.5 μs | 366.0 μs | — | — | **4.96x** | **—** |
| tensor | medium_float32_2d:read_full | read_full | 4.00 MB | CPU | on | **450.8 μs** | 465.7 μs | 1.07 ms | — | — | **2.37x** | **—** |
| tensor | medium_float32_3d:read_full | read_full | 6.25 MB | CPU | on | **684.8 μs** | 663.5 μs | 1.38 ms | — | — | **2.08x** | **—** |
| tensor | medium_float64_1d:read_full | read_full | 0.77 MB | CPU | on | **153.7 μs** | 146.5 μs | 436.2 μs | — | — | **2.98x** | **—** |
| tensor | medium_float64_2d:read_full | read_full | 8.00 MB | CPU | on | **832.5 μs** | 1.18 ms | 1.68 ms | — | — | **2.01x** | **—** |
| tensor | medium_float64_3d:read_full | read_full | 12.51 MB | CPU | on | **1.31 ms** | 1.27 ms | 2.34 ms | — | — | **1.83x** | **—** |
| tensor | medium_int16_1d:read_full | read_full | 0.20 MB | CPU | on | **69.4 μs** | 78.6 μs | 420.0 μs | — | — | **6.06x** | **—** |
| tensor | medium_int16_2d:read_full | read_full | 2.01 MB | CPU | on | **264.6 μs** | 268.0 μs | 695.7 μs | — | — | **2.63x** | **—** |
| tensor | medium_int16_3d:read_full | read_full | 3.13 MB | CPU | on | **366.3 μs** | 376.5 μs | 889.5 μs | — | — | **2.43x** | **—** |
| tensor | medium_int32_1d:read_full | read_full | 0.38 MB | CPU | on | **91.7 μs** | 73.9 μs | 364.3 μs | — | — | **4.93x** | **—** |
| tensor | medium_int32_2d:read_full | read_full | 4.00 MB | CPU | on | **466.9 μs** | 462.3 μs | 1.08 ms | — | — | **2.34x** | **—** |
| tensor | medium_int32_3d:read_full | read_full | 6.25 MB | CPU | on | **687.7 μs** | 684.1 μs | 1.39 ms | — | — | **2.04x** | **—** |
| tensor | medium_int64_1d:read_full | read_full | 0.77 MB | CPU | on | **136.9 μs** | 127.9 μs | 442.5 μs | — | — | **3.46x** | **—** |
| tensor | medium_int64_2d:read_full | read_full | 8.00 MB | CPU | on | **836.4 μs** | 854.0 μs | 1.63 ms | — | — | **1.95x** | **—** |
| tensor | medium_int64_3d:read_full | read_full | 12.51 MB | CPU | on | **1.26 ms** | 1.26 ms | 2.37 ms | — | — | **1.88x** | **—** |
| tensor | medium_int8_1d:read_full | read_full | 0.10 MB | CPU | on | **63.3 μs** | 32.9 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_2d:read_full | read_full | 1.01 MB | CPU | on | **148.8 μs** | 194.2 μs | — | — | — | **—** | **—** |
| tensor | medium_int8_3d:read_full | read_full | 1.57 MB | CPU | on | **251.9 μs** | 264.5 μs | — | — | — | **—** | **—** |
| tensor | medium_uint16_2d:read_full | read_full | 2.01 MB | CPU | on | **272.1 μs** | 284.3 μs | — | — | — | **—** | **—** |
| tensor | medium_uint32_2d:read_full | read_full | 4.00 MB | CPU | on | **744.2 μs** | 736.7 μs | — | — | — | **—** | **—** |
| tensor | mef_medium:read_full | read_full | 7.02 MB | CPU | on | **141.1 μs** | 212.5 μs | — | — | — | **—** | **—** |
| tensor | mef_small:read_full | read_full | 0.45 MB | CPU | on | **65.4 μs** | 37.4 μs | — | — | — | **—** | **—** |
| tensor | multi_mef_10ext:read_full | read_full | 2.68 MB | CPU | on | **36.3 μs** | 64.6 μs | — | — | — | **—** | **—** |
| tensor | scaled_large:read_full | read_full | 8.00 MB | CPU | on | **3.38 ms** | 2.48 ms | — | — | — | **—** | **—** |
| tensor | scaled_medium:read_full | read_full | 2.01 MB | CPU | on | **727.4 μs** | 634.2 μs | — | — | — | **—** | **—** |
| tensor | scaled_small:read_full | read_full | 0.13 MB | CPU | on | **92.3 μs** | 82.9 μs | — | — | — | **—** | **—** |
| tensor | small_float32_1d:read_full | read_full | 42.2 KB | CPU | on | **40.0 μs** | 31.4 μs | 281.9 μs | — | — | **8.96x** | **—** |
| tensor | small_float32_2d:read_full | read_full | 0.26 MB | CPU | on | **66.0 μs** | 68.0 μs | 358.8 μs | — | — | **5.44x** | **—** |
| tensor | small_float32_3d:read_full | read_full | 0.63 MB | CPU | on | **99.7 μs** | 104.4 μs | 456.8 μs | — | — | **4.58x** | **—** |
| tensor | small_float64_1d:read_full | read_full | 0.08 MB | CPU | on | **32.8 μs** | 49.5 μs | 284.7 μs | — | — | **8.69x** | **—** |
| tensor | small_float64_2d:read_full | read_full | 0.51 MB | CPU | on | **92.5 μs** | 115.9 μs | 426.1 μs | — | — | **4.61x** | **—** |
| tensor | small_float64_3d:read_full | read_full | 1.26 MB | CPU | on | **171.6 μs** | 175.6 μs | 578.4 μs | — | — | **3.37x** | **—** |
| tensor | small_int16_1d:read_full | read_full | 22.5 KB | CPU | on | **46.3 μs** | 40.7 μs | 268.1 μs | — | — | **6.58x** | **—** |
| tensor | small_int16_2d:read_full | read_full | 0.13 MB | CPU | on | **56.7 μs** | 76.0 μs | 329.1 μs | — | — | **5.80x** | **—** |
| tensor | small_int16_3d:read_full | read_full | 0.32 MB | CPU | on | **64.7 μs** | 93.9 μs | 380.6 μs | — | — | **5.88x** | **—** |
| tensor | small_int32_1d:read_full | read_full | 42.2 KB | CPU | on | **52.1 μs** | 38.5 μs | 284.7 μs | — | — | **7.40x** | **—** |
| tensor | small_int32_2d:read_full | read_full | 0.26 MB | CPU | on | **99.9 μs** | 73.1 μs | 354.3 μs | — | — | **4.85x** | **—** |
| tensor | small_int32_3d:read_full | read_full | 0.63 MB | CPU | on | **114.9 μs** | 128.8 μs | 445.3 μs | — | — | **3.87x** | **—** |
| tensor | small_int64_1d:read_full | read_full | 0.08 MB | CPU | on | **46.0 μs** | 58.4 μs | 286.0 μs | — | — | **6.22x** | **—** |
| tensor | small_int64_2d:read_full | read_full | 0.51 MB | CPU | on | **109.8 μs** | 122.1 μs | 415.4 μs | — | — | **3.78x** | **—** |
| tensor | small_int64_3d:read_full | read_full | 1.26 MB | CPU | on | **188.5 μs** | 204.6 μs | 578.7 μs | — | — | **3.07x** | **—** |
| tensor | small_int8_1d:read_full | read_full | 14.1 KB | CPU | on | **26.2 μs** | 47.0 μs | — | — | — | **—** | **—** |
| tensor | small_int8_2d:read_full | read_full | 0.07 MB | CPU | on | **47.4 μs** | 54.5 μs | — | — | — | **—** | **—** |
| tensor | small_int8_3d:read_full | read_full | 0.16 MB | CPU | on | **64.9 μs** | 63.5 μs | — | — | — | **—** | **—** |
| tensor | small_uint16_2d:read_full | read_full | 0.13 MB | CPU | on | **76.9 μs** | 74.8 μs | — | — | — | **—** | **—** |
| tensor | small_uint32_2d:read_full | read_full | 0.26 MB | CPU | on | **110.0 μs** | 70.3 μs | — | — | — | **—** | **—** |
| tensor | timeseries_frame_000:read_full | read_full | 0.26 MB | CPU | on | **67.0 μs** | 39.4 μs | 356.3 μs | — | — | **9.05x** | **—** |
| tensor | timeseries_frame_001:read_full | read_full | 0.26 MB | CPU | on | **59.4 μs** | 68.1 μs | 353.7 μs | — | — | **5.96x** | **—** |
| tensor | timeseries_frame_002:read_full | read_full | 0.26 MB | CPU | on | **62.4 μs** | 69.4 μs | 362.2 μs | — | — | **5.80x** | **—** |
| tensor | timeseries_frame_003:read_full | read_full | 0.26 MB | CPU | on | **69.6 μs** | 38.5 μs | 352.0 μs | — | — | **9.14x** | **—** |
| tensor | timeseries_frame_004:read_full | read_full | 0.26 MB | CPU | on | **63.8 μs** | 61.2 μs | 350.5 μs | — | — | **5.72x** | **—** |
| tensor | tiny_float32_1d:read_full | read_full | 8.4 KB | CPU | on | **36.1 μs** | 28.4 μs | 263.7 μs | — | — | **9.29x** | **—** |
| tensor | tiny_float32_2d:read_full | read_full | 19.7 KB | CPU | on | **25.2 μs** | 43.1 μs | 289.5 μs | — | — | **11.47x** | **—** |
| tensor | tiny_float32_3d:read_full | read_full | 25.3 KB | CPU | on | **30.3 μs** | 39.4 μs | 302.4 μs | — | — | **10.00x** | **—** |
| tensor | tiny_float64_1d:read_full | read_full | 11.2 KB | CPU | on | **28.9 μs** | 43.0 μs | 257.3 μs | — | — | **8.89x** | **—** |
| tensor | tiny_float64_2d:read_full | read_full | 36.6 KB | CPU | on | **33.7 μs** | 30.4 μs | 286.3 μs | — | — | **9.42x** | **—** |
| tensor | tiny_float64_3d:read_full | read_full | 45.0 KB | CPU | on | **45.4 μs** | 32.0 μs | 308.1 μs | — | — | **9.62x** | **—** |
| tensor | tiny_int16_1d:read_full | read_full | 5.6 KB | CPU | on | **28.8 μs** | 38.2 μs | 261.9 μs | — | — | **9.09x** | **—** |
| tensor | tiny_int16_2d:read_full | read_full | 11.2 KB | CPU | on | **32.8 μs** | 43.5 μs | 279.4 μs | — | — | **8.51x** | **—** |
| tensor | tiny_int16_3d:read_full | read_full | 14.1 KB | CPU | on | **49.5 μs** | 34.0 μs | 299.0 μs | — | — | **8.80x** | **—** |
| tensor | tiny_int32_1d:read_full | read_full | 8.4 KB | CPU | on | **38.4 μs** | 41.3 μs | 269.3 μs | — | — | **7.02x** | **—** |
| tensor | tiny_int32_2d:read_full | read_full | 19.7 KB | CPU | on | **45.3 μs** | 37.1 μs | 286.5 μs | — | — | **7.72x** | **—** |
| tensor | tiny_int32_3d:read_full | read_full | 25.3 KB | CPU | on | **34.1 μs** | 44.9 μs | 304.0 μs | — | — | **8.92x** | **—** |
| tensor | tiny_int64_1d:read_full | read_full | 11.2 KB | CPU | on | **45.1 μs** | 29.6 μs | 266.7 μs | — | — | **9.00x** | **—** |
| tensor | tiny_int64_2d:read_full | read_full | 36.6 KB | CPU | on | **37.2 μs** | 51.4 μs | 293.2 μs | — | — | **7.89x** | **—** |
| tensor | tiny_int64_3d:read_full | read_full | 45.0 KB | CPU | on | **34.7 μs** | 53.0 μs | 313.5 μs | — | — | **9.04x** | **—** |
| tensor | tiny_int8_1d:read_full | read_full | 5.6 KB | CPU | on | **31.2 μs** | 48.5 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_2d:read_full | read_full | 8.4 KB | CPU | on | **36.5 μs** | 31.4 μs | — | — | — | **—** | **—** |
| tensor | tiny_int8_3d:read_full | read_full | 8.4 KB | CPU | on | **30.3 μs** | 50.2 μs | — | — | — | **—** | **—** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | off | **318.2 μs** | 338.0 μs | 2.68 ms | 373.3 μs | — | **8.42x** | **1.17x** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | off | **315.4 μs** | 332.6 μs | 2.68 ms | 372.6 μs | — | **8.49x** | **1.18x** |
| table | ascii_10000 | projection | 0.44 MB | CPU | off | **1.00 ms** | 984.2 μs | 8.44 ms | 1.98 ms | — | **8.57x** | **2.01x** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | off | **1.00 ms** | 960.5 μs | 8.41 ms | 1.97 ms | — | **8.76x** | **2.05x** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | off | **188.4 μs** | 176.5 μs | 2.64 ms | 518.7 μs | — | **14.94x** | **2.94x** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | off | **27.6 μs** | 28.7 μs | 419.5 μs | 60.9 μs | — | **15.22x** | **2.21x** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | off | **127.8 μs** | 137.6 μs | 1.51 ms | 169.6 μs | — | **11.80x** | **1.33x** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | off | **127.8 μs** | 133.8 μs | 1.50 ms | 168.4 μs | — | **11.77x** | **1.32x** |
| table | ascii_1000 | projection | 50.6 KB | CPU | off | **189.9 μs** | 165.0 μs | 2.20 ms | 338.8 μs | — | **13.34x** | **2.05x** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | off | **189.1 μs** | 168.3 μs | 2.20 ms | 322.4 μs | — | **13.04x** | **1.92x** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | off | **119.7 μs** | 100.9 μs | 1.95 ms | 198.9 μs | — | **19.32x** | **1.97x** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | off | **27.6 μs** | 28.3 μs | 411.2 μs | 64.2 μs | — | **14.91x** | **2.33x** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | off | **14.34 ms** | 14.35 ms | 17.01 ms | 20.35 ms | — | **1.19x** | **1.42x** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | off | **10.98 ms** | 11.23 ms | 13.67 ms | 16.95 ms | — | **1.25x** | **1.54x** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | off | **11.21 ms** | 13.16 ms | 18.44 ms | 31.45 ms | — | **1.64x** | **2.81x** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | off | **31.71 ms** | 31.73 ms | 363.12 ms | 115.57 ms | — | **11.45x** | **3.64x** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | off | **306.4 μs** | 294.5 μs | 14.46 ms | 1.55 ms | — | **49.08x** | **5.28x** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | off | **33.2 μs** | 32.6 μs | 438.0 μs | 80.2 μs | — | **13.44x** | **2.46x** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | off | **1.57 ms** | 1.57 ms | 3.24 ms | 2.23 ms | — | **2.07x** | **1.43x** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | off | **1.21 ms** | 1.24 ms | 2.89 ms | 1.85 ms | — | **2.39x** | **1.53x** |
| table | mixed_100000 | projection | 5.06 MB | CPU | off | **1.37 ms** | 1.34 ms | 3.32 ms | 3.25 ms | — | **2.48x** | **2.43x** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | off | **2.56 ms** | 2.49 ms | 34.81 ms | 10.36 ms | — | **13.96x** | **4.15x** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | off | **304.2 μs** | 288.1 μs | 6.41 ms | 1.54 ms | — | **22.26x** | **5.33x** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | off | **28.1 μs** | 27.9 μs | 433.9 μs | 79.1 μs | — | **15.56x** | **2.84x** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | off | **213.5 μs** | 258.7 μs | 1.98 ms | 375.4 μs | — | **9.27x** | **1.76x** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | off | **163.9 μs** | 215.9 μs | 1.94 ms | 342.4 μs | — | **11.81x** | **2.09x** |
| table | mixed_10000 | projection | 0.51 MB | CPU | off | **188.6 μs** | 167.9 μs | 1.95 ms | 480.3 μs | — | **11.64x** | **2.86x** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | off | **295.1 μs** | 284.9 μs | 4.98 ms | 1.12 ms | — | **17.48x** | **3.92x** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | off | **139.2 μs** | 132.8 μs | 3.04 ms | 329.0 μs | — | **22.86x** | **2.48x** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | off | **28.9 μs** | 27.9 μs | 447.0 μs | 80.7 μs | — | **16.03x** | **2.89x** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | off | **69.3 μs** | 120.1 μs | 1.83 ms | 189.2 μs | — | **26.41x** | **2.73x** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | off | **67.1 μs** | 117.8 μs | 1.82 ms | 183.5 μs | — | **27.17x** | **2.73x** |
| table | mixed_1000 | projection | 0.06 MB | CPU | off | **115.7 μs** | 98.0 μs | 1.83 ms | 197.6 μs | — | **18.71x** | **2.02x** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | off | **134.4 μs** | 116.0 μs | 2.23 ms | 271.2 μs | — | **19.27x** | **2.34x** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | off | **132.4 μs** | 108.5 μs | 2.69 ms | 207.4 μs | — | **24.80x** | **1.91x** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | off | **26.2 μs** | 30.0 μs | 452.4 μs | 74.4 μs | — | **17.26x** | **2.84x** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | off | **9.36 ms** | 9.31 ms | 8.73 ms | 14.83 ms | — | **0.94x** | **1.59x** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | off | **5.92 ms** | 5.94 ms | 5.37 ms | 11.27 ms | — | **0.91x** | **1.91x** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | off | **4.59 ms** | 4.37 ms | 6.02 ms | 23.77 ms | — | **1.38x** | **5.44x** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | off | **6.12 ms** | 6.06 ms | 7.36 ms | 6.01 ms | — | **1.21x** | **0.99x** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | off | **163.1 μs** | 150.6 μs | 4.04 ms | 588.8 μs | — | **26.80x** | **3.91x** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | off | **35.1 μs** | 27.2 μs | 433.5 μs | 62.4 μs | — | **15.94x** | **2.29x** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | off | **1.06 ms** | 1.07 ms | 2.17 ms | 1.64 ms | — | **2.06x** | **1.56x** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | off | **709.2 μs** | 729.4 μs | 1.83 ms | 1.28 ms | — | **2.58x** | **1.80x** |
| table | narrow_100000 | projection | 1.25 MB | CPU | off | **562.3 μs** | 529.6 μs | 1.89 ms | 2.55 ms | — | **3.57x** | **4.82x** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | off | **710.7 μs** | 688.5 μs | 2.01 ms | 716.1 μs | — | **2.91x** | **1.04x** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | off | **171.3 μs** | 154.8 μs | 2.13 ms | 596.7 μs | — | **13.74x** | **3.85x** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | off | **29.1 μs** | 26.9 μs | 439.8 μs | 65.6 μs | — | **16.34x** | **2.44x** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | off | **143.5 μs** | 196.0 μs | 1.40 ms | 304.1 μs | — | **9.79x** | **2.12x** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | off | **110.6 μs** | 167.2 μs | 1.37 ms | 266.0 μs | — | **12.36x** | **2.40x** |
| table | narrow_10000 | projection | 0.13 MB | CPU | off | **140.8 μs** | 120.8 μs | 1.39 ms | 392.8 μs | — | **11.47x** | **3.25x** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | off | **160.1 μs** | 135.2 μs | 1.42 ms | 196.6 μs | — | **10.50x** | **1.45x** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | off | **113.4 μs** | 92.9 μs | 1.81 ms | 203.4 μs | — | **19.50x** | **2.19x** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | off | **25.8 μs** | 26.4 μs | 428.2 μs | 64.1 μs | — | **16.61x** | **2.49x** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | off | **66.5 μs** | 118.3 μs | 1.32 ms | 171.6 μs | — | **19.89x** | **2.58x** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | off | **59.4 μs** | 111.2 μs | 1.33 ms | 166.2 μs | — | **22.30x** | **2.80x** |
| table | narrow_1000 | projection | 19.7 KB | CPU | off | **106.7 μs** | 82.9 μs | 1.32 ms | 182.0 μs | — | **15.95x** | **2.19x** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | off | **111.9 μs** | 91.1 μs | 1.36 ms | 149.0 μs | — | **14.93x** | **1.64x** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | off | **103.1 μs** | 88.9 μs | 1.75 ms | 159.4 μs | — | **19.67x** | **1.79x** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | off | **26.0 μs** | 25.6 μs | 433.2 μs | 65.8 μs | — | **16.92x** | **2.57x** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | off | **810.5 μs** | 848.4 μs | 1.93 ms | 1.42 ms | — | **2.39x** | **1.75x** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | off | **825.4 μs** | 839.5 μs | 1.94 ms | 1.41 ms | — | **2.35x** | **1.71x** |
| table | typed_100000 | projection | 2.39 MB | CPU | off | **3.57 ms** | 3.50 ms | 32.97 ms | 13.28 ms | — | **9.43x** | **3.80x** |
| table | typed_100000 | read_full | 2.39 MB | CPU | off | **5.31 ms** | 5.26 ms | 33.12 ms | 14.39 ms | — | **6.30x** | **2.74x** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | off | **639.6 μs** | 625.4 μs | 5.13 ms | 1.94 ms | — | **8.20x** | **3.10x** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | off | **30.4 μs** | 28.6 μs | 442.5 μs | 71.5 μs | — | **15.49x** | **2.50x** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | off | **124.2 μs** | 180.7 μs | 1.43 ms | 286.0 μs | — | **11.51x** | **2.30x** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | off | **125.1 μs** | 167.5 μs | 1.43 ms | 290.7 μs | — | **11.46x** | **2.32x** |
| table | typed_10000 | projection | 0.24 MB | CPU | off | **461.9 μs** | 438.5 μs | 4.45 ms | 1.49 ms | — | **10.16x** | **3.39x** |
| table | typed_10000 | read_full | 0.24 MB | CPU | off | **626.7 μs** | 604.9 μs | 4.46 ms | 1.60 ms | — | **7.37x** | **2.65x** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | off | **166.6 μs** | 143.3 μs | 2.22 ms | 404.5 μs | — | **15.52x** | **2.82x** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | off | **28.2 μs** | 28.9 μs | 437.7 μs | 69.4 μs | — | **15.53x** | **2.46x** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | off | **719.9 μs** | 715.4 μs | 1.79 ms | 1.28 ms | — | **2.50x** | **1.79x** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | off | **692.3 μs** | 706.0 μs | 1.80 ms | 1.28 ms | — | **2.60x** | **1.84x** |
| table | varlen_100000 | projection | 3.06 MB | CPU | off | **72.57 ms** | 10.68 ms | 533.42 ms | 112.54 ms | — | **49.94x** | **10.53x** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | off | **72.21 ms** | 10.57 ms | 531.17 ms | 112.24 ms | — | **50.28x** | **10.62x** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | off | **7.39 ms** | 1.08 ms | 54.87 ms | 12.26 ms | — | **50.74x** | **11.34x** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | off | **28.4 μs** | 27.8 μs | 429.8 μs | 66.2 μs | — | **15.45x** | **2.38x** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | off | **117.0 μs** | 167.6 μs | 1.33 ms | 266.5 μs | — | **11.41x** | **2.28x** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | off | **113.5 μs** | 159.8 μs | 1.34 ms | 273.5 μs | — | **11.78x** | **2.41x** |
| table | varlen_10000 | projection | 0.31 MB | CPU | off | **7.08 ms** | 1.08 ms | 53.72 ms | 11.22 ms | — | **49.69x** | **10.38x** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | off | **7.17 ms** | 1.11 ms | 53.74 ms | 11.32 ms | — | **48.44x** | **10.21x** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | off | **815.3 μs** | 195.3 μs | 7.05 ms | 1.43 ms | — | **36.09x** | **7.32x** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | off | **28.0 μs** | 27.5 μs | 441.9 μs | 63.0 μs | — | **16.05x** | **2.29x** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | off | **59.2 μs** | 118.5 μs | 1.27 ms | 168.6 μs | — | **21.45x** | **2.85x** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | off | **57.8 μs** | 111.3 μs | 1.27 ms | 166.8 μs | — | **22.00x** | **2.89x** |
| table | varlen_1000 | projection | 39.4 KB | CPU | off | **813.4 μs** | 191.9 μs | 6.63 ms | 1.32 ms | — | **34.57x** | **6.88x** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | off | **792.9 μs** | 200.2 μs | 6.67 ms | 1.31 ms | — | **33.30x** | **6.53x** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | off | **202.5 μs** | 107.4 μs | 2.23 ms | 303.4 μs | — | **20.78x** | **2.82x** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | off | **27.0 μs** | 27.1 μs | 432.0 μs | 63.3 μs | — | **16.03x** | **2.35x** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | off | **3.61 ms** | 3.63 ms | 8.45 ms | 4.38 ms | — | **2.34x** | **1.21x** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | off | **3.27 ms** | 3.30 ms | 8.17 ms | 4.08 ms | — | **2.50x** | **1.25x** |
| table | wide_100000 | projection | 20.71 MB | CPU | off | **3.45 ms** | 3.41 ms | 8.65 ms | 5.41 ms | — | **2.54x** | **1.59x** |
| table | wide_100000 | read_full | 20.71 MB | CPU | off | **22.05 ms** | 19.27 ms | 146.73 ms | 42.60 ms | — | **7.62x** | **2.21x** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | off | **1.48 ms** | 1.51 ms | 23.27 ms | 5.07 ms | — | **15.67x** | **3.41x** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | off | **41.3 μs** | 39.4 μs | 562.9 μs | 252.0 μs | — | **14.29x** | **6.40x** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | off | **484.2 μs** | 506.5 μs | 5.94 ms | 782.1 μs | — | **12.26x** | **1.62x** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | off | **416.5 μs** | 458.2 μs | 5.87 ms | 754.4 μs | — | **14.10x** | **1.81x** |
| table | wide_10000 | projection | 2.08 MB | CPU | off | **443.3 μs** | 419.2 μs | 5.93 ms | 883.7 μs | — | **14.14x** | **2.11x** |
| table | wide_10000 | read_full | 2.08 MB | CPU | off | **1.50 ms** | 1.51 ms | 18.56 ms | 4.56 ms | — | **12.41x** | **3.05x** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | off | **508.4 μs** | 515.8 μs | 10.70 ms | 895.8 μs | — | **21.04x** | **1.76x** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | off | **36.5 μs** | 35.7 μs | 562.0 μs | 254.6 μs | — | **15.75x** | **7.14x** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | off | **118.6 μs** | 176.6 μs | 5.59 ms | 396.1 μs | — | **47.11x** | **3.34x** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | off | **114.1 μs** | 162.9 μs | 5.59 ms | 393.4 μs | — | **49.01x** | **3.45x** |
| table | wide_1000 | projection | 0.22 MB | CPU | off | **165.7 μs** | 136.6 μs | 5.61 ms | 409.5 μs | — | **41.03x** | **3.00x** |
| table | wide_1000 | read_full | 0.22 MB | CPU | off | **501.8 μs** | 516.3 μs | 7.22 ms | 823.8 μs | — | **14.38x** | **1.64x** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | off | **434.1 μs** | 443.7 μs | 9.33 ms | 510.5 μs | — | **21.50x** | **1.18x** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | off | **36.9 μs** | 37.5 μs | 569.3 μs | 247.6 μs | — | **15.44x** | **6.72x** |
| table | ascii_10000 | predicate_filter | 0.44 MB | CPU | on | **419.5 μs** | 423.4 μs | 2.61 ms | — | — | **6.23x** | **—** |
| table | ascii_10000 | predicate_filter_selective | 0.44 MB | CPU | on | **414.2 μs** | 422.9 μs | 2.65 ms | — | — | **6.40x** | **—** |
| table | ascii_10000 | projection | 0.44 MB | CPU | on | **1.05 ms** | 1.02 ms | 8.00 ms | — | — | **7.88x** | **—** |
| table | ascii_10000 | read_full | 0.44 MB | CPU | on | **1.02 ms** | 1.02 ms | 8.02 ms | — | — | **7.86x** | **—** |
| table | ascii_10000 | row_slice | 0.44 MB | CPU | on | **224.8 μs** | 210.2 μs | 2.56 ms | — | — | **12.17x** | **—** |
| table | ascii_10000 | scan_count | 0.44 MB | CPU | on | **26.9 μs** | 26.2 μs | 407.2 μs | — | — | **15.57x** | **—** |
| table | ascii_1000 | predicate_filter | 50.6 KB | CPU | on | **177.7 μs** | 196.2 μs | 1.51 ms | — | — | **8.52x** | **—** |
| table | ascii_1000 | predicate_filter_selective | 50.6 KB | CPU | on | **180.6 μs** | 188.4 μs | 1.51 ms | — | — | **8.34x** | **—** |
| table | ascii_1000 | projection | 50.6 KB | CPU | on | **225.5 μs** | 216.4 μs | 2.17 ms | — | — | **10.04x** | **—** |
| table | ascii_1000 | read_full | 50.6 KB | CPU | on | **226.0 μs** | 219.0 μs | 2.18 ms | — | — | **9.93x** | **—** |
| table | ascii_1000 | row_slice | 50.6 KB | CPU | on | **158.2 μs** | 146.9 μs | 1.95 ms | — | — | **13.30x** | **—** |
| table | ascii_1000 | scan_count | 50.6 KB | CPU | on | **28.3 μs** | 28.9 μs | 412.2 μs | — | — | **14.54x** | **—** |
| table | mixed_1000000 | predicate_filter | 50.55 MB | CPU | on | **5.46 ms** | 5.05 ms | 11.97 ms | — | — | **2.37x** | **—** |
| table | mixed_1000000 | predicate_filter_selective | 50.55 MB | CPU | on | **4.16 ms** | 4.21 ms | 8.78 ms | — | — | **2.11x** | **—** |
| table | mixed_1000000 | projection | 50.55 MB | CPU | on | **7.25 ms** | 7.04 ms | 13.83 ms | — | — | **1.97x** | **—** |
| table | mixed_1000000 | read_full | 50.55 MB | CPU | on | **28.61 ms** | 22.47 ms | 317.16 ms | — | — | **14.12x** | **—** |
| table | mixed_1000000 | row_slice | 50.55 MB | CPU | on | **272.9 μs** | 216.8 μs | 9.07 ms | — | — | **41.84x** | **—** |
| table | mixed_1000000 | scan_count | 50.55 MB | CPU | on | **29.3 μs** | 33.7 μs | 476.0 μs | — | — | **16.25x** | **—** |
| table | mixed_100000 | predicate_filter | 5.06 MB | CPU | on | **996.2 μs** | 742.5 μs | 2.91 ms | — | — | **3.92x** | **—** |
| table | mixed_100000 | predicate_filter_selective | 5.06 MB | CPU | on | **608.8 μs** | 620.7 μs | 2.57 ms | — | — | **4.22x** | **—** |
| table | mixed_100000 | projection | 5.06 MB | CPU | on | **868.1 μs** | 797.9 μs | 2.96 ms | — | — | **3.71x** | **—** |
| table | mixed_100000 | read_full | 5.06 MB | CPU | on | **1.83 ms** | 1.74 ms | 30.72 ms | — | — | **17.61x** | **—** |
| table | mixed_100000 | row_slice | 5.06 MB | CPU | on | **267.5 μs** | 202.5 μs | 5.74 ms | — | — | **28.32x** | **—** |
| table | mixed_100000 | scan_count | 5.06 MB | CPU | on | **29.2 μs** | 30.0 μs | 447.4 μs | — | — | **15.32x** | **—** |
| table | mixed_10000 | predicate_filter | 0.51 MB | CPU | on | **213.3 μs** | 237.5 μs | 1.97 ms | — | — | **9.23x** | **—** |
| table | mixed_10000 | predicate_filter_selective | 0.51 MB | CPU | on | **146.4 μs** | 181.6 μs | 1.92 ms | — | — | **13.12x** | **—** |
| table | mixed_10000 | projection | 0.51 MB | CPU | on | **169.4 μs** | 121.8 μs | 1.95 ms | — | — | **16.00x** | **—** |
| table | mixed_10000 | read_full | 0.51 MB | CPU | on | **259.1 μs** | 208.9 μs | 4.61 ms | — | — | **22.07x** | **—** |
| table | mixed_10000 | row_slice | 0.51 MB | CPU | on | **171.1 μs** | 116.1 μs | 3.00 ms | — | — | **25.82x** | **—** |
| table | mixed_10000 | scan_count | 0.51 MB | CPU | on | **28.7 μs** | 27.3 μs | 438.5 μs | — | — | **16.08x** | **—** |
| table | mixed_1000 | predicate_filter | 0.06 MB | CPU | on | **91.6 μs** | 140.0 μs | 1.84 ms | — | — | **20.10x** | **—** |
| table | mixed_1000 | predicate_filter_selective | 0.06 MB | CPU | on | **84.3 μs** | 133.7 μs | 1.85 ms | — | — | **21.90x** | **—** |
| table | mixed_1000 | projection | 0.06 MB | CPU | on | **132.2 μs** | 90.8 μs | 1.85 ms | — | — | **20.39x** | **—** |
| table | mixed_1000 | read_full | 0.06 MB | CPU | on | **156.3 μs** | 116.7 μs | 2.22 ms | — | — | **19.01x** | **—** |
| table | mixed_1000 | row_slice | 0.06 MB | CPU | on | **151.2 μs** | 110.1 μs | 2.71 ms | — | — | **24.61x** | **—** |
| table | mixed_1000 | scan_count | 0.06 MB | CPU | on | **30.6 μs** | 29.6 μs | 462.0 μs | — | — | **15.61x** | **—** |
| table | narrow_1000000 | predicate_filter | 12.40 MB | CPU | on | **3.43 ms** | 2.45 ms | 8.06 ms | — | — | **3.29x** | **—** |
| table | narrow_1000000 | predicate_filter_selective | 12.40 MB | CPU | on | **1.64 ms** | 1.64 ms | 4.65 ms | — | — | **2.84x** | **—** |
| table | narrow_1000000 | projection | 12.40 MB | CPU | on | **1.96 ms** | 1.91 ms | 5.02 ms | — | — | **2.62x** | **—** |
| table | narrow_1000000 | read_full | 12.40 MB | CPU | on | **3.04 ms** | 2.96 ms | 6.27 ms | — | — | **2.11x** | **—** |
| table | narrow_1000000 | row_slice | 12.40 MB | CPU | on | **169.3 μs** | 118.7 μs | 3.39 ms | — | — | **28.52x** | **—** |
| table | narrow_1000000 | scan_count | 12.40 MB | CPU | on | **27.4 μs** | 30.1 μs | 448.6 μs | — | — | **16.35x** | **—** |
| table | narrow_100000 | predicate_filter | 1.25 MB | CPU | on | **726.3 μs** | 503.9 μs | 2.08 ms | — | — | **4.13x** | **—** |
| table | narrow_100000 | predicate_filter_selective | 1.25 MB | CPU | on | **360.7 μs** | 344.6 μs | 1.74 ms | — | — | **5.06x** | **—** |
| table | narrow_100000 | projection | 1.25 MB | CPU | on | **316.4 μs** | 246.7 μs | 1.79 ms | — | — | **7.25x** | **—** |
| table | narrow_100000 | read_full | 1.25 MB | CPU | on | **414.8 μs** | 354.2 μs | 1.90 ms | — | — | **5.36x** | **—** |
| table | narrow_100000 | row_slice | 1.25 MB | CPU | on | **197.2 μs** | 107.6 μs | 2.06 ms | — | — | **19.16x** | **—** |
| table | narrow_100000 | scan_count | 1.25 MB | CPU | on | **26.3 μs** | 27.0 μs | 443.0 μs | — | — | **16.86x** | **—** |
| table | narrow_10000 | predicate_filter | 0.13 MB | CPU | on | **180.1 μs** | 213.9 μs | 1.44 ms | — | — | **7.98x** | **—** |
| table | narrow_10000 | predicate_filter_selective | 0.13 MB | CPU | on | **115.9 μs** | 159.4 μs | 1.38 ms | — | — | **11.90x** | **—** |
| table | narrow_10000 | projection | 0.13 MB | CPU | on | **143.4 μs** | 95.4 μs | 1.39 ms | — | — | **14.61x** | **—** |
| table | narrow_10000 | read_full | 0.13 MB | CPU | on | **155.9 μs** | 110.8 μs | 1.42 ms | — | — | **12.83x** | **—** |
| table | narrow_10000 | row_slice | 0.13 MB | CPU | on | **138.1 μs** | 85.1 μs | 1.81 ms | — | — | **21.26x** | **—** |
| table | narrow_10000 | scan_count | 0.13 MB | CPU | on | **26.4 μs** | 27.4 μs | 428.9 μs | — | — | **16.22x** | **—** |
| table | narrow_1000 | predicate_filter | 19.7 KB | CPU | on | **82.9 μs** | 132.4 μs | 1.36 ms | — | — | **16.35x** | **—** |
| table | narrow_1000 | predicate_filter_selective | 19.7 KB | CPU | on | **83.8 μs** | 130.5 μs | 1.33 ms | — | — | **15.87x** | **—** |
| table | narrow_1000 | projection | 19.7 KB | CPU | on | **121.4 μs** | 76.5 μs | 1.35 ms | — | — | **17.68x** | **—** |
| table | narrow_1000 | read_full | 19.7 KB | CPU | on | **131.2 μs** | 77.0 μs | 1.37 ms | — | — | **17.81x** | **—** |
| table | narrow_1000 | row_slice | 19.7 KB | CPU | on | **123.5 μs** | 76.7 μs | 1.79 ms | — | — | **23.30x** | **—** |
| table | narrow_1000 | scan_count | 19.7 KB | CPU | on | **27.4 μs** | 28.2 μs | 442.4 μs | — | — | **16.13x** | **—** |
| table | typed_100000 | predicate_filter | 2.39 MB | CPU | on | **642.5 μs** | 498.9 μs | 1.76 ms | — | — | **3.54x** | **—** |
| table | typed_100000 | predicate_filter_selective | 2.39 MB | CPU | on | **519.3 μs** | 504.5 μs | 1.76 ms | — | — | **3.49x** | **—** |
| table | typed_100000 | projection | 2.39 MB | CPU | on | **1.24 ms** | 1.16 ms | 28.68 ms | — | — | **24.81x** | **—** |
| table | typed_100000 | read_full | 2.39 MB | CPU | on | **1.34 ms** | 1.26 ms | 28.86 ms | — | — | **22.99x** | **—** |
| table | typed_100000 | row_slice | 2.39 MB | CPU | on | **268.1 μs** | 212.8 μs | 4.51 ms | — | — | **21.20x** | **—** |
| table | typed_100000 | scan_count | 2.39 MB | CPU | on | **30.4 μs** | 28.3 μs | 434.0 μs | — | — | **15.36x** | **—** |
| table | typed_10000 | predicate_filter | 0.24 MB | CPU | on | **183.3 μs** | 206.3 μs | 1.45 ms | — | — | **7.90x** | **—** |
| table | typed_10000 | predicate_filter_selective | 0.24 MB | CPU | on | **169.3 μs** | 204.5 μs | 1.43 ms | — | — | **8.43x** | **—** |
| table | typed_10000 | projection | 0.24 MB | CPU | on | **250.9 μs** | 197.5 μs | 4.04 ms | — | — | **20.46x** | **—** |
| table | typed_10000 | read_full | 0.24 MB | CPU | on | **260.0 μs** | 207.4 μs | 4.08 ms | — | — | **19.65x** | **—** |
| table | typed_10000 | row_slice | 0.24 MB | CPU | on | **149.4 μs** | 102.6 μs | 2.17 ms | — | — | **21.17x** | **—** |
| table | typed_10000 | scan_count | 0.24 MB | CPU | on | **28.2 μs** | 29.6 μs | 427.4 μs | — | — | **15.18x** | **—** |
| table | varlen_100000 | predicate_filter | 3.06 MB | CPU | on | **586.8 μs** | 464.2 μs | 1.57 ms | — | — | **3.38x** | **—** |
| table | varlen_100000 | predicate_filter_selective | 3.06 MB | CPU | on | **479.7 μs** | 482.0 μs | 1.56 ms | — | — | **3.24x** | **—** |
| table | varlen_100000 | projection | 3.06 MB | CPU | on | **73.62 ms** | 10.66 ms | 527.70 ms | — | — | **49.52x** | **—** |
| table | varlen_100000 | read_full | 3.06 MB | CPU | on | **72.49 ms** | 10.59 ms | 532.51 ms | — | — | **50.28x** | **—** |
| table | varlen_100000 | row_slice | 3.06 MB | CPU | on | **7.37 ms** | 1.16 ms | 54.54 ms | — | — | **47.04x** | **—** |
| table | varlen_100000 | scan_count | 3.06 MB | CPU | on | **27.4 μs** | 28.8 μs | 454.6 μs | — | — | **16.62x** | **—** |
| table | varlen_10000 | predicate_filter | 0.31 MB | CPU | on | **165.4 μs** | 202.8 μs | 1.34 ms | — | — | **8.11x** | **—** |
| table | varlen_10000 | predicate_filter_selective | 0.31 MB | CPU | on | **158.8 μs** | 197.1 μs | 1.34 ms | — | — | **8.41x** | **—** |
| table | varlen_10000 | projection | 0.31 MB | CPU | on | **7.32 ms** | 1.12 ms | 53.83 ms | — | — | **47.92x** | **—** |
| table | varlen_10000 | read_full | 0.31 MB | CPU | on | **7.29 ms** | 1.12 ms | 54.22 ms | — | — | **48.32x** | **—** |
| table | varlen_10000 | row_slice | 0.31 MB | CPU | on | **872.5 μs** | 283.4 μs | 7.03 ms | — | — | **24.81x** | **—** |
| table | varlen_10000 | scan_count | 0.31 MB | CPU | on | **29.8 μs** | 28.3 μs | 446.5 μs | — | — | **15.78x** | **—** |
| table | varlen_1000 | predicate_filter | 39.4 KB | CPU | on | **83.4 μs** | 137.4 μs | 1.28 ms | — | — | **15.40x** | **—** |
| table | varlen_1000 | predicate_filter_selective | 39.4 KB | CPU | on | **81.2 μs** | 138.7 μs | 1.28 ms | — | — | **15.83x** | **—** |
| table | varlen_1000 | projection | 39.4 KB | CPU | on | **873.5 μs** | 231.7 μs | 6.61 ms | — | — | **28.55x** | **—** |
| table | varlen_1000 | read_full | 39.4 KB | CPU | on | **870.6 μs** | 278.6 μs | 6.65 ms | — | — | **23.86x** | **—** |
| table | varlen_1000 | row_slice | 39.4 KB | CPU | on | **239.5 μs** | 149.5 μs | 2.25 ms | — | — | **15.02x** | **—** |
| table | varlen_1000 | scan_count | 39.4 KB | CPU | on | **26.0 μs** | 27.2 μs | 426.4 μs | — | — | **16.38x** | **—** |
| table | wide_100000 | predicate_filter | 20.71 MB | CPU | on | **1.60 ms** | 1.36 ms | 7.42 ms | — | — | **5.46x** | **—** |
| table | wide_100000 | predicate_filter_selective | 20.71 MB | CPU | on | **1.22 ms** | 1.21 ms | 7.04 ms | — | — | **5.80x** | **—** |
| table | wide_100000 | projection | 20.71 MB | CPU | on | **1.64 ms** | 1.58 ms | 7.55 ms | — | — | **4.77x** | **—** |
| table | wide_100000 | read_full | 20.71 MB | CPU | on | **13.85 ms** | 15.98 ms | 127.71 ms | — | — | **9.22x** | **—** |
| table | wide_100000 | row_slice | 20.71 MB | CPU | on | **1.02 ms** | 967.2 μs | 20.75 ms | — | — | **21.45x** | **—** |
| table | wide_100000 | scan_count | 20.71 MB | CPU | on | **41.7 μs** | 40.5 μs | 561.7 μs | — | — | **13.88x** | **—** |
| table | wide_10000 | predicate_filter | 2.08 MB | CPU | on | **311.4 μs** | 334.5 μs | 5.85 ms | — | — | **18.79x** | **—** |
| table | wide_10000 | predicate_filter_selective | 2.08 MB | CPU | on | **255.8 μs** | 277.9 μs | 5.82 ms | — | — | **22.77x** | **—** |
| table | wide_10000 | projection | 2.08 MB | CPU | on | **280.9 μs** | 228.7 μs | 5.80 ms | — | — | **25.38x** | **—** |
| table | wide_10000 | read_full | 2.08 MB | CPU | on | **1.01 ms** | 950.5 μs | 16.60 ms | — | — | **17.47x** | **—** |
| table | wide_10000 | row_slice | 2.08 MB | CPU | on | **504.0 μs** | 451.6 μs | 10.50 ms | — | — | **23.25x** | **—** |
| table | wide_10000 | scan_count | 2.08 MB | CPU | on | **38.3 μs** | 36.5 μs | 566.9 μs | — | — | **15.55x** | **—** |
| table | wide_1000 | predicate_filter | 0.22 MB | CPU | on | **127.5 μs** | 181.0 μs | 5.64 ms | — | — | **44.26x** | **—** |
| table | wide_1000 | predicate_filter_selective | 0.22 MB | CPU | on | **125.7 μs** | 172.8 μs | 5.66 ms | — | — | **45.02x** | **—** |
| table | wide_1000 | projection | 0.22 MB | CPU | on | **175.9 μs** | 132.2 μs | 5.64 ms | — | — | **42.66x** | **—** |
| table | wide_1000 | read_full | 0.22 MB | CPU | on | **499.1 μs** | 451.2 μs | 7.12 ms | — | — | **15.79x** | **—** |
| table | wide_1000 | row_slice | 0.22 MB | CPU | on | **455.8 μs** | 414.0 μs | 9.35 ms | — | — | **22.59x** | **—** |
| table | wide_1000 | scan_count | 0.22 MB | CPU | on | **38.1 μs** | 37.8 μs | 579.1 μs | — | — | **15.31x** | **—** |
<!-- BENCH_FULL_TABLE_END -->

## Performance deficits

<!-- BENCH_DEFICITS_BEGIN -->
Cases where torchfits is **not** first in its comparison family (CPU and GPU). GPU lags may reflect software or hardware limits — they are listed, not hidden.

| Platform | Domain | Case | mmap | torchfits | Peak RSS (MB) | Winner | Lag |
|---|---|---|---|---:|---:|---|---:|
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 45.69 ms | 293.6 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 45.43 ms | 309.1 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | on | 46.02 ms | 293.6 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CPU | tensor | compressed_hcompress_1 [read_full] | off | 45.44 ms | 309.1 | fitsio/fitsio_torch | 1.02× |
| Linux x86_64 / CPU | table | narrow_100000 [read_full] | off | 710.7 μs | 378.1 | fitsio/fitsio_torch | 1.15× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 6.12 ms | 397.2 | fitsio/fitsio_torch | 1.08× |
| Linux x86_64 / CPU | table | narrow_1000000 [predicate_filter_selective] | off | 5.94 ms | 401.4 | astropy/astropy | 1.11× |
| Linux x86_64 / CPU | table | narrow_1000000 [predicate_filter] | off | 9.31 ms | 401.4 | astropy/astropy | 1.07× |
| Linux x86_64 / CPU | table | narrow_1000000 [read_full] | off | 6.06 ms | 401.3 | fitsio/fitsio | 1.01× |
| Linux x86_64 / CUDA | tensor | tiny_float32_3d [read_full @ cuda] | off | 117.1 μs | 766.0 | fitsio/fitsio_torch_device | 1.07× |
| Linux x86_64 / CUDA | tensor | tiny_int32_1d [read_full @ cuda] | off | 104.1 μs | 766.0 | fitsio/fitsio_torch_device | 1.06× |
| Linux x86_64 / CUDA | tensor | medium_int8_1d [read_full @ cuda] | off | 134.3 μs | 766.0 | fitsio/fitsio_torch_device | 1.06× |
| Linux x86_64 / CUDA | tensor | tiny_float64_2d [read_full @ cuda] | off | 113.2 μs | 766.0 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | tiny_int32_2d [read_full @ cuda] | off | 108.4 μs | 766.0 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | small_int16_2d [read_full @ cuda] | off | 141.4 μs | 766.0 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | tiny_int16_1d [read_full @ cuda] | off | 102.2 μs | 766.0 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | small_float64_1d [read_full @ cuda] | off | 124.7 μs | 766.0 | fitsio/fitsio_torch_device | 1.05× |
| Linux x86_64 / CUDA | tensor | tiny_float32_2d [read_full @ cuda] | off | 112.5 μs | 766.0 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | medium_int16_1d [read_full @ cuda] | off | 153.9 μs | 766.0 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | small_int16_1d [read_full @ cuda] | off | 111.6 μs | 766.0 | fitsio/fitsio_torch_device | 1.04× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | on | 30.42 ms | 611.2 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | off | 30.45 ms | 734.2 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | small_uint16_2d [read_full @ cuda] | off | 146.0 μs | 766.0 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | on | 30.44 ms | 704.0 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | small_int8_1d [read_full @ cuda] | off | 109.4 μs | 766.0 | fitsio/fitsio_torch_device | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | off | 30.27 ms | 772.1 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | tensor | tiny_int8_2d [read_full @ cuda] | off | 121.5 μs | 766.0 | fitsio/fitsio_torch_device | 1.02× |
| Linux x86_64 / CUDA | tensor | tiny_int8_3d [read_full @ cuda] | off | 124.0 μs | 766.0 | fitsio/fitsio_torch_device | 1.02× |
| Linux x86_64 / CUDA | tensor | tiny_int64_1d [read_full @ cuda] | off | 101.9 μs | 766.0 | fitsio/fitsio_torch_device | 1.01× |
| Linux x86_64 / CUDA | tensor | small_int8_3d [read_full @ cuda] | off | 148.5 μs | 766.0 | fitsio/fitsio_torch_device | 1.01× |
| Linux x86_64 / CUDA | tensor | tiny_int8_1d [read_full @ cuda] | off | 103.5 μs | 766.0 | fitsio/fitsio_torch_device | 1.00× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | off | 30.38 ms | 772.1 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | off | 30.42 ms | 734.2 | fitsio/fitsio_torch_device_specialized | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full @ cuda] | on | 30.45 ms | 704.0 | fitsio/fitsio_torch_device_specialized | 1.03× |
| Linux x86_64 / CUDA | tensor | compressed_hcompress_1 [read_full] | on | 30.19 ms | 611.2 | fitsio/fitsio_torch | 1.03× |
| Linux x86_64 / CUDA | table | narrow_100000 [read_full] | off | 830.7 μs | 698.3 | fitsio/fitsio_torch | 1.13× |
| Linux x86_64 / CUDA | table | narrow_1000000 [read_full] | off | 7.48 ms | 718.3 | fitsio/fitsio_torch | 1.11× |
| Linux x86_64 / CUDA | table | narrow_1000000 [predicate_filter_selective] | off | 7.95 ms | 722.6 | astropy/astropy | 1.07× |
| Linux x86_64 / CUDA | table | narrow_1000000 [predicate_filter] | off | 12.04 ms | 722.6 | astropy/astropy | 1.05× |
| Linux x86_64 / CPU | tensor | scaled_large [read_full] | off | 13.57 ms | 303.6 | fitsio/fitsio_torch | 1.11× |

_…and 4 more rows in `torchfits_deficits.csv`._
<!-- BENCH_DEFICITS_END -->

### Host scorecard

| Platform | Run ID | Rows | Time deficits | Median peak RSS (MB) | Notes |
|---|---|---:|---:|---:|---|
<!-- BENCH_HOSTS_BEGIN -->
| Linux x86_64 / CPU | `exhaustive_cpu_20260806_012620` | 3057 | 9 | 293.6 | lab + mmap-matrix |
| Linux x86_64 / CUDA | `exhaustive_cuda_20260806_012651` | 4315 | 30 | 722.6 | lab + mmap-matrix + GPU |
| Linux x86_64 / CPU | `exhaustive_cpu_20260806_022603` | 3057 | 5 | 300.2 | lab + mmap-matrix |
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
