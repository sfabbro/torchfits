# Plan: perf backlog items 1–10 + output-parity gate (2026-08-04)

Status: COMPLETED with dispositions (2026-08-05). Approved by human (2026-08-04);
Wave 0 was executed first;
every perf change is gated by the Wave 0 parity suite.

## Wave 0 — Cross-library output-parity suite (correctness gate)

New `tests/test_output_parity.py` + fuzz-lite generator, extending the
`test_interop.py` / `test_fitsio_upstream_smoke.py` /
`test_astropy_upstream_smoke.py` patterns.

- Read parity, parametrized over every grid fixture (`benchmarks/bench_fits_io.py`
  fixture generator: `scaled_{s,m,l}`, `{s,m,l}_uint16_2d`, `_uint32_2d`,
  `_int8_3d` cubes, compressed RICE/GZIP/HCOMPRESS, tables with TSCAL/TZERO, VLA,
  strings): torchfits output must equal fitsio.read and astropy exactly —
  `np.array_equal` bitwise for ints; for float-scaled, **exact-against-fitsio**
  (identical CFITSIO BSCALE/BZERO math), not approximate.
- Write parity, both directions:
  - torchfits-written files (incl. `quantize=`, TSCAL/TZERO tables, LONGSTRN,
    uint64 rejection) read back by **fitsio only** for every compression type —
    fitsio is the raw CFITSIO wrapper and the cheap ground truth; astropy stays
    only for string/VLA table decode and header standard-compliance spot checks.
  - fitsio/astropy-written files read by torchfits.
- Randomized fuzz-lite: seeded random dtype/shape/BSCALE/BZERO/compression →
  cross-lib exact compare both directions (~50 seeds, small shapes; skip slow
  combos to keep ci-local fast).

Decisions: float-scaled = exact-against-fitsio (approved). Round-trip reader =
fitsio only (approved; astropy only where its semantics layer is uniquely useful).

## Wave 1 — >2D image/cube `read_full` lag cluster (profile-driven)

From `torchfits_deficits.csv` (731 rows, 2026-08-04 matrix):

| case | CPU lag | CUDA lag |
|---|---|---|
| `large_uint16_2d` | 1.149 (specialized) | 1.124 |
| `small_int8_3d` | 1.112 | 1.021 |
| `large_int8_2d` | 1.075 | — |
| `small_int16_3d` | 1.064 | 1.049 |
| `medium_int8_3d` | 1.062 | 1.001 |
| `scaled_large` | 1.050 | 1.012 |
| `large_int64_2d` / `medium_uint16_2d` | 1.047 / 1.045 | — |
| `large_int16_2d` | 1.042 | — |
| `compressed_hcompress_1` | 1.022 (162 rows) | 1.032 (84) |

Hypothesis: uint16/int8 cluster (BITPIX=16 + BZERO=32768) pays extra elementwise
pass or double copy vs fitsio numpy path; 3D cube cases may be launch/path
overhead. Profile `_read_cpu_fast_path` / `_read_generic_fast_path`
(`_read_pipeline.py`), `read_full_gpu` (`bench_gpu_transports.py:205`), image
bindings (`fits_image_mmap.cpp` / `fits_bindings.cpp`). Smallest lever; exit ≤1.03
same host.

## Wave 2 — cheap Python items

- B1 duplicate capability-check helpers (`_table/_read_schema.py:187` /
  `mutation.py` `_parse_tform`)
- P10 OrderedDict cache locks (`_io_engine/caches.py` + `image_meta.py`, one RLock)
- P3 `stack().to().unbind()` VRAM (`_io_engine/image.py:30`; chunked or
  per-tensor; gated on measured peak VRAM)

## Wave 3 — C++ cache internals

- NIT-CPP-2 tl_cache LRU + uid-bump eviction (`fits_bindings.cpp:116-128`; also
  closes R7-CPP2 stale entries)
- P5/P8 SharedReadMeta lock coalescing + `shared_mutex` (`fits_detail.h:219-262`);
  keep only what the bench confirms

## Wave 4 — remaining

- 10 CUDA small `read_full_gpu` cluster: **measure + document** (nvtx/Nsight split
  launch vs H2D vs conversion); prove/disprove overhead hypothesis; close with
  `benchmarks.md` note unless measurement shows avoidable cost. No speculative code.
- 8 combined one-open mutation: fitsio 1.4.2 has no `insert_rows` (TableHDU:
  `delete_rows`/`resize`/`write_column`/`insert_column`); astropy io.fits has none
  either — but both achieve insert+update with one open handle (fitsio:
  `resize`+`write_column` on one RW `FITS`; astropy: in-memory + single `writeto`).
  Competitive pattern exists → implement combined one-open `cpp` capsule
  (insert-shift + update writes in one `fitsfile*`) + private Python helper.
  Our arbitrary-position insert stays a superset; note in docs.
- 5 P11/P12: audit-first (device-path cache behavior on repeated GPU reads; table
  `read_torch(device=)` bounce) → implement only if audit shows measurable
  repeat-read penalty, else documented disposition.
- 9 MegaCam Rice: repro-first (same-host bench leg + `materialize` copy trace);
  touch decompress path only if repro lags AND trace shows avoidable copies; else
  close documented.
- 1 narrow-table `read_full` polish (last; profile pass after >2D closed).

## Execution rules

- Each item = one commit; `pixi run preflight-push` per item; full `pixi run
  ci-local` at wave boundaries; bench before/after same-host per item; results
  logged in `.cursor/harness/deep-review-1-faithfulness.md` resolution log.
- Wave 0 parity suite lands first and gates every subsequent diff.

## Disposition

- Wave 0: landed in `9bad157`.
- Wave 1: profiled; no stable same-host >2D gap reproduced; speculative scaling
  shortcut rejected by exact parity/performance evidence.
- Wave 2: P10 landed in `e3d61ad`; B1 was already single-sourced; P3 deferred
  pending a CUDA host.
- Wave 3: landed in `a1796f3`.
- Wave 4: measured/documented or gated no-change outcomes; no speculative
  CUDA, mutation, GPU-cache, MegaCam, or narrow-table patch landed.
