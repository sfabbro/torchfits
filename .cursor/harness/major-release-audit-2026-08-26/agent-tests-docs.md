# Major-release audit — tests, docs, examples, benchmarks

**Repo:** `/Users/fabbros/src/torchfits`
**Scope date:** 2026-08-26
**Product version in tree:** `1.1.0` (`src/torchfits/__init__.py`, `pyproject.toml`, `pixi.toml`, `scripts/torch_lanes.json`)
**This document:** tests + docs + examples + benches + `docs/assets`. No code changes.

**Verdict:** **Ship with notes — do not tag until published scorecard CSVs match the headline.** Core I/O/docs contract is largely in place (`test_docs_integrity.py` env/signature fences, malformed-FITS suite, concurrent reads, CLI coverage of 14/15 subcommands). The release-blocking gap is scientific honesty of the 1.1.0 benchmark headline: run IDs `exhaustive_cpu_20260822_213823` / `exhaustive_cuda_20260822_213846` are cited in `docs/benchmarks.md` and `docs/changelog.md` but are **not** mirrored under `docs/assets/bench/`. Repo-attached CSVs are older (20260807 / 20260719).

Severity: **BLOCKING** (tag-stopper) · **HIGH** (fix before tag if time) · **MEDIUM** (next patch) · **LOW** (nits).

---

## Issues

### BLOCKING

**B1. Headline scorecard CSVs are not in the published tree**
- `docs/benchmarks.md` headline (v1.1.0) and host table cite `exhaustive_cpu_20260822_213823` (3057 rows, 7 time deficits) and `exhaustive_cuda_20260822_213846` (4315 rows, 32 deficits). Changelog Performance section repeats those IDs.
- HTML comment in `docs/benchmarks.md` admits: “Those per-run CSVs are archived on the bench host / CANFAR VOSpace and are **pending mirroring** into `docs/assets/bench/`.”
- “Published CSVs” links still point at `exhaustive_*_20260807_013736` and `exhaustive_*_20260719_*`.
- The I/O-path table is sourced from `benchmarks_results/exhaustive_cpu_20260822_213823/results.csv` (gitignored local dir), not from a committed asset.
- Independently recomputing the “100% of significant image comparisons / 98.6–100% of table comparisons” claim from git is impossible.
- **Fix before tag:** mirror the 20260822 CSVs + `torchfits_deficits.csv` + `summary.md` into `docs/assets/bench/<run-id>/` and point Published paths at them, **or** retarget the headline to the 20260807 artifacts that *are* in git (and lower the win-rate language to match those summaries: CPU 20260807 has **3 significant** table deficits, image lags vs fitsio on HCOMPRESS marked `noise`).

**B2. `docs/examples-ml.md` cutout timings are not the published MegaCam/MegaPipe CSVs**
- Claims: 1,000 × 64×64 stamps from a 1.74 GB uncompressed MegaPipe mosaic; `open_subset_reader` **0.060 s** total (0.060 ms/stamp), astropy 0.149 s, `read_subset` 0.165 s, fitsio 0.297 s.
- Committed CSVs (`docs/assets/bench/20260719_075555/megacam_results.csv` and the MPS copy) measure **40** Rice-compressed MegaCam CCD cutouts (~0.12 s specialized torchfits, ~0.06 s *materialize* path). Different payload, N, compression, and method labels.
- No MegaPipe 1000-stamp CSV exists under `docs/assets/bench/`.
- Treat as an unsourced performance claim on a user-facing page. Either cite a committed CSV or drop the table.

### HIGH

**H1. GPU-decode language overclaims vs the honest bench page**
- `docs/benchmarks.md` correctly states every Python FITS stack here decodes on host then `.to(device)`; `disk→GPU` is empty; GPUDirect is not implemented.
- `docs/index.md` comparison table: “Native `device="cuda"` / `"mps"` decode”.
- `docs/python-workflows.md` / `docs/quickstart.md`: “Direct decode to target device”.
- `docs/install.md` / README: “Prebuilt wheels automatically enable GPU acceleration” — wheels are CPU-linked (`USE_CUDA=OFF` in scikit-build). GPU is PyTorch device placement after host decode.
- **Ask:** keep the bench honesty; soften marketing to “place result on CUDA/MPS after host decode”.

**H2. Python × PyTorch matrix champion is not the wheel lane**
- `docs/benchmarks.md` “Python & PyTorch Matrix Variance” names **PyTorch 2.12 + Python 3.11** as Pareto champion.
- Current ABI lane is **2.13.x** (`torch>=2.13,<2.14`). Variance tables still include 2.10–2.13.
- Risk: readers think 2.12 is the supported/optimal install. Either label the matrix as a historical CANFAR grid, or re-run on the 2.13 lane and say so.

**H3. Dataset return contracts disagree between `api-data.md` and `examples-ml.md`**
- `FitsCutoutDataset`: `api-data.md` and source (`data/__init__.py` ~442) return a **tensor only** (no `labels`). `examples-ml.md` “Dataset Selection Guide” says `(cutout_tensor, label)`.
- `FitsSpectrumDataset`: API “payload — no label”; ML page `(flux_tensor, label)`.
- `make_loader`: `pin_memory` default is **`False`** in source and `api-data.md`. `examples-ml.md` says “Enabled by default for CUDA environments”.
- `python-workflows.md` calls transforms “**Differentiable**”; `api-transforms.md` says they are **not** `nn.Module` and there is no certified `torch.compile` matrix. SigmaClip is lossy, not a gradient op.

**H4. `api-transforms.md` omits `fill="nan"`**
- Source `transforms/clip.py`: `SigmaClip.fill` in `{"mean","median","nan"}`; `AsymmetricSigmaClip.fill` in `{"median","nan"}`. Changelog 1.1.0 documents `fill="nan"`.
- Docs table: SigmaClip `"mean" or "median"` only; AsymmetricSigmaClip has no `fill` row.

**H5. Tests that encode a removed implementation (false confidence)**
- `tests/test_compression.py` `test_large_float_rice_parallel_matches_serial` / `test_hcompress_parallel_matches_serial` set `TORCHFITS_COMPRESSED_PARALLEL`, `_MIN_PIXELS`, `_HCOMPRESS`.
- Those names do **not** appear in `src/` (Python or C++). Both “serial” and “parallel” take the same path; the tests cannot fail if parallel decode regresses.
- Same class of smell: `tests/test_io.py` mocks `torchfits.io.cpp` (`read_full_raw_with_scale`, `read_hdus_batch`, …) — passes if the façade still calls those private names, not if CFITSIO reads are correct. Real coverage lives in `test_output_parity.py` / `test_write_fidelity.py`.

**H6. CWD pollution (pytest-xdist / crash leftovers)**
These write `test_*.fits` into the **process CWD** (repo root when invoked as `pixi run test`) and unlink in `finally`. Parallel workers collide; KeyboardInterrupt leaves files. `*.fits` is gitignored so they will not be committed, but they still pollute the working tree.

| File | CWD names |
|---|---|
| `tests/test_writing.py` | `test_write.fits`, `test_write_hdulist.fits`, `test_write_bool_table.fits`, `test_write_list_table_dict.fits`, `test_write_rich_table.fits`, `test_write_complex_tensor_table.fits`, `test_write_compressed_*.fits`, `test_write_cuda_tensor_host_copy.fits`, `test_write_mps_tensor_host_copy.fits` |
| `tests/test_read_header.py` | `test_read_header.fits` |
| `tests/test_validation.py` | `test_valid.fits` |
| `tests/test_ascii_table.py` | `test_ascii.fits` |
| `tests/test_complex_header.py` | `test_complex_header.fits` |

**H7. Unused pytest markers**
- `pyproject.toml` `[tool.pytest.ini_options] markers`: `performance`, `integration`.
- Zero `@pytest.mark.performance` / `@pytest.mark.integration` in the tree.
- `tests/test_performance.py` and `tests/test_integration.py` exist as classes but are not marked, so `-m "not performance"` cannot skip them. `test_performance.py` uses wall-clock `time` + `psutil` (CI-flaky).

**H8. C++ self-check is not in any pixi/CI gate**
- `tests/cpp/test_bracket_detection.cpp` — header-only asserts for CFITSIO extended-filename vs literal `[dir]`. Comment says run `clang++` by hand.
- No reference in `pixi.toml`, GitHub workflows, or pytest. Python `test_security.py` covers a path with `[data]` in the directory; the C++ helper itself is unguarded in CI.

**H9. CLI `table` has no functional test**
- `test_cli.py` `test_help_lists_subcommands` expects `"table"` in `--help`.
- No `_run_cli("table", …)` asserting schema/preview/JSON. `cmds_table.py` is otherwise unexercised from the CLI process.

**H10. Install / compatibility / Windows**
- `docs/install.md` documents MSVC 2019+ / Windows source builds (`=== "Windows"`).
- `docs/compatibility.md` source builds: “Linux, macOS” only. `pixi.toml` platforms: `osx-arm64`, `linux-64`. cibuildwheel skips Windows.
- Either drop the Windows recipe or mark it unsupported/experimental.

**H11. `docs/parity.md` lags 1.1.0 features**
- No row for whole-file `.bz2` reads (`HAS_BZIP2`) or tile `BZIP2_1` (tested in `test_compression_matrix.py` / `test_bz2.py`; changelog Features).
- Compression write examples use `compress="rice"` / `"gzip"` / `"hcompress"`; C++ aliases those, so they work, but CLI/docs mix `RICE_1` vs short names.
- Status column is almost all **Supported**; VLA is the only **Partial**. No **Unsupported** rows (Random Groups is a loud failure in code/`test_malformed_fits.py`, not listed).

**H12. Replay gate is vacuous**
- `benchmarks/replays/` is gitignored except `upstream_sources.json` (tracked, 277 bytes).
- Manifest `replays: []` for both fitsio and astropy. `test_upstream_parity_inventory.py` only checks that listed paths exist.
- `docs/release.md` checklist “references the parity tests that justify comparator claims” is satisfied without any replay fixtures.

**H13. Changelog `## Unreleased` still has commits after a dated 1.1.0**
- `## [1.1.0] — 2026-08-26` plus Unreleased bullets (vendor.sh sha256, changelog tag ranking). If the tag has not been cut, stamp again or fold; if it has, Unreleased is correct — confirm before `git tag v1.1.0`.

**H14. `docs/release.md` still talks about tagging SemVer 1.0.0**
- Section 4: “Before tagging a SemVer `1.0.0`”. Tree is already 1.1.0. rc example still shows `1.0.0rc5`. Update the freeze runbook for 1.x minors.

### MEDIUM

**M1. Implementation-not-intent tests (grouped)**
- `tests/test_deep_review_p0.py`, `wave2.py`, `wave4.py`, `wave5.py`: mock `_acquire_cpp_reader`, `_C`, private `_TORCH_WHERE_MAX_ROWS`, GIL/open counting. Valuable as regression pins for past review IDs; they will rot when internals move (already a pattern with COMPRESSED_PARALLEL).
- `tests/test_bug_table_hdu_cache.py`, `test_bug_table_ref.py`: named after bugs, assert cache invalidation — intent is OK; names are archaeology.
- Several tests import deprecated `torchfits.cpp` (`test_api.py`, `test_arrow_table_api.py`, `test_cache.py`, `test_table_filtering.py`, `test_dlpack_roundtrip.py`, `test_writing.py`). `test_public_boundary.py` correctly asserts the DeprecationWarning.

**M2. `/tmp` leaks (not CWD, still dirty)**
- Widespread `NamedTemporaryFile(..., delete=False)` + `os.unlink` in `finally` — fail-before-finally or skipped unlink leaves `/tmp/*.fits`.
- `tests/test_patch_bench_docs.py` writes `/tmp/pdoc_test/<run-id>/results.csv` with **no cleanup**.
- `tests/test_data.py` `tempfile.mkdtemp(prefix="torchfits_data_test_")`.

**M3. Coverage gaps vs `src/` (behavior, not line coverage)**
| Area | Tests | Gap |
|---|---|---|
| `cli/cmds_table.py` | help only | No schema/preview/JSON/error path |
| `logging.py` | none | NullHandler contract untested (low risk) |
| `tests/cpp/test_bracket_detection.cpp` | not run | See H8 |
| `FitsSpectrumDataset` labels | iterable cube/spectrum smoke in `test_data_datasets.py` | Map-style spectrum **no-label** vs ML docs |
| `table` CLI | — | H9 |
| FTP remote | `test_cli_rejects_ftp_remote_paths` | Good |
| `torchfits.logging` | — | M3 |
| Concurrent image **write** / mutate-while-read | `test_concurrent_same_file_read.py` is read-only; `test_release_semantics.py` has 4-thread cache; `test_deep_review_wave2.py` reader vs closer | No writer/reader race on one path |
| Malformed FITS | `test_malformed_fits.py` (magic, trunc image/table, garbage tail, VLA OOB, GROUPS, checksum, dup EXTNAME, fuzz) | Solid; truncated tables also `test_truncated_table_errors.py` |
| `where=` engine matrix | `test_where.py`, `test_table_filtering.py`, `test_public_where.py`, `test_arrow_table_api.py` | `read_torch` simple dialect vs `table.read` full dialect is documented; fewer tests that the **same** expression is rejected on `read_torch` for OR/IN/NOT |
| HTTP Range | `test_remote_http_range.py` | Strong (auth, resume, 206, multiprocess lock) |
| `vos_uri` | via `test_remote_http_range.py` placeholders + CLI probe | No live vos |

**M4. Architecture / bench stale versioning**
- `docs/benchmarks.md` Disk-to-GPU: “Exploring a direct path is a **1.1** candidate (see Roadmap) — not a 1.0 claim.” 1.1 is the version being shipped; GDS is roadmap **2.0**.
- `docs/architecture.md` BITPIX 8 → `int8` only; unsigned `uint8` / signed-byte XOR is described later. Easy to misread.
- Architecture C++ `_C` dump lists cache stubs (`configure_cache`, `get_cache_size`) that are documented no-ops after Option A — OK if labeled; easy to treat as live.

**M5. `compress=` short names vs CLI**
- C++ accepts `R`/`RICE`/`RICE_1`, `G`/`GZIP`/`GZIP_1`, `H`/`HCOMPRESS`/`HCOMPRESS_1`, `P`/`PLIO`/`PLIO_1` (`fits_bindings.cpp`).
- CLI `--algorithm` help omits `BZIP2_1` and `PLIO_1`; `docs/cli.md` lists `PLIO_1` but not `BZIP2_1`.
- Python `_resolve_compression_algorithm` passes strings through; unknown names fail in CFITSIO, not at the Python boundary.

**M6. Bench competitor fairness (harness is mostly honest; a few duplicates)**
- **Good:** `benchmarks/bench_timing.py` warmup + CUDA/MPS `synchronize` after every timed call; interleaved round-robin with fixed seed; medians; RSS sampler **outside** the timed window; `use_cache=False` for torchfits full reads; fitsio mmap rows skipped under `strict_mmap_fairness`; astropy CompImage excluded from compressed tensor peers (Python decompress vs CFITSIO).
- **Weak:** `bench_fits_io.py` `read_full` sets `torchfits` and `torchfits_specialized` to the **same** `torchfits.read(..., use_cache=False)` lambda — specialized column is a duplicate for that op (cutout/open-once paths are distinct elsewhere).
- **Weak:** `cfitsio_direct` `table_scan` is `fits_get_num_rows` only; `table_pred` is “column > 0 compact”. Not the same work as torchfits `scan` / SQL `where=`. Fine as a microbench if labeled; dangerous if ranked as the same `operation` in scorecards.
- **Warm cache:** OS page cache is shared after warmup (all libraries). Cold vs warm profiles exist; headline is lab/warm-ish. Documented as methodology, not a cheat, but “cold NFS” users will not see these numbers.

**M7. `docs/examples.md` / contributing invoke bare `python`**
- `python examples/test_examples.py` vs pixi-first AGENTS.md. `examples/test_examples.py` itself falls back to `pixi run python` when pixi is on PATH — OK for humans, wrong for the docs snippet.

**M8. KaTeX vendor subset**
- `katex.min.css` (v0.16.11) `@font-face`s Caligraphic, Fraktur, Size3/4, Bold, etc.
- Repo only ships five `.woff2` files (Main-Regular, Math-Italic, AMS-Regular, Size1, Size2). Typical subset; missing fonts fall back. Not a product bug; do not “review” minified KaTeX as if it were first-party.

**M9. `test_no_external_fits_backends.py` forbids top-level `numpy` (except `__init__.py`)**
- Intent: lazy numpy. Encodes import layout, not FITS behavior. Fine as a package-isolation pin; will fail on a legitimate top-level `import numpy` refactor.

### LOW

**L1.** `open_subset_reader`: `api-core-io.md` shows `reader(x1,y1,x2,y2)`; index/examples/python-workflows use `reader.read_subset(...)`. Both exist (`SubsetReader.__call__` delegates). Pick one in the hub examples.

**L2.** MPS published exhaustive CSV is **20260719**; CPU/CUDA published files include **20260807**. Host scorecard for 1.1.0 omits MPS. Either refresh MPS or say the 1.1.0 headline is Linux-only.

**L3.** `docs/index.md` “Zero-copy where dtypes allow” for Polars `from_arrow` — Arrow→Polars is often zero-copy; not a FITS decode claim. Easy to read as I/O zero-copy.

**L4.** `test_dlpack_roundtrip.py` uses `torchfits.cpp.echo_tensor` (deprecated). Tiny, CPU-only.

**L5.** `FitsTableIterableDataset` `where=` path is documented as slower (Arrow row conversion) while changelog celebrates memory-bounded `scan(where=)`. Consistent if you read both; ML page does not mention the slow path.

**L6.** `docs/contributing.md` “Do not document env vars absent from `src/`” — integrity tests enforce architecture tables; `TORCHFITS_EXAMPLE_FAST` is in `examples/` and listed as debug/bench. Fine.

**L7.** Gallery PNGs are generated artifacts (see assets ledger). `examples/cli/make_rgb_demo.py` defaults output to `docs/assets/gallery`.

---

## What is in good shape (do not churn)

- Version triplet aligned: `__version__` = `pyproject.toml` = `pixi.toml` = `torch_lanes.json` = `1.1.0`.
- `test_docs_integrity.py`: env tables ↔ `getenv`/`os.environ.get` in `src/`; Core I/O signature fences; no sky-domain ownership claims; nav paths exist.
- Malformed / truncated / checksum / security (SSRF, brackets, pipe) / concurrent same-file **read** / remote HTTP resume+lock.
- CLI: info, header, verify, diff, stats, cutout, convert (incl. RGB recipes), copy, arith, compress/decompress, transform, setkey, probe — exercised. Exit codes documented and largely matched.
- `api.md` Quick Paths match `__all__` (including `cpp` namespace, now deprecated).
- Bench timing infrastructure (warmup, device sync, interleaving, mmap fairness) is serious, not decorative.
- Examples runner (`examples/test_examples.py` + `tests/test_examples_runner.py`) with `TORCHFITS_EXAMPLE_FAST` for CI.

---

## File ledger

Counts: **88** entries under `tests/` (86 `*.py` + `transforms_reference.pyi` + `tests/cpp/test_bracket_detection.cpp`). **~673** `def test_*` + **56** `Test*` classes (~21k lines). Docs: **25** user `docs/*.md` plus **5** bench `summary.md` under assets.

### tests/

| File | n | Role / notes |
|---|---:|---|
| `conftest.py` | 0 | Autouse: clear `CACHE_ENV_SENTINELS` so HPC/cloud detection does not fire in CI |
| `transforms_reference.py` + `.pyi` | 0 | Shared numeric reference for transform tests, not collected |
| `tests/cpp/test_bracket_detection.cpp` | — | **Not in pytest/CI** (H8) |
| `test_api.py` | 2+class | Public I/O smoke; some `torchfits.cpp` raw paths |
| `test_arrow_table_api.py` | 70 | Largest table/Arrow/mutation/where matrix; `/tmp` files |
| `test_ascii_table.py` | 1 | ASCII TABLE HDU; **CWD** `test_ascii.fits` |
| `test_astropy_upstream_smoke.py` | 9 | Astropy parity (MEF, scaled, VLA, BIT, checksum, compress) |
| `test_bench_ranking_mmap.py` | 5 | Scorecard ranking / mmap skip rules (scripts) |
| `test_bench_suites.py` | 11 | `benchmarks/suites.py` registry |
| `test_bug_table_hdu_cache.py` | 2 | TableHDU schema cache vs header mutation |
| `test_bug_table_ref.py` | 4 | TableHDURef |
| `test_byteswap.py` | 8 | Endian image+table × mmap |
| `test_bz2.py` | 10 | Whole-file `.bz2` reads; write `.fits.bz2` rejected |
| `test_cache.py` | 2+5cls | Cache manager, threads, isolation copies |
| `test_cache_config.py` | 5 | HPC/cloud sentinels |
| `test_changelog_tooling.py` | 12 | `scripts/update_changelog.py` |
| `test_check_torch_extra_pins.py` | 18 | `[cpu]`/`[cuda]` extra pins vs indexes |
| `test_checksum.py` | 2 | write/verify DATASUM/CHECKSUM |
| `test_clear_all_caches.py` | 6 | disk + in-process |
| `test_cli.py` | 67 | CLI subprocess matrix; **no `table` functional test** |
| `test_cli_release_fixes.py` | 5 | arith saturate, stats uint, diff NaN |
| `test_complex_header.py` | 1 | HIERARCH/HISTORY; **CWD** |
| `test_compressed_nulls.py` | 3 | CompImage ZBLANK → NaN vs astropy |
| `test_compression.py` | 1+2cls | Includes **dead env knobs** (H5) |
| `test_compression_matrix.py` | 9 | RICE/GZIP/HCOMPRESS/PLIO/BZIP2 vs fpack |
| `test_concurrent_same_file_read.py` | 3 | Threaded reads, handle close |
| `test_cutout_performance_api.py` | 4 | subset reader / shape cache |
| `test_data.py` | 8cls | Dataset/loader; mkdtemp |
| `test_data_datasets.py` | 14 | cache roots, remote lock, cube/spectrum iterable, sharding |
| `test_deep_review_p0.py` | 4 | Mocked WHERE size skip (M1) |
| `test_deep_review_wave2.py` | 6 | HTTP missing, cache barrier, overflow header |
| `test_deep_review_wave4.py` | 11 | lazy header, PNG, stream_table |
| `test_deep_review_wave5.py` | 14 | LONGSTR, uint64 reject, GIL, NIOBUF |
| `test_dlpack_roundtrip.py` | 1 | `cpp.echo_tensor` pointer identity |
| `test_docs_code_snippets.py` | 3 | Exec fences; `chdir(tmp)` only |
| `test_docs_integrity.py` | 16 | Docs contract (env, signatures, nav, no WCS ownership) |
| `test_examples_runner.py` | 1 | Wraps `examples/test_examples.py` |
| `test_fits_schema.py` | 3 | TFORM/schema |
| `test_fitsio_upstream_smoke.py` | 11 | fitsio parity |
| `test_hdu.py` | 5 | HDUList/TensorHDU/TableHDU |
| `test_hdu_file_ops.py` | 15 | open/fromfile; 1 skipif |
| `test_hdu_str.py` | 1 | string HDU |
| `test_header_versioning.py` | 7 | LONGSTRN / CONTINUE |
| `test_http_probe_fixture.py` | 4 | probe SSRF / internal URL |
| `test_integration.py` | 2cls | Larger synthetic “real data”; **unmarked** |
| `test_interop.py` | 9 | pandas/arrow/polars/astropy |
| `test_interop_import.py` | 2 | missing optional deps |
| `test_io.py` | 1cls | **Mocked cpp** (H5) |
| `test_malformed_fits.py` | 10 | Intent: bad magic, trunc, GROUPS, fuzz |
| `test_mps.py` | 12 | skipif no MPS |
| `test_multichunk_buffered_read.py` | 2 | >16 MiB table vs astropy |
| `test_no_external_fits_backends.py` | 1 | AST import ban astropy/fitsio/numpy/torchsky |
| `test_open_table_reader.py` | 1 | handle vs cold read_torch |
| `test_output_parity.py` | 13 | Image/table/compress/fuzz vs astropy |
| `test_package_isolation.py` | 9 | import torchfits without torch; ABI cmake pins |
| `test_patch_bench_docs.py` | 4 | host labels; **/tmp/pdoc_test** |
| `test_performance.py` | 2cls | Wall-clock; **unmarked** (H7) |
| `test_public_boundary.py` | 8 | `__all__`, cpp deprecation, table destinations |
| `test_public_where.py` | 3 | `torchfits.where` public helpers |
| `test_quantize_int16.py` | 14 | robust pack image+table |
| `test_read_header.py` | 1 | **CWD** |
| `test_read_policy.py` | 5 | WHERE backend plan |
| `test_release_lane.py` | 7 | `torch_lanes.json` vs files |
| `test_release_semantics.py` | 14 | cache copy isolation, threads |
| `test_release_smoke.py` | 3 | version + open/write |
| `test_remote_http_range.py` | 12 | Range GET, auth, resume, mp lock |
| `test_rgb.py` | 14 | `rgb` / lupton / PNG |
| `test_scale_on_device.py` | 2 | BSCALE on CUDA/CPU |
| `test_security.py` | 9 | 10-axis, `[dir]`, SSRF on cpp |
| `test_security_eval.py` | 1 | TableHDU.filter no `eval` |
| `test_security_fix.py` | 2 | path guards |
| `test_skinny_meta.py` | 4 | read_keys/nrows/shape/extname |
| `test_staged_prefetch.py` | 5 | FitsStagedCutoutIterableDataset |
| `test_subset_3d.py` | 3 | cube trailing-window cutouts |
| `test_table.py` | 3+2cls | table read/write core |
| `test_table_docs_smoke.py` | 1 | docstring examples |
| `test_table_file_ops.py` | 21 | mmap update, mutations |
| `test_table_filtering.py` | 13 | where= + some cpp filtered |
| `test_transforms.py` | 29cls | Math vs `transforms_reference` |
| `test_transforms_e2e.py` | 3cls | Dataset+transform |
| `test_transforms_typing.py` | 21 | Compose mask, types |
| `test_truncated_table_errors.py` | 4 | SIGBUS-class trunc |
| `test_upstream_parity_inventory.py` | 1 | replay manifest exists |
| `test_validation.py` | 1 | HDUList.validate; **CWD** |
| `test_where.py` | 8 | parser/eval |
| `test_write_fidelity.py` | 27 | dtypes, compress, cutout write |
| `test_writing.py` | 27 | HDUList write; **CWD** + 2 skipif GPU |

### docs/*.md (user markdown; not assets)

| File | Cross-check |
|---|---|
| `api.md` | Quick Paths ≈ `__all__`; mmap table OK; `cpp` still listed as compatibility surface (deprecated in code) |
| `api-core-io.md` | Signatures fenced by `test_docs_integrity`; `read()` mmap `"auto"` vs `read_tensor` mmap `True` is real |
| `api-tables.md` | `read_torch` simple where= vs full dialect; ignored cache kwargs documented |
| `api-data.md` | Cutout/spectrum returns match source; **conflicts with examples-ml.md** (H3) |
| `api-transforms.md` | Math documented; **missing fill="nan"** (H4); rgb/lupton match |
| `cli.md` | 15 subcommands, exit 0–4, `-j` vs `-J`; compress algos omit BZIP2_1 |
| `cli-recipes.md` | Copy-paste shell; uses `/tmp`; sample fetch script |
| `install.md` | 3.10–3.14, torch 2.13, Linux+macOS arm64 wheels; **Windows source** vs compatibility (H10); GPU-wheel wording (H1) |
| `compatibility.md` | Matches cibuildwheel; no Windows |
| `release.md` | Lane 2.13 → 1.1.0; freeze text still 1.0.0 (H14); replay checklist (H12) |
| `changelog.md` | 1.1.0 dated 2026-08-26; Unreleased leftover (H13); bench IDs 20260822 (B1) |
| `parity.md` | No bz2/BZIP2/Random Groups (H11) |
| `architecture.md` | Canonical env tables; CFITSIO 4.7.0 pin; Option A cache story |
| `benchmarks.md` | Methodology good; **headline CSV missing** (B1); 2.12 champion (H2); 1.1 GDS leftover (M4) |
| `contributing.md` | Pixi-first; env-var rule; layout includes replay gates |
| `index.md` | “Native GPU decode”, 15-command CLI, zero-copy Polars (H1, L3) |
| `quickstart.md` | Matches public API; GPU wording (H1) |
| `python-workflows.md` | `compress="RICE_1"` good; “Differentiable transforms” (H3) |
| `examples.md` | Paths exist; `python examples/test_examples.py`; `reader.read_subset` |
| `examples-ml.md` | **Unsourced cutout table** (B2); dataset shapes (H3) |
| `examples-transforms.md` | Gallery; matches transforms API |
| `denoise-pipeline.md` | Points at `example_megacam_cr_denoise.py`; FAST env |
| `migration_astropy.md` / `migration_fitsio.md` | Side-by-side; GPU “direct” same overclaim as hub |
| `roadmap.md` | 1.0 shipped, 1.1 “beta soak” wording while changelog is 1.1.0; GDS in 2.0 |

### examples/*.py

Smoke: `examples/test_examples.py` discovers `*.py` + `cli/*.py`, skips `_*.py`, `test_examples.py`, `desi_shaped_spectrum.py`, `cli/make_rgb_demo.py`. Optional skip: `example_polars.py`. CI sets `TORCHFITS_EXAMPLE_FAST=1`. Pytest wrapper: `tests/test_examples_runner.py`.

| File | Notes |
|---|---|
| `_sample_data.py` | Sample cache; FAST skip |
| `_plotting.py` | Gallery helper |
| `test_examples.py` | Runner (cwd=`"."`) |
| `example_image.py` | tempfile FITS |
| `example_image_cutouts.py` | tempfile; `reader(...)` not `.read_subset` |
| `example_image_cube.py` | cube |
| `example_image_mef.py` | MEF |
| `example_image_dataset.py` | Datasets |
| `example_data_catalogs.py` | catalogs |
| `example_table.py` | Chandra-style table |
| `example_table_interop.py` | interop |
| `example_table_recipes.py` | DuckDB/Polars; FAST |
| `example_polars.py` | optional |
| `example_transforms.py` | stretches |
| `example_custom_transform.py` | FITSTransform subclass |
| `example_quantize_int16.py` | robust pack |
| `example_cutout_wcs_write.py` | CRPIX shift (no WCS engine) |
| `example_mef_header.py` | headers |
| `example_time_series.py` | tables |
| `example_streaming_cubes_spectra.py` | scan |
| `example_make_loader_vs_dataloader.py` | loaders |
| `example_staged_cutouts.py` | staged iterable |
| `example_m13_stack.py` | stack |
| `example_manga_logcube.py` | needs sample |
| `example_megacam_mef_cutouts.py` | needs MegaCam |
| `example_megapipe_cutout_collage.py` | FAST skip; gallery PNG |
| `example_megacam_cr_denoise.py` | FAST one-epoch |
| `example_ml_galaxyzoo_legacy.py` | FAST; GZ_N=8 in runner |
| `example_lupton_rgb_sdss.py` | lupton |
| `example_rgb_sky.py` | `transforms.rgb`; FAST |
| `gallery_images.py` / `gallery_tables_lc.py` | figure generators |
| `desi_shaped_spectrum.py` | **excluded** from runner (download) |
| `cli/make_rgb_demo.py` | **excluded**; writes `docs/assets/gallery` |
| `cli/imstat_imarith.sh` | shell recipe, not in py runner |

No `os.chdir` in examples. Network gated by FAST / sample cache.

### benchmarks/

| File | Warmup / sync / cache / fairness |
|---|---|
| `bench_timing.py` | Canonical timer: warmup, CUDA/MPS sync, interleaved seed `20260101`, RSS off the clock |
| `bench_all.py` | Orchestrator; `--warmup`; quick profile warmup=0 |
| `bench_fits_io.py` | user warmup=1 / lab=2; `use_cache=False`; mmap fairness; **specialized==default** on `read_full` |
| `bench_fitstable_io.py` | warmup default 1; interleaved |
| `bench_gpu_transports.py` | device sync; peer groups; skip failed warmup |
| `bench_arrow_tables.py` | `_time` warmup+iters |
| `bench_cpp_backend.py` | warmup default 3 |
| `bench_cache.py` | cache behavior |
| `bench_http_stream.py` | warmup by profile |
| `bench_megacam_cutouts.py` | warmup by profile |
| `bench_ml_loader.py` | CUDA sync |
| `bench_denoise.py` / `bench_science_pipeline.py` / `bench_median_stack.py` / `bench_fits_write.py` | profile warmup |
| `bench_gpu_memory.py` | CUDA sync around alloc samples |
| `bench_contract.py` / `suites.py` / `config.py` / `bench_fixtures.py` / `mpl_config.py` | harness, not timings |
| `run_cfitsio_direct_bench.py` | `--warmup` default 1; builds C binary |
| `cfitsio_direct/bench_cfitsio_direct.c` | CLOCK_MONOTONIC, median; ops mapped in header comment; `table_scan`/`table_pred` ≠ full torchfits ops (M6) |
| `cfitsio_direct/CMakeLists.txt` | vendored libcfitsio |
| `replays/upstream_sources.json` | fitsio + astropy.io.fits → smoke tests; **`replays: []`** |
| `__init__.py` | package marker |

---

## docs/assets — generated / vendor vs must-review

### Exclude from product review (binaries / vendor / generated figures)

Do not treat these as API/contract sources. Do not “audit” minified JS.

**KaTeX vendor (zensical math):**
- `docs/assets/katex/katex.min.css` (KaTeX 0.16.11)
- `docs/assets/katex/katex.min.js`
- `docs/assets/katex/contrib/auto-render.min.js`
- `docs/assets/katex/fonts/KaTeX_{Main-Regular,Math-Italic,AMS-Regular,Size1-Regular,Size2-Regular}.woff2`
- CSS references additional fonts (Caligraphic, Fraktur, Size3/4, Bold, …) **not** in tree — browser fallback.

**Gallery PNGs** (example/gallery scripts; regenerate, do not hand-edit):
- `docs/assets/gallery/cli_rgb_demo.png`
- `docs/assets/gallery/image_compose_pipeline.png`
- `docs/assets/gallery/image_cutout.png`
- `docs/assets/gallery/lightcurve_asymmetric_sigma_clip.png`
- `docs/assets/gallery/lightcurve_sigma_clip.png`
- `docs/assets/gallery/lupton_rgb_sdss.png`
- `docs/assets/gallery/megapipe_cutout_collage.png`
- `docs/assets/gallery/ml_gz_class_grid.png`
- `docs/assets/gallery/rgb_sky_collage.png`
- `docs/assets/gallery/rgb_vs_lupton_dwarf.png`
- `docs/assets/gallery/table_fits_scale_columns.png`

**Bench `summary.md`** (generated from CSV; review only if numbers disagree with CSV):
- `docs/assets/bench/exhaustive_{cpu,cuda,mps}_*/summary.md` (5 files)

### Must-review (numbers that back public claims)

| Path | Used by | Status vs 1.1.0 headline |
|---|---|---|
| `docs/assets/bench/exhaustive_cpu_20260807_013736/{results,torchfits_deficits}.csv` | Published CSV links | **Older than headline IDs**; CPU summary: 3 significant deficits, image HCOMPRESS noise lags |
| `docs/assets/bench/exhaustive_cuda_20260807_013736/{results,torchfits_deficits}.csv` | Published CSV links | Older than `exhaustive_cuda_20260822_*` |
| `docs/assets/bench/exhaustive_cpu_20260719_144337/*` | Published CSV links | Round-3 soak |
| `docs/assets/bench/exhaustive_cuda_20260719_144457/*` | Published CSV links | Round-3 soak |
| `docs/assets/bench/exhaustive_mps_20260719_143706/{results,torchfits_deficits,ml_results,megacam_results}.csv` | Published + companion | MPS not in 1.1.0 host table |
| `docs/assets/bench/ml_20260719_145743/ml_results.csv` | ML suite | Dated; not wired to examples-ml cutout table |
| `docs/assets/bench/20260719_075555/megacam_results.csv` | MegaCam 40-cutout Rice | **Does not support** examples-ml 1000-stamp / 0.060 s table |

**Missing must-review artifacts (cited, not in git):**
- `docs/assets/bench/exhaustive_cpu_20260822_213823/` (results + deficits + summary)
- `docs/assets/bench/exhaustive_cuda_20260822_213846/`

`benchmarks_results/` is gitignored — local-only.

---

## Explicit binary exclusion list

Out of scope for this audit’s “read every file” bar (contents not semantically reviewed):

1. All `docs/assets/katex/**` (min.js, min.css, woff2)
2. All `docs/assets/gallery/*.png`
3. Any `*.woff2` / `*.woff` / `*.ttf` under docs
4. Compiled artifacts if present: `*.so`, wheels, `extern/cfitsio/**` (gitignored vendored tree)
5. Local `benchmarks_results/**` (not in git)

CSVs and `summary.md` under `docs/assets/bench/` **are** in scope (numbers).

---

## Mapping: src façade → tests (coverage at a glance)

| Public surface | Primary tests | Hole |
|---|---|---|
| `read` / `read_tensor` / `write` | `test_output_parity`, `test_write_fidelity`, `test_api`, `test_byteswap` | `test_io` mocks only |
| `read_subset` / `open_subset_reader` | `test_subset_3d`, `test_cutout_performance_api`, CLI cutout | HTTP Range cutout vs `test_remote_http_range` |
| `table.read` / `scan` / `where=` | `test_arrow_table_api`, `test_table_filtering`, `test_where` | CLI `table` |
| `open` / HDU types | `test_hdu`, `test_hdu_file_ops`, `test_table_file_ops` | — |
| checksums | `test_checksum`, astropy smoke | — |
| cache | `test_cache`, `test_clear_all_caches`, `test_cache_config` | — |
| data.Datasets / `make_loader` | `test_data`, `test_data_datasets`, `test_staged_prefetch` | Cutout labels (none) vs ML docs |
| transforms | `test_transforms*`, `test_rgb` | `fill="nan"` docs |
| CLI | `test_cli`, `test_cli_release_fixes` | `table` subcommand |
| security | `test_security*` | C++ bracket unit not gated |
| docs contract | `test_docs_integrity`, `test_docs_code_snippets` | Does not check examples-ml numbers vs CSV |
| install/wheels | `test_package_isolation`, `test_release_lane`, `test_check_torch_extra_pins` | — |

---

## Suggested tag checklist (this slice only)

1. Commit `exhaustive_{cpu,cuda}_20260822_*` CSVs (or retarget headline to 20260807 and quote those win rates).
2. Remove or source the examples-ml 1000-cutout timing table.
3. Soften “native GPU decode” / “wheels enable GPU” copy.
4. Move CWD FITS tests to `tmp_path`; delete unused pytest markers or apply them.
5. Either restore `TORCHFITS_COMPRESSED_PARALLEL` in C++ or delete the no-op tests.
6. Add one CLI `table` smoke; optionally compile `tests/cpp/test_bracket_detection.cpp` in preflight.
7. Align `examples-ml.md` dataset returns and `pin_memory` with `api-data.md`.
8. Document `fill="nan"`; add bz2/BZIP2 (and Random Groups fail-loud) to `parity.md`.
9. Decide Unreleased vs 1.1.0 stamp before `git tag`.
