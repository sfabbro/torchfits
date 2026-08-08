# Changelog

All notable changes to torchfits are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-08-07

Version cut on the 2.13 torch ABI lane: buffered table reads through a
thread-local reader cache (process-wide eviction, stat-identity stale guard),
single-open insert/update of table rows, and torch-mask predicate
materialization. **Not yet released** — awaits collaborator docs/usability
feedback.

### Performance

- Cached table reader: `read_table`/`read_batch` reuse an open CFITSIO handle
  per thread instead of open/read/close per call; benchmark-host side effects
  drop from ~5861 to ~750 minor faults per read after warmup; isolated
  `narrow_1000000` `read_full` window 17.8 ms -> 6.7 ms (lab window,
  `exhaustive_cpu_20260807_082144_reader_cache`).
- `exhaustive_cpu_20260807_082931_reader_cache` scorecard: fitstable
  `read_full` ratio vs astropy 1.365x -> 1.072x; `predicate_filter` on
  `narrow_1000000` 16.60 -> 11.01 ms (1.21x -> 1.64x vs astropy); selective
  10.68 -> 9.33 ms (1.50x -> 1.71x) via torch-gather `where=` (mmap off, lab).
- Insert/update rows open the table once per operation (`6d2338c`).

### Fixed

- Read-then-mutate regression: appending rows after a cached read no longer
  hits CFITSIO error 104 (reader cache eviction is now process-wide, covering
  cross-thread writers).
- Stale reads: replaced files (new inode) are re-detected via stat identity;
  cached pread handles are dropped, never serving old data.
- Schema metadata with nanobind >= 2.14 (dropped str->int coercion):
  `tnull`/`bscale`/`bzero`/null values parsed from header strings instead of
  raising `std::bad_cast` on the write path.

### Docs

- Scorecards refreshed from the 2026-08-07 exhaustive runs (CPU + CUDA);
  `docs/assets/bench/` mirrors the surviving 2026-08-07 CPU/CUDA run CSVs.

## [1.0.0rc5] — 2026-08-06

Fifth release candidate on the 2.13 torch ABI lane; adds prerelease-aware release
tooling, a root cache reset entry point, and the macOS compressed-float parity fix.

### Added
- `clear_all_caches()` — root-level clear of in-process *and* disk caches
  (including `cache_root()` downloads/samples); `clear_cache()` stays
  in-process-only by default.
- CI/scripts: `release_lane.py --prerelease rc<N> --apply` renders a lane's
  release version plus a PEP 440 prerelease suffix (e.g. `1.0.0rc5` on the
  2.13 lane) across all five pinned files; `--check` / `check-lane` accept rc
  states as the lane base; unit tests cover suffix render/check/reject paths.
- Opt-in robust float→int16 packing: `write` / `write_tensor(..., quantize=)` and
  `table.write(..., quantize=)` (`"robust"` or `{"lo_q","hi_q","keep_zero"}`).
  Default remains native float (`BITPIX=-32` / float `TFORM`).
- Example: `examples/example_quantize_int16.py`.
- CLI `compress` / `decompress`: multiple inputs via `--out-dir`; `--split
  file|hdu` (one output per file or per image HDU); `-j/--jobs` = PyTorch
  intra-op threads; `-J/--file-jobs` = multi-file thread pool.
- CLI `-J/--file-jobs` on `verify` / `stats` / `arith` (and compress).
- CLI `arith`: image–image operand, multi-HDU stack+ATen, multi-file `--out-dir`.
- CLI Wave 2: batch `copy` / `transform` / `cutout` via `--out-dir` + `-J`;
  `stats` `std`/`median`; `compress --algorithm`; `header -k` wildcards;
  `setkey --delete` / `@list` via CFITSIO `fits_delete_key` (keeps compression).
- Docs: CPU-only (no CUDA libs) install recipe; “not only for ML” blurb;
  roadmap **2.0** native engine / GPU-direct (drop CFITSIO).
- Packaging: `torchfits[cpu]` / `torchfits[cuda]` extras (both **Linux-only**
  — macOS no-ops, MPS ships in the default wheel) plus one-line CUDA/CPU
  install recipes (PyPI's default torch already bundles CUDA on Linux
  x86_64; CUDA builds also run on GPU-less machines via CPU fallback).
- Docs: full docs↔code sync audit — one-line pinned installs in quickstart /
  CLI docs, `write()` payload types corrected (no top-level ndarray),
  CLI-recipe transform kwargs documented, architecture freshness rc5.
- CI/scripts: `check-torch-pins` resolves the `[cpu]` / `[cuda]` extra pins
  against the PyTorch indexes on the wheel ABI lane, run as the first step of
  both CI jobs and the wheel-build workflow so lane drift fails fast (CI lint
  job, build_wheels tests + wheel jobs, `ci-local`). macOS passes vacuously
  (both flavor pins are Linux-only); the doc-drift guard now also requires
  every extra's exact pin string (e.g. `torch==2.10.0+cpu`) to appear in
  install.md / README; unit tests cover the lane guard, the marker-skip
  path, the missing-extras failure, and the exact-pin doc-drift check.
- Lossless compressed float writes: GZIP_1 and integer RICE_1 no longer
  silently quantize (`fits_set_quantize_level(0)`); float RICE_1 /
  HCOMPRESS_1 keep CFITSIO default quantization (lossless unsupported),
  matching astropy/fitsio defaults. Documented in `io.write()`.
- Compression matrix suite (48 tests): all algorithms × int dtypes with
  astropy oracles, PLIO range rejection, float quantization bound,
  per-algorithm cutouts, fpack byte-identity.
- Output-parity suite (`tests/test_output_parity.py`, 69 tests): bitwise
  cross-library read parity (torchfits == fitsio == astropy), write
  round-trips through fitsio for every compression type + quantize +
  LONGSTRN + uint64 rejection, seeded fuzz-lite sweep.
- Bit-faithful write/read fidelity suite (55 tests) with astropy as an
  independent oracle.
- `BZIP2_1` image codec (vendored CFITSIO patch, opt-in via
  `TORCHFITS_USE_BZIP2`, default ON when libbz2 is available; not part of
  the FITS standard — astropy refuses it).
- CANFAR matrix bench mode: any python × torch lane × device grid from a VOS
  wheel bundle (`scripts/launch_canfar_matrix_grid.sh`), 41-leg grid soak.

### Changed
- `open_subset_reader` mmap path covers unsigned FITS conventions (BZERO/BSCALE).
- Warm `read_shape` hits shared image-info cache; `read_header` caches cards LRU.
- Landing one-liner + transparent nav/favicon logos; contributing / release
  checklists aligned with verify tiers (`preflight-push` / `ci-local` /
  `release-gate`).
- `torchfits header` text mode dumps **all HDUs** in fitsheader-style blocks.
- CLI parallelism docs: `-j` (torch) vs `-J` (file workers).
- `setkey --rename` / `--delete` remove keywords via CFITSIO delete (no
  decompressing rewrite); `--split hdu` rejects colliding stems.
- `table.read_torch(..., where=)` applies a torch mask after reading projected
  columns; dialect is simple compare / `BETWEEN` / `AND` only (full dialect on
  `table.read`).
- `write()` no longer rejects numpy arrays: torchfits.write(numpy_array)
  writes image HDUs (plain, quantized, compressed) instead of raising
  TypeError. Fixed the public-API bug where docs documented broken behavior.
- LONGSTRN/CONTINUE read+write: header values > 68 chars assemble via
  CONTINUE chains (`&` + bare CONTINUE) on read and
  `fits_update_key_longstr` on write.
- uint64 image/table writes raise `ValueError` with guidance (was an
  unsupported-column error); the rejection covers numpy table columns too.
- TSCAL/TZERO scaling applied to FLOAT/DOUBLE table columns; scaled tables
  fall back to buffered (non-mmap) reads with physical values.
- int8 images use the BZERO=-128 signed-byte convention instead of raw bytes
  without BZERO (values ≥ 128 corrupted on read); unsigned table prep
  registers untouched columns in the synthesized schema.
- ASCII string columns use code-first `Aw` tforms; `fits_schema.parse_tform`
  understands the ASCII `Aw` form so `update_rows` stops truncating ASCII
  strings.
- LOGICAL decode accepts `'T'` / `'1'` / `1` (CFITSIO returns converted 1/0
  on `fits_read_col(TBYTE)`); uint64 images (BZERO=2^63) detected as scaled
  instead of read raw.
- `Header.remove` fast path for huge HISTORY lists; 20k-delete regression
  smoke.
- TableHDU caches version-gated on the header (were stale after TTYPE/TFORM
  mutation).
- HCOMPRESS uses 2D 16-row tiles like fpack's default (1D tiles rejected by
  CFITSIO with status 413).
- Concurrency: Python FITS-cache LRUs and the thread-local metadata cache are
  lock-protected / size-bounded for concurrent reads; uint16 BZERO offset is
  fused into the SIMD bswap mmap path.
- Bench docs: multi-host GPU/CPU scorecard from the CANFAR matrix grid.
- Bench docs: rc5 re-soak snapshot — CANFAR CPU
  `exhaustive_cpu_20260806_012620`, CUDA `exhaustive_cuda_20260806_012651`,
  local CPU `exhaustive_cpu_20260806_022603`. CUDA 100% fits win rate
  (smart/specialized), fitstable ≥98.9%; only residual lags are
  table `read_full` / predicate rows (≤1.15×, fitsio/astropy) and
  HCOMPRESS_1 (≤1.03×).
- Bench labeling: host scorecard derives the platform from the benchmark
  data (`metadata` device field / `host` column token), not the run-id tag —
  the local bench script names every run `exhaustive_mps_*` regardless of
  platform, so a CPU run on a Linux box used to be mislabeled "macOS arm64 /
  MPS". The script now tags runs `exhaustive_cpu_*` on non-Darwin hosts.
- SSRF hardening waves: private/loopback/link-local/reserved-address guards
  on read_header, read_batch, HDU write paths, and public cpp; `scan_polars`
  guards before importing optional polars.

### Fixed
- macOS compressed-float parity: the vendored CFITSIO now builds with
  `-ffp-contract=off` (clang/aarch64 FMA contraction shifted low-ULP bits of
  decompressed float tiles, e.g. exact `0.0` read back as ~1.9e-16); the
  compressed-image parity suite additionally allows ≤1 dtype eps vs
  fitsio/astropy on macOS, staying bitwise-exact everywhere else.
- Table int16 columns with `TSCAL`/`TZERO`: disable CFITSIO auto-scale on read
  before casting, then apply scale in memory (avoids int16 overflow).
- Silent data loss / corruption fixes: compressed float writes silently
  quantized by CFITSIO defaults (documented behavior change above); int8
  images without BZERO corrupted values ≥ 128; table column ordering scrambled
  on rewrite (unordered_map → ordered vectors); ASCII string columns truncated
  to one character on `update_rows`.
- PLIO heap overflow in vendored CFITSIO (`imcomp_calc_max_elem` sizing, ASAN-
  confirmed 16-byte overwrite on incompressible data), fixed by auto-applied
  patch; bzip2 image codec re-enabled upstream.
- LOGICAL (T/F) decode accepts all of `'T'`/`'1'`/`1`.
- `scan_polars` guarded before importing optional polars.
- CI lint packaging dep fix; astropy < 6.0 uint32 tile-compression test
  failures fixed; py3.10 uint32 astropy oracle skipped below 7.0.
- `check-torch-pins` misleading pass on rc states: `main()` re-initialized the
  `failed` flag after the lane-consistency loop, so a genuine `[FAIL]` line
  never gated CI (exit 0), and `lane_for_version` rejected the `rc<N>`
  prerelease suffix — the gate re-fired a spurious FAIL on every rc cut.
  Suffix is now accepted as the lane base and lane-consistency failures
  propagate to the exit code (regression tests added).

## [1.0.0rc4] — 2026-07-20

Fourth release candidate for collaborator soak after prep / deep-review cleanup.

### Removed
- Root aliases `read_table`, `stream_table`, `read_table_rows`, `get_header`,
  `get_batch_info` — use `table.read_torch` / `table.scan_torch` / `read_header` /
  `read_batch_info`.
- Spectral and continuum transforms (`spectral.py`, `continuum.py`) — torchfits
  keeps FITS I/O–adjacent viz/ML preprocess only; spectroscopy analysis moves
  out of this package.
- Dead `core.py` / `ChecksumVerifier` — checksums go through `_C` via
  `checksum_api`.
- CLI deprecated aliases `--fitsort`, `--bytes`, `--preview` — use
  `--keyword-table`, `--header-bytes`, `-n`/`--rows`.
- Deprecated `table_module=` dual-path on cache invalidate/clear.
- `table.to_polars_lazy` — use `scan_polars` or `to_polars(...).lazy()`.
- `torchfits.cli.rgb` shim — import `lupton_rgb` / `write_rgb_image` from
  `torchfits.transforms`.
- No-op `handle_cache=` on `read_tensor` and `handle_cache_capacity=` on
  `read_subset` (persistent reuse stays on `open_subset_reader`).

### Added
- Skinny metadata: `read_nrows`, `read_keys`, `read_shape`, `read_hdu_type`,
  `read_num_hdus`, `read_colnames`, `read_extname`, `read_table_info` —
  CFITSIO structural/key queries without a full header dump.
- `open_table_reader(path, hdu=1)` — reusable table handle (mirror of
  `open_subset_reader`).
- `table.read_torch(..., where=)` — fused C++ project+predicate path.
- `FITSHeaderScale.from_path` / `FITSHeaderNormalize.from_path` via skinny keys.
- `transforms.as_module` / `AsModule` — thin `nn.Module` adapter for
  `nn.Sequential`.
- CLI `transform --name Class:key=val,...` constructor kwargs.
- HTTP Range cutouts + vos/vault remote fetch (prior unreleased work).
- Public `TensorHDU.shape_str` / `dtype_str`; optional `FitsTableDataset(labels=)`.

### Changed
- `read()` rejects unknown kwargs with `TypeError` (no silent swallow of
  leftovers like `policy=`).
- Table `read_torch` uses a thin C++ path (skips `read_unified` image probes).
- Datasets `label_key`, `get_image_meta`, benches/examples lean on skinny meta.
- Lazy root `__getattr__` uses a lock-backed attribute cache (no `globals()`
  mutation).
- Library logger uses `NullHandler` (no import-time StreamHandler).
- Bench deficit CSV always lists raw lags; `significance` is `noise` or
  `significant` (floors label only).
- Removed disconnected `benchmarks/bench_fast.py` and pixi aliases
  `bench-fast` / `bench-fast-stable` / `bench-core` (use `bench-fits`).
- Dead private `read_large_table` leftover and unused `_unsigned.py`
  (unsigned paths live in `_read_pipeline` / `write_api` / `fits_schema`).
- Table mutation: single `_mutation_cache_barrier` pre/post; dtype maps via
  `_ensure_dtype_maps()`.
- `FitsTableDataset.__getitem__` returns `(row_dict, label)` for
  `make_loader` / `fits_collate_fn` parity with image datasets.
- Lupton examples/docs use `lupton_rgb(r=..., g=..., b=...)` (reddest → R).
- Example smoke runner auto-discovers `examples/*.py` (skips `_*.py` helpers).
- DataView BITPIX 64 → `torch.int64`; C++ `write_image` supports int8/int64.

### Fixed
- HTTP Range cutouts: NumPy view `byteswap(True)` on frombuffer tensor (this
  torch build has no `Tensor.byteswap`); drop redundant cutout `.clone()`.
- SigmaClip: 0-d `new_zeros(())` fill for `torch.where(..., out=)` (no
  `zeros_like` buffer).
- Filtered table zero-match `where=`: keyed empty tensors (not `{}`).
- **`lupton_rgb`:** Astropy-parity Lupton asinh mapping (per-pixel peak clip).
  Gallery SDSS / MegaPipe figures regenerated with readable stretch.
- **`SubsetReader`:** uncompressed 2D images mmap the data segment once and
  slice+bswap into torch (MegaPipe-class mosaics); CFITSIO `fits_read_subset`
  remains the fallback for compressed / scaled / non-2D.
- Clearer `TypeError` when inferring FITS TFORM from uint16/uint32/uint64.
- `LogStretch.inverse` clamps exponents to avoid float overflow.
- Remote prefetch bookkeeping cleaned after completed downloads (locks retained).
- WHERE `BETWEEN` boundaries no longer use bare `\S+` (operators/parens).
- `fast_parse_header_cards` / `Header` keep empty comments as `""` (not `"None"`).
- Wheel builds pin torch 2.10 ABI (`--no-build-isolation` + before-build install)
  so release smoke no longer fails against a 2.13-built extension.
- Deep-review P0–P4 harden (WHERE OOM gate, batch exception narrowing, HDU close
  race, prefetch errors, mutation cache barrier, NAXIS overflow guard).

### Docs
- Removed-names table lists root `read_table` / `stream_table` /
  `read_table_rows` / `get_header` / `get_batch_info`.
- Core I/O cache section documents root vs `torchfits.cache` layers.
- Examples: `open_table_reader` + EXTNAME `table.read_torch`.
- Round-3 scorecard (post thin-I/O): MPS `exhaustive_mps_20260719_143706`,
  CANFAR CPU `exhaustive_cpu_20260719_144337`, CUDA
  `exhaustive_cuda_20260719_144457`; MegaCam `20260719_075555`; ML
  `ml_20260719_145743`.
- Slim transform gallery; real Lupton RGB figure; removed spectral/continuum docs.
- Core I/O docs point at `table.read_torch` / `table.scan_torch` (root aliases gone).
- User Guide [ML with FITS](examples-ml.md): Galaxy Zoo 1 + Legacy Survey one-epoch
  CNN train; MegaPipe mosaic collage + cutout timing.
- Canonical `TORCHFITS_*` env tables in [architecture](architecture.md); slimmed
  duplicates elsewhere.
- `release-gate` runs `docs-contract` (example sync + zensical build) and
  `docs-links` (internal hyperlink crawl of `site/`).

### Examples
- `example_ml_galaxyzoo_legacy.py`, `example_megapipe_cutout_collage.py`,
  `scripts/fetch_cfht_megapipe_sample.sh`.

## [1.0.0rc3] — 2026-07-18

Third release candidate for collaborator soak.

### Docs
- Persona pass on user-facing pages: rc honesty, corrected migration threading
  (private CFITSIO handles since rc2), API notes for EXTNAME / 3D `read_subset`,
  cache vs disk-cache / `make_loader` layering.
- Examples gallery: MaNGA LOGCUBE (`example_manga_logcube.py`), Lupton RGB from
  real SDSS g/r/i (`example_lupton_rgb_sdss.py`, stdlib `bz2` inflate before
  read), CFHT MegaCam MEF cutouts (`example_megacam_mef_cutouts.py`).
- Fill API / cache / loader doc gaps: `TORCHFITS_CACHE_DIR` vs in-process handle
  cache, when `optimize_cache` no-ops on table datasets, `make_loader` vs plain
  `DataLoader`.

### Agent / Jules
- `AGENTS.md` + `JULES.md`: weekly Jules retarget to bug/perf-only deep passes;
  ledger in `.cursor/jules-ledger.md`; out-of-scope cosmetic PRs.

### Fixed
- **String HDU / EXTNAME:** `read_tensor` and `read_subset` accept `hdu="EXTNAME"`
  (e.g. `hdu="MYDATA"`). `hdu="auto"` still raises a clear `ValueError`.
- **3D subset:** `read_subset` / `open_subset_reader` preserve the leading cube
  axis; window applies to trailing `(y, x)` only.
- **Zero-size cutout box:** a degenerate width or height keeps the other axis
  length (no longer collapses both dims to 0).
- **Table `where=` + TNULL:** filtered reads honor `apply_fits_nulls=True` so
  sentinel nulls do not leak as real values.
- **Remote prefetch race:** `resolve_local_path` waits on an in-flight prefetch
  for the same URL instead of racing a second download onto the same `.partial`.
- **Lupton RGB:** zero-size bands raise a clear error instead of a cryptic
  `RuntimeError`.
- **`.fits.bz2`:** clear `ValueError` when CFITSIO cannot read bzip2-compressed
  paths (decompress first — see Lupton example).

### Docs site
- GitHub Pages deploys **stable** (`/`, latest `v*` tag) and **edge**
  (`/edge/`, tip of `main`) from `docs.yml` — use edge to debug docs without a
  SemVer release. Docs “stable” may be an rc tag; PyPI non-prerelease can lag.

## [1.0.0rc2] — 2026-07-18

Second release candidate on the 1.0 line. CFITSIO concurrent-read correctness,
leftover API/docs/CLI, and audit cleanup. SemVer `1.0.0` still waits for soak.

### Install / compatibility
- Runtime / build metadata: `torch>=2.10` (wheels and pixi stay on the 2.10 ABI
  lane). Source builds embed the detected torch major.minor as the ABI tag.
- Docs: wheels vs source, unified GPU/accelerator install, `configure_for_environment`
  called once at import. Dropped `ipykernel` from `[dev]`.
- Disk cache root: `TORCHFITS_CACHE_DIR` (default XDG / `~/.cache/torchfits`);
  remotes and samples as subdirs; Dataset / `make_loader` honor `cache_dir=`.

### CLI
- Short options: `-e`/`--hdu`, `-f`/`--format`, `-o`/`--out`, `-w`/`--where`,
  `-c`/`--columns`, `-n`/`--rows`, `-k`/`--keyword` (and setkey `-k`/`--key`).
- `header --keyword-table` (deprecated alias `--fitsort`).
- `convert --where` / `--columns` filter+export; optional FITS table out.
- `probe --header-bytes` (alias `--bytes`) / `--timeout`.
- `probe` SSRF guard: blocks private/loopback/link-local/reserved addresses via
  `getaddrinfo` (all records) and re-validates every HTTP redirect hop.

### API / ML
- `read` / `read_header` default `hdu=0` (`hdu=None` still autodetection).
- Dataset peers: `FitsTensorDataset` (general N-D), `FitsImageDataset`,
  `FitsCubeDataset`, `FitsSpectrumDataset` (multi-arm `layout=`, IVAR companions).
- HTTP(S) remote prefetch under the configurable cache root.
- mmap guidance for DataLoader / network FS.
- Lift `_HDUInfo` / `_TableWriteProxy` to module scope with `__slots__`.
- HDU `_repr_html_` uses `scope=col` / `scope=row` for accessibility.
- Individual HDU HTML reprs: keyboard-focusable container + theme-aware borders
  (aligned with HDUList/Header).

### Correctness / bindings
- Concurrent reads open a **private** `fitsfile*` per call (CFITSIO R2); no
  shared-handle LRU across threads. SharedReadMeta + shared raw `fd` remain.
- Python table/subset paths no longer share one cpp handle across threads.
- Table writes ensure C-contiguous buffers (signed-stride safe) before
  `fits_write_col`; image writes force contiguous host tensors before
  `fits_write_img`.
- `write_table_hdu` uses RAII `vector<string>` for `fits_create_tbl` name/ttype
  pointers (no mid-throw `char**` leak).
- `evaluate_where` rejects `== NULL` / `!= NULL` on numeric arrays; prefer
  `isnull` / `notnull` or `table.read(..., where=)`.
- Remove duplicate `num_rows` binding; clean dead `analyze_table` comments.
- `append_rows` / `insert_rows` best-effort rollback via `fits_delete_rows` after
  a failed post-insert write.
- FITS header integer keys use `PyLong_Check` + overflow-checked `TLONGLONG`.
- Empty-primary MEF compressed write HDU indexing fixed.
- Jules integrations: probe SSRF (#216), hoist inner classes (#214), HDU HTML a11y
  (#213/#219).

### Benchmarks / tests
- ML loader: on-disk `size_mb` for compressed cases; pin compressed torchfits to
  `hdu=1` (matches fitsio `ext=1`); missing numeric CSV fields use `None`.
- GPU transports: pass `quick=` into table `_build_cases`.
- Table filter tests assert exact fixture row counts.
- Concurrent same-file image/table read smoke tests.
- Scorecard: new local MPS run `exhaustive_mps_20260718_180230`; Linux CPU/CUDA
  hosts remain the rc1 soak runs until CANFAR is re-driven against this tag.

### Docs / transforms
- Mermaid diagrams in architecture (zensical superfences).
- Architecture: per-read handles, deliberate skip of CFITSIO iterator/`where`.
- Roadmap: CFITSIO 1.1 leftovers + permanent design choices from the audit.
- Advanced transforms frozen for 1.0; Lupton RGB wrapper in `transforms.lupton_rgb`;
  richer multi-band RGB deferred to 1.1.
- MegaCam cutout bench: ZNAXIS-aware HDU discovery; peer fitsio ranking;
  materialize-once baseline; payload-based throughput (not whole-file MB/s).
- Docs landing: lean browse grid; nav uses mark-only logo (`torchfits-logo-mark.png`).
- Vendored CFITSIO docs pin **4.6.4**; note `fits_iterate_data` intentionally unused.
- cfitsio-direct: Rice + optional MegaCam `cutout_rep` jobs.

## [1.0.0rc1] — 2026-07-17

Release candidate for the 1.0 API. SemVer `1.0.0` waits for post-rc soak; do
not treat this tag as the final 1.0.0 freeze.

### Changed

- **`verify` / `verify_checksums`: missing checksum keywords are success.**
  Files without `DATASUM`/`CHECKSUM` now return `ok=True`,
  `status="no_checksums"`, CLI text `OK (no checksum keywords)`, **exit 0**.
  Previously CLI exited **4** (FAIL). Aligns with `fitsverify` (missing
  keywords are not corruption). Scripts that treated any nonzero verify exit
  as “bad file” must key off `status == "fail"` / exit 4 instead.

- **Root table helpers deprecated.** `read_table`, `stream_table`, and
  `read_table_rows` emit `DeprecationWarning`. Prefer `torchfits.table.read`
  / `read_torch` / `scan_torch`.

### Fixed

- **`ArcsinhStretch` / `LogStretch`: validate `a > 0` in `__init__`.**
  Previously `a=0` silently produced `NaN` (div-by-zero in `inverse` /
  `forward`). Now raises `ValueError` with a clear message. (`transforms/stretch.py`)

- **`_normalize_row_slice`: reject negative `stop` with `ValueError`.**
  Previously `slice(0, -1)` silently returned 0 rows — the function cannot
  resolve negative indices without knowing the total row count. Now raises
  `ValueError` with an actionable message. (`_table/utils.py`)

- **Empty `WHERE` / `row_slice` / `rows` results: preserve column schema.**
  Previously all empty-result paths returned `pa.table({})`, losing all column
  names/types and causing `KeyError` on valid queries (e.g. `where="ID > 9999"`
  on a table with no matches). New `_empty_table_with_schema()` helper builds
  typed empty tables from FITS header cards, preserving requested column
  ordering. When header schema is unavailable but columns were requested,
  returns null-typed empty columns instead of `{}`. (`_table/read.py`)

- **`io.write()` header type: widen to `Header | dict[str, Any] | None`.**
  Removed `TODO(1.0)` and `type:ignore[arg-type]` in `_table/write.py`. The
  runtime already accepted dicts; only the type annotation was narrow.
  (`io.py`, `_io_engine/write_api.py`, `_table/write.py`)

- **Example runner: `REQUIRED` examples can no longer silently skip.**
  Only `OPTIONAL` examples (e.g. `example_polars.py`) may skip on missing deps.
  `REQUIRED` examples always surface failures. (`examples/test_examples.py`)

### Added

- `docs/cli.md`: `### verify` section (three labels, exit codes, fitsverify note);
  CLI cold-start / process-tax note.
- `docs/compatibility.md`: Python / PyTorch / Arrow / platform matrix.
- `scripts/clean_install_smoke.sh`: local wheel → fresh venv install smoke.
- `tests/test_http_probe_fixture.py`: Range HTTP replay for `probe`.
- HTTP probe JSON records include `"source": "http"` (matches `vos` probe).
- Tests: stretch `a<=0`, empty schema preservation, verify messaging contract,
  deprecation warnings, HTTP probe fixture.

### Benchmark evidence

- Multi-host scorecard (from b1 same-day refresh, still current for rc1):
  `exhaustive_mps_20260717_040150`, `exhaustive_cpu_20260717_040146`,
  `exhaustive_cuda_20260717_042840`.
- Local release-suite (`20260717_212321`, Mac MPS, mmap matrix, `--no-gpu`):
  2,825 rows, 3 deficit rows, exit 0. No domain failures.

### Validation

848+ tests; mypy / ruff clean; docs integrity; examples runner REQUIRED green;
`bash scripts/clean_install_smoke.sh`; HTTP probe fixture.

## [1.0b1] — 2026-07-17

Beta freeze of the public FITS → tensor / dataframe story. Not a SemVer 1.0.0
API freeze (rc line followed for soak + blockers).

### Added

- `torchfits.table.read_torch` (tensor-column dataframe path) and
  `table.read_arrow` (synonym of `table.read`).
- Docs gallery: KaTeX math, transform before/after figures, CLI recipes,
  real-sample cache helpers (merged via docs gallery work).
- Release reviews: rendered docs, API adoption, deep code, real-data CLI vs
  astropy/fitsio/gnuastro/CFITSIO (FITSH skipped).

### Changed

- Docs teach FITS tables as dataframes while keeping the `torchfits.table`
  namespace; which-reader box demotes compatibility aliases.
- Landing / site_description: tensors and dataframes (columnar catalogs).

### Fixed

- `torchfits transform` on integer HDUs: promote to float before transform and
  write float outputs without reusing integer BITPIX headers.

## [0.9.3] — 2026-07-17

### Added

- `torchfits header --fitsort --keyword …` multi-file keyword table (same idea
  as qfits `dfits | fitsort`).
- Optional `vos:` / `vos://` probe when the `vos` package is installed.
- Invalid `--hdu` values exit with usage code 2 instead of a traceback.
- Lean `_repr_html_` on `TensorHDU`, `TableHDU`, and `TableHDURef` for notebooks.
- `torchfits convert --to png` Lupton RGB preview via stdlib PNG (no Pillow /
  NumPy). PPM removed.
- Table convert formats: **parquet**, **csv**, **tsv**, and **arrow** (Arrow
  IPC / Feather V2). Streaming writers for large catalogs (CSV/TSV: flat
  columns only).

### Changed

- `torchfits.transforms` is a package split by domain (`stretch`, `normalize`,
  `fits_meta`, `spectral`, `continuum`, `clip`) with the same public `__all__`.
- Transforms docs: not `nn.Module`; instance-local inverse state; Advanced notes
  for `BandMath`, `PhaseFold`, `AsymmetricLeastSquares`, `AlphaShapeContinuum`;
  invertibility + helpers tables.
- Parquet convert uses streaming `write_parquet(..., stream=True)` (out-of-core).
- Multi-host scorecard refresh (`exhaustive_mps_20260717_040150`,
  `exhaustive_cpu_20260717_040146`, `exhaustive_cuda_20260717_042840`): CUDA **0**
  deficits, CPU **1**, MPS **16**.
- `scripts/gpu-bootstrap.sh` pins `torch>=2.10,<2.11` so CANFAR cu128 installs
  do not pull PyTorch 2.11 and fail the ABI gate.

### Fixed

- Block CFITSIO `sh://` filenames (command injection via `/bin/sh`), extending
  the existing `|` checks.

## [0.9.2] — 2026-07-16

### Added

- **`torchfits` CLI** — MEF-aware shell tools: `info`, `header`, `verify`,
  `diff`, `stats`, `table`, `convert`, `copy`, `arith`, `cutout`, `compress`,
  `decompress`, `transform`, `probe`, `setkey`. JSON/JSONL output and stable
  exit codes. Guide: [`docs/cli.md`](cli.md).

### Changed

- **Public imports** — root is I/O + HDU only. Import transforms from
  `torchfits.transforms`. `torchfits.hdu` is a documented namespace.
- **Removed** `read_fast` and `read_image` (use `read` / `read_tensor`).
  Deleted the unused `_fastio` module.
- Table policy helpers (`can_use_*`, …) are no longer listed in `table.__all__`.

### Fixed

- Signed-byte (`BZERO=-128`) and unsigned smart device reads convert on the
  host then copy once to CUDA/MPS.
- `read_subset` / `SubsetReader` keep signed-byte and unsigned integer
  conventions as narrow dtypes (int8/uint16/uint32) instead of float-promoting
  every cutout.
- Automatic table `where=` with `mmap=True` uses native mmap-scan pushdown when
  safe; `mmap=False` reads then filters in Arrow/tensor space.
- CFITSIO `MINDIRECT` reset to 8640 so ~13 KB HCOMPRESS tiles use direct tile
  I/O.
- Multi-byte mmap image reads use NEON/SSSE3 endian convert for all sizes.
- Uncompressed BYTE_IMG reads use direct `pread` for mmap on and off.
- One-shot image reads use thin `cpp.read_full` instead of handle-cache
  scaffolding on the cold path.
- Repeated cutout benches use the persistent subset reader (open once).
- Deficit scorecard: images any lag above ε; Arrow tables allow ≤1.05×; fitsio
  excluded from mmap-on peers. Linux CPU/CUDA strict-gate **0** deficits; Mac
  MPS **4** on `exhaustive_mps_20260717_000853`.

### Docs

- Site logo/favicon: `torchfits-logo.png`.
- README / benchmark run IDs aligned with [`docs/benchmarks.md`](benchmarks.md).

## [0.9.1] - 2026-07-14

### Fixed

- Native wheel metadata now constrains PyTorch to the 2.10 ABI used to build
  the extension. Torchfits 0.9.0 incorrectly allowed newer incompatible
  libtorch releases, which could segfault during image or table conversion.
- Native builds and imports now reject mismatched PyTorch ABIs, and every CI
  build path installs the same PyTorch minor used by the release wheels.

## [0.9.0] - 2026-07-14

### Fixed

- Writing one FITS file no longer invalidates borrowed native handles for
  unrelated files. Native cache clearing now defers closing in-use handles,
  preventing a subsequent read from dereferencing a closed CFITSIO handle.
- Atomic table-column rewrites now close every managed `HDUList` borrower for
  the target path, so nested open contexts cannot retain an old inode and erase
  an earlier mutation.
- `write()` normalizes `os.PathLike` targets before native cache invalidation.
- Wheels no longer include the C++ build-source directory.
- Numeric tensor-to-Arrow conversion now shares the tensor's NumPy buffer
  instead of iterating through PyTorch storage one byte at a time.
- Automatic table predicates use the fast native full-read path followed by
  Arrow filtering; native row-wise pushdown remains available through the
  explicit `backend="cpp"` policy.

### Added

- **`read_polars()`** — one-call FITS-to-Polars convenience function. Calls `read()`
  with `include_fits_metadata=True`, converts via `pl.from_arrow(rechunk=False)`,
  and returns a `FITSPolarsFrame` wrapper that preserves FITS column metadata
  (TFORM, TUNIT, TDIM, TNULL, TSCAL, TZERO) alongside the `pl.DataFrame`.
  Delegates `__getattr__`, `__getitem__`, `__len__` to the wrapped DataFrame.
- **`scan_polars()`** — genuine streaming Polars path. Yields `pl.DataFrame` batches
  via `pl.from_arrow(batch, rechunk=False)` over `scan()`, without materializing
  the entire Arrow table. Unlike `to_polars_lazy()`, no full table is built.
- **`FITSPolarsFrame`** — lightweight dataclass wrapper around `pl.DataFrame` with
  `field_meta` and `table_meta` dicts for FITS metadata preservation.
- Transform masks now thread through FITS-aware normalization and clipping;
  spectral resampling uses torch-native interpolation with parity references for
  vectorized continuum, phase-folding, wavelet, and sigma-clipping paths.
- **CANFAR CUDA exhaustive (`exhaustive_cuda_0.9.0_20260714_065950`)** — 3,648
  normalized rows across the mmap on/off and CUDA matrix; 7 deficits, all at or
  below 1.439×, with no large-N deficit.

### Changed

- Removed the never-implemented `TensorHDU.stats()` and its empty native result
  from the supported `torchfits.cpp` inventory instead of inventing statistics
  semantics during the 0.8 API freeze.
- `ci-local` now runs its pre-build package-isolation checks against `src/`, so
  a clean Linux clone no longer depends on a pre-existing editable install.
- Native cache environment limits are validated before loading Torch or the
  extension module, preserving useful configuration errors in clean installs.
- Scoped extension-only visibility and semantic-interposition optimizations to
  `_C`; applying them directory-wide also changed vendored CFITSIO's C ABI and
  aborted Linux ASCII-table writes.
- Raw, unmapped image reads now support FITS `BITPIX=64` images as
  `torch.int64`, matching the mapped and scaled readers.
- GitHub workflows use the Node 24-based `actions/checkout@v5` and
  `actions/setup-python@v6`, and pin Apple Silicon testing to `macos-15`
  instead of following the rolling `macos-latest` migration.
- Removed the environment-dependent optional `torch_frame` inheritance from
  `TableHDU` and the `torchfits.hdu.TensorFrame` alias. FITS table columns stay
  as tensor/list mappings, Arrow is the interchange boundary, and Polars is the
  dataframe surface. Any legacy dataframe bridge remains outside torchfits.

- **`rechunk=False` default** on `to_polars()`, `to_polars_lazy()`, `scan_polars()`,
  `read_polars()`, and top-level `to_polars()`. Avoids Polars' unnecessary chunk
  concatenation when Arrow data is already single-chunk (the common case from
  `read()`). Pass `rechunk=True` explicitly to restore the old behavior.
- **`to_polars_lazy()` docstring** — clarified that it materializes the entire Arrow
  table eagerly before wrapping as `LazyFrame`. Users seeking true streaming should
  use `scan_polars()` instead.

### Removed

- **`"cpp_numpy"` table backend alias** — the deprecation alias introduced in
  0.7.0 is removed. Pass `backend="cpp"` instead of `"cpp_numpy"`. The
  `DeprecationWarning` is now a hard `ValueError`.
- **`should_skip_cpp_numpy_for_where`** — internal alias removed from
  `torchfits._table_engine`. Use `should_skip_cpp_for_where`.

## [0.7.0] - 2026-07-11

### Added

- **`FitsTableIterableDataset`** — constant-memory table streaming via `table.scan`
  with worker sharding by scan batch index.
- **`FitsCutoutDataset`** — map-style patch training from `(path, hdu, x, y, …)`
  cutout specs.
- **Zensical documentation site** — `zensical.toml`, `docs/index.md`, GitHub Pages
  workflow, and `pixi run docs-build` / `docs-serve`.
- **`migration_datasets.md`** — breaking-change guide for removed legacy datasets.
- **`transforms.__all__`** — explicit public transform catalog.
- **CI `release-gate` job** — upstream parity, docs contract, data, transforms,
  and security smokes on Python 3.13.
- **Lab benchmark refresh (`exhaustive_0.7.0_20260711_022156`)** — full exhaustive
  lab run (3516 rows, mmap matrix + MPS); CPU performance floor unchanged (core
  deficits ≤1.33×).
- **CANFAR CUDA exhaustive (`exhaustive_cuda_0.7.0_20260711_055635`)** — 3626 rows,
  11 deficits on staging GPU; artifacts archived to `vos:sfabbro/torchfits-gpu-bench/`.
- **CANFAR bench launcher** — headless GPU sessions on staging with VOS persistence
  via `vcp` (`scripts/launch_canfar_gpu_bench.sh`, `scripts/fetch_canfar_bench_vos.sh`).

### Changed

- **Torch-first `table.read` C++ path** — `backend="cpp"` reads via `read_fits_table_rows`
  / `TableReader.read_rows` (torch tensors) instead of the numpy hop; Arrow conversion
  stays at the PyArrow boundary only.
- **Table backend rename** — public backend `"cpp_numpy"` renamed to `"cpp"`; the old
  name still accepted with `DeprecationWarning`.
- **Legacy datasets removed** — `torchfits.FITSDataset` and
  `torchfits.IterableFITSDataset` deleted; use `torchfits.data` typed datasets.
- **`table.py` trim** — re-exports public API only (private `_` helpers no longer
  re-exported from `torchfits.table`).
- **Package description** — PyPI/README positioning for ML datasets + transforms.

### Removed

- **`src/torchfits/datasets.py`** — superseded by `torchfits.data`.

## [0.6.0] - 2026-07-09

### Changed

- **Unified C++ table chunk reads:** Refactored `_read_cpp_numpy_table` to clean up the 7-deep C++ dispatch fallback chain and `hasattr` checks, delegating directly to the modern C++ `TableReader` and `read_fits_table_rows_numpy` APIs. This successfully resolves Roadmap Track B1.
- **Version synchronization:** Unified package version triplet to `0.6.0` across `pyproject.toml`, `pixi.toml`, and package source.
- **Blocking mypy in CI** — the `mypy src/` step in GitHub Actions is now a hard gate (previously non-blocking via `|| echo`). All 103 type errors have been resolved across 18+ source files. Added `[[tool.mypy.overrides]]` in `pyproject.toml` for `pyarrow.compute` (`attr-defined`) and `pyarrow.*` (`ignore_missing_imports`).

## [0.6.0b2] - 2026-07-09

### Added

- **Predicate filter improvements (all sizes now use C++ pushdown):** The
  `predicate_filter` path delegates to C++ for all table sizes, eliminating the
  Python fallback for narrow tables.  The `read_policy.py` size threshold is
  removed — safe non-VLA tables always use C++ pushdown.  Narrow-table
  predicate_filter lag vs fitsio reduced from ~2.86× (smallest) to ≤1.07×.
- **Lightweight is_compressed check:** Compressed images use a fast O(1) header
  probe instead of opening and parsing the full HDU, reducing overhead on
  batched compressed-image reads.
- **Thread-safe caches:** CacheManager internal data structures use
  `std::shared_mutex` for concurrent reader access, safe under multi-worker
  DataLoader patterns without global GIL serialisation.
- **Parallel scan with sequential fallback:** The C++ mmap pushdown scan is now
  parallelised via `at::parallel_for` when `torch::get_num_threads() > 1`,
  with a zero-overhead sequential path when single-threaded.  Added
  `posix_madvise(POSIX_MADV_SEQUENTIAL)` to the filtered scan path for kernel
  prefetch hints.
- **Lab benchmark refresh (mmap-on+off, 0.6.0b2):** 2754 rows, **3 deficits**
  in `20260709_163739` — *down* from 0.6.0b1's 14 deficits and 0.5.0b4's 22
  deficits.  Remaining 3-deficit breakdown:
  - 3 fitstable (narrow): `predicate_filter` on `narrow_{10000,100000,1000000}`
    (1.07–1.25× behind fitsio; `narrow_1000` dropped below the deficit
    threshold).  The gap is now dominated by Python dispatch + Arrow
    conversion overhead, not the C++ scan itself (which reaches near-parity
    with fitsio at ~11.4 ms vs ~11.0 ms for 1 M rows).
  All compressed-image deficits eliminated.  The uint16/uint32 mmap-on
  regression that motivated 0.5.0b4's bswap+BZERO merge is no longer in
  the deficit table.
- **Multi-worker DataLoader coverage:** `tests/test_data.py` now exercises
  ``make_loader(..., num_workers=2)`` for both ``FitsImageDataset`` and
  ``FitsImageIterableDataset``.  Tests fork a subprocess to keep CFITSIO's
  threadpool away from pytest's own threadpool, and verify that every file is
  seen exactly once regardless of ``num_workers`` and shuffle seed.
- **End-to-end FITS round-trip coverage** in `tests/test_transforms_e2e.py`:
  - `TestEndToEndImageRoundTrip` — write / read / scale / inverse for INT16
    with custom BSCALE/BZERO, BZERO=32768 unsigned convention, and INT32 with
    rescaling.
  - `TestEndToEndTableRoundTrip` — FITS binary tables with TSCAL/TZERO use
    real on-disk encoding (via astropy), then ``FITSScaleColumns.from_header`` +
    ``TNullToNan.from_header`` round-trip is verified to within the storage
    precision.
  - `TestEndToEndFITSHeaderNormalize` — full int16 BZERO=32768 round-trip
    through the header-driven normaliser.
- **Release-gate now includes** `tests/test_data.py`, `tests/test_transforms.py`,
  and `tests/test_transforms_e2e.py`.  This closes the *torchfits.data
  documented with multi-worker test coverage* and *torchfits.transforms
  round-trip tests for scaled images and tables* gate items.
- **`AsymmetricLeastSquares(lam, p, max_iter, dim)`** — Eilers 2003 penalised
  baseline correction with asymmetric weights. Iteratively solves the Whittaker
  smoother `(W + λD^T D)z = Wy` with differential weighting (p above baseline,
  1-p below). Standard in Raman/NIR spectroscopy. D^T D penalty matrix built in
  float64 for numerical stability at large λ. Additive decomposition (invertible).
- **`AlphaShapeContinuum(half_window, iterations, dim)`** — Morphological closing
  (dilation→erosion) via `unfold` + max/min. Produces a guaranteed upper envelope
  (always ≥ signal). Practical approximation to the full alpha-shape algorithm
  (RASSINE). Additive decomposition (invertible).
- **`AsymmetricSigmaClip(n_low, n_high, dim)`** — Simple one-pass asymmetric
  sigma-clipping outlier rejection using `estimate_background` (median + MAD).
  Supports different lower/upper sigma thresholds; replaces outliers with per-group
  median. Lossy (no inverse).
- `_build_d2_matrix` internal helper for the n×n pentadiagonal second-difference
  penalty matrix D^T D used by the Whittaker smoother / AsLS.
- 27 new tests for the three transforms (201 transforms tests total, all passing).
- Example coverage for `AsymmetricLeastSquares`, `AlphaShapeContinuum`, and
  `AsymmetricSigmaClip` (later removed with the spectral/continuum hard-cut).
- All three transforms exported to the root package for direct
  `from torchfits import AsymmetricLeastSquares` access.
- Documentation for all three transforms in `docs/api.md` and `README.md`
  transform tables.

## [0.6.0b1] - 2026-07-08

### Removed

- Removed deprecated `read_large_table` function (use `stream_table` or `read_table` instead).
- **Custom WHERE AST evaluator runtime** (~120 lines) from `_where.py`: `_evaluate_cmp`,
  `_evaluate_in`, `_evaluate_between`, `_evaluate_isnull`, `_evaluate_where`, and the
  `evaluate_where` public alias. Replaced with `pyarrow.compute` native predicates via
  `_where_mask_for_table`. The parser, tokenizer, normalizers, and `where_columns_from_ast`
  stay for C++ pushdown path compatibility.
- **Compressed parallel decompression path** (~350 lines): `try_read_compressed_rows_parallel`,
  `compressed_parallel_enabled/min_pixels/min_rows_per_thread/max_threads/hcompress_enabled`
  helpers, `load_bswap` templates, `FitsHandleGuard` local class, `is_parallel_compressed_codec_cached`,
  `compressed_parallel_cache`, and `hardware_concurrency` dependency. CFITSIO's built-in decompression
  already covers this serially — the 2-thread cap meant the heuristic rarely activated.
- **Unused `read_rice_parallel`** (~320 lines) from `compression.cpp` — vendored Rice
  decompression, nanobind binding, and the entire `compression.cpp`/`compression.h` files.
  Dead after compressed parallel path removal.
- `bind_compression` from `bindings.cpp` — only bound `read_rice_parallel`.

### Changed

- **3→1 C++ read path merge:** Extracted a single `read_tensor_canonical()` in `fits_detail.h`
  and converted three read paths (`read_full_cached`, `read_full_nocache`, `FITSFile::read_tensor`)
  into thin wrappers, eliminating ~455 lines of duplication.
- **API naming consistency:** Renamed `read_image_canonical` → `read_tensor_canonical` and
  `FITSFile::read_image` → `FITSFile::read_tensor`, aligning C++ with the Python `read_tensor`/`write_tensor` API.
- **bswap+BZERO merge:** Merged the two-pass byte-swap and BZERO offset into a single `parallel_for`
  in the multi-byte mmap fast path. For unsigned images (uint16 with BZERO=32768, uint32 with
  BZERO=2147483648), `bswap + add` executes in one traversal instead of two.
- **Unsigned mmap fast path unlocked:** `_read_unsigned_image_if_needed` now defers to the C++
  path when `mmap=True`, letting `read_tensor_canonical` handle unsigned conventions natively
  (single-pass bswap+BZERO returning uint16/uint32 directly). Previously Python preempted C++
  by calling `read_full_raw` and doing a second offset pass — making the bswap+BZERO merge dead code.
  **uint32_2d: 8.3× faster (now beats fitsio); uint16_2d: 3.5× faster; 5 deficits eliminated.**
- **Vectorized string decode:** Replaced per-row Python `for` loops in `interop.py`
  (`to_pandas`, `to_arrow`), `table_hdu.py` (`get_string_column`, `to_fits`), and
  `table_hdu_ref.py` (`get_string_column`) with `np.char.decode()` + `np.char.rstrip()`
  for significant speedup on large string columns.
- **Deduplicated `fits_schema.py`:** `column_tnull_map()` delegates to `_iter_tfields_indexed()`
  instead of reimplementing the TTYPE/TNULL iteration loop.
- **Deduplicated unsigned dtype and TFORM parsing:** `_table/read.py` now delegates to
  `fits_schema.unsigned_column_dtypes_from_header()` and `fits_schema.iter_table_columns()`
  instead of reimplementing TZERO/unsigned detection and TTYPE/TFORM header walks.
- **Table schema fast path:** `table.schema()` skips data reads when `where=None`,
  inferring the Arrow schema directly from FITS TFORM header cards (≤1 header pass).
- **C++ source extraction:** Split `fits.cpp` (4552 lines) into `fits_detail.h`,
  `fits_file.h`/`.cpp`, `fits_rw.h`; split `table.cpp` (3432 lines) into
  `table_types.h`, `table_reader.h` (header-only), `table_mutation.h`/`.cpp`.
  Removed `extern "C"` linkage from table mutation functions to fix UB from
  C++ exceptions crossing C ABI boundaries. Removed dead declarations,
  unused types, stale comments, and double includes.
- **Merged cache stats:** `CacheManager.get_stats()` now pulls I/O engine metrics
  (`io_hits`, `io_misses`, `io_total_requests`) from the cache subsystem.
- **WHERE evaluator → Arrow compute:** `TableHDU.filter()` now builds a minimal Arrow
  table and delegates to `_where_mask_for_table` (pyarrow.compute native predicates)
  instead of running the old NumPy-based custom evaluator. The parser stays for
  C++ pushdown path compatibility.
- **Table read unification:** Extracted `_read_ranges_as_chunk` from `_read_cpp_numpy_table`
  into shared `_table/engine.py`, removing ~50 lines of duplicated code.
- **CI:** Added non-blocking `mypy src/` step to the GitHub Actions lint job.
- **Benchmark fairness fix:** fitsio is no longer unconditionally skipped — runs when
  `mmap=off` for fair buffered-read comparisons (449 fitsio OK rows in fits domain,
  180 in fitstable).
- `examples/example_image_dataset.py`: `optimize_for_dataset` + correct `pin_memory` when
  reading directly to CUDA.
- `scripts/run_exhaustive_bench_and_patch_docs.sh` skips rebuild when extension imports.

### Fixed

- Root I/O attributes now resolve to the actual public functions, preserving
  inspectable signatures, tracebacks, and identity while keeping bare
  `import torchfits` free of PyTorch, NumPy, Arrow, and the native extension.
- `torchfits.cpp` now has an explicit FITS-native `__all__`; future compiled
  symbols no longer become public accidentally. Direct attribute delegation is
  retained for pre-1.0 compatibility.
- Every lazy root export now has a matching `TYPE_CHECKING` declaration, so the
  shipped `py.typed` marker covers the complete documented root API.
- Removed the empty, misleading `cache` extra: adaptive cache sizing uses the
  standard library. Documentation now states that PyArrow is the core table
  runtime while Pandas, Polars, and DuckDB are optional.
- Runtime initialization no longer swallows native-load or invalid cache
  configuration errors and then marks the failed initialization as complete.
- Header-card write failures and HDU header-preservation failures are no longer
  silently ignored; callers now receive the native error instead of a
  successful return with lost metadata. A dead duplicate header helper was
  removed.
- Overwriting an existing FITS file is now transactional: the complete
  replacement is written beside the target and atomically installed only after
  success. Validation or native-write failures preserve the original bytes and
  file mode instead of deleting the user's file.
- HDU insert, replace, and delete operations use the same transactional rewrite
  rule, so a partial multi-HDU rewrite cannot replace the original file.
- Iterable HDU writes reject empty sequences, unsupported objects, header-only
  dictionaries, and non-tensor image payloads instead of silently emitting
  empty HDUs.
- Image datasets now route through the unified image reader, so their documented
  `mmap="auto"` policy works instead of reaching the bool-only `read_tensor`
  boundary. Remaining immutable column tuples are normalized at public list
  boundaries.
- `TableHDU` validates its trust boundary: non-mapping inputs and columns with
  inconsistent row counts fail immediately instead of creating an internally
  inconsistent table.
- `TableHDU.from_fits()` now uses the public `read_table()` pipeline instead of
  opening a separate native table/header path, keeping cache, validation, and
  runtime initialization behavior consistent with the rest of the package.
- Removed the duplicate `where` entry from the package root `__all__` contract.
- Scoped mypy's missing-import exceptions to optional dataframe integrations
  and the compiled extension, allowing real Python type errors to
  surface. Mypy now checks untyped function bodies and is a blocking local
  preflight and CI check; the resulting `TableHDURef` column-sequence mismatch
  was fixed at the Arrow boundary.
- Release wheels now run image and table round-trip tests against the installed
  artifact, and macOS arm64 wheels use the platform's real minimum deployment
  target (11.0). Platform documentation now matches the wheel matrix. Vendored
  CFITSIO remains statically linked but its development headers, archive, and
  CMake/pkg-config metadata are no longer copied into wheels.
- Multi-worker DataLoader tests now use a real `__main__` guard, matching the
  macOS `spawn` contract instead of recursively creating workers from
  `python -c`; timeout failures preserve worker stderr for diagnosis.
- **Vectorized NULL evaluation:** `_where.py` replaced Python-loop `np.array([v is None for v in val])`
  with vectorized `(val == None)` for element-wise null checks.
- Fixed unused imports in `tests/test_cache_config.py`.
- **Security:** Block CFITSIO pipe injection bypass via leading `!` prefix (`!|command`) in
  `check_fits_filename_security`; also enforced on unified cache open path.
- GPU `scale_on_device` preserves narrow integer H2D for FITS signed-byte (int8) and
  unsigned uint16/uint32 conventions instead of promoting through float32 or int64 on CPU.

### Added

- **Header:** O(N) construction for large dict inputs via keyed fast-path in `_set_card`
  (2000 keys ~0.002s locally vs ~2.5s pre-fix).
- **Jupyter:** Scrollable, sticky-header HTML repr for `Header` and `HDUList`.
- `tests/test_scale_on_device.py` — signed-byte, unsigned, and fitsio parity checks.
- Release gate includes `test_scale_on_device.py`.
- `.cursor/skills/release-api-freeze-review/` — pre-tag API/feature freeze audit workflow.
- **`_table/engine.py`** — shared C++ table read dispatch module with extracted
  `_read_ranges_as_chunk` helper (de-duplicated from `_read_cpp_numpy_table`).

### Performance notes

- Local `bench_ml_loader.py` diagnostic (30×512² float32, CPU, 2 epochs): Rice-compressed
  **1.12×** vs fitsio; uncompressed within ~4% (tune handle cache for your file count).
- Lab exhaustive refresh (`exhaustive_mmap_0.5.0b4_20260630_162835`, H100 MIG): **3626 rows**,
  **13 deficits** (down from 22). Integer CUDA gaps closed; remaining are marginal int8 (≤1.2×)
  and cold `large_uint32_2d` CPU vs astropy (~1.5×).
- User-profile refresh (`unsigned_mmap_fix_20260708`, CPU, mmap=on): **1,377 rows**,
  **25 deficits**. `torchfits_specialized` uint32_2d now beats fitsio (was 5–10× behind);
  uint16_2d at ~1.6× vs fitsio (was ~5×). Remaining deficits dominated by medium-size
  unsigned reads and compressed HCOMPRESS.
  Torchfits dominates table I/O (886×–2,318× vs astropy), image reads (7.92× vs astropy,
  1.76× vs fitsio on large float32), and repeated cutouts (17× vs astropy, 1.09× vs fitsio).

## [0.5.0b4] - 2026-06-30

### Changed

- Centralized FITS binary-table header parsing in `fits_schema` (TFORM/VLA/string/bit/unsigned).
- `table.read` no longer recurses for `where=`; strategy lives in `_table_engine.read_policy`.
- Table C++ handle caches moved to `_table.cache`; I/O cache invalidation no longer depends on
  importing `torchfits.table`.
- README highlights 0.5.0 features and published benchmark speedups; API docs document table
  backends and `where=` tuning environment variables.

### Added

- Unit tests for `fits_schema`, table where-read policy, and runnable example scripts.
- Public `torchfits.table.TABLE_BACKENDS` constant.
- `pixi run release-gate` task matching the release checklist parity/docs/examples gates.

## [0.5.0b3] - 2026-06-30

### Changed

- Refocused torchfits as a FITS I/O package: images, HDUs, headers, checksums,
  compression, FITS tables, caching, and table interop.
- Removed stale public claims that torchfits owns WCS, sphere geometry, HEALPix,
  sky-domain simulation, or training pipelines. Those domains belong outside
  torchfits.
- Added a roadmap and compatibility matrix that distinguish supported, partial,
  unsupported, and out-of-scope behavior.
- Replaced broad parity claims with test-backed parity tiers for common fitsio,
  Astropy, and selected CFITSIO-backed workflows.

### Added

- Extended benchmark matrix: native **uint16/uint32** 2D image fixtures, **typed**
  binary tables (BIT/complex/string columns), and **ASCII** table fixtures.
- `bench_all.py --mmap-matrix` runs mmap-on and mmap-off passes in one CSV so the
  I/O transport table can populate both `disk→CPU` and `disk→RAM→CPU` (plus GPU
  `disk→CPU→GPU` / `disk→RAM→GPU` when CUDA/MPS is available).
- `scripts/run_exhaustive_bench_and_patch_docs.sh` for lab-profile `bench-all` on
  CUDA/MPS hardware with automatic `docs/benchmarks.md` refresh.
- Lab CUDA benchmark snapshot `exhaustive_mmap_0.5.0b3_20260630_063118` (3474 rows,
  mmap on+off matrix, 720 GPU transport rows on H100).
- `docs/parity.md` for the public compatibility matrix.
- Astropy upstream smoke coverage for common image, HDU, compressed-image,
  table, ASCII table, VLA, complex column, and scaled-image workflows.
- Documentation integrity checks for stale WCS/sphere/HEALPix ownership claims.
- Supported-status promotion for in-place mmap table updates on COMPLEX
  (`1C`/`1M`), BIT (`8X`), and fixed-width STRING (`12A`-style) columns.
  `torchfits.table.update_rows(..., mmap=True)` now writes these column
  types correctly on disk. Verified via raw byte inspection and an astropy
  upstream-reader roundtrip. VLA columns remain explicitly unsupported in the mmap
  fast path by design.
- Astropy and fitsio upstream smoke coverage that exercises the
  COMPLEX / BIT / fixed-width STRING mmap-update parity shift,
  including right-padding to the declared column width and verification
  vs the upstream readers. The 8A-string assertion falls back to
  astropy because the local fitsio upstream misdecodes updated `8A`
  rows (the on-disk bytes are bit-exact to the expected layout; this
  is an upstream-reader limitation, not a torchfits writer bug).
- `tests/test_astropy_upstream_smoke.py::test_astropy_compimage_compression_variants_match_torchfits`
  exercising additional `astropy.io.fits.CompImageHDU` compression
  variants (RICE / HCOMPRESS / PLIO) round-tripped against torchfits.

### Fixed

- API docs and install guide now reference `torchfits.cache` for cache tuning
  (`configure_for_environment`, `get_cache_stats`, `clear_cache`) and the root
  I/O helpers `get_cache_performance` / `clear_file_cache` where appropriate.
- Roadmap mmap limitations updated to match the parity matrix (BIT and
  fixed-width STRING mmap updates are supported; VLA and scaled columns remain
  partial).

### Removed

- Dataset/training helper namespace from the torchfits package contract.

## [0.5.0b2] - 2026-06-30

### Fixed

- Patched `fitstable` specialised column projection and row slicing benchmark errors due to invalid `policy` argument.
- Cleaned up C++ build flags in `bench-gpu` to remove strict CUDA and Torch pins.
- Audited C++ codebase for potential memory leaks, redundant hardware heuristics, and API bounds.

### Added

- Restored core FITS benchmarks from v0.3.2: ML DataLoader performance (`bench_ml_loader.py`) and GPU Memory usage/leak validator (`bench_gpu_memory.py`).
- Added exhaustive progress print logging during benchmark execution.
- Added persistent cutout / multi-cutout repeated read benchmarks (`SubsetReader` / `open_subset_reader`) for both CPU and GPU.
- Added `read_tensor` for reading N-dimensional arrays (1D spectra, 2D images, 3D cubes, xD arrays) directly to a single PyTorch `Tensor`.
- Added `write_tensor` as the specialized PyTorch-native writer for writing single PyTorch `Tensor`s directly to FITS files.

### Deprecated

- Deprecated `read_image` in favor of the more general and PyTorch-native `read_tensor`.

## [0.5.0b1] - 2026-06-29

### Changed

- Repository home: `github.com/astroai/torchfits`.
- Default development Python is **3.13** (pixi); supported install range remains **3.10+**.
- Development Status classifier promoted to **Beta**.
- Removed obsolete diagnostic benchmarks, scratch scripts, and legacy HEALPix/WCS artifacts.
- CI rewritten: ruff-only lint, multi-OS/Python test matrix, CFITSIO vendoring via `extern/VERSIONS.txt`.
- Wheel builds: portable flags (no `-march=native`), `cp310`–`cp313` on macOS and Linux.

### Added

- GPU I/O transport benchmark rows (`bench_gpu_transports.py`) with **MPS** on Apple Silicon and **CUDA** on Linux.
- `pixi run bench-mps` for Apple Silicon accelerator benchmarks.
- Automated benchmark report workflow (`.github/workflows/bench-report.yml`).
- `scripts/render_bench_deficits.py` for documenting performance deficits without fixing them.

### Fixed

- Table mutations now invalidate FITS path caches via internal `io` helper (fixes `torchfits._invalidate_path_caches` AttributeError).

## Earlier releases

Earlier 0.1.x through 0.3.x releases included broader experimental astronomy
domains. The current package contract is FITS I/O only; consult the current
README, API reference, roadmap, and parity matrix for supported behavior.

[0.1.0]: https://github.com/astroai/torchfits/releases/tag/v0.1.0
[0.1.1]: https://github.com/astroai/torchfits/releases/tag/v0.1.1
[0.2.0]: https://github.com/astroai/torchfits/releases/tag/v0.2.0
[0.2.1]: https://github.com/astroai/torchfits/releases/tag/v0.2.1
[0.3.0]: https://github.com/astroai/torchfits/releases/tag/v0.3.0
[0.3.1]: https://github.com/astroai/torchfits/releases/tag/v0.3.1
[Unreleased]: https://github.com/astroai/torchfits/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/astroai/torchfits/compare/v1.0.0rc5...v1.0.0
[1.0.0rc5]: https://github.com/astroai/torchfits/compare/v1.0.0rc4...v1.0.0rc5
[1.0.0rc4]: https://github.com/astroai/torchfits/compare/v1.0.0rc3...v1.0.0rc4
[1.0.0rc3]: https://github.com/astroai/torchfits/compare/v1.0.0rc2...v1.0.0rc3
[1.0.0rc2]: https://github.com/astroai/torchfits/compare/v1.0.0rc1...v1.0.0rc2
[1.0.0rc1]: https://github.com/astroai/torchfits/releases/tag/v1.0.0rc1
[1.0b1]: https://github.com/astroai/torchfits/releases/tag/v1.0b1
[0.9.0]: https://github.com/astroai/torchfits/compare/v0.7.0...v0.9.0
[0.7.0]: https://github.com/astroai/torchfits/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/astroai/torchfits/releases/tag/v0.6.0
[0.6.0b1]: https://github.com/astroai/torchfits/releases/tag/v0.6.0b1
[0.5.0b3]: https://github.com/astroai/torchfits/releases/tag/v0.5.0b3
[0.5.0b2]: https://github.com/astroai/torchfits/releases/tag/v0.5.0b2
[0.5.0b1]: https://github.com/astroai/torchfits/releases/tag/v0.5.0b1
[0.3.2]: https://github.com/astroai/torchfits/releases/tag/v0.3.2
