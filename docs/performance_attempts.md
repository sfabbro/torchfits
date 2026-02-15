# Performance Attempts Log (Pre-0.2.1)

Purpose: keep track of what we already tried, what worked, and what did **not** move the remaining gaps enough.

## Remaining gaps this log is anchored to

- ML loader near parity (not a clear win).
- Narrow margins for some large uncompressed reads.

## Tried, not enough (or reverted)

1. **Broad `mmap` auto-tuning / no-mmap sweeps**
- Evidence runs: `targeted_12case_after_nommap_20260210*`, `_diag_top5_after_nommap*`, `_diag_mmap_auto_compressed`.
- Result: helped some cases, but introduced mixed regressions (especially medium/large float paths).  
- Decision: do not rely on broad global `mmap` heuristics.

2. **Compressed read fast-path tuning**
- Evidence runs: `bench_diag_compressed_after_read_fastpath_tune.log`, `bench_diag_compressed_after_revert_read_tune.log`, `targeted_compressed_*`.
- Result: gains were case-specific/noisy; global behavior not consistently better.
- Decision: keep conservative compressed path; avoid extra wrapper-level branching.

3. **Probe-removal / read-path micro-tuning for compressed + MEF**
- Evidence runs: `bench_compressed_rice_after_probe_remove.log`, `bench_mef_medium_after_probe_remove.log`.
- Result: small localized wins, not enough to shift global benchmark picture.
- Decision: partial changes retained only where clearly safe.

4. **NumPy-dispatch or alternate conversion routes in hot path**
- Evidence runs: `_diag_numpy9_after_dispatch*`, `targeted_large_float_via_numpy_*`, `full_after_numpy_dispatch`.
- Result: helped specific buckets but regressed/added overhead elsewhere.
- Decision: not used as default image read strategy.

5. **Handle-cache sweeps for compressed/hcompress**
- Evidence runs: `targeted_hcompress_handlecache*`, `_diag_hcompress_*`.
- Result: useful in repeated access patterns, but insufficient as a standalone fix for remaining gaps.
- Decision: keep moderate handle-cache defaults, avoid aggressive dependence.

6. **Validation interval tweaks as performance lever**
- Evidence runs: `_diag_compressed_validate250`, `_diag_gap_fix_validate_defaults`, `_diag_gap_recheck_*`.
- Result: little net gain; too aggressive relaxation risks stale-read correctness.
- Decision: keep correctness-first validation with modest interval.

7. **ML loader tuning-only passes**
- Evidence reference: 0.2.0 ML loader section in `docs/benchmarks.md`.
- Result: moved toward parity, but still around `~1.0x` median with run-order/cache sensitivity.
- Decision: not enough to claim clear loader win yet.

8. **Compressed `mmap='auto'` policy flip (2026-02-14)**
- Change tested: in `auto` mode, force compressed HDUs to `mmap=True` (instead of `False`).
- Evidence runs: `sentinel_iter1_auto`, `sentinel_iter1_true`, `sentinel_iter1_false`; plus focused compressed micro-benchmark with 25 repeats.
- Result: no robust global improvement. `HCOMPRESS` looked mixed, `RICE` frequently favored `mmap=False`, and variance was high.
- Decision: reverted. Keep compressed default conservative in `auto`.

9. **`BITPIX=-64` no-mmap auto expansion (2026-02-14)**
- Change tested: include float64 images in cold `nommap` auto set.
- Evidence runs: `sentinel_iter2_auto_float64nommap`, `sentinel_iter3_auto_nommap64_cache`, `sentinel_iter3b_auto_nommap64_cache_r9`, plus focused float64 micro-benchmarks.
- Result: mixed/noisy and not consistently better in controlled reruns.
- Decision: reverted (do not keep float64-specific mmap heuristic yet).

10. **Primary-HDU `read_full_nocache` simplification (2026-02-14)**
- Change tested: route primary uncompressed reads to `read_full_nocache` to avoid handle-cache overhead.
- Evidence: `sentinel_iter5_simpler_dispatch`, `sentinel_iter6_simpler_dispatch_targeted`, direct C++ probes (`read_full_nocache` vs `read_full` vs `read_full_cached`).
- Result: not robust; direct probes often showed `read_full_cached` faster for small/medium images after warmup.
- Decision: reverted (keep default cached path).

11. **Force uncompressed float reads through CFITSIO path (2026-02-14)**
- Change tested: remove float32/float64 from raw `pread+byteswap` path in C++ so uncompressed float images always use `fits_read_img`.
- Evidence: `sentinel_iter7_float_via_cfitsio_r9`, plus direct C++ probes for `read_full_cached/read_full` on `m32/m64`.
- Result: no consistent win; medium float64 and integer controls were not reliably better, and global sentinel signal regressed/noisy.
- Decision: reverted (keep existing raw float path available).

12. **`cache_capacity=0` pure-I/O sweep (2026-02-14)**
- Change tested: disable Python data cache in sentinel runs.
- Evidence: `sentinel_iter9_cache0_handle16_r9`, `sentinel_iter10_cache0_handle0_r9`, plus replicated matrix `sentrep_*`.
- Result:
  - `cache=0, handle=16` modestly improved some image medians (`medium_float64_2d`) but regressed others (`small_float32_2d`, `large_float32_2d`) and did not produce a clear global win.
  - `cache=0, handle=0` was consistently worse overall.
- Decision: do not change defaults from this alone; keep handle cache on.

13. **Throughput-mode validation throttling (outside-box, 2026-02-14)**
- Change tested:
  - full opt-out: `TORCHFITS_SHARED_META_VALIDATE=0`, `TORCHFITS_CACHE_VALIDATE=0`
  - safer variant: keep validation enabled but use long intervals (`*_VALIDATE_INTERVAL_MS=5000`).
- Evidence: `sent_throughput_novalidate_*`, `sent_throughput_longinterval_*` vs `sentrep_base_*`.
- Result:
  - `novalidate` showed the largest median uplift on image cases, but with correctness tradeoff for external-overwrite detection.
  - long-interval mode provided smaller gains with lower risk.
- Decision: promising as an opt-in throughput profile; do not force as default.

14. **Thread-local last-HDU metadata shortcut in `read_full_cached` (2026-02-14)**
- Change tested: add a direct thread-local fast path for repeated `(meta_uid, hdu)` lookups to reduce per-call hash/map overhead.
- Evidence: `sent_lowlevel_lastmeta_*` plus focused MEF loop checks.
- Result: direction was inconsistent across replicated sentinel runs (too much drift to call a robust win).
- Decision: reverted (avoid uncertain behavior before release).

15. **Cached-handle HDU position tracking in cache layer (2026-02-14)**
- Change tested: track current HDU in unified cache and skip `fits_get_hdu_num` in `read_full_cached`.
- Evidence: `sent_hdutrack_*`, `sent_hdutrack_v2_*` (seeded replicated sentinel runs).
- Result: degraded absolute torchfits latency in several core cases (small/medium image and random-MEF loops) despite mixed relative speedup noise.
- Decision: reverted.

16. **Byteswap threading threshold tuning (2026-02-14)**
- Change tested: make raw-path endian swap single-threaded below a configurable byte threshold instead of always using `at::parallel_for`.
- Evidence: `sent_bswap_*` runs plus focused TorchFits-only microbench probes (`TORCHFITS_BSWAP_PARALLEL_MIN_BYTES` sweeps).
- Result: noisy and inconsistent; no robust global improvement across sentinel controls and focused reruns.
- Decision: reverted to legacy default behavior (always-parallel unless explicitly overridden via env in experiments).

17. **`fits_get_hduaddrll` metadata caching in `read_full_cached` (2026-02-14)**
- Change tested: cache `(headstart, data_offset, dataend)` per `(file, hdu)` in shared/thread-local metadata.
- Evidence: `sent_hduaddr_*` and `sent_hduaddr_on/off_*` A/B runs.
- Result: high variance with mixed outcomes; no clear stable win after back-to-back on/off comparisons.
- Decision: reverted.

18. **CFITSIO API-choice probe: `fits_read_img` vs `fits_read_pixll` (2026-02-14)**
- Change tested: no code changes; directly benchmarked existing entrypoints (`read_full_cached`, `read_full`, `read_full_unmapped`, `read_full_nocache`) on the same medium float64 file.
- Evidence: direct local microbench via `pixi run python` probes.
- Result: `read_full_cached` (`fits_read_img` path) was already the fastest among existing C++ read entrypoints in these probes; `read_full_unmapped` (`fits_read_pixll`) was slower.
- Decision: treat API choice here as likely not the primary remaining gap; focus next on reproducible benchmark stability and remaining path-specific overheads.

19. **CFITSIO internal I/O constants tuning (`NIOBUF`, `MINDIRECT`) (2026-02-14)**
- Change tested:
  - fixed CMake wiring to target CFITSIO’s actual compile-time knobs (`NIOBUF`/`MINDIRECT`) instead of no-op `TORCHFITS_*` defines.
  - added guard handling so vendored headers can respect override defines.
  - probed one tuned build (`NIOBUF=80`, `MINDIRECT=32768`) vs default on medium float64 microbench.
- Evidence: editable build logs showing override application, plus direct `read_full_cached(..., mmap=False)` latency probe.
- Result: tested tuned pair was slower in local probe (regressed median latency vs default), so no promotion.
- Decision: keep defaults (`NIOBUF=40`, `MINDIRECT=8640`) and only revisit with a broader, controlled sweep when benchmark variance is under control.

20. **Compressed parallel gate sweeps (`MIN_PIXELS`, `MIN_ROWS_PER_THREAD`, HCOMPRESS toggle) (2026-02-14)**
- Change tested: no code change; benchmarked env-gated compressed decode options:
  - `TORCHFITS_COMPRESSED_PARALLEL_MIN_PIXELS`
  - `TORCHFITS_COMPRESSED_PARALLEL_MIN_ROWS_PER_THREAD`
  - `TORCHFITS_COMPRESSED_PARALLEL_HCOMPRESS`
- Evidence: targeted local A/B runs on `1024^2`, `2048^2`, and `4096^2` RICE/HCOMPRESS files (same-run medians vs `fitsio`).
- Result: highly data/order-sensitive. Some runs improved medium HCOMPRESS by disabling parallel decode; other runs regressed `2048^2+` behavior or increased drift.
- Decision: do not change defaults yet. Keep current parallel defaults and treat these env knobs as experiment-only until we have a more stable compressed benchmark protocol.

21. **Compressed stability protocol (randomized order) + default-toggle evaluation (2026-02-14)**
- Change tested:
  - added `benchmarks/benchmark_compressed_stability.py` with randomized per-round config order and CSV/Markdown aggregation.
  - expanded config matrix to include `library_default` (env unset) and explicit forced modes (`force_all_parallel`, `no_parallel`, `no_hcompress_parallel`).
- Evidence: `benchmark_results/compressed_stability_long/*`, `benchmark_results/compressed_stability_mid_2048_v2/*`, and smoke runs (`smoke_compressed_stability_v2*`).
- Result:
  - protocol improves traceability, but compressed results are still materially noisy and dataset/order-sensitive on this host.
  - a temporary attempt to flip HCOMPRESS parallel default OFF did not produce robust, portable evidence across follow-up runs.
- Decision:
  - keep shipping defaults unchanged (`TORCHFITS_COMPRESSED_PARALLEL=1`, `TORCHFITS_COMPRESSED_PARALLEL_HCOMPRESS=1` unless user overrides).
  - keep the new stability harness as the required gate before any future compressed-default changes.

22. **Uncompressed float64 raw-path disable probe (`DOUBLE_IMG` raw off) (2026-02-14)**
- Change tested: route uncompressed `BITPIX=-64` through CFITSIO path only (disable raw `pread+byteswap` in three image hot paths), mirroring the earlier float32 treatment.
- Evidence: focused medium-float64 microbench probes plus sentinel reruns (`hyp_double_raw_off`).
- Result: medium float64 could improve in isolated probes, but broader run behavior stayed noisy with conflicting side-effects.
- Decision: reverted. Keep current double raw path until we have stronger, low-variance evidence.

23. **`read()` intent mode selector + explicit table dispatch (2026-02-14)**
- Change:
  - added `mode='auto'|'image'|'table'` to `torchfits.read(...)`.
  - `read_table(...)` / `read_table_rows(...)` now call `read(..., mode='table')` to bypass image-fallback probing.
  - specialized APIs remain explicit-HDU only; auto-HDU stays in generic `read`.
- Evidence: API tests updated and passing; targeted table microbench with `cache_capacity=0` showed lower median for `mode='table'` vs generic auto in repeated reads.
- Decision: kept. This simplifies user control and reduces avoidable fallback overhead.

24. **Cached-handle direct HDU move in `read_full_cached` (2026-02-15)**
- Change tested: remove `fits_get_hdu_num` and always call `fits_movabs_hdu` directly in the cached-handle image path.
- Evidence: sentinel run `benchmark_results/perf_next_hyp2_movabs` and focused reruns.
- Result: no robust win in targeted medium-float64 path; mixed behavior across controls and high drift.
- Decision: reverted.

25. **`read(mode='image')` direct delegation to `read_image(...)` (2026-02-15)**
- Change tested: in generic `read(...)`, when `mode='image'`, route through `read_image(...)` directly.
- Evidence: repeated A/B microbench rounds on `1024x1024 float64` image.
- Result: high variance and no robust median win (mixed positive/negative rounds).
- Decision: reverted.

26. **Cutout path `read_subset_cached(...)` using unified C++ handle cache (2026-02-15)**
- Change tested:
  - added a new low-level C++ subset reader on top of `get_or_open_cached(...)`.
  - routed Python `read_subset(...)` to this path when handle cache was enabled.
- Evidence: repeated local A/B runs against previous `file_handle.read_subset(...)` path on randomized cutouts.
- Result: highly variable and no robust median gain across rounds.
- Decision: reverted.

27. **Compressed default flip: disable HCOMPRESS parallel decode by default (2026-02-15)**
- Change tested: switched `TORCHFITS_COMPRESSED_PARALLEL_HCOMPRESS` default from ON to OFF (users could still force ON via env).
- Evidence:
  - targeted compressed microbench sweeps showed conflicting signals depending on run order/dataset.
  - randomized default-vs-forced-ON rounds remained high-variance, with frequent regressions in medium/large compressed cases.
- Result: no robust, portable win.
- Decision: reverted (keep existing default ON; treat this as experiment-only).

28. **Python cache-hit LRU reordering removal (`move_to_end`) in hot path (2026-02-15)**
- Change tested:
  - removed `move_to_end(...)` on hit/set for `_auto_mmap_cache`, `_cold_nommap_cache`, and `_hdu_type_cache`.
  - goal: reduce Python overhead in repeated `read(..., mmap='auto')` loops.
- Evidence:
  - mixed and unstable microbench results under alternating A/B loops.
  - no reliable, low-variance improvement signal.
- Result: inconclusive with regression risk.
- Decision: reverted.

29. **Named-HDU resolution via CFITSIO `fits_movnam_hdu` (2026-02-15)**
- Change:
  - added C++ helper `resolve_hdu_name_cached(path, hdu_name)` using `fits_movnam_hdu`.
  - wired `read(..., hdu='<EXTNAME>')` and `get_header(..., hdu='<EXTNAME>')` to use this helper first (fallback scan kept).
- Evidence:
  - targeted A/B on a 50-extension FITS (named extension near end), with old Python scan emulation:
    - `read(..., hdu='SCI_49')`: median gain `~56%` (mean `~56%`, std `~5%`).
    - `get_header(..., hdu='SCI_49')`: median gain `~76%` (mean `~76%`, std `~2.7%`).
  - targeted A/B on a mixed file (40 image extensions + named table extension at end):
    - `read(..., hdu='CATALOG', mode='table')`: median gain `~18%` (mean `~18.5%`, low variance).
  - tests passing (`tests/test_api.py`, `tests/test_cache.py`).
- Decision: kept. This is a direct CFITSIO API-choice improvement and reduces wrapper overhead for named HDUs.

30. **Named-HDU lookup cache in shared metadata (2026-02-15)**
- Change:
  - added `hdu_name_cache` to shared read metadata.
  - `resolve_hdu_name_cached(...)` now normalizes HDU names (trim + uppercase) and caches resolved index per file.
- Evidence:
  - 50-extension image file, repeated reads by EXTNAME:
    - warm name-cache median `~24us` vs cold-per-call `~231us` (`~89.5%` faster).
  - repeated `get_header(..., hdu='<EXTNAME>')`:
    - warm name-cache median `~13.7us` vs cold-per-call `~199us` (`~93.1%` faster).
  - tests passing (`tests/test_api.py`, `tests/test_cache.py`).
- Decision: kept. This closes most repeated EXTNAME lookup overhead while preserving direct CFITSIO resolution semantics.

31. **Handle-based EXTNAME resolver (`fits_movnam_hdu` on open handle) (2026-02-15)**
- Change:
  - added temporary Python path to prefer `resolve_hdu_name_from_handle(file_handle, hdu_name)` before path-based resolver.
- Evidence:
  - focused A/B (old path emulated by forcing fallback to `resolve_hdu_name_cached`) showed severe regression on repeated named reads in local microbench (`~6x` to `~8x` slower median).
  - API tests still passed, but performance did not.
- Decision: reverted.

32. **HDU-type cache check after EXTNAME resolution (2026-02-15)**
- Change:
  - attempted to reuse Python `_hdu_type_cache` before calling `cpp.get_hdu_type(...)` after resolving string HDUs.
- Evidence:
  - mixed named-image/table microbenches showed no consistent signal (`~+1%` in one run, `~-0.4%` in another; within noise).
- Decision: reverted (no robust gain).

33. **Always use C++ EXTNAME resolver, even with `handle_cache_capacity=0` (2026-02-15)**
- Change:
  - in `read(...)`, removed the `handle_cache_capacity > 0` gate around `cpp.resolve_hdu_name_cached(path, hdu)`.
  - this keeps named-HDU lookup on direct CFITSIO API path for no-handle-cache mode too.
- Evidence:
  - old fallback path (header-scan loop) is unreliable in local synthetic MEF cases for some EXTNAME forms; resolver path is robust.
  - new regression test added: `test_read_by_extname_without_handle_cache`.
  - tests passing (`tests/test_api.py`, `tests/test_cache.py`).
- Decision: kept (functional correctness and simpler API path; expected lower overhead than Python header scans).

34. **Direct multi-HDU image API (`read_hdus`) on single open handle (2026-02-15)**
- Change:
  - added explicit Python API `read_hdus(path, hdus, mmap=..., return_header=...)`.
  - uses `cpp.read_hdus_batch(...)` for a one-handle C++ path; resolves named HDUs once via `resolve_hdu_name_cached`.
- Evidence:
  - focused A/B on 10-extension MEF with 80 reads (materializing all outputs in both paths):
    - `read_hdus(..., hdus=[...])` median `~3.77 ms`
    - looped `read(..., hdu=...)` median `~4.07 ms`
    - gain `~7.4%`.
  - focused A/B for named-HDUs (`SCI*` list, 20 extensions):
    - `read_hdus(...)` median `~1.19 ms`
    - looped `read(..., hdu='<name>')` median `~2.15 ms`
    - gain `~44.9%`.
- Decision: kept. This provides an explicit low-overhead user path without runtime heuristics.

35. **Persistent cutout reader (`SubsetReader`) for repeated subsets on one HDU (2026-02-15)**
- Change:
  - added C++ `SubsetReader` (binds file+HDU once, precomputes image/scaling info, repeatedly calls `fits_read_subset`).
  - added Python API `open_subset_reader(path, hdu=...)` and `SubsetReader.read_subset(...)`.
- Evidence:
  - focused A/B on 300 random `96x96` cutouts from `2048x2048` float image:
    - `open_subset_reader(...).read_subset(...)` median `~106 ms`
    - repeated `read_subset(...)` calls median `~222 ms`
    - gain `~52.2%`.
  - API/cache tests passing.
- Decision: kept. This is a direct CFITSIO-centric fast path for iterative cutout workloads.

## Tried and kept (useful)

1. **Signed-byte/int8 fast path improvements**
- Evidence runs: `targeted_signed_byte_fastpath_*`, `_diag_post_xor*`, `targeted_large_int8_after*`.
- Result: consistent wins for int8-heavy cases.
- Decision: kept.

2. **General image-path/cache behavior cleanup**
- Evidence commit: `38900d0` (`core: optimize image read paths and cache behavior`).
- Result: contributed to broad 0.2.0 win profile, but did not eliminate `medium_float64_2d` gap.
- Decision: kept; still needs focused follow-up for float64 medium case.

3. **`mmap='auto'` decision caching**
- Change: cache resolved auto mmap choice per `(path, hdu)` with existing path invalidation hooks.
- Result: reduces repeated Python-side decision overhead without changing explicit `mmap=True/False` behavior.
- Decision: kept (low risk, overhead-focused).

4. **Float raw-path simplification (`read_image_raw`)**
- Change (2026-02-14): skip BSCALE/BZERO guard setup for native floating-point images (`BITPIX=-32/-64`) in C++ raw path, and reuse one compressed-state lookup per call.
- Evidence: `benchmark_results/sent_float_guard_1`, plus focused `medium_float64_2d` microbench probes (`torchfits.cpp.read_full_raw` vs `fitsio.read`) after rebuild.
- Result: meaningful latency drop in the medium float64 hot path in local probes, with no broad regression signal in the 10-case sentinel.
- Decision: kept.

5. **Disable raw `pread+byteswap` path for uncompressed float32 image reads (2026-02-14)**
- Change: in C++ image fast-path switches, route `BITPIX=-32` (`FLOAT_IMG`) through CFITSIO (`fits_read_img`) instead of raw FD read + manual byteswap; keep integer and float64 raw paths unchanged.
- Evidence: `benchmark_results/sentinel_iter_after_float32_raw_off`, `benchmark_results/sentinel_iter_after_float32_raw_off_stable`, `benchmark_results/fast_iter_after_float32_raw_off.csv`.
- Result: robust improvement on key uncompressed gaps in this iteration, notably `large_float32_2d` and `medium_float64_2d` in sentinel runs; compressed behavior remained near-parity/noisy as before.
- Decision: kept.

6. **Float scale-cache fast-skip in `read_full_cached` (2026-02-14)**
- Change: in cached C++ image reads, skip shared scale-cache lookups entirely for native float images (`BITPIX=-32/-64`) and use identity scaling directly.
- Evidence: no-regression validation via rebuilt extension + API tests; sentinel rerun `perf-next-priority-1`.
- Result: low-risk overhead reduction in the float hot path with neutral-to-positive benchmark signal.
- Decision: kept.

7. **`read_image()` shared-handle path for repeated direct reads (2026-02-15)**
- Change:
  - `read_image(...)` now accepts `handle_cache: bool = True`.
  - when `handle_cache=True` and `raw_scale=False`, it uses `cpp.read_full_cached(...)` instead of always forcing `read_full_nocache(...)`.
  - `handle_cache=False` keeps the old uncached direct behavior.
- Evidence:
  - focused repeated-read microbench on `1024x1024 float64` image after rebuild (multi-round A/B):
    - `read_image(handle_cache=True)` consistently faster than `handle_cache=False`
  - observed gain range in local reruns: `~5%` to `~10%`.
  - API tests passing (`tests/test_api.py`).
- Decision: kept. Improves specialized direct image API without adding runtime heuristics.

8. **`read_image(return_header=True)` direct header fetch fast path (2026-02-15)**
- Change:
  - in `read_image(...)`, fetch headers directly via `cpp.read_header_dict(path, hdu)` instead of routing through `get_header(...)`.
  - keep `get_header(...)` fallback on exception.
- Evidence:
  - repeated A/B microbench on small image read+header path:
    - median gain `~2.3%` (low variance) vs old wrapper path.
  - API tests passing (`tests/test_api.py`).
- Decision: kept. Small but consistent improvement for direct API users requesting headers.
336: 
337: 9. **Route Int32 reads through CFITSIO (2026-02-14)**
338: - Change: removed `LONG_IMG` (`int32`) from raw `pread+byteswap` path, forcing it through `fits_read_img` (like `float32`).
339: - Evidence: `medium_int32_2d` went from `0.66x` speedup (regression) to `~1.45x` speedup vs `fitsio`.
340: - Result: Fixed the regression and aligned performance with other types.
341: - Decision: kept.
342: 
343: 10. **Intelligent Chunking for Large Files (2026-02-14)**
344: - Change: implemented 128MB chunking loop in `read_full_cached_fallback` to prevent massive memory spikes in CFITSIO during type conversion.
345: - Evidence: `large_int8_2d` maintained `~3.5x` speedup; `large_float32_2d` maintained `~1.15x` speedup; no overhead observed.
346: - Result: Safe handling of multi-GB files without performance penalty.
347: - Decision: kept.

## External-library hypothesis board

Derived from local source audits of `poloka-core`, `CCfits`, `EleFits`, `afw`, `gbfits`, `qfits`, `libfh`.

- `fits_movnam_hdu` for EXTNAME resolution: **done/kept** (item 29).
- EXTNAME index caching for repeated access (`hdu_name_cache`): **done/kept** (item 30).
- named EXTNAME resolution in no-handle-cache mode (`handle_cache_capacity=0`): **done/kept** (item 33).
- explicit one-handle multi-HDU image read API (`read_hdus`): **done/kept** (item 34).
- persistent one-HDU subset/cutout reader (`SubsetReader`): **done/kept** (item 35).
- `fits_get_img_equivtype` for scaling detection: **done/kept** (used in scale fast path).
- `fits_read_subset` for cutouts: **already in use** (core subset path); extra cached variant tested and **reverted** (item 26).
- compressed decode gating (`HCOMPRESS`/parallel thresholds): **tested repeatedly**, default flips **reverted** (items 20, 21, 27).
- direct HDU motion simplification (`fits_get_hdu_num` + `fits_movabs_hdu` variants): **tested**, **reverted** when not robust (item 24).
- tile/cache tuning style knobs (`NIOBUF`/`MINDIRECT`): **tested**, **reverted/no promotion** (item 19).
- low-level byteswap/threading thresholding: **tested**, **reverted** (item 16).
- single-handle cutout/full-read “open once, iterate” wrappers: **partially in use** (`read_full_cached`, `read_image(handle_cache=True)` kept); additional subset wrapper was **reverted** (item 26).

## What this implies for next iteration

- Prefer **fewer policy branches** in hot image paths.
- Use cfitsio-first direct API paths (full vs subset), with minimal wrapper decisions.
- Treat small-file overhead and medium float64 as first-class targets; avoid broad heuristic churn.
- Gate changes with a fixed 10-case sentinel suite and early-stop significance thresholds.

## Next test matrix (short loop)

Use this fixed set before any full `benchmark_all.py` run:

1. `medium_float64_2d`: verify float64 read path and any scaling-cast overhead.
Acceptance: `>= +5%` speedup vs current median and `> 2 * pooled stdev`.
2. `large_float32_2d`: verify mmap/direct-read choice and byteswap cost.
Acceptance: `>= +5%` with no regression in `large_int8_2d`.
3. `compressed_rice_1`: verify compressed decode path changes only in C++/cfitsio path.
Acceptance: `>= +3%` and stable across two reruns.
4. `compressed_hcompress_1`: same as above, with codec-specific check.
Acceptance: `>= +3%` and no increase in variance.
5. `multi_mef_random_ext_200`: verify open-handle and HDU-switch overhead remains low.
Acceptance: no regression (`>= -2%` bound).
6. `small_float32_2d`: guardrail for tiny/small-file overhead.
Acceptance: no regression (`>= -2%` bound).
7. `small_float64_2d`: guardrail for float64 overhead on small arrays.
Acceptance: no regression (`>= -2%` bound).
8. `medium_int32_2d`: control for integer medium path while touching float64 logic.
Acceptance: no regression (`>= -2%` bound).
9. `large_int8_2d`: control for signed-byte fast path.
Acceptance: no regression (`>= -2%` bound).
10. `benchmark_ml_loader.py` (uncompressed + compressed):
Acceptance: median throughput uplift `>= +3%` or skip promotion to full benchmark.
