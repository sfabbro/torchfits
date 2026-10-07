# Changelog

All notable changes to torchfits are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [1.2.0] — 2026-10-07

### Added
- Real-observation **read** coverage (`tests/test_reads_real_data.py`): the
  reads of the same CFHT sample data are cross-checked against
  `astropy.io.fits` — an independent implementation with its own Rice
  decompressor and header parser. Decompressed MegaCam extensions
  (4644x2112 uint16) match exactly, the raw variable-length tile stream is
  byte-identical row for row (4644 tiles, up to 198 distinct lengths per frame,
  longest exactly filling the declared `1PB(n)` bound), the real
  `BSCALE=1.0`/`BZERO=32768.0` convention round-trips to astropy's unsigned
  array, `read_shape` agrees on all 409 (frame, HDU) geometries, and 435
  megapixels of real mosaic match in every sampled window and in a whole-file
  strided sample. Skips when the sample data is absent.
- Real-observation coverage for the native split
  (`tests/test_core_library_real_data.py`): the three 1.6 GB CFHT MegaPipe
  mosaics and ten Rice-compressed MegaCam MEFs now prove `_core` and `_C` answer
  identically across all 409 (frame, HDU) records, that header text is exactly
  `80 * (cards + 1)` bytes of 80-column cards on every HDU, and that the whole
  9 GB corpus is readable in a fresh interpreter where `torch` and `numpy` both
  raise on import and the torch-linked extension is never imported at all. The
  tests skip when the sample data is absent
  (`scripts/fetch_cfht_megacam_sample.sh`, `scripts/fetch_cfht_megapipe_sample.sh`).
- `libtorchfits_core`: the native stack's torch-free half, shipped as its own
  shared library and bound as `torchfits._core`. It carries CFITSIO, the shared
  read-metadata cache, the FITS inspection rules, and its own `parallel_for`
  (sized by `TORCHFITS_NUM_THREADS`, which ATen reads too). The path-based
  metadata probes — `read_header`, `read_keys`, `read_colnames`, `read_nrows`,
  `read_num_hdus`, `read_hdu_type`, `read_shape`, `read_table_info` — now run
  entirely through it, so they never dlopen libtorch: cold start drops from
  332–337 ms to ~53 ms on macOS arm64. `torchfits._core.Metadata` exposes the
  same queries over an open read-only handle.
- Torch-free Arrow table transport: native fixed columns and flat VLA
  values/offsets now cross a typed, Python-owned raw-buffer boundary, and the
  `table` namespace supports schema/Arrow reads without importing Python
  `torch`. `read_torch` / `scan_torch` remain explicit tensor boundaries.
- Processing-state contract for transforms: `DataState`
  (`STORED`/`PHYSICAL`/`CONTINUUM_NORMALIZED`/`NORMALIZED`), `DataStateError`
  and the typed `Payload` container (interchangeable with the existing
  `{"flux", "ivar"?, "mask"?}` dicts). `FITSHeaderScale`,
  `FITSScaleColumns`, `FITSHeaderNormalize` and `TNullToNan` now declare the
  state they expect and raise an actionable error instead of scaling
  already-calibrated data a second time. `calibration_state(header)` reports
  what the reader produced.
- Inverse-variance propagation through every affine transform
  (`ivar / scale**2`), inverse-variance weighted statistics behind the opt-in
  `weighted=True` flag on the normalizers, clippers and `estimate_background`,
  and companion `mask` awareness (an explicit `mask=` still wins).
- `propagate_ivar=True` on `ArcsinhStretch`, `LogStretch` and `SqrtStretch`
  propagates a companion `ivar` through the nonlinear stretch by the delta
  method (`ivar / (df/dx)**2`). `SqrtStretch` on Poisson counts then recovers
  the variance-stabilised constant `ivar = 4`; clamped regions report `ivar = 0`
  (the output no longer constrains the input) instead of a spurious value.
- `Mask helpers`:`mask_from_dq`, `mask_from_ivar`, `mask_from_nan`,
  `combine_masks`, `apply_mask` — one `True` = valid convention for FITS `DQ`,
  `IVAR` and NaN data.
- Transforms: `MeshBackgroundSubtract` (SExtractor-style tile-mesh sky),
  `SigmaNormalize` (unit-variance scaling for model input), `AffineTransform`
  (explicit scale/offset with exact companion propagation), a faithful
  iterative IRAF zscale (`algorithm="iraf"`, matching
  `astropy.visualization.ZScaleInterval`) and `weighted=`/`algorithm=` options
  on the existing normalizers.
- Data module: spectra gain `wavelength_hdu`/`wavelength_column`, DQ-aware
  `mask_hdu`/`mask_column` via `mask_is_dq=`/`bad_bits=`, and
  `label_key=`/`labels=`; `FitsCubeDataset`/`FitsCubeIterableDataset` gain
  `spectral_slice=(start, stop)` for IFU spectral windows;
  `FitsTableIterableDataset` shards by contiguous row ranges and can stream
  raw tensor batches (`as_batches=True`); `discover_bands()` +
  `FitsImageDataset.from_bands()` build multi-band datasets from extension
  names with zeropoints (`BandInfo.flux_scale`, `band_zeropoints()`).
- `rgb(..., dtype=)` selects the output precision (display paths can stay in
  float32 instead of always computing in float64), and
  `AsymmetricSigmaClip(..., weighted=)`.
- `mask_is_dq=`/`bad_bits=` on `FitsTensorDataset`, `FitsImageDataset`,
  `FitsCubeDataset`, their iterable peers and `FitsStagedCutoutIterableDataset`,
  plus suffix-derived DQ decoding in `FitsImageDataset.from_bands()`.
- `examples/example_ml_training_loop.py` — raw FITS through the new datasets and
  transforms to a training step, covering band discovery, DQ masks, the
  data-state contract and spectra/IFU/catalog loaders (run by the examples
  smoke test in CI).

- data: Spectra companions, IFU windows, table sharding and bands
- transforms: Add data-state contract, IVAR/mask awareness and ML transforms
- types: Check the native extension boundary instead of assuming it
- core: Split a torch-free core library out of the torch-linked extension
### Fixed
- `read_torch(where=)` no longer builds a Python index of every kept row
  when the result columns are tensors. That list was only needed for
  variable-length and string columns, and on a dense numeric filter it cost
  more than the column read.
- CANFAR bench fetches use the pixi `vos` client (`pixi run vcp`). The fetch
  script no longer asks for a user-site `pip install vos`.
- The docs-contract gate was written down three times and nothing held the
  copies together. Found auditing the root config (round-2 unit 24).
  `test_ci_release_gate_matches_local_release_gate` holds the *release* set
  together across `pixi.toml` and `ci.yml`. The **docs-contract** set had no
  such guard and was triplicated: `pixi.toml`'s `docs-contract` task, the CI
  job of the same name, and `scripts/ci_local.sh`, which spelled out the pytest
  command and the docs build separately. All three agreed by luck.

  This was found by walking into it. Adding `tests/test_docs_snippets.py` to
  `pixi.toml` and to the CI job -- two correct edits, made in the same minute --
  silently left `scripts/ci_local.sh` gating on the old two files, and
  `pixi run docs-contract` stayed green throughout, because the divergence was
  in a file the gate does not read. `scripts/ci_local.sh` now calls
  `pixi run docs-contract` instead of restating it, so the list has one home,
  and new `test_docs_contract_gate_is_defined_once` holds the two remaining
  copies together and asserts the local script delegates rather than restating.
  Its redundant `PYTHONPATH=src` went with it: the editable install already
  resolves both `torchfits` and the compiled `_C` out of the source tree, so it
  changed nothing. Break-it **rc=1, 1 failed / 37 passed**; re-pin **5 mutations
  of `pixi.toml` / `ci.yml` / `ci_local.sh`, all caught**.
- The multi-worker `DataLoader` snippet in `docs/quickstart.md` could not be run
  as written. Found auditing the documentation (round-2 unit 23, domain docs).
  Section 8 -- the page's only ML section, and the one a reader reaches first --
  builds a `FitsImageDataset` and `make_loader(dataset, batch_size=32,
  num_workers=4, shuffle=True)` and then iterates it at module top level.
  macOS has used the `spawn` start method by default since Python 3.8, so each
  worker re-imports the snippet, builds a *second* loader, and the recursive
  spawn takes the process down:

      RuntimeError: DataLoader worker (pid 73282) exited unexpectedly with exit
      code 1. Details are lost due to multiprocessing.

  Measured: the snippet exactly as the page wrote it exits **rc=1**; the same
  snippet with `if __name__ == "__main__":` around the training loop exits
  **rc=0**. Nothing in `tests/test_docs_integrity.py` executes a docs Python
  fence -- it parses CLI commands, checks API member names and parameter-table
  completeness, and compares benchmark claims to their runs -- so a snippet
  that cannot run is invisible to all of it. The page now carries the guard with
  a one-line explanation, and new `tests/test_docs_snippets.py` **runs the
  page's own text** as a script, so removing the guard fails the suite; a second
  test guards every `docs/` page structurally so a *new* multi-worker snippet
  cannot be added without it, and a third keeps the docs in step with the house
  style (all eight `examples/` scripts that build a loader already carry the
  guard). Re-pin: 5 mutations of the page, all caught, including a single-quoted
  guard that must *not* trip the check. The four docs snippets that build a
  `num_workers=4` loader without iterating it were measured at rc=0 and are
  correctly left alone -- `DataLoader` spawns workers on the first `__iter__`,
  not at construction.
- The wheel-content contract could not pass on any Linux wheel. Found auditing
  the packaging surface (round-2 unit 22, domain packaging).
  `scripts/check_wheel_contents.py` -- the one packaging gate that runs on a
  plain PR instead of waiting for a release tag, and the check that exists
  because a wheel once shipped without `libtorchfits_core` and passed every
  functional test -- separated extension *modules* from the core *library* by
  file suffix. That is right for macOS (`libtorchfits_core.dylib`) and wrong for
  Linux, where CMake writes the same target as `libtorchfits_core.so`, an
  extension module's own suffix. All three artifacts landed in `modules` and
  `libraries` came out empty, so all three native checks fired at once:

      $ python scripts/check_wheel_contents.py <linux wheel>
      [FAIL] expected the _C and _core extension modules, got
             ['torchfits/_C.cpython-313-x86_64-linux-gnu.so',
              'torchfits/_core.cpython-313-x86_64-linux-gnu.so',
              'torchfits/libtorchfits_core.so']
      [FAIL] expected exactly two extension modules (_C, _core), got [...]
      [FAIL] expected libtorchfits_core as the only non-module library, got [].
      rc=1

  The only automated caller is the `wheel-smoke` job in `.github/workflows/ci.yml`,
  which runs on `ubuntu-latest`. macOS could never have caught it: the locally
  built wheel classifies correctly. The classifier now keys on the artifact's
  *stem* -- `_C` and `_core` are the two `nanobind_add_module` targets and are
  the same on every platform, while the suffix is not -- and the core library is
  matched by prefix so a stale second copy is still reported. The unchanged
  macOS wheel and a Linux-layout wheel both pass now. New
  `tests/test_check_wheel_contents.py` (32 tests) builds a synthetic wheel for
  four platform tag shapes from a macOS checkout and pins the complete-wheel
  pass *and* five rejection shapes (missing library, stale second library,
  missing module, module at two paths, artifact outside `torchfits/`); eight
  mutations of the checker are all caught.
, and one
  staleness guard nothing could see. Found auditing the test suite itself
  (round-2 unit 13, domain 3b).
  `test_cache.py::test_cache_performance_tracking` asserted only that
  `total_requests` grew, which is true whether or not anything is ever
  cached: deleting the whole body of `store_cached_read` left all 58 tests
  in the file green. `test_multiple_file_caching` had the same shape behind
  a `# Read them again (should hit cache)` comment. Both now pin the
  miss-then-hit transition -- and need `return_header=True`, since a bare
  image read never reaches the path that populates the cache.
  `test_optimize_for_dataset_large` mentioned `prefetch_enabled` only in a
  comment and could not have pinned it: `CacheConfig` defaults it to `True`,
  so forcing `True` in the oversized-dataset branch was green across 71
  tests. Two new tests cover both branches from a `False` starting value.
  `image_meta_cache` had no rotation test, and the AST guard meant to stop
  the staleness rule being hand-rolled only matched subscripts -- so
  replacing the validated lookup with `image_meta_cache.get(sig)`, an
  ordinary way to write that rule, kept 296 tests green while serving a
  replaced file's stale shape. The guard now matches read methods too, and a
  new test pins the rotation.
- Two remote-path guards that no test could reach, found by reading the
  coverage report over the remote/HTTP/VOS domain rather than the source
  line by line (round-2 unit 13b, domain 3c). `is_vos_path` rejects a VOS
  path containing whitespace, and `normalize_vos_uri` turns that rejection
  into a `ValueError`; neither line was executed by any test in the suite.
  Reducing the guard to the bare `isinstance` check -- so `" vos://..."` is
  accepted -- left 20 files green. `http_read_range` rejects an inverted
  byte range the same invisibly: deleting the check left 14 files green and
  sent `Range: bytes=10-5` to the server, a malformed request rather than a
  reported error. Both guards are now covered, each with a control case that
  the new tests cannot pass merely by rejecting everything.
- Two boundary guards that nothing could reach: the SQL-injection checks on
  `duckdb_query`, and the VLA offset validators at the Arrow/native boundary.
  Found auditing the table tests themselves (round-2 unit 14b, domain 5) -- but
  by reading a coverage report over the roster, not the tests: the tests are
  strong and said so. `duckdb_query` enforces "exactly one SELECT or EXPLAIN"
  in a comment that names the threat, and no test had ever passed it a second
  statement. duckdb runs both halves of
  `"SELECT ...; CREATE TABLE t AS SELECT 1"`, so the payload's tail executed;
  removing the checks left the whole suite green. `_raw_column_to_numpy`'s six
  offset checks guard the buffer arriving from C++: without them an offsets
  array that starts at 1 rather than 0 returns every row shifted by one, with
  the right shape and no error. Both are now covered, each with the acceptance
  control its rejections need to be worth anything.
- The `agent-home-check` gate could not see an unexecutable hook.
  `scripts/sync_agent_home.py` decided drift with a byte comparison alone, so
  a mirrored `*.sh` that lost its executable bit on the installed side was
  "in sync" -- and running it raises `PermissionError: [Errno 13]`, which is
  precisely the stale-copy failure the script exists to prevent. It also had no
  test at all: being run as a subprocess, `coverage run` records 0% for it and
  cannot tell "untested" from "tested via subprocess". The drift check now
  compares permission bits as well as content, and the script has a test file
  for the first time -- 17 tests covering the mirror, the manifest and both
  drift directions.
- The examples smoke runner could report a crash as a pass. An optional
  example's failure was reclassified as a legitimate skip whenever its output
  contained "skip:", "not installed" or "skipping" *anywhere* -- so a traceback
  naming `/tmp/skipping/data.fits`, or an unrelated "scipy not installed"
  warning printed just before an assertion failed, both came back
  `PASS (skipped (optional))` with the diagnostic discarded. A skip marker now
  has to begin a line, which is how every real decline in the tree is already
  written. (round-2 unit 18)
- The harness stop hook could be permanently disabled by a broken config file.
  `.cursor/hooks/harness-stop.sh` guarded its state read but not its
  `config.json` read, so a malformed config made the hook exit 1 *above* the
  state write -- leaving the reflect cadence frozen forever, since neither the
  generation short-circuit nor the turn accumulator could advance again. A
  missing config was harmless; only a broken one was fatal. The config read now
  falls back to the documented default thresholds, and the hook has tests for
  the first time. (round-2 unit 20)
- Four read-path tests that named a contract they did not assert, and one
  HTTP-subset guard with no test at all. Found auditing the test suite itself
  (round-2 unit 12, domain 3a).
  `test_read_batch_hdus_short_batch_result_falls_back_per_hdu` asserted only
  `len(out) == 2` and `calls == [0, 1]` -- never which tensor came from which
  HDU -- although its `fake_unified` fixture deliberately stamped every read
  with its own call ordinal. Reversing the per-HDU fallback results in
  `_read_batch_hdus` left the 13-test file green, and green across six files:
  exactly the silent HDU misalignment the contract exists to prevent (r4a-01).
  `test_io.py::test_read_target_device_conversion` passed no `device` and
  asserted only the dtype; its body was a byte-for-byte copy of
  `test_read_bf16`. Making `to_device` ignore the requested device outright
  left all 22 tests in that file green while failing 10 in `test_mps.py`. The
  duplicate is removed rather than left advertising coverage a mocked `cpp`
  module cannot provide.
  `test_batch_info_ignores_a_bracket_directory` asserted
  `not cfitsio_base_path(path).endswith("]") or True`, which is
  unconditionally true, and its `existing_files` count could not stand in: a
  bracket-blind helper returns the containing **directory**, which does exist,
  so the count still came back 1. It now pins the helper in both directions.
  `test_kmp_duplicate_lib_ok_set_on_import` asserted only what the environment
  already guaranteed -- pixi's activation env exports `KMP_DUPLICATE_LIB_OK`
  for every task -- so deleting the `setdefault` in `__init__.py` left the file
  fully green. The real guard is
  `test_package_isolation.py::test_import_sets_kmp_duplicate_lib_ok`, which
  pops the variable and imports in a subprocess; it is unaffected by the
  removal.
  Separately, `_bitpix_elem_bytes`'s guard against a *prior* HDU carrying a
  well-formed but unreadable `BITPIX` (28) had no test anywhere: removing the
  raise left 110 tests green across four files, because the matched-HDU branch
  calls `_torch_dtype` on the next line and so masks the regression. With the
  guard gone the HDU walk advances by a guessed 4 bytes per element instead of
  falling back to a full-file read. It is now pinned directly.
- Three `torchfits.data` tests that named a contract they did not assert: a
  dataset could yield its files reversed, could fail to shuffle at all, or
  could warm the cache for a dataset that exposes nothing to warm, and all
  three tests stayed green. Found auditing the test suite itself (round-2
  unit 10, domain 2).
  `TestFitsImageIterableDataset::test_no_shuffle_follows_file_order` asserted
  only `len(out) == 8` while its name announced file order; reversing the
  unshuffled order in `_shard_work_plan` left the whole 62-test file green.
  `test_shuffle_deterministic` compared two same-seeded datasets over a `zip`,
  which compares only the common prefix and carries no negative half, so
  disabling `shuffle` entirely satisfied it — both runs returned the file order
  and matched.
  `TestMakeLoader::test_optimize_cache_no_files_attribute` asserted only
  `isinstance(loader, DataLoader)`; the measured effect of its mutation was to
  take `optimize_for_dataset` from 0 calls to 1 on a dataset with no `files`.
  Both iterable-order mutants were caught only by the sibling
  `FitsTensorIterableDataset` tests in `test_data_ml.py`, which is why the
  defect lived in one class and not the other. Each test now asserts its own
  announced contract: order against an independent astropy read of every file,
  a different seed against the same one, and the no-op via
  `optimize_for_dataset.assert_not_called()`.
- A concurrent-close test that could not fail, and a packaging guard satisfied
  by an empty tree. Found auditing the test suite itself (round-2 unit 9).
  `test_tensor_hdu_concurrent_close_does_not_call_cpp_after_close` built its
  `mock.patch("torchfits._C") as cpp` **inside the reader thread's `with`
  block**, which exits before any statement in the test body could reach it — so
  no assertion could observe what was patched — and swallowed the `RuntimeError`
  that encodes the refusal in the same handler that swallowed every other
  outcome. Deleting the closed-refusal from `TensorHDU.to_tensor` outright left
  it green while the sequential sibling went red: it passed while C++ was being
  called after close, the exact defect in its own name. It is now four tests
  over the fact that `to_tensor` and `mark_closed` both hold `_io_lock` across
  their whole body, including one that **pins** the interleaving a scheduler
  almost never produces (the window between the closed-check and the C++ call is
  two bytecodes, and the reader won it on 50/50 rounds at every
  `sys.setswitchinterval` tried).
  `test_torchfits_contains_only_fits_native_sources` asserted only that four
  paths are *absent*, so — being a containment claim with no positive half — it
  passed with `cpp_src/` deleted or the whole package root absent; it now
  asserts the container exists and holds the FITS native sources first.
  Both classes are now detected mechanically:
  `tests/test_test_suite_quality_guards.py` walks the suite with three AST
  detectors (filesystem absence-only guards, assertions swallowed by a handler
  that catches `AssertionError`, and `mock.patch` aliases no assertion can
  reach), with two further tests pinning the detectors themselves against the
  pre-fix shapes.
- A LONGSTRN `&`+CONTINUE chain whose first segment merely *looks* like a number
  or a complex literal is now joined to **its own** keyword. `_is_string_typed`
  answers "would `_parse_card` type this value as `str`" but was handed the
  **already-unquoted** value, so it could not tell a quoted `'(1.5,2.5)…&'` or
  `'1234…&'` from an unquoted number: both came back False, `target` stayed on
  the *previous* card, and the chain's CONTINUE segments were appended to an
  unrelated keyword while the chain's own value came back **truncated with the
  `&` marker lost**. Measured on a file astropy wrote, one header keyword
  holding `(1.5,2.5)` x20: `read_header()` returned the correct 180 characters
  and `open(path)[0].header` returned **67** — while the preceding keyword came
  back with **213** characters of the wrong value's segments glued onto it. A
  trailing `&` *is* the quoted-chain signal (only a quoted LONGSTRN field can
  carry one), so it now decides on its own; the fail-safe that keeps an orphan
  CONTINUE out of a numeric card is unchanged and pinned.
- `to_arrow` / `to_polars` no longer accept a misspelled `vla_policy` when the
  table happens to have no VLA column. The policy was only checked when a list
  column was actually reached, so `vla_policy="lst"` was silently accepted for
  every table without one — including through `to_polars`, which forwards the
  value untouched. `to_pandas` has always refused up front. The check is now
  done once, before the data is walked.
- `TableHDURef.num_rows` now counts exactly the windows it can read. The
  property re-derived the row window with its own, weaker rules instead of the
  shared `_normalize_row_slice` that `read()` and `iter_rows()` call: a negative
  `start` became `0`, a negative `stop` became *zero rows*, `step` was ignored
  and a non-2-tuple raised a raw unpacking error. So `len(ref)` reported **10
  rows** for `row_slice=slice(-2, None)`, **0 rows** for `slice(0, -1)` and
  **3 rows** for `slice(0, 3, 2)` — windows `ref.read()` refuses outright with a
  precise message. The metadata-only view therefore disagreed with the data
  path about both the count and whether the request was legal at all. All three
  now go through the same helper, so they agree by construction; the six legal
  window shapes keep their exact previous counts.
- `TableHDU.iter_rows(batch_size=...)` no longer silently yields nothing. A
  non-positive `batch_size` reached `range(0, rows, -1)`, which is empty, so
  `iter_rows(-1)` on a 10-row table iterated **zero batches and raised
  nothing** — while `TableHDURef.iter_rows`, its lazy sibling in the same
  package, refuses the identical argument with `batch_size must be > 0`. Both
  now refuse it with that message.
- `TableHDURef(columns=[])` / `select([])` is refused instead of silently
  reading every column. `tuple(columns) if columns else None` could not tell an
  explicitly empty projection from an unset one, and the read path treats an
  empty projection as unset, so asking a ref for *no* columns returned the
  whole table — `select([]).columns` reported `['x']` and `select([]).read()`
  handed back the very column that had been excluded. `TableHDU.select([])`
  did honour the request, so the two peers disagreed. Both entry points now
  raise `ValueError`; a non-empty projection is unchanged.
- `Header.update(other_header)` no longer drops cards. It routed through the
  mapping view (`dict(*args)`), which holds one value per keyword, so copying a
  header through `update()` lost **every card comment** and collapsed a
  multi-line `HISTORY`/`COMMENT` block to its last line — where `Header(source)`
  keeps all of it. A `Header` argument is now applied card-by-card: commentary
  keywords append, value keywords keep the source's first occurrence (the rule
  `_set_mapping_for_card` already documents). Mapping, card-sequence and
  keyword arguments behave as before.
- `TableHDU.num_rows` now follows the header it is derived from. It was a
  `functools.cached_property`, so a **columnless** table — whose row count comes
  from the header's `NAXIS2` — kept its first-read count for the object's whole
  life: after `header["NAXIS2"] = 4` the same table still reported 10 rows in
  `num_rows`, `len(t.data)` and `repr(t)`, while `head()` (built from a fresh
  header) disagreed. It is now cached through the same `_cached` helper that
  `schema` and `string_columns` use, so it is keyed on `Header._version` like
  every other header-derived accessor on the class.
- A negative `slice_index` on a cube dataset is now refused. `FitsCubeDataset`
  and `FitsCubeIterableDataset` validated `spectral_slice` (`0 <= start < stop`)
  and then did not validate `slice_index` at all, six lines away in the same
  constructor — and `Tensor.select` *wraps* a negative index, so
  `slice_index=-1` silently returned the **last** channel of the cube: right
  shape, right dtype, no error, wrong answer, while `spectral_slice=(-1, 2)` in
  the same call raised `ValueError`. Both peers carried a byte-identical copy of
  the check, so the gap existed twice and neither copy could catch the other; the
  validation now lives in one `_resolve_spectral_selection` helper that both
  peers call. An *out-of-range* index is deliberately still an `IndexError`:
  `ds[99]` already raises `IndexError`, so that is this class's existing
  convention and converting only `slice_index` would have been a new asymmetry.
- An empty selection is now refused instead of building a zero-length dataset.
  `_resolve_paths` returned `[]` unchanged, so all eight image / cube / spectrum
  dataset classes accepted `paths=[]` and `FitsCutoutDataset` accepted
  `cutouts=[]` — while the siblings refused the identical mistake (`_as_hdu_list`
  raises `hdu sequence must be non-empty`, `from_bands` raises `no image bands
  found`). What made the gap costly is that the empty dataset did **not** fail on
  its own: `make_loader(..., shuffle=True)` raised from torch's `RandomSampler`,
  which validates `num_samples > 0`, so it looked covered — but `shuffle=False`
  uses `SequentialSampler`, which does not check, and built a loader that
  yielded **zero batches and no error**. The same dataset under the same
  `make_loader` was therefore an error or a silent zero-work training loop
  depending on one flag. All three guards now say `must be non-empty`. A glob
  that matches nothing is unchanged and still falls back to the literal pattern,
  so it keeps failing at *read* time with a real path in the message.
- `cutouts_per_file` no longer coerces an impossible count to 1.
  `FitsStagedCutoutIterableDataset` ran `max(1, int(cutouts_per_file))`, so
  `cutouts_per_file=0` and `cutouts_per_file=-5` both built a dataset that
  handed back **one** cutout per file, with no error and no warning -- and the
  `__repr__` echoed the coerced `1` as though it had been asked for. The
  sibling parameter six lines below (`cutout_size`) already *refused* `0`,
  `-1`, `(0, 4)` and `(4, -2)`, so the two halves of one constructor disagreed
  about the same mistake. The coercion is now gone and the count is rejected
  outright: `cutouts_per_file must be >= 1; got 0`. The numeric coercions the
  parameter always had are unchanged and are now pinned: `2.9` still truncates
  to 2, `"3"` still parses to 3 and `True` still gives 1.
- `FitsCutoutDataset` no longer accepts a window that selects no pixels. It
  normalized each `(path, hdu, x, y, size)` / `(path, hdu, x1, y1, x2, y2)` spec
  and checked nothing about it, so `size <= 0`, an inverted 6-tuple and a
  degenerate `x2 == x1` window each produced a valid **0-pixel** tensor with the
  right dtype and no error. Nothing downstream caught it either:
  `fits_collate_fn` stacks those shapes happily, so `make_loader` handed a model
  a `(4, 1, 0, 0)` batch whose `numel()` is 0 -- a training step that computes
  on nothing and reports a clean loss. Origin and extent are now both checked
  (`cutout origin must be non-negative`, `cutout ... selects no pixels`).
  The documented past-the-edge clamp is unchanged and still pinned: a window
  whose far edge runs off the image keeps the overlap.
- `FitsTableIterableDataset` no longer yields zero rows for a table whose
  columns are all variable-length. Its `where is None` branch inferred a
  chunk's row count by looking at **tensor** columns only --
  `next((v.shape[0] for v in chunk.values() if isinstance(v, torch.Tensor)), 0)`
  -- and defaulted to 0 when it found none. But `scan_torch` returns a
  variable-length column as a Python **list**, not a tensor, so a 5-row
  all-vlen catalog came back with every column a list, the count fell through
  to 0, and the dataset yielded **nothing at all** while its sibling
  `FitsTableDataset` in the same module reported `len=5` for the same file.
  The count is now taken from any tensor *or* list column. Equivalently: a
  dataset whose `__len__`-by-iteration disagrees with the map-style loader on
  the same file is now a fixed case, not a silent zero.
- A pre-existing flaky test is no longer flaky.
  `test_data_ml.py::TestIterableFileSharding::test_shuffle_deterministic_and_shards_disjoint`
  asserted `epoch1 != epoch2` on a shuffle of 3 elements, but 3! = 6
  permutations means a 1-in-6 chance the two epochs agree by coincidence.
  Measured **5 failures in 20 runs** on this checkout; the test fails on
  unmodified `HEAD` too, so it predates this work. Its own comment already
  documented the hazard (and had hardened a second assertion with a hand-picked
  seed) but not this one. It now compares the dataset against *itself* across
  three epochs, which tests the property actually intended -- reproducible when
  re-run, and varying per epoch -- without the coin flip. 40/40 runs pass.

- A tile-compressed image is no longer reported as a catalog by the CLI.
  `hdu_type_name` decided `TABLE` from `XTENSION=BINTABLE`, which is exactly how
  a compressed image is stored -- the image lives in a `BINTABLE` of tiles -- so
  it never consulted `ZIMAGE`, the discriminator the rest of the package already
  uses (`_io_engine.hdu_api.find_first_hdu` checks `ZIMAGE` *or* the
  `ZCMPTYPE`/`ZBITPIX`/`ZNAXIS`/`ZTILE1` cards, and `data.datasets` skips
  `BINTABLE` only when `ZIMAGE` is absent). Measured before the fix on one
  standard astropy `CompImageHDU` file: `torchfits info` labelled the 64x64
  image `type='TABLE' ncols=4 nrows=64`; `torchfits stats` printed **nothing at
  all** and still exited 0; `torchfits table` dumped the internal tile columns
  (`COMPRESSED_DATA`, `GZIP_COMPRESSED_DATA`, `ZSCALE`, `ZZERO`) as if they
  were catalog data, with a preview of the raw compressed bytes; and
  `torchfits arith` / `compress --split hdu` refused with "no image HDUs to
  process" -- so `torchfits compress` could not re-compress its own output.
  `read_tensor` was never at fault: it already decompresses such an HDU
  correctly. `info` now also reports the image geometry from `ZNAXIS*` /
  `ZBITPIX` instead of the tile table's `NAXIS*` / `BITPIX`, so a compressed
  64x64 image reads `shape='(64, 64)' dtype='float32'` rather than
  `shape='(64, 32)' dtype='uint8'`.
- `-e/--hdu` no longer accepts a repeated index. The index list is also the
  output-HDU count for `arith`, `compress` and `decompress`, so a selection's
  length no longer had to match the number of selected HDUs:
  `arith mef.fits --op add --value 1 -e 0,0 -o out.fits` wrote a **2-HDU MEF**
  from a 1-HDU selection, and `-e 0,1,0,1` wrote a 4-HDU one, with no error and
  no warning. `info` / `stats` / `verify` / `header` and `setkey` likewise
  silently processed the same HDU twice. The sibling batch guards
  (`ensure_unique_basenames`, `ensure_unique_split_stems`) already refuse the
  duplicate analogue for `--out-dir` collisions; a repeated index is now a
  usage error (exit 2) everywhere, including `setkey`'s own `-e` parser. Legal
  lists are untouched, including `-e 0,1`, `-e 0,,1` and `-e all`.
- `cutout --box` no longer writes an empty image when the box misses the image
  entirely. `_parse_box` validated the box against itself -- refusing an empty,
  inverted, or negative-origin box as "selects no pixels" -- but never against
  the image, so `--box 10,10,20,20` on a 4x4 frame produced a valid-looking 0x0
  FITS and exit 0, and `stats` renders such a file as all-NaN, so the null
  product flowed downstream silently. A box with no overlap with the image is
  now the same usage error as a degenerate box (exit 2). The documented clamp
  is unchanged: a box running off one edge still keeps the overlap, so
  `--box 0,0,100,100` on a 4x4 frame still yields the whole 4x4. On 3D+ cubes
  the box addresses the trailing `(y, x)` axes, so it is checked against
  `NAXIS2 x NAXIS1`.
- `SigmaClip` no longer accepts a threshold that no pixel can satisfy, and its
  sibling `AsymmetricSigmaClip` no longer lets one through either. `SigmaClip`
  validated `fill` but not `n_sigma` or `max_iter`, while the class directly
  below it already refused `"n_low and n_high must be > 0"` for the same
  mistake. A non-positive or NaN `n_sigma` inverts the acceptance interval
  `mean ± n*std`, so the comparison can never be true: every pixel counts as an
  outlier, the keep-mask empties, and the next iteration divides an empty group
  by `clamp_min(count, 1)`. Measured before the fix on a 10x10 frame of 100
  pixels of 10.0 plus outliers 900 / -500 / 12 / 11: `n_sigma=3.0` and
  `n_sigma=0.5` clip 4/100 pixels, and `n_sigma=0.0`, `-1.0`, `-3.0` and `nan`
  clip **100/100** and return the whole frame as the constant 0.0 (`nan` under
  `fill="nan"`, 0.0 under `fill="median"`) — the right shape and dtype, no
  error, no warning. `max_iter < 1` is the quieter version: the iteration loop
  never runs, so `forward()` hands the input straight back and still reports
  every pixel as kept in `_last_mask`. `AsymmetricSigmaClip`'s guard was one
  operator too narrow — `n_low <= 0` is False for NaN — so a NaN threshold
  reached the same inverted comparison and kept **0/100** pixels, replacing the
  frame with the median. Both now reject a threshold that is not strictly
  positive (`not (n > 0)`, which is NaN-safe by construction), and `max_iter
  must be >= 1`. Tiny-but-positive thresholds, `max_iter=1` and the existing
  non-NaN `AsymmetricSigmaClip` refusals are unchanged, and are pinned.
- `PercentileClipNormalize` now checks its percentile pair. It divided both
  arguments by 100 and used them, while `InterquantileScale` in the same file
  already refused an inverted or out-of-range quantile pair with
  `"Expected 0.0 <= q_low < q_high <= 1.0"`. Measured on a 100-pixel ramp,
  all before the fix: `(1.0, 99.0)` normalised correctly to `[0, 1]`, but
  `(99.0, 1.0)` returned a **constant 1.0** frame — with `min > max`,
  `torch.clamp` returns every element as `max`, so all 100 pixels became the
  1st-percentile value and `(clipped - lower) / (upper - lower)` is exactly
  `1.0`: nothing is clipped at all and nothing says so — and
  `(150.0, 200.0)` or `(-50.0, 50.0)` leaked `RuntimeError: quantile() q must
  be in the range [0, 1]` out of torch instead of the package's own
  `ValueError`. The weighted path took the same inverted pair. The check is
  `0.0 <= lower_pct <= upper_pct <= 100.0`, deliberately *not* the strict `<`
  of `InterquantileScale`: equal percentiles are a supported degenerate span
  here, because `forward` substitutes `1.0` for the divisor when the two
  quantiles coincide so a constant frame stays finite. `0.0`/`100.0`
  endpoints, equality and the defaults all still work.
- A zero `BSCALE`/`TSCAL` is now rejected instead of silently flattening the
  data. `FITSHeaderScale` and `FITSScaleColumns` divide by it in `inverse()`
  and nothing floored the divisor — the only such class in the package, since
  `MinMaxNormalize`, `RobustNormalize`, `SigmaNormalize`, `InterquantileScale`
  and `GlobalScalarNorm` all floor theirs and `AffineTransform` refuses
  `scale=0` outright (`"AffineTransform scale must be non-zero"`). Measured:
  `FITSHeaderScale(bscale=0.0, bzero=5.0).forward(x)` returned a frame of
  constant 5.0 and its `inverse()` returned **all-NaN**; `FITSScaleColumns`
  with a zero TSCAL wrote a constant column whose `inverse()` was all-NaN.
  `FITSHeaderScale.from_header`/`.from_path` reach the same constructor from a
  header card, so a file carrying `BSCALE = 0.0` (astropy writes it without a
  word) built a scaler that flattened every image read through it. Both
  constructors now refuse a zero scale, matching `AffineTransform`; negative
  scales stay legal (`test_differential_astropy.py` relies on `BSCALE=-0.5`)
  and `TSCAL=1`/`TZERO=0` is still filtered out before the check.
- The metadata caches no longer answer for a file that has been deleted. Every
  staleness check read `stale = stored is not None and current is not None and
  stored != current`, so a *missing* file — the one case where `stat` fails and
  the signature is `None` — was never stale. Measured: after warming a cache and
  unlinking the file, `read_header`, `read_shape`, `read_hdu_type`,
  `read_colnames`, `read_nrows` and `read_num_hdus` all returned a confident
  description of a file that no longer existed, while the payload readers
  (`read`) correctly raised. The read cache was worse: with
  `cache_capacity=8, return_header=True` it returned the *bytes* of the deleted
  file. Two root causes, both fixed. The native side
  (`core/fits_core.cpp`, `get_shared_meta_for_path`) had `if (stat(...) == 0)`
  with no `else`, so `has_stat` stayed true and the shape/rows/colnames/type
  caches went on answering; there is now a missing-`stat` branch that drops
  them and rotates the identity. The Python side compared the two signatures
  with a rule that treated "cannot stat" as "unchanged"; the rule is now simply
  `stored != current`, which also separates "gone" from "never stat-able" — a
  CFITSIO extended-syntax path such as `mef.fits[1]` cannot be stat'ed by
  construction and keeps its caches, instead of being silently uncached. Three
  hand-written copies of that rule (`hdu_api.get_header`,
  `hdu_api.autodetect_hdu`, and the read-cache check) have collapsed into the
  one implementation in `caches.signature_cached_get`, so the next reader gets
  one place to get right rather than four.
- `file.fits[N]` now means the same thing to the image metadata probes as it
  already did to the image data. CFITSIO treats a bare `[N]` as a 0-based HDU
  selector that *scopes* the file — opening `mef.fits[1]` parks the handle on
  absolute HDU 2 and makes it the new first HDU, so a caller's `hdu=0` names
  that HDU (measured on a PRIMARY/SCI/ERR/CAT file: `[0]`→PRIMARY, `[1]`→SCI,
  `[2]`→ERR, `[3]`→CAT). The data path applied that offset
  (`FITSFile::ensure_hdu` computes `hdu_num + start_hdu_`); the path-based
  metadata probes moved to an absolute `hdu + 1` and discarded the position
  CFITSIO had already given them. Measured before the fix: for the single path
  `mef.fits[1]`, `read(..., hdu=0)` returned SCI's 4x4 pixels while
  `read_header` reported `EXTNAME='PRIMARY'` and `read_shape` reported PRIMARY's
  shape. `read_header`, `read_header_string`, `read_hdu_type`, `read_nrows`,
  `read_colnames`, `read_table_info`, `read_keys` and `read_shape` now add
  CFITSIO's start HDU the same way the data path does. Deliberately unchanged:
  `read_num_hdus` stays absolute (`fits_get_num_hdus` reports the whole file —
  it is the *indexing* that is scoped, and a filtered count would make a
  `range(read_num_hdus(...))` loop stop early); the pixel-section form
  `file.fits[1:2,1:2]`, which selects pixels rather than an HDU and is what the
  `cutout` CLI uses; and the write paths (row/column mutation, header card
  writes, checksums), which open for writing and are a separate question from
  this read-side split.
- Reading a **list** of paths now obeys the same contract as reading one path.
  `read()` dispatches on `isinstance(path, (list, tuple))` and returned before
  the validation block, so a list ran none of the per-request checks and none of
  the `mode='image'` + table-options guard; it then took the batch C++
  `read_images_batch` fast path whenever `mmap=True`, and that call has no notion
  of `mode`, `columns`, a row window or `return_header` — and, measured, does
  **not** fail on a BINTABLE: it returned a zero-length tensor, so the per-file
  loop that would have applied the request never ran. Measured on two images and
  two BINTABLE files, all before the fix: `read([i1, i2], mode='bogus')` returned
  data where `read(i1, mode='bogus')` raises `ValueError`;
  `read([i1, i2], mode='image', columns=['A'])` returned full 4x4 images where
  the single path raises `ValueError`; `read([i1, i2], mode='table')` returned
  image tensors where the single path raises; `read([t1, t2], columns=['A'])`
  returned `Tensor(0,)` for a table that has five rows; and
  `return_header=True` was silently dropped, returning bare tensors where the
  single path returns `(data, header)`. A list now runs the same validation as a
  single path, and the batch fast path is taken only for a plain whole-image read
  — mirroring the guard the list-of-HDUs dispatch already used in the same file.
  A plain `read([i1, i2], mmap=True)` still takes the batch path.
- Deleting the last HDU of a file no longer destroys the file. A FITS file must
  contain at least a primary HDU, but the writers never enforced it: the C++
  writer accepts an empty payload and produces a 2880-byte file whose only card
  is `END`, which no reader can open. Because every rewrite stages through a temp
  file and `os.replace`, that invalid file was then renamed over a perfectly good
  one. Measured: `delete_hdu(path, 0)` on a single-HDU file **returned normally**
  and replaced 5760 valid bytes with 2880 unreadable ones, leaving a file that
  neither torchfits nor astropy could open — silent loss of the user's data with
  no exception raised. `write(path, HDUList([]))` left the same file behind.
  `_write_hdus_uncompressed` now refuses an empty payload, which covers every
  rewrite path (`delete_hdu`, `HDUList.write`, and anything else that stages
  through the atomic rename), and `delete_hdu` refuses earlier with an actionable
  message — so the refused delete leaves the file byte-identical (verified by
  SHA-1 before and after). `write()`'s image branch already had this check under
  the name "At least one writable HDU is required"; the branches that had drifted
  past it now share it. Deleting the last *extension* of a multi-HDU file, and
  `insert_hdu`/`replace_hdu` on a single-HDU file, are unaffected.
- A `table.read_torch(where=...)` read no longer accepts a request the same call
  without a filter rejects. The `where=` branch runs its thin read inside
  `except (RuntimeError, OSError, ValueError, TypeError, MemoryError)`, which
  cannot tell a *parameter* `ValueError` from "this table cannot be read that
  way" — so an invalid row window or `mmap` mode was caught and re-read as "thin
  path unavailable", and the fallback then re-derived the answer by a different
  route. Measured, all before the fix: `read_torch(p, mmap="sometimes")` raises
  `ValueError` but `read_torch(p, where="A > 1", mmap="sometimes")` **returned
  the filtered rows**; `read_torch(p, start_row=0)` raises
  `start_row must be >= 1` but the same call with `where=` returned all five
  rows, and `start_row=0, num_rows=3` returned rows 1-3 of the whole table
  (`_apply_row_window` clamps `start0` at 0); `num_rows=-2` raised unfiltered
  and silently returned the whole filtered table with a filter. The row-window
  and `mmap` checks now run once, before the filter branch, with the same
  messages the unfiltered path already produced. `num_rows=0` still returns
  empty columns rather than raising: the table reader's thin path has always
  answered that way, and `torchfits.read` rejecting it is a separate contract.
- A `table.read_torch(where=...)` read no longer returns a variable-length
  column at the unfiltered length. The mask gather handled `torch.Tensor` and
  `np.ndarray` and passed every other payload through untouched, but VLA
  columns arrive from the C++ readers as a Python `list` of per-row tensors —
  so a filtered read handed back one dict whose `A` column had 3 rows next to a
  `P` column with all 5. Any downstream `zip`, `stack` or `cat` then silently
  zips misaligned data. The row window already slices lists (its docstring
  says so, "tensors *and* VLA/string lists"); the mask path now gathers them
  the same way, by index, and the kept rows are the *matching* ones.
- `table.open_table_reader` now places a variable-length column on the requested
  device. `TableReaderHandle.read_torch` moved top-level tensors with its own
  comprehension and left the tensors inside a list column where they were:
  measured on MPS, `read_torch(path, device="mps")` returned `A` on `mps:0`
  and `P` on `mps:0`, while `open_table_reader(path).read_torch(device="mps")`
  returned `A` on `mps:0` and `P` on **cpu**. It now calls the existing
  `table_api._move_table_dict` helper, which already knew about list payloads,
  replacing the divergent second copy.
- Streaming a table with a variable-length column no longer fails. `stream_table`
  routes ASCII tables and scaled columns off the raw mmap row route, because
  that route cannot serve them; variable-length columns were missing from the
  list, and the C++ binding rejects them outright ("VLA columns not supported
  for mmap"), aborting the whole stream. Measured: `HDUList[1].iter_rows()` —
  whose `mmap=True` default only downgraded for ASCII tables — raised
  `RuntimeError` on a `PJ(10)` table that `mmap=False` streams fine. VLA
  columns now take the same detour, detected from the same
  `iter_table_columns` walk that detects scaling (`tform_info.vla`). Tables
  that route *can* use still use it: the fast route is asserted reachable, so
  this cannot quietly turn into N single reads.
- **`write(..., quantize="robust")` no longer destroys a sparse image.** The
  robust int16 pack takes its `lo`/`hi` from percentiles, and a percentile window
  narrower than the distribution's spread measures the *background* at both
  ends — `lo == hi == 0.0` for a masked image whose non-zero fraction is below
  `100 - hi_q`, which is 0.1% at the defaults. That is not constant data, but
  the constant-data branch cannot tell the difference: it emitted one code for
  the whole array and still reported `n_clipped == 0`. Measured end to end, a
  512x512 float image with 200 sources on a zero background (peak 500.0) wrote
  a `BITPIX=16` file holding **one distinct value, max 0.000** — every source
  read back as 0.0, `max |read - input| = 500.0`, with no error and no warning;
  the same image written without `quantize=` round-tripped perfectly. The
  threshold is sharp and is a property of the *fraction*, not the array size:
  262 non-zero pixels in 262144 collapse, 263 do not, and 2048x2048 with 2097
  non-zero (0.05%) collapses. `quantize_int16_minmax` — documented as the
  inferior "for tests / comparison only" path — got the same data right, so the
  robust pack was losing to the packing it exists to replace, which is exactly
  the "rare extremes / skewed distribution" case its own module docstring cites
  as the reason to exist. A collapsed window now asks the population
  (`finite.min()`/`finite.max()`), mirroring the guard the `keep_zero` branch
  already had for `hi <= 0`. Genuinely constant data still takes the degenerate
  branch, and the bounds are bit-for-bit unchanged whenever the window is
  informative (pinned against `torch.quantile` on the full population).
- `quantize=..., keep_zero=True` now reports the samples it flattened. When no
  positive sample survives, every finite value is written as code 0, but
  `n_clipped` counted only the non-finite ones: `quantize_int16_robust([-1, -2,
  -3], keep_zero=True)` reported **0** clipped for an array that had been
  entirely overwritten, and `[-1, -2, NaN]` reported 1 instead of 3. The count
  now uses the same in-range test as the general path, and `blank_code` tracks
  the blanks rather than reusing the clip count — the two are no longer the same
  set, which is what made the old code accidentally correct.
- The auto-mmap policy no longer depends on whether the path carries a CFITSIO
  filter. `should_use_cold_nommap` gated on `os.path.getsize(path)`, and
  `mef.fits[1]` is not a stat-able name, so the `OSError` branch reported "small
  file" for **every** extended-syntax path and the policy came out opposite to
  the one the same HDU gets when named without a filter: measured on one
  16 MiB int16 image, `read(p, hdu=1, mmap="auto")` resolved to a direct read
  while `read(p + "[1]", hdu=0, mmap="auto")` resolved to mmap — same bytes,
  same HDU, opposite answer. `paths.cfitsio_base_path` already exists for
  exactly this and its own docstring says "Existence checks must use the base
  file, not the filter"; `table_streaming` already used it. Small files still
  prefer mmap on both spellings, and the large-int16 case the gate exists for is
  asserted reachable.
- A hostile or truncated remote header no longer escapes the cutout fallback as
  a bare `KeyError`/`ValueError`. The HTTP Range reader refuses compressed and
  scaled HDUs with `HttpRangeUnsupported` so the caller materializes the whole
  file and lets CFITSIO answer, and `_data_nbytes` applies that rule to the HDUs
  it *walks past* — a missing or non-numeric card is deliberately re-raised as
  `HttpRangeUnsupported("malformed image cards: ...")` "so callers can fall back,
  never leak raw KeyError/ValueError to callers (r4a-08)". The HDU that was
  actually *matched* read its three cards bare (`int(cards["BITPIX"])`,
  `int(cards["NAXIS1"])`, `int(cards["NAXIS2"])`), so the very same header that
  falls back cleanly in position 1 raised `KeyError: 'BITPIX'` in position 0.
  Measured on the two shapes `test_subset_http_parity` already pins: before the
  fix a missing `BITPIX` raised `HttpRangeUnsupported` when it sat in a prior
  HDU and `KeyError` when it was the target, and a garbage `NAXIS1` raised
  `HttpRangeUnsupported` versus `ValueError: invalid literal for int()`.
  Because `read_subset` and `SubsetReader` both catch only
  `(HttpRangeUnsupported, HttpRangeNotSatisfied)`, neither fell back at all — a
  truncated download, a stale CDN object or a hostile server ended the call
  instead of being answered from the full file. The matched HDU's cards now pass
  through the same guard; a well-formed but unsupported `BITPIX` keeps its own
  `unsupported BITPIX=28` diagnostic rather than the generic wrapper.
- `read_batch_info` now counts a CFITSIO extended-syntax spelling by the file it
  names. `get_batch_info` used `os.path.exists(path)`, and `frame.fits[1]` is
  not a filesystem entry, so the batch preflight reported
  `existing_files: 0` for files the library then read without complaint.
  Measured on one PRIMARY/SCI file before the fix:
  `read_batch_info(["mef.fits[1]"])` returned
  `{"num_files": 1, "existing_files": 0}` while `read("mef.fits[1]", hdu=0)`
  *and* `read_batch(["mef.fits[1]"], hdu=0)` both returned the 4x4 SCI pixels —
  the same disagreement for `[0]`, `[1:2]` and `[0:2]`, so a caller screening a
  frame list by `existing_files` discarded files it could have read.
  `paths.cfitsio_base_path` exists for exactly this and its own docstring states
  the rule ("Existence checks must use the base file, not the filter"); seven
  other existence checks in the tree already follow it. Network URLs are still
  never counted (CFITSIO opens them separately), a genuinely absent file still
  counts as absent with or without a filter, and a bracketed *directory* name
  (`/tmp/[data]/f.fits`) is still not mistaken for one.


- `table.update_rows` no longer invents rows. A `row_slice` that ran past the end
  of the table was accepted, and the write grew `NAXIS2` to fit it: measured on a
  6-row table, `update_rows(path, payload_of_6, slice(4, 10))` returned normally
  and left **10 rows**, with row 5's original value overwritten and four rows that
  never existed created — on the default `mmap="auto"` path, with no warning.
  Both sibling mutations already refused the same mistake (`insert_rows`:
  "row index 99 is out of range"; `delete_rows`: "row_slice start is out of
  range"), and reads clamp instead of extending. The cause was a disagreement
  between the two writers plus a too-narrow fallback: the mmap writer refuses
  the window ("Row range exceeds table length"), `update_rows` caught that
  `RuntimeError` in its layout-fallback handler — the same handler that
  explicitly re-raises truncation to avoid "a fallback that could corrupt the
  file" — and retried through the CFITSIO writer, which does not refuse and
  grows the table instead. `update_rows` now checks the window against the
  header's `NAXIS2` before the cache barrier and either writer, and propagates
  the mmap writer's range refusal rather than retrying it; the second guard
  matters on its own when the header's row count is stale. `append_rows` is
  still how rows are added, and windows that fit still update in place.

- `table.scan` and `table.scan_torch` now reject a bad request when they are
  called, the way `table.read` already did. Both carry the comment *"Eager
  guard: a generator body would defer this until first next()"* above their path
  handling, but only `path` was hoisted — `backend`, `batch_size`, `row_slice`
  and `device` were validated inside the generator, so those errors surfaced
  from the first `next()`, after the caller had already built a pipeline around
  the scan. Measured before the fix (call without iterating): `scan` with
  `backend='bogus'`, `backend=['x']`, `batch_size=0`, `batch_size=-5`,
  `row_slice=slice(0,10,2)` and `row_slice=slice(0,-1)` all returned a generator
  and raised nothing, and `scan_torch` did the same for `batch_size`,
  `row_slice` and `device`; `read` raised for the identical arguments at call
  time. The checks are hoisted into one helper used by both entry points and
  stay in the generator bodies as well, which is deliberate: they also protect
  the internal recursive call from the `where=` branch. Scans stay lazy — the
  first batch is still not read until the caller asks for one.

- A failed cutout fallback no longer leaves a `SubsetReader` permanently broken.
  When a live HTTP Range cutout stops being servable, `SubsetReader.read_subset`
  downloads the whole file and swaps in a local `cpp.SubsetReader` — but it
  cleared `_http_url` and `_http_meta` *before* constructing the replacement,
  which is the one step of that block that can fail. Measured, against a server
  that answers every `Range` with HTTP 416 and every plain `GET` with a 20-byte
  HTML error body: the first `read_subset` raised the real error
  (`RuntimeError: Could not open FITS file: .../cache/<sha>.fits`) and every call
  after it raised `AttributeError: 'NoneType' object has no attribute 'read'`.
  The reader had been left with `_reader=None`, `_http_url=None` and a stale
  `_shape`, so `shape` and `hdu` went on answering for a file it could no longer
  open and the remote route it could still have retried was gone for good — a
  transient outage turned one recoverable failure into a permanently misleading
  reader. The replacement is now built first and the four fields are committed
  together, so a failed fallback raises the same real error every time, the
  reader recovers by itself once the remote does, and the successful fallback
  path is unchanged.
- A rejected `table.write` no longer leaves a readable table full of fabricated
  values. `write_table_hdu` built and validated every column *before*
  `fits_create_tbl`, but three rejections lived in the loop that writes the data —
  so they arrived with the HDU already created and the earlier columns written.
  Measured: a two-column write whose second column was rejected raised
  `element count 32 exceeds array size 4` and left an 8640-byte file that read
  back cleanly as a 4-row, 2-column table with `A` written and the rejected
  column silently all-zero — a caller that trusts the exception is left holding a
  plausible-looking table of invented data. A schema `TFORM` of `8X` is what
  makes this reachable with an ordinary payload: it forces the column to `TBIT`
  whatever the payload dtype is and overrides the repeat count. All three checks
  are pure payload checks and now run before the HDU exists, alongside the
  column checks already there. The header cards were one loop further on and had
  the same shape: `write_table_hdu` writes all the data and *then* calls
  `fits_update_key`, so a header value this writer refuses left a complete table
  carrying none of the caller's cards (measured: 8640 bytes, data correct, no
  cards). Header values are now decided in the same up-front position, through
  one `fits_header_value_type` helper the writing loop also uses. Both residues
  only reach the caller on the in-place write path — `table.write` with a
  `schema`, an unsigned conversion, a quantization or an ASCII table — because
  that branch calls `cpp.write_fits_table` directly, which (before the next entry)
  skipped the temp-file and `os.replace` the other branch uses. The plain branch
  was already atomic: `_io_engine/write_api.py` writes `.{name}.XXXX.tmp.fits` in
  the same directory and renames it over the target only on success, so a failed
  write there left the original byte-identical (same inode, measured).
- A rejected overwrite no longer destroys the file it was overwriting. `table.write`
  has two paths: the plain one goes through `_io_engine/write_api.py`, which writes
  a hidden temp file in the same directory and `os.replace`s it over the target
  only on success, while the other — taken whenever a `schema`, an unsigned
  conversion, a quantization or an ASCII table is involved — called
  `cpp.write_fits_table` directly and never saw that wrapper. The C++ writer
  reaches CFITSIO's `fits_create_file("!path")`, whose leading `!` *unlinks* the
  target before a byte of the payload is validated. Measured: a rejected overwrite
  replaced a good 5-row table with a 2880-byte stub that no longer read as a
  table; a crash or a full disk mid-write did the same, and the window exposed a
  missing or partial file to any concurrent reader. Both branches now share one
  `atomic_write_target` wrapper, so a rejected, failed or crashed write leaves any
  existing file byte-identical and no reader ever sees a partial file at the path.
- A rejected truncated-table mmap read or update no longer leaks a file
  descriptor. Both mmap paths did `open` → `fstat` → `ensure_extent_within_file` →
  `mmap`, and the extent check is precisely what rejects a truncated file — so at
  the only point where these calls throw, the descriptor had no owner, and
  unwinding does not close a raw `int`. Measured: 100 rejected
  `read(path, mmap=True)` calls retained 100 descriptors, `table.read_torch(
  mmap=True)` 200, and the same one-per-call on the update path; the manual
  `close(fd)` calls sat in the arms *after* the check. A long job that keeps
  meeting half-written files — the reason the check exists — exhausts its
  descriptor table and then fails every subsequent open, including CFITSIO's.
  The descriptor is now adopted by an RAII guard the moment `open` succeeds and
  handed to the mapping guard only once the mapping exists.
- A rejected CFITSIO-path `update_rows` no longer leaves the file half-updated.
  `update_rows(mmap="auto")` runs the mmap writer first and, when it rejects the
  payload for a reason that is not a truncation, deliberately retries the *same*
  payload through the CFITSIO writer (`populate_rows`) — which resolved,
  converted and validated each column as the loop reached it and wrote the ones
  before it. CFITSIO has no rollback and the handle is closed on the error path,
  so the exception reached the caller with the earlier columns already on disk.
  Measured: `update_rows` of `{A: 2x1 int32, B: 2x5 int32}` onto two `'J'`
  columns raised `update_rows repeat mismatch for B` and left A holding
  `[100, 200, 3]`. `populate_rows` now resolves, converts and validates every
  column into a write plan and issues the `fits_write_col` calls only once all of
  them are known good, so a rejected payload writes nothing at all; a non-zero
  CFITSIO status from the commit pass is an I/O failure, which no writer in this
  library rolls back.
- A rejected `update_rows_mmap` no longer leaves the file half-updated. The
  in-place mmap update validated each column as it reached it, and its bail-outs
  only `munmap`/`close` — which does not undo the columns already written into
  the mapping, because `MAP_SHARED` dirty pages are written back by the kernel
  whether or not `msync` is ever called. Measured: a two-column update whose
  second column had the wrong dtype raised `update_rows mmap dtype mismatch` and
  left the first column holding the new values. Every per-column check now runs
  before the file is mapped writable, from one shared dtype policy so the
  up-front validation and the write dispatch cannot disagree.
- A `FITSFile` opened before an out-of-band rewrite no longer republishes the
  replaced file's header into the shared read cache. `SharedReadMeta` is shared
  with every other reader, whose handles describe the file as it is *now*; a
  long-lived handle (`open_subset_reader`, any reused `FITSFile`) describes the
  file as it was when it opened, and it published unconditionally — so after the
  validator had already cleared the slot and rotated the generation, the stale
  handle put the old shape straight back. Measured: after rewriting an `(8, 8)`
  image to `(32, 32)`, the next fresh read resolved `(8, 8)` and returned the
  32×32 file's first 64 pixels as if that were the answer, with no error. A
  handle now publishes only while its own file identity still matches the
  generation the cache names; the handle itself keeps its pinned-file answers.
- `Metadata.num_hdus()` no longer leaves the handle pointing at the last HDU.
  The completeness check behind it moves the CFITSIO cursor to the final
  extension without recording that it did, and `move_to` skips
  `fits_movabs_hdu` on a cache hit — so the next read of the HDU it had just
  counted was served from the wrong extension. Measured on a 3-HDU file:
  reading HDU 0, calling `num_hdus()`, then reading HDU 0 again returned
  `(32, 32)` instead of `(8, 8)` in 2 of 3 extensions, silently. The cursor is
  now restored (and the handle's cached position dropped) on the way out.
- `libtorchfits_core`'s worker pool no longer hangs a forked child.
  `pool()` held the pool in a function-local `static`, so a `fork()` inherited
  the pointer but not the threads it names, and work queued in the child blocked
  forever on its condition variable. The project already asked for the "spawn"
  start method at its one fork site; this guards the library half.
  `tests/cpp/test_fork_after_pool.cpp` pins it at 1, 2, 4 and 8 threads.
- `_core`'s `Metadata` handle accessors no longer hold the GIL across CFITSIO
  file I/O, which the module-level entry points in the same file already
  released. Unrelated pure-Python work starved behind a metadata call:
  16,395,508 iterations in 3 s before, 32,623,045 after (module-level: 31.1M /
  32.3M), measured warm and cold.
- The CMake link gate no longer fails a correct build on older linkers, or
  report success on a failed check. `check_core_link.cmake` matched `from ` in
  `nm -m` output, which does not match the two-line form older `ld` emits
  (`...referenced from:` with the library on the next line), so a wheel whose
  CFITSIO symbols were all bound was rejected as not exporting them; and the
  Apple branch's `else()` claimed "resolves every CFITSIO symbol" after `nm`
  had failed and it had examined nothing. Both branches are now pinned by
  `tests/test_cpp_self_checks.py`, which drives the real gate with a stand-in
  `nm`.
- The shared read-metadata cache no longer holds one file descriptor per path
  forever. `SharedReadMeta` memoises each path's raw descriptor so a repeat read
  skips an `open()`, and nothing bounded the memo: a loop over N files left N
  descriptors open for the life of the process. The descriptor table is
  per-process, so the cost landed on everything else sharing it — measured on
  macOS with `RLIMIT_NOFILE=64`, 90 reads left 64/64 descriptors held and 199
  of 200 further opens failing with `EMFILE`, while `clear_all_caches()` handed
  all of them back. Retained descriptors are now capped (`TORCHFITS_MAX_CACHED_FDS`,
  default 32, LRU; `0` restores the old behaviour); the cap costs nothing
  measurable, because the memo saves one `open()` syscall (719/691/688/680 us
  per 2 MiB repeat read at caps 64/8/1/unbounded, within run-to-run spread).
  In-flight readers hold their own refcounted holder, so eviction can never
  close a descriptor under a read. Pinned by
  `tests/test_shared_meta_staleness.py::test_shared_read_cache_does_not_retain_one_descriptor_per_path`.
- A failed raw-descriptor open is no longer memoised. `get_shared_raw_fd`
  stored whatever `open()` returned, so an `fd` of -1 became a cache entry like
  any other and the path stayed on the slow CFITSIO path for the rest of the
  process — the stat identity that would have cleared it (inode, size, mtime)
  does not change when the cause does, so a restored permission or an eased
  descriptor limit left it there. A failed open is now dropped and retried on
  the next read; pinned by the new `tests/cpp/test_raw_fd_retry.cpp`.
- `torchfits._core.Metadata` is now safe to share across threads. CFITSIO keeps
  one mutable current-HDU cursor per `fitsfile`, and the accessors locked only
  around the move to an HDU, then issued the query that depended on that cursor
  with the lock released. Two threads sharing a handle could therefore read
  another thread's HDU. Measured on a 37-HDU Rice-compressed MegaCam MEF with
  four threads: wrong answers, spurious `Could not read image dimensions`, and a
  hard crash. The lock is now held across the whole move-then-read. Python could
  not reach it -- the `Metadata` bindings hold the GIL -- so it is pinned by a new
  C++ probe (`tests/cpp/test_fitsreader_threads.cpp`) rather than a threaded
  Python test.
- `libtorchfits_core`'s thread pool no longer deadlocks when a `parallel_for` is
  issued from inside a pool worker. The caller's keep-the-last-chunk rule only
  covered a nested call on the calling thread; a worker-side nested call queued
  work and waited for workers that were themselves inside user code. Measured
  deadlock at every `TORCHFITS_NUM_THREADS` >= 2. A nested call from a worker now
  runs its chunks inline on that worker, so nesting is safe (it just does not fan
  out a second time) and the throw contract still holds. The header comment that
  claimed unconditional safety now states what is actually guaranteed, and
  `tests/cpp/test_parallel_for_nesting.cpp` pins it at pool sizes 1/2/4/8/64
  under a timeout so a regression fails instead of hanging.
- `table.read(..., where=...)` no longer fails in a torch-free environment. The
  torch-accelerated filter is an optimisation that is *supposed* to return
  `None` and let the Arrow filter take over; it imported `torch` unguarded, so
  a filtered Arrow read raised `ImportError` instead of falling through.
  `examples/example_table_recipes.py` now runs with `torch` blocked, and
  `tests/test_torch_boundary.py` pins it.
- The release smoke and the cibuildwheel test command now verify the native
  split rather than only `torchfits._C`: `tests/test_release_smoke.py` asserts
  `TORCH_FREE`, that the module, library and extension build ids agree, and
  runs a metadata round trip in a child with `torch` and `numpy` blocked.    A wheel that shipped without `libtorchfits_core` passed every functional test
    before this.
  - `pixi run bench-gpu-install` deletes all three native artifacts before
    rebuilding instead of only `_C*.so`, and the exhaustive bench script checks
    build-id agreement rather than a bare `_C` import. A CPU build followed by a
    GPU build left the first build's `libtorchfits_core` in place, which
    surfaces as a build-id `ImportError` at the first metadata call rather than
    as a build failure.
  - `torchfits._core`'s shape accessors now say which axis order they return.
    `Metadata.shape()` and `read_shape()` report the row-major (torch) shape,
    while `Metadata.image_info()` reports `NAXIS1..NAXISn` as written in the
    header; for a non-square HDU the two are exact transposes, and the element
    count is identical either way, so reaching for the wrong one produced a
    silently transposed shape with nothing to hint that an order was in play.
    The two orderings are intentional and unchanged -- `image_info()` exists to
    report the header as written -- so all three accessors gained docstrings
    that state their order and cross-reference each other. The text propagates
    to the generated `_core.pyi`, and
    `tests/test_core_library.py::test_shape_and_image_info_state_their_axis_order`
    pins both the behaviour on a deliberately non-square image and the docstring,
    so removing either fails.
  - `scripts/gen_native_stub.py` no longer emits a `.pyi` that cannot be parsed.
    Its return-type rewrite re-terminated every `def` it touched with `: ...`,
    but stubgen writes a bare `:` when the binding carries a docstring and puts
    the docstring on the indented lines below — so any documented binding with a
    hand-supplied return type got a second body. The rewrite now reproduces
    whichever terminator stubgen emitted, and the generator parses its own output
    before writing it, naming the binding when it fails. This was invisible
    before because stubgen, the rewrites and the drift gate are all string-level
    and would happily commit a broken file; the axis-order docstrings above are
    the first documented bindings that also carry a return-type override, and
    `preflight-push` caught the result on the first run.
  - Writing to a file that a reader still holds open now fails with an
    actionable message instead of CFITSIO's bare `could not open the named file`.
    `open_table_reader()` returns a handle that owns its own CFITSIO handle, so
    dropping the reader *cache* cannot release it and CFITSIO keeps refusing the
    `READWRITE` open. That is a reasonable thing to do — open a reader, then
    write — and the old message named neither the cause nor the fix. The error
    now says which reader to close and what else could produce it (a missing
    parent directory, or permissions). The `FILE_NOT_OPENED` retry it depends on
    is also pinned by a new C++ probe, and the comment above it no longer names
    `fits_already_open` as if it were a status code: it is a CFITSIO function, and
    the status is the generic `FILE_NOT_OPENED`.
  - The native big-endian byte-swap helpers (`bswap16_copy`,
    `bswap16_copy_u16_offset`, `bswap32_copy`, `bswap32_copy_u32_offset`,
    `bswap64_copy`) are now tested directly against a byte-wise reference, at
    element counts straddling every SIMD vector width and the scalar tail. They
    had no direct test, and neither did the rest of the suite reach them: the
    unsigned fast paths in `read_tensor_canonical` are gated on `!compressed`
    while the real CFHT corpus is Rice-compressed throughout. Measured by
    corrupting the NEON 16-bit shuffle (`vrev16q_u8` -> `vrev32q_u8`, a
    plausible wrong-width edit): all 26 pre-existing byteswap tests and all 113
    read/write-parity tests still passed, so byte-reversed pixels could have
    shipped on every arm64 wheel with nothing failing. The new probe reports 12
    failures for that edit, and prints which SIMD branch was compiled in so a
    run that silently fell through to the scalar tail is visible in CI output.
  - C++ environment flags now share one vocabulary. `table_types.h` and
    `table_reader.h` each hand-rolled a `getenv` check against the first
    character of the value, and the two disagreed with the canonical
    `internal_utils.h` helper on real inputs: `TORCHFITS_TABLE_BUFFERED=off`
    enabled the buffered path under the local check and disabled it under the
    helper, and `TORCHFITS_VLA_HEAP_PREAD=on` did the reverse. Same variable,
    opposite answers, no error either way. Both now call the shared helpers, with
    a new `env_flag_default_false` for the opt-in flag that had no equivalent.
    The documented contract (`0`/`1`, and these knobs being explicitly outside
    the public API) is unchanged. A new drift gate in
    `tests/test_docs_integrity.py` fails if a boolean `TORCHFITS_*` flag ever
    parses `getenv` by hand again; numeric knobs stay exempt, since      `TORCHFITS_NUM_THREADS` legitimately parses with `std::stoi`.
  - A table column with more than two axes is now rejected instead of being
    silently truncated. The fixed-width column branch took its repeat count from
    `shape(1)` and packed `rows * repeat` elements, so every axis past the second
    fell out of the element count: a `(4, 3, 5)` column of 60 values was written
    as if it were `(4, 3)`, keeping 12 and discarding 48, and the resulting file
    was indistinguishable from a genuine `(4, 3)` column. The sibling VLA branch
    and `append_rows` already rejected more than two dimensions, and the public
    Python layer does too, so this only ever reached the native binding -- but
    that is the layer that talks to CFITSIO, and the error names the column and
    its shape. The shared `ensure_c_contiguous_ndarray` helper also now bounds
    the element count it is handed against the array's own size. That count is
    what the caller passes on to CFITSIO rather than something the array carries,
    and the contiguous fast path returned the source pointer without consulting
    it, so an over-large count would have read past the end of the buffer; all
    current callers derive it from the array, making the guard defence in depth.
  - Three HDU-keyed metadata lookups on the native file handle now position the
    CFITSIO cursor themselves. `get_image_info`, `get_scale_info` and
    `is_compressed_image_cached` cache on an HDU number but, on a cache miss,
    read whichever HDU the handle is currently sitting on — positioning was left
    to the caller as an unstated precondition. All twelve call sites happened to
    honour it, so nothing was wrong today; measured by removing the positioning,
    after which the whole suite still passed, which is the reason the invariant is
    now enforced in the three methods instead of assumed by their callers. The
    shared metadata cache they feed is process-wide and keyed by path, so a single
    mis-keyed entry would have been served to every later reader of that file
    rather than staying local to one handle.
  - The Galaxy Zoo example now bounds its per-cutout download. It is a *required*
    example, and it pulled one FITS cutout per row from a third-party HTTP
    service using `urllib.request.urlretrieve`, which accepts no timeout and
    blocks until the operating system gives up. Measured against an unroutable
    address, a single cutout took 75s that way; at the eight cutouts the test
    runner requests on a cold cache that is 600s inside an example whose budget
    is 300s, and the runner reports a timeout as a failure. The fetch now uses
    `urlopen` with an explicit 20s timeout, which lands in the error handler
    that was already there -- `TimeoutError` is an `OSError` -- so one slow
    cutout is skipped instead of consuming the whole budget. Observed as a
    one-off `test_example_scripts_exit_zero` failure that passed on re-run.
  - The examples runner now means what `OPTIONAL` says. Two gaps made it weaker
    than it looked: a timeout was reported as a failure for optional and required
    examples alike, so a slow network could still red the gate for an example
    whose purpose is to exercise a network path; and the skip-marker list
    (`"not installed"`, `"skipping"`) matched none of the seven examples that
    print the repository's actual `SKIP: ...` convention, so an optional example
    declining to run was still counted as a failure. `example_ml_galaxyzoo_legacy.py`
    is now optional for the same reason its downloads are bounded -- its
    availability belongs to `legacysurvey.org`, not to this repository. Required
    examples are unchanged: they still fail on a timeout, on a non-zero exit, and
    a `SKIP:` line does not excuse them.
  - `TableReader`'s mutex comment now says where the lock is actually taken. The
    header declared `io_mutex_` and described what it protects -- the CFITSIO
    cursor, the cached offsets, and the reused `scratch_buffer_` -- without saying
    that nothing in the header acquires it. The bindings that hand out a *shared*
    reader take it instead, because the only shared reader is the
    `open_fits_mmap_reader` capsule; every other path is per-call or per-thread and
    needs no lock, since the reader cache is a `ThreadLocalReaderCache` and never
    gives the same reader to two threads. Read on its own, the comment was
    misleading enough to suggest the mutex was dead code. The comment now names
    the two bindings that hold it, explains why the other paths do not, states the
    invariant for new capsule bindings, and a test enforces it.
  - A table column whose `TFORM` repeat count is zero is now rejected instead of
    being filled with a neighbouring column's bytes. The repeat count drives the
    element count, so a zero-repeat column read nothing from a row that does hold
    the other columns' data, and the result surfaced under that column's name.
    `TFORMn = '0J'` is legal -- astropy reads such a column as zero-width -- and
    on a file astropy accepts, `DDDD` came back as `[286331154, 0]`, which is the
    adjacent column `AAA`'s row-1 value, not garbage. The validation covered
    overflow and negative counts but not zero, while the row de-interleaver
    already rejected `repeat <= 0`; the two now agree, and the file is rejected at
    open time with the column named.
  - A comment in the mmap table scanner no longer describes a parallel dispatch
    that was measured and removed. It claimed the scan body was "shared between
    sequential and parallel dispatch" while the only call site is a single
    sequential pass. That mattered: the result gather has a bulk-`memcpy` fast
    path for runs of consecutive file rows, which is only correct while the scan
    emits indices in ascending order -- a coupling the stale comment hid.
  - A non-contiguous `bool` or `BIT` table payload is now written as the values
    the caller passed. The logical and `TBIT` branches of the table writer read
    the payload with a flat index over the element count, which only matches the
    caller's column when the payload is contiguous: for a strided view the
    logical elements sit at the view's strides, not at consecutive addresses
    from the base pointer, so the flat read took the base's own stride-1 order.
    A stride-2 view of `[T, F, T, F, T, F, T, F]` -- four Trues -- was written as
    `[T, F, T, F]`, silently. Every sibling write path already densified first;
    the two branches now do too.
  - The four table-mutation functions now state that they resolve column names
    case-insensitively (`fits_get_colnum(..., CASEINSEN, ...)`) while every read
    path -- and the mmap writer, which the same update API falls back *from* --
    matches names exactly. On a table with `TTYPE = 'FLUX'`, the native
    `update_fits_table_rows` and `append_fits_table_rows` accept `{'flux': ...}`
    and write into `FLUX`, while `read_fits_table_rows(['flux'])` reports
    "Column not found". The supported Python API is unaffected -- all five
    mutation entry points validate names against the header first -- so this is
    a trap for the next reader of the C++ layer, not a live bug.
  - Two unused `TableReader` helpers no longer read as live dispatch.
    `projected_bytes` and `all_fixed_numeric` fed a cross-thread fan-out that was
    measured at 3-8x slower than the sequential path and reverted; nothing in the
    tree calls either one, but their comments described the "parallel fan-out
    gate" as if it still existed. Both now say they are unreferenced, point at
    the measurements, and note that `projected_bytes` over-counts a `BIT`
    projection by up to 8x (it multiplies width by repeat, while a bit array
    occupies `ceil(repeat/8)` bytes).
  - The numpy full-image reader no longer returns a Python object with the GIL
    released. Its raw `pread`/`mmap` shortcut for byte images had two
    `return out;` statements inside a `gil_scoped_release` scope, so each
    returned an `nb::object` -- copying it increfs a Python reference without
    the GIL. Nothing observable breaks today (the array has no other reference,
    so the unsynchronised incref cannot race), but it is a latent defect and it
    broke a convention the other twenty-odd bindings observe without exception:
    release for the work, reacquire before touching Python, return afterwards.
    The I/O now lives in a helper that only reports whether it filled the
    destination, so the single return runs with the GIL held.
  - A stale one-line comment above the skinny-metadata bindings, duplicated by
    the three-line comment explaining the same thing, has been removed.
  - The command-injection regression test now covers the whole native surface.
    It pinned two of the entry points that take a FITS path -- the Python façade
    and `open_fits_file` -- while 41 of them enforce the filename guard, in 25
    places in C++ plus two constructors. All 41 are now exercised, and a
    companion test fails if a new path-taking entry point is added without a
    verdict. Removing the guard from one of them makes the new test fail.
  - Rewriting a tile-compressed FITS uncompressed no longer leaves CompImage
    dither metadata behind. `ZDITHER<n>` -- the subtractive-dither offset
    CFITSIO stamps for the `SUBTRACTIVE_DITHER_1` quantizer -- was in neither
    the exact-key nor the prefix drop set, so `ZIMAGE`, `ZCMPTYPE`, `ZBITPIX`,
    `ZNAXIS*`, `ZTILE*`, `ZQUANTIZ`, `ZNAME*` and `ZVAL*` were all stripped and
    `ZDITHER0` alone survived onto an `IMAGE` extension with no tiles and no
    dither. Opening a compressed MEF and writing it back uncompressed carried
    the card across; so did `replace_hdu` on a compressed HDU. It is dropped as
    a prefix, so every `ZDITHER<n>` is covered, and the compressed write path
    drops it too rather than replaying an offset that contradicts the tiles
    CFITSIO just wrote.
  - The two copies of the CompImage drop set are now one. `_hdu_rewrite` kept a
    private copy of the exact-key set that had already drifted from the
    write-boundary one by a card; both restated the same policy in a form that
    nothing kept in sync. `_hdu_rewrite` now imports the shared sets, so a card
    added to one is stripped by both.
  - `torchfits.read_batch` now documents what it does rather than what it did
    not. The docstring and `docs/api-core-io.md` promised that a file that
    fails to read is "skipped with a warning" so result positions would not map
    1:1 onto the input paths. Measured: with the default `strict=False` it
    raises `RuntimeError` naming the path and how many files read before it, and
    emits no warning; the result list is never silently short. The behaviour is
    the tested and intended one -- `tests/test_where_and_batch_errors.py` pins
    the raise -- so the two documentation sites were corrected to state it,
    including what `strict=True` actually changes.
  - A negative `hdu` is now rejected the same way by every public entry point,
    and a bad HDU index no longer emits a warning blaming the header parser.
    `_resolve_hdu_index` passed a negative index straight through to CFITSIO,
    so `read_shape`, `read_hdu_type`, `read_nrows`, `read_colnames`,
    `read_table_info` and `read_keys` reported "could not move to HDU" while
    `read`, `read_tensor`, `read_subset` and `read_table` all raised
    `ValueError("hdu must be a non-negative integer")`. Separately, `get_header`
    warned that its fast path had failed whenever the fast header read raised --
    including for an ordinary error the fallback cannot fix either, so
    `read_header(path, hdu=99)` emitted a misleading `RuntimeWarning` beside the
    real `OSError`. The warning now appears only when the fallback actually
    rescued the read, which keeps the diagnostic for a genuine parse failure.
  - `table.read`/`table.scan` now report an invalid `backend=` the same way
    whatever its type. The validator tests membership in a `frozenset`, which
    hashes its argument, so a list or dict backend leaked `TypeError: unhashable
    type` out of the public entry points instead of the documented
    `ValueError("backend must be one of: auto, torch, cpp")` -- and only
    unhashable types were affected, since a tuple, `bytes` or an `int` already
    produced the right error. The leak therefore depended on hashability rather
    than on validity.
  - A negative `rows=` index can no longer be read as row 0. The
    non-negativity check validated the value *after* `int()` had truncated it
    toward zero, so `rows=[-0.5]` became `0`, passed the guard, and silently
    read row 0 instead of raising. The same truncation appeared a second time
    in `np.asarray(rows, dtype=np.int64)` -- which is the only check in play
    when the header read fails -- so the two disagreed about what counts as a
    negative index. Both now share one conversion that tests the un-truncated
    value. Positive fractions, numpy integers and integral floats still
    resolve as before.
  - `stream=True` now honours the `pyarrow.Table` input that
    `write_parquet`/`write_csv`/`write_ipc`/`to_pandas`/`to_polars` all document.
    A `Table` is iterable, but it iterates its *columns*, and it exposes
    `.schema`, so the streaming branches built a writer and then fed it columns.
    `write_parquet(..., stream=True)` raised **and left a readable but zero-row
    parquet file on disk**; `to_polars(..., stream=True)` silently yielded one
    polars Series per column; `to_pandas(..., stream=True)` returned a single
    DataFrame where an iterator of per-batch frames is documented. All inputs
    are now normalized to record batches first, while `RecordBatchReader`s and
    plain batch iterables are left untouched so a streaming FITS read still
    streams, and the eager writer still takes its schema from the original input
    so an empty source still produces a valid, readable, zero-row file.
  - `NOT (...)` in a `where=` clause no longer returns rows that a NULL must
    exclude. The **dialect is unchanged** -- a null is unknown, and the 1.1.0
    rule ("negation excludes NULL rows") is what now actually runs. Three
    separate places in the Arrow evaluator broke it, and each had to be fixed
    for the rule to hold:
    (1) `_eval` collapsed nulls to `False` at every leaf, so by the time a
    `not` node ran there was nothing left to invert and it became a plain bool
    flip -- on a `TNULL` column `NOT (A == 1)` returned the NULL row while
    `A != 1` correctly did not, and 8 of 11 SQL equivalence pairs selected
    different rows;
    (2) `_in_mask` and `_between_mask` filled nulls to `False` in their
    *positive* branch, so `NOT (A IN (...))` and `NOT (A BETWEEN ...)` were
    wrong for the same reason -- their negated branches were already correct,
    because they take the negation as a flag and invert before filling;
    (3) `pyarrow.compute.is_in` is a membership test, not a comparison: it
    answers a **non-null `False`** for a null input, where `pc.equal` answers
    null. Its result is now re-marked as null before any inversion, or
    `NOT (A IN (...))` selects the very rows `A NOT IN` excludes.
    Nulls now propagate to a single top-level fill, so negation always agrees
    with the dual operator -- `NOT (x == v)` == `x != v`, `NOT (x > v)` ==
    `x <= v`, `NOT (x IN ...)` == `x NOT IN ...` -- and the Arrow path matches
    the existing `torchfits.where` reference evaluator, which already
    implemented this contract. Verified across 12 negation/dual pairs x 3 read
    backends (`auto`, `cpp`, `torch`) plus the reference evaluator: 0
    disagreements. Queries that contain no negation are unaffected.
  - The parquet/CSV/IPC exporters can now be filtered. `write_parquet`,
    `write_csv` and `write_ipc` named their *destination* parameter `where`
    forwarding `**kwargs` to the read, and `where` is one of the documented
    I/O kwargs -- so `write_parquet(dest, path, where="MAG < 20")` raised
    `TypeError: got multiple values for argument 'where'` and the row filter
    could not be reached on any of the three. The destination is now `dest`
    (`write_parquet(dest, data, ...)`); the argument is unchanged for every
    positional call, which is how all of them were written.
  - FITS column metadata no longer depends on how the read turned out. Three
    separate producers answered "what keywords does this column have"
    differently: the data path published six (`TFORM`, `TUNIT`, `TDIM`,
    `TNULL`, `TSCAL`, `TZERO`), the header-only `schema()` path published
    three, and a read that matched zero rows published none at all -- so the
    same call described its columns differently purely because the result
    happened to be empty. `TUNIT` has no field on the parsed column, which is
    why the two producers had drifted. There is now one producer and one
    column list; every read shape and backend publishes the same six keywords,
    and a plain read still publishes none.
  - **BUGFIX:** a `where=` read no longer silently drops TTYPE-less columns.
    Both filtering strategies built their output column list from a walker
    that skips columns with no `TTYPE` card, so a filtered read returned one
    column fewer than the same read without a filter. The two strategies also
    disagreed with each other, and because the C++ pushdown declines unless
    torch is already imported, *which* columns came back depended on whether
    that was the first filtered read in the process -- the first call returned
    the full column set and every later call silently returned fewer, with the
    same rows. Filtered and unfiltered reads now agree on every backend and
    across repeated calls.

- table,agents: Restore first-wins TTYPE and the agent-tree ignore
- agents: Generate the agent-home copies, and guard documented paths against untracked state
### Changed
- The wheel now ships three native artifacts: `_C` (torch-linked), `_core` (the
  torch-free metadata module) and `libtorchfits_core` itself. `_C` resolves its
  CFITSIO symbols against the core so there is exactly one CFITSIO in the
  process; a post-link `nm` guard fails the build if any of them is left
  unbound, and a build-id constant stamped into all three artifacts turns a
  mismatched install into an `ImportError` instead of a misread.
- `torchfits._C.pyi` is joined by a generated `torchfits._core.pyi`; the stub
  generator now drives both modules and `scripts/check_wheel_contents.py`
  requires all three native artifacts plus both stubs to be present.
- The cold-start boundary benchmark gained `_core` entries and its metadata
  budgets were re-tightened (400 ms → 120 ms, Arrow 900 ms → 600 ms) against
  the now torch-free path. `torchfits.open()` still loads `_C` because the
  `HDUList` handle is shared with the tensor readers; it keeps its own measured
  250 ms budget rather than borrowing the metadata one.
- The CLI parser and metadata commands (`info`, `header`, `probe`, `verify`,
  `table`, `copy`, `setkey`) are now torch-free; pixel/tensor commands declare
  their runtime explicitly so their existing thread controls remain intact.
- Dependency requirements now carry honest ranges. Every non-load-bearing
  upper cap is gone (`ruff`, `zensical`, `pyarrow`, `polars`, `duckdb`, and the
  exact `mypy` pin), and the two floors that could not be installed on a
  supported Python are fixed (`numpy>=1.26`, `pyarrow>=17.0`). The two caps
  that remain are deliberate: the `torch` range is the wheel ABI lane, and the
  `pixi` `pyarrow` cap is a conda-forge libabseil constraint tied to that lane.
- The native extension is built against nanobind 3 (`nanobind>=3.0.1`). No
  user-facing API change; the ABI lane and the Python/platform matrix are
  unchanged.
- `torchfits[docs]` now declares the documentation toolchain (`zensical`),
  which was previously pinned in CI and `pixi` only.
### Fixed
- `LogStretch` and `SqrtStretch` no longer silently truncate integer input:
  every stretch promotes integers to float32 instead of casting the stretched
  result back to the storage dtype (`LogStretch(uint16([0, 100, 1000]))` used to
  return `[0, 1, 1]`), and `ArcsinhStretch` no longer raises a cryptic dtype
  error. `float64` input stays `float64`.
- Every transform now accepts dict and `Payload` inputs, not just tensors.
  `Compose([BackgroundSubtract()])({"flux", "ivar", "mask"})` used to raise
  `AttributeError: 'dict' object has no attribute 'dtype'`, so the documented
  Dataset→transform path crashed for every statistics-based transform except
  `InterquantileScale`.
- Image `mask_hdu` companions are returned as boolean *validity* masks, and a
  FITS `DQ` bitfield must be decoded (`mask_is_dq=True`). Read verbatim, a
  perfectly clean all-zero DQ frame marked **every** pixel invalid, so every
  mask-aware statistic and `SigmaNormalize` collapsed to `NaN` — and
  `FitsImageDataset.from_bands()` attached `*_DQ` extensions as `mask`
  automatically, so the auto-discovery path hit this silently.
- A single-HDU companion now gets the same channel axis as its flux under
  `add_channel_dim=True` (`mask[0]` was row 0, not channel 0), matching what the
  staged-cutout reader already did.
- Degenerate groups (all-masked / all-NaN) no longer produce `-inf`/`1e30`
  artefacts: `MinMaxNormalize` returns NaN, `GlobalScalarNorm` falls back to an
  identity divisor, and the constant-image normalizers stay finite.
- Nonlinear transforms warn once per instance that a companion `ivar` is
  passed through unchanged (previously only the clippers did); the stretches
  can opt into exact delta-method propagation instead.

- data: Decode DQ companions into validity masks
- table: Key column assembly by index, not name
- io: Reject non-ASCII FITS text instead of silently rewriting it
- io: Accept os.PathLike across the whole read surface
- header: Resolve duplicate keywords first-wins, matching read_keys and astropy
- table: Accept os.PathLike in the table mutation API
- table: Support logical columns in where, without leaking pyarrow errors
- io: Keep header value types on every return_header route
- boundary: Generate the fixtures instead of relying on an ignored file
- native-stub: Let the stubgen child find torch's shared libraries
- native-stub: Make the committed stub invariant across the CI matrix
- http: Pin Python fetches to guard-time DNS; guard the ftp copy path
- where: Restore three-valued logic and quoted-literal parsing
- schema: Resolve case-variant FITS keywords in table schema derivation
- interop: Faithful to_astropy for dict and PathLike inputs
- cache: Remove deprecated no-op self-calls and phantom cache config
- transforms: Stretch inverse overflow and delta-method ivar contract
- transforms: Exact image RGB dtype control and Lupton astropy parity
- transforms: Exact mask promotion and typed mask errors
- transforms: MeshBackgroundSubtract honors min_tile_pixels and survives empty inputs
- transforms: Finite-only statistics with defined empty-input results
- transforms: FITS scale and TNULL helpers follow reader conventions
- transforms: Payload state is authoritative; instances are thread-safe
- data: Download cache never promotes corrupt or partial bodies
- data: Spectral-axis slicing, staged-cutout isolation, typed dataset contracts
- io: Typed stream probes and read_table error contracts
- io: LONGSTRN chain reassembly on open; negative meta caching
- io: Device validation forms and MPS downcast warnings
- io: Locked HDU registry unregister and exact-literal knob deprecations
- io: Checksum verification and quantize contract pins
- io: Write boundaries stop forging cards and corrupting compressed rewrites
- io: Typed Range errors and single-walk HTTP cutouts
- io: Strict batch errors and typed read-pipeline contracts
- table: Faithful interop — masked TNULL, width-1 scalars, byte vectors
- table: QuantizeError and KeyError contracts; mutation swallows narrowed
- table: Row selections honored and read dtype contracts pinned
- io: Orphan CONTINUE cards never fuse into unrelated keywords
- hdu: View contracts, SSRF-guarded reopens, refresh long strings
- hdu: LONGSTRN chains reassembled at HDUList.fromfile construction
- cli: Convert same-path refusal, setkey integrity, arith saturation
- cli: Diff/stats/table correctness and typed exit contracts
- cli: Internal errors exit 5 with traceback; exit matrix pinned
- cpp: Reader-cache eviction, BIT append, and table-type validation
- cpp: Table reader scale, TNULL, and moved-from correctness
- cpp: Hostile-NAXIS bounds, handle serialization, IEEE compressed reads
- cpp: NAXIS=0 empty-contract, batch attribution, and HDU-scan errors
- cpp: Byteswap tail UB, Release -O3, sanitizer propagation, CI bracket test
- tests: Isolate fixtures, mark wall-clock tests, pin saturation and signed zero
- bench: Report cutout throughput in payload bytes
- bench: Join highlight and cfitsio rows on the real case id
- examples: Keep demos in examples/output and check their claims
- docs: Record MPS warnings, cutout CRPIX, and null filters
- docs: Match the release lane table to torch_lanes.json
- packaging: Vendor CFITSIO in the conda package metadata
- docs: Drop the setkey --comment example
- io: Reassemble long strings on the slow header path
- table: Keep the first column when TTYPE is duplicated
- table: Exclude NaN rows from where comparisons
- changelog: Do not copy a stamped release back into Unreleased
### Fixed — found by adversarial probing
- `SigmaClip` and `AsymmetricSigmaClip` no longer detach the autograd graph:
  both wrapped their *return* in `torch.no_grad()`, so any pipeline containing
  a clip silently lost `requires_grad` while every normalizer kept it. The
  statistics stay constants; the kept pixels now carry gradient again.
- The declared `produces` state is now actually applied. Declaring it without
  using it left a payload labelled `stored` after it had been calibrated, so
  `FITSHeaderScale` could still be applied twice inside one pipeline. The
  header scalers now declare `produces = PHYSICAL`, `inverse()` accepts what
  `forward()` produced and restores the state it consumed, and
  `FITSHeaderNormalize` does not relabel data its float path left unchanged.
- `weighted=True` without an `ivar` is now an exact no-op. `estimate_background`
  (and the transform call sites) took the weighted inverted-CDF path with
  uniform weights, differing from the documented interpolated-median fallback
  by up to one order-statistic spacing.
- Flux-scaling transforms refuse `CONTINUUM_NORMALIZED` input with an
  actionable `DataStateError` instead of re-normalizing spectra whose common
  flux scale the normalization exists to preserve.
- `FitsSpectrumDataset(row=...)` on a rank-1 HDU raised nothing and returned a
  0-d tensor; it now raises a `ValueError` naming the offending shape.
- `FitsTableIterableDataset(as_batches=True)` now drops character/bit columns on
  **both** scanner paths (the `scan_torch` path used to hand back raw `uint8`
  byte matrices while the `where=` path dropped them).

### Fixed — IO, header and C++ engine audit
- Non-ASCII text is rejected on every write path instead of being silently
  rewritten. `sanitize_fits_string` stripped bytes outside 32..126, so
  `header={"UNI": "λ-cold"}` stored `'-cold'` — a different string, with no
  error. Header values, comments, `HISTORY`/`COMMENT` text, keywords, table
  column names and string column values now raise the way astropy does. The
  *read* path keeps the lenient stripping so files already carrying stray
  bytes stay openable.
- `os.PathLike` is accepted across the whole read surface, as
  `docs/api-core-io.md` already promised. `torchfits.write(Path(...))` worked
  while `torchfits.read(Path(...))` raised `ValueError: Path must be a string
  or list of strings`, and so did `read_header`, the skinny metadata probes,
  `table.read`, `read_batch`/`read_batch_info`, `open_subset_reader`,
  `open_table_reader`, `read_hdus`, `read_tensor`, the checksum helpers and
  `TableHDU.from_fits` — breaking any `for p in root.glob("*.fits")` loop.
- `hdu_name_cache` is now dropped when the underlying file changes. It was
  keyed by EXTNAME alone and, unlike its sibling caches, omitted from the
  stat-change invalidation, so after an out-of-band rewrite moved
  `EXTNAME="SCI"` from HDU 1 to HDU 2, `read(path, hdu="SCI")` returned the
  neighbouring array with no error or warning.
- `read_header()` resolves a repeated value keyword to its first occurrence,
  matching `read_keys()` (CFITSIO) and astropy. It previously let the last card
  win, so torchfits disagreed with itself on the same file; `HISTORY`/`COMMENT`
  keep last-wins, since those repeat by design.
- `read_header()` now exposes `COMMENT`/`HISTORY` consistently whether the
  header was built by the C++ reader or assembled in Python (the cards were in
  `.cards` but absent from the mapping for one of the two paths).
- The five skinny metadata APIs (`read_nrows`, `read_colnames`,
  `read_hdu_type`, `read_num_hdus`, `read_keys`) open and close a fresh CFITSIO
  handle per call and bypassed the shared metadata cache, so each cost 61-64 µs
  regardless of file size — more than `read_header` on a small header. Four of
  them now read from the cached open, dropping warm calls to ~2.4 µs and making
  them O(1) in header size instead of O(size).
- Removed the dead C++ shared-cache surface: 34 lines of empty `{}` bodies,
  their declarations, bindings and Python call sites (which had been calling
  no-ops after every mutation), plus an unreachable `_REMOVED_STUBS` branch.
  The live native state is `SharedReadMeta`.
- `TableHDU.filter()` / `TableHDURef.filter()` no longer raise
  `ArrowInvalid: only handle 1-dimensional arrays` on any table holding a
  packed string column (a `(rows, width)` uint8 tensor), a `TDIM` vector column
  or a variable-length column: the predicate table is now built only from
  columns Arrow can hold, string columns are decoded to text so
  `NAME == 'alpha'` matches, and the other kinds are sliced into the result
  without being offered to the predicate. Every filter on such a table
  previously failed, including a numeric predicate that never mentioned the
  offending column.
- `TableHDU.head()` truncates a zero-column table. Its row count comes from the
  header's `NAXIS2`, but with no column data to slice it returned `self`
  unchanged, so `head(2)` on a 6-row zero-column BINTABLE reported 6 rows. The
  derived header is now narrowed instead.
- `HDUList.validate()` reaches the data instead of only the header: it is
  `False` for an HDU whose handle has been detached (`close()` /
  `mark_closed()`) and for one whose file has gone away. Both guards previously
  failed open -- the `TensorHDU` branch was skipped by the very state
  `mark_closed()` creates, and a file-backed table HDU is a `TableHDURef`, which
  the `TableHDU`-only `isinstance` check never matched, so no table was ever
  validated and the method could not return `False`.
- `TableHDU.data[col]` keeps the 2-D `(rows, width)` shape of a packed uint8
  string column, matching `hdu[col]`, `to_tensor_dict()` and
  `get_string_column()`. A 1-character string column (`TFORM='1A'`) came back as
  bare char codes through `.data` and as `(N, 1)` through every other accessor.

### Fixed — transforms audit
- `mask=` is now a mask in the full sense: a pixel the caller masks is written
  back as `NaN` with `ivar = 0`, not merely excluded from the statistics that
  choose the transform's parameters. Every statistics-based transform already
  honoured the mask when computing parameters, but the output kept the
  masked-out value — on a 3×4 frame with a single masked-out `900.0` against a
  flat `10.0` sky, `RobustNormalize` returned `8.9e11`, `MinMaxNormalize`
  `8.9e7` and `SigmaNormalize` `9.0e14`, and the library's own
  `mask_from_nan` helper reported the pixel **valid**. `MinMaxNormalize`'s
  docstring already claimed masked pixels stay `NaN`; it now does. Callers
  that passed `mask=` to get a clean statistic and relied on getting the raw
  pixel back are affected, and the invalid mask is now applied in
  `PayloadView.replace()` so no transform can forget it. A transform that is a
  *declared* identity for its input (`FITSHeaderScale(bscale=1)`,
  `FITSHeaderNormalize` on a float header with `scale_floats=False`) still
  returns its input untouched, mask included: it has no statistic to
  exclude the pixel from and nothing was amplified.
- Non-finite input is blanked on the same path, so `SigmaClip(fill="mean")` no
  longer turns a `NaN`/`inf` pixel into the frame mean — the replacement value
  is applied to pixels that had data.
- `forward()` enforces the same double-scaling guard as `__call__`. The
  `produces` state stamp was applied in `FITSTransform.__call__` only, and
  `forward()` is public: `FITSHeaderScale(bscale=2.0).forward(...)` returned a
  payload still labelled `stored`, so a second `forward()` scaled the data
  again and raised nothing, while the identical call through `__call__` raised
  `DataStateError`. `FITSHeaderScale`, `FITSScaleColumns` and every
  `FITSHeaderNormalize` branch now stamp their own output.
- The one-time `ivar` warning no longer crashes on a `@dataclass` transform.
  The "already warned" set was a `weakref.WeakSet`, and a dataclass subclass
  gets `__eq__` and therefore `__hash__ = None`, so `TypeError: unhashable
  type` was raised exactly when the warning was due.

### Fixed — CLI audit
- `-j` / `--jobs` is no longer silently discarded. `run_file_jobs(..., torch_runtime=True)`
  called `torch.set_num_threads(1)` in the *serial* branch as well as in every
  worker, so the count `configure_torch_jobs` had just resolved was overwritten
  before any file was read: `transform -j 8 in.fits out.fits` and the default
  `-j 0` (all CPU cores) both ran single-threaded, on an 8-core machine. The cap
  now applies only under real fan-out, where it is the documented
  oversubscription guard — `-J 4` still runs 4 workers of 1 thread each, and
  `-j 3 -J 1` now runs 3 threads. This was a regression introduced by the lazy
  tensor-runtime refactor: the pre-refactor helper only capped inside the
  `workers > 1` path.
- A closed downstream pipe is exit `0` again, with no shutdown noise.
  `main()` mapped `BrokenPipeError` to `EXIT_OK`, but stdout is block-buffered
  whenever it is not a TTY, so the failing write happened during interpreter
  shutdown instead: `torchfits info <300 files> | head -1` exited **120** — an
  undocumented code, absent from the `docs/cli.md` table — and printed
  `Exception ignored on flushing sys.stdout: BrokenPipeError` to stderr.
  `main()` now flushes stdout itself and redirects the descriptor to
  `os.devnull` on a broken pipe, so the documented contract actually holds. A
  genuine error before the truncation point still wins: a missing input ahead
  of a `| head -1` still exits `3`.
- `convert --to png` band failures name the file and HDU. CFITSIO's bare
  `Could not move to HDU` — or a `Could not open FITS file` that says nothing
  about which band it was reading — is unattributable in a multi-band
  invocation; errors now read `g.fits: HDU 1: Could not move to HDU`.
- `table -n -1` is a usage error (exit `2`) instead of silently printing no
  preview rows. A negative count is nonsense input, and every other numeric CLI
  flag (`-j`, `-J`) already rejected it; `-n 0` remains the way to ask for the
  schema alone.

### Fixed — data layer audit
- `shuffle=` and `shuffle_buffer_size=` now change the order every epoch.
  Both were seeded from `seed + worker_id`, a constant for the life of the
  dataset, so epoch 2 replayed epoch 1 exactly: a `shuffle=True` run over
  8 files produced `[7, 0, 3, 4, 1, 5, 2, 6]` on every epoch, and
  `FitsStagedCutoutIterableDataset` cut the byte-identical patches out of the
  mosaic each epoch. Every `IterableDataset` in `torchfits.data` is affected
  (tensor, image, cube, spectrum, staged cutout, table). The seed now mixes
  a per-dataset epoch counter with `torch.initial_seed()` — measured to be
  the only way to cover all three DataLoader configurations, since
  `initial_seed()` is constant in the main process and under
  `persistent_workers=True` but varies in a freshly forked worker. **Two
  datasets built with the same `seed` still replay the same sequence of
  epochs**, so a run stays reproducible; a run that relied on a *fixed* order
  across epochs now gets a different one each epoch, which is what `shuffle`
  is for.
- `FitsTableDataset` reports the table's own row count. It derived the count
  from a materialised column, so a zero-column `BinTableHDU` reported 0 rows
  for a table torchfits itself opens as `num_rows = 6`, rejected a matching
  6-element label list, and handed back an empty dict for `ds[0]` instead of
  raising `IndexError`. Out-of-range indices now raise. A `where=` filter
  still counts the *filtered* rows.
- `FitsSpectrumDataset(column=..., ivar_hdu=...)` no longer silently discards
  the inverse variance. `mask_hdu=` and `wavelength_hdu=` were already
  rejected on that path with an actionable message; `ivar_hdu=` was missing
  from the guard and was dropped without a word, so a spectrum came back with
  no `ivar` at all. All three `*_hdu=` companions are now refused together.
- An `ivar_hdu=`/`mask_hdu=` companion whose shape does not match its flux is
  now rejected, naming both shapes and the HDU. `_stack_flux` already refuses
  flux channels of differing shapes, so a mask one row shorter than the image
  was the one mismatch that got through — and it did not fail, it
  *broadcast*: a `(3, 4)` flux with a `(1, 4)` mask silently applied that
  mask to all three rows.
- `label_key=` on a spectrum dataset is read from the spectrum's own HDU, as
  `FitsTensorDataset` already did. The spectrum datasets read HDU 0 whatever
  `hdu=` said, so a table spectrum at `hdu=1` with the label in the table's
  header raised a bare `KeyError`, and when the primary header happened to
  carry the key it produced the primary's value with no complaint.

### Fixed — tests audit
- The documented `--stdin` input path is now covered
  (`tests/test_cli_stdin_paths.py`). `docs/cli.md` lists `--stdin` in the
  global-flag table and shows the `find . -name "*.fits" | torchfits info
  --stdin -f jsonl` workflow, and seven subcommands route `args.stdin` into
  `cli.common.resolve_paths` (`info`, `header`, `verify`, `stats`, `table`,
  `probe`, `setkey`), but no test mentioned the flag at all. The tests pin
  the explicit read, the implicit read the flag table also promises ("stdin is
  also read implicitly when no paths are given and stdin is not a terminal"),
  blank and whitespace-padded lines being skipped, argv and stdin paths
  merging in order, an empty stdin staying the `no input paths (argv or
  stdin)` usage error, and a path that arrived over stdin still being named
  on failure. Every one of those seven mutations — ignore the flag, drop the
  implicit branch, stop filtering blanks — now fails a test.
- `Header.remove`'s missing-key contract is pinned on both sides. Only
  `remove_all` was varied; nothing tested a key that is absent, so neither the
  `KeyError` the default raises nor the no-op `ignore_missing=True` returns
  was checked — and `_write_helpers._merged_write_header` calls
  `remove(key, ignore_missing=True, remove_all=True)` for every overlay value
  card, including keys the base header never had. The no-op must also return
  before the version bump, so an overlay of entirely new keys leaves the
  header version untouched; a mutation that lets it fall through now fails.
- `make_loader(drop_last=False)` — the default — is now asserted. Only
  `drop_last=True` was tested, so a loader that dropped the short final batch
  unconditionally still passed. Over the 8-file fixture at `batch_size=6` the
  kept tail is asserted to be a real 2-row batch, not a repeat of the first.

### Fixed — tests audit, round 4
- The three HTTP environment variables `docs/architecture.md` documents are now
  tested (`tests/test_http_env_contracts.py`). `http_timeout` and
  `auth_headers` had **no test at all**; the only reference to any of them under
  `tests/` was one `monkeypatch.setenv("TORCHFITS_HTTP_TOKEN", ...)` that never
  asserted the header that came out. The documented precedence — the full
  `TORCHFITS_HTTP_AUTHORIZATION` value wins over `TORCHFITS_HTTP_TOKEN`, and a
  whitespace-only value counts as unset — is now pinned, along with the
  `Bearer ` prefix, the `120` default, and the *silent* fallback to the default
  on an unparseable timeout (so `TORCHFITS_HTTP_TIMEOUT=30s` still yields 120 s,
  now deliberately rather than by accident).
- The VLA predicates in `torchfits.fits_schema` are now tested
  (`tests/test_fits_schema.py`). `column_is_vla` had **no caller anywhere in the
  repository** — not in `src/`, not in `tests/`, not in `docs/` — and no test,
  while `selected_includes_vla` reimplemented the same question inline. Both are
  now pinned, *including the case that distinguishes them*: a duplicated TTYPE at
  two card indices, where the per-column form answers about the first card and
  the projection form answers whether any wanted card is a VLA.
- `selected_includes_vla`'s continue-scanning behaviour is now pinned. The cheap
  gate `_tform_might_be_vla` (any of `PpQq` anywhere in the TFORM) and the
  precise `parse_tform(...).vla` disagree for an incomplete or malformed code —
  a bare `P`, a `P` with no element type, a negative repeat — and that gap is
  the only place the scan-continues behaviour is observable. Folding the precise
  check into the gate's `if` reads as an equivalent simplification and silently
  stops at such a column, missing the real VLA column after it; that mutation
  survived the first pass and is now caught.
- `cache_subsystem_policy` is now tested (`tests/test_cache.py`). It is
  documented in `docs/api-core-io.md` and decides which `clear_file_cache`
  keywords a named subsystem clears, so it is the same *selective* contract
  TS-002 pinned one layer down — and it was equally unpinned. All four
  `fits_*` subsystems' flag sets, the `all` policy, the copy-not-reference
  return, the valid-name list in the `KeyError`, and the fact that
  `clear_cache_subsystem` does not forward the removed `handles` flag are now
  asserted. A regression making `fits_header_metadata` clear `data` too, or
  `fits_image_data` clear nothing, fails instead of passing silently.

### Fixed — tests audit, round 6
- The thread-local reader-cache **leak guard now runs on macOS**. It measured
  live malloc bytes through glibc's `mallinfo2().uordblks`, so on any
  non-glibc platform — every macOS developer and every macOS runner — it
  skipped, leaving the regression it guards (an LRU list that kept a ghost node
  per acquire) unverified exactly where a leak would be introduced. macOS has
  no `mallinfo`, but `malloc_zone_statistics(malloc_default_zone()).size_in_use`
  is the equivalent, and its noise was **measured before being trusted**: 12
  consecutive idle readings and 6 on each side of a 20,000-read run all agreed
  to the byte (spread 0). The reader cache itself is clean here — **−288 bytes
  over 20,500 cached reads** — so the guard passes rather than skips.
- A new **sensitivity control** pins the measurement itself: the probe must
  detect a synthetic 64-byte-per-iteration leak over 20,000 iterations, and
  that leak must exceed the threshold the leak test asserts. The old test had
  no such control, so a probe that quietly stopped working — a struct-layout
  change, a libc swap — would have left it reading a flat zero and passing
  without having measured anything. Verified load-bearing: replacing the probe
  with `lambda: 0` fails the control while the leak test still passes.
- Two platform-gated tests in `tests/test_package_isolation.py` **skipped by
  returning instead of skipping**, so on Linux and Windows CI they reported
  `PASSED` having run nothing — invisible, and the second one's docstring even
  promised a skip it never performed. They now raise `pytest.skip` with the
  reason, so the report says `SKIPPED (Homebrew libomp is a macOS-only
  scenario)` rather than a green tick that means nothing. Demonstrated: under a
  simulated `sys.platform == "linux"`, the bare-`return` shape reports `PASSED`
  and the  `pytest.skip` shape reports `SKIPPED (not darwin)`.
- The **fpack byte-identity gate now finds a usable fpack** instead of skipping
  all five compression-parity tests unless one hardcoded path exists.
  `test_compression_matrix.py` accepted only `$TORCHFITS_FPACK` or
  `/scratch/.tmp-sfabbro/opencode/cfitsio-full/fpack` — a path from the original
  author's HPC scratch space, present on no other machine including CI — so the
  suite's only check against CFITSIO's *own* compressor effectively ran for one
  person. The old rule ("a random system fpack may link an unpatched CFITSIO")
  conflated a *broken* fpack, which must be refused, with a *differently built*
  one, whose bytes are worth checking. Resolution is now
  `TORCHFITS_FPACK` → pinned reference → `PATH`, where a PATH hit must first
  pass a capability probe (compress a 4×4 file, exit 0, `.fz` produced); an
  fpack that cannot is "unavailable" rather than a collection error, and a stale
  `TORCHFITS_FPACK` degrades to a skip instead of raising at import. Once
  accepted it is held to the **same** byte-identity assertion. A build that
  refuses one algorithm now skips that case with fpack's exit status instead of
  disabling the other four: measured against Homebrew's fpack, RICE_1, GZIP_1,
  GZIP_2 and HCOMPRESS_1 are byte-identical while PLIO_1 exits 202
  (`required ZBITPIX compression keyword not found`), so four of the five run
  everywhere instead of none.
- Three torch-pin guards in `tests/test_check_torch_extra_pins.py` no longer
  skip on macOS. Both flavor pins in `pyproject.toml` are marked
  `sys_platform == 'linux'`, so `iter_torch_pins` yields nothing on macOS and
  the doc-drift and pin-resolution guards skipped with "no torch flavor pins
  resolve on this platform" — meaning **doc drift in the install index went
  unchecked on every developer machine**, and only Linux CI could catch it. The
  contracts those tests check are platform-independent:
  `documented_claims()` and `check_doc_drift()` never read `sys.platform`, only
  `iter_torch_pins`' marker evaluation does, and the module documents that
  platform patches are honoured precisely so this works. A new `linux_pins`
  fixture monkeypatches `sys.platform` to `"linux"` and **asserts** the pins
  resolved instead of skipping if they do not, so a future `pyproject.toml`
  that drops the markers fails loudly rather than quietly turning the three
  guards back into no-ops. Verified load-bearing on macOS: dropping the
  `cu129` index from `docs/install.md` makes both doc-drift tests fail.
- **`TORCHFITS_HTTP_TIMEOUT` now rejects unusable values loudly instead of
  failing somewhere unrelated.** The env-var audit recorded this as "a silent
  fallback, pinned as-is", and that judgement was wrong once the consequence was
  measured: only the *unparseable* case fell back. `0`, a negative, `nan` and
  `inf` were all passed straight through to `urlopen(timeout=...)`, and against
  a real socket each produced a raw low-level error that never mentions the
  variable responsible —

  | value | reached the transport as |
  |---|---|
  | `0` | `BlockingIOError: [Errno 36] Operation now in progress` (the socket goes non-blocking; the connection never establishes) |
  | `-5` | `ValueError: Timeout value out of range` |
  | `nan` | `ValueError: Invalid value NaN (not a number)` |
  | `inf` | `OverflowError: timestamp out of range for C PyTime_t` |
  | `30s` | nothing — a silent 120 s |

  A value that is not a positive, finite number now yields the default **and**
  raises a `UserWarning` naming the variable, the offending value and the
  fallback in force, so `errno 36` from a remote read is no longer a mystery.
  Unset and blank are still plain "use the default" and stay silent, as does a
  valid value.
- The five standalone C++ self-checks in `tests/cpp` are now runnable with one
  command (`tests/cpp/run_all.sh`) and their two library-linked build
  instructions no longer contain an unusable placeholder. Both link the
  torch-free core library and CFITSIO, and **neither is installed into the pixi
  environment** — they live in the per-build scratch tree
  (`.pixi/bld/torchfits/<hash>/bld`), whose hash changes on every rebuild — so
  their file headers carried a literal `<build>/libtorchfits_core.dylib` that
  had to be hand-substituted before the documented command could be used at
  all. The runner discovers the artifacts, writes the multi-extension MEF the
  two file-based checks need (neither documented how to obtain one), builds and
  runs all five, and exits non-zero if any fails. The three header-only checks
  keep their copy-pasteable commands, verified to work verbatim. All five pass
  on macOS arm64, and the runner is verified load-bearing in both directions: a
  build failure and a broken NEON byte-reverse in `internal_utils.h` each make
  it exit 1 with a `FAIL` naming the exact case and size.

### Fixed — C++ self-check audit
- Three of the five `tests/cpp` self-checks no longer verify themselves with
  `assert`. `test_bracket_detection.cpp`, `test_parallel_for_nesting.cpp` and
  `test_fitsreader_threads.cpp` expressed every check as `assert(...)`, and
  `-DNDEBUG` compiles all of that away: with `core::parallel.cpp`'s
  `run_chunks` reduced to doing nothing at all, the plain build aborted
  (SIGABRT) while the `-DNDEBUG` build printed "all checks passed" and exited
  0 at both 1 and 4 threads. They now count failures and return non-zero, the
  pattern the other two checks already used, and report the same 7 (1 thread)
  and 12 (4 threads) named failures with and without `-DNDEBUG`. This also
  removed a missing `<cassert>` include that had been resolving by accident
  through a transitive header in `test_fitsreader_threads.cpp`.
- `tests/cpp/run_all.sh` bounds every check with a timeout
  (`TORCHFITS_CPP_CHECK_TIMEOUT`, default 180 s) and reports a hung check as
  `TIMEOUT` / `FAIL (exit 124)`. The timeout is not defensive: the regression
  `test_parallel_for_nesting` guards *is* a deadlock, so a regression there
  makes the check hang forever — measured, with the worker-inline branch in
  `run_chunks` removed, it never returns at 2 or 4 threads and returns 0 at 1.
  The file's header credited this timeout to "the pytest driver", which does
  not exist for this file; the comment now says where the timeout really comes
  from.
- `test_parallel_for_nesting` now also checks that the outer chunks tile
  `[0, 4096)` exactly once, by recording each `(lo, hi)` it is handed and
  requiring them to partition the range. Its existing counter identity
  (`nested_items == nested_bodies * 64`) cannot see a skipped or repeated
  outer chunk — both counters move together — so the comment claiming it did
  was wrong about the assertion it described; it is now true.
- `test_bswap_helpers` compares against a reference that moves bytes one at a
  time instead of `__builtin_bswap*`, which is what the helpers' own scalar
  tails call (`internal::bswap_XX` is a one-line alias), so the old reference
  shared code with the code under test. The builtin does agree with a
  hand-written reversal — checked over all 65 536 16-bit values and 2M
  32/64-bit samples — so nothing was broken; the check no longer depends on
  that. It also pins the *direction*: a known big-endian FITS value must decode
  to its numeric value in host order, which a byte-permutation-vs-byte-
  permutation comparison cannot establish. That check caught two wrong
  expectations of my own while it was being written.
- `test_bswap_helpers` now runs every helper at four source/destination
  alignments, including misaligned ones. The SIMD branches use
  `vld1q`/`vst1q`/`loadu`/`storeu` and the tails use `memcpy`, all unaligned-
  safe, but nothing pinned that — and
  `SubsetReader::try_read_via_mmap` passes
  `pixel_base_ + (y * naxis1 + x1) * elem_bytes_`, so a cutout at an odd `x1`
  lands 2 or 6 bytes past a page-aligned base on a 2-byte frame (verified: an
  int16 cutout at `x1=1` and `x1=3` reads back exactly). All 19 sizes × 5
  helpers × 4 alignments pass, at `-O0`, `-O2` and `-O2 -DNDEBUG`.
- `test_fitsreader_threads` compares the whole shape vector instead of only its
  first axis, so a cursor race that returned the right width with another
  HDU's height is no longer invisible.
- `tests/test_byteswap.py` splits `CXX` with `shlex`, like
  `tests/test_security.py` already did for the same job. With
  `CXX="c++ -pipe"` — an ordinary value, since `CXX` may carry flags — the test
  raised `FileNotFoundError: 'c++ -pipe'` while its sibling passed.
  `tests/cpp/run_all.sh` had the same defect for the same reason and now reads
  `CXX` the same way (quoting inside the value is still respected, so a
  toolchain path with a space in it keeps working).
- The **x86 byte-swap branches had never been executed or checked by
  anything**. `internal_utils.h`'s five helpers have three `#if` branches —
  NEON, AVX2, SSSE3 — and a native build compiles exactly one. `CMakeLists.txt`
  adds `-mssse3` for `x86_64`, so the SSSE3 path is what every x86 user of the
  mmap cutout path actually runs, yet on this repository's arm64 machines (and
  in its CI) it is never compiled, let alone run: a wrong byte in those mask
  tables would be silent for x86 and invisible here. Two new checks, and the
  masks are correct:
  - `tests/test_simd_shuffle_masks.py` verifies the tables **statically**,
    against `_mm_shuffle_epi8`'s own semantics (`dst[i] = 0` if `mask[i] &
    0x80`, else `src[mask[i] & 0x0F]`, per 128-bit lane), plus the loop bounds
    and the BZERO add — 34 cases, no hardware needed, so the x86 branches are
    now covered on any platform. Load-bearing in seven directions: one wrong
    mask byte, a mask byte with the high bit set (which makes the instruction
    emit a **zero** byte), a lane-crossing index, a short vector loop, a
    dropped BZERO add, a mask transposed to the wrong element width, and a
    mask one byte short of its register.
  - `tests/cpp/run_all.sh` now also **runs** the SSSE3 branch on Apple Silicon
    by building an `x86_64` variant and executing it under Rosetta, and
    compiles the native bswap check with the project's own `-mssse3` baseline
    so an x86 host does not silently degrade to the scalar tail. Verified both
    ways: with an SSSE3 mask deliberately broken the native check passes (0
    failing cases) while the Rosetta one reports 97, and the whole runner exits
    1; `TORCHFITS_CPP_X86=0` skips it with a printed note. The AVX2 branch is
    left to the static check — no shipped build enables it (`-march=native` is
    deliberately avoided), and Rosetta here dies with `SIGILL` on a three-line
    program using one AVX2 intrinsic.
- The `parallel_for` deadlock regression now has **CI coverage**
  (`tests/test_cpp_self_checks.py`). `test_parallel_for_nesting.cpp` needs only a
  compiler and `core/parallel.cpp` — no built artifact — so it is compiled and
  run at pool sizes 1, 2, 4 and 8 as part of `pytest tests/ -q`, with a timeout
  because the regression it guards manifests as a **hang**, not a failure.
  Verified by removing the worker-inline branch: 3 failed, 1 passed, each
  failure naming the deadlock, and the bound is what turns a hang into a
  report. Two more tests keep the harness honest — a self-check cannot be
  dropped from `run_all.sh`'s case table without failing a test, and the two
  CFITSIO-linked checks must stay documented as manual. Those two are
  deliberately *not* wired into pytest: a skip that always fires reports
  coverage that does not exist (the TS-015 shape), and they need a standalone
  CFITSIO library, which the pixi environment does not install and a
  `pip install -e .` CI job links statically into the extension instead.
- `tests/cpp/README.md` now records that `test_bracket_detection.cpp`,
  `test_bswap_helpers.cpp` and `test_parallel_for_nesting.cpp` *are* compiled
  and run by `tests/test_security.py`, `tests/test_byteswap.py` and
  `tests/test_cpp_self_checks.py`. The audit had previously described the
  directory as unwired.

### Fixed — tests audit, round 7
- `TestTableReading` in `tests/test_table.py` now compares **values**. All five
  of its tests unpacked the fixture's `expected_data` and never used it —
  measured, `expected_data` appears on exactly the five assignment lines and in
  no assertion — so the whole class checked column presence, row counts and
  dtypes and nothing else. A reader returning correctly-shaped, correctly-typed
  arrays of zeros passed all five. Each test now compares against the column it
  wrote, including the streaming test (concatenated across batch boundaries) and
  the row-window test, which pins that `start_row` is 1-based. Evidence, with a
  control: hand the fixture back with `RA` reversed and the new assertions fail,
  while **the original assertions pass — 14 passed**.
- `test_compress_algorithm_gzip` now pins the algorithm it names. Its assertion
  was `zimage in {"T","TRUE","1"} or "GZIP" in zcmptype`, and `ZIMAGE` is `T`
  for *every* compressed image: measured against real CLI runs, a `RICE_1` and
  an `HCOMPRESS_1` output both satisfy the first clause, so the test accepted
  any compression at all. Both halves are asserted separately now, and the
  probe confirms a `RICE_1` file passes the old assertion and fails the new one.
- `test_blank_identity_promotes_to_nan_float32` now counts the NaN pixels
  instead of ruling out an all-NaN image. `assert not np.isnan(got).sum() ==
  got.size` parses as `not (X == Y)` — "not every pixel is NaN" — and its
  comment claimed "only the blank pixels are NaN". Measured: 46 NaN pixels out
  of 48 satisfies it; the assertion is now `== 2`, the two `BLANK` pixels the
  fixture sets, so a `nulval` that leaked onto ordinary data fails.

### Docs
- Transform docs cover the data-state contract, companion `ivar`/`mask`
  propagation, the mask helpers, weighted statistics and the new transforms;
  the integer-input note now matches the promotions above. Data docs cover the
  spectra companions, IFU spectral windows, table sharding / tensor-space
  streaming and band discovery.
- agents: Pixi-first, no ~/.local, Claude @AGENTS.md bridge
- Document the data-state contract, IVAR/mask and ML data features
- examples: Add an end-to-end ML training-loop example
- agents: Work on the fork's main, no feature branches
- changelog: Record the IO, header and C++ engine audit
- table: Record two deliberate divergences found in the engine review
- changelog: Record the PathLike fix for the table mutation API

- boundary: State the torch boundary, and correct the Arrow claim
### Performance
- boundary: Import torchfits.hdu and torchfits.io without torch
- Decode string windows without copying the full backing storage
- data: Make_loader and staged cutouts fetch each remote once

## [1.1.3] — 2026-09-09

### Fixed
- `FITSHeaderScale` / `FITSScaleColumns` keep integer inputs as float after
  `BSCALE*x+BZERO` (the unsigned convention `BZERO=32768` on int16 no
  longer wraps; fractional BSCALE is no longer truncated).
- `_median` / `_quantile` promote float16/bfloat16 to float32 (`torch.quantile`
  rejects half dtypes); `LogStretch` upcasts before `1 + a*x` so float16
  no longer overflows to inf.
- `SigmaClip(dim=(), fill="median")` fills with the global median instead
  of a per-element identity.
- `FITSHeaderNormalize` maps integer counts through float64 when the
  storage width is 32 bits or more, so values above `2**24` stay distinct.
- `table.read_torch(where=...)` with no `columns=` returns every column
  (matching Arrow `table.read`); `table.schema(columns=[...])` uses the
  requested field order.
- `FITSFile::read_subset` returns the image's true dtype for degenerate
  (zero-width/height) cutout boxes instead of always float32.
- `HDUList.fromfile` no longer reads every header twice (headers from
  the batch open are reused), and the `HDUInfo.header` binding preserves
  duplicate HISTORY/COMMENT cards.
- `TableHDURef.replace_column_file` mirrors the new `TFORM` into the
  in-memory header (matching `insert_column_file`).
- `Header` constructed from 3-tuples normalizes keys and scalar values
  like every other set path.
- HTTP Range HDU walks account for BINTABLE heap size (`PCOUNT`/`THEAP`),
  so cutouts past VLA columns or compressed images no longer fall back
  to a full-file download when the heap exceeds the scan cap.
- CLI `arith --hdu2` reads operand B once instead of per A-HDU;
  `arith`/`compress` `--out-dir` names strip CFITSIO `[section]`
  suffixes; `stats` upcasts integer images once.
- `bench_arrow_tables.py` uses the current `table.read` signature.
### Changed
- Derived `TableHDU`s (`filter`, `select`, `head`, `add_column`, ...) own
  a copy of the source header, so later mutations of the derived header
  no longer leak into the parent.
- CLI `transform` preserves safe header keys (WCS, EXTNAME, etc.) on
  float outputs instead of dropping the whole header; only scaling
  keywords (`BSCALE`/`BZERO`/`BLANK`, `DATAMIN`/`DATAMAX`) and checksum
  stamps are dropped.
- CLI `compress` resolves IO pairs through the shared batch resolver.
- `get_cache_stats` no longer reports counters that were never updated.
- Removed the unreachable `tzero is None` branch in `fits_schema`.

### Added
- `InterquantileScale` transform for robust scaling by interquantile range;
  `FitsStagedCutoutIterableDataset` can load companion HDUs (e.g. label or
  mask maps) alongside the image HDU (#241).

### Docs
- Documentation rewritten in plain user-facing language: internal tracking
  codes and process jargon removed, benchmark and install pages glossed.

## [1.1.1] — 2026-08-29

### Fixed
- `table.read_torch(where=)` drops TNULL sentinels and uses range-safe
  compares, matching Arrow `table.read`.
- Robust-quantize BLANK pixels are NaN on `torchfits.read`, including
  uncompressed scaled images. Native IEEE float/double HDUs
  keep Inf and signed zero (`fits_read_img` nulval is only for BLANK
  and compressed tiles; CFITSIO `fnan` would otherwise replace them).
- `torchfits copy` is a byte copy (`shutil.copy2`); same-path I/O is
  refused; `HDUList.write` to an existing path uses tempfile+replace.
- `TableHDURef.head` composes an existing row window.
- Header cache hits clone `Header` from cards; `raw_scale` reaches the
  fallback reader.
- `replace_hdu` strips stale `Z*` cards; checksum rewrites restamp when
  the input had stamps; `verify_checksums` reports `present`.
- Table `.fits.gz` / `.zip` refuse the buffered pread path; TSBYTE mmap
  matches CFITSIO; TFORM repeat overflow raises.
- CLI: unsigned `diff` min/max, Ctrl-C exit 130, JSON without NaN.
- HTTP Range cutouts of integer images with `BLANK` fall back to CFITSIO
  so missing pixels are NaN rather than the sentinel code.
- Writing an already-decoded float with a copied integer header drops
  `BSCALE`/`BZERO`/`BLANK` so CFITSIO does not scale twice (CLI cutout
  and arith).
- Quantized table `TNULL` values are Arrow nulls, so `where="V IS NULL"`
  matches. Native float NaN without `TNULLn` is unchanged.
- Import sets `KMP_DUPLICATE_LIB_OK` when unset (macOS libomp).
- build: Portable sha256 verification in vendor.sh (macOS runners)
- changelog: Rank final tags above prereleases in latest_tag
- build: Select sha256 tooling by OS, not binary presence

- Completed the 1.1 correctness review for silent NaN/TNULL/copy bugs (#237)
- Completed the 1.1.1 release-readiness review (19 tracked findings closed)

### Changed
- `torchfits._cpp` no longer re-exports undocumented `_C` names.
- Root `to_astropy(path)` delegates to `table.to_astropy`.
- `read(hdu=[...], mmap=False)` and path-list batch honor mmap.
- `quantize=` on `HDUList` + `compress=` raises instead of being ignored.
- Linux wheels install `bzip2-devel` in manylinux so `HAS_BZIP2` matches
  macOS.
- Sanitizer CI uses the pixi `test` env (not uv) and quotes cmake
  define flags so bash does not split on `;`.

### Docs
- Benchmark headline cites git-mirrored `exhaustive_*_20260807_013736`.
  GPU copy is host-decode then `device=`. Complex columns are Partial on
  Arrow. Release runbook uses OIDC, not a PyPI token. Windows is
  unsupported. `CacheConfig.max_files` is a no-op.

### Added
- 1.1.1 identity-check test suite and example-gallery additions

## [1.1.0] — 2026-08-26

Feature + correctness release on the same 2.13 torch ABI lane. New
capabilities: checksum-stamped writes, GIL-free hot reads, clean
truncated-file errors, high-fidelity Astropy interop, memory-bounded
streaming filters, multiprocess-safe remote downloads, auto-adaptive
FITS RGB, and native whole-file `.bz2` FITS reads — plus a
major-release-readiness review that cleared a round of silent-corruption,
thread-safety, and supply-chain fixes.

### Features

- **Native whole-file `.bz2` FITS reads** on builds with bzip2 support
  (capability flag `torchfits._C.HAS_BZIP2`; vendored CFITSIO links
  libbz2 from the conda prefix *or* the system): every reader entry point
  decompresses transparently, while direct-I/O fast paths automatically
  route through CFITSIO. Writing a `.bz2`-named output is rejected —
  CFITSIO would silently create an uncompressed file; use
  `compress="BZIP2_1"` for tile compression instead.

- **`transforms.rgb(*bands)`** auto-adaptive RGB from 1–7 aligned filters
  (shortest wavelength first): scarlet-style mix, per-band sky-median
  subtract + MAD equalize unless `calibrated=True` / `zeropoints=`,
  fitspng-style MAD stretch, coupled asinh, saturation, and sRGB.
  NaN mosaic holes stay black (they are not treated as sky).
  Scene classification is blow-out-safe: a bright extended object (planet
  disk, galaxy core) whose post-equalize p90 towers above the noise is
  stretched against its own p90 instead of the faint-feature anchor, so
  extended targets never clip while deep/star fields keep their stretch.
  `lupton_rgb` stays the Astropy-parity 3-band mapping (reddest first).
- **`write(..., checksum=True)`** stamps CFITSIO `DATASUM`/`CHECKSUM`
  keywords on every HDU at write time (all payload types, compressed
  included); verify later with `torchfits.verify_checksums`.
- **Multithreaded reads scale**: hot C++ paths (`read_full`,
  `read_full_numpy`, header/shape/HDU-count probes) now release the GIL
  for open + I/O, so DataLoader threads no longer serialize behind one
  Python thread during disk or network access.
- **Truncated files raise instead of crashing**: mmap table reads,
  filtered scans, and row updates validate the header-claimed extent
  against the real file size and raise a clear "truncated" error rather
  than SIGBUS-killing the interpreter.
- **Multi-chunk buffered reads are correct again**: tables whose rows
  span more than one 16 MiB scratch chunk (payload > ~16 MB with caching
  disabled) could read from an unsized staging buffer after a prefetch
  rewrite — slow, garbage results on wide/mixed projections. Buffer
  rotation now happens only while prefetching; regression tests pin
  >16 MB reads against astropy.
- **High-fidelity `table.to_astropy()`**: TNULL-bearing columns become
  real `MaskedColumn`s (no more object-dtype degradation), TUNIT maps to
  `.unit`, and fixed-size vector columns keep their `(N, repeat)` shape.
- **Memory-bounded streaming filters**: `scan(..., where=...)` now
  evaluates predicates per batch as rows stream past — peak RAM tracks
  `batch_size`, not table size. Hidden predicate columns are projected
  out after filtering, and a fully filtered-out scan still yields one
  typed empty batch.
- **Multiprocess-safe remote downloads**: concurrent DataLoader workers
  sharing a cache directory are serialized by an OS file lock (exactly
  one fetch per URL), interrupted transfers resume via `If-Range`
  validators (stale partials restart cleanly instead of producing hybrid
  files), and servers without `Content-Length` trigger an explicit
  completeness warning.

### Changed

- **`torchfits convert` PNG default is auto RGB** (`--recipe auto`): 1–7
  files, blue→red order, `--brightness` / `--saturation` /
  `--calibrated` / `--zeropoints`. `--recipe lupton` keeps the previous
  3-band reddest-first `--q` / `--stretch` mapping.
- **Scalar-column shapes are now rank-1 everywhere**: FITS repeat==1
  columns read as ``(N,)`` through every path — `hdul[n].data[col]`,
  `hdul[n][col]`, `TableHDURef`, `read_torch`, `iter_rows`,
  `to_tensor_dict`, streaming chunks, and buffered/mmap reads alike.
  Vector columns (repeat>1) keep their shapes; packed string columns are
  never squeezed. Completes the accessor-only squeeze shipped in #234.
- `read_torch(start_row=..., where=...)` now filters **within** the row
  window, matching `table.read(row_slice=..., where=...)` and the Arrow
  engine. Previously the two entry points returned different populations
  for the same query.
- WHERE negation follows SQL three-valued logic on every engine:
  `NOT (X == 5)`, `X NOT IN (...)`, and `X NOT BETWEEN ...` exclude NULL
  rows exactly like their unnegated counterparts.
- TNULL sentinel values can no longer satisfy numeric predicates on the
  C++ pushdown / torch-mask paths; sentinel-only matches are excluded so
  all engines return identical rows for the same query.
- Batch fast paths (`read(list_of_paths)`, list-of-HDUs) honor
  `fp16` / `bf16` / `raw_scale` instead of silently returning raw data.

- `torchfits.cpp` is deprecated in favor of private `torchfits._cpp`;
  every attribute access warns. No-op native cache stubs (`configure_cache`,
  `get_cache_size`, `clear_file_cache`) left the public surface.
- `TORCHFITS_CFITSIO_CACHE_MB/_FILES` env vars removed; they configured a
  native cache that no longer exists. `cache.configure_cache()` /
  `CacheManager.configure_cpp_cache()` remain as documented no-ops emitting
  DeprecationWarning for one cycle.
- `write_tensor()` accepts `checksum=` for parity with `write()`.
- `torchfits.open()` accepts only `mode="r"`; in-place update modes are
  rejected with an actionable error.
- `TensorHDU.chunks()` is implemented: lazy row-band slabs equal to slices
  of `to_tensor()`. It previously called a binding that never existed and
  always raised AttributeError.
- `DataView.dtype` reports convention dtypes (uint16/uint32/int8) matching
  reader output rather than raw storage BITPIX.
- TableHDU schema caches hold strong header refs so GC id-reuse cannot
  serve stale schemas; `select()` rejects unknown columns and
  `head()` validates its argument.
- Vendored CFITSIO fetches are sha256-pinned and verified; unpinned tags
  fail closed unless `TORCHFITS_VENDOR_ALLOW_UNPINNED=1`; "latest"
  resolution removed. Conda/pixi builds compile the same pinned+patched
  CFITSIO as the wheels — PLIO buffer fix + BZIP2_1 on every channel.

### Fixed

- **BIT (`'X'`) table columns wrote corrupted bits** when `repeat % 8 != 0`
  (e.g. `'12X'`): both the initial writer and the buffered row-update
  writer passed flat element runs to `fits_write_col(TBIT)`, which maps
  them onto raw data-unit bits ignoring per-row padding. Every write call
  is now confined to a single row; verified byte-exact against astropy for
  8X/12X/16X/23X across all three write paths.
- Buffered table reads no longer leak a closed fd after a transient pread
  failure — a recycled descriptor could later serve an unrelated file's
  bytes as table data.
- Cached reads are isolated from caller mutation: in-place edits of a
  returned tensor no longer poison subsequent identical reads (the default
  read cache stores/hands out private copies).
- `replace_hdu` header preservation drops stale `BSCALE` / `BZERO` /
  `DATASUM` / `CHECKSUM`; grafting an unsigned-convention BZERO onto new
  float data previously made every reader misinterpret the replacement.
- `SigmaClip`: a single NaN no longer wipes the whole frame (valid mask is
  seeded from finite values even without a user mask); median fill of
  fully-masked groups yields 0 instead of NaN.
- `GlobalScalarNorm`: negative statistics divide sign-preservingly instead
  of exploding to ~1e30 scales; `inverse()` round-trips exactly; NaN no
  longer poisons `mean` / `rms` statistics.
- WHERE string literals survive parsing: `NAME == 'AT&T'` matches `AT&T`
  (not `AT AND T`), FITS doubled-quote escapes work (`'O''NEIL'`), and
  unterminated quotes raise instead of mangling.
- Header parser: LONGSTRN chains keep assembling correctly when any card
  carries a comment, and ESO `HIERARCH` cards parse to typed values under
  their full keyword (`ESO TEL AMBI TEMP -> 12.5`).
- Thread-local HDU metadata cache rotates its generation id when an
  out-of-band file replacement is detected, so stale shape/dtype/scale
  cannot be paired with new bytes on the mmap path.
- Unsigned-convention detection tolerates floating-point imprecision in
  stored `BZERO`/`TZERO` uniformly for images *and* tables (a file whose
  offset was serialized as 32767.999… now reads as uint16/uint32 on every
  path).

- **Compressed-image null pixels decode as NaN** (was silent 0): the null
  probe targeted a CFITSIO symbol that exists in no upstream release.
  ZBLANK is probed directly and float CompImage reads always pass NaN
  nulval. Regression suite vs astropy: tests/test_compressed_nulls.py.
- **`torchfits arith` no longer wraps/truncates integer images**: ops run
  in int64/float64 with saturating cast-back plus warning; `--dtype`
  overrides; div produces floats under `auto`.
- **`torchfits stats` works on unsigned-convention images**: min/max run
  after upcast instead of raising on missing uint reduction kernels.
- **`quantize="robust"` maps NaN/Inf to BLANK sentinel codes** with the
  keyword written, instead of packing non-finite values into valid codes
  that dequantize as real data.
- **WHERE float equality is engine-independent**; **schema() stops
  lying about complex columns**; **row windows keep VLA/string
  columns aligned**; **scan(mmap=True) handles scaled tables and
  ASCII HDUs**; chunk buffers zero-initialized; read_batch
  warnings document skip semantics; interop kwargs no longer leak to
  pandas; FITS numeric parsing accepts D-exponents / rejects `1_0`.
- **Transforms are functional** (no caller-tensor mutation via `.to()`
  aliasing); medians interpolate like numpy/astropy;
  SigmaClip/AsymmetricSigmaClip gain `fill="nan"`.
- **Random Groups images fail loudly** instead of decoding garbage.
- **`torchfits diff` treats NaN == NaN**: byte-identical files containing
  NaN pixels compare clean instead of reporting spurious differences.
- `torchfits --transform -J` fan-out builds one transform instance per
  worker file, so stateful transforms never share `_last_state` across
  threads.
- **Thread-safety**: TensorHDU reads use private per-call handles; shared
  TableReader instances serialize I/O. Stale-cache windows after
  header mutations closed by invalidating SharedReadMeta/readers in the
  header-card/key/checksum writers.

<!-- Per-commit one-liners below are maintained by
     scripts/update_changelog.py; do not delete without checking
     `pixi run changelog-check`. -->
- packaging: SPDX license expression for license-files; boundary test tracks _cpp move
- tables: Engine-aligned WHERE floats, honest complex schema, aligned windows, streaming fallbacks
- cli: Integer-safe arith, uint stats, NaN-aware diff, per-worker transforms

### Security

- `read_hdus` enforces the same SSRF/path guards as every other entry
  point (loopback/link-local/private targets were reachable before).
- HTTP credentials (`TORCHFITS_HTTP_TOKEN` / `TORCHFITS_HTTP_AUTHORIZATION`)
  are stripped when a redirect crosses origins; kept only for same-origin
  hops and plain http->https upgrades.

### Performance

- Policy/meta caches (`image_meta`, cold-nommap, auto-mmap, hdu-type) are
  validated against the file's stat signature, eliminating stale dispatch
  after in-place rewrites. Measured cost of the validation: ~1.6 us per
  `os.stat` on this host.
- Cached reads now hand out private copies, so callers can mutate results.
  Measured on a 64 MiB float32 image (local NVMe): uncached read 46.5 ms
  (~1.4 GB/s) vs warm cache hit 45.3 ms including the isolation copy — a
  hit still beats the I/O it replaces, and repeated table hits stay
  sub-millisecond. If a cached-hot workload regresses measurably for you,
  `read(..., cache_capacity=0)` restores v1.0 semantics at the cost of
  re-reading.
- Full CPU + CUDA exhaustive benchmark re-runs on Linux CANFAR headless (published
  CSVs `exhaustive_cpu_20260807_013736` / `exhaustive_cuda_20260807_013736`
  under `docs/assets/bench/`; lab profile, mmap on+off matrix, 3057 + 4315
  rows): **100% of significant image comparisons won on both hosts**.
  Exactly one case family remains where a peer leads: narrow-table full
  reads with `mmap=False` trail fitsio by 21-36% on CPU and 8% on CUDA
  (buffered path stages whole rows; single-pass decode lands in 1.2).
  Image HCOMPRESS lags vs fitsio are sub-1.03× noise. A double-buffered
  chunk prefetch now overlaps the buffered path's I/O with decode for
  payloads >= 64 MB (gated from an earlier 4 MB threshold after CANFAR
  A/B showed thread handoff regressing warm-cache 13 MB tables).
  Benchmark harness fairness fixes: device synchronization on GPU timings,
  seeded interleaving order, medians over means, and cache-symmetric peer
  comparisons.
- Multi-HDU writes flush process-global caches once per operation instead
  of twice per HDU.
- BIT (`'X'`) writes now issue one `fits_write_col` call per row; only
  tables containing bit columns pay for the extra calls.

### Added

- Table mutations warn on silent value loss: float payloads into integer
  columns (truncation/non-finite counts), out-of-range integers, non-ASCII
  characters dropped from string columns, and over-width string clipping.

<!-- Auto-maintained one-liners (scripts/update_changelog.py). -->
- io: Native whole-file .bz2 FITS reads on bzip2-capable builds

### Dependencies

- Vendored CFITSIO updated to **4.7.0** for wheel, source, conda and pixi
  builds alike (`extern/VERSIONS.txt`, sha256-pinned): every channel ships
  the PLIO buffer fix and BZIP2_1 support (see [architecture](architecture.md)).

### Docs

- Scalar-column shape contract documented; architecture note reconciled
  with the wheel-vs-conda CFITSIO split.
- Corrected inaccurate benchmark and compatibility claims in the docs and
  toned down unverifiable ones
- changelog: Versionless Unreleased + generator tooling; refresh roadmap
- Changelog entries curated under Unreleased; changelog-check green
- io: Document BZIP2_1 availability, lossiness and interop caveat in write()

## [1.0.0] — 2026-08-09

Version cut on the 2.13 torch ABI lane: buffered table reads through a
thread-local reader cache (process-wide eviction, stat-identity stale guard),
single-open insert/update of table rows, and torch-mask predicate
materialization. Final pre-release checks complete; awaiting collaborator
docs/usability testing before the tag.

### Added

- `torchfits.to_astropy()`: Direct conversion of PyTorch tensor dictionary
  structures to Astropy `Table` instances via zero-copy Arrow intermediate buffers.
- Streaming ML Datasets: `FitsCubeIterableDataset`,
  `FitsSpectrumIterableDataset`, and `FitsStagedCutoutIterableDataset`
  with rank and world size sharding for distributed PyTorch training.
- MegaCam cosmic-ray denoise example (`example_megacam_cr_denoise.py`):
  Noise2Noise on real dark/bias calibration twins (zero-field N2N), with
  self-normalizing pair transforms, held-out CCD evaluation, and honest
  transfer metrics (CR suppression, star fluxes, background, noise-injection test).
  Full rationale and results: [Denoise pipeline](denoise-pipeline.md).
- `scripts/fetch_cfht_calib_frames.sh`: idempotent download of CFHT MegaCam
  darks/biases from the CADC data service.
- `scripts/canfar_denoise_incontainer.sh` + `scripts/launch_canfar_denoise.sh`:
  headless Skaha job running the denoise example on a CUDA GPU (defaults to
  the 4-epoch setting; longer fixed-LR runs diverge — see the pipeline page).
- Denoise example writes a before/after gallery figure
  (`examples/output/megacam_cr_denoise_dark.png`, rendered in the ML guide
  and the pipeline page).
- Exemplary science-pipeline benchmark (`bench_science_pipeline.py`: robust
  sigma-clipped coadd + ML cutout serving, torchfits vs astropy) with
  per-stage code-lines rows (`bench_contract.code_lines`); `bench-denoise`
  pixi task for the MegaCam CR-cleaning benchmark.

### Performance

- Cached table reader: `read_table`/`read_batch` reuse an open CFITSIO handle
  per thread instead of open/read/close per call; benchmark-host side effects
  drop from ~5861 to ~750 minor faults per read after warmup; isolated
  `narrow_1000000` `read_full` window 17.8 ms -> 6.7 ms (lab window,
  `exhaustive_cpu_20260807_082144_reader_cache`).
- `exhaustive_cpu_20260807_082931_reader_cache` run: fitstable
  `read_full` ratio vs astropy 1.365x -> 1.072x; `predicate_filter` on
  `narrow_1000000` 16.60 -> 11.01 ms (1.21x -> 1.64x vs astropy); selective
  10.68 -> 9.33 ms (1.50x -> 1.71x) via torch-gather `where=` (mmap off, lab).
  The `_082144` / `_082931` run CSVs were lost in a local workspace reset
  (noted in `benchmarks.md`); the numbers survive in the benchmarks page's
  generated tables, and the reader-cache direction was spot-verified on
  current HEAD 2026-08-09 (`narrow_1000000` `read_full`: torchfits 7.9 ms vs
  astropy 16.5 ms, mmap on).
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
- Tile compression: reject unsupported floating-point payloads on PLIO_1 writes with a clean ValueError.
- Table mutation & schema: strict validation for VLA and object dictionary-table columns.
- Stats transforms accept integer dtypes (BZERO-scaled `read_subset` results
  come back as UInt16, which torch cannot reduce): helpers upcast to float32
  (int64 keeps precision as float64), masked fills use dtype-safe sentinels,
  and `SigmaClip` promotes integer inputs instead of raising.

### Dependencies

- Vendored CFITSIO at the 1.0.0 tag: **4.6.4** (`extern/VERSIONS.txt`).
  (An earlier revision of this entry claimed 4.7.0; that bump landed after
  the tag and is recorded under [Unreleased] below.)

### Packaging

- Linux wheels now cover **x86_64 and aarch64** for CPython **3.10–3.14**
  (cibuildwheel 4.1; GHA `ubuntu-24.04-arm` for aarch64). macOS arm64 is
  unchanged; run `bash scripts/cibuildwheel.sh` on a Mac to build locally.
- PyPI no longer receives an sdist — `pip install` cannot fall back to a
  source compile (the rc5 trap on Ubuntu 26.04 / Python 3.14).
- Wheel cmake args no longer point `CUDA_TOOLKIT_ROOT_DIR` at conda.
  CUDA torch at runtime is the same CPU-linked wheel; verify on CANFAR with
  `scripts/verify_wheel_cuda_canfar.sh`.

### Docs

- Benchmark tables refreshed from the 2026-08-07 exhaustive runs (CPU + CUDA);
  `docs/assets/bench/` mirrors the surviving 2026-08-07 CPU/CUDA run CSVs.
- Denoise pipeline page with honest dark-vs-bias results and stated
  limitations; MegaCam CR-cleaning section in the ML guide; examples index
  row.

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
- Docs: full docs↔code sync review — one-line pinned installs in quickstart /
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
  wheel bundle (`scripts/launch_canfar_matrix_grid.sh`), 41-leg grid run.

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
- Bench docs: multi-host GPU/CPU results from the CANFAR matrix grid.
- Bench docs: rc5 re-run snapshot — CANFAR CPU
  `exhaustive_cpu_20260806_012620`, CUDA `exhaustive_cuda_20260806_012651`,
  local CPU `exhaustive_cpu_20260806_022603`. CUDA 100% fits win rate
  (smart/specialized), fitstable ≥98.9%; only residual lags are
  table `read_full` / predicate rows (≤1.15×, fitsio/astropy) and
  HCOMPRESS_1 (≤1.03×).
- Bench labeling: the per-platform results table derives the platform from the benchmark
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

Fourth release candidate for collaborator testing after prep / deep-review cleanup.

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
- July 2026 CANFAR/local benchmark refresh: MPS `exhaustive_mps_20260719_143706`,
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

Third release candidate for collaborator testing.

### Docs
- Readability pass on user-facing pages: rc honesty, corrected migration threading
  (private CFITSIO handles since rc2), API notes for EXTNAME / 3D `read_subset`,
  cache vs disk-cache / `make_loader` layering.
- Examples gallery: MaNGA LOGCUBE (`example_manga_logcube.py`), Lupton RGB from
  real SDSS g/r/i (`example_lupton_rgb_sdss.py`, stdlib `bz2` inflate before
  read), CFHT MegaCam MEF cutouts (`example_megacam_mef_cutouts.py`).
- Fill API / cache / loader doc gaps: `TORCHFITS_CACHE_DIR` vs in-process handle
  cache, when `optimize_cache` no-ops on table datasets, `make_loader` vs plain
  `DataLoader`.

### Developer workflow
- Added `AGENTS.md` + `JULES.md` agent configuration: weekly automation
  retargets to bug/perf-only passes; ledger in `.cursor/jules-ledger.md`;
  out-of-scope cosmetic PRs are out of scope.

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
leftover API/docs/CLI, and cleanup. SemVer `1.0.0` still waits for extended testing.

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
- Automation integrations: probe SSRF (#216), hoist inner classes (#214), HDU HTML a11y
  (#213/#219).

### Benchmarks / tests
- ML loader: on-disk `size_mb` for compressed cases; pin compressed torchfits to
  `hdu=1` (matches fitsio `ext=1`); missing numeric CSV fields use `None`.
- GPU transports: pass `quick=` into table `_build_cases`.
- Table filter tests assert exact fixture row counts.
- Concurrent same-file image/table read smoke tests.
- Benchmarks: new local MPS run `exhaustive_mps_20260718_180230`; Linux CPU/CUDA
  hosts remain the rc1 benchmark runs until CANFAR is re-driven against this tag.

### Docs / transforms
- Mermaid diagrams in architecture (zensical superfences).
- Architecture: per-read handles, deliberate skip of CFITSIO iterator/`where`.
- Roadmap: CFITSIO 1.1 leftovers + permanent design choices from the design review.
- Advanced transforms frozen for 1.0; Lupton RGB wrapper in `transforms.lupton_rgb`;
  richer multi-band RGB deferred to 1.1.
- MegaCam cutout bench: ZNAXIS-aware HDU discovery; peer fitsio ranking;
  materialize-once baseline; payload-based throughput (not whole-file MB/s).
- Docs landing: lean browse grid; nav uses mark-only logo (`torchfits-logo-mark.png`).
- Vendored CFITSIO docs pin **4.6.4**; note `fits_iterate_data` intentionally unused.
- cfitsio-direct: Rice + optional MegaCam `cutout_rep` jobs.

## [1.0.0rc1] — 2026-07-17

Release candidate for the 1.0 API. SemVer `1.0.0` waits for post-rc extended testing; do
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

- Multi-host benchmark results (from b1 same-day refresh, still current for rc1):
  `exhaustive_mps_20260717_040150`, `exhaustive_cpu_20260717_040146`,
  `exhaustive_cuda_20260717_042840`.
- Local release-suite (`20260717_212321`, Mac MPS, mmap matrix, `--no-gpu`):
  2,825 rows, 3 deficit rows, exit 0. No domain failures.

### Validation

848+ tests; mypy / ruff clean; docs integrity; examples runner REQUIRED green;
`bash scripts/clean_install_smoke.sh`; HTTP probe fixture.

## [1.0b1] — 2026-07-17

Beta freeze of the public FITS → tensor / dataframe story. Not a SemVer 1.0.0
API freeze (rc line followed for extended testing + blockers).

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
- Multi-host benchmark refresh (`exhaustive_mps_20260717_040150`,
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
- Deficit table: images any lag above ε; Arrow tables allow ≤1.05×; fitsio
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
- `.cursor/skills/release-api-freeze-review/` — pre-tag API/feature freeze review workflow.
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
- Reviewed C++ codebase for potential memory leaks, redundant hardware heuristics, and API bounds.

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
[Unreleased]: https://github.com/astroai/torchfits/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/astroai/torchfits/compare/v1.1.3...v1.2.0
[1.1.3]: https://github.com/astroai/torchfits/compare/v1.1.1...v1.1.3
[1.1.1]: https://github.com/astroai/torchfits/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/astroai/torchfits/compare/v1.0.0...v1.1.0
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
