# R6 slice B — `_hdu/` views: `table_hdu.py`, `table_hdu_ref.py`, `tensor_hdu.py`, `dataview.py`

Routed queue (r5c-09, r5c-15) landed first with failing-first evidence; then a full
rubric review of all 4 files. Cache policy per context spec: measured
`TableHDURef.columns` recompute = **23.1 µs/call** on a 100-card header
(1000-iteration `perf_counter_ns` loop) — above the 10 µs threshold but on a
metadata walk (repr / `data.keys()` / `insert_column_file` once), not a hot
path, so the spec's primary fix (drop the cache) applies; the `Header._mutation_seq`
fallback is NOT needed and `Header._version` (already bumped on every mutation
path; pinned by `tests/test_header_versioning.py`, confirmed with slice A) plays
that role where caches remain. No `header.py` edits; no cross-slice touch.

## Findings

| ID | Sev | Class | Files:symbols | Description (with repro) | Fix & validation | Status |
|---|---|---|---|---|---|---|
| r6b-01 | MAJOR | 5 error contract | `_hdu/table_hdu_ref.py:to_arrow/scan_arrow/reader_arrow` | **r5c-09 routed.** `hdul[1].to_arrow(columns=["A"])` raised `TypeError: torchfits._table.read.read() got multiple values for keyword argument 'columns'` — the methods forward `hdu=`/`columns=`/`row_slice=` explicitly then `**kwargs` collides with a deep, confusing message. Same for `scan_arrow`/`reader_arrow` and for `hdu=`/`row_slice=`. | REJECT-collision per contract: module-level `_reject_forwarded_kwargs(method, kwargs)` raises `TypeError: "<method>() got a duplicate argument 'columns': columns is forwarded from this TableHDURef's projection; use select()/head() to change the projection instead"` naming the duplicated kwarg; honest docstrings on all three (forwarded kwargs listed; `select()`/`head()` pointed to). Strict-xfail pin `tests/test_table_squeeze_and_ref_cache.py::test_tablehduref_to_arrow_columns_kwarg` flipped to a normal passing test (now pins the reject contract + `select(["A"]).to_arrow()` positive path); new `test_tablehduref_scan_and_reader_arrow_reject_forwarded_kwargs` covers all three methods × `hdu`/`columns`/`row_slice` + pass-through of `batch_size=`. Before: `Regex pattern did not match. Expected: "duplicate argument 'columns'"; Actual: "torchfits._table.read.read() got multiple values for keyword argument 'columns'"`. After: 42/42 green (run log below). | fixed |
| r6b-02 | CLEANUP | duplication | `_hdu/table_hdu_ref.py:_normalize_row_slice` (deleted); `_table/utils.py:91` (surviving) | **r5c-15 routed.** The `table_hdu_ref.py` method duplicated `_table/utils._normalize_row_slice` (used at read `:203` / iter_rows `:234`). | Deleted the method; both call sites import `.._table.utils._normalize_row_slice` (function-level import — module import would pull the table I/O package into `import torchfits.hdu`; matches the file's existing lazy-import idiom). Zero delta on valid inputs (pinned by `test_tablehduref_read_honors_row_window`: `head(1).read()` → rows `[1]`/`[3]`, `read(row_slice=(1,2))` → `[2]`). Intentional hostile-input delta (shared fn's better contract): `row_slice=slice(0,-1)` now raises `ValueError: row_slice negative stop is not supported…` early (the old local copy returned 0 rows and died later with `num_rows must be > 0 or -1 for all rows` from `_read_pipeline_fallback:78`); step/`start<0` messages unified with the arrow path (already pinned at `tests/test_arrow_table_api.py:1178`). Pinned: `test_tablehduref_read_rejects_negative_stop` (red before: `Expected 'negative stop'; Actual 'num_rows must be > 0 or -1 for all rows'`). | fixed |
| r6b-03 | BLOCKER | 3 lifetime/staleness | `_hdu/table_hdu_ref.py:columns` (`_all_columns_cache*`) | Caches keyed `id(self.header)` alias stale column names after header replacement: `columns` cached `(id(header), header._version)` with **no strong-ref guard**, and any freshly constructed `Header` has `_version == 0` (the constructor never bumps), so a replacement header landing on a freed header's id reads as a cache hit. Deterministic repro: cache `columns` for header `x`, replace `ref.header` twice so the second fresh header reuses `id()` → `ref.columns` returns `["x"]` while the header says `y`. Trigger measured **11/20** ad-hoc and **10/10** via an id-gated allocation loop. | Dropped the cache (spec's primary fix for metadata walks; recompute 23.1 µs/100-card is not a hot-path cost) and narrowed the `TFIELDS` parse except to `(TypeError, ValueError)`. Test `test_tablehduref_columns_follow_replaced_header_on_id_reuse` (id-gated loop; red before: `assert ['x'] == ['y']`; green after). `TableHDU._cached` (r6b-12) keeps its sound cache. | fixed |
| r6b-04 | BLOCKER | 6 security | `_hdu/tensor_hdu.py:to_tensor/chunks` | Both re-open `self._source_path` via `cpp.open_fits_file`/`cpp.SubsetReader` **without re-running `guard_fits_path`**, while `TableHDURef._refresh_file_view` and `hdu_list.py:80` do guard before every CFITSIO open. The URL was validated at HDUList open, but the re-open re-resolves DNS — a rebinding flip to a private address between open and re-open bypassed the SSRF guard entirely (C++ `check_fits_filename_security` enforces only the blocked-prefix policy). Repro: `TensorHDU(file_handle=mock, source_path="http://127.0.0.1:1/x.fits").to_tensor()` → `RuntimeError: Could not open FITS file: http://127.0.0.1:1/x.fits` (CFITSIO attempted the connection — the guard never ran). | `guard_fits_path(source)` immediately before each reopen (R1's "re-resolve + re-validate immediately before open" policy); for `chunks` the guard is hoisted **outside the generator body** via a thin `chunks()` → `_chunks_iter()` split (invariant `cfitsio-http-ssrf`: guards outside generator bodies). Test `test_tensor_hdu_reopens_revalidate_source_path` (red before: RuntimeError above; green after: `HttpBlockedError` typed). | fixed |
| r6b-05 | MAJOR | 5 error contract / 1 silent drop | `_hdu/table_hdu.py:vla_lengths`, `_hdu/table_hdu_ref.py:vla_lengths` | `except Exception: continue` swallowed **IO errors** and silently dropped columns: `TableHDURef(header=VLA-hdr, source_path=<missing>).vla_lengths` returned `{}` instead of surfacing the failed read (and any item-length error vanished the same way). Repro (before): missing-file ref returned `{}` silently. | Narrowed to `except KeyError` (schema/data disagreement skips; everything else propagates). Tests `test_tablehduref_vla_lengths_propagates_read_errors` (missing file → `RuntimeError`, red before `DID NOT RAISE`) and `test_tablehdu_vla_lengths_propagates_item_errors` (a `__len__`-raising VLA element → `RuntimeError("boom")`, red before `DID NOT RAISE`). | fixed |
| r6b-06 | MAJOR | 1 silent truncation / 5 error contract | `_hdu/table_hdu.py:head` | Contract decision (plan contingency 2): in-memory `TableHDU.head(-2)` silently truncated tail rows ("all but last 2"), a surprising pandas-ism on a materialized table; now `ValueError`. Repro (before): `TableHDU({"x": torch.zeros(10)}).head(-2)` → 8 rows, no error. | `head(n)` raises `ValueError("head(n) requires n >= 0")` + honest docstring (successive calls narrow monotonically). `TableHDURef.head` (view semantics) keeps negative = tail-truncate **within the current window** and composes (`head(4).head(-1)` → `slice(5,8)` on a `(5,10)` window) — per the assignment's scoping of `ValueError` to the in-memory class. Tests: `test_tablehdu_head_negative`/`test_tablehdu_head_numpy` (red before: `DID NOT RAISE ValueError`), `test_tablehdu_head_composes`, extended `test_tablehduref_head*` composition pins. **Asymmetry is deliberate; owed doc note (below).** | fixed |
| r6b-07 | MINOR | 3/4 stale state | `_hdu/table_hdu.py:__init__/_raw_data`, `num_rows` (`functools.cached_property`) | `TableHDU` aliased the caller's dict while `num_rows` was a `cached_property` and `columns` read the dict live — mutating the dict after construction made `num_rows`/`columns`/`hdu[col]` disagree (stale cache after mutation). Repro (before): `d={"x": zeros(10)}; h=TableHDU(d); d["x"]=zeros(3); d["y"]=zeros(3)` → `h.num_rows == 3`, `h.columns == ["x","y"]` vs data aliasing. | Snapshot the dict in `__init__` (`dict(tensor_dict)`), matching `_derived_header()`'s isolation philosophy; `cached_property num_rows` is sound against external mutation afterwards. Test `test_tablehdu_isolated_from_caller_dict_mutation` (red before: `assert 3 == 10`). All internal `TableHDU(...)` call sites pass fresh dicts (verified by grep) — no consumer relied on the alias. | fixed |
| r6b-08 | MINOR | 5 error contract | `_hdu/tensor_hdu.py:data` | After `mark_closed()`, `hdu.data` raised `ValueError("No file handle available")` — misleading (the handle existed and was closed) and inconsistent with `to_tensor()`'s `RuntimeError("…closed…")`. | `data` raises the same typed `RuntimeError("TensorHDU file handle is closed; cannot read image data")` when `_closed`; in-memory HDUs (never had a handle) keep `ValueError("No file handle available")`. Test `test_tensor_hdu_data_after_close_raises_typed_error` (red before: `ValueError: No file handle available` at the raises block). | fixed |
| r6b-09 | CLEANUP | 5 silent fallthrough | `_hdu/table_hdu.py:TableDataAccessor.__getitem__` | `try: return value.squeeze(1) except Exception: pass` silently returned the unsqueezed `(N,1)` value on any squeeze failure — unreachable for `torch.Tensor`/`np.ndarray` (`squeeze(1)` on `(N,1)` cannot raise) and a silent shape fallthrough for duck-typed columns. | Deleted the dead `try/except` (zero behavior for every real column type; a duck type without `squeeze` now fails loudly instead of silently diverging in shape). Covered indirectly by `test_table_data_accessor_preserves_rank` + `test_table_data_accessor_auto_squeeze` (green). | fixed |
| r6b-10 | MINOR | 7 edge / test determinism | `tests/test_hdu_table_contracts.py:test_quantize_nan_becomes_blank_not_lo` | Unseeded `torch.randn(16,16)*10+100` occasionally draws a >3σ pixel that `quantize="robust"` clips **by design** (~±3σ robust range), and the clipped pixel alone tripped the `rtol=2e-2` round-trip check: flaked ~1-in-3 (baseline evidence below). | Deterministic tail-free ramp data (`torch.arange(256)*0.05+100`, NaN spot kept): verified quantize round-trip max abs err 0.0127 ≪ atol=0.2. Before (baseline run): `1 failed … 69.505383 (ACTUAL) vs 67.407844 (DESIRED) …` at 3 runs / 1 fail; after: 5/5 consecutive passes + 42/42 + 52/52 suite greens. | fixed |
| r6b-11 | MINOR | 7 boundary pins | `_hdu/dataview.py:__getitem__` | Candidate A-08 ("int index clamps to empty slice") **re-derived as already fixed at HEAD**: `_normalize_index` raises `IndexError` for `s < -dim or s >= dim` and negative indices wrap; slices clamp (slice semantics), exactly per the contract. `TableHDURef.head` window-composition candidate (7841dec) likewise already correct at HEAD. | No src change. Boundary pins added: `tests/test_views_dataview_index.py` (`d[rows]`, `d[rows+1]`, `d[-rows-1]`, `d[-rows-100]`, and the x-axis equivalents all raise `IndexError`; `d[-rows]==d[0]`, `d[-1]==d[rows-1]`, `d[0,-1]==d[0,cols-1]`; block semantics `d[-1].shape == (1, cols)`; slices still clamp) + composition pins in `tests/test_table_head.py`. Green immediately (pins of current behavior). | fixed |
| r6b-12 | — (verify) | 3 lifetime | `_hdu/table_hdu.py:_cached`, `header.py:_version` | Candidate "caches keyed `id(self.header)`" re-derived for `TableHDU._cached`: sound at HEAD — key `(id(header), header._version)` **plus** a strong-ref guard (`_cache_header_ref is header`) that makes id() reuse impossible while cached. `Header._version` bumps in `__setitem__`, `__delitem__`→`remove`, `update`, `clear`, `pop`→`remove`, `popitem`, `setdefault`, `insert`, `remove`, `_append_card`, `_set_card` (every mutation path; slice A confirmed independently, `tests/test_header_versioning.py` pins it). No staleness reachable; no `Header._mutation_seq` coordination needed. | No change (keeping a correct cache is the minimal diff). Evidence: code walk above + `test_tablehduref_cache_invalidation{,_on_del}` (green). | fixed |
| r6b-13 | BLOCKER | 1 silent data loss | `_hdu/table_hdu_ref.py:_refresh_file_view` | **Routed from slice A (r4b-13 root-fix family).** `_refresh_file_view` built its fresh header from raw `cpp.read_header(handle, hdu)` triples **without** LONGSTRN `'&'`+CONTINUE reassembly — every mutation-style call returning `self._refresh_file_view()` (`append_rows_file`/`insert_rows_file`/`delete_rows_file`/`update_rows_file`/`rename_columns_file`/`drop_columns_file`) silently truncated >68-char string keywords to 67+`'&'` in the returned ref's header (while `torchfits.open()`-time headers were already whole via the r4c-15 wrapper). Probe: 80-char keyword → `open` header whole=True, `append_rows_file`-refreshed header whole=False (`'xxxxxxxx…xxxxxxxx&'`). | Wrapped with slice A's shared `card.py::_reassemble_longstr_cards` at the raw-triples site (import pattern matches `hdu_list.py:18`). Test `test_tablehduref_refresh_preserves_long_string_values` (red before: `AssertionError: assert 'xxx…xxxxxxxx&' == 'xxx…xxxxxxxxx'`; green after). Owned suite 53/53; `tests/test_hdu_file_ops.py` 15 passed / 1 skipped; adjacent 77 passed / 2 skipped. | fixed |

## Evidence (captured before → after)

Red batch before fixes (`pixi run pytest` over the 5 owned test files; 11 contract
failures — this run also contains the r6b-10 flake's baseline failure):

```
FAILED tests/test_table_head.py::test_tablehdu_head_negative - Failed: DID NOT RAISE ValueError
FAILED tests/test_table_head.py::test_tablehdu_head_numpy - Failed: DID NOT RAISE ValueError
FAILED tests/test_table_squeeze_and_ref_cache.py::test_tablehduref_to_arrow_columns_kwarg - AssertionError: Regex pattern did not match.
  Expected regex: "duplicate argument 'columns'"
  Actual message: "torchfits._table.read.read() got multiple values for keyword argument 'columns'"
FAILED tests/test_table_squeeze_and_ref_cache.py::test_tablehduref_scan_and_reader_arrow_reject_forwarded_kwargs - AssertionError: Regex pattern did not match.
  Expected regex: "duplicate argument 'hdu'"
  Actual message: "torchfits._table.read.scan() got multiple values for keyword argument 'hdu'"
FAILED tests/test_table_squeeze_and_ref_cache.py::test_tablehduref_columns_follow_replaced_header_on_id_reuse - AssertionError: assert ['x'] == ['y']
FAILED tests/test_table_squeeze_and_ref_cache.py::test_tablehduref_read_rejects_negative_stop - AssertionError: Regex pattern did not match.
  Expected regex: 'negative stop'
  Actual message: 'num_rows must be > 0 or -1 for all rows'
FAILED tests/test_hdu_table_contracts.py::test_tablehdu_vla_lengths_propagates_item_errors - Failed: DID NOT RAISE RuntimeError
FAILED tests/test_hdu_table_contracts.py::test_tablehduref_vla_lengths_propagates_read_errors - Failed: DID NOT RAISE RuntimeError
FAILED tests/test_hdu_table_contracts.py::test_tablehdu_isolated_from_caller_dict_mutation - AssertionError: assert 3 == 10
FAILED tests/test_hdu_close_and_overflow.py::test_tensor_hdu_data_after_close_raises_typed_error - ValueError: No file handle available
FAILED tests/test_hdu_close_and_overflow.py::test_tensor_hdu_reopens_revalidate_source_path - RuntimeError: Could not open FITS file: http://127.0.0.1:1/x.fits
11 failed, 31 passed in 2.35s
```

Baseline flake (unseeded quantize test, before the test fix):

```
FAILED tests/test_hdu_table_contracts.py::test_quantize_nan_becomes_blank_not_lo - AssertionError:
Not equal to tolerance rtol=0.02, atol=0.2
Mismatched elements: 1 / 255 (0.392%)
 [130]: 69.50538635253906 (ACTUAL), 67.40784454345703 (DESIRED)   # robust clip of a >3σ draw
1 failed, 37 passed, 1 xfailed in 2.46s        # 3 reruns: fail, pass, pass
```

After fixes:

```
42 passed in 2.30s                    # the 5 owned test files
52 passed in 2.45s                    # + tests/test_hdu.py, tests/test_skinny_meta.py
87 passed, 2 skipped in 2.21s        # adjacent consumers: test_hdu, test_skinny_meta, test_table,
                                      # test_table_file_ops, test_writing
20 passed in 1.56s                    # test_stream_table_and_cache, test_where_and_batch_errors
29 passed in 1.89s                    # test_docs_integrity, test_astropy_upstream_smoke
14 passed in 1.66s                    # test_table_squeeze_and_ref_cache + test_table_head rerun
5× 1 passed                           # quantize flake test, 5 consecutive green
```

Cache-hole + measurement proof (throwaway scripts):

```
stale hits in 20 attempts: 11                       # TableHDURef id-reuse aliasing (pre-fix)
id-reuse forced: 10 /10; stale: 10                  # id-gated loop (pre-fix trigger rate)
columns recompute (100-card): 23107 ns/call = 23.1 µs   # measurement for the drop-vs-keep decision
```

r6b-13 LONGSTRN refresh proof (probe + pin):

```
open header whole: True 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxx…      # open_hdulist wrapper (r4c-15)
refresh header whole: False 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx…   # _refresh_file_view (pre-fix)
FAILED tests/test_table_squeeze_and_ref_cache.py::test_tablehduref_refresh_preserves_long_string_values
  AssertionError: assert 'xxxxxxxxxxxx…xxxxxxxxxxxx&' == 'xxxxxxxxxxxx…xxxxxxxxxxxxx'
1 failed in 1.43s                                    # red before the wrap
1 passed                                             # green after (owned suite: 53 passed)
```

## Owed doc updates (docs land at R15)

- `docs/api-tables.md` (HDU/table section): `TableHDURef.to_arrow`/`scan_arrow`/`reader_arrow`
  forward `hdu`/`columns`/`row_slice` from the projection and raise `TypeError` on duplicates;
  `select()`/`head()` are the projection knobs (r6b-01).
- `docs/api-tables.md` (head): `TableHDU.head(n)` requires `n >= 0` (`ValueError`);
  `TableHDURef.head(n)` narrows the current row window and `n < 0` truncates the window's
  tail — the two contracts are deliberately different (r6b-06).
- `docs/api-core-io.md` (~`:283`): `TensorHDU.data` raises `RuntimeError` after the HDU is
  closed (same contract as `to_tensor`), `ValueError` only for never-file-backed HDUs (r6b-08).

## Deferred (site + evidence)

- `table_hdu_ref.py:num_rows/_is_ascii_table` (`except Exception` around header `int()`/`str()`
  parsing) and `_refresh_file_view`'s error-path `handle.close()` swallow: zero-observable
  narrowing — no test can distinguish the narrowed excepts (header access performs no IO;
  close-during-unwind must not mask the primary error); left untouched per `minimal-diff`.
- `TableHDURef.head(-n)` view semantics kept (tail-truncate within window) per the assignment's
  scoping of the `head(-n) = ValueError` decision to the in-memory class — revisit only if the
  API review wants symmetric contracts (r6b-06 row + owed doc).

## Per-file disposition

| file | depth | finding IDs | status |
|---|---|---|---|
| src/torchfits/_hdu/table_hdu.py | deep (full read, every method against the rubric) | r6b-05, r6b-06, r6b-07, r6b-09, r6b-12 | fixed |
| src/torchfits/_hdu/table_hdu_ref.py | deep (full read, every method against the rubric) | r6b-01, r6b-02, r6b-03, r6b-05, r6b-13 | fixed |
| src/torchfits/_hdu/tensor_hdu.py | deep (full read; ctor/data/mark_closed/to_tensor/chunks/repr) | r6b-04, r6b-08 | fixed |
| src/torchfits/_hdu/dataview.py | deep (full read; `__getitem__`/`dtype`/`shape` + A-08 re-derivation) | r6b-11 | fixed (verify-only + pins) |
| tests/test_table_squeeze_and_ref_cache.py | owned test; xfail flipped + 4 pins added | r6b-01, r6b-02, r6b-03 | fixed |
| tests/test_table_head.py | owned test; rewritten to the r6b-06 contract + composition pins | r6b-06, r6b-11 | fixed |
| tests/test_hdu_table_contracts.py | owned test; 3 contract tests + quantize determinism | r6b-05, r6b-07, r6b-10 | fixed |
| tests/test_hdu_close_and_overflow.py | owned test; 2 tensor_hdu contract tests | r6b-04, r6b-08 | fixed |
| tests/test_views_dataview_index.py | new owned test; DataView boundary contract (A-08) | r6b-11 | fixed |
| tests/test_hdu.py, tests/test_skinny_meta.py | owned tests; reviewed, no change needed | — | clean |
