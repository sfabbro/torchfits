# Table-stack major-release audit (2026-08-26)

Audit only. No fixes applied.

**Scope:** every file under `src/torchfits/_table/`, `src/torchfits/_table_engine/`, plus `src/torchfits/table.py`, `where.py`, `_where.py`. Cross-check `docs/api-tables.md` against live `inspect.signature`. Hypotheses that live just outside the tree (`table_api.py`, `TableHDURef.head`, quantize helper) are cited because `table.read_torch` / `table.write` / the backlog name them.

**Method:** full source read of the 18 scoped files, signature dump, and live repros against current code (temp FITS files, 2026-08-26).

**Verdict: Block.** Do not tag a major release on this stack until `table.read_torch(..., where=)` stops returning different rows than `table.read(..., where=)` for TNULL sentinels and out-of-range integer literals, and empty scans/readers stop yielding a zero-field schema.

---

## Backlog hypotheses (current code)

| Hypothesis | Result | Evidence |
|---|---|---|
| `schema()` reports complex `C`/`M` as float64 | **Disproved** (fixed). Header mapper returns `None` for `C`/`M`; Arrow `read`/`scan`/`schema()` raise `NotImplementedError`. Torch dict path still serves `complex64`. Residual: projecting away the complex column still fails on `read()`, while `schema(columns=["X"])` succeeds. | `_read_schema.py` 287–294; `read.py` 56–74; live: `schema()` / `read()` raise; `read_torch` → `complex64`; `schema(columns=["X"])` → `float`. |
| Empty-result reads drop requested unknown columns | **Confirmed.** Empty `where=` / `row_slice=slice(0,0)` keep only names present in the header schema (`get_field_index >= 0`). Unknown-only projection yields an empty schema. `reader(row_slice=slice(0,0))` and `scan(slice(0,0))` drop *all* columns. A non-empty read of an unknown name raises `RuntimeError`. | `_read_schema.py` 51–79; `_read_scan.py` 288–289; `read.py` 296–298; live: `columns=["RA","NOPE"], where="RA>9999"` → `["RA"]`; `reader(slice(0,0)).schema.names` → `[]`. |
| `TableHDURef.head(n)` replaces the existing row window | **Confirmed.** Always sets `row_slice=slice(0, n)`, ignoring `_row_slice`. Live: window `slice(2,5)` (`num_rows=3`) then `head(2)` → `slice(0,2)` (`num_rows=2`), not `slice(2,4)`. In-memory `TableHDU.head(-n)` now **raises** `ValueError` (that half of the old note is fixed). | `_hdu/table_hdu_ref.py` 132–141; `_hdu/table_hdu.py` 331–333. Out of `_table/` but named by the hypothesis. |
| `to_astropy`: TNULL → object dtype; TUNIT not mapped | **Disproved** for `table.to_astropy(path)`. Path input maps `TUNIT` → `.unit` and TNULL → `MaskedColumn` (int32, not object). Residual: header-only empty tables omit `fits_tunit`; root `torchfits.to_astropy(tensor_dict)` still has no units (different function). | `interop.py` 336–338, 393–420, 450–462; `tests/test_interop.py` 179–200; live: `unit=mag`, `ID.mask == [False, True]`, dtype int32. |
| `table.write(quantize=)` silent no-op when no column qualifies | **Confirmed** for blanket `"robust"` / `True` on an all-integer dict: `parse_table_quantize_spec` returns `{}`, `_prepare_quantized_table_data_for_write` returns `changed=False`, write succeeds with native `TFORM`. Named integer column raises `TypeError` (not silent). | `_io_engine/quantize.py` 114–122; `_write_helpers.py` 166–167; `write.py` 73–75; live: int-only `quantize="robust"` → `tforms=['1J','1I']`, no error. |
| Broad `except Exception` swallows real IO errors | **Confirmed** on the `read_torch` engine and mmap `update_rows`; **disproved** as a blanket `except Exception` in `_read_scan.py` (now typed + `logger.debug`). | `table_api.py` 239–289, 375–378; `mutation.py` 541–543; `_read_scan.py` 75–78 / 468–547; `read.py` 70–71. |

---

## Issues

### BLOCKER — `read` vs `read_torch` `where=` row sets disagree (TNULL)

`table.read(..., where="M > 20")` excludes a TNULL sentinel of 25 (all of `auto`/`cpp`/`torch` × `mmap` True/False → `[30]`). `table.read_torch(..., where="M > 20")` returns `[25, 30]`.

Arrow path strips sentinels before the predicate (`_read_where.py` 214–232, 462–502). `read_torch` delegates to `_io_engine/table_api.py` `read_table`, which compiles the same “simple” predicates then compares raw tensors with no TNULL mask (322–336). Docs (`docs/api-tables.md` 121–127, 269–272) present the split as *operator dialect only*.

This is a silent catalog error: null rows look like detections on the torch API and vanish on the Arrow API.

### BLOCKER — `read_torch` integer `where=` wraps the literal

`_torch_cmp_mask` (`_read_where.py` 85–106) promotes narrow integer columns to int64 when the literal is out of dtype range, matching C++ pushdown. `read_table` does not: `values > lit` on int16 vs `40000` wraps.

Live: `table.read(..., where="V > 40000")` → 0 rows; `table.read_torch(..., where="V > 40000")` → all rows `[1, 2, 3]`.

Same class of bug as TNULL: two public `where=` entry points, opposite science.

### HIGH — Empty stream / empty projection drops schema

- `scan(..., row_slice=slice(0,0))` returns immediately (`_read_scan.py` 288–289) with no empty batch.
- `reader()` then does `next(it, None)` → `RecordBatchReader` on `pa.schema([])` (`read.py` 296–298). Live: empty names.
- `dataset(path, row_slice=slice(0,0))` is an in-memory dataset with an empty schema.
- `_empty_table_with_schema` silently skips names with `get_field_index < 0` (`_read_schema.py` 51–66). Live: `columns=["RA","NOPE"], where="RA>9999"` → `["RA"]`; `columns=["NOPE"], row_slice=slice(0,0)` → `[]`.
- `schema(path, columns=["NOPE"])` returns an empty schema; `schema(..., columns=["RA","NOPE"])` drops `NOPE`.
- A *populated* unknown-column read raises `RuntimeError` (C++ also prints `Column not found: NOPE` to stderr). Empty vs non-empty is inconsistent.

`scan(where=)` that matches nothing *does* yield one typed empty batch (`_read_scan.py` 277–282); that path is fine. Zero-width windows and unknown names are not.

### HIGH — `TableHDURef.head(n)` replaces the row window

`head` always installs `slice(0, n)` (`table_hdu_ref.py` 132–141). `select()` correctly forwards `_row_slice`. Composing `ref[rows 100:200].head(10)` therefore returns file rows 0–10, not 100–110. Silent wrong slice; not in the scoped dirs but it is the named backlog item.

### HIGH — `read_torch` swallows IO behind `except Exception`

`read_table` (`table_api.py` 239–289, 375–378) catches `Exception`, sets `data = None`, and either retries or falls through. Truncation is *not* special-cased here (the Arrow torch-where path in `_read_where.py` 206–210 *does* re-raise `"truncat"`). A failed mmap/open on `read_torch(..., where=)` can surface as `RuntimeError: Failed to apply where=...` or as a different engine’s result.

`_read_scan.py` no longer uses bare `Exception` for those fallbacks (typed `RuntimeError`/`OSError` + debug log). The backlog citation of `_read_scan.py` is stale; `table_api.py` is not.

### HIGH — Dual `where=` engines, docs vs source docstring

Operator split is real and tested (`_compile_where_to_simple_predicates` returns `None` for `OR`/`IN`/`NOT`/`IS NULL`, `_read_where.py` 59–82; `read_torch` raises `ValueError`, `table_api.py` 211–213). `docs/api-tables.md` 121–127 is correct.

`read_torch`’s *own* docstring still says optional `where` uses C++ filtered reads for “simple numeric predicates **(same dialect as `table.read`)**” (`read.py` 209–211). That is a public-contract lie in the source, and integrity tests only guard the markdown (`tests/test_docs_integrity.py` 379–389).

`backend="torch"` on `table.read` still runs `_try_torch_tensor_where_filter` (C++ readers + `_torch_cmp_mask`), so the named backend is not a closed world. `read_torch` is a fourth evaluator (raw PyTorch ops) and is not a `backend=` of `table.read`.

### MEDIUM — Mutation error types are not one type

Unknown / extra column:

| Call | Missing / extra | Type |
|---|---|---|
| `replace_column` | missing | `KeyError` (`mutation.py` 148) |
| `drop_columns` | missing | `KeyError` (608) |
| `rename_columns` | missing | `KeyError` (574) |
| `insert_column` | already exists | `ValueError` (66) |
| `update_rows` | extra | `ValueError` (413) |
| `append_rows` / `_normalize_mutation_rows` | extra | `ValueError` (`_mutation_coerce.py` 203) |

Callers cannot catch a single exception. `update_rows(..., mmap="auto")` still uses `except Exception` and only re-raises if forced mmap or `"truncat"` in the message (`mutation.py` 541–543).

`NAXIS2` is parsed with `except Exception: … = 0` in insert/replace/insert_rows/delete_rows (75, 152, 257, 315): a corrupt card becomes a silent zero-row mutation rather than `ValueError`.

### MEDIUM — Header-only schema omits TUNIT / TSCAL / TZERO

`_schema_from_header` attaches only `fits_tform` / `fits_tdim` / `fits_tnull` (`_read_schema.py` 342–354). `_build_fits_metadata` (used on the data path) also writes `fits_tunit` / `fits_tscal` / `fits_tzero` (116–134).

Live: `read(..., include_fits_metadata=True)` on a MAG column has `fits_tunit=mag`; the same read with `row_slice=slice(0,0)` has only `fits_tform`. `schema(include_fits_metadata=True)` (header fast path) misses TUNIT; `schema(where=...)` (scan path) includes it.

Empty vs non-empty metadata is a schema lie even when column *names* survive.

### MEDIUM — Complex columns poison the whole Arrow HDU

`_reject_complex_columns` inspects every TFORM, not the projection (`read.py` 63–74). Live: a table with `Z` complex64 + `X` float32: `read(columns=["X"])` raises; `schema(columns=["X"])` returns `X: float`; `read_torch` returns both.

`_reject_complex_columns` also `except Exception: return` (70–71), so a header IO failure skips the guard and can fail later in Arrow conversion.

### MEDIUM — `scan_torch` does not accept EXTNAME

`scan()` / `read()` / `reader()` resolve `hdu: int | str`. `scan_torch` is typed `hdu: int` and passes the value straight into `cpp.read_fits_table_rows`. Live: `hdu="CAT"` → `TypeError` from the binding. Docs show `hdu=1` only; they do not warn that EXTNAME works on `scan` but not `scan_torch`.

### MEDIUM — `evaluate_where` vs `table.read` on integer TNULL

`table.read(..., where="M IS NULL")` returns the null row (Arrow `pc.is_null` after sentinel decode). `evaluate_where` on the raw integer array uses `value is None` for non-float `isnull` (`where.py` 106–112) → all `False`. The helper docstring tells people to use `isnull` / `table.read` for FITS nulls, but integer sentinels still look like data.

Numeric `== NULL` correctly raises (`where.py` 68–76).

### MEDIUM — Blanket `quantize="robust"` is silent when nothing is float

Documented as packing *float* columns; integer columns are skipped by design (`quantize.py` 109–122). An all-integer table plus `quantize="robust"` is still a successful write with no warning and `changed=False`. A *named* integer column raises. Asymmetric.

### MEDIUM — `update_rows` mmap auto-fallback

Already noted under error types. Disk errors that do not contain `"truncat"` become a second CFITSIO write. Hard to test; same pattern the backlog named.

### LOW — Docs vs signatures (`docs/api-tables.md`)

Matches live signatures for `read`, `read_arrow` (alias), `read_torch` (including ignored cache kwargs), `scan`, `scan_torch` (params), `reader` (`include_fits_metadata=True` vs `read`’s `False`), `schema` (`apply_fits_nulls=False`), mutation kwargs including `tnull`/`tscal`/`tzero`.

Gaps:

- `write()` example includes `schema=`; the parameter table omits `schema` and `extname` (signature has both).
- `update_rows(..., mmap="auto")` is not in the mutation section.
- `write_csv` / `write_ipc` exist on `torchfits.table` and are not documented (only `write_parquet`).
- `dataset()` / `scanner()` are one-liners; `scanner` also takes `filter`, `use_threads`, `where`.
- `duckdb_query(..., return_arrow=True)` undocumented.
- Interop section mixes `torchfits.to_astropy(tensor_dict)` (root, no TUNIT) with `table.read_astropy` / `table.to_astropy` (path, TUNIT/TNULL). Easy to apply the wrong contract.
- `read_torch` docs do not mention TNULL or integer-literal width (the BLOCKERs).
- `scan()` “peak memory bounded by `batch_size`” is true for `where=` (`_read_scan.py` 245–248). `read()` / `dataset()` / `_try_torch_tensor_where_filter` still materialize (up to `_TORCH_WHERE_MAX_ROWS` = 1e6, `_read_where.py` 37–39). `dataset()` is `reader().read_all()` (`read.py` 314).

### LOW — TSCAL / TZERO on read (not the old mmap crash)

Live scaled `TFORM=I` + `TSCAL=0.5` + `TZERO=10`: `read`, `read(mmap=False)`, `read_torch`, and `where="A > 11"` all return physical `[10.5, 11.0, 11.5, 12.0]`. Mmap is refused for scaled columns (`_read_schema.py` 197–198, 230–233; `table_streaming.py` 83–109) and CFITSIO buffered reads apply the scale. Not a current correctness hole; unsigned `TZERO` still disables the mmap row path because `reject_scaled` treats any `TZERO` as scaled.

### CLEANUP — Public tokenizer is not the parser

`_parse_where_expression` normalizes then `ast.parse` (`_where.py` 341–352). `_tokenize_where_expression` is unused by the parser, still exported from `torchfits.where` and listed in `docs/api-tables.md` 320. Two tokenizations can disagree.

`_parse_where_literal` uses `except Exception: pass` on int/float (`_where.py` 30–37) — benign after a regex, still a swallow.

`SyntaxError` from `ast.parse` is rewritten to “Unexpected trailing tokens” without `from e` (353–358).

### CLEANUP — Unnamed TTYPE skipped in full-read capability

`_can_use_full_read_path` `continue`s when `TTYPE` is missing/empty (`_read_schema.py` 224–225), so mmap/torch full-read can be approved while an unnamed field is never validated. `fits_schema._iter_tfields_indexed` skips `name is None` (114–115) but keeps `""`. Capability check and schema walk disagree.

### CLEANUP — Width-1 lists

C++ `TableReader` allocates rank-1 tensors when `repeat == 1` (`table_reader.h` ~546–548). `stream_table` still runs `_squeeze_scalar_columns`; `_iter_chunks_cpp_table` / `_read_cpp_table_chunk` do not. `_numpy_to_arrow_array` turns any 2-D array into `FixedSizeList` (`arrow_convert.py` 164–182). Main scalar path was rank-1 in this audit’s files; the squeeze hole is residual, matching the old “chunk route” note.

---

## `where=` dialect (read vs `read_torch`)

| | `table.read` / `scan` | `table.read_torch` |
|---|---|---|
| Compare / `BETWEEN` / `AND` | yes | yes (compiled to simple preds) |
| `OR` / `IN` / `NOT` / `IS NULL` | yes (Arrow `pc`) | `ValueError` |
| TNULL vs numeric pred | sentinel excluded (default `apply_fits_nulls=True`) | sentinel can match |
| Out-of-range int literal | promote to int64 | dtype wrap, can keep all rows |
| Row window | window then filter | same *intent* in `table_api` (221–225); TNULL/wrap still wrong |
| `backend=` | `auto` / `cpp` / `torch` | none (always thin torch path) |

C-style `&&` / `||` / `~` rewrite is shared (`_where.py` 112–192). Parser is `ast.parse`, not the public tokenizer.

---

## File ledger (18 scoped files)

| File | Role | Status |
|---|---|---|
| `src/torchfits/table.py` | Public `table` façade; re-exports; `read_arrow = read`; `clear_cache` | CLEAN. Surface matches `__all__` / docs list. |
| `src/torchfits/where.py` | Public parse + `evaluate_where` | MEDIUM: integer `isnull` is `is None`; numeric `== NULL` correctly rejected. |
| `src/torchfits/_where.py` | Grammar, `ast.parse`, column extract | CLEANUP: tokenizer unused; literal `except Exception`; SyntaxError flattening. |
| `src/torchfits/_table/__init__.py` | Re-exports handle/reader acquire | CLEAN. |
| `src/torchfits/_table/read.py` | `read`/`scan`/`read_torch`/`reader`/`dataset`/`scanner` | BLOCKER docstring; HIGH empty `reader` schema; MEDIUM whole-HDU complex reject + `except Exception` on header. |
| `src/torchfits/_table/_read_schema.py` | Header schema, empty tables, capability | HIGH empty/unknown drop; MEDIUM metadata subset; **complex-as-float64 fixed**; CLEANUP unnamed skip. |
| `src/torchfits/_table/_read_where.py` | Arrow/torch/C++ `where=` for `table.read` | Reference-correct for TNULL + int width on *this* API; `_TORCH_WHERE_MAX_ROWS` full materialize. |
| `src/torchfits/_table/_read_scan.py` | Scan/chunk bodies | HIGH zero-row scan yields nothing; **no bare `except Exception`**; `where=` per-batch filter is real. |
| `src/torchfits/_table/engine.py` | Coalesced C++ range assemble | CLEAN (zero-fill empty segments). |
| `src/torchfits/_table/utils.py` | `row_slice`, pyarrow, TFORM helpers | CLEAN (negative stop raises). |
| `src/torchfits/_table/cache.py` | Private per-call CFITSIO handle | CLEAN (Option A, `guard_fits_path`). |
| `src/torchfits/_table/write.py` | `table.write`, HDU resolve, rewrite | MEDIUM quantize no-op via helper; CLEANUP `except Exception` on temp unlink / handle close (close is logged). |
| `src/torchfits/_table/mutation.py` | Row/column mutations | MEDIUM KeyError vs ValueError; HIGH/MEDIUM mmap `except Exception`; `except Exception` on `NAXIS2`. |
| `src/torchfits/_table/_mutation_coerce.py` | Coerce / TNULL defaults | CLEAN for TNULL fill (raises on bad sentinel, 179–186). |
| `src/torchfits/_table/interop.py` | Arrow→pandas/polars/astropy/duckdb | **TUNIT/TNULL path fixed**; MEDIUM `except Exception` on header meta → empty map; LOW undocumented csv/ipc. |
| `src/torchfits/_table/arrow_convert.py` | Tensor/numpy → Arrow, TNULL mask | CLEAN for 1-D TNULL; residual 2-D → `FixedSizeList`; sentinel coerce `except Exception` → no nulls. |
| `src/torchfits/_table_engine/__init__.py` | Policy re-exports | CLEAN. |
| `src/torchfits/_table_engine/backend_policy.py` | `TABLE_BACKENDS` | CLEAN (`auto`,`torch`,`cpp`). |
| `src/torchfits/_table_engine/read_policy.py` | Arrow filter vs C++ pushdown | LOW: `backend="torch"` still shares C++ readers via `_read_where`; no size forks (`n_rows` ignored). |

**Out of scope, required for hypotheses**

| File | Why |
|---|---|
| `src/torchfits/_io_engine/table_api.py` | Body of `table.read_torch`; TNULL/wrap BLOCKERs; `except Exception`. |
| `src/torchfits/_io_engine/_write_helpers.py` | Quantize `changed=False` no-op. |
| `src/torchfits/_io_engine/quantize.py` | Blanket robust skips non-floats. |
| `src/torchfits/_hdu/table_hdu_ref.py` | `head(n)` replaces window. |
| `src/torchfits/_hdu/table_hdu.py` | `head(-n)` now raises (fixed). |
| `src/torchfits/fits_schema.py` | Unnamed TTYPE / complex names. |
| `docs/api-tables.md` | Contract; operator split documented, TNULL/wrap split not. |

---

## Dual backends (short)

Three Arrow strategies (`read_policy.py` 45–63): C++ mmap pushdown when safe, else read-then-Arrow-filter; `backend="cpp"` prefers pushdown; `auto` + `mmap=False` skips it. Live TNULL predicates agreed across `auto`/`cpp`/`torch` × mmap for `table.read`.

`read_torch` is not one of those backends. It is a separate thin reader. Treat “dual backends” as: Arrow family (aligned) vs torch-dict `where=` (not aligned).

Streaming: `scan(where=)` filters per batch (memory ~ `batch_size`). `read()` concatenates. `scan_torch` has no `where=` (documented). `dataset()` always materializes.

---

## Live repro snapshot (2026-08-26)

Not tests in-tree; throwaway FITS under `/tmp`.

- Complex HDU: `schema()` / `read()` → `NotImplementedError`; `schema(columns=["X"])` OK; `read(columns=["X"])` still raises; `read_torch` dtypes `{Z: complex64, X: float32}`.
- Empty unknown: `read(columns=["RA","NOPE"], where="RA>9999")` → `["RA"]`; `reader(slice(0,0)).schema` empty.
- `TableHDURef` `slice(2,5)` + `head(2)` → `slice(0,2)`.
- `table.to_astropy(path)`: `MAG.unit == mag`, `ID` masked int32.
- Int-only `table.write(..., quantize="robust")`: `tforms=['1J','1I']`, no error.
- TNULL `M>20`: Arrow `[30]`, `read_torch` `[25, 30]`.
- Int16 `V>40000`: Arrow 0 rows, `read_torch` `[1,2,3]`.
- TSCAL 0.5 / TZERO 10: both APIs return physical values; `where="A>11"` agrees.
- `scan_torch(hdu="CAT")`: `TypeError` from C++.

---

## What already looks release-ready

- Complex-as-float64 header lie is gone; Arrow refuses C/M with a clear error.
- `table.to_astropy(path)` TUNIT + MaskedColumn TNULL.
- `table.read` TNULL exclusion on every named backend (regression test `tests/test_table.py` `test_where_tnull_sentinel_excluded_on_all_engines`).
- `scan(where=)` streams and keeps schema on zero matches (`tests/test_table.py` `test_scan_where_streams_in_batches`).
- `_read_scan.py` fallbacks are typed, not `except Exception`.
- `TableHDU.head(-n)` raises.
- TSCAL physical values on read/read_torch.
- `docs/api-tables.md` operator split for `read_torch` matches the compiler, not the `read.py` docstring.
