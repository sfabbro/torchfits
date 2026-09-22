# R1 slice B — `io.py`, `table.py`, `hdu.py`, `interop.py`, `fits_schema.py`

Round R1 root façade / interop / schema review (bug classes 1–8, perf classes 1–6).
Baseline `1bb6958`. Findings prefix `r1b`. All fixes landed in-tree (no commits).

| ID | Sev | Class | Files:symbols | Description (with repro) | Fix & validation | Status |
|---|---|---|---|---|---|---|
| r1b-01 | MAJOR | 1 (silent wrong result) | `src/torchfits/interop.py:to_astropy` | Dict path silently degraded Arrow nulls and vector columns. Repro: `torchfits.to_astropy({"a": [1, None, 3]})` → unmasked `float64` `[1.0, nan, 3.0]` (int widened to float, `None` silently became `NaN`); `{"v": torch.arange(6).reshape(2, 3)}` → `(2,)` object column instead of `(2, 3)` numeric. Diverged from `table.to_astropy`'s MaskedColumn/`(N, repeat)` contract (pinned by `test_table_to_astropy_fidelity`, `test_arrow_nulls_become_masked_column`). | Replaced the hand-rolled `to_numpy`/`to_pylist` loop with `table_to_astropy(arrow_table)` — the same `torchfits._table.interop.to_astropy` the str branch already delegated to. Nulls → `MaskedColumn` (int dtype preserved), 2-D tensors keep `(N, repeat)`. Tests `test_root_to_astropy_dict_nulls_become_masked`, `test_root_to_astropy_dict_vector_keeps_shape` fail before / pass after (evidence below). Zero opens added (dict path builds the Arrow table in-process and converts it; `_astropy_fits_column_meta` short-circuits on non-path input). **Docs note (R15):** `docs/api-tables.md` Astropy section — state that dict-input Arrow nulls become `MaskedColumn` and 2-D tensors keep `(N, repeat)`. Changelog: `fix(interop):` subject auto-bullets. | fixed |
| r1b-02 | MAJOR | 7 (deterministic failure on valid input) | `src/torchfits/interop.py:to_astropy` | `torchfits.to_astropy(pathlib.Path(...))` raised `AttributeError: 'PosixPath' object has no attribute 'items'` — a Path fell through `isinstance(data, str)` into the dict branch and hit `to_arrow`'s `data.items()`. The package accepts `os.PathLike` elsewhere (`write(path: str \| os.PathLike[str])`; `tests/test_pathlike_acceptance.py` is the convention). | Path branch now matches `(str, os.PathLike)` and delegates `os.fspath(data)`; annotation widened to `Dict[str, Any] \| str \| os.PathLike[str]`. Test `test_root_to_astropy_accepts_pathlike` fails before / passes after. | fixed |
| r1b-03 | MINOR | 5 (error contract) | `src/torchfits/interop.py:to_astropy` | Dict conversion wrapped `chunked_arr.to_numpy(zero_copy_only=False)` in a bare `except Exception:` → `to_pylist()` fallback: any conversion error (including genuine bugs) silently became an object column. | Removed with r1b-01 (conversion now lives in `_table.interop._arrow_column_to_astropy`, the reviewed `_table` implementation). Code proof: the `try/except Exception` block is gone from `interop.py` (see diff at `interop.py:99-126`). | fixed |
| r1b-04 | MAJOR | 1/7 (silent wrong result, edge inputs) | `src/torchfits/fits_schema.py:_iter_tfields_indexed, iter_table_columns, column_tnull_map` | FITS keywords are case-insensitive but lookups were case-inconsistent: the `TFIELDS` fast path resolved `TTYPE`/`TFORM`/`TDIM` exact-case only (a case-variant column vanished from the schema silently), and `TNULL`/`TSCAL`/`TZERO` were always exact-case (lowercase cards silently dropped → dtype/null reinterpretation). Repro: `unsigned_column_dtypes_from_header({"TFIELDS": 1, "ttype1": "A", "tform1": "1J", "tzero1": 2147483648.0})` → `{}` (unsigned convention missed → downstream int32 instead of uint32); `column_tnull_map({"ttype1": "A", "tform1": "1J", "tnull1": 7})` → `{}` (TNULL masking silently off). | Added module-private `_kw_lookup` (exact-case `dict.get` first; on the first miss the mapping folds to uppercase once and serves case-variant keys — bounded cost, no amplification on hostile `TFIELDS`), wired into the fast path and into `iter_table_columns`/`column_tnull_map`. Tests `test_case_variant_header_keys_resolve`, `test_mixed_case_fast_path_columns_resolve` fail before / pass after. Clean all-uppercase headers take the identical exact-case path as before (fold builds only on miss — no new scan on hot `where=` preprocessing). | fixed |
| r1b-05 | MINOR | 8 (docs/API faithfulness) | `src/torchfits/interop.py:to_arrow`, `src/torchfits/io.py:get_cache_performance` | Stale docstrings contradicted the implementation: `to_arrow` claimed `bytes(tensor.untyped_storage())` + "one copy" (actual: zero-copy `pa.Array.from_buffers` over `tensor.numpy()` — zero-copy is pinned by `test_to_arrow_numeric_tensor_shares_buffer`) and "No numpy dependency" (numeric path uses the tensor's NumPy view); `get_cache_performance` claimed statistics "for the handle and metadata caches" though the handle cache was removed in Option A (`caches.get_cache_performance` returns read-cache counters only). | Docstrings corrected to describe actual behavior; zero code change. | fixed |
| r1b-06 | CLEANUP | dead code | `src/torchfits/io.py:read` | `if "mode" in kwargs: raise TypeError("read() got multiple values for argument 'mode'")` is unreachable: `mode` is a named parameter, so `**kwargs` can never contain `"mode"`; a duplicate-value call is rejected by Python argument binding with the identical message before the body executes. | Deleted the 2-line guard (zero-behavior-change deletion; `kwargs["mode"] = mode` forwarding kept). Code proof: Python call binding — keywords matching a named parameter bind there or raise `TypeError: read() got multiple values for argument 'mode'` from the interpreter. | fixed |
| r1b-07 | MINOR | 8 (docs/API faithfulness) | `src/torchfits/fits_schema.py:tform_code_and_repeat` | Docstring said "Return (code, repeat) for a scalar TFORM" but the function returns values for vector TFORMs too (`"20A" → ("A", 20)`); what it excludes is VLA/unparseable. | Reworded to "for a non-VLA TFORM, or None if unparseable or VLA". | fixed |
| r1b-08 | MINOR | 8 (docs/API faithfulness) | `docs/api-core-io.md:359`, `docs/api.md:45` vs `src/torchfits/io.py:read_extname` | Docs show `read_extname(path, hdu=1)` (call-signature fence + api.md row) but the live signature is `read_extname(path, hdu=0)`. No test pins either default (`tests/test_skinny_meta.py` passes both indices explicitly). Also: `write()`'s parameter table omits the `checksum` row (fence + Checksums section cover it) and `read_tensor()`'s table omits `fallback_get_header` (fence shows it). | No src change (the freeze review forbids defaults drift during 1.2). Deferred to the R15 docs pass: align `read_extname` fence/table to `hdu=0` (or explicitly justify `hdu=1` and treat as an API change), add the `checksum` row and a `fallback_get_header` footnote. | deferred |
| r1b-09 | MINOR | 5 (error contract) | `src/torchfits/io.py:_READ_EXC_TYPES` | The catch tuple includes `TypeError`/`AttributeError` (and `MemoryError`): `read_batch(strict=False)` skips a file raising any of them as a "failed path" warning (e.g. `read_batch([42])` warns and returns `[]` instead of surfacing `TypeError`), and `read_unified` treats them as fallback triggers. The documented skip contract covers read failures, not programming errors. | No behavior change now — narrowing the tuple is an API-visible error-contract change and interacts with the deliberate extension-skew tolerance (`_read_pipeline_fallback.py` probes `hasattr(cpp_module, "resolve_hdu_name_cached")`). Deferred: narrow to IO-plausible types (RuntimeError/OSError/ValueError/MemoryError) with a release note once the skew-tolerance policy is decided. | deferred |
| r1b-10 | MINOR | 7 (edge inputs) | `src/torchfits/_table/interop.py:_materialize_arrow_table` (outside r1b file set — R5 `_table/` round) | `torchfits.table.to_astropy(pathlib.Path(...))` raises `TypeError: 'PosixPath' object is not iterable`: `_materialize_arrow_table`'s `isinstance(data, str)` check sends a Path into `pa.Table.from_batches(list(data))`. (Root `torchfits.to_astropy` is fixed here — r1b-02 — by delegating `os.fspath(data)`.) | Hand-off to R5: accept `os.PathLike` in `_materialize_arrow_table` (mirror r1b-02's `(str, os.PathLike)` + `os.fspath`). Not fixed here — `_table/interop.py` is R5-owned; no test added (a failing test would be left red). | deferred |

## Evidence — failing first, passing after

`pixi run pytest tests/test_interop.py tests/test_fits_schema.py -q` against unfixed code
(new tests added first):

```
FAILED tests/test_interop.py::test_root_to_astropy_dict_nulls_become_masked - AssertionError: a
assert None is not None
 +  where None = getattr(<Column name='a' dtype='float64' length=3>\n1.0\nnan\n3.0, 'mask', None)
FAILED tests/test_interop.py::test_root_to_astropy_dict_vector_keeps_shape - assert (2,) == (2, 3)
FAILED tests/test_interop.py::test_root_to_astropy_accepts_pathlike - AttributeError: 'PosixPath' object has no attribute 'items'. Did you mean: 'stem'?
FAILED tests/test_fits_schema.py::test_case_variant_header_keys_resolve - AssertionError: assert {} == {'PIX': torch.uint32}
FAILED tests/test_fits_schema.py::test_mixed_case_fast_path_columns_resolve - AssertionError: assert ['A'] == ['A', 'B']
5 failed, 13 passed in 2.14s
```

After the fixes — `pixi run pytest tests/test_interop.py tests/test_fits_schema.py tests/test_io.py tests/test_hdu.py tests/test_table.py -q`:

```
...................................................................      [100%]
67 passed, 4 warnings in 2.28s
```

`test_complex_tform_schema_and_bignum_repeat` passed before and after — it pins the
re-derived correct behavior (see verified-clean below). Signature/doc guards
(`pixi run pytest tests/test_docs_integrity.py tests/test_public_boundary.py -q`): `26 passed`.

Post-fix behavior smoke (`pixi run python`):

```
ragged: object [array([1, 2]), array([3])]        # dict VLA parity with pre-fix behavior
pathlike: 2  str: 2                               # str + os.PathLike
masked: MaskedColumn int64 [False, True, False]   # r1b-01
zero-row: (0, 3) int64 | (0,) float32             # empty-result dtype/shape preserved
read_extname sig: (path: 'str', hdu: 'Any' = 0)   # r1b-08 mismatch vs docs' hdu=1
```

## Verified clean (re-derived, no finding)

- **Complex (`C`/`M`) schema derivation** (`fits_schema.py:parse_tform`, `build_table_schema_dict`): correct. Astropy ground truth: `format='C'` + 3 complex64 → `TFORM='C'`, `NAXIS1=8`; `format='2C'` → `NAXIS1=16` — the TFORM repeat counts complex values, exactly what `parse_tform("3C").repeat == 3` reports. Pinned by `test_complex_tform_schema_and_bignum_repeat`.
- **No int32 truncation of header-derived sizes in `fits_schema.py`**: repeats/`TFIELDS` parse to Python ints (`parse_tform("4294967296J").repeat == 4294967296`, pinned in the same test); no `int32` casts anywhere in the module (bug class 2 clean).
- **`to_astropy` TUNIT→`.unit` / TNULL→MaskedColumn** (plan candidate): already correct at HEAD for path input via the shared `_table.interop` machinery — pinned by existing `test_table_to_astropy_fidelity` + `test_arrow_nulls_become_masked_column`; the remaining gap was the dict path (r1b-01) and PathLike (r1b-02), both fixed.
- **`read()`/`write()`/`read_tensor()`/`read_hdus()`/`read_batch()`/`open()`/HDU-mutation/checksum/skinny-probe signature fences in `docs/api-core-io.md` match live signatures** (hand diff); `read_extname` is the sole mismatch (r1b-08). `test_docs_integrity.py::test_api_md_core_io_signatures_match_live` + `test_public_boundary.py` green.
- **`io._read_check_cache` positional forwarding** matches the single call site (`_read_pipeline_fallback.py:56`, 12 positional args) and `caches.check_read_cache`'s keyword-only names exactly (`mmap`→`args[10]`, `raw_scale`→`args[11]`). Fragile but correct; left as-is (minimal-diff).
- **`to_polars(rechunk=)`**: valid against the pinned polars (`pl.from_arrow` accepts keyword-only `rechunk`, default `True`); dict and path flows work.
- **`iter_table_columns` on non-numeric `TSCAL`/`TZERO`** raises a loud `ValueError` from `float()` — safe direction (no silent scale default); left as-is.
- **`table.py`, `hdu.py`**: thin re-export façades; `__all__` ↔ imports consistent (table.py 32/32, hdu.py lazy exports + eager `Header`/`Card`); `read_arrow = read` synonym matches `docs/api.md`. No defects.
- **Perf classes**: no measured ≥5% win available in these five files (pure façades/delegation; `fits_schema` hot path keeps its exact-case fast lookups — see r1b-04). No perf changes landed (evidence rule).

## Per-file disposition

| file | depth (full/skim) | finding IDs | status |
|---|---|---|---|
| `src/torchfits/io.py` | full | r1b-05, r1b-06, r1b-09 | fixed (r1b-05, r1b-06); deferred (r1b-09) |
| `src/torchfits/table.py` | full | — | clean |
| `src/torchfits/hdu.py` | full | — | clean |
| `src/torchfits/interop.py` | full | r1b-01, r1b-02, r1b-03, r1b-05 | fixed |
| `src/torchfits/fits_schema.py` | full | r1b-04, r1b-07 | fixed |

Cross-file observations recorded (not in r1b file set): r1b-08 (docs, R15), r1b-10 (`_table/interop.py`, R5).
