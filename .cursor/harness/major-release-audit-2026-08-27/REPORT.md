# torchfits major-release readiness audit — 2026-08-27

Tree: `main` @ `7841dec` (version 1.1.0, torch 2.13 ABI lane). Linux x86_64, Python 3.13, pixi 0.77.
Method: independent review at HEAD. The in-repo 2026-08-26 audit (`.cursor/harness/major-release-audit-2026-08-26/`) was used **only as a list of hypotheses**; every claim below was re-derived from code or live repro at HEAD. Prior-audit findings are cited only where independently confirmed.

Companion files:

- [LEDGER.md](LEDGER.md) — per-file audit ledger (440 tracked files)
- [tracked-files.txt](tracked-files.txt) — `git ls-files` snapshot

---

## 1. Executive assessment

**What it is.** `torchfits` is FITS I/O for PyTorch: image/cube → tensors, FITS tables → Arrow/torch columns, an `HDUList` façade, map/iterable Datasets, transforms (stretch/normalize/RGB/asinh/Lupton), and a shell CLI. The native core is a C++17 nanobind extension over vendored CFITSIO 4.7.0 (mmap + SIMD bswap fast paths, private per-call `fitsfile*` handles, refcounted raw fds, shared-metadata cache with UID rotation on file replacement).

**Architecture (verified).** Lazy root (`__init__.py` imports nothing heavy) → `torchfits.io` / `torchfits.table` → `_io_engine` (read dispatch, write, caches, quantize, subset, HTTP) and `_table` (+ `_table_engine` policy) → `torchfits._C` (nanobind). Python and C++ responsibilities are reasonably separated: path guarding and strategy selection in Python; byte-level decode/scale in C++.

**Strongest aspects.**

- The native read path is genuinely hardened: overflow-checked NAXIS products, truncation checks before mmap/pread, refcounted `RawFdHolder` so invalidation can never close an fd under a live mmap, atomic UID rotation on stat change, GIL released on all expensive native work, mmap tensors are always copies (never views of a mapped region) so no dangling-view lifetime class exists.
- Correctness discipline vs CFITSIO semantics: nulval handling distinguishes native-IEEE (no NaN coercion, Inf/−0 preserved) from BLANK/comp-tile reads; unsigned int16/32 conventions and signed-byte match astropy bit-for-bit (verified live); random-groups images rejected loudly.
- Release engineering is mature: lane-pinned torch ABI, `check-lane`, ASan/UBSan CI on two OSes, docs-contract and changelog tooling as tests, wheel matrix cp310–cp314, honest sdist (documented non-install path).
- Test suite: 1206 passed / 23 skipped in ~5 min at HEAD; examples run in CI (`test_examples_runner`).

**Weakest aspects (all verified at HEAD).**

1. **Silent header data loss for numpy scalars** — `np.int64`, `np.float32/64`, `np.bool_` header values are dropped without error on `write()` (repro'd; A-01). The single most user-hostile defect remaining.
2. **Release gate is red**: `pixi run preflight-push` fails at HEAD (`changelog-check`: docs/changelog.md stale) (A-02).
3. **Test-coupled production code**: the read pipeline detects mocks and caches `hasattr` results so patched tests take different paths than production; debug `print()` in hot dispatch (A-03).
4. **Fragile exception contract** in `write()` — re-raise decisions keyed on error-message substrings (`"uint64"`, `"quantize="`) (A-04).
5. Wide internal surface exported for "backward-compatible imports" (`_table.read.__all__` lists ~25 private names) — churn magnet before a major release (A-11).
6. C++ megafiles (`table_reader.h` 2818 lines, `fits_bindings.cpp` 2489, `fits_file.cpp` 1136) concentrate decode/scale/lifetime logic (A-10) — maintainability, not correctness.

**Prior-audit BLOCKER verification.** All six 08-26 BLOCKER claims were re-tested and are **fixed at HEAD**: `where=` row-set agreement across mmap/backend strategies incl. TNULL (B1/B2), quantized NaN→BLANK→NaN round-trip incl. astropy agreement (B6), `copy` is a true byte copy with same-path refusal (B3/B5), headline bench CSVs exist under `docs/assets/bench/` (B4). `_cpp` now has a closed `__all__` (H1-class); `parity.md` documents complex columns as Partial with the correct `table.read_torch` error name (H2-class).

**Maturity / verdict.** Late-1.x quality: production-capable on Linux/macOS-arm64, torch-2.13 lane, with a documented performance residual. **Not ready to tag a major release today**, but the gap is small and concrete: fix A-01 (silent header loss), A-02 (gate), then the API-surface and test-hygiene items in §7. No known silent wrong-answer remains in the paths probed (image scale/unsigned/int64, table TNULL, quantized NaN, where strategies all verified live against astropy/ground truth).

---

## 2. Repository map

```
Public façade            Internal engine                    Native (C++17/nanobind)
─────────────            ───────────────                    ───────────────────────
__init__.py (lazy)  →    _io_engine/                        cpp_src/
  read/write/open         _read_pipeline.py (dispatch,       fits_bindings.cpp (2489)
  read_tensor/read_*      fast paths, fallback)              fits_detail.h (shared read core)
  hdu.py (re-exports)     write_api.py → _write_helpers,     fits_file.{h,cpp} (FITSFile,
  table.py (namespace)    _hdu_rewrite (atomic rewrite)      SubsetReader)
  interop.py (to_*)       caches.py (Python LRU+stats)       table_reader.h (2818; decode/
  where.py (grammar)      quantize.py (robust int16+BLANK)   filter/scale/VLA/mmap update)
  cache.py                subset.py, http_subset.py          table_ops.cpp (write/VLA ops)
  cpp.py (deprecated →    table_reader_api, table_api,       table_bindings.cpp
    _cpp.py (closed))     table_streaming, batch, device     cache.cpp (no-op stubs post-Option A)
  data/ (Datasets,        image.py, image_meta, hdu_api      hardware.cpp (SIMD dispatch)
    remote prefetch)      paths.py (SSRF guard, bz2 gate)    fits_rw.h / torch_compat.h
  transforms/ (8)         _read_pipeline_fallback.py         internal_utils.h (bswap SIMD)
  cli/ (16 subcommands)   options.py, checksum_api.py        security.h ('|' / 'sh://' block)
                                                        vendored CFITSIO (extern/vendor.sh)
```

Data flow (image read): guard → `_read_unified` → CPU fast path (`read_full_cached`/`nocache`) → C++ `read_tensor_canonical` (mmap+SIMD bswap / pread / CFITSIO fallback) → scale application (uint16/32 via offset-add during bswap; generic via float32 mul/add) → device transfer. Table read: guard → `table.read` → backend policy (`_table_engine`) → C++ pushdown filter or Arrow/torch filter fallback; TNULL/unsigned/scale applied in C++; `apply_fits_nulls` forwarded to every strategy (verified).

---

## 3. Audit coverage

Full ledger in [LEDGER.md](LEDGER.md). Summary: **440 tracked files** — 414 reviewed, 26 excluded (binary/logo assets, vendored KaTeX, `pixi.lock`, bench provenance assets reviewed as data). No file unaccounted for. Depth is not uniform and is stated per group:

- **Deep read + live repros**: `__init__.py`, `io.py`, `cpp.py`, `_cpp.py`, `where.py`, `_where.py`, `_io_engine/_read_pipeline.py`, `write_api.py`, `caches.py`, `quantize.py`, `_write_helpers.py`, `paths.py`, `device.py`, `checksum_api.py`, `options.py`, `_table/read.py`, `_read_where.py`, `_table_engine/*`, `_hdu/*` (all 7), `hdu.py`, `data/datasets.py` (core), `cli/cmds_copy.py`, `cmds_cutout.py`, `cli/common.py` (core).
- **Deep read**: `cpp_src/fits_bindings.cpp` (full), `fits_file.cpp`, `fits_detail.h`, `fits_file.h`, `table_types.h`, `security.h`, `cache.{h,cpp}`, `internal_utils.h`, `table_reader.h` (constructor, scale/TNULL, read_columns, mmap filter paths), `hardware.{h,cpp}`.
- **Medium read**: `header_parser.py`, `http_util.py`, `interop.py`, `cache.py`, `vos_uri.py`, `_string_decode.py`, `_tensor_buffer.py`, `transforms/*` (interfaces + spot), `data/remote.py` (interfaces), remaining `cli/cmds_*.py` (structure + arg handling), `logging.py`.
- **Triage + provenance checks**: `tests/*` (91 files: inventory, full-suite run, targeted deep reads), `benchmarks/*` (27: methodology, suites.py, warmup/run discipline), `docs/*` (35 md: API contract vs `__all__` via docs-contract, benchmarks.md provenance, parity.md), `scripts/*` (51: release-lane, changelog, wheels, bench renderers).
- **Config/build reviewed**: `pyproject.toml`, `pixi.toml`, both `CMakeLists.txt`, 5 GHA workflows, `constraints-wheel.txt`, `.pre-commit-config.yaml`, `SDIST-README.txt`, `packaging/conda/recipe.yaml`.
- **Excluded (data/generated)**: 11 gallery PNG/SVG logos, 14 KaTeX font/css/js files, 7 bench-result CSV/MD provenance assets (existence-checked vs docs citations), `pixi.lock` (pins reviewed via `check-lane`), `site/` + `build/` (untracked output).

**Executed**: full pytest suite (1206 pass / 23 skip, 297 s); `pixi run preflight-push` (ruff+format+mypy+compileall+check-lane pass; **changelog-check fails**); 6 live repro scripts (TNULL/where-strategy matrix, quantized-NaN vs astropy, `copy` semantics, numpy-header write, scaled-int32/int64 vs astropy, `open(mode="update")`). **Not executed**: GHA wheel build (needs tag runners), ASan locally (CI-covered), GPU/MPS paths (no GPU present), docs site build, Windows (unsupported, no job).

---

## 4. Major findings

### F-1 — Silent data loss writing numpy-scalar header values (A-01, HIGH)
`torchfits.write(p, img, header={"INT_NP": np.int64(7), ...})` writes the file **without** `INT_NP`, no error or warning. Repro at HEAD:

```
written header keys: {'INT_PY': 5, 'INT_NP': None, 'FLT_NP32': None, 'FLT_NP64': None, 'BOOL_NP': None, 'STR': 'ok'}
MISSING: ['INT_NP', 'FLT_NP32', 'FLT_NP64', 'BOOL_NP']
```

Root cause: the C++ header-write loops (`fits_file.cpp` `write_hdus` / `write_hdus_compressed_images` header-dict walk, and `fits_bindings.cpp` `write_hdu_header_cards`) dispatch only on `nb::str`, `nb::bool_`, `PyLong_Check`, and `nb::isinstance<float/double>`; numpy scalar types match none and fall through silently. `Header` (a dict subclass) stores values as-is, so Python never normalizes them. Astronomy pipelines are numpy-saturated; this silently corrupts provenance metadata. Fix: normalize in one place — `Header._set_card` / `_merge_fits_write_header` converts `np.generic` via `.item()`; belt-and-braces: raise in C++ on unmatched value types instead of skipping.

### F-2 — Release gate red at HEAD (A-02, MEDIUM, hygiene)
`pixi run preflight-push` → `docs/changelog.md is stale; run pixi run changelog-update`. The project's own pre-push gate must be green before any tag; trivial fix, but nothing currently enforces the gate on `main` pushes.

### F-3 — Tests can silently exercise different code than production (A-03, MEDIUM, test-quality)
`_io_engine/_read_pipeline.py` contains `_is_cpp_module_mocked()` (checks `_mock_name`) and a process-lifetime `_CPP_ATTR_CACHE` for `hasattr` results, existing purely so tests that patch `torchfits.cpp` fall through fast paths. Consequences: (a) tests patch the **deprecated** alias rather than the engine boundary, so mock tests don't pin real dispatch behavior; (b) the attr cache is a production hazard (documented reload footgun). Also debug `print("TORCHFITS_DEBUG_SCALE: ...")` in dispatch instead of the logger. Fix: define the seam (tests patch engine functions), delete mock detection, route debug through logging.

### F-4 — `write()` exception contract keyed on message text (A-04, MEDIUM, API)
`write_api.write` catches `ValueError` and re-raises only if `"uint64" in str(e) or "quantize=" in str(e)`; everything else is re-wrapped as `RuntimeError("Failed to write FITS file ...")`. Any error whose message coincidentally contains those substrings escapes re-typing; typed error handling by callers is impossible. Fix: typed exceptions raised at validation sites; match on type.

### F-5 — Import-time global side effect: `KMP_DUPLICATE_LIB_OK=TRUE` (A-05, LOW/MEDIUM)
`__init__.py:18` does `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` before torch import — a process-wide suppression of a real linker-error class for the *embedding application*. Fixed a real macOS crash; durable fix is to scope it to extension load on Darwin (or document loudly). Decide before 2.0.

### F-6 — Scale-precision asymmetry image vs table paths (A-06, LOW)
Images: unsigned offsets applied during bswap (exact), generic BSCALE/BZERO applied in float32 (`read_full_scaled_cpu`). Tables: scaled in **float64** (`table_reader.h`). No demonstrated divergence vs astropy today (verified: BZERO=2³¹ int32 → uint32 exact; int64 images exact), but fractional-scaled LONGLONG images lose precision where the table path would not. Document; consider float64 accumulation in 2.0.

### F-7 — Public-API surface mostly sealed, with residue (A-09/A-11)
Verified good: root `__all__` closed and lazy; `torchfits._cpp` explicit closed `__all__` with SSRF guards; `torchfits.cpp` warning alias with removal named at 2.0; docs-contract test pins `docs/api.md` to the root surface. Residue: `_table/read.py` `__all__` re-exports ~25 private names; `write_api.py` similar; `ReadOptions.handle_cache_capacity` documented no-op; `clear_file_cache(handles=)` accepted-and-ignored. For the major release: strip from module `__all__`, keep attribute access one deprecation cycle.

### F-8 — C++ megafiles (A-10, MEDIUM, maintainability)
`table_reader.h` (2818) mixes schema analysis, buffered decode, mmap decode, pushdown filters, VLA, mmap row updates, thread-local reader LRU. `fits_bindings.cpp` (2489) mixes read kernels, numpy kernels, write, checksums, metadata. Reviewed logic was correct (overflow/truncation guards, RAII), but every future fix lands in a template-dense file with high regression blast radius. Split by concern before/with 2.0; not a 1.x-patch item.

---

## 5. Complete issue register

Severity reflects release risk. "Verified" = confirmed at HEAD by repro or code reading.

| ID | Sev | Category | Files | Description | Fix & validation |
|---|---|---|---|---|---|
| A-01 | **HIGH** | correctness/API (data loss) | `cpp_src/fits_file.cpp` (`write_hdus`), `cpp_src/fits_bindings.cpp` (`write_hdu_header_cards`), `_hdu/header.py`, `_io_engine/_write_helpers.py` | numpy scalar header values (`np.int64/float32/float64/bool_`) silently dropped on write. **Verified (repro).** | Normalize `np.generic→.item()` in `Header._set_card`/`_merge_fits_write_header`; raise `TypeError` on unsupported C++ value types. Test: round-trip header with each numpy scalar kind. |
| A-02 | MEDIUM | release engineering | `docs/changelog.md` | `preflight-push`/`changelog-check` fails at HEAD ("changelog stale"). **Verified.** | Run `pixi run changelog-update`; add CI job on `main` running `preflight-push`. |
| A-03 | MEDIUM | test-quality/architecture | `_io_engine/_read_pipeline.py` | Mock detection (`_is_cpp_module_mocked`), `hasattr` result cache, `print()` debug in dispatch. Tests can pass without covering production dispatch. **Verified (code).** | Remove mock sniffing; tests patch engine functions; logger for debug; delete attr cache. Migrate mock-based tests. |
| A-04 | MEDIUM | API/error semantics | `_io_engine/write_api.py` | Re-raise decisions keyed on `str(e)` substrings; non-ValueError failures re-typed RuntimeError. **Verified (code).** | Typed exceptions at validation sites; match on type. Tests assert exception types. |
| A-05 | LOW/MED | env/side effects | `__init__.py:18` | Global `KMP_DUPLICATE_LIB_OK=TRUE` setdefault affects host process. **Verified (code).** | Scope to extension load on Darwin, or document prominently. |
| A-06 | LOW | numerical consistency | `cpp_src/fits_bindings.cpp` (`read_full_scaled_cpu`), `_read_pipeline.py`, `table_reader.h` | Image scale float32 vs table scale float64. **Verified (no divergence vs astropy today).** | Doc note; float64 for LONGLONG images in 2.0 + regression test. |
| A-07 | LOW | native hygiene | `cpp_src/table_reader.h:481,508` | Error paths `std::cerr` diagnostics before throwing. | Move message into exception. |
| A-08 | LOW | API semantics | `_hdu/dataview.py:83-99` | Out-of-range int index in `DataView.__getitem__` clamps to empty slice instead of IndexError. | Raise on out-of-range ints + test. |
| A-09 | LOW | API grammar | `_where.py`, `where.py`, docs/api-tables.md | Public where-grammar uses underscore functions (`_between`, `_isnull`); `not` on float columns excludes NaN rows silently (documented but surprising). | 2.0: non-underscore aliases, keep old deprecated. Grammar-matrix tests. |
| A-10 | MEDIUM | maintainability | `cpp_src/table_reader.h`, `fits_bindings.cpp`, `fits_file.cpp` | Megafiles concentrating decode/filter/lifetime logic. | Split by concern. Validate: full suite + ASan CI green after split. |
| A-11 | MEDIUM | API surface | `_table/read.py` `__all__`, `_io_engine/write_api.py` `__all__` | ~25 private names re-exported "backward-compat" in module `__all__`. | Remove from `__all__` (keep attributes) with changelog note. |
| A-12 | LOW | error messages | `_io_engine/device.py` | `validate_device` message omits accepted `mps:N`. **Verified.** | Fix message. |
| A-13 | LOW | semantics/docs | `_io_engine/device.py` (`to_device`) | MPS silently downcasts float64→float32 / complex128→complex64. | Document in compatibility.md; optional warning. |
| A-14 | LOW | edge case | `_table/engine.py` (`_read_ranges_as_chunk`) | Empty coalesced range → tensor columns surface zeros, list columns surface `None` for those rows. | Pre-allocate all columns up-front; test with empty range. |
| A-15 | LOW | platform | CI/docs | No Windows support (documented); sanitizer workflow Linux+macOS only. | Keep documented; release gate states it. |
| A-16 | CLEANUP | legacy knobs | `caches.py` (`handles=` no-op), `options.py` (`handle_cache_capacity`), `cache.cpp` stubs | Accepted-and-ignored parameters from removed handle-cache era. | 2.0 removal; deprecation warnings now. |
| A-17 | CLEANUP | debug knobs | `TORCHFITS_DEBUG_SCALE/COLD_NOMMAP/COLD_NOCACHE/TABLE_BUFFERED/SHARED_META_VALIDATE...` | Env knobs scattered, partially undocumented. | Inventory in docs/compatibility.md. |
| A-18 | LOW | thread-safety | `_io_engine/caches.py` (`_open_hdulist_registry`) | Registry mutated without `cache_lock` (small fanout documented). | Guard with lock or weakrefs. |
| A-19 | CLEANUP | tests | `tests/` | Mock-based tests patch deprecated `torchfits.cpp` alias (see A-03); deep-review wave tests encode implementation details. | Migrate patches to engine seam. |

**Verified fixed at HEAD** (prior 08-26 register, independently re-derived): B1/B2 (`where=`/TNULL row-set agreement across mmap×backend×read/read_torch/scan — live matrix all-consistent), B3/B5 (`copy` byte-exact, same-path refused), B4 (bench headline CSVs present in git), B6 (quantized NaN→NaN on read/read_tensor, astropy agrees), H1 (`_cpp` closed surface), H2 (parity.md complex=Partial, error names `table.read_torch`).

**No issue manufactured for**: performance (documented narrow-table residual; hot paths release GIL and use SIMD copy — no speculative work recommended), zero-copy claims (mmap tensors are copies by design — verified in code; no dangling-view class exists).

---

## 6. Recommended major-release API

**Retain unchanged**: root `read/write/open/read_tensor/read_*/verify_checksums/insert_hdu/replace_hdu/delete_hdu`, `to_pandas/to_arrow/to_polars/to_astropy`, `Header/Card/HDUList/TensorHDU/TableHDU/TableHDURef`, `torchfits.table.*` (read/read_torch/scan/scanner/schema/dataset/mutations/interop), `torchfits.data` Datasets, `transforms`, CLI commands and exit codes, `where=` grammar semantics as documented (incl. TNULL behavior).

**Change (behavior fixes, pre-2.0)**: A-01 (numpy header values — bugfix, not API break), A-04 (typed write errors — breaking only for code catching `RuntimeError` broadly; net positive), A-08 (`DataView` IndexError).

**Remove at 2.0 (deprecate now)**: `torchfits.cpp` alias (already warning, removal promised), `ReadOptions.handle_cache_capacity`, `clear_file_cache(handles=)`, `_cpp`-mirror names in `cpp.py`.

**Make internal**: strip `_`-prefixed re-exports from `_table/read.py` / `write_api.py` `__all__` (A-11); `TableDataAccessor` stays public but documented as read-only view.

**Grammar**: add non-underscore `between`/`isnull` aliases (A-09); keep old ones working.

No renaming of primary entry points is warranted — the surface is consistent (`read_*` family, destination-qualified table API) and heavily contract-tested.

---

## 7. Implementation plan (ordered to avoid rework)

**Phase 1 — Correctness / release blockers**
- Issues: A-01, A-02.
- Files: `_hdu/header.py`, `_io_engine/_write_helpers.py`, `cpp_src/fits_file.cpp`, `cpp_src/fits_bindings.cpp`, `docs/changelog.md`, `.github/workflows/ci.yml`.
- Validation: new round-trip header tests (all numpy scalar kinds); `pixi run preflight-push` green; full suite green.

**Phase 2 — Error semantics & API hardening** (independent of Phase 1)
- Issues: A-04, A-08, A-12, A-11, A-09 + docs.
- Files: `_io_engine/write_api.py`, `_io_engine/device.py`, `_hdu/dataview.py`, `_table/read.py`, `_where.py`, `where.py`, `docs/api-tables.md`.
- Validation: exception-type tests; docs-contract/docs-links green.

**Phase 3 — Test seam & hygiene** (independent)
- Issues: A-03, A-19, A-18, A-14, A-07.
- Files: `_io_engine/_read_pipeline.py`, `_io_engine/caches.py`, `_table/engine.py`, `cpp_src/table_reader.h`, affected `tests/*`.
- Validation: full suite Linux+macOS; ASan workflow green.

**Phase 4 — Native maintainability** (after Phase 3 so splits are behavior-neutral)
- Issues: A-10 (splits, no logic changes); A-06 float64-scale decision documented (change in 2.0).
- Validation: byte-identical outputs on a golden-file read matrix; ASan+UBSan CI; `bench-release-scorecard` within noise.

**Phase 5 — Env/docs/packaging polish**
- Issues: A-05, A-13, A-16, A-17, A-15 doc statement.
- Files: `__init__.py`, `docs/{compatibility,install}.md`, `caches.py`, `options.py`.
- Validation: `pixi run docs-contract`, `docs-links`, `release-gate`.

**Phase 6 — Release validation**
- Full `pixi run ci-local`; wheel workflow dry-run (`workflow_dispatch`); clean-venv wheel install smoke; examples runner; §8 checklist.

---

## 8. Release gate checklist (must pass before tagging)

- [ ] `pixi run preflight-push` green on the release commit (A-02 closed; includes ruff/mypy/lane/changelog).
- [ ] Full pytest suite green on Linux + macOS, cp310–cp314 (CI matrix), incl. examples runner.
- [ ] Header write round-trips all scalar types incl. numpy (A-01 regression tests).
- [ ] Write-path exception types covered by tests (A-04).
- [ ] ASan/UBSan workflow green on the release PR.
- [ ] `where=` strategy matrix (mmap on/off × read/read_torch/scan × TNULL/unsigned/out-of-range literals) returns identical row sets.
- [ ] `torchfits copy` byte-identical (`cmp`) and same-path refused.
- [ ] Quantized NaN→BLANK→NaN round-trip vs astropy; native IEEE Inf/−0 preserved.
- [ ] Wheel matrix: Linux x86_64+aarch64, macOS arm64, cp310–cp314, torch 2.13 lane; `check-lane`/`check-torch-pins` green.
- [ ] Clean-venv `pip install torchfits` smoke; sdist documented as non-install.
- [ ] `docs-contract`, `docs-links` green; no root symbol without changelog entry; `parity.md` matches behavior.
- [ ] Known limitations stated: no Windows; MPS float64/complex128 downcast; sdist non-install; narrow-table bench residual.
- [ ] Version triplet consistent (`pyproject`/`pixi.toml`/`__init__.py`/conda recipe); changelog date stamped; release notes via `scripts/release_notes.py`.
- [ ] Headline benchmark claims cite run IDs present under `docs/assets/bench/`.
