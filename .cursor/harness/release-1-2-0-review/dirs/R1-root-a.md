# R1 slice A — `src/torchfits/` root facade (`__init__.py`, `cpp.py`, `_cpp.py`, `_C.pyi`, `cache.py`, `logging.py`)

Baseline HEAD `1bb6958` worktree. Targeted suite: `tests/test_cache.py tests/test_cache_config.py
tests/test_clear_all_caches.py tests/test_native_stub.py tests/test_public_boundary.py
tests/test_interop_import.py` — 62 passed before, 67 passed after (6 new tests added,
1 implementation-pin test deleted per repo test policy). Files edited: `src/torchfits/cache.py`,
`tests/test_cache.py` only.

## Findings

| ID | Sev | Class | Files:symbols | Description (with repro) | Fix & validation | Status |
|---|---|---|---|---|---|---|
| r1a-01 | MAJOR | 5 error contract / 8 docs (deterministic failure on valid input) | `cache.py:get_cache_manager` (`:307` pre-fix), `cache.py:configure_for_environment` (`:315`), `CacheManager.optimize_for_dataset` (`:292`) | UnifiedCache residue: three internal call sites invoked the **deprecated no-op** `CacheManager.configure_cpp_cache()`, so every process's first I/O entry (`_ensure_runtime_init` → `configure_for_environment` → `get_cache_manager`) emitted a `DeprecationWarning` from library code, and every `make_loader(optimize_cache=True)` emitted another. Under the common `-W error::DeprecationWarning` (or pytest `filterwarnings = error`) config this makes valid `torchfits.write`/`read` **raise**. Repro (pre-fix): `pixi run python -W error::DeprecationWarning -c "import torch,torchfits,tempfile; p=tempfile.mktemp(suffix='.fits'); torchfits.write_tensor(p, torch.ones(2,2), overwrite=True); torchfits.read(p)"` → traceback `File ".../cache.py", line 227, in configure_cpp_cache warnings.warn(...) / DeprecationWarning: CacheManager.configure_cpp_cache is a deprecated no-op…` | Deleted the three internal calls to the deprecated no-op (they configured nothing after Option A); `configure_cpp_cache` remains the user-facing deprecated no-op and still warns on explicit calls. Failing-first (`pixi run pytest tests/test_cache.py::TestDeprecatedNoOpContract -q`, unfixed): `test_configure_for_environment_emits_no_deprecation`, `test_get_cache_manager_creation_emits_no_deprecation`, `test_optimize_for_dataset_emits_no_deprecation` FAILED (recorded `DeprecationWarning` lists, "Left contains 2 more items…"); contract-preservation test passed. After fix: `6 passed in 1.26s`; the `-W error` repro above now prints `read ok under -W error::DeprecationWarning (2, 2)`. | fixed |
| r1a-02 | MAJOR | 1 silent wrong result / 4 race / 8 docs | `cache.py:configure_cache` (`:380-394` pre-fix) | `configure_cache` is documented a "Deprecated no-op" (docstring + `docs/changelog.md` 1.1.2: "remain as documented no-ops emitting DeprecationWarning for one cycle") but actually replaced the global `_cache_manager` with phantom-knob `CacheConfig` — observable via `get_cache_stats()["config"]` — through an **unlocked** `global _cache_manager` write racing `get_cache_manager`'s double-checked locking (registry race; two threads could hold different managers). Repro (pre-fix): `configure_cache(7, 8, 9)` then `get_cache_manager()` returns a different object with `max_files == 7`. | Made it a true no-op matching the documented contract: arguments `del`-marked ignored, DeprecationWarning preserved verbatim; the unlocked global write is gone (`_cache_manager` is now only written under `_cache_manager_lock`). Failing-first (`test_configure_cache_is_documented_noop`, unfixed): `AssertionError: assert <CacheManager …8e90> is <CacheManager …8c30>` — manager replaced. After fix: passes (`6 passed`). Explicit-call warning contract pinned by `test_explicit_deprecated_calls_still_warn`. | fixed |
| r1a-03 | MINOR | 8 docs faithfulness (stale UnifiedCache residue) | `cache.py:clear_all_caches`, `cache.py:CacheConfig`, `CacheManager.optimize_for_dataset` | Stale/phantom docstrings left by the shared-handle removal: (a) `clear_all_caches` claimed to clear "C++ handles" / "``SharedReadMeta`` and handle-pool state" — the handle pool no longer exists; (b) `CacheConfig` disclosed `max_files`/`max_memory_mb` as unused but omitted `prefetch_enabled`, which is equally phantom (no live reader in `src/` — verified by grep `prefetch_enabled` across `src/torchfits` + `tests`: only `cache.py` and tests touch it); (c) `optimize_for_dataset` said "Optimize cache settings" without disclosing the hints size no live cache. | Docstring-only fixes (code proof = text diff below): `clear_all_caches` → "in-process LRUs, C++ ``SharedReadMeta``, *and* disk"; `CacheConfig` → "`max_files``, ``max_memory_mb``, and ``prefetch_enabled`` are unused policy hints… ``disk_cache_gb`` sizes ``optimize_for_dataset``'s heuristic and describes on-disk remote/sample roots."; `optimize_for_dataset` → "Tune the policy hints… these hints do not size live caches (see :class:`CacheConfig`)". Untestable-by-design (tests must not pin docstrings); covered behaviorally by r1a-01/r1a-02 tests. | fixed |
| r1a-04 | MINOR | 5 error contract (silent `except Exception: pass`) | `cache.py:CacheManager.get_stats` (`:241-246`), `cache.py:stats` (`:371-376`) | Both stats aggregators swallowed **any** exception from `_io_engine.caches.get_cache_performance()` and silently returned zeroed/absent counters (wrong result on a broken engine). Inconsistent with this module's own `CacheManager.clear()` convention (`except (ImportError, AttributeError)`). Repro (pre-fix): monkeypatch `get_cache_performance` to raise `RuntimeError("engine broken")`; `get_cache_stats()` returns `{"hits": 0, …}` and `stats()` silently omits `"io"`. | Narrowed both to `except (ImportError, AttributeError)` (module-shape failures only); real engine errors now surface. Failing-first (`test_cache_stats_surface_engine_errors`, unfixed): `Failed: DID NOT RAISE RuntimeError`. After fix: passes (`6 passed`). | fixed |

### Failures/outputs (verbatim excerpts)

Failing-first, `pixi run pytest "tests/test_cache.py::TestDeprecatedNoOpContract" -q` on unfixed code:

```
FAILED tests/test_cache.py::TestDeprecatedNoOpContract::test_configure_for_environment_emits_no_deprecation - assert [<warnings.Wa...] == []
  Left contains 2 more items, first extra item: <warnings.WarningMessage object at 0x7fdbc1ad1a90>
FAILED tests/test_cache.py::TestDeprecatedNoOpContract::test_get_cache_manager_creation_emits_no_deprecation - assert [<warnings.Wa...] == []
  Left contains one more item: <warnings.WarningMessage object at 0x7fdb9bd6cb90>
FAILED tests/test_cache.py::TestDeprecatedNoOpContract::test_optimize_for_dataset_emits_no_deprecation - assert [<warnings.Wa...] == []
  Left contains one more item: <warnings.WarningMessage object at 0x7fdb9bda8770>
FAILED tests/test_cache.py::TestDeprecatedNoOpContract::test_configure_cache_is_documented_noop - AssertionError: assert <torchfits.cache.CacheManager object at 0x7fdb9bda8e90> is <torchfits.cache.CacheManager object at 0x7fdb9bda8c30>
 +  where <torchfits.cache.CacheManager object at 0x7fdb9bda8e90> = <function get_cache_manager at 0x7fdbc3563ec0>()
FAILED tests/test_cache.py::TestDeprecatedNoOpContract::test_cache_stats_surface_engine_errors - Failed: DID NOT RAISE RuntimeError
5 failed, 1 passed in 1.38s
```

After fix, same selector: `6 passed in 1.26s`. Full allowed set after fix:

```
67 passed, 308 warnings in 3.34s
```

`pixi run python -W error::DeprecationWarning` first-read repro after fix: `read ok under -W error::DeprecationWarning (2, 2)`.

Policy-mandated test deletions (implementation pins / mock echoes — repo test policy "assert what a
consumer observes", not wiring):
- `tests/test_cache.py::test_configure_for_environment` — asserted only `get_cache_manager` /
  `configure_cpp_cache` call wiring via `MagicMock` echo; deleted (behavior contract replaced by
  `test_configure_for_environment_emits_no_deprecation`).
- `mock_configure_cpp_cache.assert_called_once()` pins removed from the three
  `test_optimize_for_dataset_*` tests (their observable `manager.config.*` assertions kept).
- Stale comment in `test_get_cache_manager_singleton` referencing `configure_cpp_cache` removed.

## Verified — no defect (with evidence)

- **`_C.pyi` stub drift gate** (`cpp`-adjacent): `pixi run check-stub` → `[ OK ] src/torchfits/_C.pyi
  matches the extension`. Committed stub is current; no regeneration needed. `tests/test_native_stub.py`
  (live-extension equality + declared-return-type accuracy) green.
- **`cpp-seal-all` invariant holds**: `tests/test_public_boundary.py` pins `HAS_BZIP2` rejection and
  `__dir__` sealing. Throwaway AST check of `_cpp.py` vs `_C.pyi`: `__all__` = 55 unique names, every
  one present in the extension stub (and import-time `getattr(_C, _name)` fails fast on drift);
  every stub path-first callable in `__all__` is in `_PATH_FIRST`/`_PATH_LIST_FIRST` (0 unguarded);
  `FITSFile`/`SubsetReader`/`TableReader` ctors in `_PATH_CTORS`. Sealed-private `_C` names (intentionally
  not exported): `HAS_BZIP2`, `echo_tensor`, `evict_cached_reader`, `open_fits_mmap_reader`,
  `read_fits_table_rows_mmap_from_reader`, `read_fits_table_from_handle`,
  `read_fits_table_rows_from_handle`, `read_fits_table_rows_numpy_from_handle`.
  `guard_cfitsio_remote_path` coerces non-str inputs (`str(path)`) so the `TableReader(file_obj, …)`
  ctor overload passes the guard harmlessly (and the only handle-based ctor caller,
  `_io_engine/table_streaming.py:137`, uses `_C` directly). `cpp.py` delegation complete: every
  attribute resolves through `_impl` with a per-access `DeprecationWarning` (changelog contract),
  dunder probes raise `AttributeError` without warning.
- **`unified-cache-stubs` invariant current shape**: no `get_or_open_cached` anywhere in `src/`;
  `configure_cache`/`get_cache_size`/`clear_file_cache` are absent from the native surface
  (`test_public_boundary` pins this); live shared state is `SharedReadMeta`
  (`cpp_src/fits_detail.h`, cleared via `clear_shared_read_meta_cache`), documented at
  `cpp_src/fits_handle.h:9-12`. The residue was **Python-side only** — the phantom knobs /
  stale docstrings / deprecated-no-op self-calls fixed in r1a-01..r1a-03.
- **Cache invalidation on file replacement / `id()` keys (class 3)**: none of my files hold
  path-keyed or `id()`-keyed caches — `__init__._ATTR_CACHE` is name-keyed and bounded by
  `__all__`; `CacheConfig._ENV_CACHE` is keyed by the env-detection signature (bounded space).
  UID rotation / `path_signature` / `invalidate_path_caches` live in `_io_engine/caches.py`
  (`path_signature:128`, `invalidate_path_caches:432`) — R4's slice; nothing to fix here.
- **`logging.py` import side effects honest**: the only import-time effect is
  `logger.addHandler(logging.NullHandler())`, matching the changelog claim ("Library logger uses
  `NullHandler` (no import-time StreamHandler)"). No env-var handling exists or is documented
  (grep `TORCHFITS_LOG|LOG_LEVEL|torchfits.logging` in `docs/` finds only the changelog line) —
  nothing to make honest. (Bare note, not fixed: a `reload()` of the module would stack duplicate
  NullHandlers; standard-library idiom, zero impact in normal imports.)
- **`__init__.py` env side effect honest**: `KMP_DUPLICATE_LIB_OK=TRUE` `setdefault` guarded to
  Darwin with an honest comment ("harmless elsewhere but process-wide so scope to Darwin");
  matches playbook `kmp-duplicate-lib-ok` (behavior stays per plan assumption 3). Lazy
  `__getattr__` uses an `RLock` with double-checked name cache — no race, no `id()` keys.
- **Env vars in my files exist in `src/` and match their docstrings**: `TORCHFITS_CACHE_DIR`,
  `XDG_CACHE_HOME`, `TORCHFITS_REMOTE_CACHE`, `TORCHFITS_SAMPLE_CACHE` — `cache_root()`'s
  documented resolution order matches the code exactly.
- **Perf rubric (classes 1-6)**: reviewed — `_ATTR_CACHE` fast path is a dict hit after first
  resolution; `for_environment` is memoised by env signature; `_detect_gpu` delegates to torch's
  cached probe; `stats()`'s double `get_cache_performance()` is a lock+dict read (no measurable
  cost). No perf changes landed (evidence rule: nothing met the ≥5%/count-proof bar — by design
  these are cold-path policy façades).

## Cross-file notes (other slices own the fix)

- `docs/benchmarks.md:389-391` ("so handle caches stay warm") is stale handle-pool residue → R15.
- `data/__init__.py:616-618` `make_loader` docstring ("pre-warm handle and file caches") is stale
  for the same reason → R3.
- `docs/api-core-io.md` / `docs/architecture.md` cache sections become *more* accurate with
  r1a-02 (they already call `configure_cache`/`configure_cpp_cache` documented no-ops); the only
  doc delta owed at R15 is the changelog `Fixed` bullet for r1a-01/r1a-02 (generated by
  `pixi run changelog-update` from the fix commits) — no hand-editing needed here.

## Deferred

None from this slice. (Known A-19 — `tests/test_cache.py` still exercises the deprecated
`torchfits.cpp` alias via `pytest.importorskip("torchfits.cpp")`, ~300 of the 308 suite warnings —
is explicitly R11's assignment ("migrate patches to the `_io_engine._read_pipeline` seam") and was
left untouched to avoid colliding with that migration.)

## Per-file disposition

| file | depth (full/skim) | finding IDs | status |
|---|---|---|---|
| `src/torchfits/__init__.py` | full | — (verified: seal surface unchanged, KMP honest, lazy init race-free) | clean |
| `src/torchfits/cpp.py` | full | — (verified: complete delegation, per-access DeprecationWarning contract) | clean |
| `src/torchfits/_cpp.py` | full | — (verified: `cpp-seal-all`, guard coverage complete incl. ctor/list paths) | clean |
| `src/torchfits/_C.pyi` | full (via `pixi run check-stub` + `tests/test_native_stub.py`) | — (verified: zero stub drift) | clean |
| `src/torchfits/cache.py` | full | r1a-01, r1a-02, r1a-03, r1a-04 | fixed |
| `src/torchfits/logging.py` | full | — (verified: NullHandler-only import effect, no env claims) | clean |
