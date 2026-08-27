# torchfits major-release readiness audit

Date: 2026-08-26  
Tree: `main` @ `1a70cd1` (`1.1.0`, PyTorch 2.13 ABI lane)  
Working tree also has **uncommitted** macOS `KMP_DUPLICATE_LIB_OK` WIP (not in the tag).  
Scope: exhaustive review of tracked product/code/docs/CI; no product fixes in this pass.

**Verdict: Block a freeze / next tag until the BLOCKER register is closed.** `1.1.0` is already cut. Area audits plus live repros found silent catalog-row disagreements, quantized NaNs that round-trip as finite pixels, a CLI `copy` that is not a binary copy, and a bench headline citing CSVs that are not in git.

Companion files:

- [LEDGER.md](LEDGER.md) — every tracked file classified reviewed vs excluded
- [tracked-files.txt](tracked-files.txt) — `git ls-files` dump
- Area notes: [agent-table.md](agent-table.md), [agent-py-surface.md](agent-py-surface.md), [agent-tests-docs.md](agent-tests-docs.md), [agent-cpp.md](agent-cpp.md), [agent-packaging.md](agent-packaging.md), [agent-io.md](agent-io.md)

---

## 1. Executive assessment

**What it is.** `torchfits` is FITS I/O for PyTorch: image/cube tensors, Arrow/tensor tables, MEF `open()`, Datasets, a shell CLI, and vendored CFITSIO behind a nanobind extension.

**Architecture.** Lazy public root (`src/torchfits/__init__.py`) → `_io_engine` / `_table` → `torchfits._C` (C++17, CFITSIO 4.7.0 pinned) → mmap/pread or `fits_read_*`. Private handles per call (no shared `fitsfile*`). Shared metadata is `SharedReadMeta` keyed by path with UID rotation on file replacement.

**Strongest aspects.**

- Clear product boundary: FITS bytes ↔ tensors/tables; WCS/units/source extraction stay out.
- Real native engine (mmap + SIMD bswap, buffered table reads, SSRF guards, checksums, truncated-file checks, GIL release on hot paths).
- Release engineering is unusually mature: lane pins, `check-lane`, changelog tooling, docs-contract tests, ASan/UBSan workflow, wheel matrix, honesty about remaining bench deficit.
- 1.1 already landed several previous-audit items (GIL on `read_full_numpy`, table truncation, mmap-update strides, `to_astropy` MaskedColumn on the **table** path, remote `If-Range` + flock, CLI `-J` per-worker transforms).

**Weakest aspects.**

- Public surface is wide and leaky: `torchfits.cpp` still in `__all__` (deprecated), `torchfits._cpp.__getattr__` re-exports **all** `_C` symbols, Dataset zoo, dual `to_astropy`.
- Table `where=` is not only two dialects: **`read` vs `read_torch` return different row sets** for TNULL sentinels and out-of-range integer literals (live 2026-08-26).
- Quantized NaNs round-trip as finite pixels on `torchfits.read` (BLANK scaled; astropy masks).
- C++ megafiles (`table_reader.h` ~2762 lines, `fits_bindings.cpp` ~2475) concentrate lifetime/endian/scale logic.
- macOS OpenMP duplicate-lib abort is real on this machine unless `KMP_DUPLICATE_LIB_OK` is set; the import-time fix is uncommitted.
- Docs/parity overclaim complex columns as fully supported while Arrow `table.read` raises `NotImplementedError`, and the error text names a nonexistent `torchfits.read_torch()`.

**Maturity.** Late 1.x: production-capable for Linux/macOS arm64 wheels on the 2.13 lane, with a documented performance residual (narrow-table `mmap=False` vs fitsio). Not a 2.0 (that roadmap is GPUDirect / native codecs).

**Ready for a major release today?** **No.** Do not freeze or retag until BLOCKERs B1–B6 are fixed or the published claims are withdrawn. 1.1.0 already shipped with B4 (headline CSVs “pending mirroring”). Do not start 2.0 until the 1.x façade is sealed.

---

## 2. Repository map

```
Python façade          Native
─────────────          ──────
__init__.py (lazy)  →  _cpp.py wraps _C (+ path SSRF)
io.py (re-export)   →  _io_engine/*  →  cpp_src/*.cpp/.h  →  vendored CFITSIO
table.py            →  _table/* + _table_engine
hdu.py              →  _hdu/*
transforms/         →  (pure torch)
data/               →  Datasets + remote HTTP/vos
cli/                →  same I/O façades
cache.py            →  disk roots / policy; live LRU in _io_engine/caches.py
```

**Intended public API** (from `__all__`, `docs/api.md`, `tests/test_public_boundary.py`):

- Root I/O: `read`, `read_tensor`, `write`, `write_tensor`, `open`, skinny metadata, subset/batch/HDU mutation, checksums, cache clears, `to_*` interop.
- Namespaces: `table`, `hdu`, `transforms`, `data`, `cache`, `where`, deprecated `cpp`.
- CLI: `torchfits` console script (`cli/main.py`).

**Supported matrix (declared):** Python 3.10–3.14, PyTorch 2.13.x wheels (source ≥2.10), Linux x86_64/aarch64 + macOS arm64. Windows and x86_64 macOS: no wheels. Optional: pandas/polars/duckdb/astropy/fitsio/matplotlib.

**Release workflow:** `scripts/torch_lanes.json` → `release_lane.py --apply`; tag `v*` → `build_wheels.yml` → PyPI. Local gates: `preflight-push` / `ci-local` / `release-gate`.

---

## 3. Audit coverage

429 tracked files. 404 reviewed, 25 excluded. Full table: [LEDGER.md](LEDGER.md).

### Executed

| Check | Result |
|---|---|
| `pixi run preflight-push` | pass (ruff, mypy, compileall, check-lane, changelog-check) |
| `pixi run docs-contract` + `docs-links` | pass (25 tests); external-link **warnings**: fonts.gstatic 404, gnuastro TLS timeout, legacysurvey 400 |
| `pixi run -e test -- pip install -e . --no-build-isolation` | rebuilt after 27-commit FF (required: default vs test env do not share `_C`) |
| `pixi run -e test -- pytest tests/` **without** `KMP_DUPLICATE_LIB_OK` | **abort** (`Fatal Python error: Aborted`) inside `write()` ~30% — macOS libomp |
| `KMP_DUPLICATE_LIB_OK=TRUE pixi run -e test -- pytest tests/` | **1170 passed, 9 skipped**, 365 warnings, 270s (after test-env rebuild) |
| Full examples / exhaustive benches / ASan locally | not re-run (examples need network/samples; benches are multi-host; sanitizer is GHA) |
| Clean sdist pip install | not exercised; `SDIST-README.txt` says sdist is **not** an install path |

### Coverage depth (honest)

- **Python `src/torchfits/`:** every module accounted; all public façades and engine files read or grepped; largest table/I/O files read in sections.
- **C++:** native-layer pass in [agent-cpp.md](agent-cpp.md) (all `cpp_src` files; megafiles read in chunks). Residual: no sanitizer run in that pass.
- **Tests:** complete inventory; deep: `test_public_boundary`, `test_docs_integrity`, `test_package_isolation`, `conftest`, `test_examples_runner`.
- **Docs:** all `docs/*.md` inventoried; `api.md`, `architecture.md`, `install.md`, `cli.md`, `parity.md`, `release.md`, `roadmap.md`, `changelog.md` (1.1.0), `benchmarks.md` (headline) read.
- **CI/packaging/scripts/extern:** workflows, `pyproject.toml`, `pixi.toml` (tasks), conda recipe, `vendor.sh` header, `torch_lanes.json`.

### Exclusions (why)

KaTeX fonts/JS/CSS; logo PNG/SVG; `pixi.lock` pin-by-pin (lane scripts + pyproject reviewed). Vendored CFITSIO **tarball** is gitignored (`extern/cfitsio/`); pin+patches+license reviewed.

---

## 4. Major findings

### 4.1 The façade is not sealed

`tests/test_public_boundary.py` checks `__all__` and a deprecation warning on `torchfits.cpp`, but `src/torchfits/_cpp.py` `__getattr__` forwards **any** name to `torchfits._C`. New nanobind symbols become importable without a review. `__dir__` includes `dir(_C)`. A freeze that only audits `__all__` is false.

`torchfits.cpp` remains a root namespace in `__all__`. Removing it is a breaking 2.0 (or 1.2 with a hard warning window). Leaving it is accidental public API.

### 4.2 Table API tells two stories

- `table.read` / `scan`: full WHERE dialect, Arrow, **rejects** TFORM `C`/`M`.
- `table.read_torch`: simple compare / BETWEEN / AND only; tensors; complex OK.
- Error on complex Arrow reads says `use torchfits.read_torch()` — **that symbol is not on the root module**. Correct name is `torchfits.table.read_torch`.
- `docs/parity.md` lists complex columns as **Supported** without saying Arrow cannot serve them.
- `TableHDURef.head(n)` always sets `row_slice=slice(0, n)`, discarding an existing window (rows 100–200 then `head(10)` → file rows 0–10).

### 4.3 CLI exit codes collide with real failures

Documented: `1` = diff, `2` = usage, `3` = I/O, `4` = checksum fail.  
`main()` maps `KeyboardInterrupt` → `2` (usage). Uncaught `Exception` (not `CliError`/`OSError`) becomes Python exit `1`, same as `diff`. `stats --json` uses `json.dumps` default `allow_nan=True` (invalid JSON). `diff` still calls `tensor.min()` on possibly unsigned images; `stats` was already patched to upcast.

### 4.4 Native layer is concentrated and mostly hardened

1.1 work is visible: GIL release around `read_full_numpy`, `ensure_extent_within_file`, mmap-update index via `tensor.stride()`, SharedReadMeta UID rotation (R7-CPP2 addressed in `fits_detail.h`). Residual risk: `col.repeat = (int)repeat_long` truncation; C++ `has_cfitsio_extended_filename_syntax` is last-slash-aware while Python `cfitsio_base_path` / `has_cfitsio_filter` use naive `'[' in path`. Megafiles remain a change-risk for any 1.2 table decode rewrite.

### 4.5 macOS OpenMP is a product footgun

Verified: full pytest without `KMP_DUPLICATE_LIB_OK` aborted the interpreter during `write()`. `pixi.toml` `test`/`release-gate` tasks prefix the env var; `pixi run -e test -- pytest` and a user `import torch` before `import torchfits` do not. Uncommitted WIP setdefaults in `__init__.py` + pixi `activation.env`. GHA macOS jobs generally omit the var (runners may lack a second libomp).

### 4.6 Docs/CI drift, not fantasy performance

Docs-contract is strong for signatures/env vars. Remaining lies: complex parity, `torchfits.read_torch` in an error string, GHA `release-gate` job omits `test_cli.py`, `test_public_boundary.py`, `test_http_probe_fixture.py`, `test_mps.py`, `docs-contract`, `docs-links` vs local `pixi run release-gate`. `[project.optional-dependencies] test` cannot actually run the suite (no astropy/fitsio/psutil/pyarrow extras). Conda `license_file: LICENSE` omits vendored CFITSIO license that wheels ship.

### 4.7 Performance residual is documented, not hidden — except the headline CSVs

Narrow-table `read_full` `mmap=False` vs fitsio is still the one significant family. **The 1.1.0 headline cites `exhaustive_*_20260822_*` that are not under `docs/assets/bench/`** (HTML comment: pending VOSpace mirror). Git still has 20260807 / 20260719. Independently recomputing “100% of significant image comparisons” from the repo is impossible.

### 4.8 `where=` on `read_torch` is a different science, not a different return type

Live (2026-08-26): TNULL=25, values `[10,25,30]`, `where="M > 20"` → Arrow `[30]`, torch `[25, 30]`. Int16 `V > 40000` → Arrow 0 rows, torch all rows. Arrow strips TNULL and uses a range-safe compare; `table_api.read_table` compares raw tensors (`values > lit` wraps). Docs present the split as operator dialect only.

### 4.9 `torchfits copy` is an HDU rewrite

`cmds_copy.py` does `open` + `HDUList.write` → `_write_hdus_uncompressed` → `to_tensor()` / materialize tables → `cpp.write_fits_file` **on the target path with no tempfile**. Docs: “exact, lossless binary copy.” CompImage decompresses. `OUTPUT==INPUT` writes onto an open `fitsfile*`. Contrast `setkey --out` (`shutil.copy2`).

### 4.10 Packaging: `main` Lint is red; Linux wheels have no libbz2

HEAD `1a70cd1` Comprehensive CI [failed](https://github.com/astroai/torchfits/actions/runs/32933573150) on the Lint job (mypy without torch/numpy). Tests and the named release-gate job succeeded. manylinux 1.1.0 wheels do not link `libbz2` (`HAS_BZIP2=false`); macOS wheels do. Changelog qualifies “on builds with bzip2 support,” but Linux PyPI is the primary install path. `sanitizer.yml` has never run on GitHub.

### 4.11 Quantized NaNs are finite on `torchfits.read`

`write(..., quantize="robust")` stores non-finite pixels as `BLANK=-32767`. Astropy masks them. `torchfits.read` scales the sentinel (`BSCALE * (-32767) + BZERO`) because `fits_read_img` gets `nulval` only for *compressed* float HDUs (`fits_detail.h` ~589–597). Live (2026-08-26): `read[0,0] ≈ -1.87`, `isnan=False`. `tests/test_release_semantics.py::test_quantize_nan_becomes_blank_not_lo` only asserts astropy.

Default `return_header=True` cache is also unsound: first hit is `Header`, second is a shared `dict`; mutating it poisons later reads (live).

---

## 5. Complete issue register

IDs are stable for the implementation plan. Severity = release risk.

### Blockers (do not freeze / retag)

| ID | Sev | Category | Where | Description | Why it matters | Fix | Compat | Tests |
|---|---|---|---|---|---|---|---|---|
| B1 | BLOCKER | Correctness | `_io_engine/table_api.py:317-336` vs `_read_where.py` | `read_torch(where="M > 20")` keeps TNULL 25; `read` drops it | Silent extra catalog rows on the tensor API | Apply TNULL mask before torch compare (same as Arrow) | Row-set change on `read_torch` | TNULL fixture above; both APIs `[30]` |
| B2 | BLOCKER | Correctness | `table_api.py` vs `_torch_cmp_mask` | Int16 `V > 40000` keeps all rows on `read_torch`, none on `read` | Out-of-range literals wrap instead of matching C++/Arrow | Promote like `_torch_cmp_mask` | Row-set change | int16 vs 40000 |
| B3 | BLOCKER | Docs/CLI | `cli/cmds_copy.py`, `docs/cli.md:271-273` | `copy` rewrites via `HDUList.write` (decompresses CompImage) | Users believe they have a bit-identical backup | `shutil.copy2` (local) or drop “lossless binary” | CompImage outputs change if they used copy as rewrite | CompImage `copy` vs `cmp` |
| B4 | BLOCKER | Benchmarks | `docs/benchmarks.md:167-176`, changelog | Headline run IDs `exhaustive_*_20260822_*` not in `docs/assets/bench/` | Win-rate cannot be recomputed from git | Mirror CSVs or retarget headline to 20260807 | Docs only | Playbook `bench-table-from-csv` |
| B5 | BLOCKER | Safety | `_hdu_rewrite.py:189-190` | `HDUList.write` / `copy` to `OUTPUT==INPUT` writes while source handle is open | Truncate/corrupt the file being read | Tempfile+replace like `write_api`; reject same-path | In-place copy currently undefined | `copy a.fits a.fits` must fail cleanly |
| B6 | BLOCKER | Correctness | `quantize.py`, `fits_detail.h:589-597` | Robust-quantize NaNs written as `BLANK=-32767`; `torchfits.read` returns a finite scaled value | Silent masquerade of missing pixels as data near `lo` | Pass `nulval=NaN` when BLANK present (uncompressed scaled too); assert `isnan` on torchfits read | Values at BLANK pixels become NaN | Round-trip `read`/`read_tensor`/subset vs astropy |

### Genuine release issues (HIGH)

| ID | Sev | Category | Where | Description | Why it matters | Fix | Compat | Tests |
|---|---|---|---|---|---|---|---|---|
| H1 | HIGH | API leak | `_cpp.py:182-187` | `__getattr__` + `__dir__` expose every `_C` symbol, not `__all__` | Freeze inventory is incomplete; bindings become public by accident | Delete `__getattr__` or raise `AttributeError` for names not in `__all__`; `__dir__` = `__all__` only | May break unofficial `_C` peeks through `_cpp` | `test_public_boundary`: `getattr(_cpp, undocumented)` raises |
| H2 | HIGH | Docs/API | `_table/read.py:56-59`, `docs/parity.md:63` | Complex TFORM rejected by Arrow APIs; error names `torchfits.read_torch()` (missing); parity says Supported | Users follow a dead name; parity overclaims | Error → `torchfits.table.read_torch`; parity **Partial** (tensor path only) | Docs + message only | Assert message; `table.read` on C/M raises; `table.read_torch` works |
| H3 | HIGH | Correctness | `_hdu/table_hdu_ref.py:132-141` | `head(n)` replaces `row_slice` with `slice(0,n)` instead of composing | Silent wrong rows after a window | Compose with existing slice | Behavior change for anyone relying on reset | Window then `head`; compare to pandas-style head |
| H4 | HIGH | CLI | `cli/cmds_diff.py:35-42` vs `cmds_stats.py:70-73` | `diff` `tensor.min()` on uint16/uint32 can `RuntimeError` | Diff of unsigned images fails; stats already special-cased | Same upcast as stats | None | uint16 image pair `torchfits diff` |
| H5 | HIGH | CLI | `cli/main.py:61-63` | Ctrl-C → exit 2 (usage) | Scripts treat interrupt as user error | Distinct code (130) or 3; document | Exit-code change | `KeyboardInterrupt` test |
| H6 | HIGH | CLI | `cli/common.py:404-407` | `json.dumps` emits `NaN`/`Infinity` | `stats -f json` is not JSON | `allow_nan=False` or string sentinels | JSON shape | Image with NaN, parse output |
| H7 | HIGH | CI | `.github/workflows/ci.yml:141-150` vs `pixi.toml:87` | GHA release-gate omits CLI/boundary/http/mps/docs-contract | Tag CI greener than local gate | Align job with pixi `release-gate` | None | Workflow review |
| H8 | HIGH | Platform | `__init__.py` (uncommitted), `pixi.toml` test task | Duplicate `libomp.dylib` aborts process; verified pytest abort without env var | macOS `import torch` first, or raw pytest, dies | Land setdefault + pixi activation; docs already drafted locally | Env default | isolation tests in WIP |
| H9 | HIGH | Packaging | `pyproject.toml:74-77` | `[test]` extra is pytest-only; suite needs astropy/fitsio/psutil/pyarrow | `pip install torchfits[test]` cannot run tests | Mirror `dev` test deps or document | Extra contents | Optional extra smoke |
| H10 | HIGH | API | `_table/_read_where.py`, `_io_engine/table_api.py:213` | Operator dialect still split (`OR`/`IN`/`NOT` vs simple); `read_torch` docstring claims the same dialect as `read` | After B1/B2, remaining trap is language + lying docstring | Align docstring; one parser or explicit `ValueError` | If unified, some `read_torch` strings start working | Matrix of expressions |
| H11 | HIGH | Interop | `interop.py:99-129` vs `_table/interop.py` | Root `to_astropy` is Arrow→numpy Table; `table.to_astropy` does MaskedColumn/TUNIT | Changelog “high-fidelity to_astropy” is the table path only | Root should call table helper or docs must split | Possible dtype change on root | Dual-path fixture |
| H12 | HIGH | Path | `paths.py:48-67` vs `security.h:16-21` | Python treats any `[` as CFITSIO filter; C++ requires `]` after last `/` | Paths with `[` in a directory mis-stripped | Share last-component rule | Rare paths | `/tmp/[data]/file.fits` |
| H13 | HIGH | Correctness | `_read_scan.py:288-289`, `read.py:296-298` | Empty `reader`/`scan(slice(0,0))` yields zero-field schema; unknown names dropped only on empty results | Empty vs populated unknown-column behavior disagrees | Always emit header schema (incl. requested names or raise) | Empty-table consumers | `reader(slice(0,0)).schema.names` |
| H14 | HIGH | Native | `table_reader.h` `direct_io_ok` | Buffered table `pread` allows `.gz`/`.zip` (only `.bz2` refused); can throw or decode compressed bytes as cells | `.fits.gz` tables are a normal distribution format | Refuse whole-file compress suffixes; fall back to CFITSIO | Paths that accidentally “worked” | gzip a small table; mmap on/off |
| H15 | HIGH | GIL | `table_bindings.cpp:332-345` | `gil_scoped_release` before `get_fptr_from_python_object` | `nb::cast` without GIL → crash class | Extract `fitsfile*` with GIL held | Internal | Threaded close-during-read (tsan) |
| H16 | HIGH | Docs | `docs/index.md`, `install.md`, README | “Native GPU decode” / “wheels enable GPU acceleration”; benches correctly say host decode + `.to(device)` | Marketing contradicts the honest bench page | “Place result on CUDA/MPS after host decode” | Docs | docs-contract sentence if added |
| H17 | HIGH | Docs | `docs/examples-ml.md:161-177` | 1,000 MegaPipe stamps @ 0.060 s not backed by committed CSVs (git MegaCam CSVs are 40 Rice CCD cutouts) | Unsourced timing on a user page | Cite a CSV or drop the table | Docs | Playbook `bench-table-from-csv` |
| H18 | HIGH | Tests | `tests/test_compression.py` | Sets `TORCHFITS_COMPRESSED_PARALLEL` / `_MIN_PIXELS` names absent from `src/` | Parallel-vs-serial tests cannot fail | Delete or retarget to a real switch | None | grep env in `src/` |
| H19 | HIGH | CI | `.github/workflows/ci.yml` Lint | Mypy without torch/numpy; `main` @ `1a70cd1` workflow [failed](https://github.com/astroai/torchfits/actions/runs/32933573150) | README CI badge is red while tests pass | Install stubs or drop the bare job; keep pixi mypy | None | GHA lint green |
| H20 | HIGH | Packaging | `cibw_before_build.sh`, manylinux wheel | Linux PyPI wheels: `HAS_BZIP2=false` (no libbz2); macOS wheels link libbz2 | 1.1.0 leads with `.bz2` reads; Linux is the primary wheel OS | Install bzip2 in manylinux before-all **or** say “macOS/conda only” | Wheel DT_NEEDED | `test_bz2` must run in cibw, not skip |
| H21 | HIGH | Docs | `docs/release.md` vs `build_wheels.yml` | Runbook still says PyPI API token; upload is OIDC trusted publishing | Maintainers rotate the wrong secret | Document Environment + trusted publisher | Docs | workflow has no `password:` |
| H22 | HIGH | Scripts | `scripts/clean_install_smoke.sh` | Uses unbound `ROOT_DIR` (`ROOT` is set); `set -u` aborts | `docs/release.md` §8 tells maintainers to run this before tag | Rename to `ROOT` | None | `bash -n` + dry run |
| H23 | HIGH | Cache | `caches.py:332-375` | `return_header=True`: 1st hit `Header`, later hits shared `dict`; in-place edits leak | Same call, different type; poisoned headers | Clone `Header` from cards on store and hit | Dict-from-cache callers see Header again | Two reads; mutate; third unchanged |
| H24 | HIGH | Scale | `_read_pipeline.py:385-409` | `read(..., raw_scale=True)` dropped on fallback (`return_header`, named HDU) | Same kwargs, logical vs storage values | Thread `raw_scale` into `read_fallback_image` | Fallback values/dtype change | uint16 + `return_header=True` matches `read_tensor` |
| H25 | HIGH | HDU rewrite | `_hdu_rewrite.py` `replace_hdu` | CompImage surgery decompresses and keeps stale `Z*` on IMAGE | Tools key off `ZIMAGE` after pixels are uncompressed | Strip Z\* like compressed-write sanitizer; or CFITSIO in-place | Header cards after replace | Rice → `replace_hdu` → no `ZIMAGE` unless still compressed |
| H26 | HIGH | Checksum | `_hdu_rewrite.py:353-355`, `checksum_api.py` | Rewrite strips CHECKSUM; `verify_checksums` `ok=True` when absent | Archive ingest treats missing stamps as pass | Restamp after rewrite **or** `ok` false / `present` field | `ok` meaning | checksumed file → replace → loud status |
| H27 | HIGH | mmap | `_read_pipeline.py:458-551` | `read(hdu=[…], mmap=False)` 2-arg batch ignores mmap; `read([paths], mmap="auto")` always mmap | Documented mmap policy skipped | Pass `use_mmap` into `read_hdus_batch` / `read_images_batch` | Some list reads switch to CFITSIO | Patch bindings; compressed list must not mmap-only |
| H28 | HIGH | Quantize | `write_api.py:186-230` | `quantize=` ignored on `compress=` + HDUList (tensor path applies it) | Caller thinks they got robust int16; CFITSIO’s compressor ran | Raise or apply per image HDU | Raising is honest | HDUList+RICE+robust raises or ZBITPIX=16 |
| H29 | HIGH | Quantize | `quantize.py:243-265` | `keep_zero=True` turns NaN into 0, no BLANK | Same masquerade as B6 on the mask path | BLANK for non-finite, or reject | NaN positions change | keep_zero + NaN |
| H30 | HIGH | Header | `header_parser.py:345-351` | Cards parser drops Fortran `D` exponents (`1.5D-3` stays str) | `float(BSCALE)` fails; Python meta can default scale 1.0 | Use `_parse_fits_number` in `_parse_card` | String values become float | `read_header` BSCALE `1.5D-3` → 0.0015 |
| H31 | HIGH | Interop | `_tensor_buffer.py:31-59` | `to_arrow` flattens `(N,K)` to `N*K` rows | Vector FITS columns silently reshape | list/fixed_size_list or raise | Breaking for flatten users | `(3,2)` → 3 rows of pairs |
| H32 | HIGH | Native | `fits_bindings.cpp:1540-1551` | `read_full_numpy` promotes unsigned BZERO images to float32; `read_full_numpy_cached` returns uint16/32 | Two numpy helpers, two dtypes for the same file | Share dtype selection with `read_tensor_canonical` | Numpy unsigned dtype change | BITPIX=16 BZERO=32768 |
| H33 | HIGH | Native | `fits_file.cpp:393-419` | Image/`write_hdus` and VLA writes ignore ndarray contiguity (tables use `ensure_c_contiguous`) | Strided/Fortran images silently mis-write | Contig-copy before `fits_write_img` | Strided inputs start matching `.contiguous()` | `tensor.t()` vs `.t().contiguous()` |
| H34 | HIGH | Native | `table_reader.h` TSBYTE mmap | Table BYTE mmap `memcpy`s; image path XOR 0x80 for `SBYTE_IMG` | Signed-byte table columns disagree with CFITSIO/astropy | XOR 0x80 on TSBYTE mmap/buffered | TSBYTE mmap values | TFORM `1S` vs `fits_read_col` |

### Meaningful quality (fix in 1.1.1 / 1.2, not freeze-blockers)

| ID | Sev | Category | Where | Description | Fix |
|---|---|---|---|---|---|
| M1 | MEDIUM | Architecture | `cpp_src/table_reader.h`, `fits_bindings.cpp` | 2.5k–2.7k line units; 1.2 arena decode will touch them | Split reader: schema / mmap / buffered / mutate |
| M2 | MEDIUM | API | `data/datasets.py`, `data/__init__.py` | 13 Dataset classes (image/cube/spectrum × map/iterable + cutout/staged/table) | Fewer constructors + flags; keep aliases |
| M3 | MEDIUM | Cache | `cache.py` vs `_io_engine/caches.py` | Two “cache” stories; `CacheConfig.max_files` is a documented no-op | Docs + deprecate sizing fields |
| M4 | MEDIUM | Correctness | `table_reader.h:183` | `col.repeat = (int)repeat_long` truncates hostile TFORM | Guard/saturate; reject overflow |
| M5 | MEDIUM | HDU | `_hdu/dataview.py:64-75` | `__getitem__` is 2D-only while cubes are a product | Document or support N-D slices |
| M6 | MEDIUM | Packaging | `packaging/conda/recipe.yaml:79` | Conda `license_file` omits CFITSIO license wheels ship | Add `extern/licenses/CFITSIO-LICENSE.txt` |
| M7 | MEDIUM | Docs | `docs/architecture.md:69-72` | GPUDirect called a “1.1 candidate”; 1.1 shipped without it | Point at 2.0 / roadmap |
| M8 | MEDIUM | Hygiene | many `except Exception` | Strategy fallbacks swallow real I/O (`table_api`, `hdu_api`, `mutation`) | Narrow types; log |
| M9 | MEDIUM | Dead | `data/__init__.py:211-220` | `_normalize_cpp_chunk` identity | Delete |
| M10 | MEDIUM | API | `io.py:454-473`, `caches.py` | `clear_file_cache(handles=)` ignored after Option A | Drop param or warn |
| M11 | MEDIUM | CLI | `resolve_batch_io_pairs` | Same-path still unrejected for compress/convert (copy is B5) | Refuse same-file rewrite |
| M12 | MEDIUM | HTTP | `http_util.py` | Guard resolves DNS once; urllib/CFITSIO re-resolve (TOCTOU / rebinding) | Document residual; optional pin if ever needed |
| M13 | MEDIUM | Tests | `tests/conftest.py` vs CWD writes | Some suites still write fixtures to CWD (historical) | tmp_path only |
| M14 | MEDIUM | Docs CI | `scripts/check_docs_links.py` | External 404s are warnings | Fail on 404 for first-party URLs |
| M15 | MEDIUM | Windows | `docs/install.md` MSVC vs cibw skip win32 | Source-on-Windows implied; untested | “Unsupported” without MSVC recipe, or add a job |

### Cleanup / optional

| ID | Sev | Notes |
|---|---|---|
| C1 | CLEANUP | Uncommitted libomp playbook/docs/tests — land with H8 |
| C2 | CLEANUP | `FastHeaderParser._KEYWORD_PATTERN` unused |
| C3 | CLEANUP | Root `CMakeLists.txt` requires 3.15; scikit-build docs say 3.21 |
| C4 | CLEANUP | `torchfits.cpp` deprecation copy could name removal version |
| C5 | LOW | `Header` ctor silently skips unparseable cards (`header.py:46-50`) |
| C6 | LOW | `SDIST-README` correctly refuses sdist-as-install; keep |

**Not issues (verified fixed vs stale backlog):** GIL on `read_full_numpy`; table mmap truncation check; `update_rows_mmap` strides; `table.to_astropy` MaskedColumn/TUNIT; remote flock + If-Range; CLI `-J` per-worker transform instances; SharedReadMeta UID rotation; schema C/M no longer mapped as float64 (returns `None` / data-driven).

**Optional improvements (not defects):** arena decode (perf 1.2); Dataset collapse (M2); CLI wave-3 fitsverify; object-store row ranges.

---

## 6. Recommended major-release API

Goal: one façade a user can learn in an afternoon. Breaking changes belong in **2.0** unless noted as 1.2 with deprecation.

### Retain (core 1.x)

- `read_tensor`, `read`, `write`, `write_tensor`, `open`
- skinny: `read_header`, `read_shape`, `read_nrows`, `read_keys`, `read_colnames`, `read_num_hdus`, `read_hdu_type`, `read_extname`, `read_table_info`
- `read_subset`, `open_subset_reader`, `open_table_reader`, `read_hdus`, `read_batch`
- HDU mutation + checksums
- `table.read` / `read_arrow` / `scan` / `read_torch` / `scan_torch` / `write` / mutations
- `hdu.{HDUList,TensorHDU,TableHDU,TableHDURef,Header,Card}`
- `transforms` current `__all__` (no spectral revival)
- `data.make_loader` + a **small** Dataset set
- CLI subcommands and exit codes 0/1/3/4 (fix 2 vs interrupt)

### Change (1.1.1, non-breaking)

- **B1–B6 first** (TNULL/`where` wrap; quantize NaN; binary `copy`; tempfile rewrite; bench CSVs or headline rollback)
- H2 error string; H4 diff upcast; H6 JSON; H8 KMP; H7 CI align; H12 path brackets; H3 `head` compose (behavior fix — changelog as bugfix)
- Seal `_cpp` (H1) — breaking only for people poking undocumented `_C` names via `_cpp`

### Rename / document, do not churn

- Keep `table.read_arrow` alias
- Keep `read_torch` **on `table`**, never add a root `read_torch` unless 2.0 consolidates

### Deprecate (warn in 1.2, remove in 2.0)

- `torchfits.cpp` namespace
- `CacheConfig.max_files` / `max_memory_mb` / `configure_cpp_cache`
- `clear_file_cache(handles=)`
- Possibly root `to_astropy` if it cannot match `table.to_astropy` (or make it a thin wrapper — preferred)

### Remove (2.0 only)

- `torchfits.cpp`
- Undocumented `_C` names from `_cpp`
- Spectral transform names (already gone)

### Make internal

- `fits_schema`, `header_parser`, `_io_engine`, `_table`, `_where` implementation modules (already underscored except `fits_schema` / `header_parser` — consider `_fits_schema` in 2.0)
- `torchfits.logging` is fine as a logger module

### Do not break without benefit

- `read(..., scale_on_device=True)`, `mmap="auto"`, unsigned BZERO conventions, `quantize="robust"` not min-max
- Wheel ABI = one torch minor per torchfits release

---

## 7. Implementation plan

Order avoids rewriting `table_reader.h` twice.

### Phase 1 — Silent science + copy + macOS

**Issues:** B1, B2, B6, B3, B5, H8, H3, H23, H24  
**Deps:** none  
**Files:** `table_api.py`, `_read_where.py`, `quantize`/`fits_detail` nulval, `cli/cmds_copy.py`, `_hdu_rewrite.py`, `caches.py`, `__init__.py`  
**Outcome:** `read` and `read_torch` agree on TNULL and out-of-range ints; quantized BLANK pixels are NaN on `torchfits.read`; `copy` is binary or honestly named; same-path write cannot clobber an open handle; header cache clones; macOS import order safe; `head` windows compose.  
**Validate:** live TNULL/int16/quantize-NaN fixtures; CompImage `copy` vs `cmp`; `copy a.fits a.fits` fails cleanly; two `return_header=True` reads.

### Phase 2 — Façade seal + CI truth

**Issues:** H1, H7, H9, H11, H19, H20, H21, H22, M6, M10, M14  
**Deps:** H1 after any in-tree `_cpp.undocumented` uses (grep first)  
**Files:** `_cpp.py`, `tests/test_public_boundary.py`, `.github/workflows/ci.yml`, `pyproject.toml`, `interop.py` or docs, conda recipe, cibw before-all, `docs/release.md`, `clean_install_smoke.sh`  
**Outcome:** public inventory == `__all__`; GHA Lint green and gate == pixi; extras runnable; Linux wheels link libbz2 or docs say they don’t; release runbook matches OIDC.  
**Validate:** public-boundary tests; compare GHA YAML to `pixi.toml` `release-gate` string; `test_bz2` in cibw.

### Phase 3 — Table semantics + native holes

**Issues:** H10, H13, H14, H15, H25, H26, H27, H32, H33, H34, M4, M5, M8 (table sites)  
**Deps:** Phase 1 B1/B2  
**Files:** `_read_where.py`, `table_api.py`, `docs/api-tables.md`, `table_reader.h`, `table_bindings.cpp`  
**Outcome:** remaining WHERE dialect documented; empty reader keeps schema; `.fits.gz` tables use CFITSIO; GIL held around `nb::cast`.  
**Validate:** where matrix; `reader(slice(0,0))`; gzip table; rebuild test env after C++.

### Phase 4 — Native / boundary (1.2, not a patch)

**Issues:** M1, residual narrow-table bench, optional M12  
**Deps:** design note for arena/strided tensors (API-visible)  
**Files:** `table_reader.h`, `fits_bindings.cpp`, benches  
**Outcome:** split files **or** arena decode with explicit non-contiguous column tensors.  
**Validate:** existing table fidelity tests + same-host `bench_fitstable_io` case_id vs fitsio.

### Phase 5 — Tests

**Issues:** H3–H6 tests, H1, H8, M13, malformed/concurrency already strong (`test_malformed_fits`, `test_truncated_table_errors`, `test_concurrent_same_file_read`)  
**Add:** uint16 `diff`; invalid JSON regression; `_cpp` seal; `head` composition; `pip install .[test]` recipe.  
**Validate:** `pixi run test` (uses KMP).

### Phase 6 — Performance

Only after Phase 4 design. No speculative micro-opts. Keep `bench-table-from-csv` playbook rule.

### Phase 7 — Benchmarks

**Issues:** B4, H16, H17, M7  
**Outcome:** headline run IDs exist under `docs/assets/bench/` **or** the 1.1.0 table is retargeted to 20260807; MegaPipe timings sourced or removed; GPU copy matches benches (host decode + `.to`).  
**Validate:** playbook `bench-table-from-csv`.

### Phase 8 — Docs / examples

**Issues:** H2 parity, M3 cache docs, M7, examples already gated by `test_examples_runner`  
**Validate:** `pixi run docs-contract && pixi run docs-links`

### Phase 9 — Packaging / CI

**Issues:** H7, H8, H9, M6, M15  
**Validate:** `pixi run ci-local`; confirm GHA yaml; do not claim Windows.

### Phase 10 — Release validation

1. `pixi run changelog-update` + curate  
2. `pixi run release-gate`  
3. API freeze skill: `.cursor/skills/release-api-freeze-review/`  
4. Tag only after **B1–B6** plus H1–H34 closed or explicitly deferred in changelog  
5. Wheels: existing tag workflow  

---

## 8. Release gate (checklist before the next tag)

Use for **1.1.1** or **1.2.0**, not to un-ship 1.1.0.

### Correctness

- [ ] `read` vs `read_torch` `where=` same rows for TNULL and out-of-range integer literals (B1, B2)
- [ ] Quantized NaN pixels are NaN on `torchfits.read` / `read_tensor` (B6), not only astropy
- [ ] `torchfits copy` is a binary copy **or** docs no longer say lossless (B3); same-path rejected (B5)
- [ ] No known silent row/pixel wrong-answer (H3 closed or documented)
- [ ] Unsigned images: `diff` and `stats` both succeed (H4)
- [ ] Truncated tables still raise (existing `test_truncated_table_errors`)
- [ ] Complex Arrow path error names `table.read_torch` (H2)
- [ ] Hostile TFORM repeat does not truncate silently (M4) *or* explicitly deferred

### API stability

- [ ] `_cpp` cannot `getattr` undocumented `_C` names (H1)
- [ ] Root `__all__` == `docs/api.md` Quick Paths (existing integrity test)
- [ ] `torchfits.cpp` still warns; removal version named
- [ ] `where=` dialects documented in `api-tables.md` (H10)
- [ ] No new root symbols without changelog

### Testing

- [ ] `pixi run test` (full suite) on macOS **and** Linux
- [ ] `KMP_DUPLICATE_LIB_OK` not required for `import torchfits` then `import torch` (H8)
- [ ] `tests/test_cli.py`, `test_public_boundary.py`, `test_malformed_fits.py` green
- [ ] Examples: `tests/test_examples_runner.py`

### Native safety

- [ ] GHA `sanitizer.yml` green on the PR
- [ ] No new shared `fitsfile*` pooling

### Platforms

- [ ] Wheels: Linux x86_64, aarch64, macOS arm64, cp310–cp314, torch 2.13
- [ ] Windows: still “unsupported” unless a job exists (M15)

### Install

- [ ] `pip install torchfits` from a clean venv (wheel)
- [ ] Source build recipe in `docs/install.md` matches CMake (bzip2 optional)
- [ ] `[test]` extra documented or actually sufficient (H9)

### Packaging

- [ ] `check-lane` + `check-torch-pins`
- [ ] Wheel + conda license files include CFITSIO (M6)
- [ ] sdist still documented as non-install

### Examples / docs

- [ ] `docs-contract`, `docs-links`
- [ ] `parity.md` complex = Partial
- [ ] No phantom extra torch lanes

### Benchmarks

- [ ] Headline run IDs exist under `docs/assets/bench/` (B4)
- [ ] No new headline ratios without archived `benchmarks_results/` + `docs/assets/bench/`
- [ ] Narrow-table residual still described as 1.2 work

### CI / static

- [ ] GHA Lint green (H19) and release-gate == pixi `release-gate` (H7)
- [ ] Linux wheel `test_bz2` does not skip, **or** changelog says macOS/conda only (H20)
- [ ] `preflight-push` (ruff, mypy)
- [ ] ASan workflow not skipped (packaging: sanitizer.yml has **zero** GitHub runs)

### Artifacts

- [ ] Version triplet: `pyproject` / `pixi.toml` / `__init__.py` / conda recipe / changelog date
- [ ] GitHub Release body from curated changelog (`scripts/release_notes.py`)

---

## Execution notes for the next agent

1. Do **not** start with C++ splits. Land Phase 1 Python/docs/CLI/KMP first.
2. Rebuild **test** env after any C++ change: `pixi run -e test -- pip install -e . --no-build-isolation`.
3. On macOS, until H8 lands, prefix pytest with `KMP_DUPLICATE_LIB_OK=TRUE` or use `pixi run test`.
4. Ignore stale `.cursor/post-1.0-backlog.md` 1.1 deferrals that this audit marked fixed; use **this register**.
5. Working-tree libomp files are the H8 patch — include them rather than re-deriving.
6. Area notes: [agent-packaging.md](agent-packaging.md), [agent-io.md](agent-io.md), [agent-cpp.md](agent-cpp.md).
