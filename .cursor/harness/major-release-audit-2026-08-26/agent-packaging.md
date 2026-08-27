# Packaging / CI / vendor audit — torchfits 1.1.0

Date: 2026-08-26
Scope: tracked packaging, vendor, CI, scripts, agent-infra, and related root metadata.
Method: read every file in scope; inspect the published `v1.1.0` wheels and sdist; sample GitHub Actions. **No fixes applied.**

## Verdict

**Ship with notes — not packaging-clean.**

`v1.1.0` is already tagged and on PyPI (15 wheels + GitHub-only sdist; `upload_pypi` succeeded via OIDC). The published platform matrix matches the install docs (CPython 3.10–3.14 × Linux x86_64/aarch64 × macOS arm64, torch 2.13 ABI lane). Vendor pinning, license files in the wheel, and KMP guard are in good shape.

Do not treat the tree as major-release-clean until these are resolved:

1. Comprehensive CI **Lint is red on `main`** (mypy without torch/numpy). Tests, docs-contract, and the named release-gate job are green.
2. **manylinux wheels do not link libbz2**, so the 1.1.0 headline “native whole-file `.bz2` reads” is macOS-wheel-only (and pixi/conda). Linux PyPI users get `HAS_BZIP2=false`.
3. Several **docs / extras / runbook lies** (`[test]` extra, `[dev]`/`[bench]` on 3.10, Windows source tab, `docs/release.md` PyPI token vs OIDC, `vendor.sh` invocation, broken `clean_install_smoke.sh`).

conda packaging is a second, unfinished channel (`packaging/conda/recipe.yaml` “not yet upstreamed”) and still disagrees with `pixi.toml` package metadata on CFITSIO.

---

## Look-for checklist

| Topic | Result |
|---|---|
| Windows unsupported vs classifiers | No Windows OS classifier (correct). No Linux/macOS classifiers either. `cibuildwheel.skip` lists `*-win32` but **not** `*-win_amd64`. `docs/install.md` has a Windows source-build tab; `docs/compatibility.md` source column is Linux+macOS only; CMake Release flags are GCC/Clang (`-O3`, `-mssse3`). |
| Python / torch version truth | Wheels, classifiers, CI matrix, `constraints-wheel.txt`, pixi, recipe, `torch_lanes.json`: **3.10–3.14 / torch 2.13.x**. `build-system.requires` and CMake still say **torch≥2.10** (source / ABI embed). Runtime `project.dependencies` is `torch>=2.13,<2.14`. Intentional, but naive isolated builds can ABI-mismatch. |
| Wheel vs source | PyPI: wheels only (workflow comment + test assertion). GitHub Release attaches sdist (`torchfits-1.1.0.tar.gz`, 6.3 MB). Sdist has no `extern/cfitsio/` (gitignored); CMake auto-vendor is off without `.git`. |
| SPDX / license files | `license = "MIT"` + `license-files` ships `LICENSE` and `extern/licenses/CFITSIO-LICENSE.txt` in every inspected wheel. METADATA `License-Expression: MIT` does **not** mention the NASA CFITSIO text. conda recipe lists only `LICENSE`. |
| CI vs local `release-gate` | Local gate is a longer pytest list + `docs-contract` + `docs-links`. GitHub `release-gate` job omits 4 test modules and both docs tasks. GitHub `test` job runs `pytest tests/` but with a thinner dep set (skips). GitHub **Lint** does not match `preflight-push` and is currently **failing**. |
| Sanitizer workflow | File exists, workflow Active, trigger is `pull_request` to main/develop + `workflow_dispatch`. **Zero runs in GitHub history.** Direct pushes to `main` never start it. C flags / CFITSIO C code unsanitized; leak detection on macOS+Python is noisy. |
| Sdist completeness | Inspection sdist includes vendor script, pins, patches, licenses, `SDIST-README.txt`. Missing CFITSIO sources by design. Bloated with `pixi.lock`, docs bench CSVs, gallery PNGs, `.dsh/`. `.cursor/` and `.github/` excluded. |
| Secrets | No tokens in tree. PyPI uses OIDC (`id-token: write`), not `secrets.PYPI_API_TOKEN`. `bench-report.yml` uses `github.token` to open a PR. `.dsh/README.md` shows `export DEEPSEEK_API_KEY=sk-...` as a placeholder. CANFAR scripts default to `vos:sfabbro/...` (path, not a credential). |
| Unreproducible builds | No `SOURCE_DATE_EPOCH`. CMake mutates vendored `fitsio.h` / `fitsio2.h` in-tree for NIOBUF/MINDIRECT. Opportunistic IPO/LTO. `vendor.sh` rewrites `VERSIONS.txt` after every run. CMake may `CREATE_LINK` zlib into the conda prefix. macOS `repair-wheel-command = ""`. |
| `[test]` extra incomplete | `pytest` + `pytest-cov` only. Cannot run the suite. `[dev]`/`[bench]` pin `astropy>=8` / `matplotlib>=3.11`, uninstallable on Python 3.10. |
| Dead scripts | 51 tracked scripts; most are wired. `clean_install_smoke.sh` is **broken** (`ROOT_DIR` unbound). `aggregate_matrix_bench.py` has **no** in-repo callers. |
| `KMP_DUPLICATE_LIB_OK` | `setdefault` in `src/torchfits/__init__.py`; pixi `activation.env`; tests; install troubleshooting. GitHub `test` job does not export it (macos-15 matrix still passed). |
| Vendor sha256 | `extern/VERSIONS.txt` pins `cfitsio-4.7.0` + sha256. `vendor.sh` verifies before extract; Darwin uses `shasum`, Linux `sha256sum`. Unpinned fetch requires `TORCHFITS_VENDOR_ALLOW_UNPINNED=1`. `cfitsio_repo=` in VERSIONS.txt is **not** used for the download URL (hardcoded `HEASARC/cfitsio`). |

---

## Issues

Severity: **Blocker** (fix before calling CI/release packaging green) · **High** · **Medium** · **Low**.

### Blocker

#### PKG-1 — GitHub Lint job fails: mypy without torch/numpy
- **Where:** `.github/workflows/ci.yml` job `lint` (“Mypy (strict)”).
- **Evidence:** run [32933573150](https://github.com/astroai/torchfits/actions/runs/32933573150) on `main` after the `v1.1.0` tag: Lint **failure**, 83 errors / 50 files, `Cannot find implementation or library stub for module named "torch"` (also numpy, polars, duckdb). All `test` matrix cells, docs-contract, bench-smoke, and the named release-gate job **succeeded**.
- **Drift:** local `pixi run preflight-push` runs mypy inside the pixi env (pytorch present). CI Lint installs only `mypy>=2.3`. The `release-gate` CI job *does* install torch/pyarrow/polars/duckdb and re-runs mypy — so typecheck is duplicated and only the bare Lint job is red.
- **Why it matters:** the CI badge on README tracks this workflow. `main` is red while PyPI 1.1.0 is live.

#### PKG-2 — manylinux wheels ship without libbz2; 1.1.0 bz2 feature is not on Linux PyPI
- **Where:** `scripts/cibw_before_build.sh` / `[tool.cibuildwheel.linux]` `before-all` (vendor only; no bzip2 headers); `src/torchfits/cpp_src/CMakeLists.txt` leaves `USE_BZIP2=OFF` when `find_package(BZip2)` fails.
- **Evidence:** inspected `torchfits-1.1.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`:
  - DT_NEEDED: `libz.so.1`, libtorch\*, no `libbz2`.
  - No auditwheel `.libs/` bundle.
  - CFITSIO `#else` strings (`bzip2 compression is not supported in this build`) present in `_C.so`.
  - macOS arm64 wheel **does** link `/usr/lib/libbz2.1.0.dylib` and `/usr/lib/libz.1.dylib`.
- **Why it matters:** changelog 1.1.0 leads with native whole-file `.bz2` reads. Linux is the primary wheel OS. `tests/test_bz2.py` skips when incapable; cibuildwheel only runs `tests/test_release_smoke.py`, so the gap never fails the wheel job.
- **Related:** conda recipe explicitly depends on `bzip2`; pixi prefix has `bzlib.h`. Dev/CI editable installs on Ubuntu likely **have** HAS_BZIP2. Wheel ≠ CI.

### High

#### PKG-3 — `docs/release.md` documents a PyPI API token; the workflow uses trusted publishing
- **Where:** `docs/release.md` §6 and §10 (`secrets.PYPI_API_TOKEN`, “not trusted publishing”); `.github/workflows/build_wheels.yml` `upload_pypi` (`permissions.id-token: write`, `pypa/gh-action-pypi-publish@release/v1`, no password).
- **Evidence:** `v1.1.0` `upload_pypi` job **succeeded** with OIDC. `.cursor/docs-review-progress.md` repeats the stale token claim (agent infra).
- **Why it matters:** a maintainer following the runbook may create/rotate an unused secret, or believe publish is broken when the `pypi` GitHub Environment / PyPI trusted-publisher pairing is the real switch.

#### PKG-4 — `[test]` extra cannot run tests; `[dev]`/`[bench]` extras break on Python 3.10
- **Where:** `pyproject.toml` `[project.optional-dependencies]`; `docs/install.md` extras table; `.cursor/post-1.0-backlog.md` already records the `[test]` gap.
- **`[test]`:** only `pytest>=9.0`, `pytest-cov>=7.0`. Suite needs at least astropy, fitsio, pandas, matplotlib, polars, duckdb, psutil, tomli (3.10), and the core torch/numpy/pyarrow pins.
- **`[dev]` / `[bench]`:** `astropy>=8.0` and `matplotlib>=3.11` require Python ≥3.11 (pixi.toml comment, lines 72–74). Classifiers and wheels include 3.10. `pip install 'torchfits[dev]'` on 3.10 cannot resolve. Pixi workspace uses `astropy>=6.1` / `matplotlib>=3.8` so the solver can fall back.
- **`[dev]` also omits** polars, duckdb, psutil (pixi has them; many tests `importorskip`).

#### PKG-5 — Windows is undocumented-unsupported in CMake but offered as a source-build OS
- **Where:** `docs/install.md` “Windows” tab (VS 2019+ / MSVC); `docs/compatibility.md` source builds = Linux, macOS; `docs/install.md` troubleshooting tells Windows users to “build from source”; `pyproject.toml` `cibuildwheel.skip` = `*-win32` only; CMake `add_compile_options(-O3)` and `-mssse3` are not MSVC-gated (MSVC skip exists only for PGO / finite-math / `_C` visibility).
- **Classifiers:** no `Operating System :: Microsoft :: Windows` (good) and no POSIX/Linux/macOS either (PyPI OS filter is empty).
- **Why it matters:** a 1.x “production/stable” classifier plus a Windows install recipe implies a supported source path that will not configure.

#### PKG-6 — `docs/install.md` `./extern/vendor.sh` is not a valid invocation
- **Where:** `docs/install.md` build steps; `extern/vendor.sh` requires `--cfitsio-version <tag-or-file>`.
- **Evidence:** empty spec → `Failed to resolve CFITSIO version`. `SDIST-README.txt` and CI/CMake pass `--cfitsio-version extern/VERSIONS.txt` correctly. `benchmarks/cfitsio_direct/CMakeLists.txt` error text also omits the flag.

#### PKG-7 — `scripts/clean_install_smoke.sh` is broken (`ROOT_DIR` vs `ROOT`)
- **Where:** `scripts/clean_install_smoke.sh` line 38: `python "$ROOT_DIR/scripts/release_lane.py"` while the script sets `ROOT`. `set -euo pipefail` → unbound variable.
- **Why it matters:** `docs/release.md` §8 tells maintainers to run this before tag.

#### PKG-8 — conda / pixi package metadata still depends on conda-forge CFITSIO 4.6.x while compiling vendored 4.7.0
- **Where:** `pixi.toml` `[package.build-dependencies]`, `[package.host-dependencies]`, `[package.run-dependencies]`: `cfitsio = ">=4.6.3,<4.6.5"`. Vendor pin is `cfitsio-4.7.0`. Build flags force `TORCHFITS_USE_VENDORED_CFITSIO=ON` (static).
- **`packaging/conda/recipe.yaml`:** does **not** list cfitsio as a run dep (better), **does** list `bzip2`, **does not** run `extern/vendor.sh`. Relies on CMake `TORCHFITS_AUTO_VENDOR_DEPS`, which is **OFF** when `.git` is absent (sdist / detached source). `pixi run build` from a git checkout works; a clean tarball conda build will not find CFITSIO sources.
- **Two backends:** pixi-build-cmake (`pixi.toml`) vs rattler-build (`recipe.yaml`) already drift (cfitsio run dep, vendor step, license files).

### Medium

#### PKG-9 — Local `release-gate` / `ci-local` are not a mirror of GitHub Comprehensive CI
- **GitHub `release-gate` pytest list omits** (present locally): `tests/test_cli.py`, `tests/test_public_boundary.py`, `tests/test_http_probe_fixture.py`, `tests/test_mps.py`. Also omits `docs-contract` and `docs-links` (local gate runs both).
- **GitHub `test` job** runs full `pytest tests/` but installs only `numpy astropy fitsio pytest nanobind scikit-build-core psutil pyarrow tomli` — **not** pandas, matplotlib, polars, duckdb. Those tests `importorskip` and **silently skip** on the 2×5 matrix. Local pixi has the richer set.
- **`scripts/ci_local.sh`** claims to “Mirror GitHub Comprehensive CI” but does **not** run mypy, full pytest, or bench-smoke. It does run `check-lane` / `check-torch-pins` / docs-build / local release-gate.
- **GitHub Lint** does not run `check-lane`, `changelog-check`, or `compileall` (`preflight-push` does). GitHub never runs `docs-links` (`pixi run docs-links` is local-only). Docs-contract job builds zensical but does not crawl links.

#### PKG-10 — Sanitizer workflow has never run; coverage is incomplete even if it did
- **Where:** `.github/workflows/sanitizer.yml`.
- **Triggers:** PR to main/develop, `workflow_dispatch`. Pushes to `main` (the 1.1.0 cut) do not start it. `gh run list --workflow=sanitizer.yml` is empty.
- **Gaps:** Python 3.12 (not 3.13); `uv` not pixi; sanitizer flags on `CMAKE_CXX_FLAGS` only (vendored CFITSIO is C); `CMAKE_EXE_LINKER_FLAGS` not `CMAKE_SHARED_LINKER_FLAGS` / module flags; `detect_leaks=1` under CPython; test subset only; no fitsio in the venv.

#### PKG-11 — SPDX expression does not reflect vendored CFITSIO
- Wheels correctly **ship** `dist-info/licenses/LICENSE` and `dist-info/licenses/extern/licenses/CFITSIO-LICENSE.txt` (PEP 639 `license-files`).
- METADATA is `License-Expression: MIT` only. CFITSIO’s file is NASA / U.S. Government permission (not MIT). conda `about.license: MIT` / `license_file: LICENSE` drops the third-party file.
- Not a missing-file bug; it is an incomplete expression for reuse scanners.

#### PKG-12 — cibuildwheel `skip` does not exclude 64-bit Windows
- `skip = "*-musllinux_* *-manylinux_i686 *-win32 cp31?t-*"` — `cp3*-win_amd64` would still build if someone ran cibuildwheel on Windows. GHA matrix has no Windows runner, so PyPI is clean today (verified: 15 wheels, no win).

#### PKG-13 — musl / Intel macOS / Windows cannot `pip install` at all
- No sdist on PyPI (intentional: “pip must not compile”). No musllinux, macosx x86_64, or Windows wheels.
- `docs/install.md` names Windows and x86_64 macOS; it does **not** name Alpine/musl. Those users get `No matching distribution` with no compile fallback.

#### PKG-14 — GitHub sdist is an accidental install surface (~18 MB uncompressed)
- `build_sdist` does not vendor CFITSIO; pip from the GitHub tar.gz fails at configure unless the user runs `vendor.sh` first (good fail). Isolated builds would still resolve `torch>=2.10` unbounded from `build-system.requires`.
- Sdist includes `pixi.lock` (1.5 MB), docs bench CSVs (~8 MB+), gallery PNGs, katex, `.dsh/`, full tests. `.cursor/` and `.github/` excluded. Fine for inspection; heavy and not a build input.

#### PKG-15 — Unreproducible / dirty-tree build steps
- CMake string-replaces `#define NIOBUF` / `MINDIRECT` in vendored headers if the comment text matches exactly; silent no-op if upstream wording changes.
- `check_ipo_supported` enables LTO when the toolchain allows — wheel contents differ by compiler.
- `vendor.sh` always rewrites `VERSIONS.txt` (content should match if the hash matches; still a dirty git tree if line endings differ).
- ZLIB missing-path branch may `file(CREATE_LINK ...)` into the environment prefix.
- No `SOURCE_DATE_EPOCH` in cibuildwheel environment.
- macOS wheels un-delocated by design (`repair-wheel-command = ""`); they depend on `/usr/lib/libz.1.dylib`, `libbz2.1.0.dylib`, `libcurl.4.dylib` (stable on macOS 11+).

#### PKG-16 — `gpu-bootstrap.sh` torch pin is not rendered by `release_lane.py`
- Hardcoded `TORCH_SPEC=torch>=2.13,<2.14` and default index `cu129`. Lane apply rewrites pyproject/pixi/recipe/constraints/`__init__.py` only. A future lane bump will leave CANFAR GPU bootstrap on 2.13.

#### PKG-17 — CI test / wheel-test dep set vs extras
- cibuildwheel `test-requires = ["numpy", "pyarrow", "pytest"]` plus `cibw_test.sh` installs the torch pin — enough for `test_release_smoke.py` only. No bz2, no CLI, no security, no ABI-mismatch test.
- `install.strip = false` → ~1–2 MB unstripped `_C.so` (observed ~2.2 MB linux / ~1.8 MB mac).

### Low

#### PKG-18 — No OS Trove classifiers
- Add `Operating System :: POSIX :: Linux` and `Operating System :: MacOS` so PyPI does not look OS-agnostic. Do not add Windows.

#### PKG-19 — `SDIST-README.txt` is not the package long description
- Present in the tarball (good for humans who open it). PKG-INFO / PyPI still use `readme = "README.md"`. GitHub sdist users who never open `SDIST-README.txt` will try `pip install` and hit the CFITSIO fatal.

#### PKG-20 — `torch_lanes.json` is a single lane; comments still talk about 2.10–2.13 grids
- Map is only `"2.13" → 1.1.0` with `cu126/cu129/cu130`. pixi.toml comments describe python 3.10–3.14 × torch 2.10–2.13. `release_lane.py` can render experimental `+torchNNN` versions; they are not published (confirmed: no `+torch` assets on `v1.1.0`).

#### PKG-21 — `[feature.test.dependencies]` is empty
- `test` pixi env is `prod` + empty feature (a separate prefix so the extension is not shared). Not a product bug; easy to misread as “the test extra lives here.”

#### PKG-22 — Root `CMakeLists.txt` is a 5-line wrapper; conda recipe ignores it
- Recipe `-S $SRC_DIR/src/torchfits/cpp_src`. pixi-build-cmake uses the root file which `add_subdirectory`s the same project. Harmless duplication.

#### PKG-23 — `aggregate_matrix_bench.py` is unwired
- No pixi task, no workflow, no other script references. Maintainer-only / leftover. Not harmful.

#### PKG-24 — `check_duplicate_cpp.py` CI output uses emoji
- Works; noisy in logs. Pre-commit hook uses `python3` (not pixi).

#### PKG-25 — Agent-infra staleness (not product API)
- `.cursor/post-1.0-backlog.md` still says “wheels + pixi stay on **PyTorch 2.10**.”
- `.cursor/harness/config.json` `verify_full` is `pixi run pre-commit`, not `release-gate` (AGENTS.md before-tag command).
- `.cursor/docs-review-progress.md` PyPI token claim.
- `.cursor/harness/docs-audit-1.1.0.md` phantom 2.11/2.12 wheel URLs — product docs appear updated; this file may be leftover.

#### PKG-26 — `.dsh/README.md` placeholder `sk-...`
- Not a live secret. Looks like one in grep.

---

## What is in good shape (do not “fix”)

- **Vendor sha256 path:** `extern/VERSIONS.txt` + verify-before-extract + Darwin `shasum` workaround (needed; v1.1.0 wheel job failed twice on macOS before the OS-select fix). `TORCHFITS_VENDOR_ALLOW_UNPINNED` default off.
- **Patches (intent):**
  - `cfitsio-4.7.0-plio-cbuf.patch` — PLIO worst-case buffer was `nx * sizeof(int)` in 4.6.4; overflow on incompressible data. Real correctness fix.
  - `cfitsio-4.7.0-bzip2.patch` — re-enable upstream-commented `BZIP2_1` behind `HAVE_BZIP2`, with a failing stub if the library is missing. Intent is sound; **Linux wheels never define `HAVE_BZIP2`** (PKG-2).
- **KMP_DUPLICATE_LIB_OK:** `os.environ.setdefault` at import; pixi activation; isolation tests; documented. Caller can still override.
- **Wheel ABI isolation:** `--no-build-isolation` + `constraints-wheel.txt` + `cibw_before_build.sh` + `check_torch_extra_pins.py` + `release_lane.py --check`. CPU-linked extension, CUDA torch of the same minor at runtime (`USE_CUDA=OFF`).
- **PyPI has no sdist** so unmatched CPython/arch cannot compile (PKG-13 is the cost of that choice).
- **License files in wheels** (both MIT and CFITSIO).
- **Published matrix matches classifiers / README / compatibility table** (Python 3.10–3.14, 3 OS/arch cells, no win/musl/Intel Mac).
- **`py.typed`** is in the wheel.
- **Console script** `torchfits = torchfits.cli.main:main` in METADATA.
- **Secrets:** no live credentials in tracked files; Pages and PyPI use OIDC.

---

## File ledger

Status: **OK** · **ISSUE** (see id) · **NOTE** · **EXCLUDE** (not hand-audited).

### Root metadata

| File | Status |
|---|---|
| `pyproject.toml` | ISSUE PKG-2, PKG-4, PKG-5, PKG-11, PKG-12, PKG-18. Version 1.1.0; classifiers 3.10–3.14; no OS classifiers; extras incomplete; cibuildwheel skip incomplete; sdist exclude `.cursor/.pixi/.github`; torch runtime pin 2.13 vs build-requires ≥2.10. |
| `pixi.toml` | ISSUE PKG-8, PKG-9, PKG-16, PKG-20, PKG-21. Version 1.1.0; platforms `osx-arm64`, `linux-64` only (no linux-aarch64 pixi env; wheels still build aarch64). Local `release-gate` richer than GHA. `KMP_DUPLICATE_LIB_OK` in activation. Empty `[feature.test]`. |
| `pixi.lock` | EXCLUDE (26482 lines, pixi lock format v7, linux-64 + osx-arm64). Note: included in GitHub sdist (~1.5 MB). |
| `CMakeLists.txt` | NOTE PKG-22. Five-line `add_subdirectory(src/torchfits/cpp_src)`. |
| `src/torchfits/cpp_src/CMakeLists.txt` | ISSUE PKG-2, PKG-5, PKG-8, PKG-15. Real build. Vendored CFITSIO default ON; auto-vendor only with `.git`; in-tree header patch; zlib symlink side effect; BZip2 optional/silent; MSVC-hostile `-O3`/`-mssse3`; ABI ≥2.10 embed. |
| `benchmarks/cfitsio_direct/CMakeLists.txt` | NOTE (out of named “root + cpp” but present). Same `-O3`/`-mssse3`; error message omits `--cfitsio-version`. |
| `constraints-wheel.txt` | OK. `torch>=2.13,<2.14`; rendered by `release_lane.py`. |
| `LICENSE` | OK. MIT, Copyright 2026 Sébastien Fabbro. |
| `SDIST-README.txt` | NOTE PKG-19. Accurate wheels-only / clone-for-source story; vendor flag correct. |
| `AGENTS.md` | NOTE (agent infra). Verify tiers; Jules rules; docs contract. |
| `zensical.toml` | OK for docs packaging. KaTeX paths; `custom_dir = overrides`. |

### `packaging/`

| File | Status |
|---|---|
| `packaging/conda/recipe.yaml` | ISSUE PKG-8, PKG-11. Version/torch_pin match lane. No vendor.sh. license_file LICENSE only. “Not yet upstreamed.” |

### `extern/`

| File | Status |
|---|---|
| `extern/VERSIONS.txt` | OK. `HEASARC/cfitsio`, `cfitsio-4.7.0`, sha256 `f281ca29…`. Repo field unused by fetch. |
| `extern/vendor.sh` | ISSUE PKG-6, PKG-15. Pin+hash+patches-by-tag are correct; Darwin sha256 OK; always rewrites VERSIONS.txt; hardcoded repo. |
| `extern/licenses/CFITSIO-LICENSE.txt` | OK. NASA permission text. |
| `extern/patches/cfitsio-4.7.0-bzip2.patch` | OK intent (PKG-2 is apply/link, not patch quality). |
| `extern/patches/cfitsio-4.7.0-plio-cbuf.patch` | OK intent. |
| `extern/cfitsio/` | Not tracked (`.gitignore`). Generated by vendor.sh. |

### `.github/workflows/`

| File | Status |
|---|---|
| `ci.yml` | ISSUE PKG-1, PKG-9. Lint mypy bare; test dep set thin; release-gate list shorter than pixi; KMP only on release-gate job; no docs-links / check-lane / changelog-check. Matrix 3.10–3.14 × ubuntu + macos-15 is the right OS/Python grid. |
| `build_wheels.yml` | ISSUE PKG-3, PKG-12. Wheel matrix matches product; sdist GitHub-only; OIDC publish (docs disagree); tests job is Ubuntu py3.13 only. |
| `docs.yml` | OK. main + workflow_dispatch; no tag deploy (comment explains Pages concurrency). zensical only; no docs-links. |
| `sanitizer.yml` | ISSUE PKG-10. Never ran. |
| `bench-report.yml` | NOTE. `workflow_dispatch`; `GH_TOKEN` to push a docs PR. CPU-only; not a release gate. |

### `scripts/` (51)

| File | Status |
|---|---|
| `aggregate_matrix_bench.py` | ISSUE PKG-23. Unwired. |
| `bench_cfitsio_direct.sh` | OK. Vendors with correct flag. |
| `bench_deficit_focus.sh` | OK. pixi `bench-deficit-focus`. |
| `bench_exhaustive_local.sh` | OK. pixi `bench-exhaustive-local`. |
| `bench_release_scorecard.sh` | OK. pixi `bench-release-scorecard`. |
| `bench_suite.sh` | OK. pixi `bench-suite`. |
| `build_docs_pages.sh` | OK. Used by docs.yml. Timestamp in `EDGE_BUILD.txt` (docs, not wheels). |
| `build_wheels_local.sh` | OK. Local matrix; does not replace cibuildwheel manylinux. |
| `canfar_denoise_incontainer.sh` | NOTE. CANFAR; VOS path env. |
| `canfar_gpu_bench_incontainer.sh` | NOTE. CANFAR. |
| `canfar_matrix_bench_incontainer.sh` | NOTE. CANFAR. |
| `check_docs_links.py` | ISSUE PKG-9. Local-only gate. |
| `check_duplicate_cpp.py` | NOTE PKG-24. CI + pre-commit. |
| `check_torch_extra_pins.py` | OK. CI lint + ci_local. Vacuous pass on macOS (linux markers). |
| `ci_local.sh` | ISSUE PKG-9. Not a full CI mirror. |
| `cibuildwheel.sh` | OK. Wrapper around cibuildwheel 4.1.x. |
| `cibw_before_build.sh` | ISSUE PKG-2. No bzip2/zlib-devel install on linux images. |
| `cibw_test.sh` | NOTE PKG-17. Release-smoke only. |
| `clean_install_smoke.sh` | ISSUE PKG-7. Broken. |
| `fetch_canfar_bench_vos.sh` | NOTE. VOS. |
| `fetch_cfht_calib_frames.sh` | OK. Examples/bench data. |
| `fetch_cfht_megacam_sample.sh` | OK. |
| `fetch_cfht_megapipe_sample.sh` | OK. |
| `fetch_example_samples.sh` | OK. |
| `fetch_rgb_sky_samples.sh` | OK. Referenced by `examples/example_rgb_sky.py`. |
| `gpu-bootstrap.sh` | ISSUE PKG-16. Pin not lane-rendered. |
| `gpu-env-loader.sh` | OK. |
| `import_canfar_bench_artifacts.py` | OK. Wired from launcher. |
| `launch_canfar_denoise.sh` | NOTE. |
| `launch_canfar_gpu_bench.sh` | NOTE. |
| `launch_canfar_matrix_grid.sh` | NOTE. |
| `patch_bench_docs.py` | OK. Orchestrates render_*.py. |
| `patch_canfar_exhaustive_docs.sh` | OK. |
| `publish_canfar_bench_vos.sh` | NOTE. |
| `publish_canfar_wheel_bundle.sh` | NOTE. Uses `sha256sum` (Linux/CANFAR). |
| `release_lane.py` | OK. Does not cover gpu-bootstrap / docs / recipe cfitsio. |
| `release_notes.py` | OK. Used by github_release job. |
| `render_bench_deficits.py` | OK. Via patch_bench_docs. |
| `render_bench_highlights.py` | OK. |
| `render_bench_iopath_table.py` | OK. pixi `bench-table-render`. |
| `render_bench_ml.py` | OK. |
| `render_bench_quick.py` | OK. |
| `render_full_benchmarks_table.py` | OK. |
| `run_exhaustive_bench_and_patch_docs.sh` | OK. |
| `selfcheck_canfar_launcher.sh` | OK. |
| `sync_docs_examples.sh` | OK. |
| `torch_lanes.json` | NOTE PKG-20. Single 2.13 lane. |
| `update_changelog.py` | OK. preflight `changelog-check`; not in GHA lint. |
| `verify_wheel_cuda_canfar.sh` | NOTE. Optional CUDA tier. |
| `verify_wheel_cuda_canfar_incontainer.sh` | NOTE. |
| `verify_wheel_matrix.sh` | OK. |

### Repo hygiene

| File | Status |
|---|---|
| `.pre-commit-config.yaml` | OK. ruff 0.16.7 vs pixi `ruff>=0.16.1,<0.17`. duplicate-cpp local hook. |
| `.gitignore` | OK. `extern/cfitsio/`, `.cursor/reviews/`, wheels, pixi, site, `*.tar.gz`. |
| `.gitattributes` | OK. `pixi.lock merge=binary linguist-generated`. |

### `.cursor/` (agent infra — not product API)

| File | Status |
|---|---|
| `docs-review-progress.md` | NOTE PKG-3/25. Stale PyPI token. |
| `harness/.gitignore` | NOTE. Ignores trajectories/failures/state. |
| `harness/config.json` | NOTE PKG-25. `verify_full` ≠ release-gate. |
| `harness/deep-review-1-faithfulness.md` | NOTE. Historical. |
| `harness/docs-audit-1.1.0.md` | NOTE PKG-25. |
| `harness/matrix-grid-20260804.md` | NOTE. |
| `harness/plans/perf-parity-backlog-1-10.md` | NOTE. |
| `harness/playbook.md` | NOTE. Includes macos-libomp-dup (matches code). |
| `hooks.json` | NOTE. sessionStart / stop. |
| `hooks/harness-session-start.sh` | NOTE. |
| `hooks/harness-stop.sh` | NOTE. |
| `jules-ledger.md` | NOTE. |
| `jules.md` | NOTE. |
| `post-1.0-backlog.md` | NOTE PKG-4/25. `[test]` extra already listed; torch 2.10 sentence is stale. |
| `skills/docs-authoring-review/SKILL.md` | NOTE. |
| `skills/release-api-freeze-review/SKILL.md` | NOTE. Pre-tag API review (companion to this packaging audit). |
| `skills/release-api-freeze-review/scripts/inventory_public_api.py` | NOTE. |

### `.dsh/` (agent infra)

| File | Status |
|---|---|
| `README.md` | NOTE PKG-26. Included in sdist. |
| `cordis.patch.yml` | NOTE. Disables bash sandbox on win32. |
| `skills/science-core/SKILL.md` | NOTE. |
| `skills/torchfits-dev/SKILL.md` | NOTE. Mirrors AGENTS verify tiers. |

### `overrides/` (docs theme)

| File | Status |
|---|---|
| `home.html` | OK. Hero; not a packaging contract. |
| `main.html` | OK. Edge banner. |
| `partials/actions.html` | OK. Hide edit/view for generated pages. |

---

## Exclusions (not hand-audited)

| Item | Why |
|---|---|
| `pixi.lock` pin-by-pin | 26 482 lines; lock format v7; platforms linux-64 + osx-arm64. Checked header/platforms only. Solver output can drift on `pixi lock` without a packaging-policy bug. |
| `docs/assets/katex/**` | 8 tracked binaries/fonts/min.js (zensical math). Present in the GitHub sdist (~0.3 MB for `katex.min.js`). Not packaging logic. |
| `docs/assets/bench/**` CSVs and gallery PNGs | Dominate sdist size; content belongs to a docs/bench audit. |
| `src/torchfits/**` product code except `__init__.py` KMP and `cpp_src/CMakeLists.txt` | Out of this packaging scope. |
| `tests/**` except isolation/release-smoke/bz2 as evidence | Out of scope except where they document extras/CI skip behavior. |
| Live PyPI HTML / warehouse UI | Inferred from wheel METADATA + successful `upload_pypi`. |
| conda-forge upload | Recipe says not upstreamed; `pixi run build` not executed in this pass. |

---

## Published `v1.1.0` artifact snapshot (2026-08-26)

GitHub Release: 15 wheels + sdist. PyPI upload job: success.

| Kind | Count / note |
|---|---|
| manylinux x86_64 (`manylinux_2_27_x86_64` / `manylinux_2_28`) | cp310–cp314 |
| manylinux aarch64 (`manylinux_2_26_aarch64` / `manylinux_2_28`) | cp310–cp314 |
| macosx_11_0_arm64 | cp310–cp314 |
| Windows / musl / macos x86_64 | none |
| sdist on GitHub | `torchfits-1.1.0.tar.gz` 6.3 MB; **not** on PyPI |
| Wheel Requires-Dist | `torch<2.14,>=2.13`, numpy, pyarrow |
| Wheel licenses | MIT + CFITSIO file |
| Linux `_C.so` NEEDED | libz + libtorch\*; **no libbz2** |
| macOS `_C.so` | `@rpath` libtorch\*; `/usr/lib/libz.1.dylib`, `libbz2.1.0.dylib`, `libcurl.4.dylib` |

Tag history: wheel workflow failed twice on macOS `vendor.sh` sha256 tooling, then succeeded after the OS-select fix (now under changelog **Unreleased**). The `v1.1.0` ref currently points at that later commit.

---

## Suggested fix order (guidance only — not implemented)

1. Make CI Lint install the same typecheck deps as pixi (or drop mypy from the bare job and keep it on release-gate only) so `main` is green.
2. Install bzip2 headers in cibuildwheel linux `before-all` (or fail the build if `TORCHFITS_USE_BZIP2=ON` and BZip2 is missing) so Linux wheels match the 1.1.0 bz2 claim.
3. Align `docs/release.md` with OIDC; fix `vendor.sh` docs; fix `clean_install_smoke.sh`; drop or fence the Windows source tab.
4. Repair extras (`[test]` runnable, `[dev]`/`[bench]` markers for 3.10).
5. Decide conda: one backend, vendor in the build script, drop stale 4.6.x run dep, ship CFITSIO license.
6. Turn sanitizer on for `push` to main (or document it as PR-only) and add a first green run before relying on it.
7. Optionally exclude `pixi.lock` / docs bench CSVs from the GitHub sdist; add Linux/macOS Trove classifiers; skip `*-win_amd64`.
