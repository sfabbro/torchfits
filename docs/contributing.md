# Contributing

## Development setup

```bash
git clone https://github.com/astroai/torchfits.git
cd torchfits
pixi install --locked
pixi run preflight-push   # quick check while editing
pixi run test             # full unit suite
pixi run current-source-gate  # locked environment + complete local gate
```

The project uses [pixi](https://pixi.sh/) for environment management,
[ruff](https://github.com/astral-sh/ruff) for linting, and
[pytest](https://docs.pytest.org/) for testing. Detailed coding guidelines live in
[`AGENTS.md`](https://github.com/astroai/torchfits/blob/main/AGENTS.md).

The workspace pins Pixi `0.81.0` in `pixi.toml`; `pixi.lock` pins the actual
package artifacts. The reproducible complete local gate is:

```bash
pixi install --locked
pixi run current-source-gate
```

The task repeats the locked install and then runs the full `ci-local` gate.

## Verify tiers

| When | Command |
|------|---------|
| During edits | `pixi run preflight-push` |
| Before push / PR | `pixi run ci-local` |
| Before tag | `pixi run release-gate` |

## Dependency policy

**Every package must be installed through pixi.** Never run `pip install`,
`pip install -e .`, `python -m pip install`, `conda install`, `mamba install`,
`pipx install`, or any equivalent outside of an activated pixi environment.
The pixi lockfile is the reproducibility contract for this repository;
mixing it with bare-pip / bare-conda / system installs breaks the lockfile
and the CI gates. If you find yourself reaching for one of the commands
below, stop and add a `[dependencies]` entry in `pixi.toml` (or a feature's
`[feature.<name>.dependencies]`) instead, then `pixi install`.

**Do**

- `pixi install` — install the workspace as declared in `pixi.toml`.
- `pixi add <pkg>` — add a dependency to `pixi.toml` and re-lock.
- `pixi run <task>` — run scripts and tests; tasks are defined in `pixi.toml`
  and resolve dependencies from the lockfile.

**Don't**

- `pip install ...` / `python -m pip install ...` outside a pixi env.
- `conda install ...` / `mamba install ...` outside a pixi env.
- `pipx install ...` (creates a separate venv that drifts from the lockfile).
- `pip install -e .` typed directly into a shell, even "just to try it".

**Documented exceptions**

- `pip install -e .` invocations inside any `[tasks]` or
  `[feature.<name>.tasks]` block of `pixi.toml`
  run inside an activated pixi env and are
  intentional — they rebuild the C++ extension against the
  pixi-managed dependencies already on `$PATH`. They are not bare-pip
  installs. (In particular, `bench-gpu-install` under
  `[feature.gpu.tasks]` is the canonical pixi-tasked reinstallation of
  the C++ extension with `--no-deps` against the pixi-managed torch.)
- `pip install -r requirements.txt` style flows do not exist in this repo
  and must not be added; pixi is the single source of truth for the
  environment.

## Repository layout

```
src/torchfits/          Python package
src/torchfits/cpp_src/  C++ native extension (nanobind + CFITSIO)
extern/cfitsio/         Vendored CFITSIO sources (via extern/vendor.sh)
tests/                  Unit and integration tests
benchmarks/             Benchmark scripts and replay gates
docs/                   Documentation (zensical)
overrides/              Docs theme templates (zensical custom_dir)
examples/               Runnable example scripts
scripts/                CI, docs, and bench helpers
```

Local-only (gitignored): `build/`, `site/`, `benchmarks_results/` (bench run
output — publish selected CSVs under `docs/assets/bench/<run-id>/`).

## Native extension

The C++ extension is built by [scikit-build-core](https://scikit-build-core.readthedocs.io/) with [nanobind](https://nanobind.readthedocs.io/) bindings. Populate vendored sources with:

```bash
./extern/vendor.sh --cfitsio-version extern/VERSIONS.txt
```

Rebuild after C++ changes (default and test envs do **not** share the
extension — rebuild each env you will run):

```bash
pixi run -- pip install -e . --no-build-isolation
pixi run -e test -- pip install -e . --no-build-isolation
```

Or `pixi run dev` when you want the default-env editable install used by
example verification tasks.

### C++ code conventions

- **No inline RAII structs in `.cpp` files.** Use the shared guards from the headers below instead.
- `FitsHandleGuard` (`cache.h`) — RAII wrapper that always `fits_close_file`s a privately owned `fitsfile*`.
- `MMapHandle` (`hardware.h`) — RAII wrapper for `mmap` regions. Construct with a filename (open + mmap) or adopt an existing mapping via `MMapHandle(ptr, size, fd)`.
- If a new resource requires RAII, add the guard to the appropriate shared header rather than defining an inline struct at the usage site.

## Testing

Minimum before a PR:

```bash
pixi run preflight-push
pixi run -e test -- pytest tests/test_api.py tests/test_table.py tests/test_cli.py -q
```

Full suite / pre-push:

```bash
pixi run test
pixi run current-source-gate
```

Upstream parity + docs contract (also part of the release gate):

```bash
pixi run release-gate
```

### Real-observation tests

Two suites run against real CFHT data — three 1.6 GB MegaPipe mosaics
(21404x20347, single HDU) and ten Rice-compressed MegaCam MEFs (37–41 HDUs,
`1PB(nnnn)` tiles). That data is not in git, so both skip unless it is fetched:

```bash
bash scripts/fetch_cfht_megacam_sample.sh    # ~2.5 GB
bash scripts/fetch_cfht_megapipe_sample.sh  # ~5 GB
pixi run -e test -- pytest tests/test_core_library_real_data.py tests/test_reads_real_data.py -q
```

`tests/test_core_library_real_data.py` is about the **metadata** path: it
compares `torchfits._core` with the torch-linked `torchfits._C` answer for
answer on all 409 (frame, HDU) records, and walks the whole 9 GB corpus in a
fresh interpreter with `torch` and `numpy` both blocked, asserting the
torch-linked extension module is never imported.

`tests/test_reads_real_data.py` goes past the header and holds the **reads**
against `astropy.io.fits` — a separate implementation with its own Rice
decompressor and header parser, so agreement is evidence rather than
self-consistency. Decompressed extensions, the raw variable-length tile
stream, the real `BZERO` convention, and 435 megapixels of mosaic all have to
match exactly.

Run them before a PR that touches FITS parsing, header handling, the HDU
resolution rules, image or table reads, or the core/extension split. Synthetic
fixtures cannot reach the failure modes real data has: a 358-card header with
70-character `HISTORY` values, a `NAXIS=0` primary where `NAXIS1` is genuinely
absent, a compressed `IMAGE` whose HDU 1 still answers `read_table_info`, tile
lengths that vary from 35 bytes to the `1PB(n)` bound, and a file too large to
cache. See [Architecture](architecture.md#the-torch-boundary) for what the
split guarantees.

### Checking the torch boundary yourself

The boundary claims are enforced by tests, but the quickest way to find a leak
is to block the import and run something real:

```bash
python - <<'PY'
import importlib.abc, runpy, sys
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy"}:
            raise ImportError(fullname + " is blocked")
        return None
sys.meta_path.insert(0, _Block())
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[1], run_name="__main__")
PY examples/example_mef_header.py
```

Auditing `examples/` this way is how the `where=`-filter leak was found, and
the remaining `torchfits.open()` leak is recorded in
[Roadmap](roadmap.md#next).

## Benchmarks

Quick FITS benchmark sweep:

```bash
pixi run bench-all
```

Published multi-host benchmark runs (release docs): see [Release Checklist](release.md#6-benchmark-evidence)
(`bench-exhaustive-local` + CANFAR CPU/CUDA + `bench-release-scorecard`).

Include benchmark evidence in PRs that touch performance-sensitive paths.

## Documentation policy

- `README.md`: user-facing overview only.
- `docs/api.md`: public API reference. Update if a PR changes a public API.
- `docs/roadmap.md`: FITS I/O roadmap and parity tiers.
- `docs/parity.md`: compatibility matrix. Update if support status changes.
- `docs/changelog.md`: release notes, [Keep a Changelog](https://keepachangelog.com/) format.
- `docs/benchmarks.md`: benchmark methodology and results.
- Published site: **stable** = latest `v*` tag at
  [astroai.github.io/torchfits](https://astroai.github.io/torchfits/);
  **edge** = tip of `main` at
  [astroai.github.io/torchfits/edge](https://astroai.github.io/torchfits/edge/)
  (no SemVer release required). Local dual tree:
  `pixi run docs-build-pages` or `bash scripts/build_docs_pages.sh`.

## PR guidelines

- Keep PRs focused on a single concern.
- Include tests for behavior changes.
- Run `pixi run preflight-push` (and `pixi run ci-local` before merge/push).
- Do not commit local scratch files, benchmark artifacts, or `.env` files.
- Docs must match the public façade; do not document env vars absent from `src/`.

## Release process

See [release.md](release.md) for the maintainer checklist.
