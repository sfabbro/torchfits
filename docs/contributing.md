# Contributing

## Development setup

```bash
git clone https://github.com/astroai/torchfits.git
cd torchfits
pixi install
pixi run test
```

The project uses [pixi](https://pixi.sh/) for environment management, [ruff](https://github.com/astral-sh/ruff) for linting, and [pytest](https://docs.pytest.org/) for testing.

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
  `[feature.<name>.tasks]` block of `pixi.toml` and inside
  `scripts/gpu-bootstrap.sh` run inside an activated pixi env and are
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
extern/cfitsio/         Vendored CFITSIO sources
tests/                  Unit and integration tests
benchmarks/             Benchmark scripts and replay gates
docs/                   Documentation
examples/               Runnable example scripts
```

## Native extension

The C++ extension is built by [scikit-build-core](https://scikit-build-core.readthedocs.io/) with [nanobind](https://nanobind.readthedocs.io/) bindings. Populate vendored sources with:

```bash
./extern/vendor.sh
```

Rebuild after C++ changes:

```bash
pip install -e . --no-build-isolation
```

### C++ code conventions

- **No inline RAII structs in `.cpp` files.** Use the shared guards from the headers below instead.
- `FitsHandleGuard` (`cache.h`) — RAII wrapper for `fitsfile*` handles. Two modes: `cached=false` (calls `fits_close_file`) and `cached=true` (calls `release_cached`).
- `MMapHandle` (`hardware.h`) — RAII wrapper for `mmap` regions. Construct with a filename (open + mmap) or adopt an existing mapping via `MMapHandle(ptr, size, fd)`.
- If a new resource requires RAII, add the guard to the appropriate shared header rather than defining an inline struct at the usage site.

## Testing

Minimum before a PR:

```bash
pixi run pytest tests/test_api.py tests/test_table.py -q
```

Full suite:

```bash
pixi run test
```

Upstream parity gates (require comparison libraries):

```bash
pixi run fits-upstream-gate
```

## Benchmarks

Quick FITS benchmark sweep:

```bash
pixi run bench-all
```

Include benchmark evidence in PRs that touch performance-sensitive paths.

## Documentation policy

- `README.md`: user-facing overview only.
- `docs/api.md`: public API reference. Update if a PR changes a public API.
- `docs/roadmap.md`: FITS I/O roadmap and parity tiers.
- `docs/parity.md`: compatibility matrix. Update if support status changes.
- `docs/changelog.md`: release notes, [Keep a Changelog](https://keepachangelog.com/) format.
- `docs/benchmarks.md`: benchmark methodology and results.

## PR guidelines

- Keep PRs focused on a single concern.
- Include tests for behavior changes.
- Run `pixi run lint` and fix issues before submitting.
- Do not commit local scratch files, benchmark artifacts, or `.env` files.

## Release process

See [release.md](release.md) for the maintainer checklist.
