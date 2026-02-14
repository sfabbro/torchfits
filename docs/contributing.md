# Contributing

This guide is for contributors and maintainers.
User-facing usage lives in `../README.md`.

## Development Setup

```bash
pixi install
pixi run dev
```

Common commands:

```bash
pixi run test
pixi run lint
```

## Repository Layout

- Python package: `src/torchfits/`
- Native extension: `src/torchfits/cpp_src/`
- Vendored native deps: `extern/cfitsio/`, `extern/wcslib/`
- Tests: `tests/`
- Benchmarks: `benchmarks/`
- Docs: `docs/`

## Native Build Model

Default builds use vendored native libraries.
Populate/update vendored sources with:

```bash
./extern/vendor.sh
```

By default, this resolves the latest published dependency tags.
Pin explicit versions when needed:

```bash
./extern/vendor.sh --cfitsio-version cfitsio-4.6.2 --wcslib-version v8.5
```

Optional system-library mode:

- `-DTORCHFITS_USE_VENDORED_CFITSIO=OFF`
- `-DTORCHFITS_USE_VENDORED_WCSLIB=OFF`

## Test Expectations

Minimum pre-PR checks:

```bash
pixi run pytest tests/test_api.py -q
pixi run pytest tests/test_table.py -q
```

Recommended for broader changes:

```bash
pixi run pytest tests/ -q
```

## Benchmarks

Benchmark tasks currently defined in `pixi.toml`:

- `pixi run bench`
- `pixi run bench-fast`
- `pixi run bench-fast-stable`
- `pixi run bench-core`
- `pixi run bench-table`
- `pixi run bench-table-arrow`
- `pixi run bench-table-arrow-diverse`
- `pixi run bench-all`
- `pixi run bench-all-keep`
- `pixi run bench-all-runner`
- `pixi run bench-all-force`
- `pixi run bench-transforms`
- `pixi run bench-buffer`
- `pixi run bench-cache`
- `pixi run bench-focused`
- `pixi run bench-pytest`

Additional benchmark script:

- `pixi run python benchmarks/benchmark_ml_loader.py --device cpu`

Suggested regression workflow:

```bash
pixi run bench-fast-stable
# make change
pixi run bench-fast-stable
```

For investigation, narrow scope with `bench-core`, `bench-table`, `bench-cache`.
`bench-all` is exhaustive and can take over an hour on a laptop-class machine.

## Documentation Policy

- Root `README.md`: user-facing only.
- `docs/api.md`: public API reference.
- `docs/changelog.md`: release notes.
- `docs/examples.md` and `docs/benchmarks.md`: operational guides.

If a PR changes a public API, update `docs/api.md` in the same PR.

## Release Checklist (Maintainers)

Use `docs/release.md` as the canonical runbook.

Minimum gates:

1. Run release benchmarks and update `docs/benchmarks.md`.
2. Ensure `docs/changelog.md` is final and versions are synced in:
   - `pyproject.toml`
   - `pixi.toml`
   - `src/torchfits/__init__.py`
3. Ensure CI is green.
4. Build and validate artifacts:

```bash
pixi run python -m build
pixi run twine check dist/*
```

5. Smoke-test install wheel/sdist in a fresh virtualenv.
6. Tag `vX.Y.Z` and publish.

## PR Hygiene

- Keep PRs scoped.
- Include tests for behavior changes.
- Include benchmark evidence for performance-sensitive changes.
- Do not commit local scratch/benchmark artifacts.
