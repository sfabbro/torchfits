# Contributing

## Development setup

```bash
git clone https://github.com/sfabbro/torchfits.git
cd torchfits
pixi install
pixi run test
```

The project uses [pixi](https://pixi.sh/) for environment management, [ruff](https://github.com/astral-sh/ruff) for linting, and [pytest](https://docs.pytest.org/) for testing.

## Repository layout

```
src/torchfits/          Python package
src/torchfits/cpp_src/  C++ native extension (nanobind + CFITSIO)
src/torchfits/wcs/      Pure-PyTorch WCS implementation
src/torchfits/sphere/   HEALPix, spherical geometry, SHT
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
pixi run wcs-upstream-gate
pixi run fits-upstream-gate
pixi run healsparse-upstream-gate
pixi run sphere-upstream-gate
```

## Benchmarks

Quick regression check:

```bash
pixi run python benchmarks/bench_sentinel.py --initial-repeats 3 --full-repeats 9
```

Full four-domain sweep:

```bash
pixi run bench-all
```

Include benchmark evidence in PRs that touch performance-sensitive paths.

## Documentation policy

- `README.md`: user-facing overview only.
- `docs/api.md`: public API reference. Update if a PR changes a public API.
- `docs/changelog.md`: release notes, [Keep a Changelog](https://keepachangelog.com/) format.
- `docs/benchmarks.md`: benchmark methodology and results.
- `docs/sphere.md`: sphere/HEALPix API reference.

## PR guidelines

- Keep PRs focused on a single concern.
- Include tests for behavior changes.
- Run `pixi run lint` and fix issues before submitting.
- Do not commit local scratch files, benchmark artifacts, or `.env` files.

## Release process

See [release.md](release.md) for the maintainer checklist.
