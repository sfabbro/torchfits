# Release Checklist

Maintainer runbook for cutting a release.

## 1. Version sync

Confirm the version triplet matches in:

- `pyproject.toml` (`version = "X.Y.Z"`)
- `pixi.toml`
- `src/torchfits/__init__.py` (`__version__`)

## 2. Changelog

Finalize the entry in `docs/changelog.md`. Follow [Keep a Changelog](https://keepachangelog.com/) format.

## 3. Tests

```bash
pixi run test
```

All tests must pass.

## 4. Correctness gates

Run FITS upstream parity gates:

```bash
pixi run pytest tests/test_fitsio_upstream_smoke.py tests/test_astropy_upstream_smoke.py -q
pixi run pytest tests/test_package_isolation.py tests/test_docs_integrity.py -q
```

All gates must pass.

## 5. Benchmark evidence

Run the exhaustive benchmark suite:

```bash
pixi run bench-all
pixi run bench-mps    # Apple Silicon (device=mps)
# Linux CUDA: pixi run -e bench-gpu bench-gpu
```

Regenerate the I/O transport table:

```bash
pixi run bench-table-render -- --csv benchmarks_results/<run-id>/results.csv
```

Scheduled CI: `.github/workflows/bench-report.yml` (weekly + manual).

Repository: https://github.com/astroai/torchfits.

**PyPI trusted publishing:** register `astroai/torchfits` before **v0.5.0** (final).
`0.5.0b1` was published from the pre-transfer repo; no retroactive re-publish needed.

Do not make new performance claims unless the benchmark run is archived and the
comparison target is listed in `docs/parity.md`.

## 6. Parity and docs contract

- [ ] `docs/parity.md` marks every major FITS feature as supported, partial,
      unsupported, or out of scope.
- [ ] `benchmarks/replays/upstream_sources.json` references the parity tests
      that justify comparator claims.
- [ ] README and docs do not claim torchfits ownership of WCS, sphere geometry,
      HEALPix, or sky-domain simulation.

## 7. Local artifact check (optional)

```bash
pip wheel . --no-deps --no-build-isolation -w dist
twine check dist/*
```

Smoke-test the wheel in a fresh virtualenv.

## 8. Tag and push

```bash
git add -A
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

## 9. Publish

Create a GitHub release for `vX.Y.Z`.

Publishing triggers `.github/workflows/build_wheels.yml`, which:

1. Runs tests.
2. Builds wheels on Linux and macOS plus sdist.
3. Uploads to [PyPI](https://pypi.org/project/torchfits/) via trusted publishing.

## 10. Post-release verification

- [ ] `pip install torchfits==X.Y.Z` works in a fresh environment.
- [ ] `import torchfits; print(torchfits.__version__)` shows correct version.
- [ ] `torchfits.read(...)` runs without import errors.
- [ ] Changelog and release notes links resolve.
