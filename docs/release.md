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

Run all upstream parity gates:

```bash
pixi run wcs-upstream-gate
pixi run fits-upstream-gate
pixi run healsparse-upstream-gate
pixi run sphere-upstream-gate
pixi run sphere-upstream-spin-gate
pixi run sphere-upstream-interp-edge-gate
pixi run sphere-upstream-polygon-gate
pixi run real-data-gate
```

All gates must pass.

## 5. Benchmark evidence

Run the exhaustive benchmark suite:

```bash
pixi run bench-all
```

Run the ML loader benchmark:

```bash
pixi run python benchmarks/bench_ml_loader.py \
  --device cpu --shape 2048,2048 --n-files 50 \
  --batch-size 4 --num-workers 4 --epochs 3 --repeats 5 --warm-cache
```

Update `docs/benchmarks.md` with the run ID, date, and snapshot tables.

## 6. Local artifact check (optional)

```bash
pip wheel . --no-deps --no-build-isolation -w dist
twine check dist/*
```

Smoke-test the wheel in a fresh virtualenv.

## 7. Tag and push

```bash
git add -A
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

## 8. Publish

Create a GitHub release for `vX.Y.Z`.

Publishing triggers `.github/workflows/build_wheels.yml`, which:

1. Runs tests.
2. Builds wheels on Linux and macOS plus sdist.
3. Uploads to [PyPI](https://pypi.org/project/torchfits/) via trusted publishing.

## 9. Post-release verification

- [ ] `pip install torchfits==X.Y.Z` works in a fresh environment.
- [ ] `import torchfits; print(torchfits.__version__)` shows correct version.
- [ ] `torchfits.read(...)` runs without import errors.
- [ ] Changelog and release notes links resolve.
