# Release Runbook

This is the maintainer checklist for cutting a release.

## 1. Final Sanity Checks

Confirm version triplet matches:

- `pyproject.toml`
- `pixi.toml`
- `src/torchfits/__init__.py`

Confirm changelog entry is finalized in `docs/changelog.md`.

## 2. Benchmark Evidence

Run exhaustive benchmark:

```bash
pixi run python benchmarks/benchmark_all.py \
  --profile user \
  --include-tables \
  --output-dir benchmark_results/release_<version>_$(date +%Y%m%d_%H%M%S)
```

Run ML loader benchmark:

```bash
pixi run python benchmarks/benchmark_ml_loader.py \
  --device cpu \
  --shape 2048,2048 \
  --n-files 50 \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 3 \
  --repeats 5 \
  --warm-cache
```

Update `docs/benchmarks.md` with:
- run id/date
- key speedup medians and win/loss counts
- known weak spots

## 3. Optional Local Artifact Check

```bash
pixi run python -m pip wheel . --no-deps --no-build-isolation -w dist
pixi run twine check dist/*
```

Smoke-test wheel install in a fresh environment before publish.
The release workflow builds both wheels and sdist during publish.

## 4. Git + Tag Push

```bash
git add -A
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

## 5. Publish from GitHub Release

Create a GitHub release for `vX.Y.Z` (publish the release in the UI).

Publishing a release triggers `.github/workflows/build_wheels.yml`, which:
- runs tests first
- builds wheels on Linux and macOS plus sdist
- uploads to PyPI via trusted publishing

After workflow success, verify:
- PyPI version availability
- install from PyPI in fresh env
- changelog and release notes links resolve
