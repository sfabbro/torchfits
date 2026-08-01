# Release Checklist

Maintainer runbook for cutting a release.

## 1. Version sync

Confirm the version triplet matches in:

- `pyproject.toml` (`version = "X.Y.Z"`)
- `pixi.toml`
- `src/torchfits/__init__.py` (`__version__`)

For native wheels, also confirm the PyTorch minor-version range is identical in
the build-system, project runtime, and Pixi build/host/run dependencies
(wheels stay on the **2.10** ABI lane unless a new lane is intentionally cut),
and that the `[cpu]` / `[cuda]` extra pins in `pyproject.toml` match the newest
2.10.x build on the PyTorch indexes (`https://download.pytorch.org/whl/cpu`
and `/cu128`).

## 2. Changelog

Finalize the entry in `docs/changelog.md`. Follow [Keep a Changelog](https://keepachangelog.com/) format.

## 3. Tests and gates

```bash
pixi run preflight-push
pixi run test
pixi run ci-local
pixi run release-gate
```

All must pass. `release-gate` runs upstream parity smoke tests, docs integrity
checks, the docs contract (`docs-contract` + `docs-links`), and the runnable
example scripts. `ci-local` and the CI lint job also run `check-torch-pins`,
which resolves the `[cpu]` / `[cuda]` extra pins against the PyTorch indexes
on the wheel ABI lane.

## 4. Public-API freeze (SemVer 1.0 / breaking cuts)

Before tagging a SemVer `1.0.0` (or any intentional breaking cut), run the
release API freeze review under
`.cursor/skills/release-api-freeze-review/` and fix drift between
`docs/api*.md` and the exported surface.

## 5. Correctness gates

Covered by `release-gate`; re-run narrowly if needed:

```bash
pixi run pytest tests/test_fitsio_upstream_smoke.py tests/test_astropy_upstream_smoke.py -q
pixi run pytest tests/test_package_isolation.py tests/test_docs_integrity.py -q
```

## 6. Benchmark evidence

Published multi-host scorecard (MPS + CANFAR CPU + CANFAR CUDA):

```bash
pixi run bench-install
bash scripts/selfcheck_canfar_launcher.sh
# Launch CANFAR first (async), then local:
pixi run bench-exhaustive-canfar-cpu
pixi run bench-exhaustive-canfar-cuda
pixi run bench-exhaustive-local
# After CANFAR finishes:
bash scripts/fetch_canfar_bench_vos.sh exhaustive_cpu_<stamp>
bash scripts/fetch_canfar_bench_vos.sh exhaustive_cuda_<stamp>
pixi run bench-release-scorecard -- \
  benchmarks_results/exhaustive_mps_<stamp> \
  benchmarks_results/exhaustive_cpu_<stamp> \
  benchmarks_results/exhaustive_cuda_<stamp>
```

Mirror CSVs into `docs/assets/bench/<run-id>/` and update Published paths in
`docs/benchmarks.md`. Companion suites: `pixi run bench-megacam`, `bench-ml`.

Quick local smoke (not a published scorecard): `pixi run bench-all` /
`pixi run bench-mps`. Manual CI refresh: `.github/workflows/bench-report.yml`
(`workflow_dispatch` only; CPU-only).

Repository: https://github.com/astroai/torchfits.

**PyPI trusted publishing:** `astroai/torchfits` is registered; tag pushes
trigger `.github/workflows/build_wheels.yml`.

Do not make new performance claims unless the benchmark run is archived and the
comparison target is listed in `docs/parity.md`.

## 7. Parity and docs contract

- [ ] `docs/parity.md` marks every major FITS feature as supported, partial,
      unsupported, or out of scope.
- [ ] `benchmarks/replays/upstream_sources.json` references the parity tests
      that justify comparator claims.
- [ ] README and docs do not claim torchfits ownership of WCS, sphere geometry,
      HEALPix, or sky-domain simulation.
- [ ] Install docs still document CPU-only (no CUDA libs) and GPU torch index
      recipes, and the `[cpu]` / `[cuda]` extra pins track the current 2.10.x
      wheel lane.

## 8. Local artifact check (optional)

```bash
bash scripts/clean_install_smoke.sh
# or manually:
pip wheel . --no-deps --no-build-isolation -w dist
twine check dist/*
```

Smoke-test the wheel in a fresh virtualenv (the script does this).

## 9. Tag and push

```bash
git add -A
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

## 10. Publish

Create a GitHub release for `vX.Y.Z` with **user-facing** notes (not an
internal checklist). Prefer writing the body yourself over
`generate_release_notes` alone.

Suggested shape:

1. **Install** — `pip install torchfits==X.Y.Z`, Python / PyTorch versions, docs URL.
2. **Highlights** — what a user can do now, with short copy-paste examples.
3. **Breaking changes** — before/after table when needed.
4. **Links** — changelog, compare URL, PR.

Do **not** lead with review filenames, logo changes, or bench run IDs unless
they are the product. Put evidence in the changelog / docs site.

Publishing triggers `.github/workflows/build_wheels.yml`, which:

1. Runs tests.
2. Builds wheels on Linux and macOS plus sdist.
3. Uploads to [PyPI](https://pypi.org/project/torchfits/) via trusted publishing.

## 11. Post-release verification

- [ ] `pip install torchfits==X.Y.Z` works in a fresh environment.
- [ ] `import torchfits; print(torchfits.__version__)` shows correct version.
- [ ] `torchfits.read(...)` runs without import errors.
- [ ] [Stable docs](https://astroai.github.io/torchfits/) load (latest `v*` tag,
      built when `main` runs `docs.yml` after the release push).
- [ ] [Edge docs](https://astroai.github.io/torchfits/edge/) load (tip of `main`).
  Docs deploy only from `main` (not from the tag event) so Pages protection and
  concurrency do not cancel the post-release publish.
- [ ] Changelog and release notes links resolve.
