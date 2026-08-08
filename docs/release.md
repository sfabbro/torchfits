# Release Checklist

Maintainer runbook for cutting a release.

## 0. Release lane

Each torchfits release targets **one PyTorch minor** (the wheel "lane"),
because PyTorch has no stable C++ ABI across minors. Lanes live in
`scripts/torch_lanes.json` (single source of truth):

| PyTorch lane | torchfits release |
|---|---|
| 2.13.x | 1.2.3 (current cut, not yet released) |

The repo tracks the current lane. A new lane is added to the map only when a
real backport release is cut, with that release's actual version.

## 1. Version sync

Apply the lane everywhere in one step:

```bash
python scripts/release_lane.py --lane <X.Y> --apply
pixi run check-lane
```

`--apply` rewrites the version + torch pin in:

- `pyproject.toml` (`version`, `[cpu]` / `[cuda]` extra pins)
- `constraints-wheel.txt` (wheel ABI lane)
- `pixi.toml`
- `packaging/conda/recipe.yaml` (torch_pin)
- `src/torchfits/__init__.py` (`__version__`)

Release candidates (pre-tag soaks) render the lane base plus a PEP 440
prerelease suffix:

```bash
python scripts/release_lane.py --lane <X.Y> --prerelease rc<N> --apply
```

e.g. `--prerelease rc5` on lane 2.13 renders `1.2.3rc5` everywhere, and
`--check` / `check-lane` accept the rc state as the lane's base version.
A plain `--apply` (no `--prerelease`) always finalizes back to the map
version.

`check-lane` fails unless all five agree with `scripts/torch_lanes.json`.
Update the compatibility / install docs (README.md, `docs/install.md`,
`docs/compatibility.md`) when the current-lane numbers change.

## 2. Changelog

Finalize the entry in `docs/changelog.md`. Follow [Keep a Changelog](https://keepachangelog.com/) format.

## 3. Tests and gates

```bash
pixi run preflight-push
pixi run test
pixi run ci-local
pixi run release-gate
```

All must pass. `preflight-push` includes `check-lane` (all five lane files
agree). `ci-local` and the CI lint job also run `check-torch-pins`, which
resolves the `[cpu]` / `[cuda]` extra pins against the PyTorch indexes on the
wheel ABI lane.

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
# Local leg: bench-exhaustive-local prints its run-id (exhaustive_<cpu|mps>_<stamp>).
pixi run bench-release-scorecard -- \
  benchmarks_results/exhaustive_<cpu-or-mps>_<localstamp> \
  benchmarks_results/exhaustive_cpu_<stamp> \
  benchmarks_results/exhaustive_cuda_<stamp>
```

Mirror CSVs into `docs/assets/bench/<run-id>/` and update Published paths in
`docs/benchmarks.md`. Companion suites: `pixi run bench-megacam`, `bench-ml`.

Quick local smoke (not a published scorecard): `pixi run bench-all` /
`pixi run bench-mps`. Manual CI refresh: `.github/workflows/bench-report.yml`
(`workflow_dispatch` only; CPU-only).

Repository: https://github.com/astroai/torchfits.

**PyPI publishing:** `astroai/torchfits` is registered; tag pushes trigger
`.github/workflows/build_wheels.yml` (publishes via a PyPI API token,
`secrets.PYPI_API_TOKEN`).

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
      recipes, and the `[cpu]` / `[cuda]` extra pins track the current wheel
      lane (`pixi run check-torch-pins` enforces this).

## 8. Local artifact check (optional)

Wheel matrix (lanes × Python, built in tar copies so the working tree stays
clean):

```bash
bash scripts/build_wheels_local.sh --lanes <list> --pythons "3.10 3.11 3.12 3.13 3.14" dist-local
bash scripts/verify_wheel_matrix.sh --jobs 4 dist-local
```

Conda package (bare-cmake build via pixi; verify in a fresh env):

```bash
pixi run build
# verify: import + torchfits --help in a fresh pixi env
```

CANFAR CUDA tier (optional, soft-fail): see
`scripts/verify_wheel_cuda_canfar.sh` (`TORCHFITS_WHEEL_URL` for unpublished
wheels). Or a plain smoke of the published artifact:

```bash
bash scripts/clean_install_smoke.sh
# or manually:
pip wheel . --no-deps --no-build-isolation -w dist
twine check dist/*
```

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

1. Runs tests (each job resolves the lane's torch pin via
   `release_lane.py --print-pins`).
2. Builds wheels on Linux and macOS plus sdist (cp310–cp314, torch pinned to
   the lane).
3. Uploads to [PyPI](https://pypi.org/project/torchfits/) via a PyPI API
   token (`secrets.PYPI_API_TOKEN`), not trusted publishing.

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
