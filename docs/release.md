# Release Checklist

Maintainer runbook for cutting a release.

## 0. Release lane

Each torchfits release targets **one PyTorch minor** (the wheel "lane"),
because PyTorch has no stable C++ ABI across minors. Lanes live in
`scripts/torch_lanes.json` (single source of truth):

| PyTorch lane | torchfits release |
|---|---|
| **2.14.x** | **1.3.1** (lane base. CUDA flavors cu126/cu130) |

`scripts/torch_lanes.json` currently defines a single lane; auxiliary
`+torchNNN` local-version wheels are created by adding lanes to that file and
re-running `scripts/release_lane.py --apply`.

## 1. Version sync

Apply the lane everywhere in one step:

```bash
pixi run python scripts/release_lane.py --lane <X.Y> --apply
pixi run check-lane
```

`--apply` rewrites the version + torch pin in:

- `pyproject.toml` (`version`, `[cpu]` / `[cuda]` extra pins)
- `constraints-wheel.txt` (wheel ABI lane)
- `pixi.toml`
- `packaging/conda/recipe.yaml` (torch_pin)
- `src/torchfits/__init__.py` (`__version__`)

Release candidates (pre-tag validation runs) render the lane base plus a PEP 440
prerelease suffix:

```bash
pixi run python scripts/release_lane.py --lane <X.Y> --prerelease rc<N> --apply
```

e.g. `--prerelease rc5` on lane 2.14 appends `rc5` to that lane's `torchfits_version`, and
`--check` / `check-lane` accept the rc state as the lane's base version.
A plain `--apply` (no `--prerelease`) always finalizes back to the map
version.

**Release-candidate gate.** From 1.3.1 onward, every stable tag is preceded by
an `rc` tag for the same version. The rc and the stable tag carry the same
tree apart from the version strings. Publish `vX.Y.Zrc1`, run the gates and
install checks against that artifact, then finalize and tag the stable
version. `scripts/check_release_rc.py` enforces the gate before PyPI upload,
and section 9 runs it locally. `1.3.0` (2026-10-22) was published without an
rc.

`check-lane` fails unless all five agree with `scripts/torch_lanes.json`.
Update the compatibility / install docs (README.md, `docs/install.md`,
`docs/compatibility.md`) when the current-lane numbers change.

## 2. Changelog

`docs/changelog.md` follows [Keep a Changelog](https://keepachangelog.com/).
Work in flight lives under a versionless `## Unreleased` heading on top —
edge docs must never name an unreleased version.

- While developing, fold conventional commits into `## Unreleased`:
  ```bash
  pixi run changelog-update          # merge commits since the newest tag (deduped)
  pixi run changelog-check           # CI-style drift check
  ```
  Generated bullets are a starting point — curate the prose afterwards.
- When cutting a release, stamp the real version and date (this also refreshes
  the `[Unreleased]` / compare link refs at the bottom of the file and opens a
  fresh empty `## Unreleased`):
  ```bash
  pixi run changelog-release -- <version>
  ```
  Run it before `git tag`, so the tag points at the finalized changelog.

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
pixi run pytest tests/test_package_isolation.py tests/test_docs_integrity.py -q
```

Upstream parity smokes run in torchfits-harness.

## 6. Benchmark evidence

Benchmark runs, published CSVs, and the parity matrix live in
[torchfits-harness](https://github.com/astroai/torchfits-harness).

Repository: https://github.com/astroai/torchfits.

**Publishing:** `astroai/torchfits` is registered on PyPI. A release-candidate
tag push builds and validates the complete artifact matrix and creates a GitHub
prerelease, but never uploads to PyPI. A stable tag push only builds and
validates. After that workflow succeeds, run it manually from the exact stable
tag with `publish=true`; the guarded jobs publish through OIDC on the `pypi`
GitHub Environment and create the stable GitHub release. Fork runs cannot
publish.

Do not make new performance claims unless the benchmark run is archived in
torchfits-harness.

## 7. Parity and docs contract

- [x] Parity and migration notes live in torchfits-harness. This repo's
      [evaluation](evaluation.md) page points there.
- [x] Core tests and the default environment do not import astropy or fitsio.
- [x] README and docs do not claim torchfits ownership of WCS, sphere geometry,
      HEALPix, or sky-domain simulation.
- [x] Install docs still document CPU-only (no CUDA libs) and GPU torch index
      recipes, and the `[cpu]` / `[cuda]` extra pins track the current wheel
      lane (`pixi run check-torch-pins` enforces this).

## 8. Local artifact check (optional)

Wheel matrix (lanes × Python, built in tar copies so the working tree stays
clean):

```bash
bash scripts/build_wheels_local.sh --lanes <list> --pythons "3.11 3.12 3.13 3.14 3.15" dist-local
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
pixi run python scripts/check_release_rc.py --version X.Y.Z
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
2. Builds wheels (Linux x86_64 + aarch64 on cp311–cp315; macOS arm64 on
   cp311–cp314, because torch 2.14's cp315 macOS wheel is ABI-broken —
   [pytorch#200779](https://github.com/pytorch/pytorch/issues/200779); torch
   pinned to the lane via `scripts/cibw_before_build.sh`). An sdist is
   attached to the GitHub Release only — **not** uploaded to PyPI.
3. Uploads **wheels** to [PyPI](https://pypi.org/project/torchfits/) via
   OIDC trusted publishing (`id-token: write` on the `pypi` Environment).
   There is no `password:` / `PYPI_API_TOKEN` in the workflow.

`Comprehensive CI` builds one wheel per PR (job `Wheel smoke`) and asserts the
packaging contract before any tag exists: the native extension and
`torchfits/py.typed` are present, the generated `torchfits/_C.pyi` stub is
shipped and not truncated, both licence files (project + vendored CFITSIO)
are under `dist-info/licenses/`, the metadata version matches
`pyproject.toml`, and no C++ source or test tree leaked in. It then installs
that wheel into a clean venv and runs the same release smoke tests
cibuildwheel runs at tag time.

```bash
pixi run wheel-smoke   # same build + contents check, locally
```

Local / out-of-band builds (same `[tool.cibuildwheel]` config):

```bash
bash scripts/cibuildwheel.sh                 # host arch (Linux Docker or macOS)
CIBW_ARCHS=aarch64 bash scripts/cibuildwheel.sh
bash scripts/verify_wheel_cuda_canfar.sh     # CUDA torch vs the CPU-linked wheel
```

## 11. Post-release verification

These checks intentionally remain unchecked in a development checkout. Run
them only after the tagged release and its package/docs deployments exist.

- [ ] `pip install torchfits==X.Y.Z` works in a fresh environment.
- [ ] `import torchfits; print(torchfits.__version__)` shows correct version.
- [ ] `torchfits.read(...)` runs without import errors.
- [ ] [Stable docs](https://astroai.github.io/torchfits/) load (latest `v*` tag,
      built when `main` runs `docs.yml` after the release push).
- [ ] [Edge docs](https://astroai.github.io/torchfits/edge/) load (tip of `main`).
  Docs deploy only from `main` (not from the tag event) so Pages protection and
  concurrency do not cancel the post-release publish.
- [ ] Changelog and release notes links resolve.
