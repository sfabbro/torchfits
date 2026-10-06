from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "torchfits"
REPO_ROOT = PACKAGE_ROOT.parents[1]


def _requirement_name(requirement: str) -> str:
    """The distribution name of a PEP 508 requirement string.

    Splitting on the specifier/version characters rather than on a comma keeps
    ``torch>=2.13,<2.14`` intact and, more importantly, keeps ``torchvision``
    from being read as torch: a naive ``startswith("torch")`` would treat every
    extra torch-family build requirement as a lane escape hatch.
    """
    return re.split(r"[<>=!~;\[ ]", requirement.strip(), maxsplit=1)[0]


def _resolve_on_index(requirement: str) -> list[str] | None:
    """Versions pip's own resolver picks for ``requirement`` on the real index.

    Returns ``None`` when the index cannot be reached so the caller can skip
    rather than fail on a machine with no network.
    """
    report = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--quiet",
            "--ignore-installed",
            "--no-deps",
            "--report",
            "-",
            requirement,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    if report.returncode != 0:
        return None
    return [
        entry["metadata"]["version"] for entry in json.loads(report.stdout)["install"]
    ]


def _wheel_lane_spec() -> str:
    constraints = (REPO_ROOT / "constraints-wheel.txt").read_text(encoding="utf-8")
    match = re.search(r"torch(>=2\.\d+,<2\.\d+)", constraints)
    assert match, "constraints-wheel.txt must pin torch to the current ABI lane"
    return match.group(1)


def test_native_torch_abi_range_is_consistent() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    pixi = tomllib.loads((REPO_ROOT / "pixi.toml").read_text(encoding="utf-8"))
    lane_spec = _wheel_lane_spec()

    # The build requirement and the runtime requirement must be the SAME pin.
    # Under pip's default build isolation `build-system.requires` is what the
    # build environment resolves, so a floor with no upper bound compiles
    # against the newest torch on PyPI and CMake stamps that minor into the
    # extension -- which then refuses to import under the lane this release is
    # pinned to. `test_native_extension_rejects_mismatched_torch_runtime` below
    # measures that refusal; this is the assertion that keeps the two pins from
    # ever disagreeing again.
    assert f"torch{lane_spec}" in pyproject["build-system"]["requires"]
    # Published wheels carry the ABI lane pin (one torchfits release per torch
    # minor, see scripts/torch_lanes.json).
    assert f"torch{lane_spec}" in pyproject["project"]["dependencies"]
    assert "torch>=2.10" not in pyproject["build-system"]["requires"]
    assert "torch>=2.10,<2.11" not in pyproject["build-system"]["requires"]
    assert "torch>=2.10,<2.11" not in pyproject["project"]["dependencies"]
    # Dev pixi / published wheels stay on the wheel ABI lane.
    for section in ("build-dependencies", "host-dependencies", "run-dependencies"):
        assert pixi["package"][section]["pytorch"] == lane_spec
    assert pixi["dependencies"]["pytorch"] == lane_spec

    workflow_paths = (
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "build_wheels.yml",
        REPO_ROOT / ".github" / "workflows" / "bench-report.yml",
    )
    for path in workflow_paths:
        workflow = path.read_text(encoding="utf-8")
        # CI resolves the lane from scripts/torch_lanes.json at runtime.
        assert 'pip install "torch$TORCH_PIN"' in workflow
        assert "pip install torch " not in workflow

    # constraints-wheel.txt must stay in lockstep with the newest lane in
    # torch_lanes.json (the wheel ABI lane every workflow builds against).
    lanes = json.loads(
        (REPO_ROOT / "scripts" / "torch_lanes.json").read_text(encoding="utf-8")
    )
    current = max(lanes)
    major, minor = map(int, current.split("."))
    assert lane_spec == f">={major}.{minor},<{major}.{minor + 1}"

    wheel_workflow = (
        REPO_ROOT / ".github" / "workflows" / "build_wheels.yml"
    ).read_text(encoding="utf-8")
    cibw = pyproject.get("tool", {}).get("cibuildwheel", {})
    frontend = cibw.get("build-frontend")
    assert frontend == {"name": "pip", "args": ["--no-build-isolation"]} or (
        isinstance(frontend, str) and "no-build-isolation" in frontend
    )
    assert "cp314-*" in str(cibw.get("build", ""))
    assert "cp31?t-*" in str(cibw.get("skip", ""))
    assert "ubuntu-24.04-arm" in wheel_workflow
    # Require the v4 cibuildwheel line rather than one exact patch, so a
    # dependency bump does not have to touch this test (the workflow pin is
    # maintained by dependabot, and the wheel-smoke job validates behaviour).
    cibw_pin = re.search(r"pypa/cibuildwheel@v(\d+)\.", wheel_workflow)
    assert cibw_pin is not None, "cibuildwheel action pin not found"
    assert int(cibw_pin.group(1)) >= 4
    # PyPI must not get an sdist — unmatched CPython/arch would compile.
    assert "pattern: cibw-wheels-*" in wheel_workflow
    cmake_args = " ".join(pyproject["tool"]["scikit-build"]["cmake"]["args"])
    assert "CONDA_PREFIX" not in cmake_args
    assert "-DUSE_CUDA=OFF" in cmake_args
    assert (REPO_ROOT / "constraints-wheel.txt").is_file()
    assert f"torch{lane_spec}" in (REPO_ROOT / "constraints-wheel.txt").read_text(
        encoding="utf-8"
    )

    cmake = (PACKAGE_ROOT / "cpp_src" / "CMakeLists.txt").read_text(encoding="utf-8")
    bindings = (PACKAGE_ROOT / "cpp_src" / "bindings.cpp").read_text(encoding="utf-8")
    assert "TORCHFITS_BUILD_TORCH_VERSION" in cmake
    assert 'TORCHFITS_TORCH_ABI="${TORCHFITS_TORCH_ABI}"' in cmake
    assert "matching_abi" in bindings
    assert "pip should not compile" in cmake
    assert (REPO_ROOT / "scripts" / "cibw_before_build.sh").is_file()
    assert (REPO_ROOT / "scripts" / "cibw_test.sh").is_file()
    assert (REPO_ROOT / "scripts" / "cibuildwheel.sh").is_file()

    # [dev] covers test + bench + examples deps (no ipykernel).
    dev = set(pyproject["project"]["optional-dependencies"]["dev"])
    assert "ipykernel" not in dev
    assert any(x.startswith("pytest") for x in dev)
    assert any(x.startswith("astropy") for x in dev)
    assert any(x.startswith("matplotlib") for x in dev)


def test_every_tracked_file_is_re_addable() -> None:
    """No tracked file may sit under a .gitignore rule that can never re-include it.

    `git check-ignore` skips tracked paths, so an ignore rule covering a tracked
    file is invisible in normal use: the file keeps showing up in `git status`
    as tracked, and the breakage only appears on the delete/restore cycle. Git
    cannot re-include a file whose *parent directory* is excluded, so a rule like
    `benchmarks/replays/` plus `!benchmarks/replays/<manifest>` leaves the
    negation unreachable -- `git add` of that manifest then fails outright. Here
    the manifest was tracked anyway, which is why the dead `!` line went unseen.
    `--no-index` is what makes the shadowing visible, and the assertion is over
    the whole tracked set so the next one is caught where it is introduced.
    """
    probe = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=False, cwd=REPO_ROOT
    )
    if probe.returncode != 0:
        pytest.skip(f"not a git checkout: {probe.stderr.strip()}")
    tracked = probe.stdout.splitlines()
    assert tracked, "git ls-files returned nothing; the check would be vacuous"
    shadowed = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(tracked) + "\n",
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert not shadowed.stdout.strip(), (
        "these tracked files are shadowed by an unreachable .gitignore rule, so "
        "git add would refuse them after a delete/restore cycle:\n"
        f"{shadowed.stdout}"
    )


def test_sdist_readme_states_the_abi_rule_and_names_no_loose_torch_floor() -> None:
    """The packager-facing build note must not repeat the loose floor.

    SDIST-README.txt ships inside the sdist tarball, which is exactly the
    audience a wrong version claim reaches: downstream packagers reading it
    have no other statement of the PyTorch requirement. It used to say
    "pre-installed PyTorch (>= 2.10, ABI-matched)", a floor that admits every
    future minor while the extension in fact accepts exactly one -- the same
    claim that made `build-system.requires` dangerous. The note now points at
    `build-system.requires` as the authority instead of restating a version, so
    it cannot drift when the lane moves; this test keeps it that way.
    """
    readme = (REPO_ROOT / "SDIST-README.txt").read_text(encoding="utf-8")
    assert "build-system.requires" in readme, (
        "SDIST-README.txt must point at pyproject's build-system.requires as the "
        "authoritative torch pin rather than restating a version that can drift"
    )
    assert "--no-build-isolation" in readme, (
        "a packager with a pre-installed torch has no way to know they must "
        "bypass the isolated build env for the ABI stamp to match"
    )
    # No torch version claim of any shape: a named floor is what started this.
    assert not re.search(r"[Pp]y[Tt]orch\s*[\(<>=]*\s*\d", readme), (
        "SDIST-README.txt must not state a torch version; point at the pin instead"
    )
    for stale in (">= 2.10", ">=2.10", r"2\.10"):
        assert stale not in readme, f"SDIST-README.txt still mentions {stale!r}"


def test_native_extension_rejects_mismatched_torch_runtime() -> None:
    import torch

    expected_abi = ".".join(torch.__version__.split(".")[:2])
    script = """
import torch
torch.__version__ = "9.99.0"
import torchfits._C
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert (
        f"built for PyTorch {expected_abi}.x but found PyTorch 9.99.0" in result.stderr
    )


def test_build_env_torch_pin_cannot_resolve_outside_the_lane() -> None:
    """The isolated build env must not be able to pick another torch minor.

    The defect this guards is a *resolution* failure, not an import failure, so
    it is measured with pip's own resolver rather than by reading the pin: with
    the loose `torch>=2.10` floor, `pip install .` under build isolation
    resolved 2.14.0 (newest on PyPI) for the build env while the runtime pin
    admitted only 2.13.x, and the resulting extension refused to import with
    "built for PyTorch 2.14.x but found PyTorch 2.13.0".  A test that only
    asserted the string in pyproject would have passed straight through that.
    """
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    lane_spec = _wheel_lane_spec()
    build_requires = pyproject["build-system"]["requires"]
    # Every torch requirement in the build list must be the lane pin verbatim:
    # a floor, a range, or a second torch entry all let the resolver leave the
    # lane (a range only if it admits another minor).
    torch_reqs = [r for r in build_requires if _requirement_name(r) == "torch"]
    assert torch_reqs == [f"torch{lane_spec}"]

    # And the resolver agrees: nothing outside the lane satisfies it, against
    # the real index. `--ignore-installed` is load-bearing -- the test env
    # already has the lane torch, so without it pip reports an empty plan and
    # every assertion below passes vacuously. With it, `torch>=2.10` resolves
    # 2.14.0 and `torch>=2.13,<2.14` resolves 2.13.0 on the same index.
    resolved = _resolve_on_index(f"torch{lane_spec}")
    if resolved is None:
        pytest.skip("index unreachable")
    assert resolved, f"pip resolved nothing at all for torch{lane_spec}"
    lane_minor = lane_spec.split(">=")[1].split(",")[0]
    for version in resolved:
        major, minor = (int(x) for x in version.split(".")[:2])
        assert f"{major}.{minor}" == lane_minor, (
            f"build requirement torch{lane_spec} resolved to {version}, "
            "which the extension's ABI check would reject at import"
        )


def test_native_extension_rechecks_abi_after_metadata_import() -> None:
    """A metadata-first process still checks ABI before tensor entry points."""
    import torch

    expected_abi = ".".join(torch.__version__.split(".")[:2])
    script = """
import torchfits._C
import torch
torch.__version__ = "9.99.0"
import torchfits
torchfits.read
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert (
        f"built for PyTorch {expected_abi}.x but found PyTorch 9.99.0" in result.stderr
    )


def test_metadata_works_with_neither_torch_nor_numpy(tmp_path) -> None:
    """The torch-free core must not need numpy either.

    The clean-install case is the one that matters: a machine that only wants to
    look at a header has no reason to have torch, and torchfits declares numpy
    as a hard dependency for the tensor/Arrow paths, not for metadata. Reading
    a header, a shape and a column list in a process where importing either
    raises is the strongest available statement that the split is real.

    The fixture is written by this test's parent process (which does have the
    full stack) and only read by the child.
    """
    import numpy as np

    import torchfits

    image = tmp_path / "image.fits"
    torchfits.write(
        str(image), np.arange(64, dtype=np.float32).reshape(8, 8), overwrite=True
    )
    table = tmp_path / "table.fits"
    torchfits.table.write(
        str(table),
        {"ID": np.arange(3, dtype=np.int32), "MAG": np.linspace(18.0, 20.0, 3)},
        overwrite=True,
        extname="SCI",
    )

    script = """
import importlib.abc, json, sys


class _Block(importlib.abc.MetaPathFinder):
    BLOCKED = ("torch", "numpy")

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.BLOCKED:
            raise ImportError(root + " is blocked: " + fullname)
        return None


sys.meta_path.insert(0, _Block())

import torchfits
import torchfits._core as core

image, table = sys.argv[1], sys.argv[2]
print(json.dumps({
    "core_library_build_id": core.core_library_build_id(),
    "module_build_id": core.__build_id__,
    "num_hdus": torchfits.read_num_hdus(table),
    "hdu_type": torchfits.read_hdu_type(table, 1),
    "nrows": torchfits.read_nrows(table, 1),
    "colnames": torchfits.read_colnames(table, 1),
    "shape": torchfits.read_shape(image, 0),
    "keys": torchfits.read_keys(table, ["EXTNAME", "NAXIS2"], hdu=1),
    "header_size": len(torchfits.read_header(table, 1)),
    "table_info": sorted(torchfits.read_table_info(table, 1)),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(image), str(table)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["module_build_id"] == payload["core_library_build_id"]
    assert payload["num_hdus"] == 2
    assert payload["hdu_type"] == "BINARY_TABLE"
    assert payload["nrows"] == 3
    assert payload["colnames"] == ["ID", "MAG"]
    assert payload["shape"][0] == -32
    assert list(payload["shape"][1]) == [8, 8]
    assert payload["keys"] == {"EXTNAME": "SCI", "NAXIS2": 3}
    assert payload["header_size"] > 0
    assert payload["table_info"] == ["colnames", "nrows", "tforms"]


def test_torchfits_source_does_not_reference_torchsky() -> None:
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*"):
        if path.suffix not in {".py", ".cpp", ".h"} and path.name != "CMakeLists.txt":
            continue
        if "torchsky" in path.read_text(encoding="utf-8", errors="ignore").lower():
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert offenders == []


def test_torchfits_contains_only_fits_native_sources() -> None:
    native_root = PACKAGE_ROOT / "cpp_src"
    # Positive half first. "Contains only FITS native sources" is a
    # containment claim, so the container itself has to be asserted: with the
    # four absence checks alone the test is satisfied by an empty package, and
    # deleting or renaming cpp_src/ wholesale left it green. The positive
    # anchor is the house style here -- test_check_duplicate_cpp.py carries
    # `assert files, "the real cpp_src tree must not be empty"` for the
    # same tree.
    assert native_root.is_dir(), f"native source root is missing: {native_root}"
    sources = sorted(
        p.name for p in native_root.rglob("*") if p.suffix in {".cpp", ".h"}
    )
    assert sources, f"cpp_src must not be empty: {native_root}"
    assert "fits_file.cpp" in sources, (
        "the FITS native sources are what this package is allowed to contain; "
        f"found {sources}"
    )
    assert not (native_root / "wcs.cpp").exists()
    assert not (native_root / "healpix.cpp").exists()
    assert not (PACKAGE_ROOT / "wcs").exists()
    assert not (PACKAGE_ROOT / "sphere").exists()


def test_torchfits_python_sources_never_import_astropy_or_fitsio() -> None:
    """Runtime I/O must use vendored CFITSIO + _C, not Python astropy/fitsio."""
    forbidden = (
        "import astropy",
        "from astropy",
        "import fitsio",
        "from fitsio",
    )
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in forbidden:
                if pattern in stripped:
                    offenders.append(f"{path.relative_to(PACKAGE_ROOT)}: {stripped}")
    assert not offenders, "\n".join(offenders)


def test_root_import_stays_runtime_light() -> None:
    script = """
import sys
import torchfits
for name in ('torch', 'numpy', 'pyarrow', 'torchfits._C'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_import_sets_kmp_duplicate_lib_ok() -> None:
    if sys.platform != "darwin":
        pytest.skip("KMP_DUPLICATE_LIB_OK is only set by __init__ on Darwin")
    script = """
import os
os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
import torchfits
assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE", os.environ.get("KMP_DUPLICATE_LIB_OK")
"""
    env = {**os.environ}
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_duplicate_libomp_survives_after_import() -> None:
    """Homebrew libomp + PyTorch libomp abort with OMP Error #15 unless the
    import guard ran first. Skip when Homebrew's dylib is not installed."""
    if sys.platform != "darwin":
        pytest.skip("Homebrew libomp is a macOS-only scenario")
    brew_omp = next(
        (
            path
            for path in (
                Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
                Path("/usr/local/opt/libomp/lib/libomp.dylib"),
            )
            if path.is_file()
        ),
        None,
    )
    if brew_omp is None:
        pytest.skip("Homebrew libomp dylib not installed")
    script = f"""
import ctypes, os
os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
import torchfits
assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"
lib = ctypes.CDLL({str(brew_omp)!r}, mode=ctypes.RTLD_GLOBAL)
lib.omp_get_max_threads.restype = ctypes.c_int
lib.omp_get_max_threads()
import torch
print(torch.__version__, flush=True)
"""
    env = {**os.environ}
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_removed_native_cache_environment_is_ignored() -> None:
    """The removed TORCHFITS_CFITSIO_CACHE_* knobs must not break imports."""
    env = {
        **os.environ,
        "TORCHFITS_CFITSIO_CACHE_MB": "256",
        "TORCHFITS_CFITSIO_CACHE_FILES": "32",
    }
    result = subprocess.run(
        [sys.executable, "-c", "import torchfits; torchfits.read"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
