from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "torchfits"
REPO_ROOT = PACKAGE_ROOT.parents[1]


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

    # Pip metadata allows torch>=2.10 (source builds against the installed minor).
    assert "torch>=2.10" in pyproject["build-system"]["requires"]
    # Published wheels carry the ABI lane pin (one torchfits release per torch
    # minor, see scripts/torch_lanes.json).
    assert f"torch{lane_spec}" in pyproject["project"]["dependencies"]
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
    assert "pypa/cibuildwheel@v4.1.1" in wheel_workflow
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
