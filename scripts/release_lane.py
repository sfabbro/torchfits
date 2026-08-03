#!/usr/bin/env python3
"""Render or check the PyTorch release-lane pins across the repo.

torchfits wheels are ABI-matched to a single PyTorch minor — the "lane"
(PyTorch has no stable C++ ABI across minors). ``scripts/torch_lanes.json``
maps every supported torch minor to the torchfits version released for it and
the CUDA build flavors that lane ships.

This script renders or checks the lane pins in:

- ``pyproject.toml`` — project version, ``torch`` runtime dependency, and the
  ``[cpu]`` / ``[cuda]`` extra pins,
- ``constraints-wheel.txt`` — build-time torch constraint (cibuildwheel),
- ``pixi.toml`` — package version and the ``pytorch`` build/host/run/dev pins,
- ``packaging/conda/recipe.yaml`` — conda recipe version and pytorch pin,
- ``src/torchfits/__init__.py`` — ``__version__``.

The repo (main) always tracks the newest lane; older lanes are rendered from
tags with ``--apply`` when cutting a release.

Usage::

    release_lane.py                print the current lane (from committed files)
    release_lane.py --check        verify committed files match torch_lanes.json
    release_lane.py --lane 2.13    print the rendered state for a lane
    release_lane.py --lane 2.13 --apply   rewrite the files for that lane

``--check`` is wired into preflight/CI so a lane pin can never drift from the
lane map. Exit code is non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANES_FILE = ROOT / "scripts" / "torch_lanes.json"

PYPROJECT = ROOT / "pyproject.toml"
CONSTRAINTS = ROOT / "constraints-wheel.txt"
PIXI = ROOT / "pixi.toml"
RECIPE = ROOT / "packaging" / "conda" / "recipe.yaml"
INIT = ROOT / "src" / "torchfits" / "__init__.py"

_PYPROJECT_VERSION_RE = re.compile(r'^(version = ")([^"]+)(")$', re.MULTILINE)
_INIT_VERSION_RE = re.compile(r'^(__version__ = ")([^"]+)(")$', re.MULTILINE)
_PIXI_VERSION_RE = re.compile(r'^(version = ")([^"]+)(")$', re.MULTILINE)
_PIXI_PYTORCH_RE = re.compile(r'^(pytorch = ")(>=2\.\d+,<2\.\d+)(")$', re.MULTILINE)
_CONSTRAINTS_RE = re.compile(r"^(torch>=2\.\d+,<2\.\d+)$", re.MULTILINE)
_RECIPE_VERSION_RE = re.compile(r'^(  version: ")([^"]+)(")$', re.MULTILINE)
_RECIPE_TORCH_PIN_RE = re.compile(
    r'^(  torch_pin: ")(>=2\.\d+,<2\.\d+)(")$', re.MULTILINE
)

PYPROJECT_DEPS_BLOCK = """# Core runtime dependencies for the library.
# These are automatically managed by pixi for conda packages
# but needed here for PyPI compatibility
dependencies = [
    # Published wheels are ABI-matched to the {lane} lane (PyTorch has no
    # stable C++ ABI across minors). The lane pin keeps pip on the matching
    # minor; other torch minors ship as separate torchfits releases, one per
    # lane (scripts/torch_lanes.json, docs/install.md).
    "torch>={lane},<{next_lane}",
    "numpy>=1.20.0",  # Core numerical operations
    "pyarrow>=5.0",   # Table I/O (Arrow interchange)
]"""

_PYPROJECT_DEPS_RE = re.compile(
    r"^# Core runtime dependencies for the library\..*?^\]$",
    re.MULTILINE | re.DOTALL,
)

PYPROJECT_EXTRAS_HEADER = """# Torch build flavors — pick one. Extras cannot embed index URLs, so [cpu] /
# [cuda] resolve only when the matching download.pytorch.org index is
# reachable (PIP_EXTRA_INDEX_URL or pip.conf extra-index-url; see
# docs/install.md). Exact pins track the {lane} ABI lane (see
# scripts/torch_lanes.json).
"""

_PYPROJECT_EXTRAS_HEADER_RE = re.compile(
    r"^# Torch build flavors — pick one\..*?^\]$",
    re.MULTILINE | re.DOTALL,
)

PYPROJECT_CPU_BLOCK = """cpu = [
    # Thin CPU-only PyTorch: no CUDA runtime / cuDNN stack. Linux only —
    # macOS has no +cpu wheel; MPS ships inside the default macOS build.
    "torch=={lane}.0+cpu; sys_platform == 'linux'",
]"""

PYPROJECT_CUDA_BLOCK = """cuda = [
    # CUDA-enabled PyTorch ({cuda_default}). Linux only — macOS has no
    # +{cuda_default} wheels; MPS ships inside the default macOS torch build.
    "torch=={lane}.0+{cuda_default}; sys_platform == 'linux'",
]"""

_PYPROJECT_CUDA_RE = re.compile(r"^cuda = \[.*?^\]$", re.MULTILINE | re.DOTALL)


def load_lanes() -> dict[str, dict[str, object]]:
    data = json.loads(LANES_FILE.read_text(encoding="utf-8"))
    lanes: dict[str, dict[str, object]] = {}
    for lane, spec in data.items():
        if not re.fullmatch(r"\d+\.\d+", lane):
            raise SystemExit(f"torch_lanes.json: invalid lane {lane!r}")
        if not re.fullmatch(r"\d+\.\d+\.\d+", str(spec["torchfits_version"])):
            raise SystemExit(f"torch_lanes.json: bad torchfits_version for lane {lane}")
        lanes[lane] = spec
    return lanes


def next_lane(lane: str) -> str:
    major, minor = lane.split(".")
    return f"{major}.{int(minor) + 1}"


def lane_for_version(version: str, lanes: dict[str, dict[str, object]]) -> str:
    for lane, spec in lanes.items():
        if spec["torchfits_version"] == version:
            return lane
    raise SystemExit(
        f"version {version!r} is not a torchfits release-lane version "
        f"(torch_lanes.json: {', '.join(lanes)})"
    )


def current_lane() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = _PYPROJECT_VERSION_RE.search(text)
    if match is None:
        raise SystemExit("pyproject.toml has no version field")
    return lane_for_version(match.group(2), load_lanes())


def _replace_once(text: str, pattern: re.Pattern[str], new: str, what: str) -> str:
    updated, count = pattern.subn(new, text)
    if count != 1:
        raise SystemExit(f"expected exactly one match for {what}; found {count}")
    return updated


def render(lane: str, version: str | None) -> dict[Path, str]:
    lanes = load_lanes()
    if lane in lanes:
        spec = lanes[lane]
        expected_version = str(spec["torchfits_version"])
        if version is None:
            version = expected_version
        if version != expected_version:
            raise SystemExit(
                f"lane {lane} releases as torchfits {expected_version}, "
                f"not {version!r} (torch_lanes.json)"
            )
        cu_default = str(spec["cu_default"])
    else:
        # Not a release lane: render an experimental dev build. The version
        # derives from the current release with a +torch<minor> local segment
        # (PEP 440), so backport-candidate wheels never collide with the real
        # release version and verify_wheel_matrix.sh can recover the lane.
        if version is not None:
            raise SystemExit(
                f"lane {lane!r} is not a release lane in torch_lanes.json; "
                "its version derives from the current release"
            )
        base = str(lanes[current_lane()]["torchfits_version"])
        version = f"{base}.dev0+torch{lane.replace('.', '')}"
        cu_default = str(lanes[current_lane()]["cu_default"])
    nxt = next_lane(lane)

    rendered: dict[Path, str] = {}

    pyproject = PYPROJECT.read_text(encoding="utf-8")
    pyproject = _replace_once(
        pyproject, _PYPROJECT_VERSION_RE, rf"\g<1>{version}\g<3>", "pyproject version"
    )
    pyproject = _replace_once(
        pyproject,
        _PYPROJECT_DEPS_RE,
        PYPROJECT_DEPS_BLOCK.format(lane=lane, next_lane=nxt),
        "pyproject dependencies",
    )
    pyproject = _replace_once(
        pyproject,
        _PYPROJECT_EXTRAS_HEADER_RE,
        PYPROJECT_EXTRAS_HEADER.format(lane=lane)
        + PYPROJECT_CPU_BLOCK.format(lane=lane),
        "pyproject extras header + cpu block",
    )
    pyproject = _replace_once(
        pyproject,
        _PYPROJECT_CUDA_RE,
        PYPROJECT_CUDA_BLOCK.format(lane=lane, cuda_default=cu_default),
        "pyproject cuda extra",
    )
    rendered[PYPROJECT] = pyproject

    constraints = CONSTRAINTS.read_text(encoding="utf-8")
    constraints = _replace_once(
        constraints,
        _CONSTRAINTS_RE,
        f"torch>={lane},<{nxt}",
        "constraints-wheel.txt torch pin",
    )
    rendered[CONSTRAINTS] = constraints

    pixi = PIXI.read_text(encoding="utf-8")
    pixi = _replace_once(
        pixi, _PIXI_VERSION_RE, rf"\g<1>{version}\g<3>", "pixi version"
    )
    updated, count = _PIXI_PYTORCH_RE.subn(rf"\g<1>>={lane},<{nxt}\g<3>", pixi)
    if count == 0:
        raise SystemExit("no pixi pytorch pins found to render")
    pixi = updated
    rendered[PIXI] = pixi

    recipe = RECIPE.read_text(encoding="utf-8")
    recipe = _replace_once(
        recipe, _RECIPE_VERSION_RE, rf"\g<1>{version}\g<3>", "recipe version"
    )
    recipe = _replace_once(
        recipe, _RECIPE_TORCH_PIN_RE, rf"\g<1>>={lane},<{nxt}\g<3>", "recipe torch_pin"
    )
    rendered[RECIPE] = recipe

    init = INIT.read_text(encoding="utf-8")
    init = _replace_once(
        init, _INIT_VERSION_RE, rf"\g<1>{version}\g<3>", "__init__ version"
    )
    rendered[INIT] = init

    return rendered


def check() -> int:
    failed = False
    version_text = PYPROJECT.read_text(encoding="utf-8")
    match = _PYPROJECT_VERSION_RE.search(version_text)
    if match is None:
        print("[FAIL] pyproject.toml has no version field", flush=True)
        return 1
    version = match.group(2)
    try:
        lane = lane_for_version(version, load_lanes())
    except SystemExit as exc:
        print(f"[FAIL] {exc}", flush=True)
        return 1
    print(
        f"[INFO] committed state is torch lane {lane} (torchfits {version})", flush=True
    )
    try:
        expected = render(lane, version)
    except SystemExit as exc:
        print(f"[FAIL] {exc}", flush=True)
        return 1
    for path, want in expected.items():
        got = path.read_text(encoding="utf-8")
        if got == want:
            print(f"[ OK ] {path.name}", flush=True)
        else:
            print(f"[FAIL] {path.name} drifted from the {lane} lane", flush=True)
            failed = True
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", help="torch minor lane to render (e.g. 2.13)")
    parser.add_argument(
        "--apply", action="store_true", help="rewrite the pinned files in place"
    )
    parser.add_argument(
        "--check", action="store_true", help="verify committed files match the lane map"
    )
    parser.add_argument(
        "--print-pins",
        action="store_true",
        help="print lane/version/torch pins for the committed state (CI use)",
    )
    args = parser.parse_args()

    if args.check:
        return check()

    lanes = load_lanes()
    if args.print_pins:
        lane = current_lane()
        version = str(lanes[lane]["torchfits_version"])
        print(f"lane={lane} version={version} torch=>={lane},<{next_lane(lane)}")
        return 0

    if args.lane is None:
        lane = current_lane()
        print(f"current lane: {lane}", flush=True)
        return 0

    rendered = render(args.lane, None)
    version_match = _PYPROJECT_VERSION_RE.search(rendered[PYPROJECT])
    assert version_match is not None
    version = version_match.group(2)
    if not args.apply:
        for path, text in rendered.items():
            print(f"--- {path.relative_to(ROOT)} ---")
            print(text)
        return 0

    for path, text in rendered.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}", flush=True)
    print(
        f"repo now tracks torch lane {args.lane} (torchfits {version}). "
        "Re-run `pixi install` to sync the dev environment.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
