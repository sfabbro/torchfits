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
    release_lane.py --lane 2.13 --check   verify files match a rendered lane
    release_lane.py --lane 2.13 --apply   rewrite the files for that lane
    release_lane.py --lane 2.13 --prerelease rc5 --apply
                                   rewrite the files for a release candidate
                                   (e.g. torchfits 1.0.0rc5 on lane 2.13)

``--check`` is wired into preflight/CI so a lane pin can never drift from the
lane map. Exit code is non-zero on any mismatch. Pre-release states (e.g.
``1.0.0rc5``) are recognized by ``--check`` as the lane's base version plus a
PEP 440 prerelease suffix; ``--apply`` without ``--prerelease`` always renders
the plain map version (finalize).
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
_PIXI_PYARROW_RE = re.compile(
    r'^(pyarrow = ")(?:>=25\.0\.0,<26|>=24,<25)(")$', re.MULTILINE
)
_CONSTRAINTS_RE = re.compile(r"^(torch>=2\.\d+,<2\.\d+)$", re.MULTILINE)

# conda-forge pins strict libabseil minors; torch 2.10's flatbuffers build
# needs abseil 20260107.1 while every pyarrow 25.0.0 build needs abseil
# 20260526+, so pyarrow must fall back to 24.x on the 2.10 lane (pyarrow 24
# ships cp310-cp314 builds that accept the older abseil).
_LANE_PIXI_PYARROW: dict[str, str] = {
    "2.10": ">=24,<25",
}
_RECIPE_VERSION_RE = re.compile(r'^(  version: ")([^"]+)(")$', re.MULTILINE)
_RECIPE_TORCH_PIN_RE = re.compile(
    r'^(  torch_pin: ")(>=2\.\d+,<2\.\d+)(")$', re.MULTILINE
)

# PEP 440 prerelease suffixes the lane tool accepts on top of a lane version:
# 1.0.0rc5 / 1.0.0b1 / 1.0.0a2, optionally with a .postN segment. Dev local
# segments (.dev0+torch213) are stripped first so backport-candidate builds
# resolve back to their lane base too.
_PRERELEASE_SUFFIX_RE = re.compile(
    r"(?:rc\d+|a\d+|b\d+|alpha\d+|beta\d+)(?:\.post\d+)?$"
)
_DEV_LOCAL_RE = re.compile(r"\.dev0\+torch\d+$")


def strip_prerelease(version: str) -> str:
    """Return the lane base version of *version* (drop prerelease/dev-local)."""
    return _PRERELEASE_SUFFIX_RE.sub("", _DEV_LOCAL_RE.sub("", version))


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
    base = strip_prerelease(version)
    for lane, spec in lanes.items():
        if spec["torchfits_version"] == base:
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


def render(
    lane: str, version: str | None, prerelease: str | None = None
) -> dict[Path, str]:
    lanes = load_lanes()
    if lane in lanes:
        spec = lanes[lane]
        expected_version = str(spec["torchfits_version"])
        if prerelease is not None:
            # Release candidate: base version + PEP 440 prerelease suffix.
            if version is not None and strip_prerelease(version) != expected_version:
                raise SystemExit(
                    f"lane {lane} releases as torchfits {expected_version}, "
                    f"not {version!r} (torch_lanes.json)"
                )
            version = f"{expected_version}{prerelease}"
        elif version is None:
            # Plain apply/finalize: always the map version.
            version = expected_version
        elif strip_prerelease(version) != expected_version:
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
        if prerelease is not None:
            raise SystemExit(
                f"lane {lane!r} is not a release lane in torch_lanes.json; "
                "--prerelease only applies to release lanes"
            )
        # Read the release base from pyproject; after --apply the version has a
        # dev local segment (e.g. 1.0.0.dev0+torch212), so strip it first.
        match = _PYPROJECT_VERSION_RE.search(PYPROJECT.read_text(encoding="utf-8"))
        if match is None:
            raise SystemExit("pyproject.toml has no version field")
        base = strip_prerelease(match.group(2))
        base_lane = lane_for_version(base, load_lanes())
        expected_dev_version = f"{base}.dev0+torch{lane.replace('.', '')}"
        if version is not None and version != expected_dev_version:
            raise SystemExit(
                f"lane {lane!r} is not a release lane in torch_lanes.json; "
                f"expected version {expected_dev_version!r}, got {version!r}"
            )
        version = expected_dev_version
        cu_default = str(lanes[base_lane]["cu_default"])
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
    pyarrow_pin = _LANE_PIXI_PYARROW.get(lane, ">=25.0.0,<26")
    pixi = _replace_once(
        pixi,
        _PIXI_PYARROW_RE,
        rf"\g<1>{pyarrow_pin}\g<2>",
        "pixi pyarrow pin",
    )
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
    return _verify(lane, expected)


def _verify(lane: str, expected: dict[Path, str]) -> int:
    failed = False
    for path, want in expected.items():
        got = path.read_text(encoding="utf-8")
        if got == want:
            print(f"[ OK ] {path.name}", flush=True)
        else:
            print(f"[FAIL] {path.name} drifted from the {lane} lane", flush=True)
            failed = True
    return 1 if failed else 0


def committed_version() -> str | None:
    match = _PYPROJECT_VERSION_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    return match.group(2) if match is not None else None


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
        "--prerelease",
        help="PEP 440 prerelease suffix for the lane version (e.g. rc5 -> 1.0.0rc5)",
    )
    parser.add_argument(
        "--print-pins",
        action="store_true",
        help="print lane/version/torch pins for the committed state (CI use)",
    )
    args = parser.parse_args()

    if args.prerelease is not None and not re.fullmatch(
        r"(?:rc|a|b|alpha|beta)\d+", args.prerelease
    ):
        parser.error(
            f"--prerelease {args.prerelease!r} is not a PEP 440 prerelease suffix "
            "(e.g. rc5, b1)"
        )

    if args.check:
        if args.lane is not None:
            # Validate against the committed version (may carry a prerelease
            # suffix), so a checked-out rc state verifies as the lane's base.
            try:
                expected = render(args.lane, committed_version())
            except SystemExit as exc:
                print(f"[FAIL] {exc}", flush=True)
                return 1
            return _verify(args.lane, expected)
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

    rendered = render(args.lane, None, prerelease=args.prerelease)
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
