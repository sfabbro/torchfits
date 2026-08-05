#!/usr/bin/env python3
"""Verify the torch build-flavor extras in pyproject.toml still resolve.

The ``[cpu]`` / ``[cuda]`` extras pin exact PyTorch builds from the
download.pytorch.org indexes (PEP 440 local versions such as
``torch==2.10.0+cpu``). Those pins must never drift off the published-wheel
ABI lane: a pin whose public version falls outside ``constraints-wheel.txt``
(or that no longer carries a ``+local`` segment) would install a wheel that
fails the extension's ABI check at import.

The wheel lane itself is defined by ``scripts/torch_lanes.json`` (the release
train: one torchfits version per supported torch minor) and rendered into
``constraints-wheel.txt`` / ``pyproject.toml`` / ``pixi.toml`` by
``scripts/release_lane.py``. This script also verifies that rendered state has
not drifted from the lane map (version -> lane, lane -> ``[cpu]`` /
``[cuda]`` pins, constraints specifier).

For every torch pin in every extra, this script:

1. checks the public version satisfies the wheel-lane constraint from
   ``constraints-wheel.txt``,
2. checks it carries a ``+local`` version segment (an exact ``==`` pin is
   required so the build flavor resolves deterministically),
3. resolves it with ``pip install --dry-run`` against the matching PyTorch
   index (``+cpu`` -> https://download.pytorch.org/whl/cpu, ``+cuXXX`` ->
   https://download.pytorch.org/whl/cuXXX) and fails if pip cannot.

Pins whose marker is not active on the current platform (the ``cuda`` extra
is Linux-only) are skipped, matching the extras' own semantics; if every pin
is skipped, the check passes vacuously (e.g. macOS, which has no ``+cpu`` /
``+cu128`` wheels).

This script also guards against install-docs drift: every extra's index URL
and exact pin string must appear in the docs/install.md / README.md
one-liners, and the documented wheel-lane pin must match
``constraints-wheel.txt``.

Exit code is non-zero when any pin fails. Run via ``pixi run check-torch-pins``
(also wired into the CI lint job and scripts/ci_local.sh).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]

INDEX_PREFIX = "https://download.pytorch.org/whl/"


def load_lane_map() -> dict[str, dict[str, object]]:
    """The release-train lane map (torch minor -> torchfits version + flavors)."""
    path = ROOT / "scripts" / "torch_lanes.json"
    return json.loads(path.read_text(encoding="utf-8"))


def lane_for_version(version: str) -> str:
    lanes = load_lane_map()
    for lane, spec in lanes.items():
        if spec["torchfits_version"] == version:
            return lane
    raise SystemExit(
        f"pyproject version {version!r} is not a torchfits release-lane version "
        f"(torch_lanes.json: {', '.join(lanes)})"
    )


def check_lane_map_consistency(lane: SpecifierSet) -> list[str]:
    """Failures when the rendered lane state drifted from torch_lanes.json."""
    failures: list[str] = []
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    data = tomllib.loads(pyproject_text)
    version = str(data["project"]["version"])
    try:
        map_lane = lane_for_version(version)
    except SystemExit as exc:
        failures.append(f"[FAIL] {exc}")
        return failures
    spec = load_lane_map()[map_lane]
    major, minor = (int(p) for p in map_lane.split("."))
    expected_spec = SpecifierSet(f">={map_lane},<{major}.{minor + 1}")
    if lane != expected_spec:
        failures.append(
            f"[FAIL] constraints-wheel.txt pins torch{lane} but the lane map "
            f"({version}) requires {expected_spec}"
        )
    cpu_expected = f"torch=={map_lane}.0+cpu; sys_platform == 'linux'"
    cuda_expected = f"torch=={map_lane}.0+{spec['cu_default']}; sys_platform == 'linux'"
    for extra, entry in _iter_torch_entries(pyproject_text):
        expected = cpu_expected if extra == "cpu" else cuda_expected
        if entry != expected:
            failures.append(
                f"[FAIL] {extra} extra pins {entry!r} but the lane map "
                f"({version}, torch {map_lane}) requires {expected!r}"
            )
    return failures


def load_wheel_lane() -> SpecifierSet:
    """The torch range the published wheels are ABI-matched to."""
    text = (ROOT / "constraints-wheel.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        req = Requirement(stripped)
        if req.name.lower() == "torch":
            return req.specifier
    raise SystemExit("constraints-wheel.txt has no torch constraint")


def index_for_local(local: str) -> str:
    """Map a ``+local`` segment to its download.pytorch.org index."""
    if local == "cpu":
        return INDEX_PREFIX + "cpu"
    if local.startswith("cu"):
        return INDEX_PREFIX + local
    raise SystemExit(
        f"unsupported torch local version segment {local!r} (expected 'cpu' or 'cuXXX')"
    )


DOC_FILES = (ROOT / "docs" / "install.md", ROOT / "README.md")
_DOC_INDEX_RE = re.compile(
    r"--extra-index-url\s+(https://download\.pytorch\.org/whl/[a-zA-Z0-9]+)"
)
_DOC_LANE_RE = re.compile(r"torch(>=2\.\d+,<2\.\d+)")
_DOC_PIN_RE = re.compile(r"torch==\d+\.\d+\.\d+\+[a-zA-Z0-9]+")


def documented_claims() -> tuple[set[str], list[str], set[str]]:
    """Index URLs, lane pins, and exact torch pins the docs / README use."""
    indexes: set[str] = set()
    lane_specs: list[str] = []
    exact_pins: set[str] = set()
    for path in DOC_FILES:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        indexes.update(_DOC_INDEX_RE.findall(text))
        lane_specs.extend(_DOC_LANE_RE.findall(text))
        exact_pins.update(_DOC_PIN_RE.findall(text))
    return indexes, lane_specs, exact_pins


def check_doc_drift(
    lane: SpecifierSet, pins: list[tuple[str, str, Version]]
) -> list[str]:
    """Failures when documented one-liners drift from the extras / wheel lane."""
    failures: list[str] = []
    doc_indexes, lane_specs, doc_pins = documented_claims()
    doc_locals = {url.rsplit("/", 1)[-1] for url in doc_indexes}
    for extra, entry, version in pins:
        local = "".join(version.local)
        if local not in doc_locals:
            failures.append(
                f"[FAIL] {extra}: install docs never document "
                f"--extra-index-url https://download.pytorch.org/whl/{local}"
            )
        bare_pin = entry.split(";", 1)[0].strip()
        if bare_pin not in doc_pins:
            failures.append(
                f"[FAIL] {extra}: install docs/README never show the exact pin "
                f"{bare_pin} (extras pin {entry})"
            )
    if not lane_specs:
        failures.append("[FAIL] install docs/README contain no torch wheel-lane pin")
    for spec in lane_specs:
        if SpecifierSet(spec) != lane:
            failures.append(
                f"[FAIL] docs pin torch{spec} but the wheel lane is {lane} "
                "(constraints-wheel.txt)"
            )
    return failures


def _format_full_version(info: Any) -> str:
    """Full CPython implementation version (mirrors packaging's private helper)."""
    version = f"{info.major}.{info.minor}.{info.micro}"
    kind = info.releaselevel
    if kind != "final":
        version += kind[0] + str(info.serial)
    return version


def _active_marker_environment() -> dict[str, str]:
    """Live marker context for the current process.

    packaging >= 26.3 computes its default marker environment once per
    process and caches it, so platform patches (e.g. ``sys.platform`` swaps
    in the unit tests) would be ignored. Evaluate markers against an
    explicit environment instead so behavior is version-independent.
    """
    import platform

    return {
        "implementation_name": sys.implementation.name,
        "implementation_version": _format_full_version(sys.implementation.version),
        "os_name": os.name,
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "python_full_version": platform.python_version(),
        "platform_python_implementation": platform.python_implementation(),
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "sys_platform": sys.platform,
    }


def _iter_torch_entries(pyproject_text: str) -> Iterator[tuple[str, str]]:
    """Yield (extra, entry) for every torch requirement in the extras."""
    data = tomllib.loads(pyproject_text)
    extras = data["project"].get("optional-dependencies", {})
    for extra, entries in extras.items():
        for entry in entries:
            req = Requirement(entry)
            if req.name.lower() == "torch":
                yield extra, entry


def iter_torch_pins(pyproject_text: str) -> Iterator[tuple[str, str, Version]]:
    """Yield (extra, entry, Version) for active torch pins in the extras."""
    for extra, entry in _iter_torch_entries(pyproject_text):
        req = Requirement(entry)
        if req.marker is not None and not req.marker.evaluate(
            environment=_active_marker_environment()
        ):
            print(
                f"[skip] {extra}: {entry} (marker not active on this platform)",
                flush=True,
            )
            continue
        match = re.fullmatch(r"==\s*(?P<version>[^;,\s]+)", str(req.specifier))
        if match is None:
            raise SystemExit(
                f"{extra}: {entry} — torch pins must be exact (==) so the "
                "build flavor resolves deterministically"
            )
        version = Version(match.group("version"))
        if not version.local:
            raise SystemExit(
                f"{extra}: {entry} — torch pins must carry a +local segment "
                "(e.g. +cpu / +cu128) so they resolve from the PyTorch "
                "index, not PyPI"
            )
        yield extra, entry, version


def resolve(specifier: str, index: str) -> subprocess.CompletedProcess[str]:
    """Dry-run pip resolution of ``specifier`` against ``index`` + PyPI."""
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--quiet",
        "--disable-pip-version-check",
        specifier,
        "--extra-index-url",
        index,
    ]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd,
            returncode=1,
            stdout=exc.stdout or "",
            stderr="timed out resolving the pin",
        )


def main() -> int:
    lane = load_wheel_lane()
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    failed = False
    for line in check_lane_map_consistency(lane):
        print(line, flush=True)
        failed = True

    pins = list(iter_torch_pins(pyproject_text))
    if not pins:
        if not list(_iter_torch_entries(pyproject_text)):
            print("no torch flavor extras found in pyproject.toml", flush=True)
            return 1
        # Every torch flavor pin was marker-skipped (e.g. macOS: both extras
        # are Linux-only). Vacuous pass — the extras can never resolve there.
        print("all torch flavor pins skipped by platform markers — OK", flush=True)
        return 0

    failed = False
    for extra, entry, version in pins:
        label = f"{extra}: {entry}"
        if not lane.contains(version.public, prereleases=True):
            print(
                f"[FAIL] {label} public version {version.public} is outside "
                f"the wheel ABI lane {lane} (constraints-wheel.txt)",
                flush=True,
            )
            failed = True
            continue
        index = index_for_local("".join(version.local))
        result = resolve(entry, index)
        if result.returncode == 0:
            print(f"[ OK ] {label} resolves from {index}", flush=True)
        else:
            print(f"[FAIL] {label} failed to resolve from {index}", flush=True)
            print(result.stderr or result.stdout or "", flush=True)
            failed = True

    drift = check_doc_drift(lane, pins)
    if drift:
        for line in drift:
            print(line, flush=True)
        failed = True
    else:
        print(
            "[ OK ] install one-liners match the extra pins and the wheel lane",
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
