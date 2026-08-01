#!/usr/bin/env python3
"""Verify the torch build-flavor extras in pyproject.toml still resolve.

The ``[cpu]`` / ``[cuda]`` extras pin exact PyTorch builds from the
download.pytorch.org indexes (PEP 440 local versions such as
``torch==2.10.0+cpu``). Those pins must never drift off the published-wheel
ABI lane: a pin whose public version falls outside ``constraints-wheel.txt``
(or that no longer carries a ``+local`` segment) would install a wheel that
fails the extension's ABI check at import.

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
must appear in the docs/install.md / README.md one-liners, and the documented
wheel-lane pin must match ``constraints-wheel.txt``.

Exit code is non-zero when any pin fails. Run via ``pixi run check-torch-pins``
(also wired into the CI lint job and scripts/ci_local.sh).
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]

INDEX_PREFIX = "https://download.pytorch.org/whl/"


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


def documented_claims() -> tuple[set[str], list[str]]:
    """Index URLs and lane pins that the install docs / README one-liners use."""
    indexes: set[str] = set()
    lane_specs: list[str] = []
    for path in DOC_FILES:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        indexes.update(_DOC_INDEX_RE.findall(text))
        lane_specs.extend(_DOC_LANE_RE.findall(text))
    return indexes, lane_specs


def check_doc_drift(
    lane: SpecifierSet, pins: list[tuple[str, str, Version]]
) -> list[str]:
    """Failures when documented one-liners drift from the extras / wheel lane."""
    failures: list[str] = []
    doc_indexes, lane_specs = documented_claims()
    doc_locals = {url.rsplit("/", 1)[-1] for url in doc_indexes}
    for extra, _entry, version in pins:
        local = "".join(version.local)
        if local not in doc_locals:
            failures.append(
                f"[FAIL] {extra}: install docs never document "
                f"--extra-index-url https://download.pytorch.org/whl/{local}"
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
        if req.marker is not None and not req.marker.evaluate():
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
