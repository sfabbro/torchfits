#!/usr/bin/env python3
"""Fail unless a built wheel carries everything the package promises.

Wheels are otherwise only built when a release tag is pushed, which is the
worst moment to discover a packaging regression: a missing ``py.typed``, a
dropped stub, an absent third-party licence, or a metadata version that
disagrees with ``pyproject.toml``. This script is the check that runs on a
plain PR build instead. It is stdlib-only on purpose, so it works in a bare
interpreter with no project dependencies installed.

Usage::

    python scripts/check_wheel_contents.py dist/torchfits-1.1.3-*.whl
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Third-party licence text that must travel with the wheel (vendored CFITSIO).
REQUIRED_LICENCE_SUFFIXES = ("LICENSE", "CFITSIO-LICENSE.txt")

# Paths that must never ship: C++ sources, test suites, build scratch.
FORBIDDEN_SUBSTRINGS = (
    "cpp_src",
    "/tests/",
    "tests/",
    "__pycache__",
    ".pyc",
    "/build/",
)

# Symbols the stub must declare, so a truncated or empty stub fails here.
STUB_REQUIRED_SYMBOLS = ("class FITSFile", "class TableReader", "def read_full(")


def _fail(problems: list[str]) -> int:
    for problem in problems:
        print(f"[FAIL] {problem}", flush=True)
    return 1


def check(wheel: Path) -> int:
    if not wheel.is_file():
        return _fail([f"no such wheel: {wheel}"])

    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        metadata_name = next(
            (n for n in names if n.endswith(".dist-info/METADATA")), None
        )
        if metadata_name is None:
            return _fail([f"{wheel.name}: no .dist-info/METADATA"])
        metadata = zf.read(metadata_name).decode("utf-8")
        stub_name = next((n for n in names if n.endswith("torchfits/_C.pyi")), None)
        stub = zf.read(stub_name).decode("utf-8") if stub_name else ""

    problems: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    # --- compiled extension -------------------------------------------------
    shared_objects = [n for n in names if n.endswith(".so") or n.endswith(".pyd")]
    require(
        len(shared_objects) == 1 and shared_objects[0].startswith("torchfits/"),
        f"expected exactly one native extension under torchfits/, got {shared_objects}",
    )

    # --- typing support (PEP 561 + the generated native stub) ---------------
    require(
        "torchfits/py.typed" in names,
        "torchfits/py.typed is missing: the wheel would be untyped for users",
    )
    require(
        stub_name is not None,
        "torchfits/_C.pyi is missing: mypy would treat the native API as Any",
    )
    for symbol in STUB_REQUIRED_SYMBOLS:
        require(symbol in stub, f"torchfits/_C.pyi does not declare {symbol!r}")

    # --- licences -----------------------------------------------------------
    licence_entries = [n for n in names if "/licenses/" in n]
    for suffix in REQUIRED_LICENCE_SUFFIXES:
        require(
            any(n.endswith(suffix) for n in licence_entries),
            f"no {suffix} under dist-info/licenses/ (found {licence_entries})",
        )

    # --- metadata -----------------------------------------------------------
    version = _metadata_field(metadata, "Version")
    require(version is not None, "METADATA has no Version field")
    pyproject_version = _pyproject_version()
    require(
        version == pyproject_version,
        f"METADATA Version {version!r} != pyproject.toml {pyproject_version!r} "
        "(run scripts/release_lane.py --check)",
    )
    require(
        "License-Expression: MIT" in metadata,
        "METADATA lacks the PEP 639 License-Expression field",
    )
    require(
        "License :: OSI Approved" not in metadata,
        "METADATA still carries a deprecated license classifier (PEP 639 removed it)",
    )

    # --- nothing that should not ship --------------------------------------
    for pattern in FORBIDDEN_SUBSTRINGS:
        leaked = [n for n in names if pattern in n]
        require(not leaked, f"wheel contains {pattern!r}: {leaked[:5]}")

    if problems:
        return _fail(problems)
    print(
        f"[ OK ] {wheel.name}: {len(names)} entries, extension + py.typed + stub + "
        "both licences, metadata consistent",
        flush=True,
    )
    return 0


def _metadata_field(metadata: str, field: str) -> str | None:
    match = re.search(rf"^{re.escape(field)}: (.+)$", metadata, re.MULTILINE)
    return match.group(1).strip() if match else None


def _pyproject_version() -> str | None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    return max(check(wheel) for wheel in args.wheels)


if __name__ == "__main__":
    raise SystemExit(main())
