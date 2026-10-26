#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import tarfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

MAX_ARCHIVE_SIZE = 512 * 1024 * 1024
MAX_MEMBERS = 20000
MAX_MEMBER_SIZE = 256 * 1024 * 1024
MAX_TOTAL_SIZE = 2 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 250
MAX_METADATA_SIZE = 1024 * 1024
REQUIRED = {
    "CMakeLists.txt",
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "SDIST-README.txt",
    "extern/VERSIONS.txt",
    "extern/cfitsio/CMakeLists.txt",
    "extern/licenses/CFITSIO-LICENSE.txt",
    "pyproject.toml",
    "src/torchfits/__init__.py",
}
FORBIDDEN_ROOTS = {
    ".agents",
    ".cursor",
    ".git",
    ".github",
    ".pixi",
}
FORBIDDEN_PARTS = {
    "__pycache__",
    "build",
    "dist",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".dll", ".pyd"}
NAME_RE = re.compile(r"^torchfits-(?P<version>[A-Za-z0-9.!+_-]+)\.tar\.gz$")


def _safe_path(name: str) -> PurePosixPath | None:
    if not name or "\\" in name:
        return None
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path


def _link_target(member: tarfile.TarInfo, root: str) -> PurePosixPath | None:
    target = _safe_path(member.linkname)
    if target is None:
        return None
    member_path = PurePosixPath(member.name)
    resolved = target if member.islnk() else member_path.parent / target
    parts: list[str] = []
    for part in resolved.parts:
        if part == "..":
            if not parts:
                return None
            parts.pop()
        elif part != ".":
            parts.append(part)
    result = PurePosixPath(*parts)
    if not result.parts or result.parts[0] != root:
        return None
    return result


def validate(sdist: Path, version: str) -> list[str]:
    problems: list[str] = []
    match = NAME_RE.fullmatch(sdist.name)
    if match is None or match.group("version") != version:
        problems.append(f"{sdist.name}: filename is not torchfits-{version}.tar.gz")
    if not sdist.is_file():
        return problems + [f"{sdist}: no such sdist"]
    archive_size = sdist.stat().st_size
    if archive_size > MAX_ARCHIVE_SIZE:
        problems.append(f"{sdist.name}: archive size exceeds {MAX_ARCHIVE_SIZE}")
        return problems
    try:
        with tarfile.open(sdist, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) > MAX_MEMBERS:
                problems.append(
                    f"{sdist.name}: {len(members)} members exceeds {MAX_MEMBERS}"
                )
                return problems
            names: set[str] = set()
            regular_names: set[str] = set()
            roots: set[str] = set()
            link_targets: list[tuple[str, PurePosixPath]] = []
            total = 0
            resource_exceeded = False
            expected_root = f"torchfits-{version}"
            for member in members:
                path = _safe_path(member.name)
                if path is None:
                    problems.append(f"{sdist.name}: unsafe member {member.name!r}")
                    continue
                roots.add(path.parts[0])
                normalized = str(path)
                if normalized in names:
                    problems.append(f"{sdist.name}: duplicate member {normalized}")
                names.add(normalized)
                if member.isfile():
                    regular_names.add(normalized)
                if not (
                    member.isfile()
                    or member.isdir()
                    or member.issym()
                    or member.islnk()
                ):
                    problems.append(
                        f"{sdist.name}: unsupported member type {normalized}"
                    )
                if member.size < 0 or member.size > MAX_MEMBER_SIZE:
                    problems.append(f"{sdist.name}: oversized member {normalized}")
                    resource_exceeded = True
                total += max(member.size, 0)
                relative = path.parts[1:]
                if (relative and relative[0] in FORBIDDEN_ROOTS) or any(
                    part in FORBIDDEN_PARTS for part in relative
                ):
                    problems.append(f"{sdist.name}: forbidden member {normalized}")
                if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                    problems.append(f"{sdist.name}: forbidden member {normalized}")
                if member.issym() or member.islnk():
                    target = _link_target(member, expected_root)
                    if target is None:
                        problems.append(f"{sdist.name}: unsafe link {normalized}")
                    else:
                        link_targets.append((normalized, target))
            if roots != {expected_root}:
                problems.append(
                    f"{sdist.name}: expected single root {expected_root}, got {sorted(roots)}"
                )
            if total > MAX_TOTAL_SIZE:
                problems.append(
                    f"{sdist.name}: uncompressed size exceeds {MAX_TOTAL_SIZE}"
                )
                resource_exceeded = True
            if archive_size and total > archive_size * MAX_COMPRESSION_RATIO:
                problems.append(f"{sdist.name}: compression ratio exceeds limit")
                resource_exceeded = True
            if resource_exceeded:
                return problems
            for name, target in link_targets:
                if str(target) not in names:
                    problems.append(f"{sdist.name}: link target missing for {name}")
            relative_names = {
                name.removeprefix(f"{expected_root}/")
                for name in regular_names
                if name.startswith(f"{expected_root}/")
            }
            missing = sorted(REQUIRED - relative_names)
            if missing:
                problems.append(f"{sdist.name}: missing required files {missing}")
            metadata_member = (
                archive.getmember(f"{expected_root}/PKG-INFO")
                if "PKG-INFO" not in missing
                else None
            )
            if metadata_member is not None:
                if metadata_member.size > MAX_METADATA_SIZE:
                    problems.append(f"{sdist.name}: PKG-INFO exceeds size limit")
                else:
                    extracted = archive.extractfile(metadata_member)
                    if extracted is None:
                        problems.append(f"{sdist.name}: PKG-INFO is not readable")
                    else:
                        metadata = Parser().parsestr(
                            extracted.read(MAX_METADATA_SIZE + 1).decode("utf-8")
                        )
                        if metadata.get("Name") != "torchfits":
                            problems.append(
                                f"{sdist.name}: PKG-INFO Name is not torchfits"
                            )
                        if metadata.get("Version") != version:
                            problems.append(
                                f"{sdist.name}: PKG-INFO Version is not {version}"
                            )
    except (KeyError, tarfile.TarError, UnicodeDecodeError) as exc:
        problems.append(f"{sdist.name}: invalid archive: {exc}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    problems = validate(args.sdist, args.version)
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1
    print(f"[ OK ] validated sdist for {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
