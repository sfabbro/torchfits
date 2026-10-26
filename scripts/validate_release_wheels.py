#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from email.parser import Parser
from pathlib import Path

from archive_safety import zip_safety_problems

EXPECTED_CELLS = {
    *((f"cp{minor}", "manylinux", "x86_64") for minor in range(311, 316)),
    *((f"cp{minor}", "manylinux", "aarch64") for minor in range(311, 316)),
    *((f"cp{minor}", "macosx_14_0", "arm64") for minor in range(311, 315)),
}
WHEEL_RE = re.compile(
    r"^torchfits-(?P<version>[^-]+)-(?P<python>cp\d+)-(?P<abi>cp\d+)-"
    r"(?P<platform>[^-]+)\.whl$"
)


def _field(message, name: str) -> str | None:
    value = message.get(name)
    return value.strip() if value is not None else None


def _torch_requirement(message) -> str | None:
    for requirement in message.get_all("Requires-Dist", []):
        compact = requirement.replace(" ", "")
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(.*)$", compact)
        if match is not None:
            name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
            if name == "torch" and ";" not in compact:
                return match.group(2)
    return None


def _cell(python: str, platform: str) -> tuple[str, str, str] | None:
    if platform == "macosx_14_0_arm64":
        return python, "macosx_14_0", "arm64"
    if "manylinux" not in platform:
        return None
    if platform.endswith("_x86_64"):
        arch = "x86_64"
    elif platform.endswith("_aarch64"):
        arch = "aarch64"
    else:
        return None
    platform_tags = set(platform.split("."))
    required = f"manylinux_2_28_{arch}"
    allowed = {required, f"manylinux_2_26_{arch}"}
    if required not in platform_tags or not platform_tags <= allowed:
        return None
    return python, "manylinux", arch


def validate(wheel_dir: Path, version: str) -> tuple[list[Path], list[str]]:
    wheels = sorted(wheel_dir.glob("*.whl"))
    problems: list[str] = []
    cells: list[tuple[str, str, str]] = []
    if len(wheels) != 14:
        problems.append(f"expected exactly 14 wheels, found {len(wheels)}")
    for wheel in wheels:
        match = WHEEL_RE.fullmatch(wheel.name)
        if match is None:
            problems.append(f"invalid wheel filename: {wheel.name}")
            continue
        python = match.group("python")
        abi = match.group("abi")
        platform = match.group("platform")
        if match.group("version") != version:
            problems.append(f"{wheel.name}: filename version is not {version}")
        if abi != python:
            problems.append(f"{wheel.name}: ABI tag {abi} does not match {python}")
        cell = _cell(python, platform)
        if cell is None:
            problems.append(f"{wheel.name}: unexpected platform tag {platform}")
        else:
            cells.append(cell)
        with zipfile.ZipFile(wheel) as archive:
            infos = archive.infolist()
            safety = zip_safety_problems(infos)
            if safety:
                problems.extend(f"{wheel.name}: {problem}" for problem in safety)
                continue
            metadata_names = [
                info.filename
                for info in infos
                if info.filename.endswith(".dist-info/METADATA")
            ]
            wheel_names = [
                info.filename
                for info in infos
                if info.filename.endswith(".dist-info/WHEEL")
            ]
            if len(metadata_names) != 1 or len(wheel_names) != 1:
                problems.append(f"{wheel.name}: expected one METADATA and one WHEEL")
                continue
            metadata = Parser().parsestr(
                archive.read(metadata_names[0]).decode("utf-8")
            )
            wheel_metadata = Parser().parsestr(
                archive.read(wheel_names[0]).decode("utf-8")
            )
        if _field(metadata, "Version") != version:
            problems.append(f"{wheel.name}: METADATA Version is not {version}")
        if _field(metadata, "Requires-Python") != ">=3.11":
            problems.append(f"{wheel.name}: Requires-Python is not >=3.11")
        torch_requirement = _torch_requirement(metadata)
        if torch_requirement is None or set(torch_requirement.split(",")) != {
            ">=2.14",
            "<2.15",
        }:
            problems.append(f"{wheel.name}: torch requirement is not >=2.14,<2.15")
        expected_tags = {
            f"{python}-{abi}-{platform_tag}" for platform_tag in platform.split(".")
        }
        wheel_tags = set(wheel_metadata.get_all("Tag", []))
        if wheel_tags != expected_tags:
            problems.append(
                f"{wheel.name}: WHEEL tags {sorted(wheel_tags)} != "
                f"filename tags {sorted(expected_tags)}"
            )
    found = set(cells)
    if found != EXPECTED_CELLS:
        missing = sorted(EXPECTED_CELLS - found)
        extra = sorted(found - EXPECTED_CELLS)
        problems.append(f"wheel matrix mismatch; missing={missing}, extra={extra}")
    if len(cells) != len(set(cells)):
        problems.append("wheel matrix contains duplicate cells")
    return wheels, problems


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(artifacts: list[Path], destination: Path) -> None:
    lines = []
    for artifact in artifacts:
        digest = _sha256(artifact)
        lines.append(f"{digest}  {artifact.stat().st_size}  {artifact.name}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_manifest(artifacts: list[Path], manifest: Path) -> list[str]:
    expected: dict[str, tuple[str, int]] = {}
    problems: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if (
            len(fields) != 3
            or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None
            or not fields[1].isdigit()
            or Path(fields[2]).name != fields[2]
            or fields[2] in expected
        ):
            problems.append(f"invalid manifest line: {line}")
            continue
        expected[fields[2]] = (fields[0], int(fields[1]))
    if set(expected) != {artifact.name for artifact in artifacts}:
        problems.append("manifest filenames do not match validated artifacts")
        return problems
    for artifact in artifacts:
        digest, size = expected[artifact.name]
        if artifact.stat().st_size != size:
            problems.append(f"{artifact.name}: size does not match manifest")
        if _sha256(artifact) != digest:
            problems.append(f"{artifact.name}: SHA-256 does not match manifest")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_dir", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--sdist", type=Path)
    args = parser.parse_args()
    wheels, problems = validate(args.wheel_dir, args.version)
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1
    artifacts = wheels + ([args.sdist] if args.sdist is not None else [])
    if args.manifest is not None:
        write_manifest(artifacts, args.manifest)
    if args.verify_manifest is not None:
        manifest_problems = verify_manifest(artifacts, args.verify_manifest)
        if manifest_problems:
            for problem in manifest_problems:
                print(f"[FAIL] {problem}")
            return 1
    print(f"[ OK ] exact 14-wheel release matrix for {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
