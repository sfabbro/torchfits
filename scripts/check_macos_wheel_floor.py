#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from archive_safety import zip_safety_problems


def check(wheel: Path, floor: str) -> list[str]:
    problems: list[str] = []
    expected_tag = f"macosx_{floor.replace('.', '_')}_arm64"
    if expected_tag not in wheel.name:
        problems.append(f"{wheel.name}: expected {expected_tag} filename tag")
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(wheel) as archive:
            infos = archive.infolist()
            safety = zip_safety_problems(infos)
            if safety:
                problems.extend(f"{wheel.name}: {problem}" for problem in safety)
                return problems
            native = [
                info.filename
                for info in infos
                if info.filename.startswith("torchfits/")
                and info.filename.endswith((".so", ".dylib"))
            ]
            if not native:
                problems.append(f"{wheel.name}: no native Mach-O artifacts")
            for name in native:
                path = Path(archive.extract(name, directory))
                result = subprocess.run(
                    ["otool", "-l", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                floors = re.findall(
                    r"^\s*minos\s+(\d+\.\d+)", result.stdout, re.MULTILINE
                )
                if not floors or set(floors) != {floor}:
                    problems.append(
                        f"{wheel.name}:{name}: minos={floors}, expected {floor}"
                    )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--floor", default="14.0")
    args = parser.parse_args()
    problems = [
        problem for wheel in args.wheels for problem in check(wheel, args.floor)
    ]
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1
    print(f"[ OK ] macOS wheel tags and Mach-O minimum are {args.floor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
