#!/usr/bin/env python
"""Smoke runner for all example scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Examples that must succeed in the default dev environment.
REQUIRED = [
    "example_image.py",
    "example_image_cutouts.py",
    "example_image_cube.py",
    "example_image_mef.py",
    "example_image_dataset.py",
    "example_table.py",
    "example_table_interop.py",
    "example_table_recipes.py",
]

# Optional-deps examples: pass if they exit 0 or print a known skip message.
OPTIONAL = [
    "example_polars.py",
]


def _python_cmd() -> list[str]:
    if os.environ.get("PIXI_ENVIRONMENT_NAME"):
        return [sys.executable]
    if shutil.which("pixi"):
        return ["pixi", "run", "python"]
    return [sys.executable]


def _example_path(name: str) -> str:
    base_dir = "examples" if os.path.isdir("examples") else SCRIPT_DIR
    path = os.path.join(base_dir, name)
    if not os.path.exists(path):
        path = os.path.join(SCRIPT_DIR, name)
    return path


def _run_example(name: str) -> tuple[bool, str]:
    path = _example_path(name)
    if not os.path.exists(path):
        return False, f"file not found: {path}"

    result = subprocess.run(
        [*_python_cmd(), path],
        cwd=".",
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode == 0:
        return True, ""

    output = (result.stderr or "") + (result.stdout or "")
    skip_markers = ("not installed", "skipping")
    if any(marker in output.lower() for marker in skip_markers):
        return True, "skipped (optional dependency missing)"
    return False, output[:1500]


def main() -> int:
    print(f"Running examples from: {os.getcwd()}")
    success = True

    for name in REQUIRED + OPTIONAL:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        ok, detail = _run_example(name)
        if ok:
            label = "PASS"
            if detail:
                label = f"PASS ({detail})"
            print(label)
        else:
            print("FAIL")
            print(detail)
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
