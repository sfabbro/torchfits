#!/usr/bin/env python
"""Smoke runner for all example scripts."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Examples explicitly excluded from auto-discovery (e.g., they require
# external data not available in CI, or are run via a different path).
_EXCLUDE = {
    "desi_shaped_spectrum.py",  # requires DESI data download
    "cli/make_rgb_demo.py",  # auxiliary script, not a standalone example
    "test_examples.py",  # this file
}

# Optional-deps examples: pass if they exit 0 or print a known skip message.
OPTIONAL = {
    "example_polars.py",
}


def _discover_examples() -> list[str]:
    """Discover all example scripts via glob, excluding test-runner and aux files."""
    patterns = [
        os.path.join(SCRIPT_DIR, "*.py"),
        os.path.join(SCRIPT_DIR, "cli", "*.py"),
    ]
    discovered: list[str] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            name = (
                os.path.basename(path)
                if "cli" not in pattern
                else os.path.join("cli", os.path.basename(path))
            )
            base = os.path.basename(name)
            if base.startswith("_") or name in _EXCLUDE:
                continue
            discovered.append(name)
    return discovered


def _example_path(name: str) -> str:
    base_dir = "examples" if os.path.isdir("examples") else SCRIPT_DIR
    path = os.path.join(base_dir, name)
    if not os.path.exists(path):
        path = os.path.join(SCRIPT_DIR, name)
    return path


# Per-example timeout overrides (seconds); default is 180.
TIMEOUTS = {
    "example_table_recipes.py": 120,
    # Downloads up to GZ_N individual cutouts over HTTP on a cold cache.
    "example_ml_galaxyzoo_legacy.py": 300,
}


def _python_cmd() -> list[str]:
    if os.environ.get("PIXI_ENVIRONMENT_NAME"):
        return [sys.executable]
    if shutil.which("pixi"):
        return ["pixi", "run", "python"]
    return [sys.executable]


def _run_example(name: str) -> tuple[bool, str]:
    path = _example_path(name)
    if not os.path.exists(path):
        return False, f"file not found: {path}"

    timeout = TIMEOUTS.get(name, 180)
    env = os.environ.copy()
    if os.environ.get("GITHUB_ACTIONS"):
        env["TORCHFITS_EXAMPLE_FAST"] = "1"
    # The denoise example trains a U-Net + evaluates full CCDs; the smoke
    # path (1 epoch, 1 CCD, 1024x1024 probe) runs it in well under a minute.
    if name == "example_megacam_cr_denoise.py":
        env["TORCHFITS_EXAMPLE_FAST"] = "1"
    # Keep the Galaxy Zoo smoke path bounded: full DEFAULT_GZ_N=200 cutout
    # downloads routinely exceed the runner timeout on a cold cache.
    if name == "example_ml_galaxyzoo_legacy.py":
        env.setdefault("GZ_N", "8")
    try:
        result = subprocess.run(
            [*_python_cmd(), path],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"timeout after {timeout}s: {exc}"
    if result.returncode == 0:
        return True, ""

    output = (result.stderr or "") + (result.stdout or "")
    # Only OPTIONAL examples are allowed to skip on missing deps.
    if name in OPTIONAL:
        skip_markers = ("not installed", "skipping")
        if any(marker in output.lower() for marker in skip_markers):
            return True, "skipped (optional dependency missing)"
    return False, output[:1500]


def main() -> int:
    print(f"Running examples from: {os.getcwd()}")
    success = True

    required = [n for n in _discover_examples() if n not in OPTIONAL]
    optional = [n for n in _discover_examples() if n in OPTIONAL]
    all_examples = required + optional

    print(
        f"Discovered {len(all_examples)} examples ({len(required)} required, {len(optional)} optional)\n"
    )

    for name in all_examples:
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
