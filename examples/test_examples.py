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
    "cli/make_rgb_demo.py",  # auxiliary script, not a standalone example
    "test_examples.py",  # this file
}

# Optional examples: pass if they exit 0, print a known skip message, or time
# out. They still run — the point is that they must not red the gate for a reason
# outside this repository's control.
OPTIONAL = {
    "example_polars.py",
    # Fetches a catalog and one FITS cutout per row from a third-party HTTP
    # service (legacysurvey.org). Its availability depends on that service and on
    # the network, not on anything here, so a network problem must not fail CI.
    "example_ml_galaxyzoo_legacy.py",
}

# Markers an optional example prints when it declines to run. The examples use a
# "SKIP: ..." convention; the older "not installed"/"skipping" pair matched none
# of the seven examples that print one, so an optional example skipping that way
# was still reported as a failure.
SKIP_MARKERS = (
    "skip:",
    "not installed",
    "skipping",
)


def _discover_examples() -> list[str]:
    """Discover all example scripts via glob, excluding test-runner and aux files."""
    # (subdir, pattern) pairs: whether a hit is a cli/ script is decided by the
    # subdirectory it was globbed from, never by the absolute pattern text --
    # a checkout under e.g. ~/clinical/ or ~/client/ used to reclassify every
    # top-level example as cli/<name>.py, and the whole gate then failed with
    # "file not found".
    globs = [
        ("", os.path.join(SCRIPT_DIR, "*.py")),
        ("cli", os.path.join(SCRIPT_DIR, "cli", "*.py")),
    ]
    discovered: list[str] = []
    for subdir, pattern in globs:
        for path in sorted(glob.glob(pattern)):
            base = os.path.basename(path)
            name = os.path.join(subdir, base) if subdir else base
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


def _assertions_enabled() -> bool:
    """False when this process was started with ``-O`` / ``PYTHONOPTIMIZE``."""
    return __debug__


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
    # PYTHONOPTIMIZE (or `python -O` on this runner) erases every `assert` in the
    # examples, and the identity-stress and cookbook examples are built almost
    # entirely out of them: they would print "All ... checks passed" having
    # checked nothing, and the gate would call that a PASS. Assertions are the
    # point here, so the children are started with them on.
    env.pop("PYTHONOPTIMIZE", None)
    if not _assertions_enabled():
        return False, "runner started with assertions disabled (-O)"
    if os.environ.get("GITHUB_ACTIONS"):
        env["TORCHFITS_EXAMPLE_FAST"] = "1"
    # The denoise example trains a U-Net + evaluates CCDs; the smoke path
    # (1 epoch, 1 CCD, bounded 1024x1024 eval + probe windows) runs in
    # well under a minute even on single-threaded torch.
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
        # A timeout is an environment problem rather than a defect in the
        # example, which is what "optional" is meant to absorb. A required
        # example still fails here: one that cannot finish inside its budget is
        # a real signal about the repository.
        if name in OPTIONAL:
            return True, f"skipped (timed out after {timeout}s, optional)"
        return False, f"timeout after {timeout}s: {exc}"
    if result.returncode == 0:
        return True, ""

    output = (result.stderr or "") + (result.stdout or "")
    # Only OPTIONAL examples may decline to run.
    if name in OPTIONAL and any(m in output.lower() for m in SKIP_MARKERS):
        return True, "skipped (optional)"
    return False, output[:1500]


def main() -> int:
    print(f"Running examples from: {os.getcwd()}")
    success = True

    if not _assertions_enabled():
        print(
            "refusing to run: this interpreter has assertions disabled (-O / "
            "PYTHONOPTIMIZE), and the identity-stress and cookbook examples are "
            "built out of `assert`. Re-run without -O.",
            file=sys.stderr,
        )
        return 1

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
