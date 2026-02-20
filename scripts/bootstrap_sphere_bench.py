#!/usr/bin/env python3
"""Install optional sphere benchmark comparison libraries from released packages."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

CORE_SPECS: list[tuple[str, str]] = [
    ("astropy-healpix", "astropy-healpix"),
    ("hpgeom", "hpgeom"),
    ("healpix", "healpix"),
    ("mhealpy", "mhealpy"),
    ("healsparse", "healsparse"),
    ("skyproj", "skyproj"),
    ("sphgeom", "lsst-sphgeom"),
]

HARMONIC_SPECS: list[tuple[str, str]] = [
    ("torch-harmonics", "torch-harmonics"),
    ("s2fft", "s2fft"),
    ("s2wav", "s2wav"),
    ("so3", "so3"),
    ("coord", "LSSTDESC.Coord"),
]


def _parse_list(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {x.strip().lower() for x in raw.split(",") if x.strip()}
    return values or None


def _install_with_pip(package_spec: str, no_build_isolation: bool) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "pip", "install"]
    if no_build_isolation:
        cmd.append("--no-build-isolation")
    cmd.append(package_spec)
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def _install_with_uv(package_spec: str) -> subprocess.CompletedProcess[str]:
    uv = shutil.which("uv")
    if uv is None:
        return subprocess.CompletedProcess(
            args=["uv", "pip", "install", package_spec],
            returncode=127,
            stdout="",
            stderr="uv executable not found in PATH",
        )
    cmd = [uv, "pip", "install", package_spec]
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--installer",
        choices=["pip", "uv"],
        default="pip",
        help="Package installer backend",
    )
    parser.add_argument("--with-harmonics", action="store_true", help="Also install harmonic/encoder ecosystem packages")
    parser.add_argument(
        "--include",
        type=str,
        default=None,
        help="Comma-separated package aliases to include (default: all core specs)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated package aliases to skip",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any install fails")
    parser.add_argument(
        "--build-isolation",
        action="store_true",
        help="Use pip build isolation (only applies when --installer=pip)",
    )
    args = parser.parse_args()

    include = _parse_list(args.include)
    exclude = _parse_list(args.exclude) or set()
    specs = list(CORE_SPECS)
    if args.with_harmonics:
        specs.extend(HARMONIC_SPECS)

    installed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    for alias, package_spec in specs:
        name = alias.lower()
        if include is not None and name not in include:
            skipped.append(f"{alias} (not included)")
            continue
        if name in exclude:
            skipped.append(f"{alias} (excluded)")
            continue

        print(f"[install] {alias}: {package_spec}")
        if args.installer == "pip":
            proc = _install_with_pip(package_spec, no_build_isolation=not args.build_isolation)
        else:
            proc = _install_with_uv(package_spec)
        if proc.returncode == 0:
            installed.append(alias)
            continue

        failed.append(alias)
        print(f"[failed] {alias} exit={proc.returncode}")
        if proc.stdout:
            print(proc.stdout.strip().splitlines()[-1])
        if proc.stderr:
            print(proc.stderr.strip().splitlines()[-1])

    print("\nSummary")
    print(f"  installed: {', '.join(installed) if installed else 'none'}")
    print(f"  failed:    {', '.join(failed) if failed else 'none'}")
    print(f"  skipped:   {len(skipped)}")
    if skipped:
        for item in skipped:
            print(f"    - {item}")

    if failed and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
