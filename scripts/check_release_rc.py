#!/usr/bin/env python3
"""Require a release-candidate tag before a stable release tag.

A stable version (``1.3.1``) must be preceded by at least one ``v1.3.1rcN``
tag: the rc is what gets installed and validated, and the stable cut changes
only the version strings. This runs before PyPI upload in
``.github/workflows/build_wheels.yml`` and from the release runbook.

Usage::

    python scripts/check_release_rc.py --version 1.3.1
    python scripts/check_release_rc.py --version 1.3.1rc1   # no-op

Exit codes: 0 when the version is a prerelease or a matching rc tag exists,
1 when a stable version has no rc tag, 2 on a malformed version.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

_VERSION_RE = re.compile(r"^(?P<base>\d+\.\d+\.\d+)(?P<pre>rc\d+)?$")
_RC_TAG_RE = re.compile(r"^v(?P<base>\d+\.\d+\.\d+)rc(?P<number>\d+)$")
ALLOWED_STABLE_CHANGES = frozenset(
    {
        "docs/changelog.md",
        "packaging/conda/recipe.yaml",
        "pixi.toml",
        "pyproject.toml",
        "src/torchfits/__init__.py",
    }
)
VERSION_FILES = ALLOWED_STABLE_CHANGES - {"docs/changelog.md"}


def split_version(version: str) -> tuple[str, str | None]:
    """Return ``(base, prerelease)`` for a PEP 440 release version."""
    match = _VERSION_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"not a PEP 440 release version: {version!r}")
    return match.group("base"), match.group("pre")


def rc_tags(version: str, tags: list[str]) -> list[str]:
    """Return the ``v<version>rcN`` tags for a *stable* version, sorted."""
    base, prerelease = split_version(version)
    if prerelease is not None:
        return []
    found: list[str] = []
    for tag in tags:
        match = _RC_TAG_RE.fullmatch(tag)
        if match is not None and match.group("base") == base:
            found.append(tag)
    return sorted(found, key=lambda tag: int(_RC_TAG_RE.fullmatch(tag).group("number")))


def git_tags() -> list[str]:
    """Return every ``v*`` tag in the repository (full history required)."""
    result = subprocess.run(
        ["git", "tag", "--list", "v*"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_text(revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def validate_source_tree(rc_tag: str, target: str = "HEAD") -> list[str]:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", rc_tag, target],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError(f"{rc_tag} is not an ancestor of {target}")
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{rc_tag}..{target}"],
        check=True,
        capture_output=True,
        text=True,
    )
    changed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    forbidden = sorted(set(changed) - ALLOWED_STABLE_CHANGES)
    if forbidden:
        raise ValueError(f"source changed after {rc_tag}: {', '.join(forbidden)}")
    match = _RC_TAG_RE.fullmatch(rc_tag)
    if match is None:
        raise ValueError(f"not an rc tag: {rc_tag}")
    rc_version = rc_tag.removeprefix("v")
    stable_version = match.group("base")
    for path in sorted(set(changed) & VERSION_FILES):
        before = _git_text(rc_tag, path).replace(rc_version, "<VERSION>")
        after = _git_text(target, path).replace(stable_version, "<VERSION>")
        if before != after:
            raise ValueError(f"non-version content changed after {rc_tag}: {path}")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="tag version without the 'v' prefix (e.g. 1.3.1)",
    )
    parser.add_argument("--print-tag", action="store_true")
    args = parser.parse_args(argv)

    version = args.version.strip().removeprefix("v")
    try:
        base, prerelease = split_version(version)
    except ValueError as exc:
        print(f"release-rc gate: {exc}", file=sys.stderr)
        return 2

    if prerelease is not None:
        print(f"release-rc gate: {version} is a prerelease; no rc tag required")
        return 0

    found = rc_tags(version, git_tags())
    if found:
        selected = found[-1]
        try:
            changed = validate_source_tree(selected)
        except (subprocess.CalledProcessError, ValueError) as exc:
            print(f"release-rc gate: {exc}", file=sys.stderr)
            return 1
        if args.print_tag:
            print(selected)
        else:
            detail = ", ".join(changed) if changed else "no files"
            print(
                f"release-rc gate: {version} uses {selected}; "
                f"stable-only changes: {detail}"
            )
        return 0

    print(
        f"release-rc gate: no v{base}rc* tag found. Publish v{base}rc1, "
        f"validate that artifact, then tag v{base}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
