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

_VERSION_RE = re.compile(
    r"^(?P<base>\d+\.\d+\.\d+)(?P<pre>(?:rc|a|b|alpha|beta|dev)\d*)?$"
)
_RC_TAG_RE = re.compile(r"^v(?P<base>\d+\.\d+\.\d+)rc\d+$")


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
    return sorted(found)


def git_tags() -> list[str]:
    """Return every ``v*`` tag in the repository (full history required)."""
    result = subprocess.run(
        ["git", "tag", "--list", "v*"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="tag version without the 'v' prefix (e.g. 1.3.1)",
    )
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
        print(f"release-rc gate: {version} has {', '.join(found)}")
        return 0

    print(
        f"release-rc gate: no v{base}rc* tag found. Publish v{base}rc1, "
        f"validate that artifact, then tag v{base}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
