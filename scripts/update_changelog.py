#!/usr/bin/env python3
"""Merge conventional commits into docs/changelog.md and stamp releases.

The changelog keeps a single ``## Unreleased`` section on top while work is
in flight; ``docs/release.md`` stamps it with the real version + date when a
release is cut. Edge docs therefore always say "Unreleased" until the
maintainer names the version.

Usage::

    update_changelog.py                       merge new commits into Unreleased
    update_changelog.py --check               exit 1 if the merge would change the file
    update_changelog.py --since v1.1.0        merge commits from an explicit base tag
    update_changelog.py --release 1.1.0 [--date YYYY-MM-DD]
                                              stamp Unreleased -> [1.1.0] - date,
                                              refresh the link refs, open a fresh
                                              Unreleased section

Commit mapping (conventional prefixes):

    feat -> Added          fix -> Fixed          perf -> Performance
    security -> Security   revert -> Changed     docs -> Docs
    deps -> Dependencies   refactor -> Changed (--all)

``chore`` / ``ci`` / ``test`` / ``build`` / ``style`` subjects are skipped.

Exit code is non-zero on any failure or, with ``--check``, on drift.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "docs" / "changelog.md"
REPO_URL = "https://github.com/astroai/torchfits"

# Canonical Keep-a-Changelog order plus the project's extra sections.
SECTION_ORDER = [
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
    "Performance",
    "Dependencies",
    "Packaging",
    "Docs",
]

TYPE_MAP = {
    "feat": "Added",
    "fix": "Fixed",
    "perf": "Performance",
    "security": "Security",
    "revert": "Changed",
    "refactor": "Changed",
    "docs": "Docs",
    "deps": "Dependencies",
}
ALWAYS_SKIP = {"chore", "ci", "test", "build", "style"}

SUBJECT_RE = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?: (?P<body>.+)$")


def parse_subject(subject: str, include_all: bool = False) -> tuple[str, str] | None:
    """Map a conventional-commit subject to (section, bullet text)."""
    match = SUBJECT_RE.match(subject.strip())
    if match is None:
        return None
    ctype = match["type"]
    if ctype in ALWAYS_SKIP or (ctype == "refactor" and not include_all):
        return None
    section = TYPE_MAP.get(ctype)
    if section is None:
        return None
    scope = (match["scope"] or "").strip()
    body = match["body"].strip()
    prefix = (
        f"{scope}: " if scope and not body.lower().startswith(scope.lower()) else ""
    )
    bullet = prefix + body[:1].upper() + body[1:]
    return section, "- " + bullet


def _normalize(bullet: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", bullet.lower()).strip()


# Audit-style tracking IDs embedded in commit subjects / curated bullets
# (e.g. "(B1)", "(H4, H5)", "M16"). A generated bullet whose IDs are all
# already cited somewhere in Unreleased is considered merged even when its
# wording differs from the curated prose — otherwise every curation pass
# re-imports a terse duplicate.
_ISSUE_ID_RE = re.compile(r"(?<![A-Za-z])(?:B|H|M)\d{1,2}(?!\d)")


def _issue_ids(text: str) -> set[str]:
    return set(_ISSUE_ID_RE.findall(text))


def _unreleased_span(lines: list[str]) -> tuple[int, int]:
    """Return [start, end) line indexes of the Unreleased section."""
    start = None
    for i, line in enumerate(lines):
        if re.fullmatch(r"## Unreleased\s*", line):
            start = i
            break
    if start is None:
        raise SystemExit(
            "docs/changelog.md has no '## Unreleased' section; run "
            "--release to stamp the current one or re-add the heading."
        )
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return start, end


def merge_generated(text: str, bullets: list[tuple[str, str]]) -> str:
    """Insert generated bullets into their Unreleased subsections (deduped)."""
    lines = text.splitlines()
    start, end = _unreleased_span(lines)
    block = lines[start:end]

    covered_ids: set[str] = set()
    for line in block:
        if line.startswith("- ") or line.startswith("  "):
            covered_ids |= _issue_ids(line)

    pending: dict[str, list[str]] = {}
    for section, bullet in bullets:
        pending.setdefault(section, []).append(bullet)

    def _seen(section: str) -> set[str]:
        keys: set[str] = set()
        in_section = False
        for line in block:
            if line.startswith("### "):
                in_section = line[4:].strip() == section
                continue
            if in_section and line.startswith("- "):
                keys.add(_normalize(line))
        return keys

    for section in pending:
        seen = _seen(section)

        def _merged(bullet: str) -> bool:
            if _normalize(bullet) in seen:
                return True
            ids = _issue_ids(bullet)
            return bool(ids) and ids <= covered_ids

        pending[section] = [b for b in pending[section] if not _merged(b)]

    # Work on content without trailing blank lines so appends land tight.
    trail = len(block)
    while trail > 0 and not block[trail - 1].strip():
        trail -= 1
    body, tail = block[:trail], block[trail:]

    out: list[str] = []
    current: str | None = None
    for line in body:
        if line.startswith("### "):
            if current and pending.get(current):
                out.extend(pending.pop(current))
            current = line[4:].strip()
        out.append(line)
    if current and pending.get(current):
        out.extend(pending.pop(current))

    leftover = [(s, b) for s in SECTION_ORDER for b in pending.get(s, [])]
    if leftover:
        if out and out[-1].strip():
            out.append("")
        for section in SECTION_ORDER:
            if not pending.get(section):
                continue
            out.append(f"### {section}")
            out.extend(pending[section])
            del pending[section]

    lines[start:end] = out + tail
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def latest_tag(*, include_prereleases: bool = True) -> str | None:
    tags = subprocess.run(
        ["git", "tag", "--list", "v*", "--sort=-version:refname"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if tags.returncode != 0:
        return None
    for tag in tags.stdout.split():
        if include_prereleases or not re.search(r"[a-zA-Z]", tag[1:]):
            return tag
    return None


def previous_stable_version(text: str) -> str | None:
    """Newest released X.Y.Z named by the changelog's own link refs."""
    versions = re.findall(r"^\[(\d+\.\d+\.\d+)\]:", text, flags=re.M)
    if not versions:
        return None
    keys = [tuple(int(p) for p in v.split(".")) for v in versions]
    return versions[keys.index(max(keys))]


def _refresh_refs(text: str, version: str, prev: str) -> str:
    """Point [Unreleased] at HEAD past the new release and add its compare link."""
    new_refs = (
        f"[Unreleased]: {REPO_URL}/compare/v{version}...HEAD\n"
        f"[{version}]: {REPO_URL}/compare/v{prev}...v{version}"
    )
    unreleased_ref = re.compile(r"^\[Unreleased\]: .*$", flags=re.M)
    match = unreleased_ref.search(text)
    if match is not None:
        return text[: match.start()] + new_refs + text[match.end() :]
    first_ref = re.search(r"^\[[^\]]+\]: ", text, flags=re.M)
    if first_ref is None:
        return text.rstrip("\n") + "\n\n" + new_refs + "\n"
    idx = first_ref.start()
    return text[:idx] + new_refs + "\n" + text[idx:]


def stamp_release(text: str, version: str, date: str) -> str:
    """Rename Unreleased to [version] - date, refresh refs, reopen Unreleased."""
    prev = previous_stable_version(text) or "0.1.0"
    lines = text.splitlines()
    start, _ = _unreleased_span(lines)
    lines[start] = f"## [{version}] — {date}"
    lines.insert(start, "")
    lines.insert(start, "## Unreleased")
    stamped = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    stamped = re.sub(rf"^\[{re.escape(version)}\]: .*\n?", "", stamped, flags=re.M)
    stamped = stamped.replace("\n\n\n", "\n\n")
    return _refresh_refs(stamped, version, prev)


def collect_bullets(since: str, *, include_all: bool) -> list[tuple[str, str]]:
    log = subprocess.run(
        ["git", "log", "--pretty=%s", f"{since}..HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    bullets: list[tuple[str, str]] = []
    for subject in log.stdout.splitlines():
        parsed = parse_subject(subject, include_all=include_all)
        if parsed is not None:
            bullets.append(parsed)
    return bullets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update docs/changelog.md")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check", action="store_true", help="fail if generation would change the file"
    )
    group.add_argument(
        "--release", metavar="VERSION", help="stamp the Unreleased section as VERSION"
    )
    parser.add_argument("--date", help="release date (YYYY-MM-DD; default today)")
    parser.add_argument(
        "--since", help="base git tag for generation (default: newest tag)"
    )
    parser.add_argument("--all", action="store_true", help="also map refactor commits")
    args = parser.parse_args(argv)

    original = CHANGELOG.read_text(encoding="utf-8")

    if args.release:
        date = args.date or datetime.date.today().isoformat()
        updated = stamp_release(original, args.release, date)
    else:
        since = args.since or latest_tag()
        if since is None:
            print("no git tags found; nothing to generate")
            return 0
        try:
            bullets = collect_bullets(since, include_all=args.all)
        except subprocess.CalledProcessError as exc:
            print(f"git log failed for range {since}..HEAD: {exc}")
            return 1
        if not bullets:
            print(f"no new classifiable commits since {since}")
            return 0
        updated = merge_generated(original, bullets)

    if args.check:
        if updated != original:
            print("docs/changelog.md is stale; run `pixi run changelog-update`")
            return 1
        print("changelog up to date")
        return 0

    if updated != original:
        CHANGELOG.write_text(updated, encoding="utf-8")
        print(f"updated {CHANGELOG.relative_to(ROOT)}")
    else:
        print("changelog already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
