"""Extract the release-notes body for one version from docs/changelog.md.

Used by .github/workflows/build_wheels.yml (github_release job) so the
GitHub Release page carries the curated changelog section instead of only
auto-generated PR lists. Falls back to the ``## Unreleased`` section when
the requested version has not been stamped yet (prerelease cuts).

Usage:
    python scripts/release_notes.py --version 1.1.0 [--out FILE]

Prints the section body (without the heading) to stdout or writes it to
--out. Exits non-zero when neither section exists.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "docs" / "changelog.md"


def _section(text: str, heading_re: str) -> str | None:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.fullmatch(heading_re, line.strip()):
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    body = "\n".join(lines[start:end]).strip()
    return body or None


def extract(text: str, version: str) -> tuple[str, str]:
    """Return (source_label, body) for *version* ('Unreleased' fallback)."""
    escaped = re.escape(version)
    stamped = _section(text, rf"## \[{escaped}\](\s*—.*)?")
    if stamped:
        return f"[{version}]", stamped
    unreleased = _section(text, r"## Unreleased")
    if unreleased:
        return "Unreleased", unreleased
    raise SystemExit(
        f"release notes: no '## [{version}]' or '## Unreleased' section in {CHANGELOG}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="e.g. 1.1.0")
    parser.add_argument("--out", default=None, help="write body to FILE")
    args = parser.parse_args(argv)

    label, body = extract(CHANGELOG.read_text(encoding="utf-8"), args.version)
    header = (
        f"<!-- sourced from docs/changelog.md section [{args.version}] "
        f"(found: {label}; prerelease fallback) -->\n"
        if label == "Unreleased"
        else f"<!-- sourced from docs/changelog.md section [{args.version}] -->\n"
    )
    payload = header + body + "\n"
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
