#!/usr/bin/env python3
"""Fan the canonical `.cursor/` agent tree out to the installed agent homes.

`.cursor/skills/` and `.cursor/hooks/` are tracked and canonical. `.agents/` and
`.codex/` are *installed* copies that `.gitignore` keeps out of version control,
on the same model as `~/.claude/`: convenient for the client, never the source of
truth. Nothing kept the copies equal to the originals except discipline, and a
stale copy silently serves the very bug the tracked file was fixed to remove --
the 41-of-45 API-freeze inventory bug and the state-losing Stop hook both lived
on under the canonical names until
`test_untracked_client_trees_do_not_shadow_tracked_copies` was written.

This is the installer `.gitignore` claims exists ("`.cursor/` is the canonical
tracked tree; the agent-home installer fans these out"). Run it after editing the
canonical tree:

    pixi run agent-home          # sync the copies
    pixi run agent-home-check    # exit 1 if any copy drifted (for CI)

`.codex/hooks.json` is deliberately never copied: it is per-machine (it holds
absolute paths) and its schema is the client's, not ours.
"""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (canonical, installed) directory pairs. Both sides are directory roots, and
# only regular files are mirrored -- never `__pycache__`, never a manifest.
MIRRORS = (
    (ROOT / ".cursor" / "skills", ROOT / ".agents" / "skills"),
    (ROOT / ".cursor" / "hooks", ROOT / ".codex" / "hooks"),
)

# One manifest per installed root records what we wrote, so a file deleted from
# the canonical tree can be removed from the copy without touching anything else
# a user may have installed there. It sits *beside* the mirrored directory, never
# inside it: anything under an installed root that lacks a canonical counterpart
# is drift as far as the guard test is concerned.
MANIFEST = ".{name}-sync-manifest.json"

SKIP_SUFFIXES = (".pyc",)
SKIP_PARTS = ("__pycache__",)


def _mirrored_files(canonical: Path) -> list[Path]:
    if not canonical.is_dir():
        return []
    out = []
    for path in sorted(canonical.rglob("*")):
        rel = path.relative_to(canonical)
        if not path.is_file():
            continue
        if path.suffix in SKIP_SUFFIXES or any(p in SKIP_PARTS for p in rel.parts):
            continue
        out.append(path)
    return out


def _manifest_path(installed: Path) -> Path:
    return installed.parent / MANIFEST.format(name=installed.name)


def _under_root(path: Path) -> str:
    """``path`` relative to the repo root, or absolute when it lies outside.

    Both sides of a drift report need this. Guarding only the target, as the
    original did behind a ``# pragma: no cover``, left ``source.relative_to``
    unguarded -- so *reporting* drift raised ``ValueError`` instead of
    printing it. That cannot happen for the real ``MIRRORS`` (every entry is
    under ``ROOT``), which is exactly why it survived: the only way to reach it
    is a test that points ``MIRRORS`` at a temporary tree, which is how this
    function has to be tested at all.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _needs_copy(source: Path, target: Path) -> bool:
    """True when the installed copy is missing, or differs in bytes or mode.

    The mode is part of the comparison because `shutil.copy2` copies it: a
    mirrored ``*.sh`` hook that lost its executable bit on the installed side
    has byte-identical content, so a bytes-only check called it in sync and
    `agent-home-check` exited 0. Executing that copy then raises
    ``PermissionError: [Errno 13]``. Comparing the full permission bits matches
    what the sync path actually writes.
    """
    if not target.is_file():
        return True
    if target.read_bytes() != source.read_bytes():
        return True
    return stat.S_IMODE(target.stat().st_mode) != stat.S_IMODE(source.stat().st_mode)


def _read_manifest(root: Path) -> list[str]:
    path = _manifest_path(root)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    written = data.get("written_by_sync_agent_home", [])
    return [str(p) for p in written] if isinstance(written, list) else []


def _write_manifest(root: Path, written: list[str]) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_by_sync_agent_home": sorted(written),
        "source": ".cursor/",
        "regenerate": "pixi run agent-home",
    }
    _manifest_path(root).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def plan() -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Return (copies to make, stale files to remove) across every mirror."""
    copies: list[tuple[Path, Path]] = []
    stale: list[tuple[Path, Path]] = []
    for canonical, installed in MIRRORS:
        if not canonical.is_dir():
            continue
        present = set()
        for source in _mirrored_files(canonical):
            rel = source.relative_to(canonical)
            present.add(rel.as_posix())
            target = installed / rel
            if _needs_copy(source, target):
                copies.append((source, target))
        for rel_name in _read_manifest(installed):
            rel = Path(rel_name)
            if rel.as_posix() in present:
                continue
            target = installed / rel
            if target.is_file():
                stale.append((target, installed / rel))
    return copies, stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift and exit 1 without writing anything",
    )
    parser.add_argument("--quiet", action="store_true", help="print nothing on success")
    args = parser.parse_args(argv)

    copies, stale = plan()
    if args.check:
        # An absent install is not drift. The copies are gitignored, so a clean
        # checkout has neither tree; only a tree that was actually synced can
        # go stale.
        installed_roots = [installed for _, installed in MIRRORS]

        def _installed_root(target: Path) -> Path | None:
            for root in installed_roots:
                try:
                    target.relative_to(root)
                except ValueError:
                    continue
                return root
            return None

        copies = [
            pair
            for pair in copies
            if (root := _installed_root(pair[1])) is not None
            and (root.exists() or _manifest_path(root).is_file())
        ]
        for source, target in copies:
            print(
                f"drift: {_under_root(target)} does not match {_under_root(source)}",
                file=sys.stderr,
            )
        for target, _ in stale:
            print(
                f"drift: {_under_root(target)} no longer exists in .cursor/",
                file=sys.stderr,
            )
        if copies or stale:
            print("run: pixi run agent-home", file=sys.stderr)
            return 1
        if not args.quiet:
            print("agent-home copies match .cursor/")
        return 0

    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for target, _ in stale:
        target.unlink()

    for canonical, installed in MIRRORS:
        if canonical.is_dir():
            _write_manifest(
                installed,
                [
                    p.relative_to(canonical).as_posix()
                    for p in _mirrored_files(canonical)
                ],
            )

    if not args.quiet:
        print(
            f"synced {len(copies)} file(s) into {len(MIRRORS)} agent home(s)"
            + (f", removed {len(stale)} stale" if stale else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
