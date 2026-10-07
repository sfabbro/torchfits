"""``scripts/sync_agent_home.py`` — the gate that keeps ``.agents/`` and
``.codex/`` equal to the canonical ``.cursor/`` tree.

This script *is* the ``agent-home-check`` release gate, and nothing tested it.
It runs as a subprocess, so ``coverage run`` over the suite records 0% for it
and cannot distinguish "no test" from "tested via subprocess" — which is why
the blind spot below survived. Every test here drives the module directly with
``MIRRORS`` pointed at a temporary tree.

The blind spot (R2-064): ``plan()`` compared ``target.read_bytes() !=
source.read_bytes()`` and nothing else, so a mirrored ``*.sh`` hook that had
lost its executable bit on the installed side was byte-identical and therefore
"in sync". ``agent-home-check`` exited 0 on a hook the client cannot run.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    """Import the script by path -- ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "sync_agent_home_under_test", REPO / "scripts" / "sync_agent_home.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def mirror(tmp_path, monkeypatch):
    """A two-root mirror: canonical ``.cursor/hooks`` -> installed ``.codex/hooks``."""
    mod = _load()
    canonical = tmp_path / ".cursor" / "hooks"
    installed = tmp_path / ".codex" / "hooks"
    canonical.mkdir(parents=True)
    installed.mkdir(parents=True)
    monkeypatch.setattr(mod, "MIRRORS", ((canonical, installed),))
    return mod, canonical, installed


def _hook(canonical: Path, name: str = "harness-stop.sh", mode: int = 0o755) -> Path:
    path = canonical / name
    path.write_text("#!/bin/sh\nexit 0\n")
    os.chmod(path, mode)
    return path


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# --- control: a synced tree is not drift ------------------------------------


def test_synced_tree_is_not_drift(mirror):
    """Without this, every rejection below could pass by rejecting everything."""
    mod, canonical, installed = mirror
    _hook(canonical)
    assert mod.main(["--quiet"]) == 0

    copies, stale = mod.plan()
    assert (copies, stale) == ([], [])
    assert mod.main(["--check", "--quiet"]) == 0


# --- R2-064: the mode is part of the contract -------------------------------


def test_check_detects_a_lost_executable_bit(mirror):
    """The finding. Bytes are identical; only the mode drifted, and the gate
    used to call that in sync."""
    mod, canonical, installed = mirror
    _hook(canonical)
    mod.main(["--quiet"])

    copied = installed / "harness-stop.sh"
    os.chmod(copied, 0o644)

    assert _mode(copied) == 0o644
    copies, stale = mod.plan()
    assert len(copies) == 1 and not stale, (copies, stale)
    assert mod.main(["--check", "--quiet"]) == 1


def test_the_unexecutable_copy_is_genuinely_unrunnable(mirror):
    """Why the mode matters: the consequence is a hard failure, not a warning."""
    mod, canonical, installed = mirror
    _hook(canonical)
    mod.main(["--quiet"])
    copied = installed / "harness-stop.sh"

    assert subprocess.run([str(canonical / "harness-stop.sh")]).returncode == 0
    assert _mode(copied) == 0o755

    os.chmod(copied, 0o644)
    with pytest.raises(PermissionError):
        subprocess.run([str(copied)])


def test_a_non_executable_canonical_file_stays_non_executable(mirror):
    """The mirror must not demand +x from a 644 source -- SKILL.md and the
    inventory script are both 644. Only *differences* are drift."""
    mod, canonical, installed = mirror
    skill = canonical / "SKILL.md"
    skill.write_text("# skill\n")
    os.chmod(skill, 0o644)

    assert mod.main(["--quiet"]) == 0
    assert _mode(installed / "SKILL.md") == 0o644
    assert mod.main(["--check", "--quiet"]) == 0


def test_a_gained_executable_bit_is_also_drift(mirror):
    """Direction must not matter: 755 canonical, 755 copy is the only match."""
    mod, canonical, installed = mirror
    _hook(canonical)
    mod.main(["--quiet"])
    copied = installed / "harness-stop.sh"
    os.chmod(copied, 0o777)
    assert mod.main(["--check", "--quiet"]) == 1


# --- bytes, presence, and staleness -----------------------------------------


def test_check_detects_byte_drift(mirror):
    mod, canonical, installed = mirror
    hook = _hook(canonical)
    mod.main(["--quiet"])
    (installed / "harness-stop.sh").write_text("#!/bin/sh\nexit 1\n")
    assert hook.read_bytes() != (installed / "harness-stop.sh").read_bytes()
    assert mod.main(["--check", "--quiet"]) == 1


def test_check_detects_a_missing_file(mirror):
    mod, canonical, installed = mirror
    _hook(canonical)
    mod.main(["--quiet"])
    (installed / "harness-stop.sh").unlink()
    assert mod.main(["--check", "--quiet"]) == 1


def test_sync_creates_missing_parents(mirror):
    mod, canonical, installed = mirror
    nested = canonical / "sub" / "dir"
    nested.mkdir(parents=True)
    (nested / "deep.sh").write_text("#!/bin/sh\n")
    assert mod.main(["--quiet"]) == 0
    assert (installed / "sub" / "dir" / "deep.sh").read_bytes() == (
        nested / "deep.sh"
    ).read_bytes()


def test_deleted_canonical_file_is_removed_from_the_copy(mirror):
    """The manifest exists so a deletion propagates without touching anything
    else a user installed under the same root."""
    mod, canonical, installed = mirror
    keep = _hook(canonical, "keep.sh")
    gone = _hook(canonical, "gone.sh")
    mod.main(["--quiet"])
    assert (installed / "gone.sh").is_file()

    # Something the user put there themselves: not in the manifest, must survive.
    foreign = installed / "mine.sh"
    foreign.write_text("# mine\n")

    canonical.joinpath("gone.sh").unlink()
    del gone
    assert mod.main(["--quiet"]) == 0
    assert not (installed / "gone.sh").exists()
    assert (installed / "keep.sh").read_bytes() == keep.read_bytes()
    assert foreign.is_file(), "sync deleted a file it did not write"


def test_a_deleted_canonical_file_is_not_stale_without_the_manifest(mirror):
    """Removals are keyed off the manifest, so with no manifest a deleted
    canonical file leaves its copy behind unreported. Pinned so the limit is a
    known one rather than a surprise.

    Note the boundary this is *not*: a file that is merely *missing* from the
    installed side is still reported, because ``plan()`` compares every
    canonical file against its target directly.
    """
    mod, canonical, installed = mirror
    _hook(canonical)
    mod.main(["--quiet"])
    (installed.parent / ".hooks-sync-manifest.json").unlink()
    (canonical / "harness-stop.sh").unlink()

    copies, stale = mod.plan()
    assert (copies, stale) == ([], [])
    assert mod.main(["--check", "--quiet"]) == 0
    # ...and the orphaned copy is still there.
    assert (installed / "harness-stop.sh").is_file()


def test_a_missing_installed_file_is_reported_without_a_manifest(mirror):
    """The other side of the boundary the manifest test draws: a canonical
    file with no installed counterpart is drift on its own evidence."""
    mod, canonical, installed = mirror
    _hook(canonical)
    mod.main(["--quiet"])
    (installed.parent / ".hooks-sync-manifest.json").unlink()
    (installed / "harness-stop.sh").unlink()

    copies, stale = mod.plan()
    assert len(copies) == 1 and not stale
    assert mod.main(["--check", "--quiet"]) == 1


def test_corrupt_manifest_is_treated_as_empty(mirror):
    mod, canonical, installed = mirror
    _hook(canonical)
    mod.main(["--quiet"])
    (installed.parent / ".hooks-sync-manifest.json").write_text("{ not json")
    assert mod.main(["--check", "--quiet"]) == 0


# --- what is mirrored -------------------------------------------------------


def test_pyc_and_pycache_are_skipped(mirror):
    mod, canonical, installed = mirror
    _hook(canonical)
    (canonical / "junk.pyc").write_bytes(b"\x00")
    cache = canonical / "__pycache__"
    cache.mkdir()
    (cache / "x.pyc").write_bytes(b"\x00")
    (cache / "y.txt").write_text("no\n")

    assert mod.main(["--quiet"]) == 0
    assert not (installed / "junk.pyc").exists()
    assert not (installed / "__pycache__").exists()
    assert (installed / "harness-stop.sh").is_file()


def test_a_missing_canonical_root_is_not_drift(mirror, tmp_path):
    """An uninstalled agent home must not be reported as drift."""
    mod, canonical, installed = mirror
    import shutil

    shutil.rmtree(canonical)
    assert mod.main(["--check", "--quiet"]) == 0


# --- the reporting path, which this is the first thing to exercise ----------


def test_drift_report_names_both_paths_without_raising(mirror, capsys):
    """`--check` guarded `target.relative_to(ROOT)` but not
    `source.relative_to(ROOT)`, so reporting drift raised ValueError instead of
    printing it. It cannot fire for the real MIRRORS, which is exactly why it
    survived -- and why this module had to be driven with MIRRORS repointed."""
    mod, canonical, installed = mirror
    _hook(canonical)
    mod.main(["--quiet"])
    os.chmod(installed / "harness-stop.sh", 0o644)

    assert mod.main(["--check"]) == 1
    err = capsys.readouterr().err
    assert "drift:" in err
    assert "run: pixi run agent-home" in err
    assert str(installed / "harness-stop.sh") in err
    assert str(canonical / "harness-stop.sh") in err


def test_check_writes_nothing(mirror):
    """The CI path must be read-only."""
    mod, canonical, installed = mirror
    _hook(canonical)
    assert mod.main(["--check", "--quiet"]) == 1  # nothing synced yet
    assert not (installed / "harness-stop.sh").exists()

    mod.main(["--quiet"])
    before = (installed / "harness-stop.sh").read_bytes()
    os.chmod(installed / "harness-stop.sh", 0o644)
    assert mod.main(["--check", "--quiet"]) == 1
    assert (installed / "harness-stop.sh").read_bytes() == before
    assert _mode(installed / "harness-stop.sh") == 0o644, "--check repaired it"


# --- the gate itself, end to end -------------------------------------------


def test_absent_install_is_not_drift(tmp_path, monkeypatch):
    """A clean checkout has no ``.agents`` tree. That is not drift."""
    mod = _load()
    canonical = tmp_path / ".cursor" / "skills"
    installed = tmp_path / ".agents" / "skills"
    canonical.mkdir(parents=True)
    (canonical / "SKILL.md").write_text("hello\n")
    monkeypatch.setattr(mod, "MIRRORS", ((canonical, installed),))
    assert mod.main(["--check", "--quiet"]) == 0
    assert not installed.exists()


def test_the_real_mirrors_are_in_sync():
    """`agent-home-check` as CI runs it, against this repository."""
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "sync_agent_home.py"), "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stdout + rc.stderr
