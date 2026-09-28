"""extern/vendor.sh must not be able to move the repository's own CFITSIO pin.

`extern/VERSIONS.txt` is tracked and is the single place the vendored CFITSIO
tag and its sha256 live. vendor.sh ends by rewriting that file with whatever it
just downloaded -- that is what makes "pin a new tag by hand" work, and it is
also the only mechanism in the repo that can change the pin. It must therefore
be reachable *only* from an invocation that passed the pin file in.

Measured on 2026-09-27, starting from the repo's own tracked
`extern/VERSIONS.txt` (cfitsio-4.7.0 / f281ca29...):

    $ TORCHFITS_VENDOR_ALLOW_UNPINNED=1 \
        bash extern/vendor.sh --cfitsio-version cfitsio-4.6.2
    ...
    cfitsio_tag=cfitsio-4.6.2
    cfitsio_sha256=83f710f9441e7c703eeb485c9a475f50801ae7fad32ac7e356a406d318b4c02f

exit 0, no warning, tracked file rewritten. The invocation is the one
`vendor.sh --help` advertises. Worse, the patch loop keys off
`<tag>-<name>.patch`, so both `cfitsio-4.7.0-*.patch` files were skipped
silently -- including `plio-cbuf`, the fix for a CFITSIO heap overflow on
PLIO-compressed incompressible data. Nothing but a one-line diff in a pin file
records that the vendored tree lost its security patch.

The same run with the *same* tag reproduced the recorded sha256 byte for byte,
so the pin-file path is idempotent and CI cannot repoint the pin behind your
back; the hole is only reachable from a bare tag. These tests pin both halves.

Every run here is hermetic: `curl` is stubbed with a script that writes a
prebuilt tarball, so nothing touches the network and the real `tar`/`shasum`
still do the work.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VENDOR_SH = ROOT / "extern" / "vendor.sh"
TRACKED_VERSIONS = ROOT / "extern" / "VERSIONS.txt"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("tar") is None,
    reason="extern/vendor.sh needs bash and tar",
)


# GitHub serves `archive/refs/tags/<tag>.tar.gz` as a gzip tar whose single
# top-level directory is the tag. vendor.sh does `find -mindepth 1 -maxdepth 1
# -type d | head -n1` on it, so one directory and one file is the whole
# contract.
_STUB_CURL = """#!/usr/bin/env bash
set -eu
out=""
prev=""
for arg in "$@"; do
  if [[ "$prev" == "-o" ]]; then out="$arg"; fi
  prev="$arg"
done
if [[ -z "$out" ]]; then
  echo "stub curl: no -o argument in: $*" >&2
  exit 64
fi
cp "$TF_FAKE_ARCHIVE" "$out"
"""


def _fake_archive(tag: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        payload = b"/* stand-in for cfitsio-4.x/imcompress.c */\n"
        info = tarfile.TarInfo(f"cfitsio-{tag}/imcompress.c")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _run_vendor(
    tree: Path, spec: str, tag: str, archive: Path, *, allow_unpinned: bool = True
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{tree / 'bin'}{os.pathsep}{env['PATH']}"
    env["TF_FAKE_ARCHIVE"] = str(archive)
    if allow_unpinned:
        env["TORCHFITS_VENDOR_ALLOW_UNPINNED"] = "1"
    return subprocess.run(
        ["bash", str(tree / "extern" / "vendor.sh"), "--cfitsio-version", spec],
        capture_output=True,
        text=True,
        check=False,
        cwd=tree,
        env=env,
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A throwaway checkout root with vendor.sh, a stub curl and a fake tarball."""
    (tmp_path / "extern").mkdir()
    (tmp_path / "bin").mkdir()
    shutil.copy(VENDOR_SH, tmp_path / "extern" / "vendor.sh")
    shutil.copy(TRACKED_VERSIONS, tmp_path / "extern" / "VERSIONS.txt")
    curl = tmp_path / "bin" / "curl"
    curl.write_text(_STUB_CURL, encoding="utf-8")
    curl.chmod(0o755)
    return tmp_path


def _archive_for(tree: Path, tag: str) -> tuple[Path, str]:
    path = tree / f"{tag}.tar.gz"
    data = _fake_archive(tag)
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def test_bare_tag_never_rewrites_the_tracked_pin(tree: Path) -> None:
    """A bare tag is a read-only preview; only the pin file may be written.

    This is the measured repoint from the module docstring, reduced to a
    hermetic case. The stub archive is a different tag from the tracked pin, so
    a script that writes back turns the tracked file into a different tag *and*
    a different hash -- exactly what happened to `extern/VERSIONS.txt` above.
    """
    archive, _ = _archive_for(tree, "cfitsio-9.9.9")
    before = (tree / "extern" / "VERSIONS.txt").read_text(encoding="utf-8")

    result = _run_vendor(tree, "cfitsio-9.9.9", "cfitsio-9.9.9", archive)

    after = (tree / "extern" / "VERSIONS.txt").read_text(encoding="utf-8")
    assert after == before, (
        "vendor.sh rewrote the tracked pin file from a bare-tag invocation. "
        f"Before:\n{before}After:\n{after}"
    )
    # The run itself must still succeed -- this is not 'refuse to vendor', it
    # is 'vendor the requested tag without touching the pin'.
    assert result.returncode == 0, result.stderr
    # ...and the vendored tree must exist, or the guard is trivially satisfied
    # by doing nothing at all.
    assert (tree / "extern" / "cfitsio" / "imcompress.c").is_file()


def test_bare_tag_reports_the_hash_it_computed(tree: Path) -> None:
    """Refusing to write the pin must not lose the hash the operator needs.

    The `--help` text promises the hash "is then computed and recorded for the
    next run"; with the write gone, it has to be printed instead, or pinning a
    new tag by hand means hand-hashing a 2 MB download.
    """
    archive, digest = _archive_for(tree, "cfitsio-9.9.9")

    result = _run_vendor(tree, "cfitsio-9.9.9", "cfitsio-9.9.9", archive)

    assert result.returncode == 0, result.stderr
    assert digest in result.stdout + result.stderr, (
        "a bare-tag run must still report the sha256 it computed, otherwise "
        f"there is no way to pin the tag. Output:\n{result.stdout}{result.stderr}"
    )


@pytest.mark.skipif(shutil.which("patch") is None, reason="needs patch(1)")
def test_patches_for_another_tag_are_announced(tree: Path) -> None:
    """Skipping `<other-tag>-*.patch` must be loud.

    The patch loop globs `"${CFITSIO_VERSION}"-*.patch`, so vendoring any tag
    other than the one the patches were written against silently builds an
    unpatched CFITSIO. Today the only patches are `cfitsio-4.7.0-bzip2.patch`
    and `cfitsio-4.7.0-plio-cbuf.patch`, and plio-cbuf is a heap-overflow fix;
    losing it is a security regression that leaves no trace in the build log.
    """
    patches = tree / "extern" / "patches"
    patches.mkdir()
    shutil.copy(
        ROOT / "extern" / "patches" / "cfitsio-4.7.0-plio-cbuf.patch",
        patches / "cfitsio-4.7.0-plio-cbuf.patch",
    )
    archive, _ = _archive_for(tree, "cfitsio-9.9.9")

    result = _run_vendor(tree, "cfitsio-9.9.9", "cfitsio-9.9.9", archive)

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "plio-cbuf" in output, (
        "vendor.sh skipped a patch written for a different CFITSIO tag without "
        f"saying so. Output:\n{output}"
    )


def test_pin_file_run_verifies_and_records(tree: Path) -> None:
    """The control: the pin-file path still vendors and still records.

    Guards the fixes above from over-correcting into "vendor.sh never writes".
    A pin file naming tag T can only ever be rewritten with tag T (the tag is
    read out of the file), so this write is what keeps `--cfitsio-version
    extern/VERSIONS.txt` the only path that can update a pin.
    """
    tag = "cfitsio-4.7.0"
    archive, digest = _archive_for(tree, tag)
    versions = tree / "extern" / "VERSIONS.txt"
    versions.write_text(
        f"cfitsio_repo=HEASARC/cfitsio\ncfitsio_tag={tag}\ncfitsio_sha256={digest}\n",
        encoding="utf-8",
    )

    result = _run_vendor(
        tree, "extern/VERSIONS.txt", tag, archive, allow_unpinned=False
    )

    assert result.returncode == 0, result.stderr
    assert f"cfitsio_sha256={digest}" in versions.read_text(encoding="utf-8")


def test_pin_file_run_refuses_a_hash_mismatch(tree: Path) -> None:
    """A pin whose hash no longer matches upstream must not vendor, or write."""
    tag = "cfitsio-4.7.0"
    archive, _ = _archive_for(tree, tag)
    versions = tree / "extern" / "VERSIONS.txt"
    pinned = versions.read_text(encoding="utf-8")
    versions.write_text(
        pinned.rstrip("\n").rsplit("=", 1)[0] + "=" + "0" * 64 + "\n",
        encoding="utf-8",
    )

    result = _run_vendor(
        tree, "extern/VERSIONS.txt", tag, archive, allow_unpinned=False
    )

    assert result.returncode != 0, "a sha256 mismatch must fail the vendoring"
    assert "MISMATCH" in result.stderr
    assert (
        versions.read_text(encoding="utf-8")
        == pinned.rstrip("\n").rsplit("=", 1)[0] + "=" + "0" * 64 + "\n"
    ), "the pin must not be rewritten on a mismatch"
    assert not (tree / "extern" / "cfitsio").exists()


def test_no_version_argument_prints_usage(tree: Path) -> None:
    """`./extern/vendor.sh` with no arguments must explain itself.

    The bare form is what docs/install.md, docs/contributing.md and the
    cfitsio_direct CMake error message tell people to run. Measured, it exits 1
    with `Failed to resolve CFITSIO version from: ` -- an empty spec and no
    hint that `--cfitsio-version` is the required argument.
    """
    result = subprocess.run(
        ["bash", str(tree / "extern" / "vendor.sh")],
        capture_output=True,
        text=True,
        check=False,
        cwd=tree,
        env={**os.environ, "PATH": f"{tree / 'bin'}{os.pathsep}{os.environ['PATH']}"},
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "--cfitsio-version" in output, (
        f"a bare invocation must print usage. Output:\n{output}"
    )
    assert "from: \n" not in output, "the empty-spec message must be gone"
