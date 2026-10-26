from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_release_run as release_run  # noqa: E402
import validate_release_sdist as sdist_validator  # noqa: E402


def _entries(version: str = "1.3.1rc2") -> dict[str, bytes]:
    entries = {name: b"x" for name in sdist_validator.REQUIRED}
    entries["PKG-INFO"] = (
        f"Metadata-Version: 2.4\nName: torchfits\nVersion: {version}\n"
    ).encode()
    return entries


def _sdist(
    directory: Path,
    version: str = "1.3.1rc2",
    entries: dict[str, bytes] | None = None,
    extras: list[tarfile.TarInfo] | None = None,
) -> Path:
    path = directory / f"torchfits-{version}.tar.gz"
    root = f"torchfits-{version}"
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in (entries or _entries(version)).items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        for info in extras or []:
            archive.addfile(info, io.BytesIO(b"x") if info.isfile() else None)
    return path


def test_valid_sdist_passes(tmp_path: Path) -> None:
    assert sdist_validator.validate(_sdist(tmp_path), "1.3.1rc2") == []


@pytest.mark.parametrize(
    ("version", "metadata_version", "metadata_name", "message"),
    [
        ("1.3.1rc2", "1.3.1rc1", "torchfits", "PKG-INFO Version"),
        ("1.3.1rc2", "1.3.1rc2", "other", "PKG-INFO Name"),
    ],
)
def test_sdist_rejects_wrong_identity(
    tmp_path: Path,
    version: str,
    metadata_version: str,
    metadata_name: str,
    message: str,
) -> None:
    entries = _entries(version)
    entries["PKG-INFO"] = (
        f"Metadata-Version: 2.4\nName: {metadata_name}\nVersion: {metadata_version}\n"
    ).encode()
    assert any(
        message in problem
        for problem in sdist_validator.validate(
            _sdist(tmp_path, version, entries), version
        )
    )


@pytest.mark.parametrize("name", ["/absolute", "torchfits-1.3.1rc2/../escape"])
def test_sdist_rejects_unsafe_member(tmp_path: Path, name: str) -> None:
    info = tarfile.TarInfo(name)
    info.size = 1
    assert any(
        "unsafe member" in problem
        for problem in sdist_validator.validate(
            _sdist(tmp_path, extras=[info]), "1.3.1rc2"
        )
    )


def test_sdist_rejects_escaping_link(tmp_path: Path) -> None:
    info = tarfile.TarInfo("torchfits-1.3.1rc2/link")
    info.type = tarfile.SYMTYPE
    info.linkname = "../../escape"
    assert any(
        "unsafe link" in problem
        for problem in sdist_validator.validate(
            _sdist(tmp_path, extras=[info]), "1.3.1rc2"
        )
    )


def test_sdist_rejects_missing_link_target(tmp_path: Path) -> None:
    info = tarfile.TarInfo("torchfits-1.3.1rc2/link")
    info.type = tarfile.SYMTYPE
    info.linkname = "missing"
    assert any(
        "link target missing" in problem
        for problem in sdist_validator.validate(
            _sdist(tmp_path, extras=[info]), "1.3.1rc2"
        )
    )


def test_sdist_rejects_multiple_roots(tmp_path: Path) -> None:
    info = tarfile.TarInfo("other-root/file")
    info.size = 1
    assert any(
        "expected single root" in problem
        for problem in sdist_validator.validate(
            _sdist(tmp_path, extras=[info]), "1.3.1rc2"
        )
    )


def test_sdist_rejects_duplicate_member(tmp_path: Path) -> None:
    info = tarfile.TarInfo("torchfits-1.3.1rc2/README.md")
    info.size = 1
    assert any(
        "duplicate member" in problem
        for problem in sdist_validator.validate(
            _sdist(tmp_path, extras=[info]), "1.3.1rc2"
        )
    )


def test_sdist_rejects_missing_required_file(tmp_path: Path) -> None:
    entries = _entries()
    del entries["extern/cfitsio/CMakeLists.txt"]
    assert any(
        "missing required files" in problem
        for problem in sdist_validator.validate(
            _sdist(tmp_path, entries=entries), "1.3.1rc2"
        )
    )


@pytest.mark.parametrize(
    "name", ["build/output.o", ".github/workflows/publish.yml", "src/torchfits/_C.so"]
)
def test_sdist_rejects_forbidden_leakage(tmp_path: Path, name: str) -> None:
    entries = _entries()
    entries[name] = b"leak"
    assert any(
        "forbidden member" in problem
        for problem in sdist_validator.validate(
            _sdist(tmp_path, entries=entries), "1.3.1rc2"
        )
    )


def test_sdist_rejects_member_count_before_metadata_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "torchfits-1.3.1rc2.tar.gz"
    path.write_bytes(b"fixture")

    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def getmembers(self):
            return [
                tarfile.TarInfo(str(index))
                for index in range(sdist_validator.MAX_MEMBERS + 1)
            ]

        def extractfile(self, *args):
            raise AssertionError("extractfile must not be called")

    monkeypatch.setattr(sdist_validator.tarfile, "open", lambda *args: Archive())
    assert any(
        "members exceeds" in problem
        for problem in sdist_validator.validate(path, "1.3.1rc2")
    )


def test_sdist_rejects_oversized_member_before_metadata_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "torchfits-1.3.1rc2.tar.gz"
    path.write_bytes(b"fixture")
    oversized = tarfile.TarInfo("torchfits-1.3.1rc2/payload")
    oversized.size = sdist_validator.MAX_MEMBER_SIZE + 1

    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def getmembers(self):
            return [oversized]

        def extractfile(self, *args):
            raise AssertionError("extractfile must not be called")

    monkeypatch.setattr(sdist_validator.tarfile, "open", lambda *args: Archive())
    assert any(
        "oversized member" in problem
        for problem in sdist_validator.validate(path, "1.3.1rc2")
    )


def test_successful_rc_run_requires_tag_and_sha_identity() -> None:
    runs = [
        {
            "event": "push",
            "conclusion": "success",
            "headBranch": "v1.3.1",
            "headSha": "same",
        },
        {
            "event": "push",
            "conclusion": "failure",
            "headBranch": "v1.3.1rc2",
            "headSha": "same",
        },
    ]
    assert not release_run.has_successful_run(runs, "v1.3.1rc2", "same")
    runs.append(
        {
            "event": "push",
            "conclusion": "success",
            "headBranch": "v1.3.1rc2",
            "headSha": "same",
        }
    )
    assert release_run.has_successful_run(runs, "v1.3.1rc2", "same")
