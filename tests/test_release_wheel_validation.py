from __future__ import annotations

import sys
import stat
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_macos_wheel_floor as macos_floor  # noqa: E402
import validate_release_wheels as release_wheels  # noqa: E402
from archive_safety import (  # noqa: E402
    MAX_ARCHIVE_MEMBERS,
    MAX_COMPRESSION_RATIO,
    MAX_MEMBER_SIZE,
    zip_safety_problems,
)


def _wheel(directory: Path, version: str, python: str, platform: str) -> Path:
    name = f"torchfits-{version}-{python}-{python}-{platform}.whl"
    path = directory / name
    dist_info = f"torchfits-{version}.dist-info"
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: torchfits\n"
        f"Version: {version}\n"
        "Requires-Python: >=3.11\n"
        "Requires-Dist: torch<2.15,>=2.14\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        tags = "".join(
            f"Tag: {python}-{python}-{platform_tag}\n"
            for platform_tag in platform.split(".")
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            f"Wheel-Version: 1.0\n{tags}",
        )
    return path


def _matrix(directory: Path, version: str = "1.3.1rc2") -> list[Path]:
    wheels = []
    for python in ("cp311", "cp312", "cp313", "cp314", "cp315"):
        wheels.append(_wheel(directory, version, python, "manylinux_2_28_x86_64"))
        wheels.append(_wheel(directory, version, python, "manylinux_2_28_aarch64"))
    for python in ("cp311", "cp312", "cp313", "cp314"):
        wheels.append(_wheel(directory, version, python, "macosx_14_0_arm64"))
    return wheels


def test_exact_release_matrix_passes_and_writes_manifest(tmp_path: Path) -> None:
    _matrix(tmp_path)
    wheels, problems = release_wheels.validate(tmp_path, "1.3.1rc2")
    assert problems == []
    manifest = tmp_path / "WHEEL-MANIFEST.sha256"
    release_wheels.write_manifest(wheels, manifest)
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 14
    assert release_wheels.verify_manifest(wheels, manifest) == []


def test_manifest_rejects_changed_wheel(tmp_path: Path) -> None:
    wheels = _matrix(tmp_path)
    manifest = tmp_path / "WHEEL-MANIFEST.sha256"
    release_wheels.write_manifest(wheels, manifest)
    wheels[0].write_bytes(wheels[0].read_bytes() + b"changed")
    problems = release_wheels.verify_manifest(wheels, manifest)
    assert any("size does not match" in problem for problem in problems)
    assert any("SHA-256 does not match" in problem for problem in problems)


def test_torch_requirement_requires_exact_distribution_name() -> None:
    from email.parser import Parser

    for name in ("torchvision", "torch-extra", "torch.foo"):
        metadata = Parser().parsestr(f"Requires-Dist: {name}<2.15,>=2.14\n")
        assert release_wheels._torch_requirement(metadata) is None


def test_release_manifest_includes_and_verifies_sdist(tmp_path: Path) -> None:
    wheels = _matrix(tmp_path)
    sdist = tmp_path / "torchfits-1.3.1rc2.tar.gz"
    sdist.write_bytes(b"sdist")
    manifest = tmp_path / "RELEASE-MANIFEST.sha256"
    release_wheels.write_manifest(wheels + [sdist], manifest)
    assert sdist.name in manifest.read_text(encoding="utf-8")
    assert release_wheels.verify_manifest(wheels + [sdist], manifest) == []
    sdist.write_bytes(b"replaced")
    assert any(
        sdist.name in problem
        for problem in release_wheels.verify_manifest(wheels + [sdist], manifest)
    )


def test_release_matrix_rejects_old_macos_tag(tmp_path: Path) -> None:
    wheels = _matrix(tmp_path)
    mac = next(wheel for wheel in wheels if "macosx" in wheel.name)
    mac.rename(Path(str(mac).replace("macosx_14_0", "macosx_11_0")))
    _, problems = release_wheels.validate(tmp_path, "1.3.1rc2")
    assert any("unexpected platform tag macosx_11_0_arm64" in item for item in problems)
    assert any("wheel matrix mismatch" in item for item in problems)


def test_release_matrix_rejects_wrong_manylinux_floor(tmp_path: Path) -> None:
    wheels = _matrix(tmp_path)
    linux = next(wheel for wheel in wheels if "manylinux" in wheel.name)
    linux.rename(Path(str(linux).replace("manylinux_2_28", "manylinux_2_17")))
    _, problems = release_wheels.validate(tmp_path, "1.3.1rc2")
    assert any("unexpected platform tag manylinux_2_17" in item for item in problems)


def test_release_matrix_rejects_duplicate_cell(tmp_path: Path) -> None:
    _matrix(tmp_path)
    _wheel(
        tmp_path,
        "1.3.1rc2",
        "cp311",
        "manylinux_2_28_x86_64.manylinux_2_26_x86_64",
    )
    _, problems = release_wheels.validate(tmp_path, "1.3.1rc2")
    assert "wheel matrix contains duplicate cells" in problems


def test_release_matrix_accepts_compressed_linux_platform_tags(tmp_path: Path) -> None:
    _matrix(tmp_path)
    source = next(tmp_path.glob("*cp311*manylinux_2_28_x86_64.whl"))
    compressed = Path(
        str(source).replace(
            "manylinux_2_28_x86_64", "manylinux_2_28_x86_64.manylinux_2_26_x86_64"
        )
    )
    with zipfile.ZipFile(source) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    source.unlink()
    wheel_name = next(name for name in entries if name.endswith(".dist-info/WHEEL"))
    entries[wheel_name] = entries[wheel_name].replace(
        b"Tag: cp311-cp311-manylinux_2_28_x86_64\n",
        b"Tag: cp311-cp311-manylinux_2_28_x86_64\n"
        b"Tag: cp311-cp311-manylinux_2_26_x86_64\n",
    )
    with zipfile.ZipFile(compressed, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    _, problems = release_wheels.validate(tmp_path, "1.3.1rc2")
    assert problems == []


def test_macos_checker_requires_tag_and_binary_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "torchfits-1.3.1rc2-cp313-cp313-macosx_14_0_arm64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("torchfits/_C.cpython-313-darwin.so", b"native")
        archive.writestr("torchfits/libtorchfits_core.dylib", b"native")

    class Result:
        stdout = "      cmd LC_BUILD_VERSION\n    minos 14.0\n"

    monkeypatch.setattr(macos_floor.subprocess, "run", lambda *args, **kwargs: Result())
    assert macos_floor.check(wheel, "14.0") == []


def test_macos_checker_rejects_false_compatibility_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "torchfits-1.3.1rc2-cp313-cp313-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("torchfits/_C.cpython-313-darwin.so", b"native")

    class Result:
        stdout = "      cmd LC_BUILD_VERSION\n    minos 14.0\n"

    monkeypatch.setattr(macos_floor.subprocess, "run", lambda *args, **kwargs: Result())
    assert any("macosx_14_0_arm64" in item for item in macos_floor.check(wheel, "14.0"))


def test_macos_checker_rejects_oversized_member_before_extract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "torchfits-1.3.1rc2-cp313-cp313-macosx_14_0_arm64.whl"
    wheel.write_bytes(b"fixture")

    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def infolist(self):
            info = zipfile.ZipInfo("torchfits/_C.so")
            info.file_size = MAX_MEMBER_SIZE + 1
            info.compress_size = 1
            return [info]

        def extract(self, *args):
            raise AssertionError("extract must not be called")

    monkeypatch.setattr(macos_floor.zipfile, "ZipFile", lambda *args: Archive())
    assert any("uncompressed size" in item for item in macos_floor.check(wheel, "14.0"))


def test_zip_limits_reject_member_count_and_compression_ratio() -> None:
    infos = [zipfile.ZipInfo(str(index)) for index in range(MAX_ARCHIVE_MEMBERS + 1)]
    assert any("members" in problem for problem in zip_safety_problems(infos))
    compressed = zipfile.ZipInfo("payload")
    compressed.file_size = MAX_COMPRESSION_RATIO + 1
    compressed.compress_size = 1
    assert any(
        "compression ratio" in problem for problem in zip_safety_problems([compressed])
    )


@pytest.mark.parametrize(
    "name",
    ["/absolute", "../parent", "pkg/../escape", "pkg\\alias"],
)
def test_zip_safety_rejects_unsafe_member_paths(name: str) -> None:
    assert any(
        "unsafe archive member path" in problem
        for problem in zip_safety_problems([zipfile.ZipInfo(name)])
    )


def test_zip_safety_rejects_duplicates_and_non_regular_members() -> None:
    duplicate = [zipfile.ZipInfo("pkg/data"), zipfile.ZipInfo("pkg/data")]
    assert any(
        "duplicate archive member" in problem
        for problem in zip_safety_problems(duplicate)
    )
    link = zipfile.ZipInfo("pkg/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    assert any(
        "unsupported archive member type" in problem
        for problem in zip_safety_problems([link])
    )
