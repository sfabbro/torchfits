from __future__ import annotations

import stat
import zipfile
from pathlib import PurePosixPath

MAX_ARCHIVE_MEMBERS = 4096
MAX_MEMBER_SIZE = 256 * 1024 * 1024
MAX_TOTAL_SIZE = 768 * 1024 * 1024
MAX_COMPRESSION_RATIO = 250


def zip_safety_problems(infos: list[zipfile.ZipInfo]) -> list[str]:
    problems: list[str] = []
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        problems.append(
            f"archive has {len(infos)} members; limit is {MAX_ARCHIVE_MEMBERS}"
        )
    total = 0
    names: set[str] = set()
    for info in infos:
        raw_parts = info.filename.split("/")
        path = PurePosixPath(info.filename)
        if (
            not info.filename
            or "\\" in info.filename
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in raw_parts[:-1])
            or raw_parts[-1] in {".", ".."}
        ):
            problems.append(f"{info.filename!r}: unsafe archive member path")
        normalized = str(path)
        if normalized in names:
            problems.append(f"{info.filename}: duplicate archive member")
        names.add(normalized)
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if info.is_dir():
            if info.file_size != 0:
                problems.append(f"{info.filename}: directory has file data")
        elif file_type not in {0, stat.S_IFREG}:
            problems.append(f"{info.filename}: unsupported archive member type")
        total += info.file_size
        if info.file_size > MAX_MEMBER_SIZE:
            problems.append(
                f"{info.filename}: uncompressed size {info.file_size} exceeds "
                f"{MAX_MEMBER_SIZE}"
            )
        if info.file_size and (
            info.compress_size == 0
            or info.file_size > info.compress_size * MAX_COMPRESSION_RATIO
        ):
            problems.append(f"{info.filename}: compression ratio exceeds limit")
    if total > MAX_TOTAL_SIZE:
        problems.append(f"archive uncompressed size {total} exceeds {MAX_TOTAL_SIZE}")
    return problems
