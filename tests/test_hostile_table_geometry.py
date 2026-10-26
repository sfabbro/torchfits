from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import torchfits
import torchfits._C as cpp


def _fits_card(key: str, value: str | int | bool | None = None) -> bytes:
    if value is None:
        text = key
    elif isinstance(value, bool):
        text = f"{key:<8}= {'T' if value else 'F':>20}"
    elif isinstance(value, str):
        text = f"{key:<8}= {value!r:<20}"
    else:
        text = f"{key:<8}= {value:>20}"
    return text.ljust(80).encode("ascii")


def _fits_header(cards: list[bytes]) -> bytes:
    payload = b"".join([*cards, _fits_card("END")])
    return payload.ljust((len(payload) + 2879) // 2880 * 2880, b" ")


def test_zero_field_zero_width_binary_table_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "empty-table.fits"
    primary = _fits_header(
        [
            _fits_card("SIMPLE", True),
            _fits_card("BITPIX", 8),
            _fits_card("NAXIS", 0),
            _fits_card("EXTEND", True),
        ]
    )
    extension = _fits_header(
        [
            _fits_card("XTENSION", "BINTABLE"),
            _fits_card("BITPIX", 8),
            _fits_card("NAXIS", 2),
            _fits_card("NAXIS1", 0),
            _fits_card("NAXIS2", 0),
            _fits_card("PCOUNT", 0),
            _fits_card("GCOUNT", 1),
            _fits_card("TFIELDS", 0),
        ]
    )
    path.write_bytes(primary + extension)
    assert torchfits.table.read_torch(str(path), hdu=1, mmap=False) == {}
    arrow = torchfits.table.read(str(path), hdu=1, mmap=False)
    assert arrow.num_rows == 0
    assert arrow.num_columns == 0


def _header(path: Path, start: int) -> tuple[dict[str, int], int]:
    payload = path.read_bytes()
    positions: dict[str, int] = {}
    offset = start
    while True:
        card = payload[offset : offset + 80]
        key = card[:8].decode("ascii").strip()
        positions[key] = offset
        offset += 80
        if key == "END":
            break
    return positions, ((offset + 2879) // 2880) * 2880


def _table_header(path: Path) -> tuple[dict[str, int], int]:
    _, extension_start = _header(path, 0)
    return _header(path, extension_start)


def _table_extent(path: Path) -> int:
    positions, data_start = _table_header(path)
    payload = path.read_bytes()

    def value(key: str) -> int:
        card = payload[positions[key] : positions[key] + 80].decode("ascii")
        return int(card.split("=", 1)[1].split("/", 1)[0])

    return data_start + value("NAXIS1") * value("NAXIS2") + value("PCOUNT")


def _replace_card(path: Path, key: str, value: str | int) -> None:
    positions, _ = _table_header(path)
    rendered = f"'{value}'" if isinstance(value, str) else str(value)
    card = f"{key:<8}= {rendered:>20}".ljust(80).encode("ascii")
    with path.open("r+b") as stream:
        stream.seek(positions[key])
        stream.write(card)


def _vla_file(path: Path, descriptor: str) -> tuple[Path, int]:
    rows = [
        np.array([1], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([3], dtype=np.int32),
    ]
    torchfits.table.write(
        str(path),
        {"V": rows},
        schema={"V": {"format": f"1{descriptor}J"}},
        overwrite=True,
    )
    _, data_start = _table_header(path)
    return path, data_start


def _patch_descriptors(
    path: Path, descriptor: str, values: list[tuple[int, int]]
) -> Path:
    _, data_start = _table_header(path)
    code = ">ii" if descriptor == "P" else ">qq"
    with path.open("r+b") as stream:
        stream.seek(data_start)
        for repeat, offset in values:
            stream.write(struct.pack(code, repeat, offset))
    return path


def _assert_geometry_rejected_under_memory_limit(
    path: Path, mmap: bool, expected: str
) -> None:
    script = """
import resource
import sys
import torchfits._C as cpp

if sys.platform.startswith("linux"):
    status = open("/proc/self/status", encoding="ascii").read().splitlines()
    current_kib = int(
        next(line for line in status if line.startswith("VmSize:")).split()[1]
    )
    limit = (current_kib + 128 * 1024) * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, resource.RLIM_INFINITY))
try:
    cpp.read_fits_table_rows(sys.argv[1], 1, [], 1, -1, sys.argv[2] == "1")
except RuntimeError as exc:
    if sys.argv[3] not in str(exc):
        raise
else:
    raise AssertionError("malformed table was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(int(mmap)), expected],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(params=["P", "Q"])
def hostile_vla(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    descriptor = str(request.param)
    path, _ = _vla_file(tmp_path / f"hostile-{descriptor}.fits", descriptor)
    maximum = (1 << 31) - 1 if descriptor == "P" else (1 << 63) - 1
    return _patch_descriptors(path, descriptor, [(maximum, 0), (1, 0), (1, 0)])


def _tensor_full(path: Path) -> object:
    return torchfits.table.read_torch(str(path), hdu=1, mmap=False)


def _arrow_raw(path: Path) -> object:
    return torchfits.table.read(
        str(path), hdu=1, mmap=False, decode_bytes=False, apply_fits_nulls=False
    )


def _tensor_projection(path: Path) -> object:
    return torchfits.table.read_torch(str(path), hdu=1, columns=["V"], mmap=False)


def _arrow_row_slice(path: Path) -> object:
    return torchfits.table.read(
        str(path), hdu=1, columns=["V"], row_slice=(0, 2), mmap=False
    )


@pytest.mark.parametrize("descriptor", ["P", "Q"])
def test_vla_descriptor_formats_read_normally(tmp_path: Path, descriptor: str) -> None:
    path, _ = _vla_file(tmp_path / f"valid-{descriptor}.fits", descriptor)
    result = torchfits.table.read_torch(str(path), hdu=1, mmap=False)
    assert [row.tolist() for row in result["V"]] == [[1], [2], [3]]


@pytest.mark.parametrize(
    "reader", [_tensor_full, _arrow_raw, _tensor_projection, _arrow_row_slice]
)
def test_hostile_vla_geometry_fails_before_read(
    hostile_vla: Path, reader: Callable[[Path], object]
) -> None:
    with pytest.raises(
        RuntimeError, match=r"Invalid VLA geometry for column V row [12]"
    ):
        reader(hostile_vla)


@pytest.mark.parametrize("repeat,offset", [(-1, 0), (1, -1)])
def test_negative_vla_descriptor_fields_are_rejected(
    tmp_path: Path, repeat: int, offset: int
) -> None:
    path, _ = _vla_file(tmp_path / f"negative-Q-{repeat}-{offset}.fits", "Q")
    _patch_descriptors(path, "Q", [(repeat, offset), (1, 0), (1, 0)])
    with pytest.raises(
        RuntimeError, match=r"Invalid VLA geometry for column V row 1: negative"
    ):
        cpp.read_fits_table_rows(str(path), 1, ["V"], 1, 3, False)


def test_fixed_tform_repeat_outside_supported_range_fails_at_open(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixed-repeat.fits"
    torchfits.table.write(
        str(path), {"X": np.arange(3, dtype=np.int32)}, overwrite=True
    )
    _replace_card(path, "TFORM1", "2147483648J")
    with pytest.raises(RuntimeError, match=r"Invalid table geometry"):
        cpp.read_fits_table_rows(str(path), 1, ["X"], 1, 3, False)


def test_fixed_column_storage_cannot_exceed_naxis1(tmp_path: Path) -> None:
    path = tmp_path / "fixed-row-width.fits"
    torchfits.table.write(
        str(path), {"X": np.arange(3, dtype=np.int32)}, overwrite=True
    )
    _replace_card(path, "NAXIS1", 1)
    with pytest.raises(RuntimeError, match=r"Invalid table geometry"):
        cpp.read_fits_table_rows(str(path), 1, ["X"], 1, 3, False)


def test_ascii_width_cannot_exceed_naxis1(tmp_path: Path) -> None:
    path = tmp_path / "ascii-width.fits"
    torchfits.table.write(
        str(path), {"X": np.array(["a", "b", "c"])}, overwrite=True, table_type="ascii"
    )
    _replace_card(path, "TFORM1", "A2147483647")
    with pytest.raises(RuntimeError, match=r"Invalid table geometry"):
        cpp.read_fits_table_rows(str(path), 1, ["X"], 1, 3, False)


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("damage", ["truncated", "oversized"])
@pytest.mark.parametrize("kind", ["fixed", "vla"])
def test_physical_table_extent_fails_before_allocation(
    tmp_path: Path, kind: str, damage: str, mmap: bool
) -> None:
    path = tmp_path / f"{kind}-{damage}-{mmap}.fits"
    if kind == "fixed":
        torchfits.table.write(
            str(path), {"X": np.arange(3, dtype=np.int32)}, overwrite=True
        )
    else:
        _vla_file(path, "Q")

    if damage == "truncated":
        path.write_bytes(path.read_bytes()[: _table_extent(path) - 1])
    else:
        _replace_card(path, "NAXIS2", 16_777_216)

    _assert_geometry_rejected_under_memory_limit(path, mmap, "Invalid table geometry")
