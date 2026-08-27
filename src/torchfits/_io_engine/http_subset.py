"""HTTP Range cutouts for uncompressed 2D FITS images.

Compressed / scaled HDUs raise :class:`HttpRangeUnsupported` so callers can
fall back to full-file cache + CFITSIO.
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from torch import Tensor

from torchfits.header_parser import fast_parse_header_cards
from torchfits.http_util import http_read_range

_BLOCK = 2880
_HEADER_PEEK = _BLOCK * 2  # enough for typical image headers
_MAX_HDU_WALK = 512


class HttpRangeUnsupported(RuntimeError):
    """Remote HDU cannot use Range cutout; materialize the full file instead."""


def _bitpix_elem_bytes(bitpix: int) -> int:
    mapping = {8: 1, 16: 2, 32: 4, 64: 8, -32: 4, -64: 8}
    try:
        return mapping[bitpix]
    except KeyError as exc:
        raise HttpRangeUnsupported(f"unsupported BITPIX={bitpix}") from exc


def _torch_dtype(bitpix: int) -> torch.dtype:
    mapping = {
        8: torch.uint8,
        16: torch.int16,
        32: torch.int32,
        64: torch.int64,
        -32: torch.float32,
        -64: torch.float64,
    }
    try:
        return mapping[bitpix]
    except KeyError as exc:
        raise HttpRangeUnsupported(f"unsupported BITPIX={bitpix}") from exc


def _cards_dict(header_bytes: bytes) -> dict[str, Any]:
    text = header_bytes.decode("latin-1", errors="replace")
    return {key: value for key, value, _c in fast_parse_header_cards(text)}


def _parse_header_at(url: str, offset: int) -> tuple[dict[str, Any], int]:
    """Fetch and parse the HDU header starting at absolute *offset*.

    Returns (cards, data_start_absolute).
    """
    peek = http_read_range(url, offset, offset + _HEADER_PEEK - 1)
    if len(peek) < _BLOCK:
        raise HttpRangeUnsupported("header peek too short")
    pos = 0
    end_at: int | None = None
    # Grow peeks until END (rare long headers).
    while True:
        while pos + 80 <= len(peek):
            card = peek[pos : pos + 80]
            if card.startswith(b"END") and card[3:8].strip() == b"":
                end_at = pos + 80
                break
            pos += 80
        if end_at is not None:
            break
        if len(peek) >= _BLOCK * 64:
            raise HttpRangeUnsupported("HDU header too large for Range scan")
        more = http_read_range(url, offset, offset + len(peek) + _BLOCK * 4 - 1)
        if len(more) <= len(peek):
            raise HttpRangeUnsupported("END card not found in Range scan window")
        peek = more
    header_nbytes = ((end_at + _BLOCK - 1) // _BLOCK) * _BLOCK
    if header_nbytes > len(peek):
        peek = http_read_range(url, offset, offset + header_nbytes - 1)
    cards = _cards_dict(peek[:header_nbytes])
    return cards, offset + header_nbytes


def _data_nbytes(cards: dict[str, Any]) -> int:
    try:
        naxis = int(cards.get("NAXIS", 0) or 0)
    except (TypeError, ValueError):
        naxis = 0
    if naxis <= 0:
        return 0
    bitpix = int(cards["BITPIX"])
    elem = _bitpix_elem_bytes(bitpix)
    n = 1
    for i in range(1, naxis + 1):
        key = f"NAXIS{i}"
        if key not in cards:
            return 0
        n *= int(cards[key])
    return n * elem


def _padded_data_nbytes(raw: int) -> int:
    if raw <= 0:
        return 0
    return ((raw + _BLOCK - 1) // _BLOCK) * _BLOCK


def _is_compressed(cards: dict[str, Any]) -> bool:
    if cards.get("ZIMAGE") in (True, "T", "t"):
        return True
    xtension = str(cards.get("XTENSION", "") or "").strip().upper()
    if xtension == "BINTABLE" and (
        "ZCMPTYPE" in cards or "ZBITPIX" in cards or "ZNAXIS" in cards
    ):
        return True
    return False


def _is_scaled(cards: dict[str, Any]) -> bool:
    if "BLANK" in cards:
        # Identity BSCALE/BZERO + BLANK still needs CFITSIO nulval; Range
        # copies would return the sentinel code.
        return True
    bscale = cards.get("BSCALE", 1.0)
    bzero = cards.get("BZERO", 0.0)
    try:
        bs = float(bscale)
        bz = float(bzero)
    except (TypeError, ValueError):
        return True
    return bs != 1.0 or bz != 0.0


def _match_hdu(cards: dict[str, Any], hdu: int | str, index: int) -> bool:
    if isinstance(hdu, int):
        return index == hdu
    name = str(cards.get("EXTNAME", "") or "").strip().upper()
    return name == str(hdu).strip().upper()


def locate_uncompressed_2d(url: str, hdu: int | str) -> dict[str, Any]:
    """Locate an uncompressed 2D image HDU; raise if Range cutout is unsuitable."""
    offset = 0
    for index in range(_MAX_HDU_WALK):
        cards, data_start = _parse_header_at(url, offset)
        if _match_hdu(cards, hdu, index):
            if _is_compressed(cards):
                raise HttpRangeUnsupported("compressed image HDU")
            if _is_scaled(cards):
                raise HttpRangeUnsupported("scaled image HDU")
            try:
                naxis = int(cards.get("NAXIS", 0) or 0)
            except (TypeError, ValueError) as exc:
                raise HttpRangeUnsupported("bad NAXIS") from exc
            if naxis != 2:
                raise HttpRangeUnsupported(f"NAXIS={naxis} (need 2)")
            bitpix = int(cards["BITPIX"])
            return {
                "data_offset": data_start,
                "bitpix": bitpix,
                "naxis1": int(cards["NAXIS1"]),
                "naxis2": int(cards["NAXIS2"]),
                "elem_bytes": _bitpix_elem_bytes(bitpix),
                "dtype": _torch_dtype(bitpix),
            }
        raw = _data_nbytes(cards)
        offset = data_start + _padded_data_nbytes(raw)
    raise HttpRangeUnsupported(f"HDU {hdu!r} not found in Range scan")


def read_subset_http(
    url: str,
    hdu: int | str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Tensor:
    """Range-fetch one row-band cutout from an uncompressed 2D HTTP(S) image."""
    meta = locate_uncompressed_2d(url, hdu)
    naxis1 = int(meta["naxis1"])
    naxis2 = int(meta["naxis2"])
    elem = int(meta["elem_bytes"])
    data_offset = int(meta["data_offset"])
    dtype = meta["dtype"]

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(naxis1, int(x2))
    y2 = min(naxis2, int(y2))
    if x2 <= x1 or y2 <= y1:
        return torch.empty((max(0, y2 - y1), max(0, x2 - x1)), dtype=dtype)

    row_bytes = naxis1 * elem
    start = data_offset + y1 * row_bytes
    end = data_offset + y2 * row_bytes - 1
    raw = http_read_range(url, start, end)
    expected = (y2 - y1) * row_bytes
    if len(raw) < expected:
        raise OSError(
            f"{url}: short Range body ({len(raw)} bytes, expected {expected})"
        )
    buf = bytearray(raw[:expected])
    # FITS multi-byte values are big-endian; swap only on little-endian hosts.
    full = torch.frombuffer(buf, dtype=dtype).reshape(y2 - y1, naxis1)
    if elem > 1 and sys.byteorder == "little":
        full.numpy().byteswap(True)

    return full[:, x1:x2].contiguous()
