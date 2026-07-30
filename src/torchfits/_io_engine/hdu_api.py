"""HDU/header access helpers for root FITS I/O."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Callable, Optional, Union

from ..header_parser import fast_parse_header, fast_parse_header_cards
from ..hdu import HDUList, Header

from .caches import (
    _HEADER_CARDS_CACHE_MAX,
    auto_hdu_cache,
    get_cached_handle,
    get_cached_hdu_type,
    header_cards_cache,
    path_signature,
    set_cached_hdu_type,
)
from .paths import cfitsio_base_path, guard_fits_path, is_cfitsio_network_url

_log = logging.getLogger(__name__)


def read_header_fast(file_handle: Any, hdu_index: int, fast_header: bool = True) -> Any:
    """Read header using fast bulk parsing or fallback to slow method."""
    import torchfits._C as cpp

    if fast_header:
        try:
            header_string = cpp.read_header_string(file_handle, hdu_index)
            if header_string:
                return fast_parse_header(header_string)
        except (AttributeError, RuntimeError, OSError):
            pass

    return cpp.read_header(file_handle, hdu_index)


def _header_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().upper() in {"T", "TRUE", "1", "YES", "Y"}
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


def find_first_hdu(
    path: str,
    handle_cache_capacity: int = 16,
) -> Optional[int]:
    """Find first payload HDU, preferring image/compressed-image over table."""
    import torchfits._C as cpp

    file_handle, cached = get_cached_handle(path, handle_cache_capacity)
    first_table_hdu: Optional[int] = None
    try:
        num_hdus = cpp.get_num_hdus(file_handle)
        for i in range(num_hdus):
            hdu_type = get_cached_hdu_type(path, i)
            if hdu_type is None:
                try:
                    hdu_type = cpp.get_hdu_type(file_handle, i)
                    set_cached_hdu_type(path, i, hdu_type)
                except Exception:
                    hdu_type = None
            if hdu_type == "IMAGE":
                try:
                    shape = file_handle.get_shape(i)
                except Exception:
                    _log.debug(
                        "get_shape failed for %r HDU %s; skipping IMAGE candidate",
                        path,
                        i,
                        exc_info=True,
                    )
                    shape = []
                if shape and all(int(dim) > 0 for dim in shape):
                    return i
                continue

            if hdu_type in {"ASCII_TABLE", "BINARY_TABLE"}:
                try:
                    hdr = read_header_fast(file_handle, i, fast_header=True)
                except Exception:
                    hdr = {}
                zimage = _header_truthy(hdr.get("ZIMAGE"))
                has_compression_keys = any(
                    k in hdr for k in ("ZCMPTYPE", "ZBITPIX", "ZNAXIS", "ZTILE1")
                )
                if zimage or has_compression_keys:
                    return i
                if first_table_hdu is None:
                    first_table_hdu = i
    finally:
        if not cached:
            try:
                file_handle.close()
            except Exception:
                pass

    return first_table_hdu


def autodetect_hdu(path: str, handle_cache_capacity: int = 16) -> int:
    """Return the first HDU with payload, preferring image/compressed-image HDUs."""
    sig = path_signature(path)
    cache_key = (path, "payload")
    cached = auto_hdu_cache.get(cache_key)
    if cached is not None:
        cached_sig, cached_hdu = cached
        if sig is None or cached_sig is None or cached_sig == sig:
            auto_hdu_cache.move_to_end(cache_key)
            return int(cached_hdu)
        auto_hdu_cache.pop(cache_key, None)

    resolved = find_first_hdu(path, handle_cache_capacity=handle_cache_capacity)
    if resolved is None:
        return 0

    auto_hdu_cache[cache_key] = (sig, int(resolved))
    auto_hdu_cache.move_to_end(cache_key)
    while len(auto_hdu_cache) > 512:
        auto_hdu_cache.popitem(last=False)
    return int(resolved)


def open_hdulist(path: str, mode: str = "r") -> HDUList:
    """Open a FITS file for reading/writing."""
    guard_fits_path(path)
    check_path = cfitsio_base_path(path)
    # Network URLs are opened by CFITSIO itself; only local paths need exists().
    if (
        mode == "r"
        and not is_cfitsio_network_url(path)
        and not os.path.exists(check_path)
    ):
        raise FileNotFoundError(f"FITS file not found: {path}")

    try:
        return HDUList.fromfile(path, mode)
    except PermissionError:
        raise PermissionError(f"Permission denied accessing file: {path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to open FITS file '{path}': {exc}") from exc


def _resolve_hdu_index(
    path: str,
    hdu: Union[int, str, None],
    *,
    autodetect_hdu: Callable[[str, int], int],
) -> int:
    """Resolve ``hdu`` to a 0-based index (supports ``None`` / ``\"auto\"`` / EXTNAME)."""
    import torchfits._C as cpp

    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == "auto"):
        return int(autodetect_hdu(path, 16))
    if isinstance(hdu, int):
        return int(hdu)
    if not isinstance(hdu, str):
        raise TypeError(f"hdu must be int, str, None, or 'auto', got {type(hdu)!r}")

    if hasattr(cpp, "resolve_hdu_name_cached"):
        try:
            return int(cpp.resolve_hdu_name_cached(path, hdu))
        except Exception as exc:
            _log.debug(
                "_resolve_hdu_index: resolve_hdu_name_cached(%r, %r) failed: %s",
                path,
                hdu,
                exc,
            )

    # Skinny fallback: probe EXTNAME only (no full header dump).
    # Missing EXTNAME (common on primary) must continue, not abort the scan.
    try:
        n_hdus = int(cpp.read_num_hdus(path))
    except Exception:
        n_hdus = 1024
    for i in range(max(0, n_hdus)):
        try:
            keys = cpp.read_keys(path, i, ["EXTNAME"])
        except Exception:
            continue
        if keys.get("EXTNAME") == hdu:
            return i
    raise ValueError(f"HDU '{hdu}' not found")


def read_nrows(path: str, hdu: Union[int, str, None] = 1) -> int:
    """Return table row count via CFITSIO ``fits_get_num_rows`` (no full header).

    Default ``hdu=1`` (first extension). Raises if the HDU is not a table.
    """
    import torchfits._C as cpp

    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    return int(cpp.read_nrows(path, hdu_index))


def read_keys(
    path: str,
    keys: list[str] | tuple[str, ...],
    hdu: Union[int, str, None] = 0,
) -> dict[str, Any]:
    """Read named header keywords via CFITSIO ``fits_read_keyword`` (no full dump).

    Missing keys raise ``RuntimeError``. Default ``hdu=0`` matches ``read_header``.
    """
    import torchfits._C as cpp

    if not keys:
        raise ValueError("keys must be a non-empty sequence of keyword names")
    key_list = [str(k) for k in keys]
    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    return dict(cpp.read_keys(path, hdu_index, key_list))


def read_shape(
    path: str, hdu: Union[int, str, None] = 0
) -> tuple[int, tuple[int, ...]]:
    """Return ``(bitpix, shape)`` via CFITSIO image params (no full header).

    ``shape`` is torch / row-major order (reversed NAXISn). Default ``hdu=0``.
    """
    import torchfits._C as cpp

    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    bitpix, shape = cpp.read_shape(path, hdu_index)
    return int(bitpix), tuple(int(d) for d in shape)


def read_hdu_type(path: str, hdu: Union[int, str, None] = 0) -> str:
    """Return HDU type string (``IMAGE`` / ``BINARY_TABLE`` / …) without full header."""
    import torchfits._C as cpp

    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    return str(cpp.read_hdu_type(path, hdu_index))


def read_num_hdus(path: str) -> int:
    """Return number of HDUs in the file (one open; no header dump)."""
    import torchfits._C as cpp

    return int(cpp.read_num_hdus(path))


def read_colnames(path: str, hdu: Union[int, str, None] = 1) -> list[str]:
    """Return table column names (TTYPEn) without materializing the full header."""
    import torchfits._C as cpp

    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    return [str(n) for n in cpp.read_colnames(path, hdu_index)]


def read_extname(path: str, hdu: Union[int, str, None] = 0) -> str | None:
    """Return EXTNAME for an HDU, or None if absent."""
    try:
        return read_keys(path, ["EXTNAME"], hdu=hdu).get("EXTNAME")
    except RuntimeError:
        return None


def read_table_info(path: str, hdu: Union[int, str, None] = 1) -> dict[str, Any]:
    """One-open table metadata: ``nrows``, ``colnames``, ``tforms``."""
    import torchfits._C as cpp

    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    info = dict(cpp.read_table_info(path, hdu_index))
    info["nrows"] = int(info["nrows"])
    info["colnames"] = [str(n) for n in info["colnames"]]
    info["tforms"] = [str(t) for t in info["tforms"]]
    return info


def get_header(
    path: str,
    hdu: Union[int, str, None] = None,
    *,
    autodetect_hdu: Callable[[str, int], int],
) -> Header:
    """Get the header of a FITS file."""
    import torchfits._C as cpp

    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    sig = path_signature(path)
    cache_key = (path, hdu_index)
    cached = header_cards_cache.get(cache_key)
    if cached is not None:
        cached_sig, cards = cached
        if sig is None or cached_sig is None or cached_sig == sig:
            header_cards_cache.move_to_end(cache_key)
            # Fresh Header so callers can mutate without poisoning the cache.
            return Header(list(cards))
        header_cards_cache.pop(cache_key, None)

    def _read_header(path: str, hdu_index: int) -> Header:
        handle = None
        try:
            handle = cpp.open_fits_file(path, "r")
            header_string = cpp.read_header_string(handle, hdu_index)
            if header_string:
                cards = fast_parse_header_cards(header_string)
                header_cards_cache[cache_key] = (sig, tuple(cards))
                header_cards_cache.move_to_end(cache_key)
                while len(header_cards_cache) > _HEADER_CARDS_CACHE_MAX:
                    header_cards_cache.popitem(last=False)
                return Header(cards)
        except Exception as exc:
            warnings.warn(
                f"get_header: fast path failed for {path!r} hdu={hdu_index}: {exc}; "
                "falling back to read_header_dict",
                RuntimeWarning,
                stacklevel=3,
            )
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception as exc:
                    _log.debug("get_header: handle close failed: %s", exc)
        return Header(cpp.read_header_dict(path, hdu_index))

    return _read_header(path, hdu_index)
