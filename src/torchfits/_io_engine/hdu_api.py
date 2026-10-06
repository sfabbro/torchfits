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
    cache_lock,
    get_cached_handle,
    get_cached_hdu_type,
    header_cards_cache,
    path_signature,
    set_cached_hdu_type,
    signature_cached_get,
)
from .paths import (
    cfitsio_base_path,
    coerce_fits_path,
    guard_fits_path,
    is_cfitsio_network_url,
)

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

    # cpp.read_header keeps LONGSTRN '&' markers and detached CONTINUE cards.
    # The fast parser joins them; this fallback has to as well (r6a-07).
    from .._hdu.card import _reassemble_longstr_cards

    return _reassemble_longstr_cards(cpp.read_header(file_handle, hdu_index))


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

    path = coerce_fits_path(path)
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
    path = coerce_fits_path(path)
    sig = path_signature(path)
    cache_key = (path, "payload")
    # Signature validation (stored != current) lives in exactly one place:
    # signature_cached_get. auto_hdu_cache stores the same (signature, value)
    # shape it validates, so this used to be a fourth hand-written copy of a
    # rule that has to tell a permanently-unstattable extended-syntax path
    # from a file that no longer exists.
    cached_hdu = signature_cached_get(auto_hdu_cache, cache_key)
    if cached_hdu is not None:
        return int(cached_hdu)

    resolved = find_first_hdu(path, handle_cache_capacity=handle_cache_capacity)
    # Cache the negative answer too (no payload HDU -> 0): uncached, every call
    # on a payload-less file re-opened it and re-walked every HDU header. The
    # stored signature rotates the answer when the file is replaced.
    value = 0 if resolved is None else int(resolved)

    with cache_lock:
        auto_hdu_cache[cache_key] = (sig, value)
        auto_hdu_cache.move_to_end(cache_key)
        while len(auto_hdu_cache) > 512:
            auto_hdu_cache.popitem(last=False)
    return value


def _reassemble_longstr_cards(cards: Any) -> list[Any] | None:
    """Join LONGSTRN ``&``+CONTINUE chains in an ordered header card list.

    ``cpp.open_and_read_headers`` surfaces CONTINUE segments verbatim (the raw
    quoted field rides in the card's comment slot), so ``HDUList.fromfile``
    headers kept the ``&`` marker and detached CONTINUE cards while
    ``read_header`` joined them. Mirrors ``FastHeaderParser`` semantics: a
    string value ending in ``&`` joins the CONTINUE card(s) that follow (the
    ``&`` is chain notation and is restored verbatim when no CONTINUE
    follows). Returns a rebuilt card list, or ``None`` when nothing needs
    joining.
    """
    from ..header_parser import FastHeaderParser

    from .._hdu.card import _is_string_typed

    out: list[Any] = []
    target: int | None = None  # index of the string card CONTINUE extends
    marker: int | None = None  # index of the card with a pending trailing '&'
    changed = False

    def restore_marker() -> None:
        nonlocal marker, changed
        if marker is not None:
            card = out[marker]
            out[marker] = card._replace(value=card.value + "&")
            marker = None
            changed = True

    for card in cards:
        if card.key != "CONTINUE":
            # Any non-CONTINUE card ends the chain: the '&' is content again.
            restore_marker()
            out.append(card)
            if not isinstance(card.value, str):
                continue
            if card.value.endswith("&"):
                # A trailing '&' is the quoted-chain signal (see _hdu/card.py):
                # judge the card, not the shape of the unquoted value, so a
                # chain like KEY = '(1.5,2.5)...&' or a run of digits is still
                # joined to itself instead of to the preceding keyword.
                out[-1] = card._replace(value=card.value[:-1])
                marker = len(out) - 1
                changed = True
                target = len(out) - 1
            elif _is_string_typed(card.key, card.value):
                target = len(out) - 1
            continue
        field = (
            card.comment
            if str(card.comment).strip()
            else (card.value if isinstance(card.value, str) else "")
        )
        if target is None or not isinstance(out[target].value, str):
            out.append(card)  # orphan CONTINUE: keep verbatim
            continue
        comment_start = FastHeaderParser._find_comment_separator(field)
        segment = (field[:comment_start] if comment_start != -1 else field).strip()
        changed = True  # the CONTINUE card itself is consumed
        if not segment:
            continue
        has_marker = segment.startswith("'") and segment.endswith("&'")
        seg_value = (
            FastHeaderParser._parse_string_value(segment)
            if segment.startswith("'")
            else segment
        )
        if has_marker and isinstance(seg_value, str) and seg_value.endswith("&"):
            seg_value = seg_value[:-1]
        head = out[target]
        out[target] = head._replace(value=head.value + seg_value)
        marker = target if has_marker else None

    restore_marker()
    return out if changed else None


def open_hdulist(path: str, mode: str = "r") -> HDUList:
    """Open a FITS file for reading/writing."""
    path = coerce_fits_path(path)
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
        hdul = HDUList.fromfile(path, mode)
        # r4c-15: join LONGSTRN '&'+CONTINUE chains in place (the object is
        # shared with TensorHDU's DataView) so open() headers carry the same
        # string values read_header produces.
        for i in range(len(hdul)):
            header = hdul[i].header
            rebuilt = _reassemble_longstr_cards(header.cards)
            if rebuilt is not None:
                header.clear()
                for card in rebuilt:
                    header.append(card)
        return hdul
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
    path = coerce_fits_path(path)
    guard_fits_path(path)
    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == "auto"):
        return int(autodetect_hdu(path, 16))
    if isinstance(hdu, int):
        # Every sibling entry point (read, read_tensor, read_subset, read_table,
        # read_fallback) rejects a negative index up front. Do the same here so
        # a wrong HDU surfaces the same actionable ValueError instead of the
        # CFITSIO "could not move to HDU" RuntimeError.
        if hdu < 0:
            raise ValueError("hdu must be a non-negative integer")
        return int(hdu)
    if not isinstance(hdu, str):
        raise TypeError(f"hdu must be int, str, None, or 'auto', got {type(hdu)!r}")

    # Name resolution is the only branch that needs the torch-linked extension,
    # so the import sits here rather than at the top: read_header(path, 0) must
    # not pay a libtorch dlopen to reach the isinstance(hdu, int) branch above.
    import torchfits._C as cpp

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
    # File-level failures propagate: a missing/unreadable file must not be
    # reported as "HDU not found" after probing phantom HDUs.
    from .. import _core_api

    n_hdus = _core_api.read_num_hdus(path)
    for i in range(max(0, n_hdus)):
        try:
            keys = _core_api.read_keys(path, i, ["EXTNAME"])
        except Exception:
            continue
        if keys.get("EXTNAME") == hdu:
            return i
    raise ValueError(f"HDU '{hdu}' not found")


def read_nrows(path: str, hdu: Union[int, str, None] = 1) -> int:
    """Return table row count via CFITSIO ``fits_get_num_rows`` (no full header).

    Default ``hdu=1`` (first extension). Raises if the HDU is not a table.
    """
    from .. import _core_api

    path = coerce_fits_path(path)
    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    return _core_api.read_nrows(path, hdu_index)


def read_keys(
    path: str,
    keys: list[str] | tuple[str, ...],
    hdu: Union[int, str, None] = 0,
) -> dict[str, Any]:
    """Read named header keywords via CFITSIO ``fits_read_keyword`` (no full dump).

    Missing keys raise ``RuntimeError``. Default ``hdu=0`` matches ``read_header``.
    """
    from .. import _core_api

    if not keys:
        raise ValueError("keys must be a non-empty sequence of keyword names")
    path = coerce_fits_path(path)
    key_list = [str(k) for k in keys]
    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    return _core_api.read_keys(path, hdu_index, key_list)


def read_shape(
    path: str, hdu: Union[int, str, None] = 0
) -> tuple[int, tuple[int, ...]]:
    """Return ``(bitpix, shape)`` via CFITSIO image params (no full header).

    ``shape`` is torch / row-major order (reversed NAXISn). Default ``hdu=0``.
    """
    from .. import _core_api

    path = coerce_fits_path(path)
    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    return _core_api.read_shape(path, hdu_index)


def read_hdu_type(path: str, hdu: Union[int, str, None] = 0) -> str:
    """Return HDU type string (``IMAGE`` / ``BINARY_TABLE`` / …) without full header."""
    from .. import _core_api

    path = coerce_fits_path(path)
    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    return _core_api.read_hdu_type(path, hdu_index)


def read_num_hdus(path: str) -> int:
    """Return number of HDUs in the file (one open; no header dump)."""
    from .. import _core_api

    path = coerce_fits_path(path)
    guard_fits_path(path)
    return _core_api.read_num_hdus(path)


def read_colnames(path: str, hdu: Union[int, str, None] = 1) -> list[str]:
    """Return table column names (TTYPEn) without materializing the full header."""
    from .. import _core_api

    path = coerce_fits_path(path)
    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    return _core_api.read_colnames(path, hdu_index)


def read_extname(path: str, hdu: Union[int, str, None] = 0) -> str | None:
    """Return EXTNAME for an HDU, or None if absent."""
    try:
        return read_keys(path, ["EXTNAME"], hdu=hdu).get("EXTNAME")
    except RuntimeError:
        return None


def read_table_info(path: str, hdu: Union[int, str, None] = 1) -> dict[str, Any]:
    """One-open table metadata: ``nrows``, ``colnames``, ``tforms``."""
    from .. import _core_api

    path = coerce_fits_path(path)
    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    info = _core_api.read_table_info(path, hdu_index)
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
    from .. import _core_api

    path = coerce_fits_path(path)
    hdu_index = _resolve_hdu_index(path, hdu, autodetect_hdu=autodetect_hdu)
    sig = path_signature(path)
    cache_key = (path, hdu_index)
    # Signature validation (stored != current) lives in exactly one place:
    # signature_cached_get. header_cards_cache stores the same (signature, cards)
    # shape it validates, so this used to be a third hand-written copy of a rule
    # that has to tell a permanently-unstattable extended-syntax path from a file
    # that no longer exists -- and it used to answer for the deleted one.
    cached_cards = signature_cached_get(header_cards_cache, cache_key)
    if cached_cards is not None:
        # Fresh Header so callers can mutate without poisoning the cache.
        return Header(list(cached_cards))

    def _read_header(path: str, hdu_index: int) -> Header:
        # Both probes go through libtorchfits_core, which does not link
        # libtorch: reading a header is the operation an interactive session
        # runs first, and it should not pay a ~1 s dlopen to do it.
        fast_failure: Exception | None = None
        try:
            header_string = _core_api.read_header_string(path, hdu_index)
            if header_string:
                cards = fast_parse_header_cards(header_string)
                with cache_lock:
                    header_cards_cache[cache_key] = (sig, tuple(cards))
                    header_cards_cache.move_to_end(cache_key)
                    while len(header_cards_cache) > _HEADER_CARDS_CACHE_MAX:
                        header_cards_cache.popitem(last=False)
                return Header(cards)
        except Exception as exc:
            fast_failure = exc
        try:
            header = Header(_core_api.read_header_dict(path, hdu_index))
        except RuntimeError as exc:
            # read_header_dict propagates open/parse failures; surface them as
            # OSError so unreadable files match the documented read_header
            # contract that capability probes rely on.
            raise OSError(str(exc)) from exc
        # Warn only when the fallback actually rescued the read, i.e. when the
        # fast path was genuinely at fault. A plain user error (missing HDU,
        # unreadable file) fails on both probes, and blaming the fast path for
        # it emitted a misleading RuntimeWarning next to the real error.
        if fast_failure is not None:
            warnings.warn(
                f"get_header: fast path failed for {path!r} hdu={hdu_index}: "
                f"{fast_failure}; fell back to read_header_dict",
                RuntimeWarning,
                stacklevel=3,
            )
        return header

    return _read_header(path, hdu_index)
