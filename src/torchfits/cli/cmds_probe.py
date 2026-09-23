"""``torchfits probe`` — quick file/URL metadata probe."""

from __future__ import annotations

import argparse
import importlib
import math
from typing import Any

from ..header_parser import fast_parse_header_cards
from ..http_util import (
    HttpBlockedError,
    ValidatingRedirectHandler,
    http_open,
    is_internal_url,
)
from ..vos_uri import is_vos_path, normalize_vos_uri
from .cmds_info import run as info_run
from .common import (
    EXIT_OK,
    IoError,
    UsageError,
    add_emit_format_args,
    add_hdu_arg,
    emit_records,
    is_remote_path,
    resolve_emit_format,
    resolve_paths,
)

# Re-exports for tests that import private names from this module.
_is_internal_url = is_internal_url
_ValidatingRedirectHandler = ValidatingRedirectHandler

_DEFAULT_HEADER_BYTES = 2880 * 2
_DEFAULT_TIMEOUT = 30.0


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "probe",
        help="probe local files (like info) or remote HTTP(S)/vos header peek",
    )
    parser.add_argument("paths", nargs="*", help="FITS paths or URLs")
    parser.add_argument(
        "--stdin", action="store_true", help="read paths from stdin (one per line)"
    )
    add_hdu_arg(
        parser,
        help="comma-separated HDU indices for local files only (ignored for remote)",
    )
    parser.add_argument(
        "--header-bytes",
        type=int,
        default=_DEFAULT_HEADER_BYTES,
        dest="header_bytes",
        help=(
            f"bytes to fetch for remote header peek (default: {_DEFAULT_HEADER_BYTES})"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT,
        help=f"remote HTTP timeout seconds (default: {_DEFAULT_TIMEOUT})",
    )
    add_emit_format_args(parser)
    parser.set_defaults(func=run)


def _probe_vos(path: str, *, header_bytes: int) -> dict[str, Any]:
    """Header probe via optional ``vos`` client (CANFAR VOSpace)."""
    # importlib: optional dep; avoids mypy import-not-found vs import-untyped flip-flop
    try:
        vos = importlib.import_module("vos")
    except ImportError as exc:
        raise UsageError(
            "vos: probe requires the optional 'vos' package "
            "(pip/pixi install vos); HTTP(S) probe needs no extra dep"
        ) from exc
    uri = normalize_vos_uri(path)
    handle = None
    try:
        client = vos.Client()
        handle = client.open(uri, mode="rb")
        chunk = handle.read(header_bytes)
    except Exception as exc:
        raise IoError(f"{path}: {exc}") from exc
    finally:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
    if not chunk or len(chunk) < 2880:
        raise IoError(f"{path}: could not read FITS header from VOSpace")
    cards = _cards_map(chunk[:2880].decode("latin-1", errors="replace"))
    return _remote_record(path, cards, source="vos")


def _is_http_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _cards_map(header_text: str) -> dict[str, Any]:
    return {key: value for key, value, _comment in fast_parse_header_cards(header_text)}


def _remote_record(path: str, cards: dict[str, Any], *, source: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "file": path,
        "hdu": 0,
        "simple": cards.get("SIMPLE"),
        "bitpix": cards.get("BITPIX"),
        "naxis": cards.get("NAXIS"),
        "source": source,
    }
    try:
        naxis = int(cards.get("NAXIS", 0) or 0)
    except (TypeError, ValueError):
        naxis = 0
    record["type"] = "IMAGE" if naxis > 0 else "UNKNOWN"
    if naxis >= 1:
        record["naxis1"] = cards.get("NAXIS1")
    if naxis >= 2:
        record["naxis2"] = cards.get("NAXIS2")
    return record


def _probe_http(url: str, *, header_bytes: int, timeout: float) -> dict[str, Any]:
    try:
        with http_open(
            url,
            headers={"Range": f"bytes=0-{header_bytes - 1}"},
            timeout=timeout,
        ) as response:
            chunk = response.read(header_bytes)
    except HttpBlockedError as exc:
        raise IoError(str(exc)) from exc
    except Exception as exc:
        raise IoError(f"{url}: {exc}") from exc
    if len(chunk) < 2880:
        raise IoError(f"{url}: response too short for FITS header")
    cards = _cards_map(chunk[:2880].decode("latin-1", errors="replace"))
    return _remote_record(url, cards, source="http")


def run(args: argparse.Namespace) -> int:
    paths = resolve_paths(args.paths, use_stdin=args.stdin)
    header_bytes = int(args.header_bytes)
    if header_bytes < 2880:
        raise UsageError("--header-bytes must be >= 2880 (one FITS block)")
    timeout = float(args.timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise UsageError("--timeout must be a positive finite number of seconds")

    remote_paths = [path for path in paths if is_remote_path(path)]
    local_paths = [path for path in paths if not is_remote_path(path)]
    if remote_paths and local_paths:
        raise UsageError("mixing local paths and remote URLs is not supported")

    remote_records: list[dict[str, Any]] = []
    for path in remote_paths:
        if is_vos_path(path):
            remote_records.append(_probe_vos(path, header_bytes=header_bytes))
            continue
        if _is_http_path(path):
            remote_records.append(
                _probe_http(path, header_bytes=header_bytes, timeout=timeout)
            )
            continue
        raise IoError(f"remote paths are not supported: {path}")

    if remote_paths:
        emit_records(remote_records, format=resolve_emit_format(args))
        return EXIT_OK

    args.paths = local_paths
    return info_run(args)
