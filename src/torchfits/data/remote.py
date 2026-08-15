"""HTTP(S) / vos download cache for Dataset / make_loader prefetch.

Local paths pass through. Remote URLs are fetched into
``{TORCHFITS_CACHE_DIR}/remote/`` (or ``TORCHFITS_REMOTE_CACHE``) so DataLoader
workers read from disk. Optional background prefetch overlaps the next GET
with the current train step.

vos / vault short forms materialize via the optional ``vos`` client.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import threading
import os
import tempfile
import warnings
from pathlib import Path
from typing import Iterable

from torchfits.http_util import (
    HttpBlockedError,
    _parse_http_content_range,
    http_open,
    http_timeout,
)
from torchfits.vos_uri import is_vos_path as is_vos_path
from torchfits.vos_uri import normalize_vos_uri as normalize_vos_uri

from torchfits.cache import remote_cache_root

_REMOTE_PREFIXES = ("http://", "https://")
_prefetch_lock = threading.Lock()
_prefetch_threads: dict[str, threading.Thread] = {}
_prefetch_errors: dict[str, BaseException] = {}
_download_locks: dict[str, threading.Lock] = {}
_log = logging.getLogger(__name__)


def is_http_url(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith(_REMOTE_PREFIXES)


def is_remote_url(path: str) -> bool:
    return is_http_url(path) or is_vos_path(path)


def remote_cache_dir() -> Path:
    return remote_cache_root()


def ephemeral_scratch_dir() -> Path:
    """Return local fast scratch directory for staging (e.g. SLURM_TMPDIR or TMPDIR)."""
    for env_var in ("SLURM_TMPDIR", "TMPDIR"):
        val = os.environ.get(env_var, "").strip()
        if val:
            path = Path(val) / "torchfits_staging"
            path.mkdir(parents=True, exist_ok=True)
            return path
    fallback = Path(tempfile.gettempdir()) / "torchfits_staging"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def cleanup_downloaded_file(path: str | Path) -> bool:
    """Safely unlink a staged/downloaded temporary file."""
    try:
        p = Path(path)
        if p.is_file():
            p.unlink(missing_ok=True)
            return True
    except OSError as exc:
        _log.debug("cleanup_downloaded_file failed for %s: %s", path, exc)
    return False


def cache_path_for_url(url: str, *, cache_dir: Path | None = None) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    suffix = Path(url.split("?", 1)[0]).suffix or ".fits"
    if len(suffix) > 16:
        suffix = ".fits"
    return (cache_dir or remote_cache_dir()) / f"{digest}{suffix}"


def _download_http(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    existing = tmp.stat().st_size if tmp.is_file() else 0
    headers: dict[str, str] = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    try:
        with http_open(
            url, headers=headers or None, timeout=http_timeout()
        ) as response:
            status = getattr(response, "status", None) or response.getcode()
            content_range = _parse_http_content_range(
                response.headers.get("Content-Range")
            )
            # Server ignored Range → restart from scratch.
            append = (
                status == 206
                and existing > 0
                and content_range is not None
                and content_range[0] == existing
            )
            if status == 206 and content_range is None:
                raise OSError(f"{url}: invalid or missing Content-Range")
            if status == 206 and existing > 0 and not append:
                raise OSError(
                    f"{url}: resumed response starts at "
                    f"{None if content_range is None else content_range[0]}, "
                    f"expected {existing}"
                )
            if not append:
                existing = 0
            mode = "ab" if append else "wb"
            content_length = response.headers.get("Content-Length")
            expected = int(content_length) if content_length else None
            wrote = 0
            with open(tmp, mode) as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    wrote += len(chunk)
            if expected is not None and wrote != expected:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                raise OSError(
                    f"{url}: short download ({wrote} bytes, expected {expected})"
                )
            if status == 206 and content_range is not None:
                range_start, range_end, total = content_range
                range_size = range_end - range_start + 1
                final_size = existing + wrote
                if (
                    wrote != range_size
                    or total is None
                    or range_end + 1 != total
                    or final_size != total
                ):
                    raise OSError(
                        f"{url}: incomplete Range download "
                        f"({final_size} bytes, total={total})"
                    )
    except HttpBlockedError:
        raise
    tmp.replace(dest)
    return dest


def _download_vos(path: str, dest: Path) -> Path:
    try:
        vos = importlib.import_module("vos")
    except ImportError as exc:
        raise ImportError(
            "vos/vault paths require the optional 'vos' package (pip/pixi install vos)"
        ) from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    uri = normalize_vos_uri(path)
    client = vos.Client()
    client.copy(uri, str(tmp))
    if not tmp.is_file() or tmp.stat().st_size == 0:
        raise OSError(f"{path}: vos copy produced empty file")
    tmp.replace(dest)
    return dest


def _download(url: str, dest: Path) -> Path:
    if is_vos_path(url):
        return _download_vos(url, dest)
    return _download_http(url, dest)


def _download_once(cache_key: str, url: str, dest: Path) -> Path:
    # NOTE: process-local locks cover Dataset/prefetch threads; use lock files
    # if cross-process download-on-demand is added.
    with _prefetch_lock:
        lock = _download_locks.setdefault(cache_key, threading.Lock())
    with lock:
        if dest.is_file():
            return dest
        return _download(url, dest)


def _cleanup_cache_key(cache_key: str) -> None:
    """Remove per-key download bookkeeping after a completed transfer.

    Does NOT pop from ``_download_locks`` — those Lock objects are tiny and
    removing them creates a race window where two threads could enter
    ``_download_once`` with different locks and download the same file in
    parallel, corrupting ``.partial``.
    """
    with _prefetch_lock:
        _prefetch_threads.pop(cache_key, None)
        _prefetch_errors.pop(cache_key, None)


def resolve_local_path(
    path: str,
    *,
    cache_dir: Path | None = None,
    download: bool = True,
) -> str:
    """Return a local filesystem path for *path* (download HTTP(S)/vos if needed)."""
    if not is_remote_url(path):
        return path
    # Cache key: normalized vos URI so short and long forms share one file.
    cache_key = normalize_vos_uri(path) if is_vos_path(path) else path
    dest = cache_path_for_url(cache_key, cache_dir=cache_dir)
    if dest.is_file():
        _cleanup_cache_key(cache_key)
        return str(dest)
    # Prefetch threads are keyed by cache_key so vos:/vault: aliases share one
    # in-flight download and do not race the same ".partial".
    with _prefetch_lock:
        existing = _prefetch_threads.get(cache_key)
        prefetch_error = _prefetch_errors.pop(cache_key, None)
    if existing is not None and existing.is_alive():
        existing.join()
        with _prefetch_lock:
            prefetch_error = _prefetch_errors.pop(cache_key, None)
        if dest.is_file():
            _cleanup_cache_key(cache_key)
            return str(dest)
    if prefetch_error is not None:
        raise prefetch_error
    if not download:
        return str(dest)
    result = str(_download_once(cache_key, path, dest))
    _cleanup_cache_key(cache_key)
    return result


def prefetch_urls(urls: Iterable[str], *, cache_dir: Path | None = None) -> None:
    """Start background downloads for missing HTTP(S)/vos URLs (best-effort)."""
    for url in urls:
        if not is_remote_url(url):
            continue
        cache_key = normalize_vos_uri(url) if is_vos_path(url) else url
        dest = cache_path_for_url(cache_key, cache_dir=cache_dir)
        if dest.is_file():
            continue
        with _prefetch_lock:
            existing = _prefetch_threads.get(cache_key)
            if existing is not None and existing.is_alive():
                continue

            def _job(u: str = url, d: Path = dest, key: str = cache_key) -> None:
                try:
                    _download_once(key, u, d)
                    with _prefetch_lock:
                        _prefetch_errors.pop(key, None)
                except Exception as exc:
                    # NOTE: best-effort prefetch; resolve_local_path re-raises
                    # the stored error instead of retrying opaquely.
                    with _prefetch_lock:
                        _prefetch_errors[key] = exc
                    _log.error("prefetch failed for %s: %s", u, exc)
                    warnings.warn(
                        f"prefetch failed for {u}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            thread = threading.Thread(
                target=_job, name="torchfits-prefetch", daemon=True
            )
            _prefetch_threads[cache_key] = thread
            thread.start()
