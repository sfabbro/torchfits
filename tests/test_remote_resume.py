"""Download state-machine tests for ``torchfits.data.remote``.

Covers the ``.partial`` -> promote -> cache transition: HTTP 416 recovery on
an over-complete resume offset, empty/short/malformed-length bodies, and
temporary-file hygiene on platforms without ``fcntl``. The SSRF guard and the
pinned fetch itself are covered in ``tests/test_remote_http_range.py`` and
``tests/test_security.py``.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

from torchfits.data import remote


@pytest.fixture(autouse=True)
def _clean_prefetch_state():
    with remote._prefetch_lock:
        threads = list(remote._prefetch_threads.values())
    for t in threads:
        t.join(timeout=1.0)
    with remote._prefetch_lock:
        remote._prefetch_threads.clear()
        remote._prefetch_errors.clear()
    yield
    with remote._prefetch_lock:
        threads = list(remote._prefetch_threads.values())
    for t in threads:
        t.join(timeout=1.0)
    with remote._prefetch_lock:
        remote._prefetch_threads.clear()
        remote._prefetch_errors.clear()


class _FakeResp:
    def __init__(self, status, headers, payload, read_error=None):
        self.status = status
        self.headers = headers
        self._payload = payload
        self._read_error = read_error

    def read(self, n: int = -1) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        data = self._payload
        self._payload = b""
        return data

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_416_on_resume_restarts_instead_of_wedging(tmp_path):
    """A complete-but-unpromoted partial must not wedge every later attempt.

    If a crash lands between the last byte written and ``.partial -> dest``,
    the next resume sends ``Range: bytes=<size>-`` which the server answers
    with 416. That is a dead resume offset: drop the partial and fetch
    cleanly instead of retrying the same doomed Range forever.
    """
    body = b"COMPLETE-PAYLOAD" * 16
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/wedged.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)
    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.write_bytes(body)  # interrupted after the last byte, before promote
    Path(str(partial) + ".meta").write_text(json.dumps({"etag": '"v1"'}))

    seen: list[dict] = []

    def fake_http_open(u, *, headers=None, timeout=None):
        seen.append(dict(headers or {}))
        if headers and "Range" in headers:
            raise urllib.error.HTTPError(
                u, 416, "Range Not Satisfiable", None, None
            )
        return _FakeResp(200, {"Content-Length": str(len(body))}, body)

    with mock.patch(
        "torchfits.data.remote.http_open", side_effect=fake_http_open
    ):
        local = remote.resolve_local_path(url, cache_dir=cache)
    assert Path(local).read_bytes() == body
    # The dead offset was attempted first, then a clean fetch followed.
    assert seen[0]["Range"] == f"bytes={len(body)}-"
    assert "Range" not in seen[1]
    assert not partial.exists()
    assert not Path(str(partial) + ".meta").exists()


def test_empty_body_is_not_promoted_to_cache(tmp_path):
    """A zero-byte response must not become the permanent cached copy.

    With connection-close framing an empty body is indistinguishable from a
    dropped connection; promoting it poisons the cache forever because
    ``resolve_local_path`` treats any existing file as complete.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/empty.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(200, {}, b"")
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        with pytest.raises(OSError, match="empty download"):
            remote._download_http(url, dest)
    assert not dest.exists()
    assert list(cache.glob("*.partial*")) == []


def test_malformed_content_length_treated_as_unverifiable(tmp_path):
    """A garbage Content-Length must not crash the download.

    ``http.client`` treats a malformed length as unknown framing; the cache
    layer must do the same and warn about unverifiable completeness.
    """
    body = b"REAL-FITS-PAYLOAD" * 32
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/badlen.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(200, {"Content-Length": "not-a-number"}, body)
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        with pytest.warns(RuntimeWarning, match="Content-Length"):
            remote._download_http(url, dest)
    assert dest.read_bytes() == body


def test_short_download_is_not_promoted(tmp_path):
    """A body shorter than Content-Length raises and never reaches the cache."""
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/short.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(200, {"Content-Length": "100"}, b"only-10b")
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        with pytest.raises(OSError, match="short download"):
            remote._download_http(url, dest)
    assert not dest.exists()


def test_body_without_content_length_promotes_with_warning(tmp_path):
    """Connection-close framing promotes, but the warning must say so."""
    body = b"UNCHECKED-BUT-USEFUL" * 16
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/nolen.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(200, {}, body)
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        with pytest.warns(RuntimeWarning, match="Content-Length"):
            remote._download_http(url, dest)
    assert dest.read_bytes() == body


def test_failed_download_leaves_no_temporaries_without_fcntl(tmp_path, monkeypatch):
    """Token-named temporaries cannot be resumed; a failed transfer must not
    leave them behind (they would accumulate forever on no-fcntl platforms).
    """
    monkeypatch.setattr(remote, "_fcntl", None)
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/reset.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(
        200,
        {"Content-Length": "4096", "ETag": '"v1"'},
        b"",
        read_error=OSError("connection reset by peer"),
    )
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        with pytest.raises(OSError, match="connection reset"):
            remote._download_http(url, dest)
    assert list(cache.glob("*.partial*")) == []
    assert list(cache.glob("*.meta*")) == []
    assert not dest.exists()


def test_failed_download_keeps_resume_state_with_fcntl(tmp_path):
    """Canonical (fcntl) partials survive a failed transfer for later resume."""
    if remote._fcntl is None:  # pragma: no cover - non-POSIX
        pytest.skip("resume state requires the canonical fcntl partial name")
    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/resume-later.fits"
    dest = remote.cache_path_for_url(url, cache_dir=cache)

    resp = _FakeResp(
        200,
        {"Content-Length": "4096", "ETag": '"v1"'},
        b"",
        read_error=OSError("connection reset by peer"),
    )
    with mock.patch("torchfits.data.remote.http_open", return_value=resp):
        with pytest.raises(OSError, match="connection reset"):
            remote._download_http(url, dest)
    partial = dest.with_suffix(dest.suffix + ".partial")
    assert partial.exists()
    assert Path(str(partial) + ".meta").read_text() == json.dumps(
        {"etag": '"v1"', "last_modified": None}
    )
    assert not dest.exists()


class _CopyState:
    barrier = threading.Barrier(2)
    content = b"VOS-PAYLOAD" * 64


class _FakeVosClient:
    def copy(self, uri, dest):
        # Force both writers to overlap: pass the entry barrier, write, and
        # stay inside copy() until the peer has written too.
        _CopyState.barrier.wait(timeout=10)
        Path(dest).write_bytes(_CopyState.content)
        time.sleep(0.2)


def test_concurrent_vos_downloads_use_private_temporaries(tmp_path, monkeypatch):
    """Two unsynchronized writers must never share one ``.partial``.

    On platforms without ``fcntl`` the cross-process lock is a no-op and
    writer isolation comes from unique per-attempt temp names; ``_download_vos``
    must honor that or concurrent downloaders corrupt/promote each other's
    in-flight file. Two threads calling ``_download_vos`` directly model two
    processes (neither the per-key thread lock nor flock applies there).
    """
    monkeypatch.setattr(remote, "_fcntl", None)
    fake_vos = types.ModuleType("vos")
    fake_vos.Client = _FakeVosClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vos", fake_vos)

    cache = tmp_path / "cache"
    cache.mkdir()
    dest = cache / "mosaic.fits"
    _CopyState.barrier = threading.Barrier(2)

    errors: list[BaseException] = []

    def worker():
        try:
            remote._download_vos("vos:alice/data/mosaic.fits", dest)
        except BaseException as exc:  # noqa: BLE001 - recorded for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert dest.read_bytes() == _CopyState.content
    assert list(cache.glob("*.partial*")) == []
