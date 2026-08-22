"""Remote HTTP Range / vos URI tests (local fixtures only; no personal vault paths)."""

from __future__ import annotations

import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from astropy.io import fits

from torchfits import http_util
from torchfits.vos_uri import is_vos_path, normalize_vos_uri


@pytest.fixture
def allow_loopback(monkeypatch):
    monkeypatch.setattr(http_util, "is_internal_url", lambda _url: False)


class _CountingHandler(BaseHTTPRequestHandler):
    """Serve one FITS body; record Range / Auth / bytes transferred."""

    body: bytes = b""
    redirected_from: str | None = None
    require_auth: str | None = None
    transferred: int = 0
    last_range: str | None = None
    last_auth: str | None = None

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        type(self).last_range = self.headers.get("Range")
        type(self).last_auth = self.headers.get("Authorization")
        if self.require_auth and self.last_auth != self.require_auth:
            self.send_error(401, "auth required")
            return
        if self.redirected_from and self.path == self.redirected_from:
            self.send_response(302)
            self.send_header("Location", "/data.fits")
            self.end_headers()
            return
        body = self.body
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng.split("=", 1)[1]
            start_s, end_s = spec.split("-", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else len(body) - 1
            end = min(end, len(body) - 1)
            chunk = body[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            type(self).transferred += len(chunk)
            return
        self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        type(self).transferred += len(body)


@pytest.fixture
def http_fits_server(tmp_path, allow_loopback):
    path = tmp_path / "img.fits"
    # Large enough that header peek + row-band << full file.
    data = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    fits.PrimaryHDU(data).writeto(str(path), overwrite=True)
    body = path.read_bytes()

    class Handler(_CountingHandler):
        pass

    Handler.body = body
    Handler.transferred = 0
    Handler.last_range = None
    Handler.last_auth = None
    Handler.redirected_from = None
    Handler.require_auth = None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/data.fits"
    try:
        yield url, Handler, body
    finally:
        server.shutdown()


def test_normalize_vos_uri_placeholders() -> None:
    assert normalize_vos_uri("vos:alice/weights/x.fits") == (
        "vos://cadc.nrc.ca~vault/alice/weights/x.fits"
    )
    assert normalize_vos_uri("vault:alice/y.fits") == (
        "vos://cadc.nrc.ca~vault/alice/y.fits"
    )
    full = "vos://cadc.nrc.ca~vault/alice/z.fits"
    assert normalize_vos_uri(full) == full
    assert is_vos_path("vos:alice/a.fits")
    assert is_vos_path("vault:alice/a.fits")
    assert not is_vos_path("https://example.edu/a.fits")


def test_redirect_handler_strips_credentials_cross_origin(allow_loopback) -> None:
    """Tokens must not follow redirects to a different origin."""
    handler = http_util.ValidatingRedirectHandler()

    def request_with_token(url: str) -> urllib.request.Request:
        return urllib.request.Request(
            url, headers={"Authorization": "Bearer s3cret", "Cookie": "k=v"}
        )

    # Cross-host: stripped.
    req = request_with_token("http://data.example.edu/x.fits")
    new_req = handler.redirect_request(
        req, None, 302, "x", {}, "https://attacker.example/y"
    )
    assert new_req is not None
    assert "Authorization" not in new_req.headers
    assert new_req.headers.get("Authorization") is None

    # Scheme downgrade on same host: stripped.
    req = request_with_token("https://data.example.edu/x.fits")
    new_req = handler.redirect_request(
        req, None, 302, "x", {}, "http://data.example.edu/y"
    )
    assert new_req is not None
    assert new_req.headers.get("Authorization") is None

    # Same origin: kept (standard http->https upgrade or plain same scheme).
    req = request_with_token("http://data.example.edu/x.fits")
    new_req = handler.redirect_request(
        req, None, 302, "x", {}, "https://data.example.edu/y"
    )
    assert new_req is not None
    assert new_req.headers.get("Authorization") == "Bearer s3cret"

    # Different port: stripped.
    req = request_with_token("http://data.example.edu:8443/x.fits")
    new_req = handler.redirect_request(
        req, None, 302, "x", {}, "https://data.example.edu:9443/y"
    )
    assert new_req is not None
    assert new_req.headers.get("Authorization") is None


def test_full_download_auth_and_redirect(tmp_path, allow_loopback, monkeypatch):
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(
        str(path), overwrite=True
    )
    body = path.read_bytes()

    class Handler(_CountingHandler):
        pass

    Handler.body = body
    Handler.transferred = 0
    Handler.redirected_from = "/redir"
    Handler.require_auth = "Bearer test-token"
    Handler.last_auth = None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    monkeypatch.setenv("TORCHFITS_HTTP_TOKEN", "test-token")
    monkeypatch.setenv("TORCHFITS_REMOTE_CACHE", str(tmp_path / "cache"))
    from torchfits.data.remote import resolve_local_path

    try:
        local = resolve_local_path(f"http://127.0.0.1:{port}/redir")
        assert Path(local).read_bytes() == body
        assert Handler.last_auth == "Bearer test-token"
        assert Handler.transferred == len(body)
    finally:
        server.shutdown()


def test_range_cutout_transfers_row_band_only(http_fits_server):
    url, Handler, body = http_fits_server
    from torchfits import read_subset

    Handler.transferred = 0
    out = read_subset(url, 0, 2, 1, 6, 4)
    assert out.shape == (3, 4)
    expected = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)[1:4, 2:6]
    np.testing.assert_array_equal(out.numpy(), expected)
    # Ensure the returned tensor uses native byte order (from FITS big-endian)
    import sys

    sys_byteorder = "<" if sys.byteorder == "little" else ">"
    assert (
        out.numpy().dtype.byteorder == "="
        or out.numpy().dtype.byteorder == sys_byteorder
    )
    assert Handler.transferred < len(body) // 4


def test_compressed_remote_falls_back_to_full_cache(
    tmp_path, allow_loopback, monkeypatch
):
    path = tmp_path / "comp.fits"
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    # CompImageHDU alone may need a primary; use HDUList.
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data)]).writeto(
        str(path), overwrite=True
    )
    body = path.read_bytes()

    class Handler(_CountingHandler):
        pass

    Handler.body = body
    Handler.transferred = 0
    Handler.redirected_from = None
    Handler.require_auth = None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    monkeypatch.setenv("TORCHFITS_REMOTE_CACHE", str(tmp_path / "cache"))
    from torchfits import read_subset

    try:
        out = read_subset(f"http://127.0.0.1:{port}/data.fits", 1, 0, 0, 2, 2)
        assert out.shape == (2, 2)
        assert Handler.transferred >= len(body)
    finally:
        server.shutdown()


def test_range_ignored_falls_back_to_full_cache(tmp_path, allow_loopback, monkeypatch):
    """Servers that answer Range with HTTP 200 must not yield wrong pixels."""
    path = tmp_path / "img.fits"
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    fits.PrimaryHDU(data).writeto(str(path), overwrite=True)
    body = path.read_bytes()

    class Handler(_CountingHandler):
        def do_GET(self) -> None:  # noqa: N802
            type(self).last_range = self.headers.get("Range")
            type(self).last_auth = self.headers.get("Authorization")
            # Ignore Range entirely — always full 200 body.
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.body)))
            self.end_headers()
            self.wfile.write(self.body)
            type(self).transferred += len(self.body)

    Handler.body = body
    Handler.transferred = 0
    Handler.redirected_from = None
    Handler.require_auth = None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    monkeypatch.setenv("TORCHFITS_REMOTE_CACHE", str(tmp_path / "cache"))
    from torchfits import read_subset

    try:
        out = read_subset(f"http://127.0.0.1:{port}/data.fits", 0, 2, 1, 6, 4)
        assert out.shape == (3, 4)
        np.testing.assert_array_equal(out.numpy(), data[1:4, 2:6])
    finally:
        server.shutdown()


def test_download_resume_appends_partial(tmp_path, allow_loopback, monkeypatch):
    path = tmp_path / "img.fits"
    fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32)).writeto(
        str(path), overwrite=True
    )
    body = path.read_bytes()
    cache = tmp_path / "cache"
    cache.mkdir()
    from torchfits.data.remote import cache_path_for_url, resolve_local_path

    url = "http://127.0.0.1:9/resume.fits"
    dest = cache_path_for_url(url, cache_dir=cache)
    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.write_bytes(body[:10])

    class _Resp:
        status = 206
        headers = {
            "Content-Length": str(len(body) - 10),
            "Content-Range": f"bytes 10-{len(body) - 1}/{len(body)}",
        }
        _payload: bytes

        def __init__(self) -> None:
            self._payload = body[10:]

        def read(self, n: int = -1) -> bytes:
            data = self._payload
            self._payload = b""
            return data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 206

    monkeypatch.setenv("TORCHFITS_REMOTE_CACHE", str(cache))
    with mock.patch("torchfits.data.remote.http_open", return_value=_Resp()):
        local = resolve_local_path(url, cache_dir=cache)
    assert Path(local).read_bytes() == body


def test_http_read_range_rejects_wrong_content_range(monkeypatch):
    class _Resp:
        status = 206
        headers = {"Content-Range": "bytes 0-3/100"}

        def read(self, n: int = -1) -> bytes:
            return b"bad!"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 206

    monkeypatch.setattr(http_util, "http_open", lambda *args, **kwargs: _Resp())
    with pytest.raises(http_util.HttpRangeNotSatisfied, match="Content-Range"):
        http_util.http_read_range("https://example.test/data.fits", 10, 13)


def test_download_resume_rejects_incomplete_206(tmp_path, monkeypatch):
    from torchfits.data.remote import cache_path_for_url, resolve_local_path

    cache = tmp_path / "cache"
    cache.mkdir()
    url = "https://example.test/incomplete.fits"
    dest = cache_path_for_url(url, cache_dir=cache)
    partial = dest.with_suffix(dest.suffix + ".partial")
    partial.write_bytes(b"a" * 10)

    class _Resp:
        status = 206
        headers = {
            "Content-Length": "10",
            "Content-Range": "bytes 10-19/30",
        }

        def __init__(self) -> None:
            self._payload = b"b" * 10

        def read(self, n: int = -1) -> bytes:
            data = self._payload
            self._payload = b""
            return data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 206

    with mock.patch("torchfits.data.remote.http_open", return_value=_Resp()):
        with pytest.raises(OSError, match="incomplete Range download"):
            resolve_local_path(url, cache_dir=cache)
    assert not dest.exists()
    assert partial.read_bytes() == b"a" * 10 + b"b" * 10
