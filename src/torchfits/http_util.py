"""Shared HTTP(S) helpers: SSRF-safe redirects, auth env, timeouts.

Used by ``torchfits probe`` and remote Dataset/cache downloads.

Python-side fetches validate and pin: the guard resolves the host once and the
connection dials those exact addresses (original Host/SNI preserved), so a DNS
answer cannot change between validation and dial (DNS rebinding). Redirects are
re-validated and re-pinned per hop.

Residual: ``guard_cfitsio_remote_path`` gives guard-time validation only for
CFITSIO-driver URLs (``http``/``https``/``ftp``) — CFITSIO resolves internally
and its connections re-resolve after the check (residual TOCTOU). Public
http(s)/ftp still go to CFITSIO; private/loopback stays blocked at the guard.
"""

from __future__ import annotations

import errno
import http.client
import ipaddress
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping


class HttpBlockedError(OSError):
    """Raised when a URL or redirect targets a blocked (internal) host."""


class HttpRangeNotSatisfied(OSError):
    """Raised when the server does not return a usable byte Range body."""


def _parse_http_content_range(value: str | None) -> tuple[int, int, int | None] | None:
    """Parse ``Content-Range: bytes start-end/total`` or return ``None``."""
    if not value:
        return None
    unit, separator, remainder = value.strip().partition(" ")
    span, slash, total_text = remainder.partition("/")
    start_text, dash, end_text = span.partition("-")
    if unit.lower() != "bytes" or not separator or not slash or not dash:
        return None
    try:
        start = int(start_text)
        end = int(end_text)
        total = None if total_text == "*" else int(total_text)
    except ValueError:
        return None
    if start < 0 or end < start or (total is not None and total <= end):
        return None
    return start, end, total


def http_timeout(default: float = 120.0) -> float:
    raw = os.environ.get("TORCHFITS_HTTP_TIMEOUT", "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def auth_headers() -> dict[str, str]:
    """Authorization headers from env (first non-empty wins)."""
    full = os.environ.get("TORCHFITS_HTTP_AUTHORIZATION", "").strip()
    if full:
        return {"Authorization": full}
    token = os.environ.get("TORCHFITS_HTTP_TOKEN", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _resolve_public_addrs(url: str) -> tuple[str, ...]:
    """Resolve *url*'s host once and validate every answer as public.

    Single resolution point for guard and connection pin: validation and dial
    share one ``getaddrinfo`` result, so a DNS answer cannot change between
    the two (DNS rebinding). Multi-record answers fail closed — any
    non-public entry (private/loopback/link-local/reserved/multicast/
    unspecified) blocks, as does resolution failure.
    """
    blocked = HttpBlockedError(
        f"{url}: access to internal or private networks is blocked "
        "for security reasons"
    )
    try:
        hostname = urllib.parse.urlparse(url).hostname
    except Exception:
        raise blocked from None
    if not hostname:
        raise blocked
    try:
        infos = socket.getaddrinfo(hostname, None)
    except Exception:
        raise blocked from None
    addrs: list[str] = []
    for info in infos:
        ip = str(info[4][0]).split("%", 1)[0]
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            raise blocked from None
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            raise blocked
        if ip not in addrs:
            addrs.append(ip)
    return tuple(addrs)


def is_internal_url(url: str) -> bool:
    """True if *url*'s host resolves to any non-public address (or cannot resolve).

    Thin bool facade over :func:`_resolve_public_addrs`; resolving with
    :func:`socket.getaddrinfo` and rejecting when *any* returned address is
    private/loopback/link-local/reserved/multicast/unspecified closes the
    DNS-rebinding and multi-record SSRF gaps left by a single
    ``gethostbyname`` lookup. Resolution failure is treated as internal (block).
    """
    try:
        _resolve_public_addrs(url)
    except Exception:
        return True
    return False


def _pin_addrs_for(request: urllib.request.Request) -> tuple[str, ...]:
    """Guard-time addresses for *request* (as pinned by the guard hop)."""
    pinned = getattr(request, "_torchfits_pin", None)
    if pinned:
        return tuple(pinned)
    return _resolve_public_addrs(request.full_url)


def _connect_pinned(conn: http.client.HTTPConnection) -> None:
    """Create ``conn``'s socket against its pinned addresses only.

    ``conn.host`` (used for the Host header and TLS SNI) keeps the original
    hostname; only the dial target is the guard-time resolution.
    """
    addrs = getattr(conn, "_pinned_addrs", None) or (conn.host,)
    last: OSError | None = None
    for addr in addrs:
        try:
            conn.sock = conn._create_connection(
                (addr, conn.port), conn.timeout, conn.source_address
            )
            break
        except OSError as exc:
            last = exc
    else:
        raise OSError(f"failed to connect to {conn.host}: {last}") from last
    sys.audit("http.client.connect", conn, conn.host, conn.port)
    try:
        conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError as err:
        if err.errno != errno.ENOPROTOOPT:
            raise
    if conn._tunnel_host:
        conn._tunnel()


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """``HTTPConnection`` that dials the guard-time resolved addresses only."""

    def __init__(
        self,
        host: str,
        pinned_addrs: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        self._pinned_addrs = pinned_addrs
        super().__init__(host, **kwargs)

    def connect(self) -> None:
        _connect_pinned(self)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """``HTTPSConnection`` pinned like :class:`_PinnedHTTPConnection`.

    TLS wraps with ``server_hostname`` set to the original host, so SNI and
    certificate validation still see the URL's hostname.
    """

    def __init__(
        self,
        host: str,
        pinned_addrs: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        self._pinned_addrs = pinned_addrs
        super().__init__(host, **kwargs)

    def connect(self) -> None:
        _connect_pinned(self)
        server_hostname = self._tunnel_host if self._tunnel_host else self.host
        self.sock = self._context.wrap_socket(
            self.sock, server_hostname=server_hostname
        )


def _conn_factory(
    base: Any, pinned_addrs: tuple[str, ...]
) -> Callable[..., http.client.HTTPConnection]:
    def factory(host: str, **kwargs: Any) -> http.client.HTTPConnection:
        return base(host, pinned_addrs, **kwargs)

    return factory


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req: Any) -> Any:
        return self.do_open(
            _conn_factory(_PinnedHTTPConnection, _pin_addrs_for(req)), req
        )


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: Any) -> Any:
        return self.do_open(
            _conn_factory(_PinnedHTTPSConnection, _pin_addrs_for(req)),
            req,
            context=self._context,
        )


class _PinnedFTPHandler(urllib.request.FTPHandler):
    """``FTPHandler`` that dials the guard-time resolved addresses only.

    ``ftp_open`` re-resolves the host via ``gethostbyname``; that answer is
    ignored and the pinned address is dialed instead (FTP has no Host/SNI).
    """

    def ftp_open(self, req: Any) -> Any:
        self._pinned_addrs = _pin_addrs_for(req)
        return super().ftp_open(req)

    def connect_ftp(
        self,
        user: str,
        passwd: str,
        host: str,
        port: int,
        dirs: list[str],
        timeout: Any,
    ) -> Any:
        addrs = getattr(self, "_pinned_addrs", None) or (host,)
        last: OSError | None = None
        for addr in addrs:
            try:
                return urllib.request.ftpwrapper(
                    user, passwd, addr, port, dirs, timeout, persistent=False
                )
            except OSError as exc:
                last = exc
        raise last


class ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect hop so redirects cannot reach internal hosts.

    Credential-bearing headers (``Authorization``, ``Cookie``) are stripped
    whenever the redirect leaves the original origin, so an attacker-controlled
    public target cannot collect tokens issued for the data provider.
    """

    _CREDENTIAL_HEADERS = ("Authorization", "Cookie")

    @staticmethod
    def _origin(url: str) -> tuple[str, int, str] | None:
        parts = urllib.parse.urlsplit(url)
        if not parts.hostname:
            return None
        try:
            port = parts.port
        except ValueError:
            return None
        if port is None:
            port = 443 if parts.scheme == "https" else 80
        return parts.hostname.lower(), port, parts.scheme

    def _keep_credentials(self, original_url: str, newurl: str) -> bool:
        old = self._origin(original_url)
        new = self._origin(newurl)
        if old is None or new is None:
            return False
        if old[0] != new[0]:
            # Different host: never forward credentials.
            return False
        if old[2] == new[2]:
            # Same scheme: keep credentials only on the same port.
            return old[1] == new[1]
        # Scheme change: only a plain default-port TLS upgrade keeps
        # credentials; other ports can be bound by unrelated processes.
        return old[2] == "http" and new[2] == "https" and old[1] == 80 and new[1] == 443

    def redirect_request(  # type: ignore[no-untyped-def]
        self, req, fp, code, msg, headers, newurl
    ):
        try:
            pinned = _resolve_public_addrs(newurl)
        except HttpBlockedError:
            raise HttpBlockedError(
                f"{newurl}: redirect to internal or private networks is blocked "
                "for security reasons"
            ) from None
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            if not self._keep_credentials(req.full_url, newurl):
                for header in self._CREDENTIAL_HEADERS:
                    if header in new_req.headers:
                        del new_req.headers[header]
                    new_req.remove_header(header)
            # Pin this hop to the addresses just validated (per-hop re-guard).
            new_req._torchfits_pin = pinned
        return new_req


def build_http_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        ValidatingRedirectHandler(),
        _PinnedHTTPHandler(),
        _PinnedHTTPSHandler(),
        _PinnedFTPHandler(),
    )


_CFITSIO_NETWORK_SCHEMES = ("http://", "https://", "ftp://")


def _strip_leading_cfitsio_bang(path: str) -> str:
    """Strip CFITSIO forced-overwrite ``!`` prefixes (and interstitial whitespace)."""
    s = path.lstrip()
    while s.startswith("!"):
        s = s[1:].lstrip()
    return s


def is_cfitsio_network_url(path: str) -> bool:
    """True when *path* is an ``http``/``https``/``ftp`` CFITSIO filename."""
    lowered = _strip_leading_cfitsio_bang(path).lower()
    return any(lowered.startswith(scheme) for scheme in _CFITSIO_NETWORK_SCHEMES)


def guard_cfitsio_remote_path(path: str) -> None:
    """Block private/loopback CFITSIO network URLs; leave the path unchanged.

    Public ``http``/``https``/``ftp`` URLs are allowed so CFITSIO can still open
    them via its own network drivers (and so Python Range/fetch paths can keep
    using the same URL shape). Private targets raise :class:`HttpBlockedError`
    before CFITSIO runs. Local paths and ``vos:`` / ``vault:`` are untouched.

    Guard-time validation only: CFITSIO's network drivers resolve the hostname
    themselves and their connections re-resolve after this check (residual
    TOCTOU / DNS rebinding). Python-side fetches through :func:`http_open` are
    pinned to the validated addresses instead.
    """
    candidate = _strip_leading_cfitsio_bang(str(path))
    lowered = candidate.lower()
    if not any(lowered.startswith(scheme) for scheme in _CFITSIO_NETWORK_SCHEMES):
        return
    # Hostname checks ignore a trailing CFITSIO ``[...]`` section.
    if is_internal_url(candidate):
        raise HttpBlockedError(
            f"{path}: access to internal or private networks is blocked "
            "for security reasons"
        )


def http_request(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    method: str | None = None,
) -> urllib.request.Request:
    addrs = _resolve_public_addrs(url)
    merged = dict(auth_headers())
    if headers:
        merged.update(headers)
    request = urllib.request.Request(url, headers=merged, method=method)
    # Pin the connection to the addresses validated above (DNS rebinding).
    request._torchfits_pin = addrs
    return request


def http_open(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> Any:
    """Open *url* with SSRF-safe pinned redirects and optional auth.

    Every hop (initial request and each redirect) is re-validated and the
    connection dials only the guard-time resolved addresses, with the URL's
    Host/SNI preserved. Caller closes.
    """
    request = http_request(url, headers=headers)
    opener = build_http_opener()
    return opener.open(request, timeout=http_timeout() if timeout is None else timeout)


def http_read_range(
    url: str,
    start: int,
    end_inclusive: int,
    *,
    timeout: float | None = None,
) -> bytes:
    """GET ``Range: bytes=start-end`` and return those bytes.

    Requires HTTP 206, or HTTP 200 when ``start == 0`` (server ignored Range
    but the leading bytes still match). Mid-file 200 responses and HTTP 416
    (start beyond EOF) raise :class:`HttpRangeNotSatisfied` so callers can
    fall back to a full fetch.
    """
    if end_inclusive < start:
        raise ValueError("end_inclusive must be >= start")
    want = end_inclusive - start + 1
    headers = {"Range": f"bytes={start}-{end_inclusive}"}
    try:
        with http_open(url, headers=headers, timeout=timeout) as response:
            status = getattr(response, "status", None) or response.getcode()
            data = bytes(response.read(want))
            if status == 206:
                content_range = response.headers.get("Content-Range")
                parsed = _parse_http_content_range(content_range)
                if (
                    parsed is None
                    or parsed[0] != start
                    or len(data) > parsed[1] - parsed[0] + 1
                ):
                    raise HttpRangeNotSatisfied(
                        f"{url}: invalid Content-Range {content_range!r} "
                        f"for requested start={start}"
                    )
                return data
            if status == 200 and start == 0:
                return data
            raise HttpRangeNotSatisfied(
                f"{url}: Range not satisfied (HTTP {status}, start={start})"
            )
    except HttpBlockedError:
        raise
    except HttpRangeNotSatisfied:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            raise HttpRangeNotSatisfied(
                f"{url}: Range not satisfied (HTTP 416, start={start})"
            ) from exc
        raise OSError(f"{url}: HTTP {exc.code}") from exc
    except Exception as exc:
        raise OSError(f"{url}: {exc}") from exc
