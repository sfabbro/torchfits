"""Shared HTTP(S) helpers: SSRF-safe redirects, auth env, timeouts.

Used by ``torchfits probe`` and remote Dataset/cache downloads.

Residual: ``guard_cfitsio_remote_path`` resolves DNS once; urllib/CFITSIO may
re-resolve later (TOCTOU / DNS rebinding). Public http(s)/ftp still go to
CFITSIO; private/loopback stays blocked at the guard.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping


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


def is_internal_url(url: str) -> bool:
    """True if *url*'s host resolves to any non-public address (or cannot resolve).

    Resolving with :func:`socket.getaddrinfo` and rejecting when *any* returned
    address is private/loopback/link-local/reserved/multicast/unspecified closes
    the DNS-rebinding and multi-record SSRF gaps left by a single
    ``gethostbyname`` lookup. Resolution failure is treated as internal (block).
    """
    try:
        hostname = urllib.parse.urlparse(url).hostname
    except Exception:
        return True
    if not hostname:
        return True
    try:
        infos = socket.getaddrinfo(hostname, None)
    except Exception:
        return True
    for info in infos:
        ip = str(info[4][0]).split("%", 1)[0]
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            return True
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            return True
    return False


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
        if is_internal_url(newurl):
            raise HttpBlockedError(
                f"{newurl}: redirect to internal or private networks is blocked "
                "for security reasons"
            )
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None and not self._keep_credentials(req.full_url, newurl):
            for header in self._CREDENTIAL_HEADERS:
                if header in new_req.headers:
                    del new_req.headers[header]
                new_req.remove_header(header)
        return new_req


def build_http_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(ValidatingRedirectHandler())


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
    if is_internal_url(url):
        raise HttpBlockedError(
            f"{url}: access to internal or private networks is blocked "
            "for security reasons"
        )
    merged = dict(auth_headers())
    if headers:
        merged.update(headers)
    return urllib.request.Request(url, headers=merged, method=method)


def http_open(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> Any:
    """Open *url* with SSRF-safe redirects and optional auth. Caller closes."""
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
    but the leading bytes still match). Mid-file 200 responses raise
    :class:`HttpRangeNotSatisfied` so callers can fall back to a full fetch.
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
        raise OSError(f"{url}: HTTP {exc.code}") from exc
    except Exception as exc:
        raise OSError(f"{url}: {exc}") from exc
