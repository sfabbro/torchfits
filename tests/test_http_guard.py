"""HTTP fetch guard: DNS-rebinding pinning, redirect re-guard, guarded migration.

Loopback fixture servers stand in for remote hosts: DNS resolution and socket
dials are mocked so the guard sees public addresses while the dial spy records
what would have been contacted (relaying only the fixture hop to the real local
server). The safety bar under test: a private address is never dialed.
"""

from __future__ import annotations

import http.server
import ipaddress
import io
import socket
import threading
import urllib.request
from pathlib import Path

import pytest

import torchfits
import torchfits.cli.cmds_copy as cmds_copy
from torchfits import http_util

GLOBAL_A = "93.184.216.34"  # public unicast; never actually dialed
GLOBAL_B = "8.8.8.8"  # public unicast; never actually dialed


class _RedirectHandler(http.server.BaseHTTPRequestHandler):
    target = "http://10.0.0.5/evil.fits"

    def do_GET(self):
        self.send_response(302)
        self.send_header("Location", type(self).target)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture()
def redirect_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join()


def _mock_dns_and_dial(monkeypatch, names, dialed, relay=None):
    """Script DNS answers and record (optionally relay) socket dials.

    ``names`` maps hostname -> list of IPs returned in order (last repeats).
    ``relay`` maps a scripted IP -> real ``host:port`` to actually connect.
    """
    real_gai = socket.getaddrinfo
    real_connect = socket.create_connection
    state = {}

    def fake_gai(host, port, *args, **kwargs):
        ips = names.get(host)
        if ips is None:
            return real_gai(host, port, *args, **kwargs)
        idx = state.get(host, 0)
        state[host] = idx + 1
        ip = ips[min(idx, len(ips) - 1)]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, int(port or 0)))]

    def fake_create_connection(address, *args, **kwargs):
        host, port = address
        try:
            ip = str(ipaddress.ip_address(host))
        except ValueError:
            ip = fake_gai(host, port)[0][4][0]
        dialed.append(ip)
        target = (relay or {}).get(ip)
        if target is None:
            raise ConnectionRefusedError(f"dial spy: no route to {ip}")
        real_host, real_port = target.rsplit(":", 1)
        return real_connect((real_host, int(real_port)))

    monkeypatch.setattr(socket, "getaddrinfo", fake_gai)
    monkeypatch.setattr(socket, "create_connection", fake_create_connection)


def test_http_open_blocks_private_before_dial(monkeypatch):
    dialed = []
    _mock_dns_and_dial(monkeypatch, {}, dialed)
    with pytest.raises(http_util.HttpBlockedError, match="private"):
        http_util.http_open("http://10.0.0.5/x.fits")
    assert dialed == []


def test_connection_pins_guard_time_resolution(monkeypatch):
    """The dial target is the guard-time DNS answer; a rebound answer is never
    reached (DNS-rebinding defense)."""
    dialed = []
    _mock_dns_and_dial(
        monkeypatch, {"rebind.example.invalid": [GLOBAL_B, "127.0.0.1"]}, dialed
    )
    with pytest.raises(OSError):
        http_util.http_open("http://rebind.example.invalid:8080/x.fits")
    assert dialed == [GLOBAL_B]
    assert "127.0.0.1" not in dialed


def test_redirect_to_private_is_refused_end_to_end(monkeypatch, redirect_server):
    _RedirectHandler.target = "http://10.0.0.5/evil.fits"
    dialed = []
    _mock_dns_and_dial(
        monkeypatch,
        {"fixture.example.invalid": [GLOBAL_A]},
        dialed,
        relay={GLOBAL_A: redirect_server},
    )
    host, port = redirect_server.rsplit(":", 1)
    with pytest.raises(http_util.HttpBlockedError, match="redirect to internal"):
        http_util.http_open(f"http://fixture.example.invalid:{port}/start.fits")
    assert dialed == [GLOBAL_A]  # only the fixture hop was contacted


def test_redirect_hop_pinned_against_dns_rebinding(monkeypatch, redirect_server):
    """A redirect hop whose DNS rebounds must dial its re-guard-time address,
    never the private answer (redirect-to-private refused at the dial)."""
    _RedirectHandler.target = "http://rebind.example.invalid:8080/evil.fits"
    dialed = []
    _mock_dns_and_dial(
        monkeypatch,
        {
            "fixture.example.invalid": [GLOBAL_A],
            "rebind.example.invalid": [GLOBAL_B, "127.0.0.1"],
        },
        dialed,
        relay={GLOBAL_A: redirect_server},
    )
    host, port = redirect_server.rsplit(":", 1)
    with pytest.raises(OSError):
        http_util.http_open(f"http://fixture.example.invalid:{port}/start.fits")
    assert dialed == [GLOBAL_A, GLOBAL_B]
    assert "127.0.0.1" not in dialed


def test_copy_remote_uses_guarded_helper_not_urlretrieve(monkeypatch, tmp_path):
    def trap(*args, **kwargs):
        raise AssertionError("bare urllib.request.urlretrieve must not be used")

    monkeypatch.setattr(urllib.request, "urlretrieve", trap)

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    opened = []

    def fake_open(url, **kwargs):
        opened.append(url)
        return _Resp(b"fits-bytes")

    monkeypatch.setattr(cmds_copy, "http_open", fake_open)
    out = tmp_path / "out.fits"
    cmds_copy._copy_remote("ftp://example.invalid/x.fits", str(out))
    assert opened == ["ftp://example.invalid/x.fits"]
    assert out.read_bytes() == b"fits-bytes"


def test_no_bare_urlretrieve_call_sites_in_src():
    src_root = Path(torchfits.__file__).parent
    offenders = [
        str(py)
        for py in sorted(src_root.rglob("*.py"))
        if "urlretrieve" in py.read_text()
    ]
    assert offenders == []
