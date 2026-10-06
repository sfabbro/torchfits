"""Guards for the HTTP-subset fallback and batch-info contracts (round 2).

Three findings, all reachable from public API:

* **R2-025** -- the r4a-08 "malformed cards must trigger the full-file
  fallback, never leak raw ``KeyError``/``ValueError``" rule was enforced only
  for *prior* HDUs (``_data_nbytes``). The **matched** HDU read ``BITPIX``,
  ``NAXIS1`` and ``NAXIS2`` bare, so the very same hostile header shapes that
  ``test_subset_http_parity`` pins for position 1 escaped as ``KeyError`` /
  ``ValueError`` -- bypassing the ``(HttpRangeUnsupported,
  HttpRangeNotSatisfied)`` catch in both ``read_subset`` and
  ``SubsetReader`` and denying the caller the CFITSIO fallback.
* **R2-026** -- ``read_batch_info`` used ``os.path.exists(path)``, so every
  CFITSIO extended-syntax spelling (``file.fits[1]``, ``[0]``, ``[1:2]``)
  counted as missing while ``read``/``read_batch`` succeeded on the exact same
  string. ``cfitsio_base_path`` documents the opposite rule ("Existence
  checks must use the base file, not the filter") and seven other call sites
  in the repo follow it.
* **R2-027** -- ``SubsetReader.read_subset`` cleared ``_http_url`` /
  ``_http_meta`` *before* constructing the replacement ``cpp.SubsetReader``,
  the one step that can fail. One failed fallback left the reader with
  ``_reader is None`` and no remote URL: every later call raised
  ``AttributeError: 'NoneType' object has no attribute 'read'`` instead of the
  real error, and the HTTP route it could still have retried was gone for good.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from astropy.io import fits as afits

import torchfits
from torchfits import http_util
from torchfits._io_engine import batch as batch_mod
from torchfits._io_engine import http_subset
from torchfits._io_engine.http_subset import HttpRangeUnsupported, read_subset_http
from torchfits._io_engine.paths import cfitsio_base_path

URL = "http://example.com/remote.fits"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _header_block(cards: list[str]) -> bytes:
    out = b"".join(c.ljust(80)[:80].encode("latin-1") for c in cards)
    out += b"END".ljust(80)
    return out.ljust(2880, b" ")


def _serve_payload(monkeypatch, payload: bytes) -> None:
    """Mock http_read_range with inclusive-end Range semantics."""

    def fake(url, start, end):
        return payload[start : end + 1]

    monkeypatch.setattr(http_subset, "http_read_range", fake)


def _write_image(tmp_path, data, name="img.fits", **cards) -> str:
    path = str(tmp_path / name)
    hdu = afits.PrimaryHDU(np.asarray(data))
    for key, value in cards.items():
        hdu.header[key] = value
    hdu.writeto(path, overwrite=True)
    return path


# Hostile headers: exactly the two shapes test_subset_http_parity pins for a
# *prior* HDU, plus the missing/garbage variants of the third matched-HDU card.
_HOSTILE_HEADERS = {
    "missing_bitpix": _header_block(
        [
            "SIMPLE  =                    T",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
            "NAXIS2  =                    4",
        ]
    ),
    "garbage_bitpix": _header_block(
        [
            "SIMPLE  =                    T",
            "BITPIX  = not-a-number       ",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
            "NAXIS2  =                    4",
        ]
    ),
    "garbage_naxis1": _header_block(
        [
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            "NAXIS1  = garbage-not-an-int  ",
            "NAXIS2  =                    4",
        ]
    ),
    "missing_naxis2": _header_block(
        [
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
        ]
    ),
}


# The image HDU sitting *after* whatever is being walked past, so a prior-HDU
# test has something to land on when the walk is refused.
_PRIOR_HDU_TARGET = (
    _header_block(
        [
            "XTENSION= 'IMAGE   '           ",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            "NAXIS1  =                    2",
            "NAXIS2  =                    2",
        ]
    )
    + (np.zeros((2, 2), dtype=">i2")).tobytes()
)


# ---------------------------------------------------------------------------
# R2-025 -- matched-HDU malformed cards
# ---------------------------------------------------------------------------


def _serve_hostile(monkeypatch, shape: str) -> None:
    _serve_payload(monkeypatch, _HOSTILE_HEADERS[shape])


@pytest.mark.parametrize("shape", sorted(_HOSTILE_HEADERS))
def test_read_subset_http_matched_hdu_never_leaks_key_error(monkeypatch, shape):
    """A malformed *matched* HDU is a Range-unsuitable file, not a KeyError."""
    _serve_hostile(monkeypatch, shape)
    with pytest.raises(HttpRangeUnsupported):
        read_subset_http(URL, 0, 0, 0, 2, 2)


@pytest.mark.parametrize("shape", sorted(_HOSTILE_HEADERS))
def test_prior_hdu_malformed_cards_still_typed(monkeypatch, shape):
    _serve_payload(monkeypatch, _HOSTILE_HEADERS[shape] + _PRIOR_HDU_TARGET)
    with pytest.raises(HttpRangeUnsupported):
        read_subset_http(URL, 1, 0, 0, 2, 2)


def test_prior_hdu_unsupported_bitpix_stops_the_walk(monkeypatch):
    """A well-formed but unreadable BITPIX must stop the HDU walk, not be sized.

    ``_HOSTILE_HEADERS`` covers *malformed* cards (missing, garbage). BITPIX=28
    is neither: ``int()`` accepts it, so the only thing standing between the
    walk and a guessed 4-byte element size is ``_bitpix_elem_bytes``. That
    guard had no test anywhere -- deleting its raise left 110 tests green
    across this file, test_subset_http_parity.py, test_remote_http_range.py and
    test_io_invariants.py -- because the matched-HDU branch calls
    ``_torch_dtype`` on the very next line, so the matched-HDU test cannot see
    a regression here either. R2-055.
    """
    unsupported = _header_block(
        [
            "XTENSION= 'IMAGE   '           ",
            "BITPIX  =                   28",
            "NAXIS   =                    2",
            "NAXIS1  =                    2",
            "NAXIS2  =                    2",
        ]
    )
    _serve_payload(monkeypatch, unsupported + _PRIOR_HDU_TARGET)
    with pytest.raises(HttpRangeUnsupported, match="unsupported BITPIX=28"):
        read_subset_http(URL, 1, 0, 0, 2, 2)


@pytest.mark.parametrize("shape", sorted(_HOSTILE_HEADERS))
def test_public_read_subset_reaches_full_file_fallback(monkeypatch, tmp_path, shape):
    """The caller must get CFITSIO's answer, not a bare KeyError/ValueError."""
    _serve_hostile(monkeypatch, shape)
    local = _write_image(tmp_path, np.arange(16, dtype=np.float32).reshape(4, 4))
    monkeypatch.setattr(
        "torchfits.data.remote.resolve_local_path", lambda _url, *a, **k: local
    )
    out = torchfits.read_subset(URL, 0, 1, 1, 3, 3)
    assert out.shape == (2, 2)
    assert out.flatten().tolist() == [5.0, 6.0, 9.0, 10.0]


@pytest.mark.parametrize("shape", sorted(_HOSTILE_HEADERS))
def test_open_subset_reader_reaches_full_file_fallback(monkeypatch, tmp_path, shape):
    _serve_hostile(monkeypatch, shape)
    local = _write_image(tmp_path, np.arange(16, dtype=np.float32).reshape(4, 4))
    monkeypatch.setattr(
        "torchfits.data.remote.resolve_local_path", lambda _url, *a, **k: local
    )
    with torchfits.open_subset_reader(URL, hdu=0) as reader:
        assert reader.shape == (4, 4)
        assert reader.read_subset(0, 0, 2, 2).shape == (2, 2)


def test_unsupported_bitpix_message_survives_the_wrap(monkeypatch):
    """An unsupported-but-well-formed BITPIX keeps its own diagnostic."""
    bad = _header_block(
        [
            "SIMPLE  =                    T",
            "BITPIX  =                   28",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
            "NAXIS2  =                    4",
        ]
    )
    _serve_payload(monkeypatch, bad)
    with pytest.raises(HttpRangeUnsupported, match="unsupported BITPIX=28"):
        read_subset_http(URL, 0, 0, 0, 2, 2)


# ---------------------------------------------------------------------------
# R2-026 -- read_batch_info and CFITSIO extended-syntax paths
# ---------------------------------------------------------------------------


@pytest.fixture
def mef(tmp_path):
    path = str(tmp_path / "mef.fits")
    afits.HDUList(
        [
            afits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)),
            afits.ImageHDU(
                np.arange(16, dtype=np.float32).reshape(4, 4) + 7, name="SCI"
            ),
        ]
    ).writeto(path, overwrite=True)
    return path


@pytest.mark.parametrize("filter_suffix", ["", "[0]", "[1]", "[1:2]", "[0:2]"])
def test_batch_info_counts_filter_spellings_that_read_succeeds_on(mef, filter_suffix):
    """existing_files must agree with the reader for the same string."""
    spelling = mef + filter_suffix
    info = torchfits.read_batch_info([spelling])
    assert info["num_files"] == 1
    assert info["existing_files"] == 1
    # The count is not optimistic: the same spelling really does read.
    assert len(torchfits.read_batch([spelling], hdu=0)) == 1


def test_batch_info_keeps_counting_absent_files(tmp_path, mef):
    missing = str(tmp_path / "nope.fits")
    assert torchfits.read_batch_info([missing])["existing_files"] == 0
    assert torchfits.read_batch_info([missing + "[1]"])["existing_files"] == 0


def test_batch_info_ignores_a_bracket_directory(tmp_path):
    """``/tmp/[data]/f.fits`` is not a filter (cfitsio_base_path's rule)."""
    d = tmp_path / "[data]"
    d.mkdir()
    path = _write_image(d, np.zeros((2, 2), dtype=np.float32), name="f.fits")
    # R2-054: this line used to be ``assert not cfitsio_base_path(path).endswith(...)
    # or True``, which is unconditionally true and so asserted nothing at all
    # while reading like the rule check. The count below cannot stand in for
    # it either: a bracket-blind helper returns ``.../`` for this path, and that
    # directory does exist, so ``existing_files`` still came back 1. Pin the
    # helper directly, in both directions -- a bracket directory is not a
    # filter, and a real trailing filter on that same path is.
    assert cfitsio_base_path(path) == path
    assert cfitsio_base_path(path + "[1]") == path
    assert torchfits.read_batch_info([path])["existing_files"] == 1


def test_batch_info_still_excludes_network_urls(monkeypatch):
    monkeypatch.setattr(http_util, "is_internal_url", lambda _u: False)
    monkeypatch.setattr(http_util, "_resolve_public_addrs", lambda _u: ("127.0.0.1",))
    info = torchfits.read_batch_info(
        ["https://example.test/a.fits", "https://x/b.fits[1]"]
    )
    assert info == {"num_files": 2, "existing_files": 0}


def test_batch_info_counts_the_base_file_not_the_filter(tmp_path, monkeypatch):
    """Direct unit pin on the helper actually used."""
    real = str(tmp_path / "f.fits")
    afits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32)).writeto(real, overwrite=True)
    original = batch_mod.os.path.exists
    seen: list[str] = []

    def spy(p):
        seen.append(p)
        return original(p)

    monkeypatch.setattr(batch_mod.os.path, "exists", spy)
    torchfits.read_batch_info([real + "[1]"])
    assert seen == [real], seen


# ---------------------------------------------------------------------------
# R2-027 -- a failed SubsetReader fallback must not brick the reader
# ---------------------------------------------------------------------------


class _DegradingHandler(BaseHTTPRequestHandler):
    """Serves a real FITS over Range, then degrades on demand."""

    body: bytes = b""
    serve_ranges = True
    full_body_is_garbage = False

    def log_message(self, format: str, *args) -> None:  # noqa: A002, A003
        return

    def do_GET(self) -> None:  # noqa: N802
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            if not type(self).serve_ranges:
                self.send_error(416, "Range Not Satisfiable")
                return
            start_s, end_s = rng.split("=", 1)[1].split("-", 1)
            start = int(start_s) if start_s else 0
            end = min(int(end_s), len(type(self).body) - 1)
            chunk = type(self).body[start : end + 1]
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{end}/{len(type(self).body)}"
            )
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        payload = (
            b"<html>gateway error</html>"
            if type(self).full_body_is_garbage
            else type(self).body
        )
        self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def degrading_server(tmp_path, monkeypatch):
    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = _write_image(tmp_path, data, name="big.fits")

    class Handler(_DegradingHandler):
        pass

    Handler.body = open(path, "rb").read()
    Handler.serve_ranges = True
    Handler.full_body_is_garbage = False

    monkeypatch.setattr(http_util, "is_internal_url", lambda _u: False)
    monkeypatch.setattr(http_util, "_resolve_public_addrs", lambda _u: ("127.0.0.1",))
    monkeypatch.setenv("TORCHFITS_REMOTE_CACHE", str(tmp_path / "cache"))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/data.fits"
    try:
        yield url, Handler
    finally:
        server.shutdown()
        server.server_close()


def test_failed_fallback_keeps_serving_the_same_real_error(degrading_server):
    """Attempts 2 and 3 must not degrade into a None-handle AttributeError."""
    url, Handler = degrading_server
    reader = torchfits.open_subset_reader(url, hdu=0)
    assert reader.shape == (64, 64)
    Handler.serve_ranges = False
    Handler.full_body_is_garbage = True

    errors = []
    for _ in range(3):
        with pytest.raises(Exception) as excinfo:  # noqa: PT011
            reader.read_subset(0, 0, 2, 2)
        errors.append((type(excinfo.value), str(excinfo.value)))
    assert not any(name == "AttributeError" for name, _ in errors), errors
    assert len({msg for _, msg in errors}) == 1, errors
    reader.close()


def test_failed_fallback_keeps_the_remote_route_usable(degrading_server):
    """Recovery after a transient failure must work through HTTP again."""
    url, Handler = degrading_server
    reader = torchfits.open_subset_reader(url, hdu=0)
    Handler.serve_ranges = False
    Handler.full_body_is_garbage = True
    with pytest.raises(Exception):  # noqa: PT011
        reader.read_subset(0, 0, 2, 2)

    Handler.serve_ranges = True
    Handler.full_body_is_garbage = False
    out = reader.read_subset(0, 0, 2, 2)
    assert out.shape == (2, 2)
    assert out.flatten().tolist() == [0.0, 1.0, 64.0, 65.0]
    reader.close()


def test_failed_fallback_leaves_the_reader_closable_and_sized(degrading_server):
    url, Handler = degrading_server
    reader = torchfits.open_subset_reader(url, hdu=0)
    Handler.serve_ranges = False
    Handler.full_body_is_garbage = True
    with pytest.raises(Exception):  # noqa: PT011
        reader.read_subset(0, 0, 2, 2)
    assert reader.shape == (64, 64)
    assert reader.hdu == 0
    reader.close()  # must not raise


def test_successful_fallback_still_swaps_in_the_local_reader(
    monkeypatch, tmp_path, degrading_server
):
    """The recovery path itself keeps working: a 416 with a good full body."""
    url, Handler = degrading_server
    reader = torchfits.open_subset_reader(url, hdu=0)
    Handler.serve_ranges = False  # forces the full-file route
    out = reader.read_subset(0, 0, 2, 2)
    assert out.shape == (2, 2)
    assert out.flatten().tolist() == [0.0, 1.0, 64.0, 65.0]
    # Second call now goes through the local CFITSIO reader, not HTTP.
    assert reader.read_subset(1, 1, 3, 3).flatten().tolist() == [
        65.0,
        66.0,
        129.0,
        130.0,
    ]
    reader.close()
