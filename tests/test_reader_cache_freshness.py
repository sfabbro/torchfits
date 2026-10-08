"""Freshness contract for the thread-local ``TableReader`` cache (R8 r8b).

``torchfits._C`` keeps a per-thread LRU of ``TableReader`` handles so repeated
reads of the same (file, hdu) reuse the CFITSIO handle and pread fd. A cached
reader must never serve data decoded from a previous file generation:

* out-of-band ``os.replace`` is caught by the stat-identity check on acquire
  (dev/ino/size/mtime-ns);
* content replaced invisibly to stat (in-place same-size rewrite with mtime
  restored) can only be caught by explicit invalidation — ``clear_cache`` /
  ``clear_shared_read_meta_cache`` — which must reach the reader cache too
  via the shared-meta generation (uid) bump.
"""

from __future__ import annotations

import ctypes
import importlib
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torchfits  # noqa: E402


def _write_table(path: str, values: list[int]) -> None:
    torchfits.table.write(
        path,
        {"A": np.array(values, dtype=np.int32)},
        overwrite=True,
    )


def _read_binding(path: str, kind: str):
    """Read column A through the raw binding cache path."""
    m = importlib.import_module("torchfits._C")
    if kind == "full":
        return m.read_fits_table(path, 1, ["A"], False)["A"].tolist()
    if kind == "numpy":
        return m.read_fits_table_rows_numpy(path, 1, ["A"], 1, -1, False)["A"].tolist()
    return m.read_fits_table_rows(path, 1, ["A"], 1, -1, kind == "mmap")["A"].tolist()


def build_format_swap_pair(directory) -> tuple[str, bytes]:
    """Write table A at ``gen.fits`` and return same-length table B bytes.

    A: ``A:J`` (int32) + ``B:E`` (float32); B: ``A:E`` + ``B:J``. The files
    have identical total byte length (the TFORM card values merely swap), so B
    can replace A's content without changing any stat field the caches watch.
    Column ``A`` decodes as int32 [1, 2, 3] in A and float32 [7.5, 8.5, 9.5]
    in B — a stale reader silently returns bit-reinterpreted garbage instead.
    """
    from astropy.io import fits

    path = str(directory / "gen.fits")

    def _bytes(a_fmt, a_arr, b_fmt, b_arr) -> bytes:
        import io

        cols = [
            fits.Column(name="A", format=a_fmt, array=a_arr),
            fits.Column(name="B", format=b_fmt, array=b_arr),
        ]
        buf = io.BytesIO()
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
            buf
        )
        return buf.getvalue()

    a_bytes = _bytes(
        "J",
        np.array([1, 2, 3], dtype=np.int32),
        "E",
        np.array([0.5, 1.5, 2.5], dtype=np.float32),
    )
    b_bytes = _bytes(
        "E",
        np.array([7.5, 8.5, 9.5], dtype=np.float32),
        "J",
        np.array([4, 5, 6], dtype=np.int32),
    )
    assert len(a_bytes) == len(
        b_bytes,
    ), "pair must be byte-length identical"
    with open(path, "wb") as f:
        f.write(a_bytes)
    return path, b_bytes


def replace_bytes_invisible(path: str, new_bytes: bytes) -> None:
    """Replace file content in place, restoring every watched stat field."""
    st = os.stat(path)
    with open(path, "r+b") as f:
        f.seek(0)
        f.write(new_bytes)
        f.truncate()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    st2 = os.stat(path)
    assert (st2.st_ino, st2.st_size, st2.st_mtime_ns) == (
        st.st_ino,
        st.st_size,
        st.st_mtime_ns,
    ), "replacement must be invisible to the caches' stat identity check"


@pytest.mark.parametrize("kind", ["buffered", "mmap", "numpy", "full"])
def test_read_sees_os_replaced_file(tmp_path, kind):
    """A cached reader must drop when the path is replaced (new inode).

    Regression pin for the stat-identity check on cache acquire: open file A,
    read, replace with file B at the same path, read again.
    """
    path = str(tmp_path / "rep.fits")
    _write_table(path, [1, 2, 3])
    assert _read_binding(path, kind) == [1, 2, 3]

    tmp = str(tmp_path / "b.fits")
    _write_table(tmp, [7, 8, 9, 10])
    os.replace(tmp, path)

    assert _read_binding(path, kind) == [7, 8, 9, 10], "stale rows after os.replace"


def test_clear_cache_reaches_thread_local_reader_cache(tmp_path):
    """clear_cache must drop the thread-local reader cache (r8b, R7-CPP2).

    The content replacement below is invisible to stat (same inode, size and
    mtime), so the explicit invalidation documented for exactly this case —
    ``torchfits.cache.clear_cache`` — is the only freshness source. It clears
    ``SharedReadMeta``; the reader cache must observe that generation bump and
    refuse the cached handle instead of silently serving the old decode.
    """
    m = importlib.import_module("torchfits._C")
    path, b_bytes = build_format_swap_pair(tmp_path)

    first = m.read_fits_table_rows(path, 1, ["A"], 1, -1, False)["A"]
    assert first.dtype == torch.int32 and first.tolist() == [1, 2, 3]

    replace_bytes_invisible(path, b_bytes)
    torchfits.cache.clear_cache()

    got = m.read_fits_table_rows(path, 1, ["A"], 1, -1, False)["A"]
    assert got.dtype == torch.float32, f"stale decode dtype: {got.dtype}"
    assert got.tolist() == [7.5, 8.5, 9.5], f"stale reader cache data: {got.tolist()}"


def test_public_read_refetches_after_clear_cache(tmp_path):
    """Public read contract: after clear_cache every read re-opens (r8b)."""
    path, b_bytes = build_format_swap_pair(tmp_path)

    first = torchfits.table.read_torch(path, hdu=1)["A"]
    assert first.dtype == torch.int32 and first.tolist() == [1, 2, 3]

    replace_bytes_invisible(path, b_bytes)
    torchfits.cache.clear_cache()

    got = torchfits.table.read_torch(path, hdu=1)["A"]
    assert got.dtype == torch.float32, f"stale decode dtype: {got.dtype}"
    assert got.tolist() == [7.5, 8.5, 9.5], f"stale reader cache data: {got.tolist()}"


class _MallocInfo2(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


_LEAK_THRESHOLD_BYTES = 512 * 1024


class _MallocStatisticsDarwin(ctypes.Structure):
    """``malloc_statistics_t`` -- the prefix macOS's malloc_zone_statistics fills."""

    _fields_ = [
        ("blocks_in_use", ctypes.c_uint32),
        ("size_in_use", ctypes.c_size_t),
        ("max_size_in_use", ctypes.c_size_t),
        ("size_in_malloc_zone", ctypes.c_size_t),
    ]


def _live_alloc_bytes():
    """Return a zero-arg callable reporting live malloc bytes, per platform.

    glibc exposes exactly that as ``mallinfo2().uordblks``. macOS has no
    ``mallinfo`` at all; the equivalent is
    ``malloc_zone_statistics(malloc_default_zone()).size_in_use``.

    The Darwin reading is documented as a *sampled* estimate in general, so its
    noise was measured rather than assumed before being trusted: 12 consecutive
    readings with no work between them, and 6 on each side of a 20,000-read run,
    all agreed to the byte (spread 0) on macOS arm64. A platform with neither
    probe skips here rather than silently measuring nothing.
    """
    libc = ctypes.CDLL(None)
    if hasattr(libc, "mallinfo2"):
        libc.mallinfo2.restype = _MallocInfo2
        return lambda: libc.mallinfo2().uordblks
    if hasattr(libc, "malloc_zone_statistics"):
        libc.malloc_default_zone.restype = ctypes.c_void_p
        libc.malloc_zone_statistics.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        libc.malloc_zone_statistics.restype = ctypes.c_void_p

        def _darwin() -> int:
            stats = _MallocStatisticsDarwin()
            libc.malloc_zone_statistics(libc.malloc_default_zone(), ctypes.byref(stats))
            return stats.size_in_use

        return _darwin
    pytest.skip(
        "no live-allocation probe (neither mallinfo2 nor malloc_zone_statistics)"
    )


def _probe_blind_reason() -> str | None:
    """None when a leaked malloc shows up in the live-byte probe.

    Pip-installed libtorch on macOS runners can replace the default malloc
    zone, so ``malloc_zone_statistics`` stays flat. A probe that reads zero
    must not be treated as "no leak".
    """
    libc = ctypes.CDLL(None)
    if not hasattr(libc, "malloc"):
        return "no malloc(3) to build a synthetic leak with"
    live = _live_alloc_bytes()
    libc.malloc.restype = ctypes.c_void_p
    libc.malloc.argtypes = [ctypes.c_size_t]
    before = live()
    for _ in range(1000):
        libc.malloc(256 * 1024)
    seen = live() - before
    if seen < 256 * 1024 * 1000 // 2:
        return (
            f"allocator stats saw {seen} bytes of a known 256 MiB leak; "
            "this build cannot guard the reader cache"
        )
    return None


def test_live_alloc_probe_detects_a_known_leak():
    """Pin the measurement itself, or the leak test below proves nothing.

    A sensitivity control in two tiers. The second tier is the one that matters:
    the regression this file guards against was a *ghost LRU list node per
    acquire* -- tens of bytes per cached read, not a megabyte. A probe that
    cannot see a 64-byte-per-iteration leak over 20k iterations would happily
    report a flat zero for exactly that bug while still detecting a gross one,
    so the control asserts the ghost-node magnitude is visible *and* that it
    exceeds the threshold the leak test asserts.

    Without this control, a probe that quietly stopped working -- a struct-layout
    change, a ``size_in_use`` that stopped tracking, a different libc -- would
    leave ``test_reader_cache_does_not_leak_per_cached_read`` reading zero and
    passing without ever having measured anything, which is precisely the
    failure mode of a guard that only runs on one platform.
    """
    blind = _probe_blind_reason()
    if blind:
        pytest.skip(blind)
    libc = ctypes.CDLL(None)
    if not hasattr(libc, "malloc"):
        pytest.skip("no malloc(3) to build a synthetic leak with")
    live = _live_alloc_bytes()
    libc.malloc.restype = ctypes.c_void_p
    libc.malloc.argtypes = [ctypes.c_size_t]

    def leak(block: int, count: int) -> int:
        before = live()
        for _ in range(count):
            libc.malloc(block)  # deliberately leaked: never free()d
        return live() - before

    # Tier 1: the probe is live at all.
    gross = leak(256 * 1024, 1000)
    assert gross >= 256 * 1024 * 1000 // 2, (
        f"probe saw only {gross} bytes of a known 256 MiB leak"
    )

    # Tier 2: the ghost-node magnitude the real regression had. 64 B is above
    # the 16-byte minimum malloc bucket granularity, so the request is served
    # as asked rather than rounded away.
    per_read, reads = 64, 20000
    ghost = leak(per_read, reads)
    assert ghost >= per_read * reads // 2, (
        f"probe saw only {ghost} bytes of a known {per_read * reads}-byte leak"
    )
    assert ghost > _LEAK_THRESHOLD_BYTES, (
        f"a ghost-node leak ({ghost} B) would not trip the "
        f"{_LEAK_THRESHOLD_BYTES} B threshold the leak test asserts"
    )


def test_reader_cache_does_not_leak_per_cached_read(tmp_path):
    """Cache bookkeeping must not accumulate state per acquire/release (r8b).

    Each cached read exercises one acquire + release. Every cache-internal
    container must return to its steady state: a per-read growth of live
    malloc bytes is a deterministic leak (the LRU list used to keep a ghost
    node per acquire). The probe is glibc's ``mallinfo2().uordblks`` or macOS's
    ``malloc_zone_statistics().size_in_use``; both are byte-exact enough that
    the slope over 20k cached reads is deterministic (see
    ``_live_alloc_bytes`` and its sensitivity control above).
    """
    blind = _probe_blind_reason()
    if blind:
        pytest.skip(blind)
    live = _live_alloc_bytes()

    m = importlib.import_module("torchfits._C")
    path = str(tmp_path / "leak.fits")
    _write_table(path, [1, 2, 3])

    def read_once() -> None:
        m.read_fits_table_rows(path, 1, ["A"], 1, -1, False)

    for _ in range(500):  # warm every per-call allocator arena
        read_once()
    before = live()
    for _ in range(20000):
        read_once()
    grew = live() - before

    assert grew < _LEAK_THRESHOLD_BYTES, (
        f"reader cache leaked {grew} live malloc bytes over 20000 cached reads"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
