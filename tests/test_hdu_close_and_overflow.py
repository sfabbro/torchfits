"""Closed TensorHDU access, prefetch errors, and NAXIS overflow."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
from unittest import mock

import numpy as np
import pytest
import torch

import torchfits
from astropy.io import fits as afits
from torchfits._hdu.tensor_hdu import TensorHDU
from torchfits.data import remote as remote_mod


def _extension_subprocess_env() -> dict[str, str]:
    """A fresh interpreter does not inherit this process's loaded libtorch.

    The extension links libc10 without an rpath. ``scripts/cibw_test.sh``
    exports the same directory for wheel tests; crash-isolation children
    need it too, or ``import torchfits._C`` dies before the code under test.
    """
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    env = os.environ.copy()
    key = (
        "DYLD_FALLBACK_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    )
    prev = env.get(key, "")
    env[key] = lib if not prev else lib + os.pathsep + prev
    return env


def test_tensor_hdu_to_tensor_raises_after_close():
    handle = mock.Mock()
    hdu = TensorHDU(file_handle=handle, hdu_index=0)
    hdu.mark_closed()
    with pytest.raises(RuntimeError, match="closed"):
        hdu.to_tensor()
    handle.close.assert_not_called()


@contextlib.contextmanager
def _patched_cpp(side_effect=None):
    """Yield a mock standing in for ``torchfits._C``.

    ``import torchfits._C`` is what binds the extension as an attribute on the
    package, and ``mock.patch`` needs that attribute to exist -- in a process
    that never imported it, a bare ``mock.patch("torchfits._C")`` raises
    AttributeError before the test body runs.
    """
    import torchfits._C  # noqa: F401  -- binds the attribute on the package

    with mock.patch("torchfits._C") as cpp:
        cpp.read_full.side_effect = (
            side_effect
            if side_effect is not None
            else (lambda *a, **k: torch.zeros(2, 2))
        )
        yield cpp


def test_close_before_read_never_reaches_cpp():
    """close-first ordering: a refused read must not touch C++ at all.

    ``to_tensor`` and ``mark_closed`` both hold ``_io_lock`` across their
    whole body, so they are mutually exclusive. Once ``mark_closed`` has
    returned, ``to_tensor`` must raise and must not have called ``read_full``.
    """
    handle = mock.Mock()
    hdu = TensorHDU(file_handle=handle, hdu_index=0)
    closed_done = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        if not closed_done.wait(10):
            errors.append(AssertionError("closer never ran"))
            return
        try:
            hdu.to_tensor()
        except RuntimeError:
            pass  # close won: the documented outcome
        except BaseException as exc:  # noqa: BLE001 — collected, asserted below
            errors.append(exc)

    def closer() -> None:
        hdu.mark_closed()
        closed_done.set()

    with _patched_cpp() as cpp:
        threads = [threading.Thread(target=reader), threading.Thread(target=closer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
            assert not t.is_alive(), "close/read race deadlocked"
        assert not errors, f"unexpected reader outcome: {errors!r}"
        assert cpp.read_full.call_count == 0, (
            "to_tensor reached C++ after mark_closed() returned: "
            f"{cpp.read_full.call_args_list}"
        )
    assert hdu._closed


def test_read_before_close_reaches_cpp_exactly_once():
    """read-first ordering: the read happens once, and close still lands.

    The reader holds the same lock ``to_tensor`` takes, so this ordering is
    pinned rather than left to the scheduler.
    """
    handle = mock.Mock()
    hdu = TensorHDU(file_handle=handle, hdu_index=0)
    read_done = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            with hdu._io_lock:
                hdu.to_tensor()
                read_done.set()
        except BaseException as exc:  # noqa: BLE001 — collected, asserted below
            errors.append(exc)
            read_done.set()

    def closer() -> None:
        if not read_done.wait(10):
            errors.append(AssertionError("reader never ran"))
            return
        hdu.mark_closed()

    with _patched_cpp() as cpp:
        threads = [threading.Thread(target=reader), threading.Thread(target=closer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
            assert not t.is_alive(), "close/read race deadlocked"
        assert cpp.read_full.call_count == 1, (
            "the pre-close read must go through C++ exactly once: "
            f"{cpp.read_full.call_args_list}"
        )
    assert not errors, f"unexpected reader outcome: {errors!r}"
    assert hdu._closed


def test_tensor_hdu_concurrent_close_does_not_call_cpp_after_close():
    """Unsynchronised close/read: no C++ read may follow ``mark_closed``.

    The ordering here is deliberately left to the scheduler, so the lock is
    exercised the way production hits it. The invariant asserted is the one
    the two ordered tests above pin: a ``read_full`` observed after
    ``mark_closed`` returned is a violation whichever interleaving made it.

    The earlier version of this test built its ``cpp`` mock inside the reader
    thread's ``with mock.patch(...)`` block, so the mock was unreachable once
    that block exited and no assertion could have inspected it; the
    ``RuntimeError`` from a correctly refused read was swallowed by the same
    handler that swallowed every other outcome. It therefore passed while C++
    was being called after close.
    """
    violations: list[str] = []

    def one_round() -> None:
        handle = mock.Mock()
        hdu = TensorHDU(file_handle=handle, hdu_index=0)
        barrier = threading.Barrier(2)
        closed = threading.Event()
        errors: list[BaseException] = []

        def fake_read_full(*_a, **_k):
            if closed.is_set():
                violations.append("read_full called after mark_closed() returned")
            return torch.zeros(2, 2)

        def reader() -> None:
            barrier.wait(10)
            try:
                hdu.to_tensor()
            except RuntimeError:
                pass  # close won: the documented outcome
            except BaseException as exc:  # noqa: BLE001 — collected below
                errors.append(exc)

        def closer() -> None:
            barrier.wait(10)
            hdu.mark_closed()
            closed.set()

        with _patched_cpp(fake_read_full) as cpp:
            assert cpp.read_full.side_effect is fake_read_full
            threads = [
                threading.Thread(target=reader),
                threading.Thread(target=closer),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)
                assert not t.is_alive(), "close/read race deadlocked"
        assert not errors, f"unexpected reader outcome: {errors!r}"
        assert hdu._closed

    for _ in range(25):
        one_round()
    assert not violations, violations


def test_no_cpp_read_starts_after_mark_closed_returns_mid_read():
    """The interleaving a scheduler almost never produces, pinned on purpose.

    ``guard_fits_path`` is called inside ``to_tensor``'s critical section,
    between the closed-check and the ``read_full`` call -- exactly the window
    in which a dropped lock would let a close slip past. Blocking there holds
    the reader in that window:

    * if ``to_tensor`` really holds ``_io_lock`` across its whole body, then
      ``mark_closed`` cannot return until the read is over, so no ``read_full``
      can start after it;
    * if the lock is dropped, the close completes while the reader sits in the
      window and the ``read_full`` that follows is a post-close read.

    Racing the two threads never reaches this: the window is a couple of
    bytecodes and the reader wins it every time, at every ``switchinterval``.
    """
    handle = mock.Mock()
    hdu = TensorHDU(file_handle=handle, hdu_index=0, source_path="pinned.fits")
    in_window = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    violations: list[str] = []
    errors: list[BaseException] = []

    def blocking_guard(_path: str) -> str:
        in_window.set()
        release.wait(10)
        return _path

    def fake_read_full(*_a, **_k):
        if closed.is_set():
            violations.append("read_full called after mark_closed() returned")
        return torch.zeros(2, 2)

    def reader() -> None:
        try:
            hdu.to_tensor()
        except RuntimeError:
            pass  # close won: the documented outcome
        except BaseException as exc:  # noqa: BLE001 — collected, asserted below
            errors.append(exc)

    def closer() -> None:
        hdu.mark_closed()
        closed.set()

    with (
        _patched_cpp(fake_read_full),
        mock.patch(
            "torchfits._io_engine.paths.guard_fits_path", side_effect=blocking_guard
        ),
    ):
        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=closer),
        ]
        threads[0].start()
        assert in_window.wait(10), "reader never reached the pinned window"
        # The reader is inside the critical section; the close must be queued
        # behind it, not run to completion.
        threads[1].start()
        assert not closed.wait(0.25), (
            "mark_closed() returned while a read was in flight -- the read "
            "critical section is not holding _io_lock"
        )
        release.set()
        for t in threads:
            t.join(10)
            assert not t.is_alive(), "close/read race deadlocked"

    assert not errors, f"unexpected reader outcome: {errors!r}"
    assert not violations, violations
    assert closed.is_set()


def test_prefetch_error_surfaces_on_resolve(tmp_path, monkeypatch):
    url = "https://example.test/missing.fits"
    key = url
    dest = tmp_path / "cache.fits"
    monkeypatch.setattr(remote_mod, "cache_path_for_url", lambda *a, **k: dest)
    monkeypatch.setattr(remote_mod, "is_remote_url", lambda p: True)
    monkeypatch.setattr(remote_mod, "is_vos_path", lambda p: False)

    with remote_mod._prefetch_lock:
        remote_mod._prefetch_errors[key] = RuntimeError("prefetch boom")

    with pytest.raises(RuntimeError, match="prefetch boom"):
        remote_mod.resolve_local_path(url, cache_dir=tmp_path)


def test_mutation_barrier_does_not_clear_global_cache():
    from torchfits._table import _mutation_coerce as coerce_mod

    with mock.patch.object(coerce_mod, "_invalidate_path_caches") as inv:
        with mock.patch("torchfits.cache.clear") as clear:
            coerce_mod._mutation_cache_barrier("/tmp/a.fits")
    inv.assert_called_once_with("/tmp/a.fits")
    clear.assert_not_called()


def test_tensor_hdu_data_after_close_raises_typed_error():
    """The .data property must report closure like to_tensor does (r6b)."""
    handle = mock.Mock()
    hdu = TensorHDU(file_handle=handle, hdu_index=0)
    _ = hdu.data
    hdu.mark_closed()
    with pytest.raises(RuntimeError, match="closed"):
        hdu.data
    # In-memory HDUs never had a handle: that stays a ValueError.
    inmem = TensorHDU(data=torch.zeros(2))
    with pytest.raises(ValueError, match="No file handle"):
        inmem.data


def test_tensor_hdu_reopens_revalidate_source_path():
    """to_tensor()/chunks() re-open source_path; the SSRF guard must run
    before CFITSIO sees the URL (r6b; cfitsio-http-ssrf re-validate-before-open)."""
    from torchfits.http_util import HttpBlockedError

    handle = mock.Mock()
    hdu = TensorHDU(
        data=None,
        header=None,
        file_handle=handle,
        hdu_index=0,
        source_path="http://127.0.0.1:1/x.fits",
    )
    with pytest.raises(HttpBlockedError):
        hdu.to_tensor()
    with pytest.raises(HttpBlockedError):
        list(hdu.chunks((2,)))
    handle.read_subset.assert_not_called()


def test_table_data_accessor_preserves_rank():
    from torchfits._hdu.table_hdu import TableDataAccessor, TableHDU

    col = torch.ones(5, 1)
    hdu = TableHDU({"COL": col})
    acc = TableDataAccessor(hdu)
    assert acc["COL"].shape == (5,)


def test_pathological_naxis_product_raises(tmp_path):
    """Absurd NAXISn values must fail before under-allocating a buffer."""
    cards = [
        f"{'SIMPLE':<8}= {'T':>20}",
        f"{'BITPIX':<8}= {-32:>20}",
        f"{'NAXIS':<8}= {2:>20}",
        f"{'NAXIS1':<8}= {2**30:>20}",
        f"{'NAXIS2':<8}= {2**30:>20}",
        "END",
    ]
    hdr = "".join(c.ljust(80) for c in cards).encode("ascii")
    hdr += b" " * ((2880 - (len(hdr) % 2880)) % 2880)
    path = tmp_path / "overflow.fits"
    path.write_bytes(hdr)
    with pytest.raises(RuntimeError, match="NAXIS product overflow"):
        torchfits.read_tensor(str(path), hdu=0)


def test_hostile_naxis_subset_reader_never_crashes(tmp_path):
    """A 2D image whose NAXIS1*NAXIS2 byte count wraps size_t arithmetic in
    the mmap length must not segfault the interpreter on the first cutout
    (r9b-02). NAXIS1=6148914691236517205, NAXIS2=3, BITPIX=8 gives exactly
    SIZE_MAX pixels, so `map_len = nbytes + page_offset` wrapped to a few
    KB and the copy read past the mapping."""
    cards = [
        f"{'SIMPLE':<8}= {'T':>20}",
        f"{'BITPIX':<8}= {8:>20}",
        f"{'NAXIS':<8}= {2:>20}",
        f"{'NAXIS1':<8}= {6148914691236517205:>20}",
        f"{'NAXIS2':<8}= {3:>20}",
        "END",
    ]
    hdr = "".join(c.ljust(80) for c in cards).encode("ascii")
    hdr += b" " * ((2880 - (len(hdr) % 2880)) % 2880)
    path = tmp_path / "hostile_naxis.fits"
    path.write_bytes(hdr + bytes(range(256)) * 11)

    code = (
        "import torch\n"
        "import numpy as np, sys\n"
        "from torchfits import _cpp\n"
        "try:\n"
        "    reader = _cpp.SubsetReader(sys.argv[1], 0)\n"
        "    out = reader.read(0, 0, 4, 3)\n"
        "except Exception as exc:\n"
        "    print('RAISED', type(exc).__name__)\n"
        "else:\n"
        "    print('OK', tuple(np.asarray(out).shape))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        timeout=120,
        env=_extension_subprocess_env(),
    )
    assert proc.returncode == 0, (
        f"subset reader crashed the interpreter (rc={proc.returncode}): "
        f"{proc.stderr.decode(errors='replace')[-400:]}"
    )
    assert b"RAISED" in proc.stdout or b"OK" in proc.stdout


def test_fitsfile_handle_thread_safe_reads_and_close(tmp_path):
    """One FITSFile handle shared across Python threads must return each
    requested HDU's data — not whatever HDU the racing CHDU cursor landed on
    — and close() must not free the handle mid-read (r9b-03). Runs in a
    subprocess so a use-after-free cannot take down the test session."""
    path = tmp_path / "three_hdu.fits"
    afits.HDUList(
        [
            afits.PrimaryHDU(data=np.zeros((8, 8), np.float32)),
            afits.ImageHDU(data=np.ones((8, 8), np.float32)),
            afits.ImageHDU(data=np.full((8, 8), 2.0, np.float32)),
        ]
    ).writeto(path)
    code = (
        "import torch\n"
        "import numpy as np, threading, sys\n"
        "from torchfits import _cpp\n"
        "fh = _cpp.open_fits_file(sys.argv[1], 'r')\n"
        "barrier = threading.Barrier(3)\n"
        "errs = []\n"
        "def worker(h):\n"
        "    barrier.wait()\n"
        "    for _ in range(500):\n"
        "        try:\n"
        "            t = np.asarray(_cpp.read_full(fh, h, True))\n"
        "        except Exception as exc:\n"
        "            errs.append(f'hdu {h}: {type(exc).__name__}: {exc}')\n"
        "            return\n"
        "        if t.shape != (8, 8) or abs(float(t.mean()) - h) > 1e-6:\n"
        "            errs.append(f'hdu {h}: mean {float(t.mean())} shape {t.shape}')\n"
        "            return\n"
        "ts = [threading.Thread(target=worker, args=(h,)) for h in (0, 1, 2)]\n"
        "for t in ts: t.start()\n"
        "for t in ts: t.join()\n"
        "fh2 = _cpp.open_fits_file(sys.argv[1], 'r')\n"
        "def closer():\n"
        "    for _ in range(200):\n"
        "        try: _cpp.read_full(fh2, 0, True)\n"
        "        except RuntimeError: pass\n"
        "ct = threading.Thread(target=closer)\n"
        "ct.start()\n"
        "fh2.close()\n"
        "ct.join()\n"
        "print('ANOMALIES', len(errs))\n"
        "for e in errs[:5]: print(' ', e)\n"
        "raise SystemExit(1 if errs else 0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        timeout=180,
        env=_extension_subprocess_env(),
    )
    assert proc.returncode == 0, (
        f"concurrent handle use raced (rc={proc.returncode}): "
        f"{proc.stdout.decode(errors='replace')[-400:]}"
        f"{proc.stderr.decode(errors='replace')[-400:]}"
    )
