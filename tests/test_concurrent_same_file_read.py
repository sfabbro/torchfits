"""Concurrent reads of the same file across threads (CFITSIO §4 Option A).

Regression guard for sharing one cached ``fitsfile*`` across threads. Each read
path now opens a private per-call handle, so many threads hitting the same file
and alternating HDUs must return byte-identical data with no crash / corruption.
"""

import os
import tempfile
import threading

_LEAKED: list[str] = []


def _tracked_tmpfile() -> "object":
    fd, name = tempfile.mkstemp(suffix=".fits")
    os.close(fd)
    _LEAKED.append(name)

    class _F:
        pass

    f = _F()
    f.name = name
    return f


def teardown_module() -> None:
    for _name in _LEAKED:
        try:
            os.unlink(_name)
        except OSError:
            pass


import numpy as np  # noqa: E402
import torch  # noqa: E402

import torchfits  # noqa: E402


def _write_mef():
    """Write a small MEF: primary image + one named image extension."""
    from astropy.io import fits

    rng = np.random.default_rng(1234)
    d0 = rng.normal(0, 1, (48, 64)).astype(np.float32)
    d1 = rng.normal(0, 1, (32, 96)).astype(np.float32)
    f = _tracked_tmpfile()
    hdus = [fits.PrimaryHDU(d0), fits.ImageHDU(d1, name="SCI")]
    fits.HDUList(hdus).writeto(f.name, overwrite=True)
    return f.name, d0, d1


def test_concurrent_same_file_read():
    path, d0, d1 = _write_mef()

    # Single-threaded references (also warms SharedReadMeta).
    ref0 = torchfits.read(path, hdu=0)
    ref1 = torchfits.read(path, hdu=1)
    assert np.allclose(ref0.numpy(), d0)
    assert np.allclose(ref1.numpy(), d1)

    n_threads = 8
    iters = 25
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(tid: int) -> None:
        try:
            for i in range(iters):
                hdu = (tid + i) % 2
                # Alternate between the unified read and the tensor fast path.
                if i % 2 == 0:
                    t = torchfits.read(path, hdu=hdu)
                else:
                    t = torchfits.read_tensor(path, hdu=hdu)
                expected = ref0 if hdu == 0 else ref1
                if not torch.equal(t.cpu(), expected):
                    raise AssertionError(
                        f"thread {tid} iter {i} hdu {hdu}: data mismatch"
                    )
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent read failures: {errors[:3]}"


def test_concurrent_same_file_table_read():
    """Concurrent table reads of one file across threads must not crash/corrupt."""
    n = 500
    f = _tracked_tmpfile()
    torchfits.write(
        f.name,
        {
            "ID": np.arange(n, dtype=np.int64),
            "VAL": np.arange(n, dtype=np.float32),
        },
        overwrite=True,
    )
    ref = torchfits.table.read(f.name, hdu=1)
    assert ref.num_rows == n

    errors: list[Exception] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            for _ in range(20):
                t = torchfits.table.read(f.name, hdu=1)
                if t.num_rows != n:
                    raise AssertionError("row count mismatch")
                if t.column("ID").to_pylist()[:3] != [0, 1, 2]:
                    raise AssertionError("data corruption")
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent table read failures: {errors[:3]}"


def test_concurrent_read_then_mutate_fresh_data():
    """A reader cached on another thread must not serve stale data after mutation.

    The thread-local TableReader cache holds CFITSIO handles open READONLY;
    writers evict the file's readers process-wide before opening READWRITE
    (CFITSIO status 104 otherwise), and acquire re-verifies the file identity
    so a replace racing a release is caught as well.
    """
    f = _tracked_tmpfile()
    torchfits.table.write(
        f.name,
        data={"A": np.array([1, 2], dtype=np.int32)},
        overwrite=True,
    )

    ready = threading.Event()
    proceed = threading.Event()
    results: list[list[int]] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def reader_worker() -> None:
        try:
            for _ in range(3):
                torchfits.table.read(f.name, hdu=1)
            ready.set()
            proceed.wait(10)
            t = torchfits.table.read(f.name, hdu=1)
            with lock:
                results.append(t.column("A").to_pylist())
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    worker = threading.Thread(target=reader_worker)
    worker.start()
    assert ready.wait(10), "reader worker did not warm its cache"
    worker.join(0.2)
    assert worker.is_alive(), "reader worker exited before mutation"

    torchfits.table.append_rows(f.name, rows={"A": np.array([3], dtype=np.int32)})

    proceed.set()
    worker.join(10)
    assert not errors, f"reader worker failed: {errors[:3]}"
    assert results == [[1, 2, 3]], f"stale rows after in-place append: {results}"

    # Replace the file entirely (new inode): the still-cached reader must
    # reopen and serve the new payload.
    ready2 = threading.Event()
    proceed2 = threading.Event()
    results2: list[list[int]] = []
    errors2: list[Exception] = []

    def reader_worker2() -> None:
        try:
            for _ in range(3):
                torchfits.table.read(f.name, hdu=1)
            ready2.set()
            proceed2.wait(10)
            t = torchfits.table.read(f.name, hdu=1)
            with lock:
                results2.append(t.column("A").to_pylist())
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors2.append(exc)

    worker2 = threading.Thread(target=reader_worker2)
    worker2.start()
    assert ready2.wait(10), "second reader worker did not warm its cache"
    worker2.join(0.2)
    assert worker2.is_alive(), "second reader worker exited before mutation"

    torchfits.table.write(
        f.name,
        data={"A": np.array([7, 8, 9], dtype=np.int32)},
        overwrite=True,
    )

    proceed2.set()
    worker2.join(10)
    assert not errors2, f"second reader worker failed: {errors2[:3]}"
    assert results2 == [[7, 8, 9]], f"stale rows after replace: {results2}"


def test_invalidation_reaches_every_threads_reader_cache(tmp_path):
    """Shared-meta invalidation must drop every thread's cached reader (r8b).

    Each thread owns its thread-local TableReader cache; the writer-side
    evict_everywhere hook is not involved here. After a content replacement
    invisible to stat plus ``clear_shared_read_meta_cache``, every thread's
    next read must observe the new file generation (uid bump on the shared
    meta), never the decode stamped into its cached handle.
    """
    import importlib

    from test_reader_cache_freshness import (
        build_format_swap_pair,
        replace_bytes_invisible,
    )

    m = importlib.import_module("torchfits._C")
    path, b_bytes = build_format_swap_pair(tmp_path)

    n_threads = 4
    primed = threading.Barrier(n_threads + 1)
    proceed = threading.Barrier(n_threads + 1)
    errors: list[Exception] = []
    results: list[list[float]] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            first = m.read_fits_table_rows(path, 1, ["A"], 1, -1, False)["A"]
            if first.dtype != torch.int32 or first.tolist() != [1, 2, 3]:
                raise AssertionError(f"unexpected prime read: {first}")
            primed.wait(10)
            proceed.wait(10)
            got = m.read_fits_table_rows(path, 1, ["A"], 1, -1, False)["A"]
            with lock:
                results.append(got.tolist())
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)
            try:
                primed.abort()
                proceed.abort()
            except threading.BrokenBarrierError:
                pass

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    primed.wait(10)
    replace_bytes_invisible(path, b_bytes)
    m.clear_shared_read_meta_cache()
    proceed.wait(10)
    for t in threads:
        t.join(10)
        assert not t.is_alive(), "reader worker hung"

    assert not errors, f"concurrent invalidation failures: {errors[:3]}"
    assert results == [[7.5, 8.5, 9.5]] * n_threads, f"stale generations: {results}"


if __name__ == "__main__":
    test_concurrent_same_file_read()
    test_concurrent_same_file_table_read()
    test_concurrent_read_then_mutate_fresh_data()
    print("ok")
