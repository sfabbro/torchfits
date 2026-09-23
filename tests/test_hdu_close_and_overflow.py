"""Closed TensorHDU access, prefetch errors, and NAXIS overflow."""

from __future__ import annotations

import subprocess
import sys
import threading
from unittest import mock

import numpy as np
import pytest
import torch

import torchfits
from astropy.io import fits as afits
from torchfits import _cpp  # noqa: E402
from torchfits._hdu.tensor_hdu import TensorHDU
from torchfits.data import remote as remote_mod


def test_tensor_hdu_to_tensor_raises_after_close():
    handle = mock.Mock()
    hdu = TensorHDU(file_handle=handle, hdu_index=0)
    hdu.mark_closed()
    with pytest.raises(RuntimeError, match="closed"):
        hdu.to_tensor()
    handle.close.assert_not_called()


def test_tensor_hdu_concurrent_close_does_not_call_cpp_after_close():
    handle = mock.Mock()
    hdu = TensorHDU(file_handle=handle, hdu_index=0)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def reader() -> None:
        barrier.wait()
        try:
            import torchfits

            if not hasattr(torchfits, "_C"):
                torchfits._C = mock.Mock()
            with mock.patch("torchfits._C") as cpp:
                cpp.read_full.side_effect = lambda *a, **k: torch.zeros(2, 2)
                try:
                    hdu.to_tensor()
                except RuntimeError:
                    pass
        except BaseException as exc:  # noqa: BLE001 — collect race outcomes
            errors.append(exc)

    def closer() -> None:
        barrier.wait()
        hdu.mark_closed()

    threads = [threading.Thread(target=reader), threading.Thread(target=closer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert hdu._closed


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
    )
    assert proc.returncode == 0, (
        f"concurrent handle use raced (rc={proc.returncode}): "
        f"{proc.stdout.decode(errors='replace')[-400:]}"
        f"{proc.stderr.decode(errors='replace')[-400:]}"
    )
