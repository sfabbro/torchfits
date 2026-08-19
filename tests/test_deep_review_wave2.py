"""Wave-2 deep-review safety regressions."""

from __future__ import annotations

import threading
from unittest import mock

import pytest
import torch

import torchfits
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
            if not hasattr(torchfits, "_C"): torchfits._C = mock.Mock()
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


def test_table_data_accessor_preserves_rank():
    from torchfits._hdu.table_hdu import TableDataAccessor, TableHDU

    col = torch.ones(5, 1)
    hdu = TableHDU({"COL": col})
    acc = TableDataAccessor(hdu)
    assert acc["COL"].shape == (5,)


def test_pathological_naxis_product_raises(tmp_path):
    """P2-9: absurd NAXISn values must fail before under-allocating a buffer."""
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
