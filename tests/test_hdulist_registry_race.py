"""A-18 registry race pin (r4c-01): HDUList deregistration must use cache_lock.

``HDUList.close`` used to read-modify-write ``caches._open_hdulist_registry``
without the registry lock, so a close racing an open (or an invalidation) could
lose the concurrent registration or resurrect a closed handle (lost update).
Every registry access must now happen under ``caches.cache_lock``, and close's
deregistration must never clobber a concurrent registration.
"""

from __future__ import annotations

import os
import threading
from unittest import mock

import numpy as np
from astropy.io import fits

import torchfits
from torchfits._io_engine import caches
from torchfits._io_engine.hdu_api import open_hdulist


class _TrackingRegistry(dict):
    """dict proxy recording accesses made outside ``caches.cache_lock``.

    ``on_unlocked`` (when set) runs after a *snapshot* access (``get`` /
    ``items``) has captured its value but before the caller resumes — that is
    the exact window in which the pre-fix close() lost concurrent updates.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unlocked: list[str] = []
        self.on_unlocked = None

    def _check(self, op: str) -> None:
        if caches.cache_lock._is_owned():
            return
        self.unlocked.append(op)
        hook = self.on_unlocked
        if hook is not None:
            self.on_unlocked = None
            hook()

    def get(self, key, default=None):
        value = super().get(key, default)
        self._check("get")
        return value

    def items(self):
        value = list(super().items())
        self._check("items")
        return value

    def pop(self, key, *default):
        self._check("pop")
        return super().pop(key, *default)

    def setdefault(self, key, default=None):
        self._check("setdefault")
        return super().setdefault(key, default)

    def __setitem__(self, key, value):
        self._check("__setitem__")
        super().__setitem__(key, value)

    def __getitem__(self, key):
        self._check("__getitem__")
        return super().__getitem__(key)


def _write_mef(path: str) -> None:
    img = fits.ImageHDU(np.zeros((4, 4), dtype=np.float32))
    fits.HDUList([fits.PrimaryHDU(), img]).writeto(path, overwrite=True)


def test_registry_access_is_serialized_under_cache_lock(monkeypatch, tmp_path):
    """Concurrent open/close/invalidate must never touch the registry unlocked."""
    path = str(tmp_path / "race.fits")
    _write_mef(path)
    real = os.path.realpath(path)

    tracking = _TrackingRegistry()
    monkeypatch.setattr(caches, "_open_hdulist_registry", tracking)

    errors: list[BaseException] = []

    def open_close() -> None:
        try:
            for _ in range(10):
                hdul = open_hdulist(path)
                hdul.close()
        except BaseException as exc:  # noqa: BLE001 — surfaced via errors below
            errors.append(exc)

    def churn() -> None:
        try:
            for _ in range(10):
                caches.invalidate_path_caches(path)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=open_close) for _ in range(4)] + [
        threading.Thread(target=churn) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent registry churn failed: {errors[:3]}"
    assert tracking.unlocked == [], (
        f"registry accessed outside cache_lock: {tracking.unlocked[:5]}"
    )
    # End state: every handle is closed, so nothing may remain registered.
    assert tracking.get(real, []) == []
    # The file is still perfectly usable afterwards.
    data = torchfits.read(path, hdu=1)
    assert data.numel() == 16


def test_close_does_not_lose_concurrent_registration(monkeypatch, tmp_path):
    """A registration racing HDUList.close must survive close's deregistration.

    Deterministic interleave: park close(hdul1) in its registry access, run
    close(hdul2) to completion and register a new live handle, then resume —
    close(hdul1)'s stale write-back clobbers both (lost update + resurrection).
    """
    path = str(tmp_path / "lost_update.fits")
    _write_mef(path)
    real = os.path.realpath(path)

    tracking = _TrackingRegistry()
    monkeypatch.setattr(caches, "_open_hdulist_registry", tracking)

    hdul1 = open_hdulist(path)
    hdul2 = open_hdulist(path)
    entries = tracking[real]
    handle1 = next(h for h, hdul in entries if hdul is hdul1)
    handle2 = next(h for h, hdul in entries if hdul is hdul2)

    parked = threading.Event()
    resume = threading.Event()

    def hook() -> None:
        parked.set()
        resume.wait(timeout=5.0)

    tracking.on_unlocked = hook

    closer = threading.Thread(target=hdul1.close)
    closer.start()
    got_parked = parked.wait(timeout=1.0)
    # Race close(hdul1)'s read-modify-write: another close completes and a new
    # live HDUList registers before close(hdul1) writes its stale snapshot back.
    hdul2.close()
    late_hdul = mock.Mock(name="late_hdul")
    late_handle = mock.Mock(name="late_handle")
    caches._register_open_hdulist(path, late_handle, late_hdul)
    resume.set()
    closer.join()

    surviving = [h for h, _hdul in tracking.get(real, [])]
    assert late_handle in surviving, (
        "concurrent registration lost to close()'s stale write-back"
        if got_parked
        else "late registration vanished without any interleaving"
    )
    assert handle1 not in surviving, "close() failed to deregister its own handle"
    assert handle2 not in surviving, (
        "closed handle resurrected by close()'s stale write-back"
        if got_parked
        else "closed handle still registered"
    )
    assert surviving == [late_handle]
