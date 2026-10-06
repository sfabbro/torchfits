"""HDU metadata lookups: honest errors + cache freshness on replacement (r4c-08/09/13)."""

from __future__ import annotations

import ast
import os
import pathlib
import time
from collections import OrderedDict

import numpy as np
import pytest
from astropy.io import fits

import torchfits
from torchfits._io_engine import caches, hdu_api, image_meta

# SharedReadMeta re-stats a path at most once per interval (default 1000 ms),
# which is the only way it can notice a file disappear. Same wait the shared
# staleness suite uses; one sleep covers every probe below because each has
# its own path.
_VALIDATE_INTERVAL_S = 1.2


def test_named_hdu_on_missing_file_raises_io_error(tmp_path):
    """A missing file must not be reported as "HDU not found" (r4c-08).

    The EXTNAME scan used to swallow the file-level open failure, probe up to
    1024 phantom HDUs, and end with a misleading ValueError.
    """
    missing = str(tmp_path / "no_such.fits")
    with pytest.raises((OSError, RuntimeError)):
        torchfits.read_header(missing, hdu="EVENTS")


def test_named_hdu_on_existing_file_still_resolves(tmp_path):
    path = str(tmp_path / "named.fits")
    img = fits.ImageHDU(np.zeros((2, 2), dtype=np.float32), name="SCI")
    fits.HDUList([fits.PrimaryHDU(), img]).writeto(path, overwrite=True)
    hdr = torchfits.read_header(path, hdu="SCI")
    assert hdr["NAXIS1"] == 2


def test_missing_name_on_existing_file_still_raises_value_error(tmp_path):
    path = str(tmp_path / "named.fits")
    img = fits.ImageHDU(np.zeros((2, 2), dtype=np.float32), name="SCI")
    fits.HDUList([fits.PrimaryHDU(), img]).writeto(path, overwrite=True)
    with pytest.raises(ValueError, match="not found"):
        torchfits.read_header(path, hdu="NOPE")


def test_meta_and_data_caches_rotate_on_file_replacement(tmp_path):
    """Replacing a file must invalidate every path-keyed cache (r4c-13)."""
    path = str(tmp_path / "rot.fits")

    def replace_with(primary_data: np.ndarray) -> None:
        tmp = str(tmp_path / "rot_next.fits")
        fits.HDUList([fits.PrimaryHDU(primary_data)]).writeto(tmp, overwrite=True)
        os.replace(tmp, path)

    replace_with(np.full((2, 2), 1.0, dtype=np.float32))
    assert torchfits.read(path).shape == (2, 2)
    assert torchfits.read_header(path)["NAXIS1"] == 2

    replace_with(np.full((3, 3), 2.0, dtype=np.float32))
    out = torchfits.read(path)
    assert out.shape == (3, 3), "stale data cache after file replacement"
    assert float(out[0, 0]) == 2.0
    assert torchfits.read_header(path)["NAXIS1"] == 3, (
        "stale header cache after file replacement"
    )

    # A payload-extending replacement must also rotate the autodetect answer.
    tmp = str(tmp_path / "rot_mef.fits")
    img = fits.ImageHDU(np.zeros((4, 4), dtype=np.float32))
    fits.HDUList([fits.PrimaryHDU(), img]).writeto(tmp, overwrite=True)
    os.replace(tmp, path)
    assert hdu_api.autodetect_hdu(path) == 1, (
        "stale autodetect cache after file replacement"
    )
    assert torchfits.read(path, hdu="auto").shape == (4, 4)


def test_autodetect_negative_result_stays_fresh(tmp_path):
    """The cached no-payload answer must rotate when a payload appears (r4c-09)."""
    path = str(tmp_path / "empty.fits")
    fits.HDUList([fits.PrimaryHDU()]).writeto(path, overwrite=True)
    assert hdu_api.autodetect_hdu(path) == 0
    assert hdu_api.autodetect_hdu(path) == 0  # repeat: must stay 0

    tmp = str(tmp_path / "empty_next.fits")
    img = fits.ImageHDU(np.zeros((4, 4), dtype=np.float32))
    fits.HDUList([fits.PrimaryHDU(), img]).writeto(tmp, overwrite=True)
    os.replace(tmp, path)
    assert hdu_api.autodetect_hdu(path) == 1


# ---------------------------------------------------------------------------
# R2-014: a file that has gone away must not be answered from cache.
#
# Every staleness check read
#     stale = stored is not None and current is not None and stored != current
# so a *missing* file (current is None) was never stale. Once a path had been
# read, unlinking it left read_header, read_shape, read_hdu_type, read_colnames,
# read_nrows and read_num_hdus happily describing a file that no longer
# existed, while the payload readers (read) correctly raised. The rule is now
# simply `stored != current`, which separates "gone" from "never stat-able"
# (a CFITSIO extended-syntax path such as `mef.fits[1]`) without a syscall.
# ---------------------------------------------------------------------------


def test_signature_rule_separates_a_gone_file_from_a_never_statable_one(tmp_path):
    """The four (stored, current) combinations, pinned by the rule's contract."""
    real = str(tmp_path / "gone.fits")
    fits.HDUList([fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32))]).writeto(
        real, overwrite=True
    )
    # Permanently unstattable: no such file, and the path can never be stat'ed.
    unstattable = str(tmp_path / "never.fits[1]")

    assert caches.path_signature(real) is not None
    assert caches.path_signature(unstattable) is None

    # stored == current -> served, for both a stat-able and a never-stat-able
    # path. This is the case that must keep working.
    for path in (real, unstattable):
        cache: OrderedDict = OrderedDict()
        caches.signature_cached_set(cache, (path, 0), "value", 8)
        assert caches.signature_cached_get(cache, (path, 0)) == "value", path

    # A stored-only signature (written while the path was missing) must not
    # survive the file coming back.
    seeded: OrderedDict = OrderedDict()
    seeded[(real, 0)] = (None, "value")
    assert caches.signature_cached_get(seeded, (real, 0)) is None
    assert (real, 0) not in seeded, "a stored-only entry must be dropped"

    # A current-only signature (the file was unlinked) must not be served, and
    # the entry must be dropped rather than merely ignored.
    cache = OrderedDict()
    caches.signature_cached_set(cache, (real, 0), "value", 8)
    os.unlink(real)
    assert caches.signature_cached_get(cache, (real, 0)) is None
    assert (real, 0) not in cache, "the stale entry must be evicted, not left behind"


@pytest.mark.parametrize(
    ("label", "make", "probe"),
    [
        (
            "read_header",
            lambda p: fits.HDUList([fits.PrimaryHDU(np.zeros((2, 2), np.float32))]),
            lambda p: torchfits.read_header(p),
        ),
        (
            "read_shape",
            lambda p: fits.HDUList([fits.PrimaryHDU(np.zeros((2, 2), np.float32))]),
            lambda p: torchfits.read_shape(p),
        ),
        (
            "read_hdu_type",
            lambda p: fits.HDUList([fits.PrimaryHDU(np.zeros((2, 2), np.float32))]),
            lambda p: torchfits.read_hdu_type(p),
        ),
        (
            "read_num_hdus",
            lambda p: fits.HDUList([fits.PrimaryHDU(np.zeros((2, 2), np.float32))]),
            lambda p: torchfits.read_num_hdus(p),
        ),
        (
            "read_colnames",
            lambda p: fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.BinTableHDU.from_columns(
                        [fits.Column(name="A", format="J", array=np.arange(4))]
                    ),
                ]
            ),
            lambda p: torchfits.read_colnames(p),
        ),
        (
            "read_nrows",
            lambda p: fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.BinTableHDU.from_columns(
                        [fits.Column(name="A", format="J", array=np.arange(4))]
                    ),
                ]
            ),
            lambda p: torchfits.read_nrows(p),
        ),
    ],
)
def test_metadata_probes_stop_answering_once_the_file_is_gone(
    tmp_path, label, make, probe
):
    """Each probe must raise for a deleted file instead of describing it."""
    path = str(tmp_path / f"{label}.fits")
    make(path).writeto(path, overwrite=True)
    probe(path)  # warm every layer this probe consults
    os.unlink(path)
    time.sleep(_VALIDATE_INTERVAL_S)
    with pytest.raises((OSError, RuntimeError)):
        probe(path)


def test_payload_read_cache_does_not_serve_a_deleted_file(tmp_path):
    """The read cache holds bytes, so serving one for a gone file is worse."""
    path = str(tmp_path / "payload.fits")
    fits.HDUList(
        [fits.PrimaryHDU(np.arange(4, dtype=np.float32).reshape(2, 2))]
    ).writeto(path, overwrite=True)
    # return_header routes through the fallback path, which is the only one
    # that populates file_cache.
    torchfits.read(path, cache_capacity=8, return_header=True)
    assert len(caches.file_cache) > 0, "no cached payload: the check would be vacuous"

    os.unlink(path)
    with pytest.raises((OSError, RuntimeError)):
        torchfits.read(path, cache_capacity=8, return_header=True)


def test_extended_syntax_path_keeps_its_caches(tmp_path):
    """`mef.fits[1]` is never stat-able; it must stay cached, not be disabled.

    Reading a missing signature as "fresh" was the R2-014 defect, but flipping
    that the other way would silently turn off every cached metadata probe for
    filtered paths. Both signatures being None is what keeps them live.
    """
    mef = str(tmp_path / "mef.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.zeros((2, 2), dtype=np.float32), name="SCI"),
        ]
    ).writeto(mef, overwrite=True)
    filtered = f"{mef}[1]"

    first_hdr = torchfits.read_header(filtered, 0)
    first_auto = hdu_api.autodetect_hdu(filtered)
    first_meta = image_meta.get_image_meta(filtered, 0)

    warmed = [
        ("header_cards_cache", caches.header_cards_cache),
        ("auto_hdu_cache", caches.auto_hdu_cache),
        ("image_meta_cache", caches.image_meta_cache),
    ]
    for name, cache in warmed:
        with caches.cache_lock:
            keys = [k for k in cache if isinstance(k[0], str) and k[0] == filtered]
            assert keys, f"{name} cached nothing for the extended-syntax path"
            for key in keys:
                stored_sig, value = cache[key]
                assert stored_sig is None, (name, stored_sig)
                assert caches.signature_cached_get(cache, key) == value, (
                    f"{name} dropped its entry: the cache is dead for "
                    "extended-syntax paths"
                )

    assert dict(torchfits.read_header(filtered, 0)) == dict(first_hdr)
    assert hdu_api.autodetect_hdu(filtered) == first_auto
    assert image_meta.get_image_meta(filtered, 0) == first_meta


# Caches whose entries are (path_signature, value) pairs. hdu_type_cache goes
# through caches.signature_cached_get by way of the public get/set helpers, so
# it never appears under its own name and is deliberately absent here.
_SIGNATURE_CACHES = frozenset(
    {
        "image_meta_cache",
        "cold_nommap_cache",
        "auto_mmap_cache",
        "header_cards_cache",
        "auto_hdu_cache",
    }
)
# signature_cached_get plus image_meta's two thin wrappers around it.
_HELPER_NAMES = frozenset({"signature_cached_get", "_sig_get"})
# Attribute methods that *read* a cache entry. R2-059: only subscripts were
# detected before, so `cache.get(key)` was a way to hand-roll a staleness rule
# the guard could not see. The write methods (move_to_end/popitem/clear) are
# deliberately not listed -- they are legitimate on the module's own caches.
_READ_METHODS = frozenset({"get", "__getitem__"})


def test_signature_validated_reads_all_go_through_one_helper():
    """One implementation of the staleness rule, so one place to get it right.

    Four hand-written copies of the rule existed: caches.signature_cached_get,
    the payload-cache check in _check_read_cache_locked, and one each in
    hdu_api.get_header and hdu_api.autodetect_hdu. Three of them carried the
    "a missing file is never stale" defect, and all four could drift apart.
    Every cache in the package now hands its (path, key) lookup to
    caches.signature_cached_get, and nothing outside caches.py reads a
    signature-bearing cache directly at all.

    Deep-review unit 13, R2-059: the detector only flagged ``Subscript``
    loads, so ``image_meta_cache.get(sig)`` -- an ordinary way to write a
    hand-rolled staleness rule, since the entries are plain
    ``(signature, value)`` tuples -- walked straight past it. Replacing
    ``_sig_get(image_meta_cache, sig)`` with that read passed this file and
    every other image-meta test green while serving stale metadata for a
    replaced file. Attribute reads are now flagged too; the mutation that
    motivated it is the one ``test_image_meta_cache_rotates_on_replacement``
    and the ``get`` half of this test pin.
    """
    import torchfits._io_engine as engine

    validated: dict[str, int] = {}
    hand_rolled: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self, source: str) -> None:
            self.source = source
            self.stack: list[ast.AST] = []

        def _cache_names(self, node: ast.AST) -> set[str]:
            return {
                n.id
                for n in ast.walk(node)
                if isinstance(n, ast.Name) and n.id in _SIGNATURE_CACHES
            }

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id in _HELPER_NAMES:
                for name in self._cache_names(node):
                    validated[name] = validated.get(name, 0) + 1
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

        def _record_load(self, node: ast.AST, rendered: str) -> None:
            hand_rolled.append(
                f"{self.source}: {rendered} read outside signature_cached_get"
            )

        def visit_Subscript(self, node: ast.Subscript) -> None:
            # A Load is a read of the cached value; a Store is the legitimate
            # write of a (signature, value) entry.
            if (
                isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id in _SIGNATURE_CACHES
            ):
                self._record_load(node, ast.unparse(node))
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Attribute(self, node: ast.Attribute) -> None:
            # R2-059: `cache.get(key)` reads the same (signature, value) entry
            # as `cache[key]` does, so a hand-rolled rule can hide behind it.
            # Only the read methods are flagged; move_to_end/popitem/clear are
            # legitimate writes. caches.py is not exempted either -- measured
            # at zero current hits, and the file holding the canonical rule is
            # exactly where a second copy would appear.
            if (
                node.attr in _READ_METHODS
                and isinstance(node.value, ast.Name)
                and node.value.id in _SIGNATURE_CACHES
            ):
                self._record_load(node, ast.unparse(node))
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

    for path in sorted(pathlib.Path(engine.__file__).parent.glob("*.py")):
        Visitor(path.name).visit(ast.parse(path.read_text()))

    assert not hand_rolled, "signature validation is hand-rolled again:\n" + "\n".join(
        hand_rolled
    )
    # Non-vacuity: every cache is still validated through the helper, so
    # deleting the lookups cannot quietly satisfy the assertion above.
    assert set(validated) == _SIGNATURE_CACHES, (set(validated), _SIGNATURE_CACHES)


def test_image_meta_cache_rotates_on_replacement(tmp_path):
    """``image_meta_cache`` must not serve a replaced file's shape (R2-059).

    The three caches this file rotates -- data, header cards, autodetect --
    were covered; ``image_meta_cache`` was not, and nothing else covered it
    either: bypassing its signature check left 296 tests green across the
    image/meta/read selections. ``(7, 5)`` is the NAXIS1/NAXIS2 order of a
    ``(5, 7)`` numpy array.

    The sleep is load-bearing. SharedReadMeta re-stats a path at most once per
    interval (default 1000 ms) and sits *below* the Python cache, so without
    it the native layer re-serves the old shape and this test would be green
    whatever the Python cache did -- which is how it was green for the
    mutation above.
    """
    path = str(tmp_path / "meta_rot.fits")
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32)).writeto(path, overwrite=True)

    first = image_meta.get_image_meta(path, 0)
    assert first is not None
    assert tuple(first[2]) == (2, 2), first

    tmp = str(tmp_path / "meta_rot_next.fits")
    fits.PrimaryHDU(np.zeros((5, 7), dtype=np.float32)).writeto(tmp, overwrite=True)
    os.replace(tmp, path)

    time.sleep(_VALIDATE_INTERVAL_S)

    second = image_meta.get_image_meta(path, 0)
    assert second is not None
    assert tuple(second[2]) == (7, 5), (
        f"stale image_meta after file replacement: {second[2]}"
    )
