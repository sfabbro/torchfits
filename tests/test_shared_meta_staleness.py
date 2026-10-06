"""SharedReadMeta must not serve stale EXTNAME -> index resolution.

``hdu_name_cache`` maps a normalized ``EXTNAME`` to an HDU index. It is keyed
by name alone, so it has to be dropped whenever the stat check notices that the
file changed — a rewrite can move a name to another index, or reuse it for a
different extension.

It was not dropped, and the consequence was silent: after an out-of-band
rewrite that moved ``EXTNAME="SCI"`` from HDU 1 to HDU 2,
``read(path, hdu="SCI")`` returned the array belonging to ``ERR`` with no error
or warning. These tests rewrite the file with astropy (deliberately out of
band, so nothing calls ``invalidate_shared_meta``) and wait past the validator's
interval, which is the only mechanism that can notice.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torchfits  # noqa: E402
from astropy.io import fits  # noqa: E402

# SharedReadMeta re-stats a path at most once per interval (default 1000 ms).
_VALIDATE_INTERVAL_S = 1.2

SCI_VALUE = 111
ERR_VALUE = 222


def _write(tmp_path, sci_index, name="named.fits"):
    """Write a MEF with SCI/ERR, placing SCI at ``sci_index`` (1 or 2)."""
    path = str(tmp_path / name)
    sci = fits.ImageHDU(np.full((2, 2), SCI_VALUE, dtype=np.int16), name="SCI")
    err = fits.ImageHDU(np.full((2, 2), ERR_VALUE, dtype=np.int16), name="ERR")
    hdus = [fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32))]
    hdus += [sci, err] if sci_index == 1 else [err, sci]
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_extname_resolution_not_stale_after_index_moves(tmp_path):
    """A moved EXTNAME must resolve to its new HDU, not the cached old one."""
    path = _write(tmp_path, sci_index=1)
    assert torchfits.read(path, hdu="SCI").flatten()[0].item() == SCI_VALUE

    time.sleep(_VALIDATE_INTERVAL_S)
    _write(tmp_path, sci_index=2)  # out of band: nothing invalidates the meta

    got = torchfits.read(path, hdu="SCI").flatten()[0].item()
    exp = int(fits.getdata(path, "SCI").flatten()[0])
    assert exp == SCI_VALUE
    assert got == exp, f"stale EXTNAME resolution: read {got}, expected {exp}"


def test_table_metadata_cache_invalidated_by_torchfits_mutation(tmp_path):
    """A mutation through torchfits must refresh the cached table metadata."""
    path = str(tmp_path / "t.fits")
    torchfits.table.write(path, {"A": np.arange(10, dtype=np.int32)}, overwrite=True)
    assert torchfits.read_nrows(path) == 10

    torchfits.table.append_rows(path, {"A": np.array([99], dtype=np.int32)})
    assert torchfits.read_nrows(path) == 11

    torchfits.table.delete_rows(path, slice(0, 3))
    assert torchfits.read_nrows(path) == 8


@pytest.mark.parametrize("probe", ["nrows", "colnames", "hdu_type", "num_hdus"])
def test_table_metadata_cache_invalidated_out_of_band(tmp_path, probe):
    """An out-of-band rewrite must refresh every cached structural probe."""
    path = str(tmp_path / f"oob_{probe}.fits")

    def write(rows, ncols, extra_hdu):
        cols = [
            fits.Column(name=f"C{i}", format="J", array=np.zeros(rows, dtype=np.int32))
            for i in range(ncols)
        ]
        hdus = [fits.PrimaryHDU()]
        hdus.append(fits.BinTableHDU.from_columns(cols))
        if extra_hdu:
            hdus.append(fits.ImageHDU(np.zeros((2, 2), dtype=np.float32), name="EXTRA"))
        fits.HDUList(hdus).writeto(path, overwrite=True)

    def observed():
        if probe == "nrows":
            return torchfits.read_nrows(path)
        if probe == "colnames":
            return list(torchfits.read_colnames(path, 1))
        if probe == "hdu_type":
            return torchfits.read_hdu_type(path, 1)
        return torchfits.read_num_hdus(path)

    write(rows=5, ncols=2, extra_hdu=False)
    before = observed()

    time.sleep(_VALIDATE_INTERVAL_S)
    write(rows=9, ncols=4, extra_hdu=True)
    after = observed()

    if probe == "nrows":
        assert (before, after) == (5, 9)
    elif probe == "colnames":
        assert before == ["C0", "C1"]
        assert after == ["C0", "C1", "C2", "C3"]
    elif probe == "hdu_type":
        assert before == after == "BINARY_TABLE"
    else:
        assert before == 2
        assert after == 3


def test_reader_cache_generation_rotates_on_shared_meta_invalidation(tmp_path):
    """The thread-local TableReader cache must follow shared-meta invalidation.

    A content replacement invisible to stat (same inode/size/mtime) can only be
    observed through explicit invalidation: ``clear_shared_read_meta_cache``
    erases the path's SharedReadMeta and its generation (uid) is minted anew on
    the next lookup. The reader cache must stamp that generation on acquire and
    drop handles from an older one, or it keeps decoding with the stale column
    layout (silently wrong values) even after the documented clear.
    """
    import importlib

    from test_reader_cache_freshness import (
        build_format_swap_pair,
        replace_bytes_invisible,
    )

    m = importlib.import_module("torchfits._C")
    path, b_bytes = build_format_swap_pair(tmp_path)

    first = m.read_fits_table_rows(path, 1, ["A"], 1, -1, False)["A"]
    assert first.dtype == torch.int32 and first.tolist() == [1, 2, 3]

    replace_bytes_invisible(path, b_bytes)
    m.clear_shared_read_meta_cache()

    got = m.read_fits_table_rows(path, 1, ["A"], 1, -1, False)["A"]
    assert got.dtype == torch.float32, f"stale decode dtype: {got.dtype}"
    assert got.tolist() == [7.5, 8.5, 9.5], f"stale reader cache data: {got.tolist()}"


def test_extname_resolution_fails_when_name_disappears(tmp_path):
    """A name that no longer exists must raise, never resolve to a stale index."""
    path = _write(tmp_path, sci_index=1)
    assert torchfits.read(path, hdu="SCI").flatten()[0].item() == SCI_VALUE

    time.sleep(_VALIDATE_INTERVAL_S)
    rewritten = str(tmp_path / "renamed.fits")
    fits.HDUList(
        [
            fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32)),
            fits.ImageHDU(np.full((2, 2), ERR_VALUE, dtype=np.int16), name="ERR"),
        ]
    ).writeto(rewritten, overwrite=True)

    with pytest.raises(Exception):
        torchfits.read(rewritten, hdu="SCI")


# The descriptor cache is per-process state that outlives the read, so it can
# only be measured in a child: this process has to keep its own descriptors.
_FD_PROBE = textwrap.dedent(
    """
    import os, sys, tempfile
    import numpy as np
    from astropy.io import fits
    import torchfits

    n_files = int(sys.argv[1])
    cap = int(os.environ["TORCHFITS_MAX_CACHED_FDS"])

    def open_fds():
        d = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
        return len([e for e in os.listdir(d) if e.isdigit()])

    tmp = tempfile.mkdtemp(prefix="torchfits-fd-")
    data = np.arange(64, dtype=np.int16).reshape(8, 8)
    paths = []
    for i in range(n_files):
        p = os.path.join(tmp, "f%04d.fits" % i)
        fits.PrimaryHDU(data).writeto(p, overwrite=True)
        paths.append(p)

    before = open_fds()
    for p in paths:
        torchfits.read(p, hdu=0)
    retained = open_fds() - before
    print("read %d paths, retained %d descriptors (cap %d)" % (n_files, retained, cap))
    sys.exit(0 if retained <= cap else 1)
    """
)


def test_shared_read_cache_does_not_retain_one_descriptor_per_path():
    """Reading N paths must not leave N descriptors open in the process.

    SharedReadMeta memoises each path's raw descriptor so a repeat read skips
    an ``open()``, and the memo had no bound: a loop over N files left N
    descriptors open for the life of the process. The table is per-process, so
    the cost lands on everything else sharing it -- measured on macOS with
    ``RLIMIT_NOFILE=64``, 90 reads left 64/64 descriptors held and 199 of 200
    further opens failing with EMFILE, while ``clear_all_caches()`` gave all
    of them back.

    The cap is asserted through ``TORCHFITS_MAX_CACHED_FDS`` rather than a
    hardcoded number, so this pins the contract (the knob bounds retention)
    instead of the default value chosen for it.
    """
    cap = 8
    env = {**os.environ, "TORCHFITS_MAX_CACHED_FDS": str(cap)}
    result = subprocess.run(
        [sys.executable, "-c", _FD_PROBE, "60"],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"the shared read cache kept more than TORCHFITS_MAX_CACHED_FDS={cap} "
        f"descriptors open (exit {result.returncode}):\n"
        f"{result.stdout}{result.stderr}"
    )


def test_stale_open_handle_does_not_republish_into_the_shared_cache(tmp_path):
    """A handle opened before a rewrite must not write its header into the cache.

    ``SharedReadMeta`` is shared with every other reader, whose handles describe
    the file as it is *now*; a ``FITSFile`` (what ``open_subset_reader`` and any
    reused handle hold) describes the file as it was when it opened. It published
    its header view unconditionally, so after an out-of-band rewrite the
    validator would clear the slot and rotate the generation -- and the stale
    handle would put the replaced file's shape straight back, leaving the next
    fresh read to resolve an ``(8, 8)`` shape for a ``(32, 32)`` image and
    return its first 64 pixels as if that were the answer.

    ``use_cache=False`` is the read that consults ``image_info_cache``; the
    default path resolves the header from its own fresh handle, which is why
    this needs the option to be visible at all.

    Timing: the validator re-stats a path at most once per interval (1000 ms),
    so the sleep makes the middle read validate, and the last two calls have to
    land inside the following interval. The failure mode of missing that window
    is a false green, never a false red -- a slower machine revalidates and sees
    the correct file.
    """
    from torchfits import _C

    path = str(tmp_path / "stale_handle.fits")
    fits.PrimaryHDU(np.arange(64, dtype=np.int16).reshape(8, 8)).writeto(
        path, overwrite=True
    )

    handle = _C.FITSFile(path, 0)
    assert list(handle.get_shape(0)) == [8, 8]
    assert tuple(torchfits.read(path, use_cache=False).shape) == (8, 8)

    # Out-of-band rewrite: a new inode, so the validator can see it.
    fits.PrimaryHDU(np.full((32, 32), 7, dtype=np.int16)).writeto(path, overwrite=True)
    time.sleep(_VALIDATE_INTERVAL_S)

    # This read clears the caches and rotates the generation, so the shared slot
    # now belongs to the rewritten file.
    fresh = torchfits.read(path, use_cache=False)
    assert tuple(fresh.shape) == (32, 32), (
        f"the rewrite was not picked up at all: {tuple(fresh.shape)}"
    )

    # The handle predates the rewrite and still reads the old bytes -- that is
    # what a pinned handle means. It must not publish them.
    stale = handle.read_tensor(0, False)
    assert tuple(stale.shape) == (8, 8)

    after = torchfits.read(path, use_cache=False)
    assert tuple(after.shape) == (32, 32), (
        "a handle opened before the rewrite republished its header: a fresh "
        f"read resolved {tuple(after.shape)} for a (32, 32) image"
    )
    assert after.flatten()[0].item() == 7
