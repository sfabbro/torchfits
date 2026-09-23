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
