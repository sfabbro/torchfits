"""bzip2-wrapped FITS input behavior.

Two regimes, both pinned:

* Builds compiled with ``HAVE_BZIP2`` (``torchfits._C.HAS_BZIP2``) read
  whole-file ``.bz2`` FITS natively — CFITSIO decompresses on open.
* Incapable builds must fail with the actionable capability error instead
  of a confusing CFITSIO message.
"""

from __future__ import annotations

import bz2

import numpy as np
import pytest

astropy_fits = pytest.importorskip("astropy.io.fits")
torch = pytest.importorskip("torch")

import torchfits  # noqa: E402
from torchfits._io_engine.paths import has_bz2_support  # noqa: E402

HAS_BZ2 = has_bz2_support()


@pytest.fixture(scope="module")
def bz2_mef(tmp_path_factory):
    """Plain MEF (image + table) and its whole-file bzip2 twin."""
    d = tmp_path_factory.mktemp("bz2")
    hdul = astropy_fits.HDUList(
        [
            astropy_fits.PrimaryHDU(data=np.arange(64, dtype=np.float32).reshape(8, 8)),
            astropy_fits.BinTableHDU.from_columns(
                [astropy_fits.Column(name="A", format="J", array=np.arange(100))]
            ),
        ]
    )
    plain = str(d / "mef.fits")
    hdul.writeto(plain)
    bz = str(d / "mef.fits.bz2")
    with open(bz, "wb") as fh:
        fh.write(bz2.compress(open(plain, "rb").read()))
    return plain, bz


def _require_capable():
    if not HAS_BZ2:
        pytest.skip("torchfits built without bzip2 support")


# ---------------------------------------------------------------------------
# Capable builds: native transparent reads across every entry point
# ---------------------------------------------------------------------------


def test_bz2_read_tensor_matches_plain(bz2_mef):
    _require_capable()
    plain, bz = bz2_mef
    torch.testing.assert_close(torchfits.read_tensor(bz), torchfits.read_tensor(plain))


def test_bz2_root_read_returns_columns(bz2_mef):
    _require_capable()
    _, bz = bz2_mef
    got = torchfits.read(bz, hdu=1)
    assert isinstance(got, dict)
    assert got["A"].tolist() == list(range(100))


def test_bz2_table_read_torch(bz2_mef):
    _require_capable()
    _, bz = bz2_mef
    cols = torchfits.table.read_torch(bz, hdu=1)
    assert cols["A"].tolist() == list(range(100))


def test_bz2_hdulist_and_header(bz2_mef):
    _require_capable()
    plain, bz = bz2_mef
    with torchfits.open(plain) as ref, torchfits.open(bz) as hdul:
        assert len(hdul) == len(ref)
    assert int(torchfits.read_header(bz, 0)["NAXIS"]) == 2


def test_bz2_subset_reader(bz2_mef):
    _require_capable()
    _, bz = bz2_mef
    with torchfits.open_subset_reader(bz, hdu=0) as reader:
        cut = reader.read_subset(0, 0, 4, 4)
    torch.testing.assert_close(cut, torchfits.read_tensor(bz)[:4, :4])


def test_bz2_scan_batches(bz2_mef):
    _require_capable()
    _, bz = bz2_mef
    import pyarrow as pa

    batches = list(torchfits.table.scan(bz))
    assert batches
    table = pa.Table.from_batches(batches)
    assert table["A"].to_pylist() == list(range(100))


# ---------------------------------------------------------------------------
# Both builds: write-side guard (CFITSIO would write an uncompressed file)
# ---------------------------------------------------------------------------


def test_write_rejects_bz2_output_name(tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        torchfits.write(str(tmp_path / "x.fits.bz2"), torch.zeros(2, 2))


def test_table_write_rejects_bz2_output_name(tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        torchfits.table.write(str(tmp_path / "y.fits.bz2"), {"A": [1, 2]})


# ---------------------------------------------------------------------------
# Incapable builds (simulated): actionable error on every guarded entry
# ---------------------------------------------------------------------------


def test_incapable_build_raises_actionable_error(bz2_mef, monkeypatch):
    from torchfits._io_engine import paths as _paths

    monkeypatch.setattr(_paths, "has_bz2_support", lambda: False)
    _, bz = bz2_mef
    with pytest.raises(ValueError, match="without bzip2 support"):
        torchfits.read(bz)
    with pytest.raises(ValueError, match="without bzip2 support"):
        torchfits.read_tensor(bz)
    with pytest.raises(ValueError, match="without bzip2 support"):
        torchfits.read_subset(bz, hdu=0, x1=0, y1=0, x2=1, y2=1)


def test_capability_flag_is_exposed():
    assert isinstance(HAS_BZ2, bool)
    import torchfits._C as _C

    assert bool(getattr(_C, "HAS_BZIP2", False)) == HAS_BZ2
