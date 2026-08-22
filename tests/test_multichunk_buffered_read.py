"""Multi-chunk buffered reads (payload > 16 MiB scratch) must be correct.

Regression for the double-buffer prefetch rewrite: `cur` rotated even when
prefetch was disabled, pointing chunk 2's pread at an unsized scratch
vector (heap corruption -> slow + garbage reads on any table whose rows
span more than one 16 MiB chunk with caching disabled).
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits as afits

import torchfits


@pytest.fixture(scope="module")
def big_mixed_table(tmp_path_factory):
    path = tmp_path_factory.mktemp("multichunk") / "mixed_big.fits"
    n = 600_000  # ~41 B/row -> ~24 MiB payload, spans two chunks
    rng = np.random.default_rng(7)
    cols = [
        afits.Column(name="id", format="J", array=np.arange(n, dtype="<i4")),
        afits.Column(name="ra", format="D", array=rng.uniform(0.0, 360.0, n)),
        afits.Column(name="dec", format="D", array=rng.uniform(-90.0, 90.0, n)),
        afits.Column(name="flux", format="E", array=rng.normal(size=n).astype("<f4")),
        afits.Column(name="flag", format="L", array=rng.integers(0, 2, n).astype(bool)),
    ]
    afits.HDUList(
        [afits.PrimaryHDU(), afits.BinTableHDU.from_columns(cols)]
    ).writeto(str(path), overwrite=True)
    return str(path), n


def test_multichunk_nocache_read_matches_astropy(big_mixed_table):
    path, _ = big_mixed_table
    out = torchfits.read(path, hdu=1, mmap=False, cache_capacity=0)
    with afits.open(path) as hdul:
        data = hdul[1].data
        np.testing.assert_array_equal(out["id"].numpy(), data["id"])
        np.testing.assert_allclose(out["ra"].numpy(), data["ra"], rtol=0, atol=0)
        np.testing.assert_allclose(out["dec"].numpy(), data["dec"], rtol=0, atol=0)
        np.testing.assert_array_equal(out["flag"].numpy(), data["flag"])


def test_multichunk_nocache_repeated_reads_stable(big_mixed_table):
    path, n = big_mixed_table
    first = torchfits.read(path, hdu=1, mmap=False, cache_capacity=0)
    for _ in range(3):
        again = torchfits.read(path, hdu=1, mmap=False, cache_capacity=0)
        assert again["id"].shape[0] == n
        np.testing.assert_array_equal(again["flux"].numpy(), first["flux"].numpy())
