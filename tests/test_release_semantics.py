"""Release-semantics regressions (H2, H3, H7, H8).

- H2: float WHERE equality selects identical rows regardless of engine.
- H3: header-only schema falls back for complex columns instead of lying
  about float64.
- H7: windowed reads keep VLA/string columns row-aligned with tensors.
- H8: streaming a scaled table with mmap=True no longer raises mid-stream.
"""

from __future__ import annotations

import numpy as np
import pytest

astropy = pytest.importorskip("astropy")
torch = pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")

import torchfits  # noqa: E402
from astropy.io import fits as afits  # noqa: E402


def _write_float32_table(path, values):
    cols = [afits.Column(name="X", format="E", array=np.array(values, dtype=np.float32))]
    afits.BinTableHDU.from_columns(cols).writeto(str(path), overwrite=True)


def test_where_float32_equality_matches_storage_precision(tmp_path):
    path = tmp_path / "f32.fits"
    _write_float32_table(path, [0.1, 0.1, 5.0, 0.30000001])

    got = torchfits.table.read(str(path), where="X == 0.1")
    assert got.num_rows == 2, got.to_pydict()
    np.testing.assert_array_equal(got["X"].to_numpy(), np.float32(0.1) * np.ones(2))


def test_schema_complex_column_raises_clear_error(tmp_path):
    """Arrow cannot express complex; the API must say so instead of lying
    (old header-schema reported float64) or crashing cryptically."""
    path = tmp_path / "cplx.fits"
    torchfits.write(
        str(path),
        {"Z": torch.complex(torch.randn(8), torch.randn(8))},
        overwrite=True,
    )
    with pytest.raises(NotImplementedError, match="complex"):
        torchfits.table.read(str(path))
    with pytest.raises(NotImplementedError, match="complex"):
        list(torchfits.table.scan(str(path)))

    # Torch-dict readers still serve complex columns fine.
    got = torchfits.table.read_torch(str(path), hdu=1)
    assert got["Z"].dtype == torch.complex64


def test_windowed_read_keeps_vla_columns_aligned(tmp_path):
    path = tmp_path / "vla.fits"
    vla = [[float(i)] * (i % 4 + 1) for i in range(10)]
    torchfits.write(
        str(path),
        {"N": torch.arange(10, dtype=torch.int32), "V": vla},
        overwrite=True,
    )
    result = torchfits.read(
        str(path), hdu=1, start_row=4, num_rows=3, mode="table"
    )
    assert len(result["N"]) == len(result["V"]) == 3
    # Row 4 is 1-based => N value 3 (0-based storage).
    assert result["N"].tolist() == [3, 4, 5]
    assert [len(v) for v in result["V"]] == [(i % 4 + 1) for i in (3, 4, 5)]


def test_transforms_never_mutate_input(tmp_path):
    """M6: forward/inverse of scaling transforms must be functional."""
    from torchfits.transforms import FITSHeaderScale, ZScaleNormalize

    x = torch.tensor([1.0, 2.0, 3.0])
    x0 = x.clone()
    t = FITSHeaderScale(bscale=2.0, bzero=10.0)
    y = t(x)
    assert torch.equal(x, x0), "FITSHeaderScale.forward mutated its input"
    _ = t.inverse(y)
    assert torch.equal(x, x0), "FITSHeaderScale.inverse mutated its input"

    z = ZScaleNormalize()
    img = torch.rand(8, 8)
    img0 = img.clone()
    _ = z(img)
    _ = z.inverse(z(img))
    assert torch.equal(img, img0), "ZScaleNormalize mutated its input"


def test_sigma_clip_fill_nan_keeps_rejection_visible():
    from torchfits.transforms import AsymmetricSigmaClip, SigmaClip

    x = torch.full((1, 16, 16), 5.0)
    x[0, 0, 0] = 1000.0  # outlier
    out = SigmaClip(fill="nan")(x)
    assert bool(torch.isnan(out[0, 0, 0]))
    assert not bool(torch.isnan(out[0, 8, 8]))  # background pixels untouched

    out2 = AsymmetricSigmaClip(fill="nan")(x)
    assert bool(torch.isnan(out2[0, 0, 0]))
    with pytest.raises(ValueError, match="fill"):
        SigmaClip(fill="bogus")


def test_median_matches_numpy_for_even_counts():
    from torchfits.transforms.helpers import _median

    x = torch.arange(16, dtype=torch.float32)
    got = _median(x.reshape(1, -1), (-1,))
    np.testing.assert_allclose(got.item(), float(np.median(np.arange(16))))


def test_stream_scaled_table_with_mmap_succeeds(tmp_path):
    path = tmp_path / "scaled.fits"
    rng = np.random.default_rng(42)
    raw = (rng.integers(-1000, 1000, size=64)).astype(np.int16)
    cols = [afits.Column(name="RAW", format="I", array=raw)]
    hdul = afits.HDUList(
        [afits.PrimaryHDU(), afits.BinTableHDU.from_columns(cols)]
    )
    hdul.writeto(str(path), overwrite=True)
    # Attach TSCAL/TZERO to column 1 post-write (astropy's Column bscale
    # pre-scales the in-memory array, which is not what we want here).
    with afits.open(str(path), mode="update") as hdul:
        hdr = hdul[1].header
        hdr["TSCAL1"] = 0.5
        hdr["TZERO1"] = 10.0

    batches = list(torchfits.table.scan(str(path), mmap=True))
    assert batches, "streaming must produce batches"
    table = pa.Table.from_batches(batches)
    expected = raw.astype(np.float64) * 0.5 + 10.0
    np.testing.assert_allclose(table["RAW"].to_numpy(), expected, rtol=1e-6)
