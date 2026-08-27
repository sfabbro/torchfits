"""Robust int16 quantization: correctness + skewed-distribution fidelity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchfits
from torchfits._io_engine.quantize import (
    _SHRT_MAX_EFF,
    _SHRT_MIN_EFF,
    dequantize_int16,
    parse_quantize_options,
    parse_table_quantize_spec,
    quantize_int16_minmax,
    quantize_int16_robust,
)


def _skewed(n: int, rng: np.random.Generator, n_spikes: int = 5) -> np.ndarray:
    """Bulk near 1000 with σ=10; a handful of spikes at 1e6 (heavy-tailed)."""
    bulk = 1000.0 + 10.0 * rng.standard_normal(n)
    out = bulk.copy()
    idx = rng.choice(n, size=min(n_spikes, n), replace=False)
    out[idx] = 1.0e6
    return out.astype(np.float64)


def _bulk_rms(original: np.ndarray, roundtrip: np.ndarray) -> float:
    mask = original < 5_000.0
    err = roundtrip[mask] - original[mask]
    return float(np.sqrt(np.mean(err * err)))


def test_endpoint_codes_map_lo_hi():
    values = torch.linspace(10.0, 20.0, 1001, dtype=torch.float64)
    packed = quantize_int16_robust(values, lo_q=0.0, hi_q=100.0)
    assert packed.lo == pytest.approx(10.0)
    assert packed.hi == pytest.approx(20.0)
    assert int(packed.codes.min()) == _SHRT_MIN_EFF
    assert int(packed.codes.max()) == _SHRT_MAX_EFF
    rt = dequantize_int16(packed.codes, packed.scale, packed.zero, dtype=torch.float64)
    assert rt[0].item() == pytest.approx(10.0, abs=1e-9)
    assert rt[-1].item() == pytest.approx(20.0, abs=1e-9)


def test_constant_array_exact_roundtrip():
    values = torch.full((32, 8), 3.5, dtype=torch.float32)
    packed = quantize_int16_robust(values)
    assert packed.scale == 1.0
    assert packed.zero == pytest.approx(3.5)
    assert torch.all(packed.codes == 0)
    rt = dequantize_int16(packed.codes, packed.scale, packed.zero)
    assert torch.allclose(rt, values)


def test_nan_and_inf_are_clipped_not_crash():
    values = torch.tensor([1.0, 2.0, float("nan"), float("inf"), 3.0])
    packed = quantize_int16_robust(values, lo_q=0.0, hi_q=100.0)
    assert packed.n_clipped >= 2
    assert torch.isfinite(
        dequantize_int16(packed.codes, packed.scale, packed.zero)
    ).all()


def test_keep_zero_preserves_zero_and_clips_negatives():
    values = torch.tensor([-5.0, 0.0, 1.0, 10.0, 100.0], dtype=torch.float64)
    packed = quantize_int16_robust(values, keep_zero=True, hi_q=100.0)
    assert packed.zero == 0.0
    assert packed.scale > 0.0
    rt = dequantize_int16(packed.codes, packed.scale, packed.zero, dtype=torch.float64)
    assert rt[1].item() == pytest.approx(0.0, abs=1e-12)
    # Negatives collapse to 0.
    assert rt[0].item() == pytest.approx(0.0, abs=1e-12)
    assert rt[-1].item() == pytest.approx(100.0, rel=1e-6, abs=1e-6)


def test_keep_zero_all_nonpositive():
    values = torch.tensor([-2.0, -1.0, 0.0])
    packed = quantize_int16_robust(values, keep_zero=True)
    assert packed.zero == 0.0
    assert torch.all(packed.codes == 0)
    assert packed.blank_code is None


def test_keep_zero_all_nonpositive_nan_sets_blank():
    from torchfits._io_engine.quantize import BLANK_CODE

    values = torch.tensor([-2.0, 0.0, float("nan")])
    packed = quantize_int16_robust(values, keep_zero=True)
    assert packed.blank_code == BLANK_CODE
    assert int(packed.codes[2].item()) == BLANK_CODE
    assert int(packed.codes[0].item()) == 0
    assert int(packed.codes[1].item()) == 0


def test_empty_and_nonfinite_raise():
    with pytest.raises(ValueError, match="empty"):
        quantize_int16_robust(torch.empty(0, dtype=torch.float32))
    with pytest.raises(ValueError, match="no finite"):
        quantize_int16_robust(torch.tensor([float("nan"), float("inf")]))
    with pytest.raises(TypeError):
        quantize_int16_robust(torch.arange(4, dtype=torch.int16))


def test_parse_options_rejects_bad_input():
    assert parse_quantize_options(None) is None
    assert parse_quantize_options("robust") is not None
    with pytest.raises(ValueError):
        parse_quantize_options({"lo_q": 50, "hi_q": 10})
    with pytest.raises(TypeError):
        parse_quantize_options({"lo_q": 0.1, "nope": 1})
    with pytest.raises(TypeError):
        parse_quantize_options(1.5)


def test_table_quantize_skips_integer_columns_on_blanket_robust():
    data = {
        "ID": torch.arange(8, dtype=torch.int32),
        "FLUX": torch.randn(8),
    }
    spec = parse_table_quantize_spec("robust", list(data.keys()), data=data)
    assert set(spec) == {"FLUX"}


def test_robust_beats_minmax_on_flat_vector():
    rng = np.random.default_rng(0)
    values = torch.from_numpy(_skewed(50_000, rng))
    robust = quantize_int16_robust(values)
    minmax = quantize_int16_minmax(values)
    r_rt = dequantize_int16(robust.codes, robust.scale, robust.zero).numpy()
    m_rt = dequantize_int16(minmax.codes, minmax.scale, minmax.zero).numpy()
    r_rms = _bulk_rms(values.numpy(), r_rt)
    m_rms = _bulk_rms(values.numpy(), m_rt)
    assert r_rms < 0.25 * m_rms
    assert r_rms < 1.0


def test_image_write_quantize_1d_and_3d(tmp_path):
    rng = np.random.default_rng(1)
    vec = torch.from_numpy(_skewed(32_768, rng).astype(np.float32))
    cube = vec.reshape(4, 64, 128)

    for label, data in (("1d", vec), ("3d", cube)):
        path = str(tmp_path / f"{label}.fits")
        torchfits.write(path, data, overwrite=True, quantize="robust")
        bitpix, shape = torchfits.read_shape(path, hdu=0)
        assert bitpix == 16
        assert shape == tuple(data.shape)
        hdr = torchfits.read_header(path, hdu=0)
        assert "BSCALE" in hdr and "BZERO" in hdr
        assert float(hdr["BSCALE"]) < 1.0

        robust = quantize_int16_robust(data)
        minmax = quantize_int16_minmax(data)
        r_rt = dequantize_int16(robust.codes, robust.scale, robust.zero).numpy()
        m_rt = dequantize_int16(minmax.codes, minmax.scale, minmax.zero).numpy()
        assert _bulk_rms(data.numpy(), r_rt) < 0.25 * _bulk_rms(data.numpy(), m_rt)

        out = torchfits.read(path, hdu=0)
        assert out.dtype == torch.float32
        assert _bulk_rms(data.numpy(), out.numpy()) < 1.0


def test_table_write_quantize_column(tmp_path):
    rng = np.random.default_rng(2)
    flux = torch.from_numpy(_skewed(20_000, rng).astype(np.float32))
    path = str(tmp_path / "tab.fits")
    torchfits.table.write(
        path,
        {"ID": torch.arange(flux.numel(), dtype=torch.int32), "FLUX": flux},
        overwrite=True,
        quantize={"FLUX": "robust"},
    )
    info = torchfits.read_table_info(path, hdu=1)
    assert info["colnames"] == ["ID", "FLUX"]
    assert any(t.upper().rstrip("0123456789").endswith("I") for t in info["tforms"])

    cols = torchfits.table.read_torch(path, hdu=1, columns=["FLUX"])
    rt = cols["FLUX"].numpy()
    robust = quantize_int16_robust(flux)
    minmax = quantize_int16_minmax(flux)
    r_rt = dequantize_int16(robust.codes, robust.scale, robust.zero).numpy()
    m_rt = dequantize_int16(minmax.codes, minmax.scale, minmax.zero).numpy()
    assert _bulk_rms(flux.numpy(), r_rt) < 0.25 * _bulk_rms(flux.numpy(), m_rt)
    assert _bulk_rms(flux.numpy(), rt) < 1.0


def test_table_write_quantize_robust_all_floats_keeps_int_id(tmp_path):
    rng = np.random.default_rng(3)
    n = 4096
    path = str(tmp_path / "all.fits")
    torchfits.table.write(
        path,
        {
            "ID": torch.arange(n, dtype=torch.int32),
            "A": torch.from_numpy(_skewed(n, rng).astype(np.float32)),
            "B": torch.from_numpy(_skewed(n, rng, n_spikes=3).astype(np.float32)),
        },
        overwrite=True,
        quantize="robust",
    )
    info = torchfits.read_table_info(path, hdu=1)
    # ID stays J; A/B packed to I
    forms = {n: t.upper() for n, t in zip(info["colnames"], info["tforms"])}
    assert forms["ID"].endswith("J")
    assert forms["A"].rstrip("0123456789").endswith("I")
    assert forms["B"].rstrip("0123456789").endswith("I")


def test_default_write_stays_float32(tmp_path):
    data = torch.randn(16, 16, dtype=torch.float32)
    path = str(tmp_path / "native.fits")
    torchfits.write(path, data, overwrite=True)
    bitpix, _ = torchfits.read_shape(path, hdu=0)
    assert bitpix == -32


def test_ndarray_input_roundtrip():
    rng = np.random.default_rng(4)
    arr = _skewed(8192, rng).astype(np.float32).reshape(64, 128)
    packed = quantize_int16_robust(arr)
    assert packed.codes.shape == arr.shape
    rt = dequantize_int16(packed.codes, packed.scale, packed.zero).numpy()
    assert _bulk_rms(arr, rt) < 1.0
