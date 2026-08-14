"""Comprehensive test suite for Apple Silicon MPS device support."""

from __future__ import annotations

from pathlib import Path
from typing import cast
import numpy as np
import pytest
import torch
from astropy.io import fits

import torchfits
import torchfits.table as fits_table
import torchfits.transforms as T
from torchfits.data import FitsCutoutDataset
from torchfits.hdu import TensorHDU, TableHDURef
from torchfits._io_engine.quantize import quantize_int16_robust, dequantize_int16

mps_required = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS is not available on this platform/runner",
)


@mps_required
def test_mps_is_available() -> None:
    assert torch.backends.mps.is_available()
    assert torch.backends.mps.is_built()


@mps_required
def test_image_read_write_mps(tmp_path: Path) -> None:
    path = str(tmp_path / "img_mps.fits")
    data = torch.randn(64, 64, dtype=torch.float32)

    # Write MPS tensor
    data_mps = data.to("mps")
    torchfits.write(path, data_mps, overwrite=True)

    # Read with string "mps"
    out_str, hdr = torchfits.read(path, device="mps", return_header=True)
    assert out_str.device.type == "mps"
    assert torch.allclose(out_str.cpu(), data)
    assert hdr is not None

    # Read with torch.device("mps")
    out_dev = torchfits.read(path, device=torch.device("mps"))
    assert out_dev.device.type == "mps"
    assert torch.allclose(out_dev.cpu(), data)

    # Read with torch.device("mps:0")
    out_dev0 = torchfits.read(path, device=torch.device("mps:0"))
    assert out_dev0.device.type == "mps"
    assert torch.allclose(out_dev0.cpu(), data)


@mps_required
def test_image_float64_auto_downcast_mps(tmp_path: Path) -> None:
    path = str(tmp_path / "f64_mps.fits")
    data64 = torch.randn(32, 32, dtype=torch.float64)
    torchfits.write(path, data64, overwrite=True)

    # MPS does not support float64 natively; torchfits automatically adapts to float32
    out = torchfits.read(path, device="mps")
    assert out.device.type == "mps"
    assert out.dtype == torch.float32
    assert torch.allclose(out.cpu(), data64.float(), atol=1e-5)


@mps_required
def test_scale_on_device_mps(tmp_path: Path) -> None:
    # int8
    int8_data = np.array([-128, 0, 127, 42], dtype=np.int8).reshape(2, 2)
    p_int8 = str(tmp_path / "int8.fits")
    fits.PrimaryHDU(int8_data).writeto(p_int8, overwrite=True)
    r_int8 = torchfits.read(p_int8, device=torch.device("mps"), scale_on_device=True)
    assert r_int8.dtype == torch.int8
    assert r_int8.device.type == "mps"
    assert r_int8.cpu().numpy().tolist() == int8_data.tolist()

    # uint16
    uint16_data = np.array([[0, 32768, 65535]], dtype=np.uint16)
    p_uint16 = str(tmp_path / "uint16.fits")
    fits.PrimaryHDU(uint16_data).writeto(p_uint16, overwrite=True)
    r_uint16 = torchfits.read(p_uint16, device="mps", scale_on_device=True)
    assert r_uint16.dtype == torch.uint16
    assert r_uint16.device.type == "mps"
    assert r_uint16.cpu().numpy().tolist() == uint16_data.tolist()

    # scaled int16 (BSCALE/BZERO)
    hdu = fits.PrimaryHDU(np.array([[10, 20], [30, 40]], dtype=np.int16))
    hdu.header["BSCALE"] = 2.5
    hdu.header["BZERO"] = 10.0
    p_scaled = str(tmp_path / "scaled.fits")
    hdu.writeto(p_scaled, overwrite=True)
    r_scaled = torchfits.read(p_scaled, device="mps", scale_on_device=True)
    assert r_scaled.device.type == "mps"
    assert r_scaled.dtype == torch.float32
    expected = np.array([[35.0, 60.0], [85.0, 110.0]], dtype=np.float32)
    assert torch.allclose(r_scaled.cpu(), torch.from_numpy(expected))


@mps_required
def test_batch_read_mps(tmp_path: Path) -> None:
    paths = [str(tmp_path / f"batch_{i}.fits") for i in range(3)]
    tensors = [torch.randn(16, 16, dtype=torch.float32) for _ in range(3)]
    for p, t in zip(paths, tensors):
        torchfits.write(p, t, overwrite=True)

    results = torchfits.read_batch(paths, device=torch.device("mps"))
    assert len(results) == 3
    for r, orig in zip(results, tensors):
        assert r.device.type == "mps"
        assert torch.allclose(r.cpu(), orig)


@mps_required
def test_subset_reader_mps(tmp_path: Path) -> None:
    path = str(tmp_path / "subset.fits")
    data = torch.arange(10000, dtype=torch.float32).reshape(100, 100)
    torchfits.write(path, data, overwrite=True)

    with torchfits.open_subset_reader(path, device=torch.device("mps")) as reader:
        cutout = reader.read_subset(10, 20, 30, 50)
        assert cutout.device.type == "mps"
        assert cutout.shape == (30, 20)
        assert torch.allclose(cutout.cpu(), data[20:50, 10:30])

        # callable interface
        cutout2 = reader(10, 20, 30, 50)
        assert cutout2.device.type == "mps"
        assert torch.allclose(cutout2.cpu(), data[20:50, 10:30])


@mps_required
def test_hdulist_tensor_hdu_mps(tmp_path: Path) -> None:
    path = str(tmp_path / "hdulist.fits")
    data = torch.randn(32, 32, dtype=torch.float32)
    torchfits.write(path, data, overwrite=True)

    with torchfits.open(path) as hdul:
        hdu0 = cast(TensorHDU, hdul[0])
        tensor = hdu0.to_tensor(device="mps")
        assert tensor.device.type == "mps"
        assert torch.allclose(tensor.cpu(), data)


@mps_required
def test_table_read_write_mps(tmp_path: Path) -> None:
    path = str(tmp_path / "table.fits")
    table_data = {
        "RA": np.array([10.1, 10.2, 10.3], dtype=np.float64),
        "DEC": np.array([-20.1, -20.2, -20.3], dtype=np.float32),
        "MAG": np.array([18.5, 19.2, 20.1], dtype=np.float32),
        "ID": np.array([1, 2, 3], dtype=np.int64),
    }
    torchfits.write(path, table_data, overwrite=True)

    # torchfits.read(mode="table", device="mps")
    t1 = torchfits.read(path, hdu=1, mode="table", device="mps")
    assert all(
        t.device.type == "mps" for t in t1.values() if isinstance(t, torch.Tensor)
    )
    assert t1["RA"].dtype == torch.float32
    assert t1["ID"].dtype == torch.int64

    # torchfits.read(hdu="auto", device=torch.device("mps"))
    t_auto = torchfits.read(path, hdu="auto", device=torch.device("mps"))
    assert all(
        t.device.type == "mps" for t in t_auto.values() if isinstance(t, torch.Tensor)
    )

    # fits_table.read_torch(device="mps")
    t2 = fits_table.read_torch(path, hdu=1, device="mps")
    assert all(
        t.device.type == "mps" for t in t2.values() if isinstance(t, torch.Tensor)
    )

    # open_table_reader
    with torchfits.open_table_reader(path, hdu=1) as reader:
        rows = reader.read_torch(device=torch.device("mps"))
        assert all(
            t.device.type == "mps" for t in rows.values() if isinstance(t, torch.Tensor)
        )


@mps_required
def test_table_hdu_ref_mps(tmp_path: Path) -> None:
    path = str(tmp_path / "table_ref.fits")
    table_data = {
        "RA": np.array([10.1, 10.2, 10.3], dtype=np.float64),
        "ID": np.array([1, 2, 3], dtype=np.int64),
    }
    torchfits.write(path, table_data, overwrite=True)

    with torchfits.open(path) as hdul:
        hdu1 = cast(TableHDURef, hdul[1])
        data = hdu1.read(device="mps")
        assert data["RA"].device.type == "mps"
        assert data["ID"].device.type == "mps"

        mat = hdu1.materialize(device="mps")
        assert mat["RA"].device.type == "mps"
        assert mat["ID"].device.type == "mps"


@mps_required
def test_transforms_mps() -> None:
    x = torch.abs(torch.randn(4, 3, 64, 64, device="mps")) + 0.1

    transforms = [
        T.ArcsinhStretch(),
        T.LogStretch(),
        T.SqrtStretch(),
        T.ZScaleNormalize(),
        T.RobustNormalize(),
        T.BackgroundSubtract(),
        T.PercentileClipNormalize(),
        T.MinMaxNormalize(),
        T.GlobalScalarNorm(),
        T.FITSHeaderScale(bscale=2.0, bzero=1.0),
        T.SigmaClip(),
        T.AsymmetricSigmaClip(),
    ]

    for t in transforms:
        res = t(x)
        assert res.device.type == "mps"

    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    rgb = T.lupton_rgb(r, g, b)
    assert rgb.device.type == "mps"
    assert rgb.shape == (4, 64, 64, 3)

    # Helpers
    s_arc = T.safe_arcsinh(x)
    assert s_arc.device.type == "mps"
    s_log = T.safe_log(x)
    assert s_log.device.type == "mps"
    bg_med, bg_std = T.estimate_background(x)
    assert bg_med.device.type == "mps"
    assert bg_std.device.type == "mps"
    z1, z2 = T.zscale_limits(x)
    assert z1.device.type == "mps"
    assert z2.device.type == "mps"


@mps_required
def test_quantization_mps() -> None:
    x = torch.randn(32, 32, device="mps")
    q = quantize_int16_robust(x)
    assert q.codes.device.type == "mps"
    assert q.codes.dtype == torch.int16

    deq = dequantize_int16(q.codes, q.scale, q.zero)
    assert deq.device.type == "mps"
    assert deq.shape == (32, 32)


@mps_required
def test_dataset_mps(tmp_path: Path) -> None:
    path = str(tmp_path / "cutouts_img.fits")
    data = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    torchfits.write(path, data, overwrite=True)

    cutouts = [(path, 0, 0, 0, 16, 16), (path, 0, 16, 16, 32, 32)]
    ds = FitsCutoutDataset(cutouts, device="mps")
    assert len(ds) == 2

    item0 = ds[0]
    assert item0.device.type == "mps"
    assert item0.shape == (1, 16, 16)
    assert torch.allclose(item0.squeeze(0).cpu(), data[0:16, 0:16])
