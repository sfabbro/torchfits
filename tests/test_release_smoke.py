from __future__ import annotations

from importlib import metadata
from pathlib import Path
import tempfile

import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table

import torchfits


def _declared_version() -> str:
    try:
        return metadata.version("torchfits")
    except metadata.PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        import tomllib

        with pyproject.open("rb") as fh:
            data = tomllib.load(fh)
        return str(data["project"]["version"])


def test_runtime_version_matches_declared_version() -> None:
    assert torchfits.__version__ == _declared_version()


def test_release_smoke_image_read_write_roundtrip() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    header = {"OBJECT": "SMOKE"}

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        torchfits.write(path, image, header=header, overwrite=True)
        out, hdr = torchfits.read(path, return_header=True)

        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 4)
        assert torch.allclose(out.cpu(), image)
        assert str(hdr["OBJECT"]).strip() == "SMOKE"

        with torchfits.open(path) as hdul:
            reopened = hdul[0].to_tensor()
            assert torch.allclose(reopened.cpu(), image)
    finally:
        Path(path).unlink(missing_ok=True)


def test_release_smoke_table_read() -> None:
    table = Table(
        {
            "RA": np.array([10.1, 10.2, 10.3], dtype=np.float64),
            "ID": np.array([1, 2, 3], dtype=np.int64),
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        table.write(path, format="fits", overwrite=True)
        data = torchfits.read_table(path, hdu=1, columns=["RA", "ID"])

        assert set(data.keys()) == {"RA", "ID"}
        assert torch.allclose(
            data["RA"].cpu(), torch.tensor([10.1, 10.2, 10.3], dtype=torch.float64)
        )
        assert torch.equal(data["ID"].cpu(), torch.tensor([1, 2, 3], dtype=torch.int64))
    finally:
        Path(path).unlink(missing_ok=True)


def test_release_smoke_wcs_header_build() -> None:
    data = np.zeros((8, 8), dtype=np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CRPIX1"] = 4.0
    hdu.header["CRPIX2"] = 4.0
    hdu.header["CRVAL1"] = 180.0
    hdu.header["CRVAL2"] = 0.0
    hdu.header["CD1_1"] = -0.01
    hdu.header["CD1_2"] = 0.0
    hdu.header["CD2_1"] = 0.0
    hdu.header["CD2_2"] = 0.01

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        hdu.writeto(path, overwrite=True)
        wcs = torchfits.get_wcs(path, hdu=0)
        world = wcs.pixel_to_world(torch.tensor([3.0]), torch.tensor([3.0]))
        assert len(world) == 2
        assert torch.isfinite(world[0]).all()
        assert torch.isfinite(world[1]).all()
    finally:
        Path(path).unlink(missing_ok=True)
