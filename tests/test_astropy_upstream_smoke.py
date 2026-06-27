from __future__ import annotations

import numpy as np
import pytest
import torch

fits = pytest.importorskip("astropy.io.fits")

import torchfits  # noqa: E402


def test_astropy_image_hdulist_header_and_scaled_workflows_match_torchfits(
    tmp_path,
) -> None:
    image = np.arange(16, dtype=np.int16).reshape(4, 4)
    sci = (np.arange(9, dtype=np.float32).reshape(3, 3) / 3.0).astype(np.float32)

    mef_path = tmp_path / "astropy_mef.fits"
    primary = fits.PrimaryHDU(image)
    primary.header["OBJECT"] = "TORCHFITS"
    primary.header["EXPTIME"] = (12.5, "seconds")
    primary.header.add_history("created by astropy")
    primary.header.add_comment("common FITS header workflow")
    image_hdu = fits.ImageHDU(sci, name="SCI")
    image_hdu.header["BUNIT"] = "adu"
    fits.HDUList([primary, image_hdu]).writeto(mef_path, overwrite=True)

    np.testing.assert_array_equal(
        torchfits.read(mef_path.as_posix(), hdu=0, mmap=False).cpu().numpy(), image
    )
    np.testing.assert_allclose(
        torchfits.read(mef_path.as_posix(), hdu="SCI", mmap=False).cpu().numpy(),
        sci,
        rtol=0.0,
        atol=0.0,
    )
    primary_header = torchfits.get_header(mef_path.as_posix(), hdu=0)
    assert primary_header["OBJECT"] == "TORCHFITS"
    assert primary_header["EXPTIME"] == 12.5
    assert "seconds" in primary_header.comments("EXPTIME")
    assert "created by astropy" in primary_header.get_history()
    assert "common FITS header workflow" in primary_header.get_comment()

    sci_header = torchfits.get_header(mef_path.as_posix(), hdu="SCI")
    assert sci_header["EXTNAME"] == "SCI"
    assert sci_header["BUNIT"] == "adu"

    scaled_path = tmp_path / "astropy_scaled.fits"
    raw = np.array([[0, 1], [2, 3]], dtype=np.int16)
    scaled_hdu = fits.PrimaryHDU(raw)
    scaled_hdu.header["BSCALE"] = 2.0
    scaled_hdu.header["BZERO"] = 10.0
    scaled_hdu.writeto(scaled_path, overwrite=True)

    expected = raw.astype(np.float32) * 2.0 + 10.0
    scaled = torchfits.read(scaled_path.as_posix(), hdu=0, mmap=False).cpu().numpy()
    np.testing.assert_allclose(scaled, expected, rtol=0.0, atol=0.0)


def test_astropy_binary_table_vla_and_complex_workflows_match_torchfits(
    tmp_path,
) -> None:
    path = tmp_path / "astropy_table.fits"
    vla = np.array(
        [
            np.array([1, 2], dtype=np.int32),
            np.array([3], dtype=np.int32),
            np.array([4, 5, 6], dtype=np.int32),
        ],
        dtype=object,
    )
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ID", format="J", array=np.array([1, 2, 3], np.int32)),
            fits.Column(name="NAME", format="8A", array=np.array(["a", "b", "c"])),
            fits.Column(
                name="Z",
                format="C",
                array=np.array([1 + 2j, 3 + 4j, 5 + 6j], np.complex64),
            ),
            fits.Column(name="VLA", format="PJ()", array=vla),
        ],
        name="CATALOG",
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)

    with torchfits.open(path.as_posix()) as hdul:
        table_hdu = hdul["CATALOG"]
        assert table_hdu["ID"].squeeze(-1).tolist() == [1, 2, 3]
        assert table_hdu.get_string_column("NAME") == ["a", "b", "c"]
        np.testing.assert_allclose(
            table_hdu["Z"].squeeze(-1).numpy(),
            np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64),
        )
        assert [row.tolist() for row in table_hdu.get_vla_column("VLA")] == [
            [1, 2],
            [3],
            [4, 5, 6],
        ]


def test_astropy_ascii_table_and_compressed_image_workflows_match_torchfits(
    tmp_path,
) -> None:
    ascii_path = tmp_path / "astropy_ascii.fits"
    ascii_hdu = fits.TableHDU.from_columns(
        [
            fits.Column(name="A", format="I4", array=np.array([1, 2, 3])),
            fits.Column(name="B", format="F8.2", array=np.array([4.5, 5.5, 6.5])),
            fits.Column(name="LABEL", format="A8", array=np.array(["x", "y", "z"])),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), ascii_hdu]).writeto(ascii_path, overwrite=True)

    with torchfits.open(ascii_path.as_posix()) as hdul:
        table = hdul[1].data
        assert torch.equal(table["A"], torch.tensor([1, 2, 3], dtype=torch.int32))
        np.testing.assert_allclose(table["B"].numpy(), np.array([4.5, 5.5, 6.5]))
        assert table["LABEL"][0, 0].item() == ord("x")

    compressed_path = tmp_path / "astropy_compressed.fits"
    image = np.arange(64, dtype=np.int16).reshape(8, 8)
    comp = fits.CompImageHDU(image, compression_type="RICE_1", name="COMP")
    fits.HDUList([fits.PrimaryHDU(), comp]).writeto(compressed_path, overwrite=True)

    out = torchfits.read(compressed_path.as_posix(), hdu=1, mmap=False)
    np.testing.assert_array_equal(out.cpu().numpy(), image)


def test_astropy_vla_and_complex_table_mmap_limitations_are_explicit(
    tmp_path,
) -> None:
    path = tmp_path / "astropy_mmap_unsupported.fits"
    vla = np.array([np.array([1, 2], dtype=np.int32)], dtype=object)
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ID", format="J", array=np.array([1], np.int32)),
            fits.Column(
                name="Z",
                format="C",
                array=np.array([1 + 2j], np.complex64),
            ),
            fits.Column(name="VLA", format="PJ()", array=vla),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)

    # Public reads keep mmap=True ergonomic: unsupported physical layouts use
    # the safe buffered path instead of silently dropping VLA/complex data.
    table = torchfits.read_table(path.as_posix(), hdu=1, mmap=True)
    assert table["VLA"][0].tolist() == [1, 2]

    with pytest.raises(ValueError, match="mmap table updates only support"):
        torchfits.table.update_rows(
            path.as_posix(),
            {"VLA": [np.array([9, 10], dtype=np.int32)]},
            row_slice=slice(0, 1),
            hdu=1,
            mmap=True,
        )

    complex_only_path = tmp_path / "astropy_complex_mmap_unsupported.fits"
    complex_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ID", format="J", array=np.array([1], np.int32)),
            fits.Column(
                name="Z",
                format="C",
                array=np.array([1 + 2j], np.complex64),
            ),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), complex_hdu]).writeto(
        complex_only_path, overwrite=True
    )

    complex_table = torchfits.read_table(
        complex_only_path.as_posix(), hdu=1, mmap=True
    )
    np.testing.assert_allclose(
        complex_table["Z"].squeeze(-1).numpy(),
        np.array([1 + 2j], dtype=np.complex64),
    )

    with pytest.raises(ValueError, match="mmap table updates only support"):
        torchfits.table.update_rows(
            complex_only_path.as_posix(),
            {"Z": np.array([3 + 4j], dtype=np.complex64)},
            row_slice=slice(0, 1),
            hdu=1,
            mmap=True,
        )
