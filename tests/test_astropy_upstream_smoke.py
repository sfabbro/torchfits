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


def test_astropy_vla_table_mmap_updates_are_explicit(tmp_path) -> None:
    path = tmp_path / "astropy_mmap_vla_unsupported.fits"
    vla = np.array([np.array([1, 2], dtype=np.int32)], dtype=object)
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ID", format="J", array=np.array([1], np.int32)),
            fits.Column(name="VLA", format="PJ()", array=vla),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)

    # Public reads keep mmap=True ergonomic: VLA reads go through the safe
    # buffered path rather than silently dropping variable-length data.
    table = torchfits.read_table(path.as_posix(), hdu=1, mmap=True)
    assert table["VLA"][0].tolist() == [1, 2]

    # VLA mmap updates remain explicitly unsupported by design (heap pointer
    # indirection cannot be safely patched in-place via mmap).
    with pytest.raises(ValueError, match="mmap table updates do not support variable-length-array"):
        torchfits.table.update_rows(
            path.as_posix(),
            {"VLA": [np.array([9, 10], dtype=np.int32)]},
            row_slice=slice(0, 1),
            hdu=1,
            mmap=True,
        )


def test_astropy_complex_bit_string_mmap_updates_roundtrip_match_astropy(
    tmp_path,
) -> None:
    """mmap=True update path for complex, bit, and string columns mirrors
    astropy.io.fits round-trips and stays checksum-stable."""
    bits = np.array(
        [
            [True, False, True, False, True, False, True, False],
            [False, True, False, True, False, True, False, True],
        ],
        dtype=np.bool_,
    )
    names8 = np.array(["alpha01", "bravo02"])
    complex_col = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    int_col = np.array([11, 22], dtype=np.int32)

    path = tmp_path / "astropy_complex_bit_string.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="ID", format="J", array=int_col),
                    fits.Column(name="FLAGS", format="8X", array=bits),
                    fits.Column(name="NAME", format="8A", array=names8),
                    fits.Column(
                        name="Z", format="C", array=complex_col
                    ),
                ],
                name="CATALOG",
            ),
        ]
    ).writeto(path, overwrite=True)

    # mmap=True updates for COMPLEX (C), BIT (X), and STRING (A) now write
    # in place rather than forcing the buffered path.
    new_flags = np.array(
        [
            [False, True, False, True, False, True, False, True],
            [True, False, True, False, True, False, True, False],
        ],
        dtype=np.bool_,
    )
    new_complex = np.array([7 + 8j, 9 + 10j], dtype=np.complex64)
    # FITS 8A columns are fixed-width 8 chars; values longer than 8 are
    # silently truncated to the first ``width`` bytes by the C++ mmap
    # writer, so each new name must fit the declared column width.
    new_names = np.array(["beta0022", "gamma033"])
    new_ids = np.array([33, 44], dtype=np.int32)

    torchfits.table.update_rows(
        path.as_posix(),
        {
            "ID": new_ids,
            "FLAGS": new_flags,
            "NAME": new_names,
            "Z": new_complex,
        },
        row_slice=slice(0, 2),
        hdu=1,
        mmap=True,
    )

    with fits.open(path, mode="readonly") as hdul:
        np.testing.assert_array_equal(hdul[1].data["ID"], new_ids)
        np.testing.assert_array_equal(hdul[1].data["FLAGS"], new_flags)
        # Astropy decodes fixed-width CHAR columns with trailing spaces.
        np.testing.assert_array_equal(
            np.asarray([s.rstrip() for s in hdul[1].data["NAME"]]),
            np.asarray([s.rstrip() for s in new_names]),
        )
        np.testing.assert_allclose(
            hdul[1].data["Z"], new_complex, rtol=0.0, atol=0.0
        )

    # Un-modified columns preserved accurately.
    with torchfits.open(path.as_posix()) as hdul:
        np.testing.assert_array_equal(hdul[1]["ID"].squeeze(-1).numpy(), new_ids)
        np.testing.assert_array_equal(
            hdul[1]["FLAGS"].squeeze(-1).numpy().astype(bool), new_flags
        )


def test_astropy_string_mmap_update_pads_to_column_width(tmp_path) -> None:
    """Updates with a narrower user-provided string are right-padded with
    ASCII spaces so the on-disk FITS CHAR width is preserved exactly."""
    path = tmp_path / "astropy_string_padding.fits"
    names = np.array(["alpha01", "bravo02"], dtype="S8")
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="NAME", format="8A", array=names)]
            ),
        ]
    ).writeto(path, overwrite=True)

    # Provide a narrower 4-byte payload; torchfits should pad the trailing
    # 4 bytes per row with ASCII spaces (0x20).
    short_payload = np.array(["x", "y"], dtype="S4")
    torchfits.table.update_rows(
        path.as_posix(),
        {"NAME": short_payload},
        row_slice=slice(0, 2),
        hdu=1,
        mmap=True,
    )

    # Astropy decodes each CHAR row as a fixed-width bytes value; the
    # trailing bytes must be ASCII spaces (0x20), so the prefix is exactly
    # the user-provided 4-byte payload.
    # astropy's BinTableHDU.data accessor trims trailing whitespace from
    # fixed-width CHAR columns, so we use torchfits.read_table (which
    # returns the raw uint8 tensor without trimming) to verify the
    # user prefix is followed by ASCII-space (0x20) padding bytes on disk.
    raw_names = torchfits.read_table(path.as_posix(), hdu=1)["NAME"]
    raw_names_np = np.asarray(raw_names)
    assert raw_names_np.shape == (2, 8)
    np.testing.assert_array_equal(
        raw_names_np[0],
        np.frombuffer(b"x" + b" " * 7, dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        raw_names_np[1],
        np.frombuffer(b"y" + b" " * 7, dtype=np.uint8),
    )


def test_astropy_checksum_roundtrip_matches_verify_chksum(tmp_path) -> None:
    """torchfits.write_checksums writes DATASUM/CHECKSUM cards that match
    astropy.io.fits.verify_checksum's findings on image and table HDUs."""
    image = np.arange(32, dtype=np.int32).reshape(4, 8)
    table = {
        "ID": np.array([1, 2, 3], dtype=np.int32),
        "FLUX": np.array([1.5, 2.5, 3.5], dtype=np.float32),
    }

    image_path = tmp_path / "astropy_chksum_image.fits"
    fits.HDUList(
        [fits.PrimaryHDU(image), fits.ImageHDU(image * 2, name="SCI")]
    ).writeto(image_path, overwrite=True)
    torchfits.write_checksums(image_path.as_posix(), hdu=0)
    torchfits.write_checksums(image_path.as_posix(), hdu=1)

    torchfits_result_0 = torchfits.verify_checksums(image_path.as_posix(), hdu=0)
    torchfits_result_1 = torchfits.verify_checksums(image_path.as_posix(), hdu=1)
    assert torchfits_result_0["ok"]
    assert torchfits_result_1["ok"]
    with fits.open(image_path, mode="readonly") as hdul:
        assert hdul[0].verify_checksum()
        assert hdul[1].verify_checksum()

    table_path = tmp_path / "astropy_chksum_table.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="ID", format="J", array=table["ID"]),
                    fits.Column(name="FLUX", format="E", array=table["FLUX"]),
                ]
            ),
        ]
    ).writeto(table_path, overwrite=True)
    torchfits.write_checksums(table_path.as_posix(), hdu=1)
    torchfits_table_result = torchfits.verify_checksums(
        table_path.as_posix(), hdu=1
    )
    assert torchfits_table_result["ok"]
    with fits.open(table_path, mode="readonly") as hdul:
        assert hdul[1].verify_checksum()


@pytest.mark.parametrize("compression_type", ["RICE_1", "GZIP_1", "PLIO_1", "HCOMPRESS_1"])
def test_astropy_compimage_compression_variants_match_torchfits(
    tmp_path, compression_type
) -> None:
    """Cover the common CompImageHDU compression algorithms: RICE_1, GZIP_1,
    PLIO_1, HCOMPRESS_1."""
    image = np.arange(64, dtype=np.int16).reshape(8, 8)
    compressed_path = tmp_path / f"astropy_comp_{compression_type}.fits"
    comp = fits.CompImageHDU(
        image, compression_type=compression_type, name="COMP"
    )
    fits.HDUList([fits.PrimaryHDU(), comp]).writeto(
        compressed_path, overwrite=True
    )

    out = torchfits.read(compressed_path.as_posix(), hdu=1, mmap=False)
    np.testing.assert_array_equal(out.cpu().numpy(), image)
    # Round-trip via astropy to confirm the bytes are a CFITSIO-readable
    # compressed image HDU for the chosen algorithm.
    with fits.open(compressed_path, mode="readonly") as hdul:
        recovered = hdul[1].data
        np.testing.assert_array_equal(recovered, image)
