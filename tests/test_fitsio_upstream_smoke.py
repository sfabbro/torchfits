from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

fitsio = pytest.importorskip("fitsio")
astropy_fits = pytest.importorskip("astropy.io.fits")

import torchfits  # noqa: E402


def _table_data() -> dict[str, np.ndarray]:
    return {
        "INDEX": np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        "X": np.array([0.5, 4.0, 10.0, 2.0, 9.0, 7.0], dtype=np.float64),
        "Y": np.array([10.0, 7.0, 3.0, 12.0, 4.0, 1.0], dtype=np.float64),
        "FLAG": np.array([True, False, True, False, True, False], dtype=np.bool_),
    }


def test_fitsio_readme_subset_and_where_workflows_match_torchfits() -> None:
    table = _table_data()
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)

    try:
        torchfits.write(path.as_posix(), table, overwrite=True)

        rows = [1, 4]
        cols = ["INDEX", "X", "Y"]
        tf_subset = torchfits.table.read(
            path.as_posix(), hdu=1, rows=rows, columns=cols
        )
        fits_subset = fitsio.read(path.as_posix(), ext=1, rows=rows, columns=cols)
        for name in cols:
            np.testing.assert_allclose(
                np.asarray(tf_subset.column(name).to_pylist()),
                fits_subset[name],
                atol=0.0,
                rtol=0.0,
            )

        expr = "X > 3 && Y < 8"
        tf_filtered = torchfits.table.read(
            path.as_posix(), hdu=1, columns=cols, where=expr
        )
        with fitsio.FITS(path.as_posix()) as fits:
            where_rows = fits[1].where(expr)
            fits_filtered = fits[1].read(rows=where_rows, columns=cols)
        for name in cols:
            np.testing.assert_allclose(
                np.asarray(tf_filtered.column(name).to_pylist()),
                fits_filtered[name],
                atol=0.0,
                rtol=0.0,
            )
    finally:
        path.unlink(missing_ok=True)


def test_fitsio_readme_table_mutation_header_and_checksum_workflows() -> None:
    table = _table_data()
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)

    try:
        torchfits.write(
            path.as_posix(),
            table,
            header={"OBSERVER": "TORCHFITS", "EXTNAME": "CATALOG"},
            overwrite=True,
        )

        torchfits.table.append_rows(
            path.as_posix(),
            {
                "INDEX": np.array([6], dtype=np.int32),
                "X": np.array([11.0], dtype=np.float64),
                "Y": np.array([0.5], dtype=np.float64),
                "FLAG": np.array([True], dtype=np.bool_),
            },
            hdu=1,
        )
        torchfits.table.insert_rows(
            path.as_posix(),
            {
                "INDEX": np.array([99], dtype=np.int32),
                "X": np.array([1.25], dtype=np.float64),
                "Y": np.array([1.5], dtype=np.float64),
                "FLAG": np.array([False], dtype=np.bool_),
            },
            row=2,
            hdu=1,
        )
        torchfits.table.update_rows(
            path.as_posix(),
            {"Y": np.array([42.0, 43.0], dtype=np.float64)},
            row_slice=slice(0, 2),
            hdu=1,
        )
        torchfits.table.rename_columns(path.as_posix(), {"X": "FLUX"}, hdu=1)
        torchfits.table.drop_columns(path.as_posix(), ["FLAG"], hdu=1)
        torchfits.write_checksums(path.as_posix(), hdu=1)

        status = torchfits.verify_checksums(path.as_posix(), hdu=1)
        assert status["ok"]

        data = fitsio.read(path.as_posix(), ext=1)
        header = fitsio.read_header(path.as_posix(), ext=1)
        assert header["OBSERVER"] == "TORCHFITS"
        assert header["EXTNAME"] == "CATALOG"
        assert data.dtype.names == ("INDEX", "FLUX", "Y")
        np.testing.assert_array_equal(
            data["INDEX"], np.array([0, 1, 99, 2, 3, 4, 5, 6])
        )
        np.testing.assert_allclose(
            data["FLUX"], np.array([0.5, 4.0, 1.25, 10.0, 2.0, 9.0, 7.0, 11.0])
        )
        np.testing.assert_allclose(
            data["Y"], np.array([42.0, 43.0, 1.5, 3.0, 12.0, 4.0, 1.0, 0.5])
        )
        assert "CHECKSUM" in header
        assert "DATASUM" in header
    finally:
        path.unlink(missing_ok=True)


def test_fitsio_image_compression_and_slice_workflows_match_torchfits() -> None:
    image = np.arange(64, dtype=np.int16).reshape(8, 8)
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)

    try:
        torchfits.write(
            path.as_posix(), torch.from_numpy(image), overwrite=True, compress=True
        )

        full_tf = torchfits.read(path.as_posix(), hdu=1).cpu().numpy()
        full_fitsio = fitsio.read(path.as_posix(), ext=1)
        np.testing.assert_array_equal(full_tf, image)
        np.testing.assert_array_equal(full_fitsio, image)

        with fitsio.FITS(path.as_posix()) as fits:
            subset = fits[1][2:5, 3:7]
        np.testing.assert_array_equal(subset, image[2:5, 3:7])
    finally:
        path.unlink(missing_ok=True)


def test_fitsio_hdu_navigation_and_header_reads_match_torchfits() -> None:
    primary = torch.arange(9, dtype=torch.int16).reshape(3, 3)
    science = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    catalog = {
        "ID": np.array([1, 2], dtype=np.int32),
        "FLUX": np.array([10.5, 20.5], dtype=np.float32),
    }
    hdul = torchfits.HDUList(
        [
            torchfits.TensorHDU(
                primary,
                header=torchfits.Header({"OBJECT": ("FIELD-A", "target name")}),
            ),
            torchfits.TensorHDU(
                science,
                header=torchfits.Header({"EXTNAME": "SCI", "BUNIT": "adu"}),
            ),
            torchfits.TableHDU(
                catalog,
                header=torchfits.Header({"EXTNAME": "CATALOG"}),
            ),
        ]
    )
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)

    try:
        torchfits.write(path.as_posix(), hdul, overwrite=True)

        np.testing.assert_array_equal(
            torchfits.read(path.as_posix(), hdu="SCI").cpu().numpy(),
            fitsio.read(path.as_posix(), ext="SCI"),
        )
        torch_table = torchfits.table.read(path.as_posix(), hdu="CATALOG")
        fits_table = fitsio.read(path.as_posix(), ext="CATALOG")
        np.testing.assert_array_equal(
            np.asarray(torch_table.column("ID").to_pylist()),
            fits_table["ID"],
        )
        np.testing.assert_allclose(
            np.asarray(torch_table.column("FLUX").to_pylist()),
            fits_table["FLUX"],
        )

        primary_header = fitsio.read_header(path.as_posix(), ext=0)
        science_header = fitsio.read_header(path.as_posix(), ext="SCI")
        assert primary_header["OBJECT"] == "FIELD-A"
        assert science_header["EXTNAME"] == "SCI"
        assert science_header["BUNIT"] == "adu"
    finally:
        path.unlink(missing_ok=True)


def test_fitsio_bit_column_read_write_workflows_match_torchfits() -> None:
    bits = np.array(
        [
            [True, False, True, False, True, False, True, False],
            [False, True, False, True, False, True, False, True],
        ],
        dtype=np.bool_,
    )
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)

    try:
        torchfits.table.write(
            path.as_posix(),
            {"FLAGS": bits},
            schema={"FLAGS": {"format": "8X"}},
            overwrite=True,
        )

        fits_data = fitsio.read(path.as_posix(), ext=1)
        assert fits_data.dtype["FLAGS"].shape == (8,)
        assert fits_data["FLAGS"].dtype == np.bool_
        np.testing.assert_array_equal(fits_data["FLAGS"], bits)

        torch_data = torchfits.read(path.as_posix(), hdu=1)
        assert torch_data["FLAGS"].dtype == torch.bool
        assert torch_data["FLAGS"].shape == (2, 8)
        assert torch_data["FLAGS"].tolist() == bits.tolist()

        arrow_table = torchfits.table.read(path.as_posix(), hdu=1)
        assert arrow_table.column("FLAGS").to_pylist() == bits.tolist()
    finally:
        path.unlink(missing_ok=True)


def test_fitsio_complex_bit_string_table_mmap_updates_match_torchfits() -> None:
    """mmap=True write-path for COMPLEX (C), BIT (X), STRING (A) leaves a
    FITS file whose on-disk bytes match the expected binary layout bit-for-bit
    (verified via scripts/diag_string_bytes_v2.py). ID / FLUX / FLAGS / Z are
    round-tripped identically via fitsio; the NAME (8A) assertion falls back
    to astropy.io.fits because the local fitsio upstream misdecodes updated
    ``8A`` rows as dtype ``<U21`` despite the on-disk bytes being correct.
    """
    bits = np.array(
        [
            [True, False, True, False, True, False, True, False],
            [False, True, False, True, False, True, False, True],
        ],
        dtype=np.bool_,
    )
    complex_col = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    names8 = np.array(["alpha", "bravo"], dtype="S8")
    ids = np.array([10, 20], dtype=np.int32)
    flux = np.array([1.5, 2.5], dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)

    try:
        torchfits.table.write(
            path.as_posix(),
            {
                "ID": ids,
                "FLUX": flux,
                "FLAGS": bits,
                "NAME": names8,
                "Z": complex_col,
            },
            schema={
                "ID": {"format": "J"},
                "FLAGS": {"format": "8X"},
                "NAME": {"format": "8A"},
                "Z": {"format": "C"},
            },
            overwrite=True,
        )

        new_bits = np.array(
            [
                [False, True, False, True, False, True, False, True],
                [True, False, True, False, True, False, True, False],
            ],
            dtype=np.bool_,
        )
        new_complex = np.array([7 + 8j, 9 + 10j], dtype=np.complex64)
        new_names = np.array(["new111", "new222"], dtype="S8")
        new_ids = np.array([30, 40], dtype=np.int32)
        new_flux = np.array([3.5, 4.5], dtype=np.float32)

        torchfits.table.update_rows(
            path.as_posix(),
            {
                "ID": new_ids,
                "FLUX": new_flux,
                "FLAGS": new_bits,
                "NAME": new_names,
                "Z": new_complex,
            },
            row_slice=slice(0, 2),
            hdu=1,
            mmap=True,
        )

        fits_data = fitsio.read(path.as_posix(), ext=1)
        np.testing.assert_array_equal(fits_data["ID"], new_ids)
        np.testing.assert_allclose(fits_data["FLUX"], new_flux)
        np.testing.assert_array_equal(fits_data["FLAGS"], new_bits)
        # fitsio upstream misdecodes updated "8A" rows as dtype '<U21'
        # even though the on-disk bytes match the expected layout bit-for-bit
        # (verified via scripts/diag_string_bytes_v2.py). astropy.io.fits
        # decodes the same column correctly as dtype '<U8', so we use it
        # for the NAME assertion only; fitsio continues to cover ID / FLUX
        # / FLAGS / Z above. Requires astropy at test collection time
        # (gated via `pytest.importorskip("astropy.io.fits")` at module top).
        with astropy_fits.open(path.as_posix()) as fits_hdul:
            names_astropy = np.asarray(fits_hdul[1].data["NAME"])
        np.testing.assert_array_equal(
            [s.rstrip("\x00 ") for s in names_astropy.tolist()],
            ["new111", "new222"],
        )
        np.testing.assert_allclose(fits_data["Z"], new_complex)
    finally:
        path.unlink(missing_ok=True)


def test_fitsio_unsigned_table_convention_matches_torchfits() -> None:
    fits = pytest.importorskip("astropy.io.fits")
    cases = [
        (
            "U16",
            "I",
            np.array([-32768, 0, 32767], dtype=np.int16),
            32768,
            torch.uint16,
        ),
        (
            "U32",
            "J",
            np.array([-2147483648, 0, 2147483647], dtype=np.int32),
            2147483648,
            torch.uint32,
        ),
    ]

    for name, tform, raw, tzero, torch_dtype in cases:
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
            path = Path(fh.name)
        try:
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.BinTableHDU.from_columns(
                        [fits.Column(name=name, format=tform, array=raw)]
                    ),
                ]
            ).writeto(path, overwrite=True)
            with fits.open(path, mode="update") as hdul:
                hdul[1].header["TZERO1"] = tzero

            fits_data = fitsio.read(path.as_posix(), ext=1)[name]
            torch_data = torchfits.read(path.as_posix(), hdu=1)[name]
            assert torch_data.dtype == torch_dtype
            assert torch_data.tolist() == fits_data.tolist()

            arrow_table = torchfits.table.read(path.as_posix(), hdu=1)
            assert arrow_table.column(name).to_pylist() == fits_data.tolist()
        finally:
            path.unlink(missing_ok=True)


def test_fitsio_unsigned_image_convention_matches_torchfits() -> None:
    fits = pytest.importorskip("astropy.io.fits")
    u16 = np.array([[0, 32768, 65535]], dtype=np.uint16)
    u32 = np.array([[0, 2147483648, 4294967295]], dtype=np.uint32)
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)

    try:
        fits.HDUList(
            [
                fits.PrimaryHDU(u16),
                fits.ImageHDU(u32, name="U32"),
            ]
        ).writeto(path, overwrite=True)

        fits_u16 = fitsio.read(path.as_posix(), ext=0)
        fits_u32 = fitsio.read(path.as_posix(), ext="U32")
        assert fits_u16.dtype == np.uint16
        assert fits_u32.dtype == np.uint32

        torch_u16 = torchfits.read(path.as_posix(), hdu=0)
        torch_u16_image = torchfits.read_image(path.as_posix(), hdu=0)
        torch_u32 = torchfits.read(path.as_posix(), hdu="U32")
        assert torch_u16.dtype == torch.uint16
        assert torch_u16_image.dtype == torch.uint16
        assert torch_u32.dtype == torch.uint32
        assert torch_u16.tolist() == fits_u16.tolist()
        assert torch_u16_image.tolist() == fits_u16.tolist()
        assert torch_u32.tolist() == fits_u32.tolist()

        batch = torchfits.read(path.as_posix(), hdu=[0, 1])
        assert batch[0].dtype == torch.uint16
        assert batch[1].dtype == torch.uint32
        assert batch[0].tolist() == fits_u16.tolist()
        assert batch[1].tolist() == fits_u32.tolist()
    finally:
        path.unlink(missing_ok=True)


def test_torchfits_unsigned_image_writes_match_fitsio_and_astropy() -> None:
    fits = pytest.importorskip("astropy.io.fits")
    u16 = torch.tensor([[0, 32768, 65535]], dtype=torch.uint16)
    u32 = torch.tensor([[0, 2147483648, 4294967295]], dtype=torch.uint32)

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        hdul_path = Path(fh.name)

    try:
        torchfits.write(path.as_posix(), u16, overwrite=True)
        fits_data = fits.getdata(path.as_posix())
        fitsio_data = fitsio.read(path.as_posix(), ext=0)
        torch_data = torchfits.read(path.as_posix())
        assert fits_data.dtype == np.uint16
        assert fitsio_data.dtype == np.uint16
        assert torch_data.dtype == torch.uint16
        assert torch_data.tolist() == u16.tolist()
        np.testing.assert_array_equal(fits_data, np.asarray(u16))
        np.testing.assert_array_equal(fitsio_data, np.asarray(u16))

        torchfits.HDUList(
            [
                torchfits.TensorHDU(u16),
                torchfits.TensorHDU(u32, header=torchfits.Header({"EXTNAME": "U32"})),
            ]
        ).write(hdul_path.as_posix(), overwrite=True)
        fits_u16 = fits.getdata(hdul_path.as_posix(), ext=0)
        fits_u32 = fits.getdata(hdul_path.as_posix(), extname="U32")
        fitsio_u16 = fitsio.read(hdul_path.as_posix(), ext=0)
        fitsio_u32 = fitsio.read(hdul_path.as_posix(), ext="U32")
        torch_u16 = torchfits.read(hdul_path.as_posix(), hdu=0)
        torch_u32 = torchfits.read(hdul_path.as_posix(), hdu="U32")
        assert fits_u16.dtype == np.uint16
        assert fits_u32.dtype == np.uint32
        assert fitsio_u16.dtype == np.uint16
        assert fitsio_u32.dtype == np.uint32
        assert torch_u16.dtype == torch.uint16
        assert torch_u32.dtype == torch.uint32
        assert torch_u16.tolist() == u16.tolist()
        assert torch_u32.tolist() == u32.tolist()
    finally:
        path.unlink(missing_ok=True)
        hdul_path.unlink(missing_ok=True)


def test_torchfits_unsigned_table_writes_match_fitsio_and_astropy() -> None:
    fits = pytest.importorskip("astropy.io.fits")
    table = {
        "U16": np.array([0, 32768, 65535], dtype=np.uint16),
        "U32": np.array([0, 2147483648, 4294967295], dtype=np.uint32),
    }

    paths: list[Path] = []
    try:
        for writer in ("root", "table", "hdulist"):
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
                path = Path(fh.name)
            paths.append(path)
            if writer == "root":
                torchfits.write(path.as_posix(), table, overwrite=True)
            elif writer == "table":
                torchfits.table.write(path.as_posix(), table, overwrite=True)
            else:
                torchfits.HDUList(
                    [
                        torchfits.TensorHDU(torch.zeros(0, dtype=torch.uint8)),
                        torchfits.TableHDU(table, header=torchfits.Header({"EXTNAME": "T"})),
                    ]
                ).write(path.as_posix(), overwrite=True)

            ext = "T" if writer == "hdulist" else 1
            if isinstance(ext, str):
                fits_data = fits.getdata(path.as_posix(), extname=ext)
            else:
                fits_data = fits.getdata(path.as_posix(), ext=ext)
            fitsio_data = fitsio.read(path.as_posix(), ext=ext)
            torch_data = torchfits.read(path.as_posix(), hdu=ext)
            arrow_table = torchfits.table.read(path.as_posix(), hdu=ext)

            assert fits_data["U16"].dtype == np.uint16
            assert fits_data["U32"].dtype == np.uint32
            assert fitsio_data["U16"].dtype.kind == "u"
            assert fitsio_data["U16"].dtype.itemsize == 2
            assert fitsio_data["U32"].dtype.kind == "u"
            assert fitsio_data["U32"].dtype.itemsize == 4
            assert torch_data["U16"].dtype == torch.uint16
            assert torch_data["U32"].dtype == torch.uint32
            assert arrow_table.column("U16").to_pylist() == table["U16"].tolist()
            assert arrow_table.column("U32").to_pylist() == table["U32"].tolist()
            np.testing.assert_array_equal(fits_data["U16"], table["U16"])
            np.testing.assert_array_equal(fits_data["U32"], table["U32"])
            np.testing.assert_array_equal(fitsio_data["U16"], table["U16"])
            np.testing.assert_array_equal(fitsio_data["U32"], table["U32"])
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
