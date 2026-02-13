import os

import numpy as np
import torch
from astropy.io import fits

import torchfits


def test_writing():
    filename = "test_write.fits"

    # Create TensorHDU
    data_tensor = torch.randn(10, 10)
    header_tensor = torchfits.Header({"TESTKEY": "TESTVAL"})
    hdu_tensor = torchfits.TensorHDU(data=data_tensor, header=header_tensor)

    # Create TableHDU
    data_table = {
        "col1": torch.tensor([1, 2, 3], dtype=torch.int32),
        "col2": torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32),
    }
    header_table = torchfits.Header({"TBLKEY": "TBLVAL"})
    hdu_table = torchfits.TableHDU(data_table, header=header_table)

    # Create HDUList
    hdul = torchfits.HDUList([hdu_tensor, hdu_table])

    def assert_written(path: str) -> None:
        with fits.open(path) as hdul_astro:
            # Verify TensorHDU (Primary)
            # Handle endianness for PyTorch
            data_numpy = hdul_astro[0].data
            if data_numpy.dtype.byteorder == ">":
                data_numpy = data_numpy.astype(data_numpy.dtype.newbyteorder("<"))
            data_read = torch.tensor(data_numpy)
            assert torch.allclose(data_read, data_tensor)
            assert hdul_astro[0].header["TESTKEY"] == "TESTVAL"

            # Verify TableHDU (Extension 1)
            # Note: Astropy might read as BinTableHDU
            data_read_table = hdul_astro[1].data
            assert np.allclose(data_read_table["col1"], data_table["col1"].numpy())
            assert np.allclose(data_read_table["col2"], data_table["col2"].numpy())
            assert hdul_astro[1].header["TBLKEY"] == "TBLVAL"

    filenames = [filename, "test_write_hdulist.fits"]

    try:
        # Write via HDUList
        hdul.write(filenames[0], overwrite=True)
        assert_written(filenames[0])

        # Write via top-level helper
        torchfits.write(filenames[1], hdul, overwrite=True)
        assert_written(filenames[1])
    finally:
        for path in filenames:
            if os.path.exists(path):
                os.remove(path)


def test_table_write_bool_preserved():
    filename = "test_write_bool_table.fits"
    data_table = {"flag": torch.tensor([True, False, True], dtype=torch.bool)}
    try:
        torchfits.write(filename, data_table, overwrite=True)
        with fits.open(filename) as hdul:
            values = hdul[1].data["flag"]
            assert values.tolist() == [True, False, True]
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_write_accepts_list_table_dict():
    filename = "test_write_list_table_dict.fits"
    table = {"flag": [True, False, True]}
    try:
        torchfits.write(filename, table, overwrite=True)
        with fits.open(filename) as hdul:
            values = hdul[1].data["flag"]
            assert values.tolist() == [True, False, True]
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_table_write_rich_types_pending_cfitsio_impl():
    filename = "test_write_rich_table.fits"
    table = {
        "ID": np.array([1, 2, 3], dtype=np.int32),
        "NAME": ["alpha", "beta", "gamma"],
        "Z": np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64),
        "VLA": [
            np.array([1, 2], dtype=np.int32),
            np.array([3], dtype=np.int32),
            np.array([4, 5, 6], dtype=np.int32),
        ],
    }
    try:
        torchfits.write(filename, table, header={"EXTNAME": "CATALOG"}, overwrite=True)
        with torchfits.open(filename) as hdul:
            table_hdu = hdul[1]
            assert table_hdu.header.get("EXTNAME") == "CATALOG"
            assert table_hdu.get_string_column("NAME") == ["alpha", "beta", "gamma"]
            vals = table_hdu["Z"].squeeze(-1)
            assert np.allclose(
                vals.numpy(), np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
            )
            vla = table_hdu.get_vla_column("VLA")
            assert [v.tolist() for v in vla] == [[1, 2], [3], [4, 5, 6]]
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_table_write_complex_tensor_pending_cfitsio_impl():
    filename = "test_write_complex_tensor_table.fits"
    table = {
        "ID": torch.tensor([1, 2, 3], dtype=torch.int32),
        "Z": torch.tensor([1 + 2j, 3 + 4j, 5 + 6j], dtype=torch.complex64),
    }
    try:
        torchfits.write(filename, table, overwrite=True)
        with torchfits.open(filename) as hdul:
            vals = hdul[1]["Z"].squeeze(-1)
            assert np.allclose(
                vals.numpy(), np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
            )
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_write_compressed_image_roundtrip():
    filename = "test_write_compressed_image.fits"
    # Use an integer image to ensure lossless Rice compression.
    data = torch.arange(64 * 64, dtype=torch.int16).reshape(64, 64)
    try:
        torchfits.write(filename, data, overwrite=True, compress=True)
        out = torchfits.read(filename, hdu=1)
        assert torch.equal(out, data)
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_write_compressed_hdulist_mixed():
    filename = "test_write_compressed_hdulist_mixed.fits"
    image = torch.arange(16 * 16, dtype=torch.int16).reshape(16, 16)
    table = {"ID": np.array([1, 2, 3], dtype=np.int32), "NAME": ["a", "b", "c"]}
    hdul = torchfits.HDUList(
        [
            torchfits.TensorHDU(
                data=image, header=torchfits.Header({"EXTNAME": "SCI"})
            ),
            torchfits.TableHDU(table, header=torchfits.Header({"EXTNAME": "CAT"})),
        ]
    )
    try:
        torchfits.write(filename, hdul, overwrite=True, compress=True)
        with torchfits.open(filename) as opened:
            assert len(opened) == 3
            assert opened[1].header.get("EXTNAME") == "SCI"
            assert opened[2].header.get("EXTNAME") == "CAT"
        assert torch.equal(torchfits.read(filename, hdu=1), image)
        table_out = torchfits.read(filename, hdu=2)
        assert table_out["ID"].squeeze(-1).tolist() == [1, 2, 3]
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_write_compressed_hdulist_images_roundtrip():
    filename = "test_write_compressed_hdulist_images.fits"
    img0 = torch.arange(64, dtype=torch.int16).reshape(8, 8)
    img1 = torch.full((8, 8), 5, dtype=torch.int16)
    hdul = torchfits.HDUList(
        [
            torchfits.TensorHDU(
                data=img0, header=torchfits.Header({"EXTNAME": "SCI0"})
            ),
            torchfits.TensorHDU(
                data=img1, header=torchfits.Header({"EXTNAME": "SCI1"})
            ),
        ]
    )
    try:
        torchfits.write(filename, hdul, overwrite=True, compress=True)
        with torchfits.open(filename) as opened:
            assert len(opened) == 3
            assert opened[1].header.get("EXTNAME") == "SCI0"
            assert opened[2].header.get("EXTNAME") == "SCI1"
        assert torch.equal(torchfits.read(filename, hdu=1), img0)
        assert torch.equal(torchfits.read(filename, hdu=2), img1)
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_write_compressed_image_tuple_roundtrip():
    filename = "test_write_compressed_image_tuple.fits"
    img0 = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    img1 = torch.full((4, 4), 2, dtype=torch.int16)
    payload = (img0, img1)
    try:
        torchfits.write(filename, payload, overwrite=True, compress=True)
        assert torch.equal(torchfits.read(filename, hdu=1), img0)
        assert torch.equal(torchfits.read(filename, hdu=2), img1)
    finally:
        if os.path.exists(filename):
            os.remove(filename)
