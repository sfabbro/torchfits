"""``os.PathLike`` must be accepted everywhere a FITS path is.

The write side always normalized with ``os.fspath``, but the read side rejected
``pathlib.Path`` outright: ``torchfits.write(Path(...))`` worked while
``torchfits.read(Path(...))`` raised ``ValueError: Path must be a string or
list of strings``. Any ``for p in root.glob('*.fits')`` loop hit that wall, and
``torchfits.data`` pipelines traffic in ``Path`` objects.

These tests assert equal *results* for ``Path`` and ``str`` input rather than
just "did not raise", so a future regression cannot pass by silently returning
something else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torchfits  # noqa: E402
import torchfits.table  # noqa: E402

IMAGE = torch.arange(4, dtype=torch.float32).reshape(2, 2)


@pytest.fixture()
def image_and_table(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "img.fits"
    table = tmp_path / "tbl.fits"
    torchfits.write(str(image), IMAGE, overwrite=True)
    torchfits.table.write(
        str(table),
        {"A": np.arange(3, dtype=np.int32), "S": np.array(["x", "y", "z"], dtype="U1")},
        overwrite=True,
    )
    return image, table


def test_read_data_matches_str(image_and_table):
    image, _ = image_and_table
    assert torch.equal(torchfits.read(image), torchfits.read(str(image)))
    assert torch.equal(torchfits.read(image), IMAGE)


def test_read_header_matches_str(image_and_table):
    image, _ = image_and_table
    assert dict(torchfits.read_header(image)) == dict(torchfits.read_header(str(image)))


def test_skinny_metadata_accepts_path(image_and_table):
    image, table = image_and_table
    assert torchfits.read_nrows(table) == torchfits.read_nrows(str(table)) == 3
    assert (
        torchfits.read_colnames(table, 1)
        == torchfits.read_colnames(str(table), 1)
        == ["A", "S"]
    )
    assert torchfits.read_table_info(table, 1)["nrows"] == 3
    assert torchfits.read_num_hdus(image) == torchfits.read_num_hdus(str(image))
    assert torchfits.read_shape(image) == torchfits.read_shape(str(image))
    assert torchfits.read_hdu_type(image) == torchfits.read_hdu_type(str(image))
    assert torchfits.read_keys(image, ["BITPIX"]) == torchfits.read_keys(
        str(image), ["BITPIX"]
    )
    assert torchfits.read_extname(image) == torchfits.read_extname(str(image))
    assert torchfits.read_batch_info([image])["num_files"] == 1


def test_read_tensor_and_hdus_accept_path(image_and_table):
    image, _ = image_and_table
    assert torch.equal(torchfits.read_tensor(image, hdu=0), IMAGE)
    (only,) = torchfits.read_hdus(image, [0])
    assert torch.equal(only, IMAGE)
    with torchfits.open(image) as hdul:
        assert hdul is not None


def test_table_read_accepts_path(image_and_table):
    _, table = image_and_table
    via_path = torchfits.table.read(table)
    via_str = torchfits.table.read(str(table))
    assert via_path["A"].to_pylist() == via_str["A"].to_pylist() == [0, 1, 2]
    # read_torch exposes string columns as uint8 byte tensors (one row per code).
    codes = torchfits.table.read_torch(table)["S"].tolist()
    assert [bytes(int(c) for c in row).decode() for row in codes] == ["x", "y", "z"]


def test_subset_and_persistent_handles_accept_path(image_and_table):
    image, table = image_and_table
    with torchfits.open_subset_reader(image) as reader:
        assert reader is not None
    assert torchfits.read_subset(image, 0, 0, 0, 1, 1).shape == (1, 1)
    with torchfits.open_table_reader(table) as handle:
        assert handle.num_rows() == 3


def test_batch_read_accepts_pathlib_entries(image_and_table):
    image, _ = image_and_table
    tensors = torchfits.read_batch([image, image])
    assert len(tensors) == 2
    assert all(torch.equal(t, IMAGE) for t in tensors)


def test_checksums_accept_path(image_and_table):
    image, _ = image_and_table
    torchfits.write_checksums(image)
    assert torchfits.verify_checksums(image)["status"] == "ok"


def test_table_hdu_from_fits_accepts_path(image_and_table):
    _, table = image_and_table
    hdu = torchfits.TableHDU.from_fits(table, 1)
    assert hdu.num_rows == 3


_MUTATIONS = [
    (
        "update_rows",
        lambda p: torchfits.table.update_rows(
            p, {"A": np.array([7, 8], dtype=np.int32)}, (0, 2)
        ),
    ),
    (
        "append_rows",
        lambda p: torchfits.table.append_rows(p, {"A": np.array([7], dtype=np.int32)}),
    ),
    (
        "insert_rows",
        lambda p: torchfits.table.insert_rows(
            p, {"A": np.array([7], dtype=np.int32)}, row=1
        ),
    ),
    ("delete_rows", lambda p: torchfits.table.delete_rows(p, (0, 1))),
    (
        "insert_column",
        lambda p: torchfits.table.insert_column(p, "B", np.arange(3, dtype=np.int32)),
    ),
    (
        "replace_column",
        lambda p: torchfits.table.replace_column(
            p, "A", np.arange(3, dtype=np.int32) * 2
        ),
    ),
    ("rename_columns", lambda p: torchfits.table.rename_columns(p, {"A": "R"})),
    ("drop_columns", lambda p: torchfits.table.drop_columns(p, ["S"])),
]


@pytest.mark.parametrize(
    "mutate", [m[1] for m in _MUTATIONS], ids=[m[0] for m in _MUTATIONS]
)
def test_table_mutations_accept_path(tmp_path, mutate):
    """The table mutation API resolved its HDU by calling the native opener
    directly, so every mutator raised ``TypeError`` for ``pathlib.Path`` while
    accepting ``str`` -- the same asymmetry as the read surface, one layer down.
    """
    via_str = tmp_path / "as_str.fits"
    via_path = tmp_path / "as_path.fits"
    for target in (via_str, via_path):
        torchfits.table.write(
            str(target),
            {
                "A": np.arange(3, dtype=np.int32),
                "S": np.array(["x", "y", "z"], dtype="U1"),
            },
            overwrite=True,
        )
    mutate(str(via_str))
    mutate(via_path)  # the only difference: pathlib.Path argument

    a = torchfits.table.read(via_str)
    b = torchfits.table.read(via_path)
    assert a.column_names == b.column_names
    for column in a.column_names:
        assert a[column].to_pylist() == b[column].to_pylist()


def test_non_path_arguments_still_rejected():
    """Widening to PathLike must not make the APIs accept arbitrary objects."""
    for bad in (None, 123, object()):
        with pytest.raises((TypeError, ValueError)):
            torchfits.read_nrows(bad)
