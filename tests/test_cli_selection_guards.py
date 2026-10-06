"""CLI selection and classification guards (round 2, unit 5).

Three user-facing contracts the CLI used to violate, all rooted in a
degenerate endpoint that a sibling in the same package already refused:

* **R2-033** -- ``hdu_type_name`` classified a tile-compressed image as a
  ``TABLE`` because a compressed HDU is a ``BINTABLE`` of tiles. ``info``/
  ``probe`` then mislabelled it, ``stats`` printed nothing at all while
  exiting 0, ``table`` dumped the internal tile columns, and ``arith`` /
  ``compress --split hdu`` refused with "no image HDUs to process" -- so
  ``torchfits compress`` could not re-compress its own output.
* **R2-034** -- ``-e/--hdu`` accepted a repeated index, so a selection's
  length no longer matched the selected HDUs: ``arith -e 0,0`` wrote a 2-HDU
  MEF from a 1-HDU selection. The sibling batch guards
  (``ensure_unique_basenames`` / ``ensure_unique_split_stems``) already
  refuse the duplicate analogue.
* **R2-035** -- ``cutout --box`` validated the box against itself but never
  against the image, so a box wholly outside it wrote a valid-looking 0x0
  product and exited 0. ``_parse_box`` already refused the same
  "selects no pixels" condition for a degenerate box.

The compressed-image fixture is built by ``astropy``'s ``CompImageHDU``:
that is the *standard* layout (an empty primary plus a ``BINTABLE`` of
tiles), the same structure ``torchfits compress`` emits.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from astropy.io import fits

import torchfits
from torchfits.cli.common import UsageError, hdu_type_name, parse_hdu_list

EXIT_OK = 0
EXIT_USAGE = 2

# Helpers introduced by these guards (is_compressed_image_header,
# compressed_image_geometry, cmds_cutout._plane_shape /
# _check_box_intersects_image, cmds_setkey._parse_hdus) are imported inside
# the tests that use them, not at module scope, so this file still *collects*
# against pre-fix sources: a module-level import of a name that does not exist
# yet aborts collection, and a collection error is not evidence that a guard
# works. Only pre-existing names are imported at module scope.


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def _ramp(n: int = 64) -> np.ndarray:
    return np.tile(np.linspace(0.0, 10.0, n, dtype=np.float32), (n, 1))


@pytest.fixture(scope="module")
def image(tmp_path_factory) -> str:
    """A plain (uncompressed) 64x64 float32 image."""
    path = tmp_path_factory.mktemp("sel") / "plain.fits"
    torchfits.write(str(path), torch.from_numpy(_ramp()), overwrite=True)
    return str(path)


@pytest.fixture(scope="module")
def compressed(tmp_path_factory) -> str:
    """A tile-compressed 64x64 image in the standard astropy layout."""
    path = tmp_path_factory.mktemp("sel") / "comp.fits"
    fits.HDUList(
        [fits.PrimaryHDU(), fits.CompImageHDU(_ramp(), compression_type="RICE_1")]
    ).writeto(path, overwrite=True)
    return str(path)


@pytest.fixture(scope="module")
def catalog(tmp_path_factory) -> str:
    """A genuine BINTABLE catalog (2 columns, 5 rows) -- not compressed."""
    path = tmp_path_factory.mktemp("sel") / "cat.fits"
    cols = fits.ColDefs(
        [
            fits.Column(name="A", format="J", array=np.arange(5)),
            fits.Column(name="B", format="E", array=np.arange(5.0)),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
        path, overwrite=True
    )
    return str(path)


@pytest.fixture(scope="module")
def mef(tmp_path_factory) -> str:
    """Two image HDUs, so ``-e 0,1`` is a legal distinct selection."""
    path = tmp_path_factory.mktemp("sel") / "mef.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(np.arange(6, dtype=np.float32).reshape(2, 3)),
            fits.ImageHDU(np.ones((2, 2), np.float32), name="SCI"),
        ]
    ).writeto(path, overwrite=True)
    return str(path)


# --------------------------------------------------------------------------
# R2-033: a tile-compressed image is an IMAGE, not a catalog
# --------------------------------------------------------------------------


def test_compressed_image_header_is_detected(compressed):
    from torchfits.cli.common import is_compressed_image_header

    header = torchfits.read_header(compressed, 1)
    assert str(header.get("XTENSION", "")).strip().upper() == "BINTABLE"
    assert is_compressed_image_header(header) is True


def test_compression_cards_alone_identify_a_compressed_image():
    """ZIMAGE is the marker, but the Z* cards are the same signal."""
    from torchfits.cli.common import is_compressed_image_header

    header = {"XTENSION": "BINTABLE", "ZCMPTYPE": "RICE_1"}
    assert is_compressed_image_header(header) is True


@pytest.mark.parametrize(
    "header",
    [
        {"XTENSION": "BINTABLE"},
        {"XTENSION": "BINTABLE", "ZIMAGE": False},
        {"NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4, "BITPIX": -32},
        {"XTENSION": "IMAGE"},
    ],
)
def test_plain_headers_are_not_compressed_images(header):
    from torchfits.cli.common import is_compressed_image_header

    assert is_compressed_image_header(header) is False


def test_hdu_type_name_reports_compressed_image_as_image(compressed):
    with torchfits.open(compressed) as hdul:
        assert hdu_type_name(hdul[1].header, hdul[1]) == "IMAGE"


def test_hdu_type_name_still_reports_a_real_catalog_as_table(catalog):
    with torchfits.open(catalog) as hdul:
        assert hdu_type_name(hdul[1].header, hdul[1]) == "TABLE"


def test_hdu_type_name_still_reports_a_plain_image_as_image(image):
    with torchfits.open(image) as hdul:
        assert hdu_type_name(hdul[0].header, hdul[0]) == "IMAGE"


def test_info_labels_a_compressed_image_as_an_image(compressed):
    result = _run_cli("info", compressed)
    assert result.returncode == EXIT_OK, result.stderr
    assert "type='TABLE'" not in result.stdout
    assert "type='IMAGE'" in result.stdout


def test_info_reports_the_image_geometry_not_the_tile_table(compressed):
    """The ZNAXIS/ZBITPIX cards, not the tile table's NAXIS/BITPIX."""
    result = _run_cli("info", compressed)
    assert result.returncode == EXIT_OK, result.stderr
    line = next(
        line
        for line in result.stdout.splitlines()
        if "hdu=1" in line and "type=" in line
    )
    assert "shape='(64, 64)'" in line
    assert "dtype='float32'" in line
    # The tile table's own geometry would be (64, 32) uint8.
    assert "(64, 32)" not in line
    assert "uint8" not in line


def test_stats_reports_statistics_for_a_compressed_image(compressed, image):
    """Before the fix `stats` printed nothing and still exited 0."""
    result = _run_cli("stats", compressed)
    assert result.returncode == EXIT_OK, result.stderr
    assert result.stdout.strip(), "stats emitted no record for a compressed image"
    assert "shape=[64, 64]" in result.stdout
    assert "dtype='float32'" in result.stdout
    # Same numbers as the uncompressed source (tile quantization only moves
    # the minimum by ~1e-9, so compare the robust statistics).
    plain = _run_cli("stats", image)
    assert "max=10.0" in result.stdout and "max=10.0" in plain.stdout
    assert "mean=5.0" in result.stdout and "mean=5.0" in plain.stdout


def test_table_does_not_dump_the_internal_tile_columns(compressed):
    result = _run_cli("table", compressed)
    assert result.returncode == EXIT_OK, result.stderr
    assert "COMPRESSED_DATA" not in result.stdout
    assert "GZIP_COMPRESSED_DATA" not in result.stdout
    assert "ZSCALE" not in result.stdout


def test_table_still_describes_a_real_catalog(catalog):
    result = _run_cli("table", catalog, "-e", "1", "-n", "2")
    assert result.returncode == EXIT_OK, result.stderr
    assert "A:" in result.stdout and "B:" in result.stdout


def test_arith_works_on_a_compressed_image(compressed, tmp_path):
    out = tmp_path / "arith.fits"
    result = _run_cli(
        "arith", compressed, "--op", "add", "--value", "1", "-e", "1", "-o", str(out)
    )
    assert result.returncode == EXIT_OK, result.stderr
    assert out.exists()


def test_compress_can_recompress_its_own_output(image, tmp_path):
    """`compress` refused a file it had just written: no image HDUs found."""
    once = tmp_path / "once.fits"
    assert _run_cli("compress", image, "-o", str(once)).returncode == EXIT_OK
    twice_dir = tmp_path / "twice"
    result = _run_cli(
        "compress", str(once), "--split", "hdu", "--out-dir", str(twice_dir)
    )
    assert result.returncode == EXIT_OK, result.stderr
    assert sorted(p.name for p in twice_dir.iterdir())


def test_compressed_image_geometry_reads_the_z_cards(compressed):
    from torchfits.cli.common import compressed_image_geometry

    header = torchfits.read_header(compressed, 1)
    assert compressed_image_geometry(header) == ((64, 64), "float32")


def test_compressed_image_geometry_is_none_for_a_plain_image(image):
    from torchfits.cli.common import compressed_image_geometry

    assert compressed_image_geometry(torchfits.read_header(image, 0)) is None


def test_compressed_cube_geometry_uses_the_z_axis_count():
    """The Z axis count differs from the tile table's for a cube.

    For a 2-D compressed image both counts are 2, so reading ``NAXIS`` for the
    axis count happens to agree; a 3-4-5 cube has ``ZNAXIS=3`` against the tile
    table's ``NAXIS=2``, which is what makes the wrong card observable.
    """
    from torchfits.cli.common import compressed_image_geometry

    cube = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    path = Path(tempfile.mkdtemp()) / "cube.fits"
    fits.HDUList(
        [fits.PrimaryHDU(), fits.CompImageHDU(cube, compression_type="RICE_1")]
    ).writeto(path, overwrite=True)
    header = torchfits.read_header(str(path), 1)
    assert int(header["ZNAXIS"]) == 3
    assert int(header["NAXIS"]) == 2
    assert compressed_image_geometry(header) == ((5, 4, 3), "float32")


# --------------------------------------------------------------------------
# R2-034: -e/--hdu must not repeat an index
# --------------------------------------------------------------------------


def test_parse_hdu_list_rejects_a_repeated_index():
    with pytest.raises(UsageError, match="duplicate HDU index"):
        parse_hdu_list("0,0")


def test_parse_hdu_list_accepts_distinct_indices():
    assert parse_hdu_list("0,1") == [0, 1]
    assert parse_hdu_list(None) is None


def test_parse_hdu_list_still_skips_empty_pieces():
    """Empty pieces were already skipped; only repeats are new refusals."""
    assert parse_hdu_list("0,,1") == [0, 1]
    assert parse_hdu_list(" 0 , 1 ") == [0, 1]


def test_parse_hdu_list_reports_the_first_repeated_index():
    with pytest.raises(UsageError, match=r"duplicate HDU index in --hdu: 1"):
        parse_hdu_list("0,1,1")


def test_parse_hdu_list_still_rejects_a_bad_index():
    with pytest.raises(UsageError, match="invalid HDU index"):
        parse_hdu_list("0,x")


def test_arith_repeated_index_no_longer_writes_a_multihdu_mef(mef, tmp_path):
    out = tmp_path / "dup.fits"
    result = _run_cli(
        "arith", mef, "--op", "add", "--value", "1", "-e", "0,0", "-o", str(out)
    )
    assert result.returncode == EXIT_USAGE
    assert "duplicate HDU index" in result.stderr
    assert not out.exists(), "a refused selection must not write an output"


def test_arith_distinct_indices_still_write_one_hdu_each(mef, tmp_path):
    out = tmp_path / "distinct.fits"
    result = _run_cli(
        "arith", mef, "--op", "add", "--value", "1", "-e", "0,1", "-o", str(out)
    )
    assert result.returncode == EXIT_OK, result.stderr
    assert len(fits.open(out)) == 2


@pytest.mark.parametrize("command", ["info", "stats", "verify"])
def test_inventory_commands_reject_a_repeated_index(command, mef):
    result = _run_cli(command, mef, "-e", "0,0")
    assert result.returncode == EXIT_USAGE
    assert "duplicate HDU index" in result.stderr


def test_inventory_commands_still_accept_distinct_indices(mef):
    result = _run_cli("info", mef, "-e", "0,1")
    assert result.returncode == EXIT_OK, result.stderr
    assert result.stdout.count("hdu=") == 2


def test_compress_split_hdu_rejects_a_repeated_index(mef, tmp_path):
    result = _run_cli(
        "compress",
        mef,
        "--split",
        "hdu",
        "--out-dir",
        str(tmp_path / "sd"),
        "-e",
        "1,1",
    )
    assert result.returncode == EXIT_USAGE
    assert "duplicate HDU index" in result.stderr


def test_setkey_rejects_a_repeated_index(mef, tmp_path):
    result = _run_cli(
        "setkey",
        mef,
        "-k",
        "K",
        "--value",
        "1",
        "-e",
        "0,0",
        "-o",
        str(tmp_path / "k.fits"),
    )
    assert result.returncode == EXIT_USAGE
    assert "duplicate HDU index" in result.stderr


def test_setkey_still_accepts_all(mef, tmp_path):
    result = _run_cli(
        "setkey",
        mef,
        "-k",
        "K",
        "--value",
        "1",
        "-e",
        "all",
        "-o",
        str(tmp_path / "k.fits"),
    )
    assert result.returncode == EXIT_OK, result.stderr


def test_setkey_accepts_two_distinct_indices(mef, tmp_path):
    """The duplicate guard must not refuse a legal two-HDU selection."""
    out = tmp_path / "both.fits"
    result = _run_cli(
        "setkey", mef, "-k", "K", "--value", "1", "-e", "0,1", "-o", str(out)
    )
    assert result.returncode == EXIT_OK, result.stderr
    for index in (0, 1):
        assert torchfits.read_header(str(out), index).get("K") == 1


def test_setkey_parse_hdus_guards_duplicates():
    from torchfits.cli.cmds_setkey import _parse_hdus

    assert _parse_hdus("all", 2) == [0, 1]
    assert _parse_hdus("0,1", 2) == [0, 1]
    with pytest.raises(UsageError, match="duplicate HDU index"):
        _parse_hdus("0,0", 2)


def test_setkey_parse_hdus_still_rejects_out_of_range():
    from torchfits.cli.cmds_setkey import _parse_hdus

    with pytest.raises(UsageError, match="out of range"):
        _parse_hdus("0,9", 2)


# --------------------------------------------------------------------------
# R2-035: --box must intersect the image
# --------------------------------------------------------------------------


def test_cutout_box_entirely_outside_the_image_is_refused(image, tmp_path):
    out = tmp_path / "outside.fits"
    result = _run_cli("cutout", image, "--box", "100,100,120,120", "-o", str(out))
    assert result.returncode == EXIT_USAGE
    assert "lies outside" in result.stderr
    assert not out.exists()


def test_cutout_box_starting_exactly_at_the_edge_is_refused(image, tmp_path):
    """A half-open box starting at NAXIS1 selects nothing."""
    out = tmp_path / "edge.fits"
    result = _run_cli("cutout", image, "--box", "64,64,80,80", "-o", str(out))
    assert result.returncode == EXIT_USAGE
    assert not out.exists()


def test_cutout_box_past_the_edge_still_clamps(image, tmp_path):
    """The documented clamp is preserved: a box over the whole image clips."""
    out = tmp_path / "clamped.fits"
    result = _run_cli("cutout", image, "--box", "0,0,100,100", "-o", str(out))
    assert result.returncode == EXIT_OK, result.stderr
    assert fits.getdata(out).shape == (64, 64)


def test_cutout_partially_outside_box_keeps_the_overlap(image, tmp_path):
    out = tmp_path / "partial.fits"
    result = _run_cli("cutout", image, "--box", "62,62,100,100", "-o", str(out))
    assert result.returncode == EXIT_OK, result.stderr
    assert fits.getdata(out).shape == (2, 2)


def test_cutout_fully_inside_box_is_unaffected(image, tmp_path):
    out = tmp_path / "inside.fits"
    result = _run_cli("cutout", image, "--box", "1,1,3,3", "-o", str(out))
    assert result.returncode == EXIT_OK, result.stderr
    assert fits.getdata(out).shape == (2, 2)


def test_cutout_degenerate_box_is_still_refused(image, tmp_path):
    out = tmp_path / "degenerate.fits"
    result = _run_cli("cutout", image, "--box", "2,2,2,2", "-o", str(out))
    assert result.returncode == EXIT_USAGE
    assert "selects no pixels" in result.stderr


def test_plane_shape_reads_the_trailing_axes(image):
    from torchfits.cli.cmds_cutout import _plane_shape

    header = torchfits.read_header(image, 0)
    assert _plane_shape(header) == (64, 64)


def test_plane_shape_is_none_without_a_2d_plane():
    from torchfits.cli.cmds_cutout import _plane_shape

    assert _plane_shape({"NAXIS": 0, "NAXIS1": 0, "NAXIS2": 0}) is None
    assert _plane_shape({"NAXIS": 1, "NAXIS1": 8, "NAXIS2": 8}) is None


def test_plane_shape_handles_a_cube(tmp_path):
    """--box addresses the trailing (y, x) axes of a cube."""
    from torchfits.cli.cmds_cutout import _plane_shape

    path = tmp_path / "cube.fits"
    torchfits.write(str(path), torch.zeros(3, 8, 10), overwrite=True)
    assert _plane_shape(torchfits.read_header(str(path), 0)) == (8, 10)


def test_check_box_intersects_image_is_a_noop_without_a_plane():
    from torchfits.cli.cmds_cutout import _check_box_intersects_image

    # No shape to check against: the reader stays responsible, so the guard
    # must not invent a refusal.
    _check_box_intersects_image((0, 0, 4, 4), {"NAXIS": 0}, path="x", hdu=0)
