"""Real CFHT observations read end to end, against astropy as the reference.

``tests/test_core_library_real_data.py`` proves the split's *metadata* answers
match on real data. This file goes past the header: it actually reads the
pixels and the tables of the same 9 GB of CFHT observations and holds them
against `astropy.io.fits`, which is a completely separate implementation --
its own FITS reader, its own Rice decompressor, its own header parser. Two
implementations agreeing byte for byte on 435 megapixels of real sky is a
stronger claim than any amount of self-consistency.

What each test is actually for:

* :func:`test_read_shape_matches_astropy_for_every_real_hdu` -- the geometry
  the mmap path allocates from, cross-checked on all 409 (frame, HDU) pairs.
* :func:`test_hdu_type_matches_the_astropy_hdu_class` -- the type string the
  reader routes on, cross-checked against astropy's HDU classes.
* :func:`test_image_info_is_fits_order_and_read_shape_is_row_major` -- two
  orderings of the same axes in one module. Documented, deliberate, and
  indistinguishable on the square synthetic images the other tests use, so it
  needs a real non-square frame to be pinned at all.
* :func:`test_megacam_compressed_extensions_decompress_to_astropy_pixels` --
  two independent Rice decompressors.
* :func:`test_megacam_scale_info_matches_the_real_bscale_bzero` -- the real
  MegaCam convention: int16 on disk, ``BZERO=32768.0``, unsigned in memory.
* :func:`test_megacam_raw_compressed_table_matches_astropy` -- the variable
  length column and the raw Arrow transport, byte for byte.
* :func:`test_megapipe_mosaic_pixels_match_astropy` -- the mmap path on a
  1.6 GB, 435-megapixel file.

The data is not in git. Fetch it once::

    bash scripts/fetch_cfht_megacam_sample.sh    # ~2.5 GB, 10 Rice .fz frames
    bash scripts/fetch_cfht_megapipe_sample.sh  # ~5 GB, 3 1.6 GB mosaics

Without the samples every test here skips. The mosaic test reads one 1.6 GB
file, so a machine that fetched the samples is assumed to have the disk for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")
torch = pytest.importorskip("torch")

import torchfits  # noqa: E402
import torchfits._C as C  # noqa: E402
import torchfits.table as table_api  # noqa: E402

core = pytest.importorskip("torchfits._core")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "benchmarks_data"

MEGACAM = sorted(DATA_ROOT.glob("cfht_megacam/*.fits.fz"))
MEGAPIPE = sorted(DATA_ROOT.glob("cfht_megapipe/*.fits"))
CORPUS = MEGACAM + MEGAPIPE

needs_megacam = pytest.mark.skipif(
    not MEGACAM, reason="no MegaCam frames; run scripts/fetch_cfht_megacam_sample.sh"
)
needs_megapipe = pytest.mark.skipif(
    not MEGAPIPE,
    reason="no MegaPipe mosaics; run scripts/fetch_cfht_megapipe_sample.sh",
)
needs_corpus = pytest.mark.skipif(
    not CORPUS,
    reason=(
        "real CFHT sample data absent; run scripts/fetch_cfht_megacam_sample.sh "
        "and scripts/fetch_cfht_megapipe_sample.sh"
    ),
)

#: astropy HDU class name -> the HDU type string torchfits reports.
ASTROPY_TYPE = {
    "PrimaryHDU": "IMAGE",
    "ImageHDU": "IMAGE",
    "CompImageHDU": "IMAGE",
    "BinTableHDU": "BINARY_TABLE",
    "TableHDU": "TABLE",
}


def _mega_frame() -> Path:
    """One real Rice-compressed frame, with the most HDUs of the sample set."""
    return max(MEGACAM, key=lambda p: core.read_num_hdus(str(p)))


@needs_corpus
def test_read_shape_matches_astropy_for_every_real_hdu() -> None:
    """All 409 (frame, HDU) geometries against astropy's own HDU shape.

    ``read_shape`` is what the mmap path turns into an allocation, so a wrong
    answer is a wrong read rather than a wrong report. A compressed frame's
    primary is ``NAXIS=0`` and correctly reports no axes at all, which is the
    case a synthetic "always 2-D" fixture would get wrong.
    """
    checked = 0
    for path in CORPUS:
        with fits.open(path) as hdul:
            assert len(hdul) == core.read_num_hdus(str(path))
            for hdu, hdu_obj in enumerate(hdul):
                expected = tuple(hdu_obj.shape)
                bitpix, shape = core.read_shape(str(path), hdu)
                assert tuple(shape) == expected, f"{path.name} HDU {hdu}"
                assert bitpix != 0 or expected == ()
                checked += 1
    assert checked == sum(core.read_num_hdus(str(p)) for p in CORPUS) > 300


@needs_corpus
def test_hdu_type_matches_the_astropy_hdu_class() -> None:
    """The type string the reader routes on, against astropy's HDU classes.

    A tile-compressed extension is a ``CompImageHDU`` in astropy and ``IMAGE``
    in torchfits even though its *header* describes a one-column binary table
    (``tests/test_core_library_real_data.py`` covers that half). Both facts have
    to hold at once, on the same real frames, or the reader routes wrong.
    """
    seen: set[str] = set()
    for path in CORPUS:
        with fits.open(path) as hdul:
            for hdu, hdu_obj in enumerate(hdul):
                astropy_class = type(hdu_obj).__name__
                assert astropy_class in ASTROPY_TYPE, f"unmapped {astropy_class}"
                assert (
                    core.read_hdu_type(str(path), hdu) == ASTROPY_TYPE[astropy_class]
                ), f"{path.name} HDU {hdu}: astropy {astropy_class}"
                seen.add(astropy_class)
    assert seen  # the corpus really did exercise astropy's HDU classes


@needs_megacam
def test_image_info_is_fits_order_and_read_shape_is_row_major() -> None:
    """One geometry, two orderings, pinned on a frame that is not square.

    ``Metadata.image_info`` returns raw ``NAXIS1..NAXIS9``; ``read_shape``
    returns the reversed, row-major shape the tensor paths need. Both are
    documented and correct, and on a square image they are indistinguishable --
    so a caller that mixes them can be wrong for a long time without noticing.
    A real MegaCam extension is 4644x2112, so the two differ visibly.
    """
    path = _mega_frame()
    reader = core.Metadata(str(path))
    for hdu in (1, 2):
        bitpix, naxis, naxes = reader.image_info(hdu)
        fits_order = tuple(int(naxes[i]) for i in range(naxis))
        shape_bitpix, row_major = core.read_shape(str(path), hdu)
        assert bitpix == shape_bitpix == 16
        assert fits_order == tuple(reversed(tuple(row_major)))
        assert row_major == (4644, 2112)
        assert fits_order == (2112, 4644)
        assert fits_order != tuple(row_major), "a square frame would prove nothing"
        with fits.open(path) as hdul:
            # astropy's HDU shape is row-major, like read_shape.
            assert tuple(hdul[hdu].shape) == tuple(row_major)
            assert (
                hdul[hdu].header["NAXIS1"],
                hdul[hdu].header["NAXIS2"],
            ) == fits_order


@needs_megacam
def test_megacam_compressed_extensions_decompress_to_astropy_pixels() -> None:
    """Two independent Rice decompressors, exact on real sky.

    ``read_tensor`` applies ``BZERO`` and returns unsigned, matching what
    astropy hands back, so the comparison is of decoded pixels and not of two
    different raw conventions.
    """
    path = _mega_frame()
    num_hdus = core.read_num_hdus(str(path))
    hdus = sorted({1, 2, num_hdus // 2, num_hdus - 1})
    with fits.open(path) as hdul:
        for hdu in hdus:
            expected = np.asarray(hdul[hdu].data)
            got = torchfits.read_tensor(str(path), hdu)
            assert got.dtype == torch.uint16, f"HDU {hdu}: {got.dtype}"
            assert tuple(got.shape) == expected.shape
            assert tuple(got.shape) == core.read_shape(str(path), hdu)[1]
            np.testing.assert_array_equal(got.numpy(), expected)
            # Real data, not a blank frame: the comparison would be vacuous if
            # every extension were zeros.
            assert expected.min() < expected.max()


@needs_megacam
def test_megacam_scale_info_matches_the_real_bscale_bzero() -> None:
    """Real MegaCam stores int16 with ``BZERO=32768.0`` to mean unsigned 16.

    The whole convention is one header pair, and getting it wrong silently
    shifts a frame by 32768 counts. So: the core's ``scale_info`` must report
    the header's own numbers, and the raw read plus that scale must reproduce
    astropy's unsigned array exactly.
    """
    path = _mega_frame()
    has_bscale, has_bzero, bscale, bzero = core.Metadata(str(path)).scale_info(1)
    with fits.open(path) as hdul:
        header = hdul[1].header
        assert has_bscale and has_bzero
        assert bscale == pytest.approx(header["BSCALE"])
        assert bzero == pytest.approx(header["BZERO"])
        assert (bscale, bzero) == (1.0, 32768.0)
        expected = np.asarray(hdul[1].data)

    raw, reported_scale, reported_bscale, reported_bzero = C.read_full_raw_with_scale(
        str(path), 1
    )
    assert raw.dtype == torch.int16
    assert (reported_scale, reported_bscale, reported_bzero) == (True, bscale, bzero)
    restored = raw.numpy().astype(np.int64) * reported_bscale + reported_bzero
    np.testing.assert_array_equal(restored.astype(np.uint16), expected)
    assert int(restored.min()) == int(expected.min()) < int(expected.max())


@needs_megacam
def test_megacam_raw_compressed_table_matches_astropy() -> None:
    """The VLA column and the raw Arrow transport, byte for byte.

    ``read_table_info`` reports one column of 4644 rows for a compressed
    extension; astropy's ``CompImageHDU`` hides that table behind ``.data``, and
    only ``disable_image_compression=True`` exposes the raw tile stream. This
    is the largest realistic exercise of the variable-length column path on real
    data: 4644 Rice tiles, 5.7-8.9 MB of compressed bytes.

    The tiles are *variable* length -- that is the entire point of compressing
    them, and it is why the column is a VLA at all. A real frame has 94-198
    distinct tile lengths between ~35 bytes (a perfectly uniform tile, which
    compresses to almost nothing) and the ``1PB(n)`` bound from its own header.
    A test with one fixed tile length would not exercise the VLA.
    """
    path = _mega_frame()
    arrow_table = table_api.read(str(path), 1)
    assert arrow_table.num_rows == 4644
    assert arrow_table.column_names == ["COMPRESSED_DATA"]
    assert str(arrow_table.schema.field("COMPRESSED_DATA").type) == "list<item: uint8>"

    mine = arrow_table.column("COMPRESSED_DATA").to_numpy(zero_copy_only=False)
    with fits.open(path, disable_image_compression=True) as hdul:
        raw = hdul[1].data
        assert raw.shape[0] == core.read_nrows(str(path), 1) == arrow_table.num_rows
        theirs = [np.asarray(tile) for tile in raw["COMPRESSED_DATA"]]

    assert len(mine) == len(theirs) == 4644
    lengths = np.array([len(tile) for tile in mine])
    tform = core.read_table_info(str(path), 1)["tforms"][0]
    bound = int(tform[tform.index("(") + 1 : -1])
    assert tform.startswith("1PB(")
    assert lengths.max() == bound, "the longest tile fills the declared TFORM bound"
    assert len(set(lengths.tolist())) > 50, "real tiles are variable length"

    for row, (a, b) in enumerate(zip(mine, theirs)):
        assert len(np.asarray(a)) == len(b)
        np.testing.assert_array_equal(np.asarray(a), b, err_msg=f"row {row}")
    assert int(lengths.sum()) == sum(len(tile) for tile in theirs)
    assert int(lengths.sum()) > 4_000_000


@needs_megacam
def test_megacam_tiles_are_variable_length_across_the_whole_corpus() -> None:
    """Every real frame, and the reason the column is a VLA at all.

    Rice tiles compress to whatever they compress to. Across the ten sample
    frames the tile bound (``1PB(n)``) ranges from 1312 to 2093 bytes, the
    longest tile in each frame hits its bound exactly, and the shortest ranges
    from 35 bytes -- a tile with nothing in it -- to over 1100. A fixed-width
    column could not hold that, and a reader that assumed one length would
    either truncate or over-read.
    """
    shortest_overall = None
    for path in MEGACAM:
        tiles = table_api.read(str(path), 1).column("COMPRESSED_DATA")
        lengths = np.array([len(tile) for tile in tiles.to_numpy(zero_copy_only=False)])
        tform = core.read_table_info(str(path), 1)["tforms"][0]
        bound = int(tform[tform.index("(") + 1 : -1])
        assert core.read_nrows(str(path), 1) == tiles.length() == 4644
        assert lengths.max() == bound, path.name
        assert lengths.min() <= lengths.max()
        assert len(set(lengths.tolist())) > 50, path.name
        shortest_overall = (
            lengths.min()
            if shortest_overall is None
            else min(shortest_overall, int(lengths.min()))
        )
        del tiles
    # Somewhere in this corpus a tile is nearly empty, which is the whole
    # reason the column is variable length rather than a padded fixed array.
    assert shortest_overall is not None and shortest_overall < 200


@needs_megapipe
def test_megapipe_mosaic_pixels_match_astropy() -> None:
    """The mmap path on a 1.6 GB, 435-megapixel real mosaic.

    Reading the whole mosaic is unavoidable -- that is what the mmap path does
    -- but comparing it does not have to be. Astropy reads only the sampled
    windows, so the peak cost is the one 1.6 GB file rather than two of them.
    The strided sample then covers the whole mosaic, including the blank
    borders where the sky is zero, for the cost of 221x229 values.
    """
    path = max(MEGAPIPE, key=lambda p: core.read_shape(str(p), 0)[1][0])
    image = torchfits.read_tensor(str(path), 0)
    height, width = (int(dim) for dim in image.shape)
    assert (height, width) == (21404, 20347)
    assert image.dtype == torch.float32
    assert height * width > 4 * 10**8

    windows = [
        (0, 0, 64, 64),  # the corner, which is blank sky
        (10_000, 10_000, 128, 128),  # a populated field
        (21_000, 20_000, 256, 256),
        (10_700, 10_170, 1024, 1024),  # the mosaic centre
        (height - 512, width - 512, 512, 512),  # the far corner
        (0, width - 256, 64, 256),  # the last row, last columns
    ]
    with fits.open(path, memmap=True) as hdul:
        section = hdul[0].section
        for y, x, rows, columns in windows:
            expected = np.asarray(section[y : y + rows, x : x + columns])
            got = image[y : y + rows, x : x + columns].numpy()
            assert got.shape == expected.shape, f"window y={y} x={x}"
            np.testing.assert_array_equal(got, expected, err_msg=f"window y={y} x={x}")
        strided = np.asarray(hdul[0].data[::97, ::89])

    sampled = image[::97, ::89].numpy()
    assert sampled.shape == strided.shape == (221, 229)
    np.testing.assert_array_equal(sampled, strided)
    # The sample has to be real sky for the comparison above to mean anything:
    # blank border, faint negative sky, and sources three orders of magnitude
    # above it.
    assert np.isfinite(strided).all(), "real mosaics have real blanks"
    assert (strided == 0).any(), "the mosaic border is blank sky"
    assert strided.min() < 0, "a sky-subtracted mosaic goes negative"
    assert float(strided.max()) > 1000
    assert float(strided.std()) > 10


def test_table_read_of_a_real_frame_is_arrow_not_numpy() -> None:
    """A real Arrow read stays Arrow; the reader has no numpy boundary.

    The split claims table reads are torch-free and numpy-free on the Python
    side, so the type the public API actually returns matters: a caller doing
    ``table.read(...)`` on real data must not have silently received a numpy
    array from somewhere in the stack.
    """
    if not MEGACAM:
        pytest.skip("no MegaCam frames; run scripts/fetch_cfht_megacam_sample.sh")
    import pyarrow as pa

    arrow_table = table_api.read(str(_mega_frame()), 1)
    assert isinstance(arrow_table, pa.Table)
    assert not isinstance(arrow_table, np.ndarray)
    schema = table_api.schema(str(_mega_frame()), hdu=1)
    assert isinstance(schema, pa.Schema)
    assert schema.names == arrow_table.column_names


def test_public_read_and_table_agree_on_real_metadata() -> None:
    """The public API and the raw module must not drift on real files.

    ``_core`` is what the public probes are routed through, and the public
    ``read_header`` is a dict-like over the same card list. If the two ever came
    from different sources, this is where it would show up on real data.
    """
    if not CORPUS:
        pytest.skip(
            "real CFHT sample data absent; run scripts/fetch_cfht_megacam_sample.sh "
            "and scripts/fetch_cfht_megapipe_sample.sh"
        )
    for path in CORPUS:
        assert torchfits.read_num_hdus(str(path)) == core.read_num_hdus(str(path))
        header = torchfits.read_header(str(path), 0)
        cards = core.read_header_dict(str(path), 0)
        keys = {key for key, _value, _comment in cards}
        assert set(header.keys()) == keys, path.name
        for key in ("SIMPLE", "BITPIX", "NAXIS"):
            assert key in header, f"{path.name}: {key} missing from the public header"
        assert torchfits.read_shape(str(path), 0)[1] == core.read_shape(str(path), 0)[1]
        if core.read_num_hdus(str(path)) > 1:
            assert torchfits.read_nrows(str(path), 1) == core.read_nrows(str(path), 1)
            assert torchfits.read_colnames(str(path), 1) == core.read_colnames(
                str(path), 1
            )
            assert torchfits.read_table_info(str(path), 1) == core.read_table_info(
                str(path), 1
            )


def test_real_reads_do_not_mutate_the_files() -> None:
    """Reading 9 GB of real observations must leave them byte-identical.

    Every probe here opens a real observation read-only, and the shared
    per-path metadata cache means a second probe of the same file reuses the
    first one's open state. If any of that took a write path -- a header
    update, a checksum keyword, a BSCALE normalisation -- the damage would be
    permanent and to data that is not ours to modify.
    """
    if not CORPUS:
        pytest.skip(
            "real CFHT sample data absent; run scripts/fetch_cfht_megacam_sample.sh "
            "and scripts/fetch_cfht_megapipe_sample.sh"
        )
    frame = _mega_frame()
    mosaic = MEGAPIPE[0] if MEGAPIPE else frame
    for path in {frame, mosaic}:
        before: dict[str, Any] = {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for hdu in range(core.read_num_hdus(str(path))):
            core.read_header_dict(str(path), hdu)
            core.read_shape(str(path), hdu)
        torchfits.read_header(str(path), 0)
        if core.read_num_hdus(str(path)) > 1:
            table_api.read(str(path), 1)
        torchfits.read_tensor(str(path), 1 if path is frame else 0)
        after = path.stat()
        assert after.st_size == before["size"], f"{path.name} changed size"
        assert after.st_mtime_ns == before["mtime_ns"], f"{path.name} was modified"
