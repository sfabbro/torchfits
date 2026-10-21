"""Real CFHT payloads through every path the public read API exposes.

``tests/test_core_library_real_data.py`` holds the *metadata* answers of a
torch-free core against the torch-linked extension on real observations. This
file goes past the header and pins the *payload*: the pixels of a
Rice-compressed MegaCam frame and the variable-length tile table that stores
them.

Why it is worth a second real-data file:

* those pixels reach the caller through four independent paths -- a whole-frame
  read, :func:`torchfits.read_subset`, a reused
  :func:`torchfits.open_subset_reader`, and the scaled/raw pair that applies
  ``BZERO`` -- and until now nothing compared them on a file produced by an
  observatory. They agree on a 4x4 synthetic frame by construction; a
  4644x2112 frame where CFITSIO decompresses 2046-pixel Rice tiles is where an
  off-by-one in a tile boundary or a row stride actually shows.
* the raw/scaled pair is what makes MegaCam ``uint16`` in memory and ``int16``
  on disk, and the corpus carries *two* different offsets (``32768`` on the
  ``o`` frames, ``32668`` on the ``p`` frames), so the expectation is read from
  each frame's own header. Asserting the two reads differ by exactly the offset
  is stronger than asserting a dtype, and a float round-trip would break it.
* the compressed-tile table is the only real variable-length column in this
  repository's data, and ``mmap`` refuses it (CFITSIO cannot memory-map a heap).
  The fallback that makes that a non-event -- silently, with the same values --
  is a correctness property nothing else here covers.
* one test writes a real frame back out and reads it in again, which is the only
  place the *writer* meets real pixel statistics (mostly zeros with a long
  bright tail) and the pseudo-unsigned ``int16 + BZERO`` pair.

The data is not in git. Fetch it once::

    bash scripts/fetch_cfht_megacam_sample.sh    # ~2.5 GB, 10 Rice .fz frames

Without the samples this file skips, so a clone with no CFHT data still gets a
green suite.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import torchfits  # noqa: E402
import torchfits.table as table_api  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[1] / "benchmarks_data"
MEGACAM = sorted(DATA_ROOT.glob("cfht_megacam/*.fits.fz"))

pytestmark = pytest.mark.skipif(
    not MEGACAM,
    reason="real CFHT sample data absent; run scripts/fetch_cfht_megacam_sample.sh",
)

#: The extension every pixel test reads.
HDU = 1


@pytest.fixture(scope="module")
def frame_path() -> Path:
    """The frame the pixel tests agree on: the first sorted sample.

    Resolved per test rather than at import. Indexing ``MEGACAM`` at module
    level raises ``IndexError`` during collection on a clone without the
    samples -- before the skip mark above can fire -- so a missing download
    would turn the suite red instead of skipping it. The first sorted frame is
    stable across fetches, so the cost of this file does not depend on how many
    samples happen to be on disk.
    """
    assert MEGACAM, "no real CFHT frame; the module skip mark should have fired"
    return MEGACAM[0]


@pytest.fixture(scope="module")
def full_frame(frame_path: Path) -> torch.Tensor:
    """One whole science extension, read once for every test in the file."""
    frame = torchfits.read(frame_path, hdu=HDU)
    assert frame.ndim == 2
    return frame


def _assert_same(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.shape == want.shape, (
        f"{what}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    )
    assert got.dtype == want.dtype, f"{what}: dtype {got.dtype} != {want.dtype}"
    if not torch.equal(got, want):
        delta = (got.to(torch.int32) - want.to(torch.int32)).abs()
        raise AssertionError(
            f"{what}: {int((delta != 0).sum())} of {delta.numel()} pixels differ, "
            f"max |delta| = {int(delta.max())}"
        )


def test_whole_frame_read_matches_a_full_box_subset(
    frame_path: Path, full_frame: torch.Tensor
) -> None:
    """``read`` and ``read_subset`` must agree on the box that covers everything.

    They are different CFITSIO entry points -- one reads the image, the other
    iterates tiles for a window -- so this is a genuine cross-check of the
    decompression and the axis order, not a tautology.
    """
    ny, nx = full_frame.shape
    box = torchfits.read_subset(frame_path, HDU, 0, 0, nx, ny)
    _assert_same(box, full_frame, "read_subset(full box)")


def test_quadrants_stitch_back_into_the_whole_frame(
    frame_path: Path, full_frame: torch.Tensor
) -> None:
    """Four cutouts, placed by their offsets, must reproduce the full frame.

    An off-by-one in a tile edge or a transpose inside the cutout reader shows
    up here as a band of mismatched pixels at a quadrant boundary, which is
    exactly where a real observation's tile grid sits.
    """
    ny, nx = full_frame.shape
    mid_y, mid_x = ny // 2, nx // 2
    quadrants = [
        (0, 0, mid_x, mid_y),
        (mid_x, 0, nx, mid_y),
        (0, mid_y, mid_x, ny),
        (mid_x, mid_y, nx, ny),
    ]
    stitched = torch.empty_like(full_frame)
    for x1, y1, x2, y2 in quadrants:
        stitched[y1:y2, x1:x2] = torchfits.read_subset(frame_path, HDU, x1, y1, x2, y2)
    _assert_same(stitched, full_frame, "quadrant stitch")


def test_one_row_strips_match_the_same_rows_of_the_whole_frame(
    frame_path: Path,
    full_frame: torch.Tensor,
) -> None:
    """Row strips at both ends and in the middle: the ugliest read pattern.

    A one-pixel-high window is the case where a reader can silently return the
    wrong row (the first tile of the image, say) and still look plausible. The
    rows chosen are the first, the last, and one inside the middle tile.
    """
    ny, nx = full_frame.shape
    for y in (0, ny // 2, ny - 1):
        strip = torchfits.read_subset(frame_path, HDU, 0, y, nx, y + 1)
        _assert_same(strip, full_frame[y : y + 1, :], f"row strip y={y}")


def test_a_reused_subset_reader_does_not_drift(
    frame_path: Path, full_frame: torch.Tensor
) -> None:
    """The same box, read twice from one reader, with an unrelated box between.

    ``open_subset_reader`` keeps a handle and its own CFITSIO state; if a
    cutout leaked a tile offset into the next one, the second read of the same
    box would differ from the first. Reading a far-away box in between is what
    makes that visible.
    """
    ny, nx = full_frame.shape
    box = (16, 16, 272, 272)
    with torchfits.open_subset_reader(frame_path, HDU) as reader:
        first = reader.read_subset(*box)
        reader.read_subset(nx - 272, ny - 272, nx - 16, ny - 16)
        second = reader.read_subset(*box)
    _assert_same(second, first, "reused reader, second read of the same box")
    x1, y1, x2, y2 = box
    _assert_same(first, full_frame[y1:y2, x1:x2], "reused reader vs whole-frame slice")


def _frames_by_zero_point() -> dict[float, Path]:
    """One real frame per ``BZERO`` convention found in the corpus.

    Reading the header of every frame is cheap (a few hundred cards) and keeps
    this file honest about what the sample set actually contains instead of
    hardcoding a filename that a re-fetch could reorder.
    """
    by_zero: dict[float, Path] = {}
    for path in MEGACAM:
        header = torchfits.read_header(path, HDU)
        zero = float(header.get("BZERO", 0.0) or 0.0)
        if zero and zero not in by_zero:
            by_zero[zero] = path
    return by_zero


def test_scaled_reads_apply_each_real_bzero_convention_exactly() -> None:
    """Real ``BZERO`` is applied exactly, for both offsets the corpus carries.

    ``BZERO=32768`` is the whole unsigned range, so the scaled frame is a
    ``uint16``; ``BZERO=32668`` shifts past ``uint16`` and the reader must fall
    back to float. Both are CFHT output from the same instrument, so both are
    things a reader has to get right.

    Exactness is the assertion, not the dtype: a scale path that went through
    floating point would round, one that applied ``BZERO`` twice would wrap the
    low pixels to the top of the range, and a dtype check would survive both.
    ``int16 + 32768`` and ``int16 + 32668`` are integers below 2**24, so a float
    result is still exact -- which is what makes equality the honest test.
    """
    by_zero = _frames_by_zero_point()
    assert by_zero, "no real frame carries a BZERO card; the sample set is unusable"

    for zero, path in sorted(by_zero.items()):
        raw = torchfits.read_tensor(path, hdu=HDU, raw_scale=True)
        scaled = torchfits.read(path, hdu=HDU)
        assert raw.dtype == torch.int16, (
            f"{path.name}: raw read should be int16, got {raw.dtype}"
        )

        if zero + 32768.0 == 65536.0:
            expected_dtype = torch.uint16
        else:
            expected_dtype = torch.float32
        assert scaled.dtype == expected_dtype, (
            f"{path.name}: BZERO={zero} should read as {expected_dtype}, got {scaled.dtype}"
        )

        expected = (raw.to(torch.int64) + int(zero)).to(expected_dtype)
        _assert_same(scaled, expected, f"{path.name}: scaled == raw + BZERO({zero})")

        # Real science pixels, not a flat frame: a constant result would make
        # the equality above hold for any offset.
        scaled_i = scaled.to(torch.int64)
        assert int(scaled_i.min()) >= 0, f"{path.name}: scaled pixels went negative"
        assert int(scaled_i.max()) - int(scaled_i.min()) > 1000, (
            f"{path.name}: scaled pixels are flat; the equality proves nothing"
        )


def test_repeated_reads_of_a_real_frame_are_bit_identical(
    frame_path: Path,
    full_frame: torch.Tensor,
) -> None:
    """A second whole-frame read, after other work, must not drift.

    The path-level caches (header, schema, mmap plan) are keyed by path and
    HDU; a stale entry would show up as a different frame the second time.
    """
    again = torchfits.read(frame_path, hdu=HDU)
    _assert_same(again, full_frame, "second whole-frame read")


def test_threads_read_one_real_frame_without_tearing(
    frame_path: Path, full_frame: torch.Tensor
) -> None:
    """Four threads, one 300 MB frame: every answer is the whole frame.

    CFITSIO handles are not shareable across threads, so a reader that reused
    one would return garbled or truncated pixels under a race -- intermittently,
    which is the worst way to find out. Each worker also checks its own shape,
    so a short read fails inside the worker rather than at comparison time.
    """

    def read_once(_: int) -> torch.Tensor:
        frame = torchfits.read(frame_path, hdu=HDU)
        assert frame.shape == full_frame.shape
        return frame

    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(read_once, range(4)))
    for index, frame in enumerate(frames):
        _assert_same(frame, full_frame, f"thread {index}")


def test_compressed_tile_table_reads_agree_across_backends(frame_path: Path) -> None:
    """The real VLA column, through the three row-selection paths.

    ``COMPRESSED_DATA`` is the only variable-length column in the corpus. A
    full read, a contiguous slice and a scattered selection must produce the
    same bytes for the same rows; the scattered path reorders rows by index and
    maps them back, which is where a VLA read can silently pair the wrong heap
    offset with a row number.
    """
    whole = table_api.read(frame_path, hdu=HDU)
    column = whole.column_names[0]
    assert whole.num_rows > 4000

    rows = [3, 7, 11, 4000]
    scattered = table_api.read(frame_path, hdu=HDU, rows=rows)
    assert scattered.num_rows == len(rows)
    for position, row in enumerate(rows):
        assert (
            scattered.column(column)[position].as_py()
            == whole.column(column)[row].as_py()
        ), f"rows=[...] paired row {row} with the wrong heap entry"

    contiguous = table_api.read(frame_path, hdu=HDU, row_slice=(10, 20))
    assert contiguous.num_rows == 10
    for position, row in enumerate(range(10, 20)):
        assert (
            contiguous.column(column)[position].as_py()
            == whole.column(column)[row].as_py()
        ), f"row_slice(10, 20) row {position} != whole-table row {row}"


def test_mmap_refusal_on_a_real_vla_column_keeps_the_values(frame_path: Path) -> None:
    """``mmap=True`` must fall back, not lose data, on a heap-backed column.

    CFITSIO answers ``VLA columns not supported for mmap`` here, and the
    fallback is what turns that into a normal read. Comparing the mmap and
    non-mmap answers is the only way to see that the fallback returned the same
    table instead of an empty one.
    """
    import torchfits._C as cpp

    with pytest.raises(RuntimeError, match="VLA columns not supported for mmap"):
        cpp.read_fits_table_rows_raw(str(frame_path), HDU, [], 1, -1, True)

    via_mmap = table_api.read(frame_path, hdu=HDU, mmap=True)
    via_reader = table_api.read(frame_path, hdu=HDU, mmap=False)
    assert via_mmap.num_rows == via_reader.num_rows > 0
    column = via_mmap.column_names[0]
    assert via_mmap.column(column).to_pylist() == via_reader.column(column).to_pylist()


def test_a_real_frame_survives_a_lossless_write_read_round_trip(tmp_path: Path) -> None:
    """A real frame, compressed and checksummed, comes back bit for bit.

    This is the only test here that exercises the *writer* on real pixels, and
    real pixels are the hard case: CFHT frames are mostly zeros with a long
    bright tail, which is what makes RICE_1 efficient and also what makes a
    wrong row length or a mis-signed offset obvious instead of invisible.

    Both storage conventions go through it. A ``uint16`` frame is written as
    ``int16 + BZERO=32768`` (FITS has no unsigned ``BITPIX``), so the round trip
    proves the pseudo-unsigned pair is written and read back consistently -- a
    reader that dropped ``BZERO`` would return the raw negatives and a writer
    that wrote ``uint16`` straight through would overflow.

    The algorithm is chosen from the dtype rather than fixed, because the two
    real conventions need different ones to stay exact: RICE_1 is lossless for
    integers and *quantizes* floats (CFITSIO's default, and what astropy and
    fitsio do too), so a float frame -- the ``p`` convention, a third of this
    corpus -- is written with GZIP_1. Assuming RICE_1 everywhere is exactly the
    mistake this test made the first time it ran: it silently lost up to 4 ADU
    on 7.8 million of 9.8 million pixels.
    """
    by_zero = _frames_by_zero_point()
    assert by_zero, "no real frame to round-trip"

    for index, (zero, source) in enumerate(sorted(by_zero.items())):
        frame = torchfits.read(source, hdu=HDU)
        algorithm = "GZIP_1" if frame.dtype.is_floating_point else "RICE_1"
        out = tmp_path / f"roundtrip_{index}.fits"
        torchfits.write(out, frame, compress=algorithm, checksum=True)

        # A tile-compressed image cannot be a primary HDU (a primary header has
        # no room for the Z* cards), so the writer emits a null primary plus the
        # image as extension 1 -- the same layout astropy produces. Pinning it
        # here is what tells a future reader which HDU the payload is on.
        assert torchfits.read_num_hdus(out) == 2, f"{source.name}: unexpected HDU count"
        assert torchfits.read_shape(out, 0)[1] == (), (
            f"{source.name}: the primary should be an empty image, not the payload"
        )
        assert torchfits.read_shape(out, 1)[1] == tuple(frame.shape)

        status = torchfits.verify_checksums(out, hdu=1)
        assert status["present"] is True, f"{source.name}: no checksum written"
        assert status["ok"] is True, f"{source.name}: checksum verify failed: {status}"

        restored = torchfits.read(out, hdu=1)
        _assert_same(
            restored, frame, f"{source.name} {algorithm} round trip (BZERO={zero})"
        )


def test_multi_extension_read_returns_each_extension_at_its_own_shape(
    frame_path: Path,
) -> None:
    """``read_hdus`` over a real MEF: 41 HDUs of one frame, three of them read.

    The extensions of a MegaCam frame are not all the same shape (the guide
    window and the mask differ), so a reader that cached the first HDU's
    geometry would return wrongly shaped tensors for the others.
    """
    num_hdus = torchfits.read_num_hdus(frame_path)
    assert num_hdus > 5
    wanted = [1, num_hdus // 2, num_hdus - 1]
    tensors = torchfits.read_hdus(frame_path, wanted)
    assert len(tensors) == len(wanted)
    for hdu, tensor in zip(wanted, tensors):
        shape = torchfits.read_shape(frame_path, hdu)[1]
        assert tuple(tensor.shape) == tuple(shape), (
            f"HDU {hdu}: {tuple(tensor.shape)} != {shape}"
        )
