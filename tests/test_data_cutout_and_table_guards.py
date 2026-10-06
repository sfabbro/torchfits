"""Dataset guard tests: degenerate cutouts and row-count inference (round 2, unit 6).

Three contracts the ``torchfits.data`` datasets did not hold, each one a rule
the sibling **already had** on the path next to it:

* **R2-036** -- ``FitsStagedCutoutIterableDataset`` coerced
  ``cutouts_per_file`` with ``max(1, ...)`` while ``cutout_size``, six lines
  below, *rejected* its degenerate value. A dataset asked for zero cutouts per
  file handed back one per file, and its ``__repr__`` reported the coerced
  value as if it had been the request.
* **R2-037** -- ``FitsCutoutDataset`` never checked that a window selects any
  pixels, so ``size <= 0`` or an inverted 6-tuple box produced a valid
  0-pixel tensor. ``fits_collate_fn`` stacks those happily: a dataset of them
  yields a ``(4, 1, 0, 0)`` batch straight out of ``make_loader``. The package
  already refuses this condition twice -- ``cutout_size must be positive`` in
  the staged dataset, and the CLI's ``--box`` parser.
* **R2-038** -- ``FitsTableIterableDataset``'s unfiltered path inferred a
  chunk's row count by looking only at **tensor** columns. A table whose
  columns are all variable-length descriptors comes back as Python lists, so
  the count defaulted to 0 and the dataset yielded **no rows at all** for a
  non-empty catalog. The filtered branch of the same method uses the Arrow
  batch's ``num_rows``, and the map-style ``FitsTableDataset`` in the same
  module reported the true count.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from astropy.io import fits

import torchfits
from torchfits.data import (
    FitsCutoutDataset,
    FitsTableDataset,
    FitsTableIterableDataset,
)
from torchfits.data.datasets import FitsStagedCutoutIterableDataset

ROWS = 5


@pytest.fixture(scope="module")
def image(tmp_path_factory) -> str:
    """A plain 8x8 image -- small enough that an over-large box clamps."""
    path = tmp_path_factory.mktemp("data") / "img.fits"
    torchfits.write(
        str(path), torch.arange(64, dtype=torch.float32).reshape(8, 8), overwrite=True
    )
    return str(path)


@pytest.fixture(scope="module")
def numeric_table(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("data") / "num.fits"
    cols = fits.ColDefs(
        [
            fits.Column(name="A", format="J", array=np.arange(ROWS)),
            fits.Column(name="B", format="E", array=np.arange(ROWS, dtype=float)),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
        path, overwrite=True
    )
    return str(path)


@pytest.fixture(scope="module")
def char_table(tmp_path_factory) -> str:
    """Character columns: ``scan_torch`` hands these back as uint8 matrices."""
    path = tmp_path_factory.mktemp("data") / "chr.fits"
    cols = fits.ColDefs(
        [fits.Column(name="S", format="8A", array=np.array([b"a"] * ROWS, dtype="S8"))]
    )
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
        path, overwrite=True
    )
    return str(path)


@pytest.fixture(scope="module")
def vlen_table(tmp_path_factory) -> str:
    """Only a variable-length descriptor column: ``scan_torch`` yields lists."""
    path = tmp_path_factory.mktemp("data") / "vlen.fits"
    cols = fits.ColDefs(
        [
            fits.Column(
                name="V",
                format="PD()",
                array=np.array([np.array([1, 2])] * ROWS, dtype=object),
            )
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
        path, overwrite=True
    )
    return str(path)


# --------------------------------------------------------------------------
# R2-036: cutouts_per_file must be refused, not coerced
# --------------------------------------------------------------------------


def test_staged_cutout_refuses_zero_cutouts_per_file(image):
    with pytest.raises(ValueError, match="cutouts_per_file must be >= 1"):
        FitsStagedCutoutIterableDataset([image], cutouts_per_file=0, cutout_size=4)


def test_staged_cutout_refuses_negative_cutouts_per_file(image):
    with pytest.raises(ValueError, match="cutouts_per_file must be >= 1"):
        FitsStagedCutoutIterableDataset([image], cutouts_per_file=-5, cutout_size=4)


def test_staged_cutout_still_yields_exactly_what_was_asked(image):
    for n in (1, 3):
        ds = FitsStagedCutoutIterableDataset([image], cutouts_per_file=n, cutout_size=4)
        assert ds.cutouts_per_file == n
        assert sum(1 for _ in ds) == n


def test_staged_cutout_repr_reports_the_requested_count(image):
    """The coerced value used to be echoed back as if it had been requested."""
    ds = FitsStagedCutoutIterableDataset([image], cutouts_per_file=3, cutout_size=4)
    assert "cutouts_per_file=3" in repr(ds)


@pytest.mark.parametrize("value,expected", [(2.9, 2), ("3", 3), (True, 1)])
def test_cutouts_per_file_still_coerces_a_numeric_value(image, value, expected):
    """The count has always been ``int()``-coerced; the guard must not narrow it.

    R2-036 moved the coercion above the new check, and nothing pinned that it
    was still there -- a break-it that required an ``int`` type was invisible.
    """
    ds = FitsStagedCutoutIterableDataset([image], cutouts_per_file=value, cutout_size=4)
    assert ds.cutouts_per_file == expected


def test_cutouts_per_file_still_rejects_a_non_numeric_value(image):
    with pytest.raises(ValueError):
        FitsStagedCutoutIterableDataset([image], cutouts_per_file="many", cutout_size=4)


@pytest.mark.parametrize("size", [0, -1, (0, 4), (4, -2)])
def test_staged_cutout_size_sibling_still_refuses(image, size):
    """The guard this finding is measured against must not have been weakened."""
    with pytest.raises(ValueError, match="cutout_size must be positive"):
        FitsStagedCutoutIterableDataset([image], cutouts_per_file=1, cutout_size=size)


# --------------------------------------------------------------------------
# R2-037: a cutout window must select pixels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, -4])
def test_cutout_dataset_refuses_a_non_positive_size(image, size):
    with pytest.raises(ValueError, match="selects no pixels"):
        FitsCutoutDataset([(image, 0, 0, 0, size)])


def test_cutout_dataset_refuses_an_empty_explicit_box(image):
    with pytest.raises(ValueError, match="selects no pixels"):
        FitsCutoutDataset([(image, 0, 0, 0, 0, 0)])


def test_cutout_dataset_refuses_an_inverted_explicit_box(image):
    with pytest.raises(ValueError, match="selects no pixels"):
        FitsCutoutDataset([(image, 0, 2, 2, 1, 1)])


def test_cutout_dataset_refuses_a_half_open_zero_width_window(image):
    """``x2 == x1`` is empty in a half-open window even though y grows."""
    with pytest.raises(ValueError, match="selects no pixels"):
        FitsCutoutDataset([(image, 0, 0, 0, 0, 4)])


def test_cutout_dataset_refuses_a_negative_origin(image):
    with pytest.raises(ValueError, match="non-negative"):
        FitsCutoutDataset([(image, 0, -1, 0, 3, 3)])


def test_cutout_dataset_accepts_a_positive_size(image):
    ds = FitsCutoutDataset([(image, 0, 0, 0, 4)])
    assert tuple(ds[0].shape) == (1, 4, 4)
    assert ds.cutouts[0][2:] == (0, 0, 4, 4)


def test_cutout_dataset_accepts_an_explicit_box(image):
    ds = FitsCutoutDataset([(image, 0, 1, 1, 3, 3)])
    assert tuple(ds[0].shape) == (1, 2, 2)


def test_cutout_dataset_still_clamps_past_the_edge(image):
    """Past-the-edge clamps to the overlap -- the documented cutout contract."""
    ds = FitsCutoutDataset([(image, 0, 0, 0, 128)])
    assert tuple(ds[0].shape) == (1, 8, 8)


def test_cutout_dataset_still_clamps_a_partly_outside_window(image):
    ds = FitsCutoutDataset([(image, 0, 6, 6, 128, 128)])
    assert tuple(ds[0].shape) == (1, 2, 2)


def test_cutout_dataset_refusal_happens_before_any_read(image):
    """A bad window must not cost a file open, and must name the spec."""
    with pytest.raises(ValueError, match=image.split("/")[-1]):
        FitsCutoutDataset([(image, 0, 0, 0, 0)])


# --------------------------------------------------------------------------
# R2-038: the row count must not depend on a column happening to be a tensor
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["numeric_table", "char_table", "vlen_table"])
def test_table_iterable_yields_every_row(request, fixture):
    path = request.getfixturevalue(fixture)
    n_rows = int(torchfits.read_nrows(path, 1))
    assert n_rows == ROWS
    assert len(list(FitsTableIterableDataset(path, hdu=1))) == n_rows


@pytest.mark.parametrize("fixture", ["numeric_table", "char_table", "vlen_table"])
def test_map_and_iterable_table_datasets_agree_on_row_count(request, fixture):
    path = request.getfixturevalue(fixture)
    assert len(FitsTableDataset(path, hdu=1)) == len(
        list(FitsTableIterableDataset(path, hdu=1))
    )


def test_vlen_table_rows_carry_their_values(numeric_table, vlen_table):
    """Not just the right *number* of rows: each row must hold the column."""
    rows = list(FitsTableIterableDataset(vlen_table, hdu=1))
    assert len(rows) == ROWS
    assert all("V" in row for row in rows)
    assert [list(row["V"]) for row in rows] == [[1, 2]] * ROWS


def test_vlen_table_as_batches_still_works(vlen_table):
    """``as_batches`` yields the chunk dict and never infers a row count.

    This is a *consistency* guard, not a defect guard: the batch path hands
    the chunk straight to the consumer, so it was already correct for a
    variable-length catalog. It pins that the two paths still agree on the
    total number of rows after the per-row inference was repaired.
    """
    batches = list(FitsTableIterableDataset(vlen_table, hdu=1, as_batches=True))
    assert batches, "as_batches yielded nothing for a variable-length catalog"
    assert all(isinstance(b, dict) for b in batches)
    assert sum(_rows_in_chunk(b) for b in batches) == ROWS


def _rows_in_chunk(chunk: dict) -> int:
    """Rows a scan_torch column dict carries (tensors and lists alike)."""
    return next(
        (
            v.shape[0] if isinstance(v, torch.Tensor) else len(v)
            for v in chunk.values()
            if isinstance(v, (torch.Tensor, list))
        ),
        0,
    )


def test_numeric_table_row_count_is_unchanged(numeric_table):
    rows = list(FitsTableIterableDataset(numeric_table, hdu=1))
    assert len(rows) == ROWS
    assert [int(r["A"]) for r in rows] == list(range(ROWS))


def test_table_iterable_batching_still_covers_every_row(vlen_table):
    """Small batch_size forces many chunks; the count must still be exact."""
    rows = list(FitsTableIterableDataset(vlen_table, hdu=1, batch_size=2))
    assert len(rows) == ROWS


def test_filtered_path_still_reports_its_own_count(numeric_table):
    """The ``where=`` branch reads Arrow's num_rows; it must stay in step."""
    rows = list(FitsTableIterableDataset(numeric_table, hdu=1, where="A >= 0"))
    assert len(rows) == ROWS
