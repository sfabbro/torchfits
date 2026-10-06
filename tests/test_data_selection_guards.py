"""Dataset selection guards: spectral slicing and empty selections (round 2, unit 6).

Two contracts the ``torchfits.data`` datasets did not hold, each one a rule the
sibling **already had** on the path next to it:

* **R2-039** -- ``FitsCubeDataset`` / ``FitsCubeIterableDataset`` validate
  ``spectral_slice`` (``0 <= start < stop``) and then do not validate
  ``slice_index`` at all. ``tensor.select`` wraps a negative index, so
  ``slice_index=-1`` silently returned the **last** channel of the cube with no
  error, while ``spectral_slice=(-1, 2)`` in the same constructor refused.
* **R2-040** -- an empty selection is accepted by ``_resolve_paths`` and by
  ``FitsCutoutDataset``, so every image/cube/spectrum dataset builds with zero
  files. The siblings refuse the identical mistake: ``_as_hdu_list`` raises
  ``hdu sequence must be non-empty`` and ``from_bands`` raises
  ``no image bands found``. The downstream blast radius then depends on a
  sampler flag -- torch's ``RandomSampler`` rejects ``num_samples == 0`` so
  ``make_loader(..., shuffle=True)`` fails loudly, but ``SequentialSampler``
  does not check, so ``shuffle=False`` built a loader that yields **zero
  batches and no error**.

Every new name is imported inside the test that uses it, so this file collects
against unfixed sources and each test fails on its own rather than erroring at
collection (a collection error is the vacuous outcome, not evidence).
"""

from __future__ import annotations

import pytest
import torch
from astropy.io import fits

CHANNELS = 3
PLANE = 4


@pytest.fixture(scope="module")
def cube(tmp_path_factory) -> str:
    """A (3, 4, 4) cube: the spectral axis is axis 0 and has 3 channels."""
    path = tmp_path_factory.mktemp("sel") / "cube.fits"
    data = torch.arange(CHANNELS * PLANE * PLANE, dtype=torch.float32).reshape(
        CHANNELS, PLANE, PLANE
    )
    fits.PrimaryHDU(data.numpy()).writeto(path, overwrite=True)
    return str(path)


@pytest.fixture(scope="module")
def image(tmp_path_factory) -> str:
    """A plain 8x8 image, for the dataset classes that want a 2-D HDU."""
    path = tmp_path_factory.mktemp("sel") / "img.fits"
    torchfits_write(str(path))
    return str(path)


def torchfits_write(path: str) -> None:
    from torchfits import write

    write(path, torch.arange(64, dtype=torch.float32).reshape(8, 8), overwrite=True)


# --------------------------------------------------------------------------
# R2-039: a negative slice_index must be refused like spectral_slice's negative
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, -2, -3, -99])
def test_cube_dataset_refuses_a_negative_slice_index(cube, bad):
    from torchfits.data import FitsCubeDataset

    with pytest.raises(ValueError, match="slice_index"):
        FitsCubeDataset([cube], slice_index=bad)


@pytest.mark.parametrize("bad", [-1, -2, -3, -99])
def test_cube_iterable_dataset_refuses_a_negative_slice_index(cube, bad):
    from torchfits.data import FitsCubeIterableDataset

    with pytest.raises(ValueError, match="slice_index"):
        FitsCubeIterableDataset([cube], slice_index=bad)


def test_the_negative_slice_index_refusal_needs_no_file_read():
    """The check must be *pure*, which the channel count forces: the cube's
    depth is only known after a header read, so a shape-aware bound would
    either do I/O in the constructor or be deferred to access -- the two things
    this finding is about.

    Pointed at a path that does not exist, so a guard that tried to read the
    file would surface as ``OSError``/``IndexError`` instead of ``ValueError``.
    """
    from torchfits.data import FitsCubeDataset

    with pytest.raises(ValueError, match="slice_index"):
        FitsCubeDataset(["/nonexistent-dir-xyz/cube.fits"], slice_index=-1)


@pytest.mark.parametrize("good", [0, 1, 2])
def test_cube_dataset_still_accepts_an_in_range_slice_index(cube, good):
    from torchfits.data import FitsCubeDataset

    payload, _label = FitsCubeDataset([cube], slice_index=good)[0]
    assert payload.shape == (PLANE, PLANE)
    # The selected channel must be the right one, not merely *a* channel.
    expected = torch.arange(CHANNELS * PLANE * PLANE, dtype=torch.float32).reshape(
        CHANNELS, PLANE, PLANE
    )[good]
    assert torch.equal(payload, expected)


def test_cube_iterable_dataset_still_accepts_an_in_range_slice_index(cube):
    from torchfits.data import FitsCubeIterableDataset

    payload = next(iter(FitsCubeIterableDataset([cube], slice_index=2)))
    assert payload.shape == (PLANE, PLANE)


def test_cube_dataset_still_returns_the_whole_cube_without_a_selection(cube):
    """``slice_index=None`` and ``spectral_slice=None`` both mean "no slicing"."""
    from torchfits.data import FitsCubeDataset

    payload, _label = FitsCubeDataset([cube])[0]
    assert payload.shape == (CHANNELS, PLANE, PLANE)


def test_slice_index_and_spectral_slice_are_still_mutually_exclusive(cube):
    from torchfits.data import FitsCubeDataset

    with pytest.raises(ValueError, match="not both"):
        FitsCubeDataset([cube], slice_index=1, spectral_slice=(0, 2))


@pytest.mark.parametrize("bad", [(-1, 2), (0, 0), (2, 1), (-3, -1)])
def test_spectral_slice_still_refuses_its_degenerate_pairs(cube, bad):
    """The sibling's existing guard, pinned so the R2-039 fix cannot regress it."""
    from torchfits.data import FitsCubeDataset

    with pytest.raises(ValueError, match="spectral_slice"):
        FitsCubeDataset([cube], spectral_slice=bad)


def test_a_spectral_window_starting_at_zero_still_works(cube):
    """Added after **BI-82 went green**.

    Every ``spectral_slice`` case above is a *degenerate* pair, so nothing
    pinned that the documented first-channel window ``(0, n)`` still builds.
    The mutation ``start < 0`` -> ``start < 1`` (refuse a window starting at
    channel 0) passed the whole guard file -- a real gap, not an equivalent
    mutant.
    """
    from torchfits.data import FitsCubeDataset

    payload, _label = FitsCubeDataset([cube], spectral_slice=(0, 2))[0]
    assert payload.shape == (2, PLANE, PLANE)
    expected = torch.arange(CHANNELS * PLANE * PLANE, dtype=torch.float32).reshape(
        CHANNELS, PLANE, PLANE
    )[:2]
    assert torch.equal(payload, expected)


def test_an_out_of_range_slice_index_still_raises_IndexError(cube):
    """Deliberately NOT changed to ValueError.

    Measured: ``FitsCubeDataset(cube)[99]`` already raises ``IndexError: list
    index out of range``, so an out-of-range ``slice_index`` raising
    ``IndexError`` matches the class's own convention. Converting it would
    create a new asymmetry (``ds[99]`` -> IndexError but ``ds[0]`` ->
    ValueError). Pinned so that decision is not silently reversed.
    """
    from torchfits.data import FitsCubeDataset

    ds = FitsCubeDataset([cube], slice_index=99)
    with pytest.raises(IndexError):
        ds[0]


def test_both_cube_peers_route_selection_through_the_shared_helper():
    """R2-039's fix moves the validation into one helper.

    The invariant is **not** that the two bodies differ -- after the fix they
    are legitimately near-identical, because one rule written once is the goal.
    The invariant is that neither peer re-implements the rule inline, which is
    how the missing negative check survived in *both* copies at once.
    """
    import inspect

    from torchfits.data import FitsCubeDataset, FitsCubeIterableDataset

    for cls in (FitsCubeDataset, FitsCubeIterableDataset):
        body = inspect.getsource(cls.__init__)
        assert "_resolve_spectral_selection(" in body, (
            f"{cls.__name__} no longer routes its slice_index/spectral_slice"
            " validation through the shared helper"
        )
        assert "spectral_slice must be" not in body, (
            f"{cls.__name__} re-implements the spectral_slice check inline"
        )
        assert "must be >= 0" not in body, (
            f"{cls.__name__} re-implements the slice_index bound inline"
        )


# --------------------------------------------------------------------------
# R2-040: an empty selection must be refused like the hdu / from_bands siblings
# --------------------------------------------------------------------------


@pytest.mark.parametrize("empty", [[], ()])
def test_resolve_paths_refuses_an_empty_selection(empty):
    from torchfits.data.datasets import _resolve_paths

    with pytest.raises(ValueError, match="non-empty|no FITS paths"):
        _resolve_paths(empty)


@pytest.mark.parametrize(
    "class_name",
    [
        "FitsTensorDataset",
        "FitsImageDataset",
        "FitsCubeDataset",
        "FitsTensorIterableDataset",
        "FitsCubeIterableDataset",
        "FitsSpectrumDataset",
        "FitsSpectrumIterableDataset",
        "FitsStagedCutoutIterableDataset",
    ],
)
def test_every_image_dataset_refuses_an_empty_path_list(class_name):
    """All eight classes that route through ``_resolve_paths``.

    Resolved by name so one guard covers every caller of the shared helper --
    the point of the fix is that there is exactly one place to get wrong.
    """
    import torchfits.data as data

    cls = getattr(data, class_name)
    with pytest.raises(ValueError, match="non-empty|no FITS paths"):
        cls([])


def test_fits_cutout_dataset_refuses_an_empty_cutout_list():
    from torchfits.data import FitsCutoutDataset

    with pytest.raises(ValueError, match="non-empty|no FITS paths|cutout"):
        FitsCutoutDataset([])


def test_a_single_cutout_window_still_builds(image):
    """Added after **BI-88 went green**.

    The empty-list guard was pinned but a *one*-element cutout list was not, so
    the mutation ``len(normalized) < 2`` (which refuses a single patch) passed
    the whole guard file. A one-patch dataset is the smallest legal selection
    and must survive the fix.
    """
    from torchfits.data import FitsCutoutDataset

    ds = FitsCutoutDataset([(image, 0, 1, 1, 2)])
    assert len(ds) == 1
    payload = ds[0]
    assert payload.shape == (1, 2, 2)  # add_channel_dim=True by default


def test_an_empty_dataset_is_refused_before_any_sampler_sees_it():
    """The measured harm, pinned -- including the torch asymmetry that caused it.

    ``make_loader(FitsTensorDataset([]), shuffle=True)`` used to fail loudly,
    because ``RandomSampler`` validates ``num_samples``. ``shuffle=False`` uses
    ``SequentialSampler``, which does **not**, so the same dataset under the
    same ``make_loader`` built cleanly and yielded **zero batches with no
    error**. The first half of this test pins that asymmetry in torch itself,
    so the reason this guard exists is not quietly forgotten (and so a future
    torch that adds the check is understood, not mistaken for this guard
    having been unnecessary).
    """
    import torch.utils.data as tud
    from torch.utils.data import Dataset

    class _Empty(Dataset):
        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise AssertionError("never indexed")

    with pytest.raises(ValueError, match="positive integer"):
        tud.RandomSampler(_Empty())
    assert tud.SequentialSampler(_Empty()) is not None  # the hole

    # The dataset is refused at construction, so neither sampler is reachable
    # with zero files and the flag cannot change the outcome.
    from torchfits.data import FitsCutoutDataset, FitsTensorDataset

    with pytest.raises(ValueError, match="non-empty"):
        FitsTensorDataset([])
    with pytest.raises(ValueError, match="non-empty"):
        FitsCutoutDataset([])


def test_hdu_sequence_is_still_refused_when_empty(image):
    """The sibling guard that already existed, pinned."""
    from torchfits.data import FitsTensorDataset

    with pytest.raises(ValueError, match="hdu sequence must be non-empty"):
        FitsTensorDataset([image], hdu=[])


def test_from_bands_still_refuses_an_empty_band_list(image):
    """The other sibling guard, pinned."""
    from torchfits.data import FitsImageDataset

    with pytest.raises(ValueError, match="no image bands found"):
        FitsImageDataset.from_bands([image], bands=[])


def test_a_real_single_file_dataset_still_builds(image):
    from torchfits.data import FitsImageDataset

    ds = FitsImageDataset([image])
    assert len(ds) == 1
    payload, label = ds[0]
    assert payload.shape[0] == 1  # add_channel_dim=True by default
    assert label.dtype == torch.long


def test_a_real_multi_file_dataset_still_builds(image, tmp_path_factory):
    from torchfits.data import FitsImageDataset

    second = tmp_path_factory.mktemp("sel2") / "img2.fits"
    torchfits_write(str(second))
    ds = FitsImageDataset([image, str(second)])
    assert len(ds) == 2


def test_a_non_matching_glob_still_falls_back_to_the_literal_pattern(image):
    """``_resolve_paths`` must not confuse "no match" with "empty selection".

    A glob matching nothing falls back to the literal pattern so the failure
    happens at read time with a real path in the message. The new guard must
    not swallow that into "no FITS paths matched".
    """
    from torchfits.data.datasets import _resolve_paths

    resolved = _resolve_paths("/nonexistent-dir-xyz/*.fits")
    assert resolved == ["/nonexistent-dir-xyz/*.fits"]


def test_a_remote_url_is_still_a_single_path(tmp_path_factory):
    from torchfits.data.datasets import _resolve_paths

    assert _resolve_paths("https://example.invalid/x.fits") == [
        "https://example.invalid/x.fits"
    ]
