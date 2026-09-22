"""ML-value additions to :mod:`torchfits.data`.

Covers the spectral companions (wavelength, DQ mask, labels), IFU/cube
spectral-axis windows, even row-range sharding + tensor-space streaming for
tables, and auto band discovery with zeropoints.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

fits = pytest.importorskip("astropy.io.fits")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spectrum_mef(tmp_path):
    """Two-arm MEF spectrum: flux/ivar/DQ/wavelength per arm."""
    path = tmp_path / "spec.fits"
    hdus = [fits.PrimaryHDU()]
    dq = np.zeros((2, 8), dtype=np.int16)
    dq[0, 7] = 2
    dq[1, 0] = 2
    hdus.append(
        fits.ImageHDU(np.arange(16, dtype=np.float32).reshape(2, 8), name="FLUX")
    )
    hdus.append(fits.ImageHDU(np.full((2, 8), 4.0, dtype=np.float32), name="IVAR"))
    hdus.append(fits.ImageHDU(dq, name="DQ"))
    hdus.append(
        fits.ImageHDU(
            np.linspace(4000.0, 4700.0, 8, dtype=np.float32)[None, :].repeat(2, axis=0),
            name="WAVE",
        )
    )
    fits.HDUList(hdus).writeto(str(path), overwrite=True)
    return path


@pytest.fixture
def bands_fits(tmp_path):
    """Three-band image MEF with zeropoints, IVAR and DQ companions."""
    path = tmp_path / "bands.fits"
    hdus = [fits.PrimaryHDU()]
    for name, value, zp in (("G", 1.0, 25.0), ("R", 2.0, 24.0), ("Z", 3.0, 23.0)):
        hdu = fits.ImageHDU(np.full((4, 4), value, dtype=np.float32), name=name)
        hdu.header["PHOTZEROPOINT"] = zp
        hdu.header["EXPTIME"] = 100.0
        hdus.append(hdu)
    for name in ("G_IVAR", "R_IVAR", "Z_IVAR"):
        hdus.append(fits.ImageHDU(np.ones((4, 4), dtype=np.float32), name=name))
    for name in ("G_DQ", "R_DQ", "Z_DQ"):
        hdus.append(fits.ImageHDU(np.zeros((4, 4), dtype=np.int16), name=name))
    fits.HDUList(hdus).writeto(str(path), overwrite=True)
    return path


@pytest.fixture
def table_fits(tmp_path):
    """8-row binary table with float and integer columns."""
    path = tmp_path / "cat.fits"
    hdus = [fits.PrimaryHDU()]
    cols = [
        fits.Column(name="flux", format="E", array=np.arange(8, dtype=np.float32)),
        fits.Column(name="dq", format="J", array=np.arange(8, dtype=np.int32)),
    ]
    hdus.append(fits.BinTableHDU.from_columns(cols))
    fits.HDUList(hdus).writeto(str(path), overwrite=True)
    return path


# ---------------------------------------------------------------------------
# Spectra: wavelength, DQ mask, labels
# ---------------------------------------------------------------------------


class TestSpectrumCompanions:
    def test_image_arm_wavelength_mask_and_label(self, spectrum_mef):
        from torchfits.data import FitsSpectrumDataset

        payload, label = FitsSpectrumDataset(
            [str(spectrum_mef)],
            hdu="FLUX",
            ivar_hdu="IVAR",
            mask_hdu="DQ",
            mask_is_dq=True,
            bad_bits=[1],  # bit 1 == value 2
            wavelength_hdu="WAVE",
            row=0,
            labels=[5],
        )[0]

        assert payload["flux"].shape == (8,)
        assert payload["wavelength"].shape == (8,)
        assert payload["wavelength"][0].item() == pytest.approx(4000.0)
        mask = payload["mask"]
        assert mask.dtype == torch.bool
        assert mask.sum().item() == 7
        assert not mask[7]
        assert int(label) == 5

    def test_plain_payload_when_no_labels(self, spectrum_mef):
        from torchfits.data import FitsSpectrumDataset

        payload = FitsSpectrumDataset([str(spectrum_mef)], hdu="FLUX")[0]
        assert isinstance(payload, dict)  # not a (payload, label) tuple
        assert "wavelength" not in payload

    def test_shared_wavelength_hdu_stacks_with_arms(self, spectrum_mef):
        from torchfits.data import FitsSpectrumDataset

        payload = FitsSpectrumDataset(
            [str(spectrum_mef)],
            hdu=["FLUX"],
            wavelength_hdu="WAVE",
            row=1,
            layout="dict",
        )[0]
        assert payload["wavelength"].shape == payload["flux"].shape

    def test_mask_is_dq_requires_a_mask_source(self, spectrum_mef):
        from torchfits.data import FitsSpectrumDataset

        with pytest.raises(ValueError, match="mask_is_dq"):
            FitsSpectrumDataset([str(spectrum_mef)], hdu="FLUX", mask_is_dq=True)

    def test_table_spectrum_with_wavelength_and_dq(self, tmp_path):
        from torchfits.data import FitsSpectrumDataset

        path = tmp_path / "table_spec.fits"
        table = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="FLUX",
                    format="E",
                    array=np.arange(6, dtype=np.float32),
                ),
                fits.Column(
                    name="IVAR", format="E", array=np.ones(6, dtype=np.float32)
                ),
                fits.Column(name="DQ", format="J", array=np.full(6, 4, dtype=np.int32)),
                fits.Column(
                    name="WAVE",
                    format="E",
                    array=np.linspace(5000.0, 5500.0, 6, dtype=np.float32),
                ),
            ]
        )
        fits.HDUList([fits.PrimaryHDU(), table]).writeto(str(path), overwrite=True)

        payload = FitsSpectrumDataset(
            [str(path)],
            hdu=1,
            column="FLUX",
            ivar_column="IVAR",
            mask_column="DQ",
            mask_is_dq=True,
            bad_bits=[2],  # bit 2 == value 4 -> every row invalid
            wavelength_column="WAVE",
        )[0]
        assert payload["flux"].shape == (6,)
        assert payload["wavelength"][-1].item() == pytest.approx(5500.0)
        assert not payload["mask"].any()

    def test_table_spectrum_rejects_mixing_column_and_hdu(self, spectrum_mef):
        from torchfits.data import FitsSpectrumDataset

        with pytest.raises(ValueError, match="columns"):
            FitsSpectrumDataset(
                [str(spectrum_mef)],
                hdu=1,
                column="FLUX",
                wavelength_hdu="WAVE",
            )

    def test_iterable_labels_follow_the_file(self, spectrum_mef, tmp_path):
        from torchfits.data import FitsSpectrumIterableDataset

        second = tmp_path / "spec2.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(np.ones((1, 8), dtype=np.float32), name="FLUX"),
            ]
        ).writeto(str(second), overwrite=True)

        ds = FitsSpectrumIterableDataset(
            [str(spectrum_mef), str(second)],
            hdu="FLUX",
            row=0,
            labels=[0, 1],
        )
        items = list(ds)
        assert len(items) == 2
        labels = {int(label) for _, label in items}
        assert labels == {0, 1}
        assert all(payload["flux"].shape == (8,) for payload, _ in items)


# ---------------------------------------------------------------------------
# Cubes / IFU: spectral-axis windows
# ---------------------------------------------------------------------------


class TestCubeSpectralSlice:
    @pytest.fixture
    def cube_fits(self, tmp_path):
        path = tmp_path / "cube.fits"
        data = np.arange(120, dtype=np.float32).reshape(5, 4, 6)
        fits.PrimaryHDU(data).writeto(str(path), overwrite=True)
        return path

    def test_map_style_window(self, cube_fits):
        from torchfits.data import FitsCubeDataset

        data = np.arange(120, dtype=np.float32).reshape(5, 4, 6)
        cube, _ = FitsCubeDataset([str(cube_fits)], spectral_slice=(1, 4), labels=[0])[
            0
        ]
        assert cube.shape == (3, 4, 6)
        assert np.allclose(cube.numpy(), data[1:4])

    def test_iterable_window(self, cube_fits):
        from torchfits.data import FitsCubeIterableDataset

        cube = next(
            iter(FitsCubeIterableDataset([str(cube_fits)], spectral_slice=(2, 5)))
        )
        assert cube.shape == (3, 4, 6)

    def test_slice_index_and_window_are_exclusive(self, cube_fits):
        from torchfits.data import FitsCubeDataset

        with pytest.raises(ValueError, match="not both"):
            FitsCubeDataset([str(cube_fits)], slice_index=0, spectral_slice=(1, 2))

    def test_invalid_window_rejected(self, cube_fits):
        from torchfits.data import FitsCubeDataset

        with pytest.raises(ValueError, match="spectral_slice"):
            FitsCubeDataset([str(cube_fits)], spectral_slice=(3, 3))

    def test_window_applies_to_companions_too(self, tmp_path):
        from torchfits.data import FitsCubeIterableDataset

        path = tmp_path / "cube_mef.fits"
        data = np.arange(120, dtype=np.float32).reshape(5, 4, 6)
        fits.HDUList(
            [
                fits.PrimaryHDU(data),
                fits.ImageHDU(np.ones_like(data), name="IVAR"),
            ]
        ).writeto(str(path), overwrite=True)

        payload = next(
            iter(
                FitsCubeIterableDataset(
                    [str(path)], hdu=0, ivar_hdu="IVAR", spectral_slice=(0, 2)
                )
            )
        )
        assert payload["flux"].shape == (2, 4, 6)
        assert payload["ivar"].shape == (2, 4, 6)


# ---------------------------------------------------------------------------
# Tables: row-range sharding and tensor-space streaming
# ---------------------------------------------------------------------------


class TestTableSharding:
    def test_shard_ranges_are_contiguous_and_even(self):
        from torchfits.data import _shard_row_range

        assert [_shard_row_range(10, i, 3) for i in range(3)] == [
            (0, 4),
            (4, 7),
            (7, 10),
        ]
        assert [_shard_row_range(2, i, 4) for i in range(4)] == [
            (0, 1),
            (1, 2),
            (2, 2),
            (2, 2),
        ]
        assert _shard_row_range(0, 0, 4) == (0, 0)

    def test_rank_shards_partition_every_row_exactly_once(self, table_fits):
        from torchfits.data import FitsTableIterableDataset

        first = [
            row["flux"].item()
            for row in FitsTableIterableDataset(str(table_fits), rank=0, world_size=2)
        ]
        second = [
            row["flux"].item()
            for row in FitsTableIterableDataset(str(table_fits), rank=1, world_size=2)
        ]
        assert first == [0.0, 1.0, 2.0, 3.0]
        assert second == [4.0, 5.0, 6.0, 7.0]

    def test_row_window_skips_unrelated_rows(self, table_fits):
        import torchfits.table

        chunks = list(
            torchfits.table.scan_torch(
                str(table_fits), hdu=1, row_slice=(3, 6), batch_size=2
            )
        )
        values = [v for chunk in chunks for v in chunk["flux"].tolist()]
        assert values == [3.0, 4.0, 5.0]

    def test_tensor_space_streaming_yields_row_batches(self, table_fits):
        from torchfits.data import FitsTableIterableDataset

        ds = FitsTableIterableDataset(str(table_fits), batch_size=3, as_batches=True)
        chunks = list(ds)
        assert [int(chunk["flux"].shape[0]) for chunk in chunks] == [3, 3, 2]
        assert chunks[0]["flux"].tolist() == [0.0, 1.0, 2.0]

    def test_batch_mode_shards_by_row_range(self, table_fits):
        from torchfits.data import FitsTableIterableDataset

        ds = FitsTableIterableDataset(
            str(table_fits), batch_size=2, as_batches=True, rank=1, world_size=2
        )
        values = [v for chunk in ds for v in chunk["flux"].tolist()]
        assert values == [4.0, 5.0, 6.0, 7.0]

    def test_shuffle_buffer_rejects_batch_mode(self, table_fits):
        from torchfits.data import FitsTableIterableDataset

        ds = FitsTableIterableDataset(
            str(table_fits), as_batches=True, shuffle_buffer_size=4
        )
        with pytest.raises(ValueError, match="as_batches"):
            list(ds)


# ---------------------------------------------------------------------------
# Band discovery + zeropoints
# ---------------------------------------------------------------------------


class TestBandDiscovery:
    def test_discovers_named_bands_with_zeropoints(self, bands_fits):
        from torchfits.data import discover_bands

        bands = discover_bands(str(bands_fits))
        flux_bands = [b for b in bands if b.role == "flux"]
        assert [b.name for b in flux_bands] == ["G", "R", "Z"]
        assert {b.role for b in bands if b.name.endswith("_IVAR")} == {"ivar"}
        assert {b.role for b in bands if b.name.endswith("_DQ")} == {"mask"}
        assert flux_bands[0].shape == (4, 4)
        assert flux_bands[0].zeropoint == pytest.approx(25.0)
        assert flux_bands[0].exptime == pytest.approx(100.0)

    def test_flux_scale_uses_zeropoint(self, bands_fits):
        from torchfits.data import discover_bands

        g = next(b for b in discover_bands(str(bands_fits)) if b.name == "G")
        assert g.flux_scale() == pytest.approx(10.0 ** (-0.4 * 25.0))
        assert g.flux_scale(exptime_normalized=False) == pytest.approx(
            10.0 ** (-0.4 * 25.0) / 100.0
        )

    def test_flux_scale_is_none_without_zeropoint(self, tmp_path):
        from torchfits.data import discover_bands

        path = tmp_path / "plain.fits"
        fits.PrimaryHDU(np.ones((3, 3), dtype=np.float32)).writeto(
            str(path), overwrite=True
        )
        band = discover_bands(str(path))[0]
        assert band.role == "flux"
        assert band.zeropoint is None
        assert band.flux_scale() is None

    def test_from_bands_wires_flux_and_companions(self, bands_fits):
        from torchfits.data import FitsImageDataset

        ds = FitsImageDataset.from_bands(str(bands_fits), bands=["G", "R"])
        payload, label = ds[0]
        assert ds.hdus == ["G", "R"]
        assert payload["flux"].shape == (2, 4, 4)
        assert payload["ivar"].shape == (2, 4, 4)
        assert payload["mask"].shape == (2, 4, 4)
        assert int(label) == 0

    def test_from_bands_selects_single_band(self, bands_fits):
        from torchfits.data import FitsImageDataset

        ds = FitsImageDataset.from_bands(str(bands_fits), bands=["Z"])
        payload, _ = ds[0]
        assert payload["flux"].shape == (1, 4, 4)
        assert payload["flux"].mean().item() == pytest.approx(3.0)

    def test_from_bands_reports_unknown_band(self, bands_fits):
        from torchfits.data import FitsImageDataset

        with pytest.raises(ValueError, match="available"):
            FitsImageDataset.from_bands(str(bands_fits), bands=["NIR"])

    def test_band_zeropoints_map(self, bands_fits):
        from torchfits.data import FitsImageDataset

        ds = FitsImageDataset.from_bands(str(bands_fits))
        assert ds.band_zeropoints() == {"G": 25.0, "R": 24.0, "Z": 23.0}


# ---------------------------------------------------------------------------
# Regressions found by adversarial probing
# ---------------------------------------------------------------------------


class TestSpectrumRowValidation:
    """``row=`` on a 1D spectrum used to return a 0-d scalar silently."""

    @pytest.fixture
    def spectrum_1d_fits(self, tmp_path):
        path = tmp_path / "spec1d.fits"
        fits.PrimaryHDU(np.linspace(0, 1, 32, dtype=np.float32)).writeto(
            str(path), overwrite=True
        )
        return path

    def test_row_on_1d_spectrum_raises(self, spectrum_1d_fits) -> None:
        from torchfits.data import FitsSpectrumDataset

        with pytest.raises(ValueError, match="row=0 needs a multi-row"):
            FitsSpectrumDataset([str(spectrum_1d_fits)], row=0)[0]

    def test_row_on_multispectrum_still_works(self, tmp_path) -> None:
        from torchfits.data import FitsSpectrumDataset

        path = tmp_path / "multi.fits"
        fits.PrimaryHDU(np.arange(24, dtype=np.float32).reshape(3, 8)).writeto(
            str(path), overwrite=True
        )
        payload = FitsSpectrumDataset([str(path)], row=1)[0]
        assert payload["flux"].shape == (8,)
        assert payload["flux"][0].item() == pytest.approx(8.0)


class TestTableBatchModeConsistency:
    """``as_batches=True`` must drop non-numeric columns on *both* scanner paths."""

    @pytest.fixture
    def mixed_table(self, tmp_path):
        path = tmp_path / "mixed.fits"
        table = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="flux", format="E", array=np.arange(8, dtype=np.float32)
                ),
                fits.Column(
                    name="name",
                    format="6A",
                    array=np.array([f"n{i}" for i in range(8)]),
                ),
                fits.Column(name="dq", format="J", array=np.arange(8, dtype=np.int32)),
            ]
        )
        fits.HDUList([fits.PrimaryHDU(), table]).writeto(str(path), overwrite=True)
        return path

    def test_batches_keep_only_numeric_columns(self, mixed_table) -> None:
        from torchfits.data import FitsTableIterableDataset

        chunks = list(
            FitsTableIterableDataset(str(mixed_table), as_batches=True, batch_size=4)
        )
        assert all(set(chunk) == {"flux", "dq"} for chunk in chunks)
        assert all(
            isinstance(value, torch.Tensor)
            for chunk in chunks
            for value in chunk.values()
        )

    def test_where_path_drops_characters_too(self, mixed_table) -> None:
        from torchfits.data import FitsTableIterableDataset

        chunks = list(
            FitsTableIterableDataset(
                str(mixed_table), where="flux >= 2.0", as_batches=True, batch_size=3
            )
        )
        assert chunks and all("name" not in chunk for chunk in chunks)

    def test_per_row_mode_keeps_characters(self, mixed_table) -> None:
        from torchfits.data import FitsTableIterableDataset

        rows = list(FitsTableIterableDataset(str(mixed_table), batch_size=8))
        assert len(rows) == 8 and "name" in rows[0]

    def test_where_shards_cover_every_matching_row_once(self, mixed_table) -> None:
        from torchfits.data import FitsTableIterableDataset

        seen: list[float] = []
        for rank in range(3):
            ds = FitsTableIterableDataset(
                str(mixed_table), where="flux >= 2.0", rank=rank, world_size=3
            )
            seen += [row["flux"].item() for row in ds]
        assert sorted(seen) == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    def test_table_datasets_expose_no_files_attribute(self, mixed_table) -> None:
        """Table datasets read one file; ``make_loader`` cache warm-up is a no-op."""
        from torchfits.data import FitsTableIterableDataset

        assert not getattr(FitsTableIterableDataset(str(mixed_table)), "files", None)


# ---------------------------------------------------------------------------
# Image mask companions: DQ decoding and rank alignment
# ---------------------------------------------------------------------------


def _band_mef(path, dq):
    """Write a single-band MEF with an IVAR and a DQ companion."""
    shape = (4, 4)
    hdus = [
        fits.PrimaryHDU(),
        fits.ImageHDU(np.full(shape, 5.0, dtype=np.float32), name="G"),
        fits.ImageHDU(np.full(shape, 0.25, dtype=np.float32), name="G_IVAR"),
        fits.ImageHDU(np.asarray(dq, dtype=np.int16), name="G_DQ"),
    ]
    fits.HDUList(hdus).writeto(str(path), overwrite=True)
    return path


class TestImageMaskCompanions:
    def test_clean_dq_marks_every_pixel_valid(self, bands_fits) -> None:
        """An all-zero DQ frame is *all good*, not all bad.

        Read verbatim, a clean DQ extension marks every pixel invalid and each
        mask-aware statistic collapses to NaN.
        """
        from torchfits.data import FitsImageDataset
        from torchfits.transforms import SigmaNormalize, estimate_background

        ds = FitsImageDataset.from_bands(str(bands_fits))
        payload, _ = ds[0]
        assert payload["mask"].dtype == torch.bool
        assert bool(payload["mask"].all())

        med, mad = estimate_background(payload["flux"], mask=payload["mask"])
        assert torch.isfinite(med).all()
        assert torch.isfinite(mad).all()
        out = SigmaNormalize()(payload)
        assert torch.isfinite(out["flux"]).all()

    def test_dq_bitmask_is_decoded_not_trusted(self, tmp_path) -> None:
        from torchfits.data import FitsImageDataset

        dq = np.zeros((4, 4), dtype=np.int16)
        dq[0, 0] = 4  # bit 2
        dq[3, 3] = 1024  # bit 10
        path = _band_mef(tmp_path / "dq.fits", dq)

        ds = FitsImageDataset.from_bands(str(path))
        assert ds.mask_is_dq is True
        mask = ds[0][0]["mask"]
        assert mask.dtype == torch.bool
        assert not mask[0, 0, 0] and not mask[0, 3, 3]
        assert int(mask.sum()) == 14

    def test_bad_bits_names_the_fatal_bits(self, tmp_path) -> None:
        from torchfits.data import FitsImageDataset

        dq = np.zeros((4, 4), dtype=np.int16)
        dq[0, 0] = 4  # bit 2 — fatal
        dq[3, 3] = 1024  # bit 10 — harmless here
        path = _band_mef(tmp_path / "bits.fits", dq)

        ds = FitsImageDataset.from_bands(str(path), bad_bits=[2])
        mask = ds[0][0]["mask"]
        assert not mask[0, 0, 0]
        assert mask[0, 3, 3]
        assert int(mask.sum()) == 15

    def test_mask_is_dq_override_keeps_nonzero_is_valid(self, tmp_path) -> None:
        from torchfits.data import FitsImageDataset

        dq = np.zeros((4, 4), dtype=np.int16)
        dq[0, 0] = 4
        path = _band_mef(tmp_path / "raw.fits", dq)

        # Without decoding, the companion is read as a plain validity array:
        # nonzero means valid, so the *inverted* meaning of a DQ bitfield —
        # which is exactly why the flag exists.
        ds = FitsImageDataset.from_bands(str(path), mask_is_dq=False)
        mask = ds[0][0]["mask"]
        assert mask.dtype == torch.bool
        assert bool(mask[0, 0, 0]) and int(mask.sum()) == 1

    def test_explicit_mask_hdu_is_a_validity_mask(self, tmp_path) -> None:
        from torchfits.data import FitsImageDataset

        dq = np.zeros((4, 4), dtype=np.int16)
        dq[0, 0] = 2
        path = _band_mef(tmp_path / "explicit.fits", dq)

        ds = FitsImageDataset(str(path), hdu="G", mask_hdu="G_DQ", mask_is_dq=True)
        mask = ds[0][0]["mask"]
        assert mask.dtype == torch.bool
        assert int(mask.sum()) == 15

    def test_companions_match_flux_rank(self, tmp_path) -> None:
        """One band still gets a channel axis, so ``mask[0]`` is channel 0."""
        from torchfits.data import FitsImageDataset

        dq = np.zeros((4, 4), dtype=np.int16)
        dq[0, 0] = 2
        path = _band_mef(tmp_path / "rank.fits", dq)

        payload, _ = FitsImageDataset(
            str(path),
            hdu="G",
            ivar_hdu="G_IVAR",
            mask_hdu="G_DQ",
            mask_is_dq=True,
        )[0]
        assert payload["flux"].shape == (1, 4, 4)
        assert payload["ivar"].shape == (1, 4, 4)
        assert payload["mask"].shape == (1, 4, 4)

        no_channel, _ = FitsImageDataset(
            str(path), hdu="G", mask_hdu="G_DQ", mask_is_dq=True, add_channel_dim=False
        )[0]
        assert no_channel["flux"].shape == (4, 4)
        assert no_channel["mask"].shape == (4, 4)

    def test_iterable_dataset_decodes_dq(self, tmp_path) -> None:
        from torchfits.data import FitsImageIterableDataset

        dq = np.zeros((4, 4), dtype=np.int16)
        dq[1, 1] = 8
        path = _band_mef(tmp_path / "iter.fits", dq)

        ds = FitsImageIterableDataset(
            [str(path)], hdu="G", mask_hdu="G_DQ", mask_is_dq=True
        )
        payload = next(iter(ds))
        assert payload["mask"].dtype == torch.bool
        assert int(payload["mask"].sum()) == 15

    def test_mixed_mask_and_dq_companions_are_rejected(self, tmp_path) -> None:
        """Two bands, one with ``_DQ`` and one with ``_MASK``: ambiguous."""
        from torchfits.data import FitsImageDataset

        path = tmp_path / "mixed.fits"
        shape = (4, 4)
        hdus = [fits.PrimaryHDU()]
        for name in ("G", "R"):
            hdus.append(fits.ImageHDU(np.ones(shape, dtype=np.float32), name=name))
        hdus.append(fits.ImageHDU(np.zeros(shape, dtype=np.int16), name="G_DQ"))
        hdus.append(fits.ImageHDU(np.ones(shape, dtype=np.int16), name="R_MASK"))
        fits.HDUList(hdus).writeto(str(path), overwrite=True)

        with pytest.raises(ValueError, match="mix DQ bitfields and plain masks"):
            FitsImageDataset.from_bands(str(path))


# ---------------------------------------------------------------------------
# R3a review: spectral-axis correctness, label integrity, sharding exactness
# ---------------------------------------------------------------------------


def _install_fake_worker(monkeypatch, worker_id: int, num_workers: int) -> None:
    """Pretend this call runs inside DataLoader worker ``worker_id``."""
    import types

    info = types.SimpleNamespace(id=worker_id, num_workers=num_workers)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: info)


class TestCubeSpectralAxisAcrossBands:
    """``spectral_slice``/``slice_index`` act on the spectral axis of every
    band, not on the channel axis a multi-HDU stack puts in front of it."""

    @pytest.fixture
    def two_band_cubes(self, tmp_path):
        path = tmp_path / "mcube.fits"
        data = np.arange(2 * 5 * 4 * 6, dtype=np.float32).reshape(2, 5, 4, 6)
        fits.HDUList(
            [
                fits.PrimaryHDU(data[0]),
                fits.ImageHDU(data[1], name="CUBE2"),
                fits.ImageHDU(np.ones_like(data[0]), name="IVAR1"),
                fits.ImageHDU(np.ones_like(data[1]), name="IVAR2"),
            ]
        ).writeto(str(path), overwrite=True)
        return path, data

    def test_map_window_slices_spectral_axis(self, two_band_cubes):
        from torchfits.data import FitsCubeDataset

        path, data = two_band_cubes
        payload, _ = FitsCubeDataset(
            [str(path)], hdu=[0, "CUBE2"], spectral_slice=(1, 4), labels=[0]
        )[0]
        assert payload.shape == (2, 3, 4, 6)
        assert np.allclose(payload.numpy(), data[:, 1:4])

    def test_slice_index_selects_plane_from_each_band(self, two_band_cubes):
        from torchfits.data import FitsCubeDataset

        path, data = two_band_cubes
        payload, _ = FitsCubeDataset(
            [str(path)], hdu=[0, "CUBE2"], slice_index=1, labels=[0]
        )[0]
        assert payload.shape == (2, 4, 6)
        assert np.allclose(payload.numpy(), data[:, 1])

    def test_iterable_window_slices_spectral_axis(self, two_band_cubes):
        from torchfits.data import FitsCubeIterableDataset

        path, data = two_band_cubes
        payload = next(
            iter(
                FitsCubeIterableDataset(
                    [str(path)],
                    hdu=[0, "CUBE2"],
                    ivar_hdu=["IVAR1", "IVAR2"],
                    spectral_slice=(1, 4),
                )
            )
        )
        assert payload["flux"].shape == (2, 3, 4, 6)
        assert payload["ivar"].shape == (2, 3, 4, 6)
        assert np.allclose(payload["flux"].numpy(), data[:, 1:4])

    def test_window_bounds_clamp_or_reject(self, two_band_cubes):
        from torchfits.data import FitsCubeDataset

        path, data = two_band_cubes
        # stop beyond the extent clamps to the available planes
        payload, _ = FitsCubeDataset(
            [str(path)], hdu=0, spectral_slice=(3, 99), labels=[0]
        )[0]
        assert payload.shape == (2, 4, 6)
        assert np.allclose(payload.numpy(), data[0][3:])
        # start beyond the extent yields an empty spectral axis (Python slice
        # semantics) with the dtype preserved
        payload, _ = FitsCubeDataset(
            [str(path)], hdu=0, spectral_slice=(9, 12), labels=[0]
        )[0]
        assert payload.shape == (0, 4, 6)
        assert payload.dtype == torch.float32
        for bad in ((-1, 2), (3, 3), (4, 2)):
            with pytest.raises(ValueError, match="spectral_slice"):
                FitsCubeDataset([str(path)], hdu=0, spectral_slice=bad, labels=[0])


class TestSpectrumIterableLabelIntegrity:
    def test_duplicate_paths_keep_distinct_labels(self, spectrum_mef):
        from torchfits.data import FitsSpectrumIterableDataset

        ds = FitsSpectrumIterableDataset(
            [str(spectrum_mef), str(spectrum_mef)],
            hdu="FLUX",
            row=0,
            labels=[1, 2],
        )
        assert sorted(int(label) for _, label in ds) == [1, 2]

    def test_label_key_resolves_remote_through_cache(self, tmp_path):
        """``label_key=`` must read headers from the cached local copy.

        It used to hand the raw remote URI straight to ``read_keys``: vos/URL
        spectra failed at construction (and HTTP fetched the file once via
        CFITSIO and again via ``resolve_local_path``).
        """
        import shutil
        from unittest import mock

        import torchfits.data.remote as remote
        from torchfits.data import FitsSpectrumDataset

        fixture = tmp_path / "spec.fits"
        header = fits.PrimaryHDU().header
        header["CLASS"] = 3
        fits.HDUList(
            [
                fits.PrimaryHDU(header=header),
                fits.ImageHDU(
                    np.arange(8, dtype=np.float32)[None, :].repeat(2, 0), name="FLUX"
                ),
            ]
        ).writeto(str(fixture), overwrite=True)

        calls: list[str] = []

        def _serve(url, dest):
            calls.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(fixture, dest)
            return dest

        url = "vos://vos.test!spec.fits"
        with mock.patch.object(remote, "_download", side_effect=_serve):
            ds = FitsSpectrumDataset(
                [url],
                hdu="FLUX",
                row=0,
                label_key="CLASS",
                cache_dir=tmp_path / "cache",
            )
            payload, label = ds[0]

        assert payload["flux"].shape == (8,)
        assert int(label) == 3
        assert calls == [url]


class TestTableColumnSpectrumContract:
    """``column=`` table spectra are single-arm: say so at construction.

    ``hdu=[1, 2]`` with ``column=`` used to read only the first table and
    silently drop the second arm under ``layout="dict"`` (the "single arm"
    guard was layout-gated and ran at read time).
    """

    @pytest.fixture
    def two_table_spectra(self, tmp_path):
        path = tmp_path / "tables.fits"
        hdus = [fits.PrimaryHDU()]
        for _ in range(2):
            hdus.append(
                fits.BinTableHDU.from_columns(
                    [
                        fits.Column(
                            name="FLUX",
                            format="E",
                            array=np.arange(6, dtype=np.float32),
                        )
                    ]
                )
            )
        fits.HDUList(hdus).writeto(str(path), overwrite=True)
        return path

    def test_multi_hdu_table_spectra_rejected(self, two_table_spectra):
        from torchfits.data import FitsSpectrumDataset

        with pytest.raises(ValueError, match="single arm"):
            FitsSpectrumDataset([str(two_table_spectra)], hdu=[1, 2], column="FLUX")

    def test_named_hdu_table_spectra_rejected_at_construction(self, two_table_spectra):
        from torchfits.data import FitsSpectrumDataset

        with pytest.raises(ValueError, match="integer hdu"):
            FitsSpectrumDataset([str(two_table_spectra)], hdu="T1", column="FLUX")

    def test_iterable_rejects_multi_hdu_table_spectra(self, two_table_spectra):
        from torchfits.data import FitsSpectrumIterableDataset

        with pytest.raises(ValueError, match="single arm"):
            FitsSpectrumIterableDataset(
                [str(two_table_spectra)], hdu=[1, 2], column="FLUX"
            )


class TestBandZeropointContracts:
    def test_band_zeropoints_reports_selected_bands_only(self, bands_fits):
        from torchfits.data import FitsImageDataset

        ds = FitsImageDataset.from_bands(str(bands_fits), bands=["G", "R"])
        assert ds.band_zeropoints() == {"G": 25.0, "R": 24.0}

    def test_flux_scale_per_second_requires_exptime(self, tmp_path):
        from torchfits.data import discover_bands

        path = tmp_path / "zp.fits"
        hdu = fits.ImageHDU(np.ones((3, 3), dtype=np.float32), name="G")
        hdu.header["ZP"] = 25.0
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(str(path), overwrite=True)
        g = next(b for b in discover_bands(str(path)) if b.name == "G")
        assert g.flux_scale() == pytest.approx(10.0 ** (-0.4 * 25.0))
        with pytest.raises(ValueError, match="EXPTIME"):
            g.flux_scale(exptime_normalized=False)


class TestDiscoverBandsRobustness:
    def test_skips_table_hdus(self, tmp_path):
        from torchfits.data import discover_bands

        path = tmp_path / "mix.fits"
        cols = [
            fits.Column(name="flux", format="E", array=np.arange(4, dtype=np.float32))
        ]
        fits.HDUList(
            [
                fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32)),
                fits.BinTableHDU.from_columns(cols),
                fits.ImageHDU(np.ones((2, 2), dtype=np.float32), name="G"),
            ]
        ).writeto(str(path), overwrite=True)
        assert [b.name for b in discover_bands(str(path))] == ["HDU0", "G"]

    def test_surfaces_shape_failures_on_image_hdus(self, tmp_path, monkeypatch):
        """A read failure on an image extension must not silently vanish.

        The band list feeds ``from_bands``; a silently dropped extension
        builds a dataset with fewer channels than the file has.
        """
        import torchfits
        from torchfits.data import discover_bands

        path = tmp_path / "img.fits"
        fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32)).writeto(
            str(path), overwrite=True
        )
        real = torchfits.read_shape

        def _boom(p, hdu=0):
            if int(hdu) == 0:
                raise RuntimeError("Could not read image parameters")
            return real(p, hdu)

        monkeypatch.setattr(torchfits, "read_shape", _boom)
        with pytest.raises(RuntimeError):
            discover_bands(str(path))

    def test_surfaces_header_failures(self, tmp_path, monkeypatch):
        import torchfits
        from torchfits.data import discover_bands

        path = tmp_path / "img.fits"
        fits.PrimaryHDU(np.ones((2, 2), dtype=np.float32)).writeto(
            str(path), overwrite=True
        )

        def _boom(p, hdu=0):
            raise OSError("truncated")

        monkeypatch.setattr(torchfits, "read_header", _boom)
        with pytest.raises(OSError):
            discover_bands(str(path))


class TestIterableFileSharding:
    """Exact partition of files across ranks and DataLoader workers."""

    @pytest.fixture
    def tagged_images(self, tmp_path):
        paths = []
        for i in range(5):
            p = tmp_path / f"img_{i}.fits"
            fits.PrimaryHDU(np.full((2, 2), float(i), dtype=np.float32)).writeto(
                str(p), overwrite=True
            )
            paths.append(str(p))
        return paths

    @staticmethod
    def _values(items):
        return [float(t.flatten()[0]) for t in items]

    def test_partition_exact_with_workers(self, tagged_images, monkeypatch):
        from torchfits.data import FitsTensorIterableDataset

        for world_size, num_workers in ((1, 1), (2, 2), (3, 2), (8, 3)):
            seen: list[float] = []
            for rank in range(world_size):
                for worker_id in range(num_workers):
                    _install_fake_worker(monkeypatch, worker_id, num_workers)
                    ds = FitsTensorIterableDataset(
                        tagged_images, rank=rank, world_size=world_size
                    )
                    seen += self._values(ds)
            assert sorted(seen) == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_shuffle_deterministic_and_shards_disjoint(
        self, tagged_images, monkeypatch
    ):
        from torchfits.data import FitsTensorIterableDataset

        kwargs = dict(shuffle=True, shuffle_buffer_size=3, seed=11, world_size=2)
        ds0 = FitsTensorIterableDataset(tagged_images, rank=0, **kwargs)
        epoch1 = self._values(ds0)
        assert self._values(ds0) == epoch1  # same seed => same order every epoch
        assert set(epoch1) == {0.0, 2.0, 4.0}  # count preserved under shuffle

        ds1 = FitsTensorIterableDataset(tagged_images, rank=1, **kwargs)
        rank1 = self._values(ds1)
        assert set(rank1) == {1.0, 3.0}  # different ranks disjoint
        assert set(epoch1) | set(rank1) == {0.0, 1.0, 2.0, 3.0, 4.0}

        _install_fake_worker(monkeypatch, 0, 2)
        w0 = self._values(ds0)
        _install_fake_worker(monkeypatch, 0, 2)
        assert self._values(ds0) == w0  # worker shards deterministic
        _install_fake_worker(monkeypatch, 1, 2)
        w1 = self._values(ds0)
        assert not set(w0) & set(w1)  # workers disjoint
        assert sorted(w0 + w1) == [0.0, 2.0, 4.0]  # workers cover the rank shard

    def test_iterable_epoch_yield_matches_map_len(self, tagged_images):
        from torchfits.data import FitsTensorDataset, FitsTensorIterableDataset

        ds_map = FitsTensorDataset(tagged_images)
        ds_iter = FitsTensorIterableDataset(
            tagged_images, shuffle=True, shuffle_buffer_size=2, seed=1
        )
        assert len(list(ds_iter)) == len(ds_map) == 5

    def test_rank_out_of_range_raises(self, tagged_images):
        from torchfits.data import FitsTensorIterableDataset

        with pytest.raises(ValueError, match="rank"):
            list(FitsTensorIterableDataset(tagged_images, rank=2, world_size=2))
        with pytest.raises(ValueError, match="rank"):
            list(FitsTensorIterableDataset(tagged_images, rank=-1, world_size=2))
        with pytest.raises(ValueError, match="world_size"):
            list(FitsTensorIterableDataset(tagged_images, rank=0, world_size=0))
