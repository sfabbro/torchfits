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
