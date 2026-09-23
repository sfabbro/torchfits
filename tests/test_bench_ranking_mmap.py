"""Ranking groups must not mix mmap-on and mmap-off peers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_contract import annotate_rankings, compute_deficits


def _row(
    *,
    domain: str = "fits",
    case_id: str,
    family: str,
    library: str,
    method: str,
    mmap_target: str,
    time_s: float,
    comparable: bool = True,
) -> dict:
    return {
        "domain": domain,
        "case_id": case_id,
        "case_label": case_id,
        "operation": "read_full",
        "family": family,
        "library": library,
        "method": method,
        "status": "OK",
        "comparable": comparable,
        "mmap_target": mmap_target,
        "time_s": time_s,
        "n_points": 64,
        "metadata": {},
    }


def test_mmap_modes_rank_independently() -> None:
    rows = [
        _row(
            case_id="tiny_int8_1d::read_full_gpu",
            family="smart",
            library="torchfits",
            method="torchfits",
            mmap_target="off",
            time_s=7.26e-3,
        ),
        _row(
            case_id="tiny_int8_1d::read_full_gpu",
            family="smart",
            library="fitsio",
            method="fitsio_torch",
            mmap_target="off",
            time_s=7.40e-3,
        ),
        # Faster fitsio on mmap-on must not demote mmap-off torchfits.
        _row(
            case_id="tiny_int8_1d::read_full_gpu",
            family="smart",
            library="fitsio",
            method="fitsio_torch",
            mmap_target="on",
            time_s=7.20e-3,
        ),
        _row(
            case_id="tiny_int8_1d::read_full_gpu",
            family="smart",
            library="torchfits",
            method="torchfits",
            mmap_target="on",
            time_s=9.33e-3,
        ),
    ]
    annotate_rankings(rows)
    off_tf = next(
        r for r in rows if r["mmap_target"] == "off" and r["library"] == "torchfits"
    )
    on_tf = next(
        r for r in rows if r["mmap_target"] == "on" and r["library"] == "torchfits"
    )
    assert off_tf["rank_in_family"] == 1
    assert off_tf["best_in_family"] is True
    assert on_tf["rank_in_family"] == 2

    # Images: any meaningful lag is a deficit (~15.7% here).
    deficits = compute_deficits(rows, run_id="test")
    assert len(deficits) == 1
    assert deficits[0]["mmap_target"] == "on"


def test_image_deficits_count_any_lag() -> None:
    rows = [
        _row(
            case_id="noise::read_full",
            family="smart",
            library="torchfits",
            method="torchfits",
            mmap_target="off",
            time_s=2.20e-3,
        ),
        _row(
            case_id="noise::read_full",
            family="smart",
            library="fitsio",
            method="fitsio_torch",
            mmap_target="off",
            time_s=1.00e-3,
        ),
    ]
    annotate_rankings(rows)
    deficits = compute_deficits(rows, run_id="test")
    assert len(deficits) == 1
    assert deficits[0]["domain"] == "fits"


def test_timer_epsilon_absorbs_only_clock_noise() -> None:
    """Image float-timer ε is absolute (~0.2ms), never a percent-of-median floor."""
    rows = [
        _row(
            case_id="plain::read_full",
            family="smart",
            library="torchfits",
            method="torchfits",
            mmap_target="off",
            time_s=60.0e-3 + 1e-4,  # 0.1ms — under ε
        ),
        _row(
            case_id="plain::read_full",
            family="smart",
            library="fitsio",
            method="fitsio_torch",
            mmap_target="off",
            time_s=60.0e-3,
        ),
    ]
    annotate_rankings(rows)
    noise = compute_deficits(rows, run_id="test")
    assert len(noise) == 1
    assert noise[0]["significance"] == "noise"

    rows[0]["time_s"] = 60.0e-3 + 2.5e-4  # 0.25ms — above ε, is significant
    annotate_rankings(rows)
    deficits = compute_deficits(rows, run_id="test")
    assert len(deficits) == 1
    assert deficits[0]["significance"] == "significant"


def test_table_arrow_allows_1_05() -> None:
    rows = [
        _row(
            domain="fitstable",
            case_id="narrow::predicate",
            family="smart",
            library="torchfits",
            method="torchfits",
            mmap_target="off",
            time_s=1.05e-3,  # inclusive 1.05× slack
        ),
        _row(
            domain="fitstable",
            case_id="narrow::predicate",
            family="smart",
            library="fitsio",
            method="fitsio_torch",
            mmap_target="off",
            time_s=1.00e-3,
        ),
    ]
    annotate_rankings(rows)
    noise = compute_deficits(rows, run_id="test")
    assert len(noise) == 1
    assert noise[0]["significance"] == "noise"

    rows[0]["time_s"] = 1.06e-3
    annotate_rankings(rows)
    deficits = compute_deficits(rows, run_id="test")
    assert len(deficits) == 1
    assert deficits[0]["domain"] == "fitstable"
    assert deficits[0]["significance"] == "significant"


def test_deficits_require_external_peer() -> None:
    rows = [
        _row(
            case_id="gpu::read_full",
            family="specialized",
            library="torchfits",
            method="torchfits_specialized_device",
            mmap_target="off",
            time_s=2e-3,
        ),
        _row(
            case_id="gpu::read_full",
            family="specialized",
            library="torchfits",
            method="torchfits_dtype_fair_device",
            mmap_target="off",
            time_s=1e-3,
        ),
    ]
    annotate_rankings(rows)
    assert compute_deficits(rows, run_id="test") == []


def _one_second(methods, *, runs, warmup):
    _ = runs, warmup
    return {name: (1.0, None, None, None) for name in methods}


def _write_images(path: Path, n_ext: int, shape: tuple[int, int], dtype) -> None:
    from astropy.io import fits

    hdus = [fits.PrimaryHDU()]
    image = np.zeros(shape, dtype=dtype)
    for _ in range(n_ext):
        hdus.append(fits.ImageHDU(image.copy()))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def test_fits_cutout_throughput_uses_window_bytes(tmp_path, monkeypatch) -> None:
    """100x100 cutout MB/s is the window, not the whole file."""
    import benchmarks.bench_fits_io as fio

    path = tmp_path / "multi_mef_10ext.fits"
    _write_images(path, 6, (256, 256), np.float32)
    monkeypatch.setattr(fio, "time_medians_interleaved", _one_second)
    suite = fio.FITSBenchmarkSuite(output_dir=tmp_path, use_mmap=False, profile="user")
    rows = suite._benchmark_cutout_rows({"multi_mef_10ext": path}, runs=1, warmup=0)
    window_mb = (100 * 100 * 4) / (1024.0 * 1024.0)
    file_mb = path.stat().st_size / (1024.0 * 1024.0)
    assert file_mb > window_mb * 2
    assert rows[0]["size_mb"] == window_mb
    assert rows[0]["torchfits_mb_s"] == window_mb


def test_compressed_cutout_throughput_uses_zbitpix(tmp_path, monkeypatch) -> None:
    """Rice HDUs store the image type in ZBITPIX, not the tile-table BITPIX."""
    from astropy.io import fits

    import benchmarks.bench_fits_io as fio

    path = tmp_path / "compressed_rice_1.fits"
    image = np.zeros((256, 256), dtype=np.int16)
    fits.HDUList(
        [fits.PrimaryHDU(), fits.CompImageHDU(image, compression_type="RICE_1")]
    ).writeto(path, overwrite=True)
    monkeypatch.setattr(fio, "time_medians_interleaved", _one_second)
    suite = fio.FITSBenchmarkSuite(output_dir=tmp_path, use_mmap=False, profile="user")
    rows = suite._benchmark_cutout_rows({"compressed_rice_1": path}, runs=1, warmup=0)
    window_mb = (100 * 100 * 2) / (1024.0 * 1024.0)
    assert rows[0]["size_mb"] == window_mb
    assert rows[0]["torchfits_mb_s"] == window_mb


def test_repeated_cutout_throughput_counts_every_window(tmp_path, monkeypatch) -> None:
    import benchmarks.bench_fits_io as fio

    path = tmp_path / "medium_float32_2d.fits"
    from astropy.io import fits

    fits.PrimaryHDU(np.zeros((256, 256), dtype=np.float32)).writeto(
        path, overwrite=True
    )
    monkeypatch.setattr(fio, "time_medians_interleaved", _one_second)
    suite = fio.FITSBenchmarkSuite(output_dir=tmp_path, use_mmap=False, profile="user")
    rows = suite._benchmark_repeated_cutout_rows(
        {"medium_float32_2d": path}, runs=1, warmup=0
    )
    payload_mb = (50 * 100 * 100 * 4) / (1024.0 * 1024.0)
    assert rows[0]["size_mb"] == payload_mb
    assert rows[0]["torchfits_mb_s"] == payload_mb


def test_random_ext_throughput_counts_each_full_read(tmp_path, monkeypatch) -> None:
    import benchmarks.bench_fits_io as fio

    path = tmp_path / "multi_mef_10ext.fits"
    _write_images(path, 10, (8, 8), np.float32)
    monkeypatch.setattr(fio, "time_medians_interleaved", _one_second)
    suite = fio.FITSBenchmarkSuite(output_dir=tmp_path, use_mmap=False, profile="user")
    row = suite._benchmark_random_extensions(
        {"multi_mef_10ext": path}, runs=1, warmup=0
    )
    assert row is not None
    payload_mb = (200 * 8 * 8 * 4) / (1024.0 * 1024.0)
    assert row["size_mb"] == payload_mb
    assert row["torchfits_mb_s"] == payload_mb


def test_header_row_does_not_claim_ops_per_second(tmp_path, monkeypatch) -> None:
    import benchmarks.bench_fits_io as fio

    path = tmp_path / "img.fits"
    _write_images(path, 1, (8, 8), np.float32)
    monkeypatch.setattr(fio, "time_medians_interleaved", _one_second)
    suite = fio.FITSBenchmarkSuite(output_dir=tmp_path, use_mmap=False, profile="user")
    rows = fio._benchmark_headers(
        run_id="t",
        files={"img": path},
        suite=suite,
        mmap_target="off",
        runs=1,
        warmup=0,
    )
    assert rows[0]["throughput"] in ("", None)
    assert rows[0]["unit"] == ""
