# Upstream Parity Matrix

This matrix tracks the upstream correctness surfaces that `torchfits` now treats as validation contracts. It mirrors the plan categories:

- `exact-replay`: same upstream-style cases or fixture classes
- `adapter-replay`: same semantic contract through a torchfits adapter surface
- `example-smoke`: canonical upstream docs/README workflow
- `benchmark-hook`: replay or benchmark lane exists for the same workflow family

## Astropy

| Upstream surface | Source | Local coverage | Contract |
|---|---|---|---|
| WCS core fixture-heavy tests | `astropy/wcs/tests/test_wcs.py` | `tests/test_wcs_parity.py`, `tests/test_wcs_allsky.py`, `tests/test_wcs_tpv.py`, `tests/test_wcs_legacy.py`, `tests/test_wcs_zpx.py`, `benchmarks/replays/replay_upstream_astropy_wcs.py` | `exact-replay`, `benchmark-hook` |
| FITS docs/examples | `astropy.io.fits` docs | `tests/test_release_smoke.py`, `benchmarks/replays/replay_upstream_fits_workflows.py` | `adapter-replay`, `example-smoke` |

## Healpy

| Upstream surface | Source | Local coverage | Contract |
|---|---|---|---|
| HEALPix primitive tests/tutorial flows | healpy tests/tutorials | `tests/test_healpix_parity.py`, `benchmarks/replays/replay_upstream_astropy_healpy.py`, `benchmarks/replays/replay_upstream_test_functions.py`, `benchmarks/replays/replay_upstream_healpy_interp_edges.py`, `benchmarks/replays/replay_upstream_healpy_spin.py` | `exact-replay`, `benchmark-hook` |

## HealSparse

| Upstream surface | Source | Local coverage | Contract |
|---|---|---|---|
| I/O lifecycle | `tests/test_io.py` | `tests/test_healsparse_upstream.py`, `benchmarks/replays/replay_upstream_healsparse.py` | `adapter-replay`, `benchmark-hook` |
| Degrade semantics | `tests/test_degrade.py` | `tests/test_healsparse_upstream.py`, `benchmarks/replays/replay_upstream_healsparse.py` | `adapter-replay`, `benchmark-hook` |
| Geometry and map operations | `tests/test_geom.py`, `tests/test_operations.py` | `tests/test_healsparse_upstream.py`, `benchmarks/replays/replay_upstream_healsparse.py` | `adapter-replay` |
| Quickstart/basic interface | `basic_interface.html` | `tests/test_healsparse_upstream.py` | `example-smoke` |

## Fitsio

| Upstream surface | Source | Local coverage | Contract |
|---|---|---|---|
| README workflows | `fitsio/README.md` | `tests/test_fitsio_upstream_smoke.py`, `benchmarks/replays/replay_upstream_fits_workflows.py` | `example-smoke`, `benchmark-hook` |
| Image/header/table/compression tests | `fitsio/tests/test_image.py`, `test_header.py`, `test_table.py`, `test_image_compression.py` | `tests/test_fitsio_upstream_smoke.py`, `benchmarks/replays/replay_upstream_fits_workflows.py` | `adapter-replay`, `benchmark-hook` |

## Real-Data Reference Fixtures

| Real-data surface | Source | Local coverage | Contract |
|---|---|---|---|
| Installed scaled/VLA/WCS FITS fixtures | `astropy` package test data | `tests/test_real_data_validation.py`, `benchmarks/replays/replay_real_data_validation.py` | `adapter-replay`, `benchmark-hook` |
| Installed compressed image fixture | `fitsio` package test image | `tests/test_real_data_validation.py`, `benchmarks/replays/replay_real_data_validation.py` | `adapter-replay`, `benchmark-hook` |
| Installed HEALPix weight table | `healpy` package data | `tests/test_real_data_validation.py`, `benchmarks/replays/replay_real_data_validation.py` | `adapter-replay`, `benchmark-hook` |
| Public sample/reference FITS image | Astropy data server tutorial image | `benchmarks/replays/replay_real_data_validation.py` | `example-smoke`, `benchmark-hook` |

## Notes

- The authoritative source inventory lives in `benchmarks/replays/upstream_sources.json`.
- `healsparse` remains adapter-based rather than exact API replay because `torchfits` intentionally exposes sparse-map functionality through its own spherical namespace.
- `fitsio` parity is workflow-oriented: the contract is image/table/header/compression behavior, not a verbatim `fitsio.FITS` object model.
- The real-data lane is intentionally complementary to the upstream fixture replays: it validates installed package artifacts and a cached public FITS sample without replacing the stricter upstream parity gates.
