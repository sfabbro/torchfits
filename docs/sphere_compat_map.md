# Sphere Compatibility Map

Status key:
- `parity`: implemented with upstream-like semantics
- `extended`: implemented, with torchfits-native extensions/contract details
- `partial`: implemented but with known semantic gaps
- `planned`: not yet implemented

| Function | Module | Status | Notes |
|---|---|---|---|
| `ang2pix` | `torchfits.sphere.compat` | parity | ring/nest, lonlat/theta-phi paths |
| `pix2ang` | `torchfits.sphere.compat` | parity | ring/nest, lonlat/theta-phi paths |
| `ring2nest` | `torchfits.sphere.compat` | parity | CPU parity-tested |
| `nest2ring` | `torchfits.sphere.compat` | parity | CPU parity-tested |
| `get_all_neighbours` | `torchfits.sphere.compat` | parity | pixel and angular forms |
| `get_all_neighbors` | `torchfits.sphere.compat` | parity | US spelling alias |
| `get_interp_weights` | `torchfits.sphere.compat` | parity | 4-neighbor interpolation weights |
| `get_interp_val` | `torchfits.sphere.compat` | parity | map interpolation |
| `ud_grade` | `torchfits.sphere.compat` | parity | parity-tested across ordering, `power`, `pess`, and bad-mask (`badval`/NaN/Inf) semantics |
| `query_disc` | `torchfits.sphere.compat` | parity | inclusive mode supported |
| `query_circle` | `torchfits.sphere.compat` | parity | lon/lat wrapper; strict mode can dispatch to canonical `hpgeom` semantics |
| `query_polygon` | `torchfits.sphere.compat` | parity | convex polygon query path |
| `query_box` | `torchfits.sphere.compat` | parity | wrap-aware longitude intervals |
| `query_strip` | `torchfits.sphere.compat` | parity | colatitude strip query |
| `query_ellipse` | `torchfits.sphere.compat` | parity | hpgeom-style units/options (`degrees`/`lonlat`/`return_pixel_ranges`); `backend='auto'` dispatches to canonical `hpgeom` when available with torch-native fallback |
| `query_circle_vec` | `torchfits.sphere.compat` | parity | cartesian-vector circle query (`hpgeom`-style); strict mode uses canonical `hpgeom` |
| `query_polygon_vec` | `torchfits.sphere.compat` | parity | cartesian-vector polygon query (`hpgeom`-style) |
| `pixel_ranges_to_pixels` | `torchfits.sphere.compat` | parity | `hpgeom` half-open/closed range expansion semantics |
| `upgrade_pixel_ranges` | `torchfits.sphere.compat` | parity | NEST pixel-range upgrade semantics |
| `pixels_to_pixel_ranges` | `torchfits.sphere.compat` | extended | torchfits-native range compression helper |
| `boundaries` | `torchfits.sphere.compat` | parity | healpy-style boundary sampling |
| `vec2pix` | `torchfits.sphere.compat` | parity | xyz -> pixel |
| `pix2vec` | `torchfits.sphere.compat` | parity | pixel -> xyz |
| `ang2vec` | `torchfits.sphere.compat` | parity | angle -> xyz |
| `vec2ang` | `torchfits.sphere.compat` | parity | xyz -> angle |
| `max_pixrad` | `torchfits.sphere.compat` | parity | conservative radius |
| `isnsideok` | `torchfits.sphere.compat` | parity | scalar + tensor forms |
| `isnpixok` | `torchfits.sphere.compat` | parity | scalar + tensor forms |
| `spherical_polygon_contains` | `torchfits.sphere.geom` | extended | inside-point disambiguation supported |
| `spherical_polygons_intersect` | `torchfits.sphere.geom` | extended | edge crossing + containment logic |
| `SphericalBooleanRegion.union` | `torchfits.sphere.geom` | extended | controlled-error boolean algebra + optional exact backend via `to_exact()` |
| `SphericalBooleanRegion.intersection` | `torchfits.sphere.geom` | extended | controlled-error boolean algebra + optional exact backend via `to_exact()` |
| `SphericalBooleanRegion.difference` | `torchfits.sphere.geom` | extended | controlled-error boolean algebra + optional exact backend via `to_exact()` |
| `NativeExactSphericalRegion` | `torchfits.sphere.geom` | extended | dependency-free exact-backend scaffold with controlled-error area/boolean semantics (`to_exact_region(..., backend='native')`) |
| `map2alm` | `torchfits.sphere.compat` | parity | scalar + polarized (`pol=True`) transforms; supports `iter`/`use_pixel_weights`; optional healpy backend |
| `map2alm_lsq` | `torchfits.sphere.compat` | parity | iterative least-squares analysis; returns `(alm, rel_res, n_iter)`; torch-native + optional healpy backend |
| `alm2map` | `torchfits.sphere.compat` | parity | scalar + polarized (`pol=True`) synthesis; supports `pixwin`; optional healpy backend |
| `almxfl` | `torchfits.sphere.compat` | parity | harmonic transfer scaling by `ell`; short `fl` tails zero-filled like healpy |
| `alm2cl` | `torchfits.sphere.compat` | parity | auto/cross spectra from alm vectors; healpy-compatible stacked ordering |
| `anafast` | `torchfits.sphere.compat` | parity | scalar + polarized spectra, `nspec`, and `alm=True` return paths; optional healpy backend |
| `gaussian_beam` | `torchfits.sphere.compat` | parity | scalar and polarized (`pol=True`) beam transfer outputs |
| `gauss_beam` | `torchfits.sphere.compat` | parity | healpy naming alias for `gaussian_beam` |
| `pixwin` | `torchfits.sphere.compat` | parity | compatibility function is available with packaged pixel-window FITS tables (no optional healpy runtime dependency) |
| `bl2beam` | `torchfits.sphere.compat` | parity | circular beam profile synthesis from `b_l` |
| `beam2bl` | `torchfits.sphere.compat` | parity | circular beam profile analysis to `b_l` |
| `synalm` | `torchfits.sphere.compat` | parity | random alm synthesis from `C_l` for scalar/multi-field spectra (`new` ordering supported) |
| `synfast` | `torchfits.sphere.compat` | parity | map synthesis from `C_l` with scalar/polarized paths and optional `alm` return |
| `smoothalm` | `torchfits.sphere.compat` | parity | scalar and polarized alm smoothing (`beam`/`fwhm`) |
| `smoothmap` | `torchfits.sphere.compat` | parity | scalar and polarized map smoothing (`fwhm`/`sigma`/`beam_window`), optional healpy backend |
| `smoothing` | `torchfits.sphere.compat` | parity | healpy naming alias for `smoothmap`; scalar maps accepted when `pol=True` |
| `map2alm_spin` | `torchfits.sphere.compat` | parity | integer `spin>=0`, CPU-first torch backend + healpy backend parity checks on odd/even spins |
| `alm2map_spin` | `torchfits.sphere.compat` | parity | integer `spin>=0`, CPU-first torch backend + healpy backend parity checks on odd/even spins |
| `fit_monopole` | `torchfits.sphere.compat` | parity | healpy-compatible `bad`/`gal_cut` masking semantics |
| `remove_monopole` | `torchfits.sphere.compat` | parity | healpy-compatible masked subtraction (`fitval`/`copy`) |
| `fit_dipole` | `torchfits.sphere.compat` | parity | healpy-compatible `bad`/`gal_cut` masking semantics |
| `remove_dipole` | `torchfits.sphere.compat` | parity | healpy-compatible masked subtraction (`fitval`/`copy`) |
| `SparseHealpixMap` | `torchfits.sphere.sparse` | parity | coverage-aware sparse map container with dense-equivalent `ud_grade` semantics (including bad-value/non-finite handling), interpolation, and fast NEST hierarchy paths |

## Planned Compatibility Targets

| Function | Module | Status | Notes |
|---|---|---|---|
| higher-`nside` pixel windows | `torchfits.sphere.spectral` | planned | extend packaged table range beyond current HEALPix set if needed |
