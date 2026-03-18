# Sphere & HEALPix

`torchfits.sphere` is a PyTorch-native library for [HEALPix](https://healpix.jpl.nasa.gov/) pixelization, spherical geometry, and spherical harmonic transforms. It runs on CPU, CUDA, and MPS.

`torchfits.sphere.compat` provides a [healpy](https://healpy.readthedocs.io/)-compatible API surface for code migration. See [Compatibility](#compatibility) for details.

---

## Pixel & Coordinate Utilities

### `ang2pix(nside, theta, phi, nest=False, lonlat=False)`

Convert angles to HEALPix pixel indices.

- `lonlat=False` (default): `theta` = co-latitude (rad), `phi` = longitude (rad).
- `lonlat=True`: `theta` = longitude (deg), `phi` = latitude (deg).

```python
import torchfits.sphere as sphere

pix = sphere.ang2pix(128, theta_rad, phi_rad)
pix = sphere.ang2pix(128, lon_deg, lat_deg, lonlat=True, nest=True)
```

### `pix2ang(nside, ipix, nest=False, lonlat=False)`

Inverse of `ang2pix`.

### `vec2pix(nside, x, y, z, nest=False)` / `pix2vec(nside, ipix, nest=False)`

Convert between Cartesian unit vectors and pixel indices.

### `nside2npix(nside)` / `npix2nside(npix)`

Convert between NSIDE and total pixel count.

### `query_disc(nside, vec, radius, *, inclusive=False, nest=False)`

Pixels within a spherical disc. `vec` is a 3-element unit vector, `radius` in radians.

---

## Coordinate Conversions

| Function | Description |
|---|---|
| `lonlat_to_unit_xyz(lon_deg, lat_deg)` | (lon, lat) degrees to Cartesian `[..., 3]` |
| `unit_xyz_to_lonlat(vectors)` | Cartesian to (lon, lat) degrees |
| `wrap_longitude(lon_deg, *, center_deg=180)` | Wrap to `[center-180, center+180)` |
| `great_circle_distance(lon1, lat1, lon2, lat2)` | Angular distance (degrees) |
| `pairwise_angular_distance(lon, lat)` | N x N distance matrix |
| `slerp_lonlat(lon1, lat1, lon2, lat2, t)` | Spherical linear interpolation |

---

## Geometric Queries

### `query_polygon_general(nside, lon_deg, lat_deg, *, nest=False)`

Pixels inside a simple (possibly non-convex) spherical polygon.

### `query_ellipse(nside, lon, lat, semi_major, semi_minor, pa_deg, nest=False)`

Pixels within a spherical ellipse. Angles in degrees.

### `convex_polygon_contains(lon_deg, lat_deg, poly_lon_deg, poly_lat_deg)`

Fast half-space point-in-convex-polygon test.

### `spherical_polygon_contains(lon_deg, lat_deg, poly_lon_deg, poly_lat_deg)`

Point-in-polygon for simple (possibly non-convex) spherical polygons.

---

## Spherical Geometry Classes

### `SphericalPolygon`

Simple spherical polygon with containment, area, boolean operations, and pixel queries.

```python
from torchfits.sphere import SphericalPolygon

poly = SphericalPolygon(lon_deg=torch.tensor([10., 20., 20., 10.]),
                        lat_deg=torch.tensor([0., 0., 10., 10.]))
poly.area()                          # steradians
poly.contains(lon_deg=15., lat_deg=5.)
poly.query_pixels(nside=64)
poly.intersects(other_poly)
poly.union(other_poly)
```

Methods: `area`, `signed_area`, `contains`, `query_pixels`, `pixelize`, `area_estimate`, `intersects`, `union`, `intersection`, `difference`.

Validated against [spherical-geometry](https://github.com/spacetelescope/spherical_geometry) for non-convex containment and area.

### `SphericalCap`

Spherical disc region with the same interface as `SphericalPolygon`.

### `SphericalMultiPolygon`

Union-of-polygons collection.

### `PixelizedRegion`

HEALPix pixel set with boolean operations (`union`, `intersection`, `difference`).

---

## Sampling

### `sample_healpix_map(values, nside, lon_deg, lat_deg, *, interpolation='bilinear')`

Sample a HEALPix map at arbitrary sky coordinates. `interpolation`: `'nearest'` or `'bilinear'`.

### `sample_multiband_healpix(cube, nside, lon_deg, lat_deg, ...)`

Sample a multi-band HEALPix cube `[n_band, npix]`.

### `sample_multiwavelength_healpix(cube, nside, lon_deg, lat_deg, *, source_wavelength, target_wavelength, ...)`

Sample and optionally resample along wavelength axis.

### `fit_monopole_dipole` / `remove_monopole_dipole`

Fit or subtract monopole + dipole from a HEALPix map.

---

## Sparse Maps

### `SparseHealpixMap`

Sparse HEALPix map storing values only for covered pixels. Compatible with [healsparse](https://healsparse.readthedocs.io/) semantics.

```python
from torchfits.sphere import SparseHealpixMap

sparse = SparseHealpixMap(nside=64, pixels=pix_tensor, values=val_tensor)
sparse = SparseHealpixMap.from_dense(dense_map, nside=64)

sparse.coverage_fraction
sparse.to_dense()
sparse.interpolate(lon_deg=lon, lat_deg=lat, method='bilinear')
sparse.ud_grade(nside_out=32)
```

### `HealSparseMap` / `SparseMap` / `WideBitMask` / `SkyMaskPipe`

Additional sparse map types for interop with healsparse workflows.

---

## Harmonic Space (SHT)

### `map2alm(map_values, lmax=None, *, nside=None, iter=0, pol=False, use_pixel_weights=False)`

Analyze a HEALPix map into spherical harmonic coefficients.

- Scalar: `(npix,)` in, `(nalm,)` out.
- Polarized (`pol=True`): `(3, npix)` in, `(3, nalm)` out for (T, E, B).

### `alm2map(alm_values, nside, *, lmax=None, pol=False, pixwin=False)`

Synthesize map(s) from alm coefficients.

### `map2alm_lsq(map_values, lmax, *, tol=1e-10, maxiter=20)`

Iterative least-squares analysis. Returns `(alm, rel_res, n_iter)`.

### `map2alm_spin(maps, spin, *, lmax=None)` / `alm2map_spin(alm_plus, alm_minus, nside, spin, *, lmax=None)`

Spin-weighted transforms. Input maps shape: `(2, npix)`.

### Coefficient utilities

| Function | Description |
|---|---|
| `alm_size(lmax, mmax=None)` | Number of alm coefficients |
| `alm_index(ell, m, lmax)` | Flat index for mode (ell, m) |
| `almxfl(alm, fl)` | Multiply by ell-dependent filter |

---

## Power Spectra

### `anafast(map1, map2=None, *, lmax=None, pol=True, iter=0)`

Auto/cross angular power spectrum C_ell.

### `alm2cl(alms1, alms2=None, *, lmax=None)`

Power spectrum from alm coefficients.

### `synalm(cls, lmax=None)` / `synfast(cls, nside, lmax=None, *, pol=True)`

Generate random alm or maps from input C_ell.

### `smoothmap(map_values, *, fwhm_rad=None, beam_window=None, pol=False)` / `smoothalm(alm_values, *, fwhm_rad=None)`

Gaussian beam smoothing in harmonic space.

---

## Beams & Windows

| Function | Description |
|---|---|
| `gaussian_beam(fwhm_rad, lmax, *, pol=False)` | Gaussian beam transfer function B_ell |
| `pixwin(nside, *, pol=False, lmax=None)` | HEALPix pixel window function |
| `bl2beam(bl, theta)` / `beam2bl(beam, theta, lmax)` | Beam profile to/from harmonic window |

Pixel window data is packaged as FITS tables — no healpy runtime dependency.

---

## Performance

The SHT implementation uses:

- Ring FFT + Legendre recurrence blocking for register-efficient throughput.
- OpenMP parallelism over rings.
- PyTorch `torch.fft` for azimuthal transforms.
- C++ backend (default on CPU) with compiled recurrence kernels. Falls back to pure PyTorch if the extension is unavailable.

Environment knobs:

| Variable | Default | Description |
|---|---|---|
| `TORCHFITS_SCALAR_MAX_CACHE_BYTES` | 512 MiB | Basis-matrix cache budget |
| `TORCHFITS_SCALAR_RING_AUTO_MIN_BYTES` | 64 MiB | Min map size to use ring path |
| `TORCHFITS_SPIN_RING_AUTO_MIN_BYTES` | 32 MiB | Min map size for spin ring path |

---

## Compatibility

`torchfits.sphere.compat` mirrors the [healpy](https://healpy.readthedocs.io/) API. Functions listed as **parity** accept the same arguments and produce equivalent results. Functions listed as **extended** are torchfits additions with no healpy equivalent.

| Function | Status | Notes |
|---|---|---|
| `ang2pix` / `pix2ang` | parity | ring/nest, lonlat/theta-phi |
| `ring2nest` / `nest2ring` | parity | |
| `vec2pix` / `pix2vec` / `ang2vec` / `vec2ang` | parity | |
| `get_all_neighbours` / `get_interp_weights` / `get_interp_val` | parity | |
| `boundaries` | parity | healpy-style boundary sampling |
| `ud_grade` | parity | `power`, `pess`, bad-mask semantics |
| `query_disc` / `query_circle` / `query_circle_vec` | parity | inclusive mode |
| `query_polygon` / `query_polygon_vec` | parity | convex polygon |
| `query_box` / `query_strip` / `query_ellipse` | parity | |
| `isnsideok` / `isnpixok` / `max_pixrad` | parity | |
| `map2alm` / `alm2map` | parity | scalar + polarized; torch + healpy backends |
| `map2alm_lsq` | parity | returns (alm, rel_res, n_iter) |
| `map2alm_spin` / `alm2map_spin` | parity | integer spin >= 0 |
| `almxfl` / `alm2cl` / `anafast` | parity | |
| `synalm` / `synfast` | parity | scalar/multi-field |
| `gaussian_beam` / `pixwin` | parity | packaged FITS tables |
| `bl2beam` / `beam2bl` | parity | |
| `smoothalm` / `smoothmap` / `smoothing` | parity | |
| `fit_monopole` / `remove_monopole` | parity | bad/gal_cut masking |
| `fit_dipole` / `remove_dipole` | parity | bad/gal_cut masking |
| `pixel_ranges_to_pixels` / `upgrade_pixel_ranges` | parity | |
| `pixels_to_pixel_ranges` | extended | torchfits-native range compression |
| `SphericalPolygon` / `SphericalBooleanRegion` | extended | non-convex polygons, boolean algebra |
| `SparseHealpixMap` | parity | healsparse-compatible sparse maps |

Strict mode (`torchfits.sphere.compat.set_strict_mode(True)`) delegates certain operations to [hpgeom](https://github.com/LSSTDESC/hpgeom) for bit-exact results where needed.

---

## Limitations

| Area | Limitation |
|---|---|
| `map2alm` | `use_pixel_weights` and `gal_cut` are not yet supported by the torch backend. Falls back to healpy when requested. |
| `map2alm_spin` / `alm2map_spin` | Functional but slower than healpy's C implementation (~6-8x). Optimization target for 0.4.x. |
| CYP projection | Forward and inverse not yet implemented in `torchfits.wcs`. |
| Pixel windows | Packaged tables cover standard HEALPix NSIDE range. Higher NSIDE may require external data. |
| GPU | HEALPix geometry primitives and SHT run on CUDA and MPS. Polygon operations are CPU-only. |
| Complex columns | Not supported in mmap table reads or in-place updates. |

---

## See Also

- [healpy documentation](https://healpy.readthedocs.io/)
- [HEALPix home page](https://healpix.jpl.nasa.gov/)
- [hpgeom](https://github.com/LSSTDESC/hpgeom)
- [astropy-healpix](https://astropy-healpix.readthedocs.io/)
- [healsparse](https://healsparse.readthedocs.io/)
- [spherical-geometry](https://github.com/spacetelescope/spherical_geometry)
- [API reference](api.md)
- [Benchmarks](benchmarks.md)
