# Spherical Geometry & HEALPix

`torchfits.sphere` is a PyTorch-native library for spherical geometry and HEALPix maps, with high-performance C++ kernels for Spherical Harmonic Transforms (SHT).

## Table of Contents

- [Pixel & Coordinate Utilities](#pixel--coordinate-utilities)
- [Coordinate Conversions](#coordinate-conversions)
- [Geometric Queries](#geometric-queries)
- [Spherical Geometry](#spherical-geometry)
- [Sampling](#sampling)
- [Sparse Maps](#sparse-maps)
- [Harmonic Space (SHT)](#harmonic-space-sht)
- [Power Spectra](#power-spectra)
- [Beams & Windows](#beams--windows)
- [Performance Notes](#performance-notes)
- [Compatibility Map](#compatibility-map)

---

## Pixel & Coordinate Utilities

### `ang2pix(nside, theta, phi, nest=False, lonlat=False)`

Convert angles to HEALPix pixel indices.

- If `lonlat=False` (default): `theta` is co-latitude in radians, `phi` is longitude in radians.
- If `lonlat=True`: `theta` is longitude in degrees, `phi` is latitude in degrees.

```python
import torch
import torchfits.sphere as sphere

nside = 128
# (theta, phi) in radians
pix = sphere.ang2pix(nside, theta_rad, phi_rad)
# (lon, lat) in degrees
pix = sphere.ang2pix(nside, lon_deg, lat_deg, lonlat=True)
# NESTED ordering
pix = sphere.ang2pix(nside, lon_deg, lat_deg, nest=True, lonlat=True)
```

### `pix2ang(nside, ipix, nest=False, lonlat=False)`

Convert pixel indices to angles. Inverse of `ang2pix`.

```python
lon, lat = sphere.pix2ang(nside, pix, lonlat=True)
theta, phi = sphere.pix2ang(nside, pix)  # radians
```

### `vec2pix(nside, x, y, z, nest=False)`

Convert Cartesian unit vectors to pixel indices.

### `pix2vec(nside, ipix, nest=False)`

Convert pixel indices to Cartesian unit vectors. Returns `Tensor` of shape `(npix, 3)`.

### `nside2npix(nside)` / `npix2nside(npix)`

Convert between `nside` and total pixel count.

```python
npix = sphere.nside2npix(64)   # 49152
nside = sphere.npix2nside(npix)
```

### `query_disc(nside, vec, radius, *, inclusive=False, nest=False)`

Find pixel indices whose centers fall inside a spherical disc.

- `vec`: 3-element Cartesian unit vector `[x, y, z]`.
- `radius`: angular radius in **radians**.
- `inclusive=True`: conservative expansion by max pixel radius.

```python
import torch

center = torch.tensor([0.0, 0.0, 1.0])  # north pole
pixels = sphere.query_disc(nside=64, vec=center, radius=0.1)
```

---

## Coordinate Conversions

### `lonlat_to_unit_xyz(lon_deg, lat_deg)`

Convert longitude / latitude in degrees to Cartesian unit vectors. Output shape is `[..., 3]`.

### `unit_xyz_to_lonlat(vectors, *, wrap_center_deg=180.0)`

Convert Cartesian unit vectors of shape `[..., 3]` to `(lon_deg, lat_deg)`.

### `wrap_longitude(lon_deg, *, center_deg=180.0)`

Wrap longitudes to the range `[center_deg - 180, center_deg + 180)`.

- `center_deg=180` → `[0, 360)` (default)
- `center_deg=0`   → `[-180, 180)`

### `great_circle_distance(lon1_deg, lat1_deg, lon2_deg, lat2_deg, *, degrees=True)`

Numerically stable great-circle angular distance. Returns radians when `degrees=False`.

### `pairwise_angular_distance(lon_deg, lat_deg, *, degrees=True)`

Pairwise angular distance matrix for N positions. Inputs are 1-D tensors `[N]`; output is `[N, N]`.

### `slerp_lonlat(lon1_deg, lat1_deg, lon2_deg, lat2_deg, t)`

Spherical linear interpolation between two sky positions. `t=0` → start, `t=1` → end.

---

## Geometric Queries

### `query_polygon_general(nside, lon_deg, lat_deg, *, nest=False)`

Find pixels inside a **simple** (possibly non-convex) spherical polygon.

- Vertices given as lon/lat degree sequences.
- Non-convex polygons are triangulated via gnomonic-plane ear-clipping.

```python
lon = [10.0, 20.0, 20.0, 10.0]
lat = [0.0,  0.0,  10.0, 10.0]
pixels = sphere.query_polygon_general(nside=64, lon_deg=lon, lat_deg=lat)
```

### `query_ellipse(nside, lon, lat, semi_major, semi_minor, pa_deg, nest=False)`

Find pixels within a spherical ellipse.

- `lon`, `lat`: centre in degrees.
- `semi_major`, `semi_minor`: angular semi-axes in degrees.
- `pa_deg`: position angle (East of North), degrees.

### `convex_polygon_contains(lon_deg, lat_deg, poly_lon_deg, poly_lat_deg, *, atol=1e-12)`

Fast half-space point-in-convex-polygon predicate.

### `spherical_polygon_contains(lon_deg, lat_deg, poly_lon_deg, poly_lat_deg, *, inclusive=True, atol_deg=1e-10, inside_lon_deg=None, inside_lat_deg=None)`

Point-in-polygon check for simple (possibly non-convex) spherical polygons.

---

## Spherical Geometry

### `spherical_triangle_area(lon1, lat1, lon2, lat2, lon3, lat3, *, degrees=False)`

Spherical triangle area on the unit sphere (steradians by default; square degrees when `degrees=True`).

### `spherical_polygon_area(lon_deg, lat_deg, *, degrees=False, oriented=False)`

Area of a simple spherical polygon. Returns non-oriented enclosed area by default.

### `spherical_polygon_signed_area(lon_deg, lat_deg, *, degrees=False)`

Signed area following vertex winding orientation.

### `spherical_polygons_intersect(lon1_deg, lat1_deg, lon2_deg, lat2_deg, *, atol_deg=1e-8)`

Polygon intersection predicate – returns `True` if interiors intersect or boundaries cross.

### Class `SphericalPolygon`

Simple spherical polygon with containment and boolean operations.

```python
from torchfits.sphere import SphericalPolygon

poly = SphericalPolygon(lon_deg=torch.tensor([10., 20., 20., 10.]),
                        lat_deg=torch.tensor([0., 0., 10., 10.]))
area_sr   = poly.area()
area_deg2 = poly.area(degrees=True)
inside    = poly.contains(lon_deg=15.0, lat_deg=5.0)
pixels    = poly.query_pixels(nside=64)
pr        = poly.pixelize(nside=64)        # → PixelizedRegion
intersect = poly.intersects(other_poly)
union     = poly.union(other_poly)
diff      = poly.difference(other_poly)
```

**Methods:**

| Method | Description |
|---|---|
| `area(*, degrees=False, oriented=False)` | Polygon area |
| `signed_area(*, degrees=False)` | Signed winding area |
| `contains(lon_deg, lat_deg, *, inclusive=True)` | Point-in-polygon |
| `query_pixels(nside, *, nest=False)` | Pixel indices inside polygon |
| `pixelize(nside, *, nest=False)` | Returns `PixelizedRegion` |
| `area_estimate(*, nsides=(128,256,512,1024))` | Multi-resolution area estimate |
| `intersects(other)` | Intersection predicate |
| `union(other)` / `intersection(other)` / `difference(other)` | Boolean operations |

### Class `SphericalCap`

Spherical cap (disc) region with the same polygon-like interface.

### Class `SphericalMultiPolygon`

Collection of `SphericalPolygon` instances with union-of-polygons semantics.

```python
from torchfits.sphere import SphericalMultiPolygon, SphericalPolygon

multi = SphericalMultiPolygon(polygons=(poly1, poly2))
area = multi.area()
inside = multi.contains(lon_deg, lat_deg)
pixels = multi.query_pixels(nside=64)
```

### Class `PixelizedRegion`

Set of HEALPix pixels supporting boolean operations.

```python
from torchfits.sphere import PixelizedRegion

r1 = poly1.pixelize(nside=64)
r2 = poly2.pixelize(nside=64)
union_r = r1.union(r2)
inter_r = r1.intersection(r2)
diff_r  = r1.difference(r2)
area_sr = r1.area()
```

---

## Sampling

### `sample_healpix_map(values, nside, lon_deg, lat_deg, *, nest=False, interpolation='bilinear')`

Sample a HEALPix map `[..., npix]` at arbitrary sky coordinates; returns `[..., *coord_shape]`.

- `interpolation`: `'nearest'` or `'bilinear'`.

```python
m = torch.randn(sphere.nside2npix(64))
vals = sphere.sample_healpix_map(m, nside=64, lon_deg=lon, lat_deg=lat)
```

### `sample_multiband_healpix(cube, nside, lon_deg, lat_deg, *, nest=False, interpolation='bilinear')`

Sample a multi-band HEALPix cube `[..., n_band, npix]`; returns `[..., n_band, *coord_shape]`.

### `sample_multiwavelength_healpix(cube, nside, lon_deg, lat_deg, *, source_wavelength=None, target_wavelength=None, nest=False, interpolation='bilinear')`

Sample a multiwavelength cube `[..., n_wave, npix]` and optionally resample wavelengths.

### `interpolate_wavelength_axis(values, source_wavelength, target_wavelength, *, axis=-1)`

Linear interpolation along a wavelength axis.

### `fit_monopole_dipole(map_values, nside, *, nest=False, valid_mask=None)`

Fit monopole + dipole model to a HEALPix map. Returns `(monopole, dipole_xyz)`.

### `remove_monopole_dipole(map_values, nside, *, nest=False, valid_mask=None)`

Subtract fitted monopole + dipole from a HEALPix map.

---

## Sparse Maps

### Class `SparseHealpixMap`

Sparse HEALPix map that stores values only for covered pixels.

```python
from torchfits.sphere import SparseHealpixMap

# Create from a footprint (list of pixel indices)
sparse = SparseHealpixMap(nside=64, nest=False,
                          pixels=pixels_tensor, values=values_tensor)

# Create from a dense map (masking UNSEEN values)
sparse = SparseHealpixMap.from_dense(dense_map, nside=64)

# Properties
print(sparse.coverage_fraction)   # fraction of sky covered
mask = sparse.coverage_mask        # dense boolean mask
dense = sparse.to_dense()          # expand back to npix-length tensor

# Interpolate at sky coordinates
vals = sparse.interpolate(lon_deg=lon, lat_deg=lat, method='bilinear')

# Change resolution
coarser = sparse.ud_grade(nside_out=32)
finer   = sparse.ud_grade(nside_out=128)
```

**Key fields:** `nside`, `nest`, `pixels`, `values`, `fill_value` (default: `UNSEEN = -1.6375e30`).

**Key methods:**

| Method | Description |
|---|---|
| `coverage_fraction` | Fraction of sky covered |
| `coverage_mask` | Dense boolean mask tensor |
| `covered` | Pixel indices with valid values |
| `to_dense()` | Expand to `[npix]` tensor |
| `from_dense(map, *, nside, ...)` | Classmethod: build from dense map |
| `interpolate(lon, lat, *, method)` | Bilinear/nearest interpolation |
| `ud_grade(nside_out, *, pess, power)` | Upgrade / downgrade resolution |

---

## Harmonic Space (SHT)

### `map2alm(map_values, lmax=None, mmax=None, *, nside=None, nest=False, iter=0, pol=False, use_pixel_weights=False, backend='torch')`

Analyze a HEALPix map into spherical harmonic coefficients (alms).

- `map_values`: shape `(npix,)` for scalar or `(3, npix)` for polarized (`pol=True`).
- Returns `(nalm,)` for scalar or `(3, nalm)` for `(T, E, B)` polarization.
- `iter`: number of Jacobi iteration rounds (0 = single pass).

```python
lmax = 3 * nside - 1
alm = sphere.map2alm(m, lmax=lmax)
# Polarized
alm_TEB = sphere.map2alm(m_TQU, lmax=lmax, pol=True)
```

### `alm2map(alm_values, nside, *, lmax=None, mmax=None, nest=False, pol=False, pixwin=False, backend='torch')`

Synthesize HEALPix map(s) from alm coefficients.

- `alm_values`: shape `(nalm,)` for scalar or `(3, nalm)` in `(T, E, B)` order.

```python
m_reconstructed = sphere.alm2map(alm, nside=nside)
# Polarized
m_TQU = sphere.alm2map(alm_TEB, nside=nside, pol=True)
```

### `map2alm_lsq(map_values, lmax, mmax=None, *, nside=None, pol=True, tol=1e-10, maxiter=20, nest=False, backend='torch')`

Iterative least-squares map analysis. Returns `(alm, rel_res, n_iter)`.

### `map2alm_spin(maps, spin, *, nside=None, lmax=None, mmax=None, nest=False, backend='torch')`

Spin-weighted map → alm transform. Input `maps` shape must be `(2, npix)`.

### `alm2map_spin(alm_plus, alm_minus, nside, spin, *, lmax=None, mmax=None, nest=False, backend='torch')`

Synthesize two spin-component maps from `(alm_plus, alm_minus)` coefficient pairs.

### Coefficient Utilities

| Function | Description |
|---|---|
| `alm_size(lmax, mmax=None)` | Number of alm coefficients |
| `alm_index(ell, m, lmax, mmax=None)` | Flat index for mode `(ell, m)` |
| `almxfl(alm, fl, *, mmax=None, inplace=False)` | Multiply by ell-dependent filter |

```python
nalm = sphere.alm_size(lmax=383)
idx  = sphere.alm_index(ell=10, m=3, lmax=383)
alm_filtered = sphere.almxfl(alm, beam_bl)
```

---

## Power Spectra

### `anafast(map1, map2=None, *, nside=None, lmax=None, mmax=None, pol=True, iter=0, alm=False, use_pixel_weights=False, gal_cut=0.0, backend='torch')`

Compute auto/cross angular power spectrum $C_\ell$ from scalar or polarized maps.

```python
cl = sphere.anafast(m)          # auto-spectrum
cl_cross = sphere.anafast(m1, m2)  # cross-spectrum
```

### `alm2cl(alms1, alms2=None, *, lmax=None, mmax=None, lmax_out=None, nspec=None)`

Compute auto/cross power spectra directly from alm coefficients.

### `synalm(cls, lmax=None, mmax=None, *, new=False)`

Generate alm coefficient vectors from input power spectra `cls`.

### `synfast(cls, nside, lmax=None, mmax=None, *, alm=False, pol=True, pixwin=False, fwhm=0.0, sigma=None, new=False)`

Synthesize map(s) from input $C_\ell$ power spectrum/spectra.

```python
cl = torch.ones(3 * nside)
m = sphere.synfast(cl, nside=nside)
```

### `smoothmap(map_values, *, nside=None, lmax=None, fwhm_rad=None, sigma=None, beam_window=None, pol=False, iter=0, backend='torch')`

Smooth a HEALPix map in harmonic space with a Gaussian beam.

### `smoothalm(alm_values, *, lmax=None, mmax=None, fwhm_rad=None, beam=None, pol=False)`

Apply beam smoothing directly to alm coefficients.

---

## Beams & Windows

### `gaussian_beam(fwhm_rad, lmax, *, pol=False)`

Return Gaussian beam transfer function(s) $B_\ell$. With `pol=True`, returns `(T, E)` pair.

### `pixwin(nside, *, pol=False, lmax=None)`

Return HEALPix pixel window function(s). Data loaded from packaged FITS tables.

### `bl2beam(bl, theta)` / `beam2bl(beam, theta, lmax)`

Convert between harmonic window $b_\ell$ and circular beam profile $b(\theta)$.

---

## Performance Notes

The SHT kernels use:

- **Ring FFT + recurrence blocking**: separates azimuthal IFFT from Legendre recurrences, enabling register-resident L-blocking for maximum throughput.
- **OpenMP parallelism**: multi-threaded over rings.
- **PyTorch `torch.fft`**: azimuthal transforms leverage PyTorch's internal `pocketfft` with automatic workspace caching.
- **C++ backend** (default for CPU): high-throughput recurrence kernels compiled at install time. Falls back to pure-Torch when C++ extension is unavailable.

For large $N_\text{side}$ or batch transforms, `torchfits.sphere` substantially outperforms pure-Python HEALPix implementations.

### Environment Knobs

| Variable | Default | Description |
|---|---|---|
| `TORCHFITS_SCALAR_MAX_CACHE_BYTES` | `512 MiB` | Basis-matrix cache budget |
| `TORCHFITS_SCALAR_RING_AUTO_MIN_BYTES` | `64 MiB` | Min map size to use ring path |
| `TORCHFITS_SPIN_RING_AUTO_MIN_BYTES` | `32 MiB` | Min map size for spin ring path |

---

## Compatibility Map

Status: `parity` = upstream-compatible semantics · `extended` = torchfits-native additions · `partial` = known gaps · `planned` = not yet implemented.

Functions in `torchfits.sphere.compat` mirror the [healpy](https://healpy.readthedocs.io/) and [hpgeom](https://github.com/LSSTDESC/hpgeom) APIs and can be used as drop-in replacements.

| Function | Status | Notes |
|---|---|---|
| `ang2pix` | parity | ring/nest, lonlat/theta-phi |
| `pix2ang` | parity | ring/nest, lonlat/theta-phi |
| `ring2nest` / `nest2ring` | parity | CPU parity-tested |
| `vec2pix` / `pix2vec` | parity | xyz ↔ pixel |
| `ang2vec` / `vec2ang` | parity | angle ↔ xyz |
| `get_all_neighbours` / `get_all_neighbors` | parity | pixel and angular forms |
| `get_interp_weights` | parity | 4-neighbor interpolation weights |
| `get_interp_val` | parity | map interpolation |
| `boundaries` | parity | healpy-style boundary sampling |
| `ud_grade` | parity | `power`, `pess`, bad-mask semantics |
| `isnsideok` / `isnpixok` | parity | scalar + tensor forms |
| `max_pixrad` | parity | conservative radius |
| `query_disc` | parity | inclusive mode supported |
| `query_circle` | parity | lon/lat wrapper |
| `query_circle_vec` | parity | Cartesian-vector form |
| `query_polygon` | parity | convex polygon |
| `query_polygon_vec` | parity | Cartesian-vector form |
| `query_box` | parity | wrap-aware longitude intervals |
| `query_strip` | parity | colatitude strip |
| `query_ellipse` | parity | hpgeom-style units; dispatches to hpgeom when available |
| `pixel_ranges_to_pixels` | parity | half-open/closed range expansion |
| `upgrade_pixel_ranges` | parity | NEST pixel-range upgrade |
| `pixels_to_pixel_ranges` | extended | torchfits-native range compression |
| `map2alm` | parity | scalar + polarized (`pol=True`); `iter`, `use_pixel_weights` |
| `map2alm_lsq` | parity | returns `(alm, rel_res, n_iter)`; torch + healpy backends |
| `alm2map` | parity | scalar + polarized; `pixwin`; torch + healpy backends |
| `almxfl` | parity | short `fl` tails zero-filled (healpy semantics) |
| `alm2cl` | parity | auto/cross; healpy-compatible stacked ordering |
| `anafast` | parity | scalar + polarized; `nspec`, `alm=True`; torch + healpy backends |
| `synalm` | parity | scalar/multi-field; `new` ordering |
| `synfast` | parity | scalar/polarized; optional `alm` return |
| `gaussian_beam` / `gauss_beam` | parity | scalar + polarized |
| `pixwin` | parity | packaged pixel-window FITS tables (no healpy runtime dep) |
| `bl2beam` / `beam2bl` | parity | beam profile ↔ harmonic window |
| `smoothalm` | parity | scalar + polarized |
| `smoothmap` / `smoothing` | parity | scalar + polarized; torch + healpy backends |
| `map2alm_spin` / `alm2map_spin` | parity | integer `spin≥0`; torch + healpy backends |
| `fit_monopole` / `remove_monopole` | parity | `bad`/`gal_cut` masking semantics |
| `fit_dipole` / `remove_dipole` | parity | `bad`/`gal_cut` masking semantics |
| `spherical_polygon_contains` | extended | inside-point disambiguation |
| `spherical_polygons_intersect` | extended | edge crossing + containment |
| `SphericalBooleanRegion` (union/intersection/difference) | extended | controlled-error boolean algebra; optional exact backend |
| `NativeExactSphericalRegion` | extended | dependency-free exact-backend scaffold |
| `SparseHealpixMap` | parity | coverage-aware sparse map; dense-equivalent `ud_grade` |

### Planned

| Item | Notes |
|---|---|
| Higher-nside pixel windows | Extend packaged table range beyond current HEALPix set |
