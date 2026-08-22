# Examples & Tutorial Scripts

A curated catalog of runnable astronomical examples, from survey mosaic processing and IFU datacubes to X-ray event table filtering and neural network training.

All example scripts live in the `examples/` directory of the repository, and are mirrored in [`published-examples/`](published-examples/README.md).

To run the automated validation suite across all example scripts:

```bash
python examples/test_examples.py
```

---

## Sample Data Sources

Most examples run against public astronomical survey and observatory data. Sample data can be downloaded into your local cache using the repository helper scripts:

```bash
# Fetch Astropy tutorial samples (HorseHead, M13, Chandra, SDSS)
bash scripts/fetch_example_samples.sh

# Fetch SDSS MaNGA DR17 LOGCUBE (~200 MB)
bash scripts/fetch_example_samples.sh --with-manga

# Fetch CFHT MegaCam 36-CCD MEF exposures (CADC)
bash scripts/fetch_cfht_megacam_sample.sh

# Fetch CFHT MegaPipe survey mosaic stacks (~5.3 GB)
bash scripts/fetch_cfht_megapipe_sample.sh
```

---

## 1. Survey Mosaics & Multi-Extension FITS (MEF)

Astronomical cameras (such as CFHT MegaCam, Subaru Hyper Suprime-Cam, and DECam) produce Multi-Extension FITS files comprising dozens of individual CCDs.

### Extracting Cutouts from Multi-CCD Exposures

Extract postage stamps across individual CCD extensions (`EXTNAME="CCD01"`, `CCD02`, ...) using high-throughput subset readers:

```python
import torchfits

# Reusable reader handle for fast repeated extractions on a 36-CCD exposure
with torchfits.open_subset_reader("megacam_exposure.fits.fz", hdu="CCD01") as reader:
    stamp1 = reader.read_subset(100, 100, 228, 228)
    stamp2 = reader.read_subset(500, 500, 628, 628)
```

- **Script:** [`example_megacam_mef_cutouts.py`](published-examples/example_megacam_mef_cutouts.py)
- **Gigapixel Collage Demo:** [`example_megapipe_cutout_collage.py`](published-examples/example_megapipe_cutout_collage.py)

### Cutouts with World Coordinate System (WCS) Updates

When extracting a sub-region from an image, the reference pixel coordinates (`CRPIX1`, `CRPIX2`) in the FITS header must be translated to preserve astrometric calibration:

```python
import torchfits

# Extract cutout and compute translated WCS reference pixel coordinates
cutout, header = torchfits.read("spitzer_irac.fits", return_header=True)
header["CRPIX1"] -= 100
header["CRPIX2"] -= 100

torchfits.write(
    "spitzer_cutout.fits", cutout[100:356, 100:356], header=header, overwrite=True
)
```

- **Script:** [`example_cutout_wcs_write.py`](published-examples/example_cutout_wcs_write.py)

---

## 2. Spectroscopy & 3D Datacubes

### SDSS MaNGA IFU Datacubes

SDSS-IV MaNGA (Mapping Nearby Galaxies at APO) packages 3D integral field unit (IFU) spectroscopy into multi-extension FITS datacubes containing flux, inverse variance, mask planes, and wavelength grids:

```python
import torchfits

# Read 3D spectral flux cube [Wavelength, Y, X]
flux_cube = torchfits.read_tensor("manga_logcube.fits", hdu="FLUX")
ivar_cube = torchfits.read_tensor("manga_logcube.fits", hdu="IVAR")
wave_grid = torchfits.read_tensor("manga_logcube.fits", hdu="WAVE")

print(f"Datacube shape: {flux_cube.shape}")  # [4563, 74, 74]
```

- **Script:** [`example_manga_logcube.py`](published-examples/example_manga_logcube.py)
- **Radio Cube Script:** [`example_image_cube.py`](published-examples/example_image_cube.py)

---

## 3. Astronomical Event Tables & Catalogs

### Chandra X-Ray Event Lists

High-energy astrophysics instruments record individual photon arrival events. Query and filter multi-million photon event files using C++ pushdown predicates directly into PyArrow:

```python
import torchfits

# Filter Chandra X-ray events by energy band (e.g. hard X-ray band > 5 keV)
events = torchfits.table.read(
    "chandra_events.fits",
    hdu="EVENTS",
    columns=["time", "x", "y", "energy"],
    where="energy > 5000",
)
print(f"Selected {events.num_rows} hard X-ray photons.")
```

- **Script:** [`example_table.py`](published-examples/example_table.py)
- **Dataframe Interop Script:** [`example_table_interop.py`](published-examples/example_table_interop.py)
- **DuckDB & Polars Analytics:** [`example_table_recipes.py`](published-examples/example_table_recipes.py)

---

## 4. Multi-Exposure Coaddition & Image Stacking

Combine dithered telescope exposures into a high signal-to-noise coadded image tensor:

```python
import glob
import torch
import torchfits

# Stack multiple raw exposures (e.g. Messier 13 globular cluster frames)
frames = [torchfits.read_tensor(f, hdu=0).float() for f in glob.glob("m13_blue_*.fits")]
stacked_image = torch.stack(frames, dim=0).mean(dim=0)

torchfits.write("m13_coadd.fits", stacked_image, overwrite=True)
```

- **Script:** [`example_m13_stack.py`](published-examples/example_m13_stack.py)

---

## 5. Storage Optimization: Robust 16-Bit Quantization

When storage budgets or bandwidth require compressing 32-bit floating-point astronomical images and table columns into 16-bit integers (`BITPIX=16`, `TFORM=I`), standard global min-max scaling loses dynamic range to cosmic rays and noise spikes. The `quantize="robust"` parameter automatically uses percentile-based bulk packing:

```python
import torchfits

image_float = torchfits.read_tensor("raw_float.fits")

# Pack into 16-bit integer FITS with automatically calibrated BSCALE and BZERO
torchfits.write("packed_int16.fits", image_float, quantize="robust", overwrite=True)
```

- **Script:** [`example_quantize_int16.py`](published-examples/example_quantize_int16.py)

---

## 6. Complete Example Scripts Directory

### Image & Datacube Processing

| Script | Purpose & Key APIs |
|---|---|
| [`example_image.py`](published-examples/example_image.py) | Basic read and write round-trip for 2D images |
| [`example_image_cutouts.py`](published-examples/example_image_cutouts.py) | Bounding box cutouts with `read_subset` and `open_subset_reader` |
| [`example_image_cube.py`](published-examples/example_image_cube.py) | 3D radio and spectral data cube slicing |
| [`example_image_mef.py`](published-examples/example_image_mef.py) | Multi-extension inspection and named HDU extraction |
| [`example_m13_stack.py`](published-examples/example_m13_stack.py) | Multi-frame exposure stacking and median coaddition |
| [`example_mef_header.py`](published-examples/example_mef_header.py) | Navigating complex headers across multi-CCD files |
| [`example_cutout_wcs_write.py`](published-examples/example_cutout_wcs_write.py) | Extracting sub-regions and updating `CRPIX` astrometry keywords |
| [`example_manga_logcube.py`](published-examples/example_manga_logcube.py) | SDSS MaNGA IFU datacubes (`FLUX`, `IVAR`, `MASK`, `WAVE`) |
| [`example_megacam_mef_cutouts.py`](published-examples/example_megacam_mef_cutouts.py) | Parallel cutouts from CFHT MegaCam 36-CCD mosaics |
| [`example_quantize_int16.py`](published-examples/example_quantize_int16.py) | Robust 16-bit quantization for images and table columns |

### Tables & Catalogs

| Script | Purpose & Key APIs |
|---|---|
| [`example_table.py`](published-examples/example_table.py) | Arrow tables, tensor dictionary loading, and Chandra event filtering |
| [`example_table_interop.py`](published-examples/example_table_interop.py) | Zero-copy conversion between PyArrow, Pandas, and Polars |
| [`example_polars.py`](published-examples/example_polars.py) | `read_polars` and out-of-core `scan_polars` streaming |
| [`example_table_recipes.py`](published-examples/example_table_recipes.py) | SQL pushdown, DuckDB querying, and Arrow record batch iterators |

### Machine Learning & Preprocessing

| Script | Purpose & Key APIs |
|---|---|
| [`example_ml_galaxyzoo_legacy.py`](published-examples/example_ml_galaxyzoo_legacy.py) | Galaxy Zoo 1 morphology classification on Legacy Survey cutouts |
| [`example_megacam_cr_denoise.py`](published-examples/example_megacam_cr_denoise.py) | FITS-native Noise2Noise calibration, held-out dark test, and conservative science CR repair |
| [`example_megapipe_cutout_collage.py`](published-examples/example_megapipe_cutout_collage.py) | Gigapixel mosaic cutout extraction and Lupton RGB collage |
| [`example_lupton_rgb_sdss.py`](published-examples/example_lupton_rgb_sdss.py) | Lupton asinh RGB synthesis from SDSS $g, r, i$ filter frames |
| [`example_rgb_sky.py`](published-examples/example_rgb_sky.py) | Auto RGB collage: Legacy Survey Virgo, JWST Stephan's Quintet, HST OPAL Jupiter |
| [`example_transforms.py`](published-examples/example_transforms.py) | Composite stretch and normalization pipelines (`Compose`) |
| [`example_time_series.py`](published-examples/example_time_series.py) | Light curve filtering with symmetric and asymmetric sigma clipping |
| [`example_custom_transform.py`](published-examples/example_custom_transform.py) | Implementing custom `FITSTransform` subclasses |
| [`example_make_loader_vs_dataloader.py`](published-examples/example_make_loader_vs_dataloader.py) | Benchmark comparing `make_loader` cache warmup vs `DataLoader` |
| [`example_image_dataset.py`](published-examples/example_image_dataset.py) | Minimal `FitsImageDataset` + `make_loader` ([ML guide](examples-ml.md)) |
| [`example_data_catalogs.py`](published-examples/example_data_catalogs.py) | Table + cutout datasets ([ML guide](examples-ml.md)) |

### Figure generators

| Script | Output |
|---|---|
| [`gallery_images.py`](published-examples/gallery_images.py) | image before/after PNGs |
| [`gallery_tables_lc.py`](published-examples/gallery_tables_lc.py) | light-curve / table plots |

Samples use `TORCHFITS_SAMPLE_CACHE` when set, otherwise torchfits' normal
cache precedence. CI sets `TORCHFITS_EXAMPLE_FAST=1` to skip downloads.

### Out of gallery

These live under `examples/` but are not part of the published gallery /
`docs-contract` verification suite (CLI demos or specialized shapes):

- `examples/desi_shaped_spectrum.py` — DESI-shaped spectrum demo ([API data](api-data.md))
- `examples/cli/make_rgb_demo.py` — RGB collage generator for CLI recipes ([CLI recipes](cli-recipes.md))
