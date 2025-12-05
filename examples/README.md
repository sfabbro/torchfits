# torchfits Examples

Working examples demonstrating torchfits for common astronomical data workflows.

## Running Examples

```bash
cd examples/
python example_basic_reading.py
```

## Example Files

### Getting Started

**`example_basic_reading.py`** - Read FITS images and headers, access header keywords, test different cache settings

**`example_cutouts.py`** - Extract cutouts/subsets from large images without loading full file

**`example_mef.py`** - Work with Multi-Extension FITS (MEF) files, iterate through HDUs, access by name or index

### Data Types

**`example_tables.py`** - Read FITS binary tables as tensor dictionaries, column selection, row ranges

**`example_datacube.py`** - Handle 3D data cubes, extract spatial slices and spectral profiles

**`example_wcs.py`** - World Coordinate System transformations between pixel and sky coordinates

### Machine Learning

**`example_dataset.py`** - PyTorch Dataset integration for FITS data loading

**`example_mnist.py`** - Complete ML pipeline: MNIST digits stored as FITS, CNN training workflow

**`example_sdss_classification.py`** - Astronomical spectral classification (stars/galaxies/quasars) with real SDSS data

## Common Workflows

### Reading FITS Data

```python
import torchfits

# Read full image
data, header = torchfits.read("image.fits")

# Read specific HDU by name
data, header = torchfits.read("mef.fits", hdu="SCI")

# Read to GPU
data, header = torchfits.read("image.fits", device='cuda')

# Extract cutout from large file
cutout = torchfits.read_subset("large.fits", hdu=0,
                               x1=100, y1=100, x2=200, y2=200)

# Read catalog columns
table, header = torchfits.read("catalog.fits", hdu=1,
                               columns=["RA", "DEC", "MAG"])
```

### Multi-Extension FITS

```python
with torchfits.open("mef.fits") as hdul:
    # Iterate through all HDUs
    for i, hdu in enumerate(hdul):
        print(f"HDU {i}: {hdu.header.get('EXTNAME', 'PRIMARY')}")
        
    # Access by name
    sci_data = hdul['SCI'].data
    err_data = hdul['ERR'].data
```

### PyTorch Integration

```python
from torchfits import FITSDataset, create_dataloader

dataset = FITSDataset(file_paths, transform=transform, device='cuda')
dataloader = create_dataloader(dataset, batch_size=32)

for batch in dataloader:
    output = model(batch)
```

## Sample Data

### Provided Files
- `basic_example.fits` - 2D image for basic operations
- `table_example.fits` - Binary table with multiple column types
- `mef_example.fits` - Multi-extension file (empty primary + image + table extensions)
- `data_wcs_examples/` - Files with WCS headers
  - `test_image_2d.fits` - 2D image with RA/Dec WCS
  - `test_spectrum_1d.fits` - 1D spectrum with wavelength WCS
  - `test_cube_3d.fits` - 3D cube with spatial + spectral WCS

### Auto-Downloaded Data
- `example_mnist.py` - Downloads MNIST dataset (~10 MB)
- `example_sdss_classification.py` - Downloads SDSS spectral data (~50 MB)

## Learning Path

1. Start with `example_basic_reading.py` to understand core read/write operations
2. Try `example_cutouts.py` for efficient access to large files
3. Explore `example_mef.py` for multi-extension files (common in modern surveys)
4. Use `example_tables.py` for catalog data manipulation
5. Try `example_wcs.py` for coordinate transformations
6. Advanced: Machine learning examples for training on astronomical data

## Notes

- All examples create test files automatically if they don't exist
- Examples test both CPU and GPU (CUDA/MPS) when available
- Data types are preserved by default for numerical accuracy
- Cache capacity can be adjusted based on available memory

## See Also

- [Main README](../README.md) - Installation and quick start
- [API Reference](../API.md) - Complete API documentation
