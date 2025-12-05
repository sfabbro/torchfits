# Examples

This directory contains comprehensive examples demonstrating the `torchfits` library.

## Quick Start

```bash
cd examples/
python example_basic_reading.py
```

## Examples Overview

### Getting Started
* `example_basic_reading.py` - Basic FITS reading and header access
* `example_cutouts.py` - Reading image cutouts  
* `example_mef.py` - Multi-Extension FITS files

### Data Types
* `example_tables.py` - FITS binary tables
* `example_datacube.py` - 3D data cubes and spectra
* `example_wcs.py` - World Coordinate System transformations

### Machine Learning
* `example_dataset.py` - PyTorch Dataset integration
* `example_mnist.py` - MNIST classification with FITS
* `example_sdss_classification.py` - Astronomical spectral classification

## Key Usage Patterns

### Basic Reading
```python
# Read entire image
data, header = torchfits.read("basic_example.fits")

# Read cutout
cutout = torchfits.read("basic_example.fits", start=[10, 20], shape=[50, 60])

# Read table
table = torchfits.read("table_example.fits", columns=["RA", "DEC"])
```

### Object-Oriented Interface
```python
with torchfits.FITS("file.fits") as f:
    primary = f[0]
    sci_hdu = f["SCI"]
    data = primary.read()
```

### PyTorch Integration
```python
class FITSDataset(Dataset):
    def __getitem__(self, idx):
        return torchfits.read(self.files[idx])
```

## Data Files

* `basic_example.fits` - Simple 2D image
* `table_example.fits` - Binary table 
* `data_wcs_examples/` - WCS sample files

Some examples download external data automatically (MNIST, SDSS).

## Learning Path

1. Start with `example_basic_reading.py` 
2. Try `example_cutouts.py` for efficient access
3. Explore `example_tables.py` for catalog data
4. Advanced: Machine learning examples

Happy coding with torchfits! 🔭

### Prerequisites

Most examples use the provided sample data files. Some advanced examples will automatically download external datasets:

* `example_mnist.py` - Downloads MNIST dataset (~10MB)
* `example_sdss_classification.py` - Downloads SDSS spectral data (~50MB)

## Example Details

### Basic Examples

#### `example_basic_reading.py`
**What it demonstrates:** 
- Reading full FITS images and headers
- Accessing header keywords
- Basic file information

**Key features:**
```python
data, header = torchfits.read("basic_example.fits")
num_hdus = torchfits.get_num_hdus("basic_example.fits")
```

#### `example_cutouts.py`
**What it demonstrates:**
- Reading rectangular cutouts from large images
- CFITSIO-style cutout strings
- Using `start` and `shape` parameters

**Key features:**
```python
# Method 1: Using start/shape
cutout = torchfits.read("basic_example.fits", start=[10, 20], shape=[50, 60])

# Method 2: CFITSIO string syntax
cutout = torchfits.read("basic_example.fits[10:59,20:79]")
```

#### `example_mef.py` 
**What it demonstrates:**
- Multi-Extension FITS files
- Accessing HDUs by number and name
- Iterating through extensions

**Key features:**
```python
with torchfits.FITS("mef_example.fits") as f:
    for i, hdu in enumerate(f):
        print(f"HDU {i}: {hdu.name}, type: {hdu.type}")
```

### Data Type Examples

#### `example_tables.py`
**What it demonstrates:**
- Reading FITS binary tables
- Column selection and data types
- Working with string and numeric columns

**Key features:**
```python
# Read specific columns
data = torchfits.read("table_example.fits", columns=["RA", "DEC", "MAG"])
print(f"RA data type: {data['RA'].dtype}")
```

#### `example_datacube.py`
**What it demonstrates:**
- 3D data cube manipulation
- Extracting 2D slices and 1D spectra
- Working with wavelength axes

**Key features:**
```python
# Extract a spectrum at specific spatial coordinates
spectrum = torchfits.read("test_cube_3d.fits", start=[5, 10, 0], shape=[1, 1, -1])
```

#### `example_wcs.py`
**What it demonstrates:**
- World Coordinate System transformations
- Converting between pixel and world coordinates
- Different WCS types (RA/Dec, wavelength)

**Key features:**
```python
world_coords = torchfits.pixel_to_world([50.0, 50.0], header)
pixel_coords = torchfits.world_to_pixel(world_coords, header)
```

### Machine Learning Examples

#### `example_dataset.py`
**What it demonstrates:**
- Creating PyTorch Dataset classes with torchfits
- Efficient data loading with DataLoader
- Memory management and caching

**Key features:**
```python
class FITSDataset(Dataset):
    def __getitem__(self, idx):
        return torchfits.read(self.files[idx])

dataset = FITSDataset(fits_files)
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

#### `example_mnist.py` 
**What it demonstrates:**
- Complete machine learning pipeline using FITS
- Converting standard datasets to FITS format
- Training a CNN classifier

**What it does:**
1. Downloads MNIST dataset
2. Converts to individual FITS files
3. Creates PyTorch Dataset with torchfits
4. Trains a CNN classifier
5. Evaluates model performance

**Key learning points:**
- FITS as a scientific data format for ML
- Efficient batch loading from FITS files
- Integration with standard PyTorch training loops

#### `example_sdss_classification.py`
**What it demonstrates:**
- Real astronomical data processing
- Spectral classification (stars, galaxies, quasars)
- Working with 1D spectral data

**What it does:**
1. Downloads SDSS DR16 spectral data
2. Preprocesses astronomical spectra
3. Creates balanced dataset of different object types
4. Trains CNN for spectral classification
5. Evaluates on test set

**Key learning points:**
- Processing real astronomical data
- Handling variable-length spectra
- Scientific data preprocessing techniques

## Data Files

The examples use the following data files:

### Provided Sample Data
- `basic_example.fits` - Simple 2D image for basic operations
- `table_example.fits` - Binary table with various column types
- `data_wcs_examples/` - Sample files with WCS information
  - `test_image_2d.fits` - 2D image with RA/Dec WCS
  - `test_spectrum_1d.fits` - 1D spectrum with wavelength WCS  
  - `test_cube_3d.fits` - 3D cube with spatial and spectral WCS

### Downloaded Data (automatic)
- MNIST dataset (example_mnist.py)
- SDSS spectral samples (example_sdss_classification.py)

## Tips for Learning

1. **Start with basics:** Run `example_basic_reading.py` first to understand core concepts

2. **Explore incrementally:** Each example builds on previous concepts

3. **Check the output:** All examples print informative messages about what they're doing

4. **Modify and experiment:** Try changing parameters to see how they affect the results

5. **Read the comments:** Each example has detailed comments explaining the code

6. **Performance notes:** The ML examples show best practices for efficient data loading

## Common Patterns

### Error Handling
```python
try:
    data, header = torchfits.read("file.fits")
except Exception as e:
    print(f"Error reading FITS file: {e}")
```

### Memory Efficiency
```python
# For large files, use cutouts instead of reading entire images
cutout = torchfits.read("large_file.fits", start=[x, y], shape=[100, 100])

# Enable caching for repeated access
data = torchfits.read("file.fits", cache_capacity=512)  # 512 MB cache
```

### GPU Usage
```python
# Read directly to GPU (if CUDA available)
data, header = torchfits.read("file.fits", device="cuda")
```

## Getting Help

- Check the main [README.md](../README.md) for API documentation
- Each example script has detailed comments
- Use `help(torchfits.read)` for function documentation
- Report issues on the GitHub repository

## Contributing Examples

Have an interesting use case? Contributions of new examples are welcome! Please ensure your example:

1. Is self-contained and well-commented
2. Demonstrates a specific feature or use case
3. Includes sample data or downloads it automatically
4. Follows the existing code style

Happy coding with torchfits! 🔭✨
