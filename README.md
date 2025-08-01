# torchfits

**IN-PROGRESS -STILL EXPERIMENTAL.**

[![PyPI version](https://badge.fury.io/py/torchfits.svg)](https://badge.fury.io/py/torchfits) [![License](https://img.shields.io/badge/License-GPL_2.0-blue.svg)](https://opensource.org/licenses/GPL-2.0)

`torchfits` is a high-performance Python package for reading FITS files directly into PyTorch tensors. It leverages `cfitsio` and `wcslib` for speed and accuracy, making it ideal for astronomy and machine learning applications that require efficient access to large FITS datasets. It is designed to be easy to use, with a simple and consistent API.

## Features

*   **Fast FITS I/O:** Uses a highly optimized C++ backend (powered by `cfitsio`) for rapid data access.
*   **Direct PyTorch Tensors:** Reads FITS data directly into PyTorch tensors, avoiding unnecessary data copies.
*   **Flexible Cutout Reading:**  Supports CFITSIO-style cutout strings (e.g., `'myimage.fits[1][10:20,30:40]'`).
*   **Simplified Cutout Definition:** Provides an easy way to read rectangular regions using `start` and `shape` parameters.
*   **Automatic Data Type Handling:** Automatically determines the correct PyTorch data type.
*   **Image, Cube, and Table Support:** Reads data from image HDUs (1D, 2D, and 3D) and binary/ASCII table HDUs.
*   **WCS Support:** Includes functions for world-to-pixel and pixel-to-world coordinate transformations using `wcslib`. WCS information is automatically updated when reading cutouts. (Note: these functions are internal, WCS use is integrated directly into `read`).
*   **Header Access:** Provides functions to access the full FITS header as a Python dictionary, or to retrieve individual header keyword values.
*   **HDU Information:** Functions to get the number of HDUs, HDU types, and image dimensions.
*   **Designed for PyTorch Datasets:** The API is designed to integrate seamlessly with PyTorch's `Dataset` and `DataLoader` classes, including distributed data loading.
* **HDU selection**: Select HDU either by name or number.

## Installation

### Prerequisites

Before installing `torchfits`, you need to install the required system libraries:

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install libcfitsio-dev libwcs-dev
```

**Linux (Fedora/CentOS/RHEL):**
```bash
sudo yum install cfitsio-devel wcslib-devel
```

**macOS (Homebrew):**
```bash
brew install cfitsio wcslib
```

**macOS (MacPorts):**
```bash
sudo port install cfitsio wcslib
```

### Install torchfits

Once prerequisites are installed:

```bash
pip install torchfits
```

### Development Installation

For development or the latest features:

```bash
git clone https://github.com/sfabbro/torchfits.git
cd torchfits
pip install -e .
```

## Quickstart

Get started with `torchfits` in just a few lines of code! Here are the most common use cases:

### Basic Usage

```python
import torchfits

# Read an entire FITS image
data, header = torchfits.read("my_image.fits")
print(f"Image shape: {data.shape}, Data type: {data.dtype}")
print(f"NAXIS1: {header['NAXIS1']}, NAXIS2: {header['NAXIS2']}")

# Read a specific extension
sci_data, sci_header = torchfits.read("my_image.fits", hdu="SCI")

# Read a cutout (faster for large images)
cutout, cutout_header = torchfits.read("my_image.fits", start=[100, 200], shape=[50, 50])
```

### Working with Tables

```python
# Read all columns from a table
table_data = torchfits.read("catalog.fits", hdu="CATALOG")
print(f"Available columns: {list(table_data.keys())}")

# Read specific columns only
ra_dec = torchfits.read("catalog.fits", hdu="CATALOG", columns=["RA", "DEC"])
print(f"RA shape: {ra_dec['RA'].shape}")

# Read a subset of rows
first_1000 = torchfits.read("catalog.fits", hdu="CATALOG", num_rows=1000)
```

### File Information

```python
# Get basic file information
num_hdus = torchfits.get_num_hdus("my_image.fits")
dims = torchfits.get_dims("my_image.fits", hdu=1)
hdu_type = torchfits.get_hdu_type("my_image.fits", hdu=1)

print(f"File has {num_hdus} HDUs")
print(f"Primary HDU dimensions: {dims}")
print(f"Primary HDU type: {hdu_type}")
```

### Object-Oriented Approach (Recommended for Multiple Operations)

When you need to perform multiple operations on the same file, use the object-oriented interface for better performance:

```python
import torchfits

with torchfits.FITS("my_multiextension_file.fits") as f:
    print(f"File has {len(f)} HDUs")
    
    # Access HDUs by index or name
    primary = f[0]
    sci_hdu = f["SCI"]
    
    # Read data and access headers
    primary_data = primary.read()
    sci_data = sci_hdu.read(start=[0, 0], shape=[100, 100])  # Cutout
    
    print(f"Primary HDU shape: {primary_data.shape}")
    print(f"Science HDU cutout shape: {sci_data.shape}")
    print(f"EXPTIME: {sci_hdu.header['EXPTIME']}")
```

## Examples

The `examples/` directory contains several example scripts demonstrating various use cases:

* **`example_basic_reading.py`:** Basic reading of full HDUs and headers.  Demonstrates using `torchfits.read` with simple filenames and accessing header information.
* **`example_cutouts.py`:**  Shows how to read cutouts using both CFITSIO-style strings (passed directly to `torchfits.read`) and the `start`/`shape` parameters of the `torchfits.read` function.
* **`example_mef.py`:**  Illustrates working with Multi-Extension FITS (MEF) files, including iterating through HDUs and accessing HDUs by number and name.
* **`example_tables.py`:** Focuses on reading data from FITS binary tables, showing how to access individual columns.
* **`example_datacube.py`:**  Demonstrates reading slices and 1D spectra from a 3D data cube, using both CFITSIO strings and `start`/`shape` parameters.
* **`example_dataset.py`:**  A basic example of integrating `torchfits` with PyTorch's `Dataset` and `DataLoader` for efficient data loading.
* **`example_mnist.py`:** A complete, self-contained example that downloads the MNIST dataset, converts it to FITS, and trains a simple CNN classifier using `torchfits` for FITS I/O.
* **`example_sdss_classification.py`:**  A complete example that downloads a small subset of SDSS spectroscopic data, loads the spectra using `torchfits`, and trains a simple CNN classifier to distinguish between stars, galaxies, and quasars.

To run the examples, navigate to the `examples/` directory and run, for instance:

```bash
python example_basic_reading.py
```

The examples that use external datasets will automatically download and cache the data.

## API Reference

### Core Reading Function

#### `torchfits.read(filename_or_url, hdu=1, start=None, shape=None, columns=None, start_row=0, num_rows=None, cache_capacity=0, device='cpu')`

The main function for reading FITS data into PyTorch tensors. Handles images, data cubes, and tables seamlessly.

**Parameters:**

* `filename_or_url` (str or dict): Path to FITS file, CFITSIO-compatible URL, or fsspec parameters dict for remote files
* `hdu` (int or str, optional): HDU number (1-based) or name. Defaults to 1 (primary HDU)
* `start` (list[int], optional): Starting pixel coordinates (0-based) for cutouts. For 2D: `[row, col]`
* `shape` (list[int], optional): Cutout shape. Required if `start` given. Use `-1` to read to end of dimension
* `columns` (list[str], optional): Column names for table HDUs. Reads all columns if `None`
* `start_row` (int, optional): Starting row (0-based) for table reads. Default: 0
* `num_rows` (int, optional): Number of rows to read from table. Reads all if `None`
* `cache_capacity` (int, optional): Cache size in MB. Default: auto-sized. Set 0 to disable
* `device` (str, optional): Target device ('cpu' or 'cuda'). Default: 'cpu'

**Returns:**

* For images/cubes: tuple `(data, header)` where `data` is PyTorch tensor, `header` is dict
* For tables: dict with column names as keys, PyTorch tensors as values

### File Information Functions

* **`torchfits.get_header(filename, hdu=1)`**: Get complete FITS header as dictionary
* **`torchfits.get_num_hdus(filename)`**: Get total number of HDUs in file  
* **`torchfits.get_dims(filename, hdu=1)`**: Get dimensions of image/cube HDU
* **`torchfits.get_header_value(filename, hdu, key)`**: Get value of specific header keyword
* **`torchfits.get_hdu_type(filename, hdu=1)`**: Get HDU type ("IMAGE", "BINTABLE", "TABLE", "UNKNOWN")

### World Coordinate System (WCS) Functions

* **`torchfits.world_to_pixel(world_coords, header)`**: Convert world coordinates to pixel coordinates using WCS information from header
* **`torchfits.pixel_to_world(pixel_coords, header)`**: Convert pixel coordinates to world coordinates using WCS information from header

### Object-Oriented Interface

For more complex file operations, use the object-oriented interface:

#### `torchfits.FITS(filename)`

Context manager for accessing FITS files. Provides access to individual HDUs.

```python
with torchfits.FITS("file.fits") as f:
    num_hdus = len(f)              # Number of HDUs
    primary = f[0]                 # Access by index
    sci_hdu = f['SCI']             # Access by name
    data = primary.read()          # Read HDU data
    header = primary.header        # Get HDU header
```

#### `torchfits.HDU(filename, hdu_spec)`

Represents a single HDU. Has `read()` method and `header` property.

### WCS Functions

* **`torchfits.world_to_pixel(world_coords, header)`**: Converts world coordinates to pixel coordinates.
* **`torchfits.pixel_to_world(pixel_coords, header)`**: Converts pixel coordinates to world coordinates.

### Classes

* **`torchfits.FITS(filename)`**: A context manager for accessing FITS files.
* **`torchfits.HDU(filename, hdu_spec)`**: Represents a single HDU in a FITS file.

## Reading Remote FITS Files

`torchfits` supports reading FITS files from remote locations using `fsspec` parameters. This allows you to read FITS files directly from cloud storage or other remote sources without downloading them first.

### Example

```python
import torchfits

# Define fsspec parameters for reading a remote FITS file
fsspec_params = {
    'protocol': 'https',
    'host': 'example.com',
    'path': 'path/to/remote_file.fits'
}

# Read the remote FITS file
data, header = torchfits.read(fsspec_params)

print(f"Data shape: {data.shape}, Header: {header}")
```

## Contributing

Contributions are welcome! Traditional GitHub contributions style welcome.

## License

This project is licensed under the GPL-2 License - see the LICENSE file for details.

