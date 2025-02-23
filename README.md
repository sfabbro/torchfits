# torchfits

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

**Prerequisites:**

Before installing `torchfits`, you need to install `cfitsio` and `wcslib` on your system.

*   **Linux (Debian/Ubuntu):**

    ```bash
    sudo apt-get install libcfitsio-dev libwcs-dev
    ```

*   **Linux (Fedora/CentOS/RHEL):**

    ```bash
    sudo yum install cfitsio-devel wcslib-devel
    ```

*   **macOS (Homebrew):**

    ```bash
    brew install cfitsio wcslib
    ```

*   **macOS (MacPorts):**

    ```bash
    sudo port install cfitsio wcslib
    ```

*   **Windows:**
    Download and install the pre-built binaries from the CFITSIO ([https://heasarc.gsfc.nasa.gov/fitsio/](https://heasarc.gsfc.nasa.gov/fitsio/)) and WCSLIB ([http://www.atnf.csiro.au/people/mcalabre/WCS/](http://www.atnf.csiro.au/people/mcalabre/WCS/)) websites.  You may need to manually add the installation directories to your system's `PATH` environment variable.  You will likely also need to specify the include and library directories during the build process (see "Building from Source" below).

**Installation via pip (Recommended):**

Once the prerequisites are installed, you can install `torchfits` using `pip`:

```bash
pip install torchfits
```

This will automatically build the C++ extension.

1. Building from Source:

Clone the Repository:
```bash
git clone [https://github.com/sfabbro/torchfits.git](https://github.com/sfabbro/torchfits.git)
cd torchfits
```

2. Build and install
For an editable install (useful for development):
```bash
pip install -e .
```

## Quickstart
```python
import torchfits

# Read the entire primary HDU of a FITS file.
data, header = torchfits.read("my_image.fits")

# Read a 10x20 cutout from the first extension, starting at (50, 75).
cutout_data, cutout_header = torchfits.read("my_image.fits", hdu=1, start=[50, 75], shape=[10, 20])

#Get the number of HDU
num_hdu = torchfits.get_num_hdus("my_image.fits")

#Get value of header keyword
naxis = torchfits.get_header_value("my_image.fits", 1, "NAXIS")

print(f"NAXIS = {naxis}, Number of HDUs = {num_hdus}, Cutout Shape = {cutout_data.shape}")
```

## Examples

The `examples/` directory contains several example scripts demonstrating various use cases:

*   **`example_basic_reading.py`:** Basic reading of full HDUs and headers.  Demonstrates using `torchfits.read` with simple filenames and accessing header information.
*   **`example_cutouts.py`:**  Shows how to read cutouts using both CFITSIO-style strings (passed directly to `torchfits.read`) and the `start`/`shape` parameters of the `torchfits.read` function.
*   **`example_mef.py`:**  Illustrates working with Multi-Extension FITS (MEF) files, including iterating through HDUs and accessing HDUs by number and name.
*   **`example_tables.py`:** Focuses on reading data from FITS binary tables, showing how to access individual columns.
*   **`example_datacube.py`:**  Demonstrates reading slices and 1D spectra from a 3D data cube, using both CFITSIO strings and `start`/`shape` parameters.
*   **`example_dataset.py`:**  A basic example of integrating `torchfits` with PyTorch's `Dataset` and `DataLoader` for efficient data loading.
*   **`example_mnist.py`:** A complete, self-contained example that downloads the MNIST dataset, converts it to FITS, and trains a simple CNN classifier using `torchfits` for FITS I/O.
*   **`example_sdss_classification.py`:**  A complete example that downloads a small subset of SDSS spectroscopic data, loads the spectra using `torchfits`, and trains a simple CNN classifier to distinguish between stars, galaxies, and quasars.

To run the examples, navigate to the `examples/` directory and run, for instance:

```bash
python example_basic_reading.py
```

The examples that use external datasets will automatically download and cache the data.

## API Reference

*   **`torchfits.read(filename_with_cutout, hdu=None, start=None, shape=None)`:** Reads FITS data.  Handles images, cubes, and tables.  Returns either a tuple `(tensor, header)` for images/cubes, or a dictionary for tables.
    *   `filename_with_cutout` (str):  FITS file path, optionally with a CFITSIO-style cutout string (e.g., `'myimage.fits[1][10:20,30:40]'`).
    *   `hdu` (int or str, optional): HDU number (1-based) or name (string). Defaults to the primary HDU if no cutout string is provided that specifies the HDU.
    *   `start` (list[int], optional): Starting pixel coordinates (0-based) for a cutout.
    *   `shape` (list[int], optional): Shape of the cutout.  Use `-1` or `None` for a dimension to read to the end.  If `start` is given, `shape` *must* also be given.

*   **`torchfits.get_header(filename, hdu_num)`:** Returns the FITS header as a dictionary.
    *   `filename` (str):  FITS file path.
    *    `hdu_num` (int or str): HDU number (1-based) or name (string).

*   **`torchfits.get_dims(filename, hdu_num)`:** Returns the dimensions of an image/cube HDU.
    *   `filename` (str):  FITS file path.
    *    `hdu_num` (int or str): HDU number (1-based) or name (string).
*   **`torchfits.get_header_value(filename, hdu_num, key)`:** Returns the value of a single header keyword.
    *   `filename` (str):  FITS file path.
    *   `hdu_num` (int or str): HDU number (1-based) or name (string).
    *    `key` (str): The header keyword.

*   **`torchfits.get_hdu_type(filename, hdu_num)`:** Returns the HDU type as a string ("IMAGE", "BINTABLE", "TABLE", or "UNKNOWN").
    *   `filename` (str):  FITS file path.
    *   `hdu_num` (int or str): HDU number (1-based) or name (string).

*   **`torchfits.get_num_hdus(filename)`:** Returns the total number of HDUs in the FITS file.
    *  `filename` (str): The path to the FITS file.

## Contributing
Contributions are welcome! Traditional GitHub contributions style welcome.

## License
This project is licensed under the GPL-2 License - see the LICENSE file for details.

