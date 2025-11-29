# torchfits: Design and API Specification

> **Technical Vision**
> This document outlines the architecture for torchfits, a library engineered to be the fastest and most memory-efficient bridge between FITS files and the PyTorch ecosystem. Our design prioritizes raw I/O throughput, zero-copy data handling, and a simple, ML-centric Python API for all common astronomical data formats.

## 1. Core Philosophy and Goals

The central objective of torchfits is to eliminate the data loading bottleneck in astronomical machine learning. To achieve this, the library is built on three foundational principles:

- **Performance First:** The core design goal is to be demonstrably faster than the fitsio library for all common use cases. This will be achieved through a C++ backend that minimizes overhead, avoids intermediate data copies, and leverages advanced I/O techniques.
- **Simplicity and Usability:** The Python API will be intuitive and "PyTorch-native." It should feel like a natural extension of the PyTorch library, enabling users to go from any FITS data structure to a torch.Tensor on the target device with minimal boilerplate.
- **Broad Functionality:** The library will robustly support the most common FITS structures used in astronomy, including large images, multi-extension files (MEFs), massive binary tables, and n-dimensional data cubes/spectra, along with essential WCS transformations.

## 2. System Architecture: A Two-Layer Model

torchfits employs a hybrid C++/Python architecture to delegate tasks to the most suitable environment.

- **C++ Core Engine:** A compiled C++ extension that handles all performance-critical operations. It directly interfaces with cfitsio and wcslib for file I/O, byte parsing, and WCS computations. This allows for meticulous memory management, direct tensor creation, and multi-threading to bypass Python's Global Interpreter Lock (GIL).
- **Python API Layer:** A high-level, user-facing API written in Python. It provides the user-friendly interface and integrates seamlessly with PyTorch's data structures and machine learning workflows (Dataset, DataLoader).
- **Bindings:** We will continue to use pybind11 for its robust and efficient integration between C++ and Python, especially its excellent handling of PyTorch C++ tensors.

## 3. The C++ Core Engine: The Performance Driver

The C++ backend is the heart of torchfits and the key to outperforming competitors. Success hinges on the following low-level implementation details, which apply generically to all data types.

### 3.1. Zero-Copy Tensor Creation

The highest priority is to move data from the file buffer directly into a PyTorch tensor's memory space without intermediate copies.

- **Memory Pre-allocation:** The C++ layer will allocate the required memory for a torch::Tensor before reading the data from the FITS file.
- **Direct Read:** Use cfitsio functions like `fits_read_subset` (for arrays) or `fits_read_col` (for tables) to read data directly into the pre-allocated tensor's data pointer. This is the most critical performance optimization.

### 3.2. Advanced I/O and Memory Strategy

To maximize throughput, the engine will employ several strategies:

- **Memory Mapping:** For read-only access, `mmap` will be used where possible to map the file directly into memory, letting the OS handle efficient page caching.
- **Optimized cfitsio Usage:**
  - **Tiled Compression:** When reading compressed data arrays, torchfits will leverage cfitsio's tiled reading capabilities. This is essential for efficiently extracting small cutouts from large, compressed mosaics without decompressing the entire file.
  - **Columnar Table I/O:** For reading binary tables, the C++ layer will read data column-by-column, which is the most efficient access pattern for cfitsio and ML workflows.
- **Caching:** An internal cache will store fitsfile pointers and frequently accessed headers to reduce I/O overhead in iterative ML workflows.

### 3.3. Parallelism

The C++ backend will be designed to be thread-safe, allowing multiple DataLoader workers to read from different files (or different HDUs of the same file) in parallel without being blocked by the GIL.

## 4. The Python API: Simplicity and Integration

The user-facing API is designed to be minimal, intuitive, and ML-centric, with clear methods for each FITS data type.

### 4.1. High-Level File Access & Diverse Use Cases

The primary entry point is a simple `torchfits.open()` function that handles various FITS structures gracefully.

```python
import torchfits
import torch

# Use Case 1: Reading a large image and a cutout
with torchfits.open('large_image.fits') as f:
    image_tensor = f[0].read(device='cuda:0')
    cutout = f[0].read(section=(slice(100, 200), slice(100, 200)))

# Use Case 2: Reading a large catalog (BINTABLE)
with torchfits.open('huge_catalog.fits') as f:
    table_data = f[1].read_table(
        columns=['RA', 'DEC', 'FLUX_G'], 
        rows=slice(1_000_000, 2_000_000),
        device='cpu'
    )

# Use Case 3: Reading a 3D Data Cube
with torchfits.open('datacube.fits') as f:
    cube = f[0].read() # cube.shape might be (500, 1024, 1024)

# Use Case 4: Working with a Multi-Extension File (MEF)
with torchfits.open('multi_extension_file.fits') as f:
    science_image = f['SCI'].read()
    error_array = f['ERR'].read()
    source_catalog = f['CAT'].read_table(columns=['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO'])

# Use Case 5: Handling Spectra with Variance and Masks
with torchfits.open('spectrum.fits') as f:
    # A common pattern in spectroscopic files (e.g., SDSS)
    flux = f['FLUX'].read()
    ivar = f['IVAR'].read()
    mask = f['MASK'].read()
    
    # Also access metadata from a table extension
    metadata = f['METADATA'].read_table(columns=['Z', 'CLASS'])
    redshift = metadata['Z'][0]
```

### 4.2. HDU Objects

The `open()` function returns a list-like object containing `ImageHDU` and `TableHDU` objects.

- **ImageHDU** (Handles IMAGE, PRIMARY array HDUs):
  - `.read_header() -> dict`: Reads the header.
  - `.read(section: tuple | None = None, device: str = 'cpu') -> torch.Tensor`: Reads the full N-dimensional data array (image, cube, spectrum) or a specified slice directly to the target device.
  - `.wcs -> WCS`: A property that returns a WCS object.
- **TableHDU** (Handles BINTABLE HDUs):
  - `.read_header() -> dict`: Reads the table header.
  - `.read_table(columns: list[str] | None = None, rows: slice | None = None, device: str = 'cpu') -> dict[str, torch.Tensor]`: Reads specified columns and/or rows into a dictionary of tensors.

### 4.3. WCS Transformations

WCS functionality will be provided by a `WCS` object, powered by wcslib in the C++ backend. It will operate directly on PyTorch tensors.

```python
with torchfits.open('my_image.fits') as f:
    wcs = f[0].wcs
    pixel_coords = torch.tensor([[100.0, 200.0], [150.0, 250.0]], device='cuda:0')
    world_coords = wcs.pixel_to_world(pixel_coords)
```

### 4.4. Integration with torch.utils.data.Dataset

The API is designed for efficient use within a PyTorch Dataset, easily handling complex, multi-extension data like spectra.

```python
from torch.utils.data import Dataset

class SpectroscopicDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Open a multi-extension spectroscopic FITS file
        with torchfits.open(self.file_paths[idx], cache=True) as f:
            # Load the essential arrays
            flux = f['FLUX'].read()
            ivar = f['IVAR'].read()
            
            # Load the target class from a table extension
            target_info = f['METADATA'].read_table(columns=['CLASS'])
            target_class = target_info['CLASS'][0] # Assuming one object per file

        # The transform could be, e.g., normalization
        if self.transform:
            flux = self.transform(flux)
        
        # Return a dictionary of tensors, a common pattern for ML
        return {
            'flux': flux,
            'ivar': ivar,
            'target': torch.tensor(target_class)
        }
```

## 5. Writing FITS Files

torchfits will support writing for both primary data arrays and tables.

```python
import torchfits
import torch

# Example 1: Writing an image
image_tensor = torch.randn(1024, 1024, device='cuda:0')
header = {'OBJECT': 'NGC 101'}
torchfits.writeto('new_image.fits', data=image_tensor, header=header)

# Example 2: Writing a table from a dictionary of tensors
table_data = {
    'TARGET_ID': torch.arange(100),
    'RA': torch.rand(100) * 360,
    'DEC': (torch.rand(100) * 180) - 90
}
torchfits.writeto('new_catalog.fits', data=table_data, ext_name='CATALOG')
```
